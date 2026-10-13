"""Validation on the prompts the model is used with.

The trainer used to validate with one centroid point only. Production starts from an annotator's circle, so
:class:`ValidationSuite` scores the same validation split once per prompt mode (``eval.val_prompts``) with fixed,
reproducible prompts, and reports IoU and boundary F1 per volume and per category. One mode
(``eval.primary_val_prompt``) is the number that picks the best checkpoint and drives early stopping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from sam2_segmentation_trainer.data.dataset import VAL_PROMPTS, EMSegDataset, collate_em
from sam2_segmentation_trainer.metrics import boundary_f1, iou_dice_pr
from sam2_segmentation_trainer.model import predict_masks, predict_masks_mixed
from sam2_segmentation_trainer.overlays import denorm_preview, overlay_panel

DEFAULT_VAL_PROMPTS = ("circle_mask", "point", "centroid")
"""Validation prompts when ``eval.val_prompts`` is unset: production circle, interior point, legacy centroid."""


@dataclass(slots=True)
class ModeResult:
    """Scores of one validation prompt mode.

    ``iou`` and ``bf`` average the samples that are not fully masked tiles; those (``fulltile_*``) are kept
    apart because a point inside a tile the target fills scores about 1 and would inflate the headline number.
    """

    iou: float = 0.0
    bf: float = 0.0
    n: int = 0
    by_volume: dict[str, float] = field(default_factory=dict)
    by_category: dict[str, float] = field(default_factory=dict)
    fulltile_iou: float | None = None
    fulltile_n: int = 0


def summarize_mode(records: Sequence[dict[str, Any]]) -> ModeResult:
    """Collapse per-sample records (volume, category, iou, bf, full_tile) into a :class:`ModeResult`."""
    result = ModeResult()
    normal = [r for r in records if not r.get("full_tile")]
    full = [r for r in records if r.get("full_tile")]
    if full:
        result.fulltile_iou = float(np.mean([r["iou"] for r in full]))
        result.fulltile_n = len(full)
    # With nothing but fully masked tiles there is no other number to report, so fall back to them.
    scored = normal or full
    if not scored:
        return result
    result.n = len(scored)
    result.iou = float(np.mean([r["iou"] for r in scored]))
    result.bf = float(np.mean([r["bf"] for r in scored]))
    for field_name, target in (("volume", result.by_volume), ("category", result.by_category)):
        buckets: dict[str, list[float]] = {}
        for record in scored:
            name = str(record.get(field_name) or "").strip()
            if name:
                buckets.setdefault(name, []).append(float(record["iou"]))
        target.update({name: float(np.mean(values)) for name, values in sorted(buckets.items())})
    return result


def predict_batch(model, batch, device, image_size: int) -> torch.Tensor:
    """Predicted logits `[B, H, W]` for a validation batch, whatever prompts its samples carry.

    A batch with boxes or dense masks goes through :func:`predict_masks_mixed`; a plain point batch keeps the
    original :func:`predict_masks` path.
    """
    images = batch["image"].to(device, non_blocking=True)
    points = batch["point"].to(device, non_blocking=True)
    labels = batch["point_label"].to(device, non_blocking=True)
    boxes = batch.get("boxes") or []
    dense = batch.get("dense_mask") or []
    if any(b is not None for b in boxes) or any(d is not None for d in dense):
        pred, _iou = predict_masks_mixed(model, images, points, labels, boxes, dense, image_size=image_size)
    else:
        pred, _iou = predict_masks(model, images, points, labels, image_size=image_size)
    return pred


def collect_records(
    model,
    loader,
    device,
    image_size: int,
    threshold: float,
    *,
    overlays_per_volume: int = 0,
) -> tuple[list[dict[str, Any]], list[tuple[str, np.ndarray]]]:
    """Score every batch of ``loader``. Returns per-sample records and a few overlay strips.

    A batch with boxes or dense masks goes through :func:`predict_masks_mixed`; a plain point batch keeps the
    original :func:`predict_masks` path.
    """
    model.eval()
    records: list[dict[str, Any]] = []
    overlays: list[tuple[str, np.ndarray]] = []
    overlay_counts: dict[str, int] = {}
    keep_overlays = max(0, int(overlays_per_volume))
    try:
        with torch.no_grad():
            for batch in loader:
                masks = batch["mask"].to(device, non_blocking=True)
                pred = predict_batch(model, batch, device, image_size)
                pred_bin = (torch.sigmoid(pred) > threshold).to(torch.int64)
                meta = batch.get("meta") or []
                for index, (pred_i, mask_i, item) in enumerate(zip(pred_bin, masks, meta, strict=True)):
                    pred_np = pred_i.detach().cpu().numpy()
                    gt_np = mask_i.detach().cpu().numpy()
                    stats = iou_dice_pr(pred_np, gt_np)
                    item = item or {}
                    volume = str(item.get("volume", "")).strip()
                    records.append(
                        {
                            "volume": volume,
                            "category": str(item.get("category_name", "")),
                            "iou": stats["iou"],
                            "bf": boundary_f1(pred_np, gt_np),
                            "full_tile": bool(item.get("full_tile", False)),
                        }
                    )
                    if not volume or keep_overlays <= 0:
                        continue
                    seen = overlay_counts.get(volume, 0)
                    if seen >= keep_overlays:
                        continue
                    panel = overlay_panel(denorm_preview(batch["image"][index]), gt_np, pred_np)
                    overlays.append((volume, panel))
                    overlay_counts[volume] = seen + 1
    finally:
        model.train()
    return records, overlays


class ValidationSuite:
    """The validation split under every configured prompt mode.

    ``dataset`` is the primary mode's dataset (the one callers size and refresh); :meth:`replace_examples`
    keeps all modes in step when the inventory changes.
    """

    def __init__(
        self,
        examples: Sequence,
        *,
        modes: Sequence[str],
        primary: str,
        image_size: int,
        num_workers: int,
        missing_tiles=None,
        tiles=None,
        prompt_cfg=None,
        seed: int = 0,
        circle_scale: float = 0.95,
        full_tile_threshold: float | None = None,
    ) -> None:
        modes = list(dict.fromkeys(modes))
        if not modes:
            raise ValueError("eval.val_prompts must name at least one prompt")
        unknown = [m for m in modes if m not in VAL_PROMPTS]
        if unknown:
            raise ValueError(f"unknown validation prompt(s) {unknown}; choose from {list(VAL_PROMPTS)}")
        if primary not in modes:
            raise ValueError(f"eval.primary_val_prompt {primary!r} is not in eval.val_prompts {modes}")
        self.modes = modes
        self.primary = primary
        self.num_workers = int(num_workers)
        self.datasets = {
            mode: EMSegDataset(
                examples,
                split="val",
                image_size=image_size,
                missing_tiles=missing_tiles,
                tiles=tiles,
                prompt_cfg=prompt_cfg,
                seed=seed,
                val_prompt=mode,
                circle_scale=circle_scale,
                full_tile_threshold=full_tile_threshold,
            )
            for mode in modes
        }

    @property
    def dataset(self) -> EMSegDataset:
        return self.datasets[self.primary]

    def __len__(self) -> int:
        return len(self.dataset)

    def replace_examples(self, examples: Sequence) -> None:
        for dataset in self.datasets.values():
            dataset.replace_examples(examples)

    def set_tiles(self, tiles) -> None:
        for dataset in self.datasets.values():
            dataset.tiles = tiles

    def loader(self, mode: str) -> DataLoader:
        return DataLoader(
            self.datasets[mode],
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_em,
        )

    def run(
        self,
        model,
        device,
        image_size: int,
        threshold: float,
        *,
        overlays_per_volume: int = 0,
    ) -> tuple[dict[str, ModeResult], list[tuple[str, np.ndarray]]]:
        """Score every mode. Overlay strips come from the primary mode only."""
        results: dict[str, ModeResult] = {}
        overlays: list[tuple[str, np.ndarray]] = []
        for mode in self.modes:
            records, strips = collect_records(
                model,
                self.loader(mode),
                device,
                image_size,
                threshold,
                overlays_per_volume=overlays_per_volume if mode == self.primary else 0,
            )
            results[mode] = summarize_mode(records)
            if mode == self.primary:
                overlays = strips
        return results, overlays


def log_results(writer, epoch: int, results: dict[str, ModeResult], primary: str) -> str:
    """Write every mode's scores to TensorBoard and return the one-line summary for the transcript.

    ``val/iou`` and ``val_iou/{volume}`` keep their old names and now carry the primary mode, so existing
    dashboards follow the number that selects the checkpoint. Each mode is also logged under its own name.
    """
    for mode, result in results.items():
        writer.add_scalar(f"val/iou/{mode}", result.iou, epoch)
        writer.add_scalar(f"val/bf/{mode}", result.bf, epoch)
        if result.fulltile_iou is not None:
            writer.add_scalar(f"val/iou_fulltile/{mode}", result.fulltile_iou, epoch)
        for volume, score in result.by_volume.items():
            writer.add_scalar(f"val_iou/{mode}/{volume}", score, epoch)
        for category, score in result.by_category.items():
            writer.add_scalar(f"val_iou_cat/{mode}/{category}", score, epoch)
    main = results[primary]
    writer.add_scalar("val/iou", main.iou, epoch)
    for volume, score in main.by_volume.items():
        writer.add_scalar(f"val_iou/{volume}", score, epoch)
    per_volume = ", ".join(f"{name} {score:.4f}" for name, score in main.by_volume.items())
    others = ", ".join(
        f"{mode} {result.iou:.4f}" for mode, result in results.items() if mode != primary
    )
    text = f"epoch {epoch} val IoU {main.iou:.4f} [{primary}, bf {main.bf:.4f}]"
    if per_volume:
        text += f" ({per_volume})"
    if others:
        text += f"; other prompts: {others}"
    if main.fulltile_iou is not None:
        text += f"; fully masked tiles ({main.fulltile_n}): {main.fulltile_iou:.4f}"
    return text
