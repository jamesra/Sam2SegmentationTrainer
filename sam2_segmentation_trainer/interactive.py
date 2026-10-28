"""Simulated corrective clicks: iterative training rounds and the NoC@90 interaction-cost metric.

An annotator who is not happy with a prediction clicks where it is wrong and the model tries again, seeing its own
previous mask. SAM 2 trains for that by predicting, sampling a click from the error region, and re-running with
the previous low-resolution logits plus the new click. A ground-truth mask is never fed in as the prompt, so the
model cannot learn to copy its input. The same loop, run until the IoU reaches a target, gives the number of
clicks (NoC@90) the standard interactive-segmentation benchmarks report.

Clicks target the errors the project cares about, not trace noise: with a core band ``b`` the missed region is the
part of the ground truth at least ``b`` px inside its edge that the prediction lacks, and the spurious region is the
prediction more than ``b`` px outside the ground truth (see ``goal_metrics``). When both are empty the
click is the usual positive one at the deepest target pixel, never a click on edge-band noise. Only a band of 0
gives the raw regions.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

from annotation_crops.prompt_encoding import edge_distance
from sam2_segmentation_trainer.data.refine import grow_mask
from sam2_segmentation_trainer.goal_metrics import (
    CORE_RECALL_TARGET,
    MERGE_MIN_PX,
    MIN_COMPONENT_PX,
    ground_truth_terms,
    sample_goal_metrics,
)
from sam2_segmentation_trainer.model import decode_grouped, encode_images, to_device_async

MASK_LOGIT_CLIP = 20.0
"""Previous logits are clipped to this before they become the next mask prompt (the range the server feeds)."""


def error_regions(pred: np.ndarray, gt: np.ndarray, band: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """``(missed, spurious)`` regions of ``pred`` against ``gt`` (boolean, same grid).

    With ``band`` > 0 (px of this grid) they are the goal errors: ``missed`` is the part of ``gt`` at least ``band``
    inside its edge that is not predicted, ``spurious`` the prediction more than ``band`` outside ``gt``. When both
    are empty, both are returned empty: what is left is trace noise in the edge band, and :func:`next_click` then
    takes its "no error" path. Only ``band`` 0 gives the raw differences.
    """
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    if band > 0.0:
        return (edge_distance(gt) >= band) & ~pred, pred & ~grow_mask(gt, band)
    return gt & ~pred, pred & ~gt


def next_click(
    pred: np.ndarray, gt: np.ndarray, rng: np.random.Generator, band: float = 0.0
) -> tuple[int, int, int]:
    """One corrective click ``(x, y, label)`` in the grid of ``pred``.

    The region is chosen by area from the missed pixels (a positive click) and the spurious ones (a negative
    click), as :func:`error_regions` defines them for ``band``, and the click lands at its deepest interior pixel.
    With no error (with a band: no goal error) the click is a positive one at the deepest target pixel, so an
    already good prediction still gets a harmless extra hint.
    """
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    missed, spurious = error_regions(pred, gt, band)
    n_missed, n_spurious = int(missed.sum()), int(spurious.sum())
    if n_missed + n_spurious == 0:
        region, label = gt, 1
    elif rng.random() * (n_missed + n_spurious) < n_missed:
        region, label = missed, 1
    else:
        region, label = spurious, 0
    if not region.any():
        height, width = gt.shape
        return width // 2, height // 2, label
    depth = distance_transform_edt(np.pad(region, 1))[1:-1, 1:-1]
    y, x = np.unravel_index(int(np.argmax(depth)), depth.shape)
    return int(x), int(y), label


def error_clicks(
    low_res_logits: torch.Tensor, gt: torch.Tensor, rng: np.random.Generator, core_band_px: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """A corrective click per sample, from the decoder's low-resolution logits and the full-size ground truth.

    The comparison runs at the logits' own resolution, which keeps the distance transform cheap.
    ``core_band_px`` (full-size px, ``val.core_band_px``) is scaled to that grid and selects goal errors (see
    :func:`error_regions`). Returns ``(coords [B, 2] float32, labels [B] int32)`` in the ground truth's pixel space.
    """
    batch, low_h, low_w = low_res_logits.shape
    height, width = gt.shape[-2:]
    pred_small = (low_res_logits > 0).detach().cpu().numpy()
    gt_small = (F.adaptive_avg_pool2d(gt.float().unsqueeze(1), (low_h, low_w))[:, 0] >= 0.5).cpu().numpy()
    coords = np.zeros((batch, 2), dtype=np.float32)
    labels = np.zeros(batch, dtype=np.int32)
    band = float(core_band_px) * low_w / width
    for i in range(batch):
        x, y, label = next_click(pred_small[i], gt_small[i], rng, band)
        coords[i] = ((x + 0.5) * width / low_w, (y + 0.5) * height / low_h)
        labels[i] = label
    device = low_res_logits.device
    return to_device_async(torch.from_numpy(coords), device), to_device_async(torch.from_numpy(labels), device)


def _append_click(
    points: torch.Tensor, labels: torch.Tensor, coords: torch.Tensor, click_labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.cat([points, coords.to(points.dtype).unsqueeze(1)], dim=1),
        torch.cat([labels, click_labels.to(labels.dtype).unsqueeze(1)], dim=1),
    )


def _prev_mask_prompts(
    low_res: torch.Tensor,
    initial: Sequence[torch.Tensor | None],
    use_prev: Sequence[bool],
) -> list[torch.Tensor | None]:
    """Mask prompt per sample: the clipped previous logits where ``use_prev``, else the sample's initial one."""
    clipped = low_res.detach().clamp(-MASK_LOGIT_CLIP, MASK_LOGIT_CLIP)
    return [clipped[i] if use_prev[i] else initial[i] for i in range(low_res.shape[0])]


def iterative_predict_masks(
    model,
    images: torch.Tensor,
    point_coords: torch.Tensor,
    point_labels: torch.Tensor,
    boxes: Sequence[torch.Tensor | None],
    dense_masks: Sequence[torch.Tensor | None],
    gt_masks: torch.Tensor,
    *,
    image_size: int,
    rounds: int,
    prev_mask_prob: float,
    rng: np.random.Generator,
    orig_hw: tuple[int, int] | None = None,
    variant_index: torch.Tensor | None = None,
    core_band_px: float = 0.0,
):
    """Prediction after ``rounds`` corrective clicks. Only the last pass carries gradients.

    The first ``rounds`` passes run without gradient: each predicts from the prompts so far, samples a click from
    where the prediction is wrong, and feeds back its own low-resolution logits as the mask prompt (for a
    ``prev_mask_prob`` share of samples; the others keep their initial prompt, as muSAM does so the model does
    not learn to depend on a previous mask). The final pass has the loss's gradient. ``rounds`` 0 is an
    ordinary mixed prediction. Returns ``(pred_logits, iou_preds)`` like ``predict_masks_mixed``.

    With ``variant_index`` the prompts are several variants per image (see ``predict_masks_mixed``); ``gt_masks``
    stays per image and is expanded to the variants here. ``core_band_px`` makes the clicks target goal errors (see
    :func:`error_clicks`).
    """
    image_embed, high_res = encode_images(model, images)
    points, labels = point_coords, point_labels
    current = list(dense_masks)
    batch = len(current)
    if variant_index is not None:
        gt_masks = gt_masks[variant_index]
    if rounds > 0:
        with torch.no_grad():
            frozen_embed = image_embed.detach()
            frozen_high = [feat.detach() for feat in high_res]
            for _ in range(int(rounds)):
                _pred, _iou, low = decode_grouped(
                    model, frozen_embed, frozen_high, points, labels, boxes, current,
                    image_size=image_size, orig_hw=orig_hw, return_low_res=True,
                    embed_index=variant_index,
                )
                coords, click_labels = error_clicks(low, gt_masks, rng, core_band_px)
                points, labels = _append_click(points, labels, coords, click_labels)
                use_prev = (rng.random(batch) < float(prev_mask_prob)).tolist()
                current = _prev_mask_prompts(low, dense_masks, use_prev)
    return decode_grouped(
        model, image_embed, high_res, points, labels, boxes, current,
        image_size=image_size, orig_hw=orig_hw, embed_index=variant_index,
    )


def simulate_clicks(
    model,
    batch: dict,
    device,
    image_size: int,
    *,
    max_clicks: int = 20,
    target_iou: float = 0.9,
    threshold: float = 0.5,
    seed: int = 0,
    core_band_px: float = 0.0,
    core_recall_target: float = CORE_RECALL_TARGET,
    merge_min_px: int = MERGE_MIN_PX,
) -> list[dict[str, float | int | bool]]:
    """Clicks needed for each sample of a validation batch, by two stopping rules.

    ``noc`` is NoC@``target_iou``: the first click count at which the raw IoU reaches the target. ``noc_goal`` is
    the first at which the core recall reaches ``core_recall_target`` with no merge into a neighbor (the batch needs
    ``neighbor_core``, from a dataset with ``core_band_px`` > 0; without it only the recall counts; a ground truth
    with no core falls back to the IoU target). The first click is the batch's own point (use the ``point``
    validation prompt: the deepest interior pixel). Every later click is sampled from the errors (goal errors for
    ``core_band_px`` > 0, see :func:`error_regions`), with the previous logits fed back as the mask prompt, as an
    annotator correcting the model would do. A sample that never satisfies a rule reports ``max_clicks`` and
    ``reached`` (``reached_goal``) False. The loop stops when every sample has satisfied both rules.
    """
    model.eval()
    rng = np.random.default_rng(seed)
    images = batch["image"].to(device)
    gt = batch["mask"].to(device)
    points = batch["point"].to(device)[:, :1]
    labels = batch["point_label"].to(device)[:, :1]
    count = images.shape[0]
    boxes: list[torch.Tensor | None] = [None] * count
    dense: list[torch.Tensor | None] = [None] * count
    noc = [int(max_clicks)] * count
    noc_goal = [int(max_clicks)] * count
    final_iou = [0.0] * count
    final_recall: list[float | None] = [None] * count
    final_merge = [False] * count
    reached = [False] * count
    reached_goal = [False] * count
    cores = batch.get("neighbor_core")
    gt_np = gt.cpu().numpy() > 0
    truths: list[dict | None] = [None] * count
    with torch.no_grad():
        image_embed, high_res = encode_images(model, images)
        for click in range(1, int(max_clicks) + 1):
            pred, _iou, low = decode_grouped(
                model, image_embed, high_res, points, labels, boxes, dense,
                image_size=image_size, return_low_res=True,
            )
            pred_bin = torch.sigmoid(pred) > threshold
            gt_bin = gt.bool()
            inter = (pred_bin & gt_bin).flatten(1).sum(1).float()
            union = (pred_bin | gt_bin).flatten(1).sum(1).float()
            iou = (inter / (union + 1e-6)).tolist()
            pred_np = pred_bin.cpu().numpy()
            for i in range(count):
                final_iou[i] = float(iou[i])
                if not reached[i] and iou[i] >= target_iou:
                    reached[i], noc[i] = True, click
                if reached_goal[i]:
                    continue
                if truths[i] is None:
                    truths[i] = ground_truth_terms(gt_np[i], core_band_px, MIN_COMPONENT_PX)
                goal = sample_goal_metrics(
                    pred_np[i], gt_np[i], None if cores is None else cores[i].numpy(),
                    band=core_band_px, merge_min_px=merge_min_px, cache=truths[i],
                )
                final_recall[i], final_merge[i] = goal["recall"], bool(goal["merge"])
                recall_ok = iou[i] >= target_iou if goal["recall"] is None else goal["recall"] >= core_recall_target
                if recall_ok and not goal["merge"]:
                    reached_goal[i], noc_goal[i] = True, click
            if (all(reached) and all(reached_goal)) or click == int(max_clicks):
                break
            coords, click_labels = error_clicks(low, gt, rng, core_band_px)
            points, labels = _append_click(points, labels, coords, click_labels)
            dense = _prev_mask_prompts(low, dense, [True] * count)
    return [
        {
            "noc": noc[i],
            "reached": reached[i],
            "iou": final_iou[i],
            "noc_goal": noc_goal[i],
            "reached_goal": reached_goal[i],
            "core_recall": final_recall[i],
            "merge": final_merge[i],
        }
        for i in range(count)
    ]


def summarize_noc(rows: Sequence[dict], volumes: Sequence[str], max_clicks: int) -> dict[str, object]:
    """Mean clicks and the share of samples that never reached the target, overall and per volume.

    Both stopping rules are reported: ``mean_clicks`` / ``failed`` for the IoU target, ``mean_clicks_goal`` /
    ``failed_goal`` for core recall at its target with no merge (rows from :func:`simulate_clicks`).
    """
    if not rows:
        return {"n": 0}

    def block(items: Sequence[dict]) -> dict[str, float]:
        out = {
            "mean_clicks": float(np.mean([r["noc"] for r in items])),
            "failed": float(np.mean([not r["reached"] for r in items])),
            "n": len(items),
        }
        goal = [r for r in items if "noc_goal" in r]
        if goal:
            out["mean_clicks_goal"] = float(np.mean([r["noc_goal"] for r in goal]))
            out["failed_goal"] = float(np.mean([not r["reached_goal"] for r in goal]))
        return out

    by_volume: dict[str, list[dict]] = {}
    for row, volume in zip(rows, volumes, strict=True):
        by_volume.setdefault(str(volume), []).append(row)
    return {
        "max_clicks": int(max_clicks),
        "overall": block(rows),
        "by_volume": {name: block(items) for name, items in sorted(by_volume.items())},
    }
