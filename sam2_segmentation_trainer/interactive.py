"""Simulated corrective clicks: iterative training rounds and the NoC@90 interaction-cost metric.

An annotator who is not happy with a prediction clicks where it is wrong and the model tries again, seeing its own
previous mask. SAM 2 trains for that by predicting, sampling a click from the error region, and re-running with
the previous low-resolution logits plus the new click. A ground-truth mask is never fed in as the prompt, so the
model cannot learn to copy its input. The same loop, run until the IoU reaches a target, gives the number of
clicks (NoC@90) the standard interactive-segmentation benchmarks report.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

from sam2_segmentation_trainer.model import decode_grouped, encode_images

MASK_LOGIT_CLIP = 20.0
"""Previous logits are clipped to this before they become the next mask prompt (the range the server feeds)."""


def next_click(pred: np.ndarray, gt: np.ndarray, rng: np.random.Generator) -> tuple[int, int, int]:
    """One corrective click ``(x, y, label)`` in the grid of ``pred``.

    The region is chosen by area from the missed pixels (a positive click) and the spurious ones (a negative
    click), and the click lands at its deepest interior pixel. With no error the click is a positive one inside
    the target, so an already perfect prediction still gets a harmless extra hint.
    """
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    missed = gt & ~pred
    spurious = pred & ~gt
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
    low_res_logits: torch.Tensor, gt: torch.Tensor, rng: np.random.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """A corrective click per sample, from the decoder's low-resolution logits and the full-size ground truth.

    The comparison runs at the logits' own resolution, which keeps the distance transform cheap. Returns
    ``(coords [B, 2] float32, labels [B] int32)`` in the ground truth's pixel space.
    """
    batch, low_h, low_w = low_res_logits.shape
    height, width = gt.shape[-2:]
    pred_small = (low_res_logits > 0).detach().cpu().numpy()
    gt_small = (F.adaptive_avg_pool2d(gt.float().unsqueeze(1), (low_h, low_w))[:, 0] >= 0.5).cpu().numpy()
    coords = np.zeros((batch, 2), dtype=np.float32)
    labels = np.zeros(batch, dtype=np.int32)
    for i in range(batch):
        x, y, label = next_click(pred_small[i], gt_small[i], rng)
        coords[i] = ((x + 0.5) * width / low_w, (y + 0.5) * height / low_h)
        labels[i] = label
    device = low_res_logits.device
    return torch.from_numpy(coords).to(device), torch.from_numpy(labels).to(device)


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
):
    """Prediction after ``rounds`` corrective clicks. Only the last pass carries gradients.

    The first ``rounds`` passes run without gradient: each predicts from the prompts so far, samples a click from
    where the prediction is wrong, and feeds back its own low-resolution logits as the mask prompt (for a
    ``prev_mask_prob`` share of samples; the others keep their initial prompt, as muSAM does so the model does
    not learn to depend on a previous mask). The final pass has the loss's gradient. ``rounds`` 0 is an
    ordinary mixed prediction. Returns ``(pred_logits, iou_preds)`` like ``predict_masks_mixed``.

    With ``variant_index`` the prompts are several variants per image (see ``predict_masks_mixed``); ``gt_masks``
    stays per image and is expanded to the variants here.
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
                coords, click_labels = error_clicks(low, gt_masks, rng)
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
) -> list[dict[str, float | int | bool]]:
    """Clicks needed to reach ``target_iou`` for each sample of a validation batch (NoC@``target_iou``).

    The first click is the batch's own point (use the ``point`` validation prompt: the deepest interior pixel).
    Every later click is sampled from the error region, with the previous logits fed back as the mask prompt, as
    an annotator correcting the model would do. A sample that never reaches the target reports ``max_clicks`` and
    ``reached`` False.
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
    final_iou = [0.0] * count
    reached = [False] * count
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
            for i in range(count):
                final_iou[i] = float(iou[i])
                if not reached[i] and iou[i] >= target_iou:
                    reached[i], noc[i] = True, click
            if all(reached) or click == int(max_clicks):
                break
            coords, click_labels = error_clicks(low, gt, rng)
            points, labels = _append_click(points, labels, coords, click_labels)
            dense = _prev_mask_prompts(low, dense, [True] * count)
    return [{"noc": noc[i], "reached": reached[i], "iou": final_iou[i]} for i in range(count)]


def summarize_noc(rows: Sequence[dict], volumes: Sequence[str], max_clicks: int) -> dict[str, object]:
    """Mean clicks and the share of samples that never reached the target, overall and per volume."""
    if not rows:
        return {"n": 0}

    def block(items: Sequence[dict]) -> dict[str, float]:
        return {
            "mean_clicks": float(np.mean([r["noc"] for r in items])),
            "failed": float(np.mean([not r["reached"] for r in items])),
            "n": len(items),
        }

    by_volume: dict[str, list[dict]] = {}
    for row, volume in zip(rows, volumes, strict=True):
        by_volume.setdefault(str(volume), []).append(row)
    return {
        "max_clicks": int(max_clicks),
        "overall": block(rows),
        "by_volume": {name: block(items) for name, items in sorted(by_volume.items())},
    }
