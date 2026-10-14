"""Segmentation metrics for EM SAM2 eval."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np
from scipy.ndimage import distance_transform_cdt
from skimage.segmentation import find_boundaries


def confusion(pred_bin: np.ndarray, gt_bin: np.ndarray) -> tuple[int, int, int, int]:
    pred = pred_bin.astype(bool)
    gt = gt_bin.astype(bool)
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    tn = int((~pred & ~gt).sum())
    return tp, fp, fn, tn


def iou_dice_pr(pred_bin: np.ndarray, gt_bin: np.ndarray) -> dict[str, float]:
    tp, fp, fn, _tn = confusion(pred_bin, gt_bin)
    iou = tp / (tp + fp + fn + 1e-6)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
    }


def boundary_f1_multi(
    pred_bin: np.ndarray, gt_bin: np.ndarray, tolerances: Sequence[int] = (2,)
) -> list[float]:
    """Boundary F1 at each tolerance: an outline pixel counts when the other outline has a pixel within it.

    The tolerance is a chessboard chamfer distance rather than a dilation by a ``(2 * tolerance + 1)`` square,
    which gives the same answer but stays cheap at tolerances of 10 or 20 px. The outlines and the two distance
    maps are computed once, inside the bounding box of both masks grown by the largest tolerance, and shared by
    all tolerances.
    """
    pred = np.asarray(pred_bin).astype(bool)
    gt = np.asarray(gt_bin).astype(bool)
    both = pred | gt
    if not both.any():
        return [0.0 for _ in tolerances]
    reach = int(max(tolerances)) + 2
    rows = np.flatnonzero(both.any(axis=1))
    cols = np.flatnonzero(both.any(axis=0))
    window = (
        slice(max(int(rows[0]) - reach, 0), int(rows[-1]) + 1 + reach),
        slice(max(int(cols[0]) - reach, 0), int(cols[-1]) + 1 + reach),
    )
    pred_b = find_boundaries(pred[window], mode="outer")
    gt_b = find_boundaries(gt[window], mode="outer")
    to_gt = distance_transform_cdt(~gt_b, metric="chessboard") if gt_b.any() else None
    to_pred = distance_transform_cdt(~pred_b, metric="chessboard") if pred_b.any() else None
    scores = []
    for tolerance in tolerances:
        near_gt = (to_gt <= tolerance) if to_gt is not None else np.zeros_like(gt_b)
        near_pred = (to_pred <= tolerance) if to_pred is not None else np.zeros_like(pred_b)
        prec = (pred_b & near_gt).sum() / (pred_b.sum() + 1e-6)
        rec = (gt_b & near_pred).sum() / (gt_b.sum() + 1e-6)
        scores.append(float(2 * prec * rec / (prec + rec + 1e-6)))
    return scores


def boundary_f1(pred_bin: np.ndarray, gt_bin: np.ndarray, tolerance: int = 2) -> float:
    """Boundary F1 at one tolerance (see :func:`boundary_f1_multi`)."""
    return boundary_f1_multi(pred_bin, gt_bin, (tolerance,))[0]


def tolerant_iou(
    pred_bin: np.ndarray, gt_bin: np.ndarray, weight: np.ndarray | None = None, min_weight: float = 0.5
) -> float:
    """IoU that ignores the edge band: only pixels whose loss weight is at least ``min_weight`` count.

    ``weight`` is the per-pixel edge weight of the ground truth (see ``EMSegDataset``); without it this is the
    plain IoU. A few pixels of disagreement with the traced line, where hand tremor lives, then cost nothing.
    """
    if weight is None:
        return iou_dice_pr(pred_bin, gt_bin)["iou"]
    keep = np.asarray(weight, dtype=np.float32) >= min_weight
    if not keep.any():
        return iou_dice_pr(pred_bin, gt_bin)["iou"]
    return iou_dice_pr(np.asarray(pred_bin)[keep], np.asarray(gt_bin)[keep])["iou"]


def mean_iou_by_volume(samples: Iterable[tuple[str, float]]) -> tuple[float, dict[str, float]]:
    """Overall mean IoU and the mean IoU for each volume name.

    Names are the ones on the validation examples (the training-data folder,
    unless that example's manifest sets ``volume``). A blank name stays in the
    overall mean and does not get its own entry. An empty sample list returns
    ``0.0`` and an empty map.
    """
    rows = [(str(volume), float(iou)) for volume, iou in samples]
    if not rows:
        return 0.0, {}
    buckets: dict[str, list[float]] = {}
    for volume, iou in rows:
        name = volume.strip()
        if not name:
            continue
        buckets.setdefault(name, []).append(iou)
    by_volume = {name: float(np.mean(buckets[name])) for name in sorted(buckets)}
    overall = float(np.mean([iou for _volume, iou in rows]))
    return overall, by_volume


def summarize_records(records: Iterable[dict]) -> dict[str, object]:
    recs = list(records)
    if not recs:
        return {"n": 0}
    keys = ("iou", "dice", "precision", "recall", "bf")
    overall = {k: float(np.mean([r[k] for r in recs])) for k in keys}
    overall["n"] = len(recs)

    def _group(field: str) -> dict[str, dict[str, float]]:
        buckets: dict[str, list[dict]] = defaultdict(list)
        for rec in recs:
            buckets[str(rec.get(field, ""))].append(rec)
        out: dict[str, dict[str, float]] = {}
        for name, group in sorted(buckets.items()):
            out[name] = {k: float(np.mean([g[k] for g in group])) for k in keys}
            out[name]["n"] = float(len(group))
        return out

    return {
        "overall": overall,
        "by_volume": _group("volume"),
        "by_category": _group("category_name"),
        "by_downsample": _group("downsample"),
    }
