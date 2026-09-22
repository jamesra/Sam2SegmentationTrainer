"""Segmentation metrics for EM SAM2 eval."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
from scipy.ndimage import binary_dilation
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


def boundary_f1(pred_bin: np.ndarray, gt_bin: np.ndarray, tolerance: int = 2) -> float:
    pred_b = find_boundaries(pred_bin.astype(bool), mode="outer")
    gt_b = find_boundaries(gt_bin.astype(bool), mode="outer")
    struct = np.ones((2 * tolerance + 1, 2 * tolerance + 1))
    pred_d = binary_dilation(pred_b, structure=struct)
    gt_d = binary_dilation(gt_b, structure=struct)
    prec = (pred_b & gt_d).sum() / (pred_b.sum() + 1e-6)
    rec = (gt_b & pred_d).sum() / (gt_b.sum() + 1e-6)
    return float(2 * prec * rec / (prec + rec + 1e-6))


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
