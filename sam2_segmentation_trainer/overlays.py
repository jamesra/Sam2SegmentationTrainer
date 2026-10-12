"""Input / ground-truth / prediction strips for eval grids and TensorBoard.

The prediction panel keeps false positives red and false negatives blue so a
quick look shows hallucinations in empty regions and missed cells, which a
single IoU number does not.
"""

from __future__ import annotations

import numpy as np
import torch

from sam2_segmentation_trainer.data.dataset import IMAGENET_MEAN, IMAGENET_STD


def denorm_preview(image_chw: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalization. Returns HWC uint8 RGB."""
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    rgb = (image_chw.detach().cpu().float() * std + mean).clamp(0, 1)
    return (rgb.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)


def overlay_panel(rgb: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """HWC uint8 strip: input, ground truth in green, prediction.

    ``gt`` and ``pred`` are binary masks at the image resolution. On the
    prediction panel, pixels only in the prediction are red and pixels only
    in the ground truth are blue.
    """
    base = np.array(rgb, copy=True)
    gt_panel = base.copy()
    pred_panel = base.copy()
    gt_m = np.asarray(gt).astype(bool)
    pred_m = np.asarray(pred).astype(bool)
    gt_panel[gt_m] = (0.6 * gt_panel[gt_m] + np.array([0, 180, 0]) * 0.4).astype(np.uint8)
    pred_panel[pred_m] = (0.6 * pred_panel[pred_m] + np.array([0, 180, 0]) * 0.4).astype(np.uint8)
    false_pos = pred_m & ~gt_m
    false_neg = gt_m & ~pred_m
    pred_panel[false_pos] = (0.4 * pred_panel[false_pos] + np.array([220, 30, 30]) * 0.6).astype(
        np.uint8
    )
    pred_panel[false_neg] = (0.4 * pred_panel[false_neg] + np.array([30, 80, 220]) * 0.6).astype(
        np.uint8
    )
    return np.concatenate([base, gt_panel, pred_panel], axis=1)
