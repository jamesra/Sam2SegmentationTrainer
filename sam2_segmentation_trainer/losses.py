"""Loss terms that need only torch, kept apart from ``model`` so tests do not need SAM2 installed."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mask_containment_loss_per_sample(pred_logits: torch.Tensor, foreground: torch.Tensor) -> torch.Tensor:
    """Per-sample hinge penalty for predicting background where a seed said foreground (shape ``[B]``).

    ``pred_logits`` is ``[B, H, W]`` and ``foreground`` is ``[B, H, W]``: the trust of the seed at each pixel,
    its edge confidence ``c(d)`` (a 0/1 mask works too, as plain weights of 1). The penalty is the
    weight-normalized mean of ``relu(-logit)``, so it is zero once every trusted pixel is predicted object, and
    the model is free to grow beyond the seed. The weights make it act on the seed's core and barely at its
    traced edge. A sample with no foreground contributes 0.

    Only seeds are trusted. This is never applied to a mask to correct or a near-correct mask, which may be
    wrong. There is deliberately no counterpart for boxes: a box is only a hint that the object is most likely
    inside it (see ``.claude/rules/sam2-prompt-semantics.md``).
    """
    weight = foreground.to(dtype=pred_logits.dtype)
    violation = F.relu(-pred_logits) * weight
    return violation.flatten(1).sum(1) / weight.flatten(1).sum(1).clamp_min(1.0)


def edge_loss_weight(distance: torch.Tensor, tau: float, w_min: float) -> torch.Tensor:
    """Per-pixel loss weight ``max(w_min, 1 - exp(-d / tau))`` from the distance to the ground-truth edge.

    Labels stay hard 0/1; down-weighting pixels near the traced line makes a few pixels of disagreement nearly
    free there, so the image decides where the membrane is, while supervision elsewhere stays at full strength.
    ``tau <= 0`` gives weights of 1 (no edge weighting).
    """
    if tau <= 0.0:
        return torch.ones_like(distance, dtype=torch.float32)
    weight = 1.0 - torch.exp(-distance.float() / float(tau))
    return weight.clamp_min(float(w_min))


def weighted_focal_loss_per_sample(
    pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None, gamma: float = 2.0
) -> torch.Tensor:
    """Per-sample focal loss (shape ``[B]``), each pixel scaled by ``weight`` and normalized by the weight sum."""
    bce = F.binary_cross_entropy_with_logits(pred, target.float(), reduction="none")
    pt = torch.exp(-bce)
    loss = (1 - pt) ** gamma * bce
    if weight is None:
        return loss.flatten(1).mean(1)
    weight = weight.to(dtype=loss.dtype)
    return (loss * weight).flatten(1).sum(1) / weight.flatten(1).sum(1).clamp_min(1e-6)


def weighted_dice_loss(
    pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None, eps: float = 1e-6
) -> torch.Tensor:
    """Per-sample soft dice loss (shape ``[B]``) with sums of ``w*p*g``, ``w*p`` and ``w*g``."""
    prob = torch.sigmoid(pred).flatten(1)
    truth = target.float().flatten(1)
    if weight is None:
        w = torch.ones_like(prob)
    else:
        w = weight.to(dtype=prob.dtype).flatten(1)
    inter = (w * prob * truth).sum(1)
    return 1 - (2 * inter + eps) / ((w * prob).sum(1) + (w * truth).sum(1) + eps)


def tolerant_iou_target(
    pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None, *, min_weight: float = 0.5
) -> torch.Tensor:
    """IoU (shape ``[B]``) of the thresholded prediction over pixels with ``weight >= min_weight``.

    The IoU-head regression target, matching tolerant validation: pixels in the edge band are ignored.
    With no ``weight`` it is the plain IoU.
    """
    pred_bin = torch.sigmoid(pred) > 0.5
    target_b = target.bool()
    if weight is not None:
        keep = weight >= min_weight
        pred_bin, target_b = pred_bin & keep, target_b & keep
    inter = (pred_bin & target_b).flatten(1).sum(1).float()
    union = (pred_bin | target_b).flatten(1).sum(1).float()
    return inter / (union + 1e-6)


def containment_warmup_scale(epoch: int, step: int, steps_per_epoch: int, warmup_epochs: float) -> float:
    """Linear 0..1 ramp for containment weights over ``warmup_epochs``, advancing every step.

    Uses the fractional epoch ``epoch + step / steps_per_epoch`` so the first epoch is not entirely at zero.
    ``warmup_epochs <= 0`` means no warmup.
    """
    if warmup_epochs <= 0:
        return 1.0
    progress = float(epoch) + float(step) / float(max(1, steps_per_epoch))
    return max(0.0, min(1.0, progress / float(warmup_epochs)))
