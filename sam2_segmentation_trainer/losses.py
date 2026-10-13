"""Loss terms that need only torch, kept apart from ``model`` so tests do not need SAM2 installed."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mask_containment_loss_per_sample(pred_logits: torch.Tensor, foreground: torch.Tensor) -> torch.Tensor:
    """Per-sample hinge penalty for predicting background where a mask prompt said foreground (shape ``[B]``).

    ``pred_logits`` is ``[B, H, W]`` and ``foreground`` is ``[B, H, W]`` (non-zero where the dense mask prompt
    was +1). The penalty is the mean of ``relu(-logit)`` over the foreground pixels, so it is zero once every
    prompted pixel is predicted object, and the model is free to grow beyond the mask. A sample with no
    foreground contributes 0.

    A mask prompt means "everything inside is foreground, and the engine should grow beyond it", so this one
    sided term enforces that meaning. There is deliberately no counterpart for boxes: a box is only a hint
    that the object is most likely inside it (see ``.claude/rules/sam2-prompt-semantics.md``).
    """
    fg = (foreground != 0).to(dtype=pred_logits.dtype)
    violation = F.relu(-pred_logits) * fg
    return violation.flatten(1).sum(1) / fg.flatten(1).sum(1).clamp_min(1.0)


def containment_warmup_scale(epoch: int, step: int, steps_per_epoch: int, warmup_epochs: float) -> float:
    """Linear 0..1 ramp for containment weights over ``warmup_epochs``, advancing every step.

    Uses the fractional epoch ``epoch + step / steps_per_epoch`` so the first epoch is not entirely at zero.
    ``warmup_epochs <= 0`` means no warmup.
    """
    if warmup_epochs <= 0:
        return 1.0
    progress = float(epoch) + float(step) / float(max(1, steps_per_epoch))
    return max(0.0, min(1.0, progress / float(warmup_epochs)))
