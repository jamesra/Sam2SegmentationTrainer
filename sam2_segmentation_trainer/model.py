"""SAM2 image fine-tune helpers: build, freeze, prompt coords, forward, losses."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from sam2.build_sam import build_sam2

def build_em_sam2(
    config_file: str,
    checkpoint: str,
    device: torch.device,
    *,
    train: bool,
):
    mode = "train" if train else "eval"
    model = build_sam2(
        config_file,
        checkpoint,
        device=str(device),
        mode=mode,
        apply_postprocessing=False,
    )
    return model


def configure_trainable_params(model, freeze_mode: str = "none") -> tuple[int, int]:
    if freeze_mode == "none":
        for param in model.parameters():
            param.requires_grad = True
        for param in model.sam_prompt_encoder.parameters():
            param.requires_grad = False
    elif freeze_mode == "partial":
        for param in model.parameters():
            param.requires_grad = False
        for param in model.sam_mask_decoder.parameters():
            param.requires_grad = True
        trunk = model.image_encoder.trunk
        for block_idx in _block_indices_for_stages(trunk, (2, 3)):
            for param in trunk.blocks[block_idx].parameters():
                param.requires_grad = True
    else:
        raise ValueError(f"Unknown freeze_mode: {freeze_mode}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable: {trainable / 1e6:.1f}M / {total / 1e6:.1f}M params "
        f"({100 * trainable / total:.0f}%) [{freeze_mode}]"
    )
    return trainable, total


def enable_activation_checkpointing(model) -> int:
    """Recompute Hiera blocks in backward to fit 1024 / batch 4 on 32GB."""
    trunk = model.image_encoder.trunk
    n_enabled = 0
    for block in trunk.blocks:
        if getattr(block, "_em_activation_ckpt", False):
            continue
        orig_forward = block.forward

        def _forward(x, _orig=orig_forward):
            return torch.utils.checkpoint.checkpoint(_orig, x, use_reentrant=False)

        block.forward = _forward
        block._em_activation_ckpt = True
        n_enabled += 1
    return n_enabled


def _block_indices_for_stages(trunk, stages: Sequence[int]) -> list[int]:
    # Hiera `blocks` is a flat ModuleList; `stage_ends` is the last index of each stage.
    indices: list[int] = []
    prev = -1
    for stage_i, end in enumerate(trunk.stage_ends):
        if stage_i in stages:
            indices.extend(range(prev + 1, end + 1))
        prev = end
    return indices


def param_groups(model, base_lr: float, encoder_lr_factor: float) -> list[dict]:
    encoder_ids = {id(p) for p in model.image_encoder.parameters()}
    encoder_params = [
        p
        for p in model.parameters()
        if p.requires_grad and id(p) in encoder_ids
    ]
    other_params = [
        p
        for p in model.parameters()
        if p.requires_grad and id(p) not in encoder_ids
    ]
    groups = []
    if other_params:
        groups.append({"params": other_params, "lr": base_lr})
    if encoder_params:
        groups.append({"params": encoder_params, "lr": base_lr * encoder_lr_factor})
    if not groups:
        raise RuntimeError("no trainable parameters")
    return groups


def transform_prompt_coords(
    coords: torch.Tensor,
    orig_hw: tuple[int, int],
    image_size: int,
) -> torch.Tensor:
    """Pixel coords in orig_hw -> SAM2 prompt-encoder pixels at `image_size`."""
    height, width = orig_hw
    scaled = coords.clone()
    scaled[..., 0] = scaled[..., 0] / float(width)
    scaled[..., 1] = scaled[..., 1] / float(height)
    return scaled * float(image_size)


def predict_masks(
    model,
    images: torch.Tensor,
    point_coords: torch.Tensor,
    point_labels: torch.Tensor,
    *,
    image_size: int,
    orig_hw: tuple[int, int] | None = None,
):
    if orig_hw is None:
        orig_hw = (image_size, image_size)
    coords = transform_prompt_coords(point_coords, orig_hw, image_size)
    labels = point_labels
    if labels.dtype != torch.int32:
        labels = labels.to(dtype=torch.int32)

    backbone_out = model.forward_image(images)
    _, vision_feats, _, feat_sizes = model._prepare_backbone_features(backbone_out)
    if model.directly_add_no_mem_embed:
        vision_feats[-1] = vision_feats[-1] + model.no_mem_embed

    batch = images.shape[0]
    feats = [
        feat.permute(1, 2, 0).view(batch, -1, *feat_size)
        for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])
    ][::-1]
    image_embed = feats[-1]
    high_res_features = feats[:-1]

    sparse_embed, dense_embed = model.sam_prompt_encoder(
        points=(coords, labels),
        boxes=None,
        masks=None,
    )
    low_res_masks, iou_preds, _, _ = model.sam_mask_decoder(
        image_embeddings=image_embed,
        image_pe=model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embed,
        dense_prompt_embeddings=dense_embed,
        multimask_output=False,
        repeat_image=False,
        high_res_features=high_res_features,
    )
    pred_logits = F.interpolate(
        low_res_masks,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )[:, 0]
    return pred_logits, iou_preds


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-sample dice loss (shape ``[B]``)."""
    pred_s = torch.sigmoid(pred).flatten(1)
    target_f = target.float().flatten(1)
    inter = (pred_s * target_f).sum(1)
    return 1 - (2 * inter + eps) / (pred_s.sum(1) + target_f.sum(1) + eps)


def focal_loss_per_sample(
    pred: torch.Tensor, target: torch.Tensor, gamma: float = 2.0
) -> torch.Tensor:
    """Per-sample mean focal loss over spatial dims (shape ``[B]``)."""
    bce = F.binary_cross_entropy_with_logits(
        pred, target.float(), reduction="none"
    )
    pt = torch.exp(-bce)
    return ((1 - pt) ** gamma * bce).flatten(1).mean(1)


def focal_loss(
    pred: torch.Tensor, target: torch.Tensor, gamma: float = 2.0
) -> torch.Tensor:
    return focal_loss_per_sample(pred, target, gamma=gamma).mean()


def iou_prediction_loss_per_sample(
    pred: torch.Tensor, target: torch.Tensor, iou_preds: torch.Tensor
) -> torch.Tensor:
    """Per-sample IoU-head MSE (shape ``[B]``)."""
    with torch.no_grad():
        pred_bin = torch.sigmoid(pred) > 0.5
        target_b = target.bool()
        inter = (pred_bin & target_b).flatten(1).sum(1).float()
        union = (pred_bin | target_b).flatten(1).sum(1).float()
        gt_iou = inter / (union + 1e-6)
    iou_head = iou_preds[:, 0] if iou_preds.ndim > 1 else iou_preds
    return (iou_head.float() - gt_iou).pow(2)


def iou_prediction_loss(
    pred: torch.Tensor, target: torch.Tensor, iou_preds: torch.Tensor
) -> torch.Tensor:
    return iou_prediction_loss_per_sample(pred, target, iou_preds).mean()


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    iou_preds: torch.Tensor,
    *,
    focal_weight: float,
    dice_weight: float,
    iou_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the batch loss and detached GPU stats. Callers sync when they log."""
    loss_focal_ps = focal_loss_per_sample(pred, target)
    loss_dice_ps = dice_loss(pred, target)
    loss_iou_ps = iou_prediction_loss_per_sample(pred, target, iou_preds)
    sample_losses = (
        float(focal_weight) * loss_focal_ps
        + float(dice_weight) * loss_dice_ps
        + float(iou_weight) * loss_iou_ps
    )
    loss = sample_losses.mean()
    stats = {
        "loss": loss.detach(),
        "loss_focal": loss_focal_ps.mean().detach(),
        "loss_dice": loss_dice_ps.mean().detach(),
        "loss_iou": loss_iou_ps.mean().detach(),
        "sample_losses": sample_losses.detach(),
    }
    return loss, stats
