"""SAM2 image fine-tune helpers: build, freeze, prompt coords, forward, losses."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from sam2.build_sam import build_sam2

from sam2_segmentation_trainer.losses import (
    mask_containment_loss_per_sample,
    tolerant_iou_target,
    weighted_dice_loss,
    weighted_focal_loss_per_sample,
)

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


def configure_trainable_params(
    model, freeze_mode: str = "none", unfreeze_mask_cnn: bool = False
) -> tuple[int, int]:
    """Set ``requires_grad`` for the chosen ``freeze_mode`` and return (trainable, total) parameter counts.

    The prompt encoder stays frozen except, when ``unfreeze_mask_cnn`` is set, its ``mask_downscaling``
    network (about 4.7K parameters), which turns a dense mask prompt into an embedding and has to learn the
    -1/0/+1 mask semantics of mixed-prompt training. Those parameters join the non-encoder optimizer group.
    Changing this setting changes the optimizer's parameter list, so a run cannot resume across it.
    """
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

    if unfreeze_mask_cnn:
        for param in model.sam_prompt_encoder.mask_downscaling.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable: {trainable / 1e6:.1f}M / {total / 1e6:.1f}M params "
        f"({100 * trainable / total:.0f}%) [{freeze_mode}]"
    )
    return trainable, total


def checkpointed_block_indices(n_blocks: int, fraction: float) -> list[int]:
    """Which of ``n_blocks`` blocks to checkpoint for a ``fraction`` in [0, 1], spread evenly through the trunk.

    ``1.0`` is every block, ``0.5`` every other one, ``0`` none. Evenly spread keeps the saved activations from
    piling up in one stage.
    """
    count = int(round(max(0.0, min(1.0, float(fraction))) * n_blocks))
    if count <= 0:
        return []
    return sorted({int(i * n_blocks / count) for i in range(count)})


def enable_activation_checkpointing(model, fraction: float = 1.0) -> int:
    """Recompute a ``fraction`` of the Hiera blocks in backward to fit 1024 / batch 4 on 32GB (all by default)."""
    trunk = model.image_encoder.trunk
    n_enabled = 0
    chosen = set(checkpointed_block_indices(len(trunk.blocks), fraction))
    for index, block in enumerate(trunk.blocks):
        if index not in chosen:
            continue
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


def encode_images(model, images: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Run the image encoder once: returns ``(image_embed, high_res_features)`` for the mask decoder.

    Split from :func:`decode_prompts` so one batch can be decoded in several prompt groups (some samples
    with boxes or dense masks, some without) while sharing a single encoder pass.
    """
    backbone_out = model.forward_image(images)
    _, vision_feats, _, feat_sizes = model._prepare_backbone_features(backbone_out)
    if model.directly_add_no_mem_embed:
        vision_feats[-1] = vision_feats[-1] + model.no_mem_embed

    batch = images.shape[0]
    feats = [
        feat.permute(1, 2, 0).view(batch, -1, *feat_size)
        for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])
    ][::-1]
    return feats[-1], feats[:-1]


def decode_prompts(
    model,
    image_embed: torch.Tensor,
    high_res_features: list[torch.Tensor],
    point_coords: torch.Tensor | None,
    point_labels: torch.Tensor | None,
    *,
    image_size: int,
    orig_hw: tuple[int, int] | None = None,
    boxes: torch.Tensor | None = None,
    dense_masks: torch.Tensor | None = None,
    return_low_res: bool = False,
):
    """Prompt encoder + mask decoder for a batch that shares one prompt structure.

    ``point_coords`` is ``[B, N, 2]`` in ``orig_hw`` pixels with ``point_labels`` ``[B, N]`` (1 foreground,
    0 background, -1 padding that SAM2 ignores); ``N == 0`` or None means no points. ``boxes`` is ``[B, 4]``
    xyxy in ``orig_hw`` pixels, passed as ``boxes=`` (the encoder adds the corner embeddings itself, so no
    corner labels are built here). ``dense_masks`` is ``[B, 1, 256, 256]`` float ``mask_input`` logits. None
    selects the encoder's learned no-box / no-mask embedding, which an all-zero tensor would not.

    Returns ``(pred_logits [B, image_size, image_size], iou_preds [B, 1])``; with ``return_low_res`` a third
    value, the decoder's own low-resolution logits ``[B, h, w]``, which is what corrective-click rounds feed
    back as the next mask prompt.
    """
    if orig_hw is None:
        orig_hw = (image_size, image_size)
    points = None
    if point_coords is not None and point_coords.shape[1] > 0:
        coords = transform_prompt_coords(point_coords, orig_hw, image_size)
        labels = point_labels
        if labels.dtype != torch.int32:
            labels = labels.to(dtype=torch.int32)
        points = (coords, labels)
    box_prompt = None
    if boxes is not None:
        box_prompt = transform_prompt_coords(boxes.reshape(-1, 2, 2), orig_hw, image_size).reshape(-1, 4)
    if points is None and box_prompt is None:
        raise ValueError("decode_prompts needs at least one point or box")

    sparse_embed, dense_embed = model.sam_prompt_encoder(
        points=points,
        boxes=box_prompt,
        masks=dense_masks,
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
    if return_low_res:
        return pred_logits, iou_preds, low_res_masks[:, 0]
    return pred_logits, iou_preds


def predict_masks(
    model,
    images: torch.Tensor,
    point_coords: torch.Tensor,
    point_labels: torch.Tensor,
    *,
    image_size: int,
    orig_hw: tuple[int, int] | None = None,
):
    """Points-only prediction for a whole batch (validation, evaluation, benchmarks)."""
    image_embed, high_res = encode_images(model, images)
    return decode_prompts(
        model,
        image_embed,
        high_res,
        point_coords,
        point_labels,
        image_size=image_size,
        orig_hw=orig_hw,
    )


def predict_masks_mixed(
    model,
    images: torch.Tensor,
    point_coords: torch.Tensor,
    point_labels: torch.Tensor,
    boxes: Sequence[torch.Tensor | None],
    dense_masks: Sequence[torch.Tensor | None],
    *,
    image_size: int,
    orig_hw: tuple[int, int] | None = None,
    variant_index: torch.Tensor | None = None,
):
    """Prediction for a batch whose samples may have boxes and dense masks or not.

    The image encoder runs once. The prompt encoder takes a single ``boxes`` and ``masks`` tensor per call, so
    samples are grouped by (has box, has dense mask) and each group is decoded separately, then scattered
    back to batch order. ``boxes[i]`` is ``[4]`` xyxy and ``dense_masks[i]`` is ``[256, 256]`` or None.
    Returns ``(pred_logits [B, H, W], iou_preds [B, 1])`` like :func:`predict_masks`.

    With ``variant_index`` (``[N]`` long, the image each prompt variant belongs to) the prompt arguments
    describe ``N`` variants of the ``B`` images; the encoder still runs once per image and every variant reuses
    its image's embeddings. The result then has ``N`` rows.
    """
    image_embed, high_res = encode_images(model, images)
    return decode_grouped(
        model,
        image_embed,
        high_res,
        point_coords,
        point_labels,
        boxes,
        dense_masks,
        image_size=image_size,
        orig_hw=orig_hw,
        embed_index=variant_index,
    )


def decode_grouped(
    model,
    image_embed: torch.Tensor,
    high_res: list[torch.Tensor],
    point_coords: torch.Tensor,
    point_labels: torch.Tensor,
    boxes: Sequence[torch.Tensor | None],
    dense_masks: Sequence[torch.Tensor | None],
    *,
    image_size: int,
    orig_hw: tuple[int, int] | None = None,
    return_low_res: bool = False,
    embed_index: torch.Tensor | None = None,
):
    """Decode an already encoded batch whose samples may have boxes and dense masks or not.

    Samples are grouped by (has box, has dense mask) and each group is decoded separately, then scattered back
    to batch order. With ``return_low_res`` the result has a third value, the low-resolution logits
    ``[B, h, w]``. With ``embed_index`` (``[N]`` long) the ``N`` prompts are variants of the encoded images:
    prompt ``i`` is decoded against image ``embed_index[i]``, so one encoder pass serves all of them.
    """
    batch = image_embed.shape[0] if embed_index is None else int(embed_index.shape[0])
    groups: dict[tuple[bool, bool], list[int]] = {}
    for i in range(batch):
        groups.setdefault((boxes[i] is not None, dense_masks[i] is not None), []).append(i)

    device = image_embed.device
    pieces: list[tuple[list[int], torch.Tensor, torch.Tensor, torch.Tensor | None]] = []
    for (has_box, has_dense), members in groups.items():
        sel = torch.tensor(members, device=device, dtype=torch.long)
        group_boxes = (
            torch.stack([boxes[i] for i in members]).to(device, non_blocking=True) if has_box else None
        )
        group_dense = (
            torch.stack([dense_masks[i] for i in members]).unsqueeze(1).to(device, non_blocking=True)
            if has_dense
            else None
        )
        image_sel = sel if embed_index is None else embed_index[sel]
        decoded = decode_prompts(
            model,
            image_embed[image_sel],
            [feat[image_sel] for feat in high_res],
            point_coords[sel],
            point_labels[sel],
            image_size=image_size,
            orig_hw=orig_hw,
            boxes=group_boxes,
            dense_masks=group_dense,
            return_low_res=return_low_res,
        )
        pieces.append((members, decoded[0], decoded[1], decoded[2] if return_low_res else None))

    if len(pieces) == 1:
        _members, logits, iou, low = pieces[0]
        return (logits, iou, low) if return_low_res else (logits, iou)
    first_logits, first_iou, first_low = pieces[0][1], pieces[0][2], pieces[0][3]
    pred_logits = first_logits.new_empty((batch, *first_logits.shape[1:]))
    iou_preds = first_iou.new_empty((batch, *first_iou.shape[1:]))
    low_res = first_low.new_empty((batch, *first_low.shape[1:])) if return_low_res else None
    for members, logits, iou, low in pieces:
        index = torch.tensor(members, device=device, dtype=torch.long)
        pred_logits = pred_logits.index_copy(0, index, logits)
        iou_preds = iou_preds.index_copy(0, index, iou)
        if return_low_res:
            low_res = low_res.index_copy(0, index, low)
    return (pred_logits, iou_preds, low_res) if return_low_res else (pred_logits, iou_preds)


def dice_loss(
    pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None, eps: float = 1e-6
) -> torch.Tensor:
    """Per-sample dice loss (shape ``[B]``), optionally edge-weighted."""
    return weighted_dice_loss(pred, target, weight, eps)


def focal_loss_per_sample(
    pred: torch.Tensor, target: torch.Tensor, gamma: float = 2.0, weight: torch.Tensor | None = None
) -> torch.Tensor:
    """Per-sample focal loss (shape ``[B]``): the mean over pixels, or the weight-normalized sum with ``weight``."""
    return weighted_focal_loss_per_sample(pred, target, weight, gamma)


def focal_loss(
    pred: torch.Tensor, target: torch.Tensor, gamma: float = 2.0
) -> torch.Tensor:
    return focal_loss_per_sample(pred, target, gamma=gamma).mean()


def iou_prediction_loss_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    iou_preds: torch.Tensor,
    *,
    use_l1: bool = False,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-sample IoU-head loss (shape ``[B]``): squared error, or absolute error with ``use_l1`` (SAM 2).

    With an edge ``weight`` the target IoU ignores pixels whose weight is below 0.5, matching tolerant validation.
    """
    with torch.no_grad():
        gt_iou = tolerant_iou_target(pred, target, weight)
    iou_head = iou_preds[:, 0] if iou_preds.ndim > 1 else iou_preds
    error = iou_head.float() - gt_iou
    return error.abs() if use_l1 else error.pow(2)


def iou_prediction_loss(
    pred: torch.Tensor, target: torch.Tensor, iou_preds: torch.Tensor
) -> torch.Tensor:
    return iou_prediction_loss_per_sample(pred, target, iou_preds).mean()


def build_prompt_region(
    points: torch.Tensor, labels: torch.Tensor, shape: tuple[int, int], radius: float
) -> torch.Tensor:
    """Boolean disk mask (shape ``[B, H, W]``) around positive prompt points.

    ``points`` is ``[B, N, 2]`` xy pixel coords in the same space as ``shape``;
    ``labels`` is ``[B, N]`` with positive prompts marked ``> 0``.
    """
    height, width = shape
    device = points.device
    dtype = points.dtype
    ys = torch.arange(height, device=device, dtype=dtype).view(1, 1, height, 1)
    xs = torch.arange(width, device=device, dtype=dtype).view(1, 1, 1, width)
    px = points[..., 0].view(*points.shape[:2], 1, 1)
    py = points[..., 1].view(*points.shape[:2], 1, 1)
    dist2 = (xs - px) ** 2 + (ys - py) ** 2
    within = dist2 <= (float(radius) ** 2)
    positive = (labels > 0).view(*labels.shape, 1, 1)
    return (within & positive).any(dim=1)


def prompt_containment_loss_per_sample(
    pred: torch.Tensor,
    points: torch.Tensor,
    labels: torch.Tensor,
    radius: float,
) -> torch.Tensor:
    """Per-sample BCE pushing logits at prompted locations toward foreground (shape ``[B]``)."""
    region = build_prompt_region(points, labels, pred.shape[-2:], radius)
    region_f = region.to(dtype=pred.dtype)
    bce = F.binary_cross_entropy_with_logits(
        pred, torch.ones_like(pred), reduction="none"
    )
    denom = region_f.flatten(1).sum(1).clamp_min(1.0)
    return (bce * region_f).flatten(1).sum(1) / denom


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    iou_preds: torch.Tensor,
    *,
    focal_weight: float,
    dice_weight: float,
    iou_weight: float,
    points: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    containment_weight: float = 0.0,
    containment_radius: float = 8.0,
    prompt_foreground: torch.Tensor | None = None,
    mask_contain_weight: float = 0.0,
    iou_use_l1: bool = False,
    edge_weight: torch.Tensor | None = None,
    variant_index: torch.Tensor | None = None,
    num_samples: int | None = None,
    sample_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the batch loss and detached GPU stats. Callers sync when they log.

    ``prompt_foreground`` (``[B, H, W]``, the trust of a seed at each pixel, 0 elsewhere) with a positive
    ``mask_contain_weight`` adds the one-sided mask containment penalty; the caller scales the weight by its
    warmup. ``stats`` then also has ``loss_mask_contain``.

    ``edge_weight`` (``[B, H, W]``, see :func:`sam2_segmentation_trainer.losses.edge_loss_weight`) down-weights
    pixels near the ground-truth edge in the focal, dice and IoU-head terms; the labels stay hard.

    With ``variant_index`` (``[N]`` long) the rows are prompt variants and the index names each one's sample.
    The batch loss averages the variants of a sample first, then the samples, so a sample with more variants
    does not weigh more. ``num_samples`` is the number of samples (the batch size) and every one of them has at
    least one variant; passing it avoids a GPU sync to find out. ``stats["sample_losses"]`` keeps one value per
    variant.

    ``sample_weight`` (``[B]``, or ``[N]`` without ``variant_index``) scales each sample's loss, for
    ``data.approved_weight``; ``stats["sample_losses"]`` stays unweighted.

    The terms are computed in fp32 whatever the autocast dtype of the predictions: the dice and containment sums
    run over a million pixels per row, which bf16 cannot add exactly.
    """
    pred = pred.float()
    iou_preds = iou_preds.float()
    if edge_weight is not None:
        edge_weight = edge_weight.float()
    loss_focal_ps = focal_loss_per_sample(pred, target, weight=edge_weight)
    loss_dice_ps = dice_loss(pred, target, edge_weight)
    loss_iou_ps = iou_prediction_loss_per_sample(
        pred, target, iou_preds, use_l1=iou_use_l1, weight=edge_weight
    )
    sample_losses = (
        float(focal_weight) * loss_focal_ps
        + float(dice_weight) * loss_dice_ps
        + float(iou_weight) * loss_iou_ps
    )
    stats = {
        "loss_focal": loss_focal_ps.mean().detach(),
        "loss_dice": loss_dice_ps.mean().detach(),
        "loss_iou": loss_iou_ps.mean().detach(),
    }
    if float(containment_weight) > 0.0 and points is not None and labels is not None:
        # "In this domain, prompts are ground truth": punish background predicted
        # at prompted locations so the model internalizes grow-outward behavior.
        loss_containment_ps = prompt_containment_loss_per_sample(
            pred, points, labels, containment_radius
        )
        sample_losses = sample_losses + float(containment_weight) * loss_containment_ps
        stats["loss_containment"] = loss_containment_ps.mean().detach()
    if float(mask_contain_weight) > 0.0 and prompt_foreground is not None:
        loss_mask_ps = mask_containment_loss_per_sample(pred, prompt_foreground)
        sample_losses = sample_losses + float(mask_contain_weight) * loss_mask_ps
        stats["loss_mask_contain"] = loss_mask_ps.mean().detach()
    weighted_losses = sample_losses
    if sample_weight is not None:
        row_weight = sample_weight if variant_index is None else sample_weight[variant_index]
        weighted_losses = sample_losses * row_weight.to(device=sample_losses.device, dtype=sample_losses.dtype)
    if variant_index is None:
        loss = weighted_losses.mean()
    else:
        count = int(num_samples) if num_samples is not None else int(variant_index.max().item()) + 1
        totals = weighted_losses.new_zeros(count).index_add(0, variant_index, weighted_losses)
        sizes = torch.bincount(variant_index, minlength=count).to(sample_losses.dtype)
        loss = (totals / sizes.clamp(min=1.0)).mean()
    stats["loss"] = loss.detach()
    stats["sample_losses"] = sample_losses.detach()
    return loss, stats
