"""Prompt construction for mixed-prompt SAM2 training (points, boxes, dense masks, negative context).

Everything here works on numpy arrays that are already augmented, in tile pixel space with the origin at the
top-left (x right, y down), so it needs neither torch nor SAM2 and can be tested alone.

Prompt meanings follow ``.claude/rules/sam2-prompt-semantics.md``: a box is only a hint that the object is most
likely inside, so no loss here forces predictions into it; a dense mask is a strong per-pixel statement
("everything inside is foreground, or background for the -1 regions") that the engine may grow beyond.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Mapping

import numpy as np
from scipy.ndimage import distance_transform_edt

from annotation_crops.negatives import (
    KIND_CIRCLE,
    NegativeRegion,
    combine_signed,
    stamp_negative_regions,
)

LOW_RES = 256
"""Side of SAM2's ``mask_input`` for a 1024 image (4x the 64x64 image embedding)."""

MASK_PROMPT_LOGIT = 10.0
"""Logit a signed mask value of +/-1 is fed to the prompt encoder as.

SAM2's ``mask_downscaling`` network was trained on the low-resolution *logits* of earlier predictions, and the
segmentation server's ``_as_sam2_mask_input`` passes such logits clipped to +/-20. So +1 becomes
``+MASK_PROMPT_LOGIT``, -1 becomes ``-MASK_PROMPT_LOGIT`` and 0 (unknown) stays 0. Whatever later turns a 1-bit
foreground or background mask Feature into ``mask_input`` at inference must use these same values, or the
trained mask network sees a different scale than it was trained on.
"""

_MAX_CANDIDATES = 20000
"""Interior pixels considered when placing points; larger sets are randomly thinned to keep this cheap."""


class PromptType(str, Enum):
    """Prompt families a training sample can draw. The value is the name logged and stored in metadata."""

    POINTS = "points"
    BOX = "box"
    BOX_POINTS = "box_points"
    CIRCLE_MASK = "circle_mask"
    FULL_MASK = "full_mask"
    NEGATIVE_ONLY = "negative_only"


@dataclass(frozen=True, slots=True)
class PromptConfig:
    """Sampling settings from the ``prompt:`` and ``negative:`` sections of the training config."""

    prob_points_only: float = 0.10
    prob_box_only: float = 0.05
    prob_box_points: float = 0.05
    prob_circle_mask: float = 0.45
    prob_full_mask: float = 0.30
    prob_negative_only: float = 0.05
    box_jitter_frac: float = 0.15
    box_points_max: int = 3
    min_edge_px: float = 15.0
    line_frac_min: float = 0.40
    line_frac_max: float = 0.90
    bg_point_prob_circle: float = 0.25
    bg_point_prob_polygon: float = 0.15
    cache_size: int = 64
    allow_circle_targets: bool = False
    circle_target_fraction: float = 0.05

    def probabilities(self) -> dict[PromptType, float]:
        """Draw probability of each prompt type, before availability is considered."""
        return {
            PromptType.POINTS: self.prob_points_only,
            PromptType.BOX: self.prob_box_only,
            PromptType.BOX_POINTS: self.prob_box_points,
            PromptType.CIRCLE_MASK: self.prob_circle_mask,
            PromptType.FULL_MASK: self.prob_full_mask,
            PromptType.NEGATIVE_ONLY: self.prob_negative_only,
        }

    def validate(self) -> None:
        """Raise ValueError when probabilities are negative or do not sum to 1, or a range is inverted."""
        probs = self.probabilities()
        if any(p < 0.0 for p in probs.values()):
            raise ValueError(f"prompt probabilities must be >= 0: {probs}")
        total = sum(probs.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"prompt probabilities must sum to 1.0, got {total:.6f}: {probs}")
        if not 0.0 <= self.line_frac_min <= self.line_frac_max <= 1.0:
            raise ValueError("prompt.line_frac_min/line_frac_max must satisfy 0 <= min <= max <= 1")
        if self.box_points_max < 1:
            raise ValueError("prompt.box_points_max must be >= 1")
        for name in ("bg_point_prob_circle", "bg_point_prob_polygon", "circle_target_fraction"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"negative.{name} must be within [0, 1]")

    @classmethod
    def from_cfg(cls, cfg: Mapping[str, Any]) -> "PromptConfig":
        """Build from a config mapping that may have ``prompt`` and ``negative`` sections (OmegaConf works).

        Missing keys keep their defaults, so configs written before these sections existed still load.
        The result is validated.
        """
        defaults = cls()
        merged: dict[str, Any] = {}
        for section in ("prompt", "negative"):
            values = cfg.get(section) if hasattr(cfg, "get") else None
            if values:
                merged.update({key: values[key] for key in values})
        known = {field.name for field in fields(cls)}
        kwargs: dict[str, Any] = {}
        for name in known:
            if name not in merged:
                continue
            default = getattr(defaults, name)
            raw = merged[name]
            kwargs[name] = bool(raw) if isinstance(default, bool) else type(default)(raw)
        config = cls(**kwargs)
        config.validate()
        return config


@dataclass(eq=False)
class PromptSample:
    """The prompt for one training sample.

    ``point_coords``/``point_labels`` are foreground (1) and background (0) points, possibly none.
    ``box`` is (x1, y1, x2, y2) or None. ``dense_logits`` is the 256x256 float mask prompt (see
    :data:`MASK_PROMPT_LOGIT`) or None, meaning "use the learned no-mask embedding".
    ``foreground_region`` is the full-resolution +1 area of the dense mask (after negatives are carved out) for
    the mask-containment loss, or None when the dense mask has no foreground.
    """

    prompt_type: PromptType
    point_coords: np.ndarray
    point_labels: np.ndarray
    box: np.ndarray | None = None
    dense_logits: np.ndarray | None = None
    foreground_region: np.ndarray | None = None


def benford_count(rng: np.random.Generator, max_count: int = 9) -> int:
    """Draw 1..max_count with P(d) proportional to log10(1 + 1/d), so one point is about 30% of draws."""
    digits = np.arange(1, max_count + 1)
    weights = np.log10(1.0 + 1.0 / digits)
    return int(rng.choice(digits, p=weights / weights.sum()))


def sample_points_in_mask(
    mask: np.ndarray,
    count: int,
    rng: np.random.Generator,
    *,
    min_edge_px: float,
    line_frac_min: float,
    line_frac_max: float,
) -> np.ndarray:
    """Up to ``count`` (x, y) pixel points spread through the interior of a boolean mask.

    The first point is the deepest interior pixel (maximum distance transform). Each later point is placed a
    random fraction (``line_frac_min``..``line_frac_max``) of the way from the nearest chosen point toward the
    pixel farthest from all chosen points. Pixels within ``min_edge_px`` of the boundary are avoided; a mask too
    thin for that uses half its own maximum depth instead. Returns float32 of shape (n, 2), n <= count, empty
    for an empty mask.
    """
    if count <= 0 or not mask.any():
        return np.zeros((0, 2), dtype=np.float32)
    depth = distance_transform_edt(mask)
    peak = float(depth.max())
    threshold = min(float(min_edge_px), peak * 0.5)
    candidates = np.argwhere(depth >= max(threshold, 0.5))[:, ::-1].astype(np.float32)
    if candidates.shape[0] > _MAX_CANDIDATES:
        candidates = candidates[rng.choice(candidates.shape[0], _MAX_CANDIDATES, replace=False)]
    first_y, first_x = np.unravel_index(int(np.argmax(depth)), depth.shape)
    points: list[np.ndarray] = [np.array([first_x, first_y], dtype=np.float32)]
    for _ in range(count - 1):
        chosen = np.stack(points)
        gaps = np.linalg.norm(candidates[:, None, :] - chosen[None, :, :], axis=2)
        nearest_gap = gaps.min(axis=1)
        far_index = int(np.argmax(nearest_gap))
        if nearest_gap[far_index] < 1.0:
            break
        far = candidates[far_index]
        anchor = chosen[int(np.argmin(np.linalg.norm(chosen - far, axis=1)))]
        fraction = float(rng.uniform(line_frac_min, line_frac_max))
        placed = anchor + fraction * (far - anchor)
        px, py = int(round(float(placed[0]))), int(round(float(placed[1])))
        inside = 0 <= py < mask.shape[0] and 0 <= px < mask.shape[1] and bool(mask[py, px])
        points.append(placed.astype(np.float32) if inside else far.copy())
    return np.stack(points).astype(np.float32)


def jittered_box(
    mask: np.ndarray,
    rng: np.random.Generator,
    jitter_frac: float,
) -> np.ndarray | None:
    """Tight box of a boolean mask with each edge pushed out by ``uniform(0, jitter_frac * edge length)``.

    Returns float32 (x1, y1, x2, y2) clamped to the image, or None for an empty mask. The object stays inside
    the box, matching "most likely inside": the jitter only makes the box imprecise.
    """
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    height, width = mask.shape
    x1, x2 = float(xs.min()), float(xs.max() + 1)
    y1, y2 = float(ys.min()), float(ys.max() + 1)
    box_w, box_h = x2 - x1, y2 - y1
    x1 -= float(rng.uniform(0.0, jitter_frac * box_w))
    x2 += float(rng.uniform(0.0, jitter_frac * box_w))
    y1 -= float(rng.uniform(0.0, jitter_frac * box_h))
    y2 += float(rng.uniform(0.0, jitter_frac * box_h))
    return np.array(
        [max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2)], dtype=np.float32
    )


def nearest_resize(array: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbor resize of a 2-D array to ``size`` x ``size``.

    Samples the source pixel under each output pixel center, so a signed -1/0/+1 canvas keeps exactly those
    values; averaging would invent fractional values that mix foreground and background.
    """
    height, width = array.shape
    rows = np.minimum(((np.arange(size) + 0.5) * height / size).astype(np.int64), height - 1)
    cols = np.minimum(((np.arange(size) + 0.5) * width / size).astype(np.int64), width - 1)
    return array[np.ix_(rows, cols)]


def to_low_res_logits(signed: np.ndarray) -> np.ndarray:
    """Signed int8 canvas -> 256x256 float32 ``mask_input`` (see :data:`MASK_PROMPT_LOGIT`)."""
    return nearest_resize(signed, LOW_RES).astype(np.float32) * np.float32(MASK_PROMPT_LOGIT)


def choose_prompt_type(
    cfg: PromptConfig,
    rng: np.random.Generator,
    *,
    can_circle_mask: bool,
    has_negatives: bool,
) -> PromptType:
    """Draw a prompt type, dropping types the sample cannot support and renormalizing the rest.

    ``circle_mask`` needs a circle for the target whose center is in the tile; ``negative_only`` needs at least
    one other annotation. Their probability mass moves to the remaining types in proportion. If nothing is
    left the sample gets a full mask prompt.
    """
    weights = dict(cfg.probabilities())
    if not can_circle_mask:
        weights[PromptType.CIRCLE_MASK] = 0.0
    if not has_negatives:
        weights[PromptType.NEGATIVE_ONLY] = 0.0
    total = sum(weights.values())
    if total <= 0.0:
        return PromptType.FULL_MASK
    types = list(weights)
    return types[int(rng.choice(len(types), p=np.array([weights[t] for t in types]) / total))]


def _background_points(
    regions: list[NegativeRegion],
    cfg: PromptConfig,
    rng: np.random.Generator,
    *,
    force_one: bool,
) -> np.ndarray:
    """Background points from the negative regions.

    Each region is rolled independently (circle regions at ``bg_point_prob_circle``, polygon regions at
    ``bg_point_prob_polygon``) and, when it wins, yields a Benford-distributed 1..9 points. ``force_one``
    guarantees at least one region contributes, for prompts that have no foreground point.
    """
    chosen: list[NegativeRegion] = [
        region
        for region in regions
        if rng.random() < (cfg.bg_point_prob_circle if region.kind == KIND_CIRCLE else cfg.bg_point_prob_polygon)
    ]
    if force_one and not chosen and regions:
        chosen = [regions[int(rng.integers(len(regions)))]]
    sets = [
        sample_points_in_mask(
            region.mask,
            benford_count(rng),
            rng,
            min_edge_px=cfg.min_edge_px,
            line_frac_min=cfg.line_frac_min,
            line_frac_max=cfg.line_frac_max,
        )
        for region in chosen
    ]
    sets = [points for points in sets if points.shape[0]]
    return np.concatenate(sets) if sets else np.zeros((0, 2), dtype=np.float32)


def build_prompt(
    prompt_type: PromptType,
    *,
    target_mask: np.ndarray,
    circle_mask: np.ndarray | None,
    regions: list[NegativeRegion],
    cfg: PromptConfig,
    rng: np.random.Generator,
) -> PromptSample:
    """Assemble the prompt for one sample from augmented masks.

    ``target_mask`` is the ground truth (boolean). ``circle_mask`` is the target's annotation circle, required
    for :attr:`PromptType.CIRCLE_MASK`. ``regions`` are the other annotations in the tile. Whatever the type, the
    negatives always ride along: as a -1 area in the dense mask and, per the configured probabilities, as
    background points. With no negatives and no foreground mask the dense prompt is None so the encoder uses its
    learned no-mask embedding rather than an all-zero mask.
    """
    shape = target_mask.shape
    negative = stamp_negative_regions(regions, shape)
    kwargs = dict(
        min_edge_px=cfg.min_edge_px, line_frac_min=cfg.line_frac_min, line_frac_max=cfg.line_frac_max
    )
    background = _background_points(
        regions, cfg, rng, force_one=prompt_type is PromptType.NEGATIVE_ONLY
    )
    foreground_canvas: np.ndarray | None = None
    box: np.ndarray | None = None

    if prompt_type is PromptType.POINTS:
        points = sample_points_in_mask(target_mask, benford_count(rng), rng, **kwargs)
    elif prompt_type is PromptType.BOX:
        points = np.zeros((0, 2), dtype=np.float32)
        box = jittered_box(target_mask, rng, cfg.box_jitter_frac)
    elif prompt_type is PromptType.BOX_POINTS:
        count = int(rng.integers(1, cfg.box_points_max + 1))
        points = sample_points_in_mask(target_mask, count, rng, **kwargs)
        box = jittered_box(target_mask, rng, cfg.box_jitter_frac)
    elif prompt_type is PromptType.CIRCLE_MASK:
        if circle_mask is None or not circle_mask.any():
            raise ValueError("circle_mask prompt needs a non-empty circle mask")
        foreground_canvas = circle_mask.astype(np.int8)
        ys, xs = np.nonzero(circle_mask)
        points = np.array([[xs.mean(), ys.mean()]], dtype=np.float32)
    elif prompt_type is PromptType.FULL_MASK:
        foreground_canvas = target_mask.astype(np.int8)
        points = sample_points_in_mask(target_mask, 1, rng, **kwargs)
    elif prompt_type is PromptType.NEGATIVE_ONLY:
        points = np.zeros((0, 2), dtype=np.float32)
        if not background.shape[0]:
            raise ValueError("negative_only prompt needs at least one negative region")
    else:  # pragma: no cover - enum is exhaustive
        raise ValueError(f"unknown prompt type {prompt_type!r}")

    coords = np.concatenate([points, background]) if background.shape[0] else points
    labels = np.concatenate(
        [np.ones(points.shape[0], dtype=np.int32), np.zeros(background.shape[0], dtype=np.int32)]
    )
    signed = combine_signed(foreground_canvas, negative)
    has_dense = bool(signed.any())
    foreground_region = (signed > 0) if foreground_canvas is not None and bool((signed > 0).any()) else None
    return PromptSample(
        prompt_type=prompt_type,
        point_coords=coords.astype(np.float32, copy=False),
        point_labels=labels,
        box=box,
        dense_logits=to_low_res_logits(signed) if has_dense else None,
        foreground_region=foreground_region,
    )
