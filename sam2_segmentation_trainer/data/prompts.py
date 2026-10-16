"""Prompt construction for mixed-prompt SAM2 training (points, boxes, seeds, masks to correct, negative context).

Everything here works on numpy arrays that are already augmented, in tile pixel space with the origin at the
top-left (x right, y down), so it needs neither torch nor SAM2 and can be tested alone.

Prompt meanings follow ``.claude/rules/sam2-prompt-semantics.md``: a box is only a hint that the object is most
likely inside, so no loss here forces predictions into it. A dense mask is one of two things
(docs/mask_refinement_plan.md): a *seed*, trusted inside the cell, that the engine grows beyond, or a *mask to
correct*, a suggestion that may be wrong. The full ground-truth mask is never a prompt. Mask values are scaled by
the edge confidence curve, built by :mod:`annotation_crops.prompt_encoding`, which the server shares.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Collection, Mapping, Sequence

import numpy as np
from scipy.ndimage import distance_transform_edt

from annotation_crops.negatives import KIND_CIRCLE, KIND_POLYGON, NegativeRegion, rasterize_circle
from annotation_crops.prompt_encoding import (
    CORRECTION_LOGIT,
    EDGE_TAU_DEPTH_FRAC,
    EDGE_TAU_PX,
    DEFAULT_CIRCLE_TRUST,
    LOW_RES,
    SEED_LOGIT,
    CircleAnnotationTrust,
    NegativeCanvas,
    edge_confidence,
    encode_mask_input,
    negative_canvas,
    window,
)
from sam2_segmentation_trainer.data.refine import (
    correction_mask,
    fit_adjacent_seed,
    near_correct_mask,
    subset_seed,
)

__all__ = ["LOW_RES"]

MASK_PROMPT_LOGIT = SEED_LOGIT
"""Logit a fully trusted seed pixel is fed to the prompt encoder as (alias of ``SEED_LOGIT``)."""

_MAX_CANDIDATES = 20000
"""Interior pixels considered when placing points; larger sets are randomly thinned to keep this cheap."""


class PromptType(str, Enum):
    """Prompt families a training sample can draw. The value is the name logged and stored in metadata."""

    POINTS = "points"
    BOX = "box"
    BOX_POINTS = "box_points"
    CIRCLE_SEED = "circle_seed"
    ADJACENT_SEED = "adjacent_seed"
    SUBSET_SEED = "subset_seed"
    CORRECTION = "correction"
    NEAR_CORRECT = "near_correct"


SEED_PROMPTS = frozenset({PromptType.CIRCLE_SEED, PromptType.ADJACENT_SEED, PromptType.SUBSET_SEED})
"""Families whose dense mask is a trusted seed (the mask-containment loss applies to their core)."""

FULL_TILE_PROMPTS = frozenset({PromptType.POINTS, PromptType.CIRCLE_SEED})
"""Prompt types a fully masked tile may draw: it has no edge to refine, so no box, correction or cut seed."""


def _pair(value: Any) -> tuple[float, float]:
    low, high = (float(v) for v in value)
    return low, high


@dataclass(frozen=True, slots=True)
class CircleSeedJitter:
    """How an annotator-like circle seed differs from the inscribed one (``prompt.circle_seed_jitter``).

    Annotators draw circles that are smaller than the inscribed circle and often off its centre
    (docs/circle_annotator_spec.md). :func:`annotator_circle` draws one: a centre at a random depth and a random
    radius. ``prob`` is the share of *training* circle seeds replaced by one; the ``circle_annotator`` validation
    family ignores it and always draws one.
    """

    prob: float = 0.7
    """Share of training circle seeds replaced by an annotator-like circle. 0 disables it and draws nothing."""
    depth_range: tuple[float, float] = (0.3, 1.0)
    """Centre depth as a fraction of the target's maximum depth (1 is the deepest point, 0 the edge)."""
    scale_range: tuple[float, float] = (0.45, 1.0)
    """Radius as a fraction of the maximum depth, before the fit cap."""
    fit_grow_px: float = 10.0
    """The radius may exceed the centre's own depth by this much (annotators fit to the visible membrane)."""
    min_radius_px: float = 6.0
    """A smaller circle falls back to the inscribed seed."""

    _KEYS = ("prob", "depth_range", "scale_range", "fit_grow_px", "min_radius_px")

    def __post_init__(self) -> None:
        object.__setattr__(self, "prob", float(self.prob))
        object.__setattr__(self, "depth_range", _pair(self.depth_range))
        object.__setattr__(self, "scale_range", _pair(self.scale_range))
        object.__setattr__(self, "fit_grow_px", float(self.fit_grow_px))
        object.__setattr__(self, "min_radius_px", float(self.min_radius_px))
        self.validate()

    def validate(self) -> None:
        """Raise ValueError when a value is outside its range (``prompt.circle_seed_jitter.*``)."""
        if not 0.0 <= self.prob <= 1.0:
            raise ValueError(f"prompt.circle_seed_jitter.prob must be within [0, 1], got {self.prob}")
        low, high = self.depth_range
        if not 0.0 < low <= high <= 1.0:
            raise ValueError(f"prompt.circle_seed_jitter.depth_range must satisfy 0 < low <= high <= 1, got {self.depth_range}")
        low, high = self.scale_range
        if not 0.0 < low <= high:
            raise ValueError(f"prompt.circle_seed_jitter.scale_range must satisfy 0 < low <= high, got {self.scale_range}")
        if not self.fit_grow_px >= 0.0:
            raise ValueError(f"prompt.circle_seed_jitter.fit_grow_px must be >= 0, got {self.fit_grow_px}")
        if not self.min_radius_px >= 1.0:
            raise ValueError(f"prompt.circle_seed_jitter.min_radius_px must be >= 1, got {self.min_radius_px}")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "CircleSeedJitter":
        """From a config mapping; a missing key keeps its default, None gives the defaults, an unknown key is refused."""
        if not values:
            return cls()
        unknown = [key for key in values if key not in cls._KEYS]
        if unknown:
            raise ValueError(f"unknown prompt.circle_seed_jitter key(s) {unknown}; choose from {list(cls._KEYS)}")
        return cls(**{key: values[key] for key in cls._KEYS if key in values})


DEFAULT_CIRCLE_SEED_JITTER = CircleSeedJitter()


@dataclass(frozen=True, slots=True)
class PromptConfig:
    """Sampling settings from the ``prompt:`` and ``negative:`` sections of the training config."""

    prob_points_only: float = 0.10
    prob_box_only: float = 0.05
    prob_box_points: float = 0.05
    prob_circle_seed: float = 0.40
    prob_adjacent_seed: float = 0.15
    prob_subset_seed: float = 0.10
    prob_correction: float = 0.10
    prob_near_correct: float = 0.05
    variants_per_sample: int = 3
    edge_tau_px: float = EDGE_TAU_PX
    edge_tau_depth_frac: float = EDGE_TAU_DEPTH_FRAC
    seed_logit: float = SEED_LOGIT
    correction_logit: float = CORRECTION_LOGIT
    circle_annotation_trust: CircleAnnotationTrust = DEFAULT_CIRCLE_TRUST
    circle_seed_jitter: CircleSeedJitter = DEFAULT_CIRCLE_SEED_JITTER
    adjacent_fit_inside: float = 0.98
    adjacent_fit_grow_px: float = 10.0
    adjacent_scale_range: tuple[float, float] = (0.7, 1.0)
    erode_px: tuple[float, float] = (10.0, 40.0)
    dilate_px: tuple[float, float] = (5.0, 50.0)
    scale_range: tuple[float, float] = (0.8, 1.25)
    correction_wobble_px: tuple[float, float] = (10.0, 20.0)
    near_correct_wobble_px: tuple[float, float] = (5.0, 12.0)
    near_correct_excursion_px: tuple[float, float] = (15.0, 30.0)
    box_jitter_frac: float = 0.15
    box_points_max: int = 3
    min_edge_px: float = 15.0
    line_frac_min: float = 0.40
    line_frac_max: float = 0.90
    bg_point_prob_circle: float = 0.25
    bg_point_prob_polygon: float = 0.15
    cache_size: int = 64
    drop_overlapping_circles: bool = False
    context_dropout: float = 0.2
    region_dropout: float = 0.2
    corrupt_max_shift_px: int = 30
    corrupt_polygon_tolerance: float = 0.04

    def probabilities(self) -> dict[PromptType, float]:
        """Draw probability of each prompt type, before availability is considered."""
        return {
            PromptType.POINTS: self.prob_points_only,
            PromptType.BOX: self.prob_box_only,
            PromptType.BOX_POINTS: self.prob_box_points,
            PromptType.CIRCLE_SEED: self.prob_circle_seed,
            PromptType.ADJACENT_SEED: self.prob_adjacent_seed,
            PromptType.SUBSET_SEED: self.prob_subset_seed,
            PromptType.CORRECTION: self.prob_correction,
            PromptType.NEAR_CORRECT: self.prob_near_correct,
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
        if self.variants_per_sample < 1:
            raise ValueError("prompt.variants_per_sample must be >= 1")
        self.circle_seed_jitter.validate()
        if self.edge_tau_px < 0.0:
            raise ValueError("prompt.edge_tau_px must be >= 0 (0 disables the edge curve)")
        if self.seed_logit <= 0.0 or self.correction_logit <= 0.0:
            raise ValueError("prompt.seed_logit and prompt.correction_logit must be > 0")
        if not 0.0 < self.adjacent_fit_inside <= 1.0:
            raise ValueError("prompt.adjacent_fit_inside must be within (0, 1]")
        if self.adjacent_fit_grow_px < 0.0:
            raise ValueError("prompt.adjacent_fit_grow_px must be >= 0")
        if self.edge_tau_depth_frac < 0.0:
            raise ValueError("prompt.edge_tau_depth_frac must be >= 0 (0 disables the thin-process cap)")
        for name in (
            "adjacent_scale_range",
            "erode_px",
            "dilate_px",
            "scale_range",
            "correction_wobble_px",
            "near_correct_wobble_px",
            "near_correct_excursion_px",
        ):
            low, high = getattr(self, name)
            if not 0.0 < low <= high:
                raise ValueError(f"prompt.{name} must satisfy 0 < low <= high")
        if self.corrupt_max_shift_px < 1:
            raise ValueError("prompt.corrupt_max_shift_px must be >= 1")
        if self.corrupt_polygon_tolerance <= 0.0:
            raise ValueError("prompt.corrupt_polygon_tolerance must be > 0")
        if not isinstance(self.drop_overlapping_circles, bool):
            raise ValueError("negative.drop_overlapping_circles must be true or false")
        for name in ("bg_point_prob_circle", "bg_point_prob_polygon", "context_dropout", "region_dropout"):
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
            if isinstance(default, CircleAnnotationTrust):
                kwargs[name] = CircleAnnotationTrust.from_mapping(raw)  # raises on full_frac >= zero_frac
            elif isinstance(default, CircleSeedJitter):
                kwargs[name] = CircleSeedJitter.from_mapping(raw)  # raises on an inverted range or an unknown key
            elif isinstance(default, tuple):
                kwargs[name] = _pair(raw)
            elif isinstance(default, bool):
                kwargs[name] = bool(raw)
            else:
                kwargs[name] = type(default)(raw)
        config = cls(**kwargs)
        config.validate()
        return config


@dataclass(eq=False)
class PromptSample:
    """The prompt for one training sample.

    ``point_coords``/``point_labels`` are foreground (1) and background (0) points, possibly none.
    ``box`` is (x1, y1, x2, y2) or None. ``dense_logits`` is the 256x256 float mask prompt (see
    :mod:`annotation_crops.prompt_encoding`) or None, meaning "use the learned no-mask embedding".
    ``foreground_weight`` is the full-resolution trust of a seed (its edge confidence, zero where a negative
    covers it) for the mask-containment loss, or None when the prompt has no seed.
    ``input_mask`` is the full-resolution boolean seed or mask to correct, for scoring how far the model moved
    away from it; None when the prompt has no such mask.
    """

    prompt_type: PromptType
    point_coords: np.ndarray
    point_labels: np.ndarray
    box: np.ndarray | None = None
    dense_logits: np.ndarray | None = None
    foreground_weight: np.ndarray | None = None
    input_mask: np.ndarray | None = None


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
    box = window(mask, 1)
    depth = distance_transform_edt(mask[box])
    peak = float(depth.max())
    threshold = min(float(min_edge_px), peak * 0.5)
    offset = np.array([box[1].start, box[0].start], dtype=np.float32)
    candidates = np.argwhere(depth >= max(threshold, 0.5))[:, ::-1].astype(np.float32) + offset
    if candidates.shape[0] > _MAX_CANDIDATES:
        candidates = candidates[rng.choice(candidates.shape[0], _MAX_CANDIDATES, replace=False)]
    first_y, first_x = np.unravel_index(int(np.argmax(depth)), depth.shape)
    points: list[np.ndarray] = [np.array([first_x, first_y], dtype=np.float32) + offset]
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


def _padded_depth(mask: np.ndarray) -> np.ndarray:
    """Distance to the nearest background pixel with the array border counted as background (0 outside the mask).

    Only the mask's bounding box is transformed, which is exact.
    """
    depth = np.zeros(mask.shape, dtype=np.float64)
    box = window(mask, 1)
    if box is not None:
        depth[box] = distance_transform_edt(np.pad(mask[box], 1))[1:-1, 1:-1]
    return depth


def target_depth(target: np.ndarray) -> np.ndarray:
    """Distance of each pixel of the boolean ``target`` to the nearest background pixel, the tile edge counted as
    background (0 outside the target). Pass it to :func:`annotator_circle` when several circles are drawn for one
    target, so the distance transform is done once."""
    return _padded_depth(np.asarray(target, dtype=bool))


def annotator_circle(
    target: np.ndarray,
    rng: np.random.Generator,
    cfg: CircleSeedJitter,
    depth: np.ndarray | None = None,
) -> np.ndarray | None:
    """Boolean disc for an annotator-like circle seed inside the boolean ``target``, or None (the caller keeps the
    inscribed seed).

    The centre is a target pixel at a random depth (``cfg.depth_range``, as a fraction of the target's maximum depth)
    and the radius a random fraction of that maximum (``cfg.scale_range``), capped at the centre's own depth plus
    ``cfg.fit_grow_px``. Depth is measured with the tile edge counted as background, like the edge weights.

    Draws from ``rng`` in a fixed order: the depth, then the centre, then the scale. Tests pin that order; do not
    reorder it. Returns None for an empty target and when the radius falls below ``cfg.min_radius_px``.

    ``depth`` is :func:`target_depth` of ``target`` when the caller already has it (it saves a distance transform);
    it does not change what is drawn.
    """
    mask = np.asarray(target, dtype=bool)
    if not mask.any():
        return None
    if depth is None:
        depth = _padded_depth(mask)
    deepest = float(depth.max())
    if deepest <= 0.0:
        return None
    wanted = rng.uniform(cfg.depth_range[0], cfg.depth_range[1])
    gap = np.abs(depth / deepest - wanted)
    candidates = np.flatnonzero((mask & (gap <= _CENTRE_DEPTH_TOLERANCE)).ravel())
    if candidates.size == 0:
        # No pixel at that depth (a thin or small target): the pixel(s) nearest to it.
        gap = np.where(mask, gap, np.inf)
        candidates = np.flatnonzero((gap == gap.min()).ravel())
    row, col = divmod(int(candidates[rng.integers(candidates.size)]), mask.shape[1])
    scale = rng.uniform(cfg.scale_range[0], cfg.scale_range[1])
    radius = min(scale * deepest, float(depth[row, col]) + cfg.fit_grow_px)
    if radius < cfg.min_radius_px:
        return None
    return rasterize_circle(mask.shape, col + 0.5, row + 0.5, radius)


_CENTRE_DEPTH_TOLERANCE = 0.05
"""How far, as a fraction of the maximum depth, a candidate centre's depth may be from the drawn depth."""


def interior_point_xy(mask: np.ndarray) -> tuple[float, float] | None:
    """The deepest interior pixel of a boolean mask as (x, y), or None for an empty mask.

    Unlike the centroid it always lies inside the mask, which matters for C-shaped and branching cells. The mask
    is padded with background first, so a mask cut by the tile edge is measured against the edge too.
    """
    if not mask.any():
        return None
    depth = _padded_depth(mask.astype(bool))
    y, x = np.unravel_index(int(np.argmax(depth)), depth.shape)
    return float(x), float(y)


def inscribed_circle_xyr(mask: np.ndarray, scale: float = 0.95) -> tuple[float, float, float] | None:
    """Largest circle inside a boolean mask as (center_x, center_y, radius), radius times ``scale``.

    Stands in for the exporter's inscribed circle (``circle_annotations.json``, ``source = inscribed``) when a
    sample has none in its tile. The mask is padded with background, so the circle stays in the visible part.
    """
    if not mask.any():
        return None
    depth = _padded_depth(mask.astype(bool))
    y, x = np.unravel_index(int(np.argmax(depth)), depth.shape)
    # Distance is to the nearest background pixel center; half a pixel less keeps the disc off it.
    radius = max(float(depth[y, x]) - 0.5, 0.5) * float(scale)
    if radius <= 0.0:
        return None
    return float(x) + 0.5, float(y) + 0.5, radius


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


def choose_prompt_type(
    cfg: PromptConfig,
    rng: np.random.Generator,
    *,
    can_circle_seed: bool = True,
    can_adjacent_seed: bool = False,
    allowed: Collection[PromptType] | None = None,
    exclude: Collection[PromptType] = (),
) -> PromptType:
    """Draw a prompt type, dropping types the sample cannot support and renormalizing the rest.

    ``circle_seed`` needs a circle; ``adjacent_seed`` needs a linked polygon from a neighbouring section. While
    no adjacent shape is available its probability moves to the circle seed, as the plan says. With ``allowed``
    every other type is dropped too (fully masked tiles use only points and circle seeds), and ``exclude`` drops
    types already drawn for another variant of the same sample. If nothing is left the sample gets a points
    prompt when that is allowed, else a circle seed.
    """
    weights = dict(cfg.probabilities())
    if not can_adjacent_seed:
        weights[PromptType.CIRCLE_SEED] += weights[PromptType.ADJACENT_SEED]
        weights[PromptType.ADJACENT_SEED] = 0.0
    if not can_circle_seed:
        weights[PromptType.CIRCLE_SEED] = 0.0
    for kind in weights:
        if kind in exclude or (allowed is not None and kind not in allowed):
            weights[kind] = 0.0
    total = sum(weights.values())
    if total <= 0.0:
        if allowed is None or PromptType.POINTS in allowed:
            return PromptType.POINTS
        return PromptType.CIRCLE_SEED
    types = list(weights)
    return types[int(rng.choice(len(types), p=np.array([weights[t] for t in types]) / total))]


def _background_points(
    regions: list[NegativeRegion],
    cfg: PromptConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Background points from the negative regions.

    Each region is rolled independently (circle regions at ``bg_point_prob_circle``, polygon regions at
    ``bg_point_prob_polygon``) and, when it wins, yields a Benford-distributed 1..9 points.
    """
    chosen: list[NegativeRegion] = [
        region
        for region in regions
        if rng.random() < (cfg.bg_point_prob_circle if region.kind == KIND_CIRCLE else cfg.bg_point_prob_polygon)
    ]
    sets = [
        sample_points_in_mask(
            # A circle annotation's radius is unreliable, so its points come from the part that is surely inside.
            region.mask if region.weight is None else region.core,
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


def negative_region_canvases(regions: list[NegativeRegion], cfg: PromptConfig) -> list[NegativeCanvas]:
    """The mask and canvas of each negative region, for ``build_prompt(negative_canvases=...)``.

    A polygon (or inscribed circle) uses the edge-confidence curve; an annotated circle uses its trust map
    (``-seed_logit * w(rho)``), computed when the region was collected.
    """
    return [
        negative_canvas(r.mask, cfg.edge_tau_px, cfg.seed_logit, cfg.edge_tau_depth_frac)
        if r.weight is None
        else NegativeCanvas(r.mask, -np.float32(cfg.seed_logit) * r.weight)
        for r in regions
    ]


def kept_negative_indices(
    candidates: Sequence[int], cfg: PromptConfig, rng: np.random.Generator
) -> list[int]:
    """The negative regions (indices into the sample's list) a training prompt keeps after dropout.

    With probability ``context_dropout`` the prompt carries none of them (no negative area, no background points);
    otherwise each is dropped independently with probability ``region_dropout``. A rate of 0 draws nothing from
    ``rng``. The ground truth is unchanged, so the loss weights still see every neighbor.
    """
    kept = list(candidates)
    if not kept:
        return kept
    if cfg.context_dropout > 0.0 and rng.random() < cfg.context_dropout:
        return []
    if cfg.region_dropout > 0.0:
        keep = rng.random(len(kept)) >= cfg.region_dropout
        kept = [index for index, flag in zip(kept, keep) if flag]
    return kept


def build_prompt(
    prompt_type: PromptType,
    *,
    target_mask: np.ndarray,
    circle_mask: np.ndarray | None,
    regions: list[NegativeRegion],
    cfg: PromptConfig,
    rng: np.random.Generator,
    adjacent_shape: np.ndarray | None = None,
    negative_canvases: list[NegativeCanvas] | None = None,
    negative_dropout: bool = False,
) -> PromptSample:
    """Assemble the prompt for one sample from augmented masks.

    ``target_mask`` is the ground truth (boolean); every family is trained against it. ``circle_mask`` is the
    target's circle, required for :attr:`PromptType.CIRCLE_SEED`. ``adjacent_shape`` is a boolean raster of the
    linked polygon from a neighbouring section (any frame), for :attr:`PromptType.ADJACENT_SEED`; a shape that
    cannot be fitted inside the cell falls back to the circle seed. ``regions`` are the other annotations in the
    tile. Whatever the type, the negatives ride along: as a negative area in the dense mask and, per the
    configured probabilities, as background points. A neighbor merged into a mask to correct is left out of the
    negative area, otherwise the prompt would label the error. With no negatives and no dense foreground the
    dense prompt is None so the encoder uses its learned no-mask embedding rather than an all-zero mask.

    ``negative_canvases`` are :func:`negative_region_canvases` of ``regions``, computed once by a caller that
    builds several variants of one sample; they save an edge distance transform per region and variant.

    ``negative_dropout`` (training variants after the first) applies ``negative.context_dropout`` and
    ``negative.region_dropout``: the dense negative area and the background points use only the regions that
    survive (:func:`kept_negative_indices`, drawn from ``rng`` after the family's own draws). A mask to correct can
    still merge a dropped neighbor, and the seed's containment weight still treats every neighbor as a negative.
    """
    shape = target_mask.shape
    kwargs = dict(
        min_edge_px=cfg.min_edge_px, line_frac_min=cfg.line_frac_min, line_frac_max=cfg.line_frac_max
    )
    seed: np.ndarray | None = None
    correction: np.ndarray | None = None
    merged_region: NegativeRegion | None = None
    box: np.ndarray | None = None

    if prompt_type is PromptType.ADJACENT_SEED:
        if adjacent_shape is not None:
            seed = fit_adjacent_seed(
                adjacent_shape,
                target_mask,
                rng,
                fit_inside=cfg.adjacent_fit_inside,
                scale_range=cfg.adjacent_scale_range,
                grow_px=cfg.adjacent_fit_grow_px,
            )
        if seed is None or not seed.any():
            prompt_type = PromptType.CIRCLE_SEED
            seed = None

    if prompt_type is PromptType.POINTS:
        points = sample_points_in_mask(target_mask, benford_count(rng), rng, **kwargs)
    elif prompt_type is PromptType.BOX:
        points = np.zeros((0, 2), dtype=np.float32)
        box = jittered_box(target_mask, rng, cfg.box_jitter_frac)
    elif prompt_type is PromptType.BOX_POINTS:
        count = int(rng.integers(1, cfg.box_points_max + 1))
        points = sample_points_in_mask(target_mask, count, rng, **kwargs)
        box = jittered_box(target_mask, rng, cfg.box_jitter_frac)
    elif prompt_type is PromptType.CIRCLE_SEED:
        if circle_mask is None or not circle_mask.any():
            raise ValueError("circle_seed prompt needs a non-empty circle mask")
        seed = circle_mask.astype(bool)
        ys, xs = np.nonzero(seed)
        points = np.array([[xs.mean(), ys.mean()]], dtype=np.float32)
    elif prompt_type is PromptType.ADJACENT_SEED:
        points = sample_points_in_mask(seed, 1, rng, **kwargs)
    elif prompt_type is PromptType.SUBSET_SEED:
        seed = subset_seed(target_mask, rng, erode_px=cfg.erode_px)
        points = sample_points_in_mask(seed, 1, rng, **kwargs)
    elif prompt_type in (PromptType.CORRECTION, PromptType.NEAR_CORRECT):
        if prompt_type is PromptType.CORRECTION:
            polygons = [r for r in regions if r.kind == KIND_POLYGON]
            correction, merged = correction_mask(
                target_mask,
                rng,
                neighbors=[r.mask for r in polygons],
                dilate_px=cfg.dilate_px,
                scale_range=cfg.scale_range,
                polygon_tolerance=cfg.corrupt_polygon_tolerance,
                max_shift_px=cfg.corrupt_max_shift_px,
                wobble_px=cfg.correction_wobble_px,
            )
            merged_region = polygons[merged] if merged is not None else None
        else:
            correction = near_correct_mask(
                target_mask,
                rng,
                wobble_px=cfg.near_correct_wobble_px,
                excursion_px=cfg.near_correct_excursion_px,
            )
        # The click comes from the mask the user sees, as in production, never from the ground truth.
        # A merged neighbor is the part the user knows is wrong, and the deepest point of the union is often inside
        # it, so the click is taken from the rest of the mask.
        clickable = correction
        if merged_region is not None:
            rest = correction & ~np.asarray(merged_region.mask, dtype=bool)
            if rest.any():
                clickable = rest
        points = sample_points_in_mask(clickable, 1, rng, **kwargs)
        if points.shape[0] == 0:
            points = sample_points_in_mask(target_mask, 1, rng, **kwargs)
    else:  # pragma: no cover - enum is exhaustive
        raise ValueError(f"unknown prompt type {prompt_type!r}")

    # After the correction is drawn, so a neighbor merged into it never gets a background point.
    candidates = [i for i, r in enumerate(regions) if r is not merged_region]
    kept = kept_negative_indices(candidates, cfg, rng) if negative_dropout else candidates
    background = _background_points([regions[i] for i in kept], cfg, rng)
    coords = np.concatenate([points, background]) if background.shape[0] else points
    labels = np.concatenate(
        [np.ones(points.shape[0], dtype=np.int32), np.zeros(background.shape[0], dtype=np.int32)]
    )
    negatives = [regions[i].mask for i in candidates]  # every neighbor, for the containment weight
    shared = negative_canvases is not None and merged_region is None
    dense = encode_mask_input(
        shape,
        seed=seed,
        correction=correction,
        negative_canvases=(
            [negative_canvases[i] for i in kept] if shared else negative_region_canvases([regions[i] for i in kept], cfg)
        ),
        tau=cfg.edge_tau_px,
        seed_logit=cfg.seed_logit,
        correction_logit=cfg.correction_logit,
        depth_frac=cfg.edge_tau_depth_frac,
    )
    weight: np.ndarray | None = None
    if seed is not None and seed.any():
        weight = edge_confidence(seed, cfg.edge_tau_px, cfg.edge_tau_depth_frac)
        if negatives:
            weight = weight * ~np.any(np.stack(negatives), axis=0)
        if not weight.any():
            weight = None
    input_mask = seed if seed is not None else correction
    return PromptSample(
        prompt_type=prompt_type,
        point_coords=coords.astype(np.float32, copy=False),
        point_labels=labels,
        box=box,
        dense_logits=dense,
        foreground_weight=weight,
        input_mask=None if input_mask is None else np.asarray(input_mask, dtype=bool),
    )
