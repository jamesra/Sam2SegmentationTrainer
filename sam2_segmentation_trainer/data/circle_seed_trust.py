"""Evaluation-only circle-seed experiments: the radius-relative seed taper and the oversized circle family.

Neither is used by training, the exported metadata or the server; ``sam2-em-eval --goal`` turns them on
(docs/circle_seed_trust_eval.md). They test the "radius-relative seed trust" idea of docs/mask_refinement_plan.md,
which is proposed, not decided.

* **Seed taper** (``--circle-seed-taper FULL_FRAC``): a circle seed is fed as ``+seed_logit * w(rho)`` instead of
  ``+seed_logit * c(d)``. ``rho`` is the distance from the circle's centre in units of its TRUE radius (before
  ``circle_diameter_scale``) and ``w`` the cosine taper of :func:`annotation_crops.prompt_encoding.circle_trust`, the
  one circle annotations use as negatives, with ``zero_frac`` the diameter scale. The seed's claimed region is the
  disc the default encoding draws, so the overlap rule (a pixel claimed by a positive and a negative region is 0),
  the negatives, the centre point and the area-average reduction are unchanged; only the values inside the disc
  differ.
* **Oversized circle** (the ``circle_oversize`` validation family): the target's inscribed circle with its true
  radius multiplied by a factor drawn from ``--circle-oversize-range`` (1.2 to 2.0), centre unchanged, drawn like any
  seed circle (times the diameter scale). It reaches past the membrane, like an annotator's oversized circle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from annotation_crops.negatives import rasterize_circle
from annotation_crops.prompt_encoding import (
    LOW_RES,
    CircleAnnotationTrust,
    NegativeCanvas,
    area_resize,
    circle_trust,
    encode_dense_canvas,
)

DEFAULT_OVERSIZE_RANGE = (1.2, 2.0)
"""Factor range of the ``circle_oversize`` family's true radius over the inscribed one (``--circle-oversize-range``)."""

OVERSIZE_STREAM = 2
"""Trailing number of the generator ``[seed, sample, family, 2]`` that draws the ``circle_oversize`` factor, its own
stream like the annotator circle's, so no other family's draws move."""


@dataclass(frozen=True, slots=True)
class SeedCircle:
    """A seed circle's centre (tile pixels, pixel (x, y) centred at (x + 0.5, y + 0.5)) and its TRUE radius.

    The disc a prompt draws for it is ``radius * circle_diameter_scale``.
    """

    center_x: float
    center_y: float
    radius: float


def seed_taper(full_frac: float, circle_scale: float) -> CircleAnnotationTrust:
    """The seed taper's fractions: full trust inside ``full_frac`` of the true radius, 0 at ``circle_scale``.

    Raises ValueError unless ``0 <= full_frac < circle_scale``.
    """
    try:
        return CircleAnnotationTrust(full_frac, circle_scale)
    except ValueError:
        raise ValueError(
            f"--circle-seed-taper must satisfy 0 <= full_frac < circle_diameter_scale ({circle_scale}), got {full_frac}"
        ) from None


def seed_taper_weight(shape: tuple[int, int], circle: SeedCircle, taper: CircleAnnotationTrust) -> np.ndarray:
    """``w(rho)`` of ``circle`` over an image of ``shape`` (float32 in [0, 1], 0 beyond ``taper.zero_frac``)."""
    _claimed, weight = circle_trust(shape, circle.center_x, circle.center_y, circle.radius, taper)
    return weight


def tapered_seed_mask_input(
    seed: np.ndarray,
    weight: np.ndarray,
    negative_canvases: Sequence[NegativeCanvas],
    seed_logit: float,
    size: int = LOW_RES,
) -> np.ndarray | None:
    """The ``[size, size]`` ``mask_input`` of a circle seed fed as ``+seed_logit * weight``, or None when all 0.

    ``seed`` is the boolean disc and decides the overlaps, exactly as in
    :func:`annotation_crops.prompt_encoding.encode_dense_canvas`: the canvas is built by that function with the seed
    at a flat ``seed_logit`` (``tau`` 0, which leaves the prepared negative canvases as they are), and only its
    positive pixels are then scaled by ``weight``.
    """
    shape = seed.shape
    canvas = encode_dense_canvas(
        shape, seed=seed, negative_canvases=negative_canvases, tau=0.0, seed_logit=seed_logit
    )
    canvas = np.where(canvas > 0.0, canvas * np.asarray(weight, dtype=np.float32), canvas).astype(np.float32)
    if not canvas.any():
        return None
    return area_resize(canvas, size)


def oversize_factor_range(text: str) -> tuple[float, float]:
    """Parse ``LOW,HIGH`` (``--circle-oversize-range``); raises ValueError unless ``0 < LOW <= HIGH``."""
    try:
        low, high = (float(part) for part in str(text).split(","))
    except ValueError:
        raise ValueError(f"--circle-oversize-range must be LOW,HIGH, got {text!r}") from None
    check_oversize_range((low, high))
    return low, high


def check_oversize_range(factor_range: Sequence[float]) -> tuple[float, float]:
    """``factor_range`` as a float pair; raises ValueError unless ``0 < low <= high``."""
    low, high = (float(v) for v in factor_range)
    if not (np.isfinite(low) and np.isfinite(high) and 0.0 < low <= high):
        raise ValueError(f"the circle_oversize factor range must satisfy 0 < low <= high, got {factor_range}")
    return low, high


def oversize_circle(
    shape: tuple[int, int],
    circle: SeedCircle,
    circle_scale: float,
    rng: np.random.Generator,
    factor_range: Sequence[float] = DEFAULT_OVERSIZE_RANGE,
) -> tuple[np.ndarray, SeedCircle, float]:
    """``(disc, grown, factor)``: ``circle`` with its true radius times ``factor ~ U(factor_range)``, centre unchanged.

    The disc is drawn at ``grown.radius * circle_scale`` like any seed circle. One draw from ``rng``.
    """
    low, high = check_oversize_range(factor_range)
    factor = float(rng.uniform(low, high))
    grown = SeedCircle(circle.center_x, circle.center_y, circle.radius * factor)
    disc = rasterize_circle(shape, grown.center_x, grown.center_y, grown.radius * float(circle_scale))
    return disc, grown, factor
