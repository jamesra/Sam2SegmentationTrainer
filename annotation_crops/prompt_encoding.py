"""Dense mask-prompt encoding shared by the trainer and the segmentation server.

Training and production must build ``mask_input`` with this one function and these constants, or the mask
network sees a different scale at inference than it was trained on (see docs/mask_refinement_plan.md, section 2).

Human-traced masks are trusted in their interior and only loosely at their edge, so a prompt value is scaled by
the edge confidence ``c(d) = 1 - exp(-d / tau_eff)``, with ``d`` the distance in full-resolution pixels from the
mask's edge and ``tau_eff = min(tau, depth_frac * D)`` capped by the mask's own maximum depth ``D``, so a thin
process is not all edge. Three kinds of region feed the prompt:

* a *seed* (trusted, inside the cell): ``+SEED_LOGIT * c(d)``
* a *mask to correct* (may be wrong): ``+CORRECTION_LOGIT * c(d)``
* other annotations (negative context): ``-SEED_LOGIT * c(d_neighbor)``, each by its own edge distance

Everything else is 0 (unknown). Overlaps sum and are clipped to ``+/-LOGIT_CLIP``. The full-resolution canvas is
reduced to the 256x256 ``mask_input`` by area averaging, since the values are continuous.

Needs numpy and scipy only.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from scipy.ndimage import distance_transform_edt

LOW_RES = 256
"""Side of SAM2's ``mask_input`` for a 1024 image (4x the 64x64 image embedding)."""

EDGE_TAU_PX = 8.0
"""Default tau of the confidence curve: traces are about 9 px (20 nm) from the membrane on average, and
confidence reaches 95% about 24 px inside the traced line."""

EDGE_TAU_DEPTH_FRAC = 0.25
"""Thin-process cap: ``tau_eff = min(tau, EDGE_TAU_DEPTH_FRAC * D)``, ``D`` the mask's maximum depth. A process
20 px wide has ``D`` about 10 and gets ``tau_eff`` 2.5, so its whole profile is not treated as edge."""

MIN_TAU_PX = 0.5
"""Floor of ``tau_eff``, so a one-pixel-wide mask does not divide by zero."""

SEED_LOGIT = 10.0
"""M_seed. SAM2's ``mask_downscaling`` was trained on low-resolution logits and the server passes such logits
clipped to +/-20, so a trusted pixel is fed as +10."""

CORRECTION_LOGIT = 3.0
"""M_corr. Lower than a seed so the mask network learns that a mask to correct is a suggestion."""

LOGIT_CLIP = 10.0
"""Overlapping positive and negative contributions are summed and clipped to +/- this."""

PROMPT_ENCODING_VERSION = "edge-confidence-v4"
"""Name of this encoding. Bump it when the way ``mask_input`` is built changes in a way that needs a retrain; an
exported model carries it so a server can refuse a model whose encoding it does not implement."""


def encoding_metadata(circle_diameter_scale: float) -> dict[str, float | int | str]:
    """The constants a server must use to build ``mask_input`` for a model trained with this encoding."""
    return {
        "prompt_encoding": PROMPT_ENCODING_VERSION,
        "EDGE_TAU_PX": EDGE_TAU_PX,
        "EDGE_TAU_DEPTH_FRAC": EDGE_TAU_DEPTH_FRAC,
        "SEED_LOGIT": SEED_LOGIT,
        "CORRECTION_LOGIT": CORRECTION_LOGIT,
        "LOGIT_CLIP": LOGIT_CLIP,
        "LOW_RES": LOW_RES,
        "circle_diameter_scale": float(circle_diameter_scale),
    }


def window(mask: np.ndarray, margin: int) -> tuple[slice, slice] | None:
    """Bounding box of ``mask`` grown by ``margin`` pixels and clipped to the array, or None for an empty mask.

    A distance transform needs only this window: inside it every distance up to ``margin`` is exact, and the
    callers treat anything farther as "far".
    """
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(mask.any(axis=0))
    height, width = mask.shape
    return (
        slice(max(int(rows[0]) - margin, 0), min(int(rows[-1]) + 1 + margin, height)),
        slice(max(int(cols[0]) - margin, 0), min(int(cols[-1]) + 1 + margin, width)),
    )


def reach_px(tau: float) -> int:
    """Distance beyond which the confidence and the loss weight are within 5% of 1 (``3 tau`` px, 24 at 8)."""
    return int(np.ceil(3.0 * float(tau)))


def edge_distance(mask: np.ndarray) -> np.ndarray:
    """Distance in pixels from each pixel inside ``mask`` to its edge (0 outside).

    Measured from pixel centers to the boundary between a mask pixel and a background pixel, so a pixel next to
    the background is half a pixel from the edge. The border of the array is *not* an edge: a mask cut by the
    tile edge stays confident there. A mask with no background at all is infinitely far from any edge.

    Only the mask's bounding box plus one background pixel is transformed; that is exact, because the nearest
    background pixel of a mask pixel is never outside it.
    """
    mask = np.asarray(mask, dtype=bool)
    box = window(mask, 1)
    if box is None:
        return np.zeros(mask.shape, dtype=np.float32)
    if mask.all():
        return np.full(mask.shape, np.inf, dtype=np.float32)
    crop = mask[box]
    out = np.zeros(mask.shape, dtype=np.float32)
    out[box] = np.maximum(distance_transform_edt(crop) - 0.5, 0.0) * crop
    return out


def effective_tau(depth: float, tau: float = EDGE_TAU_PX, depth_frac: float = EDGE_TAU_DEPTH_FRAC) -> float:
    """``min(tau, depth_frac * depth)`` for a mask whose maximum edge distance is ``depth`` (floored at MIN_TAU_PX).

    ``tau <= 0`` stays 0 (no edge weighting). ``depth_frac <= 0`` or an infinite ``depth`` disables the cap.
    """
    if tau <= 0.0:
        return 0.0
    if depth_frac <= 0.0 or not np.isfinite(depth):
        return float(tau)
    return float(max(MIN_TAU_PX, min(tau, depth_frac * depth)))


def edge_confidence(
    mask: np.ndarray,
    tau: float = EDGE_TAU_PX,
    depth_frac: float = EDGE_TAU_DEPTH_FRAC,
    distance: np.ndarray | None = None,
) -> np.ndarray:
    """``c(d) = 1 - exp(-d / tau_eff)`` inside ``mask`` (float32), 0 outside; ``tau_eff`` is capped per mask.

    ``tau <= 0`` gives the plain 0/1 mask. ``distance`` is a precomputed :func:`edge_distance` of ``mask``.
    """
    mask = np.asarray(mask, dtype=bool)
    if tau <= 0.0:
        return mask.astype(np.float32)
    if distance is None:
        distance = edge_distance(mask)
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.float32)
    tau_eff = effective_tau(float(distance.max()), tau, depth_frac)
    return (1.0 - np.exp(-distance / tau_eff)).astype(np.float32)


def signed_edge_distance(target: np.ndarray, reach: int | None = None) -> np.ndarray:
    """Distance in pixels from every pixel to the edge of ``target``, on both sides (float32, >= 0).

    This is the ``d`` of the loss weight: pixels inside and outside the traced line both count. An all-True or
    all-False array has no edge, so every distance is infinite. With ``reach`` (see :func:`reach_px`) only the
    bounding box grown by that many pixels is transformed and everything beyond it is infinite, which the weight
    curve maps to 1.
    """
    target = np.asarray(target, dtype=bool)
    if target.all() or not target.any():
        return np.full(target.shape, np.inf, dtype=np.float32)
    box = (slice(None), slice(None)) if reach is None else window(target, int(reach) + 1)
    crop = target[box]
    inside = distance_transform_edt(crop)
    outside = distance_transform_edt(~crop)
    near = np.maximum(np.where(crop, inside, outside) - 0.5, 0.0).astype(np.float32)
    if reach is None:
        return near
    out = np.full(target.shape, np.inf, dtype=np.float32)
    out[box] = near
    return out


def area_weights(source: int, size: int) -> np.ndarray:
    """``[size, source]`` matrix that averages (or, when upsampling, replicates) a length-``source`` axis to ``size``.

    Row ``i`` holds the fractions of each source cell that fall in output cell ``i``, normalized to sum to 1.
    """
    edges_out = np.linspace(0.0, float(source), size + 1)
    edges_in = np.arange(source + 1, dtype=np.float64)
    low = np.maximum(edges_out[:-1, None], edges_in[None, :-1])
    high = np.minimum(edges_out[1:, None], edges_in[None, 1:])
    overlap = np.clip(high - low, 0.0, None)
    return (overlap / overlap.sum(axis=1, keepdims=True)).astype(np.float32)


def area_resize(array: np.ndarray, size: int = LOW_RES) -> np.ndarray:
    """Area-average a 2-D float array to ``size`` x ``size`` (float32)."""
    height, width = array.shape
    rows = area_weights(height, size)
    cols = area_weights(width, size)
    return (rows @ array.astype(np.float32, copy=False) @ cols.T).astype(np.float32)


def negative_canvas(
    mask: np.ndarray,
    tau: float = EDGE_TAU_PX,
    seed_logit: float = SEED_LOGIT,
    depth_frac: float = EDGE_TAU_DEPTH_FRAC,
) -> np.ndarray:
    """One negative region as its ``-seed_logit * c(d)`` canvas (float32, <= 0), with its own ``tau_eff``.

    A sample with several prompt variants shares its negatives between them, so it computes these once and
    passes them to :func:`encode_dense_canvas` as ``negative_canvases``.
    """
    return -np.float32(seed_logit) * edge_confidence(mask, tau, depth_frac)


def encode_dense_canvas(
    shape: tuple[int, int],
    *,
    seed: np.ndarray | None = None,
    correction: np.ndarray | None = None,
    negatives: Iterable[np.ndarray] = (),
    negative_canvases: Iterable[np.ndarray] = (),
    tau: float = EDGE_TAU_PX,
    seed_logit: float = SEED_LOGIT,
    correction_logit: float = CORRECTION_LOGIT,
    depth_frac: float = EDGE_TAU_DEPTH_FRAC,
) -> np.ndarray:
    """Full-resolution float32 logit canvas of ``shape`` (see the module docstring for the values)."""
    canvas = np.zeros(shape, dtype=np.float32)
    if seed is not None and np.any(seed):
        canvas += np.float32(seed_logit) * edge_confidence(seed, tau, depth_frac)
    if correction is not None and np.any(correction):
        canvas += np.float32(correction_logit) * edge_confidence(correction, tau, depth_frac)
    for region in negatives:
        if np.any(region):
            canvas += negative_canvas(region, tau, seed_logit, depth_frac)
    for ready in negative_canvases:
        canvas += ready
    return np.clip(canvas, -LOGIT_CLIP, LOGIT_CLIP, out=canvas)


def encode_mask_input(
    shape: tuple[int, int],
    *,
    seed: np.ndarray | None = None,
    correction: np.ndarray | None = None,
    negatives: Sequence[np.ndarray] = (),
    negative_canvases: Sequence[np.ndarray] = (),
    tau: float = EDGE_TAU_PX,
    seed_logit: float = SEED_LOGIT,
    correction_logit: float = CORRECTION_LOGIT,
    size: int = LOW_RES,
    depth_frac: float = EDGE_TAU_DEPTH_FRAC,
) -> np.ndarray | None:
    """The ``[size, size]`` float32 ``mask_input`` for a prompt, or None when every value is 0.

    None means "use the prompt encoder's learned no-mask embedding", which an all-zero tensor would not.
    """
    canvas = encode_dense_canvas(
        shape,
        seed=seed,
        correction=correction,
        negatives=negatives,
        negative_canvases=negative_canvases,
        tau=tau,
        seed_logit=seed_logit,
        correction_logit=correction_logit,
        depth_frac=depth_frac,
    )
    if not canvas.any():
        return None
    return area_resize(canvas, size)
