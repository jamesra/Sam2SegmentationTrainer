"""Per-pixel loss weights and neighbor cores (docs/mask_refinement_plan.md, sections 1, 4 and 6).

* :func:`edge_weight_map`: ``w(d) = max(w_min, 1 - exp(-d / tau_eff))`` from the distance ``d`` to the ground-truth
  edge on both sides, with ``tau_eff`` capped by the target's own depth.
* :func:`core_mask`: the pixels at least ``band`` px inside a mask, its *core*.
* :func:`leak_mask`: the pixels inside another annotation's core and outside the target. A prediction there is a
  merge, the error that breaks connectivity, so the loss weights them up (``loss.leak_weight``) and validation
  counts them (leak, merge rate).

Every distance transform runs inside the mask's bounding box grown by what it needs, not over the whole tile.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from annotation_crops.prompt_encoding import (
    EDGE_TAU_DEPTH_FRAC,
    edge_distance,
    effective_tau,
    reach_px,
    signed_edge_distance,
)


def edge_weight_map(
    target: np.ndarray,
    tau: float,
    w_min: float = 0.05,
    depth_frac: float = EDGE_TAU_DEPTH_FRAC,
) -> np.ndarray | None:
    """The edge-tolerant loss weight of a ground-truth mask (float32), or None when ``tau <= 0``.

    ``tau_eff = min(tau, depth_frac * D)`` with ``D`` the target's maximum depth, so a thin process keeps its
    supervision. Distances are measured on both sides of the edge, inside the box grown by ``3 * tau_eff``;
    beyond it the weight is 1 (the curve is at least 0.95 there).
    """
    if tau <= 0.0:
        return None
    target = np.asarray(target, dtype=bool)
    if not target.any() or target.all():
        return np.ones(target.shape, dtype=np.float32)
    tau_eff = effective_tau(float(edge_distance(target).max()), tau, depth_frac)
    distance = signed_edge_distance(target, reach_px(tau_eff))
    weight = 1.0 - np.exp(-distance / tau_eff)
    return np.maximum(weight, np.float32(w_min)).astype(np.float32)


def core_mask(mask: np.ndarray, band: float) -> np.ndarray:
    """Pixels at least ``band`` px inside ``mask`` (measured to its edge; the array border is not an edge)."""
    mask = np.asarray(mask, dtype=bool)
    if band <= 0.0:
        return mask.copy()
    return edge_distance(mask) >= float(band)


def leak_mask(neighbors: Sequence[np.ndarray], target: np.ndarray, band: float) -> np.ndarray:
    """Pixels inside any neighbor's core and outside ``target``."""
    target = np.asarray(target, dtype=bool)
    leak = np.zeros(target.shape, dtype=bool)
    for neighbor in neighbors:
        leak |= core_mask(neighbor, band)
    return leak & ~target


def neighbor_core_labels(neighbors: Sequence[np.ndarray], target: np.ndarray, band: float) -> np.ndarray:
    """Int16 map: ``i + 1`` on neighbor ``i``'s core (outside ``target``), 0 elsewhere. Later neighbors win overlaps."""
    target = np.asarray(target, dtype=bool)
    labels = np.zeros(target.shape, dtype=np.int16)
    for index, neighbor in enumerate(neighbors):
        labels[core_mask(neighbor, band) & ~target] = index + 1
    return labels


def apply_leak_weight(weight: np.ndarray | None, leak: np.ndarray, leak_weight: float, shape: tuple[int, int]):
    """``weight`` with ``leak_weight`` multiplied in on ``leak`` pixels (a fresh map of ones if ``weight`` is None).

    Returns None when there is nothing to apply (no weight and ``leak_weight`` is 1).
    """
    if leak_weight == 1.0 or not leak.any():
        return weight
    out = np.ones(shape, dtype=np.float32) if weight is None else weight.astype(np.float32, copy=True)
    out[leak] *= np.float32(leak_weight)
    return out
