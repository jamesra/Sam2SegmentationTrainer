"""Seeds and masks to correct, derived from the true mask (docs/mask_refinement_plan.md, section 3).

Every family here is trained against the true polygon of the target section:

* subset seeds (3.2): a part of the truth, always inside it, so the model grows it;
* masks to correct (3.3): grown, scaled, warped, simplified or merged masks that disagree with the truth;
* near-correct masks (3.4): the truth as a good human would trace it, a few pixels off on average with a few
  local excursions, which the model must not make worse;
* adjacent-section seeds (3.1): a neighbouring section's shape scaled to fit inside the cell.

Everything works on boolean numpy arrays in tile pixel space and needs neither torch nor SAM2. Every distance
transform runs inside the mask's bounding box grown by the radius it needs, not over the whole tile.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import (
    affine_transform,
    binary_erosion,
    distance_transform_edt,
    gaussian_filter,
    label,
    map_coordinates,
    zoom,
)
from skimage.graph import MCP_Geometric

from annotation_crops.prompt_encoding import window
from sam2_segmentation_trainer.data.corrupt import polygon_mask

SUBSET_KINDS = ("erode", "straight_cut", "shape_cut")
CORRECTION_KINDS = ("grow", "scale", "wobble", "cascade", "merge")


def _depth(mask: np.ndarray) -> np.ndarray:
    """Distance to the nearest background pixel (the array border counts as background), 0 outside the mask.

    Only the mask's bounding box is transformed, which is exact.
    """
    out = np.zeros(mask.shape, dtype=np.float64)
    box = window(mask, 1)
    if box is not None:
        out[box] = distance_transform_edt(np.pad(mask[box], 1))[1:-1, 1:-1]
    return out


def grow_mask(mask: np.ndarray, radius: float) -> np.ndarray:
    """``mask`` dilated by ``radius`` pixels (Euclidean), computed inside the bounding box grown by the radius."""
    box = window(mask, int(np.ceil(radius)) + 1)
    out = np.zeros(mask.shape, dtype=bool)
    if box is not None:
        out[box] = distance_transform_edt(~mask[box]) <= radius
    return out


def _anchor(mask: np.ndarray) -> tuple[int, int]:
    """Deepest interior pixel (y, x)."""
    y, x = np.unravel_index(int(np.argmax(_depth(mask))), mask.shape)
    return int(y), int(x)


def keep_component(mask: np.ndarray, anchor: tuple[int, int]) -> np.ndarray:
    """The connected piece of ``mask`` that contains ``anchor``; ``mask`` itself if the anchor is outside it."""
    if not mask[anchor]:
        return mask
    labels, _count = label(mask)
    return labels == labels[anchor]


def drop_fragments(mask: np.ndarray, min_share: float = 0.1) -> np.ndarray:
    """Remove pieces smaller than ``min_share`` of the whole mask (what a warp or a dent can leave behind)."""
    if not mask.any():
        return mask
    labels, count = label(mask)
    if count <= 1:
        return mask
    sizes = np.bincount(labels.ravel())[1:]
    keep = np.flatnonzero(sizes >= min_share * sizes.sum()) + 1
    return np.isin(labels, keep)


def erode_seed(mask: np.ndarray, rng: np.random.Generator, erode_px: tuple[float, float]) -> np.ndarray:
    """Shrink by ``U(erode_px)`` pixels, never by more than 0.6 of the mask's own depth so it cannot vanish."""
    depth = _depth(mask)
    shift = float(rng.uniform(*erode_px))
    return depth > min(shift, float(depth.max()) * 0.6)


def straight_cut_seed(mask: np.ndarray, rng: np.random.Generator, keep: tuple[float, float]) -> np.ndarray:
    """Keep the side of a random line through the mask that holds a ``U(keep)`` share of its area.

    Mimics a mask that stopped at a tile edge.
    """
    ys, xs = np.nonzero(mask)
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    projection = xs * np.cos(angle) + ys * np.sin(angle)
    limit = np.quantile(projection, float(rng.uniform(*keep)))
    cut = np.zeros_like(mask)
    inside = projection <= limit
    cut[ys[inside], xs[inside]] = True
    return keep_component(cut, _anchor(cut)) if cut.any() else cut


def shape_cut_seed(mask: np.ndarray, rng: np.random.Generator, keep: tuple[float, float]) -> np.ndarray:
    """Keep the pixels nearest an interior anchor, measured along the inside of the cell.

    Mimics a mask that stopped partway along a process or at a branch. The anchor is a random pixel from the
    deeper half of the mask, and the kept share of the area is ``U(keep)``.
    """
    depth = _depth(mask)
    deep = np.argwhere(mask & (depth >= np.median(depth[mask])))
    start = tuple(int(v) for v in deep[int(rng.integers(len(deep)))])
    box = window(mask, 1)
    crop = mask[box]
    costs = np.where(crop, 1.0, np.inf)
    distances, _trace = MCP_Geometric(costs).find_costs([(start[0] - box[0].start, start[1] - box[1].start)])
    reach = distances[crop]
    reach = reach[np.isfinite(reach)]
    limit = np.quantile(reach, float(rng.uniform(*keep)))
    cut = np.zeros_like(mask)
    cut[box] = crop & (distances <= limit)
    return keep_component(cut, start)


def subset_seed(
    mask: np.ndarray,
    rng: np.random.Generator,
    *,
    erode_px: tuple[float, float] = (10.0, 40.0),
    keep: tuple[float, float] = (0.2, 0.8),
) -> np.ndarray:
    """A random subset of the true mask, connected and never empty.

    The mask is first eroded by ``U(erode_px)`` pixels (at least 10 by default, so the seed stays inside the real
    cell where the trace runs outside the membrane; never more than 0.6 of the mask's own depth). Then one of
    :data:`SUBSET_KINDS` is drawn: the eroded mask itself, a straight cut through it, or a cut along the cell
    from an interior anchor. The result is always inside ``mask``; if it would be empty the eroded mask is used.
    """
    mask = np.asarray(mask, dtype=bool)
    eroded = erode_seed(mask, rng, erode_px)
    if not eroded.any():
        eroded = mask
    kind = SUBSET_KINDS[int(rng.integers(len(SUBSET_KINDS)))]
    if kind == "erode":
        out = eroded
    elif kind == "straight_cut":
        out = straight_cut_seed(eroded, rng, keep)
    else:
        out = shape_cut_seed(eroded, rng, keep)
    out = out & mask
    return out if out.any() else eroded.copy()


def _smooth_noise(shape: tuple[int, int], wavelength: float, rng: np.random.Generator) -> np.ndarray:
    """Smooth zero-mean noise with features about ``wavelength`` pixels across: a cubic upsample of a coarse grid.

    Far cheaper than a Gaussian filter with a sigma of tens of pixels.
    """
    height, width = shape
    step = max(2.0, float(wavelength))
    coarse = rng.standard_normal((int(np.ceil(height / step)) + 4, int(np.ceil(width / step)) + 4))
    return zoom(coarse, step, order=3, mode="nearest")[:height, :width]


MIN_NEAR_CORRECT_IOU = 0.3
"""A near-correct mask must overlap the truth at least this much (only small cells ever come close to it)."""


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int((a | b).sum())
    return float((a & b).sum()) / union if union else 1.0


def _boundary(mask: np.ndarray) -> np.ndarray:
    return mask & ~binary_erosion(mask, border_value=1)


def _warp(
    mask: np.ndarray,
    rng: np.random.Generator,
    wavelength: float,
    amplitude: float,
    *,
    mean_boundary: bool = False,
) -> np.ndarray:
    """Elastic warp: a smooth random displacement field, applied inside the mask's bounding box.

    By default the largest displacement is ``amplitude`` pixels. With ``mean_boundary`` the field is scaled so
    that the displacement of the outline along its normal averages ``amplitude`` pixels, which is what decides
    how far a warped outline lies from the truth (the peak of a smooth field is several times its mean).
    """
    # The box has to hold the moved boundary; the displacement stays under a few times its mean.
    margin = int(np.ceil(4.0 * amplitude if mean_boundary else amplitude)) + 2
    box = window(mask, margin)
    if box is None:
        return mask.copy()
    crop = mask[box]
    field_y = _smooth_noise(crop.shape, wavelength, rng)
    field_x = _smooth_noise(crop.shape, wavelength, rng)
    if mean_boundary:
        grad_y, grad_x = np.gradient(gaussian_filter(crop.astype(np.float32), 2.0))
        length = np.hypot(grad_y, grad_x)
        edge = _boundary(crop) & (length > 1e-3)
        if edge.any():
            normal = np.abs(field_y[edge] * grad_y[edge] + field_x[edge] * grad_x[edge]) / length[edge]
            reference = float(normal.mean())
        else:
            reference = float(np.hypot(field_y, field_x).mean())
    else:
        reference = float(np.hypot(field_y, field_x).max())
    scale = amplitude / max(reference, 1e-9)
    yy, xx = np.meshgrid(np.arange(crop.shape[0]), np.arange(crop.shape[1]), indexing="ij")
    moved = map_coordinates(
        crop.astype(np.float32), [yy + field_y * scale, xx + field_x * scale], order=1, mode="nearest"
    )
    out = np.zeros(mask.shape, dtype=bool)
    out[box] = moved > 0.5
    return out


def local_excursions(
    mask: np.ndarray,
    rng: np.random.Generator,
    count: int,
    amplitude_px: tuple[float, float],
    span_px: tuple[float, float] = (30.0, 150.0),
) -> np.ndarray:
    """``count`` local bulges or dents of ``U(amplitude_px)`` pixels, each spanning ``U(span_px)`` pixels of outline.

    A bulge adds the pixels within a smoothly tapering distance outside the mask; a dent removes the pixels within
    one inside it. Each is centred on a random outline pixel. No excursion is larger than half the mask's own
    depth, so a small cell is bent but not cut in two or swallowed. Pieces smaller than a tenth of the mask are
    dropped.
    """
    out = np.asarray(mask, dtype=bool).copy()
    if not out.any() or count <= 0:
        return out
    height, width = out.shape
    depth_cap = 0.5 * float(_depth(out).max())
    for _ in range(int(count)):
        box = window(out, 1)
        if box is None:  # an earlier dent erased the mask
            break
        ys, xs = np.nonzero(_boundary(out[box]))
        if ys.size == 0:
            break
        pick = int(rng.integers(ys.size))
        ay, ax = int(ys[pick]) + box[0].start, int(xs[pick]) + box[1].start
        amplitude = min(float(rng.uniform(*amplitude_px)), depth_cap)
        radius = float(rng.uniform(*span_px)) / 2.0
        outward = bool(rng.random() < 0.5)
        reach = int(np.ceil(radius + amplitude + 2))
        y0, y1 = max(ay - reach, 0), min(ay + reach + 1, height)
        x0, x1 = max(ax - reach, 0), min(ax + reach + 1, width)
        sub = out[y0:y1, x0:x1]
        yy, xx = np.mgrid[y0:y1, x0:x1]
        distance = np.hypot(yy - ay, xx - ax)
        taper = np.where(distance < radius, 0.5 * (1.0 + np.cos(np.pi * distance / radius)), 0.0)
        active = taper > 0.0
        if outward:
            sub |= active & (distance_transform_edt(~sub) <= amplitude * taper)
        else:
            sub &= ~(active & (distance_transform_edt(sub) <= amplitude * taper))
        out[y0:y1, x0:x1] = sub
    return drop_fragments(out)


def clip_to_band(mask: np.ndarray, truth: np.ndarray, px: float) -> np.ndarray:
    """Force ``mask`` to agree with ``truth`` more than ``px`` pixels from its outline (no excursion exceeds ``px``)."""
    inner = _depth(truth) > px
    outer = grow_mask(truth, px)
    return (mask | inner) & outer


def _scale_about(mask: np.ndarray, centre: tuple[float, float], factor: float) -> np.ndarray:
    """Scale a mask by ``factor`` about ``centre`` (y, x), like the mouse-wheel scaling in Viking."""
    cy, cx = centre
    offset = (cy - cy / factor, cx - cx / factor)
    moved = affine_transform(
        mask.astype(np.float32), [1.0 / factor, 1.0 / factor], offset=offset, order=1, mode="constant"
    )
    return moved > 0.5


def near_correct_mask(
    mask: np.ndarray,
    rng: np.random.Generator,
    *,
    wobble_px: tuple[float, float] = (5.0, 12.0),
    excursion_px: tuple[float, float] = (15.0, 30.0),
    max_excursions: int = 3,
    span_px: tuple[float, float] = (30.0, 150.0),
) -> np.ndarray:
    """The true mask as a good human trace would draw it: wobbly, with a few local excursions.

    A smooth warp whose *mean* outline displacement is ``U(wobble_px)`` (the amplitude is the mean, not the
    peak), plus 1..``max_excursions`` local bulges or dents of ``U(excursion_px)`` pixels, each spanning
    ``U(span_px)`` pixels of outline. No part of the result lies more than ``excursion_px[1]`` pixels from the
    truth's outline. The wobble mean is at most 0.25 of the mask's depth and each excursion at most 0.5, so a small
    cell is bent, not destroyed. Never empty: if the result is empty the mask is returned unchanged.
    """
    mask = np.asarray(mask, dtype=bool)
    # A trace is off by about the same number of pixels whatever the cell size, but a small cell must stay itself.
    mean = min(float(rng.uniform(*wobble_px)), 0.25 * float(_depth(mask).max()))
    wobbled = drop_fragments(
        clip_to_band(_warp(mask, rng, wavelength=max(60.0, 6.0 * mean), amplitude=mean, mean_boundary=True), mask,
                     float(excursion_px[1]))
    )
    out = wobbled
    if max_excursions > 0 and wobbled.any():
        out = local_excursions(wobbled, rng, int(rng.integers(1, max_excursions + 1)), excursion_px, span_px)
        out = drop_fragments(clip_to_band(out, mask, float(excursion_px[1])))
    # "Near-correct" must stay near: a small cell can come out as a sliver, so fall back to less perturbation.
    for candidate in (out, wobbled):
        if candidate.any() and _iou(candidate, mask) >= MIN_NEAR_CORRECT_IOU:
            return candidate
    return mask.copy()


def _simplify_and_morph(
    mask: np.ndarray, rng: np.random.Generator, polygon_tolerance: float, max_shift_px: int
) -> np.ndarray:
    """CascadePSP-style: a coarse polygon outline, then a random grow or shrink of 1..``max_shift_px`` pixels."""
    simple = polygon_mask(mask, rng, polygon_tolerance)
    shift = float(rng.integers(1, max(2, int(max_shift_px) + 1)))
    if rng.random() < 0.5:
        return grow_mask(simple, shift)
    depth = _depth(simple)
    return depth > min(shift, float(depth.max()) * 0.5)


def correction_mask(
    mask: np.ndarray,
    rng: np.random.Generator,
    *,
    neighbors: list[np.ndarray] | None = None,
    dilate_px: tuple[float, float] = (5.0, 50.0),
    scale_range: tuple[float, float] = (0.8, 1.25),
    polygon_tolerance: float = 0.04,
    max_shift_px: int = 30,
    wobble_px: tuple[float, float] = (10.0, 20.0),
    excursion_px: tuple[float, float] = (15.0, 50.0),
    merge_reach_px: float = 20.0,
) -> tuple[np.ndarray, int | None]:
    """A mask for the model to correct, and the index in ``neighbors`` of a neighbor merged into it (else None).

    One of :data:`CORRECTION_KINDS` is drawn (``merge`` only when ``neighbors`` is non-empty):
    grow (tracing outside the membrane), wheel-style scale, outline wobble (an elastic warp with a mean
    displacement of ``U(wobble_px)`` and local excursions up to ``excursion_px[1]``), a CascadePSP-style
    simplified outline followed by a random grow or shrink, or a union with part of an annotated neighbor. A merge
    picks among the neighbors within ``merge_reach_px`` of ``mask`` (the ones a tracer could plausibly leak into),
    or any neighbor when none is that close. The result always differs from ``mask`` and is never empty; a draw
    that came out equal falls back to a grow.
    """
    mask = np.asarray(mask, dtype=bool)
    kinds = [k for k in CORRECTION_KINDS if k != "merge" or neighbors]
    kind = kinds[int(rng.integers(len(kinds)))]
    merged: int | None = None
    if kind == "grow":
        out = grow_mask(mask, float(rng.uniform(*dilate_px)))
    elif kind == "scale":
        ys, xs = np.nonzero(mask)
        factor = float(rng.uniform(*scale_range))
        if abs(factor - 1.0) < 0.03:
            factor = 1.1
        out = _scale_about(mask, (float(ys.mean()), float(xs.mean())), factor)
    elif kind == "wobble":
        mean = min(float(rng.uniform(*wobble_px)), 0.5 * float(_depth(mask).max()))
        out = _warp(mask, rng, wavelength=max(80.0, 6.0 * mean), amplitude=mean, mean_boundary=True)
        out = local_excursions(out, rng, int(rng.integers(1, 4)), excursion_px)
        out = drop_fragments(clip_to_band(out, mask, float(excursion_px[1])))
    elif kind == "cascade":
        out = _simplify_and_morph(mask, rng, polygon_tolerance, max_shift_px)
    else:
        reach = grow_mask(mask, merge_reach_px)
        near = [i for i, other in enumerate(neighbors) if (reach & np.asarray(other, dtype=bool)).any()]
        pool = near or list(range(len(neighbors)))
        merged = pool[int(rng.integers(len(pool)))]
        other = np.asarray(neighbors[merged], dtype=bool)
        if rng.random() < 0.5:
            other = straight_cut_seed(other, rng, (0.3, 0.9)) if other.any() else other
        out = mask | other
        if bool((out == mask).all()):
            merged = None
    if not out.any() or bool((out == mask).all()):
        out = grow_mask(mask, float(rng.uniform(*dilate_px)))
        merged = None
    return out, merged


def fit_adjacent_seed(
    shape_mask: np.ndarray,
    target: np.ndarray,
    rng: np.random.Generator,
    *,
    fit_inside: float = 0.98,
    scale_range: tuple[float, float] = (0.7, 1.0),
    centre: tuple[float, float] | None = None,
    grow_px: float = 10.0,
) -> np.ndarray | None:
    """A neighbouring section's shape scaled to sit inside ``target`` grown by ``grow_px`` (the Viking workflow).

    ``shape_mask`` is the linked polygon rasterized in any frame; it is re-centred on ``centre`` (y, x; the
    target's inscribed-circle centre when omitted). An annotator fits the seed to the visible membrane, which can
    lie about ``grow_px`` outside the trace, so the largest scale at which at least ``fit_inside`` of the seed
    lies inside the *grown* target is found by bisection and multiplied by ``U(scale_range)``, plus a small random
    offset that keeps the share inside. Returns None when the shape is empty or cannot be fitted at any scale.
    """
    shape_mask = np.asarray(shape_mask, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if not shape_mask.any() or not target.any():
        return None
    centre = centre if centre is not None else _anchor(target)
    allowed = grow_mask(target, grow_px) if grow_px > 0 else target
    ys, xs = np.nonzero(shape_mask)
    rel = np.stack([ys - ys.mean(), xs - xs.mean()], axis=1)
    height, width = target.shape

    def inside_share(scale: float, shift: tuple[float, float] = (0.0, 0.0)) -> float:
        py = np.rint(rel[:, 0] * scale + centre[0] + shift[0]).astype(int)
        px = np.rint(rel[:, 1] * scale + centre[1] + shift[1]).astype(int)
        valid = (py >= 0) & (py < height) & (px >= 0) & (px < width)
        hit = np.zeros(py.shape, dtype=bool)
        hit[valid] = allowed[py[valid], px[valid]]
        return float(hit.mean())

    low, high = 0.0, 4.0 * max(height, width) / max(float(np.ptp(rel)), 1.0)
    if inside_share(1e-3) < fit_inside:
        return None
    for _ in range(14):
        middle = 0.5 * (low + high)
        if inside_share(middle) >= fit_inside:
            low = middle
        else:
            high = middle
    scale = low * float(rng.uniform(*scale_range))
    if scale <= 0.0:
        return None
    shift = (0.0, 0.0)
    radius = float(_depth(target).max()) * 0.25
    for _ in range(8):
        trial = (float(rng.uniform(-radius, radius)), float(rng.uniform(-radius, radius)))
        if inside_share(scale, trial) >= fit_inside:
            shift = trial
            break
    seed = _scale_about_frame(
        shape_mask, (ys.mean(), xs.mean()), (centre[0] + shift[0], centre[1] + shift[1]), scale, target.shape
    )
    return seed if seed.any() else None


def _scale_about_frame(
    mask: np.ndarray,
    source_centre: tuple[float, float],
    new_centre: tuple[float, float],
    factor: float,
    out_shape: tuple[int, int],
) -> np.ndarray:
    """``mask`` scaled by ``factor`` about its centre and placed with that centre at ``new_centre`` (y, x)."""
    sy, sx = source_centre
    ny, nx = new_centre
    offset = (sy - ny / factor, sx - nx / factor)
    moved = affine_transform(
        mask.astype(np.float32), [1.0 / factor, 1.0 / factor], offset=offset,
        order=1, mode="constant", output_shape=out_shape,
    )
    return moved > 0.5
