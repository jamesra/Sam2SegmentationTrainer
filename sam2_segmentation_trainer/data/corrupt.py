"""Deliberately imperfect mask prompts.

A full ground-truth mask as a prompt teaches the model to copy its input. A person who paints a mask by hand
leaves it too thin, too fat, patchy or coarse, so this module degrades the true mask the same ways and the
model has to find the real membrane instead of echoing the prompt.

Everything works on boolean numpy arrays in tile pixel space and needs neither torch nor SAM2.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.draw import polygon as draw_polygon
from skimage.measure import approximate_polygon, find_contours

CORRUPTIONS = ("morph", "strokes", "polygon")
"""The degradations ``corrupt_mask`` can pick: grow or shrink, partial painting, and a coarse outline."""


def morph_mask(mask: np.ndarray, rng: np.random.Generator, max_shift_px: int) -> np.ndarray:
    """Grow or shrink the mask by a random distance of 1..``max_shift_px`` pixels (a disc, so it stays round).

    Shrinking never takes more than half the mask's own depth, so it cannot erase it.
    """
    depth = distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
    shift = float(rng.integers(1, max(2, int(max_shift_px) + 1)))
    if rng.random() < 0.5:
        outside = distance_transform_edt(~mask)
        return outside <= shift
    return depth > min(shift, float(depth.max()) * 0.5)


def stroke_mask(mask: np.ndarray, rng: np.random.Generator, keep_min: float, keep_max: float) -> np.ndarray:
    """Keep only the parts of the mask that random brush strokes cover, until about ``keep`` of its area is painted."""
    area = int(mask.sum())
    if area == 0:
        return mask.copy()
    target = float(rng.uniform(keep_min, keep_max)) * area
    depth = distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
    radius = max(2.0, float(depth.max()) * float(rng.uniform(0.25, 0.6)))
    ys, xs = np.nonzero(mask)
    height, width = mask.shape
    yy, xx = np.ogrid[0:height, 0:width]
    painted = np.zeros_like(mask, dtype=bool)
    for _ in range(400):
        index = int(rng.integers(0, ys.size))
        y, x = float(ys[index]), float(xs[index])
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        for _step in range(int(rng.integers(3, 12))):
            painted |= (yy - y) ** 2 + (xx - x) ** 2 <= radius**2
            angle += float(rng.normal(0.0, 0.5))
            y += radius * np.sin(angle)
            x += radius * np.cos(angle)
        if int((painted & mask).sum()) >= target:
            break
    return painted & mask


def polygon_mask(mask: np.ndarray, rng: np.random.Generator, tolerance_frac: float) -> np.ndarray:
    """Replace the outline with a coarse polygon (holes are filled), like a quick few-click tracing."""
    padded = np.pad(mask, 1)
    contours = find_contours(padded.astype(np.uint8), 0.5)
    if not contours:
        return mask.copy()
    outline = max(contours, key=len)
    scale = float(np.sqrt(max(int(mask.sum()), 1)))
    tolerance = max(1.0, scale * float(tolerance_frac) * float(rng.uniform(0.5, 1.5)))
    simple = approximate_polygon(outline, tolerance=tolerance)
    if simple.shape[0] < 4:
        return mask.copy()
    rows, cols = draw_polygon(simple[:, 0] - 1, simple[:, 1] - 1, shape=mask.shape)
    out = np.zeros_like(mask, dtype=bool)
    out[rows, cols] = True
    return out


def corrupt_mask(
    mask: np.ndarray,
    rng: np.random.Generator,
    *,
    max_shift_px: int = 12,
    stroke_keep_min: float = 0.4,
    stroke_keep_max: float = 0.9,
    polygon_tolerance: float = 0.04,
) -> np.ndarray:
    """A degraded copy of a boolean mask: one of :data:`CORRUPTIONS`, sometimes followed by a grow or shrink.

    The result is never empty; when a degradation would erase the mask the original is returned.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return mask.copy()
    kind = CORRUPTIONS[int(rng.integers(0, len(CORRUPTIONS)))]
    if kind == "morph":
        out = morph_mask(mask, rng, max_shift_px)
    elif kind == "strokes":
        out = stroke_mask(mask, rng, stroke_keep_min, stroke_keep_max)
        if out.any() and rng.random() < 0.3:
            out = morph_mask(out, rng, max_shift_px)
    else:
        out = polygon_mask(mask, rng, polygon_tolerance)
        if out.any() and rng.random() < 0.3:
            out = morph_mask(out, rng, max_shift_px)
    return out if out.any() else mask.copy()
