"""Card geometry shared by both engines and the card builder: bounds, the level a window fits at, and prompt rings.

Kept apart from :mod:`converter.builder` so the engines can use it without importing the builder that uses them.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

from converter import contour as contour_module
from converter import odata
from converter.seg_client import SegmentationFailed

CARD_SIZE = 1024
MARGIN = 48

Bounds = Tuple[float, float, float, float]


def choose_downsample(bounds: Bounds, levels: Sequence[int], at_least: int = 1) -> int:
    """The finest level (``at_least`` or coarser) at which mosaic ``bounds`` plus a margin fit one card.

    Raises ``SegmentationFailed`` when none does.
    """
    min_x, min_y, max_x, max_y = bounds
    extent = max(max_x - min_x, max_y - min_y)
    for level in sorted(levels):
        if level >= at_least and extent / level + 2 * MARGIN <= CARD_SIZE:
            return level
    raise SegmentationFailed(f"too_large: the shape is {extent:.0f} mosaic pixels across and no tileset level fits it in {CARD_SIZE}")


def card_origin(bounds: Bounds, downsample: int) -> Tuple[int, int]:
    """Origin, in level pixels, of the card window centred on mosaic ``bounds``."""
    min_x, min_y, max_x, max_y = bounds
    cx, cy = (min_x + max_x) / 2 / downsample, (min_y + max_y) / 2 / downsample
    return int(math.floor(cx - CARD_SIZE / 2)), int(math.floor(cy - CARD_SIZE / 2))


def union_bounds(a: Bounds, b: Bounds) -> Bounds:
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def shape_bounds(shape: contour_module.Contour, downsample: int) -> Bounds:
    """Mosaic-pixel bounds of a proposed outline (its exterior; holes lie inside it)."""
    xs = [x for x, _ in shape.exterior]
    ys = [y for _, y in shape.exterior]
    return min(xs) * downsample, min(ys) * downsample, max(xs) * downsample, max(ys) * downsample


def scaled(rings: Sequence[Sequence[Tuple[float, float]]], downsample: float) -> List[List[Tuple[float, float]]]:
    return [[(x / downsample, y / downsample) for x, y in ring] for ring in rings]


def rescale(shape: contour_module.Contour, from_level: int, to_level: int) -> contour_module.Contour:
    """An outline in ``from_level`` pixels as ``to_level`` pixels."""
    if from_level == to_level:
        return shape
    factor = from_level / to_level
    return contour_module.Contour(
        exterior=np.asarray(shape.exterior, dtype=float) * factor,
        holes=[np.asarray(hole, dtype=float) * factor for hole in shape.holes],
    )


def rings_for_prompt(record: odata.LocationRecord, downsample: int) -> Tuple[List[Tuple[float, float]], List[List[Tuple[float, float]]]]:
    """The exterior and holes of the polygon to seed with: the largest part of a multi-part shape."""
    best = max(record.polygons, key=lambda rings: abs(contour_module.signed_area(np.array(rings[0], dtype=float))))
    rings = scaled(best, downsample)
    return rings[0], rings[1:]
