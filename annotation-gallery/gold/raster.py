"""Exact polygon fill by pixel centre, even-odd, for gold masks.

A pixel is inside when its centre ``(x + 0.5, y + 0.5)`` is inside the rings (the exterior first, holes after, even-odd).
Vertices are in pixel-edge coordinates, so a rectangle ``[4, 12] x [4, 10]`` fills exactly 8 x 6 pixels. (PIL's polygon fill
includes boundary pixels, which would make every retraced mask about a pixel fat.)
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np

Ring = Sequence[Tuple[float, float]]


def fill_rings(rings: Sequence[Ring], width: int, height: int) -> np.ndarray:
    """Boolean ``(height, width)`` mask of the even-odd fill of ``rings``."""
    rows: list = []
    xs: list = []
    for ring in rings:
        points = list(ring)
        if len(points) < 3:
            continue
        if points[0] != points[-1]:
            points.append(points[0])
        for (x1, y1), (x2, y2) in zip(points[:-1], points[1:]):
            if y1 == y2:
                continue
            low, high = (y1, y2) if y1 < y2 else (y2, y1)
            first = max(0, math.ceil(low - 0.5))
            last = min(height - 1, math.ceil(high - 0.5) - 1)
            if last < first:
                continue
            row = np.arange(first, last + 1)
            crossing = x1 + (row + 0.5 - y1) * (x2 - x1) / (y2 - y1)
            rows.append(row)
            xs.append(crossing)
    mask = np.zeros((height, width), dtype=bool)
    if not rows:
        return mask
    all_rows = np.concatenate(rows)
    all_xs = np.concatenate(xs)
    order = np.lexsort((all_xs, all_rows))
    all_rows, all_xs = all_rows[order], all_xs[order]
    # Crossings come in pairs per row; fill pixels whose centre lies between each pair.
    starts = np.flatnonzero(np.r_[True, all_rows[1:] != all_rows[:-1]])
    ends = np.r_[starts[1:], len(all_rows)]
    for start, end in zip(starts, ends):
        row = int(all_rows[start])
        values = all_xs[start:end]
        for index in range(0, len(values) - 1, 2):
            left = max(0, math.ceil(values[index] - 0.5))
            right = min(width, math.ceil(values[index + 1] - 0.5))
            if right > left:
                mask[row, left:right] = True
    return mask
