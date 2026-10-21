"""Logits to polygon, the way Viking's SegmentationMaskPolygonizer does it: smooth, contour at the iso level, take the
largest ring as the exterior, keep real holes, and thin the vertices without making the ring cross itself.

The logit grid is contoured *before* it is scaled (the server asks clients for that: RESPONSE_CONTENT_NO_UPSCALE), then the
vertices are multiplied by the logit scale and mapped into mosaic pixels, X right, Y up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import contourpy
import numpy as np
from scipy.ndimage import gaussian_filter

from converter.seg_client import SegmentResult

Ring = np.ndarray  # (n, 2) float64, closed: last row equals the first

# A hole must be at least this share of the exterior's area to be kept (the server fills smaller ones, too).
MIN_HOLE_FRACTION = 0.03
SMOOTHING_SIGMA = 0.5
# Samples of "certainly not object" added around the grid so a shape touching the crop edge still closes.
PAD = 2
PAD_VALUE = -20.0


@dataclass
class Contour:
    exterior: Ring
    holes: List[Ring] = field(default_factory=list)

    def vertex_count(self) -> int:
        return len(self.exterior) - 1 + sum(len(hole) - 1 for hole in self.holes)


def signed_area(ring: Ring) -> float:
    x, y = ring[:, 0], ring[:, 1]
    return float(0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def _orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def ring_self_intersects(ring: Ring) -> bool:
    """True when two non-adjacent edges of the closed ring cross or touch."""
    n = len(ring) - 1
    if n < 4:
        return False
    a = ring[:-1]
    b = ring[1:]
    for i in range(n):
        js = np.arange(i + 2, n if i > 0 else n - 1)
        if js.size == 0:
            continue
        c, d = a[js], b[js]
        p, q = a[i], b[i]
        o1 = (q[0] - p[0]) * (c[:, 1] - p[1]) - (q[1] - p[1]) * (c[:, 0] - p[0])
        o2 = (q[0] - p[0]) * (d[:, 1] - p[1]) - (q[1] - p[1]) * (d[:, 0] - p[0])
        o3 = (d[:, 0] - c[:, 0]) * (p[1] - c[:, 1]) - (d[:, 1] - c[:, 1]) * (p[0] - c[:, 0])
        o4 = (d[:, 0] - c[:, 0]) * (q[1] - c[:, 1]) - (d[:, 1] - c[:, 1]) * (q[0] - c[:, 0])
        if np.any((o1 * o2 <= 0) & (o3 * o4 <= 0)):
            return True
    return False


def _distance_to_chord(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    chord = end - start
    length2 = float(chord @ chord)
    if length2 == 0.0:
        return np.hypot(*(points - start).T)
    t = np.clip(((points - start) @ chord) / length2, 0.0, 1.0)
    return np.hypot(*(points - (start + np.outer(t, chord))).T)


def douglas_peucker(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Open-polyline Douglas-Peucker; both ends are kept."""
    keep = np.zeros(len(points), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        first, last = stack.pop()
        if last <= first + 1:
            continue
        distances = _distance_to_chord(points[first + 1:last], points[first], points[last])
        index = int(np.argmax(distances))
        if distances[index] > tolerance:
            split = first + 1 + index
            keep[split] = True
            stack.append((first, split))
            stack.append((split, last))
    return points[keep]


def simplify_ring(ring: Ring, tolerance: float) -> Ring:
    """Thin a closed ring to ``tolerance``; halves the tolerance while the result crosses itself (then gives up)."""
    if tolerance <= 0 or len(ring) < 6:
        return ring
    body = ring[:-1]
    # Split at the point farthest from the first so both halves are real chords, then close.
    far = int(np.argmax(np.hypot(*(body - body[0]).T)))
    current = tolerance
    for _ in range(5):
        first = douglas_peucker(body[: far + 1], current)
        second = douglas_peucker(np.vstack([body[far:], body[:1]]), current)
        candidate = np.vstack([first[:-1], second])
        if len(candidate) >= 4 and not ring_self_intersects(candidate):
            return candidate
        current /= 2.0
    return ring


def smooth(logits: np.ndarray, sigma: float = SMOOTHING_SIGMA) -> np.ndarray:
    return gaussian_filter(logits.astype(np.float32), sigma=sigma, mode="nearest") if sigma > 0 else logits


def _point_in_ring(point: np.ndarray, ring: Ring) -> bool:
    x, y = point
    inside = False
    for (x1, y1), (x2, y2) in zip(ring[:-1], ring[1:]):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def contour_rings(grid: np.ndarray, level: float) -> List[Ring]:
    """Closed iso-lines of ``grid`` at ``level`` as (n, 2) arrays of (column, row) in fractional sample indices."""
    generator = contourpy.contour_generator(z=grid, line_type=contourpy.LineType.Separate)
    rings = []
    for line in generator.lines(level):
        line = np.asarray(line, dtype=np.float64)
        if len(line) >= 4 and np.allclose(line[0], line[-1]):
            rings.append(line)
    return rings


def contour_from_result(
    result: SegmentResult,
    *,
    sigma: float = SMOOTHING_SIGMA,
    tolerance: float = 1.0,
    min_hole_fraction: float = MIN_HOLE_FRACTION,
) -> Optional[Contour]:
    """The object's polygon in mosaic pixels (downsample pixels, X right, Y up), or None when nothing is above the level."""
    grid = np.pad(smooth(result.logits, sigma), PAD, constant_values=PAD_VALUE)
    rings = contour_rings(grid, result.threshold)
    if not rings:
        return None

    def to_mosaic(ring: Ring) -> Ring:
        column = ring[:, 0] - PAD
        row = ring[:, 1] - PAD
        x = result.origin_x + result.logits_x + (column + 0.5) * result.scale
        y = result.origin_y + result.frame_height_pixels - (result.logits_y + (row + 0.5) * result.scale)
        return np.column_stack([x, y])

    mosaic = [to_mosaic(ring) for ring in rings]
    areas = [abs(signed_area(ring)) for ring in mosaic]
    order = int(np.argmax(areas))
    exterior = mosaic[order]
    holes = [
        ring
        for index, ring in enumerate(mosaic)
        if index != order
        and areas[index] >= min_hole_fraction * areas[order]
        and _point_in_ring(ring[0], exterior)
    ]
    exterior = simplify_ring(_ccw(exterior), tolerance)
    return Contour(exterior=_close(exterior), holes=[_close(simplify_ring(_cw(hole), tolerance)) for hole in holes])


def _ccw(ring: Ring) -> Ring:
    return ring if signed_area(ring) > 0 else ring[::-1].copy()


def _cw(ring: Ring) -> Ring:
    return ring if signed_area(ring) < 0 else ring[::-1].copy()


def _close(ring: Ring) -> Ring:
    return ring if np.array_equal(ring[0], ring[-1]) else np.vstack([ring, ring[:1]])


def to_world(ring: Ring, downsample: int) -> Ring:
    return ring * float(downsample)


def ring_wkt(ring: Ring, precision: int = 2) -> str:
    return ", ".join(f"{x:.{precision}f} {y:.{precision}f}" for x, y in ring)


def polygon_wkt(contour: Contour, downsample: int = 1, precision: int = 2) -> str:
    """``POLYGON ((x y, ...), (hole))`` in world coordinates (mosaic pixels times the downsample)."""
    rings = [to_world(contour.exterior, downsample)] + [to_world(hole, downsample) for hole in contour.holes]
    return "POLYGON (" + ", ".join(f"({ring_wkt(ring, precision)})" for ring in rings) + ")"
