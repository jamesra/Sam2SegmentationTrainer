"""Location geometry for the converter: OGC WKT rings and Viking's closed Catmull-Rom curve fit.

Vendored from nornir-buildmanager's segmentationtraining ``wkt.py`` and ``curves.py`` (same author), because the gallery
image does not carry nornir. MosaicShape WKT stores **control points**; TypeCode 6 (curve polygon) and 7 (closed curve)
are restored to their outline with a centripetal Catmull-Rom fit, exactly as Viking does after load
(``CatmullRom.FitCurve``, 10 interpolations, extra samples where the turn exceeds 10 degrees). Circles (TypeCode 1) are a
centre point plus ``Radius``.

Do not edit the ported functions; change them upstream first so the three copies (Viking C#, nornir, here) agree.
"""

from __future__ import annotations

import math
import re
from typing import Sequence

# TypeCodes that cannot be a filled mask by themselves: point, circle, ellipse, polyline.
SKIP_MASK_TYPE_CODES = frozenset({0, 1, 2, 3})

Ring = tuple[tuple[float, float], ...]
PolygonRings = tuple[Ring, ...]  # exterior first, then holes

_COORD = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_SKIP_WKT_TYPES = frozenset({
    "POINT",
    "MULTIPOINT",
    "LINESTRING",
    "MULTILINESTRING",
    "POLYLINE",
    "MULTIPOLYLINE",
    "CIRCLE",
})


def is_curve_polygon(wkt: str) -> bool:
    """True for SQL CURVEPOLYGON/CIRCULARSTRING (Viking circles), not TypeCode 6."""
    return "CURVEPOLYGON" in wkt.upper() or "CIRCULARSTRING" in wkt.upper()


def wkt_type_token(wkt: str) -> str:
    """Leading WKT type word, e.g. POINT or POLYGON."""
    index = 0
    stripped = wkt.strip()
    while index < len(stripped) and (stripped[index].isalpha() or stripped[index] == "_"):
        index += 1
    return stripped[:index].upper()


def is_skippable_wkt(wkt: str) -> bool:
    """True for point, polyline, or circle WKT that cannot form a filled mask."""
    if is_curve_polygon(wkt):
        return True
    return wkt_type_token(wkt) in _SKIP_WKT_TYPES


def is_skippable_location(type_code: int | None, wkt: str) -> bool:
    """True when TypeCode or WKT is a point, polyline, or circle."""
    if type_code is not None and int(type_code) in SKIP_MASK_TYPE_CODES:
        return True
    return is_skippable_wkt(wkt)


def parse_wkt_polygons(wkt: str) -> list[PolygonRings]:
    """Return one (exterior, *holes) tuple per POLYGON in a WKT string.

    Point, polyline, and circle WKT return an empty list so a mis-coded
    annotation does not abort the section.
    """
    stripped = wkt.strip()
    if is_skippable_wkt(stripped):
        return []
    upper = stripped.upper()
    if upper.startswith("MULTIPOLYGON"):
        body = _paren_body(stripped, "MULTIPOLYGON")
        # Each part keeps its own outer parentheses, ``((exterior),(hole),...)``;
        # strip them so the part's rings split like a POLYGON body.
        return [_parse_polygon_body(part[1:-1]) for part in _split_top_level(body)]
    if upper.startswith("POLYGON"):
        body = _paren_body(stripped, "POLYGON")
        return [_parse_polygon_body(body)]
    raise ValueError(f"Unsupported WKT type: {wkt[:40]}")


def mosaic_wkt_from_odata_shape(shape: object) -> str | None:
    """Extract WellKnownText from an OData MosaicShape payload."""
    if shape is None:
        return None
    if isinstance(shape, str):
        return shape
    if not isinstance(shape, dict):
        return None
    geometry = shape.get("Geometry") or shape.get("geometry") or shape
    if isinstance(geometry, str):
        return geometry
    if isinstance(geometry, dict):
        wkt = geometry.get("WellKnownText") or geometry.get("wellKnownText")
        if isinstance(wkt, str) and wkt:
            return wkt
        if geometry.get("type") == "Polygon" and "coordinates" in geometry:
            return _geojson_polygon_to_wkt(geometry["coordinates"])
        if geometry.get("type") == "MultiPolygon" and "coordinates" in geometry:
            parts = [_geojson_polygon_to_wkt(poly) for poly in geometry["coordinates"]]
            interiors = ",".join(p[len("POLYGON"):] for p in parts)
            return f"MULTIPOLYGON({interiors})"
    return None


def _paren_body(wkt: str, keyword: str) -> str:
    start = wkt.upper().find(keyword)
    open_paren = wkt.find("(", start)
    if open_paren < 0:
        raise ValueError("WKT missing opening parenthesis")
    depth = 0
    for index, char in enumerate(wkt[open_paren:], start=open_paren):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return wkt[open_paren + 1:index]
    raise ValueError("WKT is not balanced")


def _split_top_level(body: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(body):
        if char == "(":
            if depth == 0:
                start = index
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                parts.append(body[start:index + 1])
    return parts


def _parse_polygon_body(body: str) -> PolygonRings:
    rings_src = _split_top_level(body)
    if not rings_src:
        raise ValueError("POLYGON has no rings")
    rings = tuple(_parse_ring(ring) for ring in rings_src)
    return rings


def _parse_ring(ring: str) -> Ring:
    inner = ring.strip()
    if inner.startswith("(") and inner.endswith(")"):
        inner = inner[1:-1]
    numbers = [float(match.group(0)) for match in _COORD.finditer(inner)]
    if len(numbers) < 6 or len(numbers) % 2:
        raise ValueError("Ring does not contain coordinate pairs")
    pairs = [(numbers[i], numbers[i + 1]) for i in range(0, len(numbers), 2)]
    return tuple(pairs)


def _geojson_polygon_to_wkt(coordinates: Sequence) -> str:
    ring_text = []
    for ring in coordinates:
        coords = ",".join(f"{pt[0]} {pt[1]}" for pt in ring)
        ring_text.append(f"({coords})")
    return "POLYGON(" + ",".join(ring_text) + ")"


# Matches Geometry.Global.Epsilon (float) used by GridVector2 equality.
EPSILON = 0.001
EPSILON_SQUARED = EPSILON * EPSILON

# SqlGeometryAnnotationExtensions.ShapeSmoothingExtensions for closed rings.
NUM_CLOSED_CURVE_INTERPOLATIONS = 10

_CURVATURE_ANGLE_DEG = 10.0
_CURVATURE_DISTANCE_THRESHOLD = 0.0625  # 0.25² in TryAddTPointsAboveThreshold
_MAX_CURVE_RECURSION = 32

Point = tuple[float, float]


def hydrate_polygons(polygons: list[PolygonRings], type_code: int | None) -> list[PolygonRings]:
    """Fit closed Catmull-Rom curves when *type_code* is CURVEPOLYGON (6)."""
    if type_code not in (6, 7):
        return polygons
    return [tuple(fit_closed_curve_ring(ring) for ring in rings) for rings in polygons]


def fit_closed_curve_ring(ring: Ring, num_interpolations: int = NUM_CLOSED_CURVE_INTERPOLATIONS) -> Ring:
    """Return a closed ring after Viking ``CalculateCurvePoints(..., closeCurve=true)``."""
    points = list(ring)
    if len(points) < 3:
        return ring
    fitted = calculate_closed_curve_points(points, num_interpolations)
    if not fitted:
        return ring
    if not _points_equal(fitted[0], fitted[-1]):
        fitted.append(fitted[0])
    else:
        fitted[-1] = fitted[0]
    return tuple(fitted)


def calculate_closed_curve_points(control_points: Sequence[Point], num_interpolations: int) -> list[Point]:
    """Port of ``CurveExtensions.CalculateClosedCurvePoints``."""
    if len(control_points) <= 2:
        return list(control_points)
    if num_interpolations == 0:
        return _ensure_closed(list(control_points))
    fitted = fit_curve(control_points, num_interpolations, closed=True)
    if fitted:
        fitted[-1] = fitted[0]
    return fitted


def fit_curve(
    control_points: Sequence[Point],
    num_interpolations: int,
    *,
    closed: bool,
) -> list[Point]:
    """Port of ``Geometry.CatmullRom.FitCurve``."""
    if len(control_points) <= 2 or num_interpolations == 0:
        return list(control_points)
    padded = (
        _control_points_for_closed_curve(control_points)
        if closed
        else _control_points_for_open_curve(control_points)
    )
    samples: list[Point] = []
    for index in range(len(padded) - 3):
        samples.extend(
            _recursively_fit_curve_segment(
                padded[index],
                padded[index + 1],
                padded[index + 2],
                padded[index + 3],
                num_interpolations,
            )
        )
    return _remove_adjacent_duplicates_preserving(samples, control_points)


def _control_points_for_closed_curve(control_points: Sequence[Point]) -> list[Point]:
    """Port of ``CatmullRom.GetControlPointsForClosedCurve``.

    Four-point windows only emit the span between the middle pair, so a closed
    ring A-B-C-D-A is padded to D-A-B-C-D-A-B.
    """
    points = list(control_points)
    index = 1
    while index < len(points) - 1:
        if _distance_squared(points[index - 1], points[index]) < EPSILON_SQUARED:
            del points[index]
            continue
        index += 1
    if not _points_equal(points[0], points[-1]):
        points.insert(0, points[-1])
    after_start = points[1]
    before_start = points[-2]
    points.insert(0, before_start)
    points.append(after_start)
    return points


def _control_points_for_open_curve(control_points: Sequence[Point]) -> list[Point]:
    points = list(control_points)
    points.insert(0, _starting_point_for_open_curve(points))
    points.append(_ending_point_for_open_curve(points))
    return points


def _starting_point_for_open_curve(points: Sequence[Point]) -> Point:
    return _point_along_line(points[0], points[1], -0.5)


def _ending_point_for_open_curve(points: Sequence[Point]) -> Point:
    return _point_along_line(points[-2], points[-1], 1.5)


def _recursively_fit_curve_segment(
    p0: Point,
    p1: Point,
    p2: Point,
    p3: Point,
    num_interpolations: int,
) -> list[Point]:
    if num_interpolations <= 1:
        t_points = {0.5} if num_interpolations == 1 else {0.0, 1.0}
    else:
        t_points = {index / (num_interpolations - 1.0) for index in range(num_interpolations)}
    return _recursively_fit_curve_segment_t(p0, p1, p2, p3, t_points, depth=0)


def _recursively_fit_curve_segment_t(
    p0: Point,
    p1: Point,
    p2: Point,
    p3: Point,
    t_points: set[float],
    *,
    depth: int,
) -> list[Point]:
    ordered = sorted(t_points)
    t1, t2, t3 = _centripetal_knots(p0, p1, p2, p3)
    t_values = [t1 + scalar * (t2 - t1) for scalar in ordered]
    output = _fit_curve_segment_with_t_values(p0, p1, p2, p3, t0=0.0, t1=t1, t2=t2, t3=t3, t_values=t_values)
    if depth >= _MAX_CURVE_RECURSION:
        return output
    added, updated = _try_add_t_points_above_threshold(output, ordered)
    if not added:
        return output
    return _recursively_fit_curve_segment_t(p0, p1, p2, p3, updated, depth=depth + 1)


def _centripetal_knots(p0: Point, p1: Point, p2: Point, p3: Point) -> tuple[float, float, float]:
    t1 = _tj(0.0, p0, p1)
    t2 = _tj(t1, p1, p2)
    t3 = _tj(t2, p2, p3)
    return t1, t2, t3


def _tj(ti: float, pi: Point, pj: Point, alpha: float = 0.5) -> float:
    return (math.dist(pi, pj) ** alpha) + ti


def _fit_curve_segment_with_t_values(
    p0: Point,
    p1: Point,
    p2: Point,
    p3: Point,
    *,
    t0: float,
    t1: float,
    t2: float,
    t3: float,
    t_values: Sequence[float],
) -> list[Point]:
    """Port of ``CatmullRom.FitCurveSegmentWithTValues`` (centripetal)."""
    output: list[Point] = []
    for t in t_values:
        a1x = ((t1 - t) / (t1 - t0)) * p0[0] + ((t - t0) / (t1 - t0)) * p1[0]
        a1y = ((t1 - t) / (t1 - t0)) * p0[1] + ((t - t0) / (t1 - t0)) * p1[1]
        a2x = ((t2 - t) / (t2 - t1)) * p1[0] + ((t - t1) / (t2 - t1)) * p2[0]
        a2y = ((t2 - t) / (t2 - t1)) * p1[1] + ((t - t1) / (t2 - t1)) * p2[1]
        a3x = ((t3 - t) / (t3 - t2)) * p2[0] + ((t - t2) / (t3 - t2)) * p3[0]
        a3y = ((t3 - t) / (t3 - t2)) * p2[1] + ((t - t2) / (t3 - t2)) * p3[1]
        b1x = ((t2 - t) / (t2 - t0)) * a1x + ((t - t0) / (t2 - t0)) * a2x
        b1y = ((t2 - t) / (t2 - t0)) * a1y + ((t - t0) / (t2 - t0)) * a2y
        b2x = ((t3 - t) / (t3 - t1)) * a2x + ((t - t1) / (t3 - t1)) * a3x
        b2y = ((t3 - t) / (t3 - t1)) * a2y + ((t - t1) / (t3 - t1)) * a3y
        cx = ((t2 - t) / (t2 - t1)) * b1x + ((t - t1) / (t2 - t1)) * b2x
        cy = ((t2 - t) / (t2 - t1)) * b1y + ((t - t1) / (t2 - t1)) * b2y
        output.append((cx, cy))
    return output


def _try_add_t_points_above_threshold(
    output: list[Point],
    t_points: list[float],
    angle_threshold_degrees: float = _CURVATURE_ANGLE_DEG,
) -> tuple[bool, set[float]]:
    """Port of ``CurveExtensions.TryAddTPointsAboveThreshold``."""
    points = list(output)
    scalars = list(t_points)
    index = 1
    while index < len(points) - 1:
        if (
            _distance_squared(points[index - 1], points[index]) < EPSILON_SQUARED
            or _distance_squared(points[index], points[index + 1]) < EPSILON_SQUARED
        ):
            del points[index]
            del scalars[index]
            continue
        index += 1
    degrees = [abs(angle) for angle in _measure_curvature(points)]
    threshold = (math.pi * 2.0 / 360.0) * angle_threshold_degrees
    needs = [False] * len(scalars)
    for index in range(len(scalars) - 2, 0, -1):
        if degrees[index] > threshold:
            distance = _distance_squared(points[index - 1], points[index]) + _distance_squared(
                points[index], points[index + 1]
            )
            needs[index] = distance > _CURVATURE_DISTANCE_THRESHOLD
    updated = set(scalars)
    starting = len(updated)
    for index in range(len(needs) - 1):
        if needs[index] or needs[index + 1]:
            updated.add((scalars[index] + scalars[index + 1]) / 2.0)
    return starting != len(updated), updated


def _measure_curvature(points: Sequence[Point]) -> list[float]:
    """Port of ``CurveExtensions.MeasureCurvature`` (turn from the incoming chord)."""
    angles = [0.0] * len(points)
    if len(points) < 3:
        return angles
    for index in range(1, len(points) - 1):
        if _exact_equal(points[index - 1], points[index]):
            raise ValueError("Duplicate points found in MeasureCurvature")
        if _distance_squared(points[index], points[index - 1]) < EPSILON_SQUARED:
            angles[index] = 0.0
            continue
        origin = points[index]
        incoming = _point_along_line(points[index - 1], points[index], 2.0)
        angles[index] = _arc_angle(origin, incoming, points[index + 1])
    return angles


def _arc_angle(origin: Point, a: Point, b: Point) -> float:
    """Port of ``GridVector2.ArcAngle`` (counter-clockwise negative)."""
    ux, uy = a[0] - origin[0], a[1] - origin[1]
    vx, vy = b[0] - origin[0], b[1] - origin[1]
    angle = math.atan2(uy, ux) - math.atan2(vy, vx)
    if angle <= -math.pi:
        angle += math.pi * 2
    elif angle > math.pi:
        angle -= math.pi * 2
    return angle


def _remove_adjacent_duplicates_preserving(
    points: Sequence[Point],
    preserved: Sequence[Point],
) -> list[Point]:
    """Port of ``RemoveAdjacentDuplicates(points, preserved_path)``."""
    if not points:
        raise ValueError("Must have at least one point to remove adjacent duplicates")
    preserved_iter = iter(preserved)
    preserved_point = next(preserved_iter, None)
    result: list[Point] = []
    iterator = iter(points)
    current = next(iterator)
    nxt: Point | None = None
    advanced = False
    for nxt in iterator:
        advanced = True
        if preserved_point is not None and _exact_equal(current, preserved_point):
            while nxt is not None and _points_equal(nxt, preserved_point):
                nxt = next(iterator, None)
            result.append(current)
            preserved_point = next(preserved_iter, None)
        elif preserved_point is not None and _points_equal(current, preserved_point):
            if nxt is None or not _points_equal(nxt, preserved_point):
                result.append(preserved_point)
                preserved_point = next(preserved_iter, None)
        elif nxt is None or not _points_equal(current, nxt):
            result.append(current)
        if nxt is None:
            break
        current = nxt
    if not advanced:
        result.append(current)
    elif nxt is not None:
        result.append(nxt)
    return result


def _ensure_closed(points: list[Point]) -> list[Point]:
    if not points:
        return points
    if not _points_equal(points[0], points[-1]):
        points.append(points[0])
    return points


def _point_along_line(a: Point, b: Point, fraction: float) -> Point:
    return (a[0] + (b[0] - a[0]) * fraction, a[1] + (b[1] - a[1]) * fraction)


def _distance_squared(a: Point, b: Point) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def _points_equal(a: Point, b: Point) -> bool:
    return _distance_squared(a, b) <= EPSILON_SQUARED


def _exact_equal(a: Point, b: Point) -> bool:
    return a[0] == b[0] and a[1] == b[1]
