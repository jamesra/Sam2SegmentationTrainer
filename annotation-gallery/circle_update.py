"""Keep ``circle_annotations.json`` in step with OData when Update All runs.

The exporter writes one circle per annotation: an ``annotated`` circle for a Viking circle location, and an
``inscribed`` circle (the largest circle inside the polygon) for a location that has a mask. Both change when the
annotation is edited in Viking, so Update All has to rewrite them. This module redoes just the affected locations'
entries in place and leaves every other entry alone.

The maths mirrors ``nornir_buildmanager.operations.segmentationtraining.circles``. That module cannot be imported
here: it needs scipy and the ingest chain, which the gallery image does not carry. ``tests/test_update_all.py``
pins the shared behavior (entry format, window conversion) so a drift shows up.

Entries are in crop pixels at the crop's own downsample, y down, and radii are the true radii
(``diameter_scale`` 1.0), exactly as the exporter writes them.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw

try:  # scipy supplies the distance transform; without it inscribed circles are not recomputed
    from scipy import ndimage
except ImportError:  # pragma: no cover - exercised only on an image built without scipy
    ndimage = None

CIRCLE_FILE = "circle_annotations.json"
CIRCLE_TYPE_CODE = 1
SOURCE_ANNOTATED = "annotated"
SOURCE_INSCRIBED = "inscribed"

MAX_RASTER_SIDE = 2048
MAX_RASTER_MAGNIFICATION = 8.0
_RASTER_PAD = 2
_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"

Ring = Sequence[tuple[float, float]]


def inscribed_available() -> bool:
    return ndimage is not None


@dataclass(frozen=True)
class Circle:
    """One circle in mosaic space at downsample 1."""

    location_id: int
    center_x: float
    center_y: float
    radius: float
    label: str
    has_polygon_mask: bool
    source: str


@dataclass(frozen=True)
class CropWindowRef:
    """A crop: its key, downsample, and pixel window (origin and size in downsampled pixels)."""

    image_key: str
    downsample: int
    origin_x: int
    origin_y: int
    width: int
    height: int


# --------------------------------------------------------------- circle geometry

def circle_geometry_from_wkt(wkt: str) -> tuple[float, float, float] | None:
    """Center and radius of a ``CURVEPOLYGON(CIRCULARSTRING(...))`` or ``CIRCLE((x y), r)`` shape."""
    text = wkt.strip()
    if text.upper().startswith("CIRCLE"):
        numbers = [float(token) for token in re.findall(_NUMBER, text)]
        if len(numbers) >= 3 and numbers[2] > 0:
            return numbers[0], numbers[1], numbers[2]
        return None
    start = text.find("(", text.upper().find("CIRCULARSTRING"))
    if "CIRCULARSTRING" not in text.upper() or start < 0:
        return None
    points: list[tuple[float, float]] = []
    for chunk in text[start + 1:].replace(")", " ").replace("(", " ").split(","):
        values = [float(token) for token in re.findall(_NUMBER, chunk)]
        if len(values) >= 2:
            points.append((values[0], values[1]))
    if len(points) < 3:
        return None
    center = _circumcenter(points[0], points[1], points[2])
    if center is None:
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        center = ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)
    radius = math.hypot(points[0][0] - center[0], points[0][1] - center[1])
    if radius <= 0:
        return None
    return center[0], center[1], radius


def _circumcenter(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]):
    determinant = 2.0 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    if abs(determinant) < 1e-9:
        return None
    a2 = a[0] ** 2 + a[1] ** 2
    b2 = b[0] ** 2 + b[1] ** 2
    c2 = c[0] ** 2 + c[1] ** 2
    ux = (a2 * (b[1] - c[1]) + b2 * (c[1] - a[1]) + c2 * (a[1] - b[1])) / determinant
    uy = (a2 * (c[0] - b[0]) + b2 * (a[0] - c[0]) + c2 * (b[0] - a[0])) / determinant
    return ux, uy


def annotated_circle(location_id: int, wkt: str, stored_radius: float | None, label: str) -> Circle | None:
    """Circle for a Viking circle location. The stored ``Radius`` wins when positive, else the WKT's."""
    shape = circle_geometry_from_wkt(wkt)
    if shape is None:
        return None
    center_x, center_y, wkt_radius = shape
    radius = float(stored_radius) if stored_radius is not None and stored_radius > 0 else wkt_radius
    return Circle(location_id, center_x, center_y, radius, label, False, SOURCE_ANNOTATED)


def max_inscribed_circle(polygons: Sequence[Sequence[Ring]]) -> tuple[float, float, float] | None:
    """Largest circle inside the polygon exteriors (holes ignored): ``(center_x, center_y, radius)``.

    The polygon is rasterized in mosaic space, the distance transform seeds the center, and a pattern search on the
    exact point-to-edge distance refines it. None for a degenerate polygon, or when scipy is not installed.
    """
    if ndimage is None:
        return None
    exteriors = [rings[0] for rings in polygons if rings and len(rings[0]) >= 3]
    if not exteriors:
        return None
    all_x = [pt[0] for ring in exteriors for pt in ring]
    all_y = [pt[1] for ring in exteriors for pt in ring]
    min_x, max_x, min_y, max_y = min(all_x), max(all_x), min(all_y), max(all_y)
    extent = max(max_x - min_x, max_y - min_y)
    if extent <= 0:
        return None
    scale = min(MAX_RASTER_MAGNIFICATION, (MAX_RASTER_SIDE - 2 * _RASTER_PAD - 1) / extent)
    width = int(math.ceil((max_x - min_x) * scale)) + 2 * _RASTER_PAD + 1
    height = int(math.ceil((max_y - min_y) * scale)) + 2 * _RASTER_PAD + 1
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    for ring in exteriors:
        draw.polygon([((x - min_x) * scale + _RASTER_PAD, (y - min_y) * scale + _RASTER_PAD) for x, y in ring], fill=1)
    mask = np.asarray(image, dtype=bool)
    if not mask.any():
        return None
    clearance = ndimage.distance_transform_edt(mask)
    row, col = np.unravel_index(int(np.argmax(clearance)), clearance.shape)
    seed_x = (col - _RASTER_PAD) / scale + min_x
    seed_y = (row - _RASTER_PAD) / scale + min_y
    center_x, center_y, radius = _refine_center(seed_x, seed_y, 1.0 / scale, _edge_arrays(exteriors))
    if radius <= 0:
        return None
    return center_x, center_y, radius


def _edge_arrays(rings: Sequence[Ring]):
    ax: list[float] = []
    ay: list[float] = []
    bx: list[float] = []
    by: list[float] = []
    for ring in rings:
        points = list(ring)
        if points[0] != points[-1]:
            points.append(points[0])
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            ax.append(x0)
            ay.append(y0)
            bx.append(x1)
            by.append(y1)
    return np.array(ax), np.array(ay), np.array(bx), np.array(by)


def _signed_clearance(x: float, y: float, edges) -> float:
    ax, ay, bx, by = edges
    dx = bx - ax
    dy = by - ay
    length_sq = dx * dx + dy * dy
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(length_sq > 0, ((x - ax) * dx + (y - ay) * dy) / length_sq, 0.0)
    t = np.clip(t, 0.0, 1.0)
    distance = float(np.min(np.hypot(x - (ax + t * dx), y - (ay + t * dy))))
    crossing = (ay > y) != (by > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_at_y = ax + (y - ay) * dx / np.where(dy != 0, dy, 1.0)
    inside = int(np.count_nonzero(crossing & (x < x_at_y))) % 2 == 1
    return distance if inside else -distance


def _refine_center(x: float, y: float, step: float, edges) -> tuple[float, float, float]:
    best = _signed_clearance(x, y, edges)
    step = max(step, 1e-6)
    for _ in range(100):
        improved = False
        for dx, dy in (
            (step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step),
            (step, step), (step, -step), (-step, step), (-step, -step),
        ):
            candidate = _signed_clearance(x + dx, y + dy, edges)
            if candidate > best:
                best, x, y, improved = candidate, x + dx, y + dy, True
        if not improved:
            step /= 2.0
            if step < 1e-3:
                break
    return x, y, best


def inscribed_circle(location_id: int, polygons: Sequence[Sequence[Ring]], label: str) -> Circle | None:
    found = max_inscribed_circle(polygons)
    if found is None:
        return None
    return Circle(location_id, found[0], found[1], found[2], label, True, SOURCE_INSCRIBED)


# ------------------------------------------------------------- crop-window entries

def entries_for_circle(circle: Circle, windows: Iterable[CropWindowRef]) -> dict[str, dict[str, Any]]:
    """One entry per crop window the circle reaches, in the exporter's format. ``{image_key: entry}``."""
    found: dict[str, dict[str, Any]] = {}
    for window in windows:
        scale = float(window.downsample)
        px = (circle.center_x - window.origin_x * scale) / scale
        py = (circle.center_y - window.origin_y * scale) / scale
        radius = circle.radius / scale
        nearest_x = min(max(px, 0.0), float(window.width))
        nearest_y = min(max(py, 0.0), float(window.height))
        if not math.hypot(px - nearest_x, py - nearest_y) < radius:
            continue
        found[window.image_key] = {
            "location_id": circle.location_id,
            "center_x": round(px, 3),
            "center_y": round(py, 3),
            "radius": round(radius, 3),
            "structure_label": circle.label,
            "has_polygon_mask": circle.has_polygon_mask,
            "source": circle.source,
        }
    return found


class CircleFile:
    """``circle_annotations.json`` held in memory so a run can replace many locations and write once."""

    def __init__(self, crops: str | os.PathLike[str]) -> None:
        self.path = Path(crops) / CIRCLE_FILE
        self.cells: dict[str, list[dict[str, Any]]] = {}
        self.exists = False
        self._by_location: dict[int, set[str]] = {}
        self._dirty = False
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(payload, dict) or int(payload.get("version", 0)) != 1:
            return
        cells = payload.get("grid_cells")
        if not isinstance(cells, dict):
            return
        self.exists = True
        self.cells = {str(key): list(entries) for key, entries in cells.items()}
        for key, entries in self.cells.items():
            for entry in entries:
                try:
                    self._by_location.setdefault(int(entry["location_id"]), set()).add(key)
                except (KeyError, TypeError, ValueError):
                    continue

    def replace(self, location_id: int, entries: dict[str, dict[str, Any]]) -> None:
        """Drop every entry of this location, then add the given ones (``{image_key: entry}``)."""
        location_id = int(location_id)
        for key in self._by_location.pop(location_id, set()):
            kept = [e for e in self.cells.get(key, []) if int(e.get("location_id", -1)) != location_id]
            if kept:
                self.cells[key] = kept
            else:
                self.cells.pop(key, None)
        for key, entry in entries.items():
            row = self.cells.setdefault(key, [])
            row.append(entry)
            row.sort(key=lambda item: item["location_id"])
            self._by_location.setdefault(location_id, set()).add(key)
        self._dirty = True

    def save(self) -> bool:
        """Write the file atomically when something changed. True when the file on disk is current."""
        if not self.exists or not self._dirty:
            return True
        document = {
            "version": 1,
            "diameter_scale": 1.0,
            "grid_cells": {key: self.cells[key] for key in sorted(self.cells)},
        }
        text = json.dumps(document, indent=1)
        temporary = self.path.with_suffix(".json.tmp")
        try:
            temporary.write_text(text, encoding="utf-8")
            os.replace(temporary, self.path)
        except OSError:
            temporary.unlink(missing_ok=True)
            return False
        self._dirty = False
        return True
