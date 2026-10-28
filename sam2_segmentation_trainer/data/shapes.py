"""Adjacent-section shapes from the exporter's ``shapes.jsonl``, for the ``adjacent_poly_seed`` prompt family.

The exporter (nornir-buildmanager ``segmentationtraining/shapes.py``) writes ``{volume}/shapes.jsonl`` next to
``manifest.jsonl``: one line per polygon location that has a mask or is linked to one::

    {"location_id": 8, "z": 3, "structure_id": 77, "structure_label": "BC", "has_mask": true,
     "linked": [[7, 2], [9, 4]], "centroid": [x, y], "area": 1234.5,
     "polygons": [[[[dx, dy], ...exterior...], [[dx, dy], ...hole...]], ...]}

``centroid`` and ``area`` are in mosaic (volume) pixels at downsample 1, y down; every polygon point is relative to
the centroid. ``linked`` lists ``[location_id, z]`` of the same structure's locations within two sections.

:func:`load_volume_shapes` keeps only what the trainer can use: the links of each training target to a neighbour in
another section (``1 <= |dz| <= 2``) that has an outline, and the outlines of those neighbours. A line with an
empty ``linked`` list is skipped without being parsed (links are symmetric, so such a location is neither a target
with a neighbour nor anybody's neighbour), which on a volume with few links (RC1: 3 of 15,867) skips almost the whole
file. The outlines are packed into a few flat numpy arrays rather than one Python list per point, so the index is
small and DataLoader workers forked from the main process share its pages instead of copying them.

Coordinates: a crop's image key ends ``_D{d}_X{x0}-{x1}_Y{y0}-{y1}`` with the window in pixels at the crop's own
downsample ``d`` (``x0`` can be negative where the window hangs over the mosaic edge), so a volume point ``(X, Y)``
lands at tile pixel ``(X / d - x0, Y / d - y0)``, the same conversion the exporter uses for circle entries. This was
checked on the v3 exports: each target's own outline rasterised this way matches its mask PNG at IoU 0.97-1.0 (and
not when y is flipped).

Placement. The exporter's coordinates are each section's own MOSAIC pixels, not registered volume pixels: on the
v3 exports a linked neighbour's centroid lies a median 23,000 px (RC2; RPC1 9,000, RPC2 10,600) from its target's,
and under 1% of links are within 256 px, so an outline drawn at its raw position almost never reaches the target's
tile. :data:`PLACEMENT_PROJECTED` (the default, :mod:`~sam2_segmentation_trainer.data.projection`) carries the
outline through the volume's registration (neighbour stos, residual correction, inverse target stos) into the
target's mosaic, where an annotator's carried-forward polygon lands; it is used at that position, eroded, and only
when enough of it lies on the target (:func:`draw_projected`). A volume without registration data falls back to
:data:`PLACEMENT_CENTROID`: the outline is translated so its centroid sits on the target's own centroid (from the
target's ``shapes.jsonl`` record), and :func:`~sam2_segmentation_trainer.data.refine.fit_adjacent_poly_seed` then
re-centres and scales it, so only the neighbour's shape reaches the prompt. :data:`PLACEMENT_RAW` draws it at its
raw coordinates (a test hook).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Collection

import numpy as np
from PIL import Image, ImageDraw

SHAPES_FILENAME = "shapes.jsonl"
"""The exporter's file name, in the volume's crops folder (beside ``manifest.jsonl``)."""

LINK_MAX_DZ = 2
"""A neighbour must lie 1 to this many sections from the target (the exporter keeps the same range)."""

PLACEMENT_CENTROID = "centroid"
"""Place the neighbour's outline with its centroid on the target's centroid (translation only; the default)."""

PLACEMENT_RAW = "raw"
"""Place the neighbour's outline at its own coordinates, as written (unregistered section mosaic pixels)."""

PLACEMENT_PROJECTED = "projected"
"""Carry the neighbour's outline into the target's mosaic through the registration (the default); a volume without
the data uses :data:`PLACEMENT_CENTROID`."""

PLACEMENTS = (PLACEMENT_CENTROID, PLACEMENT_RAW)
"""Placements :func:`pick_neighbour_raster` draws (the projected one is :func:`draw_projected`)."""

ADJACENT_PLACEMENTS = (PLACEMENT_PROJECTED, PLACEMENT_CENTROID, PLACEMENT_RAW)
"""Values of ``prompt.adjacent_poly.placement``."""

_EMPTY_LINKS = '"linked":[]'
"""How the exporter writes a location without links (compact JSON separators); such a line is skipped unparsed."""

_WINDOW_RE = re.compile(r"_D(?P<d>\d+)_X(?P<x0>-?\d+)-(?P<x1>-?\d+)_Y(?P<y0>-?\d+)-(?P<y1>-?\d+)$")


def tile_window(image_key: str) -> tuple[int, int, int] | None:
    """``(downsample, x0, y0)`` of a crop from its image key (pixels at the crop's downsample), or None."""
    match = _WINDOW_RE.search(image_key)
    if match is None:
        return None
    downsample = int(match["d"])
    if downsample <= 0:
        return None
    return downsample, int(match["x0"]), int(match["y0"])


@dataclass(slots=True)
class VolumeShapes:
    """The outlines and links of one volume that adjacent-section seeds can use. Read-only after loading.

    Row ``r`` is one kept outline: ``z[r]``, ``centroid[r]`` (volume pixels, float64) and rings
    ``ring_start[r]:ring_start[r + 1]``; ring ``i`` is ``points[ring_offsets[i]:ring_offsets[i + 1]]`` (float32,
    relative to the centroid), an exterior ring when ``ring_exterior[i]`` else a hole of the exterior before it.
    ``links`` maps a target location id to the rows of its usable neighbours, nearest section first,
    ``target_centroids`` each such target to its own centroid (volume pixels) and ``target_z`` to its section.
    """

    z: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    centroid: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float64))
    ring_start: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.int64))
    ring_offsets: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.int64))
    ring_exterior: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))
    location_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    links: dict[int, tuple[int, ...]] = field(default_factory=dict)
    target_centroids: dict[int, tuple[float, float]] = field(default_factory=dict)
    target_z: dict[int, int] = field(default_factory=dict)
    projected: Any = None
    """The neighbours' outlines carried into each target's own mosaic frame
    (:class:`~sam2_segmentation_trainer.data.projection.ProjectedOutlines`), or None when the volume has no
    registration data (the centroid placement is used)."""

    def __len__(self) -> int:
        return int(self.z.shape[0])

    def neighbour_rows(self, location_id: int) -> tuple[int, ...]:
        """Rows of the target's linked outlines in sections ``1 <= |dz| <= 2`` (empty when it has none)."""
        return self.links.get(int(location_id), ())

    def rings(self, row: int) -> list[tuple[np.ndarray, bool]]:
        """The rings of ``row`` in volume pixels, as ``(points (n, 2) float64, is_exterior)``, in file order."""
        centre = self.centroid[row]
        out: list[tuple[np.ndarray, bool]] = []
        for ring in range(int(self.ring_start[row]), int(self.ring_start[row + 1])):
            pts = self.points[self.ring_offsets[ring] : self.ring_offsets[ring + 1]].astype(np.float64) + centre
            out.append((pts, bool(self.ring_exterior[ring])))
        return out

    def rasterize(
        self,
        row: int,
        image_key: str,
        shape: tuple[int, int],
        offset: tuple[float, float] = (0.0, 0.0),
    ) -> np.ndarray | None:
        """Boolean ``shape`` raster of ``row``, moved by ``offset`` (volume pixels), in the pixel frame of the crop
        ``image_key``, or None when the key has no window. Exterior rings are filled and holes cleared, in file
        order; a part outside the crop is clipped (an outline wholly outside gives an all-False raster)."""
        window = tile_window(image_key)
        if window is None:
            return None
        downsample, x0, y0 = window
        shift = np.asarray(offset, dtype=np.float64)
        height, width = int(shape[0]), int(shape[1])
        canvas = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(canvas)
        for pts, exterior in self.rings(row):
            if len(pts) < 3:
                continue
            tile = (pts + shift) / float(downsample) - np.array([x0, y0], dtype=np.float64)
            draw.polygon([(float(x), float(y)) for x, y in tile], fill=255 if exterior else 0)
        return np.asarray(canvas) > 0


def _usable_rings(polygons) -> list[tuple[list, bool]]:
    rings: list[tuple[list, bool]] = []
    for polygon in polygons or ():
        for index, ring in enumerate(polygon or ()):
            if ring and len(ring) >= 3:
                rings.append((ring, index == 0))
    return rings


def load_volume_shapes(path: Path, targets: Collection[int] | None = None) -> VolumeShapes | None:
    """Read one ``shapes.jsonl``; None when the file is missing or unreadable.

    ``targets`` are the location ids that can be training targets (the inventory's ids for this volume; None means
    every ``has_mask`` record). Only their links to a neighbour 1-2 sections away with a non-empty outline are kept,
    and only those neighbours' outlines. Malformed lines are skipped.
    """
    try:
        handle = Path(path).open(encoding="utf-8")
    except OSError:
        return None
    wanted = None if targets is None else {int(t) for t in targets}
    records: dict[int, tuple[int, list[tuple[list, bool]], tuple[float, float]]] = {}
    target_links: dict[int, tuple[int, list[tuple[int, int]]]] = {}
    target_centres: dict[int, tuple[float, float]] = {}
    with handle:
        for line in handle:
            if not line.strip() or _EMPTY_LINKS in line:
                continue
            try:
                record = json.loads(line)
                location_id = int(record["location_id"])
                z = int(record["z"])
                linked = [(int(other), int(oz)) for other, oz in record.get("linked") or ()]
            except (ValueError, TypeError, KeyError):
                continue
            if not linked:
                continue
            is_target = bool(record.get("has_mask")) if wanted is None else location_id in wanted
            if is_target:
                near = [(other, oz) for other, oz in linked if 1 <= abs(oz - z) <= LINK_MAX_DZ]
                if near:
                    target_links[location_id] = (z, near)
            rings = _usable_rings(record.get("polygons"))
            try:
                cx, cy = (float(v) for v in record["centroid"])
            except (KeyError, TypeError, ValueError):
                target_links.pop(location_id, None)
                continue
            if location_id in target_links:
                target_centres[location_id] = (cx, cy)
            if rings:
                records[location_id] = (z, rings, (cx, cy))

    needed: set[int] = set()
    for location_id, (z, near) in target_links.items():
        for other, oz in near:
            found = records.get(other)
            if found is not None and found[0] == oz:
                needed.add(other)
    rows = sorted(needed, key=lambda lid: (records[lid][0], lid))
    row_of = {lid: row for row, lid in enumerate(rows)}

    ring_counts: list[int] = []
    lengths: list[int] = []
    exterior: list[bool] = []
    chunks: list[np.ndarray] = []
    for lid in rows:
        rings = records[lid][1]
        ring_counts.append(len(rings))
        for ring, is_exterior in rings:
            pts = np.asarray(ring, dtype=np.float32).reshape(-1, 2)
            chunks.append(pts)
            lengths.append(len(pts))
            exterior.append(is_exterior)
    links: dict[int, tuple[int, ...]] = {}
    for location_id, (z, near) in target_links.items():
        usable = sorted(
            (row_of[other] for other, oz in near if other in row_of),
            key=lambda row: (abs(records[rows[row]][0] - z), records[rows[row]][0], rows[row]),
        )
        if usable:
            links[location_id] = tuple(usable)
    return VolumeShapes(
        z=np.array([records[lid][0] for lid in rows], dtype=np.int32),
        centroid=np.array([records[lid][2] for lid in rows], dtype=np.float64).reshape(-1, 2),
        ring_start=np.concatenate([[0], np.cumsum(ring_counts, dtype=np.int64)]).astype(np.int64),
        ring_offsets=np.concatenate([[0], np.cumsum(lengths, dtype=np.int64)]).astype(np.int64),
        ring_exterior=np.array(exterior, dtype=bool),
        points=np.concatenate(chunks) if chunks else np.zeros((0, 2), dtype=np.float32),
        location_ids=np.array(rows, dtype=np.int64),
        links=links,
        target_centroids={lid: target_centres[lid] for lid in links},
        target_z={lid: int(target_links[lid][0]) for lid in links},
    )


def pick_neighbour_raster(
    shapes: VolumeShapes,
    image_key: str,
    location_id: int,
    shape: tuple[int, int],
    rng: np.random.Generator | None = None,
    placement: str = PLACEMENT_CENTROID,
) -> np.ndarray | None:
    """Raster of ONE linked neighbour outline of ``location_id`` in the crop's pixel frame, or None.

    The neighbours are tried in a random order from ``rng`` (uniform over the 1-2-section neighbours; without
    ``rng``, nearest section first, then lower z, then lower id) and the first one that reaches the crop is
    returned. ``placement`` is :data:`PLACEMENT_CENTROID` (centroid on the target's centroid) or
    :data:`PLACEMENT_RAW` (as written). None when the target has no usable neighbour, the key has no window, or no
    neighbour reaches the crop.
    """
    if placement not in PLACEMENTS:
        raise ValueError(f"placement must be one of {PLACEMENTS}, got {placement!r}")
    rows = shapes.neighbour_rows(location_id)
    if not rows:
        return None
    target = shapes.target_centroids.get(int(location_id))
    order = range(len(rows)) if rng is None or len(rows) == 1 else rng.permutation(len(rows))
    for i in order:
        row = rows[int(i)]
        offset = (0.0, 0.0)
        if placement == PLACEMENT_CENTROID and target is not None:
            offset = (target[0] - float(shapes.centroid[row, 0]), target[1] - float(shapes.centroid[row, 1]))
        raster = shapes.rasterize(row, image_key, shape, offset)
        if raster is None:
            return None
        if raster.any():
            return raster
    return None


@dataclass(slots=True)
class AdjacentDraw:
    """One adjacent-section outline drawn for a sample, in the crop's pixel frame (before augmentation).

    A ``projected`` draw is used where it lies (``raster`` is the eroded outline) and only when it ``passed``: at
    least ``overlap_min`` of the outline's area in the tile lies on the target mask (``overlap``). An unprojected
    draw (the centroid fallback) is fitted inside the target later; it has ``passed`` True and ``overlap`` None.
    """

    raster: np.ndarray
    projected: bool
    passed: bool = True
    overlap: float | None = None
    erode_px: float = 0.0


def _rings_canvas(
    rings: list[tuple[np.ndarray, bool]], window: tuple[int, int, int], shape: tuple[int, int], pad: int
) -> np.ndarray:
    """Boolean raster of rings (mosaic px) over the crop grown by ``pad`` pixels on every side."""
    downsample, x0, y0 = window
    height, width = int(shape[0]) + 2 * pad, int(shape[1]) + 2 * pad
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    origin = np.array([x0 - pad, y0 - pad], dtype=np.float64)
    for pts, exterior in rings:
        if len(pts) < 3:
            continue
        tile = np.asarray(pts, dtype=np.float64) / float(downsample) - origin
        draw.polygon([(float(x), float(y)) for x, y in tile], fill=255 if exterior else 0)
    return np.asarray(canvas) > 0


def erode_outline(canvas: np.ndarray, erode_px: float, max_share: float = 0.6) -> np.ndarray:
    """``canvas`` shrunk by ``erode_px`` pixels (Euclidean), never by more than ``max_share`` of its own depth so it
    cannot vanish. The canvas edge is not background (an outline that runs off it continues), so a part cut by the
    edge is eroded only from its real outline.

    Only the outline's bounding box plus one pixel is transformed (the result is identical: the nearest background
    pixel of an outline pixel is never outside that window, see ``annotation_crops.prompt_encoding.edge_distance``),
    about 3x faster than the whole padded canvas."""
    if erode_px <= 0.0 or not canvas.any():
        return canvas
    from scipy.ndimage import distance_transform_edt

    from annotation_crops.prompt_encoding import window

    box = window(canvas, 1)
    depth = distance_transform_edt(canvas[box])
    out = np.zeros(canvas.shape, dtype=bool)
    out[box] = depth > min(float(erode_px), float(depth.max()) * max_share)
    return out


def draw_projected(
    shapes: VolumeShapes,
    image_key: str,
    location_id: int,
    target: np.ndarray,
    rng: np.random.Generator | None = None,
    *,
    overlap_min: float = 0.6,
    erode_px: tuple[float, float] = (0.0, 0.0),
) -> AdjacentDraw | None:
    """ONE linked neighbour outline of ``location_id``, carried into the target's frame (``shapes.projected``),
    eroded by ``U(erode_px)`` pixels and tested against the target mask, or None.

    The neighbours are tried in a random order from ``rng`` (as :func:`pick_neighbour_raster`); the first whose
    projected outline reaches the crop is drawn, then its erosion is drawn from ``rng``. The rule is the pull-through
    circle's: ``overlap`` is the share of the eroded outline's area inside the tile that lies on ``target`` (the
    mask is only known inside the tile), and the draw ``passed`` when it is at least ``overlap_min``. None when the
    volume has no projection, no neighbour of the target projected, the key has no window, or no projected outline
    reaches the crop.
    """
    projected = shapes.projected
    if projected is None:
        return None
    window = tile_window(image_key)
    if window is None:
        return None
    entries = projected.entries(int(location_id))
    if not entries:
        return None
    mask = np.asarray(target, dtype=bool)
    height, width = mask.shape
    pad = int(np.ceil(max(erode_px))) + 2
    order = range(len(entries)) if rng is None or len(entries) == 1 else rng.permutation(len(entries))
    for i in order:
        entry = entries[int(i)]
        if entry < 0:
            continue
        canvas = _rings_canvas(projected.rings(entry), window, (height, width), pad)
        if not canvas[pad : pad + height, pad : pad + width].any():
            continue
        erode = float(rng.uniform(*erode_px)) if rng is not None else float(erode_px[0])
        raster = erode_outline(canvas, erode)[pad : pad + height, pad : pad + width]
        area = int(raster.sum())
        if area == 0:
            continue
        overlap = float((raster & mask).sum()) / area
        return AdjacentDraw(
            raster=raster.copy(), projected=True, passed=overlap >= overlap_min, overlap=overlap, erode_px=erode
        )
    return None
