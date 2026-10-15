"""Negative context for SAM2 prompts: other annotations in the same AnnotationCrops tile.

Shared by the trainer (which turns these into a signed dense mask prompt and background points) and the
annotation gallery (which draws them). Depends on numpy and Pillow only, so the gallery image needs neither
torch nor scipy.

A tile is one ``imageKey``. Its neighbors are the other locations that have a raster mask in
``masks/{imageKey}_{locationId}.png`` plus the circle annotations in ``circle_annotations.json`` that have no
mask of their own in this tile.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Collection, Iterable, Sequence

import numpy as np
from PIL import Image

from annotation_crops.prompt_encoding import DEFAULT_CIRCLE_TRUST, CircleAnnotationTrust, circle_negative_canvas

CIRCLE_FILE_NAME = "circle_annotations.json"
#: Fraction of a circle's true diameter that a prompt circle covers (training default, ``prompt.circle_diameter_scale``).
DEFAULT_CIRCLE_DIAMETER_SCALE = 0.95
"""File the exporter writes in ``AnnotationCrops/``: circle annotations keyed by ``imageKey``."""

KIND_POLYGON = "polygon"
KIND_CIRCLE = "circle"

SOURCE_ANNOTATED = "annotated"
SOURCE_INSCRIBED = "inscribed"

_MASK_THRESHOLD = 127


@dataclass(frozen=True, slots=True)
class CircleAnnotation:
    """One circle in a tile, in that tile's pixel space (x right, y down, origin at the top-left corner).

    ``source`` is ``"annotated"`` for a location that is still a circle and ``"inscribed"`` for the largest
    circle inside a polygon location. An annotated circle's ``radius`` is its TRUE radius (its radii are unreliable,
    so it is only ever a negative, scaled by :class:`CircleAnnotationTrust`); an inscribed circle's ``radius`` is
    already multiplied by the caller's diameter scale. The center may lie outside the tile when only part of the
    circle falls in it.
    """

    location_id: int
    center_x: float
    center_y: float
    radius: float
    structure_label: str = ""
    has_polygon_mask: bool = False
    source: str = SOURCE_ANNOTATED

    def center_in(self, shape: tuple[int, int]) -> bool:
        """Whether the center lies inside an image of ``shape`` (height, width)."""
        height, width = shape
        return 0.0 <= self.center_x < float(width) and 0.0 <= self.center_y < float(height)

    def seed_radius(self, scale: float) -> float:
        """The radius of this circle drawn as a prompt seed: ``scale`` times the true radius.

        An annotated circle is stored at its true radius, so the diameter scale applies here; an inscribed one has
        it applied already.
        """
        return self.radius * float(scale) if self.source == SOURCE_ANNOTATED else self.radius


@dataclass(eq=False)
class NegativeRegion:
    """One other annotation in the tile: its location id, whether it came from a polygon mask or a circle,
    and a boolean raster of shape ``(height, width)``. Callers must not modify ``mask``.

    ``weight`` is set only for an annotated circle: its float32 trust map ``w`` (see
    :func:`annotation_crops.prompt_encoding.circle_trust`), with ``mask`` the region it claims. Its core is
    ``weight >= 1``. Both go through the same geometric augmentation as the image.
    """

    location_id: int
    kind: str
    mask: np.ndarray
    weight: np.ndarray | None = None

    @property
    def core(self) -> np.ndarray | None:
        """The fully trusted pixels of an annotated circle, or None for any other region."""
        return None if self.weight is None else self.weight >= 1.0


def circle_file_path(crops: Path) -> Path:
    """Location of the circle file inside an ``AnnotationCrops`` folder."""
    return crops / CIRCLE_FILE_NAME


def circle_file_stamp(crops: Path) -> int | None:
    """Nanosecond mtime of the circle file, or None when it is absent. Callers use it to invalidate caches."""
    try:
        return circle_file_path(crops).stat().st_mtime_ns
    except OSError:
        return None


def load_circle_annotations(
    crops: Path, scale: float | None = None
) -> dict[str, tuple[CircleAnnotation, ...]]:
    """Read ``circle_annotations.json`` into ``{imageKey: circles}``.

    The exporter never scales: it writes true radii and records ``diameter_scale`` 1.0; files from older exporters
    have the scale baked into ``radius``. Annotated circles are loaded at their TRUE radius (divided by the file's
    ``diameter_scale`` for an old file). Inscribed circles are the caller's seed circles: ``scale`` is the diameter
    scale it wants and each of their radii is multiplied by ``scale / diameter_scale``; None keeps them as written.

    A missing or unreadable file, or a version this code does not know, yields an empty mapping: circles are
    optional context, so training and the gallery run without them. Malformed entries are skipped.
    """
    try:
        payload = json.loads(circle_file_path(crops).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict) or int(payload.get("version", 0)) != 1:
        return {}
    cells = payload.get("grid_cells")
    if not isinstance(cells, dict):
        return {}
    try:
        file_scale = float(payload.get("diameter_scale", 1.0))
    except (TypeError, ValueError):
        file_scale = 1.0
    if file_scale <= 0.0:
        file_scale = 1.0
    true_factor = 1.0 / file_scale
    seed_factor = 1.0 if scale is None else float(scale) / file_scale
    index: dict[str, tuple[CircleAnnotation, ...]] = {}
    for image_key, entries in cells.items():
        circles: list[CircleAnnotation] = []
        for entry in entries or []:
            try:
                source = str(entry.get("source") or SOURCE_ANNOTATED)
                circles.append(
                    CircleAnnotation(
                        location_id=int(entry["location_id"]),
                        center_x=float(entry["center_x"]),
                        center_y=float(entry["center_y"]),
                        radius=float(entry["radius"]) * (true_factor if source == SOURCE_ANNOTATED else seed_factor),
                        structure_label=str(entry.get("structure_label") or ""),
                        has_polygon_mask=bool(entry.get("has_polygon_mask", False)),
                        source=source,
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        if circles:
            index[str(image_key)] = tuple(circles)
    return index


def load_circle_scale(crops: Path) -> float | None:
    """The ``diameter_scale`` recorded in ``circle_annotations.json``, or None when absent or unreadable."""
    try:
        payload = json.loads(circle_file_path(crops).read_text(encoding="utf-8"))
        return float(payload["diameter_scale"])
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def rasterize_circle(shape: tuple[int, int], center_x: float, center_y: float, radius: float) -> np.ndarray:
    """Boolean disc over an image of ``shape`` (height, width). A pixel is inside when its center is.

    Pixel (x, y) has its center at (x + 0.5, y + 0.5) in the tile's continuous coordinates. A disc that
    misses the image gives an all-False array.
    """
    height, width = shape
    if radius <= 0.0:
        return np.zeros((height, width), dtype=bool)
    xs = np.arange(width, dtype=np.float32) + 0.5
    ys = np.arange(height, dtype=np.float32) + 0.5
    dx2 = (xs - np.float32(center_x)) ** 2
    dy2 = (ys - np.float32(center_y)) ** 2
    return (dy2[:, None] + dx2[None, :]) <= np.float32(radius) ** 2


class MaskCache:
    """Small LRU of decoded neighbor masks keyed by file path.

    Neighbor masks are read from a network share for every sample that shares their tile, so a worker keeps
    the most recent ones. One instance per process; not thread safe. Cached arrays are shared, so callers
    must not modify them.
    """

    _entries: OrderedDict[Path, np.ndarray]

    def __init__(self, max_entries: int = 64) -> None:
        self._entries = OrderedDict()
        self._max_entries = max(0, int(max_entries))

    def get(self, path: Path) -> np.ndarray:
        """Decoded boolean mask for ``path``. Raises FileNotFoundError when it is gone."""
        cached = self._entries.get(path)
        if cached is not None:
            self._entries.move_to_end(path)
            return cached
        mask = _load_mask(path)
        if self._max_entries:
            self._entries[path] = mask
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
        return mask


def _load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("L")) > _MASK_THRESHOLD


def _fit(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor resize of a boolean mask to ``shape`` when it differs."""
    if mask.shape == shape:
        return mask
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    resized = image.resize((shape[1], shape[0]), resample=Image.NEAREST)
    return np.array(resized) > _MASK_THRESHOLD


def collect_negative_regions(
    image_key: str,
    target_location_id: int,
    neighbor_location_ids: Iterable[int],
    circles: Sequence[CircleAnnotation],
    mask_dir: Path,
    shape: tuple[int, int],
    cache: MaskCache | None = None,
    trust: CircleAnnotationTrust = DEFAULT_CIRCLE_TRUST,
) -> list[NegativeRegion]:
    """Every other annotation in the tile as its own boolean raster.

    ``neighbor_location_ids`` are the locations that have a mask PNG in ``mask_dir`` for this tile (the target
    may be included; it is skipped). A neighbor whose file has vanished is skipped. A circle is rasterized
    only when its location is neither the target nor a neighbor whose mask was loaded, so a polygon is not
    drawn twice and the inscribed circle of a polygon is not added on top of the polygon itself.

    An annotated circle (true radius, unreliable) becomes the region it claims under ``trust`` (``rho <=
    zero_frac``) with its trust map in ``weight``; an inscribed circle is rasterized at its stored radius.
    """
    loader = cache.get if cache is not None else _load_mask
    regions: list[NegativeRegion] = []
    covered: set[int] = {int(target_location_id)}
    for location_id in sorted({int(i) for i in neighbor_location_ids}):
        if location_id == int(target_location_id):
            continue
        try:
            mask = _fit(loader(mask_dir / f"{image_key}_{location_id}.png"), shape)
        except FileNotFoundError:
            continue
        if not mask.any():
            continue
        covered.add(location_id)
        regions.append(NegativeRegion(location_id, KIND_POLYGON, mask))
    for circle in circles:
        if circle.location_id in covered:
            continue
        if circle.source == SOURCE_ANNOTATED:
            claim = circle_negative_canvas(shape, circle.center_x, circle.center_y, circle.radius, trust, 1.0)
            if claim.mask.any():
                covered.add(circle.location_id)
                regions.append(NegativeRegion(circle.location_id, KIND_CIRCLE, claim.mask, -claim.canvas))
            continue
        mask = rasterize_circle(shape, circle.center_x, circle.center_y, circle.radius)
        if mask.any():
            covered.add(circle.location_id)
            regions.append(NegativeRegion(circle.location_id, KIND_CIRCLE, mask))
    return regions


def stamp_negative_regions(regions: Collection[NegativeRegion], shape: tuple[int, int]) -> np.ndarray:
    """int8 canvas of ``shape``: -1 where any region is set, 0 elsewhere (0 means unknown)."""
    canvas = np.zeros(shape, dtype=np.int8)
    for region in regions:
        canvas[region.mask] = -1
    return canvas


def composite_negative_mask(
    image_key: str,
    target_location_id: int,
    neighbor_location_ids: Iterable[int],
    circles: Sequence[CircleAnnotation],
    mask_dir: Path,
    shape: tuple[int, int],
    cache: MaskCache | None = None,
    trust: CircleAnnotationTrust = DEFAULT_CIRCLE_TRUST,
) -> np.ndarray:
    """int8 array of ``shape``: -1 at every other annotation in the tile, 0 elsewhere.

    The trainer and the gallery both call this, so what an annotator sees as background context is what
    training uses. The caller adds the +1 foreground with :func:`combine_signed` and decides how to use or
    draw the result.
    """
    regions = collect_negative_regions(
        image_key, target_location_id, neighbor_location_ids, circles, mask_dir, shape, cache, trust
    )
    return stamp_negative_regions(regions, shape)


def combine_signed(foreground: np.ndarray | None, negative: np.ndarray) -> np.ndarray:
    """Signed int8 prompt canvas: +1 foreground, -1 background, 0 unknown.

    Where foreground and background overlap the sum is 0, so a conflict becomes uncertainty instead of a
    training signal. ``foreground`` None gives the negatives alone.
    """
    if foreground is None:
        return negative.copy()
    return np.clip(negative.astype(np.int16) + foreground.astype(np.int16), -1, 1).astype(np.int8)


def circles_over_target(regions: Sequence[NegativeRegion], target: np.ndarray) -> list[NegativeRegion]:
    """The circle regions that touch the target mask. Those say background about pixels the loss calls foreground."""
    return [r for r in regions if r.kind == KIND_CIRCLE and bool((r.mask & target).any())]


def drop_circles_over_target(regions: Sequence[NegativeRegion], target: np.ndarray) -> list[NegativeRegion]:
    """``regions`` without the circles that overlap ``target``.

    A polygon neighbor is never dropped. A neighbor's circle that reaches into the target contradicts the ground
    truth (combine_signed only cancels it against a dense foreground canvas, which most prompt types lack).
    Choosing by the true mask lets a little of its extent into the prompt, so this is off by default.
    """
    over = {id(r) for r in circles_over_target(regions, target)}
    return [r for r in regions if id(r) not in over]
