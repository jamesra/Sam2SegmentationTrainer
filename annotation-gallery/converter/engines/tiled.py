"""The tiled engine: the request is made at the model's tile level (DS 1) from the tiles under the prompt, and the server
grows the object across as many 1024 tiles as it needs, the way the model was trained and the way Viking's tile grid
segments.

Nothing here is used by the single-window engine, and it never picks a coarser level to make an object fit: a large
object takes more tiles, not a smaller picture. Seeds are this module's own, matching Viking and the trainer:

* a circle is its shape alone, with its TRUE radius (the server applies the model's circle_diameter_scale). No clicks:
  a click inside an oversized circle can land on the background and say the opposite of what the shape says;
* a polygon is its shape (exterior and holes) and one click at its deepest point, Viking's one click per polygon.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, List, Optional, Sequence, Tuple

from converter import cardgeom
from converter import contour as contour_module
from converter import odata, seed
from converter.engines import Segmentation
from converter.seg_client import Prompt, SegmentationFailed, TileIdentity, circle_shape, polygon_shape
from converter.tilesource import SectionTiles


def circle_prompt(cx: float, cy: float, radius: float) -> Prompt:
    """The circle as a shape, no clicks."""
    if radius < 1.0:
        raise seed.NotSeedable("circle radius is below one pixel at this downsample")
    return Prompt(foreground_shapes=[circle_shape(cx, cy, radius)])


def polygon_prompt(ring: Sequence[Tuple[float, float]], holes: Sequence[Sequence[Tuple[float, float]]] = ()) -> Prompt:
    """The polygon and one click at its deepest point."""
    if len(ring) < 4:
        raise seed.NotSeedable("polygon has fewer than three vertices")
    clicks = seed.interior_points(seed.polygon_mask(ring, holes), count=1)
    if not clicks:
        raise seed.NotSeedable("polygon is empty at this downsample")
    return Prompt(foreground_points=clicks, foreground_shapes=[polygon_shape(ring, holes)])


def build_prompt(record: odata.LocationRecord, downsample: int) -> Prompt:
    if record.is_circle:
        cx, cy = record.center
        return circle_prompt(cx / downsample, cy / downsample, record.radius / downsample)
    exterior, holes = cardgeom.rings_for_prompt(record, downsample)
    return polygon_prompt(exterior, holes)


def _no_neighbours(cells: Sequence[Tuple[int, int]]) -> List[Any]:
    """Neighbouring annotations are not looked up yet: every cell is answered as having no background shapes."""
    return []


class TiledEngine:
    """Segments each location at the model's tile level; the card is laid out by the caller."""

    def __init__(
        self,
        volume_name: str,
        client: Any,
        section: Callable[[int, int], SectionTiles],
        levels: Callable[[int], List[int]],
        *,
        tile_downsample: int = 1,
        threshold: float = 0.5,
        semaphore: Optional[threading.Semaphore] = None,
    ) -> None:
        self.volume_name = volume_name
        self.client = client
        self.section = section
        self.levels = levels
        self.tile_downsample = tile_downsample
        self.threshold = threshold
        self.semaphore = semaphore

    def segment(self, record: odata.LocationRecord) -> Segmentation:
        level = self.tile_downsample
        if level not in self.levels(record.z):
            raise SegmentationFailed(f"no_level: section {record.z}'s tileset has no level {level}, the model's tile level")
        section = self.section(record.z, level)
        identity = TileIdentity(
            volume=self.volume_name, section=record.z, channel=section.tileset.channel,
            transform=section.tileset.name or "mosaic", downsample=level,
        )
        prompt = build_prompt(record, level)
        if self.semaphore is not None:
            self.semaphore.acquire()
        try:
            result = self.client.segment_with_retry(
                identity, prompt, section.protocol_tile, _no_neighbours,
                threshold=self.threshold, request_id=record.id & 0xFFFFFFFF,
            )
        finally:
            if self.semaphore is not None:
                self.semaphore.release()
        shape = contour_module.contour_from_result(result)
        if shape is None:
            raise SegmentationFailed("no_object: the mask is empty at the threshold")
        return Segmentation(shape=shape, level=level, result=result)
