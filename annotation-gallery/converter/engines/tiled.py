"""The tiled engine: the request is made at the model's tile level (DS 1) from the tiles under the prompt, and the server
grows the object across as many 1024 tiles as it needs, the way the model was trained and the way Viking's tile grid
segments.

Nothing here is used by the single-window engine, and it never picks a coarser level to make an object fit: a large
object takes more tiles, not a smaller picture. The request itself is built by Viking's shared prompt library
(:mod:`converter.prompts`), so it is the request Viking sends for the same location and the server answers both from one
cache row: a circle is its TRUE disc and a click at its centre, a polygon an outline to correct (the server clicks its
deepest pixel), and the neighbouring cells are the background.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, List, Optional

from converter import contour as contour_module
from converter import odata, seed
from converter.engines import Segmentation
from converter.prompts import LocationPrompts
from converter.prompts_native import PromptRejected, PromptTooLarge
from converter.seg_client import SegmentationFailed
from converter.tilesource import SectionTiles

# The shared library builds requests at full resolution only (the level every tiled model so far was trained at).
LIBRARY_LEVEL = 1


class TiledEngine:
    """Segments each location at the model's tile level; the card is laid out by the caller."""

    def __init__(
        self,
        volume_name: str,
        client: Any,
        section: Callable[[int, int], SectionTiles],
        levels: Callable[[int], List[int]],
        *,
        prompts: LocationPrompts,
        tile_downsample: int = 1,
        threshold: float = 0.5,
        semaphore: Optional[threading.Semaphore] = None,
    ) -> None:
        self.volume_name = volume_name
        self.prompts = prompts
        self.client = client
        self.section = section
        self.levels = levels
        self.tile_downsample = tile_downsample
        self.threshold = threshold
        self.semaphore = semaphore

    def segment(self, record: odata.LocationRecord) -> Segmentation:
        level = self.tile_downsample
        if level != LIBRARY_LEVEL:
            raise SegmentationFailed(f"no_level: the model's tile level is DS {level}; the shared prompt library builds DS 1 only")
        if level not in self.levels(record.z):
            raise SegmentationFailed(f"no_level: section {record.z}'s tileset has no level {level}, the model's tile level")
        if record.is_circle and record.radius < 1:
            raise seed.NotSeedable("circle radius is below one pixel at this downsample")
        section = self.section(record.z, level)
        try:
            prompt_input = self.prompts.input_for(
                [record], section.tileset.name or section.tileset.channel, model_id=self.client.model_id(),
                threshold=self.threshold, request_id=record.id,
            )
        except PromptRejected as error:
            raise seed.NotSeedable(str(error)) from error
        if self.semaphore is not None:
            self.semaphore.acquire()
        try:
            result = self.prompts.segment(self.client, prompt_input, section.protocol_tile, threshold=self.threshold)
        except PromptRejected as error:
            raise seed.NotSeedable(str(error)) from error
        except PromptTooLarge as error:
            raise SegmentationFailed(f"too_large: {error}") from error
        finally:
            if self.semaphore is not None:
                self.semaphore.release()
        shape = contour_module.contour_from_result(result)
        if shape is None:
            raise SegmentationFailed("no_object: the mask is empty at the threshold")
        return Segmentation(shape=shape, level=level, result=result)
