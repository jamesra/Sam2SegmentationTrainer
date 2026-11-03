"""The single-window engine: the annotation, then its proposal, fitted into one 1024 image at the finest level that holds
them, for a model trained on whole objects in one window.

Nothing here is used by the tiled engine. Seeds are :mod:`converter.seed`'s: a circle with four rim clicks at 0.95 r, a
polygon with up to four clicks in its deepest parts.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, List, Optional, Sequence, Tuple

from converter import cardgeom
from converter import contour as contour_module
from converter import odata, seed
from converter.engines import Segmentation
from converter.seg_client import Prompt, SegmentationFailed, TileIdentity
from converter.tilesource import SectionTiles


def build_prompt(record: odata.LocationRecord, downsample: int) -> Prompt:
    if record.is_circle:
        cx, cy = record.center
        return seed.circle_prompt(cx / downsample, cy / downsample, record.radius / downsample)
    exterior, holes = cardgeom.rings_for_prompt(record, downsample)
    return seed.polygon_prompt(exterior, holes)


def _no_neighbours(cells: Sequence[Tuple[int, int]]) -> List[Any]:
    """Neighbouring annotations are not looked up yet: every cell is answered as having no background shapes."""
    return []


class WindowEngine:
    """Segments each location at the level its card is drawn at, re-fitting once the proposal is known."""

    def __init__(
        self,
        volume_name: str,
        client: Any,
        section: Callable[[int, int], SectionTiles],
        levels: Callable[[int], List[int]],
        *,
        threshold: float = 0.5,
        semaphore: Optional[threading.Semaphore] = None,
    ) -> None:
        self.volume_name = volume_name
        self.client = client
        self.section = section
        self.levels = levels
        self.threshold = threshold
        self.semaphore = semaphore

    def segment(self, record: odata.LocationRecord) -> Segmentation:
        levels = self.levels(record.z)
        record_bounds = odata.bounds(record)
        downsample = cardgeom.choose_downsample(record_bounds, levels)
        while True:
            shape, result = self._segment(record, self.section(record.z, downsample), downsample)
            # The card must hold the proposal as well as the annotation: the proposal can reach past the annotation,
            # and a window centred on the annotation alone would crop it.
            bounds = cardgeom.union_bounds(record_bounds, cardgeom.shape_bounds(shape, downsample))
            try:
                fitted = cardgeom.choose_downsample(bounds, levels, at_least=downsample)
            except SegmentationFailed:
                # No level holds the whole proposal; keep the annotation's card rather than fail the location.
                bounds, fitted = record_bounds, downsample
            if fitted == downsample:
                return Segmentation(shape=shape, level=downsample, result=result, card_level=downsample, card_bounds=bounds)
            downsample = fitted

    def _segment(self, record: odata.LocationRecord, section: SectionTiles, downsample: int):
        identity = TileIdentity(
            volume=self.volume_name, section=record.z, channel=section.tileset.channel,
            transform=section.tileset.name or "mosaic", downsample=downsample,
        )
        prompt = build_prompt(record, downsample)
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
        return shape, result
