"""The tiled engine: the request is made at the model's tile level (DS 1) from the tiles under the prompt, and the server
grows the object across as many 1024 tiles as it needs, the way the model was trained and the way Viking's tile grid
segments.

Nothing here is used by the single-window engine, and it never picks a coarser level to make an object fit: a large
object takes more tiles, not a smaller picture. The gallery names the location and the segmentation server builds its
prompt (SegmentLocation, :meth:`converter.seg_client.SegmentationClient.segment_location`) from the database and the
images it reads itself, so it is the request Viking gets for the same location and the server answers both from one
cache row: a circle is its TRUE disc and a click at its centre, a polygon an outline to correct (the server clicks its
deepest pixel), and the neighbouring cells are the background.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

from converter import contour as contour_module
from converter import odata, seed
from converter.engines import Segmentation
from converter.seg_client import SegmentationFailed

# The server builds location prompts at full resolution only (the level every tiled model so far was trained at).
LOCATION_LEVEL = 1


def location_context(client: Any, volume_name: str) -> int:
    """The server's annotation context for ``volume_name``. SegmentLocation needs the server to read the volume's
    images itself (the gallery never uploads tiles to it); a volume it cannot read is refused here, as the build dialog
    shows it greyed out."""
    volume = client.readable_volume(volume_name)
    if volume is None:
        raise SegmentationFailed(f"no_local_images: the segmentation server is not assigned {volume_name}")
    if not volume.readable:
        raise SegmentationFailed(f"no_local_images: the segmentation server cannot read {volume_name}'s images")
    return int(volume.annotation_context_id)


class TiledEngine:
    """Segments each location at the model's tile level; the card is laid out by the caller."""

    volume_name: str
    client: Any
    annotation_context_id: int
    tile_downsample: int
    threshold: float
    job_id: int
    semaphore: Optional[threading.Semaphore]

    def __init__(
        self,
        volume_name: str,
        client: Any,
        *,
        annotation_context_id: int,
        tile_downsample: int = 1,
        threshold: float = 0.5,
        job_id: int = 0,
        semaphore: Optional[threading.Semaphore] = None,
    ) -> None:
        self.volume_name = volume_name
        self.client = client
        self.annotation_context_id = annotation_context_id
        self.tile_downsample = tile_downsample
        self.threshold = threshold
        self.job_id = job_id
        self.semaphore = semaphore

    def segment(self, record: odata.LocationRecord) -> Segmentation:
        level = self.tile_downsample
        if level != LOCATION_LEVEL:
            raise SegmentationFailed(f"no_level: the model's tile level is DS {level}; the server builds location prompts at DS 1 only")
        if record.is_circle and record.radius < 1:
            raise seed.NotSeedable("circle radius is below one pixel at this downsample")
        if self.semaphore is not None:
            self.semaphore.acquire()
        try:
            result = self.client.retrying(lambda: self.client.segment_location(
                self.annotation_context_id, record.id, threshold=self.threshold, job_id=self.job_id,
            ))
        finally:
            if self.semaphore is not None:
                self.semaphore.release()
        shape = contour_module.contour_from_result(result)
        if shape is None:
            raise SegmentationFailed("no_object: the mask is empty at the threshold")
        return Segmentation(shape=shape, level=level, result=result)
