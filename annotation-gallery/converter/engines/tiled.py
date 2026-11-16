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


def readable_context(client: Any, volume_name: str, viking_xml: str = "") -> Any:
    """The server's annotation context that reads ``viking_xml`` for ``volume_name``. SegmentLocation needs the server to
    read the volume's images itself (the gallery never uploads tiles to it); a volume it cannot read is refused here, as
    the build dialog shows it greyed out. A volume can have several contexts of one name, each with other images: without
    a VikingXML the annotation database's default context is used (the one whose tiles the cards show)."""
    volume = client.readable_volume(volume_name, viking_xml)
    if volume is None:
        where = (
            f" reading {viking_xml}" if viking_xml
            else " (several contexts read it: set the annotation database's default context in Identity)"
        )
        raise SegmentationFailed(f"no_local_images: the segmentation server has no context for {volume_name}{where}")
    if not volume.readable:
        raise SegmentationFailed(f"no_local_images: the segmentation server cannot read {volume_name}'s images")
    return volume


def location_context(client: Any, volume_name: str, viking_xml: str = "") -> int:
    """The id of :func:`readable_context`."""
    return int(readable_context(client, volume_name, viking_xml).annotation_context_id)


def is_raw(tileset: str) -> bool:
    """A raw tileset (TEM.Raw8): unleveled pixels, which the model is not trained on."""
    return tileset.split(".")[-1].lower().startswith("raw")


def ordered_tilesets(names: Any) -> list:
    """The tileset names to offer, those fit to segment (not raw) first."""
    return sorted(names, key=lambda name: (is_raw(name), name.lower()))


def choose_tileset(names: Any, wanted: Optional[str] = None) -> str:
    """The tileset to segment on. The common context has exactly one, which is used without asking. With several, the
    one asked for, else the first that is not raw. A context with none cannot be segmented."""
    offered = ordered_tilesets(names)
    if not offered:
        raise SegmentationFailed("no_tileset: the segmentation server's VikingXML for this volume has no tileset")
    if wanted:
        match = next((name for name in offered if name.lower() == wanted.lower()), None)
        if match is None:
            raise SegmentationFailed(f"no_tileset: the segmentation server has no tileset {wanted!r}; it has {offered}")
        return match
    return offered[0]


class TiledEngine:
    """Segments each location at the model's tile level; the card is laid out by the caller."""

    volume_name: str
    client: Any
    annotation_context_id: int
    tile_downsample: int
    threshold: float
    job_id: int
    tileset: Optional[str]
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
        tileset: Optional[str] = None,
        semaphore: Optional[threading.Semaphore] = None,
    ) -> None:
        self.volume_name = volume_name
        self.client = client
        self.annotation_context_id = annotation_context_id
        self.tile_downsample = tile_downsample
        self.threshold = threshold
        self.job_id = job_id
        self.tileset = tileset
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
                self.annotation_context_id, record.id, threshold=self.threshold, job_id=self.job_id, tileset=self.tileset,
            ))
        finally:
            if self.semaphore is not None:
                self.semaphore.release()
        shape = contour_module.contour_from_result(result)
        if shape is None:
            raise SegmentationFailed("no_object: the mask is empty at the threshold")
        return Segmentation(shape=shape, level=level, result=result)
