"""Location prompts for the tiled engine and the tiled score, built by Viking's shared prompt library.

The gallery gathers what the library needs (:mod:`converter.prompts_native`): the target location(s) as the database
holds them, the section's candidate neighbours, and the structures for the neighbour rule's descendant walk. The library
turns them into the ``SegmentTilesRequest``, so a location gets the same request, and the same server cache row, from
the gallery and from Viking. The rules (mosaic DS 1, circle disc plus centre click, polygon as an outline to correct,
Viking's neighbour rule) live in the library's README, not here.

Candidates come from OData once per section (every cell-level circle and polygon on it) and are cut down to the
library's reach, with a margin, before each call: the library parses its whole input on every call.
"""

from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from converter import odata
from converter.prompts_native import PromptLibrary, library as shared_library
from converter.seg_client import SegmentResult, TileProvider
from seg_proto import prompt_input_pb2 as prompt_pb

LOGGER = logging.getLogger("converter.prompts")

# Candidates are kept when their bounds meet the library's reach grown by this much (mosaic pixels): the gallery's
# bounds come from its own curve fit, the library's from the exporter's, and the library filters exactly afterwards.
REACH_MARGIN = 64.0
SECTIONS_KEPT = 8


def annotation_context_id(volume_name: str, environ: Optional[Mapping[str, str]] = None) -> int:
    """``GALLERY_ANNOTATION_CONTEXT_<VOLUME>``: the Identity annotation context of the volume, so the server reads the
    tiles itself instead of asking the gallery to upload them. 0 (unset) uploads every tile."""
    mapping = environ if environ is not None else os.environ
    key = "GALLERY_ANNOTATION_CONTEXT_" + "".join(c if c.isalnum() else "_" for c in volume_name.upper())
    try:
        return int((mapping.get(key) or "0").strip())
    except ValueError:
        LOGGER.warning("%s is not a number; tiles are uploaded", key)
        return 0


def prompt_annotation(record: odata.LocationRecord) -> "prompt_pb.PromptAnnotation":
    """A location as the library takes it: TypeCode, raw MosaicShape WKT (the library fits curves itself), X, Y and
    the TRUE mosaic radius for a circle, the structure and LastModified in ticks."""
    annotation = prompt_pb.PromptAnnotation(
        id=record.id, type_code=record.type_code, mosaic_wkt=record.wkt or "",
        structure_id=record.structure_id or 0, last_modified_ticks=odata.last_modified_ticks(record.last_modified),
    )
    if record.is_circle:
        annotation.x, annotation.y = record.center
        annotation.radius = float(record.radius)
    return annotation


def _meets(bounds: Tuple[float, float, float, float], rect: Tuple[float, float, float, float]) -> bool:
    x0, y0, x1, y1 = bounds
    return x0 <= rect[2] + REACH_MARGIN and x1 >= rect[0] - REACH_MARGIN and y0 <= rect[3] + REACH_MARGIN and y1 >= rect[1] - REACH_MARGIN


class LocationPrompts:
    """Builds and runs library prompts for one volume. Thread-safe: the converter runs a few streams at once."""

    def __init__(
        self,
        volume_name: str,
        odata_base: str,
        *,
        get_json: odata.GetJson = odata.http_get_json,
        prompt_library: Optional[PromptLibrary] = None,
        annotation_context: Optional[int] = None,
        job_id: int = 0,
    ) -> None:
        self.volume_name = volume_name
        self.odata_base = odata_base
        self.get_json = get_json
        self._library = prompt_library
        self.annotation_context = annotation_context_id(volume_name) if annotation_context is None else annotation_context
        self.job_id = job_id
        self._lock = threading.Lock()
        self._sections: "OrderedDict[int, List[odata.LocationRecord]]" = OrderedDict()
        self._structures: Dict[int, odata.StructureRecord] = {}

    @property
    def library(self) -> PromptLibrary:
        return self._library or shared_library()

    # -- inputs ------------------------------------------------------------------------------------------------------

    def section_candidates(self, z: int) -> List[odata.LocationRecord]:
        """Every candidate neighbour on section ``z``, read once and kept for the last few sections."""
        with self._lock:
            if z in self._sections:
                self._sections.move_to_end(z)
                return self._sections[z]
        records = list(odata.iter_section_neighbours(self.odata_base, z, get_json=self.get_json))
        with self._lock:
            self._sections[z] = records
            while len(self._sections) > SECTIONS_KEPT:
                self._sections.popitem(last=False)
        return records

    def _structures_for(self, records: Iterable[odata.LocationRecord]) -> List["prompt_pb.PromptStructure"]:
        """The records' structures with their ancestors, reading from OData only the ancestors no record described."""
        known: Dict[int, odata.StructureRecord] = {}
        for record in records:
            if record.structure_id:
                known[record.structure_id] = odata.StructureRecord(
                    record.structure_id, record.structure_parent_id, record.structure_type_id, record.structure_type_parent_id,
                )
        with self._lock:
            for key, value in self._structures.items():
                known.setdefault(key, value)
        missing = {s.parent_id for s in known.values() if s.parent_id and s.parent_id not in known}
        while missing:
            found = list(odata.iter_structures(self.odata_base, sorted(missing), get_json=self.get_json))
            for structure in found:
                known[structure.id] = structure
            with self._lock:
                self._structures.update((s.id, s) for s in found)
            # A parent OData does not return ends the walk there, as an unknown structure does in the library.
            missing = {s.parent_id for s in found if s.parent_id and s.parent_id not in known}
        return [
            prompt_pb.PromptStructure(id=s.id, parent_id=s.parent_id or 0, type_id=s.type_id or 0, type_parent_id=s.type_parent_id or 0)
            for s in sorted(known.values(), key=lambda s: s.id)
        ]

    def input_for(
        self,
        targets: Sequence[odata.LocationRecord],
        tileset: str,
        *,
        model_id: Optional[str] = None,
        threshold: Optional[float] = None,
        request_id: int = 0,
        refresh: bool = False,
        target_annotations: Optional[Sequence["prompt_pb.PromptAnnotation"]] = None,
    ) -> "prompt_pb.PromptInput":
        """The library input for ``targets`` on their section. ``target_annotations`` replaces the targets' own
        geometry (the score's seed circle) while the records still decide the section and the neighbours."""
        z = targets[0].z
        prompt_input = prompt_pb.PromptInput(
            volume=self.volume_name, section=z, tileset=tileset, annotation_context_id=self.annotation_context,
            request_id=request_id & 0xFFFFFFFFFFFFFFFF, refresh=refresh, job_id=self.job_id,
        )
        if model_id and model_id != "unknown":
            prompt_input.model_id = model_id
        if threshold is not None:
            prompt_input.mask_threshold = threshold
        prompt_input.targets.extend(target_annotations or [prompt_annotation(record) for record in targets])
        reach = self.library.reach(prompt_input)
        candidates = [record for record in self.section_candidates(z) if _meets(odata.bounds(record), reach)]
        prompt_input.candidates.extend(prompt_annotation(record) for record in candidates)
        prompt_input.structures.extend(self._structures_for([*targets, *candidates]))
        return prompt_input

    # -- requests ----------------------------------------------------------------------------------------------------

    def segment(
        self,
        client,
        prompt_input: "prompt_pb.PromptInput",
        tiles: TileProvider,
        *,
        threshold: float,
        attempts: int = 4,
        cache: bool = True,
    ) -> SegmentResult:
        """Build the start, run the stream and answer each background request from the same input. ``cache`` False
        drops the locations, so a prompt that is not the location's own (the score's seed circle) is never stored as
        that location's result."""
        start = self.library.build_start(prompt_input)
        if not cache:
            del start.locations[:]

        def one_stream() -> SegmentResult:
            # What this stream has sent beyond the start; a retried stream starts again from the start.
            sent: List[int] = []

            def answer(needed) -> list:
                background = self.library.background_for(prompt_input, needed, sent)
                sent.extend(background.sent_ids)
                return list(background.shapes)

            return client.segment_request(start, tiles, answer, threshold=threshold)

        return client.retrying(one_stream, attempts=attempts)
