"""The gallery's side of the shared prompt library: what it reads from OData, how it cuts candidates to the library's
reach, and how it runs a stream. The library's own rules are tested in Viking."""

from __future__ import annotations

from typing import Any, Dict, List

from converter import odata
from converter.prompts import LocationPrompts, annotation_context_id, prompt_annotation
from converter.seg_client import SegmentationClient, SegmentationFailed
from seg_proto import prompt_input_pb2 as prompt_pb
from seg_proto import segmentation_pb2 as pb
from tests.fake_prompts import FakePromptLibrary
from tests.test_converter_builder import _entity

BASE = "http://odata.test/RC1"


def _neighbour(location_id: int, x: float, y: float, structure: int, structure_parent=None) -> Dict[str, Any]:
    entity = _entity(location_id, 1, f"POINT ({x} {y})", radius=20, structure=structure)
    entity["Parent"]["ParentID"] = structure_parent
    return entity


def test_last_modified_ticks_keep_all_seven_digits_and_the_offset() -> None:
    # 2026-10-01T00:00:00Z in .NET ticks: DateTime(2026, 10, 1, 0, 0, 0, DateTimeKind.Utc).Ticks.
    midnight = 639_264_096_000_000_000
    assert odata.last_modified_ticks("2026-10-01T00:00:00Z") == midnight
    assert odata.last_modified_ticks("2026-10-01T00:00:00.1234567Z") == midnight + 1_234_567
    assert odata.last_modified_ticks("2026-10-01T02:00:00+02:00") == midnight
    assert odata.last_modified_ticks("2026-10-01T00:00:00") == midnight, "no offset is UTC"
    assert odata.last_modified_ticks("") == 0 and odata.last_modified_ticks("yesterday") == 0


def test_a_circle_annotation_is_its_true_centre_and_radius_and_a_polygon_its_raw_wkt() -> None:
    circle = prompt_annotation(odata.record_from_entity(_entity(11, 1, "POINT (1400 1600)", radius=80)))
    assert (circle.id, circle.type_code, circle.x, circle.y, circle.radius) == (11, 1, 1400, 1600, 80)
    assert circle.structure_id == 500 and circle.last_modified_ticks == odata.last_modified_ticks("2026-10-01T00:00:00Z")
    wkt = "POLYGON ((0 0, 100 0, 100 100, 0 100, 0 0))"
    curve = prompt_annotation(odata.record_from_entity(_entity(12, 6, wkt)))
    assert curve.mosaic_wkt == wkt, "the library fits the curve itself"
    assert (curve.x, curve.y, curve.radius) == (0, 0, 0)


def test_the_annotation_context_comes_from_the_volumes_variable() -> None:
    assert annotation_context_id("RC2", {"GALLERY_ANNOTATION_CONTEXT_RC2": " 7 "}) == 7
    assert annotation_context_id("rc-2", {"GALLERY_ANNOTATION_CONTEXT_RC_2": "9"}) == 9
    assert annotation_context_id("RC2", {}) == 0, "unset: the gallery uploads the tiles"
    assert annotation_context_id("RC2", {"GALLERY_ANNOTATION_CONTEXT_RC2": "x"}) == 0


def test_candidates_are_read_once_per_section_cut_to_the_reach_and_their_structures_resolved() -> None:
    asked: List[str] = []
    section = [_neighbour(21, 1000, 1000, structure=600, structure_parent=700), _neighbour(22, 9000, 9000, structure=601)]
    structures = {700: {"ID": 700, "ParentID": 800, "TypeID": 1, "Type": {"ParentID": None}},
                  800: {"ID": 800, "ParentID": None, "TypeID": 1, "Type": {"ParentID": None}}}

    def get_json(url: str):
        asked.append(url)
        if "/Structures?" in url:
            return {"value": [v for k, v in structures.items() if f"ID%20eq%20{k}" in url or f"ID eq {k}" in url]}
        return {"value": section}

    library = FakePromptLibrary(reach=(900, 900, 1500, 1500))
    prompts = LocationPrompts("RC1", BASE, get_json=get_json, prompt_library=library, annotation_context=5)
    target = odata.record_from_entity(_entity(11, 1, "POINT (1200 1200)", radius=80))

    first = prompts.input_for([target], "Leveled", model_id="v5", threshold=0.5, request_id=11)
    second = prompts.input_for([target], "Leveled", model_id="v5", threshold=0.5, request_id=11)

    assert [c.id for c in first.candidates] == [21], "22 is far outside the reach"
    assert (first.tileset, first.annotation_context_id, first.model_id, first.mask_threshold) == ("Leveled", 5, "v5", 0.5)
    assert {s.id: s.parent_id for s in first.structures} == {500: 0, 600: 700, 700: 800, 800: 0}
    assert first.SerializeToString() == second.SerializeToString()
    assert len([u for u in asked if "/Locations?" in u]) == 1, "the section is read once"
    assert len([u for u in asked if "/Structures?" in u]) == 2, "700, then its parent 800; neither again"


def test_an_unknown_model_is_left_unset() -> None:
    prompts = LocationPrompts("RC1", BASE, get_json=lambda url: {"value": []}, prompt_library=FakePromptLibrary(), annotation_context=0)
    target = odata.record_from_entity(_entity(11, 1, "POINT (1200 1200)", radius=80))
    assert prompts.input_for([target], "Leveled", model_id="unknown").model_id == ""


class _StreamClient:
    """Fails the first stream after one background answer, then succeeds."""

    retrying = staticmethod(lambda call, attempts=4: SegmentationClient.retrying(call, attempts=attempts, backoff_seconds=0))

    def __init__(self) -> None:
        self.starts: List[Any] = []
        self.streams = 0

    def segment_request(self, start, tiles, answer_background, *, threshold):
        self.starts.append(start)
        self.streams += 1
        answer_background(pb.TilesNeeded())
        if self.streams == 1:
            raise SegmentationFailed("unavailable", retryable=True)
        return "done"


class _SendingLibrary(FakePromptLibrary):
    def background_for(self, prompt_input, needed, sent_ids):
        self.background_calls.append((0, list(sent_ids)))
        return prompt_pb.PromptBackground(sent_ids=[21])


def test_a_retried_stream_starts_its_record_of_sent_neighbours_again() -> None:
    library = _SendingLibrary()
    prompts = LocationPrompts("RC1", BASE, get_json=lambda url: {"value": []}, prompt_library=library, annotation_context=0)
    target = odata.record_from_entity(_entity(11, 1, "POINT (1200 1200)", radius=80))
    client = _StreamClient()
    prompts.segment(client, prompts.input_for([target], "Leveled"), tiles=None, threshold=0.5)
    assert client.streams == 2
    assert [sent for _n, sent in library.background_calls] == [[], []], "the second stream has sent nothing yet"


def test_a_prompt_that_is_not_the_locations_own_is_never_cached() -> None:
    prompts = LocationPrompts("RC1", BASE, get_json=lambda url: {"value": []}, prompt_library=FakePromptLibrary(), annotation_context=0)
    target = odata.record_from_entity(_entity(11, 1, "POINT (1200 1200)", radius=80))
    client = _StreamClient()
    client.streams = 1  # no failure
    prompts.segment(client, prompts.input_for([target], "Leveled"), tiles=None, threshold=0.5, cache=False)
    (start,) = client.starts
    assert not start.locations
    client.starts.clear()
    prompts.segment(client, prompts.input_for([target], "Leveled"), tiles=None, threshold=0.5)
    assert [ref.id for ref in client.starts[0].locations] == [11]

