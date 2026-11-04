"""What the converter tells the segmentation server's cache: a job per build and the outlines reviewers approved."""

from __future__ import annotations

from pathlib import Path

from catalog import connect
from converter import builder as builder_module
from converter import odata
from converter import serverreview
from tests.fake_prompts import FakePromptLibrary
from tests.test_converter_builder import VOLUME, FakeSegmenter, _disc, _entity, _make, cache, volume  # noqa: F401

KEY = f"{VOLUME}_7_D1_L11"


def _built(tmp_path: Path, volume, cache) -> Path:
    segmenter = FakeSegmenter(lambda _id: _disc(1400, 1600, 80))
    _make(tmp_path, volume, cache, segmenter, [_entity(11, 1, "POINT (1400 1600)", radius=80)]).run()
    return tmp_path / "crops"


class _Failing:
    def create_job(self, *args, **kwargs):
        raise RuntimeError("unavailable")


def test_a_build_makes_its_job_once_and_keeps_it(tmp_path: Path) -> None:
    client = FakeSegmenter(lambda _id: None)
    assert serverreview.build_job(tmp_path, client, "RC1", "structure-500") == 41
    assert serverreview.build_job(tmp_path, client, "RC1", "structure-500") == 41
    assert client.jobs == [("create", "RC1", "convert")], "a later run reuses the stored job"
    assert serverreview.stored_job(tmp_path) == 41


def test_a_server_that_cannot_make_a_job_leaves_the_build_without_one(tmp_path: Path) -> None:
    assert serverreview.build_job(tmp_path, _Failing(), "RC1", "structure-500") == 0
    assert serverreview.stored_job(tmp_path) == 0, "nothing stored, so the next run asks again"


def test_the_builds_job_is_in_every_tiled_prompt(tmp_path: Path, volume, cache) -> None:
    library = FakePromptLibrary()
    params = builder_module.BuildParams(volume=VOLUME, odata_base="http://odata.test/RC1", structure_id=500)
    entities = [_entity(11, 1, "POINT (1400 1600)", radius=80)]
    builder_module.Builder(
        params, tmp_path / "crops", volume, FakeSegmenter(lambda _id: _disc(1400, 1600, 80)), cache,
        get_json=lambda url: {"@odata.count": 1, "value": entities}, prompt_library=library, job_id=41,
    ).run()
    assert [i.job_id for i in library.inputs] == [41]


def test_an_approved_card_is_stored_against_the_locations_last_modified(tmp_path: Path, volume, cache) -> None:
    crops = _built(tmp_path, volume, cache)
    client = FakeSegmenter(lambda _id: None)
    serverreview.publish(crops, "RC1", 11, KEY, keep=True, reviewer="Ann", make_client=lambda: client).join(5)
    ((name, (outline,), removed),) = client.reviewed
    assert name == "RC1" and removed == []
    assert outline.location_id == 11 and outline.state == "approved" and outline.reviewer == "Ann"
    assert outline.last_modified_ticks == odata.last_modified_ticks("2026-10-01T00:00:00Z")
    assert len(outline.outline.polygon.exterior.points) >= 3 and outline.reviewed_ticks > outline.last_modified_ticks


def test_a_retraced_card_is_stored_as_retraced_and_an_unapproved_one_removed(tmp_path: Path, volume, cache) -> None:
    crops = _built(tmp_path, volume, cache)
    connection = connect(crops)
    connection.execute("UPDATE converter_cards SET reason = 'retraced' WHERE location_id = 11")
    connection.commit()
    connection.close()
    client = FakeSegmenter(lambda _id: None)
    serverreview.publish(crops, "RC1", 11, KEY, keep=True, reviewer="Ann", make_client=lambda: client).join(5)
    serverreview.publish(crops, "RC1", 11, KEY, keep=False, reviewer="Ann", make_client=lambda: client).join(5)
    assert client.reviewed[0][1][0].state == "retraced"
    assert client.reviewed[1] == ("RC1", [], [11])


def test_a_card_built_before_cards_recorded_last_modified_is_not_stored(tmp_path: Path, volume, cache) -> None:
    crops = _built(tmp_path, volume, cache)
    connection = connect(crops)
    connection.execute("UPDATE converter_cards SET last_modified_ticks = NULL")
    connection.commit()
    connection.close()
    assert serverreview.publish(crops, "RC1", 11, KEY, keep=True, reviewer="Ann", make_client=lambda: None) is None
