"""What the converter tells the segmentation server's cache: a job per build and the outlines reviewers traced."""

from __future__ import annotations

import shutil
from pathlib import Path

from catalog import connect
from converter import builder as builder_module
from converter import odata
from converter import serverreview
from tests.test_converter_builder import VOLUME, FakeSegmenter, _disc, _entity, _make, _remember, cache, volume  # noqa: F401

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


def test_the_builds_job_is_in_every_tiled_request(tmp_path: Path, volume, cache) -> None:
    params = builder_module.BuildParams(volume=VOLUME, odata_base="http://odata.test/RC1", structure_id=500)
    entities = [_entity(11, 1, "POINT (1400 1600)", radius=80)]
    segmenter = FakeSegmenter(lambda _id: _disc(1400, 1600, 80))
    _remember(segmenter, entities)
    builder_module.Builder(
        params, tmp_path / "crops", volume, segmenter, cache,
        get_json=lambda url: {"@odata.count": 1, "value": entities}, job_id=41,
    ).run()
    assert [call.job_id for call in segmenter.located] == [41]


def _retrace(crops: Path) -> None:
    connection = connect(crops)
    connection.execute("UPDATE converter_cards SET reason = 'retraced' WHERE location_id = 11")
    connection.commit()
    connection.close()
    assert serverreview.mark_trace(crops, 11, KEY, "retrace", "Ann")


def test_a_retraced_card_is_stored_against_the_locations_last_modified(tmp_path: Path, volume, cache) -> None:
    crops = _built(tmp_path, volume, cache)
    _retrace(crops)
    client = FakeSegmenter(lambda _id: None)
    assert serverreview.sync_now(crops, "RC1", client) == 1
    ((context_id, (trace,), removed),) = client.traces
    assert context_id == 7 and removed == []
    assert trace.location_id == 11 and trace.tracer == "Ann"
    assert trace.last_modified_ticks == odata.last_modified_ticks("2026-10-01T00:00:00Z")
    assert len(trace.outline.polygon.exterior.points) >= 3 and trace.traced_ticks > trace.last_modified_ticks
    assert serverreview.sync_now(crops, "RC1", client) == 0, "nothing is pending once the server has it"


def test_unapproving_a_retraced_card_removes_its_trace_and_approving_stores_it_again(tmp_path: Path, volume, cache) -> None:
    crops = _built(tmp_path, volume, cache)
    _retrace(crops)
    client = FakeSegmenter(lambda _id: None)
    serverreview.sync_now(crops, "RC1", client)
    assert serverreview.mark_trace(crops, 11, None, "unapprove")
    serverreview.sync_now(crops, "RC1", client)
    assert serverreview.mark_trace(crops, 11, None, "approve")
    serverreview.sync_now(crops, "RC1", client)
    assert client.traces[1] == (7, [], [11])
    assert client.traces[2][1][0].tracer == "Ann", "approving keeps who traced it"


def test_an_unedited_card_sends_nothing(tmp_path: Path, volume, cache) -> None:
    crops = _built(tmp_path, volume, cache)
    assert not serverreview.mark_trace(crops, 11, None, "approve")
    assert not serverreview.mark_trace(crops, 11, None, "ignore")
    assert serverreview.pending_traces(crops) == ([], [], [])


def test_a_trace_the_server_did_not_take_is_sent_with_the_next_sync(tmp_path: Path, volume, cache) -> None:
    # Laid out as under the converter root: <root>/<structure set>/<volume>.
    crops = Path(shutil.move(str(_built(tmp_path, volume, cache)), str(tmp_path / "root" / "structure-500" / "RC1")))
    _retrace(crops)
    client = FakeSegmenter(lambda _id: None)
    client.refuse_traces = True
    serverreview.sync(crops, "RC1", lambda: client).join(5)
    assert client.traces == [] and serverreview.pending_traces(crops)[2], "still pending"

    client.refuse_traces = False
    (thread,) = serverreview.sync_all(tmp_path / "root", lambda: client)
    thread.join(5)
    assert [trace.location_id for trace in client.traces[0][1]] == [11]
    assert serverreview.pending_traces(crops) == ([], [], [])


def test_a_card_built_before_cards_recorded_last_modified_is_not_stored(tmp_path: Path, volume, cache) -> None:
    crops = _built(tmp_path, volume, cache)
    connection = connect(crops)
    connection.execute("UPDATE converter_cards SET last_modified_ticks = NULL")
    connection.commit()
    connection.close()
    _retrace(crops)
    client = FakeSegmenter(lambda _id: None)
    serverreview.sync_now(crops, "RC1", client)
    assert client.traces == [] and serverreview.pending_traces(crops)[2] == [], "dropped, not retried forever"
