"""The converter over HTTP: a build is a ``structure-<id>`` training set under GALLERY_CONVERTER_ROOT, reviewed with the
gallery's own routes, and the approved cards come out as a batch file."""

from __future__ import annotations

import json
import sys
import threading
from http.client import HTTPConnection
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pytest

GALLERY_ROOT = Path(__file__).resolve().parents[1]
for folder in (GALLERY_ROOT, Path(__file__).resolve().parent):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import test_converter_builder as fixtures  # noqa: E402
from converter import builder as builder_module  # noqa: E402
from converter import web  # noqa: E402
from server import make_server  # noqa: E402

SET = "structure-500"


@pytest.fixture
def converter_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "converter"
    monkeypatch.setenv("GALLERY_CONVERTER_ROOT", str(root))
    monkeypatch.setenv("GALLERY_VOLUME_DIR", str(tmp_path / "gallery-volumes"))
    monkeypatch.setenv("GALLERY_IDENTITY_MODE", "stub")
    monkeypatch.setenv("GALLERY_STUB_ROLE", "review")
    monkeypatch.setenv("GALLERY_SESSION_SECRET", "test-secret")
    (tmp_path / "gallery-volumes").mkdir()
    return root


@pytest.fixture
def built(converter_root: Path) -> Path:
    """A structure with two built cards and one failure, in the converter layout; returns the crops folder."""
    crops = converter_root / SET / "RC1"
    volume = fixtures.parse_viking_xml(fixtures.VIKING_XML, source="x")
    entities = [
        fixtures._entity(11, 1, "POINT (1400 1600)", radius=80),
        fixtures._entity(12, 1, "POINT (1700 1900)", radius=80),
        fixtures._entity(13, 1, "POINT (1500 1500)", radius=0.2),
    ]
    truth = {11: fixtures._disc(1400, 1600, 80), 12: fixtures._disc(1700, 1900, 80)}
    segmenter = fixtures.FakeSegmenter(lambda location_id: truth[location_id], size=256)
    params = builder_module.BuildParams(volume="RC1", odata_base="http://odata.test/RC1", structure_id=500)
    builder = builder_module.Builder(
        params, crops, volume, segmenter, fixtures.TileCache(None, fixtures.tile_http),
        get_json=lambda url: {"@odata.count": 3, "value": entities},
    )
    summary = builder.run()
    assert (summary.built, summary.failed) == (2, 1)
    (crops / "source.json").write_text(json.dumps({"kind": "converter", "odata": "https://websvc.codepharm.net/RC1/OData"}))
    return crops


@pytest.fixture
def httpd(built: Path):
    server = make_server("127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield str(host), int(port)
    finally:
        server.shutdown()
        server.server_close()


def request(httpd: Tuple[str, int], method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Tuple[int, bytes, Dict[str, str]]:
    connection = HTTPConnection(httpd[0], httpd[1], timeout=10)
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json", "Content-Length": str(len(data))} if data is not None else {}
    connection.request(method, path, body=data, headers=headers)
    response = connection.getresponse()
    raw = response.read()
    result = (response.status, raw, {k.lower(): v for k, v in response.getheaders()})
    connection.close()
    return result


def get_json(httpd, path: str) -> Tuple[int, Any]:
    status, raw, _ = request(httpd, "GET", path)
    return status, (json.loads(raw) if raw else None)


def test_config_says_the_converter_is_on_and_the_page_is_served(httpd) -> None:
    assert get_json(httpd, "/api/config")[1]["converter"] is True
    status, raw, headers = request(httpd, "GET", "/converter/")
    assert status == 200 and b"<html" in raw.lower() and "text/html" in headers["content-type"]
    assert request(httpd, "GET", "/converter")[0] == 200


def test_built_structures_are_listed_as_sets(httpd) -> None:
    status, payload = get_json(httpd, "/api/training-sets?app=converter")
    assert status == 200 and payload == {"sets": [SET], "default": SET}


def test_the_catalog_of_a_structure_set_lists_its_built_cards(httpd) -> None:
    status, payload = get_json(httpd, f"/api/volumes/RC1/catalog?set={SET}")
    assert status == 200
    ids = sorted(row["location_id"] for row in payload["rows"])
    assert ids == [11, 12]


def test_cards_report_state_reason_and_overlap(httpd) -> None:
    status, payload = get_json(httpd, f"/api/volumes/RC1/converter/cards?set={SET}")
    cards = {card["location_id"]: card for card in payload["cards"]}
    assert status == 200 and set(cards) == {11, 12, 13}
    assert cards[11]["state"] == "built" and cards[11]["iou"] > 0.5
    assert cards[13]["state"] == "failed" and "below one pixel" in cards[13]["reason"]


def test_current_outline_masks_and_images_are_served(httpd, built: Path) -> None:
    key = f"{fixtures.VOLUME}_7_D1_L11"
    for relative in (f"original/{key}_11.png", f"masks/{key}_11.png", f"images/{key}.png"):
        status, raw, headers = request(httpd, "GET", f"/api/volumes/RC1/file?set={SET}&path={relative}")
        assert status == 200 and raw[:8] == b"\x89PNG\r\n\x1a\n", relative


def test_approve_puts_a_card_in_the_batch_and_unapprove_takes_it_out(httpd) -> None:
    key = f"{fixtures.VOLUME}_7_D1_L11"
    status, _, _ = request(httpd, "POST", f"/api/volumes/RC1/approve?set={SET}", {"location_id": 11, "image_key": key})
    assert status == 200

    status, raw, headers = request(httpd, "GET", f"/api/volumes/RC1/converter/batch?set={SET}")
    assert status == 200 and "ndjson" in headers["content-type"] and "attachment" in headers["content-disposition"]
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert (row["location_id"], row["type_code"], row["shape_space"]) == (11, 6, "mosaic")
    assert row["mosaic_wkt"].startswith("POLYGON ((") and row["original_wkt"].startswith("POINT")
    assert row["source_hash"] and row["structure_id"] == 500 and row["downsample"] == 1

    request(httpd, "POST", f"/api/volumes/RC1/unapprove?set={SET}", {"location_id": 11, "image_key": key})
    assert request(httpd, "GET", f"/api/volumes/RC1/converter/batch?set={SET}")[1] == b""


def test_reject_leaves_the_card_out_of_the_batch_and_restore_brings_it_back_unapproved(httpd) -> None:
    key = f"{fixtures.VOLUME}_7_D1_L12"
    assert request(httpd, "POST", f"/api/volumes/RC1/ignore?set={SET}", {"location_id": 12, "image_key": key})[0] == 200
    status, payload = get_json(httpd, f"/api/volumes/RC1/catalog?set={SET}")
    assert {row["location_id"]: row["ignored"] for row in payload["rows"]}[12] in (1, True)
    assert request(httpd, "GET", f"/api/volumes/RC1/converter/batch?set={SET}")[1] == b""


def test_unknown_or_malformed_sets_are_not_found(httpd) -> None:
    assert request(httpd, "GET", "/api/volumes/RC1/catalog?set=structure-999")[0] == 404
    assert request(httpd, "GET", "/api/volumes/RC1/catalog?set=structure-500/../x")[0] == 404
    assert request(httpd, "GET", "/api/volumes/RC1/catalog?set=structure-abc")[0] == 404


def test_without_a_converter_root_the_converter_is_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GALLERY_CONVERTER_ROOT", raising=False)
    monkeypatch.setenv("GALLERY_VOLUME_DIR", str(tmp_path / "gallery-volumes"))
    monkeypatch.setenv("GALLERY_IDENTITY_MODE", "stub")
    monkeypatch.setenv("GALLERY_SESSION_SECRET", "test-secret")
    (tmp_path / "gallery-volumes").mkdir()
    server = make_server("127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address[:2]
    try:
        assert get_json((str(host), int(port)), "/api/config")[1]["converter"] is False
        assert request((str(host), int(port)), "GET", "/converter/")[0] == 404
        assert get_json((str(host), int(port)), "/api/training-sets?app=converter")[1] == {"sets": [], "default": ""}
        assert request((str(host), int(port)), "GET", "/api/volumes/RC1/catalog?set=structure-500")[0] == 404
    finally:
        server.shutdown()
        server.server_close()


def test_a_reader_cannot_approve(built: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GALLERY_STUB_ROLE", "read")
    server = make_server("127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address[:2]
    try:
        key = f"{fixtures.VOLUME}_7_D1_L11"
        assert request((str(host), int(port)), "POST", f"/api/volumes/RC1/approve?set={SET}", {"location_id": 11, "image_key": key})[0] == 403
        assert request((str(host), int(port)), "GET", f"/api/volumes/RC1/converter/batch?set={SET}")[0] == 200
    finally:
        server.shutdown()
        server.server_close()


def test_structure_set_helpers(converter_root: Path, built: Path) -> None:
    assert web.list_structure_sets(converter_root) == [SET]
    assert web.resolve_structure_set(converter_root, SET) == converter_root / SET
    with pytest.raises(FileNotFoundError):
        web.resolve_structure_set(converter_root, "structure-7")
    with pytest.raises(FileNotFoundError):
        web.resolve_structure_set(None, SET)
    assert web.list_structure_sets(None) == []


# ---- build jobs over HTTP ----------------------------------------------------------------------------------------

import time  # noqa: E402

from converter import jobs  # noqa: E402


@pytest.fixture
def job_env(tmp_path: Path, converter_root: Path, monkeypatch: pytest.MonkeyPatch):
    """A gallery with volume RC1 granted, and build jobs wired to the synthetic tileset and a stand-in segmenter."""
    (tmp_path / "gallery-volumes" / "RC1").mkdir(parents=True)
    entities = [
        fixtures._entity(11, 1, "POINT (1400 1600)", radius=80),
        fixtures._entity(12, 1, "POINT (1700 1900)", radius=80),
    ]
    truth = {11: fixtures._disc(1400, 1600, 80), 12: fixtures._disc(1700, 1900, 80)}
    volume = fixtures.parse_viking_xml(fixtures.VIKING_XML, source="x")

    def get_json(url: str):
        return {"@odata.count": 2, "value": entities}

    jobs._JOBS.clear()
    jobs.OVERRIDES.update(
        make_client=lambda: fixtures.FakeSegmenter(lambda location_id: truth[location_id], size=256),
        make_volume=lambda name: volume,
        make_cache=lambda: fixtures.TileCache(None, fixtures.tile_http),
        get_json=get_json,
    )
    yield
    jobs.OVERRIDES.clear()
    jobs._JOBS.clear()


def wait_until_done(httpd, set_name: str = SET, seconds: float = 20.0) -> Dict[str, Any]:
    deadline = time.time() + seconds
    while time.time() < deadline:
        status, payload = get_json(httpd, f"/api/volumes/RC1/update-all?set={set_name}")
        assert status == 200
        if not payload["running"]:
            return payload
        time.sleep(0.05)
    raise AssertionError("the build did not finish")


@pytest.fixture
def clean_httpd(converter_root: Path, job_env):
    server = make_server("127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address[:2]
    try:
        yield str(host), int(port)
    finally:
        server.shutdown()
        server.server_close()


def test_creating_a_structure_builds_it_and_reports_in_update_alls_shape(clean_httpd) -> None:
    status, raw, _ = request(clean_httpd, "POST", "/api/converter/structures", {"structureId": 500, "volume": "RC1"})
    payload = json.loads(raw)
    assert status == 202 and payload["set"] == SET

    done = wait_until_done(clean_httpd)
    assert done["phase"] == "done" and done["refreshed"] == 2 and done["failed"] == 0 and done["pages"] == 2
    assert done["headline"] == "Finished"
    assert get_json(clean_httpd, "/api/training-sets?app=converter")[1]["sets"] == [SET]
    rows = get_json(clean_httpd, f"/api/volumes/RC1/catalog?set={SET}")[1]["rows"]
    assert sorted(row["location_id"] for row in rows) == [11, 12]


def test_continuing_skips_built_cards(clean_httpd) -> None:
    request(clean_httpd, "POST", "/api/converter/structures", {"structureId": 500, "volume": "RC1"})
    wait_until_done(clean_httpd)

    status, raw, _ = request(clean_httpd, "POST", f"/api/volumes/RC1/update-all?set={SET}", {})
    assert status == 200
    done = wait_until_done(clean_httpd)
    assert done["refreshed"] == 0 and done["skipped"] == 2


def test_idle_status_of_a_structure_without_a_job(clean_httpd, converter_root: Path) -> None:
    (converter_root / "structure-9" / "RC1").mkdir(parents=True)
    status, payload = get_json(clean_httpd, "/api/volumes/RC1/update-all?set=structure-9")
    assert status == 200 and payload["phase"] == "idle" and payload["running"] is False


def test_creating_needs_review_rights_and_valid_input(clean_httpd, monkeypatch: pytest.MonkeyPatch) -> None:
    assert request(clean_httpd, "POST", "/api/converter/structures", {"structureId": "x", "volume": "RC1"})[0] == 400
    assert request(clean_httpd, "POST", "/api/converter/structures", {"structureId": 5, "volume": "../x"})[0] == 400
    assert request(clean_httpd, "POST", "/api/converter/structures", {"structureId": 5, "volume": "NOPE"})[0] == 404
    monkeypatch.setenv("GALLERY_STUB_ROLE", "read")
    assert request(clean_httpd, "POST", "/api/converter/structures", {"structureId": 5, "volume": "RC1"})[0] == 403


def test_an_unconfigured_server_reports_what_is_missing(tmp_path: Path, converter_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "gallery-volumes" / "RC1").mkdir(parents=True)
    jobs._JOBS.clear()
    jobs.OVERRIDES.clear()
    monkeypatch.delenv("GALLERY_SEGMENTATION_TARGET", raising=False)
    server = make_server("127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address[:2]
    try:
        address = (str(host), int(port))
        request(address, "POST", "/api/converter/structures", {"structureId": 500, "volume": "RC1"})
        done = wait_until_done(address)
        assert done["phase"] == "failed" and "GALLERY_SEGMENTATION_TARGET" in done["errors"][0]
    finally:
        server.shutdown()
        server.server_close()
        jobs._JOBS.clear()


def test_cancel_stops_a_running_build(clean_httpd) -> None:
    gate = threading.Event()
    original = jobs.OVERRIDES["make_client"]

    def slow_client():
        client = original()
        inner = client.segment_with_retry

        def slow(*args, **kwargs):
            gate.wait(5)
            return inner(*args, **kwargs)

        client.segment_with_retry = slow
        return client

    jobs.OVERRIDES["make_client"] = slow_client
    request(clean_httpd, "POST", "/api/converter/structures", {"structureId": 500, "volume": "RC1"})
    time.sleep(0.2)
    assert request(clean_httpd, "POST", f"/api/volumes/RC1/update-all/cancel?set={SET}", {})[0] == 200
    gate.set()
    done = wait_until_done(clean_httpd)
    assert done["phase"] == "cancelled" and done["refreshed"] == 1
