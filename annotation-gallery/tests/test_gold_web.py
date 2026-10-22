"""The gold sets over HTTP: draw, review with edits, approval pinned to a mask hash, lists, replacements and the export."""

from __future__ import annotations

import base64
import io
import json
import sys
import threading
import zipfile
from http.client import HTTPConnection
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytest
from PIL import Image

for folder in (Path(__file__).resolve().parents[1], Path(__file__).resolve().parent):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import catalog  # noqa: E402
from gold import store  # noqa: E402
from server import make_server  # noqa: E402
from test_gold_store import make_source  # noqa: E402


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    training = tmp_path / "training"
    make_source(training / "current" / "RC1", locations=40, split=(3,), rejected=(9,))
    make_source(training / "current" / "RC2", locations=30)
    monkeypatch.setenv("GALLERY_VOLUME_DIR", str(training / "current"))
    monkeypatch.setenv("GALLERY_GOLD_ROOT", str(tmp_path / "gold"))
    monkeypatch.setenv("GALLERY_IDENTITY_MODE", "stub")
    monkeypatch.setenv("GALLERY_STUB_ROLE", "review")
    monkeypatch.setenv("GALLERY_SESSION_SECRET", "test-secret")
    return tmp_path / "gold"


@pytest.fixture
def httpd(env: Path):
    server = make_server("127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address[:2]
    try:
        yield str(host), int(port)
    finally:
        server.shutdown()
        server.server_close()


def request(httpd, method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Tuple[int, bytes, Dict[str, str]]:
    connection = HTTPConnection(httpd[0], httpd[1], timeout=20)
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json", "Content-Length": str(len(data))} if data is not None else {}
    connection.request(method, path, body=data, headers=headers)
    response = connection.getresponse()
    raw = response.read()
    result = (response.status, raw, {k.lower(): v for k, v in response.getheaders()})
    connection.close()
    return result


def call(httpd, method: str, path: str, body=None) -> Tuple[int, Any]:
    status, raw, _ = request(httpd, method, path, body)
    return status, (json.loads(raw) if raw else None)


def draw_set(httpd, name: str = "g1", **extra) -> Dict[str, Any]:
    status, payload = call(httpd, "POST", "/api/gold/sets", {"name": name, "n": 6, "seed": 3, **extra})
    assert status == 201, payload
    return payload


def cards_of(httpd, name: str = "g1", volume: str = "RC1"):
    return call(httpd, "GET", f"/api/volumes/{volume}/catalog?set=gold-{name}")[1]["rows"]


def png_b64(mask: np.ndarray) -> str:
    out = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(out, format="PNG")
    return base64.b64encode(out.getvalue()).decode("ascii")


def test_config_page_and_set_listing(httpd) -> None:
    assert call(httpd, "GET", "/api/config")[1]["gold"] is True
    status, raw, headers = request(httpd, "GET", "/gold/")
    assert status == 200 and b"<html" in raw.lower()
    assert call(httpd, "GET", "/api/training-sets?app=gold")[1] == {"sets": [], "default": ""}
    draw_set(httpd)
    assert call(httpd, "GET", "/api/training-sets?app=gold")[1] == {"sets": ["gold-g1"], "default": "gold-g1"}


def test_drawing_makes_n_cards_per_volume_and_records_the_draw(httpd) -> None:
    payload = draw_set(httpd)
    assert set(payload["meta"]["volumes"]) == {"RC1", "RC2"}
    assert {v: i["drawn"] for v, i in payload["meta"]["volumes"].items()} == {"RC1": 6, "RC2": 6}
    assert payload["meta"]["seed"] == 3 and payload["meta"]["sourceSet"] == "current"
    assert len(cards_of(httpd)) == 6 and len(cards_of(httpd, volume="RC2")) == 6
    progress = call(httpd, "GET", "/api/gold/sets/gold-g1/progress")[1]["progress"]
    assert progress["RC1"] == {"target": 6, "cards": 6, "approved": 0, "rejected": 0, "edited": 0, "remaining": 6}


def test_the_same_seed_reproduces_the_sample_and_a_default_seed_is_recorded(httpd) -> None:
    one, two = draw_set(httpd, "a"), draw_set(httpd, "b")
    assert one["meta"]["volumes"]["RC1"]["used"] == two["meta"]["volumes"]["RC1"]["used"]
    status, payload = call(httpd, "POST", "/api/gold/sets", {"name": "c", "n": 4})
    assert status == 201 and isinstance(payload["meta"]["seed"], int)


def test_drawing_can_be_limited_to_one_volume_and_stratified(httpd) -> None:
    payload = draw_set(httpd, "one", volumes=["RC2"], stratify="type")
    assert list(payload["meta"]["volumes"]) == ["RC2"] and payload["meta"]["stratify"] == "type"


@pytest.mark.parametrize(
    "body",
    [{}, {"name": "../x"}, {"name": "ok", "n": 0}, {"name": "ok", "stratify": "nope"}, {"name": "ok", "sourceSet": "gold-g1"},
     {"name": "ok", "sourceSet": "structure-5"}],
)
def test_bad_draw_requests_are_refused(httpd, body) -> None:
    draw_set(httpd)
    assert call(httpd, "POST", "/api/gold/sets", body)[0] in (400, 404)


def test_an_existing_set_is_not_overwritten(httpd) -> None:
    draw_set(httpd)
    status, payload = call(httpd, "POST", "/api/gold/sets", {"name": "g1", "n": 3, "seed": 1})
    assert status == 400 and "already exists" in payload["error"]


def test_a_retraced_outline_edits_the_mask_and_leaves_the_source_alone(httpd, env: Path) -> None:
    draw_set(httpd)
    row = cards_of(httpd)[0]
    key, location_id = row["image_key"], row["location_id"]
    crops = env / "gold-g1" / "RC1"
    source_before = (crops / "source_masks" / f"{key}_{location_id}.png").read_bytes()
    rect = [[4, 4], [12, 4], [12, 10], [4, 10], [4, 4]]

    status, payload = call(httpd, "POST", "/api/volumes/RC1/gold/edit?set=gold-g1", {"location_id": location_id, "image_key": key, "rings": [rect]})

    assert status == 200 and payload["edited"] and payload["changed_pixels"] > 0
    mask = np.asarray(Image.open(crops / "masks" / f"{key}_{location_id}.png")) > 0
    assert abs(int(mask.sum()) - 8 * 6) <= 14
    assert (crops / "source_masks" / f"{key}_{location_id}.png").read_bytes() == source_before
    assert call(httpd, "GET", "/api/gold/sets/gold-g1/progress")[1]["progress"]["RC1"]["edited"] == 1


def test_a_painted_mask_replaces_the_mask(httpd, env: Path) -> None:
    draw_set(httpd)
    row = cards_of(httpd)[0]
    key, location_id = row["image_key"], row["location_id"]
    painted = np.zeros((16, 16), dtype=bool)
    painted[1:5, 1:5] = True
    status, payload = call(httpd, "POST", "/api/volumes/RC1/gold/edit?set=gold-g1", {"location_id": location_id, "image_key": key, "png": "data:image/png;base64," + png_b64(painted)})
    assert status == 200 and payload["mask_pixels"] == 16
    assert (np.asarray(Image.open(env / "gold-g1" / "RC1" / "masks" / f"{key}_{location_id}.png")) > 0).sum() == 16


@pytest.mark.parametrize(
    "payload",
    [{"rings": [[[0, 0], [1, 1]]]}, {"rings": []}, {"png": "!!!"}, {"png": "AAAA"}, {}, {"rings": [[[1, 1], [5, 1], [5, 5], [1, 1]]], "png": "AAAA"}],
)
def test_bad_edits_are_refused_and_change_nothing(httpd, env: Path, payload) -> None:
    draw_set(httpd)
    row = cards_of(httpd)[0]
    key, location_id = row["image_key"], row["location_id"]
    path = env / "gold-g1" / "RC1" / "masks" / f"{key}_{location_id}.png"
    before = path.read_bytes()
    status, answer = call(httpd, "POST", "/api/volumes/RC1/gold/edit?set=gold-g1", {"location_id": location_id, "image_key": key, **payload})
    assert status == 400 and "error" in answer and path.read_bytes() == before


def test_a_painted_mask_of_the_wrong_size_is_refused(httpd) -> None:
    draw_set(httpd)
    row = cards_of(httpd)[0]
    status, answer = call(httpd, "POST", "/api/volumes/RC1/gold/edit?set=gold-g1", {"location_id": row["location_id"], "image_key": row["image_key"], "png": png_b64(np.ones((8, 8), bool))})
    assert status == 400 and "16 x 16" in answer["error"]


def test_approval_pins_the_mask_and_an_edit_withdraws_it(httpd, env: Path) -> None:
    draw_set(httpd)
    row = cards_of(httpd)[0]
    key, location_id = row["image_key"], row["location_id"]
    crops = env / "gold-g1" / "RC1"
    assert call(httpd, "POST", "/api/volumes/RC1/approve?set=gold-g1", {"location_id": location_id, "image_key": key})[0] == 200
    card = {c["location_id"]: c for c in store.cards(crops)}[location_id]
    assert card["approved_by"] and card["approved_sha1"] == store.sha1_of(crops / "masks" / f"{key}_{location_id}.png")
    assert call(httpd, "GET", "/api/gold/sets/gold-g1/progress")[1]["progress"]["RC1"]["approved"] == 1

    rect = [[2, 2], [9, 2], [9, 9], [2, 9], [2, 2]]
    _, edit = call(httpd, "POST", "/api/volumes/RC1/gold/edit?set=gold-g1", {"location_id": location_id, "image_key": key, "rings": [rect]})
    assert edit["unapproved"] is True
    card = {c["location_id"]: c for c in store.cards(crops)}[location_id]
    assert card["approved_by"] is None and card["approved_sha1"] is None
    assert call(httpd, "GET", "/api/gold/sets/gold-g1/progress")[1]["progress"]["RC1"]["approved"] == 0

    call(httpd, "POST", "/api/volumes/RC1/approve?set=gold-g1", {"location_id": location_id, "image_key": key})
    call(httpd, "POST", "/api/volumes/RC1/unapprove?set=gold-g1", {"location_id": location_id, "image_key": key})
    assert {c["location_id"]: c for c in store.cards(crops)}[location_id]["approved_sha1"] is None


def test_export_has_approved_masks_images_and_a_manifest_with_hashes(httpd, env: Path) -> None:
    draw_set(httpd)
    rows = cards_of(httpd)
    for row in rows[:2]:
        call(httpd, "POST", "/api/volumes/RC1/approve?set=gold-g1", {"location_id": row["location_id"], "image_key": row["image_key"]})
    status, raw, headers = request(httpd, "GET", "/api/gold/sets/gold-g1/export?volume=RC1")
    assert status == 200 and headers["content-type"] == "application/zip" and "gold-g1-RC1.zip" in headers["content-disposition"]
    archive = zipfile.ZipFile(io.BytesIO(raw))
    manifest = json.loads(archive.read("manifest.json"))
    assert len(manifest["cards"]) == 2 and manifest["seed"] == 3 and manifest["problems"] == []
    names = set(archive.namelist())
    for card in manifest["cards"]:
        assert f"RC1/masks/{card['image_key']}_{card['location_id']}.png" in names
        assert card["mask_sha1"] and card["approved_by"]
    assert len([n for n in names if n.startswith("RC1/masks/")]) == 2 and any(n.startswith("RC1/images/") for n in names)


def test_export_reports_a_mask_changed_behind_the_pages_back(httpd, env: Path) -> None:
    draw_set(httpd)
    row = cards_of(httpd)[0]
    call(httpd, "POST", "/api/volumes/RC1/approve?set=gold-g1", {"location_id": row["location_id"], "image_key": row["image_key"]})
    path = env / "gold-g1" / "RC1" / "masks" / f"{row['image_key']}_{row['location_id']}.png"
    Image.fromarray(np.zeros((16, 16), np.uint8)).save(path)
    manifest = json.loads(zipfile.ZipFile(io.BytesIO(request(httpd, "GET", "/api/gold/sets/gold-g1/export?volume=RC1")[1])).read("manifest.json"))
    assert len(manifest["problems"]) == 1 and "no longer matches" in manifest["problems"][0]


def test_an_uploaded_list_extends_then_replaces_with_confirmation(httpd, env: Path) -> None:
    meta = draw_set(httpd, volumes=["RC1"])["meta"]
    used = set(meta["volumes"]["RC1"]["used"])
    extra = next(i for i in range(1, 40) if i not in used and i not in (3, 9))
    status, result = call(httpd, "POST", "/api/gold/sets/gold-g1/list", {"volume": "RC1", "mode": "extend", "text": f"# gold extras\n{extra}, 9\nzzz 99999\n"})
    assert status == 200 and result["added"] == [extra] and result["notIds"] == ["zzz"]
    assert result["skipped"] == {"rejected": [9], "notInSource": [99999]}
    assert len(cards_of(httpd)) == 7

    row = cards_of(httpd)[0]
    call(httpd, "POST", "/api/volumes/RC1/approve?set=gold-g1", {"location_id": row["location_id"], "image_key": row["image_key"]})
    status, refused = call(httpd, "POST", "/api/gold/sets/gold-g1/list", {"volume": "RC1", "mode": "replace", "text": "20 21"})
    assert status == 400 and "approved" in refused["error"]
    status, done = call(httpd, "POST", "/api/gold/sets/gold-g1/list", {"volume": "RC1", "mode": "replace", "text": "20 21", "force": True})
    assert status == 200 and sorted(r["location_id"] for r in cards_of(httpd)) == [20, 21]


def test_empty_or_junk_lists_are_refused(httpd) -> None:
    draw_set(httpd)
    status, answer = call(httpd, "POST", "/api/gold/sets/gold-g1/list", {"volume": "RC1", "text": "# nothing\nabc"})
    assert status == 400 and answer["bad"] == ["abc"]


def test_a_rejected_card_is_replaced_by_the_next_in_the_seeded_order(httpd) -> None:
    meta = draw_set(httpd)["meta"]
    row = cards_of(httpd)[0]
    call(httpd, "POST", "/api/volumes/RC1/ignore?set=gold-g1", {"location_id": row["location_id"], "image_key": row["image_key"]})
    status, result = call(httpd, "POST", "/api/gold/sets/gold-g1/replacements", {"volume": "RC1", "count": 1})
    assert status == 200 and len(result["added"]) == 1
    assert result["added"][0] not in meta["volumes"]["RC1"]["used"]
    progress = call(httpd, "GET", "/api/gold/sets/gold-g1/progress")[1]["progress"]["RC1"]
    assert progress["rejected"] == 1 and progress["cards"] == 6 and progress["target"] == 6


def test_unknown_sets_and_bad_names_are_not_found(httpd) -> None:
    draw_set(httpd)
    assert call(httpd, "GET", "/api/gold/sets/gold-nope/progress")[0] == 404
    assert call(httpd, "GET", "/api/volumes/RC1/catalog?set=gold-nope")[0] == 404
    assert call(httpd, "POST", "/api/gold/sets/gold-nope/list", {"volume": "RC1", "text": "1"})[0] == 404


def test_readers_cannot_draw_edit_approve_or_change_lists(httpd, monkeypatch: pytest.MonkeyPatch) -> None:
    draw_set(httpd)
    row = cards_of(httpd)[0]
    monkeypatch.setenv("GALLERY_STUB_ROLE", "read")
    rect = [[2, 2], [9, 2], [9, 9], [2, 9], [2, 2]]
    assert call(httpd, "POST", "/api/gold/sets", {"name": "r", "n": 2})[0] == 403
    assert call(httpd, "POST", "/api/volumes/RC1/gold/edit?set=gold-g1", {"location_id": row["location_id"], "image_key": row["image_key"], "rings": [rect]})[0] == 403
    assert call(httpd, "POST", "/api/volumes/RC1/approve?set=gold-g1", {"location_id": row["location_id"], "image_key": row["image_key"]})[0] == 403
    assert call(httpd, "POST", "/api/gold/sets/gold-g1/list", {"volume": "RC1", "text": "5"})[0] == 403
    assert call(httpd, "GET", "/api/gold/sets/gold-g1/progress")[0] == 200


def test_edits_outside_a_gold_set_are_refused(httpd) -> None:
    assert call(httpd, "POST", "/api/volumes/RC1/gold/edit", {"location_id": 1, "image_key": "k", "rings": [[[0, 0], [1, 0], [1, 1]]]})[0] == 400


def test_gold_is_off_without_a_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_source(tmp_path / "training" / "current" / "RC1", locations=5)
    monkeypatch.setenv("GALLERY_VOLUME_DIR", str(tmp_path / "training" / "current"))
    monkeypatch.delenv("GALLERY_GOLD_ROOT", raising=False)
    monkeypatch.setenv("GALLERY_IDENTITY_MODE", "stub")
    monkeypatch.setenv("GALLERY_STUB_ROLE", "review")
    server = make_server("127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    address = (str(server.server_address[0]), int(server.server_address[1]))
    try:
        assert call(address, "GET", "/api/config")[1]["gold"] is False
        assert request(address, "GET", "/gold/")[0] == 404
        assert call(address, "POST", "/api/gold/sets", {"name": "x"})[0] == 404
    finally:
        server.shutdown()
        server.server_close()


def test_the_gallery_and_its_catalog_are_untouched_by_a_draw(httpd, env: Path) -> None:
    source = env.parent / "training" / "current" / "RC1"
    before = [(r["location_id"], r["ignored"]) for r in catalog.list_catalog_rows(source)]
    draw_set(httpd)
    row = cards_of(httpd)[0]
    call(httpd, "POST", "/api/volumes/RC1/ignore?set=gold-g1", {"location_id": row["location_id"], "image_key": row["image_key"]})
    assert [(r["location_id"], r["ignored"]) for r in catalog.list_catalog_rows(source)] == before


def test_a_set_can_start_from_uploaded_list_files(httpd, env: Path) -> None:
    status, payload = call(httpd, "POST", "/api/gold/sets", {"name": "gold_v1", "lists": {"RC1": "3 5, 7\n# note\n9999", "RC2": "1,2"}})
    assert status == 201 and payload["set"] == "gold-gold_v1"
    volumes = payload["meta"]["volumes"]
    assert volumes["RC1"]["used"] == [3, 5, 7] and volumes["RC1"]["skipped"] == {"notInSource": [9999]}
    assert volumes["RC2"]["used"] == [1, 2] and payload["meta"]["fromLists"] is True
    assert sorted(r["location_id"] for r in cards_of(httpd, "gold_v1")) == [3, 5, 7]
    # The set is a normal gold set: it can be edited and approved.
    row = cards_of(httpd, "gold_v1")[0]
    assert call(httpd, "POST", "/api/volumes/RC1/approve?set=gold-gold_v1", {"location_id": row["location_id"], "image_key": row["image_key"]})[0] == 200


def test_list_files_with_junk_or_unknown_volumes_are_refused(httpd) -> None:
    status, answer = call(httpd, "POST", "/api/gold/sets", {"name": "x1", "lists": {"RC1": "12 abc"}})
    assert status == 400 and "abc" in answer["error"]
    assert call(httpd, "POST", "/api/gold/sets", {"name": "x2", "lists": {"NOPE": "1"}})[0] == 403
    assert call(httpd, "POST", "/api/gold/sets", {"name": "x3", "lists": {"RC1": "# nothing"}})[0] == 400


def test_replacements_are_refused_for_a_list_set(httpd) -> None:
    call(httpd, "POST", "/api/gold/sets", {"name": "gold_v1", "lists": {"RC1": "3 5"}})
    status, answer = call(httpd, "POST", "/api/gold/sets/gold-gold_v1/replacements", {"volume": "RC1"})
    assert status == 400 and "made from a list" in answer["error"]
