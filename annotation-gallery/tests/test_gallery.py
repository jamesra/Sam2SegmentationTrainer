"""Slim gallery HTTP: stub identity, ignore 403, path traversal."""

from __future__ import annotations

import io
import json
import os
import re
import sqlite3
import subprocess
import sys
import threading
import zipfile
from http.client import HTTPConnection
from pathlib import Path

import pytest

GALLERY_ROOT = Path(__file__).resolve().parents[1]
if str(GALLERY_ROOT) not in sys.path:
    sys.path.insert(0, str(GALLERY_ROOT))

from catalog import (  # noqa: E402
    approve_window,
    connect,
    ignore_location,
    list_catalog_rows,
    restore_location,
    review_locations,
)
from identity import OidcIdentity, map_permission_names  # noqa: E402
from server import (  # noqa: E402
    https_redirect_location,
    list_registry_names,
    make_server,
    safe_crop_file,
    structure_ids_for_tag_pattern,
    tag_tokens,
    volume_root,
)
import thumbs  # noqa: E402
from thumbs import ensure_thumbnail, remember_mask_index  # noqa: E402

_MIN_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x00\x00\x00\x00:\x7e\x9bU\x00\x00\x00\nIDATx\x9cc\xf8\x0f\x00\x01"
    b"\x01\x01\x00\x1b\xb6\xeeV\x00\x00\x00\x00IEND\xaeB`\x82"
)


def test_connect_drops_window_origin_columns(tmp_path: Path) -> None:
    crops = tmp_path / "AnnotationCrops"
    crops.mkdir()
    connection = connect(crops)
    connection.execute("ALTER TABLE locations ADD COLUMN origin_x INTEGER")
    connection.execute("ALTER TABLE locations ADD COLUMN origin_y INTEGER")
    connection.commit()
    connection.close()
    connection = connect(crops)
    columns = {row[1] for row in connection.execute("PRAGMA table_info(locations)")}
    connection.close()
    assert "origin_x" not in columns
    assert "origin_y" not in columns


def _seed_crops(crops: Path, *, location_id: int = 42) -> None:
    images = crops / "images"
    masks = crops / "masks"
    images.mkdir(parents=True)
    masks.mkdir()
    (images / "RC2_17_D1_X0_Y0.png").write_bytes(_MIN_PNG)
    (masks / f"RC2_17_D1_X0_Y0_{location_id}.png").write_bytes(_MIN_PNG)
    connection = connect(crops)
    connection.execute(
        "INSERT INTO locations (location_id, z, structure_id, structure_label, "
        "type_id, type_name, radius, image_key, image_relpath, mask_relpath, ignored) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
        [
            location_id,
            17,
            7,
            "soma",
            1,
            "Cell",
            12.5,
            "RC2_17_D1_X0_Y0",
            "images/RC2_17_D1_X0_Y0.png",
            f"masks/RC2_17_D1_X0_Y0_{location_id}.png",
        ],
    )
    connection.commit()
    connection.close()


@pytest.fixture
def registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    registry_path = tmp_path / "gallery-volumes"
    volume = registry_path / "RC2"
    crops = volume / "AnnotationCrops"
    volume.mkdir(parents=True)
    _seed_crops(crops)
    monkeypatch.setenv("GALLERY_VOLUME_DIR", str(registry_path))
    monkeypatch.setenv("GALLERY_IDENTITY_MODE", "stub")
    monkeypatch.setenv("GALLERY_STUB_ROLE", "read")
    monkeypatch.setenv("GALLERY_SESSION_SECRET", "test-secret")
    return registry_path


@pytest.fixture
def httpd(registry: Path) -> tuple[str, int]:
    del registry
    server = make_server("127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield str(host), int(port)
    finally:
        server.shutdown()
        server.server_close()


def _request(
    httpd: tuple[str, int],
    method: str,
    path: str,
    *,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
    bearer: str | None = None,
) -> tuple[int, dict]:
    connection = HTTPConnection(httpd[0], httpd[1], timeout=5)
    extra = dict(headers or {})
    if bearer:
        extra["Authorization"] = f"Bearer {bearer}"
    if body is not None:
        extra.setdefault("Content-Type", "application/json")
        extra["Content-Length"] = str(len(body))
    connection.request(method, path, body=body, headers=extra)
    response = connection.getresponse()
    raw = response.read()
    connection.close()
    payload = json.loads(raw.decode("utf-8")) if raw else {}
    return response.status, payload


def test_catalog_loss_is_latest_score_or_null(httpd: tuple[str, int], registry: Path) -> None:
    crops = registry / "RC2" / "AnnotationCrops"
    connection = connect(crops)
    connection.execute(
        "INSERT INTO locations (location_id, z, structure_id, structure_label, "
        "type_id, type_name, radius, image_key, image_relpath, mask_relpath, ignored) "
        "VALUES (99, 18, 7, 'soma', 1, 'Cell', 4.0, 'RC2_18_D1_X0_Y0', "
        "'images/RC2_18_D1_X0_Y0.png', 'masks/RC2_18_D1_X0_Y0_99.png', 0)"
    )
    connection.executemany(
        "INSERT INTO location_scores (location_id, image_key, epoch, score) VALUES (?, ?, ?, ?)",
        [
            (42, "RC2_17_D1_X0_Y0", 1, 1.5),
            (42, "RC2_17_D1_X0_Y0", 3, 0.25),
        ],
    )
    connection.commit()
    connection.close()
    status, catalog = _request(httpd, "GET", "/api/volumes/RC2/catalog", bearer="read")
    assert status == 200
    by_id = {row["location_id"]: row for row in catalog["rows"]}
    assert by_id[42]["loss"] == 0.25
    assert by_id[99]["loss"] is None


def test_catalog_loss_is_per_window_with_legacy_fallback(tmp_path: Path) -> None:
    crops = tmp_path / "AnnotationCrops"
    crops.mkdir()
    connection = sqlite3.connect(crops / "annotation_crops.sqlite")
    connection.execute(
        "CREATE TABLE locations ("
        "location_id INTEGER NOT NULL, z INTEGER NOT NULL, structure_id INTEGER, "
        "structure_label TEXT, type_id INTEGER, type_name TEXT, radius REAL, "
        "image_key TEXT NOT NULL, image_relpath TEXT, mask_relpath TEXT, "
        "ignored INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (location_id, image_key))"
    )
    connection.execute(
        "CREATE TABLE location_scores ("
        "location_id INTEGER NOT NULL, epoch INTEGER NOT NULL, score REAL NOT NULL, "
        "PRIMARY KEY (location_id, epoch))"
    )
    connection.executemany(
        "INSERT INTO locations (location_id, z, structure_id, structure_label, "
        "type_id, type_name, radius, image_key, image_relpath, mask_relpath, ignored) "
        "VALUES (?, 17, 7, 'soma', 1, 'Cell', 12.5, ?, ?, ?, 0)",
        [
            (42, "win-a", "images/a.png", "masks/a_42.png"),
            (42, "win-b", "images/b.png", "masks/b_42.png"),
            (99, "win-c", "images/c.png", "masks/c_99.png"),
        ],
    )
    connection.execute(
        "INSERT INTO location_scores (location_id, epoch, score) VALUES (42, 3, 0.25)"
    )
    connection.commit()
    connection.close()
    connection = connect(crops)
    connection.execute(
        "INSERT INTO location_scores (location_id, image_key, epoch, score) "
        "VALUES (42, 'win-a', 4, 1.5)"
    )
    connection.commit()
    connection.close()
    by_key = {row["image_key"]: row for row in list_catalog_rows(crops)}
    assert by_key["win-a"]["loss"] == 1.5
    assert by_key["win-b"]["loss"] == 0.25
    assert by_key["win-c"]["loss"] is None


def test_training_sets_defaults_to_current_outside_a_version_tree(httpd: tuple[str, int]) -> None:
    status, payload = _request(httpd, "GET", "/api/training-sets", bearer="read")
    assert status == 200
    assert payload["default"] == "current"
    assert payload["sets"] == ["current"]


def test_read_lists_volume_and_catalog(httpd: tuple[str, int]) -> None:
    status, me = _request(httpd, "GET", "/api/me", bearer="read")
    assert status == 200
    assert me["volumes"] == [{"name": "RC2", "permission": "read", "identityName": "RC2"}]
    status, catalog = _request(httpd, "GET", "/api/volumes/RC2/catalog", bearer="read")
    assert status == 200
    assert catalog["rows"][0]["location_id"] == 42


def test_read_ignore_is_403(httpd: tuple[str, int]) -> None:
    status, payload = _request(
        httpd,
        "POST",
        "/api/volumes/RC2/ignore",
        body=json.dumps({"location_id": 42}).encode("utf-8"),
        bearer="read",
    )
    assert status == 403
    assert "review" in payload["error"]


def test_review_can_ignore_and_restore(httpd: tuple[str, int], registry: Path) -> None:
    status, payload = _request(
        httpd,
        "POST",
        "/api/volumes/RC2/ignore",
        body=json.dumps({"location_id": 42}).encode("utf-8"),
        bearer="review",
    )
    assert status == 200
    assert payload["ignored"] is True
    crops = registry / "RC2" / "AnnotationCrops"
    assert (crops / "ignored" / "RC2_17_D1_X0_Y0_42.png").is_file()
    status, payload = _request(
        httpd,
        "POST",
        "/api/volumes/RC2/restore",
        body=json.dumps({"location_id": 42}).encode("utf-8"),
        bearer="review",
    )
    assert status == 200
    assert payload["ignored"] is False
    assert (crops / "masks" / "RC2_17_D1_X0_Y0_42.png").is_file()


def test_location_search_filters_and_sort_stacks() -> None:
    html = (GALLERY_ROOT / "static" / "index.html").read_text(encoding="utf-8")
    js = (GALLERY_ROOT / "static" / "app.js").read_text(encoding="utf-8")
    assert 'id="locate"' in html
    assert 'href="https://tensor.codepharm.net"' in html
    assert "Show only this location id" in html
    assert "function parseLocationIds" in js
    assert "function showLocations" in js
    assert '["structure_id", "Structure ID"]' in js
    assert '["location_id", "Location ID"]' in js
    assert "SORTS_PER_COLUMN = 5" in js
    assert 'id="sort-add"' in js
    assert "maxSortRows" not in js


def test_header_links_the_gallery_pages_to_each_other() -> None:
    html = (GALLERY_ROOT / "static" / "index.html").read_text(encoding="utf-8")
    js = (GALLERY_ROOT / "static" / "app.js").read_text(encoding="utf-8")
    nav = html.split('<nav id="page-links"', 1)[1].split("</nav>", 1)[0]
    # Each page link starts hidden; boot shows the ones the server switched on, other than the page shown now.
    for page, href in (("gallery", "/"), ("converter", "/converter/"), ("gold", "/gold/")):
        assert re.search(rf'<a [^>]*data-page="{page}" href="{re.escape(href)}" hidden', nav)
    assert 'href="https://tensor.codepharm.net"' in nav
    show = js.split("function showPageLinks()", 1)[1].split("\n  }\n", 1)[0]
    assert "state.config.converter" in show and "state.config.gold" in show
    assert "link.dataset.page === current" in show
    boot = js.split("async function boot()", 1)[1]
    assert boot.index("showPageLinks();") < boot.index("/api/training-sets")


def test_filters_collapse_to_a_summary_bar() -> None:
    html = (GALLERY_ROOT / "static" / "index.html").read_text(encoding="utf-8")
    js = (GALLERY_ROOT / "static" / "app.js").read_text(encoding="utf-8")
    css = (GALLERY_ROOT / "static" / "app.css").read_text(encoding="utf-8")
    assert 'id="filters-toggle"' in html and 'aria-controls="filters"' in html
    assert 'id="filter-summary"' in html
    # The collapsed bar replaces the controls; the filters themselves are untouched.
    assert "body.filters-collapsed .filters" in css and "display: none" in css.split("body.filters-collapsed .filters", 1)[1].split("}", 1)[0]
    assert ".filter-summary" in css and "text-overflow: ellipsis" in css
    assert "function describeFilters" in js and "function setFiltersCollapsed" in js
    assert 'COLLAPSE_KEY = "galleryFiltersCollapsed"' in js
    # The summary follows every filter change, and the cards are laid out again when the bar changes height.
    render = js.split("function render(", 1)[1].split("const scroll", 1)[0]
    assert "updateFilterSummary()" in render
    toggle = js.split('els.filtersToggle.addEventListener("click"', 1)[1].split("});", 1)[0]
    assert "render(true)" in toggle


def test_each_window_has_review_icons_and_names_thumbnail_state() -> None:
    js = (GALLERY_ROOT / "static" / "app.js").read_text(encoding="utf-8")
    css = (GALLERY_ROOT / "static" / "app.css").read_text(encoding="utf-8")
    window_html = js.split("function windowHtml", 1)[1].split("function shapeOf", 1)[0]
    assert "data-act" not in window_html
    assert "windowActions(row)" in js
    assert "data-key" in js
    assert "function packGrid" in js
    assert "function windowsForLocation" in js
    mutate = js.split("async function mutate", 1)[1].split("async function applyRememberedStatus", 1)[0]
    assert "loadCatalog(" not in mutate
    assert "paintReviewNow(locationId)" in mutate
    assert "forgetVisibleLocation(locationId)" in mutate
    assert "await afterPaint()" in mutate
    assert "render(true, true)" in mutate
    assert "function reconcileGrid" in js
    assert 'postReview(act, locationId, "")' in mutate
    assert ".card[hidden]" in css
    assert "visibleKeys" in js
    pack = js.split("function packGrid", 1)[1].split("function layoutCards", 1)[0]
    assert "byLocation" in pack
    assert "anchorStarts" in pack
    assert "windowsForLocation(" not in pack
    assert "windows[0]" not in pack
    assert "function windowOrigin" in js
    assert "Math.floor(dc / columns)" in js
    opener = js.split("async function openViewer", 1)[1].split("function viewerRow", 1)[0]
    assert "images/${imageKey}.png" in opener
    assert "loadParts" not in opener
    assert 'closest(".card")' in js
    assert "width: 32px" in css
    assert "rgba(35, 40, 51, 0.25)" in css
    assert "rgba(20, 83, 45, 0.25)" in css
    assert "Do not change this without explicit permission" in css
    assert 'document.body.dataset.cardSize = state.cardSize' in js
    assert 'const VIEW_KEY = "galleryView"' in js
    assert "sets.includes(rememberedSet)" in js
    assert "readView().volumes[state.trainingSet]" in js
    assert 'body[data-card-size="small"] .card > .icon.check' in css
    assert "scale(0.75)" in css
    assert "const GRID_GAP = 4" in js
    assert "setGridMasksHidden" in js
    assert "grid-hide-masks" in css
    assert "function assignLinkHues" in js
    assert "function assignJoins" in js
    assert "join-n" in js
    assert "join-n" in css
    assert "border-top-width: 0" in css
    assert 'names.push("solo")' in js
    assert ".card.solo::after" in css
    assert "content: none" in css.split(".card.solo::after", 1)[1].split("}", 1)[0]
    assert "function linkToneIndex" in js
    assert "link-hot" in js
    assert "link-hot" in css
    assert "refreshMarks" in js
    refresh = js.split("async function refreshLocation", 1)[1].split("// Update all: one background job", 1)[0]
    assert "loadCatalog(" not in refresh
    assert "maskGeneration" not in js
    assert "locationGenerations.set" in refresh
    assert "repaintLocation(id)" in refresh
    assert "button.refresh.busy" in css
    assert "cardPx + 20" in js
    assert "align-self: start" in css
    assert 'content: "Generating thumbnail"' in css
    assert 'content: "Image missing"' in css


def test_ignore_keeps_a_mask_for_every_window(tmp_path: Path) -> None:
    crops = tmp_path / "AnnotationCrops"
    crops.mkdir()
    keys = [
        "RC2_17_D1_X0-1024_Y0-1024",
        "RC2_17_D1_X1024-2048_Y0-1024",
    ]
    connection = sqlite3.connect(crops / "annotation_crops.sqlite")
    connection.execute(
        "CREATE TABLE locations ("
        "location_id INTEGER NOT NULL, z INTEGER NOT NULL, structure_id INTEGER, "
        "structure_label TEXT, type_id INTEGER, type_name TEXT, radius REAL, "
        "image_key TEXT NOT NULL, image_relpath TEXT, mask_relpath TEXT, "
        "ignored INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (location_id, image_key))"
    )
    connection.executemany(
        "INSERT INTO locations (location_id, z, structure_id, structure_label, "
        "type_id, type_name, radius, image_key, image_relpath, mask_relpath, ignored) "
        "VALUES (?, 17, 7, 'soma', 1, 'Cell', 12.5, ?, ?, ?, 0)",
        [
            (7, key, f"images/{key}.png", f"masks/{key}_7.png")
            for key in keys
        ],
    )
    connection.commit()
    connection.close()
    masks = crops / "masks"
    masks.mkdir()
    for key in keys:
        (masks / f"{key}_7.png").write_bytes(b"png")
    ignore_location(crops, 7)
    by_key = {row["image_key"]: row for row in list_catalog_rows(crops)}
    assert by_key[keys[0]]["mask_relpath"] == f"ignored/{keys[0]}_7.png"
    assert by_key[keys[1]]["mask_relpath"] == f"ignored/{keys[1]}_7.png"
    assert (crops / "ignored" / f"{keys[0]}_7.png").is_file()
    assert (crops / "ignored" / f"{keys[1]}_7.png").is_file()
    connection = sqlite3.connect(crops / "annotation_crops.sqlite")
    connection.execute(
        "UPDATE locations SET mask_relpath = ? WHERE location_id = 7",
        [f"ignored/{keys[0]}_7.png"],
    )
    connection.commit()
    connection.close()
    by_key = {row["image_key"]: row for row in list_catalog_rows(crops)}
    assert by_key[keys[1]]["mask_relpath"] == f"ignored/{keys[1]}_7.png"
    restore_location(crops, 7)
    by_key = {row["image_key"]: row for row in list_catalog_rows(crops)}
    assert by_key[keys[0]]["mask_relpath"] == f"masks/{keys[0]}_7.png"
    assert by_key[keys[1]]["mask_relpath"] == f"masks/{keys[1]}_7.png"
    assert (crops / "masks" / f"{keys[1]}_7.png").is_file()


def test_location_review_covers_every_window(tmp_path: Path) -> None:
    crops = tmp_path / "AnnotationCrops"
    crops.mkdir()
    connection = sqlite3.connect(crops / "annotation_crops.sqlite")
    connection.execute(
        "CREATE TABLE locations ("
        "location_id INTEGER NOT NULL, z INTEGER NOT NULL, structure_id INTEGER, "
        "structure_label TEXT, type_id INTEGER, type_name TEXT, radius REAL, "
        "image_key TEXT NOT NULL, image_relpath TEXT, mask_relpath TEXT, "
        "ignored INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (location_id, image_key))"
    )
    connection.executemany(
        "INSERT INTO locations (location_id, z, structure_id, structure_label, "
        "type_id, type_name, radius, image_key, image_relpath, mask_relpath, ignored) "
        "VALUES (?, 17, 7, 'soma', 1, 'Cell', 12.5, ?, ?, ?, 0)",
        [
            (7, "RC2_17_D1_X0-1024_Y0-1024", "images/a.png", "masks/a_7.png"),
            (7, "RC2_17_D1_X1024-2048_Y0-1024", "images/b.png", "masks/b_7.png"),
        ],
    )
    connection.commit()
    connection.close()
    ignore_location(crops, 7)
    by_key = {row["image_key"]: row for row in list_catalog_rows(crops)}
    assert by_key["RC2_17_D1_X0-1024_Y0-1024"]["ignored"] == 1
    assert by_key["RC2_17_D1_X1024-2048_Y0-1024"]["ignored"] == 1
    restore_location(crops, 7)
    approve_window(crops, 7)
    by_key = {row["image_key"]: row for row in list_catalog_rows(crops)}
    assert by_key["RC2_17_D1_X0-1024_Y0-1024"]["approved"] == 1
    assert by_key["RC2_17_D1_X1024-2048_Y0-1024"]["approved"] == 1
    assert by_key["RC2_17_D1_X0-1024_Y0-1024"]["ignored"] == 0
    assert by_key["RC2_17_D1_X1024-2048_Y0-1024"]["ignored"] == 0


def test_approve_without_image_key_covers_the_location(httpd: tuple[str, int], registry: Path) -> None:
    status, payload = _request(
        httpd,
        "POST",
        "/api/volumes/RC2/approve",
        body=json.dumps({"location_id": 42}).encode("utf-8"),
        bearer="review",
    )
    assert status == 200
    assert payload["approved"] is True
    status, catalog = _request(httpd, "GET", "/api/volumes/RC2/catalog", bearer="read")
    assert catalog["rows"][0]["approved"] == 1
    status, exported = _request(httpd, "GET", "/api/volumes/RC2/ignore-list", bearer="read")
    assert exported["approved"] == [{"location_id": 42, "image_key": "RC2_17_D1_X0_Y0"}]


def test_approve_hides_without_moving_the_mask(httpd: tuple[str, int], registry: Path) -> None:
    status, payload = _request(
        httpd,
        "POST",
        "/api/volumes/RC2/approve",
        body=json.dumps({"location_id": 42, "image_key": "RC2_17_D1_X0_Y0"}).encode("utf-8"),
        bearer="review",
    )
    assert status == 200
    assert payload["approved"] is True
    crops = registry / "RC2" / "AnnotationCrops"
    assert (crops / "masks" / "RC2_17_D1_X0_Y0_42.png").is_file()
    assert (crops / "approved.json").is_file()
    status, catalog = _request(httpd, "GET", "/api/volumes/RC2/catalog", bearer="read")
    assert catalog["rows"][0]["approved"] == 1
    assert catalog["rows"][0]["ignored"] == 0
    status, exported = _request(httpd, "GET", "/api/volumes/RC2/ignore-list", bearer="read")
    assert exported["approved"] == [{"location_id": 42, "image_key": "RC2_17_D1_X0_Y0"}]
    assert exported["rejected"] == []
    status, _payload = _request(
        httpd,
        "POST",
        "/api/volumes/RC2/unapprove",
        body=json.dumps({"location_id": 42, "image_key": "RC2_17_D1_X0_Y0"}).encode("utf-8"),
        bearer="review",
    )
    assert status == 200
    status, catalog = _request(httpd, "GET", "/api/volumes/RC2/catalog", bearer="read")
    assert catalog["rows"][0]["approved"] == 0


def test_repair_pack_contains_ignored_mask(httpd: tuple[str, int]) -> None:
    status, _payload = _request(
        httpd,
        "POST",
        "/api/volumes/RC2/ignore",
        body=json.dumps({"location_id": 42}).encode("utf-8"),
        bearer="review",
    )
    assert status == 200
    connection = HTTPConnection(httpd[0], httpd[1], timeout=5)
    connection.request("GET", "/api/volumes/RC2/repair-pack", headers={"Authorization": "Bearer read"})
    response = connection.getresponse()
    body = response.read()
    connection.close()
    assert response.status == 200
    archive = zipfile.ZipFile(io.BytesIO(body))
    names = set(archive.namelist())
    assert "manifest.json" in names
    assert any(name.startswith("masks/") for name in names)
    manifest = json.loads(archive.read("manifest.json"))
    assert manifest["locations"][0]["locationId"] == 42


def test_ignore_list_exports_and_imports(httpd: tuple[str, int], registry: Path) -> None:
    status, exported = _request(httpd, "GET", "/api/volumes/RC2/ignore-list", bearer="read")
    assert status == 200
    assert exported == {"volume": "RC2", "rejected": [], "locationIds": [], "approved": []}
    status, payload = _request(
        httpd,
        "POST",
        "/api/volumes/RC2/ignore-list",
        body=json.dumps({"volume": "RC1", "locationIds": [42, 99]}).encode("utf-8"),
        bearer="review",
    )
    assert status == 200
    assert payload["added"] == 2
    assert payload["total"] == 2
    assert payload["moved"] == 1
    crops = registry / "RC2" / "AnnotationCrops"
    assert (crops / "ignored" / "RC2_17_D1_X0_Y0_42.png").is_file()
    status, exported = _request(httpd, "GET", "/api/volumes/RC2/ignore-list", bearer="read")
    assert exported["rejected"] == [42, 99]
    assert exported["locationIds"] == [42, 99]
    assert exported["approved"] == []


def test_training_set_review_lists_cover_every_volume(httpd: tuple[str, int], registry: Path) -> None:
    other = registry / "RC1" / "AnnotationCrops"
    other.mkdir(parents=True)
    status, exported = _request(httpd, "GET", "/api/review-lists", bearer="read")
    assert status == 200
    assert {item["volume"] for item in exported["volumes"]} == {"RC1", "RC2"}
    status, first = _request(
        httpd,
        "POST",
        "/api/review-lists",
        body=json.dumps({
            "volumes": [
                {"volume": "RC2", "rejected": [42], "approved": []},
                {"volume": "RC1", "rejected": [7], "approved": []},
                {"volume": "ZZ9", "rejected": [3], "approved": []},
            ],
        }).encode("utf-8"),
        bearer="review",
    )
    assert status == 200
    assert {item["volume"] for item in first["applied"]} == {"RC1", "RC2"}
    assert all(item["replaced"] for item in first["applied"])
    assert first["held"] == ["ZZ9"]
    assert (registry / "_review" / "ZZ9.json").is_file()
    assert not (registry / "ZZ9").exists()
    status, second = _request(
        httpd,
        "POST",
        "/api/review-lists",
        body=json.dumps({"volumes": [{"volume": "RC2", "rejected": [99], "approved": []}]}).encode("utf-8"),
        bearer="review",
    )
    assert status == 200
    assert second["applied"][0]["replaced"] is False
    assert second["applied"][0]["total"] == 2
    status, exported = _request(httpd, "GET", "/api/review-lists", bearer="read")
    by_name = {item["volume"]: item for item in exported["volumes"]}
    assert by_name["RC2"]["rejected"] == [42, 99]
    assert by_name["ZZ9"]["present"] is False
    assert by_name["ZZ9"]["rejected"] == [3]


def test_settings_save_odata_url_and_exporter(httpd: tuple[str, int], registry: Path) -> None:
    status, payload = _request(httpd, "GET", "/api/volumes/RC2/settings", bearer="read")
    assert status == 200
    assert payload["exporter"] == "tiled"
    assert payload["odata"] == ""
    assert payload["placeholder"] == "https://websvc.codepharm.net/RC2/OData"
    status, denied = _request(
        httpd,
        "POST",
        "/api/volumes/RC2/settings",
        body=json.dumps({"odata": "https://example.test/RC2/OData", "exporter": "legacy"}).encode(),
        bearer="read",
    )
    assert status == 403
    status, saved = _request(
        httpd,
        "POST",
        "/api/volumes/RC2/settings",
        body=json.dumps({
            "odata": "https://example.test/RC2/OData",
            "exporter": "legacy",
            "connection": "Server=secret;Password=nope",
        }).encode(),
        bearer="review",
    )
    assert status == 200
    assert saved["odata"] == "https://example.test/RC2/OData"
    assert saved["exporter"] == "legacy"
    assert saved["kind"] == "odata"
    crops = registry / "RC2" / "AnnotationCrops"
    text = (crops / "source.json").read_text(encoding="utf-8")
    assert "secret" not in text
    assert "connection" not in text
    connection = connect(crops)
    row = connection.execute("SELECT odata, kind, exporter FROM source WHERE id = 1").fetchone()
    connection.close()
    assert row["odata"] == "https://example.test/RC2/OData"
    assert row["kind"] == "odata"
    assert row["exporter"] == "legacy"
    (crops / "source.json").write_text(json.dumps({"kind": "sql", "path": "C:/dump.jsonl"}), encoding="utf-8")
    status, saved = _request(
        httpd,
        "POST",
        "/api/volumes/RC2/settings",
        body=json.dumps({"odata": "https://example.test/RC2/OData", "exporter": "tiled"}).encode(),
        bearer="review",
    )
    assert status == 200
    assert saved["kind"] == "sql"
    assert saved["path"] == "C:/dump.jsonl"
    assert saved["odata"] == "https://example.test/RC2/OData"
    assert "connection" not in (crops / "source.json").read_text(encoding="utf-8")


def test_refresh_without_nornir_leaves_masks(
    httpd: tuple[str, int],
    registry: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NORNIR_BUILDMANAGER_DIR", str(registry / "missing-nornir"))
    crops = registry / "RC2" / "AnnotationCrops"
    status, _saved = _request(
        httpd,
        "POST",
        "/api/volumes/RC2/settings",
        body=json.dumps({"odata": "https://example.test/RC2/OData", "exporter": "tiled"}).encode(),
        bearer="review",
    )
    assert status == 200
    mask = crops / "masks" / "RC2_17_D1_X0_Y0_42.png"
    before = mask.read_bytes()
    thumb = crops / "overlays" / "thumb" / "256" / "RC2_17_D1_X0_Y0.jpg"
    thumb.parent.mkdir(parents=True)
    thumb.write_bytes(b"jpeg")
    status, payload = _request(
        httpd,
        "POST",
        "/api/volumes/RC2/refresh-location",
        body=json.dumps({"location_id": 42}).encode(),
        bearer="review",
    )
    assert status == 503
    assert "nornir-buildmanager" in payload["error"]
    assert mask.read_bytes() == before
    assert thumb.read_bytes() == b"jpeg"


def test_refresh_drops_thumbnails_for_existing_crops(
    httpd: tuple[str, int],
    registry: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("server.nornir_buildmanager_root", lambda: registry)

    def fake_fetch(url: str, location_id: int) -> dict:
        assert url == "https://example.test/RC2/OData"
        return {"ID": location_id, "MosaicShape": {}, "LastModified": "2022-01-01T00:00:00Z"}

    def fake_refresh(crops_path: Path, entity: dict, exporter: str) -> list[str]:
        assert int(entity["ID"]) == 42
        assert exporter == "legacy"
        return ["RC2_17_D1_X0_Y0"]

    monkeypatch.setattr("server.fetch_odata_location", fake_fetch)
    monkeypatch.setattr("server.refresh_location", fake_refresh)
    status, _saved = _request(
        httpd,
        "POST",
        "/api/volumes/RC2/settings",
        body=json.dumps({
            "odata": "https://example.test/RC2/OData",
            "exporter": "legacy",
            "kind": "sql",
        }).encode(),
        bearer="review",
    )
    assert status == 200
    crops = registry / "RC2" / "AnnotationCrops"
    thumb = crops / "overlays" / "thumb" / "512" / "RC2_17_D1_X0_Y0.jpg"
    thumb.parent.mkdir(parents=True)
    thumb.write_bytes(b"jpeg")
    image = crops / "images" / "RC2_17_D1_X0_Y0.png"
    image_before = image.read_bytes()
    status, payload = _request(
        httpd,
        "POST",
        "/api/volumes/RC2/refresh-location",
        body=json.dumps({"location_id": 42}).encode(),
        bearer="review",
    )
    assert status == 200
    assert payload["imageKeys"] == ["RC2_17_D1_X0_Y0"]
    assert not thumb.is_file()
    assert image.read_bytes() == image_before


def test_refresh_uses_the_default_odata_endpoint(
    httpd: tuple[str, int],
    registry: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("server.nornir_buildmanager_root", lambda: registry)
    seen: list[str] = []

    def fake_fetch(url: str, location_id: int) -> dict:
        seen.append(url)
        return {"ID": location_id, "MosaicShape": {}, "LastModified": "2022-01-01T00:00:00Z"}

    monkeypatch.setattr("server.fetch_odata_location", fake_fetch)
    monkeypatch.setattr("server.refresh_location", lambda *_args, **_kwargs: ["RC2_17_D1_X0_Y0"])
    status, payload = _request(
        httpd,
        "POST",
        "/api/volumes/RC2/refresh-location",
        body=json.dumps({"location_id": 42}).encode(),
        bearer="review",
    )
    assert status == 200
    assert payload["imageKeys"] == ["RC2_17_D1_X0_Y0"]
    assert seen == ["https://websvc.codepharm.net/RC2/OData"]


def test_mask_refresh_import_skips_the_export_pipeline() -> None:
    root = Path(r"D:\src\git\nornir\nornir-buildmanager")
    if not (root / "nornir_buildmanager" / "operations" / "segmentationtraining" / "maskrefresh.py").is_file():
        pytest.skip("nornir-buildmanager checkout is not available")
    gallery = Path(__file__).resolve().parents[1]
    script = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from refresh import load_mask_refresh
before = set(sys.modules)
load_mask_refresh(Path(sys.argv[2]))
added = set(sys.modules) - before
forbidden = (
    "nornir_imageregistration",
    "nornir_pools",
    "nornir_shared",
    "nornir_buildmanager.build",
    "nornir_buildmanager.operations.segmentationtraining.pipeline",
    "nornir_buildmanager.operations.segmentationtraining.update",
    "nornir_buildmanager.operations.segmentationtraining.stitch",
    "nornir_buildmanager.operations.segmentationtraining.ingest",
    "nornir_buildmanager.operations.segmentationtraining.catalog",
)
bad = [name for name in sorted(added) if name in forbidden or any(name.startswith(item + ".") for item in forbidden)]
if bad:
    raise SystemExit("imported " + ", ".join(bad))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(gallery), str(root)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_path_traversal_rejected(httpd: tuple[str, int], registry: Path) -> None:
    status, payload = _request(
        httpd,
        "GET",
        "/api/volumes/RC2/file?path=../../etc/passwd",
        bearer="read",
    )
    assert status == 400
    status, payload = _request(
        httpd,
        "GET",
        "/api/volumes/../RC2/catalog",
        bearer="read",
    )
    assert status == 404
    with pytest.raises(ValueError):
        safe_crop_file(registry / "RC2" / "AnnotationCrops", "../AnnotationCrops/images/x.jpg")
    with pytest.raises(ValueError):
        safe_crop_file(registry / "RC2" / "AnnotationCrops", "overlays/other/1.json")
    with pytest.raises(ValueError):
        safe_crop_file(registry / "RC2" / "AnnotationCrops", "overlays/thumb/100/x.jpg")
    with pytest.raises(ValueError):
        safe_crop_file(registry / "RC2" / "AnnotationCrops", "overlays/thumb/512/../x.jpg")
    with pytest.raises(ValueError):
        safe_crop_file(registry / "RC2" / "AnnotationCrops", "overlays/thumb/512/foo.png")
    with pytest.raises(ValueError):
        volume_root(registry, "..")
    crops = registry / "RC2" / "AnnotationCrops"
    thumb = safe_crop_file(crops, "overlays/thumb/512/RC2_17_D1_X0_Y0.jpg")
    assert thumb.name == "RC2_17_D1_X0_Y0.jpg"
    allowed = safe_crop_file(crops, "overlays/parts/42.json")
    assert allowed.name == "42.json"
    assert list_registry_names(registry) == ["RC2"]


def test_spa_hides_review_controls_without_review() -> None:
    css = (GALLERY_ROOT / "static" / "app.css").read_text(encoding="utf-8")
    assert "body.no-review button.trash" in css
    js = (GALLERY_ROOT / "static" / "app.js").read_text(encoding="utf-8")
    assert "no-review" in js
    html = (GALLERY_ROOT / "static" / "index.html").read_text(encoding="utf-8")
    assert 'data-card-size="huge"' in html
    assert 'data-card-size="large"' in html
    assert 'data-card-size="medium"' in html
    assert 'data-card-size="small"' in html
    assert 'class="jpeg"' in html or "jpeg" in js


def test_dockerfile_has_no_nornir() -> None:
    lines = [
        line
        for line in (GALLERY_ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    text = "\n".join(lines).lower()
    assert "nornir-buildmanager" not in text
    assert "imageregistration" not in text
    assert "cupy" not in text
    assert "torch" not in text
    assert "refresh-export" not in text
    assert "nornir_shared" not in text
    # The image needs Pillow and NumPy for the negative-context bitmask (annotation_crops.negatives), nothing heavier.
    assert "pip install --no-cache-dir" in text and "pillow" in text and "numpy" in text
    assert "annotation-gallery/negatives_view.py" in text
    assert "copy annotation_crops /app/annotation_crops" in text


def test_https_redirect_location_public_origin_wins() -> None:
    assert (
        https_redirect_location(
            "127.0.0.1:8080",
            "/api/me?x=1",
            http_port=8080,
            https_port=8443,
            public_origin="https://gallery.codepharm.net",
        )
        == "https://gallery.codepharm.net/api/me?x=1"
    )


def test_https_redirect_location_maps_published_ports() -> None:
    assert (
        https_redirect_location(
            "127.0.0.1:8080",
            "/",
            http_port=8080,
            https_port=8443,
        )
        == "https://127.0.0.1:8443/"
    )
    assert (
        https_redirect_location(
            "gallery.codepharm.net",
            "/static/app.js",
            http_port=80,
            https_port=443,
        )
        == "https://gallery.codepharm.net/static/app.js"
    )


def test_http_redirects_when_https_flag_set(httpd: tuple[str, int]) -> None:
    """Cleartext handler 301s once the process marks HTTPS as available."""
    # Re-bind attributes on the live test server used by the fixture's make_server.
    # The fixture does not expose the server object; open a one-off redirect server.
    del httpd
    server = make_server("127.0.0.1", 0)
    server.force_https_redirect = True
    server.gallery_http_port = 8080
    server.gallery_https_port = 8443
    server.gallery_public_https_origin = None
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        connection = HTTPConnection(str(host), int(port), timeout=5)
        connection.request("GET", "/health", headers={"Host": "127.0.0.1:8080"})
        response = connection.getresponse()
        body = response.read()
        connection.close()
        assert response.status == 301
        assert response.getheader("Location") == "https://127.0.0.1:8443/health"
        assert body == b""
    finally:
        server.shutdown()
        server.server_close()


def test_map_permission_names_read_vs_review() -> None:
    assert map_permission_names(["Read"]) == "read"
    assert map_permission_names(["Read", "Annotate"]) == "review"
    assert map_permission_names(["Admin"]) is None


def test_oidc_maps_identity_api_volumes() -> None:
    calls: list[str] = []

    def http_json(url: str, **kwargs: object) -> object:
        del kwargs
        calls.append(url)
        if url.endswith("/connect/token"):
            return {"access_token": "tok"}
        if url.endswith("/connect/userinfo"):
            return {"sub": "u1", "name": "Ada"}
        if url.endswith("/Permissions/AccessibleVolumes"):
            return {
                "7": {"Id": 7, "Name": "RC2-Pitt", "AnnotationDatabaseName": "RC2", "permissions": ["Read", "Annotate"]},
                "8": {"Id": 8, "Name": "SKIPME", "AnnotationDatabaseName": None, "permissions": ["Read"]},
            }
        raise AssertionError(url)

    oidc = OidcIdentity(
        {
            "GALLERY_OIDC_AUTHORITY": "https://identity.codepharm.net:5001",
            "GALLERY_IDENTITY_API": "https://identity.codepharm.net:6001",
            "GALLERY_OIDC_CLIENT_ID": "gallery",
        },
        http_json=http_json,
    )
    principal = oidc.complete_login("code", "http://127.0.0.1:8090/callback", ["RC2"])
    assert principal.display_name == "Ada"
    assert principal.grants == {"RC2": "review"}
    assert any("identity.codepharm.net:5001" in url for url in calls)
    assert any("identity.codepharm.net:6001" in url for url in calls)


def test_oidc_login_url_points_at_authority() -> None:
    oidc = OidcIdentity(
        {
            "GALLERY_OIDC_AUTHORITY": "https://identity.codepharm.net:5001",
            "GALLERY_OIDC_CLIENT_ID": "gallery",
        }
    )
    url = oidc.login_url("http://127.0.0.1:8090/callback", "state-1")
    assert url.startswith("https://identity.codepharm.net:5001/connect/authorize?")
    assert "client_id=gallery" in url


def test_tag_pattern_matches_structure_and_type_tags() -> None:
    assert tag_tokens("<Tags><Tag>synapse</Tag><Tag>gap</Tag></Tags>") == ["synapse", "gap"]
    assert tag_tokens("synapse;gap") == ["synapse", "gap"]
    structures = [
        {"ID": 10, "TypeID": 1, "Tags": "synapse"},
        {"ID": 11, "TypeID": 2, "Tags": None},
        {"ID": 12, "TypeID": 1, "Tags": "other"},
    ]
    types = [{"ID": 2, "Tags": None, "StructureTags": "gap;junction"}]
    assert structure_ids_for_tag_pattern(structures, types, "synapse") == [10]
    assert structure_ids_for_tag_pattern(structures, types, "^gap$") == [11]
    assert structure_ids_for_tag_pattern(structures, types, "syn|gap") == [10, 11]


def test_thumbnail_rewritten_only_when_older_than_mask(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from PIL import Image

    writes = {"n": 0}
    real_write = thumbs._write_jpeg

    def counting(source: Path, dest: Path, size: int) -> None:
        writes["n"] += 1
        real_write(source, dest, size)

    monkeypatch.setattr(thumbs, "_write_jpeg", counting)
    crops = tmp_path / "AnnotationCrops"
    _seed_crops(crops)
    image_key = "RC2_17_D1_X0_Y0"
    written = ensure_thumbnail(crops, 256, image_key)
    assert written is not None
    assert writes["n"] == 1
    with Image.open(written) as image:
        assert image.size == (256, 256)
        assert image.format == "JPEG"
    remember_mask_index(crops, {image_key: written.stat().st_mtime_ns + 1_000_000_000})
    rewritten = ensure_thumbnail(crops, 256, image_key)
    assert rewritten is not None
    assert writes["n"] == 2
    os.utime(rewritten, ns=(9_000_000_000_000_000_000, 9_000_000_000_000_000_000))
    kept = ensure_thumbnail(crops, 256, image_key)
    assert kept == rewritten
    assert writes["n"] == 2


def test_backfill_writes_every_size_from_one_decode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from PIL import Image

    opens = {"n": 0}
    real_open = thumbs.Image.open

    def counting(path, *args, **kwargs):
        opens["n"] += 1
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(thumbs.Image, "open", counting)
    crops = tmp_path / "AnnotationCrops"
    _seed_crops(crops)
    image_key = "RC2_17_D1_X0_Y0"
    (crops / "overlays" / "thumb" / "256").mkdir(parents=True)
    existing = crops / "overlays" / "thumb" / "256" / f"{image_key}.jpg"
    Image.new("RGB", (256, 256)).save(existing, format="JPEG")
    os.utime(existing, ns=(9_000_000_000_000_000_000, 9_000_000_000_000_000_000))
    written = thumbs.backfill_crops(crops)
    assert written == 2
    assert opens["n"] == 1
    for size in (384, 512):
        path = crops / "overlays" / "thumb" / str(size) / f"{image_key}.jpg"
        assert path.is_file()
        with Image.open(path) as image:
            assert image.size == (size, size)


def test_registry_signature_tracks_folder_mtime_not_thumbnails(tmp_path: Path) -> None:
    registry = tmp_path / "current"
    images = registry / "RC2" / "AnnotationCrops" / "images"
    images.mkdir(parents=True)
    before = thumbs.registry_signature(registry)
    os.utime(images, ns=(2_000_000_000, 2_000_000_000))
    after = thumbs.registry_signature(registry)
    assert after != before
    thumb = registry / "RC2" / "AnnotationCrops" / "overlays" / "thumb" / "256"
    thumb.mkdir(parents=True)
    (thumb / "kept.jpg").write_bytes(b"jpeg")
    os.utime(thumb, ns=(9_000_000_000_000_000_000, 9_000_000_000_000_000_000))
    assert thumbs.registry_signature(registry) == after


def test_registry_signature_tracks_symlink_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry = tmp_path / "current"
    (registry / "RC2" / "AnnotationCrops" / "images").mkdir(parents=True)
    targets = iter(("v1b", "v1c"))
    monkeypatch.setattr(thumbs, "_link_target", lambda _path: next(targets))
    first = thumbs.registry_signature(registry)
    second = thumbs.registry_signature(registry)
    assert first[0] == "v1b"
    assert second[0] == "v1c"
    assert first != second


def test_backfill_keeps_one_running_and_one_queued(monkeypatch: pytest.MonkeyPatch) -> None:
    gate = thumbs._BackfillGate()
    entered = threading.Event()
    holds = [threading.Event(), threading.Event()]
    calls: list[str] = []

    def fake(registry: Path) -> None:
        calls.append(str(registry))
        entered.set()
        holds[len(calls) - 1].wait(2)

    monkeypatch.setattr(thumbs, "_backfill_registry", fake)
    gate.request(Path("a"))
    assert entered.wait(2)
    entered.clear()
    gate.request(Path("b"))
    gate.request(Path("c"))
    holds[0].set()
    assert entered.wait(2)
    holds[1].set()
    assert gate._thread is not None
    gate._thread.join(2)
    assert calls == [str(Path("a")), str(Path("c"))]


def test_thumbnail_request_encodes_jpeg(httpd: tuple[str, int], registry: Path) -> None:
    connection = HTTPConnection(httpd[0], httpd[1], timeout=10)
    connection.request(
        "GET",
        "/api/volumes/RC2/file?path=overlays/thumb/384/RC2_17_D1_X0_Y0.jpg",
        headers={"Authorization": "Bearer read"},
    )
    response = connection.getresponse()
    body = response.read()
    connection.close()
    assert response.status == 200
    assert response.getheader("Content-Type") == "image/jpeg"
    assert body[:2] == b"\xff\xd8"
    saved = registry / "RC2" / "AnnotationCrops" / "overlays" / "thumb" / "384" / "RC2_17_D1_X0_Y0.jpg"
    assert saved.is_file()


def test_review_locations_touches_only_the_listed_locations(tmp_path: Path) -> None:
    crops = tmp_path / "AnnotationCrops"
    (crops / "masks").mkdir(parents=True)
    connection = sqlite3.connect(crops / "annotation_crops.sqlite")
    connection.execute(
        "CREATE TABLE locations ("
        "location_id INTEGER NOT NULL, z INTEGER NOT NULL, structure_id INTEGER, "
        "structure_label TEXT, type_id INTEGER, type_name TEXT, radius REAL, "
        "image_key TEXT NOT NULL, image_relpath TEXT, mask_relpath TEXT, "
        "ignored INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (location_id, image_key))"
    )
    connection.executemany(
        "INSERT INTO locations (location_id, z, structure_id, structure_label, "
        "type_id, type_name, radius, image_key, image_relpath, mask_relpath, ignored) "
        "VALUES (?, 17, 7, 'soma', 1, 'Cell', 12.5, ?, ?, ?, 0)",
        [(loc, f"k{loc}", f"images/k{loc}.png", f"masks/k{loc}_{loc}.png") for loc in (1, 2, 3, 4)],
    )
    connection.commit()
    connection.close()
    for loc in (1, 2, 3, 4):
        (crops / "masks" / f"k{loc}_{loc}.png").write_bytes(b"png")
    approve_window(crops, 4)

    assert review_locations(crops, [1, 2], "ignore") == 2
    by_id = {row["location_id"]: row for row in list_catalog_rows(crops)}
    assert [by_id[loc]["ignored"] for loc in (1, 2, 3, 4)] == [1, 1, 0, 0]
    assert by_id[4]["approved"] == 1
    assert (crops / "ignored" / "k1_1.png").is_file() and not (crops / "masks" / "k1_1.png").exists()
    assert (crops / "masks" / "k3_3.png").is_file()

    # Approving a rejected location restores it; location 4 keeps its earlier approval; 3 is untouched.
    assert review_locations(crops, [1, 3], "approve") == 2
    by_id = {row["location_id"]: row for row in list_catalog_rows(crops)}
    assert [by_id[loc]["ignored"] for loc in (1, 2, 3, 4)] == [0, 1, 0, 0]
    assert [by_id[loc]["approved"] for loc in (1, 2, 3, 4)] == [1, 0, 1, 1]
    assert (crops / "masks" / "k1_1.png").is_file()

    # Rejecting an approved location clears its approval.
    review_locations(crops, [4], "ignore")
    by_id = {row["location_id"]: row for row in list_catalog_rows(crops)}
    assert by_id[4]["ignored"] == 1 and by_id[4]["approved"] == 0
    assert review_locations(crops, [], "ignore") == 0
    with pytest.raises(ValueError):
        review_locations(crops, [1], "delete")


def test_cardmask_endpoint_serves_png_with_etag(httpd: tuple[str, int], registry: Path) -> None:
    import io as _io

    import numpy as np
    from PIL import Image as _Image

    crops = registry / "RC2" / "AnnotationCrops"
    key = "RC2_17_D1_X0-64_Y0-64"
    (crops / "images").mkdir(parents=True, exist_ok=True)
    (crops / "masks").mkdir(parents=True, exist_ok=True)
    (crops / "images" / f"{key}.json").write_text(
        json.dumps({"image": {"width": 64, "height": 64}, "annotations": [{"id": 42}]}), encoding="utf-8"
    )
    plane = np.zeros((64, 64), dtype=np.uint8)
    plane[8:40, 8:40] = 255
    _Image.fromarray(plane).save(crops / "masks" / f"{key}_42.png")
    path = f"/api/volumes/RC2/cardmask?size=256&key={key}&loc=42"

    def get(headers: dict[str, str] | None = None):
        connection = HTTPConnection(httpd[0], httpd[1], timeout=5)
        connection.request("GET", path, headers=headers or {})
        response = connection.getresponse()
        body = response.read()
        result = (response.status, response.getheader("Content-Type"), response.getheader("ETag"), body)
        connection.close()
        return result

    status, content_type, etag, body = get()
    assert status == 200 and content_type == "image/png" and etag
    with _Image.open(_io.BytesIO(body)) as layer:
        assert layer.size == (256, 256)
        assert np.asarray(layer.convert("RGB"))[100, 100, 0] == 255
    status, _type, _etag, body = get({"If-None-Match": etag})
    assert status == 304 and body == b""
    connection = HTTPConnection(httpd[0], httpd[1], timeout=5)
    connection.request("GET", f"/api/volumes/RC2/cardmask?size=256&key={key}&loc=7")
    missing = connection.getresponse()
    missing.read()
    connection.close()
    assert missing.status == 404


def test_update_all_endpoints(httpd: tuple[str, int], registry: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NORNIR_BUILDMANAGER_DIR", str(registry / "no-such-mount"))
    status, payload = _request(httpd, "GET", "/api/volumes/RC2/update-all", bearer="review")
    assert status == 200 and payload["phase"] == "idle" and payload["watermark"] == ""
    status, payload = _request(httpd, "POST", "/api/volumes/RC2/update-all", body=b"{}", bearer="review")
    assert status == 503 and "nornir" in payload["error"]
    status, _payload = _request(httpd, "POST", "/api/volumes/RC2/update-all", body=b"{}")
    assert status == 403
    status, payload = _request(httpd, "POST", "/api/volumes/RC2/update-all/cancel", body=b"{}", bearer="review")
    assert status == 200 and payload["running"] is False


def test_sparse_shape_bands_start_at_the_top_of_the_grid() -> None:
    # A band that uses only low rows of a sparse shape must not leave empty rows above it: the grid draws only the
    # rows near the viewport, so a card that starts rows down the page shows as a blank grid.
    js = (GALLERY_ROOT / "static" / "app.js").read_text(encoding="utf-8")
    offsets = js.split("function shapeOffsets", 1)[1].split("function cellHtml", 1)[0]
    assert "range.lo" in offsets
    assert "range.start" in offsets
    assert "span" not in offsets


# ---------------------------------------------------------------- bots, login gate, annotation database names


def _raw_get(httpd: tuple[str, int], path: str, *, agent: str | None = None) -> tuple[int, dict[str, str], bytes]:
    connection = HTTPConnection(httpd[0], httpd[1], timeout=5)
    connection.request("GET", path, headers={"User-Agent": agent} if agent else {})
    response = connection.getresponse()
    body = response.read()
    headers: dict[str, str] = {}
    for key, value in response.getheaders():
        headers[key.lower()] = f"{headers[key.lower()]}, {value}" if key.lower() in headers else value
    connection.close()
    return response.status, headers, body


def test_robots_txt_disallows_everything_and_responses_say_noindex(httpd: tuple[str, int]) -> None:
    status, headers, body = _raw_get(httpd, "/robots.txt")
    assert status == 200
    assert body.decode() == "User-agent: *\nDisallow: /\n"
    assert "noindex" in headers["x-robots-tag"]


def test_self_identified_ai_crawlers_get_403(httpd: tuple[str, int]) -> None:
    for agent in ("Mozilla/5.0 (compatible; GPTBot/1.1)", "ClaudeBot/1.0", "CCBot/2.0"):
        status, _headers, _body = _raw_get(httpd, "/", agent=agent)
        assert status == 403, agent
    status, _headers, _body = _raw_get(httpd, "/", agent="Mozilla/5.0 Firefox/130")
    assert status == 200


def test_oidc_mode_sends_anonymous_visitors_to_login(
    httpd: tuple[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GALLERY_IDENTITY_MODE", "oidc")
    status, headers, _body = _raw_get(httpd, "/")
    assert status == 302 and headers["location"] == "/login"
    status, payload = _request(httpd, "GET", "/api/volumes/RC2/catalog")
    assert status == 401 and payload["error"] == "login required"
    status, _headers, _body = _raw_get(httpd, "/health")
    assert status == 200
    status, _headers, _body = _raw_get(httpd, "/static/app.css")
    assert status == 200


def test_stub_mode_and_opt_out_keep_the_gallery_open(
    httpd: tuple[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    status, _headers, _body = _raw_get(httpd, "/")
    assert status == 200
    monkeypatch.setenv("GALLERY_IDENTITY_MODE", "oidc")
    monkeypatch.setenv("GALLERY_REQUIRE_LOGIN", "0")
    status, _headers, _body = _raw_get(httpd, "/")
    assert status == 200


def test_logout_clears_the_cookie_and_ends_the_identity_session(
    httpd: tuple[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GALLERY_IDENTITY_MODE", "oidc")
    monkeypatch.setenv("GALLERY_OIDC_AUTHORITY", "https://identity.example:5001/")
    status, headers, _body = _raw_get(httpd, "/logout")
    assert status == 302
    assert headers["location"] == "https://identity.example:5001/connect/endsession"
    assert "gallery_session=;" in headers["set-cookie"] and "Max-Age=0" in headers["set-cookie"]
    monkeypatch.setenv("GALLERY_IDENTITY_MODE", "stub")
    status, headers, _body = _raw_get(httpd, "/logout")
    assert headers["location"] == "/"


def test_cookies_are_secure_once_the_public_origin_is_https(monkeypatch: pytest.MonkeyPatch) -> None:
    from server import cookie_attributes

    monkeypatch.delenv("GALLERY_PUBLIC_HTTPS_ORIGIN", raising=False)
    assert "Secure" not in cookie_attributes()
    monkeypatch.setenv("GALLERY_PUBLIC_HTTPS_ORIGIN", "https://sem.gallery.codepharm.net")
    attrs = cookie_attributes(max_age=60)
    assert "Secure" in attrs and "HttpOnly" in attrs and "Max-Age=60" in attrs


def test_annotation_db_name_prefers_the_field_then_the_odata_url_then_the_folder(tmp_path: Path) -> None:
    from catalog import annotation_db_name, db_name_from_odata

    assert db_name_from_odata("https://websvc.codepharm.net/NeitzNM/OData") == "NeitzNM"
    assert db_name_from_odata("https://websvc.codepharm.net/NeitzNM/OData/") == "NeitzNM"
    assert db_name_from_odata("https://websvc.codepharm.net/") == ""
    crops = tmp_path / "NasalMonkey"
    crops.mkdir()
    assert annotation_db_name(crops, "NasalMonkey") == "NasalMonkey"
    (crops / "source.json").write_text(json.dumps({"odata": "https://h/NeitzNM/OData"}), encoding="utf-8")
    assert annotation_db_name(crops, "NasalMonkey") == "NeitzNM"
    (crops / "source.json").write_text(
        json.dumps({"odata": "https://h/NeitzNM/OData", "annotationDb": "Other"}), encoding="utf-8"
    )
    assert annotation_db_name(crops, "NasalMonkey") == "Other"


def test_grants_follow_the_annotation_database_name_not_the_folder() -> None:
    from identity import grants_for_registry

    grants = {"NeitzNM": "review", "RC1": "read"}
    folders = ["NasalMonkey", "RC1", "InferiorMonkey"]
    aliases = {"NasalMonkey": "NeitzNM", "InferiorMonkey": "NeitzInferiorMonkey"}
    assert grants_for_registry(grants, folders, aliases) == {"NasalMonkey": "review", "RC1": "read"}


def test_viking_link_appends_the_location_to_the_launch_code_url() -> None:
    from server import viking_link

    calls: list[tuple] = []

    def http_json(url, **kwargs):
        calls.append((url, kwargs))
        return {"code": "abc", "expires_in": 300, "viking_url": "viking://open?code=abc&volumeName=NM"}

    url = viking_link("https://identity.example:6001/", "tok", "NM", "12345", http_json)
    assert url == "viking://open?code=abc&volumeName=NM&location=12345"
    assert calls[0][0] == "https://identity.example:6001/api/viking/launch-code"
    assert calls[0][1]["headers"]["Authorization"] == "Bearer tok"
    assert json.loads(calls[0][1]["body"]) == {"volume_name": "NM"}


def test_viking_link_falls_back_to_the_plain_link_when_identity_refuses_the_volume() -> None:
    from identity import IdentityHttpError
    from server import viking_link

    def refuse(url, **kwargs):
        raise IdentityHttpError(403, "forbidden")

    assert viking_link("https://i", "tok", "NM", "7", refuse) == "viking://open?volumeName=NM&location=7"

    def expired(url, **kwargs):
        raise IdentityHttpError(401, "unauthorized")

    with pytest.raises(IdentityHttpError):
        viking_link("https://i", "tok", "NM", "7", expired)


def test_tokens_are_kept_server_side_and_expire() -> None:
    from server import forget_token, recall_token, remember_token

    sid = remember_token("secret-token", 3600)
    assert recall_token(sid) == "secret-token" and "secret-token" not in sid
    forget_token(sid)
    assert recall_token(sid) == ""
    assert recall_token(None) == "" and recall_token("nope") == ""


def test_viking_link_endpoint_needs_a_token_and_a_granted_volume(
    httpd: tuple[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    status, payload = _request(httpd, "POST", "/api/volumes/RC2/viking-link", body=b'{"location": 5}', bearer="review")
    assert status == 401 and payload["login"] == "/login"
    status, _payload = _request(httpd, "POST", "/api/volumes/NOPE/viking-link", body=b'{"location": 5}', bearer="review")
    assert status == 404


def test_catalog_is_cached_until_the_database_changes_and_can_be_gzipped(
    httpd: tuple[str, int], registry: Path
) -> None:
    import gzip as gzip_module

    connection = HTTPConnection(httpd[0], httpd[1], timeout=5)
    connection.request("GET", "/api/volumes/RC2/catalog", headers={"Authorization": "Bearer read"})
    plain = connection.getresponse()
    plain_body = plain.read()
    assert plain.status == 200 and plain.getheader("Content-Encoding") is None
    connection.request("GET", "/api/volumes/RC2/catalog", headers={"Authorization": "Bearer read", "Accept-Encoding": "gzip"})
    packed = connection.getresponse()
    packed_body = packed.read()
    connection.close()
    assert packed.getheader("Content-Encoding") == "gzip"
    assert json.loads(gzip_module.decompress(packed_body)) == json.loads(plain_body)
    assert json.loads(plain_body)["rows"], "the seeded location is listed"


def test_grants_match_database_names_regardless_of_case() -> None:
    from identity import grants_for_registry

    assert grants_for_registry({"cped": "read"}, ["CPED"], {"CPED": "CPED"}) == {"CPED": "read"}
    assert grants_for_registry({"Cped": "read", "cped": "review"}, ["CPED"]) == {"CPED": "review"}


_READ_ROUTES = [
    ("GET", "/api/volumes/{v}/catalog", None),
    ("GET", "/api/volumes/{v}/ignore-list", None),
    ("GET", "/api/volumes/{v}/settings", None),
    ("GET", "/api/volumes/{v}/repair-pack", None),
]
_REVIEW_ROUTES = [
    ("POST", "/api/volumes/{v}/ignore", {"location_id": 42}),
    ("POST", "/api/volumes/{v}/approve", {"location_id": 42}),
    ("POST", "/api/volumes/{v}/review-batch", {"action": "approve", "location_ids": [42]}),
    ("POST", "/api/volumes/{v}/ignore-list", {"locationIds": [42]}),
    ("POST", "/api/volumes/{v}/settings", {"odata": "", "exporter": "tiled"}),
    ("POST", "/api/volumes/{v}/refresh-location", {"location_id": 42}),
    ("POST", "/api/volumes/{v}/import-location", {"location_id": 42}),
]


def _call(httpd: tuple[str, int], route: tuple, volume: str, bearer: str) -> tuple[int, dict]:
    method, path, body = route
    data = None if body is None else json.dumps(body).encode("utf-8")
    return _request(httpd, method, path.format(v=volume), body=data, bearer=bearer)


@pytest.mark.parametrize("route", _READ_ROUTES + _REVIEW_ROUTES)
def test_unknown_volume_is_404(httpd: tuple[str, int], route: tuple) -> None:
    assert _call(httpd, route, "NOPE", "review") == (404, {"error": "not found"})


@pytest.mark.parametrize("route", _REVIEW_ROUTES)
def test_read_only_grant_on_write_routes_is_403(httpd: tuple[str, int], route: tuple) -> None:
    assert _call(httpd, route, "RC2", "read") == (403, {"error": "review required"})


@pytest.mark.parametrize(
    "route",
    _READ_ROUTES + [r for r in _REVIEW_ROUTES if not r[1].endswith("/settings")],
)
def test_missing_crops_dir_is_404(
    httpd: tuple[str, int], route: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    def gone(registry: Path, name: str) -> Path:
        raise FileNotFoundError(name)

    monkeypatch.setattr("server.crops_dir", gone)
    assert _call(httpd, route, "RC2", "review") == (404, {"error": "not found"})


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def test_odata_read_retries_a_failed_connection(monkeypatch):
    import server
    from urllib.error import URLError

    calls = []

    def flaky(request, timeout):
        calls.append(request.full_url)
        if len(calls) == 1:
            raise URLError(OSError(-5, "No address associated with hostname"))
        return _FakeResponse(b'{"ID": 7}')

    monkeypatch.setattr(server, "urlopen", flaky)
    monkeypatch.setattr(server, "ODATA_RETRY_DELAY_SECONDS", 0)
    assert server.fetch_odata_location("https://example.test/RPC2/OData", 7) == {"ID": 7}
    assert len(calls) == 2


def test_odata_read_gives_up_after_the_last_attempt(monkeypatch):
    import server
    from urllib.error import URLError

    calls = []

    def down(request, timeout):
        calls.append(1)
        raise URLError(OSError(-5, "No address associated with hostname"))

    monkeypatch.setattr(server, "urlopen", down)
    monkeypatch.setattr(server, "ODATA_RETRY_DELAY_SECONDS", 0)
    with pytest.raises(URLError):
        server.fetch_odata_location("https://example.test/RPC2/OData", 7)
    assert len(calls) == server.ODATA_ATTEMPTS


def test_odata_read_does_not_retry_an_http_error(monkeypatch):
    import server
    from urllib.error import HTTPError

    calls = []

    def missing(request, timeout):
        calls.append(1)
        raise HTTPError(request.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr(server, "urlopen", missing)
    monkeypatch.setattr(server, "ODATA_RETRY_DELAY_SECONDS", 0)
    with pytest.raises(HTTPError):
        server.fetch_odata_location("https://example.test/RPC2/OData", 7)
    assert len(calls) == 1
