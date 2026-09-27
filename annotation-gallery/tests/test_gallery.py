"""Slim gallery HTTP: stub identity, ignore 403, path traversal."""

from __future__ import annotations

import io
import json
import os
import sqlite3
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
)
from identity import OidcIdentity, map_permission_names  # noqa: E402
from server import (  # noqa: E402
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
        "INSERT INTO location_scores (location_id, epoch, score) VALUES (?, ?, ?)",
        [(42, 1, 1.5), (42, 3, 0.25)],
    )
    connection.commit()
    connection.close()
    status, catalog = _request(httpd, "GET", "/api/volumes/RC2/catalog", bearer="read")
    assert status == 200
    by_id = {row["location_id"]: row for row in catalog["rows"]}
    assert by_id[42]["loss"] == 0.25
    assert by_id[99]["loss"] is None


def test_training_sets_defaults_to_current_outside_a_version_tree(httpd: tuple[str, int]) -> None:
    status, payload = _request(httpd, "GET", "/api/training-sets", bearer="read")
    assert status == 200
    assert payload["default"] == "current"
    assert payload["sets"] == ["current"]


def test_read_lists_volume_and_catalog(httpd: tuple[str, int]) -> None:
    status, me = _request(httpd, "GET", "/api/me", bearer="read")
    assert status == 200
    assert me["volumes"] == [{"name": "RC2", "permission": "read"}]
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


def test_location_search_scrolls_and_sort_stacks() -> None:
    html = (GALLERY_ROOT / "static" / "index.html").read_text(encoding="utf-8")
    js = (GALLERY_ROOT / "static" / "app.js").read_text(encoding="utf-8")
    assert 'id="locate"' in html
    assert "Other cards stay in the list." in html
    assert "scrollToLocation" in js
    assert '["structure_id", "Structure ID"]' in js
    assert '["location_id", "Location ID"]' in js
    assert "SORTS_PER_COLUMN = 5" in js
    assert 'id="sort-add"' in js
    assert "maxSortRows" not in js


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
    assert "render(true, true)" in mutate
    assert "function reconcileGrid" in js
    assert 'postReview(act, locationId, "")' in mutate
    assert ".card[hidden]" in css
    assert "visibleKeys" in js
    pack = js.split("function packGrid", 1)[1].split("function packKey", 1)[0]
    assert "windows[0]" not in pack
    assert "function windowOrigin" in js
    assert "Math.floor(dc / columns)" in js
    opener = js.split("async function openViewer", 1)[1].split("function viewerRow", 1)[0]
    assert "images/${imageKey}.png" in opener
    assert "loadParts" not in opener
    assert 'closest(".card")' in js
    assert "width: 32px" in css
    assert 'content: "Generating thumbnail"' in css
    assert 'content: "Image missing"' in css


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
    assert "pip install --no-cache-dir pillow" in text


def test_map_permission_names_read_vs_review() -> None:
    assert map_permission_names(["Read"]) == "read"
    assert map_permission_names(["Read", "Review"]) == "review"
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
        if url.endswith("/api/volumes"):
            return [
                {"name": "RC2", "permissions": ["Read", "Review"]},
                {"name": "SKIPME", "permissions": ["Read"]},
            ]
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
