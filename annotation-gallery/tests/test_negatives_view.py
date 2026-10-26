"""Negative-context bitmask: the helper, its cache, and the HTTP route."""

from __future__ import annotations

import io
import json
import os
import sys
import threading
from http.client import HTTPConnection
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

GALLERY_ROOT = Path(__file__).resolve().parents[1]
if str(GALLERY_ROOT) not in sys.path:
    sys.path.insert(0, str(GALLERY_ROOT))

from negatives_view import COUNT_FIELD, cache_path, negative_bitmask  # noqa: E402
from server import make_server  # noqa: E402

KEY = "RC2_17_D1_X0-32_Y0-32"
SIZE = 32


def _png(path: Path, rows: slice, cols: slice) -> None:
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    mask[rows, cols] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(path)


def _tile(crops: Path, ids: list[int]) -> None:
    images = crops / "images"
    images.mkdir(parents=True, exist_ok=True)
    sidecar = {
        "image": {"file_name": f"{KEY}.png", "width": SIZE, "height": SIZE},
        "annotations": [{"id": i} for i in ids],
    }
    (images / f"{KEY}.json").write_text(json.dumps(sidecar), encoding="utf-8")
    Image.fromarray(np.zeros((SIZE, SIZE), dtype=np.uint8)).save(images / f"{KEY}.png")


def _decode(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as image:
        return np.array(image)


@pytest.fixture
def crops(tmp_path: Path) -> Path:
    path = tmp_path / "AnnotationCrops"
    _tile(path, [1, 2, 3])
    _png(path / "masks" / f"{KEY}_1.png", slice(8, 24), slice(8, 24))
    _png(path / "masks" / f"{KEY}_2.png", slice(2, 6), slice(2, 6))
    _png(path / "masks" / f"{KEY}_3.png", slice(26, 30), slice(26, 30))
    return path


def test_bitmask_marks_every_other_annotation(crops: Path) -> None:
    data, count = negative_bitmask(crops, KEY, 1)
    pixels = _decode(data)
    assert count == 2
    assert pixels.shape == (SIZE, SIZE)
    assert set(np.unique(pixels)) == {0, 255}
    assert pixels[3, 3] == 255 and pixels[27, 27] == 255
    assert pixels[16, 16] == 0  # the target itself is never a negative


def test_unmasked_circles_count_and_a_polygons_own_circle_does_not(crops: Path) -> None:
    payload = {
        "version": 1,
        "diameter_scale": 0.8,
        "grid_cells": {
            KEY: [
                {"location_id": 99, "center_x": 4.0, "center_y": 28.0, "radius": 3.0},
                {"location_id": 2, "center_x": 4.0, "center_y": 4.0, "radius": 9.0,
                 "has_polygon_mask": True, "source": "inscribed"},
            ]
        },
    }
    (crops / "circle_annotations.json").write_text(json.dumps(payload), encoding="utf-8")
    data, count = negative_bitmask(crops, KEY, 1)
    pixels = _decode(data)
    assert count == 3  # neighbors 2 and 3, plus circle 99
    assert pixels[28, 4] == 255
    assert pixels[10, 4] == 0  # location 2's big inscribed circle was not stamped over its small mask


def test_tile_without_neighbors_is_empty(tmp_path: Path) -> None:
    crops = tmp_path / "AnnotationCrops"
    _tile(crops, [1])
    _png(crops / "masks" / f"{KEY}_1.png", slice(8, 24), slice(8, 24))
    data, count = negative_bitmask(crops, KEY, 1)
    assert count == 0
    assert not _decode(data).any()


def test_cache_is_written_and_reused_until_a_source_changes(crops: Path) -> None:
    negative_bitmask(crops, KEY, 1)
    cached = cache_path(crops, KEY, 1)
    assert cached.is_file()
    with Image.open(cached) as image:
        assert image.info[COUNT_FIELD] == "2"

    cached.write_bytes(cached.read_bytes())  # make the cache newer than every source
    marker = cached.stat().st_mtime_ns
    negative_bitmask(crops, KEY, 1)
    assert cached.stat().st_mtime_ns == marker  # served from the cache

    newer = marker + 10_000_000_000
    os.utime(crops / "masks" / f"{KEY}_2.png", ns=(newer, newer))
    _png(crops / "masks" / f"{KEY}_2.png", slice(0, 2), slice(0, 2))
    os.utime(crops / "masks" / f"{KEY}_2.png", ns=(newer, newer))
    data, _ = negative_bitmask(crops, KEY, 1)
    assert _decode(data)[0, 0] == 255 and _decode(data)[4, 4] == 0  # recomputed from the changed mask


def test_unwritable_cache_is_ignored(crops: Path) -> None:
    (crops / "cache").write_bytes(b"a file where the folder should be")
    data, count = negative_bitmask(crops, KEY, 1)
    assert count == 2 and data.startswith(b"\x89PNG")


@pytest.mark.parametrize("key", ["../x", "a/b", "", "a b"])
def test_unsafe_keys_are_rejected(crops: Path, key: str) -> None:
    with pytest.raises(ValueError):
        negative_bitmask(crops, key, 1)


def test_missing_sidecar(crops: Path) -> None:
    with pytest.raises(FileNotFoundError):
        negative_bitmask(crops, "RC2_other_D1_X0-1_Y0-1", 1)


def test_neighbor_with_a_missing_file_is_skipped(crops: Path) -> None:
    (crops / "masks" / f"{KEY}_3.png").unlink()
    _, count = negative_bitmask(crops, KEY, 1)
    assert count == 1


@pytest.fixture
def served(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    registry = tmp_path / "gallery-volumes"
    crops = registry / "RC2" / "AnnotationCrops"
    _tile(crops, [1, 2])
    _png(crops / "masks" / f"{KEY}_1.png", slice(8, 24), slice(8, 24))
    _png(crops / "masks" / f"{KEY}_2.png", slice(2, 6), slice(2, 6))
    monkeypatch.setenv("GALLERY_VOLUME_DIR", str(registry))
    monkeypatch.setenv("GALLERY_IDENTITY_MODE", "stub")
    monkeypatch.setenv("GALLERY_STUB_ROLE", "read")
    monkeypatch.setenv("GALLERY_SESSION_SECRET", "test-secret")
    server = make_server("127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield str(host), int(port)
    finally:
        server.shutdown()
        server.server_close()


def _get(served: tuple[str, int], path: str):
    connection = HTTPConnection(served[0], served[1], timeout=5)
    connection.request("GET", path)
    response = connection.getresponse()
    body = response.read()
    headers = dict(response.getheaders())
    connection.close()
    return response.status, headers, body


def test_route_returns_the_png_and_count(served: tuple[str, int]) -> None:
    status, headers, body = _get(served, f"/api/volumes/RC2/negatives?key={KEY}&loc=1")
    assert status == 200
    assert headers["Content-Type"] == "image/png"
    assert headers["X-Negative-Count"] == "1"
    assert headers["Cache-Control"] == "no-cache"
    assert _decode(body)[3, 3] == 255


def test_route_rejects_bad_requests(served: tuple[str, int]) -> None:
    for query in (
        f"key={KEY}&loc=abc",
        f"key=..%2Fetc&loc=1",
        "key=RC2_missing_D1_X0-1_Y0-1&loc=1",
        "loc=1",
    ):
        status, _, _ = _get(served, f"/api/volumes/RC2/negatives?{query}")
        assert status == 404, query
    status, _, _ = _get(served, f"/api/volumes/NOPE/negatives?key={KEY}&loc=1")
    assert status == 404


def test_viewer_wires_the_negative_overlay() -> None:
    js = (GALLERY_ROOT / "static" / "app.js").read_text(encoding="utf-8")
    html = (GALLERY_ROOT / "static" / "index.html").read_text(encoding="utf-8")
    assert 'NEG_KEY = "galleryNegativeOverlay"' in js
    assert "/negatives?key=" in js and "X-Negative-Count" in js
    # Loaded after the location's own overlay, dropped if another location opened meanwhile, never for split parts.
    assert "void loadViewerNegatives(token, image, mask)" in js
    load = js.split("async function loadViewerNegatives", 1)[1].split("function toggleNegatives", 1)[0]
    assert "token !== viewer.token" in load
    assert "image === viewer.image" in js.split("function drawFull", 1)[1].split("function paintViewer", 1)[0]
    assert "viewer.negatives = null" in js.split("function blankViewer", 1)[1].split("}", 1)[0]
    assert 'event.key === "n"' in js and "toggleNegatives()" in js
    assert "<small>Negatives</small>" in html


def test_the_converters_existing_annotation_is_drawn_as_a_region(crops: Path) -> None:
    _png(crops / "original" / f"{KEY}_1.png", slice(10, 14), slice(10, 14))
    data, count = negative_bitmask(crops, KEY, 1)
    pixels = _decode(data)
    assert count == 3
    assert pixels[11, 11] == 255 and pixels[3, 3] == 255
    assert pixels[20, 20] == 0


def test_an_empty_or_missing_original_adds_nothing(crops: Path) -> None:
    assert negative_bitmask(crops, KEY, 1)[1] == 2
    _png(crops / "original" / f"{KEY}_1.png", slice(0, 0), slice(0, 0))
    assert negative_bitmask(crops, KEY, 1)[1] == 2
