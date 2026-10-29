"""Card-size mask layers: content, freshness, and invalidation."""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

GALLERY_ROOT = Path(__file__).resolve().parents[1]
if str(GALLERY_ROOT) not in sys.path:
    sys.path.insert(0, str(GALLERY_ROOT))

from cardmask import card_mask_path, drop_card_masks, ensure_card_mask  # noqa: E402

KEY = "RC1_5_D1_X0-64_Y0-64"


def _save_mask(path: Path, box: tuple[int, int, int, int]) -> None:
    plane = np.zeros((64, 64), dtype=np.uint8)
    x0, y0, x1, y1 = box
    plane[y0:y1, x0:x1] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(plane).save(path)


def _tile(tmp_path: Path) -> Path:
    crops = tmp_path / "AnnotationCrops"
    (crops / "images").mkdir(parents=True)
    sidecar = {"image": {"width": 64, "height": 64}, "annotations": [{"id": 1}, {"id": 2}]}
    (crops / "images" / f"{KEY}.json").write_text(json.dumps(sidecar), encoding="utf-8")
    _save_mask(crops / "masks" / f"{KEY}_1.png", (0, 0, 32, 32))
    _save_mask(crops / "masks" / f"{KEY}_2.png", (16, 16, 48, 48))
    return crops


def _decode(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as image:
        return np.asarray(image.convert("RGB"))


def test_layer_has_target_negatives_and_overlap(tmp_path: Path) -> None:
    crops = _tile(tmp_path)
    layer = _decode(ensure_card_mask(crops, 256, KEY, 1))
    assert layer.shape == (256, 256, 3)
    assert layer[40, 40, 0] == 255 and layer[40, 40, 1] == 0 and layer[40, 40, 2] == 0   # own only
    assert layer[240, 240, :].tolist() == [0, 0, 0]                                      # neither
    assert layer[100, 100].tolist() == [255, 255, 255]                                   # overlap
    assert layer[150, 150].tolist() == [0, 255, 0]                                       # neighbor only
    assert card_mask_path(crops, 256, KEY, 1).is_file()


def test_cached_layer_is_reused_until_a_source_is_newer(tmp_path: Path) -> None:
    crops = _tile(tmp_path)
    first = ensure_card_mask(crops, 256, KEY, 1)
    cached = card_mask_path(crops, 256, KEY, 1)
    stamp = cached.stat().st_mtime_ns
    assert ensure_card_mask(crops, 256, KEY, 1) == first
    assert cached.stat().st_mtime_ns == stamp

    neighbor = crops / "masks" / f"{KEY}_2.png"
    _save_mask(neighbor, (0, 40, 64, 64))
    newer = stamp + 5_000_000_000
    os.utime(neighbor, ns=(newer, newer))
    rebuilt = _decode(ensure_card_mask(crops, 256, KEY, 1))
    assert rebuilt[240, 240, 1] == 255            # the neighbor's new area is now negative
    assert cached.stat().st_mtime_ns >= stamp


def test_drop_card_masks_removes_every_layer_of_the_tile(tmp_path: Path) -> None:
    crops = _tile(tmp_path)
    for size in (256, 384):
        ensure_card_mask(crops, size, KEY, 1)
        ensure_card_mask(crops, size, KEY, 2)
    assert drop_card_masks(crops, [KEY]) == 4
    assert not card_mask_path(crops, 256, KEY, 2).exists()


def test_rejected_target_still_has_a_layer_and_bad_input_is_refused(tmp_path: Path) -> None:
    crops = _tile(tmp_path)
    (crops / "ignored").mkdir()
    (crops / "masks" / f"{KEY}_1.png").replace(crops / "ignored" / f"{KEY}_1.png")
    assert _decode(ensure_card_mask(crops, 256, KEY, 1))[40, 40, 0] == 255
    for call in (lambda: ensure_card_mask(crops, 100, KEY, 1), lambda: ensure_card_mask(crops, 256, "../x", 1)):
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")
    try:
        ensure_card_mask(crops, 256, KEY, 99)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError")


OTHER = "RC1_5_D1_X64-128_Y0-64"


def _circles(crops: Path, cells: dict[str, list[dict]], stamp_ns: int) -> None:
    path = crops / "circle_annotations.json"
    path.write_text(json.dumps({"version": 1, "grid_cells": cells}), encoding="utf-8")
    os.utime(path, ns=(stamp_ns, stamp_ns))


def _circle(location_id: int, x: float) -> dict:
    return {"location_id": location_id, "center_x": x, "center_y": 56.0, "radius": 4.0}


def test_a_rewritten_circle_file_only_rebuilds_the_tiles_whose_circles_changed(tmp_path: Path) -> None:
    """Update All rewrites the whole circle file; a layer is rebuilt only when its own tile's circles differ."""
    crops = _tile(tmp_path)
    late = 2_000_000_000_000_000_000
    _circles(crops, {KEY: [_circle(9, 8.0)], OTHER: [_circle(8, 8.0)]}, late)
    first = ensure_card_mask(crops, 256, KEY, 1)
    cached = card_mask_path(crops, 256, KEY, 1)
    marker = cached.stat().st_mtime_ns

    _circles(crops, {KEY: [_circle(9, 8.0)], OTHER: [_circle(8, 40.0)]}, late + 10_000_000_000)
    assert ensure_card_mask(crops, 256, KEY, 1) == first
    assert cached.stat().st_mtime_ns == marker, "another tile's circle moved: this layer is still valid"

    _circles(crops, {KEY: [_circle(9, 40.0)], OTHER: [_circle(8, 40.0)]}, late + 20_000_000_000)
    moved = _decode(ensure_card_mask(crops, 256, KEY, 1))
    assert cached.stat().st_mtime_ns != marker
    assert moved[224, 160, 1] > 0 and moved[224, 32, 1] == 0, "the negative circle is drawn where it is now"


def test_a_layer_from_before_circle_digests_is_rebuilt(tmp_path: Path) -> None:
    crops = _tile(tmp_path)
    ensure_card_mask(crops, 256, KEY, 1)
    cached = card_mask_path(crops, 256, KEY, 1)
    with Image.open(cached) as image:
        image.load()
        plain = image.copy()
    plain.save(cached)  # no circles field, as every layer written before the digest was
    newer = cached.stat().st_mtime_ns + 10_000_000_000
    os.utime(cached, ns=(newer, newer))
    ensure_card_mask(crops, 256, KEY, 1)
    with Image.open(cached) as image:
        assert "circles" in image.info


def _catalog(crops: Path, cards: list[tuple[int, str]]) -> None:
    from catalog import connect

    connection = connect(crops)
    for location_id, image_key in cards:
        connection.execute(
            "INSERT INTO locations (location_id, z, structure_id, structure_label, type_id, type_name, radius, "
            "image_key, image_relpath, mask_relpath, ignored) VALUES (?, 5, 7, 'c', 1, 'Cell', 5.0, ?, ?, ?, 0)",
            [location_id, image_key, f"images/{image_key}.png", f"masks/{image_key}_{location_id}.png"],
        )
    connection.commit()
    connection.close()


def test_backfill_builds_every_card_at_each_size_once(tmp_path: Path) -> None:
    from cardmask import backfill_card_layers

    crops = _tile(tmp_path)
    _catalog(crops, [(1, KEY), (2, KEY), (3, KEY)])  # 3 has no mask: skipped, not an error
    assert backfill_card_layers(crops, workers=2) == 6
    for size in (256, 384, 512):
        assert card_mask_path(crops, size, KEY, 1).is_file() and card_mask_path(crops, size, KEY, 2).is_file()
    assert backfill_card_layers(crops, workers=2) == 0, "a second pass only checks"
    assert backfill_card_layers(crops, workers=0) == 0


def test_the_watch_sees_a_rewritten_circle_file(tmp_path: Path) -> None:
    from thumbs import registry_signature

    registry = tmp_path / "set"
    crops = _tile(registry / "RC1")
    _circles(crops, {}, 1_000_000_000_000_000_000)
    before = registry_signature(registry)
    _circles(crops, {}, 1_000_000_010_000_000_000)
    assert registry_signature(registry) != before
