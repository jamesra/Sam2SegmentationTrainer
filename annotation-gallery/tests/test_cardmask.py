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
