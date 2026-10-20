"""A torn negatives cache file is never served and never left behind by an interrupted write."""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

GALLERY_ROOT = Path(__file__).resolve().parents[1]
if str(GALLERY_ROOT) not in sys.path:
    sys.path.insert(0, str(GALLERY_ROOT))

import negatives_view  # noqa: E402
from negatives_view import cache_path, negative_bitmask  # noqa: E402

KEY = "RC2_17_D1_X0-32_Y0-32"
SIZE = 32


@pytest.fixture
def crops(tmp_path: Path) -> Path:
    path = tmp_path / "AnnotationCrops"
    (path / "images").mkdir(parents=True)
    (path / "masks").mkdir()
    sidecar = {
        "image": {"file_name": f"{KEY}.png", "width": SIZE, "height": SIZE},
        "annotations": [{"id": 1}, {"id": 2}],
    }
    (path / "images" / f"{KEY}.json").write_text(json.dumps(sidecar), encoding="utf-8")
    # Noise, so the cached PNG has a long pixel stream that a torn write can cut inside.
    other = np.random.default_rng(0).integers(0, 2, (SIZE, SIZE), dtype=np.uint8) * 255
    Image.fromarray(other).save(path / "masks" / f"{KEY}_2.png")
    return path


def _decodes(data: bytes) -> bool:
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
    except (OSError, ValueError):
        return False
    return True


def test_truncated_cache_file_is_rebuilt_not_served(crops: Path) -> None:
    good, count = negative_bitmask(crops, KEY, 1)
    cached = cache_path(crops, KEY, 1)
    torn = good[: len(good) - len(good) // 4]
    with Image.open(io.BytesIO(torn)) as header_only:
        assert header_only.size == (SIZE, SIZE)  # the header and count survive, so only decoding shows the damage
    assert not _decodes(torn)
    cached.write_bytes(torn)
    # A fresh mtime is what an interrupted write leaves, so the cache looks newer than every source.
    future = cached.stat().st_mtime_ns + 10_000_000_000
    os.utime(cached, ns=(future, future))

    data, rebuilt_count = negative_bitmask(crops, KEY, 1)

    assert _decodes(data)
    assert (data, rebuilt_count) == (good, count)
    assert cached.read_bytes() == good  # the torn file was replaced on disk too


def test_interrupted_write_leaves_no_file_at_the_cache_path(crops: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original_write_bytes = Path.write_bytes

    def write_half(self: Path, data: bytes) -> int:
        original_write_bytes(self, data[: len(data) // 2])
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_bytes", write_half)

    data, _count = negative_bitmask(crops, KEY, 1)

    assert _decodes(data)
    cache_dir = cache_path(crops, KEY, 1).parent
    assert not cache_path(crops, KEY, 1).exists()
    assert not list(cache_dir.glob(".*tmp"))  # no stray temp file either


def test_cache_hit_still_served_when_intact(crops: Path) -> None:
    first = negative_bitmask(crops, KEY, 1)
    cached = cache_path(crops, KEY, 1)
    before = cached.stat().st_mtime_ns
    assert negative_bitmask(crops, KEY, 1) == first
    assert cached.stat().st_mtime_ns == before
    assert negatives_view.COUNT_FIELD
