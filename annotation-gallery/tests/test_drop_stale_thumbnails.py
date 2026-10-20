"""``refresh.drop_stale_thumbnails``: which cached card images it removes and which files it must never touch."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from hypothesis import example, given, settings
from hypothesis import strategies as st

GALLERY_ROOT = Path(__file__).resolve().parents[1]
if str(GALLERY_ROOT) not in sys.path:
    sys.path.insert(0, str(GALLERY_ROOT))

from cardmask import CARD_MASK_DIR, card_mask_path  # noqa: E402
from refresh import drop_stale_thumbnails  # noqa: E402
from thumbs import THUMB_SIZES  # noqa: E402

_KEY = st.from_regex(r"[A-Za-z0-9][A-Za-z0-9_-]{0,30}", fullmatch=True)
_UNSAFE_KEYS = ("../x", "..", "a/../x", "x/y", "", ".hidden", ".." + chr(92) + "x", "a b", "x" + chr(0) + "y")


def _thumb(crops: Path, size: int, key: str) -> Path:
    path = crops / "overlays" / "thumb" / str(size) / f"{key}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"jpeg")
    return path


def _card_mask(crops: Path, size: int, key: str, location_id: int) -> Path:
    path = card_mask_path(crops, size, key, location_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"png")
    return path


@settings(max_examples=40, deadline=None)
@given(listed=st.sets(_KEY, min_size=1, max_size=4), other=st.sets(_KEY, max_size=4))
@example(listed={"V"}, other={"v"})  # Windows file names are case-insensitive: "v" is the same file as "V"
def test_listed_keys_lose_every_size_and_unlisted_keys_stay(listed: set[str], other: set[str]) -> None:
    # Compare case-folded: on a case-insensitive filesystem keys differing only by case name the same file.
    folded = {c.casefold() for c in listed}
    kept_keys = {k for k in other if k.casefold() not in folded and not any(k.casefold().startswith(f"{c}_") for c in folded)}
    with tempfile.TemporaryDirectory() as tmp:
        crops = Path(tmp)
        gone = [_thumb(crops, size, key) for key in listed for size in THUMB_SIZES]
        kept = [_thumb(crops, size, key) for key in kept_keys for size in THUMB_SIZES]
        drop_stale_thumbnails(crops, sorted(listed))
        assert not any(path.exists() for path in gone)
        assert all(path.exists() for path in kept)


def test_every_thumb_size_is_covered() -> None:
    assert THUMB_SIZES == {512, 384, 256}, "update the size loop expectations when sizes change"
    with tempfile.TemporaryDirectory() as tmp:
        crops = Path(tmp)
        paths = [_thumb(crops, size, "RC1_5_D1_X0-64_Y0-64") for size in THUMB_SIZES]
        drop_stale_thumbnails(crops, ["RC1_5_D1_X0-64_Y0-64"])
        assert [p.exists() for p in paths] == [False] * len(paths)


def test_unsafe_keys_never_unlink_outside_the_thumbnail_folders() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        crops = Path(tmp) / "crops"
        # "../x" from overlays/thumb/<size> lands in thumb/, "../../x" in overlays/, "../../../x" in crops, and so on.
        sentinels = [
            crops / "overlays" / "thumb" / "x.jpg",
            crops / "overlays" / "x.jpg",
            crops / "x.jpg",
            Path(tmp) / "x.jpg",
        ]
        for path in sentinels:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"keep")
        keys = [*_UNSAFE_KEYS, "../../x", "../../../x", "../../../../x"]
        drop_stale_thumbnails(crops, keys)
        assert [p.exists() for p in sentinels] == [True] * len(sentinels)


def test_an_unsafe_key_does_not_stop_the_safe_ones_after_it() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        crops = Path(tmp)
        path = _thumb(crops, 256, "good_1")
        drop_stale_thumbnails(crops, ["../x", "good_1"])
        assert not path.exists()


def test_missing_files_and_missing_folders_are_tolerated() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        crops = Path(tmp)
        drop_stale_thumbnails(crops, ["nothing_1"])
        _thumb(crops, 512, "other_1")
        drop_stale_thumbnails(crops, ["nothing_1", "nothing_1"])
        assert (crops / "overlays" / "thumb" / "512" / "other_1.jpg").exists()
        drop_stale_thumbnails(crops, [])


def test_card_mask_layers_of_listed_tiles_are_dropped_too() -> None:
    key = "RC1_5_D1_X0-64_Y0-64"
    with tempfile.TemporaryDirectory() as tmp:
        crops = Path(tmp)
        gone = [_card_mask(crops, size, key, loc) for size in THUMB_SIZES for loc in (1, 2)]
        kept = [_card_mask(crops, size, "OTHER_9_D1_X0-64_Y0-64", 1) for size in THUMB_SIZES]
        drop_stale_thumbnails(crops, [key])
        assert not any(p.exists() for p in gone)
        assert all(p.exists() for p in kept)
        assert (crops.joinpath(*CARD_MASK_DIR, "512")).is_dir()
