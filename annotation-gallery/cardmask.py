"""Card-size mask layers: one small RGB PNG per card carrying the target mask and its negatives.

A grid card used to download the full 1024 mask PNG although its tissue is a 256, 384 or 512 JPEG. This module
builds ``overlays/cardmask/{size}/{imageKey}_{locationId}.png`` instead, an 8-bit RGB image the size of the
thumbnail:

* R: the location's own mask
* G: the other annotations in the tile (the trainer's negatives), excluding the target
* B: where the two overlap

Each channel is box-downsampled and keeps 0-255, so the browser can use it as soft alpha. A file is valid only
while it is newer than every source that feeds it; the gallery also deletes a tile's files outright when it
changes a mask (:func:`drop_card_masks`).
"""

from __future__ import annotations

import io
import os
import threading
from pathlib import Path

import crops_path  # noqa: F401
import numpy as np
from PIL import Image

from annotation_crops.negatives import circle_file_stamp
from negatives_view import IMAGE_KEY_PATTERN, _sidecar, negative_bitmask, original_mask_path
from thumbs import THUMB_SIZES

CARD_MASK_DIR = ("overlays", "cardmask")
_MASK_THRESHOLD = 127
_build_slots = threading.BoundedSemaphore(int(os.environ.get("GALLERY_CARDMASK_BUILDS", "4") or 4))


def card_mask_path(crops: Path, size: int, image_key: str, location_id: int) -> Path:
    """Where the layer for one card is stored."""
    return crops.joinpath(*CARD_MASK_DIR, str(size), f"{image_key}_{int(location_id)}.png")


def _mtime_ns(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


def _own_mask_path(crops: Path, image_key: str, location_id: int) -> Path | None:
    """The location's mask for this tile, from ``masks/`` else ``ignored/``."""
    name = f"{image_key}_{int(location_id)}.png"
    for folder in ("masks", "ignored"):
        path = crops / folder / name
        if path.is_file():
            return path
    return None


_ids_cache: dict[tuple[str, str], tuple[int, list[int]]] = {}
_ids_lock = threading.Lock()


def _annotation_ids(crops: Path, image_key: str) -> tuple[int, list[int]]:
    """Sidecar mtime and the annotation ids in the tile, re-read only when the sidecar changes."""
    sidecar = crops / "images" / f"{image_key}.json"
    stamp = _mtime_ns(sidecar)
    cache_key = (str(crops), image_key)
    with _ids_lock:
        hit = _ids_cache.get(cache_key)
    if hit is not None and hit[0] == stamp:
        return hit
    try:
        ids, _shape = _sidecar(crops, image_key)
    except (OSError, ValueError):
        ids = []
    with _ids_lock:
        _ids_cache[cache_key] = (stamp, ids)
        if len(_ids_cache) > 4096:
            _ids_cache.pop(next(iter(_ids_cache)))
    return stamp, ids


def _newest_source_ns(crops: Path, image_key: str, location_id: int, own: Path | None) -> int:
    """Newest mtime among everything a card layer is built from.

    That is the sidecar, the circle file, the location's own mask and every other annotation's mask in the tile.
    The tile's annotation ids come from a cached read of the sidecar, so a warm check is a handful of stats.
    """
    sidecar_ns, ids = _annotation_ids(crops, image_key)
    newest = max(sidecar_ns, circle_file_stamp(crops) or 0)
    if own is not None:
        newest = max(newest, _mtime_ns(own))
    newest = max(newest, _mtime_ns(original_mask_path(crops, image_key, location_id)))
    masks = crops / "masks"
    for other in ids:
        if other != location_id:
            newest = max(newest, _mtime_ns(masks / f"{image_key}_{other}.png"))
    return newest


def _downsample(plane: np.ndarray, size: int) -> np.ndarray:
    """Area-average a 0/255 plane down to ``size`` x ``size``, keeping 0-255."""
    image = Image.fromarray(plane, mode="L")
    return np.asarray(image.resize((size, size), Image.Resampling.BOX), dtype=np.uint8)


def _read_plane(data: bytes | Path) -> np.ndarray:
    source = io.BytesIO(data) if isinstance(data, bytes) else data
    with Image.open(source) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def build_card_mask(crops: Path, size: int, image_key: str, location_id: int) -> bytes:
    """PNG bytes for one card layer. Raises FileNotFoundError when the location has no mask in this tile."""
    own_path = _own_mask_path(crops, image_key, location_id)
    if own_path is None:
        raise FileNotFoundError(f"no mask for {image_key} {location_id}")
    own = _read_plane(own_path) > _MASK_THRESHOLD
    try:
        negative_png, _count = negative_bitmask(crops, image_key, int(location_id))
        negative = _read_plane(negative_png) > _MASK_THRESHOLD
    except FileNotFoundError:
        negative = np.zeros_like(own)
    if negative.shape != own.shape:
        negative = np.asarray(
            Image.fromarray(negative.astype(np.uint8) * 255).resize(own.shape[::-1], Image.Resampling.NEAREST)
        ) > _MASK_THRESHOLD
    planes = (own, negative, own & negative)
    rgb = np.stack([_downsample(plane.astype(np.uint8) * 255, size) for plane in planes], axis=-1)
    buffer = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def ensure_card_mask(crops: Path, size: int, image_key: str, location_id: int) -> bytes:
    """Cached layer when it is newer than every source, else rebuilt and cached.

    Raises ValueError for an unsafe key or size and FileNotFoundError when the location has no mask.
    A cache that cannot be written is ignored, so a read-only share still works.
    """
    if size not in THUMB_SIZES or IMAGE_KEY_PATTERN.fullmatch(image_key) is None:
        raise ValueError("invalid card mask request")
    cached = card_mask_path(crops, size, image_key, location_id)
    own = _own_mask_path(crops, image_key, location_id)
    if own is None:
        raise FileNotFoundError(f"no mask for {image_key} {location_id}")
    try:
        if cached.stat().st_mtime_ns >= _newest_source_ns(crops, image_key, int(location_id), own):
            return cached.read_bytes()
    except OSError:
        pass
    with _build_slots:
        data = build_card_mask(crops, size, image_key, location_id)
    temporary = cached.with_name(f".{cached.name}.{threading.get_ident()}.tmp")
    try:
        cached.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_bytes(data)
        os.replace(temporary, cached)
    except OSError:
        pass
    finally:
        temporary.unlink(missing_ok=True)
    return data


def drop_card_masks(crops: Path, image_keys: list[str]) -> int:
    """Delete every card layer of these tiles. A changed mask also changes its neighbors' negatives."""
    removed = 0
    for key in image_keys:
        if IMAGE_KEY_PATTERN.fullmatch(str(key)) is None:
            continue
        for size in THUMB_SIZES:
            folder = crops.joinpath(*CARD_MASK_DIR, str(size))
            if not folder.is_dir():
                continue
            for path in folder.glob(f"{key}_*.png"):
                path.unlink(missing_ok=True)
                removed += 1
    return removed
