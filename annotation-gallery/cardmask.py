"""Card-size mask layers: one small RGB PNG per card carrying the target mask and its negatives.

A grid card used to download the full 1024 mask PNG although its tissue is a 256, 384 or 512 JPEG. This module
builds ``overlays/cardmask/{size}/{imageKey}_{locationId}.png`` instead, an 8-bit RGB image the size of the
thumbnail:

* R: the location's own mask
* G: the other annotations in the tile (the trainer's negatives), excluding the target
* B: where the two overlap

Each channel is box-downsampled and keeps 0-255, so the browser can use it as soft alpha. A file is valid only
while it is newer than every mask and sidecar that feeds it and was built from the tile's current circles (its
``circles`` text field, :func:`negatives_view.circle_digest`); the gallery also deletes a tile's files outright when
it changes a mask (:func:`drop_card_masks`). :func:`backfill_card_layers` builds them ahead of the first view.
"""

from __future__ import annotations

import io
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import crops_path  # noqa: F401
import numpy as np
from PIL import Image, PngImagePlugin

from negatives_view import (
    CIRCLE_FIELD,
    IMAGE_KEY_PATTERN,
    _sidecar,
    circle_digest,
    negative_bitmask,
    original_mask_path,
)
from thumbs import THUMB_SIZES

CARD_MASK_DIR = ("overlays", "cardmask")
_MASK_THRESHOLD = 127
_build_slots = threading.BoundedSemaphore(int(os.environ.get("GALLERY_CARDMASK_BUILDS", "4") or 4))
# Background builders; each takes one of the build slots, so cards being viewed keep the rest. 0 turns it off.
BACKFILL_WORKERS = int(os.environ.get("GALLERY_CARDMASK_BACKFILL_WORKERS", "2") or 0)


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

    That is the sidecar, the location's own mask and every other annotation's mask in the tile. The circles are
    checked by their digest instead, since the circle file is rewritten whole. The tile's annotation ids come from a
    cached read of the sidecar, so a warm check is a handful of stats.
    """
    sidecar_ns, ids = _annotation_ids(crops, image_key)
    newest = sidecar_ns
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


def _built_from(data: bytes) -> str | None:
    """The circle digest a cached layer records, or None (an older file, or one that does not open)."""
    try:
        with Image.open(io.BytesIO(data)) as image:
            return image.info.get(CIRCLE_FIELD)
    except (OSError, ValueError):
        return None


def build_card_mask(crops: Path, size: int, image_key: str, location_id: int) -> bytes:
    """PNG bytes for one card layer. Raises FileNotFoundError when the location has no mask in this tile."""
    digest = circle_digest(crops, image_key)
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
    info = PngImagePlugin.PngInfo()
    info.add_text(CIRCLE_FIELD, digest)
    buffer = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buffer, format="PNG", pnginfo=info, optimize=True)
    return buffer.getvalue()


def ensure_card_mask(crops: Path, size: int, image_key: str, location_id: int) -> bytes:
    """Cached layer when it is still valid (see the module notes), else rebuilt and cached.

    Raises ValueError for an unsafe key or size and FileNotFoundError when the location has no mask.
    A cache that cannot be written is ignored, so a read-only share still works.
    """
    return _ensure(crops, size, image_key, location_id)[0]


def _ensure(crops: Path, size: int, image_key: str, location_id: int) -> tuple[bytes, bool]:
    """:func:`ensure_card_mask`, and whether it had to build the layer."""
    if size not in THUMB_SIZES or IMAGE_KEY_PATTERN.fullmatch(image_key) is None:
        raise ValueError("invalid card mask request")
    cached = card_mask_path(crops, size, image_key, location_id)
    own = _own_mask_path(crops, image_key, location_id)
    if own is None:
        raise FileNotFoundError(f"no mask for {image_key} {location_id}")
    try:
        if cached.stat().st_mtime_ns >= _newest_source_ns(crops, image_key, int(location_id), own):
            data = cached.read_bytes()
            if _built_from(data) == circle_digest(crops, image_key):
                return data, False
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
    return data, True


def _catalog_cards(crops: Path) -> list[tuple[int, str]]:
    """Every card of the volume, ``(location_id, image_key)``, in section order."""
    from catalog import connect

    connection = connect(crops)
    try:
        rows = connection.execute(
            "SELECT location_id, image_key, MIN(z) AS z FROM locations WHERE image_key IS NOT NULL"
            " GROUP BY location_id, image_key ORDER BY z, location_id"
        ).fetchall()
    finally:
        connection.close()
    return [(int(row[0]), str(row[1])) for row in rows]


def backfill_card_layers(crops: Path, workers: int | None = None) -> int:
    """Build every card's layer at each card size (Large first) before it is viewed. Returns how many were built.

    A layer that is still valid is only checked, so a pass after an Update All rebuilds just the tiles whose masks or
    circles changed. Cards without a mask are skipped. ``workers`` defaults to :data:`BACKFILL_WORKERS`.
    """
    workers = BACKFILL_WORKERS if workers is None else workers
    if workers <= 0:
        return 0
    cards = _catalog_cards(crops)
    sizes = sorted(THUMB_SIZES, reverse=True)

    def one(card: tuple[int, str]) -> int:
        location_id, image_key = card
        built = 0
        for size in sizes:
            try:
                built += _ensure(crops, size, image_key, location_id)[1]
            except (FileNotFoundError, ValueError):
                return built
            except OSError as exc:
                print(f"annotation-gallery card layer {image_key} {location_id}: {exc}", flush=True)
        return built

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gallery-card-layers") as pool:
        return sum(pool.map(one, cards))


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
