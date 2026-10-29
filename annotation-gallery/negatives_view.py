"""Negative-context bitmask for one annotation: what the trainer treats as background in that tile.

The gallery draws this over the tile so a reviewer sees the same other-annotation areas that
``sam2_segmentation_trainer`` stamps as -1 into its mask prompt. The pixels come from
``annotation_crops.negatives.collect_negative_regions``, the function the trainer calls, so the two cannot drift.
"""

from __future__ import annotations

import io
import json
import os
import re
import threading
from pathlib import Path

import crops_path  # noqa: F401
import numpy as np
from PIL import Image, PngImagePlugin

from annotation_crops.negatives import (
    circle_file_path,
    collect_negative_regions,
    load_circle_annotations,
    stamp_negative_regions,
)

IMAGE_KEY_PATTERN = re.compile(r"[A-Za-z0-9_.\-]+")
"""Characters an ``imageKey`` may contain; anything else is rejected before it touches a path."""

COUNT_FIELD = "negatives"
"""PNG text field in a cached bitmask that holds the number of negative annotations."""

CACHE_DIR = "cache"
"""Folder under ``AnnotationCrops`` for cached bitmasks. It is not served as a raw file."""


_CIRCLE_VOLUMES_KEPT = 4
"""Parsed circle files kept in memory, one per volume: 57 MB for RC1's 37 MB file, 91 MB for RPC1's 59 MB one."""

_circle_cache: dict[str, tuple[tuple[int, int], dict]] = {}
_circle_lock = threading.Lock()


def _circles_by_key(crops: Path) -> dict:
    """``load_circle_annotations(crops)``, parsed once per version of the file (its mtime and size).

    Every negative bitmask needs its tile's circles. Parsing the whole volume's file for each one took 0.5-1 s, about
    92% of building a card layer. One lock covers the parse, so card layers built at once wait for a single parse.
    """
    path = circle_file_path(crops)
    try:
        info = path.stat()
        stamp = (info.st_mtime_ns, info.st_size)
    except OSError:
        return {}
    name = str(path)
    with _circle_lock:
        hit = _circle_cache.get(name)
        if hit is not None and hit[0] == stamp:
            _circle_cache[name] = _circle_cache.pop(name)
            return hit[1]
        index = load_circle_annotations(crops)
        _circle_cache.pop(name, None)
        _circle_cache[name] = (stamp, index)
        while len(_circle_cache) > _CIRCLE_VOLUMES_KEPT:
            _circle_cache.pop(next(iter(_circle_cache)))
        return index


def cache_path(crops: Path, image_key: str, location_id: int) -> Path:
    """Where the bitmask for ``(image_key, location_id)`` is cached."""
    return crops / CACHE_DIR / f"neg_{image_key}_{location_id}.png"


def original_mask_path(crops: Path, image_key: str, location_id: int) -> Path:
    """The converter's current-outline mask of a location (the annotation as it is in the database). Gallery
    volumes have none; there the file does not exist and nothing changes."""
    return crops / "original" / f"{image_key}_{int(location_id)}.png"


def _with_original(bitmask, crops: Path, image_key: str, location_id: int):
    """``bitmask`` (0/255) with the converter's existing annotation added, and how many regions that adds (0 or 1).

    The existing annotation is drawn the way the other annotations are, so a reviewer sees it against the proposal:
    where only it covers is red, where both do is yellow."""
    path = original_mask_path(crops, image_key, location_id)
    try:
        with Image.open(path) as image:
            original = np.asarray(image.convert("L"))
    except (OSError, ValueError):
        return bitmask, 0
    if original.shape != bitmask.shape:
        original = np.asarray(Image.fromarray(original).resize(bitmask.shape[::-1], Image.Resampling.NEAREST))
    if not (original > 127).any():
        return bitmask, 0
    return np.where(original > 127, 255, bitmask).astype("uint8"), 1


def _mtime_ns(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


def _sidecar(crops: Path, image_key: str) -> tuple[list[int], tuple[int, int]]:
    """Annotation ids and ``(height, width)`` from ``images/{imageKey}.json``.

    Raises FileNotFoundError when the sidecar is missing and ValueError when it has no usable image size.
    """
    payload = json.loads((crops / "images" / f"{image_key}.json").read_text(encoding="utf-8"))
    ids = [int(ann.get("id", -1)) for ann in (payload.get("annotations") or [])]
    image = payload.get("image")
    if isinstance(image, list):
        image = image[0] if image else None
    if not isinstance(image, dict) or "width" not in image or "height" not in image:
        raise ValueError(f"{image_key}: sidecar has no image size")
    return ids, (int(image["height"]), int(image["width"]))


def _read_cache(path: Path, newest_source_ns: int) -> tuple[bytes, int] | None:
    """Cached PNG bytes and count when the file is at least as new as every source and decodes, else None.

    ``Image.open`` reads only the header, so a file cut short mid-pixels would pass it; ``load`` decodes the
    pixels so a torn file counts as a miss and is rebuilt.
    """
    try:
        if path.stat().st_mtime_ns < newest_source_ns:
            return None
        data = path.read_bytes()
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            return data, int(image.info.get(COUNT_FIELD, "0"))
    except (OSError, ValueError):
        return None


def negative_bitmask(crops: Path, image_key: str, location_id: int) -> tuple[bytes, int]:
    """8-bit PNG (0 = no negative, 255 = other annotation) for the tile, and how many other annotations it has.

    Computed with :func:`annotation_crops.negatives.collect_negative_regions` from the tile's other mask PNGs and
    the circle annotations, then cached under ``cache/``. The cache is reused until the sidecar, the circle file
    or any neighbor mask is newer than it. A cache that cannot be written is ignored, so a read-only share
    still works.

    Raises ValueError for an unsafe key or a sidecar without a size, and FileNotFoundError when the tile's
    sidecar does not exist.
    """
    if not IMAGE_KEY_PATTERN.fullmatch(image_key):
        raise ValueError(f"invalid image key {image_key!r}")
    ids, shape = _sidecar(crops, image_key)
    masks = crops / "masks"
    sources = [
        crops / "images" / f"{image_key}.json",
        circle_file_path(crops),
        original_mask_path(crops, image_key, location_id),
        *(masks / f"{image_key}_{other}.png" for other in ids if other != location_id),
    ]
    newest = max(_mtime_ns(source) for source in sources)
    cached_file = cache_path(crops, image_key, location_id)
    hit = _read_cache(cached_file, newest)
    if hit is not None:
        return hit

    circles = _circles_by_key(crops).get(image_key, ())
    regions = collect_negative_regions(image_key, location_id, ids, circles, masks, shape)
    canvas = stamp_negative_regions(regions, shape)
    bitmask = (canvas < 0).astype("uint8") * 255
    bitmask, added = _with_original(bitmask, crops, image_key, location_id)
    count = len(regions) + added
    info = PngImagePlugin.PngInfo()
    info.add_text(COUNT_FIELD, str(count))
    buffer = io.BytesIO()
    Image.fromarray(bitmask, mode="L").save(buffer, format="PNG", pnginfo=info, optimize=True)
    data = buffer.getvalue()
    # Write beside the cache file and rename, as cardmask.ensure_card_mask does, so a crash or a concurrent reader
    # never sees half a PNG at the cache path. cardmask imports this module, so the helper cannot be shared yet.
    temporary = cached_file.with_name(f".{cached_file.name}.{threading.get_ident()}.tmp")
    try:
        cached_file.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_bytes(data)
        os.replace(temporary, cached_file)
    except OSError:
        pass
    finally:
        temporary.unlink(missing_ok=True)
    return data, count
