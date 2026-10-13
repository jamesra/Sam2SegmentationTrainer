"""Negative-context bitmask for one annotation: what the trainer treats as background in that tile.

The gallery draws this over the tile so a reviewer sees the same other-annotation areas that
``sam2_segmentation_trainer`` stamps as -1 into its mask prompt. The pixels come from
``annotation_crops.negatives.collect_negative_regions``, the function the trainer calls, so the two cannot drift.
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path

import crops_path  # noqa: F401
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


def cache_path(crops: Path, image_key: str, location_id: int) -> Path:
    """Where the bitmask for ``(image_key, location_id)`` is cached."""
    return crops / CACHE_DIR / f"neg_{image_key}_{location_id}.png"


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
    """Cached PNG bytes and count when the file is at least as new as every source, else None."""
    try:
        if path.stat().st_mtime_ns < newest_source_ns:
            return None
        data = path.read_bytes()
        with Image.open(io.BytesIO(data)) as image:
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
        *(masks / f"{image_key}_{other}.png" for other in ids if other != location_id),
    ]
    newest = max(_mtime_ns(source) for source in sources)
    cached_file = cache_path(crops, image_key, location_id)
    hit = _read_cache(cached_file, newest)
    if hit is not None:
        return hit

    circles = load_circle_annotations(crops).get(image_key, ())
    regions = collect_negative_regions(image_key, location_id, ids, circles, masks, shape)
    canvas = stamp_negative_regions(regions, shape)
    bitmask = (canvas < 0).astype("uint8") * 255
    info = PngImagePlugin.PngInfo()
    info.add_text(COUNT_FIELD, str(len(regions)))
    buffer = io.BytesIO()
    Image.fromarray(bitmask, mode="L").save(buffer, format="PNG", pnginfo=info, optimize=True)
    data = buffer.getvalue()
    try:
        cached_file.parent.mkdir(parents=True, exist_ok=True)
        cached_file.write_bytes(data)
    except OSError:
        pass
    return data, len(regions)
