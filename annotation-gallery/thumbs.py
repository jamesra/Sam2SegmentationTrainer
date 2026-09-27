"""Cached JPEG thumbnails for gallery cards.

Huge cards and the zoom view use the original crop PNG. Large, Medium, and
Small read ``overlays/thumb/{512|384|256}/{imageKey}.jpg``. A JPEG is rewritten
when it is missing or older than that window's newest mask.

A background watch polls the registry symlink and the ``images``, ``masks``,
and ``ignored`` folder modified times. One backfill runs at a time. A change
that arrives during a run queues a single follow-up pass, which reads the tree
again when it starts. Thumbnail writes are not part of that signature, so
filling the cache does not schedule another pass. Replacing a file in place
does not change its parent folder time; that crop is still rebuilt when a card
asks for it, or on the next pass a folder change does start.
"""

from __future__ import annotations

import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

THUMB_SIZES = frozenset({512, 384, 256})
JPEG_QUALITY = 90
IMAGE_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,200}$")
_CROP_SUFFIXES = (".png", ".jpg", ".jpeg")
_ENCODE_WORKERS = 4
_WATCH_INTERVAL_S = 5.0
_WATCH_FOLDERS = ("images", "masks", "ignored")

_indexes: dict[str, dict[str, int]] = {}
_index_lock = threading.Lock()


def thumb_parts_allowed(parts: list[str]) -> bool:
    """True for ``overlays/thumb/{512|384|256}/{imageKey}.jpg`` only."""
    if len(parts) != 4 or parts[0] != "overlays" or parts[1] != "thumb":
        return False
    if parts[2] not in {str(size) for size in THUMB_SIZES}:
        return False
    name = parts[3]
    if not name.endswith(".jpg"):
        return False
    return IMAGE_KEY.fullmatch(name[:-4]) is not None


def parse_thumb_request(relative: str) -> tuple[int, str] | None:
    """Return ``(size, image_key)`` when *relative* is a thumbnail path."""
    parts = relative.replace("\\", "/").split("/")
    if not thumb_parts_allowed(parts):
        return None
    return int(parts[2]), parts[3][:-4]


def image_key_from_mask_name(name: str) -> str | None:
    """Image key from ``{imageKey}_{locationId}.png``. Other names are skipped."""
    if not name.endswith(".png"):
        return None
    key, separator, suffix = name[:-4].rpartition("_")
    if not separator or not suffix.isdigit() or IMAGE_KEY.fullmatch(key) is None:
        return None
    return key


def scan_mask_reference_ns(crops: Path) -> dict[str, int]:
    """Newest mask or ignored-mask mtime, in nanoseconds, per image key."""
    newest: dict[str, int] = {}
    for folder in ("masks", "ignored"):
        directory = crops / folder
        if not directory.is_dir():
            continue
        for entry in os.scandir(directory):
            if not entry.is_file(follow_symlinks=False):
                continue
            key = image_key_from_mask_name(entry.name)
            if key is None:
                continue
            mtime = entry.stat(follow_symlinks=False).st_mtime_ns
            previous = newest.get(key)
            if previous is None or mtime > previous:
                newest[key] = mtime
    return newest


def remember_mask_index(crops: Path, index: dict[str, int]) -> None:
    """Keep a volume's mask mtimes so later card requests do not rescan."""
    with _index_lock:
        _indexes[str(crops.resolve())] = index


def mask_reference_ns(crops: Path, image_key: str) -> int | None:
    """Mask mtime for one image key, scanning the volume once per process."""
    resolved = str(crops.resolve())
    with _index_lock:
        index = _indexes.get(resolved)
    if index is None:
        index = scan_mask_reference_ns(crops)
        remember_mask_index(crops, index)
    return index.get(image_key)


def is_fresh(jpeg_ns: int | None, reference_ns: int | None) -> bool:
    """A missing JPEG is stale. With no mask, an existing JPEG is kept."""
    if jpeg_ns is None:
        return False
    if reference_ns is None:
        return True
    return jpeg_ns >= reference_ns


def find_crop_image(crops: Path, image_key: str) -> Path | None:
    """Crop file for *image_key*, PNG first, then JPEG."""
    images = crops / "images"
    for suffix in _CROP_SUFFIXES:
        path = images / f"{image_key}{suffix}"
        if path.is_file():
            return path
    return None


def iter_crop_keys(crops: Path) -> list[str]:
    """Image keys present under ``images/``, from one directory scan."""
    images = crops / "images"
    if not images.is_dir():
        return []
    keys: set[str] = set()
    for entry in os.scandir(images):
        if not entry.is_file(follow_symlinks=False):
            continue
        suffix = Path(entry.name).suffix.lower()
        if suffix not in _CROP_SUFFIXES:
            continue
        stem = Path(entry.name).stem
        if IMAGE_KEY.fullmatch(stem):
            keys.add(stem)
    return sorted(keys)


def scan_thumb_mtime_ns(directory: Path) -> dict[str, int]:
    """JPEG mtimes under one size directory. Missing directory is an empty map."""
    found: dict[str, int] = {}
    if not directory.is_dir():
        return found
    for entry in os.scandir(directory):
        if not entry.is_file(follow_symlinks=False) or not entry.name.endswith(".jpg"):
            continue
        key = entry.name[:-4]
        if IMAGE_KEY.fullmatch(key):
            found[key] = entry.stat(follow_symlinks=False).st_mtime_ns
    return found


def _save_jpeg(rgb: Image.Image, dest: Path, size: int) -> None:
    """Write one square JPEG from an image already in memory. Replace is atomic."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_name(f".{dest.name}.{threading.get_ident()}.tmp")
    try:
        resized = rgb.resize((size, size), Image.Resampling.LANCZOS)
        resized.save(temporary, format="JPEG", quality=JPEG_QUALITY)
        os.replace(temporary, dest)
    finally:
        temporary.unlink(missing_ok=True)


def _write_jpegs(source: Path, outputs: list[tuple[Path, int]]) -> None:
    """Decode *source* once and write every ``(dest, size)`` JPEG from that bitmap."""
    if not outputs:
        return
    with Image.open(source) as image:
        rgb = image.convert("RGB")
        for dest, size in outputs:
            _save_jpeg(rgb, dest, size)


def _write_jpeg(source: Path, dest: Path, size: int) -> None:
    """Resize *source* to one square JPEG at *dest*."""
    _write_jpegs(source, [(dest, size)])


def ensure_thumbnail(
    crops: Path,
    size: int,
    image_key: str,
    *,
    reference_ns: int | None = None,
    reference_known: bool = False,
) -> Path | None:
    """Return the JPEG, encoding it when it is missing or older than the mask.

    ``None`` means the crop image is not on disk. Huge is not a thumbnail and
    is never written here.
    """
    if size not in THUMB_SIZES or IMAGE_KEY.fullmatch(image_key) is None:
        raise ValueError("invalid thumbnail")
    dest = crops / "overlays" / "thumb" / str(size) / f"{image_key}.jpg"
    if not reference_known:
        reference_ns = mask_reference_ns(crops, image_key)
    if dest.is_file() and is_fresh(dest.stat().st_mtime_ns, reference_ns):
        return dest
    source = find_crop_image(crops, image_key)
    if source is None:
        return None
    _write_jpeg(source, dest, size)
    return dest


def _thumb_dest(crops: Path, size: int, image_key: str) -> Path:
    return crops / "overlays" / "thumb" / str(size) / f"{image_key}.jpg"


def _encode_key(job: tuple[Path, str, list[int]]) -> int:
    """Write every stale size for one crop. The source file is opened once."""
    crops, image_key, sizes = job
    try:
        source = find_crop_image(crops, image_key)
        if source is None:
            return 0
        _write_jpegs(source, [(_thumb_dest(crops, size, image_key), size) for size in sizes])
    except OSError as exc:
        print(f"annotation-gallery thumbnail {image_key}: {exc}", flush=True)
        return 0
    return len(sizes)


def backfill_crops(crops: Path) -> int:
    """Encode outdated thumbnails for one AnnotationCrops tree. Returns how many were written.

    Each crop is decoded once. Every stale size (256, 384, and 512) is resized
    from that bitmap before the next crop is opened. A JPEG that is already at
    least as new as the mask is not written.
    """
    references = scan_mask_reference_ns(crops)
    remember_mask_index(crops, references)
    existing = {
        size: scan_thumb_mtime_ns(crops / "overlays" / "thumb" / str(size))
        for size in THUMB_SIZES
    }
    jobs: list[tuple[Path, str, list[int]]] = []
    for key in iter_crop_keys(crops):
        reference_ns = references.get(key)
        stale = [
            size
            for size in sorted(THUMB_SIZES)
            if not is_fresh(existing[size].get(key), reference_ns)
        ]
        if stale:
            jobs.append((crops, key, stale))
    if not jobs:
        return 0
    written = 0
    with ThreadPoolExecutor(max_workers=_ENCODE_WORKERS) as pool:
        for count in pool.map(_encode_key, jobs):
            written += count
    return written


def _backfill_registry(registry: Path) -> None:
    if not registry.is_dir():
        return
    written = 0
    for child in registry.iterdir():
        crops = child / "AnnotationCrops"
        if not crops.is_dir():
            continue
        try:
            written += backfill_crops(crops)
        except OSError as exc:
            print(f"annotation-gallery thumbnails {child.name}: {exc}", flush=True)
    print(f"annotation-gallery thumbnails: wrote {written}", flush=True)


def _dir_mtime_ns(path: Path) -> int | None:
    """Folder modified time, or ``None`` when the folder is absent."""
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def _link_target(registry: Path) -> str:
    """Symlink target of the registry path. An ordinary directory contributes nothing."""
    try:
        if registry.is_symlink():
            return os.readlink(registry)
    except OSError:
        return ""
    return ""


def registry_signature(registry: Path) -> tuple[str, tuple[tuple[str, tuple[int | None, ...]], ...]]:
    """Identity of the training tree for the thumbnail watch.

    The symlink target catches a retargeted ``current`` pointer even when the
    new tree's folder times were preserved. Each volume contributes the modified
    times of ``images``, ``masks``, and ``ignored``. ``overlays/thumb`` is
    omitted so writing a JPEG does not look like a new export.
    """
    rows: list[tuple[str, tuple[int | None, ...]]] = []
    try:
        children = sorted(registry.iterdir(), key=lambda path: path.name)
    except OSError:
        children = []
    for child in children:
        crops = child / "AnnotationCrops"
        if not crops.is_dir():
            continue
        rows.append((child.name, tuple(_dir_mtime_ns(crops / name) for name in _WATCH_FOLDERS)))
    return (_link_target(registry), tuple(rows))


class _BackfillGate:
    """One registry backfill at a time, plus a single coalesced follow-up."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running = False
        self._queued = False
        self._registry: Path | None = None
        self._thread: threading.Thread | None = None

    def request(self, registry: Path) -> None:
        """Start a pass, or remember that one more pass is needed."""
        start = False
        with self._lock:
            self._registry = registry
            if self._running:
                self._queued = True
            else:
                self._running = True
                start = True
        if start:
            thread = threading.Thread(target=self._run, name="gallery-thumbnails", daemon=True)
            with self._lock:
                self._thread = thread
            thread.start()

    def _run(self) -> None:
        while True:
            with self._lock:
                registry = self._registry
            if registry is not None:
                try:
                    _backfill_registry(registry)
                except OSError as exc:
                    print(f"annotation-gallery thumbnails: {exc}", flush=True)
            with self._lock:
                if self._queued:
                    self._queued = False
                    continue
                self._running = False
                return


_gate = _BackfillGate()


def request_backfill(registry: Path) -> None:
    """Schedule a thumbnail pass. A running pass keeps only the latest follow-up."""
    _gate.request(registry)


def _watch_registry(registry: Path, interval_s: float) -> None:
    """Poll *registry* and schedule a backfill whenever its signature changes."""
    seen: tuple[str, tuple[tuple[str, tuple[int | None, ...]], ...]] | None = None
    while True:
        try:
            signature = registry_signature(registry)
        except OSError as exc:
            print(f"annotation-gallery thumbnail watch: {exc}", flush=True)
            signature = None
        if signature != seen:
            seen = signature
            request_backfill(registry)
        time.sleep(interval_s)


def start_thumbnail_watch(registry: Path, interval_s: float = _WATCH_INTERVAL_S) -> None:
    """Watch *registry* and fill thumbnails without blocking HTTP startup.

    The first observation schedules a pass immediately. Later passes start only
    when the symlink target or a watched folder time changes.
    """
    thread = threading.Thread(
        target=_watch_registry,
        args=(registry, interval_s),
        name="gallery-thumbnail-watch",
        daemon=True,
    )
    thread.start()


def start_thumbnail_backfill(registry: Path) -> None:
    """Schedule one thumbnail pass. Prefer :func:`start_thumbnail_watch` at startup."""
    request_backfill(registry)
