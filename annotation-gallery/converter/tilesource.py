"""Tiles of a volume's section images, read over HTTP from its VikingXML, and cut into the segmentation server's 1024 grid.

The VikingXML lists each section's tilesets: a grid of PNG tiles per downsample level, named
``{prefix}X{ix:03d}_Y{iy:03d}{postfix}`` (nornir's grid tile name). Mosaic space is those images' pixel space and its Y
grows with the image row, as in nornir (``MOSAIC_Y_INCREASES_WITH_IMAGE_ROW``).

The segmentation protocol's tile (row, col) at downsample D covers mosaic pixels x in ``[col*1024, +1024)`` and y in
``[row*1024, +1024)`` (in downsample-D pixels) and its image is that region with **row 0 at the highest y**, the way
Viking renders the world with Y up. A stored tile has row 0 at the lowest y, so the 1024 tile is the stitched region
flipped vertically. :func:`flip_for_protocol` is the one place that does it; every mask goes through the same flip.
Gallery card images stay in the stored orientation, like nornir's training crops.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import re
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Tuple
from xml.etree import ElementTree

import numpy as np
from PIL import Image

LOGGER = logging.getLogger("converter.tilesource")
PROTOCOL_TILE = 1024

HttpGet = Callable[[str, Optional[Mapping[str, str]]], Tuple[int, Mapping[str, str], bytes]]


class TileSourceError(Exception):
    pass


@dataclass(frozen=True)
class Level:
    downsample: int
    path: str
    grid_x: int
    grid_y: int


@dataclass(frozen=True)
class Tileset:
    section: int
    section_path: str
    name: str
    path: str
    tile_x: int
    tile_y: int
    prefix: str
    postfix: str
    levels: Mapping[int, Level]

    @property
    def channel(self) -> str:
        if "." in self.name:
            return self.name.split(".", 1)[0]
        parts = [p for p in re.split(r"[\\/]", self.path) if p]
        return parts[0] if len(parts) >= 2 else self.name

    @property
    def filter(self) -> str:
        if "." in self.name:
            return self.name.split(".", 1)[1]
        parts = [p for p in re.split(r"[\\/]", self.path) if p]
        return parts[1] if len(parts) >= 2 else ""

    def tile_url(self, base: str, downsample: int, ix: int, iy: int) -> str:
        level = self.levels[downsample]
        folder = "/".join(
            part.strip("/") for part in (base.rstrip("/"), self.section_path, self.path, level.path) if part
        )
        return f"{folder}/{self.prefix}X{ix:03d}_Y{iy:03d}{self.postfix}".replace("\\", "/")


def _attr(node: ElementTree.Element, *names: str) -> Optional[str]:
    """An attribute under any of its spellings: newer VikingXML capitalises them (Number, Path), RC1's older file does not."""
    for name in names:
        value = node.get(name)
        if value is not None:
            return value
    lowered = {key.lower(): value for key, value in node.attrib.items()}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def _int(value: Optional[str], default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Volume:
    name: Optional[str]
    base_url: str
    tilesets: Mapping[int, Tuple[Tileset, ...]]

    def select(self, section: int, channel: Optional[str] = None, filter_name: Optional[str] = None) -> Tileset:
        """The section's tileset: the named channel/filter if given, else the first (sorted by name)."""
        candidates = self.tilesets.get(int(section), ())
        if not candidates:
            raise TileSourceError(f"section {section} has no tileset in the VikingXML")
        for tileset in candidates:
            if (channel is None or tileset.channel.lower() == channel.lower()) and (
                filter_name is None or tileset.filter.lower() == filter_name.lower()
            ):
                return tileset
        wanted = ".".join(part for part in (channel, filter_name) if part)
        raise TileSourceError(f"section {section} has no tileset {wanted!r}; has {[t.name for t in candidates]}")


def parse_viking_xml(text: str, *, source: str, host: Optional[str] = None) -> Volume:
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError as error:
        raise TileSourceError(f"{source} is not valid VikingXML: {error}") from error
    if root.tag != "Volume":
        raise TileSourceError(f"{source}: expected a Volume element, found {root.tag}")
    base = (host or root.get("path") or root.get("host") or "").strip()
    if not base:
        base = source.rsplit("/", 1)[0] if "/" in source.replace("\\", "/") else ""
    sections = root.find("Sections")
    nodes = sections.findall("Section") if sections is not None else root.findall("Section")
    found: Dict[int, list] = {}
    for section in nodes:
        number = _int(_attr(section, "Number", "number"), -1)
        if number < 0:
            continue
        for node in section.findall("Tileset"):
            levels: Dict[int, Level] = {}
            for level in node.findall("Level"):
                downsample = _int(_attr(level, "Downsample", "downsample"))
                if downsample >= 1:
                    levels[downsample] = Level(
                        downsample, _attr(level, "path", "Path") or "",
                        _int(_attr(level, "GridDimX", "griddimx")), _int(_attr(level, "GridDimY", "griddimy")),
                    )
            if levels:
                found.setdefault(number, []).append(
                    Tileset(
                        section=number, section_path=_attr(section, "Path", "path") or "", name=_attr(node, "name", "Name") or "",
                        path=_attr(node, "path", "Path") or "", tile_x=_int(_attr(node, "TileXDim", "tilexdim")),
                        tile_y=_int(_attr(node, "TileYDim", "tileydim")), prefix=_attr(node, "FilePrefix", "fileprefix") or "",
                        postfix=_attr(node, "FilePostfix", "filepostfix") or ".png", levels=levels,
                    )
                )
    return Volume(
        name=root.get("Name") or None,
        base_url=base,
        tilesets={z: tuple(sorted(items, key=lambda item: item.name)) for z, items in found.items()},
    )


RETRIES = 4
PREFETCH_WORKERS = 8


def default_http_get(url: str, headers: Optional[Mapping[str, str]] = None) -> Tuple[int, Mapping[str, str], bytes]:
    """GET one URL. A network hiccup (a DNS miss, a reset, a timeout, a 5xx) is retried with a growing wait; a 4xx is returned."""
    import time
    import urllib.error
    import urllib.request

    last: Optional[Exception] = None
    for attempt in range(RETRIES):
        request = urllib.request.Request(url, headers={"Accept": "*/*", **(headers or {})})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.status, {k.lower(): v for k, v in response.headers.items()}, response.read()
        except urllib.error.HTTPError as error:
            if error.code < 500:
                return error.code, {k.lower(): v for k, v in error.headers.items()}, b""
            last = error
        except OSError as error:  # URLError (DNS, refused, reset) and timeouts
            last = error
        if attempt < RETRIES - 1:
            time.sleep(1.5 * (2**attempt))
    raise TileSourceError(f"GET {url} failed after {RETRIES} tries: {last}")


def load_volume(source: str, *, host: Optional[str] = None, http_get: HttpGet = default_http_get) -> Volume:
    if re.match(r"^https?://", source, re.IGNORECASE):
        status, _, body = http_get(source, None)
        if status != 200:
            raise TileSourceError(f"GET {source} returned HTTP {status}")
        return parse_viking_xml(body.decode("utf-8-sig", errors="replace"), source=source, host=host)
    return parse_viking_xml(Path(source).read_text(encoding="utf-8-sig"), source=source, host=host)


class TileCache:
    """Raw tile bytes on disk, revalidated with ETag / Last-Modified, plus a small in-memory decoded cache."""

    def __init__(self, folder: Optional[Path], http_get: HttpGet = default_http_get, memory_tiles: int = 256) -> None:
        self._folder = folder
        self._get = http_get
        self._memory: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._limit = memory_tiles
        self._lock = threading.Lock()
        self.requests = 0
        self.not_modified = 0
        self.checked: Dict[str, float] = {}

    def _paths(self, url: str) -> Tuple[Optional[Path], Optional[Path]]:
        if self._folder is None:
            return None, None
        digest = hashlib.sha1(url.encode("utf-8")).hexdigest()
        return self._folder / f"{digest}.png", self._folder / f"{digest}.etag"

    def prefetch(self, urls, workers: int = PREFETCH_WORKERS) -> None:
        """Fetch ``urls`` concurrently into the caches. A failure is left for the later ``fetch`` of that URL to raise."""
        pending = [url for url in dict.fromkeys(urls) if url not in self._memory]
        if len(pending) < 2:
            return

        def one(url: str) -> None:
            try:
                self.fetch(url)
            except Exception:  # reported again, in order, by the caller's own fetch
                pass

        with ThreadPoolExecutor(max_workers=min(workers, len(pending))) as pool:
            list(pool.map(one, pending))

    def fetch(self, url: str) -> Optional[np.ndarray]:
        """The tile as 8-bit gray, or None when the server has no such tile (404)."""
        with self._lock:
            cached = self._memory.get(url)
            if cached is not None:
                self._memory.move_to_end(url)
                return cached
        data_path, etag_path = self._paths(url)
        headers: Dict[str, str] = {}
        body: Optional[bytes] = None
        if data_path is not None and data_path.is_file() and etag_path is not None and etag_path.is_file():
            headers["If-None-Match"] = etag_path.read_text(encoding="utf-8").strip()
            body = data_path.read_bytes()
        self.requests += 1
        status, response_headers, content = self._get(url, headers)
        if status == 304 and body is not None:
            self.not_modified += 1
        elif status == 200:
            body = content
            if data_path is not None and etag_path is not None and response_headers.get("etag"):
                data_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = data_path.with_suffix(".tmp")
                temporary.write_bytes(content)
                os.replace(temporary, data_path)
                etag_path.write_text(response_headers["etag"], encoding="utf-8")
        elif status == 404:
            return None
        else:
            raise TileSourceError(f"GET {url} returned HTTP {status}")
        tile = _decode_gray(body)
        with self._lock:
            self._memory[url] = tile
            while len(self._memory) > self._limit:
                self._memory.popitem(last=False)
        return tile


def _decode_gray(data: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(data))
    if image.mode in ("I;16", "I", "F"):
        array = np.asarray(image, dtype=np.float32)
        peak = max(float(array.max()), 1.0)
        return np.clip(array * (255.0 / peak), 0, 255).astype(np.uint8)
    return np.asarray(image.convert("L"))


class SectionTiles:
    """One section's tileset at one downsample: ``region`` stitches any pixel window, ``protocol_tile`` cuts the 1024 grid."""

    def __init__(self, volume: Volume, tileset: Tileset, downsample: int, cache: TileCache) -> None:
        if downsample not in tileset.levels:
            raise TileSourceError(
                f"{tileset.name} of section {tileset.section} has no level {downsample}; has {sorted(tileset.levels)}"
            )
        if tileset.tile_x <= 0 or tileset.tile_y <= 0:
            raise TileSourceError(f"{tileset.name} has no TileXDim/TileYDim")
        self.volume = volume
        self.tileset = tileset
        self.downsample = downsample
        self.cache = cache

    def region(self, x0: int, y0: int, width: int, height: int) -> np.ndarray:
        """Pixels ``[x0, x0+width) x [y0, y0+height)`` of the stitched level, stored orientation (row 0 = lowest y)."""
        tx, ty = self.tileset.tile_x, self.tileset.tile_y
        canvas = np.zeros((height, width), dtype=np.uint8)
        # A cold region needs a dozen or more tiles; fetched one by one over a slow link they can outlast the
        # segmentation server's wait for our answer, so they are fetched together (then served from the cache below).
        self.cache.prefetch([
            self.tileset.tile_url(self.volume.base_url, self.downsample, ix, iy)
            for iy in range(max(y0 // ty, 0), (y0 + height - 1) // ty + 1)
            for ix in range(max(x0 // tx, 0), (x0 + width - 1) // tx + 1)
        ])
        for iy in range(y0 // ty, (y0 + height - 1) // ty + 1):
            for ix in range(x0 // tx, (x0 + width - 1) // tx + 1):
                if ix < 0 or iy < 0:
                    continue
                tile = self.cache.fetch(self.tileset.tile_url(self.volume.base_url, self.downsample, ix, iy))
                if tile is None:
                    continue
                th, tw = tile.shape[:2]
                ax, ay = max(x0, ix * tx), max(y0, iy * ty)
                bx, by = min(x0 + width, ix * tx + tw), min(y0 + height, iy * ty + th)
                if ax >= bx or ay >= by:
                    continue
                canvas[ay - y0:by - y0, ax - x0:bx - x0] = tile[ay - iy * ty:by - iy * ty, ax - ix * tx:bx - ix * tx]
        return canvas

    def protocol_tile(self, row: int, col: int) -> Optional[bytes]:
        """PNG bytes of the segmentation server's tile (row, col): the 1024 region, flipped for the protocol."""
        region = self.region(col * PROTOCOL_TILE, row * PROTOCOL_TILE, PROTOCOL_TILE, PROTOCOL_TILE)
        if not region.any():
            return None
        out = io.BytesIO()
        Image.fromarray(flip_for_protocol(region), mode="L").save(out, format="PNG")
        return out.getvalue()

    def crop(self, x0: int, y0: int, width: int, height: int) -> np.ndarray:
        """A window in the gallery card's orientation: stored rows, the same as nornir's training crops (no flip)."""
        return self.region(x0, y0, width, height)


def flip_for_protocol(stored: np.ndarray) -> np.ndarray:
    """Stored rows (lowest y first) to protocol rows (highest y first), and back: the flip is its own inverse."""
    return np.ascontiguousarray(stored[::-1])
