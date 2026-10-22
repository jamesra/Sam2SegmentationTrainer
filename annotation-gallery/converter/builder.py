"""Build converter cards: for each location of a structure, a crop, its current outline, and the segmentation server's
proposed polygon, laid out as a normal gallery crops folder so the grid and viewer work on it unchanged.

One location at a time and resumable: a card that is already ``built`` for the same source shape and model is left alone,
a ``failed`` one is retried on request. A card is committed to the catalog last, after its files are on disk.

Coordinates: mosaic pixels (the OData ``MosaicShape``) divided by the downsample D give the protocol's X and Y (no
negation; the protocol's *images* are flipped, see tilesource.py). A card's crop is a 1024 window of the stored tileset
level, origin ``(ox, oy)`` in level-D pixels, so crop pixel = ``mosaic / D - origin`` in both axes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import catalog
from converter import contour as contour_module
from converter import odata, seed
from converter.seg_client import (
    Prompt,
    SegmentationClient,
    SegmentationFailed,
    TileIdentity,
)
from converter.tilesource import SectionTiles, TileCache, Volume

LOGGER = logging.getLogger("converter.builder")

CARD_SIZE = 1024
MARGIN = 48
SCHEMA = """
CREATE TABLE IF NOT EXISTS converter_cards (
    location_id INTEGER NOT NULL,
    image_key TEXT NOT NULL,
    state TEXT NOT NULL,
    reason TEXT,
    iou REAL,
    source_hash TEXT,
    model_id TEXT,
    z INTEGER,
    downsample INTEGER,
    origin_x INTEGER,
    origin_y INTEGER,
    original_wkt TEXT,
    candidate_wkt TEXT,
    type_code INTEGER,
    built_at TEXT,
    PRIMARY KEY (location_id, image_key)
)
"""
BUILT, FAILED = "built", "failed"


@dataclass
class BuildParams:
    volume: str
    odata_base: str
    structure_id: int
    type_codes: Sequence[int] = odata.DEFAULT_TYPE_CODES
    z_min: Optional[int] = None
    z_max: Optional[int] = None
    channel: Optional[str] = None
    filter_name: Optional[str] = None
    model_id: Optional[str] = None
    threshold: float = 0.5
    retry_failed: bool = False
    rebuild: bool = False


@dataclass
class BuildSummary:
    total: int = 0
    built: int = 0
    skipped: int = 0
    failed: int = 0
    cancelled: bool = False
    reasons: Dict[str, int] = field(default_factory=dict)

    def note_failure(self, reason: str) -> None:
        self.failed += 1
        key = reason.split(":", 1)[0]
        self.reasons[key] = self.reasons.get(key, 0) + 1


def source_hash(record: odata.LocationRecord) -> str:
    """Identifies the shape a card was built from, so a changed annotation is noticed."""
    text = f"{record.type_code}|{record.radius}|{record.wkt}"
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def image_key(volume: str, record: odata.LocationRecord, downsample: int) -> str:
    return f"{volume}_{record.z}_D{downsample}_L{record.id}"


def choose_downsample(record: odata.LocationRecord, levels: Sequence[int]) -> int:
    """The finest level at which the shape plus a margin fits one card; raises ``SegmentationFailed`` when none does."""
    min_x, min_y, max_x, max_y = odata.bounds(record)
    extent = max(max_x - min_x, max_y - min_y)
    for level in sorted(levels):
        if extent / level + 2 * MARGIN <= CARD_SIZE:
            return level
    raise SegmentationFailed(f"too_large: the shape is {extent:.0f} mosaic pixels across and no tileset level fits it in {CARD_SIZE}")


def card_origin(record: odata.LocationRecord, downsample: int) -> Tuple[int, int]:
    min_x, min_y, max_x, max_y = odata.bounds(record)
    cx, cy = (min_x + max_x) / 2 / downsample, (min_y + max_y) / 2 / downsample
    return int(math.floor(cx - CARD_SIZE / 2)), int(math.floor(cy - CARD_SIZE / 2))


def _scaled(rings: Sequence[Sequence[Tuple[float, float]]], downsample: int) -> List[List[Tuple[float, float]]]:
    return [[(x / downsample, y / downsample) for x, y in ring] for ring in rings]


def rings_for_prompt(record: odata.LocationRecord, downsample: int) -> Tuple[List[Tuple[float, float]], List[List[Tuple[float, float]]]]:
    """The exterior and holes of the polygon to seed with: the largest part of a multi-part shape."""
    best = max(record.polygons, key=lambda rings: abs(contour_module.signed_area(np.array(rings[0], dtype=float))))
    scaled = _scaled(best, downsample)
    return scaled[0], scaled[1:]


def build_prompt(record: odata.LocationRecord, downsample: int) -> Prompt:
    if record.is_circle:
        cx, cy = record.center
        return seed.circle_prompt(cx / downsample, cy / downsample, record.radius / downsample)
    exterior, holes = rings_for_prompt(record, downsample)
    return seed.polygon_prompt(exterior, holes)


def draw_mask(
    rings: Sequence[Sequence[Tuple[float, float]]], origin: Tuple[int, int], size: int = CARD_SIZE
) -> np.ndarray:
    """Even-odd fill of protocol-space rings into a card's pixels (stored orientation): exterior first, then holes.

    A pixel is set when its centre is inside, so a ring on pixel edges fills exactly its area (see gold/raster.py).
    """
    from gold.raster import fill_rings

    shifted = [[(x - origin[0], y - origin[1]) for x, y in ring] for ring in rings]
    return fill_rings(shifted, size, size)


def circle_mask(cx: float, cy: float, radius: float, origin: Tuple[int, int], size: int = CARD_SIZE) -> np.ndarray:
    ys, xs = np.mgrid[0:size, 0:size]
    return (xs + origin[0] + 0.5 - cx) ** 2 + (ys + origin[1] + 0.5 - cy) ** 2 <= radius * radius


def original_mask(record: odata.LocationRecord, downsample: int, origin: Tuple[int, int]) -> np.ndarray:
    if record.is_circle:
        return circle_mask(record.center[0] / downsample, record.center[1] / downsample, record.radius / downsample, origin)
    mask = np.zeros((CARD_SIZE, CARD_SIZE), dtype=bool)
    for rings in record.polygons:
        mask |= draw_mask(_scaled(rings, downsample), origin)
    return mask


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int(np.logical_or(a, b).sum())
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def _write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    Image.fromarray(array.astype(np.uint8), mode="L").save(temporary, format="PNG")
    os.replace(temporary, path)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(temporary, path)


def candidate_polygon_wkt(shape: contour_module.Contour, downsample: int) -> str:
    """The proposed outline as plain POLYGON control points in mosaic pixels (Viking curves it after load)."""
    return contour_module.polygon_wkt(shape, downsample=downsample, precision=2)


class Builder:
    def __init__(
        self,
        params: BuildParams,
        crops: Path,
        volume: Volume,
        client: SegmentationClient,
        cache: TileCache,
        *,
        get_json: odata.GetJson = odata.http_get_json,
        semaphore: Optional[threading.Semaphore] = None,
    ) -> None:
        self.params = params
        self.crops = Path(crops)
        self.volume = volume
        self.client = client
        self.cache = cache
        self.get_json = get_json
        self.semaphore = semaphore
        self._sections: Dict[Tuple[int, int], SectionTiles] = {}

    # -- catalog -------------------------------------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        connection = catalog.connect(self.crops)
        connection.execute(SCHEMA)
        return connection

    def _existing(self, connection: sqlite3.Connection, location_id: int) -> Optional[sqlite3.Row]:
        return connection.execute(
            "SELECT * FROM converter_cards WHERE location_id = ? ORDER BY built_at DESC LIMIT 1", [location_id]
        ).fetchone()

    def _commit_card(self, record: odata.LocationRecord, row: Dict[str, Any], area: int) -> None:
        connection = self._connect()
        try:
            key = row["image_key"]
            connection.execute("DELETE FROM converter_cards WHERE location_id = ?", [record.id])
            connection.execute(
                "INSERT INTO converter_cards (location_id, image_key, state, reason, iou, source_hash, model_id, z, downsample, "
                "origin_x, origin_y, original_wkt, candidate_wkt, type_code, built_at) "
                "VALUES (:location_id, :image_key, :state, :reason, :iou, :source_hash, :model_id, :z, :downsample, "
                ":origin_x, :origin_y, :original_wkt, :candidate_wkt, :type_code, :built_at)",
                row,
            )
            if row["state"] == BUILT:
                connection.execute("DELETE FROM locations WHERE location_id = ?", [record.id])
                radius = record.radius if record.radius is not None else (math.sqrt(area / math.pi) if area > 0 else None)
                connection.execute(
                    "INSERT INTO locations (location_id, z, structure_id, structure_label, type_id, type_name, radius, "
                    "image_key, image_relpath, mask_relpath, ignored) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
                    [
                        record.id, record.z, record.structure_id, record.structure_label, None, record.type_name, radius,
                        key, f"images/{key}.png", f"masks/{key}_{record.id}.png",
                    ],
                )
            connection.commit()
        finally:
            connection.close()

    # -- one location --------------------------------------------------------------------------------------------

    def _section(self, z: int, downsample: int) -> SectionTiles:
        key = (z, downsample)
        if key not in self._sections:
            tileset = self.volume.select(z, self.params.channel, self.params.filter_name)
            self._sections[key] = SectionTiles(self.volume, tileset, downsample, self.cache)
        return self._sections[key]

    def _levels(self, z: int) -> List[int]:
        return sorted(self.volume.select(z, self.params.channel, self.params.filter_name).levels)

    def build_one(self, record: odata.LocationRecord) -> Dict[str, Any]:
        """Build one card and commit it. Returns its row; raises ``SegmentationFailed`` for a recorded failure."""
        downsample = choose_downsample(record, self._levels(record.z))
        section = self._section(record.z, downsample)
        origin = card_origin(record, downsample)
        key = image_key(self.params.volume, record, downsample)
        prompt = build_prompt(record, downsample)

        identity = TileIdentity(
            volume=self.params.volume, section=record.z, channel=section.tileset.channel,
            transform=section.tileset.name or "mosaic", downsample=downsample,
        )
        if self.semaphore is not None:
            self.semaphore.acquire()
        try:
            result = self.client.segment_with_retry(
                identity, prompt, section.protocol_tile, self._no_neighbours,
                threshold=self.params.threshold, request_id=record.id & 0xFFFFFFFF,
            )
        finally:
            if self.semaphore is not None:
                self.semaphore.release()
        shape = contour_module.contour_from_result(result)
        if shape is None:
            raise SegmentationFailed("no_object: the mask is empty at the threshold")

        original = original_mask(record, downsample, origin)
        candidate = draw_mask([shape.exterior] + list(shape.holes), origin)
        if not candidate.any():
            raise SegmentationFailed("no_object: the proposed polygon is outside the card")
        ratio = iou(original, candidate)

        image = section.crop(origin[0], origin[1], CARD_SIZE, CARD_SIZE)
        _write_png(self.crops / "images" / f"{key}.png", image)
        _write_png(self.crops / "masks" / f"{key}_{record.id}.png", candidate * 255)
        _write_png(self.crops / "original" / f"{key}_{record.id}.png", original * 255)
        _write_json(
            self.crops / "images" / f"{key}.json",
            {
                "image": {"file_name": f"{key}.png", "width": CARD_SIZE, "height": CARD_SIZE},
                "annotations": [{"id": record.id}],
                "converter": {
                    "volume": self.params.volume, "z": record.z, "downsample": downsample,
                    "originX": origin[0], "originY": origin[1], "structureId": record.structure_id,
                    "tileset": section.tileset.name,
                },
            },
        )
        row = {
            "location_id": record.id, "image_key": key, "state": BUILT, "reason": None, "iou": ratio,
            "source_hash": source_hash(record), "model_id": self.params.model_id, "z": record.z, "downsample": downsample,
            "origin_x": origin[0], "origin_y": origin[1], "original_wkt": record.wkt,
            "candidate_wkt": candidate_polygon_wkt(shape, downsample), "type_code": record.type_code,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self._commit_card(record, row, int(candidate.sum()))
        return row

    @staticmethod
    def _no_neighbours(row: int, col: int) -> np.ndarray:
        """Neighbouring annotations are not asked for yet: every tile is answered as having none."""
        return np.zeros((1024, 1024), dtype=bool)

    def _record_failure(self, record: odata.LocationRecord, reason: str) -> None:
        row = {
            "location_id": record.id, "image_key": f"failed_{record.id}", "state": FAILED, "reason": reason, "iou": None,
            "source_hash": source_hash(record), "model_id": self.params.model_id, "z": record.z, "downsample": None,
            "origin_x": None, "origin_y": None, "original_wkt": record.wkt, "candidate_wkt": None,
            "type_code": record.type_code, "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self._commit_card(record, row, 0)

    # -- the structure -------------------------------------------------------------------------------------------

    def should_build(self, record: odata.LocationRecord) -> bool:
        connection = self._connect()
        try:
            existing = self._existing(connection, record.id)
        finally:
            connection.close()
        if existing is None or self.params.rebuild:
            return True
        if existing["source_hash"] != source_hash(record) or existing["model_id"] != self.params.model_id:
            return True
        if existing["state"] == FAILED:
            return self.params.retry_failed
        return False

    def run(
        self,
        progress: Optional[Callable[[BuildSummary, Optional[odata.LocationRecord]], None]] = None,
        cancel: Optional[threading.Event] = None,
    ) -> BuildSummary:
        summary = BuildSummary()
        filters = dict(type_codes=self.params.type_codes, z_min=self.params.z_min, z_max=self.params.z_max)
        try:
            summary.total = odata.count_locations(self.params.odata_base, self.params.structure_id, get_json=self.get_json, **filters)
        except (ValueError, OSError):
            summary.total = 0
        for record in odata.iter_locations(self.params.odata_base, self.params.structure_id, get_json=self.get_json, **filters):
            if cancel is not None and cancel.is_set():
                summary.cancelled = True
                break
            if progress:
                progress(summary, record)
            if not self.should_build(record):
                summary.skipped += 1
                continue
            try:
                self.build_one(record)
                summary.built += 1
            except (SegmentationFailed, seed.NotSeedable) as failure:
                reason = str(failure) if "_" in str(failure).split(":")[0] else f"failed: {failure}"
                self._record_failure(record, reason)
                summary.note_failure(reason)
            except Exception as error:  # a bad location must not stop the structure
                LOGGER.exception("location %s failed", record.id)
                self._record_failure(record, f"error: {error}")
                summary.note_failure("error")
        if progress:
            progress(summary, None)
        return summary
