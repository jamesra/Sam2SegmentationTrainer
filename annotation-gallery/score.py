"""Score one location with the current segmentation model, for the gallery's refresh button.

After a refresh redraws a location's masks from OData, the gallery asks the segmentation server (the converter's
``GALLERY_SEGMENTATION_TARGET``) to segment the location from a seed circle and stores how well the result matches
the annotation in the catalog's SAM2 columns, which the card shows as its "SAM2" badge:

- ``sam2GtIou``: IoU of the model's mask and the annotation (the mask that was just redrawn).
- ``sam2PredIou``: the server's own quality estimate of its mask.
- ``sam2Checkpoint``: the model that scored it (``ModelInfo.model_version``, e.g. ``v5.4``), followed by
  `` at D<n>`` when the object had to be scored at a coarser level than its crops (see below).
- ``sam2ScoredAt``: when (UTC).

The prompt is a seed circle, never the annotation itself, so the score says whether the model finds this object as
annotated, not whether it would leave a given mask alone: a circle annotation is sent as that circle; a polygon as
its largest inscribed circle at ``SEED_CIRCLE_SCALE`` of the radius (the trainer's ``prompt.circle_diameter_scale``
for seed circles). This is the trainer's ``circle_mask`` validation family and the way Viking seeds a circle.

The object is scored whole, on one card, as the converter builds its cards (:mod:`converter.builder`): at the finest
level, no finer than the level its crops were exported at, where the annotation and the model's mask both fit in
1024 pixels. Most annotations fit at their own level, the level the model trained on. A large cell (one that spans
many crop windows) does not, and is scored at a coarser level than the model ever saw it at; its score is then
lower than its training-time numbers, and the level is recorded so the card can say so. Scoring it at full
resolution instead takes tens of seconds and many tiles per click.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

import catalog
from circle_update import max_inscribed_circle
from converter import builder as builder_module
from converter import contour as contour_module
from converter import odata, seed
from converter.seg_client import SegmentationFailed, TileIdentity
from converter.tilesource import SectionTiles, TileCache, Volume

SEED_CIRCLE_SCALE = 0.95
"""Radius fraction of an inscribed circle sent as a seed (the trainer's ``prompt.circle_diameter_scale``)."""

THRESHOLD = 0.5


@dataclass(frozen=True)
class LocationScore:
    location_id: int
    gt_iou: float
    pred_iou: Optional[float]
    model: Optional[str]
    scored_at: str
    prompt: str
    downsample: int
    native_downsample: Optional[int] = None

    @property
    def coarser_than_native(self) -> bool:
        return self.native_downsample is not None and self.downsample > self.native_downsample

    @property
    def checkpoint_label(self) -> Optional[str]:
        """What ``sam2Checkpoint`` stores: the model, plus the level when it is coarser than the crops'."""
        if not self.coarser_than_native:
            return self.model
        return f"{self.model or 'unknown model'} at D{self.downsample}"

    def payload(self) -> Dict[str, Any]:
        """The card's fields (the catalog's column names) plus how it was scored."""
        return {
            "sam2GtIou": self.gt_iou,
            "sam2PredIou": self.pred_iou,
            "sam2Checkpoint": self.checkpoint_label,
            "sam2ScoredAt": self.scored_at,
            "prompt": self.prompt,
            "downsample": self.downsample,
            "nativeDownsample": self.native_downsample,
        }


def seed_circle(record: odata.LocationRecord) -> Tuple[float, float, float, str]:
    """``(cx, cy, r, kind)`` in mosaic pixels: the circle itself, or the polygon's inscribed circle scaled down."""
    if record.is_circle:
        cx, cy = record.center
        return float(cx), float(cy), float(record.radius), "circle"
    found = max_inscribed_circle(record.polygons)
    if found is None:
        raise seed.NotSeedable("no inscribed circle (a degenerate polygon, or scipy is not installed)")
    cx, cy, radius = found
    return float(cx), float(cy), float(radius) * SEED_CIRCLE_SCALE, "inscribed circle"


def _no_neighbours(cells: Sequence[Tuple[int, int]]) -> List[Any]:
    """No neighbouring annotations: every cell the server asks about is answered with no background shapes."""
    return []


def _segment_once(
    record: odata.LocationRecord, volume_name: str, volume: Volume, tileset: Any, cache: TileCache, client: Any,
    downsample: int, cx: float, cy: float, radius: float, *, semaphore: Optional[threading.Semaphore], attempts: int,
) -> Tuple[Any, Any]:
    """One request at one level: ``(result, contour or None)``."""
    section = SectionTiles(volume, tileset, downsample, cache)
    identity = TileIdentity(
        volume=volume_name, section=record.z, channel=tileset.channel,
        transform=tileset.name or "mosaic", downsample=downsample,
    )
    prompt = seed.circle_prompt(cx / downsample, cy / downsample, radius / downsample)
    if semaphore is not None:
        semaphore.acquire()
    try:
        result = client.segment_with_retry(
            identity, prompt, section.protocol_tile, _no_neighbours,
            threshold=THRESHOLD, request_id=record.id & 0xFFFFFFFF, attempts=attempts,
        )
    finally:
        if semaphore is not None:
            semaphore.release()
    return result, contour_module.contour_from_result(result)


def score_record(
    record: odata.LocationRecord,
    volume_name: str,
    volume: Volume,
    client: Any,
    cache: TileCache,
    *,
    model: Optional[str] = None,
    semaphore: Optional[threading.Semaphore] = None,
    attempts: int = 1,
    native_downsample: Optional[int] = None,
) -> LocationScore:
    """Segment ``record`` from its seed circle and compare the result with the annotation, both whole on one card.

    ``native_downsample`` (the level the location's crops were exported at) is the finest level used, and is recorded
    so a score taken at a coarser level can be told apart.

    Raises ``SegmentationFailed`` (server refused or unreachable) or ``seed.NotSeedable`` (nothing to seed with).
    An empty mask is a score of 0, not an error: the model did not find the object.
    """
    cx, cy, radius, kind = seed_circle(record)
    tileset = volume.select(record.z)
    levels = sorted(tileset.levels)
    record_bounds = odata.bounds(record)
    finest = native_downsample if native_downsample is not None and native_downsample in levels else 1
    downsample = builder_module.choose_downsample(record_bounds, levels, at_least=finest)
    while True:
        result, shape = _segment_once(
            record, volume_name, volume, tileset, cache, client, downsample, cx, cy, radius,
            semaphore=semaphore, attempts=attempts,
        )
        if shape is None:
            bounds = record_bounds
            break
        # As the converter does: the card must hold the model's mask as well as the annotation.
        bounds = builder_module.union_bounds(record_bounds, builder_module.shape_bounds(shape, downsample))
        try:
            fitted = builder_module.choose_downsample(bounds, levels, at_least=downsample)
        except SegmentationFailed:
            bounds, fitted = record_bounds, downsample
        if fitted == downsample:
            break
        downsample = fitted
    origin = builder_module.card_origin(bounds, downsample)
    original = builder_module.original_mask(record, downsample, origin)
    if shape is None:
        candidate = np.zeros_like(original)
    else:
        candidate = builder_module.draw_mask([shape.exterior] + list(shape.holes), origin)
    pred = getattr(result, "score", None)
    return LocationScore(
        location_id=record.id,
        gt_iou=builder_module.iou(original, candidate),
        pred_iou=None if pred is None else float(pred),
        model=model,
        scored_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        prompt=kind,
        downsample=downsample,
        native_downsample=native_downsample,
    )


_DOWNSAMPLE = re.compile(r"_D(\d+)_")


def native_downsample(crops: str | Path, location_id: int) -> Optional[int]:
    """The level the location's crops were exported at (``_D<n>_`` in its image key), or None when not known."""
    try:
        connection = catalog.connect(crops)
    except Exception:  # noqa: BLE001
        return None
    try:
        row = connection.execute("SELECT image_key FROM locations WHERE location_id = ?", [location_id]).fetchone()
    except Exception:  # noqa: BLE001
        return None
    finally:
        connection.close()
    found = _DOWNSAMPLE.search(str(row[0])) if row is not None and row[0] else None
    return int(found.group(1)) if found else None


def model_version(client: Any) -> Optional[str]:
    """The served model's version (``v5.4``), or its id, or None when the server does not say."""
    try:
        info = client.status().model
    except Exception:  # noqa: BLE001 - a status failure must not lose the score itself
        return None
    return (getattr(info, "model_version", "") or getattr(info, "model_id", "") or None) if info is not None else None


def write_score(crops: str | Path, score: LocationScore) -> int:
    """Store the score on the location's catalog row(s). Returns the rows updated."""
    connection = catalog.connect(crops)
    try:
        cursor = connection.execute(
            "UPDATE locations SET sam2GtIou = ?, sam2PredIou = ?, sam2Checkpoint = ?, sam2ScoredAt = ? "
            "WHERE location_id = ?",
            [score.gt_iou, score.pred_iou, score.checkpoint_label, score.scored_at, score.location_id],
        )
        connection.commit()
        return int(cursor.rowcount)
    finally:
        connection.close()


_VOLUMES: Dict[str, Volume] = {}
_VOLUMES_LOCK = threading.Lock()


def cached_volume(key: str, load: Callable[[], Volume]) -> Volume:
    """A volume's tilesets, read once per gallery process (the VikingXML is large and rarely changes)."""
    with _VOLUMES_LOCK:
        found = _VOLUMES.get(key)
    if found is not None:
        return found
    loaded = load()
    with _VOLUMES_LOCK:
        _VOLUMES.setdefault(key, loaded)
        return _VOLUMES[key]


def score_after_refresh(
    crops: str | Path,
    volume_name: str,
    entity: Dict[str, Any],
    *,
    make_client: Callable[[], Any],
    load_volume: Callable[[], Volume],
    volume_key: str,
    make_cache: Callable[[], TileCache],
    semaphore: Optional[threading.Semaphore] = None,
) -> Dict[str, Any]:
    """Score and store one refreshed location. Never raises: a failure comes back as ``{"error": ...}``."""
    try:
        record = odata.record_from_entity(entity)
    except Exception as exc:  # noqa: BLE001 - an entity the converter cannot read
        return {"error": f"the location's shape could not be read: {exc}"}
    if record is None:
        return {"error": "the location has no shape the model can be seeded from"}
    try:
        client = make_client()
    except Exception as exc:  # noqa: BLE001 - not configured, bad CA file, ...
        return {"error": f"segmentation server not available: {exc}"}
    try:
        volume = cached_volume(volume_key, load_volume)
        score = score_record(
            record, volume_name, volume, client, make_cache(), model=model_version(client), semaphore=semaphore,
            native_downsample=native_downsample(crops, record.id),
        )
        rows = write_score(crops, score)
        return {**score.payload(), "rows": rows}
    except (SegmentationFailed, seed.NotSeedable) as exc:
        return {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - tiles, VikingXML or catalog: report, the refresh itself succeeded
        return {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


__all__ = [
    "LocationScore", "SEED_CIRCLE_SCALE", "cached_volume", "model_version", "native_downsample",
    "score_after_refresh", "score_record", "seed_circle", "write_score",
]
