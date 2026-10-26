"""The converter's side of the web server: where builds live, what a card says, and the batch file.

The server treats each build as a *training set* named ``structure-<id>``: a folder ``<GALLERY_CONVERTER_ROOT>/structure-<id>``
holding one crops folder per volume (``RC1/`` and so on, the same layout the gallery uses). Every route that already takes
``?set=`` (catalog, files, approve, reject) therefore works on it unchanged. This module adds only what is converter-specific.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import catalog

SET_NAME = re.compile(r"structure-(\d+)")
BATCH_TYPE_CODE = 6
"""Approved shapes are saved as curve polygons whose WKT holds control points (what Viking's own segmentation save writes);
mosaic space stores plain polygons and Viking curves them after load. If plain polygons (4) are wanted instead, this is the
one constant to change."""


def converter_root(environ: Optional[Dict[str, str]] = None) -> Optional[Path]:
    """The folder builds are kept in, or None when the converter is switched off (``GALLERY_CONVERTER_ROOT`` unset)."""

    mapping = environ if environ is not None else os.environ
    raw = (mapping.get("GALLERY_CONVERTER_ROOT") or "").strip()
    return Path(raw) if raw else None


def structure_set_name(structure_id: int) -> str:
    return f"structure-{int(structure_id)}"


def list_structure_sets(root: Optional[Path]) -> List[str]:
    """Built structures, newest id last. Empty when the converter is off or nothing was built."""
    if root is None or not root.is_dir():
        return []
    names = [child.name for child in root.iterdir() if SET_NAME.fullmatch(child.name) and child.is_dir()]
    return sorted(names, key=lambda name: int(SET_NAME.fullmatch(name).group(1)))


def resolve_structure_set(root: Optional[Path], name: str) -> Path:
    """The set's folder. Raises ``FileNotFoundError`` for anything but an existing ``structure-<id>`` under the root."""
    if root is None or not SET_NAME.fullmatch(name):
        raise FileNotFoundError(name)
    path = root / name
    if not path.is_dir():
        raise FileNotFoundError(name)
    path.resolve().relative_to(root.resolve())
    return path


def delete_build(root: Optional[Path], set_name: str, volume: str) -> None:
    """Remove one build (a structure in one volume): its cards, masks, catalog, review marks and source record. The
    structure's folder goes too once no volume is left in it. The annotations in the database are not touched.

    Raises ``FileNotFoundError`` when there is no such build. Callers must not delete a build that is still running."""
    set_dir = resolve_structure_set(root, set_name)
    crops = set_dir / volume
    if not crops.is_dir():
        raise FileNotFoundError(f"{set_name}/{volume}")
    crops.resolve().relative_to(set_dir.resolve())
    shutil.rmtree(crops)
    if not any(set_dir.iterdir()):
        set_dir.rmdir()


def is_converter_crops(crops: Path) -> bool:
    return (catalog.sqlite_path(crops)).is_file() and _has_cards(crops)


def _has_cards(crops: Path) -> bool:
    connection = sqlite3.connect(str(catalog.sqlite_path(crops)))
    try:
        return connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'converter_cards'"
        ).fetchone() is not None
    finally:
        connection.close()


def card_rows(crops: Path) -> List[Dict[str, Any]]:
    """Everything the page needs about each card beyond the catalog row: state, why it failed, how far the proposal moved."""
    if not catalog.sqlite_path(crops).is_file():
        return []
    connection = catalog.connect(crops)
    try:
        if not _has_cards(crops):
            return []
        rows = connection.execute(
            "SELECT location_id, image_key, state, reason, iou, model_id, z, downsample, type_code, built_at, source_hash "
            "FROM converter_cards ORDER BY location_id"
        ).fetchall()
    finally:
        connection.close()
    return [dict(row) for row in rows]


def link_rows(crops: Path) -> List[List[int]]:
    """The structure's location links as ``[a, b]`` pairs; empty when the build predates link storage."""
    if not catalog.sqlite_path(crops).is_file():
        return []
    connection = catalog.connect(crops)
    try:
        if connection.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'converter_links'").fetchone() is None:
            return []
        return [[int(row[0]), int(row[1])] for row in connection.execute("SELECT a, b FROM converter_links ORDER BY a, b")]
    finally:
        connection.close()


def failed_rows(crops: Path) -> List[Dict[str, Any]]:
    return [row for row in card_rows(crops) if row["state"] == "failed"]


def approved_batch(crops: Path) -> List[Dict[str, Any]]:
    """One row per approved card: the proposed plain polygon in mosaic pixels and what a writer needs to apply it safely.

    ``source_hash`` identifies the shape the card was built from; a writer compares it with the location's current shape
    and skips (or flags) a location someone edited since. Rejected and pending cards are not in the batch.
    """
    approved = catalog.load_approved(crops)
    if not approved:
        return []
    connection = catalog.connect(crops)
    try:
        if not _has_cards(crops):
            return []
        rows = connection.execute(
            "SELECT c.location_id, c.image_key, c.candidate_wkt, c.original_wkt, c.source_hash, c.model_id, c.z, c.downsample, "
            "c.type_code AS source_type_code, c.iou, c.built_at, l.structure_id "
            "FROM converter_cards c LEFT JOIN locations l ON l.location_id = c.location_id AND l.image_key = c.image_key "
            "WHERE c.state = 'built' ORDER BY c.location_id"
        ).fetchall()
    finally:
        connection.close()
    batch = []
    for row in rows:
        if (int(row["location_id"]), str(row["image_key"])) not in approved:
            continue
        batch.append(
            {
                "location_id": int(row["location_id"]),
                "structure_id": row["structure_id"],
                "z": row["z"],
                "type_code": BATCH_TYPE_CODE,
                "shape_space": "mosaic",
                "mosaic_wkt": row["candidate_wkt"],
                "original_wkt": row["original_wkt"],
                "original_type_code": row["source_type_code"],
                "source_hash": row["source_hash"],
                "model_id": row["model_id"],
                "downsample": row["downsample"],
                "iou": row["iou"],
                "built_at": row["built_at"],
            }
        )
    return batch


def batch_jsonl(crops: Path) -> bytes:
    lines = [json.dumps(row, separators=(",", ":")) for row in approved_batch(crops)]
    return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")


MAX_EDIT_VERTICES = 20000
MIN_EDIT_AREA = 4.0


class EditRejected(Exception):
    """The edited outline cannot replace the proposal; the message is for the reviewer."""


def _ring_from(payload: Any) -> List[tuple]:
    if not isinstance(payload, list) or len(payload) < 3:
        raise EditRejected("an outline needs at least three points")
    points = []
    for item in payload:
        try:
            x, y = float(item[0]), float(item[1])
        except (TypeError, ValueError, IndexError):
            raise EditRejected("outline points must be [x, y] numbers") from None
        if x != x or y != y or abs(x) > 1e7 or abs(y) > 1e7:
            raise EditRejected("outline points are out of range")
        points.append((x, y))
    if points[0] != points[-1]:
        points.append(points[0])
    return points


def apply_edit(crops: Path, location_id: int, image_key: str, rings: Any) -> Dict[str, Any]:
    """Replace a card's proposed polygon with a retraced one.

    ``rings`` are in the card's crop pixels (pixel-edge coordinates, the stored orientation): the exterior first, then
    holes. The mask PNG, the candidate WKT (plain POLYGON control points in mosaic pixels) and the overlap with the
    current outline are rewritten, and the card's cached layers dropped. The database is not touched.
    """
    import numpy as np
    from PIL import Image

    from converter import builder as builder_module
    from converter import contour as contour_module
    from refresh import drop_stale_thumbnails

    if not isinstance(rings, list) or not rings or sum(len(r) for r in rings if isinstance(r, list)) > MAX_EDIT_VERTICES:
        raise EditRejected("an outline is required")
    parsed = [_ring_from(ring) for ring in rings]
    connection = catalog.connect(crops)
    try:
        if not _has_cards(crops):
            raise EditRejected("this is not a converter set")
        card = connection.execute(
            "SELECT * FROM converter_cards WHERE location_id = ? AND image_key = ? AND state = 'built'", [location_id, image_key]
        ).fetchone()
        if card is None:
            raise EditRejected("no such card")
        origin = (int(card["origin_x"]), int(card["origin_y"]))
        downsample = int(card["downsample"])
        protocol = [[(x + origin[0], y + origin[1]) for x, y in ring] for ring in parsed]
        mask = builder_module.draw_mask(protocol, origin)
        if int(mask.sum()) < MIN_EDIT_AREA:
            raise EditRejected("the outline is empty")
        original_path = crops / "original" / f"{image_key}_{location_id}.png"
        original = np.asarray(Image.open(original_path)) > 0 if original_path.is_file() else np.zeros_like(mask)
        ratio = builder_module.iou(original, mask)
        shape = contour_module.Contour(
            exterior=np.array(protocol[0], dtype=float), holes=[np.array(ring, dtype=float) for ring in protocol[1:]]
        )
        wkt = builder_module.candidate_polygon_wkt(shape, downsample)
        builder_module._write_png(crops / "masks" / f"{image_key}_{location_id}.png", mask * 255)
        connection.execute(
            "UPDATE converter_cards SET candidate_wkt = ?, iou = ?, reason = 'retraced' WHERE location_id = ? AND image_key = ?",
            [wkt, ratio, location_id, image_key],
        )
        connection.commit()
    finally:
        connection.close()
    drop_stale_thumbnails(crops, [image_key])
    return {"ok": True, "location_id": location_id, "iou": ratio, "candidate_wkt": wkt}


def build_summary(crops: Path) -> Dict[str, Any]:
    """Counts for the build picker: cards built and failed, and how many of the built ones are approved or rejected."""
    rows = card_rows(crops)
    built = [row for row in rows if row["state"] == "built"]
    approved = catalog.load_approved(crops)
    rejected = {int(row["location_id"]) for row in catalog.list_catalog_rows(crops) if row.get("ignored")}
    built_ids = {int(row["location_id"]) for row in built}
    stamps = [row["built_at"] for row in rows if row.get("built_at")]
    return {
        "built": len(built),
        "failed": len(rows) - len(built),
        "approved": len({location for location, _ in approved if location in built_ids}),
        "rejected": len(rejected & built_ids),
        "updated": max(stamps) if stamps else None,
    }
