"""The converter's side of the web server: where builds live, what a card says, and the batch file.

The server treats each build as a *training set* named ``structure-<id>``: a folder ``<GALLERY_CONVERTER_ROOT>/structure-<id>``
holding one crops folder per volume (``RC1/`` and so on, the same layout the gallery uses). Every route that already takes
``?set=`` (catalog, files, approve, reject) therefore works on it unchanged. This module adds only what is converter-specific.
"""

from __future__ import annotations

import json
import re
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
    import os

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
