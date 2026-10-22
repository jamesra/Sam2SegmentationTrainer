"""Gold sets on disk: ``<GALLERY_GOLD_ROOT>/gold-<name>/<volume>/`` is a crops folder like any other (images/, masks/, the
catalog, approved.json) plus ``source_masks/`` (the masks as drawn from the source set, never edited) and a ``gold_cards``
table; ``<root>/gold-<name>/gold.json`` records how the sample was drawn so it can be audited and re-created.

Nothing here touches the annotation database or the source set: cards are copied.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

import catalog
from gold import draw

SET_NAME = re.compile(r"gold-([A-Za-z0-9][A-Za-z0-9_.-]{0,39})")
META = "gold.json"
SCHEMA = """
CREATE TABLE IF NOT EXISTS gold_cards (
    location_id INTEGER PRIMARY KEY,
    image_key TEXT NOT NULL,
    edited INTEGER NOT NULL DEFAULT 0,
    changed_pixels INTEGER NOT NULL DEFAULT 0,
    mask_sha1 TEXT,
    source_mask_sha1 TEXT,
    approved_by TEXT,
    approved_at TEXT,
    approved_sha1 TEXT
)
"""


class GoldError(Exception):
    """A request the gold set cannot honour; the message is for the user."""


def gold_root(environ: Optional[Dict[str, str]] = None) -> Optional[Path]:
    mapping = environ if environ is not None else os.environ
    raw = (mapping.get("GALLERY_GOLD_ROOT") or "").strip()
    return Path(raw) if raw else None


def set_name(name: str) -> str:
    """``gold-<name>`` for a bare name; validated."""
    full = name if name.startswith("gold-") else f"gold-{name}"
    if not SET_NAME.fullmatch(full):
        raise GoldError("a gold set name uses letters, digits, '.', '_' or '-' (up to 40)")
    return full


def list_sets(root: Optional[Path]) -> List[str]:
    if root is None or not root.is_dir():
        return []
    return sorted(child.name for child in root.iterdir() if SET_NAME.fullmatch(child.name) and child.is_dir())


def resolve_set(root: Optional[Path], name: str) -> Path:
    if root is None or not SET_NAME.fullmatch(name):
        raise FileNotFoundError(name)
    path = root / name
    if not path.is_dir():
        raise FileNotFoundError(name)
    path.resolve().relative_to(root.resolve())
    return path


def volume_crops(set_dir: Path, volume: str) -> Path:
    return set_dir / volume


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_meta(set_dir: Path) -> Dict[str, Any]:
    try:
        payload = json.loads((set_dir / META).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def write_meta(set_dir: Path, meta: Dict[str, Any]) -> None:
    set_dir.mkdir(parents=True, exist_ok=True)
    catalog._write_text_atomic(set_dir / META, json.dumps(meta, indent=1))


def sha1_of(path: Path) -> Optional[str]:
    try:
        return hashlib.sha1(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _connect(crops: Path) -> sqlite3.Connection:
    connection = catalog.connect(crops)
    connection.execute(SCHEMA)
    return connection


# ---- copying cards --------------------------------------------------------------------------------------------------

def copy_card(source_crops: Path, crops: Path, row: Dict[str, Any]) -> Dict[str, Any]:
    """Copy one location's image and mask from the source set into a gold volume folder and catalog it."""
    location_id = int(row["location_id"])
    key = str(row["image_key"])
    image_rel = str(row.get("image_relpath") or f"images/{key}.png")
    mask_source = source_crops / str(row["mask_relpath"])
    image_source = source_crops / image_rel
    if not image_source.is_file() or not mask_source.is_file():
        raise GoldError(f"location {location_id}: its image or mask file is missing in the source set")
    image_target = crops / image_rel
    mask_rel = f"masks/{key}_{location_id}.png"
    for target, source in ((image_target, image_source), (crops / mask_rel, mask_source), (crops / "source_masks" / f"{key}_{location_id}.png", mask_source)):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    digest = sha1_of(mask_source)
    connection = _connect(crops)
    try:
        connection.execute("DELETE FROM locations WHERE location_id = ?", [location_id])
        connection.execute("DELETE FROM gold_cards WHERE location_id = ?", [location_id])
        connection.execute(
            "INSERT INTO locations (location_id, z, structure_id, structure_label, type_id, type_name, radius, image_key, "
            "image_relpath, mask_relpath, ignored) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
            [
                location_id, row.get("z"), row.get("structure_id"), row.get("structure_label"), row.get("type_id"),
                row.get("type_name"), row.get("radius"), key, image_rel, mask_rel,
            ],
        )
        connection.execute(
            "INSERT INTO gold_cards (location_id, image_key, mask_sha1, source_mask_sha1) VALUES (?, ?, ?, ?)",
            [location_id, key, digest, digest],
        )
        connection.commit()
    finally:
        connection.close()
    return {"location_id": location_id, "image_key": key}


def remove_card(crops: Path, location_id: int) -> None:
    connection = _connect(crops)
    try:
        row = connection.execute("SELECT image_key FROM gold_cards WHERE location_id = ?", [location_id]).fetchone()
        connection.execute("DELETE FROM locations WHERE location_id = ?", [location_id])
        connection.execute("DELETE FROM gold_cards WHERE location_id = ?", [location_id])
        connection.commit()
    finally:
        connection.close()
    if row is not None:
        key = row["image_key"]
        for folder in ("masks", "source_masks"):
            (crops / folder / f"{key}_{location_id}.png").unlink(missing_ok=True)
        catalog.unapprove_window(crops, location_id, key)


def cards(crops: Path) -> List[Dict[str, Any]]:
    if not catalog.sqlite_path(crops).is_file():
        return []
    connection = _connect(crops)
    try:
        return [dict(row) for row in connection.execute("SELECT * FROM gold_cards ORDER BY location_id")]
    finally:
        connection.close()


# ---- the set ----------------------------------------------------------------------------------------------------

def _source_pool(source_crops: Path) -> draw.Pool:
    return draw.build_pool(catalog.list_catalog_rows(source_crops))


def _mask_pixels(source_crops: Path):
    def count(row: Dict[str, Any]) -> int:
        try:
            with Image.open(source_crops / str(row["mask_relpath"])) as image:
                return int(np.count_nonzero(np.asarray(image.convert("L")) > 127))
        except OSError:
            return 0

    return count


def window_for(pool: draw.Pool, source_crops: Path, seed: int, location_id: int) -> Dict[str, Any]:
    return draw.choose_window(seed, location_id, pool.windows[location_id], _mask_pixels(source_crops))


def create_set(
    root: Path,
    name: str,
    source_set: str,
    sources: Dict[str, Path],
    *,
    n: int = draw.DEFAULT_COUNT,
    seed: int,
    stratify: str = "",
    band: int = draw.DEFAULT_Z_BAND,
) -> Dict[str, Any]:
    """Draw ``n`` locations per volume from the source set's catalogs and copy them into a new gold set.

    ``sources`` maps volume name to its source crops folder. Refuses a name that exists (a gold set is never overwritten).
    """
    full = set_name(name)
    set_dir = root / full
    if set_dir.exists():
        raise GoldError(f"{full} already exists")
    if not sources:
        raise GoldError("no volume to draw from")
    if not 1 <= int(n) <= 5000:
        raise GoldError("n must be between 1 and 5000")
    meta: Dict[str, Any] = {
        "name": full, "sourceSet": source_set, "seed": int(seed), "n": int(n), "stratify": stratify, "band": int(band),
        "drawnAt": now(), "volumes": {},
    }
    try:
        for volume, source_crops in sorted(sources.items()):
            pool = _source_pool(source_crops)
            ordered = draw.order(pool, seed, stratify, band)
            chosen = ordered[: int(n)]
            crops = volume_crops(set_dir, volume)
            crops.mkdir(parents=True, exist_ok=True)
            for location_id in chosen:
                copy_card(source_crops, crops, window_for(pool, source_crops, seed, location_id))
            meta["volumes"][volume] = {
                "fingerprint": draw.fingerprint(pool.ids), "poolSize": len(pool.ids), "drawn": len(chosen),
                "short": max(0, int(n) - len(chosen)), "multiWindow": len(pool.multi), "rejectedExcluded": pool.rejected,
                "noMaskExcluded": len(pool.no_mask), "used": list(chosen),
            }
        write_meta(set_dir, meta)
    except BaseException:
        shutil.rmtree(set_dir, ignore_errors=True)
        raise
    return meta


def _volume_meta(meta: Dict[str, Any], volume: str) -> Dict[str, Any]:
    volumes = meta.get("volumes") or {}
    if volume not in volumes:
        raise GoldError(f"{volume} is not in this gold set")
    return volumes[volume]


def apply_id_list(
    set_dir: Path, volume: str, source_crops: Path, ids: Sequence[int], mode: str, *, force: bool = False
) -> Dict[str, Any]:
    """Make an uploaded list the set's cards for ``volume`` (``replace``) or add to them (``extend``).

    Replacing removes every current card, so it is refused when one is approved unless ``force``. Ids that cannot be used
    come back with the reason; the rest are copied.
    """
    if mode not in ("replace", "extend"):
        raise GoldError("mode must be replace or extend")
    meta = read_meta(set_dir)
    info = _volume_meta(meta, volume)
    crops = volume_crops(set_dir, volume)
    pool = _source_pool(source_crops)
    current = {card["location_id"]: card for card in cards(crops)}
    if mode == "replace":
        approved = catalog.load_approved(crops)
        if approved and not force:
            raise GoldError(f"{len(approved)} card(s) are approved; replacing would discard them (confirm to force)")
        for location_id in list(current):
            remove_card(crops, location_id)
        current = {}
    added: List[int] = []
    skipped: Dict[str, List[int]] = {"notInSource": [], "rejected": [], "noMask": [], "alreadyInSet": []}
    seed = int(meta["seed"])
    for location_id in ids:
        if location_id in current:
            skipped["alreadyInSet"].append(location_id)
        elif location_id in pool.rows:
            copy_card(source_crops, crops, window_for(pool, source_crops, seed, location_id))
            added.append(location_id)
        elif location_id in pool.no_mask:
            skipped["noMask"].append(location_id)
        else:
            # Either not in the source catalog or rejected there; say which when we can.
            rejected = {int(r["location_id"]) for r in catalog.list_catalog_rows(source_crops) if r.get("ignored")}
            skipped["rejected" if location_id in rejected else "notInSource"].append(location_id)
    info["used"] = sorted(set(info.get("used") or []) | set(added))
    info["listMode"] = mode
    info["listAt"] = now()
    write_meta(set_dir, meta)
    return {"added": added, "skipped": {key: value for key, value in skipped.items() if value}, "cards": len(cards(crops))}


def draw_replacement(set_dir: Path, volume: str, source_crops: Path, count: int = 1) -> List[int]:
    """Add the next ``count`` unused ids of the set's seeded order (for rejected cards). Same seed, same replacements."""
    meta = read_meta(set_dir)
    info = _volume_meta(meta, volume)
    pool = _source_pool(source_crops)
    if draw.fingerprint(pool.ids) != info.get("fingerprint"):
        raise GoldError("the source set changed since the draw, so its order cannot be reproduced; draw a new set")
    ordered = draw.order(pool, int(meta["seed"]), str(meta.get("stratify") or ""), int(meta.get("band") or draw.DEFAULT_Z_BAND))
    chosen = draw.next_unused(ordered, info.get("used") or [], count)
    crops = volume_crops(set_dir, volume)
    for location_id in chosen:
        copy_card(source_crops, crops, window_for(pool, source_crops, int(meta["seed"]), location_id))
    info["used"] = sorted(set(info.get("used") or []) | set(chosen))
    write_meta(set_dir, meta)
    return chosen


def progress(set_dir: Path) -> Dict[str, Any]:
    """Per-volume counts: cards, approved, edited, rejected and what is left to review."""
    meta = read_meta(set_dir)
    result: Dict[str, Any] = {}
    for volume in sorted((meta.get("volumes") or {})):
        crops = volume_crops(set_dir, volume)
        rows = catalog.list_catalog_rows(crops)
        edited = {card["location_id"] for card in cards(crops) if card["edited"]}
        approved = [row for row in rows if row.get("approved")]
        rejected = [row for row in rows if row.get("ignored")]
        active = [row for row in rows if not row.get("ignored")]
        result[volume] = {
            "target": meta.get("n"), "cards": len(active), "approved": len(approved), "rejected": len(rejected),
            "edited": len([row for row in active if row["location_id"] in edited]),
            "remaining": len([row for row in active if not row.get("approved")]),
        }
    return result
