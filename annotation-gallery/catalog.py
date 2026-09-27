"""Stdlib sqlite catalog and ignore-list mask moves for the slim gallery image."""

from __future__ import annotations

import io
import json
import os
import sqlite3
import zipfile
from pathlib import Path
from typing import Any, Iterable

SAM2_COLUMNS = (
    "sam2PredIou",
    "sam2ObjectScore",
    "sam2Stability",
    "sam2GtIou",
    "sam2Checkpoint",
    "sam2ScoredAt",
)

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS locations (
    location_id INTEGER PRIMARY KEY,
    z INTEGER NOT NULL,
    structure_id INTEGER,
    structure_label TEXT,
    type_id INTEGER,
    type_name TEXT,
    radius REAL,
    image_key TEXT NOT NULL,
    image_relpath TEXT,
    mask_relpath TEXT,
    ignored INTEGER NOT NULL DEFAULT 0,
    sam2PredIou REAL,
    sam2ObjectScore REAL,
    sam2Stability REAL,
    sam2GtIou REAL,
    sam2Checkpoint TEXT,
    sam2ScoredAt TEXT
)
"""

_CREATE_SCORES_SQL = """
CREATE TABLE IF NOT EXISTS location_scores (
    location_id INTEGER NOT NULL,
    image_key TEXT NOT NULL,
    epoch INTEGER NOT NULL,
    score REAL NOT NULL,
    PRIMARY KEY (location_id, image_key, epoch)
)
"""


def sqlite_path(crops: str | os.PathLike[str]) -> Path:
    """Return `{AnnotationCrops}/annotation_crops.sqlite`."""
    return Path(crops) / "annotation_crops.sqlite"


def ignore_path(crops: str | os.PathLike[str]) -> Path:
    """Return `{AnnotationCrops}/ignore.json`."""
    return Path(crops) / "ignore.json"


def connect(crops: str | os.PathLike[str]) -> sqlite3.Connection:
    """Open (and create) the per-volume catalog database."""
    path = sqlite_path(crops)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), timeout=5.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=5000")
    _ensure_locations_schema(connection)
    return connection


def _ensure_score_schema(connection: sqlite3.Connection) -> None:
    """Create per-window scores, or keep a location-only table under an empty image key.

    Older catalogs stored one score per location per epoch. Those rows stay
    readable as a fallback until a later epoch writes a score for the window.
    """
    connection.execute(_CREATE_SCORES_SQL)
    columns = {row[1] for row in connection.execute("PRAGMA table_info(location_scores)")}
    if "image_key" in columns:
        return
    connection.execute(
        """
        CREATE TABLE location_scores_window (
            location_id INTEGER NOT NULL,
            image_key TEXT NOT NULL,
            epoch INTEGER NOT NULL,
            score REAL NOT NULL,
            PRIMARY KEY (location_id, image_key, epoch)
        )
        """
    )
    connection.execute(
        """
        INSERT INTO location_scores_window (location_id, image_key, epoch, score)
        SELECT location_id, '', epoch, score FROM location_scores
        """
    )
    connection.execute("DROP TABLE location_scores")
    connection.execute("ALTER TABLE location_scores_window RENAME TO location_scores")


def _ensure_locations_schema(connection: sqlite3.Connection) -> None:
    """Create locations and rename jpeg_relpath on catalogs from JPEG-era exports."""
    connection.execute(_CREATE_SQL)
    _ensure_score_schema(connection)
    columns = {row[1] for row in connection.execute("PRAGMA table_info(locations)")}
    if "jpeg_relpath" in columns and "image_relpath" not in columns:
        connection.execute("ALTER TABLE locations RENAME COLUMN jpeg_relpath TO image_relpath")
        connection.commit()
    for name in ("origin_x", "origin_y"):
        if name in columns:
            connection.execute(f"ALTER TABLE locations DROP COLUMN {name}")
    connection.commit()


def load_ignore_ids(crops: str | os.PathLike[str]) -> set[int]:
    """Load ignored location ids from `ignore.json` (a JSON array)."""
    path = ignore_path(crops)
    if not path.is_file():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return set()
    ids: set[int] = set()
    for item in payload:
        try:
            ids.add(int(item))
        except (TypeError, ValueError):
            continue
    return ids


def save_ignore_ids(crops: str | os.PathLike[str], ids: Iterable[int]) -> None:
    """Write `ignore.json` as a sorted JSON array of location ids."""
    ordered = sorted({int(item) for item in ids})
    ignore_path(crops).write_text(json.dumps(ordered), encoding="utf-8")


def approved_path(crops: str | os.PathLike[str]) -> Path:
    """Return `{AnnotationCrops}/approved.json`."""
    return Path(crops) / "approved.json"


def _approval_key(location_id: int, image_key: str | None) -> tuple[int, str]:
    return (int(location_id), image_key or "")


def load_approved(crops: str | os.PathLike[str]) -> set[tuple[int, str]]:
    """Load approved windows from `approved.json`. The masks stay in the dataset."""
    path = approved_path(crops)
    if not path.is_file():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return set()
    keys: set[tuple[int, str]] = set()
    for item in payload:
        if not isinstance(item, dict):
            continue
        try:
            keys.add(_approval_key(int(item.get("location_id")), item.get("image_key") or ""))
        except (TypeError, ValueError):
            continue
    return keys


def save_approved(crops: str | os.PathLike[str], keys: Iterable[tuple[int, str]]) -> None:
    """Write `approved.json` as sorted location and image-key pairs."""
    ordered = [
        {"location_id": location_id, "image_key": image_key}
        for location_id, image_key in sorted(set(keys))
    ]
    approved_path(crops).write_text(json.dumps(ordered), encoding="utf-8")


def _location_image_keys(crops: str | os.PathLike[str], location_id: int) -> list[str]:
    """Image keys for one location. Empty when the catalog has no such rows."""
    if not sqlite_path(crops).is_file():
        return []
    connection = connect(crops)
    try:
        rows = connection.execute(
            "SELECT image_key FROM locations WHERE location_id = ?",
            [int(location_id)],
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        connection.close()
    return [str(row["image_key"]) for row in rows if row["image_key"]]


def approve_window(
    crops: str | os.PathLike[str],
    location_id: int,
    image_key: str | None = None,
) -> bool:
    """Remember a window as approved, or every window when *image_key* is omitted.

    A rejected location is restored into the dataset. The card sends no image
    key so one control covers the whole location.
    """
    location_id = int(location_id)
    targets = [image_key] if image_key else (_location_image_keys(crops, location_id) or [""])
    if _any_window_ignored(crops, location_id):
        restore_location(crops, location_id, image_key)
    keys = load_approved(crops)
    for key in targets:
        keys.add(_approval_key(location_id, key))
    save_approved(crops, keys)
    return True


def unapprove_window(
    crops: str | os.PathLike[str],
    location_id: int,
    image_key: str | None = None,
) -> bool:
    """Drop one window from `approved.json`, or every window when *image_key* is omitted."""
    location_id = int(location_id)
    keys = load_approved(crops)
    if image_key:
        keys.discard(_approval_key(location_id, image_key))
    else:
        keys = {item for item in keys if item[0] != location_id}
    save_approved(crops, keys)
    return True


def read_crop_size(crops: str | os.PathLike[str]) -> int | None:
    """Return the one window size for this volume, or None on a legacy catalog."""
    if not sqlite_path(crops).is_file():
        return None
    connection = connect(crops)
    try:
        row = connection.execute("SELECT crop_size FROM crop_geometry WHERE id = 1").fetchone()
    except sqlite3.OperationalError:
        return None
    finally:
        connection.close()
    if row is None:
        return None
    return int(row[0])


def _latest_losses(
    connection: sqlite3.Connection,
) -> tuple[dict[tuple[int, str], float], dict[int, float]]:
    """Latest score per window, plus location-wide scores stored with an empty image key."""
    try:
        rows = connection.execute(
            """
            SELECT location_id, image_key, score FROM location_scores
            WHERE epoch = (
                SELECT MAX(later.epoch) FROM location_scores AS later
                WHERE later.location_id = location_scores.location_id
                  AND later.image_key = location_scores.image_key
            )
            """
        ).fetchall()
    except sqlite3.OperationalError:
        return {}, {}
    by_window: dict[tuple[int, str], float] = {}
    legacy: dict[int, float] = {}
    for row in rows:
        location_id = int(row["location_id"])
        image_key = str(row["image_key"] or "")
        score = float(row["score"])
        if image_key:
            by_window[(location_id, image_key)] = score
        else:
            legacy[location_id] = score
    return by_window, legacy


def list_catalog_rows(crops: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Return catalog rows, with ``loss`` set to the latest score or None."""
    path = sqlite_path(crops)
    if not path.is_file():
        return []
    crops_path = Path(crops)
    connection = connect(crops)
    try:
        rows = connection.execute(
            "SELECT * FROM locations ORDER BY z, location_id"
        ).fetchall()
        result = [dict(row) for row in rows]
        by_window, legacy = _latest_losses(connection)
    finally:
        connection.close()
    approved = load_approved(crops)
    for row in result:
        location_id = int(row["location_id"])
        image_key = str(row.get("image_key") or "")
        row["loss"] = by_window.get((location_id, image_key), legacy.get(location_id))
        row["approved"] = int(_approval_key(location_id, row.get("image_key") or "") in approved)
        if row.get("ignored"):
            row["approved"] = 0
            # Older rejects stored the first window's mask on every row. Prefer
            # the file named for this window so each tile keeps its own mask.
            named = f"ignored/{image_key}_{location_id}.png"
            if image_key and (crops_path / named).is_file():
                row["mask_relpath"] = named
    return result


def ignore_location(
    crops: str | os.PathLike[str],
    location_id: int,
    image_key: str | None = None,
) -> bool:
    """Add *location_id* to ignore.json and move one window mask, or every mask."""
    unapprove_window(crops, int(location_id), image_key)
    ids = load_ignore_ids(crops)
    ids.add(int(location_id))
    save_ignore_ids(crops, ids)
    if image_key:
        _move_named_mask(crops, image_key, int(location_id), to_ignored=True)
        _set_ignored_flag(crops, int(location_id), ignored=True, image_key=image_key)
    else:
        apply_ignore_moves(crops)
        _set_ignored_flag(crops, int(location_id), ignored=True)
    return True


def restore_location(
    crops: str | os.PathLike[str],
    location_id: int,
    image_key: str | None = None,
) -> bool:
    """Restore one window mask, or every mask, and drop the location when none stay ignored."""
    location_id = int(location_id)
    if image_key:
        _move_named_mask(crops, image_key, location_id, to_ignored=False)
        _set_ignored_flag(crops, location_id, ignored=False, image_key=image_key)
        if not _any_window_ignored(crops, location_id):
            ids = load_ignore_ids(crops)
            ids.discard(location_id)
            save_ignore_ids(crops, ids)
        return True
    ids = load_ignore_ids(crops)
    ids.discard(location_id)
    save_ignore_ids(crops, ids)
    masks = Path(crops) / "masks"
    for source in _masks_named_for(Path(crops) / "ignored", location_id):
        masks.mkdir(parents=True, exist_ok=True)
        destination = masks / source.name
        if destination.is_file():
            destination.unlink()
        source.replace(destination)
    _set_ignored_flag(crops, location_id, ignored=False)
    return True


def _move_named_mask(
    crops: str | os.PathLike[str],
    image_key: str,
    location_id: int,
    *,
    to_ignored: bool,
) -> None:
    """Move `{image_key}_{location_id}.png` between masks/ and ignored/."""
    root = Path(crops)
    name = f"{image_key}_{int(location_id)}.png"
    source_dir = root / ("masks" if to_ignored else "ignored")
    dest_dir = root / ("ignored" if to_ignored else "masks")
    source = source_dir / name
    if not source.is_file():
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    destination = dest_dir / name
    if destination.is_file():
        destination.unlink()
    source.replace(destination)


def _any_window_ignored(crops: str | os.PathLike[str], location_id: int) -> bool:
    if not sqlite_path(crops).is_file():
        return False
    connection = connect(crops)
    try:
        row = connection.execute(
            "SELECT 1 FROM locations WHERE location_id = ? AND ignored = 1 LIMIT 1",
            [int(location_id)],
        ).fetchone()
    finally:
        connection.close()
    return row is not None


def import_ignore_ids(crops: str | os.PathLike[str], ids: Iterable[int]) -> dict[str, int]:
    """Merge rejected ids into ignore.json and move any masks that are present."""
    return import_review_lists(crops, ids, [], replace=False)


def import_review_lists(
    crops: str | os.PathLike[str],
    rejected_ids: Iterable[int],
    approved_keys: Iterable[tuple[int, str]],
    *,
    replace: bool,
) -> dict[str, int]:
    """Merge or replace the rejected and approved lists.

    Rejected ids that leave the list on replace have their masks moved back.
    A location in the rejected list is not kept as approved.
    """
    incoming_rejected = {int(item) for item in rejected_ids}
    incoming_approved = {_approval_key(location_id, image_key) for location_id, image_key in approved_keys}
    existing_rejected = load_ignore_ids(crops)
    existing_approved = load_approved(crops)
    if replace:
        final_rejected = incoming_rejected
        final_approved = incoming_approved
    else:
        final_rejected = existing_rejected | incoming_rejected
        final_approved = existing_approved | incoming_approved
    final_approved = {key for key in final_approved if key[0] not in final_rejected}
    for location_id in existing_rejected - final_rejected:
        restore_location(crops, location_id)
    added = final_rejected - existing_rejected
    save_ignore_ids(crops, final_rejected)
    save_approved(crops, final_approved)
    moved = apply_ignore_moves(crops)
    for location_id in final_rejected:
        _set_ignored_flag(crops, location_id, ignored=True)
    return {
        "added": len(added),
        "total": len(final_rejected),
        "moved": moved,
        "approvedTotal": len(final_approved),
    }


def build_repair_pack(
    crops: str | os.PathLike[str],
    *,
    volume: str,
    viking_template: str,
) -> bytes:
    """Zip ignored masks, their crop images, and a manifest for repair.

    Returns an empty byte string when nothing is ignored.
    """
    root = Path(crops)
    ids = sorted(load_ignore_ids(root))
    if not ids:
        return b""
    rows = {int(row["location_id"]): row for row in list_catalog_rows(root)}
    buffer = io.BytesIO()
    manifest: list[dict[str, Any]] = []
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for location_id in ids:
            row = rows.get(location_id, {})
            mask = _find_mask(root / "ignored", location_id) or _find_mask(root / "masks", location_id)
            image_rel = str(row.get("image_relpath") or "")
            image = root / image_rel if image_rel else None
            mask_name = f"masks/{mask.name}" if mask is not None else ""
            image_name = f"images/{Path(image_rel).name}" if image is not None and image.is_file() else ""
            if mask is not None:
                archive.write(mask, mask_name)
            if image_name:
                archive.write(image, image_name)
            structure_id = row.get("structure_id")
            manifest.append({
                "locationId": location_id,
                "z": row.get("z"),
                "structureId": structure_id,
                "structureLabel": row.get("structure_label"),
                "typeName": row.get("type_name"),
                "image": image_name,
                "mask": mask_name,
                "vikingUrl": viking_template.replace("{volume}", volume).replace("{id}", str(location_id)),
                "sbfsemUrl": (
                    f"https://sbfsem-tools.com/open?volume={volume}&cells={structure_id}&location={location_id}"
                    if structure_id is not None
                    else ""
                ),
            })
        archive.writestr("manifest.json", json.dumps({"volume": volume, "locations": manifest}, indent=2))
    return buffer.getvalue()


def review_lists_are_empty(crops: str | os.PathLike[str]) -> bool:
    """True when this volume has no rejected ids and no approved windows."""
    return not load_ignore_ids(crops) and not load_approved(crops)


def export_volume_review(crops: str | os.PathLike[str], volume: str) -> dict[str, Any]:
    """Rejected ids and approved windows for one volume."""
    ids = sorted(load_ignore_ids(crops))
    approved = [
        {"location_id": location_id, "image_key": image_key}
        for location_id, image_key in sorted(load_approved(crops))
    ]
    return {
        "volume": volume,
        "rejected": ids,
        "locationIds": ids,
        "approved": approved,
    }


def held_review_dir(registry: str | os.PathLike[str]) -> Path:
    """Review lists for volumes that are not in this training set.

    The folder name is not a volume name, so the gallery does not list it.
    """
    return Path(registry) / "_review"


def load_held_reviews(registry: str | os.PathLike[str]) -> dict[str, dict[str, Any]]:
    """Return held review documents keyed by volume name."""
    folder = held_review_dir(registry)
    if not folder.is_dir():
        return {}
    found: dict[str, dict[str, Any]] = {}
    for path in sorted(folder.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            found[path.stem] = payload
    return found


def save_held_review(
    registry: str | os.PathLike[str],
    volume: str,
    rejected_ids: Iterable[int],
    approved_keys: Iterable[tuple[int, str]],
) -> None:
    """Store one volume's lists without creating a volume folder."""
    folder = held_review_dir(registry)
    folder.mkdir(parents=True, exist_ok=True)
    document = {
        "volume": volume,
        "rejected": sorted({int(item) for item in rejected_ids}),
        "approved": [
            {"location_id": location_id, "image_key": image_key}
            for location_id, image_key in sorted(set(approved_keys))
        ],
    }
    (folder / f"{volume}.json").write_text(json.dumps(document), encoding="utf-8")


def parse_training_review(payload: Any) -> list[tuple[str, list[int], list[tuple[int, str]]]]:
    """Volumes in a training-set review file, or one legacy single-volume object."""
    if isinstance(payload, dict) and "volumes" in payload:
        raw = payload["volumes"]
        if isinstance(raw, dict):
            entries = []
            for name, item in raw.items():
                if not isinstance(item, dict):
                    raise ValueError("each volume entry must be an object")
                entries.append((str(name), item))
        elif isinstance(raw, list):
            entries = []
            for item in raw:
                if not isinstance(item, dict) or not item.get("volume"):
                    raise ValueError("each volume entry needs a volume name")
                entries.append((str(item["volume"]), item))
        else:
            raise ValueError("volumes must be a list or an object")
        parsed: list[tuple[str, list[int], list[tuple[int, str]]]] = []
        for name, item in entries:
            rejected, approved, _replace = parse_review_import(item)
            parsed.append((name, rejected, approved))
        return parsed
    if isinstance(payload, dict) and payload.get("volume"):
        rejected, approved, _replace = parse_review_import(payload)
        return [(str(payload["volume"]), rejected, approved)]
    raise ValueError("review file must name a volume or a volumes list")


def parse_ignore_ids(payload: Any) -> list[int]:
    """Accept a JSON array or an object with ``locationIds`` or ``rejected``."""
    rejected, _approved, _replace = parse_review_import(payload)
    return rejected


def parse_review_import(payload: Any) -> tuple[list[int], list[tuple[int, str]], bool]:
    """Return rejected ids, approved windows, and whether to replace existing lists."""
    if isinstance(payload, list):
        items = payload
        approved_raw: list[Any] = []
        replace = False
    elif isinstance(payload, dict):
        items = payload.get("rejected", payload.get("locationIds", payload.get("location_ids", [])))
        approved_raw = payload.get("approved") or []
        mode = str(payload.get("mode") or "merge").strip().lower()
        if mode not in {"merge", "replace"}:
            raise ValueError("mode must be merge or replace")
        replace = mode == "replace"
    else:
        raise ValueError("review list must be an array or an object")
    if not isinstance(items, list) or not isinstance(approved_raw, list):
        raise ValueError("rejected and approved must be arrays")
    approved: list[tuple[int, str]] = []
    for item in approved_raw:
        if isinstance(item, dict):
            approved.append(_approval_key(int(item.get("location_id")), item.get("image_key") or ""))
        else:
            approved.append(_approval_key(int(item), ""))
    return [int(item) for item in items], approved, replace


def apply_ignore_moves(crops: str | os.PathLike[str]) -> int:
    """Move every ignored window mask into `ignored/`, replacing a file already there.

    A location is one mask per crop window. Moving only the first file left the
    other windows pointing at that single mask, so the gallery could not show
    the location as a group of tiles.
    """
    output = Path(crops)
    ignored_dir = output / "ignored"
    moved = 0
    for location_id in load_ignore_ids(output):
        for source in _masks_named_for(output / "masks", location_id):
            ignored_dir.mkdir(parents=True, exist_ok=True)
            destination = ignored_dir / source.name
            if destination.is_file():
                destination.unlink()
            source.replace(destination)
            moved += 1
    return moved


def _set_ignored_flag(
    crops: str | os.PathLike[str],
    location_id: int,
    *,
    ignored: bool,
    image_key: str | None = None,
) -> None:
    path = sqlite_path(crops)
    if not path.is_file():
        return
    connection = connect(crops)
    try:
        flag = 1 if ignored else 0
        folder = "ignored" if ignored else "masks"
        root = Path(crops)
        if image_key:
            mask_rel = _mask_relpath_for_id(crops, location_id, ignored=ignored)
            named = f"{folder}/{image_key}_{int(location_id)}.png"
            if (root / named).is_file():
                mask_rel = named
            connection.execute(
                "UPDATE locations SET ignored = ?, mask_relpath = ? "
                "WHERE location_id = ? AND image_key = ?",
                [flag, mask_rel, location_id, image_key],
            )
        else:
            # One path for the whole location paints the first window's mask on
            # every tile. Each window keeps the file named for its image key.
            windows = connection.execute(
                "SELECT image_key, mask_relpath FROM locations WHERE location_id = ?",
                [location_id],
            ).fetchall()
            for window in windows:
                key = str(window["image_key"] or "")
                rel = str(window["mask_relpath"] or "")
                if key:
                    named = f"{folder}/{key}_{int(location_id)}.png"
                    if (root / named).is_file():
                        rel = named
                connection.execute(
                    "UPDATE locations SET ignored = ?, mask_relpath = ? "
                    "WHERE location_id = ? AND image_key = ?",
                    [flag, rel, location_id, key],
                )
        connection.commit()
    finally:
        connection.close()


def _mask_relpath_for_id(crops: str | os.PathLike[str], location_id: int, *, ignored: bool) -> str:
    folder = Path(crops) / ("ignored" if ignored else "masks")
    found = _find_mask(folder, location_id)
    if found is not None:
        return str(found.relative_to(crops)).replace("\\", "/")
    return f"{'ignored' if ignored else 'masks'}/{location_id}.png"


def _masks_named_for(folder: Path, location_id: int) -> list[Path]:
    """Return every ``*_{location_id}.png`` in *folder*, in name order."""
    if not folder.is_dir():
        return []
    suffix = f"_{int(location_id)}.png"
    return sorted(path for path in folder.glob(f"*{suffix}") if path.is_file())


def _find_mask(folder: Path, location_id: int) -> Path | None:
    matches = _masks_named_for(folder, location_id)
    if not matches:
        return None
    return matches[0]


_SOURCE_EXPORTERS = frozenset({"tiled", "legacy"})
_SOURCE_SECRET_KEYS = frozenset({
    "connection",
    "connectionstring",
    "connection_string",
    "server",
    "password",
    "user",
    "uid",
    "pwd",
    "credentials",
})


def source_path(crops: str | os.PathLike[str]) -> Path:
    """Return `{AnnotationCrops}/source.json`."""
    return Path(crops) / "source.json"


def load_source_document(crops: str | os.PathLike[str]) -> dict[str, Any]:
    """Read source.json. Connection fields are dropped and never returned."""
    path = source_path(crops)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        key: value
        for key, value in payload.items()
        if str(key).lower() not in _SOURCE_SECRET_KEYS
    }


def save_source_settings(
    crops: str | os.PathLike[str],
    *,
    odata: str | None,
    exporter: str,
) -> dict[str, Any]:
    """Write the OData URL and exporter into source.json and the sqlite source row.

    A missing kind becomes ``odata`` when a URL is saved. ``sql`` stays ``sql``
    so the file still does not claim the export itself came from OData. A
    geometries path is left in place.
    """
    mode = (exporter or "").strip().lower()
    if mode not in _SOURCE_EXPORTERS:
        raise ValueError("exporter must be tiled or legacy")
    current = load_source_document(crops)
    current["exporter"] = mode
    url = (odata or "").strip()
    if url:
        current["odata"] = url
        if not current.get("kind"):
            current["kind"] = "odata"
    else:
        current.pop("odata", None)
    source_path(crops).write_text(json.dumps(current, indent=2), encoding="utf-8")
    _mirror_source_row(crops, current)
    return current


def _mirror_source_row(crops: str | os.PathLike[str], source: dict[str, Any]) -> None:
    """Copy the public source fields into annotation_crops.sqlite when that file exists."""
    if not sqlite_path(crops).is_file():
        return
    connection = connect(crops)
    try:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS source ("
            "id INTEGER PRIMARY KEY CHECK (id = 1), "
            "odata TEXT, kind TEXT, filter TEXT, path TEXT, ingested_at TEXT)"
        )
        columns = {row[1] for row in connection.execute("PRAGMA table_info(source)")}
        if "exporter" not in columns:
            connection.execute("ALTER TABLE source ADD COLUMN exporter TEXT")
        connection.execute(
            "INSERT INTO source (id, odata, kind, filter, path, ingested_at, exporter) "
            "VALUES (1, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET "
            "odata = excluded.odata, "
            "kind = excluded.kind, "
            "filter = excluded.filter, "
            "path = excluded.path, "
            "ingested_at = excluded.ingested_at, "
            "exporter = excluded.exporter",
            [
                source.get("odata"),
                source.get("kind"),
                source.get("filter"),
                source.get("path"),
                source.get("ingestedAt"),
                source.get("exporter"),
            ],
        )
        connection.commit()
    finally:
        connection.close()
