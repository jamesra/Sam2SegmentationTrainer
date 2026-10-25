"""Gold-set operations the server exposes: editing a card's mask, recording approvals, and the export.

An approval pins the exact mask it was given for (its SHA-1). Any later edit of an approved card un-approves it, so what is
approved is always what the manifest's hash says.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

import catalog
from gold import store

MAX_PNG_BYTES = 8 * 1024 * 1024
MAX_VERTICES = 20000


class EditRejected(Exception):
    """The edit cannot be applied; the message is for the reviewer."""


def _write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    Image.fromarray((mask.astype(np.uint8)) * 255, mode="L").save(temporary, format="PNG")
    temporary.replace(path)


def _read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 127


def _mask_from_rings(rings: Any, shape: tuple) -> np.ndarray:
    from converter.web import EditRejected as _Rejected, _ring_from
    from gold.raster import fill_rings

    if not isinstance(rings, list) or not rings or sum(len(r) for r in rings if isinstance(r, list)) > MAX_VERTICES:
        raise EditRejected("an outline is required")
    try:
        parsed = [_ring_from(ring) for ring in rings]
    except _Rejected as error:
        raise EditRejected(str(error)) from None
    height, width = shape
    return fill_rings(parsed, width, height)


def _mask_from_png(data: Any, shape: tuple) -> np.ndarray:
    if not isinstance(data, str):
        raise EditRejected("a mask image is required")
    try:
        raw = base64.b64decode(data.split(",", 1)[-1], validate=True)
    except (binascii.Error, ValueError):
        raise EditRejected("the mask image is not valid base64") from None
    if len(raw) > MAX_PNG_BYTES:
        raise EditRejected("the mask image is too large")
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            if image.size != (shape[1], shape[0]):
                raise EditRejected(f"the mask must be {shape[1]} x {shape[0]} pixels")
            return np.asarray(image.convert("L")) > 127
    except OSError:
        raise EditRejected("the mask image could not be read") from None


def apply_edit(crops: Path, location_id: int, image_key: str, *, rings: Any = None, png: Any = None) -> Dict[str, Any]:
    """Replace a card's mask with a retraced outline (``rings``, crop pixels) or a painted mask (``png``, base64).

    The original (``source_masks/``) is never touched. The card is marked edited, its changed-pixel count and hash updated,
    cached layers dropped, and an approval it held is withdrawn.
    """
    from refresh import drop_stale_thumbnails

    if (rings is None) == (png is None):
        raise EditRejected("send either rings or a mask image")
    cards = {card["location_id"]: card for card in store.cards(crops)}
    card = cards.get(int(location_id))
    if card is None or card["image_key"] != image_key:
        raise EditRejected("no such card")
    mask_path = crops / "masks" / f"{image_key}_{location_id}.png"
    source_path = crops / "source_masks" / f"{image_key}_{location_id}.png"
    if not mask_path.is_file() or not source_path.is_file():
        raise EditRejected("the card's mask file is missing")
    source = _read_mask(source_path)
    mask = _mask_from_rings(rings, source.shape) if rings is not None else _mask_from_png(png, source.shape)
    _write_mask(mask_path, mask)
    digest = store.sha1_of(mask_path)
    changed = int(np.count_nonzero(mask != source))
    was_approved = (int(location_id), image_key) in catalog.load_approved(crops)
    if was_approved:
        catalog.unapprove_window(crops, int(location_id), image_key)
    connection = store._connect(crops)
    try:
        connection.execute(
            "UPDATE gold_cards SET edited = ?, changed_pixels = ?, mask_sha1 = ?, approved_by = NULL, approved_at = NULL, "
            "approved_sha1 = NULL WHERE location_id = ?",
            [1 if changed else 0, changed, digest, int(location_id)],
        )
        connection.commit()
    finally:
        connection.close()
    drop_stale_thumbnails(crops, [image_key])
    return {
        "ok": True, "location_id": int(location_id), "changed_pixels": changed, "edited": bool(changed),
        "mask_pixels": int(mask.sum()), "unapproved": was_approved, "mask_sha1": digest,
    }


def refresh_card(crops: Path, location_id: int, entity: Dict[str, Any]) -> List[str]:
    """Redraw a card's mask from the location's current OData outline and discard its edits.

    A gold card has no per-image annotation json for nornir's refresh to find its crop window by, so the window
    comes from the card's image key and the mask is rasterized into a scratch folder. The result replaces both the
    working mask and the ``source_masks/`` original, the card goes back to unedited, and an approval it held is
    withdrawn, as an edit does. Returns the card's image key.
    """
    import tempfile

    from refresh import (
        _KEY_WINDOW, RefreshUnavailable, drop_stale_thumbnails, load_add_support, nornir_buildmanager_root,
    )

    card = {item["location_id"]: item for item in store.cards(crops)}.get(int(location_id))
    if card is None:
        raise EditRejected("no such card")
    image_key = str(card["image_key"])
    window = _KEY_WINDOW.match(image_key)
    image_path = next((path for path in (crops / "images" / f"{image_key}.png", crops / "images" / f"{image_key}.jpg")
                       if path.is_file()), None)
    if window is None or image_path is None:
        raise EditRejected("the card's crop image is missing")
    root = nornir_buildmanager_root()
    if root is None:
        raise RefreshUnavailable("Mask refresh needs the nornir-buildmanager mount. No masks were changed.")
    try:
        support = load_add_support(root)
    except ImportError as exc:
        raise RefreshUnavailable(f"Mask refresh could not import nornir-buildmanager from the mount. ({exc})") from exc
    record = support.location_from_odata_entity(entity, include_off_edge=True)
    if record is None:
        raise ValueError("OData location has no mask geometry")
    polygons = support.hydrate_polygons(support.parse_wkt_polygons(record.wkt), record.type_code)
    if not polygons:
        raise ValueError("OData location has no mask geometry")
    downsample = float(int(window["d"]))
    with Image.open(image_path) as picture:
        width, height = picture.size
    with tempfile.TemporaryDirectory() as scratch:
        job = support.mask_job_for_crop(
            Path(scratch), int(location_id), image_key, polygons,
            origin_x=int(window["x0"]) * downsample, origin_y=int(window["y0"]) * downsample,
            downsample=downsample, width=width, height=height,
        )
        if job is None:
            raise ValueError("the location's outline no longer falls inside this card's crop")
        support.run_mask_jobs([job], workers=1)
        drawn = Path(job.mask_path)
        if not drawn.is_file():
            raise ValueError("the mask could not be redrawn")
        mask = _read_mask(drawn)
    mask_name = f"{image_key}_{int(location_id)}.png"
    _write_mask(crops / "masks" / mask_name, mask)
    _write_mask(crops / "source_masks" / mask_name, mask)
    digest = store.sha1_of(crops / "masks" / mask_name)
    if (int(location_id), image_key) in catalog.load_approved(crops):
        catalog.unapprove_window(crops, int(location_id), image_key)
    connection = store._connect(crops)
    try:
        connection.execute(
            "UPDATE gold_cards SET edited = 0, changed_pixels = 0, mask_sha1 = ?, source_mask_sha1 = ?, approved_by = NULL, "
            "approved_at = NULL, approved_sha1 = NULL WHERE location_id = ?",
            [digest, digest, int(location_id)],
        )
        connection.commit()
    finally:
        connection.close()
    drop_stale_thumbnails(crops, [image_key])
    return [image_key]


def record_approval(crops: Path, location_id: int, who: str) -> Optional[Dict[str, Any]]:
    """Pin an approval to the card's current mask: who, when, and the mask's hash."""
    cards = {card["location_id"]: card for card in store.cards(crops)}
    card = cards.get(int(location_id))
    if card is None:
        return None
    digest = store.sha1_of(crops / "masks" / f"{card['image_key']}_{location_id}.png")
    stamp = store.now()
    connection = store._connect(crops)
    try:
        connection.execute(
            "UPDATE gold_cards SET approved_by = ?, approved_at = ?, approved_sha1 = ?, mask_sha1 = ? WHERE location_id = ?",
            [who, stamp, digest, digest, int(location_id)],
        )
        connection.commit()
    finally:
        connection.close()
    return {"location_id": int(location_id), "approved_by": who, "approved_at": stamp, "approved_sha1": digest}


def clear_approval(crops: Path, location_id: int) -> None:
    connection = store._connect(crops)
    try:
        connection.execute(
            "UPDATE gold_cards SET approved_by = NULL, approved_at = NULL, approved_sha1 = NULL WHERE location_id = ?",
            [int(location_id)],
        )
        connection.commit()
    finally:
        connection.close()


def approved_cards(crops: Path) -> List[Dict[str, Any]]:
    """Approved cards whose mask still matches what was approved (a stale approval is reported by ``problems``)."""
    approved = catalog.load_approved(crops)
    result = []
    for card in store.cards(crops):
        if (card["location_id"], card["image_key"]) in approved:
            result.append(card)
    return result


def problems(crops: Path) -> List[str]:
    """Approved cards that do not match their recorded approval: a mask changed behind the page's back."""
    found = []
    for card in approved_cards(crops):
        current = store.sha1_of(crops / "masks" / f"{card['image_key']}_{card['location_id']}.png")
        if not card["approved_sha1"] or current != card["approved_sha1"]:
            found.append(f"location {card['location_id']}: the mask no longer matches its approval")
    return found


def export_zip(set_dir: Path, volume: Optional[str] = None, with_images: bool = True) -> bytes:
    """A zip of the approved cards: ``<volume>/masks``, ``<volume>/images`` and ``manifest.json`` with every hash."""
    meta = store.read_meta(set_dir)
    volumes = [volume] if volume else sorted(meta.get("volumes") or {})
    manifest: Dict[str, Any] = {
        "set": meta.get("name"), "sourceSet": meta.get("sourceSet"), "seed": meta.get("seed"), "n": meta.get("n"),
        "stratify": meta.get("stratify"), "drawnAt": meta.get("drawnAt"), "exportedAt": store.now(), "cards": [], "problems": [],
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in volumes:
            crops = store.volume_crops(set_dir, name)
            rows = {int(row["location_id"]): row for row in catalog.list_catalog_rows(crops)}
            manifest["problems"].extend(f"{name}: {line}" for line in problems(crops))
            for card in approved_cards(crops):
                key, location_id = card["image_key"], card["location_id"]
                mask = crops / "masks" / f"{key}_{location_id}.png"
                archive.write(mask, f"{name}/masks/{key}_{location_id}.png")
                row = rows.get(location_id, {})
                if with_images:
                    image = crops / str(row.get("image_relpath") or f"images/{key}.png")
                    if image.is_file():
                        archive.write(image, f"{name}/images/{image.name}")
                manifest["cards"].append(
                    {
                        "volume": name, "location_id": location_id, "image_key": key, "z": row.get("z"),
                        "structure_id": row.get("structure_id"), "type_id": row.get("type_id"), "edited": bool(card["edited"]),
                        "changed_pixels": card["changed_pixels"], "mask_sha1": card["approved_sha1"],
                        "source_mask_sha1": card["source_mask_sha1"], "approved_by": card["approved_by"],
                        "approved_at": card["approved_at"],
                    }
                )
        archive.writestr("manifest.json", json.dumps(manifest, indent=1))
    return buffer.getvalue()
