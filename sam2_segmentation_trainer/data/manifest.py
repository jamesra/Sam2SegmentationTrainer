"""Index AnnotationCrops trees from manifest.jsonl (one example per locationId)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from sam2_segmentation_trainer.paths import DEFAULT_VOLUMES, annotation_crops_root, data_root


@dataclass(frozen=True, slots=True)
class CropExample:
    volume: str
    z: int
    downsample: int
    image_key: str
    location_id: int
    image_path: Path
    json_path: Path
    mask_path: Path
    split_key: str
    width: int | None = None
    height: int | None = None

    @property
    def image_ext(self) -> str:
        return self.image_path.suffix.lower()


def list_filenames(folder: Path) -> set[str]:
    """One directory listing instead of a CIFS stat per file."""
    if not folder.is_dir():
        return set()
    return {entry.name for entry in folder.iterdir() if entry.is_file()}


def sidecar_ids_and_size(json_path: Path) -> tuple[set[int] | None, tuple[int, int] | None]:
    """Return annotation ids and ``image.width`` / ``image.height``, or None if unreadable."""
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None, None
    ids = {int(ann.get("id", -1)) for ann in (payload.get("annotations") or [])}
    image = payload.get("image")
    if isinstance(image, list):
        image = image[0] if image else None
    size: tuple[int, int] | None = None
    if isinstance(image, dict) and "width" in image and "height" in image:
        try:
            size = (int(image["width"]), int(image["height"]))
        except (TypeError, ValueError):
            size = None
    return ids, size


def annotation_ids_in_json(json_path: Path) -> set[int] | None:
    """Return annotation ids in a sidecar JSON, or None if unreadable."""
    ids, _size = sidecar_ids_and_size(json_path)
    return ids


def take_model_sized(
    examples: Sequence[CropExample], image_size: int
) -> tuple[list[CropExample], list[CropExample]]:
    """Split examples by sidecar size. Rejected entries are one per tile."""
    kept: list[CropExample] = []
    rejected: list[CropExample] = []
    seen: set[tuple[str, str]] = set()
    expected = int(image_size)
    for ex in examples:
        if ex.width == expected and ex.height == expected:
            kept.append(ex)
            continue
        key = (ex.volume, ex.image_key)
        if key in seen:
            continue
        seen.add(key)
        rejected.append(ex)
    return kept, rejected


def index_volumes(
    volumes: Sequence[str] | None = None,
    root: Path | None = None,
    skip_missing: bool = True,
) -> list[CropExample]:
    base = root if root is not None else data_root()
    vol_names = tuple(volumes) if volumes is not None else DEFAULT_VOLUMES
    examples: list[CropExample] = []
    for volume in vol_names:
        crops = annotation_crops_root(volume, base)
        manifest = crops / "manifest.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(f"missing manifest: {manifest}")
        vol_examples, n_skip_ann = _index_manifest(
            volume, crops, manifest, skip_missing=skip_missing
        )
        examples.extend(vol_examples)
        if skip_missing and n_skip_ann:
            print(
                f"index {volume}: skipped {n_skip_ann} examples "
                f"(locationId missing from sidecar JSON)"
            )
    return examples


def _index_manifest(
    volume: str,
    crops: Path,
    manifest: Path,
    *,
    skip_missing: bool,
) -> tuple[list[CropExample], int]:
    image_names = list_filenames(crops / "images") if skip_missing else None
    examples: list[CropExample] = []
    skipped_missing_ann = 0
    with manifest.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            rec = json.loads(text)
            rel_image = rec.get("image") or rec.get("jpeg")
            rel_json = rec.get("json")
            image_key = rec.get("imageKey")
            if not rel_image or not rel_json or not image_key:
                raise ValueError(
                    f"{manifest}:{line_no} missing image/jpeg, json, or imageKey"
                )
            image_path = crops / rel_image
            json_path = crops / rel_json
            if image_names is not None:
                if image_path.name not in image_names or json_path.name not in image_names:
                    continue
            location_ids: Iterable[int] = rec.get("locationIds") or []
            downsample = int(rec.get("downsample", 1))
            z = int(rec.get("z", 0))
            rec_volume = str(rec.get("volume") or volume)
            present_ids: set[int] | None = None
            tile_size: tuple[int, int] | None = None
            if skip_missing:
                present_ids, tile_size = sidecar_ids_and_size(json_path)
                if present_ids is None:
                    skipped_missing_ann += sum(1 for _ in location_ids)
                    continue
            width = None if tile_size is None else tile_size[0]
            height = None if tile_size is None else tile_size[1]
            for loc_id in location_ids:
                loc_i = int(loc_id)
                if present_ids is not None and loc_i not in present_ids:
                    skipped_missing_ann += 1
                    continue
                examples.append(
                    CropExample(
                        volume=rec_volume,
                        z=z,
                        downsample=downsample,
                        image_key=str(image_key),
                        location_id=loc_i,
                        image_path=image_path,
                        json_path=json_path,
                        mask_path=crops / "masks" / f"{image_key}_{loc_i}.png",
                        split_key=f"{rec_volume}:{loc_i}",
                        width=width,
                        height=height,
                    )
                )
    return examples, skipped_missing_ann


def summarize_index(examples: Sequence[CropExample]) -> dict[str, object]:
    by_volume: dict[str, int] = {}
    by_downsample: dict[int, int] = {}
    by_ext: dict[str, int] = {}
    tiles: dict[tuple[str, str], int] = {}
    listed: dict[Path, set[str]] = {}

    def names(folder: Path) -> set[str]:
        cached = listed.get(folder)
        if cached is None:
            cached = list_filenames(folder)
            listed[folder] = cached
        return cached

    missing_image = 0
    missing_json = 0
    missing_raster = 0
    for ex in examples:
        by_volume[ex.volume] = by_volume.get(ex.volume, 0) + 1
        by_downsample[ex.downsample] = by_downsample.get(ex.downsample, 0) + 1
        by_ext[ex.image_ext] = by_ext.get(ex.image_ext, 0) + 1
        tiles[(ex.volume, ex.image_key)] = tiles.get((ex.volume, ex.image_key), 0) + 1
        image_dir_names = names(ex.image_path.parent)
        mask_dir_names = names(ex.mask_path.parent)
        if ex.image_path.name not in image_dir_names:
            missing_image += 1
        if ex.json_path.name not in image_dir_names:
            missing_json += 1
        if ex.mask_path.name not in mask_dir_names:
            missing_raster += 1
    multi_id_tiles = sum(1 for count in tiles.values() if count > 1)
    return {
        "n_examples": len(examples),
        "n_tiles": len(tiles),
        "multi_id_tiles": multi_id_tiles,
        "by_volume": dict(sorted(by_volume.items())),
        "by_downsample": {str(k): v for k, v in sorted(by_downsample.items())},
        "by_image_ext": dict(sorted(by_ext.items())),
        "missing_image": missing_image,
        "missing_json": missing_json,
        "missing_raster_mask": missing_raster,
    }
