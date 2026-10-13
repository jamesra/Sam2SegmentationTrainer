"""One-location mask refresh through a read-only nornir-buildmanager mount.

The gallery image does not install Nornir. ``segmentationtraining`` no longer
imports the export pipeline at package import, so this module can load
``maskrefresh`` without stubbing that package. The volume-pipeline package
inits (``nornir_buildmanager`` and ``operations``) still are not executed:
they import build, tile, and imageregistration. A missing mount raises
``RefreshUnavailable`` and does not write masks.
"""

from __future__ import annotations

import json
import os
import re
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from cardmask import drop_card_masks
from thumbs import IMAGE_KEY, THUMB_SIZES

class RefreshUnavailable(RuntimeError):
    """The nornir-buildmanager mount or its imports are not available."""


def nornir_buildmanager_root() -> Path | None:
    """Checkout root that contains the ``nornir_buildmanager`` package, or None."""
    raw = (os.environ.get("NORNIR_BUILDMANAGER_DIR") or "/opt/nornir-buildmanager").strip()
    path = Path(raw)
    if (path / "nornir_buildmanager").is_dir():
        return path
    return None


def _register_package(name: str, path: Path) -> None:
    """Register *name* so its ``__init__.py`` does not run.

    ``nornir_buildmanager`` and ``operations`` start the volume pipeline.
    ``segmentationtraining`` is imported for real; its init does not load that pipeline.
    """
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    module.__package__ = name
    sys.modules[name] = module


def load_mask_refresh(root: Path) -> tuple[Callable[..., Any], Callable[..., list[str]]]:
    """Import the mask redraw API from *root* without the export pipeline.

    *root* is the nornir-buildmanager checkout (the directory that contains
    the ``nornir_buildmanager`` package).
    """
    package = root / "nornir_buildmanager"
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    _register_package("nornir_buildmanager", package)
    _register_package("nornir_buildmanager.operations", package / "operations")
    from nornir_buildmanager.operations.segmentationtraining.maskrefresh import (
        location_from_odata_entity,
        refresh_existing_location_masks,
    )

    return location_from_odata_entity, refresh_existing_location_masks


def refresh_location(crops: str | os.PathLike[str], entity: dict[str, Any], exporter: str) -> list[str]:
    """Redraw masks for one OData location on crops that already exist.

    Returns the image keys that were rewritten. Does not create a TEM image.
    """
    root = nornir_buildmanager_root()
    if root is None:
        raise RefreshUnavailable(
            "Mask refresh needs the nornir-buildmanager mount. "
            "No masks were changed."
        )
    try:
        location_from_odata_entity, refresh_existing_location_masks = load_mask_refresh(root)
    except ImportError as exc:
        raise RefreshUnavailable(
            "Mask refresh could not import nornir-buildmanager from the mount. "
            f"No masks were changed. ({exc})"
        ) from exc
    record = location_from_odata_entity(entity, include_off_edge=True)
    if record is None:
        raise ValueError("OData location has no mask geometry")
    return refresh_existing_location_masks(crops, record, exporter=exporter)


def drop_stale_thumbnails(crops: str | os.PathLike[str], image_keys: list[str]) -> None:
    """Delete cached card JPEGs and card mask layers so the next paint rebuilds them from the new mask."""
    root = Path(crops)
    drop_card_masks(root, image_keys)
    for key in image_keys:
        if IMAGE_KEY.fullmatch(str(key)) is None:
            continue
        for size in THUMB_SIZES:
            path = root / "overlays" / "thumb" / str(size) / f"{key}.jpg"
            path.unlink(missing_ok=True)


_KEY_WINDOW = re.compile(
    r"^.+_(?P<z>\d+)_D(?P<d>\d+)_X(?P<x0>-?\d+)-(?P<x1>-?\d+)_Y(?P<y0>-?\d+)-(?P<y1>-?\d+)$"
)
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


@dataclass(frozen=True)
class TileWindow:
    """A crop image that is on disk: its key, downsample, and pixel window (downsampled pixels, as in the key)."""

    key: str
    downsample: int
    x0: int
    x1: int
    y0: int
    y1: int

    def mosaic_rect(self) -> tuple[float, float, float, float]:
        scale = float(self.downsample)
        return self.x0 * scale, self.y0 * scale, self.x1 * scale, self.y1 * scale


class TileIndex:
    """The crop windows whose image is on disk, by section. One scan of ``images/``, built when first needed."""

    def __init__(self, crops: str | os.PathLike[str]) -> None:
        self._by_z: dict[int, list[TileWindow]] = {}
        images = Path(crops) / "images"
        if not images.is_dir():
            return
        for entry in os.scandir(images):
            name = entry.name
            suffix = os.path.splitext(name)[1].lower()
            if suffix not in _IMAGE_SUFFIXES or not entry.is_file(follow_symlinks=False):
                continue
            stem = name[: -len(suffix)]
            match = _KEY_WINDOW.match(stem)
            if match is None:
                continue
            window = TileWindow(
                key=stem,
                downsample=int(match["d"]),
                x0=int(match["x0"]),
                x1=int(match["x1"]),
                y0=int(match["y0"]),
                y1=int(match["y1"]),
            )
            if window.downsample > 0 and window.x1 > window.x0 and window.y1 > window.y0:
                self._by_z.setdefault(int(match["z"]), []).append(window)

    def windows(self, z: int) -> list[TileWindow]:
        return self._by_z.get(int(z), [])


def covers_box(box: tuple[float, float, float, float], rects: list[tuple[float, float, float, float]]) -> bool:
    """True when the rectangles together cover every point of *box* (min_x, min_y, max_x, max_y)."""
    min_x, min_y, max_x, max_y = box
    xs = sorted({min_x, max_x, *(v for r in rects for v in (r[0], r[2]) if min_x < v < max_x)})
    ys = sorted({min_y, max_y, *(v for r in rects for v in (r[1], r[3]) if min_y < v < max_y)})
    for left, right in zip(xs, xs[1:]):
        for top, bottom in zip(ys, ys[1:]):
            cx, cy = (left + right) / 2.0, (top + bottom) / 2.0
            if not any(r[0] <= cx <= r[2] and r[1] <= cy <= r[3] for r in rects):
                return False
    return True


def load_add_support(root: Path) -> types.SimpleNamespace:
    """The nornir pieces that rasterize a location into existing crop windows, imported without the pipeline."""
    load_mask_refresh(root)  # registers the package stubs and puts *root* on sys.path
    from nornir_buildmanager.operations.segmentationtraining.curves import hydrate_polygons
    from nornir_buildmanager.operations.segmentationtraining.geometry import mosaic_bbox
    from nornir_buildmanager.operations.segmentationtraining.masks import mask_job_for_crop, run_mask_jobs
    from nornir_buildmanager.operations.segmentationtraining.sam2.write import sa1b_annotation
    from nornir_buildmanager.operations.segmentationtraining.wkt import (
        location_from_odata_entity,
        parse_wkt_polygons,
    )

    return types.SimpleNamespace(
        location_from_odata_entity=location_from_odata_entity,
        parse_wkt_polygons=parse_wkt_polygons,
        hydrate_polygons=hydrate_polygons,
        mosaic_bbox=mosaic_bbox,
        mask_job_for_crop=mask_job_for_crop,
        run_mask_jobs=run_mask_jobs,
        sa1b_annotation=sa1b_annotation,
    )


def add_new_location(
    crops: str | os.PathLike[str],
    entity: dict[str, Any],
    index: TileIndex,
    support: types.SimpleNamespace | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Add a location the dataset does not have yet, using only crop images already on disk.

    Needs every existing window that overlaps the polygon, and those windows together must cover its bounding
    box; otherwise nothing is written and the reason comes back for the skip log. On success the masks are
    rasterized, each window's sidecar JSON gains the annotation, and the result carries what the catalog needs:
    ``{"record": record, "windows": [{"image_key", "area"}]}``. Nothing is downloaded or stitched. The section
    watermark is left alone; the next nornir ``-Update`` reconciles it.
    """
    if support is None:
        root = nornir_buildmanager_root()
        if root is None:
            raise RefreshUnavailable("Adding locations needs the nornir-buildmanager mount. No masks were changed.")
        try:
            support = load_add_support(root)
        except ImportError as exc:
            raise RefreshUnavailable(f"Could not import nornir-buildmanager from the mount. ({exc})") from exc
    output = Path(crops)
    record = support.location_from_odata_entity(entity, include_off_edge=True)
    if record is None:
        return None, "no mask geometry"
    polygons = support.hydrate_polygons(support.parse_wkt_polygons(record.wkt), record.type_code)
    if not polygons:
        return None, "no polygon"
    min_x, min_y, max_x, max_y = support.mosaic_bbox(polygons)
    box = (min_x, min_y, max_x, max_y)
    available = index.windows(record.z)
    if not available:
        return None, f"no tile image on disk for section {record.z}"
    touching = [
        window for window in available
        if window.mosaic_rect()[0] < max_x and window.mosaic_rect()[2] > min_x
        and window.mosaic_rect()[1] < max_y and window.mosaic_rect()[3] > min_y
    ]
    if not touching:
        return None, "no tile image covers it"
    if not covers_box(box, [window.mosaic_rect() for window in touching]):
        return None, "tile image missing for part of it"
    for window in touching:
        if not (output / "images" / f"{window.key}.json").is_file():
            return None, f"no sidecar for tile {window.key}"
    jobs = []
    for window in touching:
        job = support.mask_job_for_crop(
            output,
            record.id,
            window.key,
            polygons,
            origin_x=window.x0 * float(window.downsample),
            origin_y=window.y0 * float(window.downsample),
            downsample=float(window.downsample),
            width=window.x1 - window.x0,
            height=window.y1 - window.y0,
        )
        if job is not None:
            jobs.append(job)
    if not jobs:
        return None, "polygon lies outside every tile"
    results = support.run_mask_jobs(jobs, workers=1)
    windows: list[dict[str, Any]] = []
    for position, item in enumerate(sorted(results, key=lambda row: str(row["image_key"]))):
        image_key = str(item["image_key"])
        json_path = output / "images" / f"{image_key}.json"
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        annotations = [row for row in (payload.get("annotations") or []) if int(row.get("id", -1)) != record.id]
        rle = json.loads(Path(str(item["rle_path"])).read_text(encoding="utf-8"))
        annotations.append(
            support.sa1b_annotation(
                record,
                rle=rle,
                bbox=list(item["bbox"]),
                area=int(item["area"]),
                window_index=position,
                window_count=len(results),
            )
        )
        payload["annotations"] = annotations
        json_path.write_text(json.dumps(payload), encoding="utf-8")
        windows.append({"image_key": image_key, "area": int(item["area"])})
    return {"record": record, "windows": windows}, None
