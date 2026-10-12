"""One-location mask refresh through a read-only nornir-buildmanager mount.

The gallery image does not install Nornir. ``segmentationtraining`` no longer
imports the export pipeline at package import, so this module can load
``maskrefresh`` without stubbing that package. The volume-pipeline package
inits (``nornir_buildmanager`` and ``operations``) still are not executed:
they import build, tile, and imageregistration. A missing mount raises
``RefreshUnavailable`` and does not write masks.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Any, Callable

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
    """Delete cached card JPEGs so the next paint rebuilds them from the new mask."""
    root = Path(crops)
    for key in image_keys:
        if IMAGE_KEY.fullmatch(str(key)) is None:
            continue
        for size in THUMB_SIZES:
            path = root / "overlays" / "thumb" / str(size) / f"{key}.jpg"
            path.unlink(missing_ok=True)
