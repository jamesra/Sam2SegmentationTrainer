"""One-location mask refresh through a read-only nornir-buildmanager mount.

The gallery image does not install Nornir. This module imports the export
wrapper only when that checkout is mounted. A missing mount raises
``RefreshUnavailable`` and does not write masks.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

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
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    try:
        from nornir_buildmanager.operations.segmentationtraining.ingest import (
            location_from_odata_entity,
        )
        from nornir_buildmanager.operations.segmentationtraining.update import (
            refresh_existing_location_masks,
        )
    except ImportError as exc:
        raise RefreshUnavailable(
            "Mask refresh could not import nornir-buildmanager from the mount. "
            "No masks were changed."
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
