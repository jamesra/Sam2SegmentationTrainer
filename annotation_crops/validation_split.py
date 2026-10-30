"""``validation_split.json``: which locations of a volume a training run validates on and which it trains on.

A training run writes one beside each volume's crops whenever it records its training data (start, and each
inventory rebuild), and ``python -m sam2_segmentation_trainer.publish_split RUN`` writes them from a run's
``split.json`` (for a run that started before this existed), so the gallery can filter cards by split without
reading the run folder. A location in neither list is used by neither side: a guard section beside a validation
slab, or added after the run started.

    {"run": "sam2_em_v5", "mode": "zslab", "writtenAt": "2026-10-30T07:40:00Z",
     "validation": [189464, ...], "training": [1430316, ...]}
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

FILE_NAME = "validation_split.json"


def split_path(crops: str | os.PathLike[str]) -> Path:
    return Path(crops) / FILE_NAME


def write_split_file(
    crops: str | os.PathLike[str],
    *,
    run: str,
    mode: str,
    validation: Iterable[int],
    training: Iterable[int],
) -> Path:
    """Write the file atomically (temp file, then ``os.replace``) so the gallery never reads half of it."""
    payload = {
        "run": str(run),
        "mode": str(mode),
        "writtenAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "validation": sorted({int(i) for i in validation}),
        "training": sorted({int(i) for i in training}),
    }
    path = split_path(crops)
    temp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        temp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        os.replace(temp, path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    return path


def read_split_file(crops: str | os.PathLike[str]) -> dict[str, Any] | None:
    """The file's contents, or None when it is missing or unreadable (the gallery then offers no split filter)."""
    path = split_path(crops)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return {
            "run": str(payload.get("run") or ""),
            "mode": str(payload.get("mode") or ""),
            "writtenAt": str(payload.get("writtenAt") or ""),
            "validation": [int(i) for i in payload.get("validation") or []],
            "training": [int(i) for i in payload.get("training") or []],
        }
    except (TypeError, ValueError):
        return None


def ids_by_volume(keys: Iterable[str]) -> dict[str, set[int]]:
    """``{"RC1": {189464, ...}}`` from split keys such as ``"RC1:189464"`` (a run's ``split.json``)."""
    out: dict[str, set[int]] = {}
    for key in keys:
        volume, _, location = str(key).rpartition(":")
        if volume and location.isdigit():
            out.setdefault(volume, set()).add(int(location))
    return out
