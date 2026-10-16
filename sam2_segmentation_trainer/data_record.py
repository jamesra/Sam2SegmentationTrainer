"""The training-data record: which data a run trained on, kept in the run folder and carried into model exports.

Nothing else says which data a model saw: ``data.root`` in ``config.yaml`` is usually the ``current`` symlink, and the
inventory is rebuilt every epoch, so the data can change under a run. The record names the version folder
(``v3``, ``v1b``) and fingerprints what is in it, so two models trained on "v3" at different times can be told apart.

``<run>/training_data.json`` is written at the start of a run and again after every inventory refresh. An export copies
it into the model manifest (``model_manifest.py``) as ``training_data``, separate from the model's own version.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from sam2_segmentation_trainer.paths import annotation_crops_root

RECORD_FILE = "training_data.json"
"""File name of the record in a run directory."""

_HASHED_FILES = ("manifest.jsonl", "approved.json", "ignore.json")
_CHUNK = 1 << 20


def _sha256(path: Path) -> tuple[str, int] | None:
    """``(hex digest, line count)`` of ``path``, or None when it is absent or unreadable."""
    digest = hashlib.sha256()
    lines = 0
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(_CHUNK):
                digest.update(chunk)
                lines += chunk.count(b"\n")
    except OSError:
        return None
    return digest.hexdigest(), lines


def volume_fingerprint(volume: str, data_dir: Path) -> dict[str, Any]:
    """Digest and line count of a volume's ``manifest.jsonl``, ``approved.json`` and ``ignore.json``.

    A missing file is recorded as null, which is itself information (no approved list, no ignore list).
    """
    crops = annotation_crops_root(volume, data_dir)
    files: dict[str, Any] = {}
    for name in _HASHED_FILES:
        found = _sha256(crops / name)
        files[name] = None if found is None else {"sha256": found[0], "lines": found[1]}
    return files


def build_data_record(
    data_dir: Path,
    volumes: Sequence[str],
    *,
    train_counts: Mapping[str, int],
    val_counts: Mapping[str, int],
    split_seed: int | None = None,
    train_fraction: float | None = None,
) -> dict[str, Any]:
    """The record for ``data_dir``.

    ``set`` is the version folder after resolving a ``current`` symlink; ``requested`` is the name that was asked
    for. ``fingerprint`` (12 hex digits) hashes every file digest, so it changes when any volume's inventory,
    approved list or ignore list changes.
    """
    data_dir = Path(data_dir)
    per_volume = {volume: volume_fingerprint(volume, data_dir) for volume in volumes}
    combined = hashlib.sha256(json.dumps(per_volume, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "schema_version": 1,
        "set": Path(os.path.realpath(data_dir)).name,
        "requested": data_dir.name,
        "fingerprint": combined[:12],
        "volumes": per_volume,
        "train_examples": {key: int(value) for key, value in train_counts.items()},
        "val_examples": {key: int(value) for key, value in val_counts.items()},
        "split_seed": split_seed,
        "train_fraction": train_fraction,
    }


def write_data_record(run_dir: Path, record: Mapping[str, Any]) -> Path:
    """Write ``record`` to ``<run_dir>/training_data.json``."""
    path = Path(run_dir) / RECORD_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(record), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def read_data_record(run_dir: Path) -> dict[str, Any] | None:
    """The run's record, or None when it has none or it cannot be read."""
    path = Path(run_dir) / RECORD_FILE
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return record if isinstance(record, dict) else None


UNKNOWN_DATA = {"schema_version": 1, "set": "unknown", "requested": "unknown", "fingerprint": "unknown"}
"""What a manifest holds for a model whose run has no record (an older run)."""
