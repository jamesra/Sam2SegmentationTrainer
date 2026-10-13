"""Put the trainer's ``annotation_crops`` package on ``sys.path`` when it sits beside this folder."""

from __future__ import annotations

import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
_repo = _here.parent
if (_repo / "annotation_crops" / "__init__.py").is_file():
    root = str(_repo)
    if root not in sys.path:
        sys.path.insert(0, root)

LEGACY_CROPS_SUBDIR = "AnnotationCrops"
_FLAT_MARKERS = ("manifest.jsonl", "annotation_crops.sqlite")


def crops_dir_for(volume_root: Path) -> Path:
    """Folder holding a volume's manifest, images/, and masks/.

    Training versions are flat (``{volume}/manifest.jsonl``). Older mirrors and the NAS export
    nest everything under ``{volume}/AnnotationCrops``. The flat layout wins when the volume
    folder holds a manifest or catalog; otherwise the nested path is returned, whether or not
    it exists yet.
    """
    volume_root = Path(volume_root)
    if any((volume_root / name).is_file() for name in _FLAT_MARKERS):
        return volume_root
    return volume_root / LEGACY_CROPS_SUBDIR
