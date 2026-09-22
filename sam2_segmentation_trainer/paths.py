"""Container path defaults (SAM2_DATA_ROOT / SAM2_OUTPUT_ROOT / SAM2_CHECKPOINT_ROOT)."""

from __future__ import annotations

import os
import re
from pathlib import Path

DEFAULT_DATA_ROOT = "/storage4"
DEFAULT_OUTPUT_ROOT = "/outputs"
# Large last.pt / best_model writes: CIFS, not the Windows /outputs bind (rename is unsafe there).
DEFAULT_CHECKPOINT_ROOT = "/storage4/Sam2Trainer"
DEFAULT_VOLUMES = ("RC1", "RC2", "RPC1", "RPC2")
DEFAULT_CHECKPOINT_NAME = "sam2.1_hiera_large.pt"


def data_root(explicit: str | os.PathLike[str] | None = None) -> Path:
    if explicit is not None:
        return Path(explicit)
    return Path(os.environ.get("SAM2_DATA_ROOT", DEFAULT_DATA_ROOT))


def output_root(explicit: str | os.PathLike[str] | None = None) -> Path:
    if explicit is not None:
        return Path(explicit)
    return Path(os.environ.get("SAM2_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT))


def checkpoint_root(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Directory for run checkpoints (last.pt, best_model.pt, epoch snaps, TB events)."""
    if explicit is not None:
        return Path(explicit)
    return Path(os.environ.get("SAM2_CHECKPOINT_ROOT", DEFAULT_CHECKPOINT_ROOT))


def pretrained_checkpoint(output: Path | None = None) -> Path:
    root = output if output is not None else output_root()
    return root / "checkpoints" / DEFAULT_CHECKPOINT_NAME


def annotation_crops_root(volume: str, root: Path | None = None) -> Path:
    base = root if root is not None else data_root()
    return base / volume / "AnnotationCrops"


def run_dir_for(run_name: str, *, ckpt_root: Path | None = None) -> Path:
    base = ckpt_root if ckpt_root is not None else checkpoint_root()
    return base / "runs" / str(run_name)


def legacy_run_dir(run_name: str, *, out_root: Path | None = None) -> Path:
    """Previous location under the Windows /outputs bind."""
    base = out_root if out_root is not None else output_root()
    return base / "runs" / str(run_name)


def next_run_name(prefix: str, roots: list[Path]) -> str:
    """Next `prefix_vN` from existing run directories under each root's `runs/` folder."""
    pattern = re.compile(rf"^{re.escape(prefix)}_v(\d+)$")
    highest = 0
    for root in roots:
        runs = root / "runs"
        if not runs.is_dir():
            continue
        for child in runs.iterdir():
            match = pattern.match(child.name)
            if match:
                highest = max(highest, int(match.group(1)))
    return f"{prefix}_v{highest + 1}"
