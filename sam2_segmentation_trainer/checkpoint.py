"""Atomic checkpoint I/O and deterministic epoch index slicing."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import torch


def tmp_path(path: Path) -> Path:
    return path.with_name(path.name + ".tmp")


def bak_path(path: Path) -> Path:
    return path.with_name(path.name + ".bak")


def _fsync_file(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_torch_save(obj: Any, path: Path | str) -> None:
    """Write via .tmp, keep previous good file as .bak, then replace.

    Prefer a CIFS or Linux filesystem for large checkpoints. On Windows bind
    mounts, rename is not POSIX-atomic; the .bak slot is the recoverable copy
    if replace is interrupted.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = tmp_path(dest)
    bak = bak_path(dest)
    torch.save(obj, tmp)
    _fsync_file(tmp)
    if dest.is_file():
        bak.unlink(missing_ok=True)
        dest.replace(bak)
        try:
            _fsync_file(bak)
        except OSError:
            pass
    tmp.replace(dest)
    try:
        _fsync_file(dest)
    except OSError:
        pass
    try:
        dir_fd = os.open(str(dest.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass


def load_latest_checkpoint(path: Path | str, map_location: str | torch.device = "cpu") -> Any:
    """Load path, then .bak if the primary is missing or unreadable. Never load .tmp."""
    dest = Path(path)
    errors: list[Exception] = []
    for candidate in (dest, bak_path(dest)):
        if not candidate.is_file():
            continue
        try:
            return torch.load(candidate, map_location=map_location, weights_only=False)
        except Exception as exc:  # noqa: BLE001 — corrupt checkpoint is expected
            errors.append(exc)
    if errors:
        raise RuntimeError(
            f"could not load checkpoint {dest} or {bak_path(dest)}: {errors[-1]}"
        ) from errors[-1]
    return None


def migrate_run_artifacts(legacy_dir: Path, run_dir: Path) -> list[str]:
    """Copy resume/weights files from a legacy Windows-bind run dir onto CIFS.

    Only copies when the destination file is missing so an active CIFS run is
    not overwritten by a stale Windows copy.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    if not legacy_dir.is_dir():
        return []
    try:
        if legacy_dir.resolve() == run_dir.resolve():
            return []
    except OSError:
        pass
    moved: list[str] = []
    names = [
        "last.pt",
        "last.pt.bak",
        "best_model.pt",
        "config.yaml",
    ]
    names.extend(path.name for path in legacy_dir.glob("checkpoint_epoch*.pt"))
    for name in names:
        src = legacy_dir / name
        dst = run_dir / name
        if src.is_file() and not dst.is_file():
            shutil.copy2(src, dst)
            moved.append(name)
    return moved


def epoch_sample_indices(n: int, seed: int, epoch: int) -> list[int]:
    generator = torch.Generator()
    generator.manual_seed(int(seed) + int(epoch) * 1009)
    return torch.randperm(n, generator=generator).tolist()


def remaining_epoch_indices(
    n: int,
    seed: int,
    epoch: int,
    step_in_epoch: int,
    batch_size: int,
    *,
    drop_last: bool = True,
) -> list[int]:
    order = epoch_sample_indices(n, seed, epoch)
    start = int(step_in_epoch) * int(batch_size)
    rest = order[start:]
    if drop_last:
        usable = (len(rest) // int(batch_size)) * int(batch_size)
        rest = rest[:usable]
    return rest
