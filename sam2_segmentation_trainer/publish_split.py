"""Publish a run's validation / training split beside each volume's crops, for the gallery's Split filter.

    python -m sam2_segmentation_trainer.publish_split sam2_em_v5 [--root /data-local/v3]

Reads the run's ``split.json`` and ``config.yaml`` (for ``data.root`` and ``data.split_mode``) and writes
``validation_split.json`` into each volume's crops folder (see :mod:`annotation_crops.validation_split`). A running
trainer writes the same files itself from the next restart or inventory rebuild on; this is for a run already going.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from annotation_crops.validation_split import ids_by_volume, write_split_file
from sam2_segmentation_trainer.paths import annotation_crops_root, data_root, run_dir_for


def publish(run: str, root: Path | None = None) -> dict[str, tuple[int, int]]:
    """Write the files; returns ``{volume: (validation locations, training locations)}``."""
    from omegaconf import OmegaConf

    run_dir = run_dir_for(run)
    split = json.loads((run_dir / "split.json").read_text(encoding="utf-8"))
    cfg = OmegaConf.load(run_dir / "config.yaml")
    root = Path(root) if root is not None else data_root(OmegaConf.select(cfg, "data.root", default=None))
    mode = str(split.get("mode") or OmegaConf.select(cfg, "data.split_mode", default="location"))
    val = ids_by_volume(split.get("val_keys") or [])
    train = ids_by_volume(split.get("train_keys") or [])
    out: dict[str, tuple[int, int]] = {}
    for volume in sorted(set(val) | set(train)):
        crops = annotation_crops_root(volume, root)
        if not crops.is_dir():
            print(f"{volume}: no crops folder at {crops}; skipped")
            continue
        path = write_split_file(crops, run=run, mode=mode, validation=val.get(volume, ()), training=train.get(volume, ()))
        out[volume] = (len(val.get(volume, ())), len(train.get(volume, ())))
        print(f"{volume}: {out[volume][0]} validation, {out[volume][1]} training locations -> {path}")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", help="Run name, e.g. sam2_em_v5")
    parser.add_argument("--root", type=Path, default=None, help="Data root (default: the run's data.root)")
    args = parser.parse_args(argv)
    publish(args.run, args.root)


if __name__ == "__main__":
    main()
