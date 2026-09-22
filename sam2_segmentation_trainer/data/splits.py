"""90/10 train/val split keyed by (volume, locationId), stratified by volume."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from sam2_segmentation_trainer.data.manifest import CropExample


@dataclass(frozen=True, slots=True)
class SplitLists:
    train_keys: tuple[str, ...]
    val_keys: tuple[str, ...]

    def partition(
        self, examples: Sequence[CropExample]
    ) -> tuple[list[CropExample], list[CropExample]]:
        train_set = set(self.train_keys)
        val_set = set(self.val_keys)
        train = [ex for ex in examples if ex.split_key in train_set]
        val = [ex for ex in examples if ex.split_key in val_set]
        return train, val


def make_location_split(
    examples: Sequence[CropExample],
    *,
    train_fraction: float = 0.9,
    seed: int = 42,
    even_per_volume: bool = False,
) -> SplitLists:
    by_volume: dict[str, list[str]] = {}
    for ex in examples:
        by_volume.setdefault(ex.volume, [])
        if ex.split_key not in by_volume[ex.volume]:
            by_volume[ex.volume].append(ex.split_key)

    per_train: dict[str, list[str]] = {}
    per_val: dict[str, list[str]] = {}
    rng = random.Random(seed)
    for volume in sorted(by_volume):
        keys = list(by_volume[volume])
        rng.shuffle(keys)
        if not keys:
            continue
        n_train = int(round(len(keys) * train_fraction))
        n_train = min(max(n_train, 0), len(keys))
        # Keep at least one val example when a volume has two or more keys.
        if len(keys) >= 2 and n_train == len(keys):
            n_train = len(keys) - 1
        if len(keys) >= 2 and n_train == 0:
            n_train = 1
        per_train[volume] = keys[:n_train]
        per_val[volume] = keys[n_train:]

    if even_per_volume:
        if not per_train:
            raise RuntimeError("even split requested but no volumes were present")
        cap_train = min(len(keys) for keys in per_train.values())
        cap_val = min(len(keys) for keys in per_val.values())
        if cap_train < 1 or cap_val < 1:
            raise RuntimeError(
                "even split needs at least one train and one val example in every volume; "
                f"train cap={cap_train} val cap={cap_val}"
            )
        for volume in per_train:
            per_train[volume] = per_train[volume][:cap_train]
            per_val[volume] = per_val[volume][:cap_val]

    train_keys: list[str] = []
    val_keys: list[str] = []
    for volume in sorted(per_train):
        train_keys.extend(per_train[volume])
        val_keys.extend(per_val[volume])
    return SplitLists(train_keys=tuple(train_keys), val_keys=tuple(val_keys))


def save_split(split: SplitLists, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train_keys": list(split.train_keys),
        "val_keys": list(split.val_keys),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_split(path: Path) -> SplitLists:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return SplitLists(
        train_keys=tuple(payload["train_keys"]),
        val_keys=tuple(payload["val_keys"]),
    )
