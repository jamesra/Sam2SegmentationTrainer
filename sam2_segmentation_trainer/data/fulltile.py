"""Tiles that the target fills completely.

A tile with no boundary teaches focal and dice loss one lesson, predict foreground everywhere, and with a mask
prompt it is a pure copy. Production still meets such tiles (the segmentation server walks big cells across
tiles), so training keeps a deterministic fraction of them, gives them only point or circle prompts, and
validation scores them apart from the headline number.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

from sam2_segmentation_trainer.data.manifest import CropExample


def is_full_tile(example: CropExample, threshold: float | None) -> bool:
    """True when the target covers at least ``threshold`` of its tile. Unknown coverage counts as not full."""
    if threshold is None or example.area_frac is None:
        return False
    return example.area_frac >= float(threshold)


def keep_hash(key: str, seed: int) -> float:
    """A stable number in [0, 1) for ``key``: the same location always gets the same draw."""
    digest = hashlib.sha256(f"{seed}|{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def cap_full_tiles(
    examples: Sequence[CropExample],
    *,
    threshold: float | None,
    keep_fraction: float,
    seed: int = 0,
) -> list[CropExample]:
    """Drop all but about ``keep_fraction`` of the full tiles, chosen per location so the choice is repeatable.

    Tiles below ``threshold`` and tiles of unknown coverage are untouched. ``keep_fraction`` 1.0 or a
    ``threshold`` of None keeps everything.
    """
    if threshold is None or keep_fraction >= 1.0:
        return list(examples)
    kept: list[CropExample] = []
    # TODO(v6): the draw depends only on the location and the run seed, so the SAME ~keep_fraction of full tiles
    # is kept every epoch and the rest are never seen in the whole run (v5: 3,809 kept, 12,148 never trained on).
    # For v6, mix the epoch into the draw (e.g. keep_hash(f"{example.split_key}|{epoch}", seed)) so a different
    # quarter is used each epoch: same cost per epoch, every full tile seen over the run. Needs the epoch passed in
    # here and the cap re-applied at each epoch start (train.py calls _cap_full_tiles only at start-up and on an
    # inventory refresh). Do not change this mid-run: it would alter a live run's training set.
    for example in examples:
        if is_full_tile(example, threshold) and keep_hash(example.split_key, seed) >= keep_fraction:
            continue
        kept.append(example)
    return kept
