"""Train/val splits keyed by (volume, locationId).

Two modes:

* ``location``: 90/10 at random over location keys inside each volume (the original split).
* ``zslab``: validation is a few contiguous runs of sections ("z-slabs") spaced evenly through each volume, with a
  guard band of sections around every slab that neither side uses. A location lives in one section, so every
  crop window of a section lands on one side: neither nearly identical adjacent sections nor overlapping windows
  of different locations cross the split.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from sam2_segmentation_trainer.data.manifest import CropExample


@dataclass(frozen=True, slots=True)
class SplitLists:
    train_keys: tuple[str, ...]
    val_keys: tuple[str, ...]
    val_z: dict[str, tuple[int, ...]] = field(default_factory=dict)
    """``zslab`` mode: validation sections per volume. Empty for a ``location`` split."""
    guard_z: dict[str, tuple[int, ...]] = field(default_factory=dict)
    """``zslab`` mode: guard sections per volume, used for neither training nor validation."""

    @property
    def mode(self) -> str:
        return "zslab" if self.val_z else "location"

    def partition(
        self, examples: Sequence[CropExample]
    ) -> tuple[list[CropExample], list[CropExample]]:
        train_set = set(self.train_keys)
        val_set = set(self.val_keys)
        train = [ex for ex in examples if ex.split_key in train_set]
        val = [ex for ex in examples if ex.split_key in val_set]
        return train, val


def _train_count(n: int, fraction: float) -> int:
    """How many of *n* keys go to train.

    A group of two or more keeps at least one train key and one val key.
    """
    n_train = int(round(n * fraction))
    n_train = min(max(n_train, 0), n)
    if n >= 2 and n_train == n:
        n_train = n - 1
    if n >= 2 and n_train == 0:
        n_train = 1
    return n_train


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
        n_train = _train_count(len(keys), train_fraction)
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


def include_new_keys(
    split: SplitLists,
    examples: Sequence[CropExample],
    *,
    train_fraction: float,
    seed: int,
) -> SplitLists:
    """Add location keys that were not in *split* when the inventory was first built.

    Existing train and val membership stays put. New keys are assigned with *seed*
    so a location restored after startup joins the same side on later epochs.
    """
    if split.val_z:
        return _include_new_keys_by_z(split, examples)
    known = set(split.train_keys) | set(split.val_keys)
    fresh: list[str] = []
    seen: set[str] = set()
    for example in examples:
        if example.split_key in known or example.split_key in seen:
            continue
        seen.add(example.split_key)
        fresh.append(example.split_key)
    if not fresh:
        return split
    rng = random.Random(seed)
    rng.shuffle(fresh)
    n_train = _train_count(len(fresh), train_fraction)
    return SplitLists(
        train_keys=split.train_keys + tuple(fresh[:n_train]),
        val_keys=split.val_keys + tuple(fresh[n_train:]),
    )


def _include_new_keys_by_z(split: SplitLists, examples: Sequence[CropExample]) -> SplitLists:
    """Place keys that appeared after the split by their section: validation slab, guard (dropped) or train."""
    known = set(split.train_keys) | set(split.val_keys)
    val_z = {volume: set(zs) for volume, zs in split.val_z.items()}
    guard_z = {volume: set(zs) for volume, zs in split.guard_z.items()}
    add_train: list[str] = []
    add_val: list[str] = []
    seen: set[str] = set()
    for example in examples:
        key = example.split_key
        if key in known or key in seen:
            continue
        seen.add(key)
        z = int(example.z)
        if z in val_z.get(example.volume, ()):
            add_val.append(key)
        elif z not in guard_z.get(example.volume, ()):
            add_train.append(key)
    if not add_train and not add_val:
        return split
    return SplitLists(
        train_keys=split.train_keys + tuple(add_train),
        val_keys=split.val_keys + tuple(add_val),
        val_z=split.val_z,
        guard_z=split.guard_z,
    )


def make_zslab_split(
    examples: Sequence[CropExample],
    *,
    val_fraction: float = 0.1,
    slab_count: int = 5,
    guard_sections: int = 1,
) -> SplitLists:
    """Validation = ``slab_count`` contiguous z-slabs spaced evenly through each volume, about ``val_fraction`` of
    its examples (crop-window samples, not locations) in total.

    Slab ``i`` is centered at ``(i + 0.5) / slab_count`` of the volume's z-range and grows one section at a time
    (the side nearer the center first) until it holds about ``val_fraction / slab_count`` of the volume's examples,
    stopping at whichever whole-section size is nearest that share.
    Sections are those that have examples; a gap in z is skipped. ``guard_sections`` z values on each side of a
    slab are excluded from both sides. Slabs never grow into another slab or its guard band. Deterministic: no
    seed is involved.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    if slab_count < 1:
        raise ValueError(f"slab_count must be >= 1, got {slab_count}")
    guard = max(0, int(guard_sections))
    keys_by_volume_z: dict[str, dict[int, list[str]]] = {}
    samples_by_volume_z: dict[str, dict[int, int]] = {}
    seen: set[str] = set()
    for example in examples:
        z = int(example.z)
        per_z = samples_by_volume_z.setdefault(example.volume, {})
        per_z[z] = per_z.get(z, 0) + 1  # every crop window is a training sample
        if example.split_key in seen:
            continue
        seen.add(example.split_key)
        keys_by_volume_z.setdefault(example.volume, {}).setdefault(z, []).append(example.split_key)

    train_keys: list[str] = []
    val_keys: list[str] = []
    val_z: dict[str, tuple[int, ...]] = {}
    guard_z: dict[str, tuple[int, ...]] = {}
    for volume in sorted(keys_by_volume_z):
        by_z = keys_by_volume_z[volume]
        slabs = _place_slabs(
            samples_by_volume_z[volume],
            val_fraction=val_fraction,
            slab_count=slab_count,
            guard=guard,
        )
        v_z = sorted({z for lo, hi in slabs for z in by_z if lo <= z <= hi})
        g_z = sorted(
            {
                z
                for lo, hi in slabs
                for z in list(range(lo - guard, lo)) + list(range(hi + 1, hi + guard + 1))
            }
            - set(v_z)
        )
        val_z[volume] = tuple(v_z)
        guard_z[volume] = tuple(g_z)
        v_set, g_set = set(v_z), set(g_z)
        for z in sorted(by_z):
            if z in v_set:
                val_keys.extend(by_z[z])
            elif z not in g_set:
                train_keys.extend(by_z[z])
    return SplitLists(
        train_keys=tuple(train_keys), val_keys=tuple(val_keys), val_z=val_z, guard_z=guard_z
    )


def _place_slabs(
    counts: dict[int, int],
    *,
    val_fraction: float,
    slab_count: int,
    guard: int,
) -> list[tuple[int, int]]:
    """``[(lo_z, hi_z), ...]`` inclusive slab bounds for one volume. ``counts`` maps section z to example count."""
    sections = sorted(counts)
    if not sections:
        return []
    total = sum(counts.values())
    target = val_fraction * total / slab_count
    z_min, z_max = sections[0], sections[-1]
    slabs: list[tuple[int, int]] = []

    def clear(z: int) -> bool:
        return all(z < lo - guard or z > hi + guard for lo, hi in slabs)

    for i in range(slab_count):
        center = z_min + (i + 0.5) / slab_count * (z_max - z_min)
        candidates = [z for z in sections if clear(z)]
        if not candidates:
            break
        start = min(candidates, key=lambda z: (abs(z - center), z))
        index = sections.index(start)
        lo_i = hi_i = index
        held = counts[start]
        while held < target:
            below = sections[lo_i - 1] if lo_i > 0 else None
            above = sections[hi_i + 1] if hi_i + 1 < len(sections) else None
            options = [z for z in (below, above) if z is not None and clear(z)]
            if not options:
                break
            nxt = min(options, key=lambda z: (abs(z - center), z))
            if held + counts[nxt] - target > target - held:
                break  # stopping here lands nearer the target than taking one more section
            if nxt == below:
                lo_i -= 1
            else:
                hi_i += 1
            held += counts[nxt]
        slabs.append((sections[lo_i], sections[hi_i]))
    return sorted(slabs)


def save_split(split: SplitLists, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "mode": split.mode,
        "train_keys": list(split.train_keys),
        "val_keys": list(split.val_keys),
    }
    if split.val_z:
        payload["val_z"] = {volume: list(zs) for volume, zs in split.val_z.items()}
        payload["guard_z"] = {volume: list(zs) for volume, zs in split.guard_z.items()}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_split(path: Path) -> SplitLists:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return SplitLists(
        train_keys=tuple(payload["train_keys"]),
        val_keys=tuple(payload["val_keys"]),
        val_z={volume: tuple(int(z) for z in zs) for volume, zs in (payload.get("val_z") or {}).items()},
        guard_z={volume: tuple(int(z) for z in zs) for volume, zs in (payload.get("guard_z") or {}).items()},
    )
