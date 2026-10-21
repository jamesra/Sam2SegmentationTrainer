"""Frozen validation splits and gold sets: validation data that stays the same from one run to the next.

A run normally builds its train/validation split from the data it sees, so two runs on different data revisions
validate on different locations and their scores cannot be compared. Two files fix that:

* **Frozen split** (``{valsets}/splits/<name>.json``): a run's ``split.json`` saved under a name. A ``zslab`` split
  is defined by its validation and guard sections, so locations added by a later data revision are placed by their
  section (validation, guard or train) and never cross the split. New runs load it with ``data.frozen_split``.
* **Gold set** (``{valsets}/gold/<name>.json``): location ids per volume, drawn at random from a frozen split's
  validation locations and reviewed in the gallery. Training never uses a gold location; validation always scores
  every one and reports them as their own row (``val_subset_gold/...``). Location ids are only unique within a
  volume, so every entry carries its volume.

``{valsets}`` is ``{SAM2_CHECKPOINT_ROOT}/valsets`` (``/storage4/Sam2Trainer/valsets``). Both files are immutable once
published: a frozen split is never rewritten, and a gold set marked ``final`` is never changed; make a new name
(``gold_v1.1``) instead. See ``docs/validation_sets.md``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from sam2_segmentation_trainer.data.splits import SplitLists, load_split
from sam2_segmentation_trainer.paths import checkpoint_root

SCHEMA_VERSION = 1
GOLD_KIND = "gold_set"
SPLIT_KIND = "frozen_split"
NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
GOLD_LISTS = ("candidates", "gold", "rejected", "pending")
"""Per-volume lists in a gold file. ``candidates`` is the random draw; review sorts each candidate into ``gold``
(accepted), ``rejected`` (bad mask) or ``pending`` (not reviewed yet)."""


def valsets_root(explicit: str | Path | None = None) -> Path:
    """``{SAM2_CHECKPOINT_ROOT}/valsets`` unless given."""
    return Path(explicit) if explicit is not None else checkpoint_root() / "valsets"


def check_name(name: str) -> str:
    name = str(name).strip()
    if not NAME_RE.fullmatch(name):
        raise ValueError(f"invalid name {name!r}: use letters, digits, '_', '.', '-' (for example gold_v1)")
    return name


def resolve_split_path(ref: str | Path, root: Path | None = None) -> Path:
    """A frozen split's file from its name (``v5``) or a path to a ``.json`` file."""
    return _resolve(ref, "splits", root)


def resolve_gold_path(ref: str | Path, root: Path | None = None) -> Path:
    """A gold set's file from its name (``gold_v1``) or a path to a ``.json`` file."""
    return _resolve(ref, "gold", root)


def _resolve(ref: str | Path, sub: str, root: Path | None) -> Path:
    text = str(ref).strip()
    if text.endswith(".json") or "/" in text or "\\" in text:
        return Path(text)
    return valsets_root(root) / sub / f"{check_name(text)}.json"


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def split_key(volume: str, location_id: int) -> str:
    """The key splits use for a location: ``{volume}:{location_id}``."""
    return f"{volume}:{int(location_id)}"


def parse_split_key(key: str) -> tuple[str, int]:
    volume, _, location = str(key).rpartition(":")
    return volume, int(location)


def val_locations_by_volume(split: SplitLists) -> dict[str, list[int]]:
    """A split's validation location ids per volume, sorted."""
    out: dict[str, set[int]] = {}
    for key in split.val_keys:
        volume, location = parse_split_key(key)
        out.setdefault(volume, set()).add(location)
    return {volume: sorted(ids) for volume, ids in sorted(out.items())}


# --- frozen splits ---------------------------------------------------------------------------------------------


def freeze_split(
    split: SplitLists,
    path: Path,
    *,
    name: str,
    source: str,
    description: str = "",
) -> dict:
    """Write ``split`` as a frozen split at ``path``. Refuses to overwrite: a published split never changes.

    The payload is the run's ``split.json`` (so :func:`load_split` reads it) plus a ``frozen`` block.
    """
    if path.exists():
        raise FileExistsError(f"{path} exists; a frozen split is never rewritten (choose a new name)")
    payload: dict = {
        "mode": split.mode,
        "train_keys": list(split.train_keys),
        "val_keys": list(split.val_keys),
    }
    if split.val_z:
        payload["val_z"] = {volume: list(zs) for volume, zs in split.val_z.items()}
        payload["guard_z"] = {volume: list(zs) for volume, zs in split.guard_z.items()}
    payload["frozen"] = {
        "kind": SPLIT_KIND,
        "schema_version": SCHEMA_VERSION,
        "name": check_name(name),
        "source": str(source),
        "created_at": now_utc(),
        "description": str(description),
        "val_locations": {v: len(ids) for v, ids in val_locations_by_volume(split).items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return payload["frozen"]


def frozen_info(path: Path) -> dict:
    """The ``frozen`` block of a frozen split (empty for a plain ``split.json``)."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return dict(payload.get("frozen") or {})


def load_frozen_split(path: Path) -> SplitLists:
    if not Path(path).is_file():
        raise FileNotFoundError(f"frozen split {path} not found")
    return load_split(Path(path))


# --- gold sets -------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GoldSet:
    """Accepted gold locations as ``(volume, location_id)`` pairs."""

    name: str = ""
    status: str = "draft"
    split: str = ""
    locations: frozenset = field(default_factory=frozenset)
    pending: int = 0
    """Candidates not reviewed yet (a draft only)."""

    def __len__(self) -> int:
        return len(self.locations)

    def __bool__(self) -> bool:
        return bool(self.locations)

    def contains(self, volume: str, location_id: int) -> bool:
        return (str(volume), int(location_id)) in self.locations

    def matches(self, example) -> bool:
        """Whether a crop example (any object with ``volume`` and ``location_id``) is a gold location."""
        return self.contains(example.volume, example.location_id)

    def by_volume(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for volume, _ in self.locations:
            counts[volume] = counts.get(volume, 0) + 1
        return dict(sorted(counts.items()))


def new_gold_payload(
    *, name: str, split: str, candidates: dict[str, Sequence[int]], per_volume: int, seed: int, description: str = ""
) -> dict:
    stamp = now_utc()
    return {
        "kind": GOLD_KIND,
        "schema_version": SCHEMA_VERSION,
        "name": check_name(name),
        "split": str(split),
        "status": "draft",
        "created_at": stamp,
        "updated_at": stamp,
        "description": str(description),
        "sampling": {"per_volume": int(per_volume), "seed": int(seed)},
        "volumes": {
            volume: {
                "candidates": sorted(int(i) for i in ids),
                "gold": [],
                "rejected": [],
                "pending": sorted(int(i) for i in ids),
            }
            for volume, ids in sorted(candidates.items())
        },
    }


def read_gold_payload(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("kind") != GOLD_KIND:
        raise ValueError(f"{path} is not a gold set (kind {payload.get('kind')!r})")
    if int(payload.get("schema_version", 0)) > SCHEMA_VERSION:
        raise ValueError(f"{path} has schema_version {payload.get('schema_version')}; this code reads {SCHEMA_VERSION}")
    return payload


def write_gold_payload(path: Path, payload: dict, *, create: bool = False) -> None:
    """Write a gold file. ``create`` refuses an existing file; otherwise a ``final`` file on disk is never changed."""
    path = Path(path)
    if path.exists():
        if create:
            raise FileExistsError(f"{path} exists; choose a new name")
        if read_gold_payload(path).get("status") == "final":
            raise PermissionError(f"{path} is final and never changes; make a new gold set (for example _v1.1)")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def gold_from_payload(payload: dict) -> GoldSet:
    locations: set[tuple[str, int]] = set()
    pending = 0
    for volume, lists in (payload.get("volumes") or {}).items():
        locations.update((str(volume), int(i)) for i in lists.get("gold") or ())
        pending += len(lists.get("pending") or ())
    return GoldSet(
        name=str(payload.get("name", "")),
        status=str(payload.get("status", "draft")),
        split=str(payload.get("split", "")),
        locations=frozenset(locations),
        pending=pending,
    )


def load_gold(path: Path) -> GoldSet:
    if not Path(path).is_file():
        raise FileNotFoundError(f"gold set {path} not found")
    return gold_from_payload(read_gold_payload(Path(path)))


def classify_candidates(
    candidates: Iterable[int],
    *,
    windows: dict[int, set[str]],
    approved_windows: set[tuple[int, str]],
    approved_whole: set[int],
    ignored: set[int],
) -> dict[str, list[int]]:
    """Sort one volume's candidates by their review in the gallery.

    ``rejected``: the location is in ``ignore.json``, or it has no crop windows left in the data. ``gold``: the
    whole location is approved, or every one of its current windows is. ``pending``: anything else (not reviewed,
    or only some windows approved). ``windows`` maps a location id to its current image keys.
    """
    out: dict[str, list[int]] = {"gold": [], "rejected": [], "pending": []}
    for location in sorted({int(c) for c in candidates}):
        keys = windows.get(location, set())
        if location in ignored or not keys:
            out["rejected"].append(location)
        elif location in approved_whole or all((location, key) in approved_windows for key in keys):
            out["gold"].append(location)
        else:
            out["pending"].append(location)
    return out


def partition_with_gold(gold: GoldSet | None, train_ex: Sequence, val_ex: Sequence, *, log=print) -> tuple[list, list]:
    """Keep gold locations out of training, and say which gold locations validation cannot score.

    A gold location in the training partition (it should not happen when the gold set was drawn from the run's
    frozen split) is dropped from training with a warning; a gold location in neither partition (guard section,
    or gone from the data) is reported.
    """
    train_ex, val_ex = list(train_ex), list(val_ex)
    if not gold:
        return train_ex, val_ex
    leaked = [ex for ex in train_ex if gold.matches(ex)]
    if leaked:
        names = sorted({split_key(ex.volume, ex.location_id) for ex in leaked})
        log(
            f"WARNING: {len(names)} gold location(s) of gold set {gold.name!r} are in the training split; they are "
            f"left out of training (first: {names[:5]}). Draw gold sets from the run's frozen split."
        )
        train_ex = [ex for ex in train_ex if not gold.matches(ex)]
    in_val = {(ex.volume, int(ex.location_id)) for ex in val_ex}
    unscored = sorted(set(gold.locations) - in_val)
    if unscored:
        log(
            f"WARNING: {len(unscored)} gold location(s) are not in the validation split and will not be scored "
            f"(first: {[split_key(v, i) for v, i in unscored[:5]]})"
        )
    return train_ex, val_ex
