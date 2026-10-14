"""The gallery's approved masks, read per volume.

The gallery writes ``approved.json`` into each volume's crops folder (``{data_root}/{volume}/approved.json``, or
under ``AnnotationCrops`` in the nested layout). Entries are ``{"location_id": ..., "image_key": ...}``. An empty
``image_key`` approves the location in every window of that volume. Location ids are only unique within a volume, so
a match always includes the volume.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from sam2_segmentation_trainer.paths import annotation_crops_root

APPROVED_FILE = "approved.json"


@dataclass(frozen=True, slots=True)
class ApprovedSet:
    """Approved masks as ``(volume, location_id, image_key)`` windows and ``(volume, location_id)`` whole locations."""

    windows: frozenset = field(default_factory=frozenset)
    locations: frozenset = field(default_factory=frozenset)

    def __len__(self) -> int:
        return len(self.windows) + len(self.locations)

    def matches(self, volume: str, location_id: int, image_key: str) -> bool:
        return (volume, int(location_id), image_key) in self.windows or (volume, int(location_id)) in self.locations

    def entries_by_volume(self) -> dict[str, int]:
        """Approved entries per volume (a whole-location entry counts once)."""
        counts: dict[str, int] = {}
        for volume, *_ in list(self.windows) + list(self.locations):
            counts[volume] = counts.get(volume, 0) + 1
        return dict(sorted(counts.items()))


def as_approved(approved):
    """An :class:`ApprovedSet` stays as it is; anything else becomes a frozenset of ``(location_id, image_key)``."""
    return approved if isinstance(approved, ApprovedSet) else frozenset(approved or ())


def is_approved(approved, example) -> bool:
    """Whether ``example`` is approved. ``approved`` is an :class:`ApprovedSet`, or a set of ``(location_id,
    image_key)`` pairs that applies to every volume."""
    if not approved:
        return False
    if isinstance(approved, ApprovedSet):
        return approved.matches(example.volume, example.location_id, example.image_key)
    return (int(example.location_id), example.image_key) in approved


def parse_approved(payload) -> tuple[set[tuple[int, str]], set[int]]:
    """Windows and whole locations from the content of one ``approved.json`` (junk entries are skipped)."""
    windows: set[tuple[int, str]] = set()
    whole: set[int] = set()
    for item in payload if isinstance(payload, list) else []:
        try:
            if isinstance(item, dict):
                location_id, image_key = int(item["location_id"]), str(item.get("image_key") or "")
            else:
                location_id, image_key = item
                location_id, image_key = int(location_id), str(image_key or "")
        except (KeyError, TypeError, ValueError):
            continue
        if image_key:
            windows.add((location_id, image_key))
        else:
            whole.add(location_id)
    return windows, whole


def approved_files(raw, volumes: Sequence[str], root: Path) -> list[tuple[str | None, Path]]:
    """The files to read, each with its volume (None when the file does not say: every volume).

    ``raw`` is ``data.approved_path``: null (each configured volume's own ``approved.json``), a file, a directory
    (``{dir}/approved.json`` for a flat file, or ``{dir}/{volume}/approved.json``) or a list of those.
    """
    if not raw:
        return [(volume, annotation_crops_root(volume, root) / APPROVED_FILE) for volume in volumes]
    entries = [raw] if isinstance(raw, (str, Path)) else list(raw)
    files: list[tuple[str | None, Path]] = []
    for entry in entries:
        path = Path(str(entry))
        if path.is_dir():
            per_volume = [(v, annotation_crops_root(v, path) / APPROVED_FILE) for v in volumes]
            found = [(v, p) for v, p in per_volume if p.is_file()]
            if found:
                files.extend(found)
            else:
                files.append((None, path / APPROVED_FILE))
        else:
            # A file named by the user. When it sits in a configured volume's folder it belongs to that volume.
            owner = next(
                (v for v in volumes if path.resolve() == (annotation_crops_root(v, root) / APPROVED_FILE).resolve()),
                None,
            )
            files.append((owner, path))
    return files


def load_approved(raw, volumes: Sequence[str], root: Path, *, log=print) -> ApprovedSet:
    """Read every configured volume's approved masks and log the counts per volume.

    A file that names no volume applies to all configured volumes. Missing files are skipped; a total of 0 warns.
    """
    windows: set[tuple[str, int, str]] = set()
    whole: set[tuple[str, int]] = set()
    per_file: dict[str, int] = {}
    for volume, path in approved_files(raw, volumes, root):
        if not path.is_file():
            if raw:
                log(f"WARNING: data.approved_path {path} not found")
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            log(f"WARNING: could not read {path}: {exc}")
            continue
        file_windows, file_whole = parse_approved(payload)
        for target in (volume,) if volume is not None else tuple(volumes):
            windows.update((target, loc, key) for loc, key in file_windows)
            whole.update((target, loc) for loc in file_whole)
            per_file[target] = per_file.get(target, 0) + len(file_windows) + len(file_whole)
    approved = ApprovedSet(frozenset(windows), frozenset(whole))
    if len(approved) == 0:
        log(
            "WARNING: no approved masks found (0 entries in any volume's approved.json). The Approved validation row "
            "will be empty and approved_weight has no effect."
        )
    else:
        log(f"approved masks: {len(approved)} entries; per volume {dict(sorted(per_file.items()))}")
    return approved


def approved_counts(examples: Iterable, approved) -> dict[str, int]:
    """How many of ``examples`` are approved, per volume."""
    counts: dict[str, int] = {}
    for ex in examples:
        if is_approved(approved, ex):
            counts[ex.volume] = counts.get(ex.volume, 0) + 1
    return dict(sorted(counts.items()))
