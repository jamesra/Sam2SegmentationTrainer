"""Drawing a gold-standard sample: which locations, in what reproducible order. Pure functions, no files.

The *order* is the primitive. A seed, the stratification and the source pool fully determine one ordering of the whole pool;
the sample is its first ``n`` ids and a replacement for a rejected card is the next unused id. Re-computing the order from
the same seed and pool therefore reproduces the sample and every replacement.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

DEFAULT_COUNT = 100
DEFAULT_Z_BAND = 25
STRATIFY_CHOICES = ("", "type", "z")


@dataclass
class Pool:
    """The locations a gold set may draw from, and what was left out and why."""

    ids: List[int] = field(default_factory=list)
    rows: Dict[int, Dict[str, Any]] = field(default_factory=dict)  # one row per location (first window), for strata
    windows: Dict[int, List[Dict[str, Any]]] = field(default_factory=dict)  # every usable window of each location
    multi: List[int] = field(default_factory=list)  # locations with more than one crop window (one card is chosen)
    rejected: int = 0
    no_mask: List[int] = field(default_factory=list)


def build_pool(rows: Iterable[Dict[str, Any]]) -> Pool:
    """Pool of locations from a source catalog: not rejected and with at least one window that has a mask.

    A location spans several crop windows when it is large (about half of RC1's). Excluding those would bias the gold set
    to small profiles, so they stay in the pool and contribute one window each (see ``choose_window``).
    """
    by_location: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        by_location.setdefault(int(row["location_id"]), []).append(row)
    pool = Pool()
    for location_id in sorted(by_location):
        windows = by_location[location_id]
        if any(row.get("ignored") for row in windows):
            pool.rejected += 1
            continue
        usable = [row for row in windows if row.get("mask_relpath") and row.get("image_key")]
        if not usable:
            pool.no_mask.append(location_id)
            continue
        pool.ids.append(location_id)
        pool.rows[location_id] = usable[0]
        pool.windows[location_id] = usable
        if len(usable) > 1:
            pool.multi.append(location_id)
    return pool


def fingerprint(ids: Sequence[int]) -> str:
    """Identifies a pool: the same ids in any order give the same fingerprint."""
    digest = hashlib.sha1(",".join(str(i) for i in sorted(ids)).encode("ascii")).hexdigest()
    return digest[:16]


def _stratum_of(row: Dict[str, Any], stratify: str, band: int) -> Any:
    if stratify == "type":
        return row.get("type_id") if row.get("type_id") is not None else (row.get("type_name") or "")
    if stratify == "z":
        return int(row.get("z") or 0) // max(1, band)
    return 0


def order(pool: Pool, seed: int, stratify: str = "", band: int = DEFAULT_Z_BAND) -> List[int]:
    """Every id of the pool in drawing order.

    Without stratification: a seeded shuffle. With it: each stratum is shuffled on its own, then the items are merged by
    their relative position ``(rank + u) / stratum size``, with ``u`` a seeded offset per stratum, so the first ``n`` ids
    contain each stratum in proportion to its size, and later ids (replacements) keep the proportions.
    """
    if stratify not in STRATIFY_CHOICES:
        raise ValueError(f"stratify must be one of {STRATIFY_CHOICES}")
    rng = random.Random(f"gold:{seed}:{stratify}:{band}")
    if not stratify:
        ids = list(pool.ids)
        rng.shuffle(ids)
        return ids
    strata: Dict[Any, List[int]] = {}
    for location_id in pool.ids:
        strata.setdefault(_stratum_of(pool.rows[location_id], stratify, band), []).append(location_id)
    keyed: List[Tuple[float, int, int]] = []
    for position, key in enumerate(sorted(strata, key=lambda value: (str(type(value)), value))):
        members = strata[key]
        rng.shuffle(members)
        offset = rng.random()
        size = len(members)
        for rank, location_id in enumerate(members):
            keyed.append(((rank + offset) / size, position, location_id))
    keyed.sort()
    return [location_id for _, _, location_id in keyed]


def sample(pool: Pool, n: int, seed: int, stratify: str = "", band: int = DEFAULT_Z_BAND) -> List[int]:
    return order(pool, seed, stratify, band)[: max(0, int(n))]


def next_unused(ordered: Sequence[int], used: Iterable[int], count: int = 1) -> List[int]:
    taken = set(used)
    return [location_id for location_id in ordered if location_id not in taken][: max(0, count)]


MIN_WINDOW_SHARE = 0.25


def choose_window(seed: int, location_id: int, windows: Sequence[Dict[str, Any]], mask_pixels: Callable[[Dict[str, Any]], int]) -> Dict[str, Any]:
    """The window a location's gold card is made from.

    A location with one window uses it. With several, the windows holding at least a quarter of the largest window's mask
    are candidates (a sliver at a crop's edge is not worth a person's time) and the seed picks one; the choice depends only
    on the seed, the id and the masks, so it can be re-derived.
    """
    if len(windows) == 1:
        return windows[0]
    ordered = sorted(windows, key=lambda row: str(row["image_key"]))
    counts = [mask_pixels(row) for row in ordered]
    peak = max(counts)
    candidates = [row for row, count in zip(ordered, counts) if peak > 0 and count >= MIN_WINDOW_SHARE * peak] or ordered
    return random.Random(f"gold-window:{seed}:{location_id}").choice(candidates)


_ID = re.compile(r"[0-9]+")
_RANGE = re.compile(r"([0-9]+)-([0-9]+)")
# "5 - 8" is the range 5-8; "0 -4" stays two tokens, an id and a junk "-4".
_RANGE_SPACES = re.compile(r"(?<=[0-9])\s+-\s+(?=[0-9])")


def parse_id_list(text: str, limit: int = 100000) -> Tuple[List[int], List[str]]:
    """Location ids from a .txt: separated by newlines, commas, spaces or semicolons; ``#`` starts a comment.

    Follows the Viking / nornir id-list convention (``nornir_shared.argparse_helpers.IntegerList``): a hyphen between
    two ids is an inclusive range, so ``1, 3, 5-8`` is 1, 3, 5, 6, 7 and 8. A backwards range (``8-5``) is not ids.

    Returns ``(ids, bad)``: ids in order of first appearance without repeats, and the tokens that were not ids, so an upload
    never silently drops a line. More than ``limit`` ids (a range counts every id in it) raises ``ValueError``.
    """
    ids: List[int] = []
    seen = set()
    bad: List[str] = []

    def add(value: int) -> None:
        if value not in seen:
            if len(ids) >= limit:
                raise ValueError(f"more than {limit} ids")
            seen.add(value)
            ids.append(value)

    for line in text.splitlines():
        line = _RANGE_SPACES.sub("-", line.split("#", 1)[0])
        for token in re.split(r"[,\s;]+", line.strip()):
            if not token:
                continue
            span = _RANGE.fullmatch(token)
            if _ID.fullmatch(token) and int(token) > 0:
                add(int(token))
            elif span and 0 < int(span.group(1)) <= int(span.group(2)):
                lo, hi = int(span.group(1)), int(span.group(2))
                if hi - lo + 1 > limit:
                    raise ValueError(f"more than {limit} ids")
                for value in range(lo, hi + 1):
                    add(value)
            else:
                bad.append(token)
    return ids, bad
