"""Same-section merges: one card per object.

A structure can hold two locations of one object on one section (two people traced the same neurite, or a circle was
placed twice). Viking merges such siblings when a polygon is saved (``SameStructureMerge`` in the Viking client): the
same structure, the same section, outlines that overlap in area (edge contact is not an overlap), chains of overlaps
included. The union goes to the survivor, the location with the most links (then the lowest id); the others are deleted
and the survivor gains their links.

The converter applies the same rule to the proposals of a build, so the page shows one card per object:

- the survivor's proposal becomes the union (as a retrace, so it reaches Viking like one);
- the others become ``merged`` cards (``merged_into`` names the survivor). They leave the catalog the page lists, their
  links are drawn to the survivor, and the batch names them for deletion with the survivor's row.

Proposals are compared as the build's masks: pixel centres inside the outline, so outlines that only share an edge do
not overlap. Only cards cut at the same downsample are compared (one build uses one). A part of a merged card outside
the survivor's crop is lost from the union; duplicates sit almost on top of each other, so their crops nearly coincide.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import catalog

LOGGER = logging.getLogger("converter.merge")

MERGED = "merged"
# The union outline is simplified to within this many crop pixels before it replaces the survivor's proposal.
UNION_TOLERANCE = 0.75


def ensure_column(connection: sqlite3.Connection) -> None:
    columns = {row[1] for row in connection.execute("PRAGMA table_info(converter_cards)")}
    if "merged_into" not in columns:
        connection.execute("ALTER TABLE converter_cards ADD COLUMN merged_into INTEGER")


def choose_survivor(members: Sequence[int], link_counts: Dict[int, int]) -> int:
    """Viking's ``LocationSiblingMerge.ChooseSurvivorId``: the most links, then the lowest id."""
    return min(members, key=lambda location_id: (-link_counts.get(location_id, 0), location_id))


def _shift(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """``mask`` moved ``dx`` columns right and ``dy`` rows down into a frame of the same size; what leaves it is dropped."""
    out = np.zeros_like(mask)
    height, width = mask.shape
    if abs(dx) >= width or abs(dy) >= height:
        return out
    src = mask[max(0, -dy):height - max(0, dy), max(0, -dx):width - max(0, dx)]
    out[max(0, dy):max(0, dy) + src.shape[0], max(0, dx):max(0, dx) + src.shape[1]] = src
    return out


def _load_mask(crops: Path, card: sqlite3.Row) -> Optional[np.ndarray]:
    path = crops / "masks" / f"{card['image_key']}_{card['location_id']}.png"
    if not path.is_file():
        return None
    return np.asarray(Image.open(path)) > 127


def _in_frame(mask: np.ndarray, card: sqlite3.Row, frame: sqlite3.Row) -> np.ndarray:
    """``card``'s mask in ``frame``'s crop pixels (both crops are at the same downsample, origins in its pixels)."""
    return _shift(mask, int(card["origin_x"]) - int(frame["origin_x"]), int(card["origin_y"]) - int(frame["origin_y"]))


def _overlap(a: np.ndarray, card_a: sqlite3.Row, b: np.ndarray, card_b: sqlite3.Row) -> bool:
    return bool(np.logical_and(a, _in_frame(b, card_b, card_a)).any())


def find_groups(crops: Path, connection: sqlite3.Connection) -> List[List[int]]:
    """Location ids of every chain of overlapping built proposals on one section, two or more per group."""
    cards = connection.execute(
        "SELECT location_id, image_key, z, downsample, origin_x, origin_y FROM converter_cards "
        "WHERE state = 'built' AND origin_x IS NOT NULL ORDER BY location_id"
    ).fetchall()
    by_section: Dict[Tuple[int, int], List[sqlite3.Row]] = {}
    for card in cards:
        by_section.setdefault((int(card["z"]), int(card["downsample"])), []).append(card)
    groups: List[List[int]] = []
    for section_cards in by_section.values():
        if len(section_cards) < 2:
            continue
        masks = {}
        boxes = {}
        for card in section_cards:
            mask = _load_mask(crops, card)
            if mask is None or not mask.any():
                continue
            rows = np.flatnonzero(mask.any(axis=1))
            cols = np.flatnonzero(mask.any(axis=0))
            x0, y0 = int(card["origin_x"]), int(card["origin_y"])
            masks[card["location_id"]] = (card, mask)
            boxes[card["location_id"]] = (x0 + cols[0], y0 + rows[0], x0 + cols[-1], y0 + rows[-1])
        ids = list(masks)
        parent = {location_id: location_id for location_id in ids}

        def root(location_id: int) -> int:
            while parent[location_id] != location_id:
                parent[location_id] = parent[parent[location_id]]
                location_id = parent[location_id]
            return location_id

        for i, a in enumerate(ids):
            ax0, ay0, ax1, ay1 = boxes[a]
            for b in ids[i + 1:]:
                bx0, by0, bx1, by1 = boxes[b]
                if bx0 > ax1 or bx1 < ax0 or by0 > ay1 or by1 < ay0:
                    continue
                if root(a) != root(b) and _overlap(masks[a][1], masks[a][0], masks[b][1], masks[b][0]):
                    parent[root(a)] = root(b)
        chains: Dict[int, List[int]] = {}
        for location_id in ids:
            chains.setdefault(root(location_id), []).append(location_id)
        groups.extend(sorted(chain) for chain in chains.values() if len(chain) > 1)
    return groups


def _union_rings(mask: np.ndarray) -> List[List[Tuple[float, float]]]:
    """The union mask's outline in crop pixels (pixel-edge coordinates): the largest ring first, then its holes."""
    from converter import contour as contour_module

    padded = np.pad(mask.astype(float), 1)
    rings = contour_module.contour_rings(padded, 0.5)
    if not rings:
        return []
    # contourpy gives (column, row) of sample centres in the padded grid; a pixel's edge is half a sample from its
    # centre and the pad shifts by one, so crop pixel-edge coordinates are the sample coordinates less 0.5.
    rings = [ring - 1 + 0.5 for ring in rings]
    areas = [abs(contour_module.signed_area(ring)) for ring in rings]
    order = int(np.argmax(areas))
    exterior = contour_module.simplify_ring(rings[order], UNION_TOLERANCE)
    holes = [
        contour_module.simplify_ring(ring, UNION_TOLERANCE)
        for index, ring in enumerate(rings)
        if index != order and contour_module._point_in_ring(ring[0], rings[order])
    ]
    return [[(float(x), float(y)) for x, y in ring] for ring in [exterior, *holes]]


def merge_build(crops: Path) -> List[Dict[str, object]]:
    """Fold every group of overlapping proposals into its survivor's card. Returns one entry per merge made."""
    from converter import web

    if not catalog.sqlite_path(crops).is_file():
        return []
    connection = catalog.connect(crops)
    try:
        if connection.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'converter_cards'").fetchone() is None:
            return []
        ensure_column(connection)
        connection.commit()
        groups = find_groups(crops, connection)
        if not groups:
            return []
        link_counts: Dict[int, int] = {}
        if connection.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'converter_links'").fetchone():
            for a, b in connection.execute("SELECT a, b FROM converter_links"):
                link_counts[int(a)] = link_counts.get(int(a), 0) + 1
                link_counts[int(b)] = link_counts.get(int(b), 0) + 1
        plans = []
        for group in groups:
            survivor = choose_survivor(group, link_counts)
            rows = {
                int(row["location_id"]): row
                for row in connection.execute(
                    f"SELECT * FROM converter_cards WHERE state = 'built' AND location_id IN ({','.join('?' * len(group))})", group
                )
            }
            union = _load_mask(crops, rows[survivor]).copy()
            for member in group:
                if member != survivor:
                    union |= _in_frame(_load_mask(crops, rows[member]), rows[member], rows[survivor])
            plans.append((survivor, str(rows[survivor]["image_key"]), [m for m in group if m != survivor], union))
    finally:
        connection.close()

    merges = []
    for survivor, image_key, victims, union in plans:
        rings = _union_rings(union)
        if not rings:
            continue
        try:
            web.apply_edit(crops, survivor, image_key, rings)
        except web.EditRejected as error:
            LOGGER.warning("merge into %s skipped: %s", survivor, error)
            continue
        connection = catalog.connect(crops)
        try:
            connection.executemany(
                "UPDATE converter_cards SET state = ?, reason = ?, merged_into = ? WHERE location_id = ? AND state = 'built'",
                [(MERGED, f"merged into {survivor}", survivor, victim) for victim in victims],
            )
            connection.commit()
        finally:
            connection.close()
        merges.append({"survivor": survivor, "merged": victims})
    if merges:
        LOGGER.info("%s: %d same-section merges", crops, len(merges))
    return merges


def merged_map(crops: Path) -> Dict[int, int]:
    """Merged location id -> its survivor's, for the cards still merged (empty for a build without merges)."""
    if not catalog.sqlite_path(crops).is_file():
        return {}
    connection = catalog.connect(crops)
    try:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(converter_cards)")}
        if "merged_into" not in columns:
            return {}
        return {
            int(row[0]): int(row[1])
            for row in connection.execute(f"SELECT location_id, merged_into FROM converter_cards WHERE state = '{MERGED}'")
        }
    finally:
        connection.close()


def fold_links(links: Sequence[Sequence[int]], merged: Dict[int, int]) -> List[List[int]]:
    """Links with every merged location replaced by its survivor; links inside one merge and repeats are dropped."""
    out = set()
    for a, b in links:
        a, b = merged.get(int(a), int(a)), merged.get(int(b), int(b))
        if a != b:
            out.add((min(a, b), max(a, b)))
    return [list(pair) for pair in sorted(out)]
