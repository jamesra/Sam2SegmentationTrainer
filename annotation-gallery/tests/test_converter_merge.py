"""converter.merge: overlapping proposals of one structure on one section fold into one card, as Viking merges them."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

GALLERY_ROOT = Path(__file__).resolve().parents[1]
for folder in (GALLERY_ROOT, Path(__file__).resolve().parent):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import test_converter_builder as fixtures  # noqa: E402
from converter import builder as builder_module  # noqa: E402
from converter import merge, web  # noqa: E402
import catalog  # noqa: E402


def _build(tmp_path: Path, discs, links=()):
    """Cards for ``discs`` {id: (x, y, r)} on one section; the segmenter answers each with its disc."""
    crops = tmp_path / "structure-500" / "RC1"
    volume = fixtures.parse_viking_xml(fixtures.VIKING_XML, source="x")
    entities = [fixtures._entity(i, 1, f"POINT ({x} {y})", radius=r) for i, (x, y, r) in discs.items()]
    truth = {i: fixtures._disc(x, y, r) for i, (x, y, r) in discs.items()}
    segmenter = fixtures.FakeSegmenter(lambda location_id: truth[location_id], size=256)
    fixtures._remember(segmenter, entities)
    params = builder_module.BuildParams(volume="RC1", odata_base="http://odata.test/RC1", structure_id=500)

    def get_json(url):
        if "StructureLocationLinks" in url:
            return {"value": [{"A": a, "B": b} for a, b in links]}
        return {"@odata.count": len(entities), "value": entities}

    builder = builder_module.Builder(params, crops, volume, segmenter, fixtures.TileCache(None, fixtures.tile_http), get_json=get_json)
    builder.run()
    return crops


def _cards(crops: Path):
    return {row["location_id"]: row for row in web.card_rows(crops)}


def _mask_area(crops: Path, location_id: int) -> int:
    key = _cards(crops)[location_id]["image_key"]
    return int((np.asarray(Image.open(crops / "masks" / f"{key}_{location_id}.png")) > 127).sum())


def test_overlapping_proposals_fold_into_the_survivor_with_their_union(tmp_path: Path) -> None:
    crops = _build(tmp_path, {21: (1400, 1600, 80), 22: (1440, 1620, 80), 23: (1800, 1900, 60)})
    cards = _cards(crops)
    assert cards[21]["state"] == "built" and cards[23]["state"] == "built"
    assert cards[22]["state"] == merge.MERGED and cards[22]["reason"] == "merged into 21"
    assert merge.merged_map(crops) == {22: 21}
    # The survivor's proposal is the union: larger than one disc, and it is a retrace so it reaches Viking like one.
    assert _mask_area(crops, 21) > _mask_area(crops, 23) * (80 / 60) ** 2 * 1.2
    assert cards[21]["reason"] == "retraced"


def test_the_survivor_is_the_location_with_the_most_links(tmp_path: Path) -> None:
    crops = _build(tmp_path, {21: (1400, 1600, 80), 22: (1440, 1620, 80), 30: (1800, 1900, 60)}, links=[(22, 30)])
    assert merge.merged_map(crops) == {21: 22}


def test_separate_proposals_are_not_merged(tmp_path: Path) -> None:
    crops = _build(tmp_path, {21: (1400, 1600, 60), 22: (1700, 1900, 60)})
    assert merge.merged_map(crops) == {}


def test_a_merged_card_leaves_the_catalog_and_its_links_go_to_the_survivor(tmp_path: Path) -> None:
    crops = _build(tmp_path, {21: (1400, 1600, 80), 22: (1440, 1620, 80), 30: (1800, 1900, 60)}, links=[(22, 30), (21, 30)])
    # 21 and 22 both link to 30: one link survives, from the survivor.
    survivor = merge.merged_map(crops)[22] if 22 in merge.merged_map(crops) else 22
    assert web.link_rows(crops) == [[min(survivor, 30), max(survivor, 30)]]
    from server import catalog_body

    rows = json.loads(catalog_body(crops, "RC1", "review", 512)[0])["rows"]
    assert {int(row["location_id"]) for row in rows} == {survivor, 30}


def test_the_batch_names_the_merged_locations_with_the_survivor(tmp_path: Path) -> None:
    crops = _build(tmp_path, {21: (1400, 1600, 80), 22: (1440, 1620, 80)})
    key = _cards(crops)[21]["image_key"]
    catalog.approve_window(crops, 21, key)
    batch = web.approved_batch(crops)
    assert [row["location_id"] for row in batch] == [21]
    assert batch[0]["merge_location_ids"] == [22]


def test_fold_links_drops_links_inside_a_merge_and_repeats() -> None:
    assert merge.fold_links([[1, 2], [2, 3], [1, 3], [3, 4]], {2: 1}) == [[1, 3], [3, 4]]


def test_survivor_rule_matches_viking() -> None:
    assert merge.choose_survivor([5, 3, 9], {5: 2, 3: 2, 9: 1}) == 3
    assert merge.choose_survivor([5, 3, 9], {9: 4}) == 9
