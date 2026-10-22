"""The converter builder end to end over fakes: OData pages, a synthetic tileset, and a stand-in segmentation client."""

from __future__ import annotations

import io
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
from PIL import Image

import catalog
from converter import builder as builder_module
from converter import contour, geo, odata, seed
from converter.seg_client import SegmentationFailed, SegmentResult
from converter.tilesource import SectionTiles, TileCache, flip_for_protocol, parse_viking_xml

VOLUME = "RC1"
TILE = 512
GRID = 12

VIKING_XML = f"""<Volume Name="{VOLUME}" path="http://tiles.test/{VOLUME}">
  <Sections>
    <Section Number="7" Path="0007">
      <Tileset name="TEM.Leveled" path="TEM/Leveled" TileXDim="{TILE}" TileYDim="{TILE}" FilePrefix="" FilePostfix=".png">
        <Level Downsample="1" path="001" GridDimX="{GRID}" GridDimY="{GRID}"/>
        <Level Downsample="2" path="002" GridDimX="{GRID // 2}" GridDimY="{GRID // 2}"/>
        <Level Downsample="4" path="004" GridDimX="{GRID // 4}" GridDimY="{GRID // 4}"/>
      </Tileset>
    </Section>
  </Sections>
</Volume>"""


def pixel_value(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """What the synthetic tileset holds at absolute level pixel (x, y): different for every nearby pixel."""
    return ((x * 3 + y * 5) % 251 + 1).astype(np.uint8)


def tile_http(url: str, headers: Any = None):
    """Serves ``.../00N/X0xx_Y0yy.png`` tiles of the synthetic tileset; any other URL is a 404."""
    name = url.rsplit("/", 1)[-1]
    if not (name.startswith("X") and name.endswith(".png")):
        return 404, {}, b""
    ix, iy = int(name[1:4]), int(name[6:9])
    if ix >= GRID or iy >= GRID:
        return 404, {}, b""
    ys, xs = np.mgrid[0:TILE, 0:TILE]
    out = io.BytesIO()
    Image.fromarray(pixel_value(xs + ix * TILE, ys + iy * TILE), mode="L").save(out, format="PNG")
    return 200, {"etag": f'"{ix}-{iy}"'}, out.getvalue()


@pytest.fixture
def volume():
    return parse_viking_xml(VIKING_XML, source="http://tiles.test/RC1/RC1.VikingXML")


@pytest.fixture
def cache():
    return TileCache(None, tile_http)


# ---- tile source -------------------------------------------------------------------------------------------------

def test_viking_xml_is_read_and_tile_urls_are_built(volume) -> None:
    tileset = volume.select(7, "TEM", "Leveled")
    assert tileset.levels[2].path == "002" and tileset.tile_x == TILE
    assert tileset.tile_url(volume.base_url, 2, 3, 4) == "http://tiles.test/RC1/0007/TEM/Leveled/002/X003_Y004.png"


def test_a_section_without_a_tileset_is_an_error(volume) -> None:
    with pytest.raises(Exception, match="no tileset"):
        volume.select(99)


def test_region_stitches_tiles_in_stored_orientation(volume, cache) -> None:
    section = SectionTiles(volume, volume.select(7), 1, cache)
    region = section.region(500, 1000, 40, 30)  # spans four tiles
    ys, xs = np.mgrid[0:30, 0:40]
    assert np.array_equal(region, pixel_value(xs + 500, ys + 1000))


def test_protocol_tile_is_the_1024_region_flipped_vertically(volume, cache) -> None:
    section = SectionTiles(volume, volume.select(7), 1, cache)
    png = section.protocol_tile(row=1, col=0)
    tile = np.asarray(Image.open(io.BytesIO(png)))
    assert tile.shape == (1024, 1024)
    stored = section.region(0, 1024, 1024, 1024)
    assert np.array_equal(tile, flip_for_protocol(stored))
    assert tile[0, 0] == pixel_value(np.array(0), np.array(1024 + 1023))  # row 0 of the tile is the highest y


def test_tiles_off_the_grid_are_black_and_a_blank_tile_is_unavailable(volume, cache) -> None:
    section = SectionTiles(volume, volume.select(7), 1, cache)
    assert section.protocol_tile(row=50, col=50) is None


def test_missing_level_is_an_error(volume, cache) -> None:
    with pytest.raises(Exception, match="no level 3"):
        SectionTiles(volume, volume.select(7), 3, cache)


def test_etag_revalidation_uses_304(tmp_path: Path, volume) -> None:
    calls: List[Dict[str, str]] = []

    def http(url: str, headers=None):
        calls.append(dict(headers or {}))
        if headers and headers.get("If-None-Match"):
            return 304, {}, b""
        return tile_http(url, headers)

    first = TileCache(tmp_path, http)
    url = "http://tiles.test/RC1/0007/TEM/Leveled/001/X001_Y001.png"
    tile = first.fetch(url)
    second = TileCache(tmp_path, http)
    again = second.fetch(url)
    assert np.array_equal(tile, again) and second.not_modified == 1 and calls[-1]["If-None-Match"] == '"1-1"'


# ---- OData -------------------------------------------------------------------------------------------------------

def _entity(location_id: int, type_code: int, wkt: str, radius=None, z=7, structure=500) -> Dict[str, Any]:
    return {
        "ID": location_id, "ParentID": structure, "TypeCode": type_code, "Z": z, "OffEdge": False, "MosaicShape": wkt,
        "LastModified": "2026-10-01T00:00:00Z", "Radius": radius,
        "Parent": {"ID": structure, "Label": "cell 5", "TypeID": 1, "Type": {"ID": 1, "Name": "Cell"}},
    }


def test_filter_text_and_url() -> None:
    text = odata.filter_text(500, type_codes=(1, 6), z_min=3, z_max=9)
    assert text == "ParentID eq 500 and (TypeCode eq 1 or TypeCode eq 6) and Z ge 3 and Z le 9 and OffEdge eq false"
    url = odata.locations_url("http://host/odata/RC1/", text)
    assert url.startswith("http://host/odata/RC1/Locations?$filter=") and "$top" not in url and "$count=true" in url


def test_records_for_circles_polygons_and_curves_and_the_unseedable_are_skipped() -> None:
    circle = odata.record_from_entity(_entity(1, 1, "POINT (1000 2000)", radius=40))
    polygon = odata.record_from_entity(_entity(2, 4, "POLYGON ((0 0, 100 0, 100 100, 0 100, 0 0))"))
    curved = odata.record_from_entity(_entity(3, 6, "POLYGON ((0 0, 100 0, 100 100, 0 100, 0 0))"))
    assert circle.center == (1000.0, 2000.0) and circle.radius == 40
    assert len(polygon.polygons[0][0]) == 5
    assert len(curved.polygons[0][0]) > 5, "a curve polygon is restored to its smooth outline"
    assert odata.record_from_entity(_entity(4, 0, "POINT (1 1)")) is None
    assert odata.record_from_entity(_entity(5, 1, "POINT (1 1)", radius=None)) is None
    assert odata.record_from_entity(_entity(6, 5, "LINESTRING (0 0, 1 1)")) is None


def test_paging_follows_next_links() -> None:
    pages = {
        "http://h/odata/Locations?first": {"value": [_entity(1, 1, "POINT (1 1)", 5)], "@odata.nextLink": "Locations?second"},
        "http://h/odata/Locations?second": {"value": [_entity(2, 1, "POINT (2 2)", 5)]},
    }
    assert [d["value"][0]["ID"] for d in odata.iter_pages("http://h/odata/Locations?first", pages.__getitem__)] == [1, 2]


# ---- the builder -------------------------------------------------------------------------------------------------

class FakeSegmenter:
    """Returns logits that are positive in the *true* shape grown by ``grow`` pixels, in the protocol's frame."""

    def __init__(self, truth_for, grow: float = 2.0, fail: Dict[int, str] | None = None, size: int = 128) -> None:
        self.size = size
        self.truth_for = truth_for
        self.grow = grow
        self.fail = fail or {}
        self.calls: List[Any] = []

    def segment_with_retry(self, identity, prompt, tiles, background, *, threshold, request_id, **_):
        self.calls.append((identity, prompt, request_id))
        location_id = request_id
        if location_id in self.fail:
            raise SegmentationFailed(self.fail[location_id])
        assert tiles(identity.downsample and 0, 0) is None or True  # the provider must be callable
        size = self.size  # samples, 4 px each
        xs0 = min(x for x, _ in prompt.foreground_points) - size * 4 // 2
        ys0 = min(y for _, y in prompt.foreground_points) - size * 4 // 2
        rows = np.arange(size)
        cols = np.arange(size)
        X = xs0 + (cols[None, :] + 0.5) * 4
        Y = ys0 + size * 4 - (rows[:, None] + 0.5) * 4
        inside = self.truth_for(location_id)(X, Y, self.grow)
        logits = np.where(inside, 6.0, -6.0).astype(np.float32)
        return SegmentResult(
            logits=logits, scale=4.0, logits_x=0, logits_y=0, origin_x=int(xs0), origin_y=int(ys0), frame_height=size,
            score=0.9, request_id=request_id, threshold=threshold,
        )


def _disc(cx, cy, r):
    return lambda X, Y, grow: (X - cx) ** 2 + (Y - cy) ** 2 <= (r + grow) ** 2


def _make(tmp_path, volume, cache, segmenter, entities, **params):
    pages = {"value": entities}

    def get_json(url: str):
        if "$top=1" in url:
            return {"@odata.count": len(entities), "value": entities[:1]}
        return pages

    build_params = builder_module.BuildParams(volume=VOLUME, odata_base="http://odata.test/RC1", structure_id=500, **params)
    return builder_module.Builder(build_params, tmp_path / "crops", volume, segmenter, cache, get_json=get_json)


def test_a_circle_is_built_into_a_gallery_card(tmp_path: Path, volume, cache) -> None:
    # Circle of radius 80 mosaic px at (1400, 1600): at D=1 it fits a card with margin.
    entities = [_entity(11, 1, "POINT (1400 1600)", radius=80)]
    segmenter = FakeSegmenter(lambda _id: _disc(1400, 1600, 80))
    builder = _make(tmp_path, volume, cache, segmenter, entities)

    summary = builder.run()

    assert (summary.built, summary.failed, summary.total) == (1, 0, 1)
    crops = tmp_path / "crops"
    key = f"{VOLUME}_7_D1_L11"
    assert (crops / "images" / f"{key}.png").is_file()
    assert (crops / "masks" / f"{key}_11.png").is_file() and (crops / "original" / f"{key}_11.png").is_file()
    sidecar = json.loads((crops / "images" / f"{key}.json").read_text(encoding="utf-8"))
    assert sidecar["image"]["width"] == 1024 and sidecar["annotations"] == [{"id": 11}]
    rows = catalog.list_catalog_rows(crops)
    assert [(r["location_id"], r["image_key"]) for r in rows] == [(11, key)]
    connection = sqlite3.connect(crops / "annotation_crops.sqlite")
    connection.row_factory = sqlite3.Row
    card = connection.execute("SELECT * FROM converter_cards").fetchone()
    assert card["state"] == "built" and card["iou"] > 0.9 and card["candidate_wkt"].startswith("POLYGON ((")
    # The crop is the stored tileset region at the card's origin.
    image = np.asarray(Image.open(crops / "images" / f"{key}.png"))
    ox, oy = card["origin_x"], card["origin_y"]
    ys, xs = np.mgrid[0:64, 0:64]
    assert ox >= 0 and oy >= 0
    assert np.array_equal(image[:64, :64], pixel_value(xs + ox, ys + oy))


def test_candidate_polygon_is_where_the_object_is_in_mosaic_pixels(tmp_path: Path, volume, cache) -> None:
    entities = [_entity(12, 1, "POINT (1400 1600)", radius=80)]
    builder = _make(tmp_path, volume, cache, FakeSegmenter(lambda _id: _disc(1400, 1600, 80), grow=3.0), entities)
    builder.run()
    connection = sqlite3.connect(tmp_path / "crops" / "annotation_crops.sqlite")
    wkt = connection.execute("SELECT candidate_wkt FROM converter_cards").fetchone()[0]
    ring = geo.parse_wkt_polygons(wkt)[0][0]
    xs, ys = zip(*ring)
    assert abs(np.mean(xs) - 1400) < 3 and abs(np.mean(ys) - 1600) < 3
    assert abs((max(xs) - min(xs)) / 2 - 83) < 4


def test_a_polygon_at_a_coarser_level_when_it_is_big(tmp_path: Path, volume, cache) -> None:
    # 1500 x 1500 mosaic px does not fit a 1024 card at D=1 with margin, so D=2 is chosen.
    wkt = "POLYGON ((800 800, 2300 800, 2300 2300, 800 2300, 800 800))"
    # The segmenter works in protocol pixels: the square's centre (1550, 1550) mosaic is (775, 775) at D=2, half side 375.
    square = lambda X, Y, grow: (abs(X - 775) <= 375 + grow) & (abs(Y - 775) <= 375 + grow)
    builder = _make(tmp_path, volume, cache, FakeSegmenter(lambda _id: square, size=384), [_entity(13, 4, wkt)])
    summary = builder.run()
    assert summary.built == 1
    row = sqlite3.connect(tmp_path / "crops" / "annotation_crops.sqlite").execute(
        "SELECT downsample, iou FROM converter_cards"
    ).fetchone()
    assert row[0] == 2 and row[1] > 0.9


def test_failures_are_recorded_with_a_reason_and_do_not_stop_the_structure(tmp_path: Path, volume, cache) -> None:
    entities = [
        _entity(21, 1, "POINT (1400 1600)", radius=80),
        _entity(22, 1, "POINT (1500 1700)", radius=80),
        _entity(23, 1, "POINT (1600 1800)", radius=0.2),
    ]
    segmenter = FakeSegmenter(lambda _id: _disc(1500, 1700, 80), fail={21: "RESOURCE_EXHAUSTED: tiles over limit 64"})
    summary = _make(tmp_path, volume, cache, segmenter, entities).run()

    assert (summary.built, summary.failed) == (1, 2)
    assert summary.reasons == {"RESOURCE_EXHAUSTED": 1, "failed": 1}
    connection = sqlite3.connect(tmp_path / "crops" / "annotation_crops.sqlite")
    states = dict(connection.execute("SELECT location_id, state FROM converter_cards"))
    assert states[21] == "failed" and states[22] == "built"
    reason = connection.execute("SELECT reason FROM converter_cards WHERE location_id = 21").fetchone()[0]
    assert reason == "RESOURCE_EXHAUSTED: tiles over limit 64"
    tiny = connection.execute("SELECT reason FROM converter_cards WHERE location_id = 23").fetchone()[0]
    assert "below one pixel" in tiny


def test_a_second_run_skips_built_cards_and_retries_failures_only_on_request(tmp_path: Path, volume, cache) -> None:
    entities = [_entity(31, 1, "POINT (1400 1600)", radius=80), _entity(32, 1, "POINT (1500 1700)", radius=80)]
    segmenter = FakeSegmenter(lambda _id: _disc(1450, 1650, 120), fail={32: "UNAVAILABLE: gpu lost"})
    builder = _make(tmp_path, volume, cache, segmenter, entities)
    builder.run()
    first_calls = len(segmenter.calls)

    again = _make(tmp_path, volume, cache, segmenter, entities).run()
    assert again.skipped == 2 and len(segmenter.calls) == first_calls

    segmenter.fail = {}
    retry = _make(tmp_path, volume, cache, segmenter, entities, retry_failed=True).run()
    assert retry.built == 1 and retry.skipped == 1


def test_a_changed_shape_is_rebuilt(tmp_path: Path, volume, cache) -> None:
    segmenter = FakeSegmenter(lambda _id: _disc(1400, 1600, 80))
    _make(tmp_path, volume, cache, segmenter, [_entity(41, 1, "POINT (1400 1600)", radius=80)]).run()
    changed = _make(tmp_path, volume, cache, segmenter, [_entity(41, 1, "POINT (1400 1600)", radius=85)]).run()
    assert changed.built == 1


def test_cancel_stops_before_the_next_location(tmp_path: Path, volume, cache) -> None:
    entities = [_entity(51, 1, "POINT (1400 1600)", radius=80), _entity(52, 1, "POINT (1500 1700)", radius=80)]
    cancel = threading.Event()
    segmenter = FakeSegmenter(lambda _id: _disc(1450, 1650, 120))
    original = segmenter.segment_with_retry

    def cancelling(*args, **kwargs):
        cancel.set()
        return original(*args, **kwargs)

    segmenter.segment_with_retry = cancelling
    summary = _make(tmp_path, volume, cache, segmenter, entities).run(cancel=cancel)
    assert summary.cancelled and summary.built == 1


def test_choose_downsample_and_card_origin() -> None:
    small = odata.record_from_entity(_entity(1, 1, "POINT (5000 6000)", radius=100))
    assert builder_module.choose_downsample(small, [1, 2, 4]) == 1
    big = odata.record_from_entity(_entity(2, 4, "POLYGON ((0 0, 3000 0, 3000 3000, 0 3000, 0 0))"))
    assert builder_module.choose_downsample(big, [1, 2, 4]) == 4
    huge = odata.record_from_entity(_entity(3, 4, "POLYGON ((0 0, 9000 0, 9000 9000, 0 9000, 0 0))"))
    with pytest.raises(SegmentationFailed, match="too_large"):
        builder_module.choose_downsample(huge, [1, 2, 4])
    assert builder_module.card_origin(small, 1) == (5000 - 512, 6000 - 512)


OLD_FORMAT_XML = """<?xml version="1.0" ?>
<Volume Name="RC1" host="http://rogue1.test/" path="http://rogue1.test/RABBIT" num_sections="2">
  <stos GroupName="grid16" controlSection="1" mappedSection="2" path="0002-0001_grid_16.stos" pixelSpacing="16" type="grid"/>
  <Section name="GABA 1" number="1" path="0001">
    <Pyramid name="8-bit" path="8-bit"><Level Downsample="1" path="001"/></Pyramid>
    <Tileset FilePostfix=".png" FilePrefix="0001_" TileXDim="256" TileYDim="256" name="TEM" path="TEM">
      <Level Downsample="1" GridDimX="4" GridDimY="3" path="001"/>
      <Level Downsample="2" GridDimX="2" GridDimY="2" path="002"/>
    </Tileset>
  </Section>
</Volume>"""


def test_the_older_lowercase_vikingxml_of_rc1_is_read() -> None:
    volume = parse_viking_xml(OLD_FORMAT_XML, source="http://rogue1.test/RABBIT/rogue1.vikingxml")
    tileset = volume.select(1, "TEM")
    assert tileset.tile_x == 256 and tileset.prefix == "0001_" and sorted(tileset.levels) == [1, 2]
    assert tileset.levels[2].grid_x == 2
    assert tileset.tile_url(volume.base_url, 1, 3, 2) == "http://rogue1.test/RABBIT/0001/TEM/001/0001_X003_Y002.png"


def test_a_volume_specific_vikingxml_address_wins(monkeypatch) -> None:
    from converter import jobs

    seen = []
    monkeypatch.setattr(jobs, "load_volume", lambda url: seen.append(url) or "v")
    env = {"GALLERY_CONVERTER_VIKINGXML": "http://h/<volume>/S.xml", "GALLERY_CONVERTER_VIKINGXML_RC1": "http://odd/RABBIT/r.xml"}
    jobs.default_volume("RC1", env)
    jobs.default_volume("RC2", env)
    assert seen == ["http://odd/RABBIT/r.xml", "http://h/RC2/S.xml"]
