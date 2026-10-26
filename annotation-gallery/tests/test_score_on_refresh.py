"""The refresh button scores the location with the current model and stores it as the card's SAM2 badge (score.py)."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import catalog
import score as score_module
from converter import odata
from converter.seg_client import SegmentationFailed
from converter.tilesource import TileCache, parse_viking_xml
from tests.test_converter_builder import VIKING_XML, VOLUME, FakeSegmenter, _disc, _entity, tile_http


@pytest.fixture
def volume():
    return parse_viking_xml(VIKING_XML, source="http://tiles.test/RC1/RC1.VikingXML")


@pytest.fixture
def cache():
    return TileCache(None, tile_http)


class ScoringClient(FakeSegmenter):
    """The converter's fake segmenter with the status call the scorer asks for the model version."""

    def __init__(self, *args, version: str = "v5.4", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.version = version
        self.closed = False

    def status(self):
        return SimpleNamespace(model=SimpleNamespace(model_version=self.version, model_id="v5"))

    def close(self) -> None:
        self.closed = True


def _square(x0, y0, x1, y1):
    return lambda X, Y, grow: (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)


def _catalog_with(crops: Path, location_id: int, key: str = "K1") -> None:
    connection = catalog.connect(crops)
    try:
        connection.execute("INSERT INTO locations (location_id, z, image_key) VALUES (?, 7, ?)", [location_id, key])
        connection.commit()
    finally:
        connection.close()


# ---- the seed -----------------------------------------------------------------------------------------------------

def test_a_circle_is_seeded_as_itself() -> None:
    record = odata.record_from_entity(_entity(11, 1, "POINT (1400 1600)", radius=80))
    assert score_module.seed_circle(record) == (1400.0, 1600.0, 80.0, "circle")


def test_a_polygon_is_seeded_with_its_inscribed_circle_at_the_seed_scale() -> None:
    record = odata.record_from_entity(_entity(12, 4, "POLYGON ((1000 1000, 1200 1000, 1200 1200, 1000 1200, 1000 1000))"))
    cx, cy, radius, kind = score_module.seed_circle(record)
    assert kind == "inscribed circle"
    assert abs(cx - 1100) < 3 and abs(cy - 1100) < 3
    assert radius == pytest.approx(100 * score_module.SEED_CIRCLE_SCALE, abs=3)


# ---- scoring -------------------------------------------------------------------------------------------------------

def test_a_model_that_finds_the_annotated_circle_scores_near_one(volume, cache) -> None:
    record = odata.record_from_entity(_entity(11, 1, "POINT (1400 1600)", radius=80))
    client = ScoringClient(lambda _id: _disc(1400, 1600, 80), grow=0.0)

    score = score_module.score_record(record, VOLUME, volume, client, cache, model="v5.4")

    assert score.gt_iou > 0.9 and score.pred_iou == pytest.approx(0.9) and score.model == "v5.4"
    assert score.prompt == "circle" and score.downsample == 1
    _identity, prompt, request_id = client.calls[0]
    assert request_id == 11 and prompt.foreground_mask is None, "a circle goes as a shape, not a bitmap"
    (circle,) = [shape.circle for shape in prompt.foreground_shapes]
    assert (circle.center.x, circle.center.y, circle.radius) == (1400, 1600, 80), "the seed circle, its true radius"


def test_a_polygon_the_model_only_half_covers_scores_about_a_half(volume, cache) -> None:
    record = odata.record_from_entity(_entity(12, 4, "POLYGON ((1000 1000, 1200 1000, 1200 1200, 1000 1200, 1000 1000))"))
    client = ScoringClient(lambda _id: _square(1000, 1000, 1100, 1200), grow=0.0)

    score = score_module.score_record(record, VOLUME, volume, client, cache)

    assert 0.4 < score.gt_iou < 0.6 and score.prompt == "inscribed circle"


def test_a_small_object_is_scored_at_its_crops_level_and_says_nothing_more(volume, cache) -> None:
    record = odata.record_from_entity(_entity(11, 1, "POINT (1400 1600)", radius=80))
    client = ScoringClient(lambda _id: _disc(1400, 1600, 80), grow=0.0)

    score = score_module.score_record(record, VOLUME, volume, client, cache, model="v5.4", native_downsample=1)

    assert score.downsample == 1 and not score.coarser_than_native and score.checkpoint_label == "v5.4"


def test_an_object_too_large_for_one_card_is_scored_coarser_and_labelled_so(volume, cache) -> None:
    # 1500 px across: no 1024-pixel card holds it at D1, so it is scored whole at D2. The fake segmenter works in the
    # level's pixels: the square's centre (1550, 1550) mosaic is (775, 775) at D2, half side 375 (as in the converter test).
    record = odata.record_from_entity(_entity(14, 4, "POLYGON ((800 800, 2300 800, 2300 2300, 800 2300, 800 800))"))
    square = lambda X, Y, grow: (abs(X - 775) <= 375 + grow) & (abs(Y - 775) <= 375 + grow)  # noqa: E731
    client = ScoringClient(lambda _id: square, grow=0.0, size=384)

    score = score_module.score_record(record, VOLUME, volume, client, cache, model="v5.4", native_downsample=1)

    assert score.downsample == 2 and score.coarser_than_native
    assert score.checkpoint_label == "v5.4 at D2" and score.payload()["sam2Checkpoint"] == "v5.4 at D2"
    assert score.gt_iou > 0.9


def test_an_empty_mask_is_a_score_of_zero_not_an_error(volume, cache) -> None:
    record = odata.record_from_entity(_entity(13, 1, "POINT (1400 1600)", radius=80))
    client = ScoringClient(lambda _id: (lambda X, Y, grow: np.zeros(np.broadcast(X, Y).shape, dtype=bool)))

    assert score_module.score_record(record, VOLUME, volume, client, cache).gt_iou == 0.0


# ---- after a refresh: stored, reported, and never fatal ---------------------------------------------------------------

def _after(tmp_path, volume, cache, entity, client=None, make_client=None):
    return score_module.score_after_refresh(
        tmp_path,
        VOLUME,
        entity,
        make_client=make_client or (lambda: client),
        load_volume=lambda: volume,
        volume_key=f"test-{tmp_path}",
        make_cache=lambda: cache,
    )


def test_the_crops_level_comes_from_the_catalogs_image_key(tmp_path: Path) -> None:
    _catalog_with(tmp_path, 11, "RC1_7_D4_X0-1024_Y0-1024")
    _catalog_with(tmp_path, 12, "no-level-here")
    assert score_module.native_downsample(tmp_path, 11) == 4
    assert score_module.native_downsample(tmp_path, 12) is None
    assert score_module.native_downsample(tmp_path, 99) is None


def test_the_score_is_stored_in_the_catalog_the_gallery_serves(tmp_path: Path, volume, cache) -> None:
    _catalog_with(tmp_path, 11)
    _catalog_with(tmp_path, 12, "K2")
    client = ScoringClient(lambda _id: _disc(1400, 1600, 80), grow=0.0, version="v5.4")

    answer = _after(tmp_path, volume, cache, _entity(11, 1, "POINT (1400 1600)", radius=80), client)

    assert "error" not in answer and answer["rows"] == 1 and answer["sam2Checkpoint"] == "v5.4"
    assert answer["sam2GtIou"] > 0.9 and answer["prompt"] == "circle"
    assert client.closed
    connection = sqlite3.connect(catalog.sqlite_path(tmp_path))
    stored = connection.execute(
        "SELECT sam2GtIou, sam2PredIou, sam2Checkpoint, sam2ScoredAt FROM locations WHERE location_id = 11"
    ).fetchone()
    assert stored[0] == pytest.approx(answer["sam2GtIou"]) and stored[1] == pytest.approx(0.9)
    assert stored[2] == "v5.4" and stored[3] == answer["sam2ScoredAt"]
    other = connection.execute("SELECT sam2GtIou FROM locations WHERE location_id = 12").fetchone()
    assert other[0] is None, "only the refreshed location is scored"
    shown = {row["location_id"]: row for row in catalog.list_catalog_rows(tmp_path)}
    assert shown[11]["sam2GtIou"] == pytest.approx(answer["sam2GtIou"]), "the catalog the gallery serves has it"


def test_a_server_failure_is_reported_and_stores_nothing(tmp_path: Path, volume, cache) -> None:
    _catalog_with(tmp_path, 11)
    client = ScoringClient(lambda _id: _disc(1400, 1600, 80), fail={11: "unavailable: the server is down"})

    answer = _after(tmp_path, volume, cache, _entity(11, 1, "POINT (1400 1600)", radius=80), client)

    assert answer == {"error": "unavailable: the server is down"}
    connection = sqlite3.connect(catalog.sqlite_path(tmp_path))
    assert connection.execute("SELECT COUNT(*) FROM locations WHERE sam2GtIou IS NOT NULL").fetchone()[0] == 0


def test_an_unconfigured_server_or_an_unreadable_shape_is_an_error_not_an_exception(tmp_path: Path, volume, cache) -> None:
    def unconfigured():
        raise RuntimeError("GALLERY_SEGMENTATION_TARGET is not set")

    entity = _entity(11, 1, "POINT (1400 1600)", radius=80)
    assert "not available" in _after(tmp_path, volume, cache, entity, make_client=unconfigured)["error"]
    assert "error" in _after(tmp_path, volume, cache, {"ID": 42, "MosaicShape": {}}, make_client=unconfigured)
