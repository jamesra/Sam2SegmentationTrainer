"""The converter's segmentation client, seeds and contouring, against a fake server that speaks the documented
SegmentTilesStream conversation (start with grid, cells, shapes and background shapes; the painted bitmap's mask tiles;
TilesNeeded with tiles and background_cells; a TilesAnswer listing only what could not be supplied; one result per
island)."""

from __future__ import annotations

import io
from concurrent import futures
from typing import List, Optional, Sequence

import grpc
import numpy as np
import pytest
from PIL import Image

from converter import contour, seed
from converter.seg_client import (
    TILE_SIZE,
    MaskArray,
    Prompt,
    SegmentationClient,
    SegmentationFailed,
    SegmentResult,
    TileIdentity,
    circle_shape,
    foreground_tiles,
    polygon_shape,
    sample_centres,
    shape_cells,
)
from seg_proto import segmentation_pb2 as pb
from seg_proto import segmentation_pb2_grpc as pb_grpc
from seg_proto.Common import geometry_pb2 as geo

IDENTITY = TileIdentity(volume="RC1", section=7, channel="TEM", transform="stos", downsample=2)


def _png_tile(value: int = 128) -> bytes:
    out = io.BytesIO()
    Image.new("L", (TILE_SIZE, TILE_SIZE), value).save(out, format="PNG")
    return out.getvalue()


def _decode_mask(tile: "pb.MaskTile") -> np.ndarray:
    """A MaskTile as a Y-up bool array."""
    if tile.full:
        return np.ones((TILE_SIZE, TILE_SIZE), dtype=bool)
    if not tile.png:
        return np.zeros((TILE_SIZE, TILE_SIZE), dtype=bool)
    return np.flipud(np.asarray(Image.open(io.BytesIO(tile.png))) > 0)


def _disc_logits(size: int, cx: float, cy: float, radius: float) -> np.ndarray:
    """Logit grid (row 0 at the top) with a disc of positive logits; (cx, cy) are sample indices from the top-left."""
    ys, xs = np.mgrid[0:size, 0:size]
    return np.where((xs + 0.5 - cx) ** 2 + (ys + 0.5 - cy) ** 2 <= radius**2, 6.0, -6.0).astype(np.float32)


def _segment(logits: np.ndarray, island: int = 1, score: float = 0.9, scale: int = 4, origin=(0, 0)) -> "pb.SegmentResult":
    return pb.SegmentResult(
        island=island,
        score=score,
        logits=logits.astype("<f2").tobytes(),
        logits_size=geo.IntSize(width=logits.shape[1], height=logits.shape[0]),
        logits_origin=geo.IntPoint(x=origin[0], y=origin[1]),
        logits_scale=float(scale),
    )


EXTRA = (0, 5)  # the cell the fake grows into: it asks for its image tile and its background shapes


class FakeSegmentation(pb_grpc.SegmentationServiceServicer):
    """Replies with fixed segments after asking for one extra image tile and the background of that cell."""

    def __init__(self, result_logits, ask_more=True, segments: Optional[Sequence["pb.SegmentResult"]] = None):
        self.segments = list(segments) if segments is not None else [_segment(result_logits, origin=(512, 1624))]
        self.ask_more = ask_more
        self.uploads: List["pb.UploadTileRequest"] = []
        self.starts: List["pb.SegmentTilesRequest"] = []
        self.mask_tiles_received: List["pb.MaskTile"] = []
        self.answers: List["pb.TilesAnswer"] = []
        self.order: List[str] = []

    def UploadTile(self, request, context):
        self.uploads.append(request)
        return pb.UploadTileResponse(already_cached=False)

    def SegmentTilesStream(self, request_iterator, context):
        requests = iter(request_iterator)
        first = next(requests)
        assert first.WhichOneof("body") == "start"
        start = first.start
        self.starts.append(start)
        self.order.append("start")
        while len(self.mask_tiles_received) < start.mask_tile_count:
            message = next(requests)
            assert message.WhichOneof("body") == "mask_tiles"
            self.order.append("mask_tiles")
            self.mask_tiles_received.extend(message.mask_tiles.tiles)
        if self.ask_more:
            extra = geo.GridCell(row=EXTRA[0], col=EXTRA[1])
            yield pb.SegmentTilesStreamResponse(needed=pb.TilesNeeded(tiles=[extra], background_cells=[extra]))
            self.order.append("needed")
            message = next(requests)
            assert message.WhichOneof("body") == "answer"
            self.order.append("answer")
            self.answers.append(message.answer)
        yield pb.SegmentTilesStreamResponse(
            result=pb.SegmentationResponse(request_id=start.request_id, segments=self.segments)
        )


@pytest.fixture
def serve():
    servers = []

    def start(servicer):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        pb_grpc.add_SegmentationServiceServicer_to_server(servicer, server)
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        servers.append(server)
        return SegmentationClient(f"127.0.0.1:{port}", total_timeout_seconds=20)

    yield start
    for server in servers:
        server.stop(0)


def _tiles(row: int, col: int):
    return _png_tile(100 + col)


def _no_neighbours(cells):
    return []


# ---- seeds -------------------------------------------------------------------------------------------------------

def test_disc_mask_area_and_position() -> None:
    mask = seed.disc_mask(500.0, 700.0, 40.0)
    assert abs(int(mask.pixels.sum()) - np.pi * 40**2) < 0.02 * np.pi * 40**2
    ys, xs = np.nonzero(mask.pixels)
    assert abs((xs.mean() + mask.x0 + 0.5) - 500) < 0.6
    assert abs((ys.mean() + mask.y0 + 0.5) - 700) < 0.6


def test_circle_prompt_is_the_circle_at_its_true_radius_plus_four_rim_clicks() -> None:
    prompt = seed.circle_prompt(500, 700, 40)
    assert prompt.foreground_mask is None, "a circle is geometry, not a painted bitmap"
    (shape,) = prompt.foreground_shapes
    assert shape.WhichOneof("kind") == "circle"
    # The server applies circle_diameter_scale; the client sends the radius as annotated.
    assert (shape.circle.center.x, shape.circle.center.y, shape.circle.radius) == (500, 700, 40)
    assert len(prompt.foreground_points) == 4
    for x, y in prompt.foreground_points:
        assert abs(np.hypot(x - 500, y - 700) - 38) <= 1.0


def test_circle_below_a_pixel_is_not_seedable() -> None:
    with pytest.raises(seed.NotSeedable):
        seed.circle_prompt(10, 10, 0.4)


def test_polygon_mask_matches_its_area_and_is_y_up() -> None:
    ring = [(100, 200), (160, 200), (160, 260), (100, 260), (100, 200)]
    mask = seed.polygon_mask(ring)
    assert abs(int(mask.pixels.sum()) - 60 * 60) <= 130
    ys, xs = np.nonzero(mask.pixels)
    assert abs((ys.mean() + mask.y0 + 0.5) - 230) < 1.0
    assert abs((xs.mean() + mask.x0 + 0.5) - 130) < 1.0


def test_polygon_prompt_is_the_polygon_shape_with_its_holes_and_clicks_avoid_them() -> None:
    outer = [(0, 0), (200, 0), (200, 200), (0, 200), (0, 0)]
    hole = [(60, 60), (140, 60), (140, 140), (60, 140), (60, 60)]
    prompt = seed.polygon_prompt(outer, [hole])
    assert prompt.foreground_mask is None
    (shape,) = prompt.foreground_shapes
    polygon = shape.polygon
    # A proto ring does not repeat its first vertex.
    assert [(p.x, p.y) for p in polygon.exterior.points] == outer[:-1]
    assert [[(p.x, p.y) for p in ring.points] for ring in polygon.holes] == [hole[:-1]]
    raster = seed.polygon_mask(outer, [hole])
    for x, y in prompt.foreground_points:
        assert raster.pixels[y - raster.y0, x - raster.x0]
        assert not (60 <= x <= 140 and 60 <= y <= 140)


def test_interior_points_are_deepest_first() -> None:
    mask = seed.disc_mask(300, 300, 50)
    points = seed.interior_points(mask)
    assert points and np.hypot(points[0][0] - 300, points[0][1] - 300) < 3


def test_shape_cells_are_the_grid_cells_its_bounds_reach() -> None:
    assert shape_cells(circle_shape(1300, 900, 40)) == [(0, 1)]
    assert shape_cells(circle_shape(1100, 1020, 10)) == [(0, 1), (1, 1)]
    assert sorted(shape_cells(polygon_shape([(-10, 5), (1030, 5), (1030, 20)]))) == [(0, -1), (0, 0), (0, 1)]


# ---- painted bitmaps ---------------------------------------------------------------------------------------------

def test_mask_across_tiles_is_cut_on_the_1024_grid_with_full_and_png_tiles() -> None:
    pixels = np.ones((1500, 1500), dtype=bool)
    mask = MaskArray(pixels=pixels, x0=600, y0=600)  # pixels 600..2099: tiles 0, 1 and 2 on each axis
    tiles = foreground_tiles(mask)
    cells = {(t.cell.row, t.cell.col): t for t in tiles}
    assert set(cells) == {(r, c) for r in (0, 1, 2) for c in (0, 1, 2)}
    assert cells[(1, 1)].full and not cells[(1, 1)].png, "tile 1 is 1024..2047, wholly inside the mask"
    assert cells[(0, 0)].png and not cells[(0, 0)].full
    assert all(t.kind == pb.MASK_KIND_FOREGROUND for t in tiles)
    restored = _decode_mask(cells[(0, 0)])
    assert restored[600:, 600:].all() and not restored[:600, :].any() and not restored[:, :600].any()


def test_a_wholly_covered_tile_is_sent_as_full() -> None:
    mask = MaskArray(pixels=np.ones((TILE_SIZE, TILE_SIZE), dtype=bool), x0=TILE_SIZE, y0=0)
    tiles = foreground_tiles(mask)
    assert len(tiles) == 1 and tiles[0].full and not tiles[0].png


# ---- the conversation --------------------------------------------------------------------------------------------

def test_segment_runs_the_documented_conversation(serve) -> None:
    fake = FakeSegmentation(_disc_logits(256, 100, 120, 40))
    client = serve(fake)
    prompt = seed.circle_prompt(1300, 900, 40)  # tile row 0, col 1 (ds 2 mosaic pixels)
    here = circle_shape(1500, 900, 30)  # a neighbour in the prompt's cell
    there = circle_shape(5300, 900, 30)  # a neighbour in the cell the object grows into

    def neighbours(cells):
        return [here] + ([there] if EXTRA in cells else [])

    result = client.segment(IDENTITY, prompt, _tiles, neighbours, request_id=77)

    start = fake.starts[0]
    assert start.grid == geo.TileGrid(
        volume="RC1", section=7, channel="TEM", transform="stos", downsample=2, space=geo.COORDINATE_SPACE_MOSAIC
    )
    assert [(c.row, c.col) for c in start.tiles] == [(0, 1)]
    assert start.mask_threshold == 0.5 and start.request_id == 77
    assert start.mask_tile_count == 0 and not fake.mask_tiles_received, "geometry goes as shapes, not mask tiles"
    assert [s.circle.radius for s in start.foreground_feature.shapes] == [40]
    assert len(start.foreground_feature.points) == 4
    assert list(start.background_feature.shapes) == [here]
    assert fake.order == ["start", "needed", "answer"]
    # The tiles in `start` need not be uploaded; the asked-for tile is, on the call's grid.
    assert [((u.cell.row, u.cell.col), u.grid.downsample, u.grid.space) for u in fake.uploads] == [
        (EXTRA, 2, geo.COORDINATE_SPACE_MOSAIC)
    ]
    answer = fake.answers[0]
    assert not answer.unavailable
    assert list(answer.background_shapes) == [there], "only shapes not sent before"
    assert result.request_id == 77 and result.tiles_uploaded == 1 and result.background_shapes_sent == 2
    assert result.island == 1 and result.islands == 1


def test_a_painted_bitmap_goes_as_mask_tiles_right_after_start(serve) -> None:
    fake = FakeSegmentation(_disc_logits(64, 32, 32, 10), ask_more=False)
    client = serve(fake)
    pixels = np.ones((40, 40), dtype=bool)
    prompt = Prompt(foreground_mask=MaskArray(pixels=pixels, x0=1010, y0=100))  # straddles cells (0, 0) and (0, 1)

    client.segment(IDENTITY, prompt, _tiles, _no_neighbours)

    start = fake.starts[0]
    assert start.mask_tile_count == 2 == len(fake.mask_tiles_received)
    assert fake.order == ["start", "mask_tiles"]
    assert {(t.cell.row, t.cell.col) for t in fake.mask_tiles_received} == {(0, 0), (0, 1)}
    assert [(c.row, c.col) for c in start.tiles] == [(0, 0), (0, 1)]
    assert not start.foreground_feature.shapes


def test_unavailable_tiles_are_listed_in_the_answer(serve) -> None:
    fake = FakeSegmentation(_disc_logits(256, 100, 120, 40))
    client = serve(fake)

    def tiles(row: int, col: int):
        return None if col == EXTRA[1] else _png_tile()

    client.segment(IDENTITY, seed.circle_prompt(1300, 900, 40), tiles, _no_neighbours)

    answer = fake.answers[0]
    assert [(c.row, c.col) for c in answer.unavailable] == [EXTRA]
    assert not answer.background_shapes
    assert not fake.uploads


def test_island_one_is_kept_when_the_prompt_splits(serve) -> None:
    small = _segment(_disc_logits(16, 8, 8, 3), island=2, score=0.99, origin=(0, 64))
    main = _segment(_disc_logits(64, 32, 32, 10), island=1, score=0.8, origin=(512, 1624))
    client = serve(FakeSegmentation(None, ask_more=False, segments=[small, main]))

    result = client.segment(IDENTITY, seed.circle_prompt(1300, 900, 40), _tiles, _no_neighbours)

    assert result.island == 1 and result.islands == 2 and result.score == pytest.approx(0.8)
    assert result.logits.shape == (64, 64) and (result.origin_x, result.origin_y) == (512, 1624)
    assert result.notes and "2 islands" in result.notes[0]


def test_server_status_errors_become_segmentation_failed(serve) -> None:
    class Refusing(FakeSegmentation):
        def SegmentTilesStream(self, request_iterator, context):
            next(iter(request_iterator))
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "tiles over limit 64")
            yield  # pragma: no cover

    client = serve(Refusing(_disc_logits(8, 4, 4, 2)))
    with pytest.raises(SegmentationFailed) as failure:
        client.segment(IDENTITY, seed.circle_prompt(1300, 900, 40), _tiles, _no_neighbours)
    assert failure.value.code == grpc.StatusCode.RESOURCE_EXHAUSTED and not failure.value.retryable


def test_a_seed_with_nothing_in_it_is_refused_before_any_rpc(serve) -> None:
    fake = FakeSegmentation(_disc_logits(8, 4, 4, 2))
    client = serve(fake)
    with pytest.raises(SegmentationFailed, match="empty_seed"):
        client.segment(IDENTITY, Prompt(), _tiles, _no_neighbours)
    assert not fake.uploads and not fake.starts


def test_a_prompt_over_the_tile_limit_is_refused_before_any_rpc(serve) -> None:
    fake = FakeSegmentation(_disc_logits(8, 4, 4, 2))
    client = serve(fake)
    with pytest.raises(SegmentationFailed, match="too_large"):
        client.segment(IDENTITY, Prompt(foreground_shapes=[circle_shape(5000, 5000, 4500)]), _tiles, _no_neighbours)
    assert not fake.starts


# ---- placing the logits ------------------------------------------------------------------------------------------

def _result(logits, scale=4, origin=(0, 0)) -> SegmentResult:
    """``origin`` is logits_origin: the top-left pixel of sample (0, 0), Y up."""
    return SegmentResult(
        logits=logits, scale=float(scale), origin_x=origin[0], origin_y=origin[1], score=0.9, request_id=0, threshold=0.5,
    )


def test_sample_centres_run_right_and_down_from_the_top_left_origin() -> None:
    xs, ys = sample_centres(_result(np.zeros((3, 2), dtype=np.float32), scale=4, origin=(100, 500)))
    assert list(xs) == [102, 106]
    # Sample (i, j) covers pixel rows origin_y - (j + 1) * s + 1 .. origin_y - j * s: centres 499, 495, 491.
    assert list(ys) == [499, 495, 491]


def test_contour_of_a_disc_is_a_ccw_ring_of_the_right_size_and_place() -> None:
    # Disc radius 40 samples (160 px) centred 100 samples from the left and 120 from the top of a 256-sample grid whose
    # top-left corner is (512, 1624).
    result = _result(_disc_logits(256, 100, 120, 40), origin=(512, 1624))
    shape = contour.contour_from_result(result)

    assert shape is not None and not shape.holes
    assert contour.signed_area(shape.exterior) > 0
    area = abs(contour.signed_area(shape.exterior))
    assert abs(area - np.pi * 160**2) < 0.03 * np.pi * 160**2
    cx, cy = shape.exterior[:-1].mean(axis=0)
    # x = origin + 100 samples * 4; y = origin - 120 samples * 4.
    assert abs(cx - (512 + 400)) < 8 and abs(cy - (1624 - 480)) < 8
    assert np.array_equal(shape.exterior[0], shape.exterior[-1])


def test_contour_flips_y_so_the_top_row_is_the_high_y() -> None:
    high = contour.contour_from_result(_result(_disc_logits(256, 128, 30, 12), origin=(0, 1024)))
    low = contour.contour_from_result(_result(_disc_logits(256, 128, 220, 12), origin=(0, 1024)))
    assert high.exterior[:, 1].mean() > low.exterior[:, 1].mean()


def test_logits_origin_moves_the_contour() -> None:
    base = contour.contour_from_result(_result(_disc_logits(64, 32, 32, 10), origin=(0, 0)))
    moved = contour.contour_from_result(_result(_disc_logits(64, 32, 32, 10), origin=(400, -200)))
    assert abs((moved.exterior[:, 0].mean() - base.exterior[:, 0].mean()) - 400) < 1
    assert abs((base.exterior[:, 1].mean() - moved.exterior[:, 1].mean()) - 200) < 1


def test_a_shape_touching_the_grid_edge_still_closes() -> None:
    logits = np.full((64, 64), -6.0, dtype=np.float32)
    logits[:20, :20] = 6.0
    shape = contour.contour_from_result(_result(logits))
    assert shape is not None
    assert abs(abs(contour.signed_area(shape.exterior)) - (20 * 4) ** 2) < 0.1 * (20 * 4) ** 2


def test_real_holes_are_kept_and_tiny_ones_dropped() -> None:
    logits = np.full((200, 200), -6.0, dtype=np.float32)
    logits[20:180, 20:180] = 6.0
    logits[80:120, 80:120] = -6.0  # a hole of 1,600 of 25,600 samples (6%)
    logits[40:42, 40:42] = -6.0    # a 4-sample pin hole
    shape = contour.contour_from_result(_result(logits))
    assert len(shape.holes) == 1
    assert contour.signed_area(shape.holes[0]) < 0


def test_nothing_above_the_level_gives_none() -> None:
    assert contour.contour_from_result(_result(np.full((32, 32), -6.0, dtype=np.float32))) is None


def test_simplified_ring_never_crosses_itself() -> None:
    angles = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    wobble = 100 + 3 * np.sin(angles * 17)
    ring = np.column_stack([wobble * np.cos(angles), wobble * np.sin(angles)])
    ring = np.vstack([ring, ring[:1]])
    for tolerance in (0.5, 2.0, 8.0):
        thinned = contour.simplify_ring(ring, tolerance)
        assert len(thinned) <= len(ring)
        assert not contour.ring_self_intersects(thinned)
    assert len(contour.simplify_ring(ring, 2.0)) < 0.5 * len(ring)


def test_polygon_wkt_is_in_world_pixels() -> None:
    shape = contour.Contour(exterior=np.array([[0, 0], [10, 0], [10, 10], [0, 0]], dtype=float))
    assert contour.polygon_wkt(shape, downsample=2, precision=0) == "POLYGON ((0 0, 20 0, 20 20, 0 0))"


def test_the_servers_answer_timeout_is_retryable_but_a_call_deadline_is_not() -> None:
    class Error(grpc.RpcError):
        def __init__(self, code, details):
            self._code, self._details = code, details

        def code(self):
            return self._code

        def details(self):
            return self._details

    slow = SegmentationClient._failure(Error(grpc.StatusCode.DEADLINE_EXCEEDED, "No TilesAnswer within 60s of TilesNeeded."))
    assert slow.retryable
    assert not SegmentationClient._failure(Error(grpc.StatusCode.DEADLINE_EXCEEDED, "Deadline Exceeded")).retryable


def test_a_regions_tiles_are_fetched_together(tmp_path) -> None:
    import threading
    import time

    from converter.tilesource import TileCache

    active, peak, lock = 0, 0, threading.Lock()

    def slow_get(url, headers):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        import io
        from PIL import Image
        out = io.BytesIO()
        Image.new("L", (4, 4), 9).save(out, format="PNG")
        return 200, {}, out.getvalue()

    cache = TileCache(None, http_get=slow_get)
    cache.prefetch([f"http://t/{i}.png" for i in range(8)])
    assert peak > 1
    assert cache.fetch("http://t/3.png") is not None and cache.requests == 8
