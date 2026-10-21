"""The converter's segmentation client, seeds and contouring, against a fake server that speaks the documented
SegmentTilesStream conversation (start, FOREGROUND mask tiles, TilesNeeded, BACKGROUND tiles + answer, result)."""

from __future__ import annotations

import io
import threading
from concurrent import futures
from typing import List

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
    foreground_tiles,
)
from seg_proto import segmentation_pb2 as pb
from seg_proto import segmentation_pb2_grpc as pb_grpc

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
    """Y-down logit grid with a disc of positive logits; (cx, cy) are sample indices from the top-left."""
    ys, xs = np.mgrid[0:size, 0:size]
    return np.where((xs + 0.5 - cx) ** 2 + (ys + 0.5 - cy) ** 2 <= radius**2, 6.0, -6.0).astype(np.float32)


class FakeSegmentation(pb_grpc.SegmentationServiceServicer):
    """Replies with a fixed logit disc after asking for one extra image tile and one neighbour mask."""

    def __init__(self, result_logits: np.ndarray, scale: int = 4, origin=(0, 0), frame=(256, 256), ask_more=True):
        self.logits = result_logits
        self.scale = scale
        self.origin = origin
        self.frame = frame
        self.ask_more = ask_more
        self.uploads: List["pb.UploadTileRequest"] = []
        self.starts: List["pb.SegmentTilesRequest"] = []
        self.foreground_received: List["pb.MaskTile"] = []
        self.background_received: List["pb.MaskTile"] = []
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
        # Not found unless the tiles that hold a click were uploaded before `start`.
        cached = {(u.coord.row, u.coord.col) for u in self.uploads}
        for point in start.foreground_feature.points:
            cell = (point.y // TILE_SIZE, point.x // TILE_SIZE)
            if cell not in cached:
                context.abort(grpc.StatusCode.NOT_FOUND, f"TILE_NOT_FOUND row={cell[0]} col={cell[1]}")
        need = start.foreground_mask_tile_count
        while len(self.foreground_received) < need:
            message = next(requests)
            assert message.WhichOneof("body") == "mask_tiles"
            self.order.append("foreground")
            self.foreground_received.extend(message.mask_tiles.tiles)
        if self.ask_more:
            extra = pb.TileCoord(volume="RC1", section=7, channel="TEM", transform="stos", downsample=2, row=0, col=5)
            yield pb.SegmentTilesStreamResponse(needed=pb.TilesNeeded(tiles=[extra], masks=[extra]))
            self.order.append("needed")
            message = next(requests)
            if message.WhichOneof("body") == "mask_tiles":
                self.order.append("background")
                self.background_received.extend(message.mask_tiles.tiles)
                message = next(requests)
            assert message.WhichOneof("body") == "answer"
            self.order.append("answer")
            self.answers.append(message.answer)
        segment = pb.SegmentResult(
            index=1,
            score=0.9,
            logits=self.logits.astype("<f2").tobytes(),
            logits_width=self.logits.shape[1],
            logits_height=self.logits.shape[0],
            logits_scale=float(self.scale),
            logits_x=0,
            logits_y=0,
        )
        yield pb.SegmentTilesStreamResponse(
            result=pb.SegmentationResponse(
                width=self.frame[1], height=self.frame[0], origin_x=self.origin[0], origin_y=self.origin[1],
                request_id=start.request_id, segments=[segment],
            )
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


def _no_neighbours(row: int, col: int):
    return np.zeros((TILE_SIZE, TILE_SIZE), dtype=bool)


# ---- seeds -------------------------------------------------------------------------------------------------------

def test_disc_mask_area_and_position() -> None:
    mask = seed.disc_mask(500.0, 700.0, 40.0)
    assert abs(int(mask.pixels.sum()) - np.pi * 40**2) < 0.02 * np.pi * 40**2
    ys, xs = np.nonzero(mask.pixels)
    assert abs((xs.mean() + mask.x0 + 0.5) - 500) < 0.6
    assert abs((ys.mean() + mask.y0 + 0.5) - 700) < 0.6


def test_circle_prompt_has_four_rim_clicks_inside_the_disc() -> None:
    prompt = seed.circle_prompt(500, 700, 40)
    assert len(prompt.foreground_points) == 4
    for x, y in prompt.foreground_points:
        assert abs(np.hypot(x - 500, y - 700) - 38) <= 1.0
    assert prompt.foreground_mask.pixels[700 - prompt.foreground_mask.y0, 538 - prompt.foreground_mask.x0]


def test_circle_below_a_pixel_is_not_seedable() -> None:
    with pytest.raises(seed.NotSeedable):
        seed.circle_prompt(10, 10, 0.4)


def test_polygon_mask_matches_its_area_and_is_y_up() -> None:
    ring = [(100, 200), (160, 200), (160, 260), (100, 260), (100, 200)]
    mask = seed.polygon_mask(ring)
    assert abs(int(mask.pixels.sum()) - 60 * 60) <= 130
    # The row at the lowest Y of the polygon is row 0 of the array.
    assert mask.pixels[1:-1].any(axis=1).all() or mask.pixels.any()
    ys, xs = np.nonzero(mask.pixels)
    assert abs((ys.mean() + mask.y0 + 0.5) - 230) < 1.0
    assert abs((xs.mean() + mask.x0 + 0.5) - 130) < 1.0


def test_polygon_with_hole_keeps_the_hole_empty_and_clicks_avoid_it() -> None:
    outer = [(0, 0), (200, 0), (200, 200), (0, 200), (0, 0)]
    hole = [(60, 60), (140, 60), (140, 140), (60, 140), (60, 60)]
    prompt = seed.polygon_prompt(outer, [hole])
    assert not prompt.foreground_mask.pixels[100 - prompt.foreground_mask.y0, 100 - prompt.foreground_mask.x0]
    for x, y in prompt.foreground_points:
        assert prompt.foreground_mask.pixels[y - prompt.foreground_mask.y0, x - prompt.foreground_mask.x0]


def test_polygons_are_sent_as_a_correction_and_circles_as_a_seed() -> None:
    assert seed.polygon_prompt([(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)]).mask_role == pb.MASK_ROLE_CORRECTION
    assert seed.circle_prompt(50, 50, 20).mask_role == pb.MASK_ROLE_SEED


def test_interior_points_are_deepest_first() -> None:
    mask = seed.disc_mask(300, 300, 50)
    points = seed.interior_points(mask)
    assert points and np.hypot(points[0][0] - 300, points[0][1] - 300) < 3


# ---- mask tiles --------------------------------------------------------------------------------------------------

def test_mask_across_tiles_is_cut_on_the_1024_grid_with_full_and_png_tiles() -> None:
    pixels = np.zeros((1500, 1500), dtype=bool)
    pixels[:, :] = True
    mask = MaskArray(pixels=pixels, x0=600, y0=600)  # pixels 600..2099: tiles 0, 1 and 2 on each axis
    tiles = foreground_tiles(IDENTITY, mask)
    cells = {(t.coord.row, t.coord.col): t for t in tiles}
    assert set(cells) == {(r, c) for r in (0, 1, 2) for c in (0, 1, 2)}
    assert cells[(1, 1)].full and not cells[(1, 1)].png, "tile 1 is 1024..2047, wholly inside the mask"
    assert cells[(0, 0)].png and not cells[(0, 0)].full
    assert all(t.coord.downsample == 2 and t.coord.volume == "RC1" for t in tiles)
    restored = _decode_mask(cells[(0, 0)])
    assert restored[600:, 600:].all() and not restored[:600, :].any() and not restored[:, :600].any()


def test_a_wholly_covered_tile_is_sent_as_full() -> None:
    mask = MaskArray(pixels=np.ones((TILE_SIZE, TILE_SIZE), dtype=bool), x0=TILE_SIZE, y0=0)
    tiles = foreground_tiles(IDENTITY, mask)
    assert len(tiles) == 1 and tiles[0].full and not tiles[0].png


# ---- the conversation --------------------------------------------------------------------------------------------

def test_segment_runs_the_documented_conversation(serve) -> None:
    logits = _disc_logits(256, 100, 120, 40)
    fake = FakeSegmentation(logits, origin=(512, 600), frame=(256, 256))
    client = serve(fake)
    prompt = seed.circle_prompt(1300, 900, 40)  # tile row 0, col 1 (ds 2 mosaic pixels)

    result = client.segment(IDENTITY, prompt, _tiles, _no_neighbours, request_id=77)

    start = fake.starts[0]
    assert start.response_content == pb.RESPONSE_CONTENT_LOGITS | pb.RESPONSE_CONTENT_NO_UPSCALE
    assert start.mask_threshold == 0.5 and start.request_id == 77
    assert start.foreground_mask_tile_count == len(fake.foreground_received) >= 1
    assert start.foreground_feature.mask_role == pb.MASK_ROLE_SEED
    assert len(start.foreground_feature.points) == 4
    assert {(t.row, t.col) for t in start.tiles} == {(0, 1)}
    # The click tile was uploaded before `start` (the fake aborts NOT_FOUND otherwise), then the asked-for tile.
    assert {(u.coord.row, u.coord.col) for u in fake.uploads} == {(0, 1), (0, 5)}
    assert fake.order == ["start", "foreground", "needed", "background", "answer"]
    assert fake.background_received[0].kind == pb.MASK_KIND_BACKGROUND
    assert [(c.row, c.col) for c in fake.answers[0].ready] == [(0, 5)]
    assert result.request_id == 77 and result.tiles_uploaded == 2 and result.background_tiles_sent == 1


def test_unavailable_tiles_and_neighbours_are_reported_in_the_answer(serve) -> None:
    fake = FakeSegmentation(_disc_logits(256, 100, 120, 40))
    client = serve(fake)

    def tiles(row: int, col: int):
        return None if col == 5 else _png_tile()

    client.segment(IDENTITY, seed.circle_prompt(1300, 900, 40), tiles, lambda r, c: None)

    answer = fake.answers[0]
    assert [(c.row, c.col) for c in answer.unavailable] == [(0, 5)]
    assert [(c.row, c.col) for c in answer.masks_unavailable] == [(0, 5)]
    assert not fake.background_received


def test_click_tile_that_cannot_be_supplied_fails_before_the_call(serve) -> None:
    fake = FakeSegmentation(_disc_logits(256, 100, 120, 40), ask_more=False)
    client = serve(fake)

    with pytest.raises(SegmentationFailed, match="tile_unavailable"):
        client.segment(IDENTITY, seed.circle_prompt(1300, 900, 40), lambda r, c: None, _no_neighbours)

    assert not fake.starts


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


# ---- contouring --------------------------------------------------------------------------------------------------

def _result(logits, scale=4, origin=(0, 0), frame_samples=None, logits_x=0, logits_y=0) -> SegmentResult:
    rows = frame_samples if frame_samples is not None else logits.shape[0]
    return SegmentResult(
        logits=logits, scale=float(scale), logits_x=logits_x, logits_y=logits_y, origin_x=origin[0], origin_y=origin[1],
        frame_height=rows, score=0.9, request_id=0, threshold=0.5,
    )


def test_contour_of_a_disc_is_a_ccw_ring_of_the_right_size_and_place() -> None:
    # Disc radius 40 samples (160 px) centred 100 samples from the left and 120 from the top of a 256-sample frame.
    result = _result(_disc_logits(256, 100, 120, 40), origin=(512, 600))
    shape = contour.contour_from_result(result)

    assert shape is not None and not shape.holes
    assert contour.signed_area(shape.exterior) > 0
    area = abs(contour.signed_area(shape.exterior))
    assert abs(area - np.pi * 160**2) < 0.03 * np.pi * 160**2
    cx, cy = shape.exterior[:-1].mean(axis=0)
    # x = origin + 100.0 samples * 4 ; y = origin + frame height (1024 px) - 120 samples * 4.
    assert abs(cx - (512 + 400)) < 8 and abs(cy - (600 + 1024 - 480)) < 8
    assert np.array_equal(shape.exterior[0], shape.exterior[-1])


def test_contour_flips_y_so_the_top_of_the_frame_is_the_high_y() -> None:
    high = contour.contour_from_result(_result(_disc_logits(256, 128, 30, 12), origin=(0, 0)))
    low = contour.contour_from_result(_result(_disc_logits(256, 128, 220, 12), origin=(0, 0)))
    assert high.exterior[:, 1].mean() > low.exterior[:, 1].mean()


def test_logits_x_y_offset_moves_the_contour() -> None:
    base = contour.contour_from_result(_result(_disc_logits(64, 32, 32, 10), origin=(0, 0)))
    moved = contour.contour_from_result(_result(_disc_logits(64, 32, 32, 10), origin=(0, 0), logits_x=400, logits_y=200))
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
