"""Client for the segmentation server's tiles-only protocol (SegmentTilesStream), for one object at a time.

The conversation (the comments of seg_proto/segmentation.proto are the specification): one ``start`` naming the call's
TileGrid, the cells under the prompt, the prompt (clicks and shapes) and the background shapes that reach those cells;
then the mask tiles of a painted bitmap, if any (``mask_tile_count``). The server then streams ``TilesNeeded``: the
client uploads the tiles it asks for (UploadTile) and answers with the tiles it could not supply and the background
shapes reaching ``background_cells`` that it has not sent before. The last message is the result: one SegmentResult per
island of the object prompt.

Geometry goes as shapes (a circle with its TRUE radius, a polygon with its holes): the server softens foreground shapes
itself (for a circle, its radius times the model's circle_diameter_scale), so the client never scales or rasterises them.
Only a painted bitmap goes as mask tiles; it is used as drawn.

All coordinates here are mosaic pixels at the request's downsample, X right, **Y up** (world / downsample); the logits
that come back are rows from the top (largest Y) down and are placed by :func:`sample_centres`.
"""

from __future__ import annotations

import io
import logging
import math
import queue
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import grpc
import numpy as np
from PIL import Image

from seg_proto import segmentation_pb2 as pb
from seg_proto import segmentation_pb2_grpc as pb_grpc
from seg_proto.Common import geometry_pb2 as geo

LOGGER = logging.getLogger("converter.seg_client")

TILE_SIZE = 1024

# Request limits of the server (proto comments): refused with RESOURCE_EXHAUSTED, never worth retrying.
MAX_IMAGE_TILES = 64
MAX_MASK_TILES = 256
MAX_SHAPES = 512

Cell = Tuple[int, int]  # (row, col) of the 1024 grid: row = floor(y / 1024), col = floor(x / 1024)


class SegmentationFailed(Exception):
    """The server refused or could not finish this object. ``retryable`` is True only for a lost GPU or connection."""

    def __init__(self, message: str, code: Optional[grpc.StatusCode] = None, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True)
class TileIdentity:
    """The call's TileGrid: everything of a tile except its cell. The converter works in mosaic space only."""

    volume: str
    section: int
    channel: str
    transform: str
    downsample: int
    space: int = geo.COORDINATE_SPACE_MOSAIC

    def grid(self) -> "geo.TileGrid":
        return geo.TileGrid(
            volume=self.volume,
            section=self.section,
            channel=self.channel,
            transform=self.transform,
            downsample=self.downsample,
            space=self.space,
        )


def grid_cell(row: int, col: int) -> "geo.GridCell":
    return geo.GridCell(row=int(row), col=int(col))


def _cell_of(cell: "geo.GridCell") -> Cell:
    return int(cell.row), int(cell.col)


# ---- shapes ----------------------------------------------------------------------------------------------------------

def _vector(x: float, y: float) -> "geo.Vector2":
    return geo.Vector2(x=float(x), y=float(y))


def circle_shape(cx: float, cy: float, radius: float) -> "geo.Shape":
    """A circle with its TRUE radius (the server applies the model's circle_diameter_scale to a foreground circle)."""
    return geo.Shape(circle=geo.Circle(center=_vector(cx, cy), radius=float(radius)))


def _ring(points: Sequence[Tuple[float, float]]) -> "geo.Ring":
    points = list(points)
    if len(points) > 1 and tuple(points[0]) == tuple(points[-1]):
        points = points[:-1]  # a proto ring does not repeat its first vertex
    return geo.Ring(points=[_vector(x, y) for x, y in points])


def polygon_shape(
    exterior: Sequence[Tuple[float, float]], holes: Sequence[Sequence[Tuple[float, float]]] = ()
) -> "geo.Shape":
    """A straight-sided polygon with holes (rings in mosaic pixels at the downsample, Y up)."""
    return geo.Shape(polygon=geo.Polygon(exterior=_ring(exterior), holes=[_ring(hole) for hole in holes]))


def shape_bounds(shape: "geo.Shape") -> Optional[Tuple[float, float, float, float]]:
    """``(x_min, y_min, x_max, y_max)`` of a shape, None for an empty one."""
    kind = shape.WhichOneof("kind")
    if kind == "circle":
        c, r = shape.circle.center, shape.circle.radius
        return c.x - r, c.y - r, c.x + r, c.y + r
    if kind == "ellipse":
        c, a = shape.ellipse.center, max(shape.ellipse.semi_major, shape.ellipse.semi_minor)
        return c.x - a, c.y - a, c.x + a, c.y + a
    if kind == "point":
        return shape.point.x, shape.point.y, shape.point.x, shape.point.y
    if kind == "line":
        points, pad = list(shape.line.points), shape.line.width / 2.0
    elif kind == "polygon":
        points, pad = list(shape.polygon.exterior.points), 0.0
    else:
        return None
    if not points:
        return None
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    return min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad


def cells_in_bounds(bounds: Tuple[float, float, float, float]) -> List[Cell]:
    x_min, y_min, x_max, y_max = bounds
    return [
        (row, col)
        for row in range(math.floor(y_min / TILE_SIZE), math.floor(y_max / TILE_SIZE) + 1)
        for col in range(math.floor(x_min / TILE_SIZE), math.floor(x_max / TILE_SIZE) + 1)
    ]


def shape_cells(shape: "geo.Shape") -> List[Cell]:
    """The grid cells a shape's bounds reach."""
    bounds = shape_bounds(shape)
    return cells_in_bounds(bounds) if bounds is not None else []


# ---- painted bitmaps -------------------------------------------------------------------------------------------------

# A Y-up boolean array whose low corner (row 0 is the lowest Y) is mosaic pixel (x0, y0).
@dataclass(frozen=True)
class MaskArray:
    pixels: np.ndarray
    x0: int
    y0: int

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])

    def tile(self, row: int, col: int) -> np.ndarray:
        """The 1024 x 1024 Y-up window of this mask at grid cell (row, col); False where the mask does not reach."""
        tile = np.zeros((TILE_SIZE, TILE_SIZE), dtype=bool)
        ya, yb = max(self.y0, row * TILE_SIZE), min(self.y0 + self.height, (row + 1) * TILE_SIZE)
        xa, xb = max(self.x0, col * TILE_SIZE), min(self.x0 + self.width, (col + 1) * TILE_SIZE)
        if ya < yb and xa < xb:
            tile[ya - row * TILE_SIZE:yb - row * TILE_SIZE, xa - col * TILE_SIZE:xb - col * TILE_SIZE] = (
                self.pixels[ya - self.y0:yb - self.y0, xa - self.x0:xb - self.x0]
            )
        return tile

    def tile_cells(self) -> List[Cell]:
        if self.width == 0 or self.height == 0:
            return []
        return [
            (row, col)
            for row in range(self.y0 // TILE_SIZE, (self.y0 + self.height - 1) // TILE_SIZE + 1)
            for col in range(self.x0 // TILE_SIZE, (self.x0 + self.width - 1) // TILE_SIZE + 1)
        ]


def encode_mask_png(tile_up: np.ndarray) -> bytes:
    """A mask tile as a greyscale PNG, Y down like the image tile (non-zero is in the mask)."""
    out = io.BytesIO()
    Image.fromarray(np.flipud(tile_up).astype(np.uint8) * 255, mode="L").save(out, format="PNG", optimize=False)
    return out.getvalue()


def make_mask_tile(row: int, col: int, kind: int, tile_up: np.ndarray) -> "pb.MaskTile":
    """One MaskTile: ``full`` when wholly set, an empty message when nothing is set, else a PNG."""
    cell = grid_cell(row, col)
    if not tile_up.any():
        return pb.MaskTile(cell=cell, kind=kind)
    if tile_up.all():
        return pb.MaskTile(cell=cell, kind=kind, full=True)
    return pb.MaskTile(cell=cell, kind=kind, png=encode_mask_png(tile_up))


def bitmap_tiles(mask: MaskArray, kind: int = pb.MASK_KIND_FOREGROUND) -> List["pb.MaskTile"]:
    """The mask tiles of a painted bitmap. A tile with nothing set is left out (the server needs only set tiles)."""
    tiles = []
    for row, col in mask.tile_cells():
        tile = make_mask_tile(row, col, kind, mask.tile(row, col))
        if tile.full or tile.png:
            tiles.append(tile)
    return tiles


def foreground_tiles(mask: MaskArray) -> List["pb.MaskTile"]:
    return bitmap_tiles(mask, pb.MASK_KIND_FOREGROUND)


# Where a call's image tiles and neighbouring annotations come from.
TileProvider = Callable[[int, int], Optional[bytes]]  # (row, col) -> PNG/JPEG bytes of the 1024 tile, None if unavailable
# cells -> the background shapes (neighbouring annotations) that reach any of them. The client drops the ones it has
# already sent in this call, so a provider may return a shape again.
BackgroundProvider = Callable[[Sequence[Cell]], Iterable["geo.Shape"]]


def no_background(cells: Sequence[Cell]) -> List["geo.Shape"]:
    """A background provider with no neighbours."""
    return []


@dataclass
class Prompt:
    """What to segment. Points are (x, y) mosaic pixels, Y up; ``foreground_shapes`` are geometry (circles, polygons)
    the server softens; ``foreground_mask`` is a painted bitmap, used as drawn."""

    foreground_points: Sequence[Tuple[float, float]] = ()
    background_points: Sequence[Tuple[float, float]] = ()
    foreground_shapes: Sequence["geo.Shape"] = ()
    foreground_mask: Optional[MaskArray] = None

    def cells(self) -> List[Cell]:
        """The cells under the object prompt, in first-seen order: clicks, shapes, then the bitmap's set tiles."""
        cells: List[Cell] = []
        for x, y in self.foreground_points:
            cells.append((math.floor(y / TILE_SIZE), math.floor(x / TILE_SIZE)))
        for shape in self.foreground_shapes:
            cells.extend(shape_cells(shape))
        if self.foreground_mask is not None:
            cells.extend(_cell_of(tile.cell) for tile in foreground_tiles(self.foreground_mask))
        return list(dict.fromkeys(cells))


@dataclass
class SegmentResult:
    """The finished object: one island's logits, rows from the top (largest Y) down.

    ``origin_x, origin_y`` is the top-left pixel of sample (0, 0), a grid-space pixel index (Y up, at the downsample;
    pixel (x, y) covers ``[x, x + 1) x [y, y + 1)``): sample (i, j) covers pixel columns ``origin_x + i * scale`` ..
    ``+ scale - 1`` and pixel rows ``origin_y - (j + 1) * scale + 1 .. origin_y - j * scale``. See :func:`sample_centres`.
    """

    logits: np.ndarray  # float32, shape (rows, columns)
    scale: float
    origin_x: int
    origin_y: int
    score: float
    request_id: int
    threshold: float
    island: int = 1
    islands: int = 1  # how many islands the server returned; only ``island`` is kept
    tiles_uploaded: int = 0
    background_shapes_sent: int = 0
    notes: List[str] = field(default_factory=list)


def decode_logits(segment: "pb.SegmentResult") -> np.ndarray:
    """The float16 little-endian logit grid of a segment as float32, shape (rows, columns), row 0 at the top."""
    height, width = segment.logits_size.height, segment.logits_size.width
    if not segment.logits or height <= 0 or width <= 0:
        raise SegmentationFailed("the server returned no logits")
    flat = np.frombuffer(segment.logits, dtype="<f2")
    if flat.size != height * width:
        raise SegmentationFailed(f"logits size {flat.size} does not match {width}x{height}")
    return flat.reshape(height, width).astype(np.float32)


def sample_centres(result: SegmentResult) -> Tuple[np.ndarray, np.ndarray]:
    """Mosaic coordinates (Y up, downsample pixels) of every logit sample centre: ``(xs, ys)``, of length columns / rows.

    Sample (i, j) spans ``scale`` pixels right of ``origin_x + i * scale`` and down from the top of pixel row
    ``origin_y - j * scale`` (at ``origin_y - j * scale + 1``), so its centre is
    ``(origin_x + (i + 0.5) * scale, origin_y + 1 - (j + 0.5) * scale)``.
    """
    rows, cols = result.logits.shape
    xs = result.origin_x + (np.arange(cols) + 0.5) * result.scale
    ys = result.origin_y + 1 - (np.arange(rows) + 0.5) * result.scale
    return xs, ys


def pick_island(segments: Sequence["pb.SegmentResult"]) -> "pb.SegmentResult":
    """The segment of island 1 (the largest trusted area of the object prompt), else the lowest island number.

    The converter prompts one object per location (one circle, or the largest part of a polygon), so a prompt that
    splits means its softened shape fell apart: the main piece is the location; the others are dropped and counted.
    """
    return min(segments, key=lambda segment: (segment.island if segment.island > 0 else 1 << 31, -segment.score))


def _shape_key(shape: "geo.Shape") -> bytes:
    return shape.SerializeToString(deterministic=True)


class SegmentationClient:
    """One gRPC channel to a segmentation server; ``segment`` runs one object through SegmentTilesStream."""

    def __init__(
        self,
        target: str,
        *,
        channel: Optional[grpc.Channel] = None,
        credentials: Optional[grpc.ChannelCredentials] = None,
        model_id: Optional[str] = None,
        idle_timeout_seconds: float = 120.0,
        total_timeout_seconds: float = 900.0,
    ) -> None:
        if channel is None:
            options = [
                ("grpc.max_receive_message_length", 256 * 1024 * 1024),
                ("grpc.max_send_message_length", 256 * 1024 * 1024),
            ]
            channel = (
                grpc.secure_channel(target, credentials, options=options)
                if credentials is not None
                else grpc.insecure_channel(target, options=options)
            )
        self._channel = channel
        self._stub = pb_grpc.SegmentationServiceStub(channel)
        self._model_id = model_id
        self._model_probed = model_id is not None
        self._idle_timeout = idle_timeout_seconds
        self._total_timeout = total_timeout_seconds

    def close(self) -> None:
        self._channel.close()

    def status(self) -> "pb.ServerStatusResponse":
        return self._stub.GetServerStatus(pb.ServerStatusRequest(), timeout=15)

    def model_id(self) -> Optional[str]:
        """The model id requests carry: the one given, else the one the server reports (it refuses requests without it
        when its manifest says so). A server that cannot say leaves it unset."""
        if self._model_id is None and not self._model_probed:
            self._model_probed = True
            try:
                self._model_id = self.status().model.model_id or None
            except grpc.RpcError as error:
                LOGGER.warning("could not read the server's model id: %s", error)
        return self._model_id

    def segment(
        self,
        identity: TileIdentity,
        prompt: Prompt,
        tiles: TileProvider,
        background: BackgroundProvider = no_background,
        *,
        threshold: float = 0.5,
        request_id: int = 0,
    ) -> SegmentResult:
        """Segment one object. ``tiles`` supplies image tiles on demand, ``background`` the neighbours' shapes."""
        foreground = foreground_tiles(prompt.foreground_mask) if prompt.foreground_mask is not None else []
        if len(foreground) > MAX_MASK_TILES:
            raise SegmentationFailed(f"too_large: {len(foreground)} foreground mask tiles (limit {MAX_MASK_TILES})")
        if not (prompt.foreground_points or prompt.foreground_shapes or foreground):
            raise SegmentationFailed("empty_seed: the prompt has no foreground click, shape or mask")
        if len(prompt.foreground_shapes) > MAX_SHAPES:
            raise SegmentationFailed(f"too_large: {len(prompt.foreground_shapes)} foreground shapes (limit {MAX_SHAPES})")
        start_cells = prompt.cells()
        if len(start_cells) > MAX_IMAGE_TILES:
            raise SegmentationFailed(f"too_large: the prompt covers {len(start_cells)} tiles (limit {MAX_IMAGE_TILES})")

        sent_shapes: Set[bytes] = set()

        def new_background(cells: Sequence[Cell]) -> List["geo.Shape"]:
            fresh = []
            for shape in background(list(cells)) or ():
                key = _shape_key(shape)
                if key not in sent_shapes:
                    sent_shapes.add(key)
                    fresh.append(shape)
            return fresh

        start = pb.SegmentTilesRequest(
            grid=identity.grid(),
            tiles=[grid_cell(row, col) for row, col in start_cells],
            mask_threshold=threshold,
            request_id=request_id,
            foreground_feature=pb.Feature(
                points=[_vector(x, y) for x, y in prompt.foreground_points],
                shapes=list(prompt.foreground_shapes),
            ),
            mask_tile_count=len(foreground),
        )
        start.background_feature.points.extend(_vector(x, y) for x, y in prompt.background_points)
        start.background_feature.shapes.extend(new_background(start_cells))
        model_id = self.model_id()
        if model_id and model_id != "unknown":
            start.model_id = model_id

        outbound: "queue.Queue[Optional[pb.SegmentTilesStreamRequest]]" = queue.Queue()

        def requests() -> Iterator["pb.SegmentTilesStreamRequest"]:
            while True:
                item = outbound.get()
                if item is None:
                    return
                yield item

        outbound.put(pb.SegmentTilesStreamRequest(start=start))
        if foreground:
            outbound.put(pb.SegmentTilesStreamRequest(mask_tiles=pb.MaskTiles(tiles=foreground)))

        uploaded = 0
        deadline = time.monotonic() + self._total_timeout
        call = self._stub.SegmentTilesStream(requests(), timeout=self._total_timeout)
        try:
            for message in call:
                kind = message.WhichOneof("update")
                if kind == "result":
                    return self._finish(message.result, threshold, uploaded, len(sent_shapes))
                if kind != "needed":
                    continue
                if time.monotonic() > deadline:
                    raise SegmentationFailed("segmentation exceeded its time limit", grpc.StatusCode.DEADLINE_EXCEEDED)
                needed = message.needed
                ready, unavailable = self._upload(identity, needed.tiles, tiles)
                uploaded += ready
                shapes = new_background([_cell_of(cell) for cell in needed.background_cells])
                outbound.put(
                    pb.SegmentTilesStreamRequest(answer=pb.TilesAnswer(unavailable=unavailable, background_shapes=shapes))
                )
        except grpc.RpcError as error:
            raise self._failure(error) from error
        finally:
            outbound.put(None)
        raise SegmentationFailed("the stream ended without a result")

    def _upload(
        self, identity: TileIdentity, cells: Iterable["geo.GridCell"], tiles: TileProvider
    ) -> Tuple[int, List["geo.GridCell"]]:
        """Upload the cells' tiles: ``(how many were uploaded, the cells that could not be supplied)``."""
        ready = 0
        unavailable: List["geo.GridCell"] = []
        grid = identity.grid()
        for cell in cells:
            data = tiles(cell.row, cell.col)
            if not data:
                unavailable.append(grid_cell(cell.row, cell.col))
                continue
            try:
                self._stub.UploadTile(
                    pb.UploadTileRequest(grid=grid, cell=grid_cell(cell.row, cell.col), image_data=data), timeout=60
                )
            except grpc.RpcError as error:
                raise self._failure(error) from error
            ready += 1
        return ready, unavailable

    def _finish(self, response: "pb.SegmentationResponse", threshold: float, uploaded: int, sent: int) -> SegmentResult:
        if not response.segments:
            raise SegmentationFailed("no_object: the server found no mask for this prompt")
        chosen = pick_island(response.segments)
        notes = []
        if len(response.segments) > 1:
            notes.append(f"the prompt split into {len(response.segments)} islands; kept island {chosen.island}")
            LOGGER.info("request %s: %s", response.request_id, notes[-1])
        return SegmentResult(
            logits=decode_logits(chosen),
            scale=float(chosen.logits_scale) or 1.0,
            origin_x=int(chosen.logits_origin.x),
            origin_y=int(chosen.logits_origin.y),
            score=float(chosen.score),
            request_id=int(response.request_id),
            threshold=threshold,
            island=int(chosen.island) or 1,
            islands=len(response.segments),
            tiles_uploaded=uploaded,
            background_shapes_sent=sent,
            notes=notes,
        )

    @staticmethod
    def _failure(error: grpc.RpcError) -> SegmentationFailed:
        code = error.code() if hasattr(error, "code") else None
        detail = error.details() if hasattr(error, "details") else str(error)
        # The server's own "No TilesAnswer within 60s" means we were slow supplying tiles (a cold or hiccuping tile
        # source); the tiles are cached by then, so the repeat is quick. The call's overall deadline is not retried.
        slow_answer = code == grpc.StatusCode.DEADLINE_EXCEEDED and "TilesAnswer" in str(detail)
        retryable = code in (grpc.StatusCode.UNAVAILABLE,) or slow_answer
        return SegmentationFailed(f"{code.name if code else 'RPC'}: {detail}", code, retryable)

    def segment_with_retry(self, *args, attempts: int = 4, backoff_seconds: float = 5.0, **kwargs) -> SegmentResult:
        """``segment`` retried with growing waits when the server is unavailable or timed out waiting for our tiles;
        nothing else is retried."""
        for attempt in range(attempts):
            try:
                return self.segment(*args, **kwargs)
            except SegmentationFailed as failure:
                if not failure.retryable or attempt == attempts - 1:
                    raise
                wait = backoff_seconds * (2**attempt)
                LOGGER.warning("segmentation unavailable (%s); retrying in %.0fs", failure, wait)
                time.sleep(wait)
        raise AssertionError("unreachable")
