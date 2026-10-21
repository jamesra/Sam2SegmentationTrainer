"""Client for the segmentation server's tiles-only protocol (SegmentTilesStream), for one object at a time.

The conversation (see Servers/SegmentationServer/README.md and the proto comments): one ``start`` with the prompts, then
its FOREGROUND mask tiles; the server then streams ``TilesNeeded``. For each it asks for, the client uploads the image
tiles (UploadTile), sends BACKGROUND mask tiles, and answers; the last message is the result.

All coordinates here are mosaic pixels at the request's downsample, X right, **Y up** (world / downsample); the logits
that come back are Y-down samples and are mapped to Y-up by :func:`logits_to_grid`.
"""

from __future__ import annotations

import io
import logging
import queue
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, List, Optional, Sequence, Tuple

import grpc
import numpy as np
from PIL import Image

from seg_proto import segmentation_pb2 as pb
from seg_proto import segmentation_pb2_grpc as pb_grpc

LOGGER = logging.getLogger("converter.seg_client")

TILE_SIZE = 1024

# Request limits of the server (README): refused with RESOURCE_EXHAUSTED, never worth retrying.
MAX_IMAGE_TILES = 64
MAX_MASK_TILES = 256


class SegmentationFailed(Exception):
    """The server refused or could not finish this object. ``retryable`` is True only for a lost GPU or connection."""

    def __init__(self, message: str, code: Optional[grpc.StatusCode] = None, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True)
class TileIdentity:
    """Everything of a TileCoord except row and column."""

    volume: str
    section: int
    channel: str
    transform: str
    downsample: int

    def coord(self, row: int, col: int) -> "pb.TileCoord":
        return pb.TileCoord(
            volume=self.volume,
            section=self.section,
            channel=self.channel,
            transform=self.transform,
            downsample=self.downsample,
            row=row,
            col=col,
        )


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

    def tile_cells(self) -> List[Tuple[int, int]]:
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


def make_mask_tile(identity: TileIdentity, row: int, col: int, kind: int, tile_up: np.ndarray) -> "pb.MaskTile":
    """One MaskTile: ``full`` when wholly set, an empty message when nothing is set, else a PNG."""
    coord = identity.coord(row, col)
    if not tile_up.any():
        return pb.MaskTile(coord=coord, kind=kind)
    if tile_up.all():
        return pb.MaskTile(coord=coord, kind=kind, full=True)
    return pb.MaskTile(coord=coord, kind=kind, png=encode_mask_png(tile_up))


def foreground_tiles(identity: TileIdentity, mask: MaskArray) -> List["pb.MaskTile"]:
    """The FOREGROUND tiles a mask touches. A tile with nothing set is left out (the server needs only set tiles)."""
    tiles = []
    for row, col in mask.tile_cells():
        tile = make_mask_tile(identity, row, col, pb.MASK_KIND_FOREGROUND, mask.tile(row, col))
        if tile.full or tile.png:
            tiles.append(tile)
    return tiles


# Where a window's image tile or neighbour mask comes from.
TileProvider = Callable[[int, int], Optional[bytes]]  # (row, col) -> PNG/JPEG bytes of the 1024 tile, None if unavailable
BackgroundProvider = Callable[[int, int], Optional[np.ndarray]]  # (row, col) -> Y-up bool tile, None if unavailable


@dataclass
class Prompt:
    """What to segment. Points are (x, y) mosaic pixels, Y up."""

    foreground_points: Sequence[Tuple[int, int]] = ()
    background_points: Sequence[Tuple[int, int]] = ()
    foreground_mask: Optional[MaskArray] = None
    mask_role: int = pb.MASK_ROLE_SEED


@dataclass
class SegmentResult:
    """The finished object. ``logits`` is Y-down (row 0 is the top); see :func:`logits_to_grid` for the world mapping."""

    logits: np.ndarray  # float32, shape (logits_height, logits_width)
    scale: float
    logits_x: int
    logits_y: int
    origin_x: int
    origin_y: int
    frame_height: int  # response.height, in the reduced frame the logits live in
    score: float
    request_id: int
    threshold: float
    tiles_uploaded: int = 0
    background_tiles_sent: int = 0
    notes: List[str] = field(default_factory=list)

    @property
    def frame_height_pixels(self) -> int:
        return int(round(self.frame_height * self.scale))


def decode_logits(segment: "pb.SegmentResult") -> np.ndarray:
    """The float16 little-endian logit grid of a segment as float32, shape (height, width)."""
    height, width = segment.logits_height, segment.logits_width
    if not segment.logits or height <= 0 or width <= 0:
        raise SegmentationFailed("the server returned no logits (is the model's response LOGITS-capable?)")
    flat = np.frombuffer(segment.logits, dtype="<f2")
    if flat.size != height * width:
        raise SegmentationFailed(f"logits size {flat.size} does not match {width}x{height}")
    return flat.reshape(height, width).astype(np.float32)


def sample_centres(result: SegmentResult) -> Tuple[np.ndarray, np.ndarray]:
    """Mosaic coordinates (Y up, downsample pixels) of every logit sample centre: ``(xs, ys)``, each of length w / h.

    Sample (i, j) covers image pixels ``[logits_x + i * scale, +scale)`` across and ``[logits_y + j * scale, +scale)``
    down from the top of the frame, so its centre is ``(i + 0.5) * scale`` in from its edge, and the frame is flipped about
    ``origin_y + frame height`` (the mosaic's top), the way SegmentationTileGrid.MosaicPixelToWorld does it.
    """
    rows, cols = result.logits.shape
    px = result.logits_x + (np.arange(cols) + 0.5) * result.scale
    py = result.logits_y + (np.arange(rows) + 0.5) * result.scale
    return result.origin_x + px, result.origin_y + result.frame_height_pixels - py


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
        self._idle_timeout = idle_timeout_seconds
        self._total_timeout = total_timeout_seconds

    def close(self) -> None:
        self._channel.close()

    def status(self) -> "pb.ServerStatusResponse":
        return self._stub.GetServerStatus(pb.ServerStatusRequest(), timeout=15)

    def segment(
        self,
        identity: TileIdentity,
        prompt: Prompt,
        tiles: TileProvider,
        background: BackgroundProvider,
        *,
        threshold: float = 0.5,
        request_id: int = 0,
    ) -> SegmentResult:
        """Segment one object. ``tiles`` supplies image tiles on demand, ``background`` the neighbours' masks."""
        foreground = foreground_tiles(identity, prompt.foreground_mask) if prompt.foreground_mask is not None else []
        if len(foreground) > MAX_MASK_TILES:
            raise SegmentationFailed(f"too_large: {len(foreground)} foreground mask tiles (limit {MAX_MASK_TILES})")
        if not (prompt.foreground_points or foreground):
            raise SegmentationFailed("empty_seed: the prompt has no foreground click and no foreground mask")

        start = pb.SegmentTilesRequest(
            multimask_output=False,
            omit_labeled_image=True,
            mask_threshold=threshold,
            request_id=request_id,
            response_content=pb.RESPONSE_CONTENT_LOGITS | pb.RESPONSE_CONTENT_NO_UPSCALE,
            foreground_feature=pb.Feature(
                points=[pb.Point(x=int(x), y=int(y)) for x, y in prompt.foreground_points],
                mask_role=prompt.mask_role if foreground else pb.MASK_ROLE_UNSPECIFIED,
            ),
            foreground_mask_tile_count=len(foreground),
        )
        if prompt.background_points:
            start.background_feature.points.extend(pb.Point(x=int(x), y=int(y)) for x, y in prompt.background_points)
        if self._model_id:
            start.model_id = self._model_id
        # A tile holding a foreground click must be cached before `start` (else NOT_FOUND), and every tile listed in
        # start.tiles is pinned for the call. A mask-only prompt lists its first foreground tile.
        start_cells: List[Tuple[int, int]] = []
        for x, y in prompt.foreground_points:
            cell = (int(y) // TILE_SIZE, int(x) // TILE_SIZE)
            if cell not in start_cells:
                start_cells.append(cell)
        if not start_cells:
            start_cells.append((foreground[0].coord.row, foreground[0].coord.col))
        start_coords = [identity.coord(row, col) for row, col in start_cells]
        start.tiles.extend(start_coords)
        ready_at_start, missing_at_start = self._upload(identity, start_coords, tiles)
        if missing_at_start:
            gone = missing_at_start[0]
            raise SegmentationFailed(f"tile_unavailable: image tile row={gone.row} col={gone.col} could not be supplied")

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

        uploaded = len(ready_at_start)
        sent_background = 0
        deadline = time.monotonic() + self._total_timeout
        call = self._stub.SegmentTilesStream(requests(), timeout=self._total_timeout)
        try:
            for message in call:
                kind = message.WhichOneof("update")
                if kind == "result":
                    return self._finish(message.result, threshold, uploaded, sent_background)
                if kind != "needed":
                    continue
                if time.monotonic() > deadline:
                    raise SegmentationFailed("segmentation exceeded its time limit", grpc.StatusCode.DEADLINE_EXCEEDED)
                needed = message.needed
                ready, unavailable = self._upload(identity, needed.tiles, tiles)
                uploaded += len(ready)
                masks_unavailable: List["pb.TileCoord"] = []
                mask_messages: List["pb.MaskTile"] = []
                for coord in needed.masks:
                    tile_up = background(coord.row, coord.col)
                    if tile_up is None:
                        masks_unavailable.append(coord)
                    else:
                        mask_messages.append(
                            make_mask_tile(identity, coord.row, coord.col, pb.MASK_KIND_BACKGROUND, tile_up)
                        )
                if mask_messages:
                    outbound.put(pb.SegmentTilesStreamRequest(mask_tiles=pb.MaskTiles(tiles=mask_messages)))
                    sent_background += len(mask_messages)
                outbound.put(
                    pb.SegmentTilesStreamRequest(
                        answer=pb.TilesAnswer(ready=ready, unavailable=unavailable, masks_unavailable=masks_unavailable)
                    )
                )
        except grpc.RpcError as error:
            raise self._failure(error) from error
        finally:
            outbound.put(None)
        raise SegmentationFailed("the stream ended without a result")

    def _upload(
        self, identity: TileIdentity, coords: Iterable["pb.TileCoord"], tiles: TileProvider
    ) -> Tuple[List["pb.TileCoord"], List["pb.TileCoord"]]:
        ready: List["pb.TileCoord"] = []
        unavailable: List["pb.TileCoord"] = []
        for coord in coords:
            data = tiles(coord.row, coord.col)
            if not data:
                unavailable.append(coord)
                continue
            try:
                self._stub.UploadTile(
                    pb.UploadTileRequest(coord=coord, image_data=data, width=TILE_SIZE, height=TILE_SIZE), timeout=60
                )
            except grpc.RpcError as error:
                raise self._failure(error) from error
            ready.append(coord)
        return ready, unavailable

    def _finish(self, response: "pb.SegmentationResponse", threshold: float, uploaded: int, sent: int) -> SegmentResult:
        if not response.segments:
            raise SegmentationFailed("no_object: the server found no mask for this prompt")
        best = max(response.segments, key=lambda segment: segment.score)
        return SegmentResult(
            logits=decode_logits(best),
            scale=float(best.logits_scale) or 1.0,
            logits_x=int(best.logits_x),
            logits_y=int(best.logits_y),
            origin_x=int(response.origin_x),
            origin_y=int(response.origin_y),
            frame_height=int(response.height),
            score=float(best.score),
            request_id=int(response.request_id),
            threshold=threshold,
            tiles_uploaded=uploaded,
            background_tiles_sent=sent,
        )

    @staticmethod
    def _failure(error: grpc.RpcError) -> SegmentationFailed:
        code = error.code() if hasattr(error, "code") else None
        detail = error.details() if hasattr(error, "details") else str(error)
        retryable = code in (grpc.StatusCode.UNAVAILABLE,)
        return SegmentationFailed(f"{code.name if code else 'RPC'}: {detail}", code, retryable)

    def segment_with_retry(self, *args, attempts: int = 4, backoff_seconds: float = 5.0, **kwargs) -> SegmentResult:
        """``segment`` retried with growing waits when the server is unavailable; nothing else is retried."""
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
