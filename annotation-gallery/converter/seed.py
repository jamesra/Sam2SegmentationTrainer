"""Seeds for the segmentation server: what we already know about a location, as a mask plus a few clicks.

Everything is in mosaic pixels at the request's downsample, X right, Y up, like the protocol.

* A circle (LocationType 1) is a disc mask plus four clicks on its rim at 0.95 r, the encoding Viking's own segmentation
  command uses for circles.
* A polygon, curve polygon or closed curve (4, 6, 7) is the filled polygon plus clicks inside it: the point farthest from
  the edge and one more in each of up to three well-separated deep parts.
* Points and open curves are not seedable.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

from converter.seg_client import MaskArray, Prompt
from seg_proto import segmentation_pb2 as pb

POINT, CIRCLE, ELLIPSE, POLYLINE, POLYGON, OPENCURVE, CURVEPOLYGON, CLOSEDCURVE = range(8)
SEEDABLE = (CIRCLE, POLYGON, CURVEPOLYGON, CLOSEDCURVE)

RIM_FRACTION = 0.95
# Clicks stay at least this many pixels inside the edge, so a thin shape still gets one.
MIN_EDGE_DISTANCE = 2.0


class NotSeedable(Exception):
    """The location cannot be turned into a seed; the message is the reason recorded on the card."""


def disc_mask(cx: float, cy: float, radius: float) -> MaskArray:
    """A filled disc of the given radius centred on mosaic pixel ``(cx, cy)`` (Y up)."""
    r = max(1.0, radius)
    x0 = int(math.floor(cx - r)) - 1
    y0 = int(math.floor(cy - r)) - 1
    size = int(math.ceil(2 * r)) + 3
    ys, xs = np.mgrid[0:size, 0:size]
    pixels = (xs + x0 + 0.5 - cx) ** 2 + (ys + y0 + 0.5 - cy) ** 2 <= r * r
    return MaskArray(pixels=pixels, x0=x0, y0=y0)


def polygon_mask(ring: Sequence[Tuple[float, float]], holes: Sequence[Sequence[Tuple[float, float]]] = ()) -> MaskArray:
    """A filled polygon (exterior minus holes) rasterised by pixel centre, Y up. ``ring`` is mosaic pixels."""
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    x0, y0 = int(math.floor(min(xs))) - 1, int(math.floor(min(ys))) - 1
    width = int(math.ceil(max(xs))) + 2 - x0
    height = int(math.ceil(max(ys))) + 2 - y0
    # Drawn Y-down in a PIL image, then flipped back to Y up. A pixel centre sits at +0.5 from its edge.
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)

    def to_image(polygon: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
        return [(x - x0 - 0.5, (height - 1) - (y - y0 - 0.5)) for x, y in polygon]

    draw.polygon(to_image(ring), fill=255)
    for hole in holes:
        draw.polygon(to_image(hole), fill=0)
    pixels = np.flipud(np.asarray(image) > 0)
    return MaskArray(pixels=pixels, x0=x0, y0=y0)


def interior_points(mask: MaskArray, count: int = 4) -> List[Tuple[int, int]]:
    """Clicks inside a mask: the pole of inaccessibility, then up to ``count - 1`` more in separate deep parts."""
    padded = np.pad(mask.pixels, 1)
    depth = distance_transform_edt(padded)[1:-1, 1:-1]
    if depth.max() <= 0:
        return []
    points: List[Tuple[int, int]] = []
    remaining = depth.copy()
    separation = None
    for _ in range(count):
        peak = float(remaining.max())
        if peak < MIN_EDGE_DISTANCE and points:
            break
        y, x = np.unravel_index(int(np.argmax(remaining)), remaining.shape)
        points.append((int(x) + mask.x0, int(y) + mask.y0))
        if separation is None:
            separation = max(peak, 1.0)
        yy, xx = np.ogrid[:remaining.shape[0], :remaining.shape[1]]
        remaining[(yy - y) ** 2 + (xx - x) ** 2 <= (0.8 * separation) ** 2] = 0
        if not remaining.any():
            break
    return points


def circle_prompt(cx: float, cy: float, radius: float) -> Prompt:
    """Disc seed plus four rim clicks at 0.95 r (left, right, top, bottom)."""
    if radius < 1.0:
        raise NotSeedable("circle radius is below one pixel at this downsample")
    rim = RIM_FRACTION * radius
    clicks = [
        (int(round(cx - rim)), int(round(cy))),
        (int(round(cx + rim)), int(round(cy))),
        (int(round(cx)), int(round(cy - rim))),
        (int(round(cx)), int(round(cy + rim))),
    ]
    return Prompt(foreground_points=clicks, foreground_mask=disc_mask(cx, cy, radius), mask_role=pb.MASK_ROLE_SEED)


def polygon_prompt(ring: Sequence[Tuple[float, float]], holes: Sequence[Sequence[Tuple[float, float]]] = ()) -> Prompt:
    if len(ring) < 4:
        raise NotSeedable("polygon has fewer than three vertices")
    mask = polygon_mask(ring, holes)
    clicks = interior_points(mask)
    if not clicks:
        raise NotSeedable("polygon is empty at this downsample")
    return Prompt(foreground_points=clicks, foreground_mask=mask, mask_role=pb.MASK_ROLE_SEED)
