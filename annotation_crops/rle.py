"""Uncompressed column-major COCO RLE for annotation masks.

String ``counts`` stay on the trainer side, where ``pycocotools`` is installed.
This module does not import it.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def encode_coco_rle(mask: np.ndarray) -> dict[str, Any]:
    """Uncompressed COCO RLE (Fortran order).

    ``size`` is ``[height, width]``. A mask that starts on foreground gets a
    leading 0. An empty array is ``counts: []``.
    """
    height = int(mask.shape[0])
    width = int(mask.shape[1])
    if mask.size == 0:
        return {"counts": [], "size": [height, width]}
    pixels = np.asfortranarray(mask).astype(np.uint8, copy=False).ravel(order="F")
    change = np.flatnonzero(pixels[1:] != pixels[:-1]) + 1
    bounds = np.empty(change.size + 2, dtype=np.intp)
    bounds[0] = 0
    bounds[1:-1] = change
    bounds[-1] = pixels.size
    counts = np.diff(bounds).tolist()
    if int(pixels[0]) == 1:
        counts.insert(0, 0)
    return {"counts": counts, "size": [height, width]}


def decode_coco_rle(segmentation: dict[str, Any]) -> np.ndarray:
    """Return an HxW uint8 ``{0, 1}`` mask from uncompressed list ``counts``.

    ``decode(encode(mask))`` equals ``(mask > 0)`` as uint8, Fortran order.
    """
    size = segmentation["size"]
    height, width = int(size[0]), int(size[1])
    counts = segmentation["counts"]
    if not isinstance(counts, list):
        raise TypeError("string RLE counts are decoded by the trainer, not this leaf")
    return _decode_uncompressed(counts, height, width)


def bbox_area_from_rle(rle: dict[str, Any]) -> tuple[list[int], int]:
    """Foreground bounding box and area from a column-major COCO RLE.

    Column is ``index // height`` and row is ``index % height``. A foreground
    run is counted only when ``height`` is non-zero.
    """
    counts = [int(c) for c in rle.get("counts") or []]
    size = rle.get("size") or [0, 0]
    height, _width = int(size[0]), int(size[1])
    area = 0
    value = 0
    offset = 0
    x0 = y0 = x1 = y1 = 0
    found = False
    for run in counts:
        if value == 1 and height and run > 0:
            area += run
            first, last = offset, offset + run - 1
            col_first, col_last = first // height, last // height
            # A run crossing a column boundary covers every row; otherwise
            # it stays inside one column and only its own rows count.
            if col_last > col_first:
                row_lo, row_hi = 0, height - 1
            else:
                row_lo, row_hi = first % height, last % height
            if found:
                x0, x1 = min(x0, col_first), max(x1, col_last + 1)
                y0, y1 = min(y0, row_lo), max(y1, row_hi + 1)
            else:
                x0, x1, y0, y1 = col_first, col_last + 1, row_lo, row_hi + 1
                found = True
        offset += run
        value = 1 - value
    if not found:
        return [0, 0, 0, 0], 0
    return [x0, y0, x1 - x0, y1 - y0], area


def _decode_uncompressed(counts: list[int], height: int, width: int) -> np.ndarray:
    total = height * width
    flat = np.empty(total, dtype=np.uint8)
    idx = 0
    value = 0
    for run in counts:
        run_i = int(run)
        end = idx + run_i
        if end > total:
            raise ValueError(f"RLE overflow: filled {idx}+{run_i} of {total} pixels")
        flat[idx:end] = value
        idx = end
        value = 1 - value
    if idx != total:
        raise ValueError(f"RLE underflow: filled {idx} of {total} pixels")
    return np.ascontiguousarray(flat.reshape((height, width), order="F"))
