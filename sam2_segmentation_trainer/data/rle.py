"""COCO RLE decode for AnnotationCrops sidecar JSON."""

from __future__ import annotations

from typing import Any

import numpy as np


def decode_coco_rle(segmentation: dict[str, Any]) -> np.ndarray:
    """Return a HxW uint8 {0,1} mask from a COCO RLE dict (`counts` + `size`)."""
    size = segmentation["size"]
    height, width = int(size[0]), int(size[1])
    counts = segmentation["counts"]
    if isinstance(counts, list):
        return _decode_uncompressed(counts, height, width)
    from pycocotools import mask as mask_utils

    rle = {"size": [height, width], "counts": counts}
    if isinstance(counts, str):
        rle = mask_utils.frPyObjects(rle, height, width)
    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = decoded[:, :, 0]
    return (decoded > 0).astype(np.uint8)


def _decode_uncompressed(counts: list[int], height: int, width: int) -> np.ndarray:
    # COCO uncompressed RLE is column-major (Fortran order).
    total = height * width
    flat = np.empty(total, dtype=np.uint8)
    idx = 0
    value = 0
    for run in counts:
        run_i = int(run)
        end = idx + run_i
        if end > total:
            raise ValueError(
                f"RLE overflow: filled {idx}+{run_i} of {total} pixels"
            )
        flat[idx:end] = value
        idx = end
        value = 1 - value
    if idx != total:
        raise ValueError(f"RLE underflow: filled {idx} of {total} pixels")
    return np.ascontiguousarray(flat.reshape((height, width), order="F"))
