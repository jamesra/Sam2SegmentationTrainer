"""COCO RLE decode for AnnotationCrops sidecar JSON."""

from __future__ import annotations

from typing import Any

import numpy as np

from annotation_crops.rle import decode_coco_rle as decode_uncompressed_rle


def decode_coco_rle(segmentation: dict[str, Any]) -> np.ndarray:
    """Return a HxW uint8 {0,1} mask from a COCO RLE dict (`counts` + `size`)."""
    counts = segmentation["counts"]
    if isinstance(counts, list):
        return decode_uncompressed_rle(segmentation)
    size = segmentation["size"]
    height, width = int(size[0]), int(size[1])
    from pycocotools import mask as mask_utils

    rle = {"size": [height, width], "counts": counts}
    if isinstance(counts, str):
        rle = mask_utils.frPyObjects(rle, height, width)
    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = decoded[:, :, 0]
    return (decoded > 0).astype(np.uint8)
