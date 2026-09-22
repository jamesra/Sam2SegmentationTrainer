from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def encode_uncompressed(mask: np.ndarray) -> list[int]:
    flat = np.asarray(mask).reshape(-1, order="F")
    counts: list[int] = []
    value = 0
    run = 0
    for pix in flat:
        bit = 1 if pix else 0
        if bit == value:
            run += 1
        else:
            counts.append(run)
            value = bit
            run = 1
    counts.append(run)
    return counts


def write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def make_crops(root: Path, volume: str, n: int = 8) -> None:
    crops = root / volume / "AnnotationCrops"
    images = crops / "images"
    masks = crops / "masks"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    rows = []
    loc_id = 1
    for i in range(n):
        downsample = 1 if i < n // 2 else 4
        key = f"{volume}_{i}_D{downsample}_X0-1_Y0-1"
        gray = np.full((32, 32), 40 + i, dtype=np.uint8)
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:24, 8:24] = 1
        extra = i % 3 == 0
        loc_ids = [loc_id]
        write_png(images / f"{key}.png", gray)
        write_png(masks / f"{key}_{loc_id}.png", mask * 255)
        anns = [
            {
                "id": loc_id,
                "segmentation": {
                    "counts": encode_uncompressed(mask),
                    "size": [32, 32],
                },
                "bbox": [8, 8, 16, 16],
                "area": int(mask.sum()),
                "category_name": "MC" if i % 2 == 0 else "HC",
            }
        ]
        loc_id += 1
        if extra:
            mask2 = np.zeros((32, 32), dtype=np.uint8)
            mask2[2:6, 2:6] = 1
            loc_ids.append(loc_id)
            write_png(masks / f"{key}_{loc_id}.png", mask2 * 255)
            anns.append(
                {
                    "id": loc_id,
                    "segmentation": {
                        "counts": encode_uncompressed(mask2),
                        "size": [32, 32],
                    },
                    "bbox": [2, 2, 4, 4],
                    "area": int(mask2.sum()),
                    "category_name": "Cell",
                }
            )
            loc_id += 1
        sidecar = {
            "image": {
                "file_name": f"{key}.png",
                "width": 32,
                "height": 32,
                "downsample": downsample,
                "volume": volume,
            },
            "annotations": anns,
        }
        (images / f"{key}.json").write_text(json.dumps(sidecar), encoding="utf-8")
        rows.append(
            {
                "z": i + 1,
                "volume": volume,
                "imageKey": key,
                "downsample": downsample,
                "image": f"images/{key}.png",
                "json": f"images/{key}.json",
                "locationIds": loc_ids,
            }
        )
    (crops / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
