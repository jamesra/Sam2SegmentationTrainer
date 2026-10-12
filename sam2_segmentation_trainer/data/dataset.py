"""PyTorch Dataset over AnnotationCrops examples."""

from __future__ import annotations

import json
import multiprocessing
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from sam2_segmentation_trainer.data.augment import (
    build_train_transform,
    build_val_transform,
    to_rgb_uint8,
)
from sam2_segmentation_trainer.data.manifest import CropExample
from sam2_segmentation_trainer.data.meta import SampleMeta

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class MissingTileFlag:
    """Set when a listed tile is missing so the next epoch rebuilds the inventory.

    DataLoader workers share the flag. It stays set until the training loop takes it.
    """

    _missing: multiprocessing.synchronize.Synchronized

    def __init__(self) -> None:
        self._missing = multiprocessing.Value("b", 0)

    def note(self) -> None:
        """Record that a listed tile could not be opened."""
        self._missing.value = 1

    def take(self) -> bool:
        """Return whether a tile was missing since the last take, then clear the flag."""
        with self._missing.get_lock():
            flagged = bool(self._missing.value)
            self._missing.value = 0
            return flagged


def _load_gray_uint8(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("L"))


def _annotation_for_id(json_path: Path, location_id: int) -> dict[str, Any]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    for ann in payload.get("annotations") or []:
        if int(ann.get("id", -1)) == location_id:
            return ann
    raise KeyError(f"{json_path} has no annotation id={location_id}")


def load_binary_mask(example: CropExample) -> tuple[np.ndarray, dict[str, Any]]:
    """Load the raster mask PNG. Missing files are not trained (no RLE fallback)."""
    ann = _annotation_for_id(example.json_path, example.location_id)
    raw = _load_gray_uint8(example.mask_path)
    return (raw > 127).astype(np.uint8), ann


def mask_centroid_xy(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def bbox_center_xy(ann: dict[str, Any], width: int, height: int) -> tuple[float, float]:
    bbox = ann.get("bbox")
    if bbox is not None and len(bbox) == 4:
        x, y, w, h = (float(v) for v in bbox)
        return x + w / 2.0, y + h / 2.0
    return width / 2.0, height / 2.0


def imagenet_normalize(rgb_uint8: np.ndarray) -> torch.Tensor:
    arr = rgb_uint8.astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


class EMSegDataset(Dataset):
    def __init__(
        self,
        examples: Sequence[CropExample],
        *,
        split: str = "train",
        image_size: int = 1024,
        missing_tiles: MissingTileFlag | None = None,
    ):
        self.examples = list(examples)
        self.split = split
        self.image_size = image_size
        self.missing_tiles = missing_tiles
        if split == "train":
            self.transform = build_train_transform()
        else:
            self.transform = build_val_transform()

    def __len__(self) -> int:
        return len(self.examples)

    def replace_examples(self, examples: Sequence[CropExample]) -> None:
        """Swap the inventory in place. The next DataLoader uses this list."""
        self.examples = list(examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # Indexing already drops missing locationIds/masks; still skip rare races.
        n = len(self.examples)
        if n == 0:
            raise RuntimeError("EMSegDataset is empty")
        last_err: Exception | None = None
        for offset in range(n):
            example = self.examples[(int(idx) + offset) % n]
            try:
                return self._item_from_example(example)
            except (KeyError, FileNotFoundError) as exc:
                if isinstance(exc, FileNotFoundError) and self.missing_tiles is not None:
                    self.missing_tiles.note()
                last_err = exc
                continue
        assert last_err is not None
        raise last_err

    def _item_from_example(self, example: CropExample) -> dict[str, Any]:
        gray = _load_gray_uint8(example.image_path)
        rgb = to_rgb_uint8(gray)
        mask, ann = load_binary_mask(example)
        if mask.shape[:2] != rgb.shape[:2]:
            mask = np.array(
                Image.fromarray(mask * 255, mode="L").resize(
                    (rgb.shape[1], rgb.shape[0]), resample=Image.NEAREST
                )
            )
            mask = (mask > 127).astype(np.uint8)

        augmented = self.transform(image=rgb, mask=mask)
        image_aug = augmented["image"]
        mask_aug = augmented["mask"]
        if mask_aug.ndim == 3:
            mask_aug = mask_aug[:, :, 0]
        mask_bin = (np.asarray(mask_aug) > 0).astype(np.uint8)

        centroid = mask_centroid_xy(mask_bin)
        if centroid is None:
            h, w = mask_bin.shape
            centroid = bbox_center_xy(ann, w, h)

        image = imagenet_normalize(np.ascontiguousarray(image_aug))
        point = torch.tensor([[centroid[0], centroid[1]]], dtype=torch.float32)
        meta = SampleMeta(
            volume=example.volume,
            image_key=example.image_key,
            location_id=example.location_id,
            downsample=example.downsample,
            category_name=str(ann.get("category_name") or ann.get("structure_label") or ""),
            split_key=example.split_key,
        )
        return {
            "image": image,
            "mask": torch.from_numpy(mask_bin.astype(np.int64)),
            "point": point,
            "point_label": torch.tensor([1], dtype=torch.int32),
            "orig_hw": (int(mask_bin.shape[0]), int(mask_bin.shape[1])),
            "sample_meta": meta,
            **meta.as_dict(),
        }


def collate_em(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "mask": torch.stack([item["mask"] for item in batch], dim=0),
        "point": torch.stack([item["point"] for item in batch], dim=0),
        "point_label": torch.stack([item["point_label"] for item in batch], dim=0),
        "orig_hw": [item["orig_hw"] for item in batch],
        "meta": [item["sample_meta"].as_dict() for item in batch],
    }
