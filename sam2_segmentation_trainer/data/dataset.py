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

from annotation_crops.negatives import (
    MaskCache,
    NegativeRegion,
    collect_negative_regions,
    drop_circles_over_target,
    rasterize_circle,
)

from sam2_segmentation_trainer.data.augment import (
    build_train_transform,
    build_val_transform,
    to_rgb_uint8,
)
from sam2_segmentation_trainer.data.manifest import CropExample
from sam2_segmentation_trainer.data.meta import SampleMeta
from sam2_segmentation_trainer.data.prompts import (
    PromptConfig,
    PromptType,
    FULL_TILE_PROMPTS,
    build_prompt,
    choose_prompt_type,
    inscribed_circle_xyr,
    interior_point_xy,
)
from sam2_segmentation_trainer.data.fulltile import is_full_tile
from sam2_segmentation_trainer.data.tiles import TileContext

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

VAL_PROMPTS = ("centroid", "point", "circle_mask")
"""Validation prompts: the legacy mask centroid, the deepest interior point, and the inscribed circle (production)."""


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


class EpochCounter:
    """Epoch number shared with DataLoader workers.

    Persistent workers keep a copy of the dataset from the first epoch, so a plain attribute set in the main
    process would never reach them. The training loop calls :meth:`set` at the start of every epoch and the
    dataset reads it to seed its prompt draws, which makes a resumed epoch replay the same prompts.
    """

    _value: multiprocessing.synchronize.Synchronized

    def __init__(self) -> None:
        self._value = multiprocessing.Value("i", 0)

    def set(self, epoch: int) -> None:
        """Publish the current epoch."""
        self._value.value = int(epoch)

    def get(self) -> int:
        """The epoch last published, 0 before the first :meth:`set`."""
        return int(self._value.value)


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
    """Crops with a prompt per sample.

    Without ``tiles`` and ``prompt_cfg`` every sample is the original single-centroid-point prompt, which is
    what validation and evaluation use so their IoU stays comparable across runs. With both, each sample draws
    a prompt type (points, box, box and points, circle mask, full mask, negative only) and carries the other
    annotations in its tile as negative context; see :mod:`sam2_segmentation_trainer.data.prompts`.

    Prompt draws are seeded from ``(seed, epoch, index)`` so a resumed epoch sees the same prompts. The mask
    cache is per process, so each DataLoader worker keeps its own.
    """

    def __init__(
        self,
        examples: Sequence[CropExample],
        *,
        split: str = "train",
        image_size: int = 1024,
        missing_tiles: MissingTileFlag | None = None,
        tiles: TileContext | None = None,
        prompt_cfg: PromptConfig | None = None,
        seed: int = 0,
        epoch: EpochCounter | None = None,
        val_prompt: str | None = None,
        circle_scale: float = 0.95,
        full_tile_threshold: float | None = None,
    ):
        if val_prompt is not None and val_prompt not in VAL_PROMPTS:
            raise ValueError(f"val_prompt must be one of {VAL_PROMPTS}, got {val_prompt!r}")
        self.val_prompt = val_prompt
        self.circle_scale = float(circle_scale)
        self.full_tile_threshold = None if full_tile_threshold is None else float(full_tile_threshold)
        self.examples = list(examples)
        self.split = split
        self.image_size = image_size
        self.missing_tiles = missing_tiles
        self.tiles = tiles
        self.prompt_cfg = prompt_cfg
        self.seed = int(seed)
        self.epoch = epoch
        self._mask_cache = MaskCache(prompt_cfg.cache_size if prompt_cfg is not None else 0)
        if split == "train":
            self.transform = build_train_transform()
        else:
            self.transform = build_val_transform()

    @property
    def mixed_prompts(self) -> bool:
        """Whether samples use the mixed prompt mix rather than the single centroid point."""
        return self.tiles is not None and self.prompt_cfg is not None

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
            index = (int(idx) + offset) % n
            example = self.examples[index]
            try:
                if self.val_prompt == "circle_mask":
                    return self._mixed_item(example, index, forced=PromptType.CIRCLE_MASK)
                if self.val_prompt is not None:
                    return self._item_from_example(example, self.val_prompt)
                if self.mixed_prompts:
                    return self._mixed_item(example, index)
                return self._item_from_example(example)
            except (KeyError, FileNotFoundError) as exc:
                if isinstance(exc, FileNotFoundError) and self.missing_tiles is not None:
                    self.missing_tiles.note()
                last_err = exc
                continue
        assert last_err is not None
        raise last_err

    def _item_from_example(self, example: CropExample, point_mode: str = "centroid") -> dict[str, Any]:
        """One sample with a single foreground point: the mask centroid (legacy) or its deepest interior pixel."""
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

        centroid = interior_point_xy(mask_bin) if point_mode == "point" else mask_centroid_xy(mask_bin)
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
            full_tile=is_full_tile(example, self.full_tile_threshold),
        )
        return {
            "image": image,
            "mask": torch.from_numpy(mask_bin.astype(np.int64)),
            "point": point,
            "point_label": torch.tensor([1], dtype=torch.int32),
            "boxes": None,
            "dense_mask": None,
            "prompt_fg": None,
            "prompt_type": point_mode,
            "orig_hw": (int(mask_bin.shape[0]), int(mask_bin.shape[1])),
            "sample_meta": meta,
            **meta.as_dict(),
        }

    def _mixed_item(
        self, example: CropExample, index: int, *, forced: PromptType | None = None
    ) -> dict[str, Any]:
        """One sample with a drawn prompt type, negative context, and the matching dense mask and points.

        Raises KeyError for a sample that cannot form a prompt (empty target after augmentation, or a circle
        target with no other annotation), so ``__getitem__`` moves on to the next example.
        """
        tiles = self.tiles if self.tiles is not None else TileContext()
        prompt_cfg = self.prompt_cfg if self.prompt_cfg is not None else PromptConfig()
        epoch = 0 if forced is not None else (self.epoch.get() if self.epoch is not None else 0)
        rng = np.random.default_rng([self.seed, epoch, int(index)])
        key = (example.volume, example.image_key)

        gray = _load_gray_uint8(example.image_path)
        rgb = to_rgb_uint8(gray)
        shape = (int(rgb.shape[0]), int(rgb.shape[1]))
        ann: dict[str, Any] = {}
        circle_mask: np.ndarray | None = None
        if example.target_circle is not None:
            target = rasterize_circle(shape, *example.target_circle)
            own = tiles.circle_for(key, example.location_id)
            ann = {"category_name": own.structure_label if own is not None else ""}
        else:
            mask, ann = load_binary_mask(example)
            if mask.shape[:2] != shape:
                mask = np.array(
                    Image.fromarray(mask * 255, mode="L").resize((shape[1], shape[0]), resample=Image.NEAREST)
                )
                mask = (mask > 127).astype(np.uint8)
            target = mask > 0
            own = tiles.circle_for(key, example.location_id)
            if own is not None and own.center_in(shape):
                disc = rasterize_circle(shape, own.center_x, own.center_y, own.radius)
                circle_mask = disc if disc.any() else None
            if circle_mask is None and forced is PromptType.CIRCLE_MASK:
                fitted = inscribed_circle_xyr(target, self.circle_scale)
                if fitted is None:
                    raise KeyError(f"{example.split_key} has no mask to fit a circle to")
                circle_mask = rasterize_circle(shape, *fitted)

        regions = collect_negative_regions(
            example.image_key,
            example.location_id,
            tiles.neighbor_ids(key),
            tiles.circles_in(key),
            example.mask_path.parent,
            shape,
            self._mask_cache,
        )
        if prompt_cfg.drop_overlapping_circles:
            regions = drop_circles_over_target(regions, target)
        if forced is not None:
            if example.target_circle is not None:
                raise KeyError(f"circle-only target {example.split_key} cannot be validated with {forced.value}")
            prompt_type = forced
        elif example.target_circle is not None:
            if not regions:
                raise KeyError(f"circle target {example.split_key} has no other annotation in its tile")
            prompt_type = PromptType.NEGATIVE_ONLY
        else:
            prompt_type = choose_prompt_type(
                prompt_cfg,
                rng,
                can_circle_mask=circle_mask is not None,
                has_negatives=bool(regions),
                allowed=FULL_TILE_PROMPTS if is_full_tile(example, self.full_tile_threshold) else None,
            )
        use_circle = prompt_type is PromptType.CIRCLE_MASK

        # The target, every negative and the circle go through the same geometric transform as the image, so
        # prompts stay registered to it (elastic warps and flips included).
        layers = [target.astype(np.uint8)] + [r.mask.astype(np.uint8) for r in regions]
        if use_circle and circle_mask is not None:
            layers.append(circle_mask.astype(np.uint8))
        augmented = self.transform(image=rgb, masks=layers)
        image_aug = augmented["image"]
        masks_aug = [np.asarray(m) > 0 for m in augmented["masks"]]
        target_aug = masks_aug[0]
        if not target_aug.any():
            raise KeyError(f"{example.split_key} target is empty after augmentation")
        regions_aug = [
            NegativeRegion(region.location_id, region.kind, mask)
            for region, mask in zip(regions, masks_aug[1 : 1 + len(regions)])
            if mask.any()
        ]
        circle_aug = masks_aug[-1] if use_circle and circle_mask is not None else None
        if use_circle and (circle_aug is None or not circle_aug.any()):
            prompt_type = (
                PromptType.POINTS if is_full_tile(example, self.full_tile_threshold) else PromptType.FULL_MASK
            )
            circle_aug = None
        if prompt_type is PromptType.NEGATIVE_ONLY and not regions_aug:
            raise KeyError(f"{example.split_key} lost every negative region after augmentation")

        prompt = build_prompt(
            prompt_type,
            target_mask=target_aug,
            circle_mask=circle_aug,
            regions=regions_aug,
            cfg=prompt_cfg,
            rng=rng,
        )
        meta = SampleMeta(
            volume=example.volume,
            image_key=example.image_key,
            location_id=example.location_id,
            downsample=example.downsample,
            category_name=str(ann.get("category_name") or ann.get("structure_label") or ""),
            split_key=example.split_key,
            full_tile=is_full_tile(example, self.full_tile_threshold),
        )
        return {
            "image": imagenet_normalize(np.ascontiguousarray(image_aug)),
            "mask": torch.from_numpy(target_aug.astype(np.int64)),
            "point": torch.from_numpy(prompt.point_coords).reshape(-1, 2),
            "point_label": torch.from_numpy(prompt.point_labels),
            "boxes": None if prompt.box is None else torch.from_numpy(prompt.box),
            "dense_mask": None if prompt.dense_logits is None else torch.from_numpy(prompt.dense_logits),
            "prompt_fg": (
                None
                if prompt.foreground_region is None
                else torch.from_numpy(prompt.foreground_region.astype(np.uint8))
            ),
            "prompt_type": prompt.prompt_type.value,
            "orig_hw": (int(target_aug.shape[0]), int(target_aug.shape[1])),
            "sample_meta": meta,
            **meta.as_dict(),
        }


def collate_em(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack a batch whose samples may carry different numbers of points.

    ``point``/``point_label`` are padded to the largest count with label -1, which SAM2 treats as "not a
    point". ``boxes`` and ``dense_mask`` stay lists with None for samples without one, because the prompt
    encoder takes one tensor per call; the training loop groups samples by which prompts they have.
    ``prompt_fg`` is the +1 region of each dense mask at full resolution (zeros when there is none), used by
    the mask-containment loss. ``meta`` keeps its original keys; ``prompt_type`` is a parallel list.
    """
    count = len(batch)
    most = max(int(item["point"].shape[0]) for item in batch)
    points = torch.zeros((count, most, 2), dtype=torch.float32)
    labels = torch.full((count, most), -1, dtype=torch.int32)
    for row, item in enumerate(batch):
        n = int(item["point"].shape[0])
        if n:
            points[row, :n] = item["point"]
            labels[row, :n] = item["point_label"]
    masks = torch.stack([item["mask"] for item in batch], dim=0)
    prompt_fg = torch.stack(
        [
            item["prompt_fg"] if item.get("prompt_fg") is not None else torch.zeros_like(item["mask"], dtype=torch.uint8)
            for item in batch
        ],
        dim=0,
    )
    return {
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "mask": masks,
        "point": points,
        "point_label": labels,
        "boxes": [item.get("boxes") for item in batch],
        "dense_mask": [item.get("dense_mask") for item in batch],
        "prompt_fg": prompt_fg,
        "prompt_type": [item.get("prompt_type", "centroid") for item in batch],
        "orig_hw": [item["orig_hw"] for item in batch],
        "meta": [item["sample_meta"].as_dict() for item in batch],
    }
