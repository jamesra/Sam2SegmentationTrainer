"""PyTorch Dataset over AnnotationCrops examples."""

from __future__ import annotations

import json
import multiprocessing
import zlib
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
from annotation_crops.prompt_encoding import EDGE_TAU_DEPTH_FRAC

from sam2_segmentation_trainer.data.approved import as_approved, is_approved
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
    negative_region_canvases,
    inscribed_circle_xyr,
    interior_point_xy,
)
from sam2_segmentation_trainer.data.fulltile import is_full_tile
from sam2_segmentation_trainer.goal_metrics import MIN_COMPONENT_PX, ground_truth_terms
from sam2_segmentation_trainer.data.tiles import TileContext
from sam2_segmentation_trainer.data.weights import (
    apply_leak_weight,
    edge_weight_map,
    leak_mask,
    neighbor_core_labels,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

VAL_PROMPTS = ("centroid", "point", "circle_mask", "correction", "near_correct", "circle_mask_noneg")
"""Validation prompts: the legacy mask centroid, the deepest interior point, the inscribed circle seed
(production), a mask to correct, a near-correct mask (the "do no harm" check) and the circle seed with every
negative region and background point removed (how much the model leans on neighbors being sent)."""

VAL_FAMILY_CODES = {name: code for code, name in enumerate(VAL_PROMPTS)}
"""A fixed number per validation family, so a family's prompt seed does not depend on which families run."""

VAL_MIXED_TYPES = {
    "circle_mask": PromptType.CIRCLE_SEED,
    "correction": PromptType.CORRECTION,
    "near_correct": PromptType.NEAR_CORRECT,
    "circle_mask_noneg": PromptType.CIRCLE_SEED,
}
"""Validation prompts that go through the mixed pipeline, and the prompt family each forces."""

NO_NEGATIVE_PROMPTS = frozenset({"circle_mask_noneg"})
"""Validation prompts whose prompt carries no negative context (the loss weights and metrics still see the neighbors)."""

CIRCLE_PROMPTS = frozenset({"circle_mask", "circle_mask_noneg"})
"""Validation prompts that need the target's circle."""


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


def _variant_from_prompt(prompt) -> dict[str, Any]:
    """The collate-ready dict of one built prompt (see :func:`build_prompt`)."""
    return {
        "point": torch.from_numpy(prompt.point_coords).reshape(-1, 2),
        "point_label": torch.from_numpy(prompt.point_labels),
        "boxes": None if prompt.box is None else torch.from_numpy(prompt.box),
        "dense_mask": None if prompt.dense_logits is None else torch.from_numpy(prompt.dense_logits),
        "prompt_fg": (
            None
            if prompt.foreground_weight is None
            else torch.from_numpy(prompt.foreground_weight.astype(np.float16))
        ),
        "prompt_input": None if prompt.input_mask is None else torch.from_numpy(prompt.input_mask.astype(np.uint8)),
        "prompt_type": prompt.prompt_type.value,
    }


class EMSegDataset(Dataset):
    """Crops with a prompt per sample.

    Without ``tiles`` and ``prompt_cfg`` every sample is the original single-centroid-point prompt, which is
    what validation and evaluation use so their IoU stays comparable across runs. With both, each sample draws
    ``prompt_cfg.variants_per_sample`` prompt variants against one augmented image (points, box, box and
    points, circle seed, subset seed, mask to correct, near-correct mask) and carries the other annotations in
    its tile as negative context; see :mod:`sam2_segmentation_trainer.data.prompts`. The image is encoded once
    and every variant reuses its embeddings.

    Variant ``k`` is seeded from ``(seed, epoch, index, k)`` so a resumed epoch sees the same prompts. The mask
    cache is per process, so each DataLoader worker keeps its own.

    ``edge_tau_px > 0`` adds the per-pixel edge weight ``max(edge_weight_min, 1 - exp(-d / tau))`` of the
    ground-truth mask to each item (``edge_weight``), for the edge-tolerant loss and tolerant validation; ``tau``
    is capped per mask by ``edge_tau_depth_frac`` times its depth. ``leak_weight > 1`` multiplies that weight on
    pixels at least ``leak_core_px`` inside another annotation and outside the target (a merge costs more than an
    equal area of edge error). ``core_band_px > 0`` adds ``neighbor_core``, an int16 map of the neighbors' cores,
    for the validation metrics. ``approved`` is a set of ``(location_id, image_key)`` pairs: those items are
    flagged in their meta and get ``approved_weight`` as their training weight.

    ``val_families`` (validation) makes each item carry one prompt variant per family, all against one loaded
    image, so the validator encodes the image once and decodes every family from it. Each family's prompt is
    seeded from ``(seed, sample, family)`` where the sample is identified by its volume, image key and location
    id, not by its position, so results do not depend on batch composition, family order or which samples are in
    the pass. Items also carry ``gt_terms``, the ground-truth side of the goal metrics, computed once here in
    the worker.
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
        edge_tau_px: float = 0.0,
        edge_weight_min: float = 0.05,
        edge_tau_depth_frac: float = EDGE_TAU_DEPTH_FRAC,
        leak_weight: float = 1.0,
        leak_core_px: float = 10.0,
        core_band_px: float = 0.0,
        approved: frozenset | set | None = None,
        approved_weight: float = 1.0,
        val_families: Sequence[str] | None = None,
    ):
        if val_prompt is not None and val_prompt not in VAL_PROMPTS:
            raise ValueError(f"val_prompt must be one of {VAL_PROMPTS}, got {val_prompt!r}")
        families = tuple(dict.fromkeys(val_families or ()))
        unknown = [f for f in families if f not in VAL_PROMPTS]
        if unknown:
            raise ValueError(f"val_families must be among {VAL_PROMPTS}, got {unknown}")
        if families and val_prompt is not None:
            raise ValueError("pass either val_prompt (one family per item) or val_families (all of them)")
        self.val_prompt = val_prompt
        self.val_families = families
        self.edge_tau_px = float(edge_tau_px)
        self.edge_weight_min = float(edge_weight_min)
        self.edge_tau_depth_frac = float(edge_tau_depth_frac)
        self.leak_weight = float(leak_weight)
        self.leak_core_px = float(leak_core_px)
        self.core_band_px = float(core_band_px)
        self.approved = as_approved(approved)
        self.approved_weight = float(approved_weight)
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
                if self.val_families:
                    return self._families_item(example)
                if self.val_prompt in VAL_MIXED_TYPES:
                    return self._mixed_item(
                        example,
                        index,
                        forced=VAL_MIXED_TYPES[self.val_prompt],
                        no_negatives=self.val_prompt in NO_NEGATIVE_PROMPTS,
                    )
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

        shape = (int(rgb.shape[0]), int(rgb.shape[1]))
        regions = self._other_annotations(example, shape) if self._needs_neighbors() else []
        augmented = self.transform(image=rgb, masks=[mask] + [r.mask.astype(np.uint8) for r in regions])
        image_aug = augmented["image"]
        mask_bin = (np.asarray(augmented["masks"][0]) > 0).astype(np.uint8)
        regions_aug = [
            NegativeRegion(region.location_id, region.kind, np.asarray(m) > 0)
            for region, m in zip(regions, augmented["masks"][1:])
            if np.asarray(m).any()
        ]

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
            approved=self._is_approved(example),
        )
        variant = {
            "point": point,
            "point_label": torch.tensor([1], dtype=torch.int32),
            "boxes": None,
            "dense_mask": None,
            "prompt_fg": None,
            "prompt_input": None,
            "prompt_type": point_mode,
        }
        return {
            "image": image,
            "mask": torch.from_numpy(mask_bin.astype(np.int64)),
            **self._weights(mask_bin > 0, regions_aug, meta),
            "variants": [variant],
            "orig_hw": (int(mask_bin.shape[0]), int(mask_bin.shape[1])),
            "sample_meta": meta,
            **meta.as_dict(),
        }

    def _families_item(self, example: CropExample) -> dict[str, Any]:
        """One validation sample with a prompt variant per family in ``val_families``, against one loaded image."""
        tiles = self.tiles if self.tiles is not None else TileContext()
        prompt_cfg = self.prompt_cfg if self.prompt_cfg is not None else PromptConfig()
        key = (example.volume, example.image_key)
        gray = _load_gray_uint8(example.image_path)
        rgb = to_rgb_uint8(gray)
        shape = (int(rgb.shape[0]), int(rgb.shape[1]))
        mask, ann = load_binary_mask(example)
        if mask.shape[:2] != shape:
            mask = np.array(
                Image.fromarray(mask * 255, mode="L").resize((shape[1], shape[0]), resample=Image.NEAREST)
            )
            mask = (mask > 127).astype(np.uint8)
        target = mask > 0
        circle_mask: np.ndarray | None = None
        needs_circle = any(f in CIRCLE_PROMPTS for f in self.val_families)
        if needs_circle:
            own = tiles.circle_for(key, example.location_id)
            if own is not None and own.center_in(shape):
                disc = rasterize_circle(shape, own.center_x, own.center_y, own.radius)
                circle_mask = disc if disc.any() else None
        regions = self._other_annotations(example, shape)
        if prompt_cfg.drop_overlapping_circles:
            regions = drop_circles_over_target(regions, target)

        layers = [target.astype(np.uint8)] + [r.mask.astype(np.uint8) for r in regions]
        if circle_mask is not None:
            layers.append(circle_mask.astype(np.uint8))
        augmented = self.transform(image=rgb, masks=layers)
        image_aug = augmented["image"]
        masks_aug = [np.asarray(m) > 0 for m in augmented["masks"]]
        target_aug = masks_aug[0]
        if not target_aug.any():
            raise KeyError(f"{example.split_key} target is empty after augmentation")
        regions_aug = [
            NegativeRegion(region.location_id, region.kind, m)
            for region, m in zip(regions, masks_aug[1 : 1 + len(regions)])
            if m.any()
        ]
        circle_aug = masks_aug[-1] if circle_mask is not None else None
        if needs_circle and (circle_aug is None or not circle_aug.any()):
            fitted = inscribed_circle_xyr(target_aug, self.circle_scale)
            if fitted is None:
                raise KeyError(f"{example.split_key} has no mask to fit a circle to")
            circle_aug = rasterize_circle(shape, *fitted)

        sample_id = zlib.crc32(f"{example.volume}|{example.image_key}|{example.location_id}".encode("utf-8"))
        with_negatives = [f for f in self.val_families if f in VAL_MIXED_TYPES and f not in NO_NEGATIVE_PROMPTS]
        canvases = negative_region_canvases(regions_aug, prompt_cfg) if with_negatives and regions_aug else None
        variants: list[dict[str, Any]] = []
        for family in self.val_families:
            if family in VAL_MIXED_TYPES:
                prompt = build_prompt(
                    VAL_MIXED_TYPES[family],
                    target_mask=target_aug,
                    circle_mask=circle_aug,
                    regions=[] if family in NO_NEGATIVE_PROMPTS else regions_aug,
                    cfg=prompt_cfg,
                    rng=np.random.default_rng([self.seed, sample_id, VAL_FAMILY_CODES[family]]),
                    negative_canvases=None if family in NO_NEGATIVE_PROMPTS else canvases,
                )
                variant = _variant_from_prompt(prompt)
            else:
                point = interior_point_xy(target_aug) if family == "point" else mask_centroid_xy(target_aug)
                if point is None:
                    point = bbox_center_xy(ann, shape[1], shape[0])
                variant = {
                    "point": torch.tensor([[point[0], point[1]]], dtype=torch.float32),
                    "point_label": torch.tensor([1], dtype=torch.int32),
                    "boxes": None,
                    "dense_mask": None,
                    "prompt_fg": None,
                    "prompt_input": None,
                    "prompt_type": family,
                }
            variant["val_family"] = family
            variants.append(variant)

        meta = SampleMeta(
            volume=example.volume,
            image_key=example.image_key,
            location_id=example.location_id,
            downsample=example.downsample,
            category_name=str(ann.get("category_name") or ann.get("structure_label") or ""),
            split_key=example.split_key,
            full_tile=is_full_tile(example, self.full_tile_threshold),
            approved=self._is_approved(example),
            has_negatives=bool(regions_aug),
        )
        terms = (
            ground_truth_terms(target_aug, self.core_band_px, MIN_COMPONENT_PX) if self.core_band_px > 0.0 else None
        )
        return {
            "image": imagenet_normalize(np.ascontiguousarray(image_aug)),
            "mask": torch.from_numpy(target_aug.astype(np.int64)),
            **self._weights(target_aug, regions_aug, meta),
            "variants": variants,
            "gt_terms": terms,
            "orig_hw": (int(target_aug.shape[0]), int(target_aug.shape[1])),
            "sample_meta": meta,
            **meta.as_dict(),
        }

    def _is_approved(self, example: CropExample) -> bool:
        return is_approved(self.approved, example)

    def _needs_neighbors(self) -> bool:
        """Whether this dataset reads the tile's other annotations even without a mixed prompt (leak, cores)."""
        return self.tiles is not None and (self.core_band_px > 0.0 or self.leak_weight != 1.0)

    def _other_annotations(self, example: CropExample, shape: tuple[int, int]) -> list[NegativeRegion]:
        tiles = self.tiles if self.tiles is not None else TileContext()
        key = (example.volume, example.image_key)
        regions = collect_negative_regions(
            example.image_key,
            example.location_id,
            tiles.neighbor_ids(key),
            tiles.circles_in(key),
            example.mask_path.parent,
            shape,
            self._mask_cache,
        )
        return regions

    def _weights(
        self, target: np.ndarray, regions: Sequence[NegativeRegion], meta: SampleMeta
    ) -> dict[str, Any]:
        """The loss-side items of a sample: edge weight, training weight and (validation) neighbor cores.

        ``edge_weight`` is the tolerant weight of the ground truth (float16) times ``leak_weight`` on neighbor
        cores; None when both are off. ``neighbor_core`` is None unless ``core_band_px > 0``.
        """
        weight = edge_weight_map(target, self.edge_tau_px, self.edge_weight_min, self.edge_tau_depth_frac)
        masks = [r.mask for r in regions]
        if self.leak_weight != 1.0 and masks:
            weight = apply_leak_weight(
                weight, leak_mask(masks, target, self.leak_core_px), self.leak_weight, target.shape
            )
        cores = None
        if self.core_band_px > 0.0:
            cores = torch.from_numpy(neighbor_core_labels(masks, target, self.core_band_px))
        return {
            "edge_weight": None if weight is None else torch.from_numpy(weight.astype(np.float16)),
            "sample_weight": self.approved_weight if meta.approved else 1.0,
            "neighbor_core": cores,
        }

    def _plan_types(
        self,
        cfg: PromptConfig,
        index: int,
        epoch: int,
        *,
        full_tile: bool,
        forced: PromptType | None,
        has_adjacent: bool = False,
    ) -> list[PromptType]:
        """The prompt family of each variant. Variant 0 is the production prompt: a circle seed, or, when the
        sample has an adjacent-section shape, an adjacent seed with probability ``a / (a + c)``.

        The others are drawn from the mix without replacement (so the ``prob_*`` shares steer variants 1 onwards
        only); a fully masked tile has only two families to draw from, so it gets at most two variants.
        """
        if forced is not None:
            return [forced]
        allowed = FULL_TILE_PROMPTS if full_tile else None
        first = PromptType.CIRCLE_SEED
        adjacent, circle = cfg.prob_adjacent_seed, cfg.prob_circle_seed
        if has_adjacent and not full_tile and adjacent + circle > 0.0:
            draw = np.random.default_rng([self.seed, epoch, int(index), 0, 1]).random()
            if draw < adjacent / (adjacent + circle):
                first = PromptType.ADJACENT_SEED
        chosen = [first]
        for k in range(1, cfg.variants_per_sample):
            rng = np.random.default_rng([self.seed, epoch, int(index), k, 1])
            kind = choose_prompt_type(
                cfg, rng, allowed=allowed, exclude=set(chosen) | {PromptType.ADJACENT_SEED}
            )
            if kind in chosen:
                break
            chosen.append(kind)
        return chosen

    def _mixed_item(
        self,
        example: CropExample,
        index: int,
        *,
        forced: PromptType | None = None,
        no_negatives: bool = False,
    ) -> dict[str, Any]:
        """One sample with prompt variants, negative context, and the matching dense masks and points.

        Raises KeyError for a sample that cannot form a prompt (empty target after augmentation), so
        ``__getitem__`` moves on to the next example.
        """
        tiles = self.tiles if self.tiles is not None else TileContext()
        prompt_cfg = self.prompt_cfg if self.prompt_cfg is not None else PromptConfig()
        epoch = 0 if forced is not None else (self.epoch.get() if self.epoch is not None else 0)
        key = (example.volume, example.image_key)
        full_tile = is_full_tile(example, self.full_tile_threshold)

        gray = _load_gray_uint8(example.image_path)
        rgb = to_rgb_uint8(gray)
        shape = (int(rgb.shape[0]), int(rgb.shape[1]))
        mask, ann = load_binary_mask(example)
        if mask.shape[:2] != shape:
            mask = np.array(
                Image.fromarray(mask * 255, mode="L").resize((shape[1], shape[0]), resample=Image.NEAREST)
            )
            mask = (mask > 127).astype(np.uint8)
        target = mask > 0
        own = tiles.circle_for(key, example.location_id)
        circle_mask: np.ndarray | None = None
        if own is not None and own.center_in(shape):
            disc = rasterize_circle(shape, own.center_x, own.center_y, own.radius)
            circle_mask = disc if disc.any() else None

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
        adjacent_shape = None if full_tile else tiles.adjacent_shape(key, example.location_id)
        types = self._plan_types(
            prompt_cfg, index, epoch, full_tile=full_tile, forced=forced, has_adjacent=adjacent_shape is not None
        )

        # The target, every negative and the circle go through the same geometric transform as the image, so
        # prompts stay registered to it (elastic warps and flips included).
        layers = [target.astype(np.uint8)] + [r.mask.astype(np.uint8) for r in regions]
        # An adjacent seed that cannot be fitted falls back to the circle seed, so it needs the circle too.
        needs_circle = PromptType.CIRCLE_SEED in types or PromptType.ADJACENT_SEED in types
        use_circle = needs_circle and circle_mask is not None
        if use_circle:
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
        circle_aug = masks_aug[-1] if use_circle else None
        if needs_circle and (circle_aug is None or not circle_aug.any()):
            # No annotation circle in this tile (or it left the image): fit the largest circle inside the target.
            fitted = inscribed_circle_xyr(target_aug, self.circle_scale)
            if fitted is None:
                raise KeyError(f"{example.split_key} has no mask to fit a circle to")
            circle_aug = rasterize_circle(shape, *fitted)

        canvases = (
            negative_region_canvases(regions_aug, prompt_cfg)
            if len(types) > 1 and regions_aug and not no_negatives
            else None
        )
        # Negative-context dropout applies to training variants after the first (variant 0 is the production prompt).
        train_dropout = forced is None and self.split == "train"
        variants: list[dict[str, Any]] = []
        for k, kind in enumerate(types):
            prompt = build_prompt(
                kind,
                target_mask=target_aug,
                circle_mask=circle_aug,
                regions=[] if no_negatives else regions_aug,
                cfg=prompt_cfg,
                rng=np.random.default_rng([self.seed, epoch, int(index), k]),
                adjacent_shape=adjacent_shape,
                negative_canvases=canvases,
                negative_dropout=train_dropout and k >= 1,
            )
            variants.append(
                {
                    "point": torch.from_numpy(prompt.point_coords).reshape(-1, 2),
                    "point_label": torch.from_numpy(prompt.point_labels),
                    "boxes": None if prompt.box is None else torch.from_numpy(prompt.box),
                    "dense_mask": None if prompt.dense_logits is None else torch.from_numpy(prompt.dense_logits),
                    "prompt_fg": (
                        None
                        if prompt.foreground_weight is None
                        else torch.from_numpy(prompt.foreground_weight.astype(np.float16))
                    ),
                    "prompt_input": (
                        None if prompt.input_mask is None else torch.from_numpy(prompt.input_mask.astype(np.uint8))
                    ),
                    "prompt_type": prompt.prompt_type.value,
                }
            )
        meta = SampleMeta(
            volume=example.volume,
            image_key=example.image_key,
            location_id=example.location_id,
            downsample=example.downsample,
            category_name=str(ann.get("category_name") or ann.get("structure_label") or ""),
            split_key=example.split_key,
            full_tile=full_tile,
            approved=self._is_approved(example),
            has_negatives=bool(regions_aug),
        )
        return {
            "image": imagenet_normalize(np.ascontiguousarray(image_aug)),
            "mask": torch.from_numpy(target_aug.astype(np.int64)),
            **self._weights(target_aug, regions_aug, meta),
            "variants": variants,
            "orig_hw": (int(target_aug.shape[0]), int(target_aug.shape[1])),
            "sample_meta": meta,
            **meta.as_dict(),
        }


def collate_em(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack a batch of samples, each with one or more prompt variants.

    ``image``, ``mask`` and ``edge_weight`` (None unless every sample has one) are per sample, ``[B, ...]``.
    Everything else is per variant, ``[N, ...]``, with ``variant_index`` (long ``[N]``) naming the sample each
    variant belongs to. With one variant per sample ``N == B`` and ``variant_index`` is ``arange(B)``.

    ``point``/``point_label`` are padded to the largest count with label -1, which SAM2 treats as "not a
    point". ``boxes`` and ``dense_mask`` stay lists with None for variants without one, because the prompt
    encoder takes one tensor per call; the training loop groups variants by which prompts they have.
    ``prompt_fg`` is the trust of each seed at full resolution (float16, zeros when there is none), used by the
    mask-containment loss. ``prompt_input`` is the seed or mask to correct as uint8 (None when no variant has
    one). ``meta`` and ``orig_hw`` are repeated for each variant of a sample; ``prompt_type`` is a parallel list.
    """
    rows = [(b, variant) for b, item in enumerate(batch) for variant in item["variants"]]
    count = len(rows)
    most = max(int(variant["point"].shape[0]) for _b, variant in rows)
    points = torch.zeros((count, most, 2), dtype=torch.float32)
    labels = torch.full((count, most), -1, dtype=torch.int32)
    for row, (_b, variant) in enumerate(rows):
        n = int(variant["point"].shape[0])
        if n:
            points[row, :n] = variant["point"]
            labels[row, :n] = variant["point_label"]
    masks = torch.stack([item["mask"] for item in batch], dim=0)
    blank = torch.zeros_like(batch[0]["mask"], dtype=torch.uint8)
    prompt_fg = torch.stack(
        [
            variant["prompt_fg"] if variant.get("prompt_fg") is not None else blank.to(torch.float16)
            for _b, variant in rows
        ],
        dim=0,
    )
    prompt_input = None
    if any(variant.get("prompt_input") is not None for _b, variant in rows):
        prompt_input = torch.stack(
            [variant["prompt_input"] if variant.get("prompt_input") is not None else blank for _b, variant in rows],
            dim=0,
        )
    edge_weight = None
    if all(item.get("edge_weight") is not None for item in batch):
        edge_weight = torch.stack([item["edge_weight"] for item in batch], dim=0)
    neighbor_core = None
    if all(item.get("neighbor_core") is not None for item in batch):
        neighbor_core = torch.stack([item["neighbor_core"] for item in batch], dim=0)
    return {
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "mask": masks,
        "edge_weight": edge_weight,
        "sample_weight": torch.tensor([float(item.get("sample_weight", 1.0)) for item in batch], dtype=torch.float32),
        "neighbor_core": neighbor_core,
        "gt_terms": [item.get("gt_terms") for item in batch],
        "val_family": [variant.get("val_family") for _b, variant in rows],
        "variant_index": torch.tensor([b for b, _v in rows], dtype=torch.long),
        "point": points,
        "point_label": labels,
        "boxes": [variant.get("boxes") for _b, variant in rows],
        "dense_mask": [variant.get("dense_mask") for _b, variant in rows],
        "prompt_fg": prompt_fg,
        "prompt_input": prompt_input,
        "prompt_type": [variant.get("prompt_type", "centroid") for _b, variant in rows],
        "orig_hw": [batch[b]["orig_hw"] for b, _v in rows],
        "meta": [batch[b]["sample_meta"].as_dict() for b, _v in rows],
    }
