"""PyTorch Dataset over AnnotationCrops examples."""

from __future__ import annotations

import json
import multiprocessing
import os
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
from annotation_crops.prompt_encoding import EDGE_TAU_DEPTH_FRAC, CircleAnnotationTrust

from sam2_segmentation_trainer.data.approved import as_approved, is_approved
from sam2_segmentation_trainer.data.circle_seed_trust import (
    DEFAULT_OVERSIZE_RANGE,
    OVERSIZE_STREAM,
    SeedCircle,
    check_oversize_range,
    oversize_circle,
    seed_taper,
    seed_taper_weight,
    tapered_seed_mask_input,
)
from sam2_segmentation_trainer.data.augment import (
    build_train_transform,
    build_val_transform,
    to_rgb_uint8,
)
from sam2_segmentation_trainer.data.manifest import CropExample
from sam2_segmentation_trainer.data.meta import SampleMeta
from sam2_segmentation_trainer.data.prompts import (
    CIRCLE_PROMPT_TYPES,
    PromptConfig,
    PromptType,
    FULL_TILE_PROMPTS,
    annotator_circle,
    build_prompt,
    choose_prompt_type,
    negative_region_canvases,
    inscribed_circle_xyr,
    interior_point_xy,
    pull_through_circle,
    target_depth,
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

VAL_PROMPTS = (
    "centroid", "point", "circle_mask", "correction", "near_correct", "circle_mask_noneg", "circle_annotator",
    "circle_oversize",
)
"""Validation prompts: the legacy mask centroid, the deepest interior point, the inscribed circle seed
(production), a mask to correct, a near-correct mask (the "do no harm" check), the circle seed with every
negative region and background point removed (how much the model leans on neighbors being sent), an
annotator-like circle seed (a random centre depth and radius, see ``annotator_circle``) and an oversized circle
(evaluation only, never a default: the inscribed circle's radius times a random factor, see
:func:`~sam2_segmentation_trainer.data.circle_seed_trust.oversize_circle`).

New families go at the END: a family's number is its position, and its prompt seed uses that number."""

VAL_FAMILY_CODES = {name: code for code, name in enumerate(VAL_PROMPTS)}
"""A fixed number per validation family, so a family's prompt seed does not depend on which families run."""

VAL_MIXED_TYPES = {
    "circle_mask": PromptType.CIRCLE_SEED,
    "correction": PromptType.CORRECTION,
    "near_correct": PromptType.NEAR_CORRECT,
    "circle_mask_noneg": PromptType.CIRCLE_SEED,
    "circle_annotator": PromptType.CIRCLE_ANNOTATOR,
    "circle_oversize": PromptType.CIRCLE_SEED,
}
"""Validation prompts that go through the mixed pipeline, and the prompt family each forces."""

ADJACENT_PICK_STREAM = 4
"""Trailing number of the generator ``[seed, epoch, index, 0, 4]`` that picks which linked neighbour outline a
sample's adjacent polygon seed uses, and then its erosion (the variant streams end in 1, 2 and 3, or in the variant
number)."""

ADJACENT_REDRAW_STREAM = 5
"""Trailing number of the generator ``[seed, epoch, index, k, 5]`` that re-draws variant ``k`` when the mix drew
``adjacent_poly_seed`` but the projected outline failed ``prompt.adjacent_poly.overlap_min``."""

NO_NEGATIVE_PROMPTS = frozenset({"circle_mask_noneg"})
"""Validation prompts whose prompt carries no negative context (the loss weights and metrics still see the neighbors)."""

CIRCLE_PROMPTS = frozenset({"circle_mask", "circle_mask_noneg", "circle_annotator", "circle_oversize"})
"""Validation prompts that need the target's circle."""

FAMILIES_ONLY_PROMPTS = frozenset({"circle_annotator", "circle_oversize"})
"""Validation prompts built only by the families path (``val_families``): the one-prompt path has no such prompt."""


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


WORKER_THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
"""Thread-count variables set to 1 in every DataLoader worker (for any process a worker starts)."""

_OPENBLAS_SET_THREADS = (
    "scipy_openblas_set_num_threads64_",  # numpy's scipy-openblas64
    "scipy_openblas_set_num_threads",  # scipy's scipy-openblas
    "openblas_set_num_threads64_",
    "openblas_set_num_threads",  # OpenCV's bundled OpenBLAS
)


def _loaded_openblas_paths() -> list[str]:
    """Paths of the OpenBLAS libraries loaded in this process (Linux ``/proc/self/maps``; empty elsewhere)."""
    try:
        with open("/proc/self/maps", encoding="utf-8") as maps:
            paths = {line.split()[-1] for line in maps if "openblas" in line.lower() and "/" in line}
    except OSError:
        return []
    return sorted(path for path in paths if path.startswith("/"))


def limit_openblas_threads(threads: int = 1) -> int:
    """Set every loaded OpenBLAS (numpy's, scipy's, OpenCV's each bundle one) to ``threads`` threads.

    The environment variables are read only when a library loads, and workers fork from a parent that has loaded
    them all, so this calls each library's own setter. Returns how many libraries were set.
    """
    import ctypes

    count = 0
    for path in _loaded_openblas_paths():
        try:
            lib = ctypes.CDLL(path)
        except OSError:
            continue
        for name in _OPENBLAS_SET_THREADS:
            setter = getattr(lib, name, None)
            if setter is not None:
                setter(int(threads))
                count += 1
                break
    return count


def limit_worker_threads(_worker_id: int | None = None) -> None:
    """``worker_init_fn`` of every DataLoader over :class:`EMSegDataset`: one thread per worker.

    Each worker runs its own sample, so library thread pools only oversubscribe the CPU: with 4 workers on 32 cores,
    OpenBLAS's pool alone cut the training loader from about 8.9 to 4.0 samples/s on set v3. Sets OpenBLAS (every
    copy loaded), OpenCV and torch to one thread and the OMP/MKL/OpenBLAS variables to 1, without depending on how
    the trainer was launched. The main process keeps its own threading.
    """
    for name in WORKER_THREAD_ENV:
        os.environ[name] = "1"
    torch.set_num_threads(1)
    try:
        import cv2

        # Calling setNumThreads in a forked child deadlocks (OpenCV's pthread pool is not fork-safe) when the parent
        # has already run a parallel cv2 operation. The parent calls limit_main_process_cv2() before the first
        # fork, so the children inherit 1 and skip the call.
        if cv2.getNumThreads() != 1:
            cv2.setNumThreads(1)
    except ImportError:  # pragma: no cover - albumentations needs OpenCV
        pass
    limit_openblas_threads(1)


def limit_main_process_cv2() -> None:
    """Set OpenCV to one thread in the MAIN process, before any DataLoader forks workers.

    Forked workers inherit the setting, so :func:`limit_worker_threads` never has to call into OpenCV's thread pool
    from a child (that call hangs forever if the parent has ever run a parallel cv2 operation). The trainer itself
    does not use OpenCV; only the augmentations in the workers do.
    """
    try:
        import cv2

        cv2.setNumThreads(1)
    except ImportError:  # pragma: no cover
        pass


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


_MEAN32 = np.asarray(IMAGENET_MEAN, dtype=np.float32)
_STD32 = np.asarray(IMAGENET_STD, dtype=np.float32)


def imagenet_normalize(rgb_uint8: np.ndarray) -> torch.Tensor:
    """``[3, H, W]`` float32 ``(rgb / 255 - mean) / std`` of an ``[H, W, 3]`` uint8 image.

    One float32 pass per channel, written straight into the channel-first array: the same float32 operations in the
    same order as ``(torch.from_numpy(rgb.astype(float32) / 255).permute(2, 0, 1) - mean) / std``, so bit-identical,
    without the float copy, the permute copy and two temporaries (about 5x faster on a 1024 tile).
    """
    height, width = int(rgb_uint8.shape[0]), int(rgb_uint8.shape[1])
    out = np.empty((3, height, width), dtype=np.float32)
    for c in range(3):
        channel = out[c]
        np.divide(rgb_uint8[:, :, c], np.float32(255.0), out=channel, dtype=np.float32)
        channel -= _MEAN32[c]
        channel /= _STD32[c]
    return torch.from_numpy(out)


def augment_scene(
    transform,
    rgb: np.ndarray,
    target: np.ndarray,
    regions: Sequence[NegativeRegion],
    extras: Sequence[np.ndarray] = (),
):
    """Run the image, target, negative regions and extra boolean masks through one geometric transform.

    A region that is an annotated circle also carries a float32 trust map; it rides along as its own layer, so the
    claimed mask and the map stay registered to the image (flips, rotations and elastic warps). Returns
    ``(image, target, regions, extras)`` with boolean masks, and drops regions that left the image.
    """
    weighted = [r for r in regions if r.weight is not None]
    layers = (
        [target.astype(np.uint8)]
        + [r.mask.astype(np.uint8) for r in regions]
        + [r.weight.astype(np.float32, copy=False) for r in weighted]
        + [e.astype(np.uint8) for e in extras]
    )
    augmented = transform(image=rgb, masks=layers)
    out = augmented["masks"]
    count = len(regions)
    region_masks = [np.asarray(m) > 0 for m in out[1 : 1 + count]]
    weights = iter(np.asarray(w, dtype=np.float32) for w in out[1 + count : 1 + count + len(weighted)])
    regions_aug: list[NegativeRegion] = []
    for region, mask in zip(regions, region_masks):
        weight = None if region.weight is None else next(weights)  # advance for every weighted region, kept or not
        if mask.any():
            regions_aug.append(NegativeRegion(region.location_id, region.kind, mask, weight))
    extras_aug = [np.asarray(m) > 0 for m in out[1 + count + len(weighted) :]]
    return augmented["image"], np.asarray(out[0]) > 0, regions_aug, extras_aug


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
    points, the three circle seeds, adjacent polygon seed, subset seed, mask to correct, near-correct mask), all drawn
    from the ``prob_*`` mix without replacement, and carries the other annotations in
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

    Two evaluation-only options act on the families path alone (docs/circle_seed_trust_eval.md):
    ``circle_seed_taper`` (a ``full_frac``, None for off) feeds every circle-seed family's disc as
    ``+seed_logit * w(rho)``, the radius-relative cosine taper with ``zero_frac`` the ``circle_scale``, instead of the
    edge confidence; ``circle_oversize_range`` is the factor range of the ``circle_oversize`` family.
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
        circle_seed_taper: float | None = None,
        circle_oversize_range: Sequence[float] = DEFAULT_OVERSIZE_RANGE,
        elastic_warp: bool = False,
    ):
        if val_prompt is not None and val_prompt not in VAL_PROMPTS:
            raise ValueError(f"val_prompt must be one of {VAL_PROMPTS}, got {val_prompt!r}")
        if val_prompt in FAMILIES_ONLY_PROMPTS:
            raise ValueError(
                f"val_prompt {val_prompt!r} is only available as a family: pass val_families=[{val_prompt!r}] "
                "(the one-prompt path would score it as a plain circle_mask)"
            )
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
        # Evaluation only: the families path draws circles with the validation transform, which is the identity, so a
        # circle's centre and radius still describe its disc after augment_scene.
        if circle_seed_taper is not None and (split != "val" or val_prompt is not None):
            raise ValueError("circle_seed_taper is evaluation only: it needs split='val' and val_families")
        self.circle_seed_taper = (
            None if circle_seed_taper is None else seed_taper(float(circle_seed_taper), self.circle_scale)
        )
        self.circle_oversize_range = check_oversize_range(circle_oversize_range)
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
            self.transform = build_train_transform(elastic_warp=elastic_warp)
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
        image_aug, target_aug, regions_aug, _extras = augment_scene(self.transform, rgb, mask > 0, regions)
        mask_bin = target_aug.astype(np.uint8)

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
        seed_circle: SeedCircle | None = None  # its centre and true radius, for the evaluation-only options
        needs_circle = any(f in CIRCLE_PROMPTS for f in self.val_families)
        if needs_circle:
            own = tiles.circle_for(key, example.location_id)
            if own is not None and own.center_in(shape):
                radius = own.seed_radius(self.circle_scale)
                disc = rasterize_circle(shape, own.center_x, own.center_y, radius)
                circle_mask = disc if disc.any() else None
                seed_circle = SeedCircle(own.center_x, own.center_y, radius / self.circle_scale)
        regions = self._other_annotations(example, shape)
        if prompt_cfg.drop_overlapping_circles:
            regions = drop_circles_over_target(regions, target)

        image_aug, target_aug, regions_aug, extras = augment_scene(
            self.transform, rgb, target, regions, [] if circle_mask is None else [circle_mask]
        )
        if not target_aug.any():
            raise KeyError(f"{example.split_key} target is empty after augmentation")
        circle_aug = extras[0] if circle_mask is not None else None
        if needs_circle and (circle_aug is None or not circle_aug.any()):
            fitted = inscribed_circle_xyr(target_aug, self.circle_scale)
            if fitted is None:
                raise KeyError(f"{example.split_key} has no mask to fit a circle to")
            circle_aug = rasterize_circle(shape, *fitted)
            seed_circle = SeedCircle(fitted[0], fitted[1], fitted[2] / self.circle_scale)

        sample_id = zlib.crc32(f"{example.volume}|{example.image_key}|{example.location_id}".encode("utf-8"))
        with_negatives = [f for f in self.val_families if f in VAL_MIXED_TYPES and f not in NO_NEGATIVE_PROMPTS]
        canvases = negative_region_canvases(regions_aug, prompt_cfg) if with_negatives and regions_aug else None
        variants: list[dict[str, Any]] = []
        for family in self.val_families:
            if family in VAL_MIXED_TYPES:
                family_circle, family_type = circle_aug, VAL_MIXED_TYPES[family]
                family_seed_circle = seed_circle
                if family_type is PromptType.CIRCLE_ANNOTATOR:
                    # An annotator-like circle, from its own generator, so no other family's draws move; None (too
                    # small, empty) keeps the inscribed seed and the variant is built (and reported) as circle_seed.
                    drawn: dict = {}
                    disc = annotator_circle(
                        target_aug,
                        np.random.default_rng([self.seed, sample_id, VAL_FAMILY_CODES[family], 2]),
                        prompt_cfg.circle_seed_jitter,
                        info=drawn,
                    )
                    if disc is not None:
                        family_circle = disc
                        # Its disc is drawn at the drawn radius, so the taper's true radius is that over the scale and
                        # the taper claims the same disc as the default encoding.
                        family_seed_circle = SeedCircle(
                            drawn["col"] + 0.5, drawn["row"] + 0.5, drawn["radius"] / self.circle_scale
                        )
                    else:
                        family_type = PromptType.CIRCLE_SEED
                elif family == "circle_oversize" and seed_circle is not None:
                    # Evaluation only: the inscribed circle grown by a random factor, from its own generator.
                    disc, grown, _factor = oversize_circle(
                        shape,
                        seed_circle,
                        self.circle_scale,
                        np.random.default_rng([self.seed, sample_id, VAL_FAMILY_CODES[family], OVERSIZE_STREAM]),
                        self.circle_oversize_range,
                    )
                    if disc.any():
                        family_circle, family_seed_circle = disc, grown
                prompt = build_prompt(
                    family_type,
                    target_mask=target_aug,
                    circle_mask=family_circle,
                    regions=[] if family in NO_NEGATIVE_PROMPTS else regions_aug,
                    cfg=prompt_cfg,
                    rng=np.random.default_rng([self.seed, sample_id, VAL_FAMILY_CODES[family]]),
                    negative_canvases=None if family in NO_NEGATIVE_PROMPTS else canvases,
                )
                variant = _variant_from_prompt(prompt)
                if self.circle_seed_taper is not None and family_type in CIRCLE_PROMPT_TYPES:
                    variant["dense_mask"] = self._tapered_dense(
                        family_circle, family_seed_circle, [] if family in NO_NEGATIVE_PROMPTS else regions_aug,
                        None if family in NO_NEGATIVE_PROMPTS else canvases, prompt_cfg,
                    )
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
            self._circle_trust(),
        )
        return regions

    def _tapered_dense(
        self,
        disc: np.ndarray,
        circle: SeedCircle | None,
        regions: list[NegativeRegion],
        canvases,
        prompt_cfg: PromptConfig,
    ) -> torch.Tensor | None:
        """The dense prompt of a circle-seed variant under ``circle_seed_taper`` (evaluation only): ``disc`` fed as
        ``+seed_logit * w(rho)`` of ``circle``, with the same negatives :func:`build_prompt` used (validation never
        drops any, and a circle seed merges none)."""
        if circle is None:
            raise ValueError("circle_seed_taper needs the seed circle's centre and radius")
        if canvases is None:
            canvases = negative_region_canvases(regions, prompt_cfg) if regions else []
        weight = seed_taper_weight(disc.shape, circle, self.circle_seed_taper)
        dense = tapered_seed_mask_input(disc, weight, canvases, prompt_cfg.seed_logit)
        return None if dense is None else torch.from_numpy(dense)

    def _circle_trust(self) -> CircleAnnotationTrust:
        """The circle-annotation trust the prompts use (``prompt.circle_annotation_trust``)."""
        return (self.prompt_cfg if self.prompt_cfg is not None else PromptConfig()).circle_annotation_trust

    def _weights(
        self, target: np.ndarray, regions: Sequence[NegativeRegion], meta: SampleMeta
    ) -> dict[str, Any]:
        """The loss-side items of a sample: edge weight, training weight and (validation) neighbor cores.

        ``edge_weight`` is the tolerant weight of the ground truth (float16) times ``leak_weight`` on neighbor
        cores; None when both are off. ``neighbor_core`` is None unless ``core_band_px > 0``. A polygon's core is
        ``leak_core_px`` (loss) or ``core_band_px`` (metric) deep; an annotated circle's is ``rho <= full_frac``.
        """
        weight = edge_weight_map(target, self.edge_tau_px, self.edge_weight_min, self.edge_tau_depth_frac)
        masks = list(regions)
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
        """The prompt family of each variant: every variant, the first included, is drawn from the ``prob_*`` mix
        without replacement (no family is forced on a sample), each from its own generator
        ``[seed, epoch, index, k, 1]``, renormalised over the families still available (:func:`choose_prompt_type`).

        ``adjacent_poly_seed`` is available only with an adjacent-section shape and not on a fully masked tile;
        otherwise its share moves to the three circle peers in proportion to theirs. A fully masked tile draws only
        points, the plain circle seed and the annotator-like circle (:data:`FULL_TILE_PROMPTS`: three families, so with
        four variants it draws all three and the shortfall rule stops the plan there). Shortfall (unchanged
        rule): once no family with a positive share is left, :func:`choose_prompt_type` returns points (the plain
        circle seed when points are not allowed); if that family is already in the plan the plan stops there, else it
        is added once (even at share 0) and the next draw stops it. So a sample can have fewer variants.
        """
        if forced is not None:
            return [forced]
        allowed = FULL_TILE_PROMPTS if full_tile else None
        can_adjacent = has_adjacent and not full_tile
        chosen: list[PromptType] = []
        for k in range(cfg.variants_per_sample):
            rng = np.random.default_rng([self.seed, epoch, int(index), k, 1])
            kind = choose_prompt_type(cfg, rng, can_adjacent_poly_seed=can_adjacent, allowed=allowed, exclude=set(chosen))
            if kind in chosen:
                break
            chosen.append(kind)
        return chosen

    def _redraw_rejected_adjacent(
        self,
        cfg: PromptConfig,
        index: int,
        epoch: int,
        types: list[PromptType],
        *,
        full_tile: bool,
    ) -> tuple[list[PromptType], list[PromptType]]:
        """``(built, requested)`` families after a projected adjacent outline failed the overlap rule.

        The variant that drew ``adjacent_poly_seed`` is drawn again, from its own generator
        ``[seed, epoch, index, k, 5]``, from the families the sample can still use (the adjacent share moves to the
        circle peers, as when no outline exists, and the families already in the plan are excluded). It keeps
        ``adjacent_poly_seed`` as its requested family, so ``train/prompt_fallback/adjacent_poly_seed`` counts it.
        When nothing new is left (:func:`choose_prompt_type` returns a family already planned) the variant is
        dropped; :meth:`_mixed_item` then records it on the sample (``dropped_prompt_types``, and the failed rule in
        its ``adjacent_rule``), so the pass rate and the fallback count still see it.
        """
        built = list(types)
        requested = list(types)
        k = built.index(PromptType.ADJACENT_POLY_SEED)
        rng = np.random.default_rng([self.seed, epoch, int(index), k, ADJACENT_REDRAW_STREAM])
        kind = choose_prompt_type(
            cfg, rng, can_adjacent_poly_seed=False,
            allowed=FULL_TILE_PROMPTS if full_tile else None, exclude=set(built),
        )
        if kind in built:
            del built[k], requested[k]
        else:
            built[k] = kind
        return built, requested

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
            disc = rasterize_circle(shape, own.center_x, own.center_y, own.seed_radius(self.circle_scale))
            circle_mask = disc if disc.any() else None

        regions = self._other_annotations(example, shape)
        if prompt_cfg.drop_overlapping_circles:
            regions = drop_circles_over_target(regions, target)
        # One linked outline from a section 1-2 away, in this tile's pixel frame (None: the family is unavailable for
        # this sample and its share moves to the circle peers). Which neighbour, when there are several, and its
        # erosion come from its own generator [seed, epoch, index, 0, 4], so a resumed epoch draws the same one.
        # A projected outline sits where the registration carries it and counts only when it passes the overlap rule
        # against the (unaugmented) target; one that fails is not a prompt, and its variant is drawn again.
        adjacent_poly = prompt_cfg.adjacent_poly
        adjacent = (
            None
            if full_tile
            else tiles.adjacent_draw(
                key, example.location_id, target=target,
                rng=np.random.default_rng([self.seed, epoch, int(index), 0, ADJACENT_PICK_STREAM]),
                overlap_min=adjacent_poly.overlap_min, erode_px=adjacent_poly.erode_px,
            )
        )
        types = self._plan_types(
            prompt_cfg, index, epoch, full_tile=full_tile, forced=forced, has_adjacent=adjacent is not None
        )
        requested = list(types)
        adjacent_rule: str | None = None
        dropped: list[PromptType] = []  # families the mix asked for whose variant was dropped
        if adjacent is not None and adjacent.projected and PromptType.ADJACENT_POLY_SEED in types:
            adjacent_rule = "pass" if adjacent.passed else "fail"
            if not adjacent.passed:
                types, requested = self._redraw_rejected_adjacent(
                    prompt_cfg, index, epoch, types, full_tile=full_tile
                )
                if PromptType.ADJACENT_POLY_SEED not in requested:
                    dropped.append(PromptType.ADJACENT_POLY_SEED)
        adjacent_shape = None if adjacent is None else adjacent.raster
        adjacent_placed = adjacent is not None and adjacent.projected

        # The target, every negative, the circle and the adjacent outline go through the same geometric transform as
        # the image, so prompts stay registered to it (elastic warps and flips included; a flipped target keeps the
        # orientation of its neighbour's outline).
        # An adjacent polygon seed that cannot be fitted falls back to the circle seed, so it needs the circle too.
        needs_circle = any(kind in CIRCLE_PROMPT_TYPES or kind is PromptType.ADJACENT_POLY_SEED for kind in types)
        use_circle = needs_circle and circle_mask is not None
        use_adjacent = adjacent_shape is not None and PromptType.ADJACENT_POLY_SEED in types
        image_aug, target_aug, regions_aug, extras = augment_scene(
            self.transform, rgb, target, regions,
            ([circle_mask] if use_circle else []) + ([adjacent_shape] if use_adjacent else []),
        )
        if not target_aug.any():
            raise KeyError(f"{example.split_key} target is empty after augmentation")
        circle_aug = extras[0] if use_circle else None
        if use_adjacent:
            adjacent_shape = extras[-1]
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
        # Negative-context dropout applies to every training variant (no variant is a fixed production prompt).
        train_dropout = forced is None and self.split == "train"
        depth = None  # the target's distance transform, done once for however many circles are drawn
        variants: list[dict[str, Any]] = []
        for k, (kind, asked) in enumerate(zip(types, requested)):
            # The three circle peers share one seed prompt; they differ in the disc. An annotator-like or pull-through
            # disc has its own generator (trailing 2 or 3), so no other draw moves. A disc that cannot be drawn falls
            # back to the plain circle, and the variant is then reported as circle_seed (requested_prompt_type keeps
            # what the mix asked for).
            circle_k, built = circle_aug, kind
            drawn: dict = {}
            if kind in (PromptType.CIRCLE_ANNOTATOR, PromptType.CIRCLE_PULL_THROUGH):
                if depth is None:
                    depth = target_depth(target_aug)
                if kind is PromptType.CIRCLE_ANNOTATOR:
                    disc = annotator_circle(
                        target_aug, np.random.default_rng([self.seed, epoch, int(index), k, 2]),
                        prompt_cfg.circle_seed_jitter, depth, drawn,
                    )
                else:
                    disc = pull_through_circle(
                        target_aug, np.random.default_rng([self.seed, epoch, int(index), k, 3]),
                        prompt_cfg.circle_pull_through, depth, drawn,
                    )
                if disc is not None:
                    circle_k = disc
                else:
                    built = PromptType.CIRCLE_SEED
            prompt = build_prompt(
                built,
                target_mask=target_aug,
                circle_mask=circle_k,
                regions=[] if no_negatives else regions_aug,
                cfg=prompt_cfg,
                rng=np.random.default_rng([self.seed, epoch, int(index), k]),
                adjacent_shape=adjacent_shape,
                negative_canvases=canvases,
                negative_dropout=train_dropout,
                adjacent_placed=adjacent_placed,
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
                    # What the mix asked for; differs from prompt_type when a circle fell back to the plain one, an
                    # adjacent polygon seed could not be fitted, or a projected one failed the overlap rule.
                    "requested_prompt_type": asked.value,
                    # The overlap rule of a projected adjacent outline, on the variant that asked for one: "pass" or
                    # "fail" (that variant was drawn again); None otherwise.
                    "adjacent_rule": adjacent_rule if asked is PromptType.ADJACENT_POLY_SEED else None,
                    # Share of the projected outline in the tile that lies on the target (else None).
                    "adjacent_overlap": (
                        float(adjacent.overlap)
                        if asked is PromptType.ADJACENT_POLY_SEED and adjacent is not None and adjacent.overlap is not None
                        else None
                    ),
                    # For an annotator-like circle: whether the fit cap d(p) + fit_grow_px set its radius, not s * D.
                    "circle_fit_capped": bool(
                        prompt.prompt_type is PromptType.CIRCLE_ANNOTATOR and drawn.get("capped", False)
                    ),
                    # For a pull-through circle: the fraction of the disc inside the target (else None).
                    "circle_overlap": (
                        float(drawn["overlap"]) if prompt.prompt_type is PromptType.CIRCLE_PULL_THROUGH else None
                    ),
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
            # The overlap rule of the sample's projected adjacent outline ("pass", "fail" or None), kept on the sample
            # so a failure still counts when its variant was dropped (train/adjacent_poly/pass_rate).
            "adjacent_rule": adjacent_rule,
            # Families the mix asked for whose variant was dropped (train/prompt_fallback/{family}/dropped).
            "dropped_prompt_types": [kind.value for kind in dropped],
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
    one). ``meta`` and ``orig_hw`` are repeated for each variant of a sample; ``prompt_type`` (the family built) and
    ``requested_prompt_type`` (the family the mix drew; they differ after a fallback) are parallel lists.
    ``adjacent_rule`` is per sample (the overlap rule of its projected adjacent outline, or None), and
    ``dropped_prompt_type`` lists the families asked for whose variant was dropped, over the whole batch.
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
        "requested_prompt_type": [
            variant.get("requested_prompt_type", variant.get("prompt_type", "centroid")) for _b, variant in rows
        ],
        "adjacent_rule": [item.get("adjacent_rule") for item in batch],
        "dropped_prompt_type": [kind for item in batch for kind in item.get("dropped_prompt_types") or ()],
        "orig_hw": [batch[b]["orig_hw"] for b, _v in rows],
        "meta": [batch[b]["sample_meta"].as_dict() for b, _v in rows],
    }
