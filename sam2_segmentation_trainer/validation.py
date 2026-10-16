"""Validation on the prompts the model is used with, scored against the project goal.

The goal is to identify cells and their connectivity: at least 97.5% of each cell's interior captured, no merges
into neighbors (docs/mask_refinement_plan.md, section 6). :class:`ValidationSuite` scores every sample under each
prompt family (``eval.val_prompts``) with fixed, reproducible prompts and reports, per family, the goal metrics of
:mod:`sam2_segmentation_trainer.goal_metrics` (core recall, leak, merge rate, spill, fragmentation) with tolerant
IoU and boundary F1 at 10 and 20 px as diagnostics. Masks in ``approved.json`` are reported as a separate row.

One validation pass loads each sample once, runs the image encoder once, and decodes all the families from that
embedding, like training does with its prompt variants. The metrics run in a thread pool while the GPU works on
the next batch. Every epoch scores a fixed subset (``val.epoch_subset`` of each volume, stratified, always holding
every Approved sample) without the legacy centroid family; the full split, centroid included, runs every
``val.full_every_n_epochs`` epochs and after the last. One family (``eval.primary_val_prompt``) of the subset pass
picks the best checkpoint and drives early stopping, through :func:`epoch_selection`.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from annotation_crops.prompt_encoding import window
from sam2_segmentation_trainer.data.approved import as_approved, is_approved
from sam2_segmentation_trainer.data.dataset import VAL_PROMPTS, EMSegDataset, collate_em
from sam2_segmentation_trainer.goal_metrics import (
    CORE_BAND_PX,
    CORE_RECALL_TARGET,
    MERGE_MIN_PX,
    MIN_COMPONENT_PX,
    DepthBin,
    GoalSummary,
    SelectionKey,
    add_goal_fields,
    ground_truth_terms,
    sample_goal_metrics,
    sample_no_harm,
    selection_key,
    summarize_by_depth,
    summarize_goal,
)
from sam2_segmentation_trainer.metrics import boundary_f1_multi, iou_dice_pr, tolerant_iou
from sam2_segmentation_trainer.model import predict_masks, predict_masks_mixed
from sam2_segmentation_trainer.overlays import denorm_preview, overlay_panel

DEFAULT_VAL_PROMPTS = (
    "circle_mask", "correction", "near_correct", "point", "centroid", "circle_mask_noneg", "circle_annotator",
)
"""Validation prompts when ``eval.val_prompts`` is unset: the production circle seed, a mask to correct, a
near-correct mask ("do no harm"), the interior point, the legacy centroid, the circle seed without negative
context and an annotator-like circle seed (both diagnostics; neither selects a checkpoint)."""

NEG_SPLIT_PROMPTS = ("circle_mask", "circle_mask_noneg", "circle_annotator")
"""Circle-seed families whose results are also split by whether the sample has negative context (``with_neg`` and
``no_neg``)."""

RAW_IOU_PROMPTS = frozenset({"centroid"})
"""Prompts scored with the plain IoU, kept for continuity with earlier runs. The others use the tolerant IoU."""

REFINEMENT_PROMPTS = frozenset({"correction", "near_correct"})
"""Prompts whose input mask is scored too, to measure the change from it and whether the model left a good mask alone."""

LEGACY_PROMPTS = frozenset({"centroid"})
"""Families that only keep continuity with earlier runs; the per-epoch subset pass leaves them out."""

BOUNDARY_TOLERANCES = (10, 20)
"""Boundary F1 tolerances in pixels (diagnostics only; the traced edge is about 9 px from the membrane)."""

CROP_MARGIN = max(BOUNDARY_TOLERANCES) + 4
"""Pixels kept around the masks when a sample is scored in their bounding box (enough for the outlines and tolerances)."""


@dataclass(slots=True)
class ModeResult:
    """Scores of one validation prompt mode.

    The numbers average the samples that are not fully masked tiles; those (``fulltile_*``) are kept apart because
    a point inside a tile the target fills scores about 1 and would inflate the headline. ``goal`` is the pooled
    goal metrics; for the refinement prompts ``input_goal`` holds the same metrics for the mask the prompt carried.
    ``approved`` is the Approved subset of the same samples (``approved_n`` of them).
    """

    n: int = 0
    iou: float = 0.0
    """Tolerant IoU (plain IoU for the legacy prompts). A diagnostic."""
    raw_iou: float = 0.0
    bf10: float = 0.0
    bf20: float = 0.0
    goal: GoalSummary = field(default_factory=GoalSummary)
    input_goal: GoalSummary | None = None
    approved: GoalSummary | None = None
    approved_input_goal: GoalSummary | None = None
    approved_n: int = 0
    input_iou: float | None = None
    gain: float | None = None
    no_harm: float | None = None
    by_volume: dict[str, float] = field(default_factory=dict)
    by_category: dict[str, float] = field(default_factory=dict)
    goal_by_volume: dict[str, GoalSummary] = field(default_factory=dict)
    goal_by_category: dict[str, GoalSummary] = field(default_factory=dict)
    recall_by_depth: dict[str, DepthBin] = field(default_factory=dict)
    """Core recall by the ground truth's maximum depth (see ``goal_metrics.DEPTH_BINS``)."""
    approved_by_volume: dict[str, int] = field(default_factory=dict)
    with_neg: GoalSummary | None = None
    """Goal metrics over the samples whose tile has negative context (None when the records do not say)."""
    no_neg: GoalSummary | None = None
    """... and over the samples without any."""
    with_neg_n: int = 0
    no_neg_n: int = 0
    fulltile_iou: float | None = None
    fulltile_n: int = 0


def summarize_mode(records: Sequence[dict[str, Any]], *, target: float = CORE_RECALL_TARGET) -> ModeResult:
    """Collapse per-sample records into a :class:`ModeResult`.

    Records carry the goal-metric counts (see :func:`sample_goal_metrics`), the tolerant ``iou``, ``bf10`` and
    ``bf20``. Records with ``input_core_gt`` etc. also give the input mask's goal metrics, and the share of
    samples that pass "do no harm" (:func:`sample_no_harm`).
    """
    result = ModeResult()
    normal = [r for r in records if not r.get("full_tile")]
    full = [r for r in records if r.get("full_tile")]
    if full:
        result.fulltile_iou = float(np.mean([r["iou"] for r in full]))
        result.fulltile_n = len(full)
    # With nothing but fully masked tiles there is no other number to report, so fall back to them.
    scored = normal or full
    if not scored:
        return result
    result.n = len(scored)
    result.iou = float(np.mean([r["iou"] for r in scored]))
    result.raw_iou = float(np.mean([r.get("raw_iou", r["iou"]) for r in scored]))
    result.bf10 = float(np.mean([r.get("bf10", 0.0) for r in scored]))
    result.bf20 = float(np.mean([r.get("bf20", 0.0) for r in scored]))
    result.goal = summarize_goal(scored, target=target)
    result.recall_by_depth = summarize_by_depth(scored)
    for field_name, goals in (("volume", result.goal_by_volume), ("category", result.goal_by_category)):
        groups: dict[str, list[dict[str, Any]]] = {}
        for record in scored:
            name = str(record.get(field_name) or "").strip()
            if name:
                groups.setdefault(name, []).append(record)
        goals.update({name: summarize_goal(rows, target=target) for name, rows in sorted(groups.items())})
    with_input = [r for r in scored if "input_core_gt" in r]
    if with_input:
        result.input_goal = summarize_goal(with_input, prefix="input_", target=target)
        result.input_iou = float(np.mean([r["input_iou"] for r in with_input]))
        result.gain = float(np.mean([r["iou"] - r["input_iou"] for r in with_input]))
        result.no_harm = float(np.mean([sample_no_harm(r) for r in with_input]))
    if any("has_neg" in r for r in scored):
        with_neg = [r for r in scored if r.get("has_neg")]
        no_neg = [r for r in scored if "has_neg" in r and not r["has_neg"]]
        result.with_neg_n, result.no_neg_n = len(with_neg), len(no_neg)
        result.with_neg = summarize_goal(with_neg, target=target) if with_neg else None
        result.no_neg = summarize_goal(no_neg, target=target) if no_neg else None
    approved = [r for r in scored if r.get("approved")]
    result.approved_n = len(approved)
    if approved:
        result.approved = summarize_goal(approved, target=target)
        approved_inputs = [r for r in approved if "input_core_gt" in r]
        if approved_inputs:
            result.approved_input_goal = summarize_goal(approved_inputs, prefix="input_", target=target)
        for record in approved:
            name = str(record.get("volume") or "").strip()
            if name:
                result.approved_by_volume[name] = result.approved_by_volume.get(name, 0) + 1
    for field_name, bucket in (("volume", result.by_volume), ("category", result.by_category)):
        buckets: dict[str, list[float]] = {}
        for record in scored:
            name = str(record.get(field_name) or "").strip()
            if name:
                buckets.setdefault(name, []).append(float(record["iou"]))
        bucket.update({name: float(np.mean(values)) for name, values in sorted(buckets.items())})
    return result


def epoch_selection(
    results: dict[str, ModeResult], primary: str, target: float = CORE_RECALL_TARGET
) -> SelectionKey:
    """The key that orders epochs for ``best_model.pt`` and early stopping, from the primary mode's goal metrics."""
    return selection_key(results[primary].goal, target)


def predict_batch(model, batch, device, image_size: int) -> torch.Tensor:
    """Predicted logits `[B, H, W]` for a validation batch, whatever prompts its samples carry.

    A batch with boxes or dense masks goes through :func:`predict_masks_mixed`; a plain point batch keeps the
    original :func:`predict_masks` path.
    """
    images = batch["image"].to(device, non_blocking=True)
    points = batch["point"].to(device, non_blocking=True)
    labels = batch["point_label"].to(device, non_blocking=True)
    boxes = batch.get("boxes") or []
    dense = batch.get("dense_mask") or []
    if any(b is not None for b in boxes) or any(d is not None for d in dense):
        pred, _iou = predict_masks_mixed(model, images, points, labels, boxes, dense, image_size=image_size)
    else:
        pred, _iou = predict_masks(model, images, points, labels, image_size=image_size)
    return pred


def predict_variants(model, batch, device, image_size: int) -> torch.Tensor:
    """Predicted logits `[N, H, W]` for every prompt variant of a batch: one encoder pass per image.

    ``batch["variant_index"]`` names the image of each variant, so a batch of ``B`` images with ``F`` families
    gives ``B * F`` rows from ``B`` encodings.
    """
    images = batch["image"].to(device, non_blocking=True)
    points = batch["point"].to(device, non_blocking=True)
    labels = batch["point_label"].to(device, non_blocking=True)
    index = batch["variant_index"].to(device, non_blocking=True)
    pred, _iou = predict_masks_mixed(
        model, images, points, labels, batch["boxes"], batch["dense_mask"], image_size=image_size, variant_index=index
    )
    return pred


def _crop(truth: dict[str, Any] | None, box) -> dict[str, Any] | None:
    if truth is None:
        return None
    cropped = dict(truth)
    cropped["core"] = truth["core"][box]
    cropped["grown"] = truth["grown"][box]
    return cropped


def score_sample(
    pred_np: np.ndarray,
    gt_np: np.ndarray,
    item: dict[str, Any],
    *,
    weight_np: np.ndarray | None,
    neighbor_core: np.ndarray | None,
    input_np: np.ndarray | None,
    mode: str | None,
    band: float,
    merge_min_px: int = MERGE_MIN_PX,
    truth: dict[str, Any] | None = None,
    crop: bool = True,
) -> dict[str, Any]:
    """The record of one validation sample (see :func:`summarize_mode` for the fields).

    ``truth`` is the ground-truth side of the goal metrics (:func:`ground_truth_terms`), computed once per sample
    and shared by every family and the input mask. With ``crop`` the work is restricted to the bounding box of the
    prediction, the ground truth, the neighbor cores and the input mask, grown by :data:`CROP_MARGIN`; nothing
    outside it can change a score, so the result is the same as on the whole tile.
    """
    pred_np = np.asarray(pred_np) > 0
    gt_np = np.asarray(gt_np) > 0
    if crop:
        union = pred_np | gt_np
        if neighbor_core is not None:
            union = union | (neighbor_core > 0)
        if input_np is not None:
            union = union | input_np
        box = window(union, CROP_MARGIN)
        if box is not None:
            pred_np, gt_np = pred_np[box], gt_np[box]
            weight_np = None if weight_np is None else weight_np[box]
            neighbor_core = None if neighbor_core is None else neighbor_core[box]
            input_np = None if input_np is None else input_np[box]
            truth = _crop(truth, box)
    raw = iou_dice_pr(pred_np, gt_np)["iou"]
    score = raw if mode in RAW_IOU_PROMPTS else tolerant_iou(pred_np, gt_np, weight_np)
    bf10, bf20 = boundary_f1_multi(pred_np, gt_np, BOUNDARY_TOLERANCES)
    record: dict[str, Any] = {
        "volume": str(item.get("volume", "")).strip(),
        "category": str(item.get("category_name", "")),
        "key": (str(item.get("volume", "")), str(item.get("image_key", "")), int(item.get("location_id", 0) or 0)),
        "iou": score,
        "raw_iou": raw,
        "bf10": bf10,
        "bf20": bf20,
        "full_tile": bool(item.get("full_tile", False)),
        "approved": bool(item.get("approved", False)),
        "has_neg": bool(item.get("has_negatives", False)),
    }
    if truth is None:
        truth = ground_truth_terms(gt_np, band, MIN_COMPONENT_PX)
    record["gt_depth"] = truth.get("depth")
    add_goal_fields(record, sample_goal_metrics(pred_np, gt_np, neighbor_core, band=band, merge_min_px=merge_min_px, cache=truth))
    if input_np is not None and bool(input_np.any()):
        record["input_iou"] = tolerant_iou(input_np, gt_np, weight_np)
        add_goal_fields(
            record, sample_goal_metrics(input_np, gt_np, neighbor_core, band=band, merge_min_px=merge_min_px, cache=truth),
            "input_"
        )
    return record


def collect_records(
    model,
    loader,
    device,
    image_size: int,
    threshold: float,
    *,
    overlays_per_volume: int = 0,
    mode: str | None = None,
    core_band_px: float = CORE_BAND_PX,
    merge_min_px: int = MERGE_MIN_PX,
    timings: dict[str, float] | None = None,
) -> tuple[list[dict[str, Any]], list[tuple[str, np.ndarray]]]:
    """Score every batch of ``loader``, one prompt per sample. Returns per-sample records and overlay strips.

    The simple serial path (one family, one image per row); the validation suite uses
    :func:`collect_family_records`. A batch that carries ``edge_weight`` is also scored with the tolerant IoU
    (pixels in the edge band ignored) and ``iou`` is that number, except for the legacy prompts in
    :data:`RAW_IOU_PROMPTS`, which keep the raw IoU. A batch that carries ``neighbor_core`` gets leak and merge
    scored too. With ``mode`` in :data:`REFINEMENT_PROMPTS` each record also has the input mask's scores
    (``input_*``). With ``timings`` the seconds spent in the per-sample metrics accumulate under ``"metrics_s"``.
    """
    model.eval()
    records: list[dict[str, Any]] = []
    overlays: list[tuple[str, np.ndarray]] = []
    overlay_counts: dict[str, int] = {}
    keep_overlays = max(0, int(overlays_per_volume))
    try:
        with torch.no_grad():
            for batch in loader:
                masks = batch["mask"].to(device, non_blocking=True)
                pred = predict_batch(model, batch, device, image_size)
                pred_bin = (torch.sigmoid(pred) > threshold).to(torch.int64)
                meta = batch.get("meta") or []
                weights = batch.get("edge_weight")
                cores = batch.get("neighbor_core")
                inputs = batch.get("prompt_input") if mode in REFINEMENT_PROMPTS else None
                for index, (pred_i, mask_i, item) in enumerate(zip(pred_bin, masks, meta, strict=True)):
                    pred_np = pred_i.detach().cpu().numpy()
                    gt_np = mask_i.detach().cpu().numpy()
                    item = item or {}
                    started = time.perf_counter()
                    record = score_sample(
                        pred_np,
                        gt_np,
                        item,
                        weight_np=None if weights is None else weights[index].float().numpy(),
                        neighbor_core=None if cores is None else cores[index].numpy(),
                        input_np=None if inputs is None else inputs[index].numpy() > 0,
                        mode=mode,
                        band=core_band_px,
                        merge_min_px=merge_min_px,
                    )
                    if timings is not None:
                        timings["metrics_s"] = timings.get("metrics_s", 0.0) + time.perf_counter() - started
                    records.append(record)
                    volume = record["volume"]
                    if not volume or keep_overlays <= 0:
                        continue
                    seen = overlay_counts.get(volume, 0)
                    if seen >= keep_overlays:
                        continue
                    panel = overlay_panel(denorm_preview(batch["image"][index]), gt_np, pred_np)
                    overlays.append((volume, panel))
                    overlay_counts[volume] = seen + 1
    finally:
        model.train()
    return records, overlays


def collect_family_records(
    model,
    loader,
    device,
    image_size: int,
    threshold: float,
    *,
    core_band_px: float = CORE_BAND_PX,
    merge_min_px: int = MERGE_MIN_PX,
    metric_threads: int = 4,
    overlays_per_volume: int = 0,
    overlay_family: str | None = None,
    timings: dict[str, float] | None = None,
    max_pending: int = 64,
) -> tuple[dict[str, list[dict[str, Any]]], list[tuple[str, np.ndarray]]]:
    """Score every family of every sample in ``loader``: one encoder pass per image, metrics in a thread pool.

    ``loader`` yields batches from a dataset built with ``val_families``: each image has one prompt variant per
    family, ``batch["variant_index"]`` names the image of each variant and ``batch["val_family"]`` its family.
    The prediction of a batch comes back to the CPU and its rows are handed to the pool, so the metrics of one
    batch overlap with the GPU work on the next; at most ``max_pending`` rows wait at a time. Returns the records
    of each family in sample order and a few overlay strips of ``overlay_family``. With ``timings``, the seconds the
    main thread waited for the metrics accumulate under ``"metrics_wait_s"``.
    """
    model.eval()
    records: dict[str, list[dict[str, Any]]] = {}
    overlays: list[tuple[str, np.ndarray]] = []
    overlay_counts: dict[str, int] = {}
    keep_overlays = max(0, int(overlays_per_volume))
    pending: deque[tuple[str, Future]] = deque()

    def drain(limit: int) -> None:
        while len(pending) > limit:
            family, future = pending.popleft()
            records.setdefault(family, []).append(future.result())

    pool = ThreadPoolExecutor(max_workers=max(1, int(metric_threads)))
    try:
        with torch.no_grad():
            for batch in loader:
                logits = predict_variants(model, batch, device, image_size)
                pred_bin = (torch.sigmoid(logits) > threshold).to(torch.uint8).cpu().numpy()
                masks = batch["mask"].numpy() > 0
                weights = batch.get("edge_weight")
                weights_np = None if weights is None else weights.float().numpy()
                cores = batch.get("neighbor_core")
                cores_np = None if cores is None else cores.numpy()
                inputs = batch.get("prompt_input")
                terms = batch.get("gt_terms") or [None] * masks.shape[0]
                families = batch["val_family"]
                for row, (b, family) in enumerate(zip(batch["variant_index"].tolist(), families, strict=True)):
                    input_np = None
                    if inputs is not None and family in REFINEMENT_PROMPTS:
                        input_np = inputs[row].numpy() > 0
                    future = pool.submit(
                        score_sample,
                        pred_bin[row],
                        masks[b],
                        batch["meta"][row] or {},
                        weight_np=None if weights_np is None else weights_np[b],
                        neighbor_core=None if cores_np is None else cores_np[b],
                        input_np=input_np,
                        mode=family,
                        band=core_band_px,
                        merge_min_px=merge_min_px,
                        truth=terms[b],
                    )
                    pending.append((family, future))
                    volume = str((batch["meta"][row] or {}).get("volume", "")).strip()
                    if keep_overlays > 0 and family == overlay_family and volume:
                        seen = overlay_counts.get(volume, 0)
                        if seen < keep_overlays:
                            panel = overlay_panel(denorm_preview(batch["image"][b]), masks[b], pred_bin[row])
                            overlays.append((volume, panel))
                            overlay_counts[volume] = seen + 1
                started = time.perf_counter()
                drain(max_pending)
                if timings is not None:
                    timings["metrics_wait_s"] = timings.get("metrics_wait_s", 0.0) + time.perf_counter() - started
            started = time.perf_counter()
            drain(0)
            if timings is not None:
                timings["metrics_wait_s"] = timings.get("metrics_wait_s", 0.0) + time.perf_counter() - started
    finally:
        pool.shutdown(wait=True)
        model.train()
    return records, overlays


def full_pass_due(epoch: int, num_epochs: int, every: int) -> bool:
    """Whether epoch ``epoch`` (0-based) validates the full split instead of the subset: every ``every``-th epoch
    (counting from 1) and the last one."""
    return (int(epoch) + 1) % max(1, int(every)) == 0 or int(epoch) == int(num_epochs) - 1


def select_validation_subset(
    examples: Sequence, fraction: float, seed: int, approved: frozenset | set | None = None
) -> list:
    """A fixed, stratified subset of the validation examples, always including every Approved one.

    Each volume contributes its ``ceil(fraction * n)`` examples with the smallest hash of
    ``(seed, volume, image key, location id)``, so the draw is reproducible from the run seed, stable when the
    inventory changes a little, and does not depend on the order of ``examples``. A ``fraction`` of 1 or more keeps
    everything. The result keeps the order of ``examples``.
    """
    examples = list(examples)
    if fraction >= 1.0 or not examples:
        return examples
    approved = as_approved(approved)

    def draw(example) -> float:
        text = f"{seed}|{example.volume}|{example.image_key}|{example.location_id}".encode("utf-8")
        return int.from_bytes(hashlib.sha1(text).digest()[:8], "big") / float(1 << 64)

    by_volume: dict[str, list] = {}
    for example in examples:
        by_volume.setdefault(example.volume, []).append(example)
    chosen: set[int] = set()
    for members in by_volume.values():
        count = max(1, math.ceil(max(0.0, fraction) * len(members)))
        for example in sorted(members, key=draw)[:count]:
            chosen.add(id(example))
    for example in examples:
        if is_approved(approved, example):
            chosen.add(id(example))
    return [example for example in examples if id(example) in chosen]


@dataclass(slots=True)
class SuiteResult:
    """One validation pass. ``subset`` is always set (the per-epoch numbers); ``full`` only on a full pass."""

    subset: dict[str, ModeResult]
    full: dict[str, ModeResult] | None = None
    overlays: list[tuple[str, np.ndarray]] = field(default_factory=list)
    seconds: float = 0.0
    n_samples: int = 0
    timings: dict[str, float] = field(default_factory=dict)


class ValidationSuite:
    """The validation split under every configured prompt family, scored with one encoder pass per sample.

    ``dataset`` is the full-split dataset (the one callers size and refresh); :meth:`replace_examples` keeps the
    subset in step when the inventory changes. ``approved`` is the set of ``(location_id, image_key)`` pairs from
    ``approved.json``; ``core_band_px`` is the band the goal metrics ignore. ``epoch_subset`` is the share of each
    volume validated every epoch (1 = the whole split every time).
    """

    def __init__(
        self,
        examples: Sequence,
        *,
        modes: Sequence[str],
        primary: str,
        image_size: int,
        num_workers: int,
        missing_tiles=None,
        tiles=None,
        prompt_cfg=None,
        seed: int = 0,
        circle_scale: float = 0.95,
        full_tile_threshold: float | None = None,
        edge_tau_px: float = 8.0,
        edge_weight_min: float = 0.05,
        edge_tau_depth_frac: float = 0.25,
        core_band_px: float = CORE_BAND_PX,
        merge_min_px: int = MERGE_MIN_PX,
        core_recall_target: float = CORE_RECALL_TARGET,
        approved: frozenset | set | None = None,
        epoch_subset: float = 1.0,
        batch_size: int = 1,
        metric_threads: int = 4,
    ) -> None:
        modes = list(dict.fromkeys(modes))
        if not modes:
            raise ValueError("eval.val_prompts must name at least one prompt")
        unknown = [m for m in modes if m not in VAL_PROMPTS]
        if unknown:
            raise ValueError(f"unknown validation prompt(s) {unknown}; choose from {list(VAL_PROMPTS)}")
        if primary not in modes:
            raise ValueError(f"eval.primary_val_prompt {primary!r} is not in eval.val_prompts {modes}")
        self.modes = modes
        self.subset_modes = [m for m in modes if m not in LEGACY_PROMPTS] or list(modes)
        if primary not in self.subset_modes:
            raise ValueError(f"eval.primary_val_prompt {primary!r} cannot be a legacy family")
        self.primary = primary
        self.num_workers = int(num_workers)
        self.batch_size = max(1, int(batch_size))
        self.metric_threads = max(1, int(metric_threads))
        self.core_band_px = float(core_band_px)
        self.merge_min_px = int(merge_min_px)
        self.core_recall_target = float(core_recall_target)
        self.epoch_subset = float(epoch_subset)
        self.seed = int(seed)
        self.approved = as_approved(approved)
        self._dataset_args = dict(
            split="val",
            image_size=image_size,
            missing_tiles=missing_tiles,
            tiles=tiles,
            prompt_cfg=prompt_cfg,
            seed=seed,
            circle_scale=circle_scale,
            full_tile_threshold=full_tile_threshold,
            edge_tau_px=edge_tau_px,
            edge_weight_min=edge_weight_min,
            edge_tau_depth_frac=edge_tau_depth_frac,
            core_band_px=core_band_px,
            approved=approved,
        )
        examples = list(examples)
        self.full_dataset = EMSegDataset(examples, val_families=self.modes, **self._dataset_args)
        self.subset_dataset = EMSegDataset(
            select_validation_subset(examples, self.epoch_subset, self.seed, self.approved),
            val_families=self.subset_modes,
            **self._dataset_args,
        )
        self._subset_keys: set[tuple[str, str, int]] = set()
        self._refresh_keys()

    def _refresh_keys(self) -> None:
        self._subset_keys = {(e.volume, e.image_key, int(e.location_id)) for e in self.subset_dataset.examples}

    @property
    def dataset(self) -> EMSegDataset:
        """The full-split dataset, for sizing and emptiness checks."""
        return self.full_dataset

    def __len__(self) -> int:
        return len(self.full_dataset)

    @property
    def subset_size(self) -> int:
        return len(self.subset_dataset)

    def replace_examples(self, examples: Sequence) -> None:
        examples = list(examples)
        self.full_dataset.replace_examples(examples)
        self.subset_dataset.replace_examples(
            select_validation_subset(examples, self.epoch_subset, self.seed, self.approved)
        )
        self._refresh_keys()

    def set_tiles(self, tiles) -> None:
        self.full_dataset.tiles = tiles
        self.subset_dataset.tiles = tiles

    def loader(self, full: bool = False, limit: int | None = None) -> DataLoader:
        dataset = self.full_dataset if full else self.subset_dataset
        if limit is not None:
            dataset = torch.utils.data.Subset(dataset, list(range(min(int(limit), len(dataset)))))
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_em,
        )

    def run(
        self,
        model,
        device,
        image_size: int,
        threshold: float,
        *,
        full: bool = False,
        overlays_per_volume: int = 0,
    ) -> SuiteResult:
        """One pass over the subset, or over the full split (which also yields the subset numbers).

        A full pass scores every family, centroid included; the subset results are filtered out of it, so a
        full pass costs no extra subset pass. Overlay strips come from the primary family.
        """
        started = time.perf_counter()
        timings: dict[str, float] = {}
        records, overlays = collect_family_records(
            model,
            self.loader(full=full),
            device,
            image_size,
            threshold,
            core_band_px=self.core_band_px,
            merge_min_px=self.merge_min_px,
            metric_threads=self.metric_threads,
            overlays_per_volume=overlays_per_volume,
            overlay_family=self.primary,
            timings=timings,
        )
        target = self.core_recall_target
        full_results = None
        if full:
            full_results = {mode: summarize_mode(records.get(mode, []), target=target) for mode in self.modes}
            subset_records = {
                mode: [r for r in records.get(mode, []) if r["key"] in self._subset_keys]
                for mode in self.subset_modes
            }
        else:
            subset_records = {mode: records.get(mode, []) for mode in self.subset_modes}
        subset_results = {mode: summarize_mode(recs, target=target) for mode, recs in subset_records.items()}
        n_samples = len(self.full_dataset if full else self.subset_dataset)
        return SuiteResult(
            subset=subset_results,
            full=full_results,
            overlays=overlays,
            seconds=time.perf_counter() - started,
            n_samples=n_samples,
            timings=timings,
        )


_GOAL_TAGS = (
    ("core_recall", "core_recall"),
    ("core_recall_p10", "core_recall_p10"),
    ("core_recall_ok", "core_recall_ok"),
    ("leak", "leak"),
    ("merge_rate", "merge_rate"),
    ("spill", "spill"),
    ("frag", "frag"),
)


def _log_goal(writer, root: str, mode: str, goal: GoalSummary, epoch: int) -> None:
    for tag, attribute in _GOAL_TAGS:
        writer.add_scalar(f"{root}/{tag}/{mode}", getattr(goal, attribute), epoch)


BREAKDOWN_TAGS = ("core_recall", "core_recall_p10", "merge_rate", "spill")
"""The goal metrics logged per category and per volume on full passes."""


def _log_breakdown(writer, root: str, mode: str, result: ModeResult, epoch: int) -> None:
    """Per-category (``<root>_cat``), per-volume (``<root>_vol``) and per-depth (``<root>_depth``) goal metrics."""
    for suffix, groups in (("cat", result.goal_by_category), ("vol", result.goal_by_volume)):
        for name, goal in groups.items():
            for tag in BREAKDOWN_TAGS:
                writer.add_scalar(f"{root}_{suffix}/{tag}/{mode}/{name}", getattr(goal, tag), epoch)
    for name, bin_ in result.recall_by_depth.items():
        writer.add_scalar(f"{root}_depth/n/{mode}/{name}", bin_.n, epoch)
        writer.add_scalar(f"{root}_depth/with_core/{mode}/{name}", bin_.with_core, epoch)
        for tag, value in (("core_recall", bin_.core_recall), ("mean_recall", bin_.mean_recall), ("core_recall_p10", bin_.p10)):
            writer.add_scalar(f"{root}_depth/{tag}/{mode}/{name}", value, epoch)


def log_results(
    writer,
    epoch: int,
    results: dict[str, ModeResult],
    primary: str,
    target: float = CORE_RECALL_TARGET,
    *,
    root: str = "val",
    label: str | None = None,
) -> str:
    """Write every mode's scores to TensorBoard and return the one-line summary for the transcript.

    The goal metrics go to ``<root>/<metric>/<mode>``; for the refinement prompts the input mask's values go to
    ``<root>/input/<metric>/<mode>`` and the change to ``<root>/delta_<metric>/<mode>``; Approved masks go to
    ``<root>_approved/<metric>/<mode>``. The per-epoch subset pass logs under ``val_subset`` and the full pass
    under ``val_full``, so the two never mix; with ``root="val_subset"`` the old names ``val/iou`` and
    ``val_iou/{volume}`` are also written, carrying the primary mode's tolerant IoU (a diagnostic) for existing
    dashboards. ``label`` names the pass in the summary line. A full pass (``root="val_full"``) also logs the
    goal metrics per category (``val_full_cat/<metric>/<mode>/<category>``), per volume (``val_full_vol/...``) and
    core recall by the ground truth's maximum depth (``val_full_depth/<metric>/<mode>/<bin>``).
    """
    for mode, result in results.items():
        writer.add_scalar(f"{root}/iou/{mode}", result.iou, epoch)
        writer.add_scalar(f"{root}/raw_iou/{mode}", result.raw_iou, epoch)
        writer.add_scalar(f"{root}/bf10/{mode}", result.bf10, epoch)
        writer.add_scalar(f"{root}/bf20/{mode}", result.bf20, epoch)
        _log_goal(writer, root, mode, result.goal, epoch)
        if root == "val_full":
            _log_breakdown(writer, root, mode, result, epoch)
        if result.input_goal is not None:
            _log_goal(writer, f"{root}/input", mode, result.input_goal, epoch)
            for tag, attribute in _GOAL_TAGS:
                delta = getattr(result.goal, attribute) - getattr(result.input_goal, attribute)
                writer.add_scalar(f"{root}/delta_{tag}/{mode}", delta, epoch)
            writer.add_scalar(f"{root}/input_iou/{mode}", result.input_iou, epoch)
            writer.add_scalar(f"{root}/gain/{mode}", result.gain, epoch)
        if result.no_harm is not None and mode == "near_correct":
            writer.add_scalar(f"{root}/no_harm/{mode}", result.no_harm, epoch)
        if mode in NEG_SPLIT_PROMPTS:
            for name, goal, count in (
                ("with_neg", result.with_neg, result.with_neg_n),
                ("no_neg", result.no_neg, result.no_neg_n),
            ):
                writer.add_scalar(f"{root}_neg/n/{mode}/{name}", count, epoch)
                if goal is not None:
                    for tag, attribute in _GOAL_TAGS:
                        writer.add_scalar(f"{root}_neg/{tag}/{mode}/{name}", getattr(goal, attribute), epoch)
        if result.approved is not None:
            _log_goal(writer, f"{root}_approved", mode, result.approved, epoch)
        writer.add_scalar(f"{root}_approved/n/{mode}", result.approved_n, epoch)
        for volume, count in result.approved_by_volume.items():
            writer.add_scalar(f"{root}_approved_n/{mode}/{volume}", count, epoch)
        if result.fulltile_iou is not None:
            writer.add_scalar(f"{root}/iou_fulltile/{mode}", result.fulltile_iou, epoch)
        for volume, score in result.by_volume.items():
            writer.add_scalar(f"{root}_iou/{mode}/{volume}", score, epoch)
        for category, score in result.by_category.items():
            writer.add_scalar(f"{root}_iou_cat/{mode}/{category}", score, epoch)
    main = results[primary]
    if root == "val_subset":
        writer.add_scalar("val/iou", main.iou, epoch)
        for volume, score in main.by_volume.items():
            writer.add_scalar(f"val_iou/{volume}", score, epoch)
    return format_summary(epoch, results, primary, target, label=label, detail=root == "val_full")


def _goal_text(goal: GoalSummary) -> str:
    return (
        f"core recall {goal.core_recall:.4f} (p10 {goal.core_recall_p10:.3f}, "
        f"{goal.core_recall_ok:.0%} of samples at the target), merge {goal.merge_rate:.3f}, "
        f"leak {goal.leak:.4f}, spill {goal.spill:.3f}, frag {goal.frag:.3f}"
    )


def _depth_text(bins: dict[str, DepthBin]) -> str:
    names = {"lt20": "<20 px", "20_60": "20-60 px", "ge60": ">=60 px"}
    return ", ".join(
        f"{names.get(name, name)}: {b.mean_recall:.3f} (n={b.n}, {b.n - b.with_core} without a core)"
        for name, b in bins.items()
    )


def format_summary(
    epoch: int,
    results: dict[str, ModeResult],
    primary: str,
    target: float,
    *,
    label: str | None = None,
    detail: bool = False,
) -> str:
    """The one-line transcript summary of an epoch's validation; ``detail`` adds the recall by ground-truth depth."""
    main = results[primary]
    name = f"{label} " if label else ""
    text = (
        f"epoch {epoch} {name}val [{primary}, n={main.n}] {_goal_text(main.goal)}; "
        f"diagnostics: IoU {main.iou:.4f}, bf10 {main.bf10:.3f}"
    )
    if main.approved is not None:
        text += f"; approved (n={main.approved_n}): {_goal_text(main.approved)}"
    for name, goal, count in (("with_neg", main.with_neg, main.with_neg_n), ("no_neg", main.no_neg, main.no_neg_n)):
        if goal is not None and primary in NEG_SPLIT_PROMPTS:
            text += f"; {name} (n={count}): {_goal_text(goal)}"
    others = []
    for mode, result in results.items():
        if mode == primary:
            continue
        piece = f"{mode} core recall {result.goal.core_recall:.4f}, merge {result.goal.merge_rate:.3f}"
        if result.input_goal is not None:
            piece += f" (input {result.input_goal.core_recall:.4f}/{result.input_goal.merge_rate:.3f}"
            if mode == "near_correct" and result.no_harm is not None:
                piece += f", no harm {result.no_harm:.2f}"
            piece += ")"
        others.append(piece)
    if others:
        text += "; other prompts: " + "; ".join(others)
    if main.fulltile_iou is not None:
        text += f"; fully masked tiles ({main.fulltile_n}): IoU {main.fulltile_iou:.4f}"
    if detail and main.recall_by_depth:
        text += f"; mean per-sample core recall by depth, {_depth_text(main.recall_by_depth)}"
    return text
