"""Train entry: OmegaConf YAML plus Hydra-style key=value overrides.

@hydra.main cannot be used: importing `sam2` initializes GlobalHydra for SAM2's
config module, and a second Hydra initialize fails.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import sqlite3
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from sam2_segmentation_trainer.checkpoint import (
    atomic_torch_save,
    load_latest_checkpoint,
    migrate_run_artifacts,
    remaining_epoch_indices,
)
from sam2_segmentation_trainer.data.dataset import (
    EMSegDataset,
    EpochCounter,
    MissingTileFlag,
    collate_em,
)
from sam2_segmentation_trainer.data.approved import ApprovedSet, approved_counts, load_approved
from sam2_segmentation_trainer.data.prompts import PromptConfig
from sam2_segmentation_trainer.data.tiles import (
    TileContext,
    build_tile_context,
)
from sam2_segmentation_trainer.losses import containment_warmup_scale
from annotation_crops.negatives import DEFAULT_CIRCLE_DIAMETER_SCALE
from annotation_crops.scores import ensure_location_scores_schema

from sam2_segmentation_trainer.data.manifest import (
    ignore_list_stamps,
    ignore_lists_newer,
    index_volumes,
    load_ignore_ids,
    move_ignored_masks,
    take_model_sized,
    volume_counts,
)
from sam2_segmentation_trainer.data.fulltile import cap_full_tiles, is_full_tile
from sam2_segmentation_trainer.goal_metrics import EARLY_STOPPING_MIN_DELTA, MERGE_MIN_PX, SelectionKey, improves
from sam2_segmentation_trainer.interactive import iterative_predict_masks
from sam2_segmentation_trainer.validation import (
    DEFAULT_VAL_PROMPTS,
    ValidationSuite,
    collect_family_records,
    collect_records,
    epoch_selection,
    full_pass_due,
    log_results,
)
from sam2_segmentation_trainer.data.splits import (
    include_new_keys,
    load_split,
    make_location_split,
    make_zslab_split,
    save_split,
)
from sam2_segmentation_trainer.metrics import iou_dice_pr, mean_iou_by_volume
from sam2_segmentation_trainer.model import (
    build_em_sam2,
    combined_loss,
    configure_trainable_params,
    enable_activation_checkpointing,
    param_groups,
    predict_masks,
    predict_masks_mixed,
)
from sam2_segmentation_trainer.mqtt_progress import StepAnomalyMonitor, TrainingProgress
from sam2_segmentation_trainer.overlays import denorm_preview, overlay_panel
from sam2_segmentation_trainer.records import RunInputs, TrainerRoots, TrainSnapshot
from sam2_segmentation_trainer.paths import (
    annotation_crops_root,
    checkpoint_root,
    data_root,
    latest_resumable_run,
    legacy_run_dir,
    next_run_name,
    output_root,
    pretrained_checkpoint,
    run_dir_for,
)
from sam2_segmentation_trainer.tb_profiler import TbStepProfiler


class StopFlag:
    def __init__(self) -> None:
        self.stop = False

    def request(self, *_args) -> None:
        self.stop = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _resolve_paths(cfg: DictConfig) -> TrainerRoots:
    root = data_root(cfg.data.root)
    out = output_root(cfg.training.output_dir)
    ckpt_root_cfg = cfg.training.get("checkpoint_dir")
    ckpt_root = checkpoint_root(None if ckpt_root_cfg in (None, "") else ckpt_root_cfg)
    ckpt = cfg.model.checkpoint
    checkpoint = Path(ckpt) if ckpt else pretrained_checkpoint(out)
    return TrainerRoots(data=root, output=out, checkpoint=ckpt_root, pretrained=checkpoint)


def cosine_warmup_lambdas(optimizer, warmup_steps: int, total_steps: int, min_lr: float):
    lambdas = []
    for group in optimizer.param_groups:
        base_lr = group["lr"]

        def make_fn(base: float):
            min_ratio = min_lr / base if base > 0 else 0.0

            def fn(step: int) -> float:
                if step < warmup_steps:
                    return float(step + 1) / float(max(1, warmup_steps))
                progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                progress = min(1.0, max(0.0, progress))
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_ratio + (1.0 - min_ratio) * cosine

            return fn

        lambdas.append(make_fn(base_lr))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambdas)


def _amp_setup(device: torch.device, enabled: bool):
    use_cuda = device.type == "cuda"
    use_bf16 = bool(use_cuda and torch.cuda.is_bf16_supported())
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=bool(enabled and use_cuda and not use_bf16))
    return dtype, scaler


def enable_cudnn_benchmark(device: torch.device, enabled: bool = True) -> bool:
    """Let cuDNN pick convolution algorithms for a fixed tile size (always 1024×1024).

    ``training.cudnn_benchmark: false`` turns the autotuner off: it can pick an algorithm with a very large
    workspace for one batch shape (seen as a 21 GB peak at 3 prompt variants, 12 GB at 1, 2 and 4).
    """
    if device.type != "cuda" or not enabled:
        return False
    torch.backends.cudnn.benchmark = True
    return True


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds - hours * 3600 - minutes * 60
    if hours:
        return f"{hours}h {minutes}m {secs:.0f}s"
    if minutes:
        return f"{minutes}m {secs:.1f}s"
    return f"{secs:.2f}s"


def project_full_run(
    *,
    train_s: float,
    val_s: float,
    batches_per_epoch: int,
    n_val: int,
    num_epochs: int,
    val_every_n_epochs: int,
) -> dict[str, float]:
    val_every = max(1, int(val_every_n_epochs))
    val_runs = sum(1 for epoch in range(int(num_epochs)) if epoch % val_every == 0)
    train_epoch_s = float(train_s) * int(batches_per_epoch)
    val_pass_s = float(val_s) * int(n_val)
    return {
        "train_epoch_s": train_epoch_s,
        "val_pass_s": val_pass_s,
        "full_run_s": train_epoch_s * int(num_epochs) + val_pass_s * val_runs,
    }


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _log_cuda_mem(tag: str) -> None:
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(
        f"CUDA mem {tag}: {alloc:.2f}G allocated, "
        f"{reserved:.2f}G reserved, {total:.2f}G total"
    )


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    accelerator_err = getattr(torch, "AcceleratorError", None)
    if accelerator_err is not None and isinstance(exc, accelerator_err):
        return "out of memory" in str(exc).lower()
    text = str(exc).lower()
    # Under memory pressure some drivers report a failed mapping (seen as "CUDA driver error: device not ready")
    # instead of "out of memory".
    return isinstance(exc, RuntimeError) and ("out of memory" in text or "failed to create gpu mapping" in text)


def _summarize_times(times: list[float]) -> str:
    if not times:
        return "n/a"
    mean = statistics.fmean(times)
    median = statistics.median(times)
    return (
        f"{mean:.3f} s/it (median {median:.3f}, "
        f"min {min(times):.3f}, max {max(times):.3f}, n={len(times)})"
    )


def _time_validation_suite(
    model, val_suite: ValidationSuite, device, image_size: int, threshold: float, warmup: int, steps: int
) -> dict[str, tuple[float, float]]:
    """Seconds per validation sample of a subset pass and of a full pass of the real suite.

    Each pass scores the first ``warmup + steps`` samples with every family it runs; the batches holding the first
    ``warmup`` samples are not timed. A time is the interval between two batches reaching the scoring loop divided
    by the batch size, so it covers data loading, the one encoder pass, all the decodes and the metrics (which run
    in threads while the GPU works on the next batch). Returns ``{"subset": (seconds, metrics_wait), "full": ...}``
    per sample, where ``metrics_wait`` is the part the main thread spent waiting for the metrics.
    """
    result: dict[str, tuple[float, float]] = {}
    batch_size = val_suite.batch_size
    for name, full in (("subset", False), ("full", True)):
        dataset = val_suite.full_dataset if full else val_suite.subset_dataset
        take = min(len(dataset), max(2, warmup + steps))
        if take == 0:
            continue
        stamps: list[float] = []
        timings: dict[str, float] = {}
        base = val_suite.loader(full=full, limit=take)

        def stamped(batches=base, marks=stamps):
            for batch in batches:
                marks.append(time.perf_counter())
                yield batch

        collect_family_records(
            model,
            stamped(),
            device,
            image_size,
            threshold,
            core_band_px=val_suite.core_band_px,
            merge_min_px=val_suite.merge_min_px,
            metric_threads=val_suite.metric_threads,
            timings=timings,
        )
        stamps.append(time.perf_counter())
        batches = len(stamps) - 1
        skip = min(math.ceil(warmup / batch_size), max(0, batches - 1))
        intervals = [b - a for a, b in zip(stamps[skip:-1], stamps[skip + 1 :])]
        seconds = statistics.fmean(intervals) / batch_size if intervals else 0.0
        wait = timings.get("metrics_wait_s", 0.0) / max(1, take)
        result[name] = (seconds, wait)
    return result


def run_throughput_benchmark(
    *,
    model,
    train_ds,
    val_ds,
    device: torch.device,
    image_size: int,
    batch_size: int,
    num_workers: int,
    n_train: int,
    n_val: int,
    seed: int,
    batches_per_epoch: int,
    num_epochs: int,
    val_every_n_epochs: int,
    accum: int,
    amp_dtype,
    scaler,
    mixed_precision: bool,
    cfg: DictConfig,
    optimizer,
    scheduler,
    train_steps: int,
    val_steps: int,
    warmup_train: int = 10,
    warmup_val: int = 3,
    progress: TrainingProgress | None = None,
    val_suite: ValidationSuite | None = None,
    time_validation: bool = True,
) -> dict[str, float]:
    # Wall-clock includes DataLoader wait; that is the tqdm s/it the full run will see.
    model.train()
    optimizer.zero_grad(set_to_none=True)
    need_train = warmup_train + train_steps
    train_indices = remaining_epoch_indices(
        n_train, seed, 0, 0, batch_size, drop_last=True
    )
    if len(train_indices) < batch_size:
        raise RuntimeError("not enough train examples to benchmark a batch")
    if len(train_indices) < need_train * batch_size:
        need_train = max(1, len(train_indices) // batch_size)
        warmup_train = min(warmup_train, max(0, need_train // 5))
        train_steps = max(1, need_train - warmup_train)
        need_train = warmup_train + train_steps
    train_indices = train_indices[: need_train * batch_size]
    train_dl = _make_loader(
        train_ds,
        train_indices,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        prefetch_factor=int(cfg.training.prefetch_factor),
        persistent_workers=bool(cfg.training.persistent_workers),
    )

    def train_one(batch, step_i: int) -> float:
        images = batch["image"].to(device, non_blocking=True)
        points = batch["point"].to(device, non_blocking=True)
        labels = batch["point_label"].to(device, non_blocking=True)
        inputs = _loss_inputs(batch, device)
        with torch.amp.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=bool(mixed_precision) and device.type == "cuda",
        ):
            pred, iou_preds = predict_masks_mixed(
                model,
                images,
                points,
                labels,
                batch["boxes"],
                batch["dense_mask"],
                image_size=image_size,
                variant_index=inputs["variant_index"],
            )
            loss, stats = combined_loss(
                pred,
                inputs["masks"],
                iou_preds,
                focal_weight=float(cfg.loss.focal_weight),
                dice_weight=float(cfg.loss.dice_weight),
                iou_weight=float(cfg.loss.iou_weight),
                points=points,
                labels=labels,
                containment_weight=float(cfg.loss.get("containment_weight", 0.0)),
                containment_radius=float(cfg.loss.get("containment_radius", 8.0)),
                prompt_foreground=inputs["prompt_fg"],
                mask_contain_weight=_mask_contain_weight(cfg, 1.0),
                iou_use_l1=_iou_use_l1(cfg),
                edge_weight=inputs["edge_weight"],
                variant_index=inputs["variant_index"],
                num_samples=images.shape[0],
                sample_weight=inputs["sample_weight"],
            )
            loss = loss / accum
        scaler.scale(loss).backward()
        do_step = ((step_i + 1) % accum == 0) or ((step_i + 1) == need_train)
        if do_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                float(cfg.training.gradient_clip),
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        return float(stats["loss"])

    train_iter = iter(train_dl)
    try:
        for i in range(warmup_train):
            train_one(next(train_iter), i)
            _sync_device(device)

        train_times: list[float] = []
        wait_times: list[float] = []
        variant_counts: list[int] = []
        log_every = int(cfg.training.log_every_n_steps)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for j in range(train_steps):
            i = warmup_train + j
            t0 = time.perf_counter()
            batch = next(train_iter)  # time spent here is the GPU waiting for the DataLoader
            wait_times.append(time.perf_counter() - t0)
            variant_counts.append(int(batch["point"].shape[0]))
            step_loss = train_one(batch, i)
            _sync_device(device)
            train_times.append(time.perf_counter() - t0)
            if progress is not None:
                progress.report_step(
                    step=j,
                    batches_per_epoch=train_steps,
                    global_step=j,
                    log_every=log_every,
                    loss=step_loss,
                    lr=float(optimizer.param_groups[0]["lr"]),
                )
    except Exception as exc:
        if not _is_oom(exc):
            raise
        _log_cuda_mem("oom")
        raise RuntimeError(
            f"CUDA OOM during benchmark at batch_size={batch_size} image_size={image_size}. "
            "Retry with training.batch_size=2 or training.batch_size=1 "
            "(keep gradient_accumulation_steps so effective batch stays similar)."
        ) from exc

    val_times: list[float] = []
    need_val = warmup_val + val_steps
    # With a validation suite the real pass is timed below (its dataset carries one prompt variant per family, which
    # this single-prompt timing loop cannot decode); without one, time a plain point-prompt forward.
    n_val_take = 0 if (val_suite is not None or not time_validation) else min(len(val_ds), need_val)
    if n_val_take > 0:
        if n_val_take < need_val:
            warmup_val = min(warmup_val, max(0, n_val_take // 5))
            val_steps = max(1, n_val_take - warmup_val)
            n_val_take = warmup_val + val_steps
        val_dl = DataLoader(
            Subset(val_ds, list(range(n_val_take))),
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_em,
        )

        def val_one(batch) -> None:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            points = batch["point"].to(device, non_blocking=True)
            labels = batch["point_label"].to(device, non_blocking=True)
            pred, _iou = predict_masks(
                model, images, points, labels, image_size=image_size
            )
            pred_bin = (torch.sigmoid(pred) > float(cfg.data.mask_threshold)).to(
                torch.int64
            )
            for pred_i, mask_i in zip(pred_bin, masks, strict=True):
                iou_dice_pr(
                    pred_i.detach().cpu().numpy(),
                    mask_i.detach().cpu().numpy(),
                )

        model.eval()
        val_iter = iter(val_dl)
        with torch.no_grad():
            for _ in range(warmup_val):
                val_one(next(val_iter))
                _sync_device(device)
            for _ in range(val_steps):
                t0 = time.perf_counter()
                val_one(next(val_iter))
                _sync_device(device)
                val_times.append(time.perf_counter() - t0)
        model.train()

    train_s = statistics.fmean(train_times) if train_times else 0.0
    val_s = statistics.fmean(val_times) if val_times else 0.0
    suite_times: dict[str, tuple[float, float]] = {}
    if time_validation and val_suite is not None and len(val_suite) > 0:
        suite_times = _time_validation_suite(
            model, val_suite, device, image_size, float(cfg.data.mask_threshold), warmup_val, val_steps
        )
    projected = project_full_run(
        train_s=train_s,
        val_s=val_s,
        batches_per_epoch=batches_per_epoch,
        n_val=n_val,
        num_epochs=num_epochs,
        val_every_n_epochs=val_every_n_epochs,
    )
    if suite_times:
        # Every epoch validates the subset; every `full_every` epochs and after the last, the full split instead.
        full_every = max(1, int(OmegaConf.select(cfg, "val.full_every_n_epochs", default=3)))
        subset_pass_s = suite_times["subset"][0] * val_suite.subset_size
        full_pass_s = suite_times["full"][0] * len(val_suite)
        n_full = sum(1 for e in range(num_epochs) if full_pass_due(e, num_epochs, full_every))
        projected["val_pass_s"] = subset_pass_s
        projected["val_full_pass_s"] = full_pass_s
        projected["full_run_s"] = (
            projected["train_epoch_s"] * num_epochs + subset_pass_s * (num_epochs - n_full) + full_pass_s * n_full
        )
        projected["n_full_passes"] = float(n_full)
    gpu = (
        torch.cuda.get_device_name(0)
        if device.type == "cuda" and torch.cuda.is_available()
        else str(device)
    )
    print("Benchmark (no checkpoints written, last.pt not loaded)")
    print(f"  GPU: {gpu}")
    print(
        f"  timed {len(train_times)} train batches after {warmup_train} warmup; "
        + (
            f"{warmup_val + val_steps} validation samples per pass ({warmup_val} warmup)"
            if suite_times
            else "validation not timed"
            if not time_validation
            else f"{len(val_times)} val after {warmup_val} warmup"
        )
    )
    print(
        f"  train={n_train} val={n_val} batch_size={batch_size} image_size={image_size} "
        f"batches/epoch={batches_per_epoch} epochs={num_epochs}"
    )
    print(f"  train {_summarize_times(train_times)}")
    if wait_times:
        wait_s = statistics.fmean(wait_times)
        share = wait_s / statistics.fmean(train_times) if train_times else 0.0
        print(
            f"  data wait {_summarize_times(wait_times)}; {share:.0%} of the step "
            f"({num_workers} workers, prefetch {int(cfg.training.prefetch_factor)}; "
            f"{statistics.fmean(variant_counts) / batch_size:.1f} prompt variants per sample)"
        )
        print(
            "    -> data-bound: more workers or cheaper prompts help"
            if share > 0.2
            else "    -> GPU-bound: the DataLoader keeps up"
        )
    if device.type == "cuda" and torch.cuda.is_available():
        print(
            f"  peak CUDA memory in the timed train steps: "
            f"{torch.cuda.max_memory_allocated() / 1024**3:.2f}G allocated, "
            f"{torch.cuda.max_memory_reserved() / 1024**3:.2f}G reserved, "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}G total"
        )
    if val_times:
        print(f"  val   {_summarize_times(val_times)}  (one family, point prompt, no goal metrics)")
    if suite_times:
        for name in ("subset", "full"):
            if name in suite_times:
                seconds, wait = suite_times[name]
                families = val_suite.subset_modes if name == "subset" else val_suite.modes
                size = val_suite.subset_size if name == "subset" else len(val_suite)
                print(
                    f"  val {name:6s} {seconds:.3f} s/sample ({len(families)} families, one encoder pass, batch "
                    f"{val_suite.batch_size}, {val_suite.metric_threads} metric threads; waited {wait:.3f} s/sample "
                    f"for the metrics) x {size} samples"
                )
    print(f"  estimated train epoch: {format_duration(projected['train_epoch_s'])}")
    if suite_times:
        print(f"  estimated val pass, subset (every epoch):       {format_duration(projected['val_pass_s'])}")
        print(
            f"  estimated val pass, full ({int(projected['n_full_passes'])} of {num_epochs} epochs):  "
            f"{format_duration(projected['val_full_pass_s'])}"
        )
    elif time_validation:
        print(f"  estimated val pass:    {format_duration(projected['val_pass_s'])}")
    estimate = (
        f"  estimated {num_epochs}-epoch run: "
        f"{format_duration(projected['full_run_s'])} (no early stopping)"
    )
    if progress is not None:
        progress.transcript(estimate)
    else:
        print(estimate)
    return projected


def _mixed_prompts_enabled(cfg: DictConfig) -> bool:
    """Whether training uses the mixed prompt mix (``prompt.enabled``, default on) or the single centroid point."""
    return bool(OmegaConf.select(cfg, "prompt.enabled", default=True))


def _circle_scale(cfg: DictConfig) -> float:
    """Diameter scale of the circles drawn as prompts and negatives (``prompt.circle_diameter_scale``).

    The exporter writes true radii; the loader rescales each file to this value.
    """
    return float(OmegaConf.select(cfg, "prompt.circle_diameter_scale", default=DEFAULT_CIRCLE_DIAMETER_SCALE))


def _circle_guard(cfg: DictConfig, prompt_cfg: PromptConfig) -> dict:
    """Keyword arguments for :func:`build_tile_context`: the circle scale, and a failure when circles are absent."""
    return {
        "require_circles": prompt_cfg.prob_circle_seed > 0.0
        and bool(OmegaConf.select(cfg, "data.require_circle_file", default=True)),
        "circle_scale": _circle_scale(cfg),
    }


def prompt_mix_fractions(counts: Counter[str]) -> dict[str, float]:
    """Share of each prompt type among the samples of one epoch, largest first."""
    total = sum(counts.values())
    if total == 0:
        return {}
    return {ptype: count / total for ptype, count in counts.most_common()}


def _log_prompt_mix(writer: SummaryWriter, epoch: int, counts: Counter[str]) -> None:
    """Print and log the realized prompt mix (share and count per family), so a missing circle file or a dropped
    type shows at once. Variant 0 of every sample is fixed, so this differs from the ``prob_*`` shares."""
    fractions = prompt_mix_fractions(counts)
    if not fractions:
        return
    summary = ", ".join(f"{ptype} {share:.1%}" for ptype, share in fractions.items())
    print(f"epoch {epoch} prompt mix ({sum(counts.values())} samples): {summary}")
    for ptype, share in fractions.items():
        writer.add_scalar(f"train/prompt_mix/{ptype}", share, epoch)
        writer.add_scalar(f"train/prompt_count/{ptype}", counts[ptype], epoch)


def _iterative_rounds(cfg: DictConfig, seed: int, epoch: int, step: int) -> int:
    """Corrective-click rounds for this batch: 0 for an ordinary step, else a draw from ``iterative.rounds_min..max``.

    ``iterative.batch_fraction`` of the batches take the iterative path. The draw is seeded from
    ``(seed, epoch, step)``, so a resumed epoch repeats it.
    """
    fraction = float(OmegaConf.select(cfg, "iterative.batch_fraction", default=0.0))
    if fraction <= 0.0:
        return 0
    low = max(1, int(OmegaConf.select(cfg, "iterative.rounds_min", default=2)))
    high = max(low, int(OmegaConf.select(cfg, "iterative.rounds_max", default=6)))
    rng = np.random.default_rng([int(seed), int(epoch), int(step), 8])
    if rng.random() >= fraction:
        return 0
    return int(rng.integers(low, high + 1))


def _full_tile_settings(cfg: DictConfig) -> tuple[float | None, float]:
    """``(threshold, keep_fraction)`` for fully masked tiles; a null threshold turns the handling off."""
    threshold = OmegaConf.select(cfg, "data.full_tile_threshold", default=0.99)
    keep = float(OmegaConf.select(cfg, "data.full_tile_keep_fraction", default=0.25))
    return (None if threshold is None else float(threshold)), keep


def _cap_full_tiles(train_ex: list, cfg: DictConfig) -> list:
    """Keep a deterministic fraction of the training tiles that the target fills completely."""
    threshold, keep = _full_tile_settings(cfg)
    capped = cap_full_tiles(train_ex, threshold=threshold, keep_fraction=keep, seed=int(cfg.training.seed))
    dropped = len(train_ex) - len(capped)
    if dropped:
        print(f"fully masked tiles: kept {sum(is_full_tile(e, threshold) for e in capped)}, dropped {dropped} from training")
    return capped


def _iou_use_l1(cfg: DictConfig) -> bool:
    """SAM 2 trains the IoU head with L1 (``iou_use_l1_loss``); ``loss.iou_use_l1: false`` restores MSE."""
    return bool(OmegaConf.select(cfg, "loss.iou_use_l1", default=True))


def _edge_loss_settings(cfg: DictConfig) -> tuple[float, float]:
    """``(loss.edge_tau_px, loss.edge_weight_min)``: tau of the edge-tolerant loss weight (0 disables it) and its floor."""
    tau = float(OmegaConf.select(cfg, "loss.edge_tau_px", default=8.0))
    return tau, float(OmegaConf.select(cfg, "loss.edge_weight_min", default=0.05))


def _tau_depth_frac(cfg: DictConfig) -> float:
    """``prompt.edge_tau_depth_frac``: the thin-process cap of tau, shared by the prompt encoding and the loss weight."""
    return float(OmegaConf.select(cfg, "prompt.edge_tau_depth_frac", default=0.25))


def _leak_settings(cfg: DictConfig) -> tuple[float, float]:
    """``(loss.leak_weight, val.core_band_px)``: the weight on neighbor cores (1 disables) and how deep inside a
    neighbor a pixel must be to count as its core."""
    return (
        float(OmegaConf.select(cfg, "loss.leak_weight", default=3.0)),
        float(OmegaConf.select(cfg, "val.core_band_px", default=10.0)),
    )


def _tolerance_tau(cfg: DictConfig) -> float:
    """Tau of the band the tolerant IoU ignores. It follows the loss weight, or 8 px when edge weighting is off."""
    tau = _edge_loss_settings(cfg)[0]
    return tau if tau > 0.0 else 8.0


def load_approved_set(cfg: DictConfig, data_dir: Path | None = None) -> ApprovedSet:
    """The gallery's approved masks (see :mod:`sam2_segmentation_trainer.data.approved`).

    ``data.approved_path`` null reads ``{data_root}/{volume}/approved.json`` for every configured volume; it may
    also be a file, a directory, or a list of those. Counts are logged per volume, and a total of 0 warns.
    """
    raw = OmegaConf.select(cfg, "data.approved_path", default=None)
    if raw is not None and not isinstance(raw, str):
        raw = [str(item) for item in raw]
    return load_approved(raw, [str(v) for v in cfg.data.volumes], data_dir if data_dir is not None else data_root())


def _loss_inputs(batch: dict, device) -> dict:
    """Device tensors shared by every loss call: per-variant targets and weights, and the variant index."""
    index = batch["variant_index"].to(device, non_blocking=True)
    edge = batch.get("edge_weight")
    return {
        "variant_index": index,
        "masks": batch["mask"].to(device, non_blocking=True)[index],
        "edge_weight": None if edge is None else edge.to(device, non_blocking=True)[index],
        "prompt_fg": batch["prompt_fg"].to(device, non_blocking=True),
        "sample_weight": batch["sample_weight"].to(device, non_blocking=True),
    }


def _mask_contain_weight(cfg: DictConfig, scale: float) -> float:
    """Configured mask-containment weight times the warmup ``scale`` (0..1)."""
    return float(OmegaConf.select(cfg, "loss.mask_contain_weight", default=0.0)) * float(scale)


def _make_loader(
    dataset,
    indices: list[int],
    *,
    batch_size: int,
    num_workers: int,
    drop_last: bool,
    prefetch_factor: int,
    persistent_workers: bool,
) -> DataLoader:
    subset = Subset(dataset, indices) if indices is not None else dataset
    # DataLoader rejects these kwargs when the loading process is the main process.
    worker_kwargs: dict = {}
    if num_workers > 0:
        worker_kwargs["prefetch_factor"] = int(prefetch_factor)
        worker_kwargs["persistent_workers"] = bool(persistent_workers)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=collate_em,
        **worker_kwargs,
    )


def due_for_val_images(global_step: int, last_step: int | None, every_n_steps: int) -> bool:
    """Whether this validation pass should write TensorBoard overlay images.

    The first pass of a process logs. Later passes log once at least
    ``every_n_steps`` training steps have elapsed. ``every_n_steps <= 0``
    leaves images off.
    """
    if every_n_steps <= 0:
        return False
    if last_step is None:
        return True
    return int(global_step) - int(last_step) >= int(every_n_steps)


@torch.no_grad()
def run_validation(
    model,
    loader,
    device,
    image_size: int,
    threshold: float,
    *,
    overlays_per_volume: int = 0,
) -> tuple[float, dict[str, float], list[tuple[str, np.ndarray]]]:
    """Mean IoU, per-volume means, and a few overlay strips.

    Volume names come from the examples in the loader. ``overlays_per_volume``
    keeps that many strips per name (input | ground truth | prediction) from
    the start of the loader. The prediction is the mask decoder output after
    ``sigmoid > threshold`` (0.5 in the default config).
    """
    records, overlays = collect_records(
        model, loader, device, image_size, threshold, overlays_per_volume=overlays_per_volume
    )
    overall, by_volume = mean_iou_by_volume([(r["volume"], r["iou"]) for r in records])
    return overall, by_volume, overlays


class _LossWindow:
    """Detached GPU losses, copied to the CPU in one transfer when drained.

    Besides the four always-present losses, the optional prompt-containment and mask-containment losses are
    carried when ``stats`` has them. The per-sample losses ride along with each sample's prompt type so the
    log can show which prompt families are hard.
    """

    _OPTIONAL = ("loss_containment", "loss_mask_contain")

    def __init__(self) -> None:
        self._rows: list[torch.Tensor] = []
        self._meta: list[tuple[int, int, int, float, list, list[str]]] = []
        self._extras: tuple[str, ...] = ()

    def add(
        self,
        stats: dict[str, torch.Tensor],
        *,
        epoch: int,
        step: int,
        global_step: int,
        duration_s: float,
        meta: list,
        prompt_types: list[str] | None = None,
    ) -> None:
        self._extras = tuple(key for key in self._OPTIONAL if key in stats)
        head = [
            stats["loss"].reshape(1),
            stats["loss_focal"].reshape(1),
            stats["loss_dice"].reshape(1),
            stats["loss_iou"].reshape(1),
        ]
        head.extend(stats[key].reshape(1) for key in self._extras)
        packed = torch.cat(head + [stats["sample_losses"].reshape(-1)])
        self._rows.append(packed.detach())
        self._meta.append((epoch, step, global_step, duration_s, meta, list(prompt_types or [])))

    def drain(self) -> list[dict]:
        if not self._rows:
            return []
        extras = self._extras
        head_len = 4 + len(extras)
        packed = torch.stack(self._rows).float().cpu()
        records = []
        for i, (epoch, step, global_step, duration_s, meta, prompt_types) in enumerate(self._meta):
            row = packed[i].tolist()
            record = {
                "epoch": epoch,
                "step": step,
                "global_step": global_step,
                "duration_s": duration_s,
                "meta": meta,
                "prompt_types": prompt_types,
                "loss": row[0],
                "loss_focal": row[1],
                "loss_dice": row[2],
                "loss_iou": row[3],
                "sample_losses": row[head_len:],
            }
            for offset, key in enumerate(extras):
                record[key] = row[4 + offset]
            records.append(record)
        self._rows.clear()
        self._meta.clear()
        return records


def _ensure_score_schema(connection: sqlite3.Connection) -> None:
    """Create per-window scores, or keep a location-only table under an empty image key."""
    ensure_location_scores_schema(connection)


def record_location_scores(data_dir: Path, rows: list[dict]) -> None:
    """Upsert one combined loss per mask image per epoch into each volume catalog.

    A sample with no image key is skipped so a run cannot write another
    location-wide row. The same window twice in one epoch keeps the later score.
    """
    by_volume: dict[str, list[tuple[int, str, int, float]]] = {}
    for row in rows:
        meta = row.get("meta") or []
        sample_losses = row.get("sample_losses") or []
        epoch = int(row["epoch"])
        for item, score in zip(meta, sample_losses, strict=False):
            volume = item.get("volume")
            location_id = item.get("location_id")
            image_key = item.get("image_key")
            if volume is None or location_id is None or not image_key:
                continue
            by_volume.setdefault(str(volume), []).append(
                (int(location_id), str(image_key), epoch, float(score))
            )
    for volume, scores in by_volume.items():
        path = annotation_crops_root(volume, data_dir) / "annotation_crops.sqlite"
        if not path.is_file():
            continue
        connection = sqlite3.connect(str(path), timeout=5.0)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA busy_timeout=5000")
            _ensure_score_schema(connection)
            connection.executemany(
                "INSERT OR REPLACE INTO location_scores "
                "(location_id, image_key, epoch, score) VALUES (?, ?, ?, ?)",
                scores,
            )
            connection.commit()
        except sqlite3.Error as exc:
            print(f"location_scores write skipped for {volume}: {exc}")
        finally:
            connection.close()


def _snapshot(
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    step_in_epoch: int,
    global_step: int,
    opt_step: int,
    best_val_iou: float,
    epochs_without_improve: int,
    freeze_mode: str,
    unfreeze_mask_cnn: bool = False,
    best_selection: list | None = None,
) -> dict:
    return TrainSnapshot(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=epoch,
        step_in_epoch=step_in_epoch,
        global_step=global_step,
        opt_step=opt_step,
        best_val_iou=best_val_iou,
        epochs_without_improve=epochs_without_improve,
        freeze_mode=freeze_mode,
        rng=capture_rng(),
        unfreeze_mask_cnn=unfreeze_mask_cnn,
        best_selection=best_selection,
    ).as_checkpoint()


def _save_last(run_dir: Path, payload: dict) -> None:
    atomic_torch_save(payload, run_dir / "last.pt")


def _resolve_run_name(cfg: DictConfig, ckpt_root: Path, out_dir: Path) -> str:
    raw = cfg.training.get("run_name")
    name = "" if raw is None else str(raw).strip()
    if name and name not in ("auto", "None"):
        return name
    roots = [ckpt_root, out_dir]
    if bool(cfg.training.get("resume", True)):
        latest = latest_resumable_run(roots)
        if latest:
            print(f"resuming latest run {latest}")
            return latest
    prefix = str(cfg.training.get("run_name_prefix", "sam2_em"))
    return next_run_name(prefix, roots)


def _z_runs(zs) -> str:
    """Compact text for a set of section numbers: ``"10-14, 40-43"``."""
    ordered = sorted(int(z) for z in zs)
    if not ordered:
        return "none"
    runs: list[str] = []
    start = prev = ordered[0]
    for z in ordered[1:] + [None]:
        if z is not None and z == prev + 1:
            prev = z
            continue
        runs.append(f"{start}-{prev}" if start != prev else f"{start}")
        if z is not None:
            start = prev = z
    return ", ".join(runs)


def _counts_by_volume(examples) -> dict[str, int]:
    return volume_counts(examples)


def _move_ignored_masks(volumes: list[str], data_dir: Path) -> None:
    """Move rejected masks before a training index. A read-only index does not."""
    for volume in volumes:
        crops = annotation_crops_root(volume, data_dir)
        moved = move_ignored_masks(crops, load_ignore_ids(crops))
        if moved:
            print(f"index {volume}: {moved} ignored mask(s) moved")


def _write_run_inputs(path: Path, inputs: RunInputs) -> None:
    if path.is_file():
        return
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    path.write_text(json.dumps(inputs.as_json(created_utc=created), indent=2) + "\n", encoding="utf-8")


def train_from_cfg(cfg: DictConfig) -> Path:
    """Fine-tune SAM2 and publish progress when the Nornir MQTT helpers are installed."""
    progress = TrainingProgress()
    try:
        return _run_training(cfg, progress)
    except BaseException as exc:
        if progress.status == "completed":
            if isinstance(exc, KeyboardInterrupt):
                progress.mark_stopped()
                progress.transcript("training interrupted")
            else:
                progress.mark_failed(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        progress.close()


def _run_training(cfg: DictConfig, progress: TrainingProgress) -> Path:
    set_seed(int(cfg.training.seed))
    roots = _resolve_paths(cfg)
    data_dir, out_dir, ckpt_root, checkpoint = (
        roots.data,
        roots.output,
        roots.checkpoint,
        roots.pretrained,
    )
    run_name = _resolve_run_name(cfg, ckpt_root, out_dir)
    cfg.training.run_name = run_name
    run_dir = run_dir_for(run_name, ckpt_root=ckpt_root)
    legacy = legacy_run_dir(run_name, out_root=out_dir)
    run_existed = run_dir.is_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    progress.start(run_name=run_name, volumepath=str(data_dir))
    migrated = migrate_run_artifacts(legacy, run_dir)
    if migrated:
        print(f"migrated from {legacy} -> {run_dir}: {', '.join(migrated)}")
    print(f"run directory (checkpoints): {run_dir}")
    OmegaConf.save(cfg, run_dir / "config.yaml")

    volumes = list(cfg.data.volumes)
    image_size = int(cfg.data.image_size)
    _move_ignored_masks(volumes, data_dir)
    examples = index_volumes(volumes=volumes, root=data_dir, skip_missing=True)
    ignore_stamps = ignore_list_stamps(volumes, data_dir)
    missing_tiles = MissingTileFlag()
    examples, rejected_tiles = take_model_sized(examples, image_size)
    for ex in rejected_tiles:
        recorded = (
            "missing"
            if ex.width is None or ex.height is None
            else f"{ex.width}x{ex.height}"
        )
        progress.log_error(
            f"skip tile {ex.volume}/{ex.image_key} size {recorded}, "
            f"expected {image_size}x{image_size}"
        )
    if rejected_tiles:
        print(
            f"skipped {len(rejected_tiles)} tiles whose size metadata is not "
            f"{image_size}x{image_size}"
        )
    if not examples:
        raise RuntimeError(f"no usable AnnotationCrops examples under {data_dir} {volumes}")

    mixed_prompts = _mixed_prompts_enabled(cfg)
    prompt_cfg = PromptConfig.from_cfg(cfg)
    tile_context: TileContext | None = None
    if mixed_prompts:
        tile_context = build_tile_context(examples, **_circle_guard(cfg, prompt_cfg))
        n_with_negatives = sum(
            1
            for ex in examples
            if len(tile_context.neighbor_ids((ex.volume, ex.image_key))) > 1
            or tile_context.circles_in((ex.volume, ex.image_key))
        )
        print(
            f"mixed prompts on: {n_with_negatives}/{len(examples)} examples have other "
            "annotations in their tile"
        )

    even = bool(cfg.data.get("even_per_volume", False))
    run_split = run_dir / "split.json"
    legacy_split = out_dir / "splits" / f"{cfg.data.split_name}.json"
    if run_split.is_file():
        split = load_split(run_split)
        split_path = run_split
        even_applied = False
    elif run_existed and legacy_split.is_file():
        split = load_split(legacy_split)
        split_path = run_split
        save_split(split, run_split)
        even_applied = False
        print(f"reused existing split {legacy_split}")
    else:
        missing = [volume for volume in volumes if not any(ex.volume == volume for ex in examples)]
        if even and missing:
            raise RuntimeError(
                f"even split requires every volume to have usable examples; missing {missing}"
            )
        split_mode = str(cfg.data.get("split_mode", "location")).strip().lower()
        if split_mode == "zslab":
            zslab = cfg.data.get("zslab") or {}
            if even:
                print("data.even_per_volume is ignored for split_mode=zslab")
            split = make_zslab_split(
                examples,
                val_fraction=1.0 - float(cfg.data.train_split),
                slab_count=int(zslab.get("slab_count", 5)),
                guard_sections=int(zslab.get("guard_sections", 1)),
            )
            even = False
            for volume in sorted(split.val_z):
                print(
                    f"  {volume}: val sections {_z_runs(split.val_z[volume])}; "
                    f"guard {_z_runs(split.guard_z.get(volume, ()))}"
                )
        elif split_mode == "location":
            split = make_location_split(
                examples,
                train_fraction=float(cfg.data.train_split),
                seed=int(cfg.training.seed),
                even_per_volume=even,
            )
        else:
            raise ValueError(f"data.split_mode must be zslab or location, got {split_mode!r}")
        split_path = run_split
        save_split(split, run_split)
        even_applied = even
        print(f"wrote split {run_split} mode={split.mode} even_per_volume={even}")
    train_ex, val_ex = split.partition(examples)
    train_ex = _cap_full_tiles(train_ex, cfg)
    if not train_ex or not val_ex:
        raise RuntimeError(
            f"empty split: train={len(train_ex)} val={len(val_ex)} (file {split_path})"
        )
    train_counts = _counts_by_volume(train_ex)
    val_counts = _counts_by_volume(val_ex)
    print(f"train per volume: {train_counts}")
    print(f"val per volume: {val_counts}")
    _write_run_inputs(
        run_dir / "inputs.json",
        RunInputs(
            run_name=run_name,
            data_dir=data_dir,
            volumes=volumes,
            even_per_volume=even_applied,
            split_path=split_path,
            usable_counts=_counts_by_volume(examples),
            train_counts=train_counts,
            val_counts=val_counts,
            seed=int(cfg.training.seed),
            train_fraction=float(cfg.data.train_split),
            image_size=int(cfg.data.image_size),
            batch_size=int(cfg.training.batch_size),
            num_epochs=int(cfg.training.num_epochs),
            model_config=str(cfg.model.config),
            pretrained=checkpoint,
        ),
    )

    approved = load_approved_set(cfg, data_dir)
    core_recall_target = float(OmegaConf.select(cfg, "val.core_recall_target", default=0.975))
    if approved:
        print(
            f"approved samples: train per volume {approved_counts(train_ex, approved)}; "
            f"val per volume {approved_counts(val_ex, approved)}"
        )
    batch_size = int(cfg.training.batch_size)
    num_workers = int(cfg.training.num_workers)
    prefetch_factor = int(cfg.training.prefetch_factor)
    persistent_workers = bool(cfg.training.persistent_workers)
    seed = int(cfg.training.seed)
    freeze_mode = str(cfg.model.freeze_mode)
    epoch_counter = EpochCounter()
    train_ds = EMSegDataset(
        train_ex,
        split="train",
        image_size=image_size,
        missing_tiles=missing_tiles,
        tiles=tile_context,
        prompt_cfg=prompt_cfg if mixed_prompts else None,
        seed=seed,
        epoch=epoch_counter,
        full_tile_threshold=_full_tile_settings(cfg)[0],
        edge_tau_px=_edge_loss_settings(cfg)[0],
        edge_weight_min=_edge_loss_settings(cfg)[1],
        edge_tau_depth_frac=_tau_depth_frac(cfg),
        leak_weight=_leak_settings(cfg)[0],
        leak_core_px=_leak_settings(cfg)[1],
        approved=approved,
        approved_weight=float(OmegaConf.select(cfg, "data.approved_weight", default=1.0)),
    )
    val_modes = [str(m) for m in OmegaConf.select(cfg, "eval.val_prompts", default=list(DEFAULT_VAL_PROMPTS))]
    val_primary = str(OmegaConf.select(cfg, "eval.primary_val_prompt", default=val_modes[0]))
    val_suite = ValidationSuite(
        val_ex,
        modes=val_modes,
        primary=val_primary,
        image_size=image_size,
        num_workers=num_workers,
        missing_tiles=missing_tiles,
        tiles=tile_context,
        prompt_cfg=prompt_cfg if mixed_prompts else None,
        seed=seed,
        circle_scale=_circle_scale(cfg),
        full_tile_threshold=_full_tile_settings(cfg)[0],
        edge_tau_px=_tolerance_tau(cfg),
        edge_weight_min=_edge_loss_settings(cfg)[1],
        edge_tau_depth_frac=_tau_depth_frac(cfg),
        core_band_px=_leak_settings(cfg)[1],
        merge_min_px=int(OmegaConf.select(cfg, "val.merge_min_px", default=MERGE_MIN_PX)),
        core_recall_target=core_recall_target,
        approved=approved,
        epoch_subset=float(OmegaConf.select(cfg, "val.epoch_subset", default=0.25)),
        batch_size=int(OmegaConf.select(cfg, "val.batch_size", default=4)),
        metric_threads=int(OmegaConf.select(cfg, "val.metric_threads", default=4)),
    )
    val_ds = val_suite.dataset
    print(
        f"validation: every epoch the fixed subset of {val_suite.subset_size} of {len(val_ds)} samples "
        f"({', '.join(val_suite.subset_modes)}); the full split ({', '.join(val_suite.modes)}) every "
        f"{int(OmegaConf.select(cfg, 'val.full_every_n_epochs', default=3))} epochs and after the last"
    )
    print(f"validation prompts: {val_modes} (primary {val_primary})")

    n_train = len(train_ds)
    batches_per_epoch = max(1, n_train // batch_size)
    accum = max(1, int(cfg.training.gradient_accumulation_steps))
    steps_per_epoch = max(1, batches_per_epoch // accum)
    total_steps = steps_per_epoch * int(cfg.training.num_epochs)
    warmup_steps = steps_per_epoch * int(cfg.lr_schedule.warmup_epochs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if enable_cudnn_benchmark(device, bool(cfg.training.get("cudnn_benchmark", True))):
        print("cuDNN benchmark on (fixed image size)")
    else:
        print("cuDNN benchmark off")
    model = build_em_sam2(cfg.model.config, str(checkpoint), device, train=True)
    unfreeze_mask_cnn = mixed_prompts and bool(
        OmegaConf.select(cfg, "model.unfreeze_mask_cnn", default=True)
    )
    configure_trainable_params(
        model, freeze_mode=freeze_mode, unfreeze_mask_cnn=unfreeze_mask_cnn
    )
    if bool(cfg.model.get("activation_checkpointing", True)):
        fraction = float(cfg.model.get("activation_checkpointing_fraction", 1.0))
        n_ckpt = enable_activation_checkpointing(model, fraction)
        print(f"Activation checkpointing on {n_ckpt} Hiera blocks (fraction {fraction:g})")
    else:
        print("Activation checkpointing off")
    _log_cuda_mem("after model")

    optimizer = AdamW(
        param_groups(
            model,
            base_lr=float(cfg.optimizer.lr),
            encoder_lr_factor=float(cfg.differential_lr.encoder_lr_factor),
        ),
        weight_decay=float(cfg.optimizer.weight_decay),
        betas=tuple(cfg.optimizer.betas),
    )
    scheduler = cosine_warmup_lambdas(
        optimizer, warmup_steps, total_steps, float(cfg.lr_schedule.min_lr)
    )
    amp_dtype, scaler = _amp_setup(device, bool(cfg.training.mixed_precision))
    if bool(cfg.training.get("benchmark", False)):
        run_throughput_benchmark(
            val_suite=val_suite,
            model=model,
            train_ds=train_ds,
            val_ds=val_ds,
            device=device,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            n_train=n_train,
            n_val=len(val_ds),
            seed=seed,
            batches_per_epoch=batches_per_epoch,
            num_epochs=int(cfg.training.num_epochs),
            val_every_n_epochs=int(cfg.training.val_every_n_epochs),
            accum=accum,
            amp_dtype=amp_dtype,
            scaler=scaler,
            mixed_precision=bool(cfg.training.mixed_precision),
            cfg=cfg,
            optimizer=optimizer,
            scheduler=scheduler,
            train_steps=max(1, int(cfg.training.get("benchmark_steps", 50))),
            val_steps=max(1, int(cfg.training.get("benchmark_val_steps", 20))),
            progress=progress,
            time_validation=bool(cfg.training.get("benchmark_val", True)),
        )
        print(f"Run directory: {run_dir} (benchmark only; no last.pt written)")
        return run_dir

    writer = SummaryWriter(log_dir=str(run_dir / "tb"))
    profiler = TbStepProfiler(
        cfg,
        run_dir / "tb",
        device,
        worker_name=run_name,
    )
    best_val_iou = -1.0
    best_selection = None  # SelectionKey values of the best epoch so far
    epochs_without_improve = 0
    global_step = 0
    opt_step = 0
    start_epoch = 0
    start_step = 0
    patience = int(cfg.training.early_stopping_patience)
    min_delta = float(cfg.training.get("early_stopping_min_delta", EARLY_STOPPING_MIN_DELTA))
    full_every = max(1, int(OmegaConf.select(cfg, "val.full_every_n_epochs", default=3)))
    save_every = max(1, int(cfg.training.get("save_every_n_steps", 200)))
    resume_enabled = bool(cfg.training.get("resume", True))

    payload = None
    if resume_enabled:
        payload = load_latest_checkpoint(run_dir / "last.pt", map_location="cpu")
    if payload is not None:
        saved_freeze = str(payload.get("freeze_mode", freeze_mode))
        saved_groups = int(payload.get("n_param_groups", len(optimizer.param_groups)))
        if saved_freeze != freeze_mode:
            raise RuntimeError(
                f"cannot resume: freeze_mode {saved_freeze!r} != {freeze_mode!r} "
                "(pass -refresh to start from scratch)"
            )
        saved_mask_cnn = bool(payload.get("unfreeze_mask_cnn", False))
        if saved_mask_cnn != unfreeze_mask_cnn:
            raise RuntimeError(
                f"cannot resume: unfreeze_mask_cnn {saved_mask_cnn} != {unfreeze_mask_cnn} "
                "(the optimizer's parameters differ; pass -refresh to start from scratch)"
            )
        if saved_groups != len(optimizer.param_groups):
            raise RuntimeError(
                f"cannot resume: optimizer param groups {saved_groups} != "
                f"{len(optimizer.param_groups)} (pass -refresh to start from scratch)"
            )
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        scaler.load_state_dict(payload["scaler"])
        start_epoch = int(payload["epoch"])
        start_step = int(payload["step_in_epoch"])
        global_step = int(payload["global_step"])
        opt_step = int(payload["opt_step"])
        best_val_iou = float(payload["best_val_iou"])
        saved_selection = payload.get("best_selection")
        best_selection = None if saved_selection is None else SelectionKey.from_list(saved_selection)
        epochs_without_improve = int(payload["epochs_without_improve"])
        restore_rng(payload["rng"])
        print(
            f"resumed {run_dir / 'last.pt'} at epoch {start_epoch} "
            f"step {start_step} (opt_step {opt_step})"
        )

    stop = StopFlag()
    signal.signal(signal.SIGINT, stop.request)
    signal.signal(signal.SIGTERM, stop.request)

    def save_now(epoch: int, step_in_epoch: int) -> None:
        _save_last(
            run_dir,
            _snapshot(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                step_in_epoch=step_in_epoch,
                global_step=global_step,
                opt_step=opt_step,
                best_val_iou=best_val_iou,
                epochs_without_improve=epochs_without_improve,
                freeze_mode=freeze_mode,
                unfreeze_mask_cnn=unfreeze_mask_cnn,
                best_selection=None if best_selection is None else best_selection.to_list(),
            ),
        )

    num_epochs = int(cfg.training.num_epochs)
    val_images_every = int(cfg.training.get("val_images_every_n_steps", 2500))
    val_images_per_volume = max(0, int(cfg.training.get("val_images_per_volume", 3)))
    last_val_image_step: int | None = None
    anomaly = StepAnomalyMonitor(
        warmup=int(OmegaConf.select(cfg, "training.warn_anomaly_warmup", default=30)),
        window=int(OmegaConf.select(cfg, "training.warn_anomaly_window", default=100)),
        slow_factor=float(OmegaConf.select(cfg, "training.warn_slow_factor", default=2.5)),
        slow_seconds=float(OmegaConf.select(cfg, "training.warn_slow_seconds", default=3.0)),
        high_loss_factor=float(
            OmegaConf.select(cfg, "training.warn_high_loss_factor", default=5.0)
        ),
        high_loss_absolute=float(
            OmegaConf.select(cfg, "training.warn_high_loss_absolute", default=3.0)
        ),
    )
    for epoch in range(start_epoch, num_epochs):
        stamps = ignore_list_stamps(volumes, data_dir)
        if missing_tiles.take() or ignore_lists_newer(ignore_stamps, stamps):
            _move_ignored_masks(volumes, data_dir)
            examples = index_volumes(volumes=volumes, root=data_dir, skip_missing=True)
            examples, rejected_tiles = take_model_sized(examples, image_size)
            if mixed_prompts:
                tile_context = build_tile_context(examples, **_circle_guard(cfg, prompt_cfg))
                train_ds.tiles = tile_context
            grown = include_new_keys(
                split,
                examples,
                train_fraction=float(cfg.data.train_split),
                seed=seed,
            )
            if grown != split:
                split = grown
                save_split(split, run_split)
            train_ex, val_ex = split.partition(examples)
            train_ex = _cap_full_tiles(train_ex, cfg)
            if not train_ex:
                raise RuntimeError(
                    f"inventory refresh left no training examples under {data_dir}"
                )
            train_ds.replace_examples(train_ex)
            val_suite.replace_examples(val_ex)
            if mixed_prompts:
                val_suite.set_tiles(tile_context)
            n_train = len(train_ds)
            batches_per_epoch = max(1, n_train // batch_size)
            ignore_stamps = ignore_list_stamps(volumes, data_dir)
            print(
                f"epoch {epoch}: inventory rebuilt, "
                f"train={len(train_ex)} val={len(val_ex)}"
            )
        step0 = start_step if epoch == start_epoch else 0
        epoch_counter.set(epoch)
        indices = remaining_epoch_indices(
            n_train,
            seed,
            epoch,
            step0,
            batch_size,
            drop_last=True,
        )
        if not indices:
            start_step = 0
            continue
        train_dl = _make_loader(
            train_ds,
            indices,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        progress.report_epoch(epoch, num_epochs)
        pbar = tqdm(
            train_dl,
            desc=f"epoch {epoch}",
            leave=False,
            initial=step0,
            total=batches_per_epoch,
            ncols=100,
            mininterval=1.0,
            dynamic_ncols=False,
            bar_format="{desc} {percentage:5.1f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )
        interrupted = False
        t_prev = time.perf_counter()
        loss_window = _LossWindow()
        epoch_prompt_mix: Counter[str] = Counter()
        log_every = max(1, int(cfg.training.log_every_n_steps))

        def flush_losses() -> None:
            rows = loss_window.drain()
            if not rows:
                return
            for row in rows:
                epoch_prompt_mix.update(row["prompt_types"])
                anomaly.observe(
                    progress=progress,
                    epoch=row["epoch"],
                    step=row["step"],
                    duration_s=row["duration_s"],
                    loss=row["loss"],
                    meta=row["meta"],
                    sample_losses=row["sample_losses"],
                )
                writer.add_scalar("train/loss", row["loss"], row["global_step"])
                writer.add_scalar("train/loss_focal", row["loss_focal"], row["global_step"])
                writer.add_scalar("train/loss_dice", row["loss_dice"], row["global_step"])
                writer.add_scalar("train/loss_iou", row["loss_iou"], row["global_step"])
                if "loss_containment" in row:
                    writer.add_scalar(
                        "train/loss_containment", row["loss_containment"], row["global_step"]
                    )
                if "loss_mask_contain" in row:
                    writer.add_scalar(
                        "train/loss_mask_contain", row["loss_mask_contain"], row["global_step"]
                    )
                by_type: dict[str, list[float]] = {}
                for ptype, score in zip(row["prompt_types"], row["sample_losses"], strict=False):
                    by_type.setdefault(ptype, []).append(score)
                for ptype, scores in by_type.items():
                    writer.add_scalar(
                        f"train/loss_by_prompt/{ptype}", sum(scores) / len(scores), row["global_step"]
                    )
                writer.add_scalar(
                    "train/lr", optimizer.param_groups[0]["lr"], row["global_step"]
                )
            record_location_scores(data_dir, rows)
            last = rows[-1]
            pbar.set_postfix(loss=f"{last['loss']:.4f}")
            progress.report_step(
                step=last["step"],
                batches_per_epoch=batches_per_epoch,
                global_step=last["global_step"],
                log_every=1,
                loss=last["loss"],
                lr=float(optimizer.param_groups[0]["lr"]),
            )

        for rel_i, batch in enumerate(pbar):
            step = step0 + rel_i
            images = batch["image"].to(device, non_blocking=True)
            points = batch["point"].to(device, non_blocking=True)
            labels = batch["point_label"].to(device, non_blocking=True)
            inputs = _loss_inputs(batch, device)
            contain_scale = containment_warmup_scale(
                epoch,
                step,
                batches_per_epoch,
                float(OmegaConf.select(cfg, "loss.containment_warmup_epochs", default=2)),
            )
            with torch.amp.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=bool(cfg.training.mixed_precision) and device.type == "cuda",
            ):
                correction_rounds = _iterative_rounds(cfg, seed, epoch, step)
                if correction_rounds > 0:
                    pred, iou_preds = iterative_predict_masks(
                        model,
                        images,
                        points,
                        labels,
                        batch["boxes"],
                        batch["dense_mask"],
                        batch["mask"].to(device, non_blocking=True),
                        image_size=image_size,
                        rounds=correction_rounds,
                        prev_mask_prob=float(OmegaConf.select(cfg, "iterative.prev_mask_prob", default=0.5)),
                        rng=np.random.default_rng([seed, epoch, step, 9]),
                        variant_index=inputs["variant_index"],
                    )
                else:
                    pred, iou_preds = predict_masks_mixed(
                        model,
                        images,
                        points,
                        labels,
                        batch["boxes"],
                        batch["dense_mask"],
                        image_size=image_size,
                        variant_index=inputs["variant_index"],
                    )
                loss, stats = combined_loss(
                    pred,
                    inputs["masks"],
                    iou_preds,
                    focal_weight=float(cfg.loss.focal_weight),
                    dice_weight=float(cfg.loss.dice_weight),
                    iou_weight=float(cfg.loss.iou_weight),
                    points=points,
                    labels=labels,
                    containment_weight=float(cfg.loss.get("containment_weight", 0.0)),
                    containment_radius=float(cfg.loss.get("containment_radius", 8.0)),
                    prompt_foreground=inputs["prompt_fg"],
                    mask_contain_weight=_mask_contain_weight(cfg, contain_scale),
                    iou_use_l1=_iou_use_l1(cfg),
                    edge_weight=inputs["edge_weight"],
                    variant_index=inputs["variant_index"],
                    num_samples=images.shape[0],
                    sample_weight=inputs["sample_weight"],
                )
                loss = loss / accum
            scaler.scale(loss).backward()
            # Relative to the unsliced epoch, last batch still flushes leftover grads.
            epoch_batch_count = batches_per_epoch
            do_step = ((step + 1) % accum == 0) or ((step + 1) == epoch_batch_count)
            if do_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    float(cfg.training.gradient_clip),
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                opt_step += 1
                if opt_step % save_every == 0:
                    save_now(epoch, step + 1)
                    pbar.write(f"  saved last.pt (epoch {epoch} step {step + 1})")
            t_done = time.perf_counter()
            loss_window.add(
                stats,
                epoch=epoch,
                step=step,
                global_step=global_step,
                duration_s=t_done - t_prev,
                meta=batch.get("meta") or [],
                prompt_types=batch.get("prompt_type") or [],
            )
            t_prev = t_done
            if (global_step + 1) % log_every == 0:
                flush_losses()
            global_step += 1
            profiler.step()
            if stop.stop:
                save_now(epoch, step + 1)
                progress.mark_stopped()
                progress.transcript(
                    f"stop requested; saved last.pt at epoch {epoch} step {step + 1}"
                )
                interrupted = True
                break

        flush_losses()
        _log_prompt_mix(writer, epoch, epoch_prompt_mix)

        if interrupted:
            profiler.close()
            writer.close()
            print(f"Run directory: {run_dir}")
            return run_dir

        if epoch % int(cfg.training.val_every_n_epochs) == 0 and len(val_ds) == 0:
            progress.transcript(f"epoch {epoch}: no validation examples after inventory refresh")
        elif epoch % int(cfg.training.val_every_n_epochs) == 0:
            progress.stage("validate", "start")
            capture_overlays = (
                val_images_per_volume
                if due_for_val_images(global_step, last_val_image_step, val_images_every)
                else 0
            )
            full_due = full_pass_due(epoch, num_epochs, full_every)
            try:
                suite_result = val_suite.run(
                    model,
                    device,
                    image_size,
                    float(cfg.data.mask_threshold),
                    full=full_due,
                    overlays_per_volume=capture_overlays,
                )
            finally:
                progress.stage("validate", "end")
            # The subset numbers (also taken out of a full pass) select the checkpoint and stop the run.
            val_results, overlays = suite_result.subset, suite_result.overlays
            val_iou = val_results[val_suite.primary].iou
            val_key = epoch_selection(val_results, val_suite.primary, core_recall_target)
            val_summary = log_results(
                writer,
                epoch,
                val_results,
                val_suite.primary,
                core_recall_target,
                root="val_subset",
                label=f"subset ({val_suite.subset_size} of {len(val_ds)})",
            )
            writer.add_scalar("val_pass_seconds/subset" if not full_due else "val_pass_seconds/full",
                              suite_result.seconds, epoch)
            if suite_result.full is not None:
                progress.transcript(
                    log_results(
                        writer, epoch, suite_result.full, val_suite.primary, core_recall_target,
                        root="val_full", label=f"full ({len(val_ds)}, {suite_result.seconds / 60:.0f} min)",
                    )
                )
            if overlays:
                slots: dict[str, int] = {}
                for volume, panel in overlays:
                    slot = slots.get(volume, 0)
                    slots[volume] = slot + 1
                    writer.add_image(
                        f"val_overlay/{volume}/{slot}",
                        panel,
                        global_step,
                        dataformats="HWC",
                    )
                last_val_image_step = global_step
                writer.flush()
                progress.transcript(
                    f"logged {len(overlays)} val overlays at step {global_step} "
                    f"({', '.join(sorted(slots))})"
                )
            progress.transcript(val_summary)
            if improves(val_key, best_selection, min_delta):
                best_val_iou = val_iou
                best_selection = val_key
                epochs_without_improve = 0
                atomic_torch_save(model.state_dict(), run_dir / "best_model.pt")
                goal = val_results[val_suite.primary].goal
                progress.transcript(
                    f"  saved best_model.pt (core recall {goal.core_recall:.4f}, merge rate {goal.merge_rate:.3f}, "
                    f"spill {goal.spill:.3f}; {'target reached' if val_key.gated else 'below the target'})"
                )
            else:
                epochs_without_improve += 1
                if patience > 0 and epochs_without_improve >= patience:
                    if suite_result.full is None:
                        # The last validation of a run is always the full split.
                        final = val_suite.run(
                            model, device, image_size, float(cfg.data.mask_threshold), full=True
                        )
                        progress.transcript(
                            log_results(
                                writer, epoch, final.full, val_suite.primary, core_recall_target,
                                root="val_full", label=f"full ({len(val_ds)}, {final.seconds / 60:.0f} min)",
                            )
                        )
                    save_now(epoch + 1, 0)
                    progress.transcript(f"early stopping at epoch {epoch}")
                    break
            progress.stage("train", "start")
            progress.report_epoch(epoch, num_epochs)

        if epoch % int(cfg.training.save_every_n_epochs) == 0:
            atomic_torch_save(
                model.state_dict(), run_dir / f"checkpoint_epoch{epoch}.pt"
            )
        save_now(epoch + 1, 0)
        start_step = 0
        if stop.stop:
            progress.mark_stopped()
            progress.transcript("stop requested after epoch save")
            break

    profiler.close()
    writer.close()
    if progress.status == "stopped":
        print(f"Run directory: {run_dir}")
    else:
        progress.transcript(
            f"Training complete. Best epoch by the core-recall rule: "
            f"{'none' if best_selection is None else best_selection.values}. Run directory: {run_dir}"
        )
    return run_dir


def load_train_cfg(overrides: list[str] | None = None) -> DictConfig:
    cfg_path = Path(__file__).resolve().parent / "config" / "em_finetune.yaml"
    cfg = OmegaConf.load(cfg_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    PromptConfig.from_cfg(cfg)  # fail on bad prompt probabilities before any data or model loads
    return cfg


def parse_train_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune SAM2 on AnnotationCrops")
    parser.add_argument(
        "-refresh",
        "--refresh",
        action="store_true",
        help="Ignore last.pt and start from pretrained weights",
    )
    parser.add_argument(
        "-benchmark",
        "--benchmark",
        action="store_true",
        help="Time a short train/val sample and exit without writing checkpoints",
    )
    parser.add_argument(
        "--benchmark-steps",
        type=int,
        default=50,
        metavar="N",
        help="Timed train batches after warmup (default 50)",
    )
    parser.add_argument(
        "--benchmark-val-steps",
        type=int,
        default=20,
        metavar="N",
        help="Timed val forwards after warmup (default 20)",
    )
    parser.add_argument(
        "-benchmark-no-val",
        "--benchmark-no-val",
        action="store_true",
        help="With --benchmark: time only the training steps (no validation timing, no projection of it)",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra-style overrides, e.g. training.num_epochs=8 training.batch_size=2",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_train_args(argv)
    overrides = list(args.overrides)
    if args.refresh:
        overrides.append("training.resume=false")
    if args.benchmark:
        overrides.append("training.benchmark=true")
        overrides.append(f"training.benchmark_steps={args.benchmark_steps}")
        overrides.append(f"training.benchmark_val_steps={args.benchmark_val_steps}")
        if args.benchmark_no_val:
            overrides.append("training.benchmark_val=false")
    train_from_cfg(load_train_cfg(overrides))


if __name__ == "__main__":
    main()
