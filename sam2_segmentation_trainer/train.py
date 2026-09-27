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
from sam2_segmentation_trainer.data.dataset import EMSegDataset, collate_em
from sam2_segmentation_trainer.data.manifest import index_volumes, take_model_sized
from sam2_segmentation_trainer.data.splits import load_split, make_location_split, save_split
from sam2_segmentation_trainer.metrics import iou_dice_pr
from sam2_segmentation_trainer.mqtt_progress import StepAnomalyMonitor, TrainingProgress
from sam2_segmentation_trainer.model import (
    build_em_sam2,
    combined_loss,
    configure_trainable_params,
    enable_activation_checkpointing,
    param_groups,
    predict_masks,
)
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


def _resolve_paths(cfg: DictConfig) -> tuple[Path, Path, Path, Path]:
    root = data_root(cfg.data.root)
    out = output_root(cfg.training.output_dir)
    ckpt_root_cfg = cfg.training.get("checkpoint_dir")
    ckpt_root = checkpoint_root(None if ckpt_root_cfg in (None, "") else ckpt_root_cfg)
    ckpt = cfg.model.checkpoint
    checkpoint = Path(ckpt) if ckpt else pretrained_checkpoint(out)
    return root, out, ckpt_root, checkpoint


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
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _summarize_times(times: list[float]) -> str:
    if not times:
        return "n/a"
    mean = statistics.fmean(times)
    median = statistics.median(times)
    return (
        f"{mean:.3f} s/it (median {median:.3f}, "
        f"min {min(times):.3f}, max {max(times):.3f}, n={len(times)})"
    )


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
        masks = batch["mask"].to(device, non_blocking=True)
        points = batch["point"].to(device, non_blocking=True)
        labels = batch["point_label"].to(device, non_blocking=True)
        with torch.amp.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=bool(mixed_precision) and device.type == "cuda",
        ):
            pred, iou_preds = predict_masks(
                model, images, points, labels, image_size=image_size
            )
            loss, stats = combined_loss(
                pred,
                masks,
                iou_preds,
                focal_weight=float(cfg.loss.focal_weight),
                dice_weight=float(cfg.loss.dice_weight),
                iou_weight=float(cfg.loss.iou_weight),
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
        log_every = int(cfg.training.log_every_n_steps)
        for j in range(train_steps):
            i = warmup_train + j
            t0 = time.perf_counter()
            step_loss = train_one(next(train_iter), i)
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
    n_val_take = min(len(val_ds), need_val)
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
    projected = project_full_run(
        train_s=train_s,
        val_s=val_s,
        batches_per_epoch=batches_per_epoch,
        n_val=n_val,
        num_epochs=num_epochs,
        val_every_n_epochs=val_every_n_epochs,
    )
    gpu = (
        torch.cuda.get_device_name(0)
        if device.type == "cuda" and torch.cuda.is_available()
        else str(device)
    )
    print("Benchmark (no checkpoints written, last.pt not loaded)")
    print(f"  GPU: {gpu}")
    print(
        f"  timed {len(train_times)} train batches after {warmup_train} warmup; "
        f"{len(val_times)} val after {warmup_val} warmup"
    )
    print(
        f"  train={n_train} val={n_val} batch_size={batch_size} "
        f"batches/epoch={batches_per_epoch} epochs={num_epochs}"
    )
    print(f"  train {_summarize_times(train_times)}")
    print(f"  val   {_summarize_times(val_times)}")
    print(f"  estimated train epoch: {format_duration(projected['train_epoch_s'])}")
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


@torch.no_grad()
def run_validation(model, loader, device, image_size: int, threshold: float) -> float:
    model.eval()
    ious = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        points = batch["point"].to(device, non_blocking=True)
        labels = batch["point_label"].to(device, non_blocking=True)
        pred, _iou = predict_masks(
            model, images, points, labels, image_size=image_size
        )
        pred_bin = (torch.sigmoid(pred) > threshold).to(torch.int64)
        for pred_i, mask_i in zip(pred_bin, masks, strict=True):
            stats = iou_dice_pr(
                pred_i.detach().cpu().numpy(),
                mask_i.detach().cpu().numpy(),
            )
            ious.append(stats["iou"])
    model.train()
    if not ious:
        return 0.0
    return float(np.mean(ious))


class _LossWindow:
    """Detached GPU losses, copied to the CPU in one transfer when drained."""

    def __init__(self) -> None:
        self._rows: list[torch.Tensor] = []
        self._meta: list[tuple[int, int, int, float, list]] = []

    def add(
        self,
        stats: dict[str, torch.Tensor],
        *,
        epoch: int,
        step: int,
        global_step: int,
        duration_s: float,
        meta: list,
    ) -> None:
        packed = torch.cat(
            [
                stats["loss"].reshape(1),
                stats["loss_focal"].reshape(1),
                stats["loss_dice"].reshape(1),
                stats["loss_iou"].reshape(1),
                stats["sample_losses"].reshape(-1),
            ]
        )
        self._rows.append(packed.detach())
        self._meta.append((epoch, step, global_step, duration_s, meta))

    def drain(self) -> list[dict]:
        if not self._rows:
            return []
        packed = torch.stack(self._rows).float().cpu()
        records = []
        for i, (epoch, step, global_step, duration_s, meta) in enumerate(self._meta):
            row = packed[i].tolist()
            records.append(
                {
                    "epoch": epoch,
                    "step": step,
                    "global_step": global_step,
                    "duration_s": duration_s,
                    "meta": meta,
                    "loss": row[0],
                    "loss_focal": row[1],
                    "loss_dice": row[2],
                    "loss_iou": row[3],
                    "sample_losses": row[4:],
                }
            )
        self._rows.clear()
        self._meta.clear()
        return records


_SCORE_SQL = """
CREATE TABLE IF NOT EXISTS location_scores (
    location_id INTEGER NOT NULL,
    image_key TEXT NOT NULL,
    epoch INTEGER NOT NULL,
    score REAL NOT NULL,
    PRIMARY KEY (location_id, image_key, epoch)
)
"""


def _ensure_score_schema(connection: sqlite3.Connection) -> None:
    """Create per-window scores, or keep a location-only table under an empty image key."""
    connection.execute(_SCORE_SQL)
    columns = {row[1] for row in connection.execute("PRAGMA table_info(location_scores)")}
    if "image_key" in columns:
        return
    connection.execute(
        """
        CREATE TABLE location_scores_window (
            location_id INTEGER NOT NULL,
            image_key TEXT NOT NULL,
            epoch INTEGER NOT NULL,
            score REAL NOT NULL,
            PRIMARY KEY (location_id, image_key, epoch)
        )
        """
    )
    connection.execute(
        """
        INSERT INTO location_scores_window (location_id, image_key, epoch, score)
        SELECT location_id, '', epoch, score FROM location_scores
        """
    )
    connection.execute("DROP TABLE location_scores")
    connection.execute("ALTER TABLE location_scores_window RENAME TO location_scores")


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
) -> dict:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,
        "global_step": global_step,
        "opt_step": opt_step,
        "best_val_iou": best_val_iou,
        "epochs_without_improve": epochs_without_improve,
        "freeze_mode": freeze_mode,
        "n_param_groups": len(optimizer.param_groups),
        "rng": capture_rng(),
    }


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


def _counts_by_volume(examples) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ex in examples:
        counts[ex.volume] = counts.get(ex.volume, 0) + 1
    return dict(sorted(counts.items()))


def _write_run_inputs(
    path: Path,
    *,
    run_name: str,
    data_dir: Path,
    volumes: list[str],
    even_per_volume: bool,
    split_path: Path,
    usable_counts: dict[str, int],
    train_counts: dict[str, int],
    val_counts: dict[str, int],
    seed: int,
    train_fraction: float,
    image_size: int,
    batch_size: int,
    num_epochs: int,
    model_config: str,
    pretrained: Path,
) -> None:
    if path.is_file():
        return
    payload = {
        "run_name": run_name,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_root": str(data_dir),
        "volumes_requested": volumes,
        "even_per_volume": even_per_volume,
        "split_file": str(split_path),
        "usable_per_volume": usable_counts,
        "train_per_volume": train_counts,
        "val_per_volume": val_counts,
        "n_usable": sum(usable_counts.values()),
        "n_train": sum(train_counts.values()),
        "n_val": sum(val_counts.values()),
        "seed": seed,
        "train_fraction": train_fraction,
        "image_size": image_size,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "model_config": model_config,
        "pretrained_checkpoint": str(pretrained),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


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
    data_dir, out_dir, ckpt_root, checkpoint = _resolve_paths(cfg)
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
    examples = index_volumes(volumes=volumes, root=data_dir, skip_missing=True)
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
        split = make_location_split(
            examples,
            train_fraction=float(cfg.data.train_split),
            seed=int(cfg.training.seed),
            even_per_volume=even,
        )
        split_path = run_split
        save_split(split, run_split)
        even_applied = even
        print(f"wrote split {run_split} even_per_volume={even}")
    train_ex, val_ex = split.partition(examples)
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
    )

    batch_size = int(cfg.training.batch_size)
    num_workers = int(cfg.training.num_workers)
    prefetch_factor = int(cfg.training.prefetch_factor)
    persistent_workers = bool(cfg.training.persistent_workers)
    seed = int(cfg.training.seed)
    freeze_mode = str(cfg.model.freeze_mode)
    train_ds = EMSegDataset(train_ex, split="train", image_size=image_size)
    val_ds = EMSegDataset(val_ex, split="val", image_size=image_size)

    n_train = len(train_ds)
    batches_per_epoch = max(1, n_train // batch_size)
    accum = max(1, int(cfg.training.gradient_accumulation_steps))
    steps_per_epoch = max(1, batches_per_epoch // accum)
    total_steps = steps_per_epoch * int(cfg.training.num_epochs)
    warmup_steps = steps_per_epoch * int(cfg.lr_schedule.warmup_epochs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_em_sam2(cfg.model.config, str(checkpoint), device, train=True)
    configure_trainable_params(model, freeze_mode=freeze_mode)
    if bool(cfg.model.get("activation_checkpointing", True)):
        n_ckpt = enable_activation_checkpointing(model)
        print(f"Activation checkpointing on {n_ckpt} Hiera blocks")
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
        )
        print(f"Run directory: {run_dir} (benchmark only; no last.pt written)")
        return run_dir

    writer = SummaryWriter(log_dir=str(run_dir / "tb"))
    val_dl = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_em,
    )

    best_val_iou = -1.0
    epochs_without_improve = 0
    global_step = 0
    opt_step = 0
    start_epoch = 0
    start_step = 0
    patience = int(cfg.training.early_stopping_patience)
    min_delta = float(cfg.training.get("early_stopping_min_delta", 0.0))
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
            ),
        )

    num_epochs = int(cfg.training.num_epochs)
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
        step0 = start_step if epoch == start_epoch else 0
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
        log_every = max(1, int(cfg.training.log_every_n_steps))

        def flush_losses() -> None:
            rows = loss_window.drain()
            if not rows:
                return
            for row in rows:
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
            masks = batch["mask"].to(device, non_blocking=True)
            points = batch["point"].to(device, non_blocking=True)
            labels = batch["point_label"].to(device, non_blocking=True)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=bool(cfg.training.mixed_precision) and device.type == "cuda",
            ):
                pred, iou_preds = predict_masks(
                    model, images, points, labels, image_size=image_size
                )
                loss, stats = combined_loss(
                    pred,
                    masks,
                    iou_preds,
                    focal_weight=float(cfg.loss.focal_weight),
                    dice_weight=float(cfg.loss.dice_weight),
                    iou_weight=float(cfg.loss.iou_weight),
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
            )
            t_prev = t_done
            if (global_step + 1) % log_every == 0:
                flush_losses()
            global_step += 1
            if stop.stop:
                save_now(epoch, step + 1)
                progress.mark_stopped()
                progress.transcript(
                    f"stop requested; saved last.pt at epoch {epoch} step {step + 1}"
                )
                interrupted = True
                break

        flush_losses()

        if interrupted:
            writer.close()
            print(f"Run directory: {run_dir}")
            return run_dir

        if epoch % int(cfg.training.val_every_n_epochs) == 0:
            progress.stage("validate", "start")
            try:
                val_iou = run_validation(
                    model,
                    val_dl,
                    device,
                    image_size,
                    float(cfg.data.mask_threshold),
                )
            finally:
                progress.stage("validate", "end")
            writer.add_scalar("val/iou", val_iou, epoch)
            progress.transcript(f"epoch {epoch} val IoU {val_iou:.4f}")
            if val_iou > best_val_iou + min_delta:
                best_val_iou = val_iou
                epochs_without_improve = 0
                atomic_torch_save(model.state_dict(), run_dir / "best_model.pt")
                progress.transcript(f"  saved best_model.pt (IoU {best_val_iou:.4f})")
            else:
                epochs_without_improve += 1
                if patience > 0 and epochs_without_improve >= patience:
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

    writer.close()
    if progress.status == "stopped":
        print(f"Run directory: {run_dir}")
    else:
        progress.transcript(
            f"Training complete. Best val IoU: {best_val_iou:.4f}. Run directory: {run_dir}"
        )
    return run_dir


def load_train_cfg(overrides: list[str] | None = None) -> DictConfig:
    cfg_path = Path(__file__).resolve().parent / "config" / "em_finetune.yaml"
    cfg = OmegaConf.load(cfg_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
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
    train_from_cfg(load_train_cfg(overrides))


if __name__ == "__main__":
    main()
