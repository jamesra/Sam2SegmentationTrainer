"""Training-run records. Checkpoint bytes and ``inputs.json`` stay the same shape."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TrainerRoots:
    """Data, output, checkpoint, and pretrained paths from one config."""

    data: Path
    output: Path
    checkpoint: Path
    pretrained: Path


@dataclass(frozen=True, slots=True)
class RunInputs:
    """Fields written to ``inputs.json``. ``created_utc`` is filled at write time."""

    run_name: str
    data_dir: Path
    volumes: list[str]
    even_per_volume: bool
    split_path: Path
    usable_counts: dict[str, int]
    train_counts: dict[str, int]
    val_counts: dict[str, int]
    seed: int
    train_fraction: float
    image_size: int
    batch_size: int
    num_epochs: int
    model_config: str
    pretrained: Path

    def as_json(self, *, created_utc: str) -> dict[str, Any]:
        """JSON object written to disk. Key names stay stable."""
        return {
            "run_name": self.run_name,
            "created_utc": created_utc,
            "data_root": str(self.data_dir),
            "volumes_requested": self.volumes,
            "even_per_volume": self.even_per_volume,
            "split_file": str(self.split_path),
            "usable_per_volume": self.usable_counts,
            "train_per_volume": self.train_counts,
            "val_per_volume": self.val_counts,
            "n_usable": sum(self.usable_counts.values()),
            "n_train": sum(self.train_counts.values()),
            "n_val": sum(self.val_counts.values()),
            "seed": self.seed,
            "train_fraction": self.train_fraction,
            "image_size": self.image_size,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "model_config": self.model_config,
            "pretrained_checkpoint": str(self.pretrained),
        }


@dataclass
class TrainSnapshot:
    """Constructor for the ``last.pt`` dict. This is not a new file format."""

    model: Any
    optimizer: Any
    scheduler: Any
    scaler: Any
    epoch: int
    step_in_epoch: int
    global_step: int
    opt_step: int
    best_val_iou: float
    epochs_without_improve: int
    freeze_mode: str
    rng: dict
    unfreeze_mask_cnn: bool = False

    def as_checkpoint(self) -> dict[str, Any]:
        """Dict ``torch.save`` already understands."""
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": self.epoch,
            "step_in_epoch": self.step_in_epoch,
            "global_step": self.global_step,
            "opt_step": self.opt_step,
            "best_val_iou": self.best_val_iou,
            "epochs_without_improve": self.epochs_without_improve,
            "freeze_mode": self.freeze_mode,
            "unfreeze_mask_cnn": self.unfreeze_mask_cnn,
            "n_param_groups": len(self.optimizer.param_groups),
            "rng": self.rng,
        }
