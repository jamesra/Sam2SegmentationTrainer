from __future__ import annotations

import contextlib
import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import torch
from torch import nn

from sam2_segmentation_trainer.data.dataset import EMSegDataset
from sam2_segmentation_trainer.data.manifest import index_volumes
from sam2_segmentation_trainer.data.prompts import PromptConfig
from sam2_segmentation_trainer.data.tiles import build_tile_context
from sam2_segmentation_trainer.train import _amp_setup, load_train_cfg, run_throughput_benchmark
from sam2_segmentation_trainer.validation import ValidationSuite

try:
    from tests.helpers import make_crops
except ImportError:
    from helpers import make_crops


class _Standin(nn.Module):
    """One trainable scalar; the prediction functions are patched to use it, so backward and the optimizer work."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(()))


def _train_logits(model, images, points, labels, boxes, dense, *, image_size, variant_index=None, **_):
    rows = points.shape[0]
    return model.scale * torch.ones(rows, image_size, image_size), torch.zeros(rows, 1)


def _val_logits(model, batch, device, image_size):
    return batch["mask"][batch["variant_index"]].float() * 20.0 - 10.0


class BenchmarkRunTests(unittest.TestCase):
    """``run_throughput_benchmark`` end to end on the CPU with a validation suite (it once crashed decoding the
    suite's several-prompts-per-image items with the single-prompt timing loop)."""

    def test_prints_the_summary_with_a_validation_suite(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_crops(root, "RC1", n=8)
            examples = index_volumes(volumes=["RC1"], root=root)
            tiles = build_tile_context(examples, load_circles=False)
            cfg = load_train_cfg(["training.log_every_n_steps=1", "training.num_workers=0"])
            prompt_cfg = PromptConfig.from_cfg(cfg)
            train_ds = EMSegDataset(
                examples, split="train", image_size=32, tiles=tiles, prompt_cfg=prompt_cfg, seed=1,
                edge_tau_px=8.0,
            )
            suite = ValidationSuite(
                examples, modes=["circle_mask", "correction", "near_correct", "point", "centroid"],
                primary="circle_mask", image_size=32, num_workers=0, tiles=tiles, prompt_cfg=prompt_cfg,
                seed=1, core_band_px=2.0, batch_size=2, epoch_subset=0.5, metric_threads=2,
            )
            model = _Standin()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
            dtype, scaler = _amp_setup(torch.device("cpu"), False)
            out = io.StringIO()
            with mock.patch("sam2_segmentation_trainer.train.predict_masks_mixed", _train_logits), mock.patch(
                "sam2_segmentation_trainer.validation.predict_variants", _val_logits
            ), contextlib.redirect_stdout(out):
                projected = run_throughput_benchmark(
                    model=model, train_ds=train_ds, val_ds=suite.dataset, val_suite=suite,
                    device=torch.device("cpu"), image_size=32, batch_size=2, num_workers=0,
                    n_train=len(train_ds), n_val=len(suite), seed=1, batches_per_epoch=len(train_ds) // 2,
                    num_epochs=6, val_every_n_epochs=1, accum=2, amp_dtype=dtype, scaler=scaler,
                    mixed_precision=False, cfg=cfg, optimizer=optimizer, scheduler=scheduler,
                    train_steps=3, val_steps=4, warmup_train=1, warmup_val=2,
                )
        text = out.getvalue()
        self.assertIn("Benchmark (no checkpoints written", text)
        self.assertIn("val subset", text)
        self.assertIn("val full", text)
        self.assertIn("estimated val pass, subset", text)
        self.assertIn("estimated val pass, full (2 of 6 epochs)", text)  # full passes after epochs 3 and 6
        self.assertGreater(projected["val_full_pass_s"], 0.0)
        self.assertGreater(projected["val_pass_s"], 0.0)
        # The run is 4 subset passes and 2 full ones on top of the training epochs.
        expected = (
            projected["train_epoch_s"] * 6 + 4 * projected["val_pass_s"] + 2 * projected["val_full_pass_s"]
        )
        self.assertAlmostEqual(projected["full_run_s"], expected, places=6)


    def test_training_only_mode_skips_the_validation_timing(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_crops(root, "RC1", n=8)
            examples = index_volumes(volumes=["RC1"], root=root)
            tiles = build_tile_context(examples, load_circles=False)
            cfg = load_train_cfg(["training.log_every_n_steps=1", "training.num_workers=0"])
            prompt_cfg = PromptConfig.from_cfg(cfg)
            train_ds = EMSegDataset(
                examples, split="train", image_size=32, tiles=tiles, prompt_cfg=prompt_cfg, seed=1,
                edge_tau_px=8.0,
            )
            suite = ValidationSuite(
                examples, modes=["circle_mask", "point"], primary="circle_mask", image_size=32, num_workers=0,
                tiles=tiles, prompt_cfg=prompt_cfg, seed=1, core_band_px=2.0, batch_size=2, epoch_subset=0.5,
            )
            model = _Standin()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
            dtype, scaler = _amp_setup(torch.device("cpu"), False)
            out = io.StringIO()
            timed = mock.Mock(side_effect=AssertionError("validation must not be timed"))
            with mock.patch("sam2_segmentation_trainer.train.predict_masks_mixed", _train_logits), mock.patch(
                "sam2_segmentation_trainer.train._time_validation_suite", timed
            ), contextlib.redirect_stdout(out):
                projected = run_throughput_benchmark(
                    model=model, train_ds=train_ds, val_ds=suite.dataset, val_suite=suite,
                    device=torch.device("cpu"), image_size=32, batch_size=2, num_workers=0,
                    n_train=len(train_ds), n_val=len(suite), seed=1, batches_per_epoch=len(train_ds) // 2,
                    num_epochs=6, val_every_n_epochs=1, accum=2, amp_dtype=dtype, scaler=scaler,
                    mixed_precision=False, cfg=cfg, optimizer=optimizer, scheduler=scheduler,
                    train_steps=3, val_steps=4, warmup_train=1, warmup_val=2, time_validation=False,
                )
        text = out.getvalue()
        self.assertIn("validation not timed", text)
        self.assertNotIn("val subset", text)
        self.assertNotIn("estimated val pass", text)
        self.assertEqual(projected["val_pass_s"], 0.0)
        timed.assert_not_called()
        self.assertIn("train ", text)


if __name__ == "__main__":
    unittest.main()
