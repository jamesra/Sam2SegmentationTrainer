"""PyTorch profiler schedule defaults and a one-step TensorBoard trace."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from sam2_segmentation_trainer.tb_profiler import TbStepProfiler, profiler_options
from sam2_segmentation_trainer.train import load_train_cfg


class TbProfilerTests(unittest.TestCase):
    def test_missing_section_stays_on(self) -> None:
        cfg = load_train_cfg()
        del cfg.training.profiler
        opts = profiler_options(cfg)
        self.assertTrue(opts["enabled"])
        self.assertEqual(opts["skip_first"], 5)
        self.assertEqual(opts["active"], 3)

    def test_override_disables(self) -> None:
        cfg = load_train_cfg(["training.profiler.enabled=false"])
        self.assertFalse(profiler_options(cfg)["enabled"])

    def test_disabled_writes_nothing(self) -> None:
        cfg = load_train_cfg(["training.profiler.enabled=false"])
        with TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            profiler = TbStepProfiler(
                cfg, log_dir, torch.device("cpu"), worker_name="unit"
            )
            profiler.step()
            profiler.close()
            profiler.close()
            self.assertEqual(list(log_dir.glob("*.pt.trace.json*")), [])

    def test_one_step_writes_trace(self) -> None:
        cfg = load_train_cfg(
            [
                "training.profiler.skip_first=0",
                "training.profiler.wait=0",
                "training.profiler.warmup=0",
                "training.profiler.active=1",
                "training.profiler.repeat=1",
                "training.profiler.record_shapes=false",
                "training.profiler.profile_memory=false",
                "training.profiler.with_stack=false",
                "training.profiler.with_modules=false",
            ]
        )
        with TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            profiler = TbStepProfiler(
                cfg, log_dir, torch.device("cpu"), worker_name="unit"
            )
            layer = torch.nn.Linear(4, 2)
            loss = layer(torch.randn(2, 4)).sum()
            loss.backward()
            profiler.step()
            profiler.close()
            traces = list(log_dir.glob("*.pt.trace.json*"))
            self.assertTrue(traces, "expected a chrome trace for the TensorBoard plugin")
            self.assertTrue(traces[0].stat().st_size > 0)


if __name__ == "__main__":
    unittest.main()
