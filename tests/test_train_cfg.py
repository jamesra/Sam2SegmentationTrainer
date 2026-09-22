from __future__ import annotations

import unittest
from pathlib import Path

from sam2_segmentation_trainer.train import load_train_cfg


class TrainCfgTests(unittest.TestCase):
    def test_default_yaml(self) -> None:
        cfg = load_train_cfg()
        self.assertEqual(cfg.model.freeze_mode, "none")
        self.assertEqual(list(cfg.data.volumes), ["RC1", "RC2", "RPC1", "RPC2"])
        self.assertEqual(cfg.model.config, "configs/sam2.1/sam2.1_hiera_l.yaml")
        self.assertTrue(bool(cfg.model.activation_checkpointing))
        self.assertTrue(bool(cfg.training.resume))
        self.assertEqual(int(cfg.training.save_every_n_steps), 200)
        self.assertIsNone(cfg.training.checkpoint_dir)
        self.assertEqual(str(cfg.training.run_name), "auto")
        self.assertTrue(bool(cfg.data.even_per_volume))

    def test_next_run_name_increments(self) -> None:
        from tempfile import TemporaryDirectory

        from sam2_segmentation_trainer.paths import next_run_name

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "runs" / "sam2_em_v1").mkdir(parents=True)
            (root / "runs" / "sam2_em_v3").mkdir()
            (root / "runs" / "other").mkdir()
            self.assertEqual(next_run_name("sam2_em", [root]), "sam2_em_v4")
            self.assertEqual(next_run_name("sam2_em", [root / "missing"]), "sam2_em_v1")

    def test_refresh_flag_disables_resume(self) -> None:
        from sam2_segmentation_trainer.train import parse_train_args

        args = parse_train_args(["-refresh", "training.batch_size=2"])
        self.assertTrue(args.refresh)
        self.assertEqual(args.overrides, ["training.batch_size=2"])
        cfg = load_train_cfg(list(args.overrides) + ["training.resume=false"])
        self.assertFalse(bool(cfg.training.resume))

    def test_overrides(self) -> None:
        cfg = load_train_cfg(["training.num_epochs=3", "training.batch_size=2"])
        self.assertEqual(int(cfg.training.num_epochs), 3)
        self.assertEqual(int(cfg.training.batch_size), 2)

    def test_benchmark_flag(self) -> None:
        from sam2_segmentation_trainer.train import parse_train_args

        args = parse_train_args(["-benchmark", "training.batch_size=2"])
        self.assertTrue(args.benchmark)
        self.assertEqual(args.benchmark_steps, 50)
        self.assertEqual(args.benchmark_val_steps, 20)
        self.assertEqual(args.overrides, ["training.batch_size=2"])
        cfg = load_train_cfg(
            list(args.overrides)
            + [
                "training.benchmark=true",
                f"training.benchmark_steps={args.benchmark_steps}",
            ]
        )
        self.assertTrue(bool(cfg.training.benchmark))
        self.assertEqual(int(cfg.training.benchmark_steps), 50)

    def test_project_full_run(self) -> None:
        from sam2_segmentation_trainer.train import format_duration, project_full_run

        projected = project_full_run(
            train_s=1.0,
            val_s=0.25,
            batches_per_epoch=100,
            n_val=40,
            num_epochs=15,
            val_every_n_epochs=1,
        )
        self.assertEqual(projected["train_epoch_s"], 100.0)
        self.assertEqual(projected["val_pass_s"], 10.0)
        self.assertEqual(projected["full_run_s"], 15 * 100.0 + 15 * 10.0)
        self.assertEqual(format_duration(3661), "1h 1m 1s")

    def test_activation_checkpointing_wraps_blocks(self) -> None:
        import torch

        from sam2_segmentation_trainer.model import enable_activation_checkpointing

        class Block(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = torch.nn.Linear(4, 4)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.lin(x)

        class Bag:
            pass

        model = Bag()
        model.image_encoder = Bag()
        model.image_encoder.trunk = Bag()
        model.image_encoder.trunk.blocks = torch.nn.ModuleList([Block(), Block()])
        self.assertEqual(enable_activation_checkpointing(model), 2)
        self.assertEqual(enable_activation_checkpointing(model), 0)
        x = torch.randn(2, 4, requires_grad=True)
        loss = model.image_encoder.trunk.blocks[0](x).sum()
        loss.backward()
        self.assertIsNotNone(model.image_encoder.trunk.blocks[0].lin.weight.grad)


if __name__ == "__main__":
    unittest.main()
