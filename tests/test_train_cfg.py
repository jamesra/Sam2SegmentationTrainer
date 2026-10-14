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
        self.assertFalse(bool(cfg.model.activation_checkpointing))
        self.assertEqual(int(cfg.training.batch_size), 2)
        self.assertEqual(int(cfg.training.gradient_accumulation_steps), 8)
        self.assertEqual(float(cfg.training.early_stopping_min_delta), 0.002)
        self.assertEqual(float(cfg.val.epoch_subset), 0.25)
        self.assertEqual(int(cfg.val.full_every_n_epochs), 3)
        self.assertEqual(int(cfg.val.batch_size), 4)
        self.assertEqual(int(cfg.val.metric_threads), 4)
        self.assertEqual(int(cfg.training.val_every_n_epochs), 1)
        self.assertTrue(bool(cfg.training.resume))
        self.assertEqual(int(cfg.training.save_every_n_steps), 200)
        self.assertEqual(int(cfg.training.early_stopping_patience), 2)
        self.assertIsNone(cfg.training.checkpoint_dir)
        self.assertEqual(str(cfg.training.run_name), "auto")
        self.assertFalse(bool(cfg.data.even_per_volume))
        self.assertTrue(bool(cfg.training.profiler.enabled))
        self.assertEqual(int(cfg.training.profiler.active), 3)
        self.assertEqual(int(cfg.training.profiler.repeat), 1)
        self.assertFalse(bool(cfg.training.profiler.with_stack))
        self.assertEqual(int(cfg.training.val_images_every_n_steps), 2500)
        self.assertEqual(int(cfg.training.val_images_per_volume), 3)

    def test_prompt_mix_defaults(self) -> None:
        cfg = load_train_cfg()
        self.assertTrue(bool(cfg.prompt.enabled))
        probs = [
            cfg.prompt.prob_points_only,
            cfg.prompt.prob_box_only,
            cfg.prompt.prob_box_points,
            cfg.prompt.prob_circle_seed,
            cfg.prompt.prob_adjacent_seed,
            cfg.prompt.prob_subset_seed,
            cfg.prompt.prob_correction,
            cfg.prompt.prob_near_correct,
        ]
        self.assertAlmostEqual(sum(float(p) for p in probs), 1.0)
        self.assertEqual(float(cfg.prompt.prob_circle_seed), 0.40)
        self.assertEqual(float(cfg.prompt.prob_adjacent_seed), 0.15)
        self.assertEqual(int(cfg.prompt.variants_per_sample), 3)
        self.assertEqual(float(cfg.prompt.edge_tau_px), 8.0)
        self.assertEqual(float(cfg.prompt.edge_tau_depth_frac), 0.25)
        self.assertEqual(list(cfg.prompt.erode_px), [10, 40])
        self.assertEqual(list(cfg.prompt.dilate_px), [5, 50])
        self.assertEqual(list(cfg.prompt.correction_wobble_px), [10, 20])
        self.assertEqual(list(cfg.prompt.near_correct_wobble_px), [5, 12])
        self.assertEqual(list(cfg.prompt.near_correct_excursion_px), [15, 30])
        self.assertEqual(float(cfg.prompt.adjacent_fit_grow_px), 10.0)
        self.assertEqual(float(cfg.prompt.seed_logit), 10.0)
        self.assertEqual(float(cfg.prompt.correction_logit), 3.0)
        self.assertEqual(float(cfg.loss.edge_tau_px), 8.0)
        self.assertEqual(float(cfg.loss.leak_weight), 3.0)
        self.assertEqual(float(cfg.val.core_band_px), 10.0)
        self.assertEqual(float(cfg.val.core_recall_target), 0.975)
        self.assertIsNone(cfg.data.approved_path)
        self.assertEqual(float(cfg.data.approved_weight), 1.0)
        self.assertFalse(bool(cfg.training.cudnn_benchmark))
        self.assertEqual(float(cfg.model.activation_checkpointing_fraction), 1.0)
        self.assertEqual(float(cfg.loss.edge_weight_min), 0.05)
        for removed in ("prob_full_mask", "prob_negative_only", "prob_circle_mask", "prob_corrupted_mask"):
            self.assertNotIn(removed, cfg.prompt)
        for removed in ("allow_circle_targets", "circle_target_fraction"):
            self.assertNotIn(removed, cfg.negative)
        self.assertEqual(float(cfg.loss.mask_contain_weight), 0.5)
        self.assertEqual(float(cfg.loss.containment_warmup_epochs), 2.0)
        self.assertTrue(bool(cfg.model.unfreeze_mask_cnn))
        self.assertEqual(int(cfg.negative.cache_size), 64)
        self.assertEqual(
            list(cfg.eval.val_prompts), ["circle_mask", "correction", "near_correct", "point", "centroid"]
        )

    def test_out_of_memory_is_recognised_including_the_driver_mapping_failure(self) -> None:
        from sam2_segmentation_trainer.train import _is_oom

        self.assertTrue(_is_oom(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")))
        self.assertTrue(
            _is_oom(
                RuntimeError(
                    "CUDA driver error: device not ready. The CUDA driver logged these messages: "
                    "Failed to create GPU mapping"
                )
            )
        )
        self.assertFalse(_is_oom(RuntimeError("shape mismatch")))

    def test_prompt_probabilities_must_sum_to_one(self) -> None:
        with self.assertRaises(ValueError):
            load_train_cfg(["prompt.prob_points_only=0.5"])
        cfg = load_train_cfg(["prompt.prob_points_only=0.20", "prompt.prob_circle_seed=0.30"])
        self.assertAlmostEqual(float(cfg.prompt.prob_points_only), 0.20)

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

    def test_latest_resumable_run_uses_newest_checkpoint(self) -> None:
        import os
        from tempfile import TemporaryDirectory

        from sam2_segmentation_trainer.paths import latest_resumable_run

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            older = root / "runs" / "sam2_em_v1"
            newer = root / "runs" / "sam2_em_rpc1"
            older.mkdir(parents=True)
            newer.mkdir()
            (older / "last.pt").write_bytes(b"old")
            (newer / "last.pt").write_bytes(b"new")
            os.utime(older / "last.pt", (1_000, 1_000))
            os.utime(newer / "last.pt", (2_000, 2_000))
            self.assertEqual(latest_resumable_run([root]), "sam2_em_rpc1")

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
        self.assertFalse(args.benchmark_no_val)
        self.assertTrue(parse_train_args(["--benchmark", "--benchmark-no-val"]).benchmark_no_val)
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

    def test_val_image_cadence(self) -> None:
        from sam2_segmentation_trainer.train import due_for_val_images

        self.assertFalse(due_for_val_images(10_000, None, 0))
        self.assertTrue(due_for_val_images(100, None, 2500))
        self.assertFalse(due_for_val_images(3000, 1000, 2500))
        self.assertTrue(due_for_val_images(3500, 1000, 2500))

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

    def test_cudnn_benchmark_skips_cpu(self) -> None:
        import torch

        from sam2_segmentation_trainer.train import enable_cudnn_benchmark

        previous = bool(torch.backends.cudnn.benchmark)
        try:
            torch.backends.cudnn.benchmark = False
            self.assertFalse(enable_cudnn_benchmark(torch.device("cpu")))
            self.assertFalse(bool(torch.backends.cudnn.benchmark))
            if torch.cuda.is_available():
                self.assertTrue(enable_cudnn_benchmark(torch.device("cuda")))
                self.assertTrue(bool(torch.backends.cudnn.benchmark))
        finally:
            torch.backends.cudnn.benchmark = previous

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
