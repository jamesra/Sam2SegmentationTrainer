from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from sam2_segmentation_trainer.checkpoint import (
    atomic_torch_save,
    bak_path,
    epoch_sample_indices,
    load_latest_checkpoint,
    remaining_epoch_indices,
    tmp_path,
)


class AtomicCheckpointTests(unittest.TestCase):
    def test_roundtrip_and_bak_fallback(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "last.pt"
            atomic_torch_save({"v": 1}, path)
            self.assertEqual(load_latest_checkpoint(path)["v"], 1)
            atomic_torch_save({"v": 2}, path)
            self.assertEqual(load_latest_checkpoint(path)["v"], 2)
            self.assertTrue(bak_path(path).is_file())
            path.write_bytes(b"not a pickle")
            loaded = load_latest_checkpoint(path)
            self.assertEqual(loaded["v"], 1)

    def test_never_loads_tmp(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "last.pt"
            torch.save({"v": 99}, tmp_path(path))
            self.assertIsNone(load_latest_checkpoint(path))

    def test_missing_returns_none(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "last.pt"
            self.assertIsNone(load_latest_checkpoint(path))


class EpochIndexTests(unittest.TestCase):
    def test_deterministic_and_resume_slice(self) -> None:
        n, seed, epoch, batch = 25, 42, 3, 4
        full = epoch_sample_indices(n, seed, epoch)
        self.assertEqual(full, epoch_sample_indices(n, seed, epoch))
        self.assertNotEqual(full, epoch_sample_indices(n, seed, epoch + 1))
        usable = (n // batch) * batch
        from_zero = remaining_epoch_indices(n, seed, epoch, 0, batch, drop_last=True)
        self.assertEqual(from_zero, full[:usable])
        step = 2
        resumed = remaining_epoch_indices(n, seed, epoch, step, batch, drop_last=True)
        self.assertEqual(resumed, full[step * batch : usable])
        self.assertEqual(len(resumed) % batch, 0)

    def test_resume_past_end_is_empty(self) -> None:
        rest = remaining_epoch_indices(10, 0, 0, 50, 4, drop_last=True)
        self.assertEqual(rest, [])


class MigrateRunArtifactsTests(unittest.TestCase):
    def test_copies_missing_only(self) -> None:
        from sam2_segmentation_trainer.checkpoint import migrate_run_artifacts

        with TemporaryDirectory() as tmp:
            legacy = Path(tmp) / "legacy"
            dest = Path(tmp) / "dest"
            legacy.mkdir()
            (legacy / "last.pt").write_bytes(b"aaa")
            (legacy / "best_model.pt").write_bytes(b"bbb")
            dest.mkdir()
            (dest / "best_model.pt").write_bytes(b"keep")
            moved = migrate_run_artifacts(legacy, dest)
            self.assertEqual(moved, ["last.pt"])
            self.assertEqual((dest / "last.pt").read_bytes(), b"aaa")
            self.assertEqual((dest / "best_model.pt").read_bytes(), b"keep")


class CheckpointRootTests(unittest.TestCase):
    def test_default_and_override(self) -> None:
        from sam2_segmentation_trainer.paths import (
            DEFAULT_CHECKPOINT_ROOT,
            checkpoint_root,
            run_dir_for,
        )

        self.assertEqual(checkpoint_root("/tmp/ck"), Path("/tmp/ck"))
        self.assertTrue(str(DEFAULT_CHECKPOINT_ROOT).startswith("/storage4"))
        self.assertEqual(
            run_dir_for("sam2_em_v1", ckpt_root=Path("/storage4/Sam2Trainer")),
            Path("/storage4/Sam2Trainer/runs/sam2_em_v1"),
        )


if __name__ == "__main__":
    unittest.main()
