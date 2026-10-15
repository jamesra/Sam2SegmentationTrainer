from __future__ import annotations

import unittest

import torch

from sam2_segmentation_trainer.train import _LossWindow


def _stats(samples: int, *, mask_contain: float | None = None) -> dict[str, torch.Tensor]:
    stats = {
        "loss": torch.tensor(1.0),
        "loss_focal": torch.tensor(0.1),
        "loss_dice": torch.tensor(0.2),
        "loss_iou": torch.tensor(0.3),
        "sample_losses": torch.arange(samples, dtype=torch.float32),
    }
    if mask_contain is not None:
        stats["loss_mask_contain"] = torch.tensor(mask_contain)
    return stats


def _add(window: _LossWindow, stats: dict[str, torch.Tensor], step: int) -> None:
    window.add(stats, epoch=0, step=step, global_step=step, duration_s=0.5, meta=[], prompt_types=[])


class LossWindowTests(unittest.TestCase):
    def test_drains_rows_of_one_shape(self) -> None:
        window = _LossWindow()
        _add(window, _stats(6), 0)
        _add(window, _stats(6), 1)
        rows = window.drain()
        self.assertEqual([r["sample_losses"] for r in rows], [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]] * 2)
        self.assertNotIn("loss_mask_contain", rows[0])
        self.assertEqual(window.drain(), [])

    def test_an_optional_loss_that_appears_mid_window_does_not_break_the_drain(self) -> None:
        # The mask-containment term is logged once its warmup scale leaves 0: 4 + 6 numbers, then 4 + 1 + 6.
        window = _LossWindow()
        _add(window, _stats(6), 0)
        _add(window, _stats(6, mask_contain=0.7), 1)
        first, second = window.drain()
        self.assertNotIn("loss_mask_contain", first)
        self.assertEqual(first["sample_losses"], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(second["loss_mask_contain"], 0.7, places=6)
        self.assertEqual(second["sample_losses"], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(second["loss_iou"], 0.3, places=6)

    def test_rows_with_different_sample_counts_keep_their_own_losses(self) -> None:
        window = _LossWindow()
        _add(window, _stats(4), 0)
        _add(window, _stats(6, mask_contain=0.1), 1)
        first, second = window.drain()
        self.assertEqual(first["sample_losses"], [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(len(second["sample_losses"]), 6)


if __name__ == "__main__":
    unittest.main()
