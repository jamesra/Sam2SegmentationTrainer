from __future__ import annotations

import unittest

import numpy as np
import torch

from sam2_segmentation_trainer.metrics import boundary_f1, iou_dice_pr, summarize_records
from sam2_segmentation_trainer.model import transform_prompt_coords


class CoordTests(unittest.TestCase):
    def test_identity_when_already_model_size(self) -> None:
        pts = torch.tensor([[[100.0, 200.0], [512.0, 10.0]]])
        out = transform_prompt_coords(pts, (1024, 1024), 1024)
        torch.testing.assert_close(out, pts)

    def test_scales_from_native_hw(self) -> None:
        pts = torch.tensor([[[32.0, 16.0]]])
        out = transform_prompt_coords(pts, orig_hw=(64, 128), image_size=1024)
        torch.testing.assert_close(out, torch.tensor([[[256.0, 256.0]]]))


class MetricTests(unittest.TestCase):
    def test_perfect_overlap(self) -> None:
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5:15, 5:15] = 1
        stats = iou_dice_pr(mask, mask)
        self.assertAlmostEqual(stats["iou"], 1.0, places=5)
        self.assertAlmostEqual(stats["dice"], 1.0, places=5)
        self.assertGreater(boundary_f1(mask, mask), 0.99)

    def test_summarize_groups(self) -> None:
        recs = [
            {
                "iou": 0.5,
                "dice": 0.6,
                "precision": 0.5,
                "recall": 0.5,
                "bf": 0.4,
                "volume": "RC1",
                "category_name": "MC",
                "downsample": 1,
            },
            {
                "iou": 1.0,
                "dice": 1.0,
                "precision": 1.0,
                "recall": 1.0,
                "bf": 1.0,
                "volume": "RC1",
                "category_name": "HC",
                "downsample": 2,
            },
        ]
        summary = summarize_records(recs)
        self.assertEqual(summary["overall"]["n"], 2)
        self.assertAlmostEqual(summary["overall"]["iou"], 0.75)
        self.assertIn("RC1", summary["by_volume"])
        self.assertIn("1", summary["by_downsample"])


if __name__ == "__main__":
    unittest.main()
