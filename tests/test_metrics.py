from __future__ import annotations

import unittest

import numpy as np
import torch

from scipy.ndimage import binary_dilation
from skimage.segmentation import find_boundaries

from sam2_segmentation_trainer.metrics import (
    boundary_f1,
    boundary_f1_multi,
    iou_dice_pr,
    mean_iou_by_volume,
    summarize_records,
)
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

    def test_boundary_f1_matches_a_square_dilation_at_every_tolerance(self) -> None:
        def reference(pred, gt, tolerance):
            pred_b = find_boundaries(pred, mode="outer")
            gt_b = find_boundaries(gt, mode="outer")
            struct = np.ones((2 * tolerance + 1, 2 * tolerance + 1))
            prec = (pred_b & binary_dilation(gt_b, structure=struct)).sum() / (pred_b.sum() + 1e-6)
            rec = (gt_b & binary_dilation(pred_b, structure=struct)).sum() / (gt_b.sum() + 1e-6)
            return float(2 * prec * rec / (prec + rec + 1e-6))

        gt = np.zeros((120, 140), dtype=bool)
        gt[30:90, 40:100] = True
        pred = np.zeros_like(gt)
        pred[34:96, 36:104] = True
        pred[10:20, 120:130] = True
        tolerances = (2, 4, 10, 20)
        multi = boundary_f1_multi(pred, gt, tolerances)
        for tolerance, value in zip(tolerances, multi):
            self.assertAlmostEqual(value, reference(pred, gt, tolerance), places=5)
            self.assertAlmostEqual(boundary_f1(pred, gt, tolerance), value, places=6)
        self.assertEqual(sorted(multi), multi)  # a looser tolerance never scores lower

    def test_boundary_f1_with_a_mask_touching_the_array_border_and_empty_masks(self) -> None:
        gt = np.zeros((60, 60), dtype=bool)
        gt[:30, :30] = True
        self.assertGreater(boundary_f1(gt, gt, 10), 0.99)
        empty = np.zeros_like(gt)
        self.assertEqual(boundary_f1(empty, gt, 10), 0.0)
        self.assertEqual(boundary_f1(empty, empty, 10), 0.0)

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

    def test_mean_iou_by_volume_uses_example_names(self) -> None:
        overall, by_volume = mean_iou_by_volume(
            [
                ("RC1", 0.2),
                ("RC1", 0.4),
                ("RC2", 0.9),
                ("NewVolume", 0.1),
                ("", 0.0),
            ]
        )
        self.assertAlmostEqual(overall, (0.2 + 0.4 + 0.9 + 0.1 + 0.0) / 5)
        self.assertEqual(list(by_volume), ["NewVolume", "RC1", "RC2"])
        self.assertAlmostEqual(by_volume["RC1"], 0.3)
        self.assertAlmostEqual(by_volume["RC2"], 0.9)
        self.assertAlmostEqual(by_volume["NewVolume"], 0.1)

    def test_mean_iou_by_volume_empty(self) -> None:
        overall, by_volume = mean_iou_by_volume([])
        self.assertEqual(overall, 0.0)
        self.assertEqual(by_volume, {})


if __name__ == "__main__":
    unittest.main()
