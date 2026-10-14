from __future__ import annotations

import unittest

import torch

from sam2_segmentation_trainer.model import (
    build_prompt_region,
    combined_loss,
    prompt_containment_loss_per_sample,
)


class BuildPromptRegionTests(unittest.TestCase):
    def test_disk_around_positive_point(self) -> None:
        points = torch.tensor([[[5.0, 5.0]]])
        labels = torch.tensor([[1]])
        region = build_prompt_region(points, labels, (10, 10), radius=1.0)
        self.assertEqual(region.shape, (1, 10, 10))
        self.assertTrue(bool(region[0, 5, 5]))
        self.assertTrue(bool(region[0, 5, 6]))
        self.assertFalse(bool(region[0, 5, 8]))

    def test_negative_point_excluded(self) -> None:
        points = torch.tensor([[[5.0, 5.0]]])
        labels = torch.tensor([[0]])
        region = build_prompt_region(points, labels, (10, 10), radius=2.0)
        self.assertFalse(bool(region.any()))


class LossPrecisionTests(unittest.TestCase):
    def test_the_loss_terms_are_computed_in_fp32_for_bf16_predictions(self) -> None:
        from sam2_segmentation_trainer.model import combined_loss

        torch.manual_seed(0)
        pred32 = torch.randn(2, 256, 256) * 4.0
        target = (torch.rand(2, 256, 256) > 0.5).float()
        iou32 = torch.rand(2, 1)
        points = torch.tensor([[[128.0, 128.0]], [[64.0, 64.0]]])
        labels = torch.ones(2, 1, dtype=torch.int32)
        kwargs = dict(
            focal_weight=20.0, dice_weight=1.0, iou_weight=1.0, points=points, labels=labels, containment_weight=1.0,
            edge_weight=torch.rand(2, 256, 256).half(),
        )
        loss_bf16, stats = combined_loss(pred32.bfloat16(), target, iou32.bfloat16(), **kwargs)
        self.assertEqual(loss_bf16.dtype, torch.float32)
        for name in ("loss_focal", "loss_dice", "loss_iou", "loss_containment"):
            self.assertEqual(stats[name].dtype, torch.float32, name)
        # Same as running the already rounded predictions in fp32: no further rounding inside the loss.
        loss_ref, _ = combined_loss(
            pred32.bfloat16().float(), target, iou32.bfloat16().float(), **{**kwargs, "edge_weight": kwargs["edge_weight"].float()}
        )
        self.assertAlmostEqual(float(loss_bf16), float(loss_ref), places=6)


class ContainmentLossTests(unittest.TestCase):
    def test_penalizes_background_logit_at_prompt(self) -> None:
        pred = torch.full((1, 10, 10), -10.0)
        points = torch.tensor([[[5.0, 5.0]]])
        labels = torch.tensor([[1]])
        loss = prompt_containment_loss_per_sample(pred, points, labels, radius=1.0)
        self.assertGreater(float(loss[0]), 5.0)

    def test_zero_when_already_foreground(self) -> None:
        pred = torch.full((1, 10, 10), 10.0)
        points = torch.tensor([[[5.0, 5.0]]])
        labels = torch.tensor([[1]])
        loss = prompt_containment_loss_per_sample(pred, points, labels, radius=1.0)
        self.assertLess(float(loss[0]), 1e-3)

    def test_combined_loss_without_containment_matches_baseline(self) -> None:
        pred = torch.zeros((2, 8, 8))
        target = torch.zeros((2, 8, 8), dtype=torch.int64)
        iou_preds = torch.zeros((2, 1))
        loss, stats = combined_loss(
            pred, target, iou_preds, focal_weight=1.0, dice_weight=1.0, iou_weight=1.0
        )
        self.assertNotIn("loss_containment", stats)
        self.assertTrue(torch.isfinite(loss))

    def test_combined_loss_adds_containment_term(self) -> None:
        pred = torch.full((1, 8, 8), -10.0)
        target = torch.zeros((1, 8, 8), dtype=torch.int64)
        iou_preds = torch.zeros((1, 1))
        points = torch.tensor([[[4.0, 4.0]]])
        labels = torch.tensor([[1]])

        loss_plain, stats_plain = combined_loss(
            pred, target, iou_preds, focal_weight=1.0, dice_weight=1.0, iou_weight=1.0
        )
        loss_with, stats_with = combined_loss(
            pred,
            target,
            iou_preds,
            focal_weight=1.0,
            dice_weight=1.0,
            iou_weight=1.0,
            points=points,
            labels=labels,
            containment_weight=1.0,
            containment_radius=1.0,
        )
        self.assertIn("loss_containment", stats_with)
        self.assertGreater(float(loss_with), float(loss_plain))


if __name__ == "__main__":
    unittest.main()
