from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from omegaconf import OmegaConf

from annotation_crops.negatives import load_circle_scale
from sam2_segmentation_trainer.data.manifest import index_volumes
from sam2_segmentation_trainer.data.tiles import build_tile_context
from sam2_segmentation_trainer.model import combined_loss, iou_prediction_loss_per_sample

try:
    from tests.helpers import make_crops
except ImportError:
    from helpers import make_crops


def _write_circles(crops: Path, scale: float, count: int = 1, source: str = "annotated") -> None:
    cells = {
        "RC1_0_D1_X0-1_Y0-1": [
            {"location_id": 1000 + i, "center_x": 8.0, "center_y": 8.0, "radius": 4.0, "source": source}
            for i in range(count)
        ]
    }
    (crops / "circle_annotations.json").write_text(
        json.dumps({"version": 1, "diameter_scale": scale, "grid_cells": cells}), encoding="utf-8"
    )


class CircleFileGuardTests(unittest.TestCase):
    def test_missing_circle_file_raises_when_required(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_crops(root, "RC1", n=3)
            examples = index_volumes(volumes=["RC1"], root=root)
            with self.assertRaisesRegex(RuntimeError, "RC1"):
                build_tile_context(examples, require_circles=True)

    def test_missing_circle_file_is_allowed_when_not_required(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_crops(root, "RC1", n=3)
            examples = index_volumes(volumes=["RC1"], root=root)
            context = build_tile_context(examples, require_circles=False, circle_scale=0.95)
            self.assertEqual(context.circles, {})

    def _radius_after_loading(
        self, file_scale: float, circle_scale: float | None, source: str = "inscribed"
    ) -> float:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            crops = make_crops(root, "RC1", n=3)
            _write_circles(crops, file_scale, source=source)
            self.assertEqual(load_circle_scale(crops), file_scale)
            examples = index_volumes(volumes=["RC1"], root=root)
            context = build_tile_context(examples, require_circles=True, circle_scale=circle_scale)
            (circles,) = context.circles.values()
            return circles[0].radius

    def test_true_radii_are_shrunk_to_the_prompt_scale_for_inscribed_circles(self) -> None:
        self.assertAlmostEqual(self._radius_after_loading(1.0, 0.95), 4.0 * 0.95)

    def test_annotated_radii_are_not_shrunk(self) -> None:
        # Circle annotations load at their true radius whatever the diameter scale; the trust fractions apply later.
        self.assertAlmostEqual(self._radius_after_loading(1.0, 0.95, "annotated"), 4.0)
        self.assertAlmostEqual(self._radius_after_loading(1.0, None, "annotated"), 4.0)

    def test_annotated_radii_of_an_old_file_are_divided_by_its_scale(self) -> None:
        self.assertAlmostEqual(self._radius_after_loading(0.8, 0.95, "annotated"), 4.0 / 0.8)

    def test_baked_in_scale_is_rescaled_not_applied_twice(self) -> None:
        # An older file already has 0.8 in its radii; asking for 0.95 gives 0.95/0.8 of them.
        self.assertAlmostEqual(self._radius_after_loading(0.8, 0.95), 4.0 * 0.95 / 0.8)
        self.assertAlmostEqual(self._radius_after_loading(0.95, 0.95), 4.0)

    def test_no_circle_scale_keeps_radii_as_written(self) -> None:
        self.assertAlmostEqual(self._radius_after_loading(1.0, None), 4.0)

    def test_scale_is_none_without_a_file(self) -> None:
        with TemporaryDirectory() as tmp:
            self.assertIsNone(load_circle_scale(Path(tmp)))


class IouLossTests(unittest.TestCase):
    def test_the_iou_head_loss_is_the_absolute_error(self) -> None:
        pred = torch.full((1, 4, 4), 10.0)  # every pixel predicted foreground, and the target is all foreground: IoU 1
        target = torch.ones(1, 4, 4)
        iou_preds = torch.tensor([[0.5]])
        loss = iou_prediction_loss_per_sample(pred, target, iou_preds)
        self.assertAlmostEqual(float(loss), 0.5, places=5)  # |0.5 - 1|, not (0.5 - 1) ** 2 = 0.25

    def test_an_error_above_one_costs_its_size_not_its_square(self) -> None:
        pred = torch.full((2, 4, 4), 10.0)
        target = torch.ones(2, 4, 4)
        loss = iou_prediction_loss_per_sample(pred, target, torch.tensor([[0.2], [0.9]]))
        self.assertAlmostEqual(float(loss[0]), 0.8, places=5)  # a squared error would be 0.64
        self.assertAlmostEqual(float(loss[1]), 0.1, places=5)  # a squared error would be 0.01
        # Linear: the badly wrong sample costs 8x the nearly right one, not 64x.
        self.assertAlmostEqual(float(loss[0] / loss[1]), 8.0, places=3)

    def test_combined_loss_uses_it_with_weight_one(self) -> None:
        pred = torch.zeros(2, 4, 4)
        target = torch.ones(2, 4, 4)
        iou_preds = torch.full((2, 1), 0.9)
        kwargs = dict(focal_weight=0.0, dice_weight=0.0, iou_weight=1.0)
        loss, stats = combined_loss(pred, target, iou_preds, **kwargs)
        self.assertAlmostEqual(float(loss), 0.9, places=5)  # |0.9 - 0|: the prediction is all background, IoU 0
        self.assertAlmostEqual(float(stats["loss_iou"]), 0.9, places=5)

    def test_there_is_no_switch_back_to_squared_error(self) -> None:
        import inspect

        self.assertNotIn("use_l1", inspect.signature(iou_prediction_loss_per_sample).parameters)
        self.assertNotIn("iou_use_l1", inspect.signature(combined_loss).parameters)
        cfg = OmegaConf.load(Path(__file__).resolve().parent.parent / "sam2_segmentation_trainer" / "config" / "em_finetune.yaml")
        self.assertNotIn("iou_use_l1", cfg.loss)
        self.assertEqual(float(cfg.loss.iou_weight), 1.0)


class PromptMixTests(unittest.TestCase):
    def test_fractions_sum_to_one_and_sort(self) -> None:
        from sam2_segmentation_trainer.train import prompt_mix_fractions

        shares = prompt_mix_fractions(Counter({"circle_mask": 3, "points": 1}))
        self.assertEqual(list(shares), ["circle_mask", "points"])
        self.assertAlmostEqual(sum(shares.values()), 1.0)
        self.assertEqual(prompt_mix_fractions(Counter()), {})


if __name__ == "__main__":
    unittest.main()
