"""Overlay strips used by eval grids and TensorBoard Images."""

from __future__ import annotations

import unittest

import numpy as np

from sam2_segmentation_trainer.data.dataset import imagenet_normalize
from sam2_segmentation_trainer.overlays import denorm_preview, overlay_panel


class OverlayTests(unittest.TestCase):
    def test_denorm_inverts_imagenet_normalize(self) -> None:
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        rgb[..., 0] = 10
        rgb[..., 1] = 20
        rgb[..., 2] = 30
        back = denorm_preview(imagenet_normalize(rgb))
        np.testing.assert_allclose(back, rgb, atol=1)

    def test_prediction_panel_marks_extra_and_missed(self) -> None:
        rgb = np.full((4, 4, 3), 128, dtype=np.uint8)
        gt = np.zeros((4, 4), dtype=np.uint8)
        pred = np.zeros((4, 4), dtype=np.uint8)
        gt[0, 0] = 1
        pred[0, 1] = 1
        gt[0, 2] = 1
        pred[0, 2] = 1
        panel = overlay_panel(rgb, gt, pred)
        self.assertEqual(panel.shape, (4, 12, 3))
        gt_px = panel[0, 4]
        self.assertGreater(int(gt_px[1]), int(gt_px[0]))
        false_pos = panel[0, 8 + 1]
        self.assertGreater(int(false_pos[0]), int(false_pos[1]))
        false_neg = panel[0, 8]
        self.assertGreater(int(false_neg[2]), int(false_neg[0]))
        np.testing.assert_array_equal(panel[:, :4], rgb)


if __name__ == "__main__":
    unittest.main()
