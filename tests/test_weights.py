from __future__ import annotations

import unittest

import numpy as np

from sam2_segmentation_trainer.data.weights import (
    apply_leak_weight,
    core_mask,
    edge_weight_map,
    leak_mask,
    neighbor_core_labels,
)


def _box(size: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


class EdgeWeightTests(unittest.TestCase):
    def test_curve_on_both_sides_of_the_edge(self) -> None:
        target = _box(400, 100, 300, 100, 300)  # 200 px: tau stays 8
        weight = edge_weight_map(target, 8.0, 0.05)
        # d = 0.5 px next to the edge on either side, 8.5 px at the first full pixel of the mask interior
        self.assertAlmostEqual(float(weight[200, 100]), max(0.05, 1 - np.exp(-0.5 / 8)), places=4)
        self.assertAlmostEqual(float(weight[200, 99]), max(0.05, 1 - np.exp(-0.5 / 8)), places=4)
        self.assertAlmostEqual(float(weight[200, 108]), 1 - np.exp(-8.5 / 8), places=4)
        self.assertAlmostEqual(float(weight[200, 91]), 1 - np.exp(-8.5 / 8), places=4)
        self.assertGreater(float(weight[200, 200]), 0.999)
        self.assertEqual(float(weight[0, 0]), 1.0)  # far outside: beyond 3 tau the map says 1

    def test_the_floor(self) -> None:
        weight = edge_weight_map(_box(100, 20, 80, 20, 80), 8.0, 0.3)
        self.assertGreaterEqual(float(weight.min()), 0.3 - 1e-6)

    def test_a_thin_target_keeps_its_supervision(self) -> None:
        bar = _box(200, 90, 110, 20, 180)  # 20 px wide
        capped = edge_weight_map(bar, 8.0, 0.05, depth_frac=0.25)
        uncapped = edge_weight_map(bar, 8.0, 0.05, depth_frac=0.0)
        self.assertGreater(float(capped[100, 100]), 0.95)
        self.assertLess(float(uncapped[100, 100]), 0.75)
        self.assertGreater(float(capped[95, 100]), float(uncapped[95, 100]))

    def test_disabled_and_degenerate_targets(self) -> None:
        self.assertIsNone(edge_weight_map(_box(50, 10, 20, 10, 20), 0.0))
        self.assertEqual(float(edge_weight_map(np.zeros((20, 20), bool), 8.0).min()), 1.0)
        self.assertEqual(float(edge_weight_map(np.ones((20, 20), bool), 8.0).min()), 1.0)


class LeakTests(unittest.TestCase):
    def test_core_is_ten_pixels_inside(self) -> None:
        mask = _box(200, 40, 160, 40, 160)
        core = core_mask(mask, 10.0)
        self.assertTrue(core[100, 100])
        self.assertFalse(core[100, 45])  # 5 px inside
        self.assertFalse(core[100, 49])  # 9.5 px inside is still short of 10
        self.assertTrue(core[100, 51])
        self.assertTrue(np.array_equal(core_mask(mask, 0.0), mask))

    def test_leak_is_neighbor_cores_outside_the_target(self) -> None:
        target = _box(300, 20, 100, 20, 100)
        neighbor = _box(300, 20, 140, 90, 250)  # overlaps the target's right edge
        leak = leak_mask([neighbor], target, 10.0)
        self.assertFalse(leak[60, 50])  # inside the target only
        self.assertFalse(leak[60, 95])  # inside both: the target owns it
        self.assertTrue(leak[60, 150])  # deep in the neighbor
        self.assertTrue(leak[60, 105])  # 15 px inside the neighbor and outside the target
        self.assertFalse(leak[25, 150])  # only 5 px from the neighbor's top edge: its rim, not its core
        self.assertFalse(leak[200, 200])
        self.assertFalse((leak & target).any())

    def test_labels_name_each_neighbor(self) -> None:
        target = _box(300, 20, 60, 20, 60)
        a = _box(300, 20, 100, 100, 180)
        b = _box(300, 150, 250, 100, 250)
        labels = neighbor_core_labels([a, b], target, 10.0)
        self.assertEqual(int(labels[60, 140]), 1)
        self.assertEqual(int(labels[200, 170]), 2)
        self.assertEqual(int(labels[60, 100]), 0)  # the rim of a neighbor is not its core
        self.assertEqual(labels.dtype, np.int16)

    def test_apply_leak_weight(self) -> None:
        base = np.full((10, 10), 0.5, dtype=np.float32)
        leak = np.zeros((10, 10), dtype=bool)
        leak[2:4, 2:4] = True
        out = apply_leak_weight(base, leak, 3.0, base.shape)
        self.assertAlmostEqual(float(out[2, 2]), 1.5)
        self.assertAlmostEqual(float(out[0, 0]), 0.5)
        self.assertAlmostEqual(float(base[2, 2]), 0.5)  # the input is not modified
        none = apply_leak_weight(None, leak, 3.0, base.shape)
        self.assertAlmostEqual(float(none[2, 2]), 3.0)
        self.assertAlmostEqual(float(none[0, 0]), 1.0)
        self.assertIsNone(apply_leak_weight(None, leak, 1.0, base.shape))
        self.assertIs(apply_leak_weight(base, np.zeros_like(leak), 3.0, base.shape), base)


if __name__ == "__main__":
    unittest.main()
