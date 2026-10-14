from __future__ import annotations

import unittest

import numpy as np

from sam2_segmentation_trainer.data.corrupt import (
    CORRUPTIONS,
    corrupt_mask,
    morph_mask,
    polygon_mask,
    stroke_mask,
)
from sam2_segmentation_trainer.data.prompts import PromptConfig


def _blob() -> np.ndarray:
    yy, xx = np.mgrid[0:96, 0:96]
    return (yy - 48) ** 2 + (xx - 48) ** 2 <= 30**2


class CorruptionTests(unittest.TestCase):
    def test_morph_changes_the_area_by_a_bounded_amount(self) -> None:
        mask = _blob()
        for seed in range(12):
            out = morph_mask(mask, np.random.default_rng(seed), max_shift_px=6)
            self.assertTrue(out.any())
            self.assertFalse(np.array_equal(out, mask))

    def test_strokes_keep_a_subset(self) -> None:
        mask = _blob()
        out = stroke_mask(mask, np.random.default_rng(3), 0.4, 0.9)
        self.assertTrue(out.any())
        self.assertTrue(not (out & ~mask).any())
        self.assertLess(int(out.sum()), int(mask.sum()))

    def test_polygon_is_a_filled_coarse_outline(self) -> None:
        mask = _blob()
        out = polygon_mask(mask, np.random.default_rng(1), 0.08)
        self.assertTrue(out.any())
        overlap = int((out & mask).sum()) / int((out | mask).sum())
        self.assertGreater(overlap, 0.8)

    def test_corrupt_mask_is_deterministic_and_never_empty(self) -> None:
        mask = _blob()
        a = corrupt_mask(mask, np.random.default_rng(5))
        b = corrupt_mask(mask, np.random.default_rng(5))
        self.assertTrue(np.array_equal(a, b))
        for seed in range(30):
            self.assertTrue(corrupt_mask(mask, np.random.default_rng(seed)).any())
        self.assertFalse(corrupt_mask(np.zeros((8, 8), dtype=bool), np.random.default_rng(0)).any())

    def test_every_corruption_is_reachable(self) -> None:
        self.assertEqual(set(CORRUPTIONS), {"morph", "strokes", "polygon"})


class RefinementConfigTests(unittest.TestCase):
    def test_cascade_settings_are_validated(self) -> None:
        PromptConfig().validate()
        with self.assertRaises(ValueError):
            PromptConfig(corrupt_max_shift_px=0).validate()
        with self.assertRaises(ValueError):
            PromptConfig(corrupt_polygon_tolerance=0.0).validate()


if __name__ == "__main__":
    unittest.main()
