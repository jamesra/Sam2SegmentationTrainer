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
from sam2_segmentation_trainer.data.prompts import PromptConfig, PromptType, build_prompt


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


class CorruptedPromptTests(unittest.TestCase):
    def test_prompt_carries_a_dense_mask_but_no_containment_region(self) -> None:
        mask = _blob()
        prompt = build_prompt(
            PromptType.CORRUPTED_MASK,
            target_mask=mask,
            circle_mask=None,
            regions=[],
            cfg=PromptConfig(),
            rng=np.random.default_rng(2),
        )
        self.assertEqual(prompt.prompt_type, PromptType.CORRUPTED_MASK)
        self.assertIsNotNone(prompt.dense_logits)
        self.assertEqual(prompt.dense_logits.shape, (256, 256))
        self.assertIsNone(prompt.foreground_region)
        self.assertEqual(prompt.point_coords.shape[0], 1)
        self.assertEqual(int(prompt.point_labels[0]), 1)

    def test_full_mask_still_reports_its_containment_region(self) -> None:
        prompt = build_prompt(
            PromptType.FULL_MASK,
            target_mask=_blob(),
            circle_mask=None,
            regions=[],
            cfg=PromptConfig(),
            rng=np.random.default_rng(2),
        )
        self.assertIsNotNone(prompt.foreground_region)

    def test_probabilities_include_the_new_type_and_validate(self) -> None:
        cfg = PromptConfig(
            prob_points_only=0.15, prob_box_only=0.05, prob_box_points=0.10,
            prob_circle_mask=0.45, prob_full_mask=0.0, prob_negative_only=0.0, prob_corrupted_mask=0.25,
        )
        cfg.validate()
        self.assertEqual(cfg.probabilities()[PromptType.CORRUPTED_MASK], 0.25)
        with self.assertRaises(ValueError):
            PromptConfig(corrupt_stroke_keep_min=0.9, corrupt_stroke_keep_max=0.4).validate()
        with self.assertRaises(ValueError):
            PromptConfig(corrupt_max_shift_px=0).validate()


if __name__ == "__main__":
    unittest.main()
