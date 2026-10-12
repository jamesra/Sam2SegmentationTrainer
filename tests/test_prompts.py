from __future__ import annotations

import unittest

import numpy as np

from annotation_crops.negatives import KIND_CIRCLE, KIND_POLYGON, NegativeRegion
from sam2_segmentation_trainer.data.prompts import (
    LOW_RES,
    MASK_PROMPT_LOGIT,
    PromptConfig,
    PromptType,
    benford_count,
    build_prompt,
    choose_prompt_type,
    jittered_box,
    nearest_resize,
    sample_points_in_mask,
    to_low_res_logits,
)

CFG = PromptConfig()
POINT_ARGS = dict(min_edge_px=15, line_frac_min=0.4, line_frac_max=0.9)


def _square(size: int, lo: int, hi: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[lo:hi, lo:hi] = True
    return mask


def _only(prompt_type: PromptType) -> PromptConfig:
    values = {
        "prob_points_only": 0.0,
        "prob_box_only": 0.0,
        "prob_box_points": 0.0,
        "prob_circle_mask": 0.0,
        "prob_full_mask": 0.0,
        "prob_negative_only": 0.0,
    }
    name = {
        PromptType.POINTS: "prob_points_only",
        PromptType.BOX: "prob_box_only",
        PromptType.BOX_POINTS: "prob_box_points",
        PromptType.CIRCLE_MASK: "prob_circle_mask",
        PromptType.FULL_MASK: "prob_full_mask",
        PromptType.NEGATIVE_ONLY: "prob_negative_only",
    }[prompt_type]
    values[name] = 1.0
    return PromptConfig(**values)


class PromptConfigTests(unittest.TestCase):
    def test_defaults_sum_to_one(self) -> None:
        CFG.validate()
        self.assertAlmostEqual(sum(CFG.probabilities().values()), 1.0)

    def test_rejects_bad_sum_negative_and_inverted_range(self) -> None:
        with self.assertRaises(ValueError):
            PromptConfig(prob_points_only=0.5).validate()
        with self.assertRaises(ValueError):
            PromptConfig(prob_points_only=-0.1, prob_circle_mask=0.65).validate()
        with self.assertRaises(ValueError):
            PromptConfig(line_frac_min=0.9, line_frac_max=0.4).validate()
        with self.assertRaises(ValueError):
            PromptConfig(box_points_max=0).validate()

    def test_from_cfg_merges_sections_and_keeps_defaults(self) -> None:
        cfg = {
            "prompt": {"prob_points_only": 0.15, "prob_circle_mask": 0.40, "box_points_max": 2},
            "negative": {"cache_size": 8, "allow_circle_targets": True},
        }
        config = PromptConfig.from_cfg(cfg)
        self.assertEqual(config.prob_points_only, 0.15)
        self.assertEqual(config.box_points_max, 2)
        self.assertEqual(config.cache_size, 8)
        self.assertIs(config.allow_circle_targets, True)
        self.assertEqual(config.prob_full_mask, 0.30)
        self.assertEqual(PromptConfig.from_cfg({}), CFG)

    def test_from_cfg_validates(self) -> None:
        with self.assertRaises(ValueError):
            PromptConfig.from_cfg({"prompt": {"prob_points_only": 0.9}})


class PointSamplingTests(unittest.TestCase):
    def test_benford_prefers_one_and_stays_in_range(self) -> None:
        rng = np.random.default_rng(0)
        counts = np.array([benford_count(rng) for _ in range(4000)])
        self.assertTrue(counts.min() >= 1 and counts.max() <= 9)
        self.assertAlmostEqual(float((counts == 1).mean()), 0.301, delta=0.04)

    def test_first_point_is_deepest_and_all_points_inside(self) -> None:
        mask = _square(64, 8, 56)
        rng = np.random.default_rng(1)
        points = sample_points_in_mask(mask, 5, rng, **POINT_ARGS)
        self.assertEqual(points.shape, (5, 2))
        self.assertEqual(points.dtype, np.float32)
        for x, y in points:
            self.assertTrue(mask[int(round(y)), int(round(x))])
        self.assertTrue(24 <= points[0, 0] <= 40 and 24 <= points[0, 1] <= 40)
        self.assertGreater(len({tuple(p) for p in points.tolist()}), 1)

    def test_thin_mask_relaxes_edge_distance(self) -> None:
        mask = np.zeros((40, 40), dtype=bool)
        mask[18:22, 5:35] = True  # 4 px thick, shallower than min_edge_px
        points = sample_points_in_mask(mask, 3, np.random.default_rng(2), **POINT_ARGS)
        self.assertGreaterEqual(points.shape[0], 1)
        for x, y in points:
            self.assertTrue(mask[int(round(y)), int(round(x))])

    def test_empty_mask_and_zero_count(self) -> None:
        rng = np.random.default_rng(3)
        self.assertEqual(sample_points_in_mask(np.zeros((8, 8), bool), 3, rng, **POINT_ARGS).shape, (0, 2))
        self.assertEqual(sample_points_in_mask(_square(8, 1, 7), 0, rng, **POINT_ARGS).shape, (0, 2))

    def test_deterministic_for_a_seed(self) -> None:
        mask = _square(64, 8, 56)
        a = sample_points_in_mask(mask, 4, np.random.default_rng(9), **POINT_ARGS)
        b = sample_points_in_mask(mask, 4, np.random.default_rng(9), **POINT_ARGS)
        self.assertTrue(np.array_equal(a, b))


class BoxTests(unittest.TestCase):
    def test_box_contains_object_and_is_clamped(self) -> None:
        mask = _square(64, 10, 50)
        for seed in range(20):
            box = jittered_box(mask, np.random.default_rng(seed), 0.15)
            self.assertTrue(box[0] <= 10 and box[1] <= 10 and box[2] >= 50 and box[3] >= 50)
            self.assertTrue(box[0] >= 0 and box[1] >= 0 and box[2] <= 64 and box[3] <= 64)
            self.assertTrue(box[0] >= 10 - 0.15 * 40 - 1e-4)

    def test_border_object_is_clamped(self) -> None:
        mask = np.zeros((32, 32), dtype=bool)
        mask[0:20, 0:32] = True
        box = jittered_box(mask, np.random.default_rng(0), 0.5)
        self.assertEqual((box[0], box[1], box[2]), (0.0, 0.0, 32.0))

    def test_empty_mask_has_no_box(self) -> None:
        self.assertIsNone(jittered_box(np.zeros((8, 8), bool), np.random.default_rng(0), 0.1))


class DenseMaskTests(unittest.TestCase):
    def test_nearest_resize_keeps_exact_values(self) -> None:
        signed = np.zeros((1024, 1024), dtype=np.int8)
        signed[:512] = 1
        signed[600:700, 100:200] = -1
        small = nearest_resize(signed, LOW_RES)
        self.assertEqual(small.shape, (LOW_RES, LOW_RES))
        self.assertEqual(set(np.unique(small)), {-1, 0, 1})
        self.assertEqual(int(small[0, 0]), 1)
        self.assertEqual(int(small[160, 40]), -1)

    def test_logit_scale(self) -> None:
        signed = np.zeros((512, 512), dtype=np.int8)
        signed[:256] = 1
        signed[300:400] = -1
        logits = to_low_res_logits(signed)
        self.assertEqual(logits.dtype, np.float32)
        self.assertEqual(set(np.unique(logits)), {-MASK_PROMPT_LOGIT, 0.0, MASK_PROMPT_LOGIT})


class ChoosePromptTypeTests(unittest.TestCase):
    def test_unavailable_types_are_dropped(self) -> None:
        rng = np.random.default_rng(0)
        drawn = {
            choose_prompt_type(CFG, rng, can_circle_mask=False, has_negatives=False) for _ in range(500)
        }
        self.assertNotIn(PromptType.CIRCLE_MASK, drawn)
        self.assertNotIn(PromptType.NEGATIVE_ONLY, drawn)
        self.assertIn(PromptType.FULL_MASK, drawn)

    def test_everything_available_draws_every_type(self) -> None:
        rng = np.random.default_rng(1)
        drawn = {
            choose_prompt_type(CFG, rng, can_circle_mask=True, has_negatives=True) for _ in range(4000)
        }
        self.assertEqual(drawn, set(PromptType))

    def test_nothing_left_falls_back_to_full_mask(self) -> None:
        only_circle = _only(PromptType.CIRCLE_MASK)
        self.assertIs(
            choose_prompt_type(only_circle, np.random.default_rng(0), can_circle_mask=False, has_negatives=True),
            PromptType.FULL_MASK,
        )


class BuildPromptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.target = _square(64, 16, 48)
        self.circle = np.zeros((64, 64), dtype=bool)
        self.circle[24:40, 24:40] = True
        neighbor = np.zeros((64, 64), dtype=bool)
        neighbor[2:12, 2:12] = True
        circle_neighbor = np.zeros((64, 64), dtype=bool)
        circle_neighbor[50:60, 50:60] = True
        self.regions = [
            NegativeRegion(10, KIND_POLYGON, neighbor),
            NegativeRegion(11, KIND_CIRCLE, circle_neighbor),
        ]

    def _build(self, prompt_type: PromptType, regions=None, circle=True, seed: int = 0):
        return build_prompt(
            prompt_type,
            target_mask=self.target,
            circle_mask=self.circle if circle else None,
            regions=self.regions if regions is None else regions,
            cfg=_only(prompt_type),
            rng=np.random.default_rng(seed),
        )

    def test_points_prompt(self) -> None:
        prompt = self._build(PromptType.POINTS, regions=[])
        self.assertGreaterEqual(prompt.point_coords.shape[0], 1)
        self.assertTrue((prompt.point_labels == 1).all())
        self.assertIsNone(prompt.box)
        self.assertIsNone(prompt.dense_logits)  # no negatives: learned no-mask embedding, not zeros
        self.assertIsNone(prompt.foreground_region)

    def test_points_prompt_carries_negatives_as_dense_mask(self) -> None:
        prompt = self._build(PromptType.POINTS)
        self.assertIsNotNone(prompt.dense_logits)
        self.assertEqual(prompt.dense_logits.shape, (LOW_RES, LOW_RES))
        self.assertEqual(set(np.unique(prompt.dense_logits)), {-MASK_PROMPT_LOGIT, 0.0})
        self.assertIsNone(prompt.foreground_region)

    def test_box_prompt_has_box_and_no_foreground_points(self) -> None:
        prompt = self._build(PromptType.BOX, regions=[])
        self.assertIsNotNone(prompt.box)
        self.assertEqual(prompt.point_coords.shape, (0, 2))
        self.assertEqual(prompt.point_labels.shape, (0,))

    def test_box_points_prompt_uses_one_to_max_points(self) -> None:
        counts = set()
        for seed in range(30):
            prompt = self._build(PromptType.BOX_POINTS, regions=[], seed=seed)
            self.assertIsNotNone(prompt.box)
            counts.add(prompt.point_coords.shape[0])
        self.assertTrue(counts <= {1, 2, 3} and len(counts) > 1)

    def test_circle_mask_prompt(self) -> None:
        prompt = self._build(PromptType.CIRCLE_MASK)
        self.assertEqual(prompt.point_coords.shape[0] - int((prompt.point_labels == 0).sum()), 1)
        x, y = prompt.point_coords[0]
        self.assertAlmostEqual(float(x), 31.5, places=3)
        self.assertAlmostEqual(float(y), 31.5, places=3)
        self.assertEqual(int(prompt.point_labels[0]), 1)
        self.assertIsNone(prompt.box)
        self.assertEqual(set(np.unique(prompt.dense_logits)), {-MASK_PROMPT_LOGIT, 0.0, MASK_PROMPT_LOGIT})
        self.assertEqual(int(prompt.foreground_region.sum()), 16 * 16)

    def test_circle_mask_overlapping_negative_becomes_unknown(self) -> None:
        overlap = np.zeros((64, 64), dtype=bool)
        overlap[24:32, 24:40] = True
        regions = [NegativeRegion(12, KIND_POLYGON, overlap)]
        prompt = self._build(PromptType.CIRCLE_MASK, regions=regions)
        self.assertEqual(int(prompt.foreground_region.sum()), 8 * 16)
        self.assertFalse(prompt.foreground_region[24:32, 24:40].any())

    def test_circle_mask_requires_a_circle(self) -> None:
        with self.assertRaises(ValueError):
            self._build(PromptType.CIRCLE_MASK, circle=False)

    def test_full_mask_prompt(self) -> None:
        prompt = self._build(PromptType.FULL_MASK, regions=[])
        self.assertEqual(prompt.point_coords.shape[0], 1)
        self.assertEqual(int(prompt.foreground_region.sum()), 32 * 32)
        self.assertEqual(set(np.unique(prompt.dense_logits)), {0.0, MASK_PROMPT_LOGIT})

    def test_negative_only_prompt_has_only_background_points(self) -> None:
        prompt = self._build(PromptType.NEGATIVE_ONLY)
        self.assertGreaterEqual(prompt.point_coords.shape[0], 1)
        self.assertTrue((prompt.point_labels == 0).all())
        self.assertIsNone(prompt.box)
        self.assertIsNone(prompt.foreground_region)
        self.assertIsNotNone(prompt.dense_logits)
        for x, y in prompt.point_coords:
            inside_neighbor = 2 <= x < 12 and 2 <= y < 12
            inside_circle = 50 <= x < 60 and 50 <= y < 60
            self.assertTrue(inside_neighbor or inside_circle)

    def test_negative_only_needs_a_negative(self) -> None:
        with self.assertRaises(ValueError):
            self._build(PromptType.NEGATIVE_ONLY, regions=[])

    def test_background_points_ride_along_by_probability(self) -> None:
        always = PromptConfig(
            prob_points_only=1.0,
            prob_box_only=0.0,
            prob_box_points=0.0,
            prob_circle_mask=0.0,
            prob_full_mask=0.0,
            prob_negative_only=0.0,
            bg_point_prob_circle=1.0,
            bg_point_prob_polygon=1.0,
        )
        prompt = build_prompt(
            PromptType.POINTS,
            target_mask=self.target,
            circle_mask=None,
            regions=self.regions,
            cfg=always,
            rng=np.random.default_rng(5),
        )
        self.assertTrue((prompt.point_labels == 0).any())
        self.assertTrue((prompt.point_labels == 1).any())
        never = PromptConfig(bg_point_prob_circle=0.0, bg_point_prob_polygon=0.0)
        prompt = build_prompt(
            PromptType.POINTS,
            target_mask=self.target,
            circle_mask=None,
            regions=self.regions,
            cfg=never,
            rng=np.random.default_rng(5),
        )
        self.assertTrue((prompt.point_labels == 1).all())


if __name__ == "__main__":
    unittest.main()
