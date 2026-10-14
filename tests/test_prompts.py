from __future__ import annotations

import unittest

import numpy as np
from scipy.ndimage import distance_transform_edt

from annotation_crops.negatives import KIND_CIRCLE, KIND_POLYGON, NegativeRegion
from sam2_segmentation_trainer.data.prompts import (
    FULL_TILE_PROMPTS,
    LOW_RES,
    PromptConfig,
    PromptType,
    benford_count,
    build_prompt,
    choose_prompt_type,
    jittered_box,
    sample_points_in_mask,
)

CFG = PromptConfig()
POINT_ARGS = dict(min_edge_px=15, line_frac_min=0.4, line_frac_max=0.9)
ZERO = {
    "prob_points_only": 0.0,
    "prob_box_only": 0.0,
    "prob_box_points": 0.0,
    "prob_circle_seed": 0.0,
    "prob_adjacent_seed": 0.0,
    "prob_subset_seed": 0.0,
    "prob_correction": 0.0,
    "prob_near_correct": 0.0,
}
_FIELD = {
    PromptType.POINTS: "prob_points_only",
    PromptType.BOX: "prob_box_only",
    PromptType.BOX_POINTS: "prob_box_points",
    PromptType.CIRCLE_SEED: "prob_circle_seed",
    PromptType.ADJACENT_SEED: "prob_adjacent_seed",
    PromptType.SUBSET_SEED: "prob_subset_seed",
    PromptType.CORRECTION: "prob_correction",
    PromptType.NEAR_CORRECT: "prob_near_correct",
}


def _square(size: int, lo: int, hi: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[lo:hi, lo:hi] = True
    return mask


def _only(prompt_type: PromptType, **extra) -> PromptConfig:
    values = dict(ZERO)
    values[_FIELD[prompt_type]] = 1.0
    values.update(extra)
    return PromptConfig(**values)


class PromptConfigTests(unittest.TestCase):
    def test_defaults_sum_to_one(self) -> None:
        CFG.validate()
        self.assertAlmostEqual(sum(CFG.probabilities().values()), 1.0)
        self.assertEqual(CFG.variants_per_sample, 3)

    def test_rejects_bad_sum_negative_and_inverted_range(self) -> None:
        with self.assertRaises(ValueError):
            PromptConfig(prob_points_only=0.5).validate()
        with self.assertRaises(ValueError):
            PromptConfig(prob_points_only=-0.1, prob_circle_seed=0.65).validate()
        with self.assertRaises(ValueError):
            PromptConfig(line_frac_min=0.9, line_frac_max=0.4).validate()
        with self.assertRaises(ValueError):
            PromptConfig(box_points_max=0).validate()
        with self.assertRaises(ValueError):
            PromptConfig(variants_per_sample=0).validate()
        with self.assertRaises(ValueError):
            PromptConfig(erode_px=(30.0, 3.0)).validate()

    def test_from_cfg_merges_sections_and_keeps_defaults(self) -> None:
        cfg = {
            "prompt": {
                "prob_points_only": 0.15,
                "prob_circle_seed": 0.35,
                "box_points_max": 2,
                "erode_px": [2, 20],
            },
            "negative": {"cache_size": 8, "drop_overlapping_circles": True},
        }
        config = PromptConfig.from_cfg(cfg)
        self.assertEqual(config.prob_points_only, 0.15)
        self.assertEqual(config.box_points_max, 2)
        self.assertEqual(config.erode_px, (2.0, 20.0))
        self.assertEqual(config.cache_size, 8)
        self.assertIs(config.drop_overlapping_circles, True)
        self.assertEqual(config.prob_adjacent_seed, 0.15)
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


class ChoosePromptTypeTests(unittest.TestCase):
    def test_adjacent_share_moves_to_the_circle_seed_until_shapes_exist(self) -> None:
        rng = np.random.default_rng(0)
        drawn = [choose_prompt_type(CFG, rng) for _ in range(6000)]
        self.assertNotIn(PromptType.ADJACENT_SEED, drawn)
        share = drawn.count(PromptType.CIRCLE_SEED) / len(drawn)
        self.assertAlmostEqual(share, 0.55, delta=0.04)

    def test_everything_available_draws_every_type(self) -> None:
        rng = np.random.default_rng(1)
        drawn = {choose_prompt_type(CFG, rng, can_adjacent_seed=True) for _ in range(4000)}
        self.assertEqual(drawn, {kind for kind, share in CFG.probabilities().items() if share > 0})

    def test_exclude_draws_without_replacement(self) -> None:
        rng = np.random.default_rng(2)
        drawn = {
            choose_prompt_type(CFG, rng, exclude={PromptType.CIRCLE_SEED, PromptType.POINTS})
            for _ in range(1000)
        }
        self.assertFalse(drawn & {PromptType.CIRCLE_SEED, PromptType.POINTS})

    def test_allowed_limits_the_draw_and_falls_back(self) -> None:
        rng = np.random.default_rng(3)
        drawn = {choose_prompt_type(CFG, rng, allowed=FULL_TILE_PROMPTS) for _ in range(400)}
        self.assertEqual(drawn, set(FULL_TILE_PROMPTS))
        only_box = _only(PromptType.BOX)
        self.assertIs(choose_prompt_type(only_box, rng, allowed=FULL_TILE_PROMPTS), PromptType.POINTS)
        self.assertIs(
            choose_prompt_type(only_box, rng, allowed={PromptType.CIRCLE_SEED}), PromptType.CIRCLE_SEED
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

    def _build(self, prompt_type: PromptType, regions=None, circle=True, seed: int = 0, **extra):
        return build_prompt(
            prompt_type,
            target_mask=self.target,
            circle_mask=self.circle if circle else None,
            regions=self.regions if regions is None else regions,
            cfg=_only(prompt_type, **extra),
            rng=np.random.default_rng(seed),
        )

    def test_points_prompt(self) -> None:
        prompt = self._build(PromptType.POINTS, regions=[])
        self.assertGreaterEqual(prompt.point_coords.shape[0], 1)
        self.assertTrue((prompt.point_labels == 1).all())
        self.assertIsNone(prompt.box)
        self.assertIsNone(prompt.dense_logits)  # no negatives: learned no-mask embedding, not zeros
        self.assertIsNone(prompt.foreground_weight)
        self.assertIsNone(prompt.input_mask)

    def test_points_prompt_carries_negatives_as_dense_mask(self) -> None:
        prompt = self._build(PromptType.POINTS)
        self.assertEqual(prompt.dense_logits.shape, (LOW_RES, LOW_RES))
        self.assertLess(float(prompt.dense_logits.min()), -5.0)
        self.assertLessEqual(float(prompt.dense_logits.max()), 0.0)
        self.assertIsNone(prompt.foreground_weight)

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

    def test_circle_seed_prompt(self) -> None:
        prompt = self._build(PromptType.CIRCLE_SEED, regions=[])
        self.assertEqual(prompt.point_coords.shape[0], 1)
        x, y = prompt.point_coords[0]
        self.assertAlmostEqual(float(x), 31.5, places=3)
        self.assertAlmostEqual(float(y), 31.5, places=3)
        self.assertEqual(int(prompt.point_labels[0]), 1)
        self.assertIsNone(prompt.box)
        # The seed core is fully trusted; its traced edge is not.
        self.assertGreater(float(prompt.dense_logits.max()), 5.0)
        self.assertLessEqual(float(prompt.dense_logits.max()), 10.0)
        self.assertTrue(np.array_equal(prompt.input_mask, self.circle))
        weight = prompt.foreground_weight
        self.assertGreater(float(weight[31, 31]), 0.95)
        self.assertLess(float(weight[24, 31]), 0.3)
        self.assertEqual(float(weight[0, 0]), 0.0)

    def test_circle_seed_overlapping_negative_loses_its_trust(self) -> None:
        overlap = np.zeros((64, 64), dtype=bool)
        overlap[24:32, 24:40] = True
        regions = [NegativeRegion(12, KIND_POLYGON, overlap)]
        prompt = self._build(PromptType.CIRCLE_SEED, regions=regions)
        self.assertFalse(prompt.foreground_weight[24:32, 24:40].any())
        self.assertTrue(prompt.foreground_weight[34:38, 28:36].all())

    def test_circle_seed_requires_a_circle(self) -> None:
        with self.assertRaises(ValueError):
            self._build(PromptType.CIRCLE_SEED, circle=False)

    def test_subset_seed_stays_inside_the_truth_and_is_trusted(self) -> None:
        for seed in range(12):
            prompt = self._build(PromptType.SUBSET_SEED, regions=[], seed=seed)
            seed_mask = prompt.input_mask
            self.assertTrue(seed_mask.any())
            self.assertFalse((seed_mask & ~self.target).any())
            self.assertIsNotNone(prompt.foreground_weight)
            x, y = prompt.point_coords[0]
            self.assertTrue(seed_mask[int(round(y)), int(round(x))])

    def test_correction_differs_from_truth_and_is_not_trusted(self) -> None:
        for seed in range(12):
            prompt = self._build(PromptType.CORRECTION, regions=[], seed=seed)
            self.assertFalse(np.array_equal(prompt.input_mask, self.target))
            self.assertIsNone(prompt.foreground_weight)  # the containment loss never sees a mask to correct
            self.assertLessEqual(float(prompt.dense_logits.max()), 3.0 + 1e-4)
            self.assertGreater(float(prompt.dense_logits.max()), 0.0)

    def test_near_correct_stays_within_three_pixels_of_the_truth(self) -> None:
        inside = distance_transform_edt(self.target)
        outside = distance_transform_edt(~self.target)
        for seed in range(12):
            prompt = self._build(PromptType.NEAR_CORRECT, regions=[], seed=seed)
            self.assertIsNone(prompt.foreground_weight)
            moved = prompt.input_mask ^ self.target
            # No excursion goes more than 30 px (plus a pixel of rounding) from the truth's outline.
            self.assertTrue(np.all(np.where(self.target, inside, outside)[moved] <= 31.0))

    def test_correction_click_comes_from_the_mask_to_correct_not_the_truth(self) -> None:
        for kind in (PromptType.CORRECTION, PromptType.NEAR_CORRECT):
            for seed in range(25):
                prompt = self._build(kind, regions=[], seed=seed)
                x, y = prompt.point_coords[0]
                shown = prompt.input_mask
                self.assertTrue(shown[int(round(y)), int(round(x))], (kind, seed))
                # It is the deepest interior pixel of what the user sees.
                depth = distance_transform_edt(shown)  # the array border is not an edge
                self.assertAlmostEqual(float(depth[int(round(y)), int(round(x))]), float(depth.max()), places=4)

    def test_correction_click_can_lie_outside_the_truth(self) -> None:
        # A mask to correct that swallowed a big neighbor has its deepest point in the neighbor, not the truth.
        target = _square(120, 10, 30)
        region = NegativeRegion(10, KIND_POLYGON, _square(120, 40, 110))
        cfg = _only(PromptType.CORRECTION)
        outside = 0
        for seed in range(80):
            prompt = build_prompt(
                PromptType.CORRECTION, target_mask=target, circle_mask=None, regions=[region], cfg=cfg,
                rng=np.random.default_rng(seed),
            )
            x, y = prompt.point_coords[0]
            self.assertTrue(prompt.input_mask[int(round(y)), int(round(x))])
            outside += int(not target[int(round(y)), int(round(x))])
        self.assertGreater(outside, 0)

    def test_background_points_never_land_in_the_merged_neighbor(self) -> None:
        target = _square(200, 20, 60)
        neighbor = _square(200, 150, 190)
        other = _square(200, 90, 120)
        regions = [NegativeRegion(10, KIND_POLYGON, neighbor), NegativeRegion(11, KIND_POLYGON, other)]
        cfg = _only(PromptType.CORRECTION, bg_point_prob_polygon=1.0, bg_point_prob_circle=1.0)
        merged_seen = 0
        for seed in range(80):
            prompt = build_prompt(
                PromptType.CORRECTION, target_mask=target, circle_mask=None, regions=regions, cfg=cfg,
                rng=np.random.default_rng(seed),
            )
            if not (prompt.input_mask & neighbor).any():
                continue
            merged_seen += 1
            background = prompt.point_coords[prompt.point_labels == 0]
            self.assertGreater(len(background), 0)  # the other neighbor still gets its points
            self.assertFalse(any(neighbor[int(round(y)), int(round(x))] for x, y in background), seed)
        self.assertGreater(merged_seen, 3)

    def test_neighbor_merge_leaves_the_merged_neighbor_out_of_the_negative_context(self) -> None:
        target = _square(200, 20, 60)
        neighbor = _square(200, 150, 190)  # too far away for a grow or scale to reach it
        region = NegativeRegion(10, KIND_POLYGON, neighbor)
        cfg = _only(PromptType.CORRECTION)
        merged_seen = 0
        for seed in range(80):
            prompt = build_prompt(
                PromptType.CORRECTION,
                target_mask=target,
                circle_mask=None,
                regions=[region],
                cfg=cfg,
                rng=np.random.default_rng(seed),
            )
            if (prompt.input_mask & neighbor).any():
                merged_seen += 1
                # The neighbor is part of the error, so it is not also labelled background.
                self.assertGreaterEqual(float(prompt.dense_logits.min()), 0.0)
            else:
                self.assertLess(float(prompt.dense_logits.min()), -5.0)
        self.assertGreater(merged_seen, 3)

    def test_adjacent_seed_fits_inside_or_falls_back_to_the_circle(self) -> None:
        shape = np.zeros((30, 30), dtype=bool)
        shape[5:25, 8:22] = True
        prompt = build_prompt(
            PromptType.ADJACENT_SEED,
            target_mask=self.target,
            circle_mask=self.circle,
            regions=[],
            cfg=_only(PromptType.ADJACENT_SEED),
            rng=np.random.default_rng(0),
            adjacent_shape=shape,
        )
        self.assertIs(prompt.prompt_type, PromptType.ADJACENT_SEED)
        grown = distance_transform_edt(~self.target) <= 10.0
        self.assertGreaterEqual(float((prompt.input_mask & grown).sum()) / prompt.input_mask.sum(), 0.97)
        missing = self._build(PromptType.ADJACENT_SEED, regions=[])
        self.assertIs(missing.prompt_type, PromptType.CIRCLE_SEED)

    def test_background_points_ride_along_by_probability(self) -> None:
        always = _only(PromptType.POINTS, bg_point_prob_circle=1.0, bg_point_prob_polygon=1.0)
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
