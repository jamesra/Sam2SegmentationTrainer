"""Evaluation-only circle-seed options (docs/circle_seed_trust_eval.md): the radius-relative seed taper, the oversized
circle and the comparison script. CPU only, numpy only; the validation families that use them are tested in
``test_circle_seed_trust_families.py``."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np

from annotation_crops.negatives import rasterize_circle
from annotation_crops.prompt_encoding import (
    SEED_LOGIT,
    area_resize,
    circle_negative_canvas,
    encode_mask_input,
    negative_canvas,
)
from sam2_segmentation_trainer.data.circle_seed_trust import (
    DEFAULT_OVERSIZE_RANGE,
    OVERSIZE_STREAM,
    SeedCircle,
    check_oversize_range,
    oversize_circle,
    oversize_factor_range,
    seed_taper,
    seed_taper_weight,
    tapered_seed_mask_input,
)

SCALE = 0.95


def rho_map(shape, circle: SeedCircle) -> np.ndarray:
    """Distance of each pixel centre from the circle's centre, in units of its true radius."""
    ys, xs = np.mgrid[: shape[0], : shape[1]]
    return np.hypot(xs + 0.5 - circle.center_x, ys + 0.5 - circle.center_y) / circle.radius


class SeedTaperTests(unittest.TestCase):
    SHAPE = (120, 120)
    CIRCLE = SeedCircle(60.0, 60.0, 40.0)

    def test_the_taper_is_full_inside_full_frac_zero_beyond_zero_frac_and_a_half_cosine_between(self) -> None:
        taper = seed_taper(0.5, SCALE)
        weight = seed_taper_weight(self.SHAPE, self.CIRCLE, taper)
        rho = rho_map(self.SHAPE, self.CIRCLE)
        inner, outer = rho <= 0.5 - 1e-4, rho >= SCALE + 1e-4
        ramp = ~inner & ~outer & (rho > 0.5 + 1e-4) & (rho < SCALE - 1e-4)
        self.assertTrue(inner.any() and outer.any() and ramp.any())
        np.testing.assert_array_equal(weight[inner], 1.0)
        np.testing.assert_array_equal(weight[outer], 0.0)
        expected = 0.5 * (1.0 + np.cos(np.pi * (rho[ramp] - 0.5) / (SCALE - 0.5)))
        np.testing.assert_allclose(weight[ramp], expected, atol=1e-5)
        self.assertTrue(((weight >= 0.0) & (weight <= 1.0)).all())

    def test_the_fractions_are_validated_against_the_diameter_scale(self) -> None:
        self.assertEqual((seed_taper(0.5, SCALE).full_frac, seed_taper(0.5, SCALE).zero_frac), (0.5, SCALE))
        for bad in (SCALE, 1.0, -0.1):
            with self.assertRaisesRegex(ValueError, "circle-seed-taper"):
                seed_taper(bad, SCALE)

    def full_res(self, seed, weight, canvases=()) -> np.ndarray:
        # Area-averaging to the canvas's own size is the identity, so this is the full-resolution canvas.
        return tapered_seed_mask_input(seed, weight, list(canvases), SEED_LOGIT, size=self.SHAPE[0])

    def test_the_seed_is_fed_at_seed_logit_times_the_taper_over_the_scaled_disc(self) -> None:
        disc = rasterize_circle(self.SHAPE, self.CIRCLE.center_x, self.CIRCLE.center_y, self.CIRCLE.radius * SCALE)
        weight = seed_taper_weight(self.SHAPE, self.CIRCLE, seed_taper(0.5, SCALE))
        canvas = self.full_res(disc, weight)
        np.testing.assert_allclose(canvas[disc], SEED_LOGIT * weight[disc], rtol=0, atol=1e-5)
        np.testing.assert_array_equal(canvas[~disc], 0.0)
        self.assertAlmostEqual(float(canvas.max()), SEED_LOGIT, places=5)
        # The claimed region is the scaled disc: w is 0 outside it.
        np.testing.assert_array_equal(weight[~disc], 0.0)

    def test_a_pixel_claimed_by_the_seed_and_a_negative_is_unknown(self) -> None:
        disc = rasterize_circle(self.SHAPE, self.CIRCLE.center_x, self.CIRCLE.center_y, self.CIRCLE.radius * SCALE)
        weight = seed_taper_weight(self.SHAPE, self.CIRCLE, seed_taper(0.5, SCALE))
        neighbour = np.zeros(self.SHAPE, dtype=bool)
        neighbour[50:70, 50:110] = True  # over the trusted core and out past the disc
        polygon = negative_canvas(neighbour)
        circle = circle_negative_canvas(self.SHAPE, 10.0, 10.0, 20.0)
        canvas = self.full_res(disc, weight, [polygon, circle])
        overlap = disc & (neighbour | circle.mask)
        self.assertTrue(overlap.any())
        np.testing.assert_array_equal(canvas[overlap], 0.0)
        only_negative = neighbour & ~disc
        np.testing.assert_allclose(canvas[only_negative], polygon.canvas[only_negative], atol=1e-6)
        only_seed = disc & ~neighbour & ~circle.mask
        np.testing.assert_allclose(canvas[only_seed], SEED_LOGIT * weight[only_seed], atol=1e-5)

    def test_the_reduction_is_the_shared_area_average(self) -> None:
        disc = rasterize_circle(self.SHAPE, self.CIRCLE.center_x, self.CIRCLE.center_y, self.CIRCLE.radius * SCALE)
        weight = seed_taper_weight(self.SHAPE, self.CIRCLE, seed_taper(0.5, SCALE))
        low = tapered_seed_mask_input(disc, weight, [], SEED_LOGIT)
        self.assertEqual(low.shape, (256, 256))
        np.testing.assert_array_equal(low, area_resize(self.full_res(disc, weight), 256))

    def test_a_flat_weight_gives_the_untapered_mask_and_nothing_gives_none(self) -> None:
        disc = rasterize_circle(self.SHAPE, 60.0, 60.0, 30.0)
        neighbour = np.zeros(self.SHAPE, dtype=bool)
        neighbour[0:20, 0:20] = True
        ones = np.ones(self.SHAPE, dtype=np.float32)
        expected = encode_mask_input(self.SHAPE, seed=disc, negative_canvases=[negative_canvas(neighbour)], tau=0.0)
        got = tapered_seed_mask_input(disc, ones, [negative_canvas(neighbour)], SEED_LOGIT)
        np.testing.assert_array_equal(got, expected)
        self.assertIsNone(tapered_seed_mask_input(np.zeros(self.SHAPE, dtype=bool), ones, [], SEED_LOGIT))


class OversizeCircleTests(unittest.TestCase):
    SHAPE = (100, 100)
    CIRCLE = SeedCircle(50.5, 40.5, 20.0)

    def test_the_radius_is_the_true_radius_times_a_factor_in_range_and_the_centre_is_kept(self) -> None:
        factors = []
        for seed in range(200):
            disc, grown, factor = oversize_circle(
                self.SHAPE, self.CIRCLE, SCALE, np.random.default_rng([seed, 7, OVERSIZE_STREAM])
            )
            self.assertGreaterEqual(factor, DEFAULT_OVERSIZE_RANGE[0])
            self.assertLessEqual(factor, DEFAULT_OVERSIZE_RANGE[1])
            self.assertEqual((grown.center_x, grown.center_y), (self.CIRCLE.center_x, self.CIRCLE.center_y))
            self.assertAlmostEqual(grown.radius, self.CIRCLE.radius * factor)
            np.testing.assert_array_equal(disc, rasterize_circle(self.SHAPE, 50.5, 40.5, grown.radius * SCALE))
            factors.append(factor)
        self.assertGreater(max(factors) - min(factors), 0.5)  # the draw spans the range

    def test_the_same_generator_gives_the_same_circle(self) -> None:
        a = oversize_circle(self.SHAPE, self.CIRCLE, SCALE, np.random.default_rng([3, 1, 7, 2]), (1.3, 1.7))
        b = oversize_circle(self.SHAPE, self.CIRCLE, SCALE, np.random.default_rng([3, 1, 7, 2]), (1.3, 1.7))
        np.testing.assert_array_equal(a[0], b[0])
        self.assertEqual(a[1:], b[1:])

    def test_the_range_is_parsed_and_checked(self) -> None:
        self.assertEqual(oversize_factor_range("1.2,2.0"), (1.2, 2.0))
        self.assertEqual(oversize_factor_range("1.5,1.5"), (1.5, 1.5))
        for bad in ("2.0,1.2", "0,1", "x", "1.2", "1,2,3"):
            with self.assertRaises(ValueError):
                oversize_factor_range(bad)
        with self.assertRaises(ValueError):
            check_oversize_range((-1.0, 2.0))


class CompareScriptTests(unittest.TestCase):
    @staticmethod
    def script():
        path = Path(__file__).resolve().parents[1] / "scripts" / "compare_circle_seed_trust.py"
        spec = importlib.util.spec_from_file_location("compare_circle_seed_trust", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def summary(recall: float, leak: float, taper) -> dict:
        goal = {"n": 10, "core_recall": recall, "core_recall_p10": recall, "core_recall_ok": recall, "leak": leak,
                "merge_rate": 0.1, "spill": 0.2, "frag": 0.0}
        return {
            "model": "m.pt", "split": "s.json", "samples": 10,
            "circle_seed": {"taper": taper, "oversize_range": [1.2, 2.0]},
            "families": {"circle_mask": {"n": 10, "iou": 0.8, "goal": goal, "with_neg": None, "no_neg": None}},
        }

    def test_it_reports_each_metric_and_marks_better_and_worse(self) -> None:
        module = self.script()
        off = self.summary(0.90, 0.02, None)
        on = self.summary(0.95, 0.03, {"full_frac": 0.5, "zero_frac": 0.95, "ramp": "cosine"})
        text = module.compare(off, on, ["circle_mask", "circle_oversize"])
        self.assertIn("taper full_frac 0.5 zero_frac 0.95", text)
        recall = next(line for line in text.splitlines() if " core_recall " in line)
        leak = next(line for line in text.splitlines() if " leak " in line)
        merge = next(line for line in text.splitlines() if " merge_rate " in line)
        self.assertTrue(recall.endswith("+0.0500 +"), recall)
        self.assertTrue(leak.endswith("+0.0100 -"), leak)
        self.assertTrue(merge.rstrip().endswith("+0.0000"), merge)
        self.assertIn("circle_oversize: missing from the first run", text)


if __name__ == "__main__":
    unittest.main()
