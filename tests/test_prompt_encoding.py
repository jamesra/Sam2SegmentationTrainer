from __future__ import annotations

import math
import unittest

import numpy as np

from annotation_crops.prompt_encoding import (
    CORRECTION_LOGIT,
    EDGE_TAU_PX,
    LOGIT_CLIP,
    LOW_RES,
    SEED_LOGIT,
    negative_canvas,
    reach_px,
    area_resize,
    edge_confidence,
    edge_distance,
    effective_tau,
    encode_dense_canvas,
    encode_mask_input,
    signed_edge_distance,
)


def _square(size: int, lo: int, hi: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[lo:hi, lo:hi] = True
    return mask


class EdgeConfidenceTests(unittest.TestCase):
    def test_curve_values(self) -> None:
        mask = _square(400, 20, 380)
        conf = edge_confidence(mask, EDGE_TAU_PX)
        self.assertEqual(EDGE_TAU_PX, 8.0)
        self.assertEqual(float(conf[0, 0]), 0.0)  # outside
        # d = 0.5 px for the pixel next to the background, then one more pixel per step inward
        self.assertAlmostEqual(float(conf[200, 20]), 1 - math.exp(-0.5 / EDGE_TAU_PX), places=5)
        self.assertAlmostEqual(float(conf[200, 20 + 3]), 1 - math.exp(-3.5 / EDGE_TAU_PX), places=5)
        self.assertGreater(float(conf[200, 20 + 24]), 0.95)  # 3 tau inside the line
        self.assertGreater(float(conf[200, 200]), 0.999)

    def test_a_thin_process_gets_a_smaller_tau(self) -> None:
        bar = np.zeros((100, 200), dtype=bool)
        bar[40:60, 10:190] = True  # 20 px wide: maximum depth about 10
        depth = float(edge_distance(bar).max())
        self.assertAlmostEqual(depth, 9.5, places=3)
        self.assertAlmostEqual(effective_tau(depth), 0.25 * depth, places=5)
        self.assertLess(effective_tau(depth), 2.5)
        capped = edge_confidence(bar)
        uncapped = edge_confidence(bar, depth_frac=0.0)
        # The center of the bar: fully trusted with the cap, still partly edge without it.
        self.assertGreater(float(capped[50, 100]), 0.95)
        self.assertLess(float(uncapped[50, 100]), 0.75)
        # Five pixels in from the edge the difference is larger.
        self.assertGreater(float(capped[45, 100]), 0.85)
        self.assertLess(float(uncapped[45, 100]), 0.55)
        big = _square(400, 20, 380)
        self.assertEqual(effective_tau(float(edge_distance(big).max())), 8.0)

    def test_every_mask_gets_its_own_cap(self) -> None:
        bar = np.zeros((100, 300), dtype=bool)
        bar[40:60, 10:100] = True  # thin
        bar[10:90, 150:290] = True  # wide
        wide = bar.copy()
        wide[:, :120] = False
        thin = bar.copy()
        thin[:, 120:] = False
        canvas = encode_dense_canvas(bar.shape, seed=thin, negatives=[wide])
        # The thin seed reaches full trust at its center; the wide negative uses its own, larger tau.
        self.assertGreater(float(canvas[50, 55]), 0.8 * SEED_LOGIT)
        self.assertLess(float(canvas[50, 160]), 0.0)

    def test_the_array_border_is_not_an_edge(self) -> None:
        mask = np.zeros((64, 64), dtype=bool)
        mask[:, :32] = True  # touches three borders of the tile
        conf = edge_confidence(mask, EDGE_TAU_PX)
        self.assertGreater(float(conf[0, 0]), 0.97)  # far from the one real edge, and not near the border
        self.assertLess(float(conf[10, 31]), 0.2)

    def test_degenerate_masks(self) -> None:
        self.assertEqual(float(edge_confidence(np.zeros((8, 8), bool)).max()), 0.0)
        self.assertEqual(float(edge_confidence(np.ones((8, 8), bool)).min()), 1.0)
        self.assertTrue(np.isinf(edge_distance(np.ones((4, 4), bool))).all())

    def test_tau_zero_gives_the_plain_mask(self) -> None:
        mask = _square(32, 8, 24)
        self.assertTrue(np.array_equal(edge_confidence(mask, 0.0), mask.astype(np.float32)))

    def test_signed_edge_distance_counts_both_sides(self) -> None:
        mask = _square(64, 16, 48)
        d = signed_edge_distance(mask)
        self.assertAlmostEqual(float(d[32, 16]), 0.5)
        self.assertAlmostEqual(float(d[32, 15]), 0.5)
        self.assertAlmostEqual(float(d[32, 13]), 2.5)
        self.assertTrue(np.isinf(signed_edge_distance(np.ones((4, 4), bool))).all())


class CroppedDistanceTests(unittest.TestCase):
    """The bounding-box computations must match a transform of the whole tile."""

    def _shapes(self) -> list[np.ndarray]:
        yy, xx = np.mgrid[0:200, 0:240]
        disc = (yy - 90) ** 2 + (xx - 120) ** 2 <= 40**2
        corner = np.zeros((200, 240), dtype=bool)
        corner[:60, 190:] = True  # touches two borders
        ring = disc & ~((yy - 90) ** 2 + (xx - 120) ** 2 <= 15**2)
        return [disc, corner, ring]

    def test_edge_distance_is_exact(self) -> None:
        from scipy.ndimage import distance_transform_edt

        for mask in self._shapes():
            expected = np.maximum(distance_transform_edt(mask) - 0.5, 0.0) * mask
            np.testing.assert_allclose(edge_distance(mask), expected, atol=1e-5)

    def test_signed_distance_with_reach_matches_inside_the_reach(self) -> None:
        reach = reach_px(EDGE_TAU_PX)

        def weight(d: np.ndarray) -> np.ndarray:
            return np.maximum(0.05, 1.0 - np.exp(-d / EDGE_TAU_PX))

        for mask in self._shapes():
            full = signed_edge_distance(mask)
            limited = signed_edge_distance(mask, reach)
            near = full <= reach
            np.testing.assert_allclose(limited[near], full[near], atol=1e-5)
            # Beyond 3 tau the weight is within 5% of 1, and the limited map says exactly 1 there.
            self.assertLess(float(np.abs(weight(limited) - weight(full)).max()), 0.051)

    def test_reach_is_three_tau(self) -> None:
        self.assertEqual(reach_px(2.5), 8)
        self.assertEqual(reach_px(8.0), 24)


class DenseCanvasTests(unittest.TestCase):
    def test_seed_correction_and_negative_scales(self) -> None:
        seed = _square(200, 20, 100)
        other = _square(200, 120, 180)
        canvas = encode_dense_canvas(seed.shape, seed=seed, negatives=[other], tau=2.5)
        self.assertAlmostEqual(float(canvas[60, 60]), SEED_LOGIT, delta=0.2)
        self.assertAlmostEqual(float(canvas[150, 150]), -SEED_LOGIT, delta=0.2)
        self.assertEqual(float(canvas[5, 5]), 0.0)
        # Each region is scaled by its own edge distance.
        self.assertLess(float(canvas[60, 20]), 0.3 * SEED_LOGIT)
        self.assertGreater(float(canvas[150, 120]), -0.3 * SEED_LOGIT)
        corrected = encode_dense_canvas(seed.shape, correction=seed, tau=2.5)
        self.assertAlmostEqual(float(corrected[60, 60]), CORRECTION_LOGIT, delta=0.1)

    def test_overlap_sums_and_clips(self) -> None:
        a = _square(100, 10, 90)
        canvas = encode_dense_canvas(a.shape, seed=a, correction=a, tau=2.5)
        self.assertAlmostEqual(float(canvas[50, 50]), LOGIT_CLIP, places=4)  # 10 + 3 clipped
        both = encode_dense_canvas(a.shape, seed=a, negatives=[a], tau=2.5)
        self.assertAlmostEqual(float(both[50, 50]), 0.0, places=4)

    def test_area_downsampling_averages(self) -> None:
        canvas = np.zeros((1024, 1024), dtype=np.float32)
        canvas[:, :2] = 8.0
        small = area_resize(canvas)
        self.assertEqual(small.shape, (LOW_RES, LOW_RES))
        self.assertAlmostEqual(float(small[0, 0]), 4.0, places=4)  # 2 of the 4 columns in the cell
        self.assertEqual(float(small[0, 5]), 0.0)
        flat = area_resize(np.full((100, 100), 3.0, dtype=np.float32))
        np.testing.assert_allclose(flat, 3.0, atol=1e-5)

    def test_shared_negative_canvases_match_recomputing_them(self) -> None:
        seed = _square(200, 20, 100)
        others = [_square(200, 120, 180), _square(200, 110, 130)]
        direct = encode_mask_input(seed.shape, seed=seed, negatives=others)
        shared = encode_mask_input(
            seed.shape, seed=seed, negative_canvases=[negative_canvas(o) for o in others]
        )
        np.testing.assert_allclose(direct, shared, atol=1e-6)

    def test_all_zero_prompt_is_none(self) -> None:
        self.assertIsNone(encode_mask_input((64, 64)))
        encoded = encode_mask_input((64, 64), seed=_square(64, 8, 56))
        self.assertEqual(encoded.shape, (LOW_RES, LOW_RES))
        self.assertEqual(encoded.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
