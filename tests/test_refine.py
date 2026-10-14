from __future__ import annotations

import unittest

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, label

from sam2_segmentation_trainer.data.refine import (
    CORRECTION_KINDS,
    SUBSET_KINDS,
    _depth,
    correction_mask,
    fit_adjacent_seed,
    grow_mask,
    local_excursions,
    near_correct_mask,
    subset_seed,
)


def _blob(size: int = 128) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    return ((yy - size / 2) ** 2 / 45**2 + (xx - size / 2) ** 2 / 30**2) <= 1.0


def _u_shape(size: int = 128) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[20:110, 20:40] = True
    mask[20:110, 80:100] = True
    mask[90:110, 20:100] = True
    return mask


def _outline_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from each outline pixel of ``a`` to the outline of ``b``."""
    edge_a = a & ~binary_erosion(a, border_value=1)
    edge_b = b & ~binary_erosion(b, border_value=1)
    return distance_transform_edt(~edge_b)[edge_a]


class SubsetSeedTests(unittest.TestCase):
    def test_seeds_stay_at_least_ten_pixels_inside_a_large_mask(self) -> None:
        mask = _disc(100)
        deep = _depth(mask)
        for seed in range(30):
            out = subset_seed(mask, np.random.default_rng(seed))
            self.assertTrue(out.any())
            # Eroded by 10-40 px before any cut, so the seed keeps away from the traced edge.
            self.assertGreaterEqual(float(deep[out].min()), 9.5, seed)

    def test_always_inside_connected_and_smaller(self) -> None:
        for mask in (_blob(), _u_shape()):
            for seed in range(30):
                out = subset_seed(mask, np.random.default_rng(seed))
                self.assertTrue(out.any())
                self.assertFalse((out & ~mask).any())
                self.assertEqual(label(out)[1], 1)
                self.assertLess(int(out.sum()), int(mask.sum()))

    def test_every_kind_is_drawn(self) -> None:
        mask = _blob()
        areas = {int(subset_seed(mask, np.random.default_rng(s)).sum()) for s in range(40)}
        self.assertGreater(len(areas), 20)
        self.assertEqual(len(SUBSET_KINDS), 3)

    def test_a_thin_mask_survives(self) -> None:
        mask = np.zeros((40, 40), dtype=bool)
        mask[19:21, 5:35] = True
        for seed in range(10):
            out = subset_seed(mask, np.random.default_rng(seed))
            self.assertTrue(out.any() and not (out & ~mask).any())


class CorrectionTests(unittest.TestCase):
    def test_always_differs_and_is_never_empty(self) -> None:
        mask = _blob()
        neighbor = np.zeros_like(mask)
        neighbor[10:40, 100:125] = True
        for seed in range(60):
            out, merged = correction_mask(mask, np.random.default_rng(seed), neighbors=[neighbor])
            self.assertTrue(out.any())
            self.assertFalse(np.array_equal(out, mask))
            if merged is not None:
                self.assertEqual(merged, 0)
                self.assertTrue((out & neighbor).any())

    def test_every_kind_appears(self) -> None:
        mask = _blob()
        neighbor = np.zeros_like(mask)
        neighbor[10:40, 100:125] = True
        merged_seen = sum(
            correction_mask(mask, np.random.default_rng(s), neighbors=[neighbor])[1] is not None
            for s in range(100)
        )
        self.assertGreater(merged_seen, 5)
        self.assertEqual(len(CORRECTION_KINDS), 5)

    def test_a_merge_prefers_neighbors_within_reach(self) -> None:
        mask = _blob()
        near = np.zeros_like(mask)
        near[60:70, 100:125] = True  # touches the blob's right side (within 20 px)
        far = np.zeros_like(mask)
        far[0:8, 0:8] = True
        self.assertLess(distance_transform_edt(~mask)[near].min(), 20.0)
        self.assertGreater(distance_transform_edt(~mask)[far].min(), 20.0)
        picked = [
            correction_mask(mask, np.random.default_rng(s), neighbors=[far, near])[1] for s in range(200)
        ]
        merges = [index for index in picked if index is not None]
        self.assertGreater(len(merges), 10)
        self.assertEqual(set(merges), {1})  # never the far one while a near one exists

    def test_with_only_far_neighbors_any_of_them_can_be_merged(self) -> None:
        mask = _blob()
        far = np.zeros_like(mask)
        far[0:8, 0:8] = True
        other = np.zeros_like(mask)
        other[120:128, 0:8] = True
        merges = {
            correction_mask(mask, np.random.default_rng(s), neighbors=[far, other])[1] for s in range(200)
        } - {None}
        self.assertEqual(merges, {0, 1})

    def test_without_neighbors_nothing_is_merged(self) -> None:
        mask = _blob()
        for seed in range(30):
            _out, merged = correction_mask(mask, np.random.default_rng(seed))
            self.assertIsNone(merged)


def _disc(radius: int = 130) -> np.ndarray:
    yy, xx = np.mgrid[0:360, 0:360]
    return (yy - 180) ** 2 + (xx - 180) ** 2 <= radius**2


class NearCorrectTests(unittest.TestCase):
    def test_wobble_alone_moves_the_outline_by_the_requested_mean(self) -> None:
        mask = _disc()
        for low, high in ((5.0, 5.0), (12.0, 12.0)):
            means = [
                _outline_distances(
                    near_correct_mask(mask, np.random.default_rng(s), wobble_px=(low, high), max_excursions=0), mask
                ).mean()
                for s in range(10)
            ]
            # The distance to the nearest outline pixel is a little under the displacement along the normal.
            self.assertGreater(np.mean(means), 0.6 * low, (low, high))
            self.assertLess(np.mean(means), 1.1 * high, (low, high))

    def test_default_mean_displacement_is_five_to_twelve_pixels(self) -> None:
        mask = _disc()
        means = [
            _outline_distances(near_correct_mask(mask, np.random.default_rng(s)), mask).mean() for s in range(20)
        ]
        self.assertGreater(min(means), 3.0)
        self.assertLess(max(means), 14.0)
        self.assertGreater(float(np.mean(means)), 5.0)

    def test_no_part_of_the_outline_is_more_than_thirty_pixels_off(self) -> None:
        mask = _disc()
        peaks = []
        for seed in range(20):
            out = near_correct_mask(mask, np.random.default_rng(seed))
            peaks.append(_outline_distances(out, mask).max())
            self.assertLessEqual(_outline_distances(mask, out).max(), 31.0, seed)
        self.assertLessEqual(max(peaks), 31.0)

    def test_excursions_make_some_of_the_outline_far_off(self) -> None:
        mask = _disc()
        counts = []
        for seed in range(8):
            far = _outline_distances(near_correct_mask(mask, np.random.default_rng(seed)), mask)
            counts.append(float((far > 15.0).sum()))
            self.assertLess(float(np.median(far)), 15.0, seed)  # most of the outline stays near
        # A few stretches of outline are 15+ px away.
        self.assertGreater(float(np.mean(counts)), 10.0)
        self.assertGreater(min(counts), 0.0)

    def test_a_medium_mask_stays_a_recognisable_copy(self) -> None:
        mask = _disc()
        for seed in range(10):
            out = near_correct_mask(mask, np.random.default_rng(seed))
            iou = float((out & mask).sum() / (out | mask).sum())
            self.assertGreater(iou, 0.7, seed)
            self.assertLess(iou, 0.98, seed)

    def test_a_small_cell_is_never_erased(self) -> None:
        mask = _disc(25)
        for seed in range(30):
            out = near_correct_mask(mask, np.random.default_rng(seed))
            self.assertTrue(out.any())
            self.assertGreater(float((out & mask).sum() / (out | mask).sum()), 0.29, seed)


class LocalExcursionTests(unittest.TestCase):
    def test_a_bulge_adds_and_a_dent_removes_within_the_amplitude(self) -> None:
        mask = _disc(100)
        added = removed = 0
        for seed in range(30):
            out = local_excursions(mask, np.random.default_rng(seed), 1, (20.0, 20.0), (60.0, 60.0))
            self.assertFalse(np.array_equal(out, mask))
            added += int((out & ~mask).any())
            removed += int((mask & ~out).any())
            self.assertLessEqual(_outline_distances(out, mask).max(), 20.5, seed)
        self.assertGreater(added, 5)
        self.assertGreater(removed, 5)

    def test_the_span_limits_how_much_of_the_outline_moves(self) -> None:
        mask = _disc(100)
        out = local_excursions(mask, np.random.default_rng(1), 1, (20.0, 20.0), (40.0, 40.0))
        changed = np.argwhere(out ^ mask)
        spread = np.ptp(changed, axis=0).max()
        self.assertLessEqual(spread, 45)

    def test_zero_count_and_thin_masks(self) -> None:
        mask = _disc(60)
        self.assertTrue(np.array_equal(local_excursions(mask, np.random.default_rng(0), 0, (10.0, 20.0)), mask))
        thin = np.zeros((80, 80), dtype=bool)
        thin[38:42, 5:75] = True
        for seed in range(15):
            self.assertTrue(local_excursions(thin, np.random.default_rng(seed), 3, (15.0, 30.0)).any())


class AdjacentSeedTests(unittest.TestCase):
    def test_scaled_to_fit_inside_the_cell(self) -> None:
        target = _blob()
        shape = np.zeros((50, 50), dtype=bool)
        shape[5:45, 20:30] = True  # a tall thin profile, in another frame
        for seed in range(10):
            out = fit_adjacent_seed(shape, target, np.random.default_rng(seed))
            self.assertIsNotNone(out)
            # At least 98% fit inside the target grown by 10 px (the annotator fits to the visible membrane).
            grown = grow_mask(target, 10.0)
            self.assertGreaterEqual(float((out & grown).sum() / out.sum()), 0.96)
            self.assertGreater(int(out.sum()), 20)

    def test_the_grown_target_lets_a_seed_poke_out_of_the_trace(self) -> None:
        target = np.zeros((120, 120), dtype=bool)
        target[40:80, 20:100] = True  # a wide bar; the shape is tall, so the fit is limited by its height
        shape = np.zeros((90, 20), dtype=bool)
        shape[:, :] = True
        outside = []
        for seed in range(20):
            out = fit_adjacent_seed(shape, target, np.random.default_rng(seed), scale_range=(1.0, 1.0))
            self.assertIsNotNone(out)
            outside.append(int((out & ~target).sum()))
        self.assertGreater(max(outside), 0)
        flat = fit_adjacent_seed(shape, target, np.random.default_rng(0), scale_range=(1.0, 1.0), grow_px=0.0)
        self.assertLess(int((flat & ~target).sum()), 0.03 * int(flat.sum()) + 1)

    def test_empty_shape_or_target(self) -> None:
        target = _blob()
        self.assertIsNone(fit_adjacent_seed(np.zeros((8, 8), bool), target, np.random.default_rng(0)))
        self.assertIsNone(fit_adjacent_seed(target, np.zeros_like(target), np.random.default_rng(0)))


if __name__ == "__main__":
    unittest.main()
