from __future__ import annotations

import unittest

import numpy as np

from annotation_crops.prompt_encoding import edge_distance
from sam2_segmentation_trainer.data.refine import grow_mask
from sam2_segmentation_trainer.data.weights import neighbor_core_labels
from sam2_segmentation_trainer.goal_metrics import (
    EARLY_STOPPING_MIN_DELTA,
    RECALL_MIN_DELTA,
    GoalSummary,
    SelectionKey,
    add_goal_fields,
    improves,
    sample_goal_metrics,
    sample_no_harm,
    selection_key,
    summarize_goal,
)


def _box(size: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


GT = _box(300, 60, 160, 60, 160)  # 100 x 100: its core (>= 10 px inside) is 80 x 80 (the edge distance is 9.5 at the rim + 10)


class SampleMetricsTests(unittest.TestCase):
    def test_a_perfect_prediction(self) -> None:
        m = sample_goal_metrics(GT, GT)
        self.assertEqual(m["core_hit"], m["core_gt"])
        self.assertGreater(m["core_gt"], 0)
        self.assertEqual(m["recall"], 1.0)
        self.assertEqual((m["leak_px"], m["spill_px"]), (0, 0))
        self.assertFalse(m["merge"])
        self.assertFalse(m["frag"])

    def test_core_recall_ignores_the_edge_band(self) -> None:
        core_depth = float(edge_distance(GT).max())
        self.assertGreater(core_depth, 40)
        inside_band = GT & (edge_distance(GT) >= 8.0)  # misses the outer 8 px only
        self.assertEqual(sample_goal_metrics(inside_band, GT)["recall"], 1.0)
        too_small = GT & (edge_distance(GT) >= 18.0)
        recall = sample_goal_metrics(too_small, GT)["recall"]
        self.assertLess(recall, 0.8)
        self.assertGreater(recall, 0.4)

    def test_a_thin_ground_truth_has_no_core(self) -> None:
        thin = _box(300, 100, 112, 20, 280)  # 12 px wide: nothing is 10 px inside
        m = sample_goal_metrics(thin, thin)
        self.assertEqual(m["core_gt"], 0)
        self.assertIsNone(m["recall"])

    def test_spill_counts_only_pixels_more_than_the_band_outside(self) -> None:
        near = grow_mask(GT, 9.0)
        self.assertEqual(sample_goal_metrics(near, GT)["spill_px"], 0)
        far = grow_mask(GT, 25.0)
        m = sample_goal_metrics(far, GT)
        self.assertGreater(m["spill_px"], 0)
        outside_band = far & ~grow_mask(GT, 10.0)
        self.assertEqual(m["spill_px"], int(outside_band.sum()))
        self.assertGreater(m["spill_px"] / m["gt_area"], 0.5)

    def test_leak_and_merge_use_the_neighbor_cores(self) -> None:
        neighbor = _box(300, 60, 160, 180, 280)
        labels = neighbor_core_labels([neighbor], GT, 10.0)
        core = labels == 1
        self.assertGreater(int(core.sum()), 5000)
        rows, cols = np.nonzero(core)
        # Covers 3% of the neighbor's core: leaks, but is not a merge (needs 5%).
        few = GT.copy()
        order = np.argsort(rows)
        take = int(0.03 * rows.size)
        few[rows[order][:take], cols[order][:take]] = True
        m = sample_goal_metrics(few, GT, labels)
        self.assertEqual(m["leak_px"], take)
        self.assertFalse(m["merge"])
        many = GT.copy()
        take = int(0.10 * rows.size)
        many[rows[order][:take], cols[order][:take]] = True
        m = sample_goal_metrics(many, GT, labels)
        self.assertTrue(m["merge"])
        self.assertEqual(m["leak_px"], take)

    def test_merge_needs_a_share_of_one_neighbor_not_of_all(self) -> None:
        a = _box(300, 20, 60, 20, 100)
        b = _box(300, 220, 260, 20, 100)
        labels = neighbor_core_labels([a, b], GT, 10.0)
        pred = GT.copy()
        for index in (1, 2):
            rows, cols = np.nonzero(labels == index)
            order = np.argsort(rows)
            take = int(0.04 * rows.size)  # 4% of each: under the 5% bar for either
            pred[rows[order][:take], cols[order][:take]] = True
        self.assertFalse(sample_goal_metrics(pred, GT, labels)["merge"])

    def test_a_stray_blob_into_a_tile_clipped_neighbor_is_not_a_merge(self) -> None:
        # The neighbor runs off the tile: only a 10 x 40 strip of its core is visible (400 px).
        labels = np.zeros(GT.shape, dtype=np.int16)
        labels[100:140, 280:290] = 1
        blob = GT.copy()
        blob[100:106, 280:290] = True  # 60 px: 15% of the visible core, far from a real leak
        m = sample_goal_metrics(blob, GT, labels)
        self.assertEqual(m["leak_px"], 60)
        self.assertFalse(m["merge"])
        # A neighbor with a big visible core needs at least merge_min_px, whatever 5% of it is.
        labels = np.zeros(GT.shape, dtype=np.int16)
        labels[20:120, 170:290] = 1  # 12,000 px: 5% is 600
        for covered, expected in ((550, False), (610, True)):
            pred = GT.copy()
            flat = np.flatnonzero(labels == 1)[:covered]
            pred.reshape(-1)[flat] = True
            self.assertEqual(sample_goal_metrics(pred, GT, labels)["merge"], expected, covered)
        # 5% of a 8,000 px core is 400: the 500 px floor decides.
        labels = np.zeros(GT.shape, dtype=np.int16)
        labels[20:120, 170:250] = 1
        for covered, expected in ((450, False), (520, True)):
            pred = GT.copy()
            pred.reshape(-1)[np.flatnonzero(labels == 1)[:covered]] = True
            self.assertEqual(sample_goal_metrics(pred, GT, labels)["merge"], expected, covered)
        pred = GT.copy()
        pred.reshape(-1)[np.flatnonzero(labels == 1)[:450]] = True
        self.assertTrue(sample_goal_metrics(pred, GT, labels, merge_min_px=100)["merge"])

    def test_fragmentation_counts_components_of_at_least_200_pixels(self) -> None:
        pred = GT.copy()
        pred[10:30, 200:220] = True  # 400 px: a second piece
        self.assertTrue(sample_goal_metrics(pred, GT)["frag"])
        small = GT.copy()
        small[10:15, 200:210] = True  # 50 px: ignored
        self.assertFalse(sample_goal_metrics(small, GT)["frag"])
        two = GT.copy()
        two[200:260, 200:260] = True
        gt_two = two.copy()
        self.assertFalse(sample_goal_metrics(two, gt_two)["frag"])

    def test_a_shared_cache_changes_nothing(self) -> None:
        other = grow_mask(GT, 15.0)
        cache: dict = {}
        first = sample_goal_metrics(other, GT, cache=cache)
        self.assertTrue(cache)
        second = sample_goal_metrics(GT & (edge_distance(GT) >= 18.0), GT, cache=cache)
        self.assertEqual(first, sample_goal_metrics(other, GT))
        self.assertEqual(second, sample_goal_metrics(GT & (edge_distance(GT) >= 18.0), GT))

    def test_empty_prediction(self) -> None:
        m = sample_goal_metrics(np.zeros_like(GT), GT)
        self.assertEqual(m["core_hit"], 0)
        self.assertEqual(m["recall"], 0.0)
        self.assertEqual(m["pred_area"], 0)


class SummaryTests(unittest.TestCase):
    def _record(self, hit, total, leak=0, pred=1000, spill=0, gt_area=1000, merge=False, frag=False, prefix=""):
        row = {}
        add_goal_fields(
            row,
            {
                "core_gt": total, "core_hit": hit, "recall": hit / total if total else None, "pred_area": pred,
                "leak_px": leak, "gt_area": gt_area, "spill_px": spill, "merge": merge, "frag": frag,
            },
            prefix,
        )
        return row

    def test_pooling(self) -> None:
        rows = [
            self._record(100, 100),
            self._record(90, 100, leak=50, spill=100, merge=True, frag=True),
            self._record(0, 0),  # no core: counts nowhere in recall
        ]
        s = summarize_goal(rows)
        self.assertEqual(s.n, 3)
        self.assertAlmostEqual(s.core_recall, 190 / 200)
        self.assertAlmostEqual(s.core_recall_ok, 0.5)  # 1.0 passes, 0.9 does not, the no-core sample is skipped
        self.assertAlmostEqual(s.core_recall_p10, float(np.percentile([1.0, 0.9], 10)))
        self.assertAlmostEqual(s.leak, 50 / 3000)
        self.assertAlmostEqual(s.merge_rate, 1 / 3)
        self.assertAlmostEqual(s.spill, 100 / 3000)
        self.assertAlmostEqual(s.frag, 1 / 3)

    def test_the_input_prefix_pools_the_input_fields(self) -> None:
        row = self._record(100, 100)
        add_goal_fields(row, {k[0:]: v for k, v in self._record(80, 100, merge=True).items()}, "input_")
        self.assertAlmostEqual(summarize_goal([row], prefix="input_").core_recall, 0.8)
        self.assertEqual(summarize_goal([row], prefix="input_").merge_rate, 1.0)
        self.assertEqual(summarize_goal([]).n, 0)

    def test_no_harm(self) -> None:
        def pair(out_hit, in_hit, out_spill, in_spill):
            row = self._record(out_hit, 1000, spill=out_spill)
            add_goal_fields(row, self._record(in_hit, 1000, spill=in_spill), "input_")
            return row

        self.assertTrue(sample_no_harm(pair(995, 1000, 0, 0)))  # down 0.5 points exactly
        self.assertFalse(sample_no_harm(pair(990, 1000, 0, 0)))  # down 1 point
        self.assertTrue(sample_no_harm(pair(1000, 900, 0, 100)))  # better on both
        self.assertTrue(sample_no_harm(pair(1000, 1000, 5, 0)))  # spill up 0.5% of the ground truth
        self.assertFalse(sample_no_harm(pair(1000, 1000, 20, 0)))  # spill up 2%
        no_core = self._record(0, 0, spill=0)
        add_goal_fields(no_core, self._record(0, 0, spill=0), "input_")
        self.assertTrue(sample_no_harm(no_core))


class SelectionTests(unittest.TestCase):
    def _key(self, recall, merge=0.0, spill=0.0):
        return selection_key(GoalSummary(core_recall=recall, merge_rate=merge, spill=spill), 0.975)

    def test_reaching_the_target_beats_not_reaching_it(self) -> None:
        self.assertTrue(self._key(0.976, merge=0.5, spill=2.0) > self._key(0.974, merge=0.0))

    def test_among_epochs_at_the_target_the_lowest_merge_rate_wins(self) -> None:
        self.assertTrue(self._key(0.976, merge=0.02) > self._key(0.99, merge=0.03))
        self.assertFalse(self._key(0.99, merge=0.03) > self._key(0.976, merge=0.02))

    def test_spill_breaks_a_merge_rate_tie(self) -> None:
        self.assertTrue(self._key(0.98, merge=0.02, spill=0.01) > self._key(0.98, merge=0.02, spill=0.05))

    def test_below_the_target_the_highest_recall_wins(self) -> None:
        self.assertTrue(self._key(0.95, merge=0.9) > self._key(0.94, merge=0.0))

    def test_round_trip_and_float_noise(self) -> None:
        key = self._key(0.98, merge=0.02, spill=0.01)
        again = SelectionKey.from_list(key.to_list())
        self.assertFalse(again > key)
        self.assertFalse(key > again)
        # A change far below the fourth decimal is noise, not an improvement that resets the patience.
        self.assertFalse(self._key(0.98, merge=0.02, spill=0.01 - 1e-7) > key)

    def test_patience_two_stops_after_two_epochs_without_a_better_key(self) -> None:
        patience, best, waited, stopped_at = 2, None, 0, None
        epochs = [
            self._key(0.96),                       # first: saved
            self._key(0.98, merge=0.05),           # reaches the target: better
            self._key(0.98, merge=0.06),           # worse: wait 1
            self._key(0.99, merge=0.07),           # worse: wait 2 -> stop
            self._key(0.99, merge=0.0),            # never reached
        ]
        for epoch, key in enumerate(epochs):
            if best is None or key > best:
                best, waited = key, 0
            else:
                waited += 1
                if waited >= patience:
                    stopped_at = epoch
                    break
        self.assertEqual(stopped_at, 3)


class NoiseFloorTests(unittest.TestCase):
    """``improves``: the selection ordering with a noise floor, so jitter does not save checkpoints or reset patience."""

    def _key(self, recall, merge=0.0, spill=0.0):
        return selection_key(GoalSummary(core_recall=recall, merge_rate=merge, spill=spill), 0.975)

    def test_the_first_epoch_always_counts(self) -> None:
        self.assertTrue(improves(self._key(0.5), None))

    def test_reaching_the_gate_counts_and_losing_it_does_not(self) -> None:
        below = self._key(0.97, merge=0.0)
        above = self._key(0.976, merge=0.5, spill=2.0)
        self.assertTrue(improves(above, below))
        self.assertFalse(improves(below, above))

    def test_merge_rate_must_fall_by_the_floor(self) -> None:
        best = self._key(0.98, merge=0.050, spill=0.10)
        self.assertFalse(improves(self._key(0.98, merge=0.049, spill=0.10), best))  # 0.001: noise
        self.assertFalse(improves(self._key(0.98, merge=0.0485, spill=0.10), best))  # 0.0015
        self.assertTrue(improves(self._key(0.98, merge=0.048, spill=0.10), best))  # exactly 0.002
        self.assertTrue(improves(self._key(0.98, merge=0.030, spill=0.50), best))  # a real gain beats worse spill
        self.assertFalse(improves(self._key(0.99, merge=0.052, spill=0.0), best))  # worse merge, better recall

    def test_a_merge_tie_within_the_floor_is_decided_by_spill_with_the_same_floor(self) -> None:
        best = self._key(0.98, merge=0.050, spill=0.100)
        self.assertTrue(improves(self._key(0.98, merge=0.0505, spill=0.097), best))  # spill down 0.003
        self.assertFalse(improves(self._key(0.98, merge=0.0505, spill=0.099), best))  # spill down 0.001
        self.assertFalse(improves(self._key(0.98, merge=0.050, spill=0.100), best))  # identical
        self.assertTrue(improves(self._key(0.98, merge=0.049, spill=0.090), best))  # merge within floor, spill -0.01
        self.assertFalse(improves(self._key(0.98, merge=0.0495, spill=0.150), best))

    def test_below_the_gate_recall_must_rise_by_half_a_tenth_of_a_point(self) -> None:
        best = self._key(0.9500)
        self.assertFalse(improves(self._key(0.9503), best))
        self.assertTrue(improves(self._key(0.9505), best))
        self.assertFalse(improves(self._key(0.9490), best))

    def test_small_steady_gains_count_once_they_add_up_against_the_best(self) -> None:
        # Each epoch gains 0.0003 over the last. Measured against the best so far, a gain only counts once it
        # reaches 0.0005, so the checkpoint is saved every second epoch, not every epoch.
        best, waited, saved, longest = None, 0, [], 0
        for epoch in range(6):
            key = self._key(0.9500 + 0.0003 * epoch)
            if improves(key, best):
                best, waited = key, 0
                saved.append(epoch)
            else:
                waited += 1
                longest = max(longest, waited)
        self.assertEqual(saved, [0, 2, 4])
        self.assertEqual(longest, 1)

    def test_a_plateau_inside_the_noise_floor_stops_the_run_with_patience_two(self) -> None:
        best, waited, stopped = None, 0, None
        for epoch, merge in enumerate([0.040, 0.0395, 0.0405, 0.039, 0.0392]):
            key = self._key(0.98, merge=merge, spill=0.1)
            if improves(key, best):
                best, waited = key, 0
            else:
                waited += 1
                if waited >= 2:
                    stopped = epoch
                    break
        self.assertEqual(stopped, 2)

    def test_the_defaults(self) -> None:
        self.assertEqual(EARLY_STOPPING_MIN_DELTA, 0.002)
        self.assertEqual(RECALL_MIN_DELTA, 0.0005)


if __name__ == "__main__":
    unittest.main()
