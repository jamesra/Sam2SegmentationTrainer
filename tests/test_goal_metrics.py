from __future__ import annotations

import unittest

import numpy as np

from annotation_crops.prompt_encoding import edge_distance
from sam2_segmentation_trainer.data.refine import grow_mask
from sam2_segmentation_trainer.data.weights import neighbor_core_labels
from sam2_segmentation_trainer.goal_metrics import (
    SELECTION_FLOORS,
    GoalSummary,
    SelectionKey,
    add_goal_fields,
    describe_selection,
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


def _key(recall=0.96, iou=0.86, spill=0.04, merge=0.001, frag=0.09):
    return selection_key(GoalSummary(core_recall=recall, merge_rate=merge, spill=spill, frag=frag), iou)


class SelectionTests(unittest.TestCase):
    """``improves``: better by its noise floor in at least one metric, worse by its floor in none."""

    def test_the_first_epoch_always_counts(self) -> None:
        self.assertTrue(improves(_key(), None))

    def test_one_metric_beyond_its_floor_counts(self) -> None:
        best = _key()
        self.assertTrue(improves(_key(recall=0.964), best))  # +0.004
        self.assertTrue(improves(_key(iou=0.866), best))
        self.assertTrue(improves(_key(spill=0.034), best))
        self.assertTrue(improves(_key(merge=0.0), best, {**SELECTION_FLOORS, "merge_rate": 0.001}))
        self.assertTrue(improves(_key(frag=0.078), best))

    def test_changes_inside_the_floors_are_noise(self) -> None:
        best = _key()
        self.assertFalse(improves(_key(recall=0.9635, iou=0.865, spill=0.035, merge=0.0, frag=0.08), best))
        self.assertFalse(improves(best, best))
        # Noise in other metrics does not block a real gain.
        self.assertTrue(improves(_key(recall=0.957, iou=0.875, spill=0.044, frag=0.10), best))

    def test_a_trade_beyond_the_floors_is_not_an_improvement(self) -> None:
        best = _key()
        self.assertFalse(improves(_key(recall=0.99, spill=0.06), best))
        self.assertFalse(improves(_key(iou=0.90, merge=0.004), best))

    def test_the_core_recall_target_is_not_a_gate(self) -> None:
        # Crossing 97.5% with worse spill is a trade; better IoU and spill below the target count.
        self.assertFalse(improves(_key(recall=0.976, spill=0.10), _key(recall=0.974)))
        self.assertTrue(improves(_key(recall=0.960, iou=0.872, spill=0.026, merge=0.0, frag=0.065),
                                 _key(recall=0.9605, iou=0.858, spill=0.045, merge=0.0011, frag=0.095)))

    def test_floors_can_be_overridden_per_metric(self) -> None:
        best = _key()
        self.assertFalse(improves(_key(iou=0.866), best, {**SELECTION_FLOORS, "iou": 0.01}))
        self.assertTrue(improves(_key(iou=0.871), best, {**SELECTION_FLOORS, "iou": 0.01}))

    def test_round_trip_and_float_noise(self) -> None:
        key = _key()
        again = SelectionKey.from_dict(key.to_dict())
        self.assertEqual(again.metrics, key.metrics)
        self.assertEqual(set(key.to_dict()), set(SELECTION_FLOORS))
        # Rounded to 4 decimals: a change far below that is no change.
        self.assertEqual(_key(spill=0.04 - 1e-7).metrics, key.metrics)

    def test_a_key_needs_every_metric(self) -> None:
        with self.assertRaises(ValueError):
            SelectionKey({"core_recall": 0.9})

    def test_describe_names_the_metrics_that_moved(self) -> None:
        line = describe_selection(_key(iou=0.88, spill=0.06), _key())
        self.assertIn("better beyond the noise floor: iou 0.8600 -> 0.8800", line)
        self.assertIn("worse: spill 0.0400 -> 0.0600", line)
        self.assertEqual(describe_selection(_key(), None), "first validated epoch")

    def test_small_steady_gains_count_once_they_add_up_against_the_best(self) -> None:
        # Each epoch gains 0.003 IoU over the last. Measured against the best so far, a gain only counts once it
        # reaches the 0.006 floor, so the checkpoint is saved every second epoch.
        best, waited, saved, longest = None, 0, [], 0
        for epoch in range(6):
            key = _key(iou=0.850 + 0.003 * epoch)
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
        for epoch, iou in enumerate([0.860, 0.862, 0.858, 0.864, 0.870]):
            key = _key(iou=iou)
            if improves(key, best):
                best, waited = key, 0
            else:
                waited += 1
                if waited >= 2:
                    stopped = epoch
                    break
        self.assertEqual(stopped, 2)

    def test_the_default_floors(self) -> None:
        self.assertEqual(
            SELECTION_FLOORS, {"core_recall": 0.004, "iou": 0.006, "spill": 0.006, "merge_rate": 0.002, "frag": 0.012}
        )


if __name__ == "__main__":
    unittest.main()
