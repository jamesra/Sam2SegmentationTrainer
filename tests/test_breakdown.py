from __future__ import annotations

import unittest

import numpy as np

from sam2_segmentation_trainer.validation import log_results, score_sample, summarize_mode
from sam2_segmentation_trainer.goal_metrics import DEPTH_BINS, ground_truth_terms, summarize_by_depth


def _record(volume, category, depth, hit, total, merge=False, spill=0, gt_area=1000):
    return {
        "volume": volume, "category": category, "iou": 0.9, "raw_iou": 0.9, "bf10": 0.8, "bf20": 0.9,
        "gt_depth": depth, "core_gt": total, "core_hit": hit, "recall": (hit / total) if total else None,
        "pred_area": 1000, "leak_px": 0, "gt_area": gt_area, "spill_px": spill, "merge": merge, "frag": False,
    }


RECORDS = [
    _record("RC1", "MC", 8.0, 0, 0),                 # a thin profile: no core at all
    _record("RC1", "MC", 15.0, 30, 40),               # thin
    _record("RC1", "MC", 40.0, 900, 1000, merge=True),
    _record("RC2", "NC", 40.0, 990, 1000),
    _record("RC2", "NC", 90.0, 19_000, 20_000, spill=100, gt_area=30_000),
    _record("RC2", "NC", 120.0, 39_900, 40_000),
]


class Writer:
    def __init__(self) -> None:
        self.tags: dict[str, float] = {}

    def add_scalar(self, tag, value, step) -> None:
        self.tags[tag] = value


class DepthTests(unittest.TestCase):
    def test_bins_split_the_samples_by_maximum_depth(self) -> None:
        bins = summarize_by_depth(RECORDS)
        self.assertEqual([name for name, _lo, _hi in DEPTH_BINS], list(bins))
        self.assertEqual((bins["lt20"].n, bins["lt20"].with_core), (2, 1))  # the 8 px sample has no core
        self.assertEqual(bins["20_60"].n, 2)
        self.assertEqual(bins["ge60"].n, 2)
        self.assertAlmostEqual(bins["lt20"].mean_recall, 0.75)
        self.assertAlmostEqual(bins["lt20"].core_recall, 0.75)
        self.assertAlmostEqual(bins["20_60"].mean_recall, (0.9 + 0.99) / 2)
        self.assertAlmostEqual(bins["ge60"].core_recall, (19_000 + 39_900) / 60_000)

    def test_thin_processes_count_equally_in_the_per_sample_mean_but_not_in_the_pooled_figure(self) -> None:
        rows = [_record("RC1", "MC", 15.0, 20, 40), _record("RC1", "MC", 100.0, 99_000, 100_000)]
        thin_and_large = summarize_by_depth(rows)
        pooled = sum(r["core_hit"] for r in rows) / sum(r["core_gt"] for r in rows)
        self.assertGreater(pooled, 0.98)  # the thin sample, half missed, barely moves it
        self.assertAlmostEqual(thin_and_large["lt20"].mean_recall, 0.5)  # but it is visible in its own bin

    def test_empty_bins_and_unknown_depth_are_harmless(self) -> None:
        bins = summarize_by_depth([{"recall": 1.0, "core_hit": 1, "core_gt": 1}])
        self.assertTrue(all(b.n == 0 for b in bins.values()))

    def test_score_sample_records_the_ground_truth_depth(self) -> None:
        gt = np.zeros((120, 120), dtype=bool)
        gt[20:100, 20:100] = True  # 80 px square: maximum depth 39.5
        record = score_sample(
            gt, gt, {"volume": "RC1"}, weight_np=None, neighbor_core=None, input_np=None, mode="circle_mask",
            band=10.0,
        )
        self.assertAlmostEqual(record["gt_depth"], 39.5, places=3)
        self.assertAlmostEqual(ground_truth_terms(gt, 10.0, 200)["depth"], 39.5, places=3)
        empty = ground_truth_terms(np.zeros((8, 8), dtype=bool), 10.0, 200)
        self.assertEqual(empty["depth"], 0.0)


class BreakdownTests(unittest.TestCase):
    def test_goal_metrics_per_category_and_volume(self) -> None:
        result = summarize_mode(RECORDS)
        self.assertEqual(sorted(result.goal_by_category), ["MC", "NC"])
        self.assertEqual(sorted(result.goal_by_volume), ["RC1", "RC2"])
        mc = result.goal_by_category["MC"]
        self.assertEqual(mc.n, 3)
        self.assertAlmostEqual(mc.core_recall, (30 + 900) / 1040)
        self.assertAlmostEqual(mc.merge_rate, 1 / 3)
        nc = result.goal_by_category["NC"]
        self.assertAlmostEqual(nc.merge_rate, 0.0)
        self.assertAlmostEqual(nc.spill, 100 / (1000 + 30_000 + 1000))
        self.assertEqual(result.goal_by_volume["RC1"].n, 3)
        self.assertEqual(result.n, 6)

    def test_only_a_full_pass_logs_the_breakdowns(self) -> None:
        results = {"circle_mask": summarize_mode(RECORDS)}
        full, subset = Writer(), Writer()
        text = log_results(full, 3, results, "circle_mask", root="val_full", label="full")
        log_results(subset, 3, results, "circle_mask", root="val_subset", label="subset")
        for metric in ("core_recall", "core_recall_p10", "merge_rate", "spill"):
            self.assertIn(f"val_full_cat/{metric}/circle_mask/MC", full.tags)
            self.assertIn(f"val_full_vol/{metric}/circle_mask/RC2", full.tags)
        for name in ("lt20", "20_60", "ge60"):
            self.assertIn(f"val_full_depth/mean_recall/circle_mask/{name}", full.tags)
            self.assertIn(f"val_full_depth/n/circle_mask/{name}", full.tags)
        self.assertEqual(full.tags["val_full_cat/merge_rate/circle_mask/MC"], 1 / 3)
        self.assertFalse(any("_cat/" in tag and "iou" not in tag or "_vol/" in tag or "_depth/" in tag for tag in subset.tags))
        self.assertIn("by depth", text)
        self.assertIn("<20 px", text)


if __name__ == "__main__":
    unittest.main()
