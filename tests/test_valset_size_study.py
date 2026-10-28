"""scripts/valset_size_study.py and the per-sample records it reads (``sam2-em-eval --goal --records``)."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sam2_segmentation_trainer.data.approved import ApprovedSet
from sam2_segmentation_trainer.evaluate import read_records, write_records
from sam2_segmentation_trainer.validation import score_sample, select_validation_subset, summarize_mode

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "valset_size_study.py"
_spec = importlib.util.spec_from_file_location("valset_size_study", _SCRIPT)
study_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(study_mod)


def _records(n_per_volume: int = 30, seed: int = 0) -> dict[str, list[dict]]:
    """Real ``score_sample`` records for two families on small synthetic discs, varied predictions."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:64, 0:64]
    out: dict[str, list[dict]] = {"circle_annotator": [], "point": []}
    for volume in ("RC1", "RPC1"):
        for i in range(n_per_volume):
            radius = int(rng.integers(8, 20))
            gt = ((yy - 32) ** 2 + (xx - 32) ** 2 <= radius**2).astype(np.uint8)
            item = {
                "volume": volume,
                "image_key": f"{volume}_{i}_D1_X0-1024_Y0-1024",
                "location_id": 1000 + i,
                "approved": i == 0,
                "has_negatives": bool(i % 2),
            }
            for family, shrink in (("circle_annotator", int(rng.integers(0, 6))), ("point", int(rng.integers(0, 9)))):
                pred = ((yy - 32) ** 2 + (xx - 32 + shrink) ** 2 <= max(radius - shrink, 1) ** 2).astype(np.uint8)
                out[family].append(
                    score_sample(pred, gt, item, weight_np=None, neighbor_core=None, input_np=None, mode=family, band=2.0)
                )
    return out


class RecordsFileTests(unittest.TestCase):
    def test_write_then_read_gives_the_records_back(self) -> None:
        records = _records(5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl.gz"
            self.assertEqual(write_records(path, records), 20)
            back = read_records(path)
        self.assertEqual(sorted(back), sorted(records))
        for family in records:
            for a, b in zip(records[family], back[family], strict=True):
                self.assertEqual(a["key"], b["key"])
                self.assertAlmostEqual(float(a["iou"]), b["iou"], places=12)
                self.assertEqual(a["core_gt"], b["core_gt"])


class StudyTests(unittest.TestCase):
    def test_full_fraction_reproduces_summarize_mode(self) -> None:
        records = _records()
        rows = study_mod.study(records, families=["circle_annotator", "point"], fractions=[1.0], draws=5)
        for family in ("circle_annotator", "point"):
            expected = summarize_mode(records[family])
            got = {r["metric"]: r for r in rows if r["family"] == family}
            self.assertAlmostEqual(got["iou"]["full"], expected.iou, places=12)
            self.assertAlmostEqual(got["core_recall"]["full"], expected.goal.core_recall, places=12)
            self.assertAlmostEqual(got["merge_rate"]["full"], expected.goal.merge_rate, places=12)
            self.assertEqual(got["iou"]["p95_abs_err"], 0.0)
            self.assertEqual(got["iou"]["draws"], 1)

    def test_draws_are_deterministic_and_shrink_with_size(self) -> None:
        records = _records()
        kwargs = dict(families=["circle_annotator"], fractions=[0.1, 0.5], draws=40)
        first = study_mod.study(records, **kwargs)
        second = study_mod.study(records, **kwargs)
        self.assertEqual(first, second)
        err = {r["fraction"]: r["p95_abs_err"] for r in first if r["metric"] == "iou"}
        self.assertLess(err[0.5], err[0.1])

    def test_approved_samples_are_in_every_draw(self) -> None:
        records = _records()
        samples = sorted({tuple(r["key"]) for r in records["circle_annotator"]})
        approved_keys = {tuple(r["key"]) for r in records["circle_annotator"] if r["approved"]}
        approved = ApprovedSet(windows=frozenset((v, int(loc), key) for v, key, loc in approved_keys))
        examples = [study_mod.Sample(*key) for key in samples]
        self.assertEqual(len(approved_keys), 2)
        for seed in range(10):
            chosen = {tuple(c) for c in select_validation_subset(examples, 0.05, seed, approved)}
            self.assertTrue(approved_keys <= chosen)

    def test_report_marks_converged_cells(self) -> None:
        records = _records()
        rows = study_mod.study(records, families=["circle_annotator"], fractions=[0.5], draws=10)
        text = study_mod.report(rows, ["circle_annotator"], dict(study_mod.TOLERANCE), {"draws": 10, "records": "x"})
        self.assertIn("## circle_annotator", text)
        self.assertIn("Within tolerance from:", text)
        self.assertIn("Expected merges per pass", text)


if __name__ == "__main__":
    unittest.main()
