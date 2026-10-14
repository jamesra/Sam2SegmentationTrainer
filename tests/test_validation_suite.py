from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np
import torch

from sam2_segmentation_trainer.data.dataset import EMSegDataset, collate_em
from sam2_segmentation_trainer.data.manifest import index_volumes
from sam2_segmentation_trainer.data.prompts import PromptType, inscribed_circle_xyr, interior_point_xy
from sam2_segmentation_trainer.data.tiles import build_tile_context
from sam2_segmentation_trainer.goal_metrics import GoalSummary
from sam2_segmentation_trainer.validation import (
    ModeResult,
    ValidationSuite,
    collect_records,
    log_results,
    summarize_mode,
)

try:
    from tests.helpers import make_crops
except ImportError:
    from helpers import make_crops


def _c_shape() -> np.ndarray:
    mask = np.zeros((40, 40), dtype=bool)
    mask[5:35, 5:12] = True
    mask[5:12, 5:35] = True
    mask[28:35, 5:35] = True
    return mask


class InteriorGeometryTests(unittest.TestCase):
    def test_interior_point_is_inside_where_the_centroid_is_not(self) -> None:
        mask = _c_shape()
        ys, xs = np.nonzero(mask)
        centroid = (int(round(xs.mean())), int(round(ys.mean())))
        self.assertFalse(mask[centroid[1], centroid[0]])
        x, y = interior_point_xy(mask)
        self.assertTrue(mask[int(y), int(x)])

    def test_inscribed_circle_stays_inside_the_mask(self) -> None:
        mask = np.zeros((50, 50), dtype=bool)
        mask[10:40, 10:40] = True
        cx, cy, radius = inscribed_circle_xyr(mask, scale=1.0)
        self.assertAlmostEqual(radius, 15.0, delta=1.0)
        yy, xx = np.mgrid[0:50, 0:50]
        disc = (xx + 0.5 - cx) ** 2 + (yy + 0.5 - cy) ** 2 <= radius**2
        self.assertTrue(mask[disc].all())
        smaller = inscribed_circle_xyr(mask, scale=0.95)[2]
        self.assertAlmostEqual(smaller, radius * 0.95, places=4)

    def test_empty_mask_has_no_point_or_circle(self) -> None:
        empty = np.zeros((8, 8), dtype=bool)
        self.assertIsNone(interior_point_xy(empty))
        self.assertIsNone(inscribed_circle_xyr(empty))


class ValidationDatasetTests(unittest.TestCase):
    def _dataset(self, root: Path, mode: str) -> EMSegDataset:
        examples = index_volumes(volumes=["RC1"], root=root)
        return EMSegDataset(examples, split="val", image_size=32, val_prompt=mode, seed=3)

    def test_each_mode_builds_its_prompt(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_crops(root, "RC1", n=4)
            centroid = self._dataset(root, "centroid")[0]
            point = self._dataset(root, "point")[0]
            circle = self._dataset(root, "circle_mask")[0]
            correction = self._dataset(root, "correction")[0]
            near = self._dataset(root, "near_correct")[0]
            kinds = lambda item: [v["prompt_type"] for v in item["variants"]]  # noqa: E731
            self.assertEqual(kinds(centroid), ["centroid"])
            self.assertEqual(kinds(point), ["point"])
            self.assertEqual(kinds(circle), [PromptType.CIRCLE_SEED.value])
            self.assertEqual(kinds(correction), [PromptType.CORRECTION.value])
            self.assertEqual(kinds(near), [PromptType.NEAR_CORRECT.value])
            self.assertIsNone(centroid["variants"][0]["dense_mask"])
            for item in (circle, correction, near):
                self.assertEqual(tuple(item["variants"][0]["dense_mask"].shape), (256, 256))
                self.assertIsNotNone(item["variants"][0]["prompt_input"])
            self.assertIsNone(correction["variants"][0]["prompt_fg"])  # never trusted
            self.assertIsNotNone(circle["variants"][0]["prompt_fg"])

    def test_circle_mask_is_reproducible(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_crops(root, "RC1", n=4)
            first = self._dataset(root, "circle_mask")[1]
            second = self._dataset(root, "circle_mask")[1]
            a, b = first["variants"][0], second["variants"][0]
            self.assertTrue(torch.equal(a["dense_mask"], b["dense_mask"]))
            self.assertTrue(torch.equal(a["point"], b["point"]))

    def test_correction_and_near_correct_are_reproducible(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_crops(root, "RC1", n=4)
            for mode in ("correction", "near_correct"):
                a = self._dataset(root, mode)[1]["variants"][0]
                b = self._dataset(root, mode)[1]["variants"][0]
                self.assertTrue(torch.equal(a["prompt_input"], b["prompt_input"]))

    def test_unknown_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            EMSegDataset([], split="val", val_prompt="box")

    def test_circle_mask_with_an_empty_tile_context(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_crops(root, "RC1", n=4)
            examples = index_volumes(volumes=["RC1"], root=root)
            context = build_tile_context(examples, load_circles=False)
            dataset = EMSegDataset(
                examples, split="val", image_size=32, val_prompt="circle_mask", tiles=context
            )
            self.assertEqual(dataset[0]["variants"][0]["prompt_type"], PromptType.CIRCLE_SEED.value)


class SuiteTests(unittest.TestCase):
    def test_bad_configuration_is_rejected(self) -> None:
        kwargs = dict(image_size=32, num_workers=0)
        with self.assertRaises(ValueError):
            ValidationSuite([], modes=[], primary="point", **kwargs)
        with self.assertRaises(ValueError):
            ValidationSuite([], modes=["point"], primary="circle_mask", **kwargs)
        with self.assertRaises(ValueError):
            ValidationSuite([], modes=["nonsense"], primary="nonsense", **kwargs)

    def test_primary_dataset_and_replace(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_crops(root, "RC1", n=4)
            examples = index_volumes(volumes=["RC1"], root=root)
            suite = ValidationSuite(
                examples, modes=["point", "circle_mask"], primary="circle_mask", image_size=32, num_workers=0
            )
            self.assertEqual(suite.dataset.val_families, ("point", "circle_mask"))
            self.assertEqual(len(suite), len(examples))
            suite.replace_examples(examples[:2])
            self.assertEqual((len(suite.full_dataset), len(suite.subset_dataset)), (2, 2))

    def test_the_subset_leaves_out_the_legacy_family_and_needs_a_modern_primary(self) -> None:
        kwargs = dict(image_size=32, num_workers=0)
        suite = ValidationSuite(
            [], modes=["circle_mask", "centroid", "point"], primary="circle_mask", **kwargs
        )
        self.assertEqual(suite.subset_modes, ["circle_mask", "point"])
        self.assertEqual(suite.modes, ["circle_mask", "centroid", "point"])
        with self.assertRaises(ValueError):
            ValidationSuite([], modes=["circle_mask", "centroid"], primary="centroid", **kwargs)


class BenchmarkTimingTests(unittest.TestCase):
    def test_subset_and_full_passes_are_timed(self) -> None:
        from sam2_segmentation_trainer.train import _time_validation_suite

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_crops(root, "RC1", n=6)
            examples = index_volumes(volumes=["RC1"], root=root)
            suite = ValidationSuite(
                examples, modes=["point", "centroid"], primary="point", image_size=32, num_workers=0,
                epoch_subset=0.5, batch_size=2,
            )

            def predict(model, batch, device, image_size):
                return batch["mask"][batch["variant_index"]].float() * 20.0 - 10.0

            with mock.patch("sam2_segmentation_trainer.validation.predict_variants", predict):
                times = _time_validation_suite(mock.MagicMock(), suite, "cpu", 32, 0.5, warmup=1, steps=4)
        self.assertEqual(sorted(times), ["full", "subset"])
        for seconds, wait in times.values():
            self.assertGreater(seconds, 0.0)
            self.assertGreaterEqual(wait, 0.0)

    def test_collect_records_reports_the_goal_metrics_and_the_approved_flag(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_crops(root, "RC1", n=3)
            examples = index_volumes(volumes=["RC1"], root=root)
            approved = {(examples[0].location_id, examples[0].image_key)}
            suite = ValidationSuite(
                examples, modes=["point"], primary="point", image_size=32, num_workers=0, approved=approved,
                core_band_px=2.0,
            )
            masks = [batch["mask"][0] for batch in suite.loader("point")]
            calls = {"n": 0}

            def predict(model, images, points, labels, *, image_size):
                mask = masks[calls["n"]]
                calls["n"] += 1
                return (mask.float() * 20.0 - 10.0).unsqueeze(0), torch.zeros(1, 1)

            with mock.patch("sam2_segmentation_trainer.validation.predict_masks", predict):
                records, _ = collect_records(
                    mock.MagicMock(), suite.loader("point"), "cpu", 32, 0.5, mode="point", core_band_px=2.0
                )
        self.assertEqual(sum(bool(r["approved"]) for r in records), 1)
        for record in records:
            self.assertEqual(record["core_hit"], record["core_gt"])
            self.assertEqual(record["spill_px"], 0)
            self.assertFalse(record["merge"])
        result = summarize_mode(records)
        self.assertEqual(result.approved_n, 1)
        self.assertEqual(result.approved_by_volume, {"RC1": 1})
        self.assertAlmostEqual(result.goal.core_recall, 1.0)
        self.assertAlmostEqual(result.approved.core_recall, 1.0)


class ScoringTests(unittest.TestCase):
    def test_full_tiles_are_kept_out_of_the_headline(self) -> None:
        records = [
            {"volume": "RC1", "category": "MC", "iou": 0.5, "bf10": 0.4, "full_tile": False},
            {"volume": "RC1", "category": "MC", "iou": 0.7, "bf10": 0.6, "full_tile": False},
            {"volume": "RC2", "category": "HC", "iou": 1.0, "bf10": 1.0, "full_tile": True},
        ]
        result = summarize_mode(records)
        self.assertAlmostEqual(result.iou, 0.6)
        self.assertEqual(result.n, 2)
        self.assertAlmostEqual(result.fulltile_iou, 1.0)
        self.assertEqual(result.by_volume, {"RC1": 0.6})
        self.assertEqual(result.by_category, {"MC": 0.6})

    def test_only_full_tiles_falls_back_to_them(self) -> None:
        result = summarize_mode([{"volume": "A", "category": "", "iou": 1.0, "bf10": 1.0, "full_tile": True}])
        self.assertEqual(result.n, 1)
        self.assertEqual(summarize_mode([]).n, 0)

    def test_refinement_scores_gain_and_no_harm(self) -> None:
        def record(iou, input_iou, core_hit, input_core_hit, spill_px, input_spill_px):
            row = {
                "volume": "A", "category": "", "iou": iou, "bf10": 0.9, "input_iou": input_iou, "full_tile": False,
            }
            for prefix, hit, spill in (("", core_hit, spill_px), ("input_", input_core_hit, input_spill_px)):
                row.update({
                    f"{prefix}core_gt": 1000, f"{prefix}core_hit": hit, f"{prefix}recall": hit / 1000,
                    f"{prefix}pred_area": 1500, f"{prefix}leak_px": 0, f"{prefix}gt_area": 1000,
                    f"{prefix}spill_px": spill, f"{prefix}merge": False, f"{prefix}frag": False,
                })
            return row

        records = [
            record(0.9, 0.7, 990, 900, 0, 50),    # improves: passes
            record(0.8, 0.81, 996, 1000, 5, 0),   # recall down 0.4 points, spill up 0.5%: passes
            record(0.6, 0.9, 900, 990, 0, 0),     # recall down 9 points: fails
            record(0.8, 0.8, 1000, 1000, 20, 0),  # spill up 2% of the ground truth: fails
        ]
        result = summarize_mode(records)
        self.assertAlmostEqual(result.input_iou, (0.7 + 0.81 + 0.9 + 0.8) / 4)
        self.assertAlmostEqual(result.gain, (0.2 - 0.01 - 0.3 + 0.0) / 4)
        self.assertAlmostEqual(result.no_harm, 2 / 4)
        self.assertAlmostEqual(result.goal.core_recall, (990 + 996 + 900 + 1000) / 4000)
        self.assertAlmostEqual(result.input_goal.core_recall, (900 + 1000 + 990 + 1000) / 4000)
        self.assertIsNone(summarize_mode([{"volume": "A", "category": "", "iou": 1.0, "bf10": 1.0}]).gain)

    def test_collect_records_ignores_the_edge_band_for_tolerant_iou(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_crops(root, "RC1", n=2)
            examples = index_volumes(volumes=["RC1"], root=root)[:1]
            dataset = EMSegDataset(
                examples, split="val", image_size=32, val_prompt="point", edge_tau_px=8.0
            )
            loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collate_em)
            masks = [batch["mask"][0] for batch in loader]
            calls = {"n": 0}

            def predict(model, images, points, labels, *, image_size):
                mask = masks[calls["n"]].clone()
                calls["n"] += 1
                top = int(torch.nonzero(mask.any(dim=1))[0])
                mask[top : top + 1] = 0  # one pixel short along one edge: inside the tolerance band
                return (mask.float() * 20.0 - 10.0).unsqueeze(0), torch.zeros(1, 1)

            with mock.patch("sam2_segmentation_trainer.validation.predict_masks", predict):
                records, _ = collect_records(mock.MagicMock(), loader, "cpu", 32, 0.5, mode="point")
            for record in records:
                self.assertLess(record["raw_iou"], 0.95)
                self.assertGreater(record["iou"], record["raw_iou"])
                self.assertGreater(record["iou"], 0.99)

    def test_collect_records_scores_perfect_predictions(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_crops(root, "RC1", n=3)
            examples = index_volumes(volumes=["RC1"], root=root)
            dataset = EMSegDataset(examples, split="val", image_size=32, val_prompt="point")
            loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collate_em)
            masks = [batch["mask"][0] for batch in loader]
            calls = {"n": 0}

            def predict(model, images, points, labels, *, image_size):
                mask = masks[calls["n"]]
                calls["n"] += 1
                return (mask.float() * 20.0 - 10.0).unsqueeze(0), torch.zeros(1, 1)

            with mock.patch("sam2_segmentation_trainer.validation.predict_masks", predict):
                records, overlays = collect_records(mock.MagicMock(), loader, "cpu", 32, 0.5)
            self.assertEqual(len(records), len(examples))
            self.assertTrue(all(abs(r["iou"] - 1.0) < 1e-3 for r in records))
            self.assertEqual(overlays, [])

    def test_log_results_names_every_mode(self) -> None:
        class Writer:
            def __init__(self) -> None:
                self.tags: list[str] = []

            def add_scalar(self, tag, value, step) -> None:
                self.tags.append(tag)

        writer = Writer()
        results = {
            "circle_mask": ModeResult(
                iou=0.8, bf10=0.7, n=5, by_volume={"RC1": 0.8}, by_category={"MC": 0.8},
                goal=GoalSummary(n=5, core_recall=0.98, merge_rate=0.02, spill=0.01),
                approved=GoalSummary(n=2, core_recall=0.99), approved_n=2, approved_by_volume={"RC1": 2},
            ),
            "near_correct": ModeResult(
                iou=0.9, n=5, goal=GoalSummary(n=5, core_recall=0.97, merge_rate=0.03), no_harm=0.96,
                input_goal=GoalSummary(n=5, core_recall=0.96, merge_rate=0.04), input_iou=0.85, gain=0.05,
            ),
            "point": ModeResult(iou=0.75, bf10=0.6, n=5, fulltile_iou=0.99, fulltile_n=2),
        }
        text = log_results(writer, 3, results, "circle_mask", root="val_subset", label="subset (5 of 20)")
        for tag in (
            "val/iou", "val_iou/RC1", "val_subset/iou/point", "val_subset_iou/circle_mask/RC1",
            "val_subset_iou_cat/circle_mask/MC", "val_subset/bf10/circle_mask",
            "val_subset/bf20/circle_mask", "val_subset/raw_iou/point", "val_subset/iou_fulltile/point",
            "val_subset/core_recall/circle_mask",
            "val_subset/core_recall_p10/circle_mask", "val_subset/core_recall_ok/circle_mask",
            "val_subset/leak/circle_mask",
            "val_subset/merge_rate/circle_mask", "val_subset/spill/circle_mask", "val_subset/frag/circle_mask",
            "val_subset/input/core_recall/near_correct", "val_subset/delta_core_recall/near_correct",
            "val_subset/delta_merge_rate/near_correct", "val_subset/no_harm/near_correct",
            "val_subset_approved/core_recall/circle_mask",
            "val_subset_approved/n/circle_mask", "val_subset_approved_n/circle_mask/RC1",
        ):
            self.assertIn(tag, writer.tags)
        self.assertIn("subset (5 of 20) val", text)
        # The full pass logs under its own tags and does not touch the legacy names.
        full_writer = Writer()
        log_results(full_writer, 3, results, "circle_mask", root="val_full", label="full")
        self.assertIn("val_full/core_recall/circle_mask", full_writer.tags)
        self.assertNotIn("val/iou", full_writer.tags)
        self.assertFalse(any(tag.startswith("val_subset") for tag in full_writer.tags))
        self.assertIn("core recall 0.9800", text)
        self.assertIn("approved (n=2)", text)
        self.assertIn("near_correct core recall 0.9700", text)
        self.assertIn("no harm 0.96", text)


if __name__ == "__main__":
    unittest.main()
