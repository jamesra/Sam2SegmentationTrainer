from __future__ import annotations

import random
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from sam2_segmentation_trainer.data.dataset import EMSegDataset, collate_em
from sam2_segmentation_trainer.data.manifest import index_volumes
from sam2_segmentation_trainer.data.tiles import build_tile_context
from sam2_segmentation_trainer.data.weights import neighbor_core_labels
from sam2_segmentation_trainer.goal_metrics import ground_truth_terms
from sam2_segmentation_trainer.validation import (
    ValidationSuite,
    collect_family_records,
    collect_records,
    full_pass_due,
    score_sample,
    select_validation_subset,
)

try:
    from tests.helpers import make_crops
except ImportError:
    from helpers import make_crops

FAMILIES = ["circle_mask", "correction", "near_correct", "point", "centroid"]


def _fake(volume: str, i: int, location_id: int = 1):
    return SimpleNamespace(volume=volume, image_key=f"{volume}_{i}", location_id=location_id)


def _shifted_prediction(model, batch, device, image_size):
    """A deterministic stand-in for the model: each variant's ground truth moved one pixel, so scores are not all 1."""
    truth = batch["mask"][batch["variant_index"]].float()
    return torch.roll(truth, shifts=1, dims=-1) * 20.0 - 10.0


class SubsetSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.examples = [_fake("A", i) for i in range(100)] + [_fake("B", i) for i in range(40)]

    def test_stratified_by_volume(self) -> None:
        subset = select_validation_subset(self.examples, 0.25, seed=7)
        self.assertEqual(sum(e.volume == "A" for e in subset), 25)
        self.assertEqual(sum(e.volume == "B" for e in subset), 10)

    def test_reproducible_independent_of_order_and_seed_dependent(self) -> None:
        first = select_validation_subset(self.examples, 0.25, seed=7)
        shuffled = list(self.examples)
        random.Random(1).shuffle(shuffled)
        again = select_validation_subset(shuffled, 0.25, seed=7)
        self.assertEqual({(e.volume, e.image_key) for e in first}, {(e.volume, e.image_key) for e in again})
        other = select_validation_subset(self.examples, 0.25, seed=8)
        self.assertNotEqual({e.image_key for e in first}, {e.image_key for e in other})

    def test_keeps_the_input_order(self) -> None:
        subset = select_validation_subset(self.examples, 0.25, seed=7)
        positions = [self.examples.index(e) for e in subset]
        self.assertEqual(positions, sorted(positions))

    def test_every_approved_sample_is_included(self) -> None:
        approved = frozenset({(1, "A_3"), (1, "B_39"), (1, "A_77")})
        plain = select_validation_subset(self.examples, 0.1, seed=7)
        subset = select_validation_subset(self.examples, 0.1, seed=7, approved=approved)
        keys = {(e.location_id, e.image_key) for e in subset}
        self.assertTrue(approved <= keys)
        self.assertGreaterEqual(len(subset), len(plain))
        # The draw itself is unchanged: approved samples are added to it.
        self.assertTrue({(e.location_id, e.image_key) for e in plain} <= keys)

    def test_small_volumes_keep_one_and_full_fraction_keeps_all(self) -> None:
        tiny = [_fake("C", i) for i in range(3)]
        self.assertEqual(len(select_validation_subset(tiny, 0.01, seed=1)), 1)
        self.assertEqual(len(select_validation_subset(self.examples, 1.0, seed=1)), len(self.examples))
        self.assertEqual(select_validation_subset([], 0.25, seed=1), [])

    def test_full_pass_schedule(self) -> None:
        due = [e for e in range(15) if full_pass_due(e, 15, 3)]
        self.assertEqual(due, [2, 5, 8, 11, 14])
        self.assertEqual([e for e in range(8) if full_pass_due(e, 8, 3)], [2, 5, 7])  # the last epoch always
        self.assertTrue(all(full_pass_due(e, 4, 1) for e in range(4)))


class FamilyDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        make_crops(self.root, "RC1", n=6)
        make_crops(self.root, "RC2", n=6)
        self.examples = index_volumes(volumes=["RC1", "RC2"], root=self.root)
        self.tiles = build_tile_context(self.examples, load_circles=False)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _dataset(self, examples, families, **extra) -> EMSegDataset:
        return EMSegDataset(
            examples, split="val", image_size=32, tiles=self.tiles, seed=11, val_families=families,
            core_band_px=2.0, **extra,
        )

    def test_one_item_carries_every_family(self) -> None:
        item = self._dataset(self.examples, FAMILIES)[0]
        self.assertEqual([v["val_family"] for v in item["variants"]], FAMILIES)
        by_family = {v["val_family"]: v for v in item["variants"]}
        for name in ("point", "centroid"):
            self.assertIsNone(by_family[name]["dense_mask"])
            self.assertEqual(tuple(by_family[name]["point"].shape), (1, 2))
        for name in ("circle_mask", "correction", "near_correct"):
            self.assertEqual(tuple(by_family[name]["dense_mask"].shape), (256, 256))
        for name in ("correction", "near_correct"):
            self.assertIsNotNone(by_family[name]["prompt_input"])
        self.assertEqual(set(item["gt_terms"]), {"core", "grown", "components", "area", "depth"})
        self.assertEqual(tuple(item["neighbor_core"].shape), (32, 32))

    def test_prompts_do_not_depend_on_family_order_or_position(self) -> None:
        forward = self._dataset(self.examples, FAMILIES)
        reverse = self._dataset(list(reversed(self.examples)), list(reversed(FAMILIES)))
        for index, example in enumerate(self.examples):
            a = {v["val_family"]: v for v in forward[index]["variants"]}
            b = {v["val_family"]: v for v in reverse[len(self.examples) - 1 - index]["variants"]}
            for family in FAMILIES:
                self.assertTrue(torch.equal(a[family]["point"], b[family]["point"]), (example.image_key, family))
                for key in ("dense_mask", "prompt_input"):
                    if a[family][key] is None:
                        self.assertIsNone(b[family][key])
                    else:
                        self.assertTrue(torch.equal(a[family][key], b[family][key]), (example.image_key, family, key))

    def test_the_seed_changes_the_masks_to_correct(self) -> None:
        # 32 px cells are tiny, so use the larger of two seeds' worth of draws and require some difference.
        a = self._dataset(self.examples, ["correction", "near_correct"])
        b = EMSegDataset(
            self.examples, split="val", image_size=32, tiles=self.tiles, seed=12,
            val_families=["correction", "near_correct"], core_band_px=2.0,
        )
        differs = 0
        for index in range(len(self.examples)):
            for va, vb in zip(a[index]["variants"], b[index]["variants"]):
                differs += int(not torch.equal(va["prompt_input"], vb["prompt_input"]))
        self.assertGreater(differs, 0)

    def test_bad_arguments(self) -> None:
        with self.assertRaises(ValueError):
            self._dataset(self.examples, ["box"])
        with self.assertRaises(ValueError):
            EMSegDataset(self.examples, split="val", val_prompt="point", val_families=["point"])


class PassTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        make_crops(self.root, "RC1", n=8)
        make_crops(self.root, "RC2", n=8)
        self.examples = index_volumes(volumes=["RC1", "RC2"], root=self.root)
        self.tiles = build_tile_context(self.examples, load_circles=False)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _suite(self, **extra) -> ValidationSuite:
        args = dict(
            modes=FAMILIES, primary="circle_mask", image_size=32, num_workers=0, tiles=self.tiles, seed=5,
            core_band_px=2.0,
        )
        args.update(extra)
        return ValidationSuite(self.examples, **args)

    def _records(self, suite: ValidationSuite, **kwargs):
        with mock.patch("sam2_segmentation_trainer.validation.predict_variants", _shifted_prediction):
            records, _ = collect_family_records(
                mock.MagicMock(), suite.loader(full=True), "cpu", 32, 0.5, core_band_px=suite.core_band_px, **kwargs
            )
        return records

    def test_one_encoder_pass_serves_every_family(self) -> None:
        suite = self._suite(batch_size=3)
        calls = []

        def spy(model, images, points, labels, boxes, dense, *, image_size, variant_index=None):
            calls.append((images.shape[0], points.shape[0], len(boxes), variant_index.tolist()))
            return torch.zeros(points.shape[0], 32, 32), torch.zeros(points.shape[0], 1)

        with mock.patch("sam2_segmentation_trainer.validation.predict_masks_mixed", spy):
            collect_family_records(
                mock.MagicMock(), suite.loader(full=True), "cpu", 32, 0.5, core_band_px=2.0
            )
        self.assertEqual(sum(images for images, *_ in calls), len(self.examples))
        for images, rows, boxes, index in calls:
            self.assertEqual(rows, images * len(FAMILIES))
            self.assertEqual(boxes, rows)
            self.assertEqual(index, [b for b in range(images) for _ in FAMILIES])

    def test_every_family_gets_a_record_per_sample_in_order(self) -> None:
        records = self._records(self._suite(batch_size=4))
        self.assertEqual(sorted(records), sorted(FAMILIES))
        order = [(e.volume, e.image_key, int(e.location_id)) for e in self.examples]
        for family in FAMILIES:
            self.assertEqual([r["key"] for r in records[family]], order)

    def test_results_do_not_depend_on_batch_size_threads_or_backlog(self) -> None:
        reference = self._records(self._suite(batch_size=1), metric_threads=1, max_pending=1)
        for batch_size, threads, pending in ((4, 4, 64), (3, 2, 2), (8, 4, 1)):
            other = self._records(self._suite(batch_size=batch_size), metric_threads=threads, max_pending=pending)
            self.assertEqual(sorted(other), sorted(reference))
            for family in FAMILIES:
                self.assertEqual(len(other[family]), len(reference[family]))
                for a, b in zip(reference[family], other[family]):
                    self.assertEqual(a, b, (batch_size, family))

    def test_matches_the_one_family_at_a_time_path_for_deterministic_prompts(self) -> None:
        """The single-pass pipeline reproduces the old per-family loaders exactly (circle, point, centroid)."""
        suite = self._suite(batch_size=4)
        new = self._records(suite, metric_threads=3)

        def old_prediction(model, batch, device, image_size):
            return torch.roll(batch["mask"].float(), shifts=1, dims=-1) * 20.0 - 10.0

        for family in ("circle_mask", "point", "centroid"):
            dataset = EMSegDataset(
                self.examples, split="val", image_size=32, tiles=self.tiles, seed=5, val_prompt=family,
                edge_tau_px=8.0, core_band_px=2.0,
            )
            loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collate_em)
            with mock.patch("sam2_segmentation_trainer.validation.predict_batch", old_prediction):
                old, _ = collect_records(
                    mock.MagicMock(), loader, "cpu", 32, 0.5, mode=family, core_band_px=2.0
                )
            self.assertEqual(len(old), len(new[family]))
            for a, b in zip(old, new[family]):
                self.assertEqual(a, b, family)

    def test_the_subset_is_the_same_whether_run_alone_or_taken_from_a_full_pass(self) -> None:
        suite = self._suite(batch_size=4, epoch_subset=0.5)
        with mock.patch("sam2_segmentation_trainer.validation.predict_variants", _shifted_prediction):
            alone = suite.run(mock.MagicMock(), "cpu", 32, 0.5, full=False)
            full = suite.run(mock.MagicMock(), "cpu", 32, 0.5, full=True)
        self.assertNotIn("centroid", alone.subset)
        self.assertIsNone(alone.full)
        self.assertIn("centroid", full.full)
        self.assertNotIn("centroid", full.subset)
        self.assertLess(suite.subset_size, len(suite))
        self.assertEqual(alone.n_samples, suite.subset_size)
        self.assertEqual(full.n_samples, len(suite))
        for family in alone.subset:
            a, b = alone.subset[family], full.subset[family]
            self.assertEqual(a.n, b.n, family)
            self.assertEqual(a.goal, b.goal, family)
            self.assertAlmostEqual(a.iou, b.iou, places=9)
            self.assertAlmostEqual(a.bf10, b.bf10, places=9)
        self.assertGreater(full.full["circle_mask"].n, alone.subset["circle_mask"].n)

    def test_approved_samples_are_always_in_the_subset_and_reported(self) -> None:
        picked = self.examples[3]
        suite = self._suite(
            batch_size=2, epoch_subset=0.05, approved=frozenset({(int(picked.location_id), picked.image_key)})
        )
        self.assertIn(picked, suite.subset_dataset.examples)
        with mock.patch("sam2_segmentation_trainer.validation.predict_variants", _shifted_prediction):
            result = suite.run(mock.MagicMock(), "cpu", 32, 0.5, full=False)
        self.assertGreaterEqual(result.subset["circle_mask"].approved_n, 1)

    def test_inventory_refresh_keeps_the_subset_in_step(self) -> None:
        suite = self._suite(epoch_subset=0.5)
        before = suite.subset_size
        suite.replace_examples(self.examples[:4])
        self.assertEqual(len(suite), 4)
        self.assertLessEqual(suite.subset_size, 4)
        self.assertLess(suite.subset_size, before + 1)


class RealModelPathTests(unittest.TestCase):
    """Run the unpatched model path (encode once, decode every family) against a small fake SAM."""

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        make_crops(self.root, "RC1", n=6)
        self.examples = index_volumes(volumes=["RC1"], root=self.root)
        self.tiles = build_tile_context(self.examples, load_circles=False)
        self.suite = ValidationSuite(
            self.examples, modes=FAMILIES, primary="circle_mask", image_size=32, num_workers=0,
            tiles=self.tiles, seed=1, core_band_px=2.0, batch_size=4, epoch_subset=0.5,
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_every_family_decodes_against_its_own_image(self) -> None:
        from tests.test_mixed_prompts import _FakeSam

        model = _FakeSam()
        records, _ = collect_family_records(
            model, self.suite.loader(full=True), "cpu", 32, 0.5, core_band_px=2.0, metric_threads=2
        )
        self.assertEqual(sorted(records), sorted(FAMILIES))
        for family in FAMILIES:
            self.assertEqual(len(records[family]), len(self.examples))
        # One prompt-encoder call per (box?, dense mask?) group per batch, not one per family and image.
        self.assertLessEqual(len(model.sam_prompt_encoder.calls), 2 * 2)

    def test_suite_run_and_the_benchmark_timing_use_the_real_path(self) -> None:
        from sam2_segmentation_trainer.train import _time_validation_suite
        from tests.test_mixed_prompts import _FakeSam

        model = _FakeSam()
        result = self.suite.run(model, "cpu", 32, 0.5, full=True)
        self.assertEqual(sorted(result.full), sorted(FAMILIES))
        self.assertEqual(sorted(result.subset), sorted(self.suite.subset_modes))
        times = _time_validation_suite(model, self.suite, "cpu", 32, 0.5, warmup=1, steps=4)
        self.assertEqual(sorted(times), ["full", "subset"])


class CropTests(unittest.TestCase):
    def _case(self, seed: int):
        rng = np.random.default_rng(seed)
        size = 220
        yy, xx = np.mgrid[:size, :size]
        gt = (yy - 100) ** 2 + (xx - 90) ** 2 < 40**2
        pred = (yy - 104) ** 2 + (xx - 96) ** 2 < 44**2
        neighbor = (yy - 100) ** 2 + (xx - 180) ** 2 < 30**2
        pred = pred | ((yy - 100) ** 2 + (xx - 172) ** 2 < 8**2)  # leaks into the neighbor
        inp = (yy - 98) ** 2 + (xx - 88) ** 2 < 48**2
        weight = rng.uniform(0.05, 1.0, size=(size, size)).astype(np.float32)
        labels = neighbor_core_labels([neighbor], gt, 10.0)
        return pred, gt, weight, labels, inp

    def test_scoring_in_the_bounding_box_changes_nothing(self) -> None:
        for seed in range(4):
            pred, gt, weight, labels, inp = self._case(seed)
            truth = ground_truth_terms(gt, 10.0, 200)
            kwargs = dict(weight_np=weight, neighbor_core=labels, input_np=inp, mode="near_correct", band=10.0)
            cropped = score_sample(pred, gt, {"volume": "A"}, truth=truth, crop=True, **kwargs)
            whole = score_sample(pred, gt, {"volume": "A"}, truth=truth, crop=False, **kwargs)
            self.assertEqual(cropped.keys(), whole.keys())
            for key in whole:
                if isinstance(whole[key], float):
                    self.assertAlmostEqual(cropped[key], whole[key], places=9, msg=key)
                else:
                    self.assertEqual(cropped[key], whole[key], key)
            self.assertTrue(whole["merge"] or whole["leak_px"] > 0)

    def test_without_a_precomputed_truth(self) -> None:
        pred, gt, weight, labels, _inp = self._case(0)
        kwargs = dict(weight_np=weight, neighbor_core=labels, input_np=None, mode="point", band=10.0)
        shared = score_sample(pred, gt, {}, truth=ground_truth_terms(gt, 10.0, 200), **kwargs)
        alone = score_sample(pred, gt, {}, truth=None, **kwargs)
        self.assertEqual(shared, alone)


if __name__ == "__main__":
    unittest.main()
