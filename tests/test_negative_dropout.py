"""Negative-context dropout (training), the circle_mask_noneg family and the with_neg/no_neg split (validation).

docs/mask_refinement_plan.md, sections 3.6 and 6.
"""

from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from annotation_crops.negatives import KIND_POLYGON, NegativeRegion
from sam2_segmentation_trainer.data.dataset import EMSegDataset, EpochCounter, collate_em
from sam2_segmentation_trainer.data.manifest import index_volumes
from sam2_segmentation_trainer.data.prompts import (
    PromptConfig,
    PromptType,
    build_prompt,
    kept_negative_indices,
    negative_region_canvases,
)
from sam2_segmentation_trainer.data.tiles import build_tile_context
from sam2_segmentation_trainer.validation import (
    ValidationSuite,
    format_summary,
    log_results,
    summarize_mode,
)

try:
    from tests.helpers import make_crops
except ImportError:
    from helpers import make_crops


def _square(size: int, lo: int, hi: int, lo2: int | None = None, hi2: int | None = None) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[lo:hi, (lo if lo2 is None else lo2) : (hi if hi2 is None else hi2)] = True
    return mask


def _cfg(**extra) -> PromptConfig:
    # Only the plain circle seed and points, so every draw is a known family.
    base = dict(
        prob_circle_seed=0.5, prob_circle_annotator=0.0, prob_circle_pull_through=0.0,
        prob_points_only=0.5, prob_adjacent_poly_seed=0.0, prob_box_only=0.0,
        prob_box_points=0.0, prob_subset_seed=0.0, prob_correction=0.0, prob_near_correct=0.0,
        bg_point_prob_circle=1.0, bg_point_prob_polygon=1.0,
    )
    base.update(extra)
    return PromptConfig(**base)


class KeptIndicesTests(unittest.TestCase):
    def test_zero_rates_keep_everything_and_draw_nothing(self) -> None:
        rng = np.random.default_rng(3)
        before = rng.bit_generator.state
        self.assertEqual(kept_negative_indices([0, 2, 5], _cfg(context_dropout=0.0, region_dropout=0.0), rng), [0, 2, 5])
        self.assertEqual(rng.bit_generator.state, before)

    def test_full_context_dropout_keeps_nothing(self) -> None:
        cfg = _cfg(context_dropout=1.0, region_dropout=0.0)
        self.assertEqual(kept_negative_indices([0, 1, 2], cfg, np.random.default_rng(0)), [])

    def test_full_region_dropout_keeps_nothing_and_none_keeps_all(self) -> None:
        self.assertEqual(
            kept_negative_indices([0, 1, 2], _cfg(context_dropout=0.0, region_dropout=1.0), np.random.default_rng(0)), []
        )
        self.assertEqual(
            kept_negative_indices([0, 1, 2], _cfg(context_dropout=0.0, region_dropout=0.0), np.random.default_rng(0)),
            [0, 1, 2],
        )

    def test_rates_match_their_probabilities(self) -> None:
        cfg = _cfg(context_dropout=0.2, region_dropout=0.3)
        empty = kept = total = 0
        for seed in range(4000):
            result = kept_negative_indices(list(range(5)), cfg, np.random.default_rng(seed))
            empty += int(not result)
            if result:
                kept += len(result)
                total += 5
        self.assertAlmostEqual(empty / 4000, 0.2 + 0.8 * 0.3**5, delta=0.03)
        self.assertAlmostEqual(kept / total, 0.7, delta=0.03)  # given the context survived

    def test_no_regions_draws_nothing(self) -> None:
        rng = np.random.default_rng(1)
        before = rng.bit_generator.state
        self.assertEqual(kept_negative_indices([], _cfg(context_dropout=1.0), rng), [])
        self.assertEqual(rng.bit_generator.state, before)

    def test_rates_are_validated(self) -> None:
        self.assertEqual((PromptConfig().context_dropout, PromptConfig().region_dropout), (0.2, 0.2))
        for name in ("context_dropout", "region_dropout"):
            with self.assertRaises(ValueError):
                replace(PromptConfig(), **{name: 1.2}).validate()
            with self.assertRaises(ValueError):
                replace(PromptConfig(), **{name: -0.1}).validate()
        loaded = PromptConfig.from_cfg({"negative": {"context_dropout": 0.5, "region_dropout": 0.1}})
        self.assertEqual((loaded.context_dropout, loaded.region_dropout), (0.5, 0.1))


class BuildPromptDropoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.target = _square(64, 16, 48)
        self.circle = _square(64, 24, 40)
        self.regions = [
            NegativeRegion(10, KIND_POLYGON, _square(64, 2, 12)),
            NegativeRegion(11, KIND_POLYGON, _square(64, 52, 62)),
            NegativeRegion(12, KIND_POLYGON, _square(64, 2, 12, 50, 62)),
        ]

    def _prompt(self, cfg, regions=None, seed=0, canvases=False, **extra):
        regions = self.regions if regions is None else regions
        return build_prompt(
            PromptType.CIRCLE_SEED,
            target_mask=self.target,
            circle_mask=self.circle,
            regions=regions,
            cfg=cfg,
            rng=np.random.default_rng(seed),
            negative_canvases=negative_region_canvases(regions, cfg) if canvases else None,
            **extra,
        )

    def test_dropout_is_off_unless_asked_for(self) -> None:
        cfg = _cfg(context_dropout=1.0, region_dropout=1.0)
        plain = self._prompt(cfg)
        self.assertLess(float(plain.dense_logits.min()), -5.0)
        self.assertTrue((plain.point_labels == 0).any())  # background points ride along

    def test_context_dropout_removes_the_negative_area_and_the_background_points(self) -> None:
        cfg = _cfg(context_dropout=1.0)
        dropped = self._prompt(cfg, negative_dropout=True)
        bare = self._prompt(cfg, regions=[])
        self.assertTrue((dropped.point_labels == 1).all())
        np.testing.assert_array_equal(dropped.dense_logits, bare.dense_logits)
        self.assertGreaterEqual(float(dropped.dense_logits.min()), 0.0)

    def test_region_dropout_keeps_the_surviving_regions_whole(self) -> None:
        cfg = _cfg(context_dropout=0.0, region_dropout=0.5)
        outcomes = set()
        for seed in range(40):
            candidates = list(range(len(self.regions)))
            kept = kept_negative_indices(candidates, cfg, np.random.default_rng(seed))
            outcomes.add(tuple(kept))
            expected = self._prompt(_cfg(context_dropout=0.0, region_dropout=0.0), regions=[self.regions[i] for i in kept])
            # The surviving regions' dense prompt matches a prompt built from just those regions, with or without
            # the shared canvases (which are computed for all regions and must be picked by index).
            for shared in (False, True):
                got = self._prompt(cfg, seed=seed, canvases=shared, negative_dropout=True)
                np.testing.assert_allclose(got.dense_logits, expected.dense_logits, atol=1e-5, err_msg=str((seed, shared)))
        self.assertGreater(len(outcomes), 3)  # different seeds drop different regions

    def test_shared_canvases_and_masks_give_the_same_dropped_prompt(self) -> None:
        cfg = _cfg(context_dropout=0.0, region_dropout=0.5)
        for seed in range(30):
            a = self._prompt(cfg, seed=seed, canvases=True, negative_dropout=True)
            b = self._prompt(cfg, seed=seed, canvases=False, negative_dropout=True)
            np.testing.assert_allclose(a.dense_logits, b.dense_logits, atol=1e-5)
            np.testing.assert_array_equal(a.point_labels, b.point_labels)

    def test_same_rng_same_prompt(self) -> None:
        cfg = _cfg(context_dropout=0.3, region_dropout=0.4)
        a = self._prompt(cfg, seed=5, negative_dropout=True)
        b = self._prompt(cfg, seed=5, negative_dropout=True)
        np.testing.assert_array_equal(a.dense_logits, b.dense_logits)
        np.testing.assert_array_equal(a.point_coords, b.point_coords)

    def test_the_containment_weight_still_sees_every_neighbor(self) -> None:
        # A neighbor that overlaps the circle seed: dropped from the prompt, still a negative for the loss.
        overlap = NegativeRegion(13, KIND_POLYGON, _square(64, 28, 34, 20, 44))
        regions = [overlap] + self.regions
        reference = self._prompt(_cfg(), regions=regions)
        self.assertFalse(reference.foreground_weight[28:34, 24:40].any())
        for cfg in (_cfg(context_dropout=1.0), _cfg(context_dropout=0.0, region_dropout=1.0)):
            for shared in (False, True):
                dropped = self._prompt(cfg, regions=regions, canvases=shared, negative_dropout=True)
                np.testing.assert_array_equal(dropped.foreground_weight, reference.foreground_weight)
                self.assertTrue((dropped.dense_logits[28 * 4 : 34 * 4, 24 * 4 : 40 * 4] > 0).any())  # seed shows

    def test_a_mask_to_correct_can_still_merge_a_dropped_neighbor(self) -> None:
        cfg = _cfg(prob_circle_seed=0.0, prob_correction=1.0, prob_points_only=0.0, context_dropout=1.0)
        near = NegativeRegion(20, KIND_POLYGON, _square(64, 20, 44, 46, 60))
        merged_any = False
        for seed in range(60):
            prompt = build_prompt(
                PromptType.CORRECTION,
                target_mask=self.target,
                circle_mask=None,
                regions=[near],
                cfg=cfg,
                rng=np.random.default_rng(seed),
                negative_dropout=True,
            )
            merged_any |= bool((prompt.input_mask & near.mask).any())
            self.assertTrue((prompt.point_labels == 1).all())  # nothing of the context survives
        self.assertTrue(merged_any)


class DatasetDropoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        make_crops(self.root, "RC1", n=6)
        self.examples = index_volumes(volumes=["RC1"], root=self.root)
        self.tiles = build_tile_context(self.examples)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _dataset(self, cfg: PromptConfig, split: str = "train", **extra) -> EMSegDataset:
        return EMSegDataset(
            self.examples, split=split, image_size=32, tiles=self.tiles, prompt_cfg=cfg, seed=7,
            epoch=EpochCounter(), **extra,
        )

    def _items_with_negatives(self, dataset: EMSegDataset):
        for index in range(len(dataset)):
            for epoch in range(6):
                dataset.epoch.value = epoch
                item = dataset[index]
                if item["sample_meta"].has_negatives and len(item["variants"]) > 1:
                    yield item

    def test_every_training_variant_can_lose_its_negatives(self) -> None:
        # No variant is a fixed production prompt, so the first variant drops its context like the others.
        cfg = _cfg(context_dropout=1.0, variants_per_sample=3)
        seen = 0
        for item in self._items_with_negatives(self._dataset(cfg)):
            for variant in item["variants"]:
                dense = variant["dense_mask"]
                self.assertTrue(dense is None or float(dense.min()) >= 0.0)
                self.assertTrue((variant["point_label"] == 1).all())
            seen += 1
        self.assertGreater(seen, 3)

    def test_no_dropout_keeps_the_negatives_in_every_variant(self) -> None:
        cfg = _cfg(context_dropout=0.0, region_dropout=0.0, variants_per_sample=3)
        seen = 0
        for item in self._items_with_negatives(self._dataset(cfg)):
            for variant in item["variants"]:
                self.assertIsNotNone(variant["dense_mask"])
                self.assertLess(float(variant["dense_mask"].min()), -1.0)
            seen += 1
        self.assertGreater(seen, 3)

    def test_validation_never_drops(self) -> None:
        cfg = _cfg(context_dropout=1.0, region_dropout=1.0)
        dataset = self._dataset(cfg, split="val", val_prompt="circle_mask")
        seen = 0
        for index in range(len(dataset)):
            item = dataset[index]
            if item["sample_meta"].has_negatives:
                self.assertLess(float(item["variants"][0]["dense_mask"].min()), -1.0)
                seen += 1
        self.assertGreater(seen, 0)

    def test_the_loss_weights_do_not_change_with_dropout(self) -> None:
        kwargs = dict(leak_weight=3.0, core_band_px=2.0, edge_tau_px=8.0)
        a = self._dataset(_cfg(context_dropout=1.0), **kwargs)
        b = self._dataset(_cfg(context_dropout=0.0, region_dropout=0.0), **kwargs)
        for dataset in (a, b):  # the training transform is random; the weights must not depend on dropout alone
            dataset.transform = lambda image, masks: {"image": image, "masks": masks}
        for index in range(len(a)):
            x, y = a[index], b[index]
            self.assertTrue(torch.equal(x["mask"], y["mask"]))
            for key in ("edge_weight", "neighbor_core"):
                if x[key] is None:
                    self.assertIsNone(y[key])
                else:
                    self.assertTrue(torch.equal(x[key], y[key]), key)


FAMILIES = ["circle_mask", "circle_mask_noneg", "point"]


class NoNegativeFamilyTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        make_crops(self.root, "RC1", n=9)
        self.examples = index_volumes(volumes=["RC1"], root=self.root)
        self.tiles = build_tile_context(self.examples)
        self.cfg = _cfg(context_dropout=1.0, region_dropout=1.0)  # validation must ignore these

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _dataset(self, **extra) -> EMSegDataset:
        return EMSegDataset(
            self.examples, split="val", image_size=32, tiles=self.tiles, prompt_cfg=self.cfg, seed=3,
            core_band_px=2.0, val_families=FAMILIES, **extra,
        )

    def test_the_family_is_the_circle_seed_without_any_negative(self) -> None:
        dataset = self._dataset()
        with_neg = without = 0
        for index in range(len(dataset)):
            item = dataset[index]
            by_family = {v["val_family"]: v for v in item["variants"]}
            plain, bare = by_family["circle_mask"], by_family["circle_mask_noneg"]
            self.assertEqual(bare["prompt_type"], "circle_seed")
            self.assertTrue((bare["point_label"] == 1).all())
            self.assertGreaterEqual(float(bare["dense_mask"].min()), 0.0)
            self.assertTrue(torch.equal(plain["prompt_fg"] > 0, bare["prompt_fg"] > 0) or item["sample_meta"].has_negatives)
            self.assertIsNotNone(item["neighbor_core"])  # the metrics still see the neighbors
            if item["sample_meta"].has_negatives:
                with_neg += 1
                self.assertLess(float(plain["dense_mask"].min()), -1.0)
                self.assertFalse(torch.equal(plain["dense_mask"], bare["dense_mask"]))
            else:
                without += 1
                self.assertTrue(torch.equal(plain["dense_mask"], bare["dense_mask"]))
        self.assertGreater(with_neg, 0)
        self.assertGreater(without, 0)

    def test_same_circle_seed_and_the_same_target_in_both_families(self) -> None:
        item = self._dataset()[0]
        by_family = {v["val_family"]: v for v in item["variants"]}
        # The foreground click is the circle's center in both; only the plain family adds background points.
        self.assertTrue(torch.equal(by_family["circle_mask"]["point"][0], by_family["circle_mask_noneg"]["point"][0]))
        self.assertTrue(
            torch.equal(by_family["circle_mask"]["prompt_input"], by_family["circle_mask_noneg"]["prompt_input"])
        )

    def test_single_family_loader_agrees_with_the_family_pass(self) -> None:
        family = self._dataset()
        alone = EMSegDataset(
            self.examples, split="val", image_size=32, tiles=self.tiles, prompt_cfg=self.cfg, seed=3,
            core_band_px=2.0, val_prompt="circle_mask_noneg",
        )
        for index in range(len(family)):
            a = {v["val_family"]: v for v in family[index]["variants"]}["circle_mask_noneg"]
            (b,) = alone[index]["variants"]
            self.assertGreaterEqual(float(b["dense_mask"].min()), 0.0)
            self.assertTrue((b["point_label"] == 1).all())
            self.assertTrue(torch.equal(a["dense_mask"], b["dense_mask"]))
            self.assertIsNotNone(alone[index]["neighbor_core"])

    def test_the_suite_scores_it_in_the_subset_and_the_full_pass_but_selects_on_the_primary(self) -> None:
        suite = ValidationSuite(
            self.examples, modes=FAMILIES, primary="circle_mask", image_size=32, num_workers=0, tiles=self.tiles,
            prompt_cfg=self.cfg, seed=3, core_band_px=2.0, epoch_subset=0.5, batch_size=3,
        )
        self.assertIn("circle_mask_noneg", suite.subset_modes)
        with self.assertRaises(ValueError):
            ValidationSuite(
                self.examples, modes=FAMILIES, primary="circle_mask_noneg_x", image_size=32, num_workers=0,
            )


def _record(has_neg: bool, hit: int, merge: bool = False, **extra):
    row = {
        "volume": "RC1", "category": "MC", "iou": 0.9, "raw_iou": 0.9, "bf10": 0.8, "bf20": 0.9, "gt_depth": 40.0,
        "core_gt": 100, "core_hit": hit, "recall": hit / 100, "pred_area": 1000, "leak_px": 0, "gt_area": 1000,
        "spill_px": 0, "merge": merge, "frag": False, "has_neg": has_neg,
    }
    row.update(extra)
    return row


class WriterStub:
    def __init__(self) -> None:
        self.tags: dict[str, float] = {}

    def add_scalar(self, tag, value, step) -> None:
        self.tags[tag] = value


class SplitTests(unittest.TestCase):
    RECORDS = [_record(True, 90, merge=True), _record(True, 80), _record(True, 100), _record(False, 100), _record(False, 98)]

    def test_the_split_partitions_the_samples(self) -> None:
        result = summarize_mode(self.RECORDS)
        self.assertEqual((result.with_neg_n, result.no_neg_n), (3, 2))
        self.assertEqual(result.with_neg_n + result.no_neg_n, result.n)
        self.assertAlmostEqual(result.with_neg.core_recall, 270 / 300)
        self.assertAlmostEqual(result.with_neg.merge_rate, 1 / 3)
        self.assertAlmostEqual(result.no_neg.core_recall, 198 / 200)
        self.assertAlmostEqual(result.no_neg.merge_rate, 0.0)

    def test_records_without_the_flag_give_no_split(self) -> None:
        rows = [{k: v for k, v in r.items() if k != "has_neg"} for r in self.RECORDS]
        result = summarize_mode(rows)
        self.assertIsNone(result.with_neg)
        self.assertIsNone(result.no_neg)

    def test_one_side_empty(self) -> None:
        result = summarize_mode([_record(True, 100), _record(True, 90)])
        self.assertIsNone(result.no_neg)
        self.assertEqual((result.with_neg_n, result.no_neg_n), (2, 0))

    def test_the_circle_families_log_the_split_and_the_summary_names_it(self) -> None:
        results = {"circle_mask": summarize_mode(self.RECORDS), "circle_mask_noneg": summarize_mode(self.RECORDS)}
        for root in ("val_subset", "val_full"):
            writer = WriterStub()
            text = log_results(writer, 3, results, "circle_mask", root=root)
            for mode in ("circle_mask", "circle_mask_noneg"):
                for side, n in (("with_neg", 3), ("no_neg", 2)):
                    self.assertEqual(writer.tags[f"{root}_neg/n/{mode}/{side}"], n)
                    self.assertIn(f"{root}_neg/core_recall/{mode}/{side}", writer.tags)
                    self.assertIn(f"{root}_neg/merge_rate/{mode}/{side}", writer.tags)
            self.assertIn("with_neg (n=3)", text)
            self.assertIn("no_neg (n=2)", text)
            self.assertIn("circle_mask_noneg core recall", text)  # listed among the other prompts
        other = WriterStub()
        log_results(other, 0, {"point": summarize_mode(self.RECORDS)}, "point", root="val_full")
        self.assertFalse(any("_neg/" in tag for tag in other.tags))  # only the circle families are split

    def test_the_primary_summary_is_unchanged_for_a_non_circle_primary(self) -> None:
        text = format_summary(0, {"point": summarize_mode(self.RECORDS)}, "point", 0.975)
        self.assertNotIn("with_neg", text)


if __name__ == "__main__":
    unittest.main()
