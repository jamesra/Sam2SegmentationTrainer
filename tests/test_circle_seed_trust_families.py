"""The ``circle_oversize`` validation family and the evaluation-only seed taper in the families path, the validation
suite and ``sam2-em-eval --goal`` (docs/circle_seed_trust_eval.md). CPU only, small images."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import unittest
import zlib
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from omegaconf import OmegaConf

from annotation_crops.negatives import rasterize_circle
from annotation_crops.prompt_encoding import SEED_LOGIT
from sam2_segmentation_trainer.data.circle_seed_trust import (
    DEFAULT_OVERSIZE_RANGE,
    OVERSIZE_STREAM,
    SeedCircle,
    oversize_circle,
    seed_taper_weight,
    tapered_seed_mask_input,
)
from sam2_segmentation_trainer.data.dataset import (
    CIRCLE_PROMPTS,
    FAMILIES_ONLY_PROMPTS,
    NO_NEGATIVE_PROMPTS,
    VAL_FAMILY_CODES,
    VAL_MIXED_TYPES,
    VAL_PROMPTS,
    EMSegDataset,
)
from sam2_segmentation_trainer.data.manifest import index_volumes
from sam2_segmentation_trainer.data.prompts import (
    CircleSeedJitter,
    PromptConfig,
    PromptType,
    annotator_circle,
    inscribed_circle_xyr,
    negative_region_canvases,
)
from sam2_segmentation_trainer.data.tiles import build_tile_context

try:
    from tests.helpers import make_crops
except ImportError:
    from helpers import make_crops

HAS_SAM2 = importlib.util.find_spec("sam2") is not None
CONFIG = Path(__file__).resolve().parents[1] / "sam2_segmentation_trainer" / "config" / "em_finetune.yaml"
SCALE = 0.95
TENSORS = ("point", "point_label", "boxes", "dense_mask", "prompt_fg", "prompt_input")
FAMILIES = ["centroid", "point", "circle_mask", "correction", "near_correct", "circle_mask_noneg", "circle_annotator"]
CIRCLE_FAMILIES = ["circle_mask", "circle_mask_noneg", "circle_annotator", "circle_oversize"]

DEFAULT_FINGERPRINTS = {
    "plain": "1f87abccc0efca2009d136ffdf8a7675eb046af10a1bce6ea2cd3134af1bd26b",
    "circles": "16c44568cdfa2bb7e8ea15c66c90c2c78bf053488527c930c837ccd7bfc5e36f",
}
"""Digests of every family's prompt tensors on the two fixtures below, taken with the code before these options
existed: the default path must stay bit-identical."""


def write_circles(crops: Path) -> None:
    """An inscribed circle for every other tile's first location, and an annotated circle (a negative) in some."""
    rows = [json.loads(line) for line in (crops / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line]
    cells = {}
    for i, row in enumerate(rows):
        entries = []
        if i % 2 == 0:
            entries.append({"location_id": row["locationIds"][0], "center_x": 16.0, "center_y": 16.5, "radius": 7.6,
                            "structure_label": "", "has_polygon_mask": True, "source": "inscribed"})
        if i % 4 == 1:
            entries.append({"location_id": 900 + i, "center_x": 27.0, "center_y": 5.0, "radius": 4.0,
                            "structure_label": "", "has_polygon_mask": False, "source": "annotated"})
        if entries:
            cells[row["imageKey"]] = entries
    (crops / "circle_annotations.json").write_text(
        json.dumps({"version": 1, "diameter_scale": SCALE, "grid_cells": cells}), encoding="utf-8"
    )


class _Fixture(unittest.TestCase):
    CIRCLES = False

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        root = Path(self._tmp.name)
        crops = make_crops(root, "RC1", n=9)
        if self.CIRCLES:
            write_circles(crops)
        self.examples = index_volumes(volumes=["RC1"], root=root)
        self.tiles = build_tile_context(self.examples, circle_scale=SCALE)
        # The synthetic targets are 16 px squares, too small for the default 6 px minimum radius.
        self.cfg = PromptConfig(circle_seed_jitter=CircleSeedJitter(min_radius_px=1.0))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def dataset(self, families, **options) -> EMSegDataset:
        return EMSegDataset(
            self.examples, split="val", image_size=32, tiles=self.tiles, prompt_cfg=self.cfg, seed=3,
            core_band_px=2.0, val_families=families, **options,
        )

    @staticmethod
    def by_family(item) -> dict:
        return {variant["val_family"]: variant for variant in item["variants"]}

    @staticmethod
    def fingerprint(dataset) -> str:
        digest = hashlib.sha256()
        for index in range(len(dataset)):
            for variant in dataset[index]["variants"]:
                digest.update(variant["val_family"].encode())
                digest.update(variant["prompt_type"].encode())
                for key in TENSORS:
                    tensor = variant[key]
                    digest.update(b"None" if tensor is None else tensor.numpy().tobytes())
        return digest.hexdigest()

    def seed_circle(self, dataset, index) -> SeedCircle:
        """The inscribed seed's centre and true radius, as the dataset finds them (the tile's circle, else fitted)."""
        example = dataset.examples[index]
        own = self.tiles.circle_for((example.volume, example.image_key), example.location_id)
        if own is not None:
            return SeedCircle(own.center_x, own.center_y, own.seed_radius(SCALE) / SCALE)
        target = dataset[index]["mask"].numpy().astype(bool)
        x, y, r = inscribed_circle_xyr(target, SCALE)
        return SeedCircle(x, y, r / SCALE)

    @staticmethod
    def sample_id(example) -> int:
        return zlib.crc32(f"{example.volume}|{example.image_key}|{example.location_id}".encode("utf-8"))


class FamilyRegistrationTests(unittest.TestCase):
    def test_circle_oversize_is_last_so_every_other_family_keeps_its_number(self) -> None:
        self.assertEqual(VAL_PROMPTS[-1], "circle_oversize")
        self.assertEqual(VAL_FAMILY_CODES["circle_oversize"], 7)
        self.assertEqual([VAL_FAMILY_CODES[name] for name in FAMILIES], list(range(7)))

    def test_it_is_a_circle_seed_with_negatives_built_only_by_the_families_path(self) -> None:
        self.assertIs(VAL_MIXED_TYPES["circle_oversize"], PromptType.CIRCLE_SEED)
        self.assertIn("circle_oversize", CIRCLE_PROMPTS)
        self.assertIn("circle_oversize", FAMILIES_ONLY_PROMPTS)
        self.assertNotIn("circle_oversize", NO_NEGATIVE_PROMPTS)
        with self.assertRaisesRegex(ValueError, "val_families"):
            EMSegDataset([], split="val", val_prompt="circle_oversize")

    def test_the_shipped_config_does_not_validate_it(self) -> None:
        cfg = OmegaConf.load(CONFIG)
        self.assertNotIn("circle_oversize", list(cfg.eval.val_prompts))

    @unittest.skipUnless(HAS_SAM2, "validation imports sam2")
    def test_the_suite_defaults_leave_it_out(self) -> None:
        from sam2_segmentation_trainer.validation import DEFAULT_VAL_PROMPTS, NEG_SPLIT_PROMPTS

        self.assertNotIn("circle_oversize", DEFAULT_VAL_PROMPTS)
        self.assertIn("circle_oversize", NEG_SPLIT_PROMPTS)


class DefaultPathTests(_Fixture):
    def test_the_default_families_are_bit_identical_to_before(self) -> None:
        self.assertEqual(self.fingerprint(self.dataset(FAMILIES)), DEFAULT_FINGERPRINTS["plain"])

    def test_an_explicit_off_is_the_default(self) -> None:
        self.assertEqual(
            self.fingerprint(self.dataset(FAMILIES, circle_seed_taper=None, circle_oversize_range=(1.5, 1.6))),
            DEFAULT_FINGERPRINTS["plain"],
        )

    def test_scoring_circle_oversize_alongside_does_not_move_any_other_family(self) -> None:
        before, after = self.dataset(FAMILIES), self.dataset(FAMILIES + ["circle_oversize"])
        for index in range(len(before)):
            old, new = self.by_family(before[index]), self.by_family(after[index])
            for family in FAMILIES:
                for key in TENSORS:
                    a, b = old[family][key], new[family][key]
                    self.assertTrue(b is None if a is None else torch.equal(a, b), (index, family, key))

    def test_the_taper_is_evaluation_only(self) -> None:
        with self.assertRaisesRegex(ValueError, "evaluation only"):
            EMSegDataset(self.examples, split="train", tiles=self.tiles, prompt_cfg=self.cfg, circle_seed_taper=0.5)
        with self.assertRaisesRegex(ValueError, "circle-seed-taper"):
            self.dataset(["circle_mask"], circle_seed_taper=SCALE)


class DefaultPathWithCircleFileTests(DefaultPathTests):
    CIRCLES = True

    def test_the_default_families_are_bit_identical_to_before(self) -> None:
        self.assertEqual(self.fingerprint(self.dataset(FAMILIES)), DEFAULT_FINGERPRINTS["circles"])

    def test_an_explicit_off_is_the_default(self) -> None:
        self.assertEqual(
            self.fingerprint(self.dataset(FAMILIES, circle_seed_taper=None)), DEFAULT_FINGERPRINTS["circles"]
        )


class TaperedFamiliesTests(_Fixture):
    CIRCLES = True

    def test_only_the_circle_seeds_dense_prompt_changes(self) -> None:
        families = FAMILIES + ["circle_oversize"]
        off, on = self.dataset(families), self.dataset(families, circle_seed_taper=0.5)
        changed = 0
        for index in range(len(off)):
            a_item, b_item = off[index], on[index]
            self.assertTrue(torch.equal(a_item["mask"], b_item["mask"]))  # the same target and scoring
            a, b = self.by_family(a_item), self.by_family(b_item)
            for family in families:
                for key in TENSORS:
                    x, y = a[family][key], b[family][key]
                    same = y is None if x is None else (y is not None and torch.equal(x, y))
                    if key == "dense_mask" and family in CIRCLE_FAMILIES:
                        changed += not same
                    else:
                        self.assertTrue(same, (index, family, key))
                self.assertEqual(a[family]["prompt_type"], b[family]["prompt_type"])
        self.assertGreater(changed, 0)

    def test_the_dense_prompt_is_the_taper_of_each_familys_own_circle(self) -> None:
        dataset = self.dataset(CIRCLE_FAMILIES, circle_seed_taper=0.5)
        taper = dataset.circle_seed_taper
        self.assertEqual((taper.full_frac, taper.zero_frac), (0.5, SCALE))
        checked = 0
        for index in range(len(dataset)):
            item = dataset[index]
            by = self.by_family(item)
            example = dataset.examples[index]
            shape = tuple(item["mask"].shape)
            regions = dataset._other_annotations(example, shape)
            canvases = negative_region_canvases(regions, self.cfg) if regions else []
            inscribed = self.seed_circle(dataset, index)
            for family in CIRCLE_FAMILIES:
                variant = by[family]
                disc = variant["prompt_input"].numpy().astype(bool)
                circle = inscribed
                if family == "circle_annotator" and variant["prompt_type"] == "circle_annotator":
                    # Its centre pixel and drawn radius; the taper's true radius is the drawn one over the scale.
                    drawn: dict = {}
                    rng = np.random.default_rng([3, self.sample_id(example), VAL_FAMILY_CODES[family], 2])
                    expected_disc = annotator_circle(
                        item["mask"].numpy().astype(bool), rng, self.cfg.circle_seed_jitter, info=drawn
                    )
                    np.testing.assert_array_equal(disc, expected_disc)
                    circle = SeedCircle(drawn["col"] + 0.5, drawn["row"] + 0.5, drawn["radius"] / SCALE)
                elif family == "circle_oversize":
                    rng = np.random.default_rng([3, self.sample_id(example), VAL_FAMILY_CODES[family], OVERSIZE_STREAM])
                    expected_disc, circle, _factor = oversize_circle(shape, inscribed, SCALE, rng)
                    np.testing.assert_array_equal(disc, expected_disc)
                else:
                    np.testing.assert_array_equal(
                        disc, rasterize_circle(shape, circle.center_x, circle.center_y, circle.radius * SCALE)
                    )
                expected = tapered_seed_mask_input(
                    disc, seed_taper_weight(shape, circle, taper),
                    [] if family in NO_NEGATIVE_PROMPTS else canvases, SEED_LOGIT,
                )
                np.testing.assert_array_equal(variant["dense_mask"].numpy(), expected, err_msg=f"{index} {family}")
                checked += 1
        self.assertGreater(checked, 2 * len(dataset))

    def test_the_annotator_circles_taper_claims_its_own_disc(self) -> None:
        dataset = self.dataset(["circle_annotator"], circle_seed_taper=0.5)
        drawn = 0
        for index in range(len(dataset)):
            variant = self.by_family(dataset[index])["circle_annotator"]
            if variant["prompt_type"] != "circle_annotator":
                continue
            disc = variant["prompt_input"].numpy()
            positive = variant["dense_mask"].numpy() > 0
            # 32 px replicated to 256: every positive low-res cell lies over the disc.
            self.assertTrue(np.repeat(np.repeat(disc, 8, axis=0), 8, axis=1)[positive].all())
            drawn += 1
        self.assertGreater(drawn, 0)


class OversizeFamilyTests(_Fixture):
    def test_the_circle_is_the_inscribed_one_grown_by_a_factor_in_range_and_deterministic(self) -> None:
        first = self.dataset(["circle_mask", "circle_oversize"], circle_oversize_range=(1.4, 1.8))
        alone = self.dataset(["circle_oversize"], circle_oversize_range=(1.4, 1.8))
        for index in range(len(first)):
            item = first[index]
            by = self.by_family(item)
            example = first.examples[index]
            shape = tuple(item["mask"].shape)
            rng = np.random.default_rng([3, self.sample_id(example), 7, OVERSIZE_STREAM])
            disc, grown, factor = oversize_circle(shape, self.seed_circle(first, index), SCALE, rng, (1.4, 1.8))
            self.assertTrue(1.4 <= factor <= 1.8)
            np.testing.assert_array_equal(by["circle_oversize"]["prompt_input"].numpy().astype(bool), disc)
            # Larger than the inscribed seed it grows from.
            self.assertGreater(int(disc.sum()), int(by["circle_mask"]["prompt_input"].sum()))
            self.assertEqual(by["circle_oversize"]["prompt_type"], "circle_seed")
            again = self.by_family(alone[index])["circle_oversize"]
            for key in TENSORS:
                a, b = by["circle_oversize"][key], again[key]
                self.assertTrue(b is None if a is None else torch.equal(a, b), (index, key))


class OversizeFamilyWithCircleFileTests(OversizeFamilyTests):
    CIRCLES = True


@unittest.skipUnless(HAS_SAM2, "evaluate and validation import sam2")
class EvaluateOptionTests(_Fixture):
    def test_the_suite_passes_the_options_to_both_datasets(self) -> None:
        from sam2_segmentation_trainer.validation import ValidationSuite

        suite = ValidationSuite(
            self.examples, modes=["circle_mask", "circle_oversize"], primary="circle_mask", image_size=32,
            num_workers=0, tiles=self.tiles, prompt_cfg=self.cfg, circle_scale=SCALE, core_band_px=2.0,
            circle_seed_taper=0.5, circle_oversize_range=(1.3, 1.4),
        )
        for dataset in (suite.full_dataset, suite.subset_dataset):
            self.assertEqual((dataset.circle_seed_taper.full_frac, dataset.circle_seed_taper.zero_frac), (0.5, SCALE))
            self.assertEqual(dataset.circle_oversize_range, (1.3, 1.4))
        plain = ValidationSuite(
            self.examples, modes=["circle_mask"], primary="circle_mask", image_size=32, num_workers=0,
            tiles=self.tiles, prompt_cfg=self.cfg,
        )
        self.assertIsNone(plain.full_dataset.circle_seed_taper)

    def test_the_goal_summary_records_the_options_only_when_used(self) -> None:
        from sam2_segmentation_trainer import evaluate as evaluate_module
        from sam2_segmentation_trainer.validation import ValidationSuite

        request = self.request()
        plain = ValidationSuite(
            self.examples, modes=["circle_mask"], primary="circle_mask", image_size=32, num_workers=0,
            tiles=self.tiles, prompt_cfg=self.cfg,
        )
        self.assertEqual(evaluate_module._circle_seed_options(request, plain), {})
        tapered = ValidationSuite(
            self.examples, modes=["circle_mask", "circle_oversize"], primary="circle_mask", image_size=32,
            num_workers=0, tiles=self.tiles, prompt_cfg=self.cfg, circle_seed_taper=0.5,
        )
        self.assertEqual(
            evaluate_module._circle_seed_options(request, tapered),
            {"taper": {"full_frac": 0.5, "zero_frac": SCALE, "ramp": "cosine"}, "oversize_range": [1.2, 2.0]},
        )

    def test_the_one_prompt_evaluation_refuses_the_taper(self) -> None:
        from sam2_segmentation_trainer.evaluate import evaluate

        with self.assertRaisesRegex(ValueError, "--goal"):
            evaluate(self.request(circle_seed_taper=0.5))

    def test_the_command_line_parses_the_options(self) -> None:
        from unittest import mock

        from sam2_segmentation_trainer import evaluate as evaluate_module

        seen = {}

        def capture(request):
            seen["request"] = request
            return {}

        with mock.patch.object(evaluate_module, "evaluate", capture):
            evaluate_module.main(["--zero-shot", "--goal", "--circle-seed-taper", "0.5", "--circle-oversize-range",
                                  "1.3,1.9", "--split", str(Path(self._tmp.name) / "split.json")])
        self.assertEqual(seen["request"].circle_seed_taper, 0.5)
        self.assertEqual(seen["request"].circle_oversize_range, (1.3, 1.9))
        with mock.patch.object(evaluate_module, "evaluate", capture):
            evaluate_module.main(["--zero-shot", "--goal", "--split", str(Path(self._tmp.name) / "split.json")])
        self.assertIsNone(seen["request"].circle_seed_taper)
        self.assertEqual(seen["request"].circle_oversize_range, DEFAULT_OVERSIZE_RANGE)

    def request(self, **kwargs):
        from sam2_segmentation_trainer.evaluate import EvalRequest

        return EvalRequest(
            finetuned=None, base_ckpt=Path("b.pt"), model_cfg="x", data_dir=Path("."), volumes=["RC1"],
            split_path=Path("s.json"), image_size=32, threshold=0.5, out_dir=Path("o"), grid_every=0, max_grids=0,
            num_workers=0, prompts=("circle_mask",), **kwargs,
        )


if __name__ == "__main__":
    unittest.main()
