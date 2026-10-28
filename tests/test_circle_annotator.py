"""Annotator-like circle seeds (docs/circle_annotator_spec.md): the helper, the config block, the ``circle_annotator``
validation family (Step 1) and the training rule in ``_mixed_item`` (Step 2)."""

from __future__ import annotations

import contextlib
import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.ndimage import distance_transform_edt

from annotation_crops import negatives as negatives_module
from sam2_segmentation_trainer.data import dataset as dataset_module
from sam2_segmentation_trainer.data import prompts as prompts_module
from sam2_segmentation_trainer.data.dataset import (
    CIRCLE_PROMPTS,
    FAMILIES_ONLY_PROMPTS,
    NO_NEGATIVE_PROMPTS,
    VAL_FAMILY_CODES,
    VAL_MIXED_TYPES,
    VAL_PROMPTS,
    EMSegDataset,
    EpochCounter,
)
from sam2_segmentation_trainer.data.manifest import index_volumes
from sam2_segmentation_trainer.data.prompts import (
    CircleSeedJitter,
    PromptConfig,
    PromptType,
    annotator_circle,
)
from sam2_segmentation_trainer.data.tiles import build_tile_context
from sam2_segmentation_trainer.evaluate import PER_PROMPT_DEFAULTS, EvalRequest, evaluate
from sam2_segmentation_trainer.train import load_train_cfg
from sam2_segmentation_trainer.validation import (
    DEFAULT_VAL_PROMPTS,
    NEG_SPLIT_PROMPTS,
    ValidationSuite,
    log_results,
    summarize_mode,
)

try:
    from tests.helpers import make_crops
except ImportError:
    from helpers import make_crops

SIZE = 160


def disc_target(radius: int = 60) -> np.ndarray:
    ys, xs = np.mgrid[:SIZE, :SIZE]
    return (xs - SIZE / 2) ** 2 + (ys - SIZE / 2) ** 2 <= radius**2


def ellipse_target() -> np.ndarray:
    """A 3:1 ellipse."""
    ys, xs = np.mgrid[:SIZE, :SIZE]
    return ((xs - SIZE / 2) / 66.0) ** 2 + ((ys - SIZE / 2) / 22.0) ** 2 <= 1.0


def bar_target(height: int = 24) -> np.ndarray:
    """An elongated bar, ``height`` px high."""
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    top = SIZE // 2 - height // 2
    mask[top : top + height, 20:140] = True
    return mask


def c_target() -> np.ndarray:
    """A C shape: a thick ring with its right side cut away."""
    ys, xs = np.mgrid[:SIZE, :SIZE]
    radius2 = (xs - SIZE / 2) ** 2 + (ys - SIZE / 2) ** 2
    ring = (radius2 <= 62**2) & (radius2 >= 38**2)
    ring[:, int(SIZE / 2 + 20):] &= ys[:, int(SIZE / 2 + 20):] < 0  # open on the right
    return ring


TARGETS = {"disc": disc_target, "ellipse": ellipse_target, "bar": bar_target, "c_shape": c_target}
THIN_BAR = lambda: bar_target(12)  # noqa: E731  deepest point 6 px: at the minimum radius, so most draws fall back


def depth_of(mask: np.ndarray) -> np.ndarray:
    return distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]


class _RecordingRng:
    """A generator that logs which draws are made, in order."""

    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)
        self.log: list[tuple] = []

    def uniform(self, low, high):
        self.log.append(("uniform", float(low), float(high)))
        return self._rng.uniform(low, high)

    def integers(self, high):
        self.log.append(("integers", int(high)))
        return self._rng.integers(high)


@contextlib.contextmanager
def recorded_circles():
    """Record the (center_x, center_y, radius) each annotator_circle call rasterizes."""
    calls: list[tuple[float, float, float]] = []
    real = negatives_module.rasterize_circle

    def spy(shape, cx, cy, radius):
        calls.append((cx, cy, radius))
        return real(shape, cx, cy, radius)

    with mock.patch.object(prompts_module, "rasterize_circle", spy):
        yield calls


class AnnotatorCircleTests(unittest.TestCase):
    CFG = CircleSeedJitter()

    def test_the_centre_depth_radius_and_fit_hold_over_many_draws(self) -> None:
        for name, make in TARGETS.items():
            target = make()
            depth = depth_of(target)
            deepest = depth.max()
            outside = distance_transform_edt(~target)  # distance of every pixel to the target
            returned = 0
            for seed in range(120):
                with recorded_circles() as calls:
                    disc = annotator_circle(target, np.random.default_rng(seed), self.CFG)
                if disc is None:
                    self.assertEqual(calls, [], name)
                    continue
                returned += 1
                (cx, cy, radius), = calls
                row, col = int(cy - 0.5), int(cx - 0.5)
                self.assertTrue(target[row, col], f"{name} seed {seed}: the centre is not on the target")
                fraction = depth[row, col] / deepest
                low, high = self.CFG.depth_range
                self.assertGreaterEqual(fraction, low - 0.05 - 1e-9, f"{name} seed {seed}")
                self.assertLessEqual(fraction, high + 0.05 + 1e-9, f"{name} seed {seed}")
                self.assertGreaterEqual(radius, self.CFG.min_radius_px, f"{name} seed {seed}")
                self.assertLessEqual(radius, self.CFG.scale_range[1] * deepest + 1e-9)
                self.assertLessEqual(radius, depth[row, col] + self.CFG.fit_grow_px + 1e-9)
                # The disc stays within fit_grow_px of the target, give or take a pixel of rasterization.
                self.assertLessEqual(float(outside[disc].max()), self.CFG.fit_grow_px + 1.0, f"{name} seed {seed}")
            self.assertGreater(returned, 20, f"{name}: almost every draw fell back, so nothing was tested")

    def test_a_thin_bar_mostly_falls_back_and_never_returns_a_circle_below_the_minimum(self) -> None:
        target = THIN_BAR()
        returned = 0
        for seed in range(200):
            with recorded_circles() as calls:
                disc = annotator_circle(target, np.random.default_rng(seed), self.CFG)
            if disc is not None:
                returned += 1
                self.assertGreaterEqual(calls[0][2], self.CFG.min_radius_px)
        self.assertLess(returned, 40)

    def test_an_empty_target_gives_none(self) -> None:
        self.assertIsNone(annotator_circle(np.zeros((SIZE, SIZE), dtype=bool), np.random.default_rng(0), self.CFG))

    def test_a_target_too_small_for_the_minimum_radius_gives_none(self) -> None:
        tiny = np.zeros((SIZE, SIZE), dtype=bool)
        tiny[70:73, 70:73] = True
        for seed in range(20):
            self.assertIsNone(annotator_circle(tiny, np.random.default_rng(seed), self.CFG))

    def test_a_smaller_minimum_lets_a_small_target_through(self) -> None:
        tiny = np.zeros((SIZE, SIZE), dtype=bool)
        tiny[70:80, 70:80] = True
        cfg = CircleSeedJitter(min_radius_px=1.0)
        self.assertTrue(any(annotator_circle(tiny, np.random.default_rng(s), cfg) is not None for s in range(20)))

    def test_the_draws_are_depth_then_centre_then_scale(self) -> None:
        rng = _RecordingRng(4)
        annotator_circle(disc_target(), rng, self.CFG)  # type: ignore[arg-type]
        kinds = [entry[0] for entry in rng.log]
        self.assertEqual(kinds, ["uniform", "integers", "uniform"])
        self.assertEqual(rng.log[0][1:], self.CFG.depth_range)
        self.assertEqual(rng.log[2][1:], self.CFG.scale_range)

    def test_the_same_generator_state_gives_the_same_disc(self) -> None:
        target = ellipse_target()
        for key in ([5, 0, 12, 0, 2], [5, 3, 12, 1, 2]):
            first = annotator_circle(target, np.random.default_rng(key), self.CFG)
            again = annotator_circle(target, np.random.default_rng(key), self.CFG)
            self.assertIsNotNone(first)
            self.assertTrue(np.array_equal(first, again))

    def test_a_different_variant_index_gives_an_independent_draw(self) -> None:
        target = disc_target()
        discs = [
            annotator_circle(target, np.random.default_rng([5, 0, 12, k, 2]), self.CFG) for k in range(8)
        ]
        distinct = {disc.tobytes() for disc in discs if disc is not None}
        self.assertGreater(len(distinct), 4, "variants of one sample should rarely share a circle")

    def test_the_target_is_not_modified(self) -> None:
        target = c_target()
        before = target.copy()
        annotator_circle(target, np.random.default_rng(1), self.CFG)
        self.assertTrue(np.array_equal(target, before))

    def test_a_target_touching_the_tile_edge_is_measured_against_the_edge(self) -> None:
        # A half disc against the top edge: its depth must not treat the edge as inside, so every centre is at least
        # one pixel from the edge row and the deepest point is the half-disc's radius, not the full disc's.
        target = disc_target(50)
        target[: SIZE // 2 - 0] = False
        target[SIZE // 2 :, :] |= False
        shifted = np.zeros_like(target)
        shifted[:50] = disc_target(50)[SIZE // 2 - 0 : SIZE // 2 + 50]  # push the lower half against the top edge
        with recorded_circles() as calls:
            annotator_circle(shifted, np.random.default_rng(2), self.CFG)
        for _cx, cy, radius in calls:
            self.assertLessEqual(radius, depth_of(shifted).max() + self.CFG.fit_grow_px + 1e-9)
            self.assertGreaterEqual(cy - radius, -self.CFG.fit_grow_px - 1.0)


class CircleSeedJitterConfigTests(unittest.TestCase):
    def test_the_defaults_are_the_specs(self) -> None:
        jitter = CircleSeedJitter()
        self.assertEqual(
            (jitter.depth_range, jitter.scale_range, jitter.fit_grow_px, jitter.min_radius_px),
            ((0.3, 1.0), (0.45, 1.0), 10.0, 6.0),
        )
        self.assertFalse(hasattr(jitter, "prob"), "the mix owns the probability (prob_circle_annotator)")
        self.assertEqual(PromptConfig().circle_seed_jitter, jitter)

    def test_the_shipped_config_carries_the_block_with_those_defaults(self) -> None:
        cfg = load_train_cfg()
        self.assertEqual(PromptConfig.from_cfg(cfg).circle_seed_jitter, CircleSeedJitter())
        self.assertNotIn("prob", cfg.prompt.circle_seed_jitter)
        self.assertEqual(float(cfg.prompt.prob_circle_annotator), 0.20)

    def test_a_config_block_is_read_and_a_missing_key_keeps_its_default(self) -> None:
        cfg = OmegaConf.create({"prompt": {"circle_seed_jitter": {"depth_range": [0.4, 0.9]}}})
        jitter = PromptConfig.from_cfg(cfg).circle_seed_jitter
        self.assertEqual(jitter.depth_range, (0.4, 0.9))
        self.assertEqual(jitter.scale_range, (0.45, 1.0))

    def test_a_config_without_the_block_loads_with_the_defaults(self) -> None:
        self.assertEqual(PromptConfig.from_cfg(OmegaConf.create({"prompt": {}})).circle_seed_jitter, CircleSeedJitter())

    def test_each_rule_is_enforced(self) -> None:
        bad = [
            {"depth_range": [0.0, 1.0]}, {"depth_range": [0.8, 0.5]}, {"depth_range": [0.3, 1.2]},
            {"scale_range": [0.0, 1.0]}, {"scale_range": [0.9, 0.5]},
            {"fit_grow_px": -1}, {"min_radius_px": 0.5},
        ]
        for values in bad:
            with self.assertRaises(ValueError, msg=str(values)):
                PromptConfig.from_cfg(OmegaConf.create({"prompt": {"circle_seed_jitter": values}}))
        # The bounds themselves are allowed.
        CircleSeedJitter(depth_range=(1.0, 1.0), scale_range=(0.5, 5.0), fit_grow_px=0, min_radius_px=1)

    def test_an_old_configs_retired_prob_key_is_ignored(self) -> None:
        # Configs saved before circle_annotator became a prompt type carry circle_seed_jitter.prob; it is ignored.
        old = {"prob": 0.7, "depth_range": [0.3, 1.0], "scale_range": [0.45, 1.0], "fit_grow_px": 10, "min_radius_px": 6}
        self.assertEqual(CircleSeedJitter.from_mapping(old), CircleSeedJitter())
        self.assertEqual(
            PromptConfig.from_cfg(OmegaConf.create({"prompt": {"circle_seed_jitter": old}})).circle_seed_jitter,
            CircleSeedJitter(),
        )

    def test_an_old_prompt_mix_is_refused_by_the_sum_check(self) -> None:
        # An old run's prompt section (prob_circle_seed 0.35, no circle peers) merged over today's defaults sums to
        # 1.25 and is refused. Nothing loads an old run's config.yaml into PromptConfig: evaluation starts from the
        # shipped config plus the model's encoding record, and export reads only the encoding keys.
        old = {"prob_points_only": 0.10, "prob_box_only": 0.05, "prob_box_points": 0.05, "prob_circle_seed": 0.35,
               "prob_adjacent_poly_seed": 0.15, "prob_subset_seed": 0.15, "prob_correction": 0.10,
               "prob_near_correct": 0.05, "circle_seed_jitter": {"prob": 0.7}}
        with self.assertRaisesRegex(ValueError, "sum to 1.0"):
            PromptConfig.from_cfg(OmegaConf.create({"prompt": old}))
        old.update(prob_circle_annotator=0.0, prob_circle_pull_through=0.0)
        self.assertEqual(PromptConfig.from_cfg(OmegaConf.create({"prompt": old})).circle_share(), 0.35)

    def test_a_misspelled_key_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "fit_grow"):
            PromptConfig.from_cfg(OmegaConf.create({"prompt": {"circle_seed_jitter": {"fit_grow": 5}}}))


class _FamilyFixture(unittest.TestCase):
    OLD_FAMILIES = ["centroid", "point", "circle_mask", "correction", "near_correct", "circle_mask_noneg"]

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        make_crops(self.root, "RC1", n=9)
        self.examples = index_volumes(volumes=["RC1"], root=self.root)
        self.tiles = build_tile_context(self.examples)
        # The synthetic targets are 16 px squares, too small for the default 6 px minimum radius.
        self.cfg = PromptConfig(circle_seed_jitter=CircleSeedJitter(min_radius_px=1.0))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def dataset(self, families, *, epoch=None, cfg=None) -> EMSegDataset:
        return EMSegDataset(
            self.examples, split="val", image_size=32, tiles=self.tiles, prompt_cfg=cfg or self.cfg, seed=3,
            core_band_px=2.0, val_families=families, epoch=epoch,
        )

    @staticmethod
    def by_family(item) -> dict:
        return {variant["val_family"]: variant for variant in item["variants"]}


class FamilyRegistrationTests(unittest.TestCase):
    def test_the_new_family_is_last_so_every_existing_family_keeps_its_number(self) -> None:
        self.assertEqual(VAL_PROMPTS[6], "circle_annotator")  # only the evaluation-only circle_oversize follows
        self.assertEqual(
            {name: VAL_FAMILY_CODES[name] for name in _FamilyFixture.OLD_FAMILIES},
            {"centroid": 0, "point": 1, "circle_mask": 2, "correction": 3, "near_correct": 4, "circle_mask_noneg": 5},
        )
        self.assertEqual(VAL_FAMILY_CODES["circle_annotator"], 6)

    def test_it_is_a_circle_seed_family_that_keeps_its_negatives(self) -> None:
        self.assertEqual(VAL_MIXED_TYPES["circle_annotator"], PromptType.CIRCLE_ANNOTATOR)
        self.assertIn("circle_annotator", CIRCLE_PROMPTS)
        self.assertNotIn("circle_annotator", NO_NEGATIVE_PROMPTS)

    def test_the_suite_defaults_and_the_split_include_it_and_it_is_the_primary(self) -> None:
        self.assertEqual(DEFAULT_VAL_PROMPTS[-2:], ("circle_mask_noneg", "circle_annotator"))
        self.assertIn("circle_annotator", NEG_SPLIT_PROMPTS)
        cfg = load_train_cfg()
        self.assertEqual(list(cfg.eval.val_prompts)[-2:], ["circle_mask_noneg", "circle_annotator"])
        self.assertEqual(str(cfg.eval.primary_val_prompt), "circle_annotator")
        self.assertEqual(tuple(cfg.eval.val_prompts), DEFAULT_VAL_PROMPTS)

    def test_the_one_prompt_evaluation_leaves_it_out_and_refuses_it(self) -> None:
        self.assertNotIn("circle_annotator", PER_PROMPT_DEFAULTS)
        self.assertEqual(
            set(DEFAULT_VAL_PROMPTS) - set(PER_PROMPT_DEFAULTS), set(FAMILIES_ONLY_PROMPTS) & set(DEFAULT_VAL_PROMPTS)
        )
        request = EvalRequest(
            finetuned=None, base_ckpt=Path("b.pt"), model_cfg="x", data_dir=Path("."), volumes=["RC1"],
            split_path=Path("s.json"), image_size=32, threshold=0.5, out_dir=Path("o"), grid_every=0, max_grids=0,
            num_workers=0, prompts=("circle_mask", "circle_annotator"),
        )
        with self.assertRaisesRegex(ValueError, "--goal"):
            evaluate(request)

    def test_the_one_prompt_dataset_path_refuses_it_instead_of_scoring_a_plain_circle(self) -> None:
        with self.assertRaisesRegex(ValueError, "val_families"):
            EMSegDataset([], split="val", val_prompt="circle_annotator")

    def test_the_split_is_logged_for_it(self) -> None:
        records = [
            {"volume": "RC1", "category": "MC", "iou": 0.9, "raw_iou": 0.9, "bf10": 0.8, "bf20": 0.9, "gt_depth": 40.0,
             "core_gt": 100, "core_hit": hit, "recall": hit / 100, "pred_area": 1000, "leak_px": 0, "gt_area": 1000,
             "spill_px": 0, "merge": False, "frag": False, "has_neg": has_neg}
            for hit, has_neg in ((90, True), (100, True), (98, False))
        ]

        class Writer:
            tags: dict = {}

            def __init__(self) -> None:
                self.tags = {}

            def add_scalar(self, tag, value, step) -> None:
                self.tags[tag] = value

        writer = Writer()
        text = log_results(writer, 0, {"circle_annotator": summarize_mode(records)}, "circle_annotator", root="val_full")
        self.assertEqual(writer.tags["val_full_neg/n/circle_annotator/with_neg"], 2)
        self.assertEqual(writer.tags["val_full_neg/n/circle_annotator/no_neg"], 1)
        self.assertIn("with_neg (n=2)", text)


class AnnotatorFamilyTests(_FamilyFixture):
    def test_existing_families_are_unchanged_by_running_alongside_the_new_one(self) -> None:
        before = self.dataset(self.OLD_FAMILIES)
        after = self.dataset(self.OLD_FAMILIES + ["circle_annotator"])
        for index in range(len(before)):
            old, new = self.by_family(before[index]), self.by_family(after[index])
            for family in self.OLD_FAMILIES:
                for key in ("point", "point_label", "boxes", "dense_mask", "prompt_fg", "prompt_input"):
                    a, b = old[family][key], new[family][key]
                    if a is None:
                        self.assertIsNone(b, (family, key))
                    else:
                        self.assertTrue(torch.equal(a, b), (index, family, key))

    def test_it_is_a_circle_seed_with_negatives_and_a_different_seed_from_circle_mask(self) -> None:
        dataset = self.dataset(["circle_mask", "circle_mask_noneg", "circle_annotator"])
        differs = with_negatives = 0
        for index in range(len(dataset)):
            item = dataset[index]
            by = self.by_family(item)
            annotator = by["circle_annotator"]
            self.assertEqual(annotator["prompt_type"], "circle_annotator")  # 1 px minimum: always drawn here
            self.assertTrue((annotator["point_label"][:1] == 1).all())
            if not torch.equal(annotator["prompt_input"], by["circle_mask"]["prompt_input"]):
                differs += 1
            if item["sample_meta"].has_negatives:
                with_negatives += 1
                self.assertLess(float(annotator["dense_mask"].min()), -1.0)  # negatives are kept
                self.assertFalse(torch.equal(annotator["dense_mask"], by["circle_mask_noneg"]["dense_mask"]))
        self.assertGreater(differs, 0, "the jitter never changed a seed")
        self.assertGreater(with_negatives, 0)

    def test_it_is_identical_across_two_epochs_and_two_loads(self) -> None:
        from sam2_segmentation_trainer.data.dataset import EpochCounter

        counter_a, counter_b = EpochCounter(), EpochCounter()
        counter_a.set(0)
        counter_b.set(7)
        first = self.dataset(["circle_annotator"], epoch=counter_a)
        second = self.dataset(["circle_annotator"], epoch=counter_b)  # a fresh load, in a later epoch
        for index in range(len(first)):
            a = self.by_family(first[index])["circle_annotator"]
            b = self.by_family(second[index])["circle_annotator"]
            for key in ("point", "dense_mask", "prompt_fg", "prompt_input"):
                self.assertTrue(torch.equal(a[key], b[key]), (index, key))

    def test_the_circle_does_not_depend_on_which_families_run_with_it(self) -> None:
        alone = self.dataset(["circle_annotator"])
        crowd = self.dataset(["point", "circle_mask", "circle_annotator", "correction"])
        for index in range(len(alone)):
            a = self.by_family(alone[index])["circle_annotator"]
            b = self.by_family(crowd[index])["circle_annotator"]
            self.assertTrue(torch.equal(a["prompt_input"], b["prompt_input"]), index)

    def test_it_ignores_the_training_mix(self) -> None:
        # A training share of 0 for circle_annotator does not turn the validation family off.
        jitter = CircleSeedJitter(min_radius_px=1.0)
        off = PromptConfig(circle_seed_jitter=jitter, prob_circle_annotator=0.0, prob_circle_seed=0.30)
        on = PromptConfig(circle_seed_jitter=jitter)
        a, b = self.dataset(["circle_annotator"], cfg=off), self.dataset(["circle_annotator"], cfg=on)
        for index in range(len(a)):
            self.assertTrue(
                torch.equal(
                    self.by_family(a[index])["circle_annotator"]["prompt_input"],
                    self.by_family(b[index])["circle_annotator"]["prompt_input"],
                )
            )

    def test_when_no_circle_can_be_drawn_the_inscribed_seed_is_kept(self) -> None:
        dataset = self.dataset(["circle_mask", "circle_annotator"])
        with mock.patch.object(dataset_module, "annotator_circle", lambda *args, **kwargs: None):
            for index in range(len(dataset)):
                by = self.by_family(dataset[index])
                # The seed is the same; the families draw their background points from their own generators.
                self.assertTrue(
                    torch.equal(by["circle_annotator"]["point"][:1], by["circle_mask"]["point"][:1]), index
                )
                for key in ("dense_mask", "prompt_input"):
                    self.assertTrue(torch.equal(by["circle_annotator"][key], by["circle_mask"][key]), (index, key))

    def test_the_default_minimum_radius_falls_back_on_these_small_targets(self) -> None:
        # With the spec's 6 px minimum the 16 px test squares often cannot hold a circle; those samples keep the
        # inscribed seed, so the family never fails a sample.
        dataset = self.dataset(["circle_mask", "circle_annotator"], cfg=PromptConfig())
        fell_back = 0
        for index in range(len(dataset)):
            by = self.by_family(dataset[index])
            # A circle that cannot be drawn is built, and reported, as the plain circle_seed.
            if by["circle_annotator"]["prompt_type"] == "circle_seed":
                fell_back += 1
                self.assertTrue(torch.equal(by["circle_annotator"]["prompt_input"], by["circle_mask"]["prompt_input"]))
            else:
                self.assertEqual(by["circle_annotator"]["prompt_type"], "circle_annotator")
        self.assertGreater(fell_back, 0)

    def test_the_suite_runs_it_in_the_subset_and_the_full_pass_but_selects_on_circle_mask(self) -> None:
        suite = ValidationSuite(
            self.examples, modes=["circle_mask", "circle_annotator"], primary="circle_mask", image_size=32,
            num_workers=0, tiles=self.tiles, prompt_cfg=self.cfg, seed=3, core_band_px=2.0, epoch_subset=0.5,
            batch_size=3,
        )
        self.assertIn("circle_annotator", suite.modes)
        self.assertIn("circle_annotator", suite.subset_modes)
        self.assertEqual(suite.primary, "circle_mask")
        with self.assertRaises(ValueError):
            ValidationSuite(
                self.examples, modes=["circle_mask", "circle_annotator"], primary="circle_annotator_x",
                image_size=32, num_workers=0,
            )


MIX_FINGERPRINT = "8da12e8ae7f330a646de6b5ae6c437828fa4351e2a1f528926db919ffec62e13"
"""SHA-256 of every tensor ``_mixed_item`` produced for the 9 synthetic samples (training split, identity transform,
default ``PromptConfig``, seed 3, epochs 0 and 3). It pins the whole training prompt pipeline.

History: first recorded (as PROB_ZERO_FINGERPRINT) from the code before the circle jitter existed (commit 74c303f) and
kept bit for bit while the jitter's prob was 0; re-recorded for the ``erode`` mask-to-correct kind and for the
0.35 / 0.15 circle_seed / subset_seed mix. Re-recorded Oct 18 2026 when the circle prompts became three peer prompt
types (circle_seed 0.10, circle_annotator 0.20, circle_pull_through 0.05, so ten prob_* entries), variant 0 stopped being
a forced circle seed (every variant is drawn from the mix without replacement), variants_per_sample went from 3 to 4,
negative-context dropout started applying to variant 0 too, and the per-variant jitter replacement
(``circle_seed_jitter.prob``) was removed. Every one of those changes the prompt-type plan, so the old value cannot be
reproduced by patching; the suite passed on the old value before these edits."""


def only(kind: PromptType, **extra) -> PromptConfig:
    """A mix with every share 0 but ``kind`` (1.0), one variant, and a 1 px circle minimum for the 16 px fixture."""
    shares = {name: 0.0 for name in PROB_FIELDS.values()}
    shares[PROB_FIELDS[kind]] = 1.0
    values = dict(shares, variants_per_sample=1, circle_seed_jitter=CircleSeedJitter(min_radius_px=1.0))
    values.update(extra)
    return PromptConfig(**values)


PROB_FIELDS = {
    PromptType.POINTS: "prob_points_only",
    PromptType.BOX: "prob_box_only",
    PromptType.BOX_POINTS: "prob_box_points",
    PromptType.CIRCLE_SEED: "prob_circle_seed",
    PromptType.CIRCLE_ANNOTATOR: "prob_circle_annotator",
    PromptType.CIRCLE_PULL_THROUGH: "prob_circle_pull_through",
    PromptType.ADJACENT_POLY_SEED: "prob_adjacent_poly_seed",
    PromptType.SUBSET_SEED: "prob_subset_seed",
    PromptType.CORRECTION: "prob_correction",
    PromptType.NEAR_CORRECT: "prob_near_correct",
}


class MixFixture(unittest.TestCase):
    """Nine synthetic 32 px tiles (16 px square targets), training split, no random augmentation."""

    SEED = 3

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        make_crops(self.root, "RC1", n=9)
        self.examples = index_volumes(volumes=["RC1"], root=self.root)
        self.tiles = build_tile_context(self.examples)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def dataset(self, cfg: PromptConfig | None = None, *, split: str = "train", **extra):
        counter = EpochCounter()
        dataset = EMSegDataset(
            self.examples, split=split, image_size=32, tiles=self.tiles, prompt_cfg=cfg or PromptConfig(),
            seed=self.SEED, core_band_px=2.0, epoch=counter, **extra,
        )
        dataset.transform = lambda image, masks: {"image": image, "masks": masks}  # no random augmentation
        dataset.counter = counter  # type: ignore[attr-defined]
        return dataset

    def variants(self, dataset, epochs=(0, 1, 2)):
        """(epoch, index, k, item, variant) of every variant."""
        for epoch in epochs:
            dataset.counter.set(epoch)
            for index in range(len(dataset)):
                item = dataset[index]
                for k, variant in enumerate(item["variants"]):
                    yield epoch, index, k, item, variant

    @staticmethod
    def fingerprint(dataset, epochs=(0, 3)) -> tuple[str, int]:
        digest, count = hashlib.sha256(), 0
        for epoch in epochs:
            dataset.counter.set(epoch)
            for index in range(len(dataset)):
                item = dataset[index]
                digest.update(item["mask"].numpy().tobytes())
                for variant in item["variants"]:
                    digest.update(variant["prompt_type"].encode())
                    for key in ("point", "point_label", "boxes", "dense_mask", "prompt_fg", "prompt_input"):
                        tensor = variant[key]
                        digest.update(b"None" if tensor is None else tensor.numpy().tobytes())
                        count += 1
        return digest.hexdigest(), count


class MixFingerprintTests(MixFixture):
    def test_the_default_training_pipeline_is_pinned(self) -> None:
        digest, count = self.fingerprint(self.dataset())
        dataset = self.dataset()
        self.assertEqual(count, 2 * len(dataset) * 4 * 6)  # four variants on every sample, six tensors each
        self.assertEqual(digest, MIX_FINGERPRINT)


class AnnotatorTrainingTests(MixFixture):
    """``circle_annotator`` is a prompt type of its own: drawn when the mix picks it, with its own generator."""

    def test_the_seed_is_exactly_the_helpers_draw_for_that_sample_and_variant(self) -> None:
        dataset = self.dataset(only(PromptType.CIRCLE_ANNOTATOR))
        checked = 0
        for epoch, index, k, item, variant in self.variants(dataset):
            self.assertEqual(variant["requested_prompt_type"], "circle_annotator")
            if variant["prompt_type"] != "circle_annotator":
                continue
            target = item["mask"].numpy().astype(bool)
            rng = np.random.default_rng([self.SEED, epoch, index, k, 2])
            expected = annotator_circle(target, rng, dataset.prompt_cfg.circle_seed_jitter)
            self.assertTrue(np.array_equal(variant["prompt_input"].numpy().astype(bool), expected))
            checked += 1
        self.assertGreater(checked, 20)

    def test_its_generator_is_used_only_by_circle_annotator_variants(self) -> None:
        keys: list = []
        real = np.random.default_rng

        def spy(seed=None):
            keys.append(list(seed) if isinstance(seed, (list, tuple)) else seed)
            return real(seed)

        dataset = self.dataset()
        dataset.counter.set(5)
        for index in range(len(dataset)):
            keys.clear()
            with mock.patch.object(np.random, "default_rng", spy):
                item = dataset[index]
            annotator_keys = [k for k in keys if isinstance(k, list) and len(k) == 5 and k[-1] == 2]
            wanted = [k for k, v in enumerate(item["variants"]) if v["requested_prompt_type"] == "circle_annotator"]
            self.assertEqual(sorted(key[3] for key in annotator_keys), wanted)
            self.assertTrue(all(key[:3] == [self.SEED, 5, index] for key in annotator_keys))

    def test_a_circle_that_cannot_be_drawn_falls_back_to_the_plain_one_and_is_reported_as_circle_seed(self) -> None:
        plain = self.dataset(only(PromptType.CIRCLE_SEED))
        strict = self.dataset(only(PromptType.CIRCLE_ANNOTATOR, circle_seed_jitter=CircleSeedJitter(min_radius_px=50)))
        for index in range(len(plain)):
            (a,), (b,) = plain[index]["variants"], strict[index]["variants"]
            self.assertEqual((b["requested_prompt_type"], b["prompt_type"]), ("circle_annotator", "circle_seed"))
            for key in ("point", "point_label", "dense_mask", "prompt_fg", "prompt_input"):
                self.assertTrue(torch.equal(a[key], b[key]), (index, key))

    def test_the_seed_still_drives_the_dense_prompt_and_the_containment_core(self) -> None:
        dataset = self.dataset(only(PromptType.CIRCLE_ANNOTATOR))
        overhanging = 0
        for _epoch, _index, _k, item, variant in self.variants(dataset):
            if variant["prompt_type"] != "circle_annotator":
                continue
            target = item["mask"].numpy().astype(bool)
            seed = variant["prompt_input"].numpy().astype(bool)
            trust = variant["prompt_fg"].numpy()
            self.assertGreater(float(variant["dense_mask"].max()), 0.0)
            self.assertTrue(((trust > 0) <= seed).all(), "trust must lie inside the seed")
            self.assertEqual(float(trust[~target].max()), 0.0)
            point = variant["point"][0].numpy()
            self.assertTrue(seed[int(point[1]), int(point[0])], "the click is the disc's centre, inside it")
            overhanging += int((seed & ~target).any())
        self.assertGreater(overhanging, 0, "no annotator seed overhung the test targets, so nothing was tested")

    def test_the_same_epoch_sample_and_variant_always_give_the_same_circle_and_epochs_differ(self) -> None:
        dataset = self.dataset(only(PromptType.CIRCLE_ANNOTATOR))
        dataset.counter.set(1)
        first = [dataset[i]["variants"][0]["prompt_input"] for i in range(len(dataset))]
        again = [dataset[i]["variants"][0]["prompt_input"] for i in range(len(dataset))]
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, again)))
        dataset.counter.set(2)
        later = [dataset[i]["variants"][0]["prompt_input"] for i in range(len(dataset))]
        self.assertFalse(all(torch.equal(a, b) for a, b in zip(first, later)), "a new epoch should draw new circles")

    def test_the_distance_transform_is_done_once_per_sample(self) -> None:
        dataset = self.dataset(PromptConfig(circle_seed_jitter=CircleSeedJitter(min_radius_px=1.0)))
        calls = 0
        real = dataset_module.target_depth

        def counting(target):
            nonlocal calls
            calls += 1
            return real(target)

        with mock.patch.object(dataset_module, "target_depth", counting):
            for index in range(len(dataset)):
                before = calls
                item = dataset[index]
                drawn = {v["requested_prompt_type"] for v in item["variants"]}
                expected = int(bool(drawn & {"circle_annotator", "circle_pull_through"}))
                self.assertEqual(calls - before, expected, f"sample {index}")

    def test_an_adjacent_poly_seed_that_falls_back_uses_the_plain_circle(self) -> None:
        dataset = self.dataset(only(PromptType.CIRCLE_SEED))
        with mock.patch.object(EMSegDataset, "_plan_types", lambda self, *a, **k: [PromptType.ADJACENT_POLY_SEED]):
            item = dataset[0]
        (variant,) = item["variants"]
        self.assertEqual((variant["requested_prompt_type"], variant["prompt_type"]), ("adjacent_poly_seed", "circle_seed"))
        (plain,) = dataset[0]["variants"]
        self.assertTrue(torch.equal(variant["prompt_input"], plain["prompt_input"]))

    def test_the_training_variant_carries_the_fit_cap_flag(self) -> None:
        dataset = self.dataset(only(PromptType.CIRCLE_ANNOTATOR))
        checked = 0
        for epoch, index, k, item, variant in self.variants(dataset, epochs=(0, 1, 2, 3)):
            if variant["prompt_type"] != "circle_annotator":
                self.assertFalse(variant["circle_fit_capped"])
                continue
            info: dict = {}
            rng = np.random.default_rng([self.SEED, epoch, index, k, 2])
            annotator_circle(item["mask"].numpy().astype(bool), rng, dataset.prompt_cfg.circle_seed_jitter, info=info)
            self.assertEqual(variant["circle_fit_capped"], info["capped"])
            self.assertFalse(info["capped"], "the 16 px fixture targets are too small for the fit cap to bind")
            checked += 1
        self.assertGreater(checked, 20)

    def test_validation_and_forced_prompts_do_not_read_the_training_mix(self) -> None:
        off = only(PromptType.CIRCLE_SEED)
        on = only(PromptType.CIRCLE_ANNOTATOR)
        for extra in ({"val_prompt": "circle_mask"}, {"val_prompt": "circle_mask_noneg"}):
            a, b = self.dataset(off, split="val", **extra), self.dataset(on, split="val", **extra)
            for index in range(len(a)):
                (x,), (y,) = a[index]["variants"], b[index]["variants"]
                self.assertEqual(y["prompt_type"], "circle_seed")
                self.assertTrue(torch.equal(x["prompt_input"], y["prompt_input"]))


class ContainmentWeightTests(unittest.TestCase):
    """The mask-containment weight counts only where the seed overlaps the ground-truth target."""

    SHAPE = (96, 96)

    @staticmethod
    def square_target() -> np.ndarray:
        target = np.zeros((96, 96), dtype=bool)
        target[30:70, 30:70] = True  # 40 x 40
        return target

    @staticmethod
    def disc(cx: float, cy: float, radius: float) -> np.ndarray:
        ys, xs = np.mgrid[:96, :96]
        return (xs + 0.5 - cx) ** 2 + (ys + 0.5 - cy) ** 2 <= radius**2

    def build(self, target: np.ndarray, seed: np.ndarray, **kwargs):
        return prompts_module.build_prompt(
            PromptType.CIRCLE_SEED, target_mask=target, circle_mask=seed, regions=[], cfg=PromptConfig(),
            rng=np.random.default_rng(7), **kwargs,
        )

    def test_a_seed_that_reaches_past_the_target_has_no_weight_outside_it_and_unchanged_weight_inside(self) -> None:
        target = self.square_target()
        seed = self.disc(50, 50, 30)  # radius 30 around the centre: reaches 10 px past the 40 px square on every side
        self.assertTrue((seed & ~target).any() and (seed & target).any(), "the premise: the seed overhangs the target")
        cfg = PromptConfig()

        prompt = self.build(target, seed)

        weight = prompt.foreground_weight
        self.assertIsNotNone(weight)
        self.assertEqual(float(weight[~target].max()), 0.0)
        expected = prompts_module.edge_confidence(seed, cfg.edge_tau_px, cfg.edge_tau_depth_frac)
        self.assertTrue(np.array_equal(weight[target], expected[target]), "inside the target the weight is unchanged")
        self.assertGreater(float(weight[target].max()), 0.5)

    def test_the_prompt_itself_is_not_changed_and_the_seed_disc_is_not_clipped(self) -> None:
        target = self.square_target()
        seed = self.disc(50, 50, 30)
        everywhere = np.ones_like(target)  # a target covering the tile cannot clip anything: the old behaviour

        clipped_weight = self.build(target, seed)
        unrestricted = self.build(everywhere, seed)

        self.assertTrue(np.array_equal(clipped_weight.input_mask, seed), "the seed is not clipped to the target")
        self.assertTrue(np.array_equal(clipped_weight.input_mask, unrestricted.input_mask))
        self.assertTrue(np.array_equal(clipped_weight.dense_logits, unrestricted.dense_logits))
        self.assertTrue(np.array_equal(clipped_weight.point_coords, unrestricted.point_coords))
        self.assertTrue(np.array_equal(clipped_weight.point_labels, unrestricted.point_labels))
        self.assertGreater(float(unrestricted.foreground_weight[~target].max()), 0.0, "the unrestricted weight had trust outside")

    def test_a_seed_inside_the_target_is_unaffected(self) -> None:
        target = self.square_target()
        seed = self.disc(50, 50, 15)
        cfg = PromptConfig()
        weight = self.build(target, seed).foreground_weight
        expected = prompts_module.edge_confidence(seed, cfg.edge_tau_px, cfg.edge_tau_depth_frac)
        self.assertTrue(np.array_equal(weight, expected))

    def test_a_seed_wholly_outside_the_target_has_no_containment_weight(self) -> None:
        self.assertIsNone(self.build(self.square_target(), self.disc(10, 10, 6)).foreground_weight)

    def test_the_loss_ignores_background_predicted_where_the_seed_overhangs(self) -> None:
        from sam2_segmentation_trainer.losses import mask_containment_loss_per_sample

        target = self.square_target()
        seed = self.disc(50, 50, 30)
        prediction = torch.where(torch.from_numpy(target), torch.tensor(5.0), torch.tensor(-5.0))[None]  # the truth

        restricted = torch.from_numpy(self.build(target, seed).foreground_weight.astype(np.float32))[None]
        unrestricted = torch.from_numpy(self.build(np.ones_like(target), seed).foreground_weight.astype(np.float32))[None]

        self.assertEqual(float(mask_containment_loss_per_sample(prediction, restricted)), 0.0)
        self.assertGreater(float(mask_containment_loss_per_sample(prediction, unrestricted)), 0.0)


class FitCapTests(unittest.TestCase):
    """``annotator_circle`` can say whether the fit cap d(p) + fit_grow_px, not s * D, set the radius."""

    def test_info_reports_the_draw_and_whether_the_cap_set_the_radius(self) -> None:
        cfg = CircleSeedJitter()
        seen = {True: 0, False: 0}
        for name, make in (("disc", disc_target), ("ellipse", ellipse_target), ("c_shape", c_target)):
            target = make()
            for seed in range(80):
                info: dict = {}
                disc = annotator_circle(target, np.random.default_rng(seed), cfg, info=info)
                if disc is None:
                    continue
                uncapped = info["scale"] * info["max_depth"]
                capped_radius = info["depth_at_centre"] + cfg.fit_grow_px
                self.assertAlmostEqual(info["radius"], min(uncapped, capped_radius), places=9, msg=f"{name} {seed}")
                self.assertEqual(info["capped"], capped_radius < uncapped, f"{name} {seed}")
                seen[info["capped"]] += 1
        self.assertGreater(seen[True], 0)
        self.assertGreater(seen[False], 0)

    def test_info_does_not_change_what_is_drawn(self) -> None:
        target, cfg = ellipse_target(), CircleSeedJitter()
        for seed in range(20):
            plain = annotator_circle(target, np.random.default_rng(seed), cfg)
            informed = annotator_circle(target, np.random.default_rng(seed), cfg, info={})
            self.assertTrue((plain is None and informed is None) or np.array_equal(plain, informed))
