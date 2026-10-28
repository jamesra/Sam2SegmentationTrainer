"""Pull-through circles (``prompt.circle_pull_through``, docs/circle_annotator_spec.md "Pull-through circles"): an
annotator circle drawn across the cell edge with most of it inside. The helper, the config block, and the
``circle_pull_through`` prompt type in ``_mixed_item``."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
from omegaconf import OmegaConf

from sam2_segmentation_trainer.data import dataset as dataset_module
from sam2_segmentation_trainer.data import prompts as prompts_module
from sam2_segmentation_trainer.data.dataset import collate_em
from sam2_segmentation_trainer.data.prompts import (
    PULL_THROUGH_CENTRE_DEPTH,
    SEED_PROMPTS,
    CirclePullThrough,
    PromptConfig,
    PromptType,
    pull_through_circle,
)
from sam2_segmentation_trainer.train import load_train_cfg

try:
    from tests.test_circle_annotator import (
        SIZE,
        MixFixture,
        only,
        _RecordingRng,
        bar_target,
        c_target,
        depth_of,
        disc_target,
        ellipse_target,
        recorded_circles,
    )
except ImportError:
    from test_circle_annotator import (  # type: ignore[no-redef]
        SIZE,
        MixFixture,
        only,
        _RecordingRng,
        bar_target,
        c_target,
        depth_of,
        disc_target,
        ellipse_target,
        recorded_circles,
    )

TARGETS = {
    "disc": disc_target,
    "ellipse": ellipse_target,
    "c_shape": c_target,
    "thin_bar": lambda: bar_target(12),
}


def fractions(disc: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """(inside, outside) fractions of the disc's area."""
    area = float(disc.sum())
    inside = float((disc & target).sum()) / area
    return inside, 1.0 - inside


class PullThroughCircleTests(unittest.TestCase):
    CFG = CirclePullThrough()

    def test_every_disc_crosses_the_edge_with_most_of_it_inside(self) -> None:
        for name, make in TARGETS.items():
            target = make()
            depth = depth_of(target)
            returned = 0
            for seed in range(80):
                info: dict = {}
                with recorded_circles() as calls:
                    disc = pull_through_circle(target, np.random.default_rng(seed), self.CFG, info=info)
                if disc is None:
                    self.assertEqual(info, {}, name)
                    continue
                returned += 1
                inside, outside = fractions(disc, target)
                self.assertGreaterEqual(inside, self.CFG.overlap_min, f"{name} {seed}")
                self.assertGreaterEqual(outside, self.CFG.outside_min, f"{name} {seed}")
                self.assertAlmostEqual(info["overlap"], inside, places=9)
                # Not clipped: it is the full raster of the recorded circle, reaching past the target.
                ((cx, cy, radius),) = calls
                self.assertTrue(np.array_equal(disc, prompts_module.rasterize_circle(target.shape, cx, cy, radius)))
                self.assertTrue((disc & ~target).any())
                row, col = int(cy - 0.5), int(cx - 0.5)
                self.assertEqual((row, col, radius), (info["row"], info["col"], info["radius"]))
                self.assertTrue(target[row, col], f"{name} {seed}: the centre is on the target")
                low, high = PULL_THROUGH_CENTRE_DEPTH
                self.assertGreaterEqual(depth[row, col], low * radius - 1e-9)
                self.assertLessEqual(depth[row, col], high * radius + 1e-9)
                deepest = float(depth.max())
                self.assertGreaterEqual(radius, max(self.CFG.radius_range[0] * deepest, self.CFG.min_radius_px) - 1e-9)
                self.assertLessEqual(radius, max(self.CFG.radius_range[1] * deepest, self.CFG.min_radius_px) + 1e-9)
            self.assertGreater(returned, 40, f"{name}: almost every draw gave None, so nothing was tested")

    def test_the_same_generator_state_gives_the_same_disc(self) -> None:
        for make in TARGETS.values():
            target = make()
            for key in ([5, 0, 12, 0, 3], [5, 3, 12, 1, 3]):
                first = pull_through_circle(target, np.random.default_rng(key), self.CFG)
                again = pull_through_circle(target, np.random.default_rng(key), self.CFG)
                self.assertTrue((first is None and again is None) or np.array_equal(first, again))

    def test_the_draws_are_radius_then_centre_for_each_try(self) -> None:
        rng = _RecordingRng(4)
        info: dict = {}
        self.assertIsNotNone(pull_through_circle(disc_target(), rng, self.CFG, info=info))  # type: ignore[arg-type]
        kinds = [entry[0] for entry in rng.log]
        self.assertEqual(kinds, ["uniform", "integers"] * info["tries"])
        self.assertEqual(rng.log[0][1:], self.CFG.radius_range)

    def test_info_does_not_change_what_is_drawn_and_the_target_is_not_modified(self) -> None:
        target = c_target()
        before = target.copy()
        for seed in range(10):
            plain = pull_through_circle(target, np.random.default_rng(seed), self.CFG)
            informed = pull_through_circle(
                target, np.random.default_rng(seed), self.CFG, depth=prompts_module.target_depth(target), info={}
            )
            self.assertTrue((plain is None and informed is None) or np.array_equal(plain, informed))
        self.assertTrue(np.array_equal(target, before))

    def test_an_empty_target_gives_none(self) -> None:
        self.assertIsNone(pull_through_circle(np.zeros((SIZE, SIZE), bool), np.random.default_rng(0), self.CFG))

    def test_none_when_no_disc_can_cross_the_edge(self) -> None:
        # A target filling the tile has no edge inside the tile to cross: every disc is all inside.
        full = np.ones((SIZE, SIZE), dtype=bool)
        for seed in range(5):
            self.assertIsNone(pull_through_circle(full, np.random.default_rng(seed), self.CFG))
        # A target smaller than the minimum disc: no centre has the depth band, or the disc is mostly outside.
        tiny = np.zeros((SIZE, SIZE), dtype=bool)
        tiny[70:73, 70:73] = True
        for seed in range(5):
            self.assertIsNone(pull_through_circle(tiny, np.random.default_rng(seed), self.CFG))

    def test_the_tries_are_bounded(self) -> None:
        rng = _RecordingRng(0)
        full = np.ones((SIZE, SIZE), dtype=bool)
        pull_through_circle(full, rng, CirclePullThrough(max_tries=3))  # type: ignore[arg-type]
        self.assertEqual(sum(entry[0] == "uniform" for entry in rng.log), 3)


class CirclePullThroughConfigTests(unittest.TestCase):
    def test_the_defaults(self) -> None:
        pull = CirclePullThrough()
        self.assertEqual(
            (pull.overlap_min, pull.outside_min, pull.radius_range, pull.min_radius_px, pull.max_tries),
            (0.60, 0.05, (0.4, 0.9), 6.0, 20),
        )
        self.assertEqual(PromptConfig().circle_pull_through, pull)
        self.assertEqual(PromptConfig().prob_circle_pull_through, 0.05)

    def test_the_shipped_config_carries_the_block_with_those_defaults(self) -> None:
        cfg = load_train_cfg()
        self.assertEqual(PromptConfig.from_cfg(cfg).circle_pull_through, CirclePullThrough())
        self.assertNotIn("prob", cfg.prompt.circle_pull_through)
        self.assertEqual(float(cfg.prompt.prob_circle_pull_through), 0.05)

    def test_a_block_is_read_and_a_missing_key_keeps_its_default(self) -> None:
        cfg = OmegaConf.create({"prompt": {"circle_pull_through": {"overlap_min": 0.7, "radius_range": [0.5, 0.8]}}})
        pull = PromptConfig.from_cfg(cfg).circle_pull_through
        self.assertEqual((pull.overlap_min, pull.radius_range, pull.outside_min), (0.7, (0.5, 0.8), 0.05))
        self.assertEqual(PromptConfig.from_cfg(OmegaConf.create({"prompt": {}})).circle_pull_through, CirclePullThrough())

    def test_it_is_a_seed_prompt_type_in_the_mix(self) -> None:
        self.assertEqual(PromptType.CIRCLE_PULL_THROUGH.value, "circle_pull_through")
        self.assertIn(PromptType.CIRCLE_PULL_THROUGH, PromptConfig().probabilities())
        self.assertIn(PromptType.CIRCLE_PULL_THROUGH, SEED_PROMPTS)

    def test_each_rule_is_enforced(self) -> None:
        bad = [
            {"overlap_min": 0.0}, {"overlap_min": 1.1},
            {"outside_min": -0.1}, {"outside_min": 1.0},
            {"overlap_min": 0.97, "outside_min": 0.05},
            {"radius_range": [0.0, 0.9]}, {"radius_range": [0.9, 0.4]},
            {"min_radius_px": 0.5}, {"max_tries": 0},
            {"prob": 0.1},  # the mix owns the probability: prob_circle_pull_through
        ]
        for values in bad:
            with self.assertRaises(ValueError, msg=str(values)):
                PromptConfig.from_cfg(OmegaConf.create({"prompt": {"circle_pull_through": values}}))
        CirclePullThrough(overlap_min=0.95, outside_min=0.05, radius_range=(2.0, 2.0), min_radius_px=1, max_tries=1)
        CirclePullThrough(overlap_min=1.0, outside_min=0.0)

    def test_a_misspelled_key_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "overlap"):
            PromptConfig.from_cfg(OmegaConf.create({"prompt": {"circle_pull_through": {"overlap": 0.5}}}))


class PullThroughTrainingTests(MixFixture):
    """The ``circle_pull_through`` prompt type in ``_mixed_item``."""

    def pulled(self, dataset, epochs=(0, 1, 2)):
        for epoch, index, k, item, variant in self.variants(dataset, epochs):
            if variant["prompt_type"] == "circle_pull_through":
                yield epoch, index, k, item, variant

    def test_the_seed_is_exactly_the_helpers_draw_and_is_not_clipped(self) -> None:
        dataset = self.dataset(only(PromptType.CIRCLE_PULL_THROUGH))
        checked = 0
        for epoch, index, k, item, variant in self.pulled(dataset):
            target = item["mask"].numpy().astype(bool)
            info: dict = {}
            rng = np.random.default_rng([self.SEED, epoch, index, k, 3])
            expected = pull_through_circle(target, rng, dataset.prompt_cfg.circle_pull_through, info=info)
            seed = variant["prompt_input"].numpy().astype(bool)
            self.assertTrue(np.array_equal(seed, expected))
            self.assertTrue((seed & ~target).any(), "the prompt reaches past the target: it is not clipped")
            self.assertAlmostEqual(variant["circle_overlap"], info["overlap"], places=9)
            self.assertGreaterEqual(variant["circle_overlap"], 0.60)
            checked += 1
        self.assertGreater(checked, 20)

    def test_the_containment_weight_is_zero_outside_the_target(self) -> None:
        dataset = self.dataset(only(PromptType.CIRCLE_PULL_THROUGH))
        checked = 0
        for _epoch, _index, _k, item, variant in self.pulled(dataset):
            target = item["mask"].numpy().astype(bool)
            seed = variant["prompt_input"].numpy().astype(bool)
            trust = variant["prompt_fg"].numpy()
            self.assertEqual(float(trust[~target].max()), 0.0)
            self.assertTrue(((trust > 0) <= (seed & target)).all())
            self.assertGreater(float(trust.max()), 0.0)
            # The dense prompt is built from the whole disc, the part outside the target included.
            self.assertGreater(float(variant["dense_mask"].max()), 0.0)
            point = variant["point"][0].numpy()
            self.assertTrue(target[int(point[1]), int(point[0])], "the click, the disc's centre, is on the target")
            checked += 1
        self.assertGreater(checked, 20)

    def test_a_disc_that_cannot_be_drawn_falls_back_to_the_plain_circle(self) -> None:
        dataset = self.dataset(only(PromptType.CIRCLE_PULL_THROUGH))
        with mock.patch.object(dataset_module, "pull_through_circle", lambda *a, **k: None):
            item = dataset[0]
        (variant,) = item["variants"]
        self.assertEqual((variant["requested_prompt_type"], variant["prompt_type"]), ("circle_pull_through", "circle_seed"))
        self.assertIsNone(variant["circle_overlap"])
        (plain,) = self.dataset(only(PromptType.CIRCLE_SEED))[0]["variants"]
        self.assertTrue(np.array_equal(variant["prompt_input"].numpy(), plain["prompt_input"].numpy()))

    def test_its_generator_is_used_only_by_circle_pull_through_variants(self) -> None:
        keys: list = []
        real = np.random.default_rng

        def spy(seed=None):
            keys.append(list(seed) if isinstance(seed, (list, tuple)) else seed)
            return real(seed)

        dataset = self.dataset()
        for index in range(len(dataset)):
            keys.clear()
            with mock.patch.object(np.random, "default_rng", spy):
                item = dataset[index]
            pull_keys = [k for k in keys if isinstance(k, list) and len(k) == 5 and k[-1] == 3]
            wanted = [k for k, v in enumerate(item["variants"]) if v["requested_prompt_type"] == "circle_pull_through"]
            self.assertEqual(sorted(key[3] for key in pull_keys), wanted)

    def test_the_batch_carries_requested_and_built_types(self) -> None:
        dataset = self.dataset()
        batch = collate_em([dataset[i] for i in range(len(dataset))])
        self.assertEqual(len(batch["requested_prompt_type"]), len(batch["prompt_type"]))
        for asked, built in zip(batch["requested_prompt_type"], batch["prompt_type"]):
            self.assertTrue(asked == built or built == "circle_seed", (asked, built))

    def test_validation_has_no_pull_through(self) -> None:
        dataset = self.dataset(
            only(PromptType.CIRCLE_PULL_THROUGH), split="val", val_families=["circle_mask", "circle_annotator"]
        )
        for index in range(len(dataset)):
            for variant in dataset[index]["variants"]:
                self.assertNotEqual(variant["prompt_type"], "circle_pull_through")


if __name__ == "__main__":
    unittest.main()
