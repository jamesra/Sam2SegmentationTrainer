from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from sam2_segmentation_trainer.data.dataset import EMSegDataset
from sam2_segmentation_trainer.data.fulltile import cap_full_tiles, is_full_tile, keep_hash
from sam2_segmentation_trainer.data.manifest import index_volumes
from sam2_segmentation_trainer.data.prompts import (
    FULL_TILE_PROMPTS,
    PromptConfig,
    PromptType,
    choose_prompt_type,
)
from sam2_segmentation_trainer.data.tiles import build_tile_context

try:
    from tests.helpers import make_crops
except ImportError:
    from helpers import make_crops


class AreaFractionTests(unittest.TestCase):
    def test_index_records_the_covered_fraction(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_crops(root, "RC1", n=3)
            examples = index_volumes(volumes=["RC1"], root=root)
            fractions = {round(e.area_frac, 3) for e in examples if e.area_frac is not None}
            self.assertIn(0.25, fractions)  # the 16x16 mask on a 32x32 tile
            self.assertTrue(all(e.area_frac is not None for e in examples))

    def test_unknown_coverage_is_not_a_full_tile(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_crops(root, "RC1", n=2)
            example = replace(index_volumes(volumes=["RC1"], root=root)[0], area_frac=None)
            self.assertFalse(is_full_tile(example, 0.99))
            self.assertFalse(is_full_tile(replace(example, area_frac=1.0), None))
            self.assertTrue(is_full_tile(replace(example, area_frac=0.995), 0.99))


class CapTests(unittest.TestCase):
    def _examples(self, root: Path) -> list:
        make_crops(root, "RC1", n=12)
        base = index_volumes(volumes=["RC1"], root=root)
        # Make every example a full tile with a distinct location, so the cap has something to choose from.
        return [replace(e, area_frac=1.0, split_key=f"RC1:{i}") for i, e in enumerate(base)]

    def test_keeps_about_the_fraction_and_is_repeatable(self) -> None:
        with TemporaryDirectory() as tmp:
            full = self._examples(Path(tmp))
            first = cap_full_tiles(full, threshold=0.99, keep_fraction=0.5, seed=7)
            second = cap_full_tiles(full, threshold=0.99, keep_fraction=0.5, seed=7)
            self.assertEqual([e.split_key for e in first], [e.split_key for e in second])
            self.assertLess(len(first), len(full))
            self.assertTrue(all(keep_hash(e.split_key, 7) < 0.5 for e in first))

    def test_leaves_other_tiles_alone(self) -> None:
        with TemporaryDirectory() as tmp:
            full = self._examples(Path(tmp))
            mixed = [replace(e, area_frac=0.5) for e in full[:5]] + full[5:]
            kept = cap_full_tiles(mixed, threshold=0.99, keep_fraction=0.0, seed=1)
            self.assertEqual(len(kept), 5)
            self.assertEqual(len(cap_full_tiles(mixed, threshold=None, keep_fraction=0.0)), len(mixed))
            self.assertEqual(len(cap_full_tiles(mixed, threshold=0.99, keep_fraction=1.0)), len(mixed))


class PromptRestrictionTests(unittest.TestCase):
    def test_allowed_types_limit_the_draw(self) -> None:
        cfg = PromptConfig()
        rng = np.random.default_rng(0)
        drawn = {
            choose_prompt_type(cfg, rng, allowed=FULL_TILE_PROMPTS)
            for _ in range(200)
        }
        self.assertTrue(drawn <= FULL_TILE_PROMPTS)
        self.assertEqual(drawn, set(FULL_TILE_PROMPTS))

    def test_falls_back_to_points_when_nothing_allowed_has_weight(self) -> None:
        cfg = PromptConfig(
            prob_points_only=0.0, prob_box_only=0.5, prob_box_points=0.0, prob_circle_seed=0.0,
            prob_circle_annotator=0.0, prob_circle_pull_through=0.0, prob_adjacent_poly_seed=0.0, prob_subset_seed=0.5, prob_correction=0.0, prob_near_correct=0.0,
        )
        rng = np.random.default_rng(0)
        kind = choose_prompt_type(cfg, rng, allowed=FULL_TILE_PROMPTS)
        self.assertIs(kind, PromptType.POINTS)

    def test_dataset_gives_full_tiles_only_points_or_circles(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_crops(root, "RC1", n=6)
            examples = [replace(e, area_frac=1.0) for e in index_volumes(volumes=["RC1"], root=root)]
            context = build_tile_context(examples, load_circles=False)
            cfg = PromptConfig(
                prob_points_only=0.0, prob_box_only=0.0, prob_box_points=0.0, prob_circle_seed=0.0,
                prob_circle_annotator=0.0, prob_circle_pull_through=0.0,
                prob_adjacent_poly_seed=0.0, prob_subset_seed=1.0, prob_correction=0.0, prob_near_correct=0.0,
            )
            dataset = EMSegDataset(
                examples, split="train", image_size=32, tiles=context, prompt_cfg=cfg,
                seed=1, full_tile_threshold=0.99,
            )
            item = dataset[0]
            # A fully masked tile cannot draw a subset seed, and nothing it may draw has a share: the shortfall rule
            # gives one points variant and stops.
            self.assertEqual([v["prompt_type"] for v in item["variants"]], [PromptType.POINTS.value])
            self.assertTrue(item["sample_meta"].full_tile)

    def test_full_tiles_draw_no_pull_through_and_have_three_variants(self) -> None:
        import torch

        from sam2_segmentation_trainer.data.dataset import EpochCounter, collate_em
        from sam2_segmentation_trainer.model import combined_loss

        self.assertNotIn(PromptType.CIRCLE_PULL_THROUGH, FULL_TILE_PROMPTS)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_crops(root, "RC1", n=6)
            normal = index_volumes(volumes=["RC1"], root=root)
            full = [replace(e, area_frac=1.0) for e in normal]
            counter = EpochCounter()
            dataset = EMSegDataset(
                full, split="train", image_size=32, tiles=build_tile_context(full, load_circles=False),
                prompt_cfg=PromptConfig(), seed=1, full_tile_threshold=0.99, epoch=counter,
            )
            for epoch in range(3):
                counter.set(epoch)
                for index in range(len(dataset)):
                    variants = dataset[index]["variants"]
                    asked = [v["requested_prompt_type"] for v in variants]
                    # Points, circle_seed and circle_annotator once each (the default mix has 4 variants): no
                    # pull-through that would decode the circle_seed prompt a second time.
                    self.assertEqual(sorted(asked), ["circle_annotator", "circle_seed", "points"])
            # A 3-variant full tile batches with a 4-variant normal sample; the loss averages each sample's variants.
            mixed = EMSegDataset(
                normal, split="train", image_size=32, tiles=build_tile_context(normal, load_circles=False),
                prompt_cfg=PromptConfig(), seed=1, full_tile_threshold=0.99,
            )
            batch = collate_em([dataset[0], mixed[0]])
            index = batch["variant_index"]
            self.assertEqual(torch.bincount(index).tolist(), [3, 4])
            pred = torch.randn(index.numel(), 32, 32)
            target = batch["mask"][index].float()
            loss, stats = combined_loss(
                pred, target, torch.rand(index.numel()), focal_weight=20.0, dice_weight=1.0, iou_weight=1.0,
                variant_index=index, num_samples=2,
            )
            per_variant = stats["sample_losses"]
            expected = (per_variant[index == 0].mean() + per_variant[index == 1].mean()) / 2
            self.assertTrue(torch.isfinite(loss))
            self.assertAlmostEqual(float(loss), float(expected), places=5)

    def test_normal_tiles_keep_the_configured_mix(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_crops(root, "RC1", n=6)
            examples = index_volumes(volumes=["RC1"], root=root)
            context = build_tile_context(examples, load_circles=False)
            cfg = PromptConfig(
                prob_points_only=0.0, prob_box_only=0.0, prob_box_points=0.0, prob_circle_seed=0.0,
                prob_circle_annotator=0.0, prob_circle_pull_through=0.0,
                prob_adjacent_poly_seed=0.0, prob_subset_seed=1.0, prob_correction=0.0, prob_near_correct=0.0,
            )
            dataset = EMSegDataset(
                examples, split="train", image_size=32, tiles=context, prompt_cfg=cfg,
                seed=1, full_tile_threshold=0.99,
            )
            item = dataset[0]
            # No family is forced: the only family with a share comes first, then the shortfall points variant.
            self.assertEqual(
                [v["prompt_type"] for v in item["variants"]],
                [PromptType.SUBSET_SEED.value, PromptType.POINTS.value],
            )
            self.assertFalse(item["sample_meta"].full_tile)


if __name__ == "__main__":
    unittest.main()
