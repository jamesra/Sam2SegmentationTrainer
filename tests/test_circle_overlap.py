from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from annotation_crops.negatives import (
    KIND_CIRCLE,
    KIND_POLYGON,
    NegativeRegion,
    circles_over_target,
    drop_circles_over_target,
)
from sam2_segmentation_trainer.data.dataset import EMSegDataset
from sam2_segmentation_trainer.data.manifest import index_volumes
from sam2_segmentation_trainer.data.prompts import PromptConfig
from sam2_segmentation_trainer.data.tiles import build_tile_context
from sam2_segmentation_trainer.validate import circle_overlap_stats

try:
    from tests.helpers import make_crops
except ImportError:
    from helpers import make_crops


def _region(kind: str, rows: slice, cols: slice, location_id: int) -> NegativeRegion:
    mask = np.zeros((32, 32), dtype=bool)
    mask[rows, cols] = True
    return NegativeRegion(location_id, kind, mask)


class DropOverlappingCirclesTests(unittest.TestCase):
    def test_only_circles_that_touch_the_target_are_dropped(self) -> None:
        target = np.zeros((32, 32), dtype=bool)
        target[8:24, 8:24] = True
        touching = _region(KIND_CIRCLE, slice(10, 14), slice(10, 14), 1)
        clear = _region(KIND_CIRCLE, slice(0, 4), slice(0, 4), 2)
        polygon = _region(KIND_POLYGON, slice(10, 14), slice(10, 14), 3)
        regions = [touching, clear, polygon]
        self.assertEqual([r.location_id for r in circles_over_target(regions, target)], [1])
        self.assertEqual([r.location_id for r in drop_circles_over_target(regions, target)], [2, 3])


def _write_circles(crops: Path, image_key: str, location_id: int, cx: float, cy: float, radius: float) -> None:
    entry = {
        "location_id": location_id, "center_x": cx, "center_y": cy, "radius": radius,
        "structure_label": "", "has_polygon_mask": False, "source": "annotated",
    }
    (crops / "circle_annotations.json").write_text(
        json.dumps({"version": 1, "diameter_scale": 0.95, "grid_cells": {image_key: [entry]}}),
        encoding="utf-8",
    )


class OverlapMeasurementTests(unittest.TestCase):
    def test_overlap_report_counts_a_circle_on_the_target(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            crops = make_crops(root, "RC1", n=4)
            examples = index_volumes(volumes=["RC1"], root=root)
            first = examples[0]
            _write_circles(crops, first.image_key, 99999, 16.0, 16.0, 4.0)
            report = circle_overlap_stats(examples, sample=50)
            self.assertGreaterEqual(report["examples_with_circle_negatives"], 1)
            self.assertGreaterEqual(report["examples_where_a_circle_overlaps_the_target"], 1)
            self.assertGreater(report["share_of_examples"], 0.0)
            self.assertIn("RC1", report["by_volume"])

    def test_no_circle_file_reports_nothing(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_crops(root, "RC1", n=3)
            examples = index_volumes(volumes=["RC1"], root=root)
            report = circle_overlap_stats(examples, sample=10)
            self.assertEqual(report["examples_sampled"], 0)
            self.assertEqual(report["share_of_examples"], 0.0)


class CircleStatsTests(unittest.TestCase):
    def test_each_volumes_circle_file_is_parsed_once_not_once_per_example(self) -> None:
        from unittest import mock

        from sam2_segmentation_trainer import validate

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            crops = make_crops(root, "RC1", n=6)
            examples = index_volumes(volumes=["RC1"], root=root)
            _write_circles(crops, examples[0].image_key, 99999, 16.0, 16.0, 4.0)
            with mock.patch.object(validate, "load_circle_scale", wraps=validate.load_circle_scale) as spy:
                report = validate._circle_stats(examples)
            self.assertEqual(spy.call_count, 1)
            self.assertEqual(report["circle_diameter_scale_by_volume"], {"RC1": 0.95})


class DatasetOptionTests(unittest.TestCase):
    def test_option_removes_the_overlapping_circle_from_the_prompt(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            crops = make_crops(root, "RC1", n=3)
            examples = index_volumes(volumes=["RC1"], root=root)
            first = examples[0]
            _write_circles(crops, first.image_key, 99999, 16.0, 16.0, 4.0)
            context = build_tile_context(examples)
            def dense_for(drop: bool):
                cfg = PromptConfig(variants_per_sample=1, drop_overlapping_circles=drop)
                ds = EMSegDataset([first], split="val", image_size=32, tiles=context, prompt_cfg=cfg, seed=1)
                return ds[0]["variants"][0]["dense_mask"]

            kept = dense_for(False)
            dropped = dense_for(True)
            # The circle sits inside the target, so with it kept the dense prompt carries its negative.
            self.assertFalse(np.array_equal(kept.numpy(), dropped.numpy()))


if __name__ == "__main__":
    unittest.main()
