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
            self.assertEqual(centroid["prompt_type"], "centroid")
            self.assertEqual(point["prompt_type"], "point")
            self.assertEqual(circle["prompt_type"], PromptType.CIRCLE_MASK.value)
            self.assertIsNone(centroid["dense_mask"])
            self.assertIsNotNone(circle["dense_mask"])
            self.assertEqual(tuple(circle["dense_mask"].shape), (256, 256))

    def test_circle_mask_is_reproducible(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_crops(root, "RC1", n=4)
            first = self._dataset(root, "circle_mask")[1]
            second = self._dataset(root, "circle_mask")[1]
            self.assertTrue(torch.equal(first["dense_mask"], second["dense_mask"]))
            self.assertTrue(torch.equal(first["point"], second["point"]))

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
            self.assertEqual(dataset[0]["prompt_type"], "circle_mask")


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
            self.assertEqual(suite.dataset.val_prompt, "circle_mask")
            self.assertEqual(len(suite), len(examples))
            suite.replace_examples(examples[:2])
            self.assertEqual({len(d) for d in suite.datasets.values()}, {2})


class ScoringTests(unittest.TestCase):
    def test_full_tiles_are_kept_out_of_the_headline(self) -> None:
        records = [
            {"volume": "RC1", "category": "MC", "iou": 0.5, "bf": 0.4, "full_tile": False},
            {"volume": "RC1", "category": "MC", "iou": 0.7, "bf": 0.6, "full_tile": False},
            {"volume": "RC2", "category": "HC", "iou": 1.0, "bf": 1.0, "full_tile": True},
        ]
        result = summarize_mode(records)
        self.assertAlmostEqual(result.iou, 0.6)
        self.assertEqual(result.n, 2)
        self.assertAlmostEqual(result.fulltile_iou, 1.0)
        self.assertEqual(result.by_volume, {"RC1": 0.6})
        self.assertEqual(result.by_category, {"MC": 0.6})

    def test_only_full_tiles_falls_back_to_them(self) -> None:
        result = summarize_mode([{"volume": "A", "category": "", "iou": 1.0, "bf": 1.0, "full_tile": True}])
        self.assertEqual(result.n, 1)
        self.assertEqual(summarize_mode([]).n, 0)

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
            "circle_mask": ModeResult(iou=0.8, bf=0.7, n=5, by_volume={"RC1": 0.8}, by_category={"MC": 0.8}),
            "point": ModeResult(iou=0.75, bf=0.6, n=5, fulltile_iou=0.99, fulltile_n=2),
        }
        text = log_results(writer, 3, results, "circle_mask")
        self.assertIn("val/iou", writer.tags)
        self.assertIn("val/iou/point", writer.tags)
        self.assertIn("val_iou/RC1", writer.tags)
        self.assertIn("val_iou_cat/circle_mask/MC", writer.tags)
        self.assertIn("val/iou_fulltile/point", writer.tags)
        self.assertIn("circle_mask", text)
        self.assertIn("point 0.7500", text)


if __name__ == "__main__":
    unittest.main()
