from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from sam2_segmentation_trainer.data.manifest import index_volumes, summarize_index
from sam2_segmentation_trainer.data.rle import decode_coco_rle
from sam2_segmentation_trainer.data.splits import load_split, make_location_split, save_split
from sam2_segmentation_trainer.validate import build_report

try:
    from tests.helpers import encode_uncompressed, make_crops
except ImportError:
    from helpers import encode_uncompressed, make_crops

import numpy as np
from PIL import Image


class RleTests(unittest.TestCase):
    def test_roundtrip_uncompressed(self) -> None:
        mask = np.zeros((16, 20), dtype=np.uint8)
        mask[2:10, 3:8] = 1
        counts = encode_uncompressed(mask)
        decoded = decode_coco_rle({"counts": counts, "size": [16, 20]})
        np.testing.assert_array_equal(decoded, mask)


class ManifestSplitTests(unittest.TestCase):
    def test_index_expands_location_ids(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_crops(root, "RC1", n=6)
            examples = index_volumes(volumes=["RC1"], root=root)
            self.assertGreater(len(examples), 6)
            summary = summarize_index(examples)
            self.assertGreaterEqual(summary["multi_id_tiles"], 1)
            self.assertEqual(summary["missing_image"], 0)
            keys = {ex.split_key for ex in examples}
            self.assertEqual(len(keys), len(examples))

    def test_split_no_key_overlap_and_persist(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_crops(root, "RC1", n=10)
            make_crops(root, "RC2", n=8)
            examples = index_volumes(volumes=["RC1", "RC2"], root=root)
            split = make_location_split(examples, train_fraction=0.9, seed=0)
            train_set = set(split.train_keys)
            val_set = set(split.val_keys)
            self.assertFalse(train_set & val_set)
            self.assertEqual(train_set | val_set, {ex.split_key for ex in examples})
            path = root / "splits" / "v1.json"
            save_split(split, path)
            loaded = load_split(path)
            self.assertEqual(loaded.train_keys, split.train_keys)
            train, val = loaded.partition(examples)
            self.assertTrue(train)
            self.assertTrue(val)

    def test_even_per_volume_caps_to_smallest(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_crops(root, "RC1", n=12)
            make_crops(root, "RC2", n=6)
            examples = index_volumes(volumes=["RC1", "RC2"], root=root)
            split = make_location_split(
                examples, train_fraction=0.9, seed=1, even_per_volume=True
            )
            train, val = split.partition(examples)
            train_n = {}
            val_n = {}
            for ex in train:
                train_n[ex.volume] = train_n.get(ex.volume, 0) + 1
            for ex in val:
                val_n[ex.volume] = val_n.get(ex.volume, 0) + 1
            self.assertEqual(len(set(train_n.values())), 1)
            self.assertEqual(len(set(val_n.values())), 1)
            self.assertGreater(train_n["RC1"], 0)
            self.assertEqual(train_n["RC1"], train_n["RC2"])
            self.assertLess(len(train) + len(val), len(examples))

    def test_validate_report(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_crops(root, "RC1", n=4)
            report = build_report(
                root=root, volumes=["RC1"], check_json=True, json_limit=None
            )
            self.assertGreater(report["usable_examples"], 0)
            self.assertEqual(report["indexed_including_missing"]["missing_image"], 0)
            self.assertEqual(report["json_stats"]["empty_area"], 0)

    def test_skips_location_ids_missing_from_json(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_crops(root, "RC1", n=4)
            crops = root / "RC1" / "AnnotationCrops"
            # Append a manifest row whose JSON has no matching annotation id.
            key = "RC1_orphan_D1_X0-1_Y0-1"
            gray = np.full((32, 32), 10, dtype=np.uint8)
            Image.fromarray(gray).save(crops / "images" / f"{key}.png")
            sidecar = {
                "image": {
                    "file_name": f"{key}.png",
                    "width": 32,
                    "height": 32,
                    "downsample": 1,
                    "volume": "RC1",
                },
                "annotations": [],
            }
            (crops / "images" / f"{key}.json").write_text(
                json.dumps(sidecar), encoding="utf-8"
            )
            with (crops / "manifest.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "z": 99,
                            "volume": "RC1",
                            "imageKey": key,
                            "downsample": 1,
                            "image": f"images/{key}.png",
                            "json": f"images/{key}.json",
                            "locationIds": [999001],
                        }
                    )
                    + "\n"
                )
            all_ex = index_volumes(volumes=["RC1"], root=root, skip_missing=False)
            usable = index_volumes(volumes=["RC1"], root=root, skip_missing=True)
            self.assertTrue(any(ex.location_id == 999001 for ex in all_ex))
            self.assertFalse(any(ex.location_id == 999001 for ex in usable))
            self.assertLess(len(usable), len(all_ex))

    def test_skips_examples_without_raster_mask(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_crops(root, "RC1", n=4)
            examples = index_volumes(volumes=["RC1"], root=root, skip_missing=True)
            target = examples[0]
            target.mask_path.unlink()
            all_ex = index_volumes(volumes=["RC1"], root=root, skip_missing=False)
            usable = index_volumes(volumes=["RC1"], root=root, skip_missing=True)
            self.assertTrue(any(ex.split_key == target.split_key for ex in all_ex))
            self.assertFalse(any(ex.split_key == target.split_key for ex in usable))
            self.assertEqual(len(usable), len(examples) - 1)
            summary = summarize_index(all_ex)
            self.assertEqual(summary["missing_raster_mask"], 1)


if __name__ == "__main__":
    unittest.main()
