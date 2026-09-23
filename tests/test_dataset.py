from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from sam2_segmentation_trainer.data.dataset import EMSegDataset, collate_em
from sam2_segmentation_trainer.data.manifest import index_volumes, take_model_sized

try:
    from tests.helpers import make_crops
except ImportError:
    from helpers import make_crops


class DatasetTests(unittest.TestCase):
    def test_item_shapes_and_normalize(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_crops(root, "RC1", n=4)
            examples = index_volumes(volumes=["RC1"], root=root)
            ds = EMSegDataset(examples, split="val", image_size=32)
            item = ds[0]
            self.assertEqual(tuple(item["image"].shape), (3, 32, 32))
            self.assertEqual(tuple(item["mask"].shape), (32, 32))
            self.assertEqual(tuple(item["point"].shape), (1, 2))
            self.assertTrue(item["image"].abs().mean() > 0)
            batch = collate_em([ds[0], ds[1]])
            self.assertEqual(tuple(batch["image"].shape), (2, 3, 32, 32))
            self.assertEqual(batch["point"].dtype, torch.float32)
            self.assertTrue(item["category_name"])

    def test_train_augment_keeps_shapes(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_crops(root, "RC1", n=3)
            examples = index_volumes(volumes=["RC1"], root=root)
            ds = EMSegDataset(examples, split="train", image_size=32)
            item = ds[0]
            self.assertEqual(tuple(item["image"].shape), (3, 32, 32))
            self.assertEqual(int(item["mask"].max().item()) in (0, 1), True)

    def test_size_metadata_skips_tile_once(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_crops(root, "RC1", n=4)
            examples = index_volumes(volumes=["RC1"], root=root)
            self.assertTrue(all(ex.width == 32 and ex.height == 32 for ex in examples))
            kept, rejected = take_model_sized(examples, 1024)
            self.assertEqual(kept, [])
            self.assertEqual(len(rejected), 4)
            self.assertEqual(len({ex.image_key for ex in rejected}), 4)
            kept32, rejected32 = take_model_sized(examples, 32)
            self.assertEqual(len(kept32), len(examples))
            self.assertEqual(rejected32, [])


if __name__ == "__main__":
    unittest.main()
