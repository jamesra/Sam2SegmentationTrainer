from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from sam2_segmentation_trainer.data.dataset import EMSegDataset, collate_em
from sam2_segmentation_trainer.data.manifest import index_volumes

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
            ds = EMSegDataset(examples, split="val", image_size=64)
            item = ds[0]
            self.assertEqual(tuple(item["image"].shape), (3, 64, 64))
            self.assertEqual(tuple(item["mask"].shape), (64, 64))
            self.assertEqual(tuple(item["point"].shape), (1, 2))
            self.assertTrue(item["image"].abs().mean() > 0)
            batch = collate_em([ds[0], ds[1]])
            self.assertEqual(tuple(batch["image"].shape), (2, 3, 64, 64))
            self.assertEqual(batch["point"].dtype, torch.float32)
            self.assertTrue(item["category_name"])

    def test_train_augment_keeps_shapes(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_crops(root, "RC1", n=3)
            examples = index_volumes(volumes=["RC1"], root=root)
            ds = EMSegDataset(examples, split="train", image_size=64)
            item = ds[0]
            self.assertEqual(tuple(item["image"].shape), (3, 64, 64))
            self.assertEqual(int(item["mask"].max().item()) in (0, 1), True)


if __name__ == "__main__":
    unittest.main()
