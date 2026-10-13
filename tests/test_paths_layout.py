"""annotation_crops_root accepts the flat training layout and the legacy nested one."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from sam2_segmentation_trainer.data.manifest import index_volumes
from sam2_segmentation_trainer.paths import annotation_crops_root

try:
    from tests.helpers import make_crops
except ImportError:
    from helpers import make_crops


class CropsLayoutTests(unittest.TestCase):
    def test_flat_layout_is_the_volume_folder(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_crops(root, "RC1", n=4)
            self.assertEqual(annotation_crops_root("RC1", root), root / "RC1")
            self.assertEqual(len(index_volumes(volumes=["RC1"], root=root)), 6)

    def test_nested_layout_still_works(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_crops(root, "RC1", n=4, nested=True)
            self.assertEqual(annotation_crops_root("RC1", root), root / "RC1" / "AnnotationCrops")
            self.assertEqual(len(index_volumes(volumes=["RC1"], root=root)), 6)

    def test_flat_wins_when_both_exist(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_crops(root, "RC1", n=4)
            (root / "RC1" / "AnnotationCrops").mkdir()
            self.assertEqual(annotation_crops_root("RC1", root), root / "RC1")

    def test_missing_volume_defaults_to_nested_path(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(annotation_crops_root("RC9", root), root / "RC9" / "AnnotationCrops")


if __name__ == "__main__":
    unittest.main()
