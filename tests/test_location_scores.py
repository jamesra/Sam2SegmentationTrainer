"""Per-epoch location scores written into the volume catalog."""

from __future__ import annotations

import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from sam2_segmentation_trainer.train import record_location_scores


class LocationScoreTests(unittest.TestCase):
    def test_upsert_replaces_same_epoch_and_keeps_others(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            crops = root / "RC1" / "AnnotationCrops"
            crops.mkdir(parents=True)
            db = crops / "annotation_crops.sqlite"
            db.write_bytes(b"")
            row = {
                "epoch": 2,
                "sample_losses": [1.5, 0.25],
                "meta": [
                    {"volume": "RC1", "location_id": 10},
                    {"volume": "RC1", "location_id": 11},
                ],
            }
            record_location_scores(root, [row])
            record_location_scores(
                root,
                [
                    {
                        "epoch": 2,
                        "sample_losses": [9.0],
                        "meta": [{"volume": "RC1", "location_id": 10}],
                    },
                    {
                        "epoch": 3,
                        "sample_losses": [0.5],
                        "meta": [{"volume": "RC1", "location_id": 10}],
                    },
                ],
            )
            connection = sqlite3.connect(db)
            try:
                stored = connection.execute(
                    "SELECT location_id, epoch, score FROM location_scores "
                    "ORDER BY location_id, epoch"
                ).fetchall()
            finally:
                connection.close()
        self.assertEqual(stored, [(10, 2, 9.0), (10, 3, 0.5), (11, 2, 0.25)])


if __name__ == "__main__":
    unittest.main()
