"""Per-window training scores written into the volume catalog."""

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
                "sample_losses": [1.5, 0.25, 0.1],
                "meta": [
                    {"volume": "RC1", "location_id": 10, "image_key": "win-a"},
                    {"volume": "RC1", "location_id": 10, "image_key": "win-b"},
                    {"volume": "RC1", "location_id": 11, "image_key": "win-c"},
                ],
            }
            record_location_scores(root, [row])
            record_location_scores(
                root,
                [
                    {
                        "epoch": 2,
                        "sample_losses": [9.0, 4.0],
                        "meta": [
                            {"volume": "RC1", "location_id": 10, "image_key": "win-a"},
                            {"volume": "RC1", "location_id": 10},
                        ],
                    },
                    {
                        "epoch": 3,
                        "sample_losses": [0.5],
                        "meta": [{"volume": "RC1", "location_id": 10, "image_key": "win-a"}],
                    },
                ],
            )
            connection = sqlite3.connect(db)
            try:
                stored = connection.execute(
                    "SELECT location_id, image_key, epoch, score FROM location_scores "
                    "ORDER BY location_id, image_key, epoch"
                ).fetchall()
            finally:
                connection.close()
        self.assertEqual(
            stored,
            [
                (10, "win-a", 2, 9.0),
                (10, "win-a", 3, 0.5),
                (10, "win-b", 2, 0.25),
                (11, "win-c", 2, 0.1),
            ],
        )

    def test_migrates_location_only_scores(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            crops = root / "RC1" / "AnnotationCrops"
            crops.mkdir(parents=True)
            db = crops / "annotation_crops.sqlite"
            connection = sqlite3.connect(db)
            connection.execute(
                "CREATE TABLE location_scores ("
                "location_id INTEGER NOT NULL, epoch INTEGER NOT NULL, "
                "score REAL NOT NULL, PRIMARY KEY (location_id, epoch))"
            )
            connection.execute(
                "INSERT INTO location_scores (location_id, epoch, score) VALUES (10, 1, 1.25)"
            )
            connection.commit()
            connection.close()
            record_location_scores(
                root,
                [
                    {
                        "epoch": 2,
                        "sample_losses": [0.4],
                        "meta": [{"volume": "RC1", "location_id": 10, "image_key": "win-a"}],
                    }
                ],
            )
            connection = sqlite3.connect(db)
            try:
                stored = connection.execute(
                    "SELECT location_id, image_key, epoch, score FROM location_scores "
                    "ORDER BY epoch"
                ).fetchall()
            finally:
                connection.close()
        self.assertEqual(stored, [(10, "", 1, 1.25), (10, "win-a", 2, 0.4)])


if __name__ == "__main__":
    unittest.main()
