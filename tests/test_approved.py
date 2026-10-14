from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from sam2_segmentation_trainer.data.approved import ApprovedSet, approved_counts, is_approved, load_approved
from sam2_segmentation_trainer.train import load_approved_set, load_train_cfg


class Example:
    def __init__(self, volume: str, location_id: int, image_key: str) -> None:
        self.volume, self.location_id, self.image_key = volume, location_id, image_key


def _write(folder: Path, payload) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "approved.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class DiscoveryTests(unittest.TestCase):
    def test_each_volumes_own_file_is_read_flat_or_nested(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "RC1").mkdir()
            (root / "RC1" / "manifest.jsonl").write_text("", encoding="utf-8")  # flat layout
            _write(root / "RC1", [{"location_id": 7, "image_key": "RC1_a"}])
            _write(root / "RC2" / "AnnotationCrops", [{"location_id": 8, "image_key": "RC2_b"}])  # nested layout
            (root / "RC3").mkdir()  # no file at all
            logs: list[str] = []
            approved = load_approved(None, ["RC1", "RC2", "RC3"], root, log=logs.append)
        self.assertTrue(approved.matches("RC1", 7, "RC1_a"))
        self.assertTrue(approved.matches("RC2", 8, "RC2_b"))
        self.assertFalse(approved.matches("RC1", 8, "RC2_b"))
        self.assertEqual(approved.entries_by_volume(), {"RC1": 1, "RC2": 1})
        self.assertTrue(any("RC1" in line and "RC2" in line for line in logs))
        self.assertFalse(any("WARNING" in line for line in logs))

    def test_an_empty_image_key_approves_every_window_of_the_location_in_that_volume(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "RC1" / "AnnotationCrops", [{"location_id": 9, "image_key": ""}, [10], {"location_id": 11}])
            approved = load_approved(None, ["RC1"], root, log=lambda _m: None)
        for location in (9, 11):
            self.assertTrue(approved.matches("RC1", location, "RC1_0_x"))
            self.assertTrue(approved.matches("RC1", location, "RC1_5_y"))
        self.assertFalse(approved.matches("RC1", 12, "RC1_0_x"))

    def test_the_same_location_id_in_two_volumes_is_kept_apart(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "RC1" / "AnnotationCrops", [{"location_id": 5, "image_key": ""}])
            _write(root / "RC2" / "AnnotationCrops", [{"location_id": 6, "image_key": "RC2_k"}])
            approved = load_approved(None, ["RC1", "RC2"], root, log=lambda _m: None)
        self.assertTrue(is_approved(approved, Example("RC1", 5, "RC1_k")))
        self.assertFalse(is_approved(approved, Example("RC2", 5, "RC2_k")))
        self.assertTrue(is_approved(approved, Example("RC2", 6, "RC2_k")))
        self.assertFalse(is_approved(approved, Example("RC1", 6, "RC2_k")))
        counts = approved_counts(
            [Example("RC1", 5, "a"), Example("RC1", 5, "b"), Example("RC2", 5, "a"), Example("RC2", 6, "RC2_k")], approved
        )
        self.assertEqual(counts, {"RC1": 2, "RC2": 1})

    def test_path_may_be_a_file_a_directory_or_a_list(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            single = _write(root / "elsewhere", [{"location_id": 1, "image_key": "k1"}])
            _write(root / "RC2" / "AnnotationCrops", [{"location_id": 2, "image_key": "k2"}])
            quiet = lambda _m: None  # noqa: E731
            by_file = load_approved(str(single), ["RC1", "RC2"], root, log=quiet)
            by_dir = load_approved(str(root), ["RC1", "RC2"], root, log=quiet)
            by_list = load_approved([str(single), str(root)], ["RC1", "RC2"], root, log=quiet)
        self.assertTrue(by_file.matches("RC1", 1, "k1") and by_file.matches("RC2", 1, "k1"))  # names no volume: all
        self.assertTrue(by_dir.matches("RC2", 2, "k2"))
        self.assertFalse(by_dir.matches("RC1", 2, "k2"))
        self.assertTrue(by_list.matches("RC1", 1, "k1") and by_list.matches("RC2", 2, "k2"))

    def test_zero_entries_warns_and_junk_is_skipped(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "RC1" / "AnnotationCrops", ["junk", {"x": 1}])
            logs: list[str] = []
            approved = load_approved(None, ["RC1"], root, log=logs.append)
        self.assertEqual(len(approved), 0)
        self.assertFalse(approved)
        self.assertTrue(any(line.startswith("WARNING") for line in logs))

    def test_train_entry_point_reads_the_configured_volumes(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "RC1" / "AnnotationCrops", [{"location_id": 3, "image_key": ""}])
            cfg = load_train_cfg(["data.volumes=[RC1]"])
            approved = load_approved_set(cfg, root)
        self.assertIsInstance(approved, ApprovedSet)
        self.assertTrue(approved.matches("RC1", 3, "anything"))

    def test_plain_pairs_still_apply_to_every_volume(self) -> None:
        pairs = frozenset({(4, "k")})
        self.assertTrue(is_approved(pairs, Example("RC1", 4, "k")))
        self.assertFalse(is_approved(pairs, Example("RC1", 4, "other")))
        self.assertFalse(is_approved(frozenset(), Example("RC1", 4, "k")))


if __name__ == "__main__":
    unittest.main()
