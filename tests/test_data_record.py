from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from sam2_segmentation_trainer.data_record import (
    RECORD_FILE,
    build_data_record,
    read_data_record,
    write_data_record,
)


def _volume(root: Path, name: str, manifest: str, approved: str | None = None) -> None:
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "manifest.jsonl").write_text(manifest, encoding="utf-8")
    if approved is not None:
        (folder / "approved.json").write_text(approved, encoding="utf-8")


class DataRecordTests(unittest.TestCase):
    def test_the_record_names_the_set_and_fingerprints_the_files(self) -> None:
        with TemporaryDirectory() as tmp:
            version = Path(tmp) / "v3"
            _volume(version, "RC1", '{"a":1}\n{"a":2}\n', approved="{}")
            record = build_data_record(
                version, ["RC1"], train_counts={"RC1": 8}, val_counts={"RC1": 2}, split_seed=42, train_fraction=0.8
            )
        self.assertEqual(record["set"], "v3")
        self.assertEqual(record["volumes"]["RC1"]["manifest.jsonl"]["lines"], 2)
        self.assertIsNone(record["volumes"]["RC1"]["ignore.json"])  # absent is recorded, not skipped
        self.assertEqual(record["train_examples"], {"RC1": 8})
        self.assertEqual(record["split_seed"], 42)
        self.assertEqual(len(record["fingerprint"]), 12)

    def test_the_fingerprint_changes_when_the_inventory_changes(self) -> None:
        with TemporaryDirectory() as tmp:
            version = Path(tmp) / "v3"
            _volume(version, "RC1", '{"a":1}\n')
            first = build_data_record(version, ["RC1"], train_counts={}, val_counts={})["fingerprint"]
            same = build_data_record(version, ["RC1"], train_counts={}, val_counts={})["fingerprint"]
            (version / "RC1" / "manifest.jsonl").write_text('{"a":1}\n{"a":2}\n', encoding="utf-8")
            changed = build_data_record(version, ["RC1"], train_counts={}, val_counts={})["fingerprint"]
        self.assertEqual(first, same)
        self.assertNotEqual(first, changed)

    def test_a_current_symlink_resolves_to_its_version_folder(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _volume(root / "v4", "RC1", "{}\n")
            try:
                (root / "current").symlink_to(root / "v4", target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("symlinks are not available here")
            record = build_data_record(root / "current", ["RC1"], train_counts={}, val_counts={})
        self.assertEqual(record["set"], "v4")
        self.assertEqual(record["requested"], "current")

    def test_write_and_read_round_trip(self) -> None:
        with TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            written = write_data_record(run, {"set": "v3", "fingerprint": "abc"})
            self.assertEqual(written, run / RECORD_FILE)
            self.assertEqual(read_data_record(run), {"set": "v3", "fingerprint": "abc"})
            self.assertIsNone(read_data_record(Path(tmp) / "none"))
            (run / RECORD_FILE).write_text("not json", encoding="utf-8")
            self.assertIsNone(read_data_record(run))


if __name__ == "__main__":
    unittest.main()
