"""Each launch's code snapshot rebuilds the working tree it ran, and never stops training when git is unavailable."""

import json
import subprocess
import tarfile
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from sam2_segmentation_trainer.code_snapshot import write_code_snapshot


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout


class CodeSnapshotTests(unittest.TestCase):
    def _repo(self, tmp: Path) -> Path:
        root = tmp / "repo"
        root.mkdir()
        _git(root, "init", "-q")
        _git(root, "config", "user.email", "t@example.com")
        _git(root, "config", "user.name", "t")
        (root / ".gitignore").write_text("ignored.txt\n")
        (root / "a.py").write_text("one\n")
        _git(root, "add", ".")
        _git(root, "commit", "-q", "-m", "base")
        return root

    def test_snapshot_rebuilds_the_working_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            root = self._repo(tmp)
            (root / "a.py").write_text("two\n")
            (root / "new.py").write_text("untracked\n")
            (root / "ignored.txt").write_text("not recorded\n")
            run = tmp / "run"
            folder = write_code_snapshot(run, "a: 1\n", root=root, now=datetime(2026, 11, 20, 1, 2, 3, tzinfo=timezone.utc))

            self.assertEqual(folder, run / "code" / "20261120T010203Z")
            self.assertEqual((folder / "base").read_text().strip(), _git(root, "rev-parse", "HEAD").strip())
            self.assertEqual((folder / "config.yaml").read_text(), "a: 1\n")
            with tarfile.open(folder / "untracked.tgz") as archive:
                self.assertEqual(archive.getnames(), ["new.py"])
            record = json.loads((folder / "launch.json").read_text())
            self.assertEqual(record["parts"]["untracked"], {"files": 1})

            clone = tmp / "clone"
            subprocess.run(["git", "clone", "-q", str(root), str(clone)], check=True)
            _git(clone, "checkout", "-q", (folder / "base").read_text().strip())
            _git(clone, "apply", str(folder / "patch"))
            self.assertEqual((clone / "a.py").read_text(), "two\n")

    def test_each_launch_gets_its_own_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            root = self._repo(tmp)
            first = write_code_snapshot(tmp / "run", root=root, now=datetime(2026, 1, 1, tzinfo=timezone.utc))
            second = write_code_snapshot(tmp / "run", root=root, now=datetime(2026, 1, 2, tzinfo=timezone.utc))
            self.assertNotEqual(first, second)
            self.assertTrue((first / "base").is_file() and (second / "base").is_file())

    def test_not_a_checkout_is_recorded_not_raised(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            plain = tmp / "plain"
            plain.mkdir()
            folder = write_code_snapshot(tmp / "run", root=plain)
            record = json.loads((folder / "launch.json").read_text())
            self.assertIn("error", record["parts"]["base"])
            self.assertIn("error", record["parts"]["patch"])


if __name__ == "__main__":
    unittest.main()
