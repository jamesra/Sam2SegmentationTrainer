"""Training set names (v2, v2.1, the old v1b) as the WSL scripts accept and order them."""

from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "WSL" / "training_data_common.sh"


def _bash(snippet: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str] | None:
    """Run ``snippet`` in bash with the helpers sourced; None when bash cannot source the script here."""
    exe = shutil.which("bash")
    if exe is None:
        return None
    try:
        return subprocess.run(
            [exe, "-c", f"source '{SCRIPT.as_posix()}' && {snippet}"],
            capture_output=True, text=True, timeout=60, cwd=cwd,
        )
    except OSError:
        return None


def _available() -> bool:
    probe = _bash("echo ready")
    return probe is not None and probe.returncode == 0 and probe.stdout.strip() == "ready"


@unittest.skipUnless(_available(), "bash cannot source scripts/WSL/training_data_common.sh here")
class VersionNameTests(unittest.TestCase):
    def accepted(self, name: str) -> bool:
        result = _bash(f"version_name_ok '{name}'")
        assert result is not None
        return result.returncode == 0

    def test_a_set_and_its_revisions_are_accepted(self) -> None:
        for name in ("v1", "v2", "v10", "v2.1", "v2.10", "v12.3"):
            self.assertTrue(self.accepted(name), name)

    def test_the_old_letter_suffix_is_still_accepted(self) -> None:
        self.assertTrue(self.accepted("v1b"))

    def test_other_names_are_rejected(self) -> None:
        # v2.0 is just v2, a revision has one number, and a pointer such as current is not a version.
        for name in ("v2.0", "v2.01", "v2.", "v2.1.1", "v2.1b", "2", "vx", "current", "v", ""):
            self.assertFalse(self.accepted(name), name)

    def test_a_leading_zero_is_not_read_as_octal(self) -> None:
        result = _bash("version_sort_key v08")
        self.assertEqual((result.returncode, result.stdout.strip()), (0, "00008.00000"))

    def test_the_highest_version_is_chosen_numerically(self) -> None:
        names = ["v1", "v1b", "v2", "v2.1", "v2.2", "v2.10", "v3", "v3.1", "notes", "v2.0", "v2.1b"]
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in names:
                (root / name).mkdir()
            result = _bash("highest_version_dir .", cwd=root)
            self.assertEqual((result.returncode, result.stdout.strip()), (0, "v3.1"))
            (root / "v3.1").rmdir()
            result = _bash("highest_version_dir .", cwd=root)
            self.assertEqual(result.stdout.strip(), "v3")
            (root / "v3").rmdir()
            result = _bash("highest_version_dir .", cwd=root)
            self.assertEqual(result.stdout.strip(), "v2.10")  # v2.10 is above v2.2, not below it

    def test_a_revision_sorts_between_its_set_and_the_next(self) -> None:
        keys = {}
        for name in ("v2", "v2.1", "v2.10", "v3"):
            result = _bash(f"version_sort_key '{name}'")
            keys[name] = result.stdout.strip()
        self.assertLess(keys["v2"], keys["v2.1"])
        self.assertLess(keys["v2.1"], keys["v2.10"])
        self.assertLess(keys["v2.10"], keys["v3"])


@unittest.skipUnless(_available(), "bash cannot source scripts/WSL/training_data_common.sh here")
class MirrorTaskTests(unittest.TestCase):
    def tasks(self, src: Path, dest: Path) -> list[str]:
        result = _bash(f"list_mirror_tasks '{src.as_posix()}' '{dest.as_posix()}' | tr '\\0' '\\n'")
        assert result is not None
        self.assertEqual(result.returncode, 0, result.stderr)
        return [line.split("\t")[1] for line in result.stdout.splitlines() if line]

    def test_version_root_cache_and_review_folders_are_not_mirrored(self) -> None:
        with TemporaryDirectory() as tmp:
            src = Path(tmp) / "src" / "v1"
            for folder in ("RC1/images", "RC1/_work", "_work_cache/RC1/images", "_review/notes"):
                (src / folder).mkdir(parents=True)
            (src / "RC1" / "manifest.jsonl").write_text("{}\n")
            (src / "_work_cache" / "manifest.jsonl").write_text("{}\n")  # looks like a volume, is not one

            rels = self.tasks(src, Path(tmp) / "dest" / "v1")

            self.assertIn("RC1/images", rels)
            self.assertNotIn("RC1/_work", rels)
            self.assertFalse([r for r in rels if r.startswith(("_work_cache", "_review"))], rels)

    def test_a_cache_folder_does_not_fail_the_layout_check(self) -> None:
        with TemporaryDirectory() as tmp:
            src, dest = Path(tmp) / "src", Path(tmp) / "dest"
            (src / "RC1").mkdir(parents=True)
            (src / "RC1" / "manifest.jsonl").write_text("{}\n")
            (dest / "RC1" / "AnnotationCrops").mkdir(parents=True)  # nested on one side: a real mismatch
            (src / "_work_cache").mkdir()
            (src / "_work_cache" / "manifest.jsonl").write_text("{}\n")
            (dest / "_work_cache" / "AnnotationCrops").mkdir(parents=True)

            result = _bash(f"check_layouts_match '{src.as_posix()}' '{dest.as_posix()}'")

            self.assertEqual(result.returncode, 1)
            self.assertIn("RC1", result.stderr)
            self.assertNotIn("_work_cache", result.stderr)


if __name__ == "__main__":
    unittest.main()
