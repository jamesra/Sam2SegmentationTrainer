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


if __name__ == "__main__":
    unittest.main()
