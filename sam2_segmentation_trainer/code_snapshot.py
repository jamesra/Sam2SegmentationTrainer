"""A record of the exact code each launch of a run trained with, start and every resume alike.

A resume runs whatever is in the working tree at that moment, so the code of a run's later epochs can differ from its
first launch's. Each launch writes ``<run_dir>/code/<UTC time>/``:

* ``base``: the full SHA of ``HEAD``;
* ``patch``: ``git diff --binary HEAD``, the tracked changes not yet committed;
* ``untracked.tgz``: the untracked files git does not ignore;
* ``config.yaml``: the launch's resolved config (the run's own ``config.yaml`` is overwritten by every launch);
* ``launch.json``: when, the command line, and whether each part was written.

The repo state is rebuilt with ``git checkout <base>``, ``git apply patch`` and ``tar xzf untracked.tgz``.

Training never stops for this: a part that cannot be written (no git, not a checkout) is recorded as an error in
``launch.json`` and a warning is printed.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SNAPSHOT_FOLDER = "code"
_TIMEOUT_SECONDS = 120


def repo_root() -> Path:
    """The checkout this package was imported from (an editable install), whether or not it is a git repo."""
    return Path(__file__).resolve().parent.parent


def _git(root: Path, *args: str) -> bytes:
    """Run git in ``root`` and return its stdout. ``safe.directory`` covers a bind-mounted repo owned by another user."""
    return subprocess.run(
        ["git", "-c", f"safe.directory={root.as_posix()}", *args],
        cwd=root,
        capture_output=True,
        timeout=_TIMEOUT_SECONDS,
        check=True,
    ).stdout


def _untracked(root: Path) -> list[str]:
    out = _git(root, "ls-files", "--others", "--exclude-standard", "-z")
    return [name for name in out.decode("utf-8", "surrogateescape").split("\0") if name]


def write_code_snapshot(
    run_dir: Path,
    config_text: str | None = None,
    *,
    root: Path | None = None,
    now: datetime | None = None,
) -> Path | None:
    """Write this launch's snapshot under ``run_dir/code/`` and return its folder (None if the folder could not be made)."""
    root = repo_root() if root is None else Path(root)
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    folder = Path(run_dir) / SNAPSHOT_FOLDER / stamp
    try:
        folder.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        print(f"WARNING: code snapshot not written: {folder}: {error}")
        return None

    record: dict[str, Any] = {"time": stamp, "argv": list(sys.argv), "repo": str(root), "parts": {}}
    parts = record["parts"]

    def attempt(name: str, write) -> None:
        try:
            parts[name] = write()
        except (OSError, subprocess.SubprocessError, ValueError, tarfile.TarError) as error:
            detail = error.stderr.decode("utf-8", "replace").strip() if isinstance(error, subprocess.CalledProcessError) and error.stderr else str(error)
            parts[name] = {"error": detail[:500]}
            print(f"WARNING: code snapshot part {name!r} not written: {detail[:200]}")

    def base() -> str:
        sha = _git(root, "rev-parse", "HEAD").decode().strip()
        (folder / "base").write_text(sha + "\n", encoding="utf-8")
        return sha

    def patch() -> dict[str, int]:
        diff = _git(root, "diff", "--binary", "HEAD")
        (folder / "patch").write_bytes(diff)
        return {"bytes": len(diff)}

    def untracked() -> dict[str, int]:
        names = _untracked(root)
        with tarfile.open(folder / "untracked.tgz", "w:gz") as archive:
            for name in names:
                if (root / name).is_file():
                    archive.add(root / name, arcname=name, recursive=False)
        return {"files": len(names)}

    attempt("base", base)
    attempt("patch", patch)
    attempt("untracked", untracked)
    if config_text is not None:

        def config() -> dict[str, int]:
            return {"bytes": (folder / "config.yaml").write_text(config_text, encoding="utf-8")}

        attempt("config", config)

    try:
        (folder / "launch.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    except OSError as error:
        print(f"WARNING: code snapshot record not written: {error}")
    print(f"code snapshot: {folder}")
    return folder
