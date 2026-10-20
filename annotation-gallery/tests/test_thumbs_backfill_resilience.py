"""Thumbnail backfill keeps going after a crop or a whole pass raises something other than ``OSError``."""

from __future__ import annotations

import sys
import tempfile
import threading
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from PIL import Image

GALLERY_ROOT = Path(__file__).resolve().parents[1]
if str(GALLERY_ROOT) not in sys.path:
    sys.path.insert(0, str(GALLERY_ROOT))

import thumbs  # noqa: E402

JOIN_TIMEOUT_S = 5.0


def _join(gate: thumbs._BackfillGate) -> None:
    """Wait for the gate's current thread and fail if it is still running."""
    thread = gate._thread
    assert thread is not None
    thread.join(JOIN_TIMEOUT_S)
    assert not thread.is_alive()


def _bomb_for(bad: set[str]):
    """Return an ``Image.open`` that raises Pillow's decompression-bomb error for the keys in *bad*."""
    real_open = Image.open

    def opener(path, *args, **kwargs):
        if Path(path).stem in bad:
            raise Image.DecompressionBombError(f"{Path(path).name} is too large")
        return real_open(path, *args, **kwargs)

    return opener


@settings(max_examples=25, deadline=None)
@given(
    flags=st.lists(st.booleans(), min_size=1, max_size=12),
    workers=st.integers(min_value=1, max_value=4),
)
def test_bad_crop_is_skipped_and_the_others_are_written(flags: list[bool], workers: int) -> None:
    keys = [f"RC2_{index}_D1_X0_Y0" for index in range(len(flags))]
    bad = {key for key, is_bad in zip(keys, flags) if is_bad}
    good = [key for key in keys if key not in bad]
    with tempfile.TemporaryDirectory() as root, pytest.MonkeyPatch.context() as patch:
        crops = Path(root) / "AnnotationCrops"
        (crops / "images").mkdir(parents=True)
        for key in keys:
            Image.new("L", (8, 8), 128).save(crops / "images" / f"{key}.png")
        patch.setattr(thumbs, "_ENCODE_WORKERS", workers)
        patch.setattr(thumbs.Image, "open", _bomb_for(bad))
        written = thumbs.backfill_crops(crops)
        assert written == len(good) * len(thumbs.THUMB_SIZES)
        for size in thumbs.THUMB_SIZES:
            folder = crops / "overlays" / "thumb" / str(size)
            present = {path.stem for path in folder.glob("*.jpg")} if folder.is_dir() else set()
            assert present == set(good)


def test_bad_crop_is_logged_with_its_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    crops = tmp_path / "AnnotationCrops"
    (crops / "images").mkdir(parents=True)
    Image.new("L", (8, 8)).save(crops / "images" / "RC2_1_D1_X0_Y0.png")
    monkeypatch.setattr(thumbs.Image, "open", _bomb_for({"RC2_1_D1_X0_Y0"}))
    assert thumbs.backfill_crops(crops) == 0
    output = capsys.readouterr().out
    assert "RC2_1_D1_X0_Y0" in output
    assert "DecompressionBombError" in output


@pytest.mark.parametrize(
    ("failure", "escapes"),
    [(ValueError("decoder broke"), False), (SystemExit(3), True)],
)
def test_gate_starts_a_new_pass_after_one_raises(
    monkeypatch: pytest.MonkeyPatch, failure: BaseException, escapes: bool
) -> None:
    """A failed pass clears the running flag, so the next request (the watch's or a handler's) still runs.

    An ``Exception`` is logged inside the thread. A ``BaseException`` such as
    ``SystemExit`` is not caught; it still ends the thread.
    """
    escaped: list[type[BaseException]] = []
    monkeypatch.setattr(threading, "excepthook", lambda args: escaped.append(args.exc_type))
    gate = thumbs._BackfillGate()
    calls: list[Path] = []
    ran = threading.Event()

    def fake(registry: Path) -> None:
        calls.append(registry)
        if len(calls) == 1:
            raise failure
        ran.set()

    monkeypatch.setattr(thumbs, "_backfill_registry", fake)
    gate.request(Path("a"))
    _join(gate)
    assert not gate._running
    gate.request(Path("b"))
    assert ran.wait(JOIN_TIMEOUT_S)
    _join(gate)
    assert calls == [Path("a"), Path("b")]
    assert not gate._running
    assert escaped == ([type(failure)] if escapes else [])


def test_failed_pass_is_logged_with_a_traceback(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    gate = thumbs._BackfillGate()

    def fake(_registry: Path) -> None:
        raise ValueError("decoder broke")

    monkeypatch.setattr(thumbs, "_backfill_registry", fake)
    gate.request(Path("a"))
    _join(gate)
    output = capsys.readouterr().out
    assert "Traceback" in output
    assert "ValueError: decoder broke" in output


def test_failed_pass_still_runs_the_queued_follow_up(monkeypatch: pytest.MonkeyPatch) -> None:
    """A change that arrived during the failed pass is not dropped."""
    gate = thumbs._BackfillGate()
    entered = threading.Event()
    release = threading.Event()
    calls: list[Path] = []

    def fake(registry: Path) -> None:
        calls.append(registry)
        if len(calls) == 1:
            entered.set()
            release.wait(JOIN_TIMEOUT_S)
            raise ValueError("decoder broke")

    monkeypatch.setattr(thumbs, "_backfill_registry", fake)
    gate.request(Path("a"))
    assert entered.wait(JOIN_TIMEOUT_S)
    gate.request(Path("b"))
    release.set()
    _join(gate)
    assert calls == [Path("a"), Path("b")]
