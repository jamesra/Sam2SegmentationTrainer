"""Review lists (approved.json, ignore.json) under concurrent handler threads and interrupted writes."""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

GALLERY_ROOT = Path(__file__).resolve().parents[1]
if str(GALLERY_ROOT) not in sys.path:
    sys.path.insert(0, str(GALLERY_ROOT))

import catalog  # noqa: E402

# Long enough that unlocked threads overlap inside one read-modify-write, short enough to keep the suite fast.
RACE_PAUSE = 0.002


def _slow_after(function):
    """Wrap *function* so it pauses after returning, widening a read-modify-write window."""
    def wrapper(*args, **kwargs):
        result = function(*args, **kwargs)
        time.sleep(RACE_PAUSE)
        return result
    return wrapper


def _run_together(targets) -> list[BaseException]:
    """Start every target at once, wait for all, and return what they raised."""
    barrier = threading.Barrier(len(targets))
    errors: list[BaseException] = []

    def run(target):
        barrier.wait()
        try:
            target()
        except BaseException as exc:  # noqa: BLE001 - the test reports every failure.
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(target,)) for target in targets]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return errors


def test_concurrent_review_clicks_keep_every_location(tmp_path, monkeypatch):
    """Every review action from many handler threads lands in its list."""
    catalog.save_approved(tmp_path, [(i, "K") for i in range(51, 57)])
    catalog.save_ignore_ids(tmp_path, range(151, 157))
    monkeypatch.setattr(catalog, "load_approved", _slow_after(catalog.load_approved))
    monkeypatch.setattr(catalog, "load_ignore_ids", _slow_after(catalog.load_ignore_ids))
    targets = [lambda i=i: catalog.approve_window(tmp_path, i, "K") for i in range(1, 9)]
    targets += [lambda i=i: catalog.ignore_location(tmp_path, i) for i in range(101, 109)]
    targets += [lambda i=i: catalog.unapprove_window(tmp_path, i, "K") for i in range(51, 57)]
    targets += [lambda i=i: catalog.restore_location(tmp_path, i) for i in range(151, 157)]
    targets += [lambda: catalog.import_review_lists(tmp_path, [300], [(301, "K")], replace=False)]

    assert _run_together(targets) == []
    assert catalog.load_approved(tmp_path) == {(i, "K") for i in range(1, 9)} | {(301, "K")}
    assert catalog.load_ignore_ids(tmp_path) == set(range(101, 109)) | {300}


def test_batch_and_single_review_do_not_lose_each_other(tmp_path, monkeypatch):
    """A batch reject running beside single approvals keeps both results."""
    monkeypatch.setattr(catalog, "load_approved", _slow_after(catalog.load_approved))
    targets = [lambda: catalog.review_locations(tmp_path, range(200, 220), "ignore")]
    targets += [lambda i=i: catalog.approve_window(tmp_path, i, "K") for i in range(1, 9)]

    assert _run_together(targets) == []
    assert catalog.load_ignore_ids(tmp_path) == set(range(200, 220))
    assert catalog.load_approved(tmp_path) == {(i, "K") for i in range(1, 9)}


def _truncate_then_write(path: Path, text: str, encoding: str | None = None, **_kwargs) -> int:
    """Write the way ``Path.write_text`` does (truncate first), slowly, so a reader can land in the gap."""
    with open(path, "w", encoding=encoding) as handle:
        handle.flush()
        time.sleep(RACE_PAUSE)
        return handle.write(text)


def _read_while_writing(read, write_all) -> tuple[list[BaseException], list[int]]:
    """Run *write_all* in one thread while another calls *read* until it ends.

    Returns what either thread raised and the size of each set the reader saw.
    """
    errors: list[BaseException] = []
    sizes: list[int] = []
    done = threading.Event()

    def writer():
        try:
            write_all()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            done.set()

    def reader():
        try:
            while not done.is_set():
                sizes.append(len(read()))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return errors, sizes


def test_reader_never_sees_a_half_written_approved_list(tmp_path, monkeypatch):
    """The catalog route reading approved.json during an approval gets a whole list, not an error."""
    monkeypatch.setattr(Path, "write_text", _truncate_then_write)

    def approve_all():
        for location_id in range(1, 21):
            catalog.approve_window(tmp_path, location_id, "K")

    errors, sizes = _read_while_writing(lambda: catalog.load_approved(tmp_path), approve_all)

    assert errors == []
    assert sizes == sorted(sizes), "approvals only grow, so a shrinking read was a torn file"


@pytest.mark.parametrize("through", ["ignore_location", "save_ignore_ids"])
def test_reader_never_sees_a_half_written_ignore_list(tmp_path, monkeypatch, through):
    """An export reading ignore.json during a rejection never gets an emptied list."""
    original = catalog._save_ignore_ids

    def truncate_then_save(crops, ids):
        catalog.ignore_path(crops).write_text("", encoding="utf-8")
        time.sleep(RACE_PAUSE)
        original(crops, ids)

    monkeypatch.setattr(catalog, "_save_ignore_ids", truncate_then_save)

    def reject_all():
        for location_id in range(1, 21):
            if through == "ignore_location":
                catalog.ignore_location(tmp_path, location_id)
            else:
                catalog.save_ignore_ids(tmp_path, range(1, location_id + 1))

    errors, sizes = _read_while_writing(lambda: catalog.load_ignore_ids(tmp_path), reject_all)

    assert errors == []
    assert sizes == sorted(sizes), "rejections only grow, so a shrinking read was a torn file"


def test_concurrent_saves_of_one_file_all_succeed(tmp_path, monkeypatch):
    """Saves of the same file from several threads do not trip over one another's temp file."""
    monkeypatch.setattr(Path, "write_text", _truncate_then_write)
    targets = [lambda i=i: catalog.save_approved(tmp_path, [(i, "K")]) for i in range(1, 7)]
    targets += [lambda i=i: catalog.save_held_review(tmp_path, "RC1", [i], []) for i in range(1, 7)]

    assert _run_together(targets) == []
    assert len(catalog.load_approved(tmp_path)) == 1
    assert len(catalog.load_held_reviews(tmp_path)["RC1"]["rejected"]) == 1


def _write_half_then_fail(path: Path, text: str, encoding: str | None = None, **_kwargs) -> int:
    """Stand in for a process killed halfway through a write."""
    with open(path, "w", encoding=encoding) as handle:
        handle.write(text[: len(text) // 2])
    raise OSError("simulated crash mid-write")


def test_interrupted_approved_save_keeps_the_previous_file(tmp_path, monkeypatch):
    """A crash while saving approved.json leaves the old list readable and no temp file behind."""
    catalog.save_approved(tmp_path, [(1, "A"), (2, "B")])
    before = catalog.approved_path(tmp_path).read_bytes()
    monkeypatch.setattr(Path, "write_text", _write_half_then_fail)

    with pytest.raises(OSError):
        catalog.save_approved(tmp_path, [(1, "A"), (2, "B"), (3, "C")])

    assert catalog.approved_path(tmp_path).read_bytes() == before
    assert sorted(path.name for path in tmp_path.iterdir()) == ["approved.json"]


def test_interrupted_held_review_save_keeps_the_previous_file(tmp_path, monkeypatch):
    """A crash while saving a held review leaves the old document in place."""
    catalog.save_held_review(tmp_path, "RC1", [5, 6], [(7, "K")])
    held = catalog.held_review_dir(tmp_path) / "RC1.json"
    before = held.read_bytes()
    monkeypatch.setattr(Path, "write_text", _write_half_then_fail)

    with pytest.raises(OSError):
        catalog.save_held_review(tmp_path, "RC1", [5, 6, 8], [])

    assert held.read_bytes() == before
    assert catalog.load_held_reviews(tmp_path)["RC1"]["rejected"] == [5, 6]
    assert sorted(path.name for path in held.parent.iterdir()) == ["RC1.json"]


def test_saved_files_keep_their_format(tmp_path):
    """Atomic saves write the same JSON the gallery always wrote."""
    catalog.save_approved(tmp_path, [(2, "B"), (1, "A"), (1, "A")])
    catalog.save_held_review(tmp_path, "RC1", [6, 5, 5], [(7, "K")])

    assert catalog.approved_path(tmp_path).read_text(encoding="utf-8") == json.dumps(
        [{"location_id": 1, "image_key": "A"}, {"location_id": 2, "image_key": "B"}]
    )
    assert (catalog.held_review_dir(tmp_path) / "RC1.json").read_text(encoding="utf-8") == json.dumps(
        {"volume": "RC1", "rejected": [5, 6], "approved": [{"location_id": 7, "image_key": "K"}]}
    )


approval_lists = st.lists(
    st.tuples(st.integers(min_value=1, max_value=10**9), st.text(alphabet="RCX_0123456789-", max_size=24)),
    min_size=1,
    max_size=20,
)


@settings(max_examples=40, deadline=None)
@given(keys=approval_lists, data=st.data())
def test_damaged_approved_list_is_never_overwritten(keys, data):
    """A damaged approved.json (any strict prefix of a saved one) fails loudly and every mutator leaves it as is.

    Treating it as empty would let the next click replace every earlier approval with one window.
    """
    with tempfile.TemporaryDirectory() as folder:
        crops = Path(folder)
        catalog.save_approved(crops, keys)
        text = catalog.approved_path(crops).read_text(encoding="utf-8")
        cut = data.draw(st.integers(min_value=0, max_value=len(text) - 1), label="cut")
        catalog.approved_path(crops).write_text(text[:cut], encoding="utf-8")
        damaged = catalog.approved_path(crops).read_bytes()
        location_id, image_key = keys[0]

        with pytest.raises(json.JSONDecodeError):
            catalog.load_approved(crops)
        for mutate in (
            lambda: catalog.approve_window(crops, location_id, image_key or "K"),
            lambda: catalog.unapprove_window(crops, location_id, image_key or "K"),
            lambda: catalog.review_locations(crops, [location_id], "ignore"),
            lambda: catalog.import_review_lists(crops, [], [(location_id, "K")], replace=False),
        ):
            with pytest.raises(json.JSONDecodeError):
                mutate()
            assert catalog.approved_path(crops).read_bytes() == damaged
        assert sorted(path.name for path in crops.iterdir()) == ["approved.json"]
