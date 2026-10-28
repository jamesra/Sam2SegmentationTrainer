"""Update All paging: which OData pages are read, which rows are handled, progress ticks, cancel, and the state file.

Each test runs the job against a stub ``get_json`` that serves fixed pages and records every request, and stub
``refresh`` / ``add`` / ``drop_stale`` calls that append to one trace. The trace pins the order of page reads, row
handling and stale-thumbnail drops, with the job's progress counters at each drop.
"""

from __future__ import annotations

import json
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

GALLERY_ROOT = Path(__file__).resolve().parents[1]
if str(GALLERY_ROOT) not in sys.path:
    sys.path.insert(0, str(GALLERY_ROOT))

import update_all  # noqa: E402
from catalog import connect  # noqa: E402
from update_all import RETRY_BATCH, Dependencies, UpdateJob, save_state  # noqa: E402

KEY = "RC2_17_D1_X0-1024_Y0-1024"
WATERMARK = "2026-01-01T00:00:00.000000Z"
ODATA = "http://odata.invalid"


class _FixedNow(datetime):
    """``datetime`` whose ``now`` is one fixed moment, so ``last_run`` in update_state.json is reproducible."""

    @classmethod
    def now(cls, tz=None):  # noqa: D102
        return datetime(2026, 10, 19, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _fixed_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(update_all, "datetime", _FixedNow)


def _crops(tmp_path: Path, ids: tuple[int, ...]) -> Path:
    crops = tmp_path / "AnnotationCrops"
    (crops / "masks").mkdir(parents=True)
    connection = connect(crops)
    for location_id in ids:
        connection.execute(
            "INSERT INTO locations (location_id, z, structure_id, structure_label, type_id, type_name, radius, "
            "image_key, image_relpath, mask_relpath, ignored) VALUES (?, 17, 7, 'old', 1, 'Cell', 5.0, ?, ?, ?, 0)",
            [location_id, KEY, f"images/{KEY}.png", f"masks/{KEY}_{location_id}.png"],
        )
    connection.commit()
    connection.close()
    return crops


def _entity(location_id: int, modified: str = "2026-03-01T00:00:00Z") -> dict:
    return {
        "ID": location_id, "Z": 17, "ParentID": 9, "LastModified": modified,
        "Parent": {"ID": 9, "TypeID": 3, "Label": "new", "Type": {"ID": 3, "Name": "Cell"}},
    }


class _Script:
    """Stub OData and row handlers for one job, recording everything they are asked into ``trace``.

    ``update`` is the list of pages the watermark query returns; ``retry`` maps the first id of a retry batch to that
    batch's pages. A page is a response body (``@odata.nextLink`` is added), or an exception the request raises.
    Ids in ``skip`` are not added (tile missing); handling an id in ``cancel_at`` cancels the job afterwards.
    """

    def __init__(
        self,
        update: list[Any],
        retry: dict[int, list[Any]] | None = None,
        skip: frozenset[int] = frozenset(),
        cancel_at: frozenset[int] = frozenset(),
    ) -> None:
        self.update = update
        self.retry = retry or {}
        self.skip = skip
        self.cancel_at = cancel_at
        self.trace: list[tuple] = []
        self.job: UpdateJob | None = None

    def _pages(self, url: str) -> tuple[str, list[Any], int]:
        parsed = urlparse(url)
        if parsed.path.startswith("/next/"):
            _, _, kind, number = parsed.path.split("/")
            pages = self.update if kind == "update" else self.retry[int(kind)]
            return kind, pages, int(number)
        filter_text = parse_qs(parsed.query)["$filter"][0]
        if filter_text.startswith("ID in ("):
            first = filter_text.split("(", 1)[1].split(",")[0].rstrip(")").strip()
            return first, self.retry[int(first)], 0
        return "update", self.update, 0

    def get_json(self, url: str) -> dict:
        kind, pages, number = self._pages(url)
        self.trace.append(("get", kind, number))
        page = pages[number]
        if isinstance(page, Exception):
            raise page
        body = dict(page)
        if number + 1 < len(pages):
            body["@odata.nextLink"] = f"{ODATA}/next/{kind}/{number + 1}"
        return body

    def _after(self, location_id: int) -> None:
        if location_id in self.cancel_at:
            self.job.cancel()

    def refresh(self, crops, entity, exporter) -> list[str]:
        self.trace.append(("refresh", entity["ID"]))
        self._after(entity["ID"])
        return [KEY]

    def add(self, crops, entity, index):
        self.trace.append(("add", entity["ID"]))
        self._after(entity["ID"])
        if entity["ID"] in self.skip:
            return None, "tile image missing"
        record = types.SimpleNamespace(
            id=entity["ID"], z=17, parent_id=9, structure_label="new", type_id=3, type_name="Cell", radius=None
        )
        return {"record": record, "windows": [{"image_key": f"K{entity['ID']}", "area": 100}]}, None

    def drop(self, crops, keys) -> None:
        job = self.job
        self.trace.append(("drop", list(keys), job.page, job.pages, job.entities_seen))

    def run(self, crops: Path, page_size: int = 2) -> dict:
        deps = Dependencies(
            get_json=self.get_json, refresh=self.refresh, add=self.add, drop_stale=self.drop,
            tile_index=lambda crops: None,
        )
        self.job = UpdateJob(volume="paging", crops=crops, odata=ODATA, exporter="tiled", page_size=page_size,
                             deps=deps)
        self.job.start()
        self.job.join(20)
        assert not self.job.running
        return self.job.snapshot()


def _state_text(crops: Path) -> str:
    return (crops / "update_state.json").read_text(encoding="utf-8")


def _page(*entities: dict, count: int | None = None) -> dict:
    body: dict = {"value": list(entities)}
    if count is not None:
        body["@odata.count"] = count
    return body


def test_update_pages_tick_drop_and_write_the_state_file(tmp_path: Path) -> None:
    crops = _crops(tmp_path, ids=(1, 2))
    save_state(crops, {"watermark": WATERMARK})
    script = _Script(
        update=[
            _page(_entity(1, "2026-03-01T00:00:00Z"), _entity(2, "2026-03-02T00:00:00Z"), count=5),
            _page(_entity(3, "2026-03-03T00:00:00Z"), _entity(4, "2026-03-05T00:00:00Z")),
            _page(_entity(5, "2026-03-04T00:00:00Z")),
        ],
        skip=frozenset({4}),
    )
    result = script.run(crops)
    assert script.trace == [
        ("get", "update", 0), ("refresh", 1), ("refresh", 2), ("drop", [KEY], 0, 3, 2),
        ("get", "update", 1), ("add", 3), ("add", 4), ("drop", ["K3"], 1, 3, 4),
        ("get", "update", 2), ("add", 5), ("drop", ["K5"], 2, 3, 5),
    ]
    assert (result["phase"], result["page"], result["pages"], result["entitiesSeen"]) == ("done", 3, 3, 5)
    # Protected format: key order, indent=1 and every value.
    assert _state_text(crops) == (
        '{\n "watermark": "2026-03-05T00:00:00.000000Z",\n "last_run": "2026-10-19T12:00:00+00:00",\n'
        ' "last_counts": {\n  "refreshed": 2,\n  "added": 2,\n  "skipped": 1,\n  "circles": 0,\n'
        '  "unchanged": 0,\n  "failed": 0,\n  "clean": true\n },\n "skipped_ids": [\n  4\n ]\n}'
    )


def test_empty_pages_still_tick_and_rows_without_an_id_are_ignored(tmp_path: Path) -> None:
    crops = _crops(tmp_path, ids=(1,))
    save_state(crops, {"watermark": WATERMARK})
    script = _Script(
        update=[
            {"value": None},                       # no count and no rows: pages is at least one
            _page({"Z": 17}, "not a row", _entity(1)),
            _page(),
        ],
    )
    result = script.run(crops)
    assert script.trace == [
        ("get", "update", 0), ("get", "update", 1), ("refresh", 1), ("drop", [KEY], 1, 1, 1),
        ("get", "update", 2),
    ]
    assert (result["phase"], result["page"], result["pages"], result["entitiesSeen"]) == ("done", 3, 1, 1)
    state = json.loads(_state_text(crops))
    assert state["watermark"] == "2026-03-01T00:00:00.000000Z" and state["skipped_ids"] == []


def test_cancel_mid_page_stops_the_rows_reads_one_more_page_and_keeps_the_watermark(tmp_path: Path) -> None:
    crops = _crops(tmp_path, ids=(1, 2, 3, 4, 5))
    save_state(crops, {"watermark": WATERMARK, "skipped_ids": []})
    script = _Script(
        update=[_page(_entity(1), _entity(2), count=6), _page(_entity(3), _entity(4)), _page(_entity(5))],
        cancel_at=frozenset({3}),
    )
    result = script.run(crops)
    assert script.trace == [
        ("get", "update", 0), ("refresh", 1), ("refresh", 2), ("drop", [KEY], 0, 3, 2),
        ("get", "update", 1), ("refresh", 3), ("drop", [KEY], 1, 3, 3),
        ("get", "update", 2),                      # read, then the cancel check stops before its rows
    ]
    assert (result["phase"], result["page"], result["entitiesSeen"], result["cancelled"]) == ("cancelled", 2, 3, True)
    assert _state_text(crops) == (
        '{\n "watermark": "2026-01-01T00:00:00.000000Z",\n "last_run": "2026-10-19T12:00:00+00:00",\n'
        ' "last_counts": {\n  "refreshed": 3,\n  "added": 0,\n  "skipped": 0,\n  "circles": 0,\n'
        '  "unchanged": 0,\n  "failed": 0,\n  "clean": false\n },\n "skipped_ids": []\n}'
    )


def test_retry_batches_page_and_a_cancel_stops_the_rest(tmp_path: Path) -> None:
    previous = list(range(100, 100 + RETRY_BATCH + 2))      # two batches: RETRY_BATCH ids, then two
    crops = _crops(tmp_path, ids=(1,))
    save_state(crops, {"watermark": WATERMARK, "skipped_ids": previous})
    script = _Script(
        update=[_page(_entity(1), count=1)],
        retry={
            100: [_page(_entity(100), _entity(101), _entity(102)), _page(_entity(103))],
            100 + RETRY_BATCH: [_page(_entity(100 + RETRY_BATCH))],
        },
        skip=frozenset({100}),
        cancel_at=frozenset({101}),
    )
    result = script.run(crops)
    assert script.trace == [
        ("get", "100", 0), ("add", 100), ("add", 101), ("drop", ["K101"], 0, 0, 0),
        ("get", "100", 1),                         # the batch's next page is still read, but none of its rows
        ("get", "update", 0),                      # the second batch is not asked for; the update page stops
    ]
    assert (result["phase"], result["page"], result["pages"], result["entitiesSeen"]) == ("cancelled", 0, 0, 0)
    assert (result["added"], result["skipped"]) == (1, 1)
    state = json.loads(_state_text(crops))
    assert state["watermark"] == WATERMARK
    # A cancelled run keeps the whole earlier list except the id it added.
    assert state["skipped_ids"] == [item for item in previous if item != 101]


def test_retry_rows_do_not_move_the_watermark_or_tick(tmp_path: Path) -> None:
    crops = _crops(tmp_path, ids=(1,))
    save_state(crops, {"watermark": WATERMARK, "skipped_ids": [7, 8]})
    script = _Script(
        update=[_page(count=0)],
        retry={7: [_page(_entity(7, "2026-05-01T00:00:00Z")), _page(_entity(8, "2026-05-02T00:00:00Z"))]},
        skip=frozenset({8}),
    )
    result = script.run(crops)
    assert script.trace == [
        ("get", "7", 0), ("add", 7), ("drop", ["K7"], 0, 0, 0), ("get", "7", 1), ("add", 8),
        ("get", "update", 0),
    ]
    assert (result["phase"], result["page"], result["pages"], result["entitiesSeen"]) == ("done", 1, 1, 0)
    state = json.loads(_state_text(crops))
    assert state["watermark"] == WATERMARK and state["skipped_ids"] == [8]


@pytest.mark.parametrize("phase", ["update", "retry"])
def test_a_page_that_raises_stops_the_job_and_leaves_the_state_file(tmp_path: Path, phase: str) -> None:
    crops = _crops(tmp_path, ids=(1, 2))
    skipped = [9] if phase == "retry" else []
    save_state(crops, {"watermark": WATERMARK, "skipped_ids": skipped})
    before = _state_text(crops)
    failing = [_page(_entity(1), count=2), OSError("OData went away")]
    script = _Script(
        update=failing if phase == "update" else [_page(_entity(2), count=1)],
        retry={9: [_page(_entity(1)), OSError("OData went away")]},
    )
    result = script.run(crops, page_size=1)
    # The update phase has counted its pages and the first row by the time it drops; the retry phase never ticks.
    kind, pages, seen = ("update", 2, 1) if phase == "update" else ("9", 0, 0)
    assert script.trace == [("get", kind, 0), ("refresh", 1), ("drop", [KEY], 0, pages, seen), ("get", kind, 1)]
    assert result["phase"] == "failed" and result["failed"] == 1
    assert result["errors"] == ["job stopped: OData went away"]
    assert _state_text(crops) == before
