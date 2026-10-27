"""Card scores in the gallery: the catalog's per-card IoU, core recall and spill (``annotation_crops.card_scores``),
the old and new score after a re-score, the sort keys, and queueing a location for the scorer service.

The gallery sorts the whole catalog in the browser (``compareSort`` in app.js), so the sort tests run that code in node
(skipped without node); the server's part is the fields it sorts on, checked against ``MISMATCH_SQL``.
"""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
import subprocess
from pathlib import Path

import pytest

from tests.test_gallery import _request, httpd, registry  # noqa: F401  (fixtures)

import catalog
from annotation_crops.card_scores import (
    DEFAULT_FAMILY,
    MISMATCH_SQL,
    TRAIN_FAMILY,
    CardScore,
    beat,
    claim_requests,
    ensure_card_scores_schema,
    finish_request,
    write_card_scores,
)

APP_JS = Path(__file__).resolve().parents[1] / "static" / "app.js"
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")

KEY = "RC2_17_D1_X0_Y0"  # location 42's card in the gallery fixture


def _crops(registry: Path) -> Path:
    return registry / "RC2" / "AnnotationCrops"


def _add_location(crops: Path, location_id: int, image_key: str, z: int = 17) -> None:
    connection = catalog.connect(crops)
    try:
        connection.execute(
            "INSERT INTO locations (location_id, z, structure_id, type_name, image_key, image_relpath, mask_relpath) "
            "VALUES (?, ?, 7, 'Cell', ?, ?, ?)",
            [location_id, z, image_key, f"images/{image_key}.png", f"masks/{image_key}_{location_id}.png"],
        )
        connection.commit()
    finally:
        connection.close()


def _score(location_id: int, image_key: str, scored_at: str, iou, core, spill, model: str = "v5.4") -> CardScore:
    return CardScore(
        location_id=location_id, image_key=image_key, family=DEFAULT_FAMILY, model=model, scored_at=scored_at,
        iou=iou, raw_iou=iou, core_recall=core, spill=spill, bf10=None, core_gt=None, gt_area=1000,
    )


def _write(crops: Path, *scores: CardScore) -> None:
    connection = sqlite3.connect(catalog.sqlite_path(crops))
    try:
        ensure_card_scores_schema(connection)
        write_card_scores(connection, scores)
        connection.commit()
    finally:
        connection.close()


def _scorer(crops: Path) -> sqlite3.Connection:
    """A connection as the scorer service opens it (autocommit, for claim_requests)."""
    return sqlite3.connect(catalog.sqlite_path(crops), isolation_level=None)


def _beat(crops: Path) -> None:
    """A scorer service serving this catalog (its heartbeat), so the gallery queues re-scores."""
    connection = sqlite3.connect(catalog.sqlite_path(crops))
    try:
        ensure_card_scores_schema(connection)
        beat(connection, spec="best:sam2_em_v5", model="v5.4")
        connection.commit()
    finally:
        connection.close()


@pytest.fixture
def crops(registry: Path) -> Path:  # noqa: F811
    path = _crops(registry)
    _beat(path)
    return path


def _catalog_rows(httpd, bearer: str = "read") -> dict[int, dict]:  # noqa: F811
    status, payload = _request(httpd, "GET", "/api/volumes/RC2/catalog", bearer=bearer)
    assert status == 200
    return {row["location_id"]: row for row in payload["rows"]}


def _post(httpd, path: str, body: dict, bearer: str = "review"):  # noqa: F811
    return _request(httpd, "POST", path, body=json.dumps(body).encode("utf-8"), bearer=bearer)


# ---- the catalog: newest score per card, the one before it, unscored cards without fields ------------------------

def test_the_catalog_carries_each_cards_newest_score_and_the_one_before(httpd, crops: Path) -> None:  # noqa: F811
    _add_location(crops, 43, "RC2_18_D1_X0_Y0")
    _add_location(crops, 44, "RC2_19_D1_X0_Y0")
    _write(
        crops,
        _score(42, KEY, "2026-10-25T10:00:00Z", 0.82, 0.90, 0.10, model="v5.3"),
        _score(42, KEY, "2026-10-26T10:00:00Z", 0.91, 0.97, 0.03),
        _score(43, "RC2_18_D1_X0_Y0", "2026-10-26T10:00:00Z", 0.40, None, 1.5),
    )

    rows = _catalog_rows(httpd)

    rescored = rows[42]
    assert rescored["cardIou"] == pytest.approx(0.91) and rescored["cardCore"] == pytest.approx(0.97)
    assert rescored["cardSpill"] == pytest.approx(0.03) and rescored["cardModel"] == "v5.4"
    assert rescored["cardScoredAt"] == "2026-10-26T10:00:00Z"
    assert rescored["cardMismatch"] == pytest.approx(0.03), "max(1 - 0.97, 0.03)"
    old = rescored["cardOld"]
    assert old["iou"] == pytest.approx(0.82) and old["core"] == pytest.approx(0.90) and old["spill"] == pytest.approx(0.10)
    assert old["model"] == "v5.3" and old["scoredAt"] == "2026-10-25T10:00:00Z"
    assert old["mismatch"] == pytest.approx(0.10)

    thin = rows[43]
    assert thin["cardOld"] is None, "scored once: nothing to compare with"
    assert thin["cardCore"] is None, "a thin annotation has no core"
    assert thin["cardSpill"] == pytest.approx(1.5) and thin["cardMismatch"] == pytest.approx(1.5), "spill can exceed 1"

    assert not any(key in rows[44] for key in catalog.CARD_SCORE_FIELDS), "an unscored card has no card fields"
    assert rows[44]["loss"] is None and "cardRescore" not in rows[44]


def test_the_training_loss_is_still_in_the_catalog(httpd, crops: Path) -> None:  # noqa: F811
    connection = catalog.connect(crops)
    connection.execute("INSERT INTO location_scores (location_id, image_key, epoch, score) VALUES (42, ?, 3, 0.25)", [KEY])
    connection.commit()
    connection.close()
    _write(crops, _score(42, KEY, "2026-10-26T10:00:00Z", 0.91, 0.97, 0.03))

    row = _catalog_rows(httpd)[42]

    assert row["loss"] == 0.25 and row["cardIou"] == pytest.approx(0.91)


def test_the_catalog_mismatch_orders_cards_as_mismatch_sql_does(httpd, crops: Path) -> None:  # noqa: F811
    cards = {43: (0.95, 0.99, 0.02), 44: (0.50, 0.40, 0.05), 45: (0.60, 0.98, 0.70), 46: (0.70, None, 0.20)}
    for location_id in cards:
        _add_location(crops, location_id, f"RC2_{location_id}_D1_X0_Y0")
    _add_location(crops, 47, "RC2_47_D1_X0_Y0")  # never scored
    _write(crops, *(_score(i, f"RC2_{i}_D1_X0_Y0", "2026-10-26T10:00:00Z", *v) for i, v in cards.items()))

    rows = _catalog_rows(httpd)

    scored = sorted((row for row in rows.values() if "cardMismatch" in row), key=lambda row: -row["cardMismatch"])
    connection = sqlite3.connect(catalog.sqlite_path(crops))
    by_sql = [
        row[0] for row in connection.execute(
            f"SELECT s.location_id FROM card_scores AS s ORDER BY {MISMATCH_SQL.format(p='s.')} DESC"
        )
    ]
    connection.close()
    assert [row["location_id"] for row in scored] == by_sql == [45, 44, 46, 43]
    assert rows[44]["cardMismatch"] == pytest.approx(0.60), "a big core miss"
    assert rows[45]["cardMismatch"] == pytest.approx(0.70), "a big spill"
    assert rows[46]["cardMismatch"] == pytest.approx(0.20), "no core: only the spill counts"


# ---- the card-scores endpoint ---------------------------------------------------------------------------------------

def test_card_scores_pairs_old_and_new_for_the_asked_locations(httpd, crops: Path) -> None:  # noqa: F811
    _add_location(crops, 43, "RC2_18_D1_X0_Y0")
    _write(
        crops,
        _score(42, KEY, "2026-10-25T10:00:00Z", 0.82, 0.90, 0.10),
        _score(42, KEY, "2026-10-26T10:00:00Z", 0.91, 0.97, 0.03),
        _score(42, KEY, "2026-10-24T10:00:00Z", 0.10, 0.10, 0.90),  # older than both: not shown
        _score(43, "RC2_18_D1_X0_Y0", "2026-10-26T10:00:00Z", 0.40, 0.50, 0.20),
    )

    status, payload = _request(httpd, "GET", "/api/volumes/RC2/card-scores?ids=42", bearer="read")

    assert status == 200
    (card,) = payload["cards"]
    assert card["location_id"] == 42 and card["image_key"] == KEY
    assert card["cardIou"] == pytest.approx(0.91) and card["cardOld"]["iou"] == pytest.approx(0.82)

    status, payload = _request(httpd, "GET", "/api/volumes/RC2/card-scores?ids=42,43,99", bearer="read")
    assert status == 200 and [card["location_id"] for card in payload["cards"]] == [42, 43]


@pytest.mark.parametrize("query", ["", "?ids=", "?ids=abc", "?ids=" + ",".join(str(i) for i in range(1001))])
def test_card_scores_and_status_need_location_ids(httpd, query: str) -> None:  # noqa: F811
    for what in ("card-scores", "rescore-status"):
        status, payload = _request(httpd, "GET", f"/api/volumes/RC2/{what}{query}", bearer="read")
        assert status == 400 and payload["error"]


def test_card_score_routes_of_an_unknown_volume_are_404(httpd) -> None:  # noqa: F811
    assert _request(httpd, "GET", "/api/volumes/NOPE/card-scores?ids=42", bearer="review")[0] == 404
    assert _request(httpd, "GET", "/api/volumes/NOPE/rescore-status?ids=42", bearer="review")[0] == 404
    assert _post(httpd, "/api/volumes/NOPE/rescore", {"location_id": 42})[0] == 404


# ---- queueing: after a refresh, or on its own --------------------------------------------------------------------

@pytest.fixture
def refreshable(registry: Path, monkeypatch: pytest.MonkeyPatch) -> list[int]:  # noqa: F811
    """A refresh that redraws without OData or nornir, and no SAM2 scoring (score.py has its own tests)."""
    fetched: list[int] = []

    def fake_fetch(url: str, location_id: int) -> dict:
        fetched.append(location_id)
        return {"ID": location_id, "MosaicShape": {}, "LastModified": "2022-01-01T00:00:00Z"}

    monkeypatch.setattr("server.nornir_buildmanager_root", lambda: registry)
    monkeypatch.setattr("server.fetch_odata_location", fake_fetch)
    monkeypatch.setattr("server.refresh_location", lambda *_args, **_kwargs: [KEY])
    monkeypatch.setenv("GALLERY_SCORE_ON_REFRESH", "0")
    return fetched


def _requests(crops: Path) -> list[tuple]:
    connection = sqlite3.connect(catalog.sqlite_path(crops))
    try:
        ensure_card_scores_schema(connection)
        return connection.execute("SELECT id, location_id, state FROM card_score_requests ORDER BY id").fetchall()
    finally:
        connection.close()


def test_a_refresh_queues_the_location_for_re_scoring(httpd, crops: Path, refreshable) -> None:  # noqa: F811
    status, payload = _post(httpd, "/api/volumes/RC2/refresh-location", {"location_id": 42})

    assert status == 200 and payload["imageKeys"] == [KEY] and payload["score"] is None
    request = payload["rescore"]["request"]
    assert request["state"] == "pending" and request["requested_at"]
    assert _requests(crops) == [(request["id"], 42, "pending")]

    # A second refresh while the first request waits does not queue the location twice.
    status, again = _post(httpd, "/api/volumes/RC2/refresh-location", {"location_id": 42})
    assert status == 200 and again["rescore"]["request"]["id"] == request["id"]
    assert len(_requests(crops)) == 1


def test_a_failed_refresh_queues_nothing(httpd, crops: Path, refreshable, monkeypatch) -> None:  # noqa: F811
    def down(url: str, location_id: int) -> dict:
        raise OSError("OData is down")

    monkeypatch.setattr("server.fetch_odata_location", down)
    status, payload = _post(httpd, "/api/volumes/RC2/refresh-location", {"location_id": 42})

    assert status == 502 and "OData" in payload["error"]
    assert _requests(crops) == []


def test_re_scoring_on_refresh_can_be_turned_off(httpd, crops: Path, refreshable, monkeypatch) -> None:  # noqa: F811
    monkeypatch.setenv("GALLERY_RESCORE_ON_REFRESH", "0")
    status, payload = _post(httpd, "/api/volumes/RC2/refresh-location", {"location_id": 42})

    assert status == 200 and payload["rescore"] is None and _requests(crops) == []


def test_rescore_queues_without_a_redraw(httpd, crops: Path, refreshable) -> None:  # noqa: F811
    status, payload = _post(httpd, "/api/volumes/RC2/rescore", {"location_id": 42})

    assert status == 200 and payload["ok"] and payload["request"]["state"] == "pending"
    assert refreshable == [], "no OData lookup, no redraw"
    assert _requests(crops) == [(payload["request"]["id"], 42, "pending")]


def test_rescore_needs_review_a_location_id_and_a_known_location(httpd, crops: Path) -> None:  # noqa: F811
    assert _post(httpd, "/api/volumes/RC2/rescore", {"location_id": 42}, bearer="read") == (
        403, {"error": "review required"}
    )
    assert _post(httpd, "/api/volumes/RC2/rescore", {})[0] == 400
    status, payload = _post(httpd, "/api/volumes/RC2/rescore", {"location_id": 99})
    assert status == 404 and "99" in payload["error"]
    assert _requests(crops) == []


# ---- the status endpoint, through the scorer's states ---------------------------------------------------------------

def _status(httpd, ids: str) -> dict:  # noqa: F811
    status, payload = _request(httpd, "GET", f"/api/volumes/RC2/rescore-status?ids={ids}", bearer="read")
    assert status == 200
    return payload["status"]


def test_rescore_status_follows_the_request_from_pending_to_done(httpd, crops: Path) -> None:  # noqa: F811
    assert _status(httpd, "42") == {"42": None}, "never queued"

    request_id = _post(httpd, "/api/volumes/RC2/rescore", {"location_id": 42})[1]["request"]["id"]
    assert _status(httpd, "42")["42"]["state"] == "pending"
    assert _catalog_rows(httpd)[42]["cardRescore"] == {"id": request_id, "state": "pending"}

    scorer = _scorer(crops)
    assert claim_requests(scorer) == [(request_id, 42)]
    assert _status(httpd, "42")["42"]["state"] == "running"
    assert _catalog_rows(httpd)[42]["cardRescore"]["state"] == "running"

    write_card_scores(scorer, [_score(42, KEY, "2026-10-26T12:00:00Z", 0.91, 0.97, 0.03)])
    finish_request(scorer, request_id, model="v5.4", cards=1)
    scorer.close()

    done = _status(httpd, "42,43")
    assert done["42"]["state"] == "done" and done["42"]["cards"] == 1 and done["42"]["model"] == "v5.4"
    assert done["43"] is None
    row = _catalog_rows(httpd)[42]
    assert "cardRescore" not in row and row["cardIou"] == pytest.approx(0.91)


def test_a_failed_rescore_reports_its_error(httpd, crops: Path) -> None:  # noqa: F811
    request_id = _post(httpd, "/api/volumes/RC2/rescore", {"location_id": 42})[1]["request"]["id"]
    scorer = _scorer(crops)
    claim_requests(scorer)
    finish_request(scorer, request_id, model="v5.4", cards=0, error="CUDA out of memory")
    scorer.close()

    failed = _status(httpd, "42")["42"]
    assert failed["state"] == "failed" and failed["error"] == "CUDA out of memory"
    assert "cardRescore" not in _catalog_rows(httpd)[42], "only pending and running show as re-scoring"


def test_the_newest_request_decides_the_status(httpd, crops: Path) -> None:  # noqa: F811
    first = _post(httpd, "/api/volumes/RC2/rescore", {"location_id": 42})[1]["request"]["id"]
    scorer = _scorer(crops)
    claim_requests(scorer)
    finish_request(scorer, first, model="v5.4", cards=0, error="boom")
    scorer.close()

    second = _post(httpd, "/api/volumes/RC2/rescore", {"location_id": 42})[1]["request"]

    assert second["id"] != first and second["state"] == "pending"
    assert _status(httpd, "42")["42"]["id"] == second["id"]
    assert _catalog_rows(httpd)[42]["cardRescore"]["id"] == second["id"]


# ---- app.js: the sort and the badge (node) --------------------------------------------------------------------------

def _js_functions(*names: str) -> str:
    source = APP_JS.read_text(encoding="utf-8")
    parts = []
    for name in names:
        match = re.search(rf"  (?:async )?function {name}\(.*?\n  \}}\n", source, re.S)
        assert match, f"{name} not found in app.js"
        parts.append(match.group(0))
    for name in ("CARD_SORT_WORST_DESC", "TRAIN_SCORE_NOTE", "TREND_SERIES", "TREND_ICON"):
        match = re.search(rf"  const {name} = .*?;\n", source, re.S)
        assert match, f"{name} not found in app.js"
        parts.append(match.group(0))
    return "\n".join(parts)


def _run_node(script: str):
    result = subprocess.run([NODE, "-e", script], capture_output=True, text=True, timeout=60, check=False,
                            encoding="utf-8")
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


_SORT_ROWS = [
    {"location_id": 1, "cardIou": 0.95, "cardCore": 0.99, "cardSpill": 0.02, "cardMismatch": 0.02},
    {"location_id": 2, "cardIou": 0.50, "cardCore": 0.40, "cardSpill": 0.05, "cardMismatch": 0.60},
    {"location_id": 3},
    {"location_id": 4, "cardIou": 0.60, "cardCore": 0.98, "cardSpill": 0.70, "cardMismatch": 0.70},
    {"location_id": 5, "cardIou": 0.70, "cardCore": None, "cardSpill": 0.20, "cardMismatch": 0.20},
]


def _sorted_ids(key: str, desc: bool) -> list[int]:
    script = f"""
const vm = require("vm");
const state = {{ sorts: [{{ key: {json.dumps(key)}, desc: {json.dumps(desc)} }}] }};
const context = {{ state }};
vm.createContext(context);
vm.runInContext({json.dumps(_js_functions("compareValues", "compareSort"))} + "; this.compareSort = compareSort;",
                context);
const rows = {json.dumps(_SORT_ROWS)};
console.log(JSON.stringify(rows.slice().sort(context.compareSort).map((row) => row.location_id)));
"""
    return _run_node(script)


@needs_node
def test_model_mismatch_sorts_worst_first_and_unscored_last() -> None:
    assert _sorted_ids("cardMismatch", True) == [4, 2, 5, 1, 3]
    assert _sorted_ids("cardMismatch", False) == [1, 5, 2, 4, 3], "unscored last in either direction"


@needs_node
def test_iou_core_and_spill_sort_with_unscored_and_coreless_last() -> None:
    assert _sorted_ids("cardIou", False) == [2, 4, 5, 1, 3]
    assert _sorted_ids("cardCore", False) == [2, 4, 1, 3, 5], "no core sorts with the unscored"
    assert _sorted_ids("cardSpill", True) == [4, 5, 2, 1, 3]


def _badges(rows: list[dict], marks: dict, permission: str = "read") -> list[str]:
    script = f"""
const vm = require("vm");
const state = {{ converter: false, permission: {json.dumps(permission)} }};
const rescoreMarks = new Map(Object.entries({json.dumps(marks)}).map(([id, mark]) => [Number(id), mark]));
const context = {{ state, rescoreMarks }};
vm.createContext(context);
vm.runInContext({json.dumps(_js_functions("attrText", "scoreText", "scorePair", "cardScoreTitle", "cardScoreBadge"))}
                + "; this.cardScoreBadge = cardScoreBadge;", context);
console.log(JSON.stringify({json.dumps(rows)}.map((row) => context.cardScoreBadge(row))));
"""
    return _run_node(script)


@needs_node
def test_the_badge_shows_three_numbers_iou_alone_when_narrow_and_old_beside_new() -> None:
    once = {"location_id": 1, "cardIou": 0.91, "cardCore": 0.97, "cardSpill": 0.03, "cardMismatch": 0.03,
            "cardOld": None, "loss": 0.123}
    rescored = dict(once, location_id=2, cardOld={"iou": 0.82, "core": 0.9, "spill": 0.1, "mismatch": 0.1})
    coreless = dict(once, location_id=3, cardCore=None)
    unscored_with_loss = {"location_id": 4, "loss": 0.2}
    nothing = {"location_id": 5}

    badges = _badges([once, rescored, coreless, unscored_with_loss, nothing], {})

    assert '<span class="cs-full">IoU 0.91 · core 0.97 · spill 0.03</span><span class="cs-short">IoU 0.91</span>' in badges[0]
    assert "training loss 0.123" in badges[0] and "loss" not in badges[0].split('title="')[0]
    assert "IoU 0.82→0.91 · core 0.90→0.97 · spill 0.10→0.03" in badges[1]
    assert '<span class="cs-short">IoU 0.82→0.91</span>' in badges[1]
    assert "IoU 0.91 · core – · spill 0.03" in badges[2] and "too thin to have a core" in badges[2]
    assert "unscored" in badges[3] and '<span class="cs-full">–</span>' in badges[3]
    assert badges[4] == ""


@needs_node
def test_the_badge_says_re_scoring_and_shows_a_failure_in_its_tooltip() -> None:
    row = {"location_id": 7, "cardIou": 0.91, "cardCore": 0.97, "cardSpill": 0.03, "cardMismatch": 0.03}
    pending, failed = _badges(
        [row, dict(row, location_id=8)],
        {"7": {"state": "pending", "error": ""}, "8": {"state": "failed", "error": 'CUDA "out" of memory'}},
        permission="review",
    )

    assert '<span class="cs-full">re-scoring…</span>' in pending and "rescoring" in pending
    assert 'data-act="rescore"' in pending, "a reviewer can click it to re-score"
    assert "failed" in failed and "Re-score failed: CUDA &quot;out&quot; of memory" in failed
    assert "IoU 0.91 · core 0.97 · spill 0.03" in failed, "the scores stay; the failure is in the tooltip"


# ---- training-pass scores and the trend graph -------------------------------------------------------------------

def _train_score(location_id: int, image_key: str, scored_at: str, iou, core, spill, epoch: int) -> CardScore:
    return CardScore(
        location_id=location_id, image_key=image_key, family=TRAIN_FAMILY, model=f"v5.{epoch}", scored_at=scored_at,
        iou=iou, raw_iou=iou, core_recall=core, spill=spill, bf10=None, core_gt=None, gt_area=1000, epoch=epoch,
        prompt="circle_mask,point",
    )


def test_a_validation_score_wins_over_training_pass_scores(httpd, crops: Path) -> None:  # noqa: F811
    _add_location(crops, 43, "RC2_18_D1_X0_Y0")
    _write(
        crops,
        _score(42, KEY, "2026-10-25T10:00:00Z", 0.91, 0.97, 0.03),
        _train_score(42, KEY, "2026-10-26T10:00:00Z", 0.70, 0.80, 0.20, epoch=6),
        _train_score(43, "RC2_18_D1_X0_Y0", "2026-10-25T10:00:00Z", 0.60, 0.70, 0.10, epoch=5),
        _train_score(43, "RC2_18_D1_X0_Y0", "2026-10-26T10:00:00Z", 0.65, 0.75, 0.05, epoch=6),
    )

    rows = _catalog_rows(httpd)

    assert rows[42]["cardKind"] == "validation" and rows[42]["cardIou"] == pytest.approx(0.91)
    assert rows[43]["cardKind"] == "train" and rows[43]["cardEpoch"] == 6
    assert rows[43]["cardOld"]["iou"] == pytest.approx(0.60), "the previous epoch's training-pass score"


def test_card_history_returns_both_kinds_oldest_first(httpd, crops: Path) -> None:  # noqa: F811
    _write(
        crops,
        _train_score(42, KEY, "2026-10-25T10:00:00Z", 0.60, 0.70, 0.10, epoch=5),
        _train_score(42, KEY, "2026-10-26T10:00:00Z", 0.65, 0.75, 0.05, epoch=6),
        _score(42, KEY, "2026-10-26T11:00:00Z", 0.91, 0.97, 0.03),
    )

    status, payload = _request(httpd, "GET", f"/api/volumes/RC2/card-history?id=42&key={KEY}", bearer="read")

    assert status == 200
    assert [p["epoch"] for p in payload["train"]] == [5, 6]
    assert len(payload["validation"]) == 1 and payload["validation"][0]["iou"] == pytest.approx(0.91)
    status, _payload = _request(httpd, "GET", "/api/volumes/RC2/card-history?id=x", bearer="read")
    assert status == 400


@needs_node
def test_a_training_pass_score_is_tagged_train_with_a_trend_button() -> None:
    row = {"location_id": 9, "image_key": "k", "cardIou": 0.7, "cardCore": 0.8, "cardSpill": 0.2, "cardMismatch": 0.2,
           "cardKind": "train", "cardEpoch": 6, "cardModel": "v5.6"}
    script = f"""
const vm = require("vm");
const state = {{ converter: false, permission: "read" }};
const context = {{ state, rescoreMarks: new Map() }};
vm.createContext(context);
vm.runInContext({json.dumps(_js_functions("attrText", "scoreText", "scorePair", "cardScoreTitle", "cardScoreBadge",
                                          "trendButton"))} + "; this.cardScoreBadge = cardScoreBadge; this.trendButton = trendButton;",
                context);
const row = {json.dumps(row)};
console.log(JSON.stringify([context.cardScoreBadge(row), context.trendButton(row), context.trendButton({{ location_id: 1 }})]));
"""
    badge, trend, none = _run_node(script)
    assert 'class="cs-tag">train</span>' in badge and "card-score train" in badge
    assert "Training-pass score" in badge and "not comparable with validation scores" in badge
    assert 'data-act="trend"' in trend and 'data-id="9"' in trend and 'data-key="k"' in trend
    assert none == "", "an unscored card has no trend button"


@needs_node
def test_the_trend_chart_draws_validation_solid_and_training_pass_dashed() -> None:
    history = {
        "validation": [{"epoch": 4, "iou": 0.9, "core_recall": 0.97, "spill": 0.03, "model": "v5.4"}],
        "train": [{"epoch": 5, "iou": 0.6, "core_recall": None, "spill": 1.5, "model": "v5.5"},
                  {"epoch": 6, "iou": 0.7, "core_recall": 0.8, "spill": 0.1, "model": "v5.6"}],
        "loss": [],
    }
    script = f"""
const vm = require("vm");
const context = {{}};
vm.createContext(context);
vm.runInContext({json.dumps(_js_functions("attrText", "trendPoints", "trendSvg"))} + "; this.trendSvg = trendSvg;", context);
console.log(JSON.stringify([context.trendSvg({json.dumps(history)}), context.trendSvg({{ validation: [], train: [] }})]));
"""
    svg, empty = _run_node(script)
    assert svg.startswith('<svg class="trend-chart"')
    assert svg.count("<polyline") == 6, "validation: IoU, core, spill; training pass: IoU, core (one point), spill"
    assert svg.count('stroke-dasharray="3 3"') == 3, "the training-pass lines are dashed"
    assert "spill 1.500 · train" in svg, "spill above 1 keeps its value in the tooltip"
    assert "No scores yet" in empty


def _stop_scorer(crops: Path) -> None:
    connection = sqlite3.connect(catalog.sqlite_path(crops))
    try:
        connection.execute("UPDATE card_scorer_heartbeat SET seen_at = '2026-01-01T00:00:00Z'")
        connection.commit()
    finally:
        connection.close()


def test_nothing_is_queued_where_no_scorer_runs(httpd, crops: Path, refreshable) -> None:  # noqa: F811
    _stop_scorer(crops)

    status, payload = _post(httpd, "/api/volumes/RC2/rescore", {"location_id": 42})
    assert status == 503 and "no card scorer" in payload["error"]

    connection = sqlite3.connect(catalog.sqlite_path(crops))
    try:
        assert connection.execute("SELECT COUNT(*) FROM card_score_requests").fetchone()[0] == 0
    finally:
        connection.close()
