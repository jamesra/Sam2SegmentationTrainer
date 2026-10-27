"""Per-card scores (``card_scores``) and the re-score queue (``card_score_requests``).

A card is one crop window of a location, ``(location_id, image_key)``. Every score carries the run's own measures:

* ``iou``: the tolerant IoU the run reports as IoU (pixels in the traced-edge band count for neither side);
* ``core_recall``: the share of the annotation's core (at least ``val.core_band_px`` inside its edge) predicted;
  None when the annotation is too thin to have a core;
* ``spill``: predicted area more than the band outside the annotation, as a share of the annotation's area.

``mismatch`` is the worse of the two problems, ``max(1 - core_recall, spill)``: a card that misses its core and a
card that spills both rise to the top of one view.

Scores come in two kinds (``kind``):

* **validation** (``family`` is a validation family, ``circle_annotator`` by default): the checkpoint in eval mode,
  no augmentation, the family's fixed prompt; exactly a training run's validation of that sample. Written by the
  run's own validation at the end of each epoch (its validation cards) and by ``sam2-em-score-cards`` (any card,
  e.g. the gallery's re-score).
* **train** (``family == "train"``): measured on the training pass as the card is trained: augmented image, the
  prompt types drawn for it that step (``prompt``, pooled over them), weights part-way through epoch ``epoch``.
  Fresh every epoch for every training card, but noisier and not comparable with validation scores.

Every score is kept, so a card shows its newest score next to the one before (``latest_card_scores``) and its
trend across epochs (``card_score_history``). The gallery has no GPU: it queues a location in
``card_score_requests`` and the scorer service (``sam2-em-score-cards --serve``) scores it. Only sqlite is needed
here, so the gallery imports this module too.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

TRAIN_FAMILY = "train"
DEFAULT_FAMILY = "circle_annotator"

_CARD_SCORES_SQL = """
CREATE TABLE IF NOT EXISTS card_scores (
    location_id INTEGER NOT NULL,
    image_key TEXT NOT NULL,
    family TEXT NOT NULL,
    model TEXT NOT NULL,
    scored_at TEXT NOT NULL,
    iou REAL,
    raw_iou REAL,
    core_recall REAL,
    spill REAL,
    bf10 REAL,
    core_gt INTEGER,
    gt_area INTEGER,
    epoch INTEGER,
    prompt TEXT,
    PRIMARY KEY (location_id, image_key, family, scored_at, model)
)
"""

_ADDED_COLUMNS = (("epoch", "INTEGER"), ("prompt", "TEXT"))
"""Columns added after the table first shipped (``ensure_card_scores_schema`` adds them to an older table)."""

_CARD_INDEX_SQL = "CREATE INDEX IF NOT EXISTS card_scores_card ON card_scores (location_id, image_key, scored_at)"

MISMATCH_SQL = "MAX(1.0 - COALESCE({p}core_recall, 1.0), COALESCE({p}spill, 0.0))"
"""SQL for :func:`mismatch` over a row (``{p}`` is a column prefix such as ``"s."``), for sorting in sqlite."""

CURRENT_VIEW = "card_scores_current"
"""One row per scored card, the score a card shows: its newest validation score, else its newest training-pass
score. Columns: ``location_id, image_key, kind ('validation'|'train'), family, model, epoch, scored_at, iou,
core_recall, spill, mismatch``. Join it to ``locations`` on ``(location_id, image_key)`` to sort cards."""

_CURRENT_VIEW_SQL = f"""
CREATE VIEW IF NOT EXISTS {CURRENT_VIEW} AS
SELECT location_id, image_key, kind, family, model, epoch, scored_at, iou, core_recall, spill,
       {MISMATCH_SQL.format(p="")} AS mismatch
FROM (
    SELECT *, CASE WHEN family = '{TRAIN_FAMILY}' THEN 'train' ELSE 'validation' END AS kind,
           ROW_NUMBER() OVER (
               PARTITION BY location_id, image_key
               ORDER BY (family = '{TRAIN_FAMILY}') ASC, scored_at DESC, rowid DESC
           ) AS rank
    FROM card_scores
)
WHERE rank = 1
"""

_REQUESTS_SQL = """
CREATE TABLE IF NOT EXISTS card_score_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    location_id INTEGER NOT NULL,
    requested_at TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'pending',
    started_at TEXT,
    finished_at TEXT,
    model TEXT,
    cards INTEGER,
    error TEXT
)
"""

_REQUESTS_INDEX_SQL = "CREATE INDEX IF NOT EXISTS card_score_requests_state ON card_score_requests (state, id)"


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_card_scores_schema(connection: sqlite3.Connection) -> None:
    """Create ``card_scores`` (adding columns an older table lacks), its view and the request queue."""
    connection.execute(_CARD_SCORES_SQL)
    columns = {row[1] for row in connection.execute("PRAGMA table_info(card_scores)")}
    for name, kind in _ADDED_COLUMNS:
        if name not in columns:
            connection.execute(f"ALTER TABLE card_scores ADD COLUMN {name} {kind}")
    connection.execute(_CARD_INDEX_SQL)
    connection.execute(_CURRENT_VIEW_SQL)
    connection.execute(_REQUESTS_SQL)
    connection.execute(_REQUESTS_INDEX_SQL)
    connection.execute(_HEARTBEAT_SQL)


def mismatch(core_recall: float | None, spill: float | None) -> float:
    """The worse problem of a card: the share of its core missed, or its spill. A card with no core misses nothing."""
    missed = 0.0 if core_recall is None else 1.0 - float(core_recall)
    return max(missed, 0.0 if spill is None else float(spill))


def kind_of(family: str) -> str:
    return "train" if family == TRAIN_FAMILY else "validation"


@dataclass(frozen=True, slots=True)
class CardScore:
    location_id: int
    image_key: str
    family: str
    model: str
    scored_at: str
    iou: float | None
    raw_iou: float | None
    core_recall: float | None
    spill: float | None
    bf10: float | None
    core_gt: int | None
    gt_area: int | None
    epoch: int | None = None
    prompt: str | None = None

    @classmethod
    def from_record(cls, record: Mapping[str, Any], *, family: str, model: str, scored_at: str,
                    epoch: int | None = None) -> "CardScore":
        """From a validation record (``validation.score_sample``): ``key`` is ``(volume, image_key, location_id)``."""
        return cls.pooled([record], family=family, model=model, scored_at=scored_at, epoch=epoch)

    @classmethod
    def pooled(cls, records: Sequence[Mapping[str, Any]], *, family: str, model: str, scored_at: str,
               epoch: int | None = None, prompt: str | None = None) -> "CardScore":
        """One score from several records of one card (the prompt variants of a training step), pooled as a run
        pools samples: core recall and spill over their pixel counts, IoU and boundary F1 averaged."""
        if not records:
            raise ValueError("no records to pool")
        _volume, image_key, location_id = records[0]["key"]
        core_gt = sum(int(r.get("core_gt") or 0) for r in records)
        core_hit = sum(int(r.get("core_hit") or 0) for r in records)
        gt_area = sum(int(r.get("gt_area") or 0) for r in records)
        spill_px = sum(int(r.get("spill_px") or 0) for r in records)
        if core_gt:
            recall = core_hit / core_gt
        elif len(records) == 1 and records[0].get("recall") is not None:
            recall = float(records[0]["recall"])  # a record that carries recall without its counts
        else:
            recall = None
        has_spill = any(r.get("spill_px") is not None for r in records)
        return cls(
            location_id=int(location_id),
            image_key=str(image_key),
            family=family,
            model=model,
            scored_at=scored_at,
            iou=_mean(records, "iou"),
            raw_iou=_mean(records, "raw_iou"),
            core_recall=recall,
            spill=(spill_px / max(gt_area, 1)) if has_spill else None,
            bf10=_mean(records, "bf10"),
            core_gt=core_gt if any(r.get("core_gt") is not None for r in records) else None,
            gt_area=gt_area if any(r.get("gt_area") is not None for r in records) else None,
            epoch=epoch,
            prompt=prompt,
        )

    @property
    def mismatch(self) -> float:
        return mismatch(self.core_recall, self.spill)

    @property
    def kind(self) -> str:
        return kind_of(self.family)


def _mean(records: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(r[key]) for r in records if r.get(key) is not None]
    return sum(values) / len(values) if values else None


_COLUMNS = ("location_id", "image_key", "family", "model", "scored_at", "iou", "raw_iou", "core_recall", "spill",
            "bf10", "core_gt", "gt_area", "epoch", "prompt")


def write_card_scores(connection: sqlite3.Connection, scores: Iterable[CardScore]) -> int:
    """Append ``scores`` (earlier rows are kept); returns how many were written."""
    rows = [tuple(getattr(s, c) for c in _COLUMNS) for s in scores]
    if rows:
        connection.executemany(
            f"INSERT OR REPLACE INTO card_scores ({', '.join(_COLUMNS)}) VALUES ({', '.join('?' * len(_COLUMNS))})",
            rows,
        )
    return len(rows)


_READ_COLUMNS = ("location_id", "image_key", "family", "model", "scored_at", "iou", "core_recall", "spill", "bf10",
                 "epoch", "prompt")


def _score_dict(row: Sequence[Any]) -> dict[str, Any]:
    out = dict(zip(_READ_COLUMNS, row))
    out["kind"] = kind_of(out["family"])
    out["mismatch"] = mismatch(out.get("core_recall"), out.get("spill"))
    return out


def _location_filter(location_ids: Sequence[int] | None) -> tuple[str, list[Any]] | None:
    if location_ids is None:
        return "1 = 1", []
    ids = [int(i) for i in location_ids]
    if not ids:
        return None
    return f"location_id IN ({', '.join('?' * len(ids))})", ids


def latest_card_scores(
    connection: sqlite3.Connection, location_ids: Sequence[int] | None = None
) -> dict[tuple[int, str], dict[str, Any]]:
    """``{(location_id, image_key): {"kind", "new", "old"}}`` for every scored card of ``location_ids`` (None: all).

    ``new`` is the score the card shows: its newest validation score, else its newest training-pass score (``kind``
    says which); ``old`` is the score of the same kind before it (a re-score's or the previous epoch's), or None."""
    where = _location_filter(location_ids)
    if where is None:
        return {}
    clause, params = where
    sql = f"""
        SELECT {', '.join(_READ_COLUMNS)} FROM (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY location_id, image_key, (family = '{TRAIN_FAMILY}')
                ORDER BY scored_at DESC, rowid DESC
            ) AS rank
            FROM card_scores WHERE {clause}
        ) WHERE rank <= 2
        ORDER BY location_id, image_key, (family = '{TRAIN_FAMILY}') ASC, rank
    """
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for row in connection.execute(sql, params):
        score = _score_dict(row)
        key = (int(score["location_id"]), str(score["image_key"]))
        slot = out.get(key)
        if slot is None:
            out[key] = {"kind": score["kind"], "new": score, "old": None}
        elif slot["kind"] == score["kind"] and slot["old"] is None:
            slot["old"] = score
    return out


def card_score_history(connection: sqlite3.Connection, location_id: int, image_key: str) -> dict[str, Any]:
    """A card's scores over time, oldest first, for the trend graph: ``{"validation": [...], "train": [...],
    "loss": [{"epoch", "score"}, ...]}``. Score points carry ``epoch, model, scored_at, iou, core_recall, spill,
    mismatch``; ``loss`` is the training loss per epoch (``location_scores``, when the table exists)."""
    rows = connection.execute(
        f"SELECT {', '.join(_READ_COLUMNS)} FROM card_scores WHERE location_id = ? AND image_key = ? "
        "ORDER BY scored_at, rowid",
        (int(location_id), str(image_key)),
    ).fetchall()
    history: dict[str, Any] = {"validation": [], "train": [], "loss": []}
    for row in rows:
        score = _score_dict(row)
        history[score["kind"]].append(score)
    tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    if "location_scores" in tables:
        history["loss"] = [
            {"epoch": int(epoch), "score": float(score)}
            for epoch, score in connection.execute(
                "SELECT epoch, score FROM location_scores WHERE location_id = ? AND image_key IN (?, '') "
                "ORDER BY epoch",
                (int(location_id), str(image_key)),
            )
        ]
    return history


def queue_location(connection: sqlite3.Connection, location_id: int) -> int:
    """Ask the scorer service to re-score every card of ``location_id``; returns the request id. A location that
    already waits is not queued twice (its pending request's id is returned)."""
    row = connection.execute(
        "SELECT id FROM card_score_requests WHERE location_id = ? AND state = 'pending' ORDER BY id LIMIT 1",
        (int(location_id),),
    ).fetchone()
    if row is not None:
        return int(row[0])
    cursor = connection.execute(
        "INSERT INTO card_score_requests (location_id, requested_at) VALUES (?, ?)", (int(location_id), now_iso())
    )
    return int(cursor.lastrowid)


def claim_requests(connection: sqlite3.Connection, limit: int = 32) -> list[tuple[int, int]]:
    """Mark up to ``limit`` pending requests running and return them as ``[(request_id, location_id), ...]``.

    Runs in one write transaction, so two scorers never claim the same request. ``connection`` must be in autocommit
    mode (``isolation_level=None``), as the scorer opens it."""
    connection.execute("BEGIN IMMEDIATE")
    try:
        rows = connection.execute(
            "SELECT id, location_id FROM card_score_requests WHERE state = 'pending' ORDER BY id LIMIT ?",
            (int(limit),),
        ).fetchall()
        started = now_iso()
        connection.executemany(
            "UPDATE card_score_requests SET state = 'running', started_at = ? WHERE id = ?",
            [(started, int(r[0])) for r in rows],
        )
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    return [(int(r[0]), int(r[1])) for r in rows]


def finish_request(
    connection: sqlite3.Connection, request_id: int, *, model: str | None, cards: int, error: str | None = None
) -> None:
    """Close a claimed request: ``done`` with the number of cards scored, or ``failed`` with ``error``."""
    connection.execute(
        "UPDATE card_score_requests SET state = ?, finished_at = ?, model = ?, cards = ?, error = ? WHERE id = ?",
        ("failed" if error else "done", now_iso(), model, int(cards), error, int(request_id)),
    )


def requeue_running(connection: sqlite3.Connection) -> int:
    """Put requests a stopped scorer left ``running`` back in the queue; returns how many."""
    cursor = connection.execute(
        "UPDATE card_score_requests SET state = 'pending', started_at = NULL WHERE state = 'running'"
    )
    return int(cursor.rowcount or 0)


def request_status(connection: sqlite3.Connection, location_id: int) -> dict[str, Any] | None:
    """The newest request for ``location_id`` (state, times, model, cards, error), or None."""
    row = connection.execute(
        "SELECT id, state, requested_at, started_at, finished_at, model, cards, error FROM card_score_requests "
        "WHERE location_id = ? ORDER BY id DESC LIMIT 1",
        (int(location_id),),
    ).fetchone()
    if row is None:
        return None
    names = ("id", "state", "requested_at", "started_at", "finished_at", "model", "cards", "error")
    return dict(zip(names, row))


# ---- the scorer service's heartbeat -----------------------------------------------------------------------------

SCORER_STALE_SECONDS = 120.0
"""A scorer whose heartbeat is older than this is taken to be stopped (it beats every few seconds while it runs)."""

_HEARTBEAT_SQL = """
CREATE TABLE IF NOT EXISTS card_scorer_heartbeat (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    seen_at TEXT NOT NULL,
    spec TEXT,
    model TEXT
)
"""


def beat(connection: sqlite3.Connection, *, spec: str, model: str | None) -> None:
    """Record that a scorer service serves this catalog now (``spec``: what it scores with, e.g. ``best:sam2_em_v5``)."""
    connection.execute(_HEARTBEAT_SQL)
    connection.execute(
        "INSERT INTO card_scorer_heartbeat (id, seen_at, spec, model) VALUES (1, ?, ?, ?) "
        "ON CONFLICT (id) DO UPDATE SET seen_at = excluded.seen_at, spec = excluded.spec, model = excluded.model",
        (now_iso(), spec, model),
    )


def scorer_alive(connection: sqlite3.Connection, max_age_s: float = SCORER_STALE_SECONDS) -> dict[str, Any] | None:
    """The serving scorer (``{"seen_at", "spec", "model"}``) when its heartbeat is fresh, else None."""
    connection.execute(_HEARTBEAT_SQL)
    row = connection.execute("SELECT seen_at, spec, model FROM card_scorer_heartbeat WHERE id = 1").fetchone()
    if row is None:
        return None
    seen = datetime.strptime(str(row[0]), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    if (datetime.now(timezone.utc) - seen).total_seconds() > float(max_age_s):
        return None
    return {"seen_at": row[0], "spec": row[1], "model": row[2]}
