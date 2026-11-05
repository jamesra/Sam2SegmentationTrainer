"""What the converter tells the segmentation server's cache beyond the requests themselves.

- **A job per build.** The server keeps every result linked to a job (``SegmentTilesRequest.job_id``) until the job is
  deleted, so a build's proposals are still cached when Viking opens one of its cards later. The job id is kept in the
  build's folder and reused by every later run of the build; deleting the build deletes the job.
- **Manual traces.** An outline a reviewer retraced belongs in the annotation database, which the gallery cannot
  write. Until it can, the trace is stored on the server (``PutManualTraces``) as the location's cached answer, against
  the LastModified the card was built from: Viking proposes it instead of a segmentation until the location is edited,
  and clears it when it saves the location (``ForgetLocations``). Retracing or approving a retraced card stores it;
  unapproving or rejecting it removes it. An unedited card sends nothing: its model result is already cached under the
  build's job.

Neither may cost the build or the review. A job the server cannot make leaves the build unprotected. A trace is
recorded on its card as pending (``trace_pending``: ``put`` or ``remove``) before it is sent and stays pending until
the server confirms it, so a server that is down gets it at the next review action or gallery start.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import catalog
from converter import geo
from converter.seg_client import polygon_shape
from seg_proto import segmentation_pb2 as pb

LOGGER = logging.getLogger("converter.serverreview")

JOB_FILE = "segmentation_job.json"
RETRACED = "retraced"
PUT, REMOVE = "put", "remove"
# Review actions and what each asks of a retraced card's trace; other cards send nothing.
_TRACE_ACTIONS = {"retrace": PUT, "approve": PUT, "unapprove": REMOVE, "ignore": REMOVE}
# Columns the trace adds to converter_cards: who retraced the card and when, and what the server still has to be told.
TRACE_COLUMNS = (("traced_by", "TEXT"), ("traced_ticks", "INTEGER"), ("trace_pending", "TEXT"))
# One sync per build at a time, so two review clicks never send the same trace twice or clear each other's marks.
_SYNC_LOCKS: Dict[str, threading.Lock] = {}
_SYNC_LOCKS_GUARD = threading.Lock()
_EPOCH = datetime(1, 1, 1, tzinfo=timezone.utc)


def _now_ticks() -> int:
    """Now, UTC, in .NET ticks."""
    elapsed = datetime.now(timezone.utc) - _EPOCH
    return (elapsed.days * 86_400 + elapsed.seconds) * 10_000_000 + elapsed.microseconds * 10


def stored_job(crops: Path) -> int:
    """The build's job id, 0 when it has none."""
    try:
        return int(json.loads((Path(crops) / JOB_FILE).read_text(encoding="utf-8"))["job_id"])
    except (OSError, ValueError, KeyError, TypeError):
        return 0


def build_job(crops: Path, client: Any, volume: str, set_name: str) -> int:
    """The build's job id, made on its first run. 0 (no job: results may be evicted) when the server cannot make one."""
    job_id = stored_job(crops)
    if job_id:
        return job_id
    try:
        job_id = client.create_job(volume, title=f"Converter {set_name} {volume}", kind="convert")
    except Exception as error:  # noqa: BLE001 - a job only protects cache rows; the build goes on without one
        LOGGER.warning("no segmentation job for %s/%s: %s", set_name, volume, error)
        return 0
    path = Path(crops) / JOB_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"job_id": job_id, "volume": volume}), encoding="utf-8")
    return job_id


def delete_job(make_client: Callable[[], Any], volume: str, job_id: int) -> None:
    """Delete a deleted build's job, so its results can be evicted. Runs in the background."""
    if not job_id:
        return

    def run() -> None:
        client = None
        try:
            client = make_client()
            client.delete_job(volume, job_id)
        except Exception as error:  # noqa: BLE001
            LOGGER.warning("segmentation job %s of %s not deleted: %s", job_id, volume, error)
        finally:
            if client is not None and hasattr(client, "close"):
                client.close()

    threading.Thread(target=run, name=f"delete-job-{job_id}", daemon=True).start()


def ensure_trace_columns(connection: sqlite3.Connection) -> None:
    """Add the trace columns to a converter_cards table built before them."""
    columns = {row[1] for row in connection.execute("PRAGMA table_info(converter_cards)")}
    for name, kind in TRACE_COLUMNS:
        if name not in columns:
            connection.execute(f"ALTER TABLE converter_cards ADD COLUMN {name} {kind}")


def mark_trace(crops: Path, location_id: int, image_key: Optional[str], action: str, tracer: str = "") -> bool:
    """Record what a review action asks of the location's trace on the server; ``retrace`` also records who traced it.

    Only retraced cards are marked. ``image_key`` None is every card of the location (the card's approve and reject
    controls cover the whole location). Returns whether anything is now pending.
    """
    pending = _TRACE_ACTIONS.get(action)
    if pending is None:
        return False
    connection = catalog.connect(crops)
    try:
        ensure_trace_columns(connection)
        where = "location_id = ? AND state = 'built' AND reason = ?"
        arguments: List[Any] = [int(location_id), RETRACED]
        if image_key:
            where += " AND image_key = ?"
            arguments.append(image_key)
        if action == "retrace":
            changed = connection.execute(
                f"UPDATE converter_cards SET trace_pending = ?, traced_by = ?, traced_ticks = ? WHERE {where}",
                [pending, tracer, _now_ticks(), *arguments],
            ).rowcount
        else:
            changed = connection.execute(
                f"UPDATE converter_cards SET trace_pending = ? WHERE {where}", [pending, *arguments]
            ).rowcount
        connection.commit()
        return changed > 0
    except sqlite3.OperationalError as error:  # a folder without converter cards
        LOGGER.debug("no converter card for %s: %s", location_id, error)
        return False
    finally:
        connection.close()


def pending_traces(crops: Path) -> Tuple[List["pb.ManualTrace"], List[int], List[Tuple[int, str, str]]]:
    """What the server still has to be told: traces to store, locations to remove, and the (location, image key,
    pending) marks that cover them, to clear once it is told.

    A location is stored when any of its cards is pending ``put`` (the latest trace wins), else removed. A card built
    before cards recorded the location's LastModified cannot be stored; its mark is cleared with the rest.
    """
    if not catalog.sqlite_path(crops).is_file():
        return [], [], []
    connection = catalog.connect(crops)
    try:
        ensure_trace_columns(connection)
        rows = connection.execute(
            "SELECT location_id, image_key, trace_pending, candidate_wkt, last_modified_ticks, traced_by, traced_ticks "
            "FROM converter_cards WHERE trace_pending IS NOT NULL ORDER BY location_id, COALESCE(traced_ticks, 0)"
        ).fetchall()
    except sqlite3.OperationalError as error:
        LOGGER.debug("no converter cards in %s: %s", crops, error)
        return [], [], []
    finally:
        connection.close()
    by_location: Dict[int, List[Any]] = {}
    for row in rows:
        by_location.setdefault(int(row["location_id"]), []).append(row)
    traces: List["pb.ManualTrace"] = []
    removals: List[int] = []
    for location_id, cards in by_location.items():
        puts = [card for card in cards if card["trace_pending"] == PUT]
        if not puts:
            removals.append(location_id)
            continue
        trace = _trace(location_id, puts[-1])
        if trace is not None:
            traces.append(trace)
    marks = [(int(row["location_id"]), str(row["image_key"]), str(row["trace_pending"])) for row in rows]
    return traces, removals, marks


def _trace(location_id: int, card: Any) -> Optional["pb.ManualTrace"]:
    if not card["candidate_wkt"] or not card["last_modified_ticks"]:
        return None
    polygons = geo.parse_wkt_polygons(card["candidate_wkt"])
    if not polygons:
        return None
    exterior, *holes = polygons[0]
    return pb.ManualTrace(
        location_id=location_id, last_modified_ticks=int(card["last_modified_ticks"]),
        outline=polygon_shape(exterior, holes), tracer=card["traced_by"] or "",
        traced_ticks=int(card["traced_ticks"] or 0),
    )


def _clear_marks(crops: Path, marks: Iterable[Tuple[int, str, str]]) -> None:
    """Clear the marks the server was told about. A mark changed meanwhile (a later click) stays for the next sync."""
    connection = catalog.connect(crops)
    try:
        connection.executemany(
            "UPDATE converter_cards SET trace_pending = NULL "
            "WHERE location_id = ? AND image_key = ? AND trace_pending = ?",
            list(marks),
        )
        connection.commit()
    finally:
        connection.close()


def _sync_lock(crops: Path) -> threading.Lock:
    with _SYNC_LOCKS_GUARD:
        return _SYNC_LOCKS.setdefault(str(Path(crops).resolve()), threading.Lock())


def sync_now(crops: Path, volume: str, client: Any) -> int:
    """Send the build's pending traces and removals in one call. Returns how many locations the server was told about;
    raises when it could not be told (the marks stay)."""
    with _sync_lock(crops):
        traces, removals, marks = pending_traces(crops)
        if not marks:
            return 0
        if traces or removals:
            readable = client.readable_volume(volume)
            if readable is None:
                raise RuntimeError(f"the segmentation server is not assigned {volume}")
            client.put_manual_traces(int(readable.annotation_context_id), traces, removals)
        _clear_marks(crops, marks)
        LOGGER.info("%s: %d trace(s) stored and %d removed on the segmentation server", crops, len(traces), len(removals))
        return len(traces) + len(removals)


def sync(crops: Path, volume: str, make_client: Callable[[], Any]) -> threading.Thread:
    """:func:`sync_now` in the background, so a slow server never holds up the review; returns the thread (tests join
    it)."""

    def run() -> None:
        client = None
        try:
            client = make_client()
            sync_now(crops, volume, client)
        except Exception as error:  # noqa: BLE001 - the marks stay and are sent with the next sync
            LOGGER.warning("manual traces of %s not sent to the segmentation server: %s", crops, error)
        finally:
            if client is not None and hasattr(client, "close"):
                client.close()

    thread = threading.Thread(target=run, name=f"traces-{Path(crops).name}", daemon=True)
    thread.start()
    return thread


def sync_all(root: Optional[Path], make_client: Callable[[], Any]) -> List[threading.Thread]:
    """Sync every build under the converter root that still has a pending trace (at gallery start)."""
    threads: List[threading.Thread] = []
    if root is None or not Path(root).is_dir():
        return threads
    for set_dir in sorted(Path(root).iterdir()):
        if not set_dir.is_dir():
            continue
        for crops in sorted(set_dir.iterdir()):
            if crops.is_dir() and pending_traces(crops)[2]:
                threads.append(sync(crops, crops.name, make_client))
    return threads
