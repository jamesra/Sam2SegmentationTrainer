"""Update All: bring a volume's masks and annotations up to date with OData in one background job.

One watermark per volume (``update_state.json`` in ``AnnotationCrops``) says how far the volume has been checked.
The job asks OData for every location changed since it, walks those rows page by page, and moves the watermark
only when the whole run finished without a failure. A result too long for one page is read to the end by following
each page's ``@odata.nextLink``; the request carries no ``$top``, which would cap the whole result and end the walk
after the first page (see ``locations_url``).

For each row OData returns:

* a location the catalog already has is redrawn (``refresh.refresh_location``) and its label / type refreshed;
* a location the catalog does not have is added when the crop images it needs are already on disk
  (``refresh.add_new_location``); otherwise it is skipped, logged, and counted. Nothing is downloaded or stitched.

Skipped ids are remembered and retried at the start of the next run, because they do not hold the watermark back.
"""

from __future__ import annotations

import functools
import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import takewhile
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib.parse import quote, urljoin, urlparse
from urllib.request import Request, urlopen

from catalog import connect
from circle_update import (
    CIRCLE_TYPE_CODE,
    Circle,
    CircleFile,
    CropWindowRef,
    annotated_circle,
    entries_for_circle,
    inscribed_available,
    inscribed_circle,
)
from refresh import (
    TileIndex,
    add_new_location,
    drop_stale_thumbnails,
    load_add_support,
    nornir_buildmanager_root,
    refresh_location,
)

_logger = logging.getLogger(__name__)

STATE_FILE = "update_state.json"
LOG_FILE = "update_all.log"
PAGE_SIZE = int(os.environ.get("GALLERY_UPDATE_PAGE_SIZE", "500") or 500)
RETRY_BATCH = 50
# Update Visible asks for its ids in lists this long: one OData request each, a few hundred bytes of URL.
VISIBLE_BATCH = 200
SKIP_LINES_KEPT = 200
ERRORS_KEPT = 20

_SELECT = "ID,ParentID,TypeCode,Z,X,Y,OffEdge,MosaicShape,LastModified,Radius"
_EXPAND = "Parent($select=ID,TypeID,Label;$expand=Type($select=ID,Name,ParentID))"

GetJson = Callable[[str], dict[str, Any]]


# ---------------------------------------------------------------- time helpers

def parse_odata_datetime(value: Any) -> datetime | None:
    """OData or ISO timestamp as an aware UTC datetime, or None when it cannot be read."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.startswith("/Date(") and text.endswith(")/"):
            millis = int(text[6:-2].split("+")[0].split("-")[0])
            return datetime.fromtimestamp(millis / 1000.0, tz=timezone.utc)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def odata_literal(moment: datetime) -> str:
    """A UTC instant written the way an OData ``$filter`` compares dates."""
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# ------------------------------------------------------------------ state file

def state_path(crops: str | os.PathLike[str]) -> Path:
    return Path(crops) / STATE_FILE


def load_state(crops: str | os.PathLike[str]) -> dict[str, Any]:
    try:
        payload = json.loads(state_path(crops).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def save_state(crops: str | os.PathLike[str], state: dict[str, Any]) -> bool:
    """Write the state file atomically. False when the share does not let us."""
    path = state_path(crops)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(state, indent=1), encoding="utf-8")
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        return False
    return True


def initial_watermark(crops: str | os.PathLike[str]) -> datetime:
    """Where the first run starts: the older of the masks folder's mtime and the oldest mask creation time.

    Older is the safe side: the first run may redraw more than needed, but cannot miss a change.
    """
    masks = Path(crops) / "masks"
    stamps: list[float] = []
    try:
        stamps.append(masks.stat().st_mtime)
        for entry in os.scandir(masks):
            if not entry.is_file(follow_symlinks=False):
                continue
            info = entry.stat(follow_symlinks=False)
            # Copies reset the creation time and keep the mtime (or the reverse), so take the older of the two.
            stamps.append(min(info.st_mtime, getattr(info, "st_birthtime", None) or info.st_mtime))
    except OSError:
        pass
    if not stamps:
        return datetime.now(timezone.utc)
    return datetime.fromtimestamp(min(stamps), tz=timezone.utc)


# ------------------------------------------------------------------ OData paging

def http_get_json(url: str, *, max_page_size: int | None = None) -> dict[str, Any]:
    """GET one OData document.

    ``max_page_size`` is sent as ``Prefer: odata.maxpagesize``: the standard way to ask a service for pages of about
    that many rows. The service answers with ``@odata.nextLink`` while rows remain.
    """
    headers = {"Accept": "application/json"}
    if max_page_size:
        headers["Prefer"] = f"odata.maxpagesize={int(max_page_size)}"
    request = Request(url, headers=headers)
    with urlopen(request, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("OData response is not an object")
    return payload


def locations_url(base: str, filter_text: str) -> str:
    """The locations matching ``filter_text`` in ID order.

    There is deliberately no ``$top``. In OData ``$top`` caps the *total* rows of the answer, it is not a page size:
    ``$top=500`` returned 500 rows and no ``@odata.nextLink`` however many locations matched, so a run only ever saw
    the first 500 and still recorded itself as clean. Without it the service pages by itself (its own maximum, or the
    ``Prefer`` size from :func:`http_get_json`) and every page but the last carries the link to the next.
    """
    query = (
        f"$filter={quote(filter_text, safe='()/')}"
        f"&$orderby=ID&$select={quote(_SELECT, safe=',')}"
        f"&$expand={quote(_EXPAND, safe='();$,')}"
        f"&$count=true"
    )
    return f"{base.rstrip('/')}/Locations?{query}"


def id_in_clause(ids: list[int]) -> str:
    """``ID in (1,2,3)``: the locations with these ids.

    Not ``ID eq 1 or ID eq 2 or ...``: Viking's OData rejects a ``$filter`` of more than 100 nodes (HTTP 400, "The
    node count limit of '100' has been exceeded"), which 25 ``or``-ed ids already pass. An ``in`` list stays a few
    nodes however long it is (RC1's 137 skipped ids went through in one request).
    """
    return "ID in (" + ",".join(str(int(item)) for item in ids) + ")"


def iter_pages(url: str, get_json: GetJson) -> Iterator[dict[str, Any]]:
    """Every page of a result: follow ``@odata.nextLink`` (the continuation) until a page leaves it out."""
    next_url: str | None = url
    while next_url:
        document = get_json(next_url)
        yield document
        link = document.get("@odata.nextLink")
        if not link:
            return
        next_url = str(link) if urlparse(str(link)).scheme else urljoin(next_url, str(link))


def _entities(document: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in (document.get("value") or []) if isinstance(item, dict) and "ID" in item]


# ------------------------------------------------------------------- catalog I/O

def _catalog_ids(crops: Path) -> set[int]:
    connection = connect(crops)
    try:
        return {int(row[0]) for row in connection.execute("SELECT DISTINCT location_id FROM locations")}
    finally:
        connection.close()


def _update_metadata(crops: Path, entity: dict[str, Any]) -> None:
    """Label and type of a location the catalog already has, from the expanded Parent. Missing values are kept."""
    parent = entity.get("Parent") or {}
    parent_type = parent.get("Type") or {}
    fields = {
        "structure_id": entity.get("ParentID", parent.get("ID")),
        "structure_label": parent.get("Label"),
        "type_id": parent.get("TypeID", parent_type.get("ID")),
        "type_name": parent_type.get("Name"),
    }
    fields = {name: value for name, value in fields.items() if value is not None}
    if not fields:
        return
    assignments = ", ".join(f"{name} = ?" for name in fields)
    connection = connect(crops)
    try:
        connection.execute(
            f"UPDATE locations SET {assignments} WHERE location_id = ?",
            [*fields.values(), int(entity["ID"])],
        )
        connection.commit()
    finally:
        connection.close()


def _insert_rows(crops: Path, added: dict[str, Any]) -> None:
    """Catalog rows for a location that was just added, one per window. SAM2 columns stay NULL."""
    record = added["record"]
    connection = connect(crops)
    try:
        for window in added["windows"]:
            key = window["image_key"]
            area = window["area"]
            radius = record.radius if record.radius is not None else (
                math.sqrt(area / math.pi) if area > 0 else None
            )
            image = next(
                (f"images/{key}{suffix}" for suffix in (".png", ".jpg", ".jpeg")
                 if (crops / "images" / f"{key}{suffix}").is_file()),
                f"images/{key}.png",
            )
            connection.execute(
                "DELETE FROM locations WHERE location_id = ? AND image_key = ?", [record.id, key]
            )
            connection.execute(
                "INSERT INTO locations (location_id, z, structure_id, structure_label, type_id, type_name, "
                "radius, image_key, image_relpath, mask_relpath, ignored) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
                [
                    record.id, record.z, record.parent_id, record.structure_label, record.type_id,
                    record.type_name, radius, key, image, f"masks/{key}_{record.id}.png",
                ],
            )
        connection.commit()
    finally:
        connection.close()


# ------------------------------------------------------------------ circle entries

_support_cache: list[Any] = []


def _label_of(entity: dict[str, Any]) -> str:
    """The category the exporter writes: structure label, else type name, else empty."""
    parent = entity.get("Parent") or {}
    return str(parent.get("Label") or (parent.get("Type") or {}).get("Name") or "").strip()


def circle_for_entity(entity: dict[str, Any]) -> Circle | None:
    """The circle the exporter would write for this OData location.

    A Viking circle (``TypeCode`` 1) gives its annotated circle; any other shape gives the largest circle inside
    its polygon. None when the shape cannot be read, or scipy is missing for the polygon case.
    """
    if not _support_cache:
        root = nornir_buildmanager_root()
        if root is None:
            return None
        _support_cache.append(load_add_support(root))
    support = _support_cache[0]
    location_id = int(entity["ID"])
    wkt = support.mosaic_wkt_from_odata_shape(entity.get("MosaicShape"))
    if not wkt:
        return None
    type_code = entity.get("TypeCode")
    type_code = None if type_code in (None, "") else int(type_code)
    label = _label_of(entity)
    if type_code == CIRCLE_TYPE_CODE:
        radius = entity.get("Radius")
        x, y = entity.get("X"), entity.get("Y")
        return annotated_circle(
            location_id,
            None if x in (None, "") else float(x),
            None if y in (None, "") else float(y),
            None if radius in (None, "") else float(radius),
            wkt,
            label,
        )
    polygons = support.hydrate_polygons(support.parse_wkt_polygons(wkt), type_code)
    return inscribed_circle(location_id, polygons, label)


def is_circle_entity(entity: dict[str, Any]) -> bool:
    code = entity.get("TypeCode")
    return code not in (None, "") and int(code) == CIRCLE_TYPE_CODE


def window_refs(index: Any, z: int) -> list[CropWindowRef]:
    """Crop windows whose image is on disk for section *z*, as circle-entry geometry."""
    return [
        CropWindowRef(w.key, w.downsample, w.x0, w.y0, w.x1 - w.x0, w.y1 - w.y0)
        for w in index.windows(z)
    ]


# ----------------------------------------------------------------------- the job

@dataclass
class Dependencies:
    """What the job calls out to; tests replace these."""

    get_json: GetJson = http_get_json
    refresh: Callable[[Path, dict[str, Any], str], list[str]] = lambda crops, entity, exporter: refresh_location(
        crops, entity, exporter
    )
    add: Callable[[Path, dict[str, Any], TileIndex], tuple[dict[str, Any] | None, str | None]] = (
        lambda crops, entity, index: add_new_location(crops, entity, index)
    )
    drop_stale: Callable[[Path, list[str]], None] = lambda crops, keys: drop_stale_thumbnails(crops, keys)
    tile_index: Callable[[Path], TileIndex] = lambda crops: TileIndex(crops)
    circle_for: Callable[[dict[str, Any]], Circle | None] = lambda entity: circle_for_entity(entity)


@dataclass
class UpdateJob:
    volume: str
    crops: Path
    odata: str
    exporter: str
    page_size: int
    deps: Dependencies
    # Update Visible: only these locations, never the watermark walk or the skipped-id retry. None is Update All.
    ids: list[int] | None = None
    phase: str = "starting"
    page: int = 0
    pages: int = 0
    entities_seen: int = 0
    refreshed: int = 0
    unchanged: int = 0
    added: int = 0
    skipped: int = 0
    circles: int = 0
    failed: int = 0
    notes: list[str] = field(default_factory=list)
    skip_lines: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    started: float = 0.0
    finished: float = 0.0
    cancelled: bool = False
    watermark_before: str = ""
    watermark_after: str = ""
    state_saved: bool = True
    refreshed_ids: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None

    # -- control
    def start(self) -> None:
        self.started = time.time()
        self._thread = threading.Thread(target=self._run, name=f"update-all-{self.volume}", daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        self._cancel.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    @property
    def running(self) -> bool:
        return self.phase not in ("done", "failed", "cancelled")

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "volume": self.volume,
                "phase": self.phase,
                "running": self.running,
                "page": self.page,
                "pages": self.pages,
                "entitiesSeen": self.entities_seen,
                "refreshed": self.refreshed,
                "unchanged": self.unchanged,
                "added": self.added,
                "skipped": self.skipped,
                "circles": self.circles,
                "notes": list(self.notes),
                "failed": self.failed,
                "skipLines": list(self.skip_lines),
                "errors": list(self.errors),
                "started": self.started,
                "finished": self.finished,
                "cancelled": self.cancelled,
                "watermarkBefore": self.watermark_before,
                "watermarkAfter": self.watermark_after,
                "stateSaved": self.state_saved,
                "refreshedIds": list(self.refreshed_ids),
                "scope": "all" if self.ids is None else "visible",
                "requested": 0 if self.ids is None else len(self.ids),
                **({} if self.ids is None else {"unit": "batch"}),
            }

    def _set(self, **values: Any) -> None:
        with self._lock:
            for name, value in values.items():
                setattr(self, name, value)

    # -- the run
    def _run(self) -> None:
        try:
            self._walk()
        except Exception as exc:  # noqa: BLE001 - a job must always end in a reported state
            _logger.exception("update-all %s failed", self.volume)
            with self._lock:
                self.errors.append(f"job stopped: {exc}")
                self.failed += 1
            self._set(phase="failed", finished=time.time())

    def _log(self, line: str) -> None:
        try:
            with (self.crops / LOG_FILE).open("a", encoding="utf-8") as handle:
                handle.write(f"{datetime.now(timezone.utc).isoformat()} {line}\n")
        except OSError:
            pass

    def _walk(self) -> None:
        self._circle_file: CircleFile | None = None
        try:
            self._walk_rows()
        finally:
            # The circle entries replaced so far are real changes, so they are written even when the run stops.
            if self._circle_file is not None and not self._circle_file.save():
                with self._lock:
                    self.errors.append("could not write circle_annotations.json")

    def _update_circle(self, entity: dict[str, Any], index_for: Callable[[], TileIndex]) -> None:
        """Replace this location's circle entries from its current OData shape."""
        if self._circle_file is None:
            self._circle_file = CircleFile(self.crops)
            if not self._circle_file.exists:
                with self._lock:
                    self.notes.append(
                        "circle_annotations.json is missing, so circles were not updated; export circles first."
                    )
        if not self._circle_file.exists:
            return
        circle = self.deps.circle_for(entity)
        if circle is None and not is_circle_entity(entity) and not inscribed_available():
            if not any("scipy" in note for note in self.notes):
                with self._lock:
                    self.notes.append("scipy is not installed, so inscribed circles were not updated.")
            return
        windows = window_refs(index_for(), int(entity["Z"]))
        entries = entries_for_circle(circle, windows) if circle is not None else {}
        self._circle_file.replace(int(entity["ID"]), entries)
        with self._lock:
            self.circles += 1

    def _rows(self, url: str) -> Iterator[tuple[dict[str, Any], Iterator[dict[str, Any]]]]:
        """Yield each OData page of ``url`` with an iterator over its rows that stops once the job is cancelled."""
        get_json = self.deps.get_json
        if get_json is http_get_json:
            # The real getter asks for pages of this job's size; a replacement (a test's) is called as it always was.
            get_json = functools.partial(http_get_json, max_page_size=self.page_size)
        for document in iter_pages(url, get_json):
            yield document, takewhile(lambda _row: not self._cancel.is_set(), _entities(document))

    def _walk_rows(self) -> None:
        state = load_state(self.crops)
        before = parse_odata_datetime(state.get("watermark")) or initial_watermark(self.crops)
        visible = self.ids is not None
        self._set(watermark_before="" if visible else odata_literal(before), phase="reading catalog")
        known = _catalog_ids(self.crops)
        newest = before
        index: list[TileIndex] = []
        previous_skips = [int(item) for item in state.get("skipped_ids") or []]
        skipped_now: set[int] = set()
        resolved: set[int] = set()
        touched_keys: list[str] = []

        def tile_index() -> TileIndex:
            if not index:
                index.append(self.deps.tile_index(self.crops))
            return index[0]

        def handle(entity: dict[str, Any], retry: bool) -> None:
            nonlocal newest
            location_id = int(entity["ID"])
            modified = parse_odata_datetime(entity.get("LastModified"))
            if modified is not None and modified > newest and not retry:
                newest = modified
            try:
                if is_circle_entity(entity):
                    # A circle has no mask. It only feeds circle_annotations.json.
                    self._update_circle(entity, tile_index)
                    return
                if location_id in known:
                    keys = self.deps.refresh(self.crops, entity, self.exporter)
                    _update_metadata(self.crops, entity)
                    with self._lock:
                        if keys:
                            self.refreshed += 1
                            self.refreshed_ids.append(location_id)
                        else:
                            self.unchanged += 1
                    touched_keys.extend(keys)
                    if keys:
                        self._update_circle(entity, tile_index)
                    return
                added, reason = self.deps.add(self.crops, entity, tile_index())
                if added is None:
                    line = f"skip {location_id} z={entity.get('Z')}: {reason}"
                    self._log(line)
                    skipped_now.add(location_id)
                    with self._lock:
                        self.skipped += 1
                        if len(self.skip_lines) < SKIP_LINES_KEPT:
                            self.skip_lines.append(line)
                    return
                _insert_rows(self.crops, added)
                self._update_circle(entity, tile_index)
                known.add(location_id)
                resolved.add(location_id)
                touched_keys.extend(window["image_key"] for window in added["windows"])
                with self._lock:
                    self.added += 1
                    self.refreshed_ids.append(location_id)
            except Exception as exc:  # noqa: BLE001 - one bad row must not stop the run
                message = f"{location_id}: {exc}"
                self._log(f"error {message}")
                if retry:
                    # Usually older than the watermark, so only skipped_ids brings it back.
                    skipped_now.add(location_id)
                with self._lock:
                    self.failed += 1
                    if len(self.errors) < ERRORS_KEPT:
                        self.errors.append(message)

        def flush_page() -> None:
            if touched_keys:
                self.deps.drop_stale(self.crops, sorted(set(touched_keys)))
                touched_keys.clear()

        if visible:
            self._walk_ids(handle, flush_page)
            return

        # Ids skipped by an earlier run are asked for again, in small batches.
        if previous_skips:
            self._set(phase="retrying skipped")
            for start in range(0, len(previous_skips), RETRY_BATCH):
                if self._cancel.is_set():
                    break
                batch = previous_skips[start:start + RETRY_BATCH]
                for _document, rows in self._rows(locations_url(self.odata, id_in_clause(batch))):
                    for entity in rows:
                        handle(entity, retry=True)
                    flush_page()

        # Everything OData changed since the watermark: one tick per page.
        self._set(phase="updating")
        url = locations_url(self.odata, f"LastModified ge {odata_literal(before)}")
        for document, rows in self._rows(url):
            if self._cancel.is_set():
                break
            if self.pages == 0:
                try:
                    total = int(document.get("@odata.count"))
                except (TypeError, ValueError):
                    total = len(_entities(document))
                # The service may use its own page size, so the first page says how many rows a page holds.
                per_page = len(_entities(document)) if document.get("@odata.nextLink") else 0
                self._set(pages=max(1, math.ceil(total / max(1, per_page or self.page_size))))
            for entity in rows:
                with self._lock:
                    self.entities_seen += 1
                handle(entity, retry=False)
            flush_page()
            with self._lock:
                self.page += 1

        clean = not self._cancel.is_set() and self.failed == 0
        # Retried ids that came back skipped or failed are in skipped_now; ones OData no longer returns are dropped.
        # A cancelled run may not have reached every retry, so it keeps the whole earlier list.
        carried = previous_skips if self._cancel.is_set() else []
        remaining = sorted((set(carried) | skipped_now) - resolved)
        after = newest if clean else before
        saved = save_state(
            self.crops,
            {
                "watermark": odata_literal(after),
                "last_run": datetime.now(timezone.utc).isoformat(),
                "last_counts": {
                    "refreshed": self.refreshed, "added": self.added, "skipped": self.skipped,
                    "circles": self.circles, "unchanged": self.unchanged, "failed": self.failed, "clean": clean,
                },
                "skipped_ids": remaining,
            },
        )
        self._set(
            watermark_after=odata_literal(after),
            state_saved=saved,
            cancelled=self._cancel.is_set(),
            phase="cancelled" if self._cancel.is_set() else ("done" if clean else "failed"),
            finished=time.time(),
        )
        if not saved:
            with self._lock:
                self.errors.append("could not write update_state.json; the watermark was not saved")

    def _walk_ids(
        self, handle: Callable[[dict[str, Any], bool], None], flush_page: Callable[[], None]
    ) -> None:
        """Update Visible: redraw just ``self.ids``, one tick per batch.

        ``update_state.json`` is left alone: the watermark says how far *every* change has been checked, which a run
        over some locations does not show, and the skipped ids wait for the next Update All.
        """
        ids = sorted(set(self.ids or []))
        batches = [ids[start:start + VISIBLE_BATCH] for start in range(0, len(ids), VISIBLE_BATCH)]
        self._set(phase="updating visible", pages=max(1, len(batches)))
        seen: set[int] = set()
        for batch in batches:
            if self._cancel.is_set():
                break
            for _document, rows in self._rows(locations_url(self.odata, id_in_clause(batch))):
                for entity in rows:
                    seen.add(int(entity["ID"]))
                    with self._lock:
                        self.entities_seen += 1
                    handle(entity, True)
                flush_page()
            with self._lock:
                self.page += 1
        cancelled = self._cancel.is_set()
        missing = len(ids) - len(seen)
        if missing and not cancelled:
            with self._lock:
                self.notes.append(
                    f"{missing} of the {len(ids)} locations are no longer in OData; their cards were left as they are."
                )
        self._set(
            cancelled=cancelled,
            phase="cancelled" if cancelled else ("done" if self.failed == 0 else "failed"),
            finished=time.time(),
        )


# ------------------------------------------------------------------- registry

_jobs: dict[str, UpdateJob] = {}
_jobs_lock = threading.Lock()


def start_job(
    volume: str,
    crops: Path,
    odata: str,
    exporter: str,
    *,
    page_size: int | None = None,
    deps: Dependencies | None = None,
    ids: list[int] | None = None,
) -> tuple[UpdateJob, bool]:
    """Start the volume's job, or return the one already running. The flag is True for a new job.

    ``ids`` makes it an Update Visible job over just those locations (see :meth:`UpdateJob._walk_ids`).
    """
    with _jobs_lock:
        current = _jobs.get(volume)
        if current is not None and current.running:
            return current, False
        job = UpdateJob(
            volume=volume,
            crops=Path(crops),
            odata=odata,
            exporter=exporter,
            page_size=page_size or PAGE_SIZE,
            deps=deps or Dependencies(),
            ids=None if ids is None else [int(item) for item in ids],
        )
        _jobs[volume] = job
    job.start()
    return job, True


def current_job(volume: str) -> UpdateJob | None:
    with _jobs_lock:
        return _jobs.get(volume)


def idle_status(crops: str | os.PathLike[str]) -> dict[str, Any]:
    """What the dialog shows before a run: the saved watermark and the last result, if any."""
    state = load_state(crops)
    return {
        "phase": "idle",
        "running": False,
        "watermark": state.get("watermark") or "",
        "lastRun": state.get("last_run") or "",
        "lastCounts": state.get("last_counts") or {},
        "skippedPending": len(state.get("skipped_ids") or []),
    }


# ------------------------------------------------------------ one location on demand

def import_location(
    crops: str | os.PathLike[str],
    entity: dict[str, Any],
    *,
    deps: Dependencies | None = None,
) -> dict[str, Any]:
    """Bring one location OData has and the volume lacks into the volume, as Update All would.

    The result's ``status`` is ``imported`` (rows, masks and circle entries written), ``exists`` (the catalog
    already has it), ``circle`` (a circle: only ``circle_annotations.json`` changes), or ``no_tiles`` (a crop
    image the annotation needs is not on disk; nothing was written, ``reason`` says why).
    """
    deps = deps or Dependencies()
    crops = Path(crops)
    location_id = int(entity["ID"])
    if location_id in _catalog_ids(crops):
        return {"status": "exists", "location_id": location_id}
    index = deps.tile_index(crops)

    def update_circle() -> bool:
        circle_file = CircleFile(crops)
        if not circle_file.exists:
            return False
        circle = deps.circle_for(entity)
        windows = window_refs(index, int(entity["Z"]))
        circle_file.replace(location_id, entries_for_circle(circle, windows) if circle is not None else {})
        return circle_file.save()

    if is_circle_entity(entity):
        return {"status": "circle", "location_id": location_id, "circles": update_circle()}
    added, reason = deps.add(crops, entity, index)
    if added is None:
        return {"status": "no_tiles", "location_id": location_id, "reason": reason or "no tile image"}
    _insert_rows(crops, added)
    keys = [window["image_key"] for window in added["windows"]]
    update_circle()
    deps.drop_stale(crops, sorted(set(keys)))
    return {"status": "imported", "location_id": location_id, "imageKeys": keys}
