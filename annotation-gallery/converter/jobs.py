"""Background build jobs: one per (structure set, volume), reporting in the shape of Update All's status so the page's
progress strip, Cancel and polling serve both.

A job runs on a server thread, so it goes on when the page is closed or reloaded (it is found again by set and volume).
It also survives a server restart: a running job's settings are kept in ``job.json`` in its build folder, and
:func:`resume_unfinished` starts every build still marked running when the gallery starts. The resumed build skips every
card already built for the same shape and model (see builder.should_build). A cancelled, finished or failed build is not
resumed.
At most ``GALLERY_CONVERTER_STREAMS`` (default 1) segmentation streams run at once across all jobs, so a build cannot
starve the interactive users of the segmentation server.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from converter import builder as builder_module
from converter import engines, odata
from converter import serverreview
from converter.engines.tiled import ordered_tilesets
from converter.seg_client import SegmentationClient, context_for
from converter.tilesource import TileCache, Volume, load_volume, viking_xml_url

LOGGER = logging.getLogger("converter.jobs")

_JOBS: Dict[Tuple[str, str], "BuildJob"] = {}
_JOBS_LOCK = threading.Lock()
_STREAMS: Optional[threading.Semaphore] = None
_STREAMS_LOCK = threading.Lock()
# A build's last known state, beside its catalog; "running" there after a restart means the build was cut short.
STATE_FILE = "job.json"


class ConverterNotConfigured(Exception):
    """A setting the build needs is missing; the message says which."""


def streams(environ: Optional[Dict[str, str]] = None) -> threading.Semaphore:
    global _STREAMS
    with _STREAMS_LOCK:
        if _STREAMS is None:
            mapping = environ if environ is not None else os.environ
            _STREAMS = threading.Semaphore(max(1, int(mapping.get("GALLERY_CONVERTER_STREAMS") or "1")))
        return _STREAMS


def default_client(environ: Optional[Dict[str, str]] = None) -> SegmentationClient:
    """The segmentation client from the environment: ``GALLERY_SEGMENTATION_TARGET`` (host:port, TLS), an optional
    ``GALLERY_SEGMENTATION_CA`` file of trusted roots and ``GALLERY_SEGMENTATION_TLS_NAME`` to verify the certificate
    against when the host is reached by another name."""
    import grpc

    mapping = environ if environ is not None else os.environ
    target = (mapping.get("GALLERY_SEGMENTATION_TARGET") or "").strip()
    if not target:
        raise ConverterNotConfigured("GALLERY_SEGMENTATION_TARGET is not set (host:port of the segmentation server)")
    ca = (mapping.get("GALLERY_SEGMENTATION_CA") or "").strip()
    roots = Path(ca).read_bytes() if ca else None
    options = [
        ("grpc.max_receive_message_length", 256 * 1024 * 1024),
        ("grpc.max_send_message_length", 256 * 1024 * 1024),
    ]
    name = (mapping.get("GALLERY_SEGMENTATION_TLS_NAME") or "").strip()
    if name:
        options.append(("grpc.ssl_target_name_override", name))
    channel = grpc.secure_channel(target, grpc.ssl_channel_credentials(root_certificates=roots), options=options)
    return SegmentationClient(target, channel=channel)


def default_volume(volume_name: str, environ: Optional[Dict[str, str]] = None) -> Volume:
    """The volume's tilesets from its VikingXML: ``GALLERY_CONVERTER_VIKINGXML`` is a URL template with ``{volume}``
    (or ``<volume>``, which survives compose's variable substitution)."""
    url = working_viking_xml(volume_name, environ)
    if not url:
        raise ConverterNotConfigured("GALLERY_CONVERTER_VIKINGXML is not set (URL template of the VikingXML, with {volume})")
    return load_volume(url)


def tile_cache(environ: Optional[Dict[str, str]] = None) -> TileCache:
    mapping = environ if environ is not None else os.environ
    folder = (mapping.get("GALLERY_CONVERTER_TILE_CACHE") or "").strip()
    return TileCache(Path(folder) if folder else None)


class BuildJob:
    def __init__(
        self,
        set_name: str,
        volume: str,
        crops: Path,
        params: builder_module.BuildParams,
        *,
        make_client: Callable[[], SegmentationClient] = default_client,
        make_volume: Callable[[str], Volume] = default_volume,
        make_cache: Callable[[], TileCache] = tile_cache,
        get_json: odata.GetJson = odata.http_get_json,
    ) -> None:
        self.set_name = set_name
        self.volume = volume
        self.crops = crops
        self.params = params
        self._make_client = make_client
        self._make_volume = make_volume
        self._make_cache = make_cache
        self._get_json = get_json
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        # The running build's client, so a stop also ends the segmentation calls in flight.
        self._client: Optional[SegmentationClient] = None
        self.phase = "starting"
        self.total = 0
        self.processed = 0
        self.summary = builder_module.BuildSummary()
        self.errors: List[str] = []
        self.started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.finished: Optional[str] = None
        self.cancelled = False
        self.built_ids: List[int] = []
        self._thread = threading.Thread(target=self._run, name=f"converter-{set_name}-{volume}", daemon=True)

    @property
    def running(self) -> bool:
        return self.phase not in ("done", "failed", "cancelled")

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: Optional[float] = None) -> None:
        self._thread.join(timeout)

    def cancel(self) -> None:
        self._cancel.set()
        with self._lock:
            client = self._client
        stop = getattr(client, "cancel", None)
        if callable(stop):
            stop()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            summary = self.summary
            reasons = [f"{count} × {reason}" for reason, count in sorted(summary.reasons.items())]
            return {
                "volume": self.volume,
                "set": self.set_name,
                "phase": self.phase,
                "running": self.running,
                "page": self.processed,
                "pages": self.total,
                "unit": "location",
                "entitiesSeen": self.processed,
                "refreshed": summary.built,
                "unchanged": summary.skipped,
                "added": 0,
                "skipped": summary.skipped,
                "circles": 0,
                "notes": [f"Building structure {self.params.structure_id} of {self.volume}"],
                "failed": summary.failed,
                "skipLines": reasons,
                "errors": list(self.errors),
                "started": self.started,
                "finished": self.finished,
                "cancelled": self.cancelled,
                "headline": self._headline(),
                "refreshedIds": list(self.built_ids[-200:]),
                "stateSaved": True,
            }

    def _headline(self) -> str:
        if self.phase == "building":
            return f"Building structure {self.params.structure_id}"
        return {"starting": "Starting", "done": "Finished", "cancelled": "Cancelled", "failed": "Stopped"}.get(self.phase, "")

    def _progress(self, summary: builder_module.BuildSummary, record: Optional[odata.LocationRecord]) -> None:
        with self._lock:
            self.summary = summary
            self.total = max(summary.total, summary.built + summary.skipped + summary.failed)
            self.processed = summary.built + summary.skipped + summary.failed
            if record is not None and self.phase == "starting":
                self.phase = "building"

    def _run(self) -> None:
        save_state(self.crops, "running", self.params)
        try:
            client = self._make_client()
            with self._lock:
                self._client = client
            if self._cancel.is_set():
                self.cancel()
            if self.params.viking_xml:
                # The volume's own record of where its tiles are (source.json), not a guess from a template.
                volume = load_volume(self.params.viking_xml, host=self.params.tile_base or None)
            else:
                volume = self._make_volume(self.volume)
            cache = self._make_cache()
            job_id = serverreview.build_job(self.crops, client, self.params.volume, self.set_name)
            builder = builder_module.Builder(
                self.params, self.crops, volume, client, cache, get_json=self._get_json, semaphore=streams(),
                job_id=job_id,
            )
            with self._lock:
                self.phase = "building"
            summary = builder.run(progress=self._progress, cancel=self._cancel)
            with self._lock:
                self.summary = summary
                self.processed = summary.built + summary.skipped + summary.failed
                self.total = max(summary.total, self.processed)
                self.cancelled = summary.cancelled
                self.phase = "cancelled" if summary.cancelled else ("failed" if summary.failed and not summary.built and not summary.skipped else "done")
        except ConverterNotConfigured as missing:
            with self._lock:
                self.errors.append(str(missing))
                self.phase = "failed"
        except Exception as error:  # a job must always end in a reported state
            LOGGER.exception("converter build %s/%s failed", self.set_name, self.volume)
            with self._lock:
                self.errors.append(f"job stopped: {error}")
                self.phase = "failed"
        finally:
            with self._lock:
                self.finished = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                phase = self.phase
            save_state(self.crops, phase, self.params)


def current_job(set_name: str, volume: str) -> Optional[BuildJob]:
    with _JOBS_LOCK:
        return _JOBS.get((set_name, volume))


def forget_job(set_name: str, volume: str) -> None:
    """Drop the finished job of a build that was deleted, so a later build of the same structure starts clean."""
    with _JOBS_LOCK:
        existing = _JOBS.get((set_name, volume))
        if existing is not None and not existing.running:
            del _JOBS[(set_name, volume)]


def start_job(job: BuildJob) -> BuildJob:
    """Start ``job`` unless one is already running for the same set and volume (that one is returned)."""
    key = (job.set_name, job.volume)
    with _JOBS_LOCK:
        existing = _JOBS.get(key)
        if existing is not None and existing.running:
            return existing
        _JOBS[key] = job
    job.start()
    return job


def save_state(crops: Path, state: str, params: builder_module.BuildParams) -> None:
    """Record a build's state and settings, so a restart can tell a build cut short from a finished one."""
    document = {
        "state": state,
        "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "params": dataclasses.asdict(params),
    }
    try:
        crops.mkdir(parents=True, exist_ok=True)
        temporary = crops / (STATE_FILE + ".tmp")
        temporary.write_text(json.dumps(document, indent=1), encoding="utf-8")
        temporary.replace(crops / STATE_FILE)
    except OSError:
        LOGGER.exception("could not save the build state in %s", crops)


def load_params(document: Dict[str, Any]) -> builder_module.BuildParams:
    """The settings a ``job.json`` was saved with. Raises ``TypeError`` or ``ValueError`` for a document that is not."""
    raw = dict(document.get("params") or {})
    known = {item.name for item in dataclasses.fields(builder_module.BuildParams)}
    params = builder_module.BuildParams(**{key: value for key, value in raw.items() if key in known})
    return dataclasses.replace(params, type_codes=tuple(int(code) for code in params.type_codes))


def resume_unfinished(root: Optional[Path]) -> List[BuildJob]:
    """Start again every build whose ``job.json`` still says running: the server stopped while it ran."""
    if root is None or not root.is_dir():
        return []
    resumed: List[BuildJob] = []
    for state_path in sorted(root.glob(f"structure-*/*/{STATE_FILE}")):
        crops = state_path.parent
        try:
            document = json.loads(state_path.read_text(encoding="utf-8"))
            if document.get("state") != "running":
                continue
            params = load_params(document)
        except (OSError, TypeError, ValueError) as error:
            LOGGER.warning("not resuming %s: %s", state_path, error)
            continue
        set_name, volume = crops.parent.name, crops.name
        LOGGER.warning("resuming converter build %s/%s, cut short by a restart", set_name, volume)
        resumed.append(start_job(new_job(set_name, volume, crops, params)))
    return resumed


def write_source(
    crops: Path, odata_base: str, structure_id: int, viking_xml: Optional[str] = None, tile_base: Optional[str] = None
) -> None:
    """The source document the gallery reads a volume's database name from (the OData URL names it)."""
    crops.mkdir(parents=True, exist_ok=True)
    path = crops / "source.json"
    if not path.is_file():
        document: Dict[str, Any] = {"kind": "converter", "odata": odata_base, "structureId": int(structure_id)}
        if viking_xml:
            document["tiles"] = {"vikingXml": viking_xml, "base": tile_base or ""}
        path.write_text(json.dumps(document, indent=1), encoding="utf-8")


MISSING_IMAGES_SECONDS = 60.0
_MISSING_LOCK = threading.Lock()
# (when read, the lower-case names of the volumes the server reads; None when every volume can be built)
_READABLE: Optional[Tuple[float, Optional[List[Any]]]] = None


def missing_local_images(volumes: List[str]) -> List[str]:
    """The volumes a build cannot run in because the segmentation server does not read their images itself, which
    the tiled engine's SegmentLocation needs (the single-window engine uploads its own tiles). The server's answer is
    kept for a minute. A server that cannot be asked marks nothing: the build then reports why it failed."""
    held = server_volumes()
    known = None if held is None else {v.name.lower() for v in held if v.readable}
    return [volume for volume in volumes if known is not None and volume.lower() not in known]


def volume_tilesets(volume: str, viking_xml: str = "") -> List[str]:
    """The tilesets of the context the gallery works in for ``volume`` (see :func:`working_context`), fit ones first.
    Empty when the server has no such context or cannot be asked."""
    context = working_context(volume, viking_xml)
    return ordered_tilesets(getattr(context, "tilesets", ())) if context is not None else []


def working_context(volume: str, viking_xml: str = "") -> Optional[Any]:
    """The server's annotation context the gallery works in for ``volume``, so builds, segmentation and Viking links
    all use it: the annotation database's default context (set in Identity), or the only context of that name. A
    ``viking_xml`` (a volume's own source.json tiles) narrows it to the contexts reading that file. None when the server
    has no such context or cannot be asked."""
    return context_for(server_volumes() or [], volume, viking_xml)


def working_viking_xml(volume: str, environ: Optional[Dict[str, str]] = None) -> str:
    """The VikingXML the gallery reads ``volume``'s tiles from: the working context's (:func:`working_context`), so the
    cards show the images that context segments; ``GALLERY_CONVERTER_VIKINGXML`` (a URL template) when the server names
    none. Empty when neither is known."""
    context = working_context(volume)
    url = str(getattr(context, "viking_xml", "") or "").strip() if context is not None else ""
    return url or viking_xml_url(volume, environ)


def server_volumes() -> Optional[List[Any]]:
    """The segmentation server's contexts when the tiled engine is served, kept for a minute; None when every volume
    can be built (the window engine uploads its own tiles) or the server cannot be asked."""
    global _READABLE
    with _MISSING_LOCK:
        held = _READABLE
    if held is None or time.monotonic() - held[0] > MISSING_IMAGES_SECONDS:
        readable: Optional[List[Any]] = None
        try:
            client = OVERRIDES.get("make_client", default_client)()
            try:
                if engines.choose(client).kind == engines.TILED:
                    readable = list(client.status().readable_volumes)
            finally:
                close = getattr(client, "close", None)
                if callable(close):
                    close()
        except Exception as error:  # noqa: BLE001 - not configured or unreachable
            LOGGER.info("could not ask the segmentation server which volumes it reads: %s", error)
        held = (time.monotonic(), readable)
        with _MISSING_LOCK:
            _READABLE = held
    return held[1]


# Tests replace the pieces a job reaches over the network: {"make_client": ..., "make_volume": ..., "make_cache": ..., "get_json": ...}.
OVERRIDES: Dict[str, Any] = {}


def new_job(set_name: str, volume: str, crops: Path, params: builder_module.BuildParams) -> BuildJob:
    return BuildJob(set_name, volume, crops, params, **OVERRIDES)


def idle_status(set_name: str, volume: str) -> Dict[str, Any]:
    return {"volume": volume, "set": set_name, "phase": "idle", "running": False}
