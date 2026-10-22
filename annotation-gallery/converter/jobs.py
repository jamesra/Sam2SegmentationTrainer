"""Background build jobs: one per (structure set, volume), reporting in the shape of Update All's status so the page's
progress strip, Cancel and polling serve both.

A job survives a page reload (it is found again by set and volume) but not a server restart; a restart is resumed by
starting the build again, which skips every card already built for the same shape and model (see builder.should_build).
At most ``GALLERY_CONVERTER_STREAMS`` (default 1) segmentation streams run at once across all jobs, so a build cannot
starve the interactive users of the segmentation server.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from converter import builder as builder_module
from converter import odata
from converter.seg_client import SegmentationClient
from converter.tilesource import TileCache, Volume, load_volume

LOGGER = logging.getLogger("converter.jobs")

_JOBS: Dict[Tuple[str, str], "BuildJob"] = {}
_JOBS_LOCK = threading.Lock()
_STREAMS: Optional[threading.Semaphore] = None
_STREAMS_LOCK = threading.Lock()


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
    mapping = environ if environ is not None else os.environ
    # One volume can have its own address (RC1's VikingXML is not at the usual place): GALLERY_CONVERTER_VIKINGXML_RC1.
    specific = "GALLERY_CONVERTER_VIKINGXML_" + "".join(c if c.isalnum() else "_" for c in volume_name.upper())
    template = (mapping.get(specific) or mapping.get("GALLERY_CONVERTER_VIKINGXML") or "").strip()
    if not template:
        raise ConverterNotConfigured("GALLERY_CONVERTER_VIKINGXML is not set (URL template of the VikingXML, with {volume})")
    return load_volume(template.replace("{volume}", volume_name).replace("<volume>", volume_name))


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
        try:
            client = self._make_client()
            volume = self._make_volume(self.volume)
            cache = self._make_cache()
            builder = builder_module.Builder(
                self.params, self.crops, volume, client, cache, get_json=self._get_json, semaphore=streams()
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


def current_job(set_name: str, volume: str) -> Optional[BuildJob]:
    with _JOBS_LOCK:
        return _JOBS.get((set_name, volume))


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


def write_source(crops: Path, odata_base: str, structure_id: int) -> None:
    """The source document the gallery reads a volume's database name from (the OData URL names it)."""
    crops.mkdir(parents=True, exist_ok=True)
    path = crops / "source.json"
    if not path.is_file():
        path.write_text(
            json.dumps({"kind": "converter", "odata": odata_base, "structureId": int(structure_id)}, indent=1),
            encoding="utf-8",
        )


# Tests replace the pieces a job reaches over the network: {"make_client": ..., "make_volume": ..., "make_cache": ..., "get_json": ...}.
OVERRIDES: Dict[str, Any] = {}


def new_job(set_name: str, volume: str, crops: Path, params: builder_module.BuildParams) -> BuildJob:
    return BuildJob(set_name, volume, crops, params, **OVERRIDES)


def idle_status(set_name: str, volume: str) -> Dict[str, Any]:
    return {"volume": volume, "set": set_name, "phase": "idle", "running": False}
