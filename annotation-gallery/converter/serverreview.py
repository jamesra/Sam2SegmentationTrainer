"""What the converter tells the segmentation server's cache beyond the requests themselves.

- **A job per build.** The server keeps every result linked to a job (``SegmentTilesRequest.job_id``) until the job is
  deleted, so a build's proposals are still cached when Viking opens one of its cards later. The job id is kept in the
  build's folder and reused by every later run of the build; deleting the build deletes the job.
- **Reviewed outlines.** When a reviewer approves or retraces a card, its outline is stored on the server
  (``PutReviewedOutlines``) against the LastModified the card was built from; the server sends it with every later
  segmentation of that location until the location is edited, so Viking shows the reviewed outline as the proposal.
  Unapproving or rejecting the card removes it.

Both are best effort: a server that is down or refuses only costs the cache, never the build or the review.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import catalog
from converter import geo
from converter.seg_client import polygon_shape
from seg_proto import segmentation_pb2 as pb

LOGGER = logging.getLogger("converter.serverreview")

JOB_FILE = "segmentation_job.json"
APPROVED, RETRACED = "approved", "retraced"
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


def reviewed_outline(crops: Path, location_id: int, image_key: Optional[str], reviewer: str) -> Optional["pb.ReviewedOutline"]:
    """The card's current outline as a reviewed outline: "retraced" when a reviewer drew it, else "approved". None when
    the card has no outline or was built before cards recorded the location's LastModified."""
    connection = catalog.connect(crops)
    try:
        query = "SELECT candidate_wkt, reason, last_modified_ticks FROM converter_cards WHERE location_id = ? AND state = 'built'"
        arguments: list = [location_id]
        if image_key:
            query += " AND image_key = ?"
            arguments.append(image_key)
        row = connection.execute(query + " ORDER BY built_at DESC LIMIT 1", arguments).fetchone()
    except Exception as error:  # noqa: BLE001 - a folder without converter cards
        LOGGER.debug("no converter card for %s: %s", location_id, error)
        row = None
    finally:
        connection.close()
    if row is None or not row["candidate_wkt"] or not row["last_modified_ticks"]:
        return None
    polygons = geo.parse_wkt_polygons(row["candidate_wkt"])
    if not polygons:
        return None
    exterior, *holes = polygons[0]
    return pb.ReviewedOutline(
        location_id=location_id, last_modified_ticks=int(row["last_modified_ticks"]),
        outline=polygon_shape(exterior, holes), state=RETRACED if row["reason"] == RETRACED else APPROVED,
        reviewed_ticks=_now_ticks(), reviewer=reviewer,
    )


def publish(
    crops: Path,
    volume: str,
    location_id: int,
    image_key: Optional[str],
    *,
    keep: bool,
    reviewer: str,
    make_client: Callable[[], Any],
) -> Optional[threading.Thread]:
    """Store the card's outline on the server (``keep``: approved or retraced) or remove it (unapproved, rejected).
    Runs in the background so a slow server never holds up the review; returns the thread (tests join it)."""
    outline = reviewed_outline(crops, location_id, image_key, reviewer) if keep else None
    if keep and outline is None:
        return None

    def run() -> None:
        client = None
        try:
            client = make_client()
            if outline is not None:
                client.put_reviewed_outlines(volume, [outline])
            else:
                client.put_reviewed_outlines(volume, [], [location_id])
        except Exception as error:  # noqa: BLE001
            LOGGER.warning("reviewed outline of %s not %s on the server: %s", location_id, "stored" if keep else "removed", error)
        finally:
            if client is not None and hasattr(client, "close"):
                client.close()

    thread = threading.Thread(target=run, name=f"reviewed-{location_id}", daemon=True)
    thread.start()
    return thread
