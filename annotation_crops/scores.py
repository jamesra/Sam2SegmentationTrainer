"""Training-loss table ``location_scores``. Not the ``locations.sam2*`` columns.

One row per crop window per epoch, written by the training run as it goes; the gallery shows each window's row of the
highest epoch. A data folder outlives runs (a new version is cloned from the last and keeps its scores), so a new run
starts at epoch 0 under rows an older run wrote for epochs it has not reached yet, and those would win. A run writes
its epochs in increasing order, so a window's row of a higher epoch than the one just written can only be an older
run's: the trigger below moves such rows to ``location_scores_superseded`` (kept, with when) as each new score lands.
"""

from __future__ import annotations

import sqlite3

_SCORE_SQL = """
CREATE TABLE IF NOT EXISTS location_scores (
    location_id INTEGER NOT NULL,
    image_key TEXT NOT NULL,
    epoch INTEGER NOT NULL,
    score REAL NOT NULL,
    PRIMARY KEY (location_id, image_key, epoch)
)
"""

_SUPERSEDED_SQL = """
CREATE TABLE IF NOT EXISTS location_scores_superseded (
    location_id INTEGER NOT NULL,
    image_key TEXT NOT NULL,
    epoch INTEGER NOT NULL,
    score REAL NOT NULL,
    superseded_at TEXT NOT NULL
)
"""

SUPERSEDE_TRIGGER = "location_scores_supersede_later_epochs"

_TRIGGER_SQL = f"""
CREATE TRIGGER IF NOT EXISTS {SUPERSEDE_TRIGGER}
AFTER INSERT ON location_scores
BEGIN
    INSERT INTO location_scores_superseded (location_id, image_key, epoch, score, superseded_at)
    SELECT location_id, image_key, epoch, score, strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
    FROM location_scores
    WHERE location_id = NEW.location_id AND image_key = NEW.image_key AND epoch > NEW.epoch;
    DELETE FROM location_scores
    WHERE location_id = NEW.location_id AND image_key = NEW.image_key AND epoch > NEW.epoch;
END
"""


def _ensure_superseding(connection: sqlite3.Connection) -> None:
    connection.execute(_SUPERSEDED_SQL)
    connection.execute(_TRIGGER_SQL)


def ensure_location_scores_schema(connection: sqlite3.Connection) -> None:
    """Create per-window scores, or copy a location-only table under an empty image key; and the superseding trigger.

    Older catalogs stored one score per location per epoch. Those rows stay
    readable with ``image_key=''`` until a later epoch writes the window.
    """
    connection.execute(_SCORE_SQL)
    columns = {row[1] for row in connection.execute("PRAGMA table_info(location_scores)")}
    if "image_key" in columns:
        _ensure_superseding(connection)
        return
    connection.execute(
        """
        CREATE TABLE location_scores_window (
            location_id INTEGER NOT NULL,
            image_key TEXT NOT NULL,
            epoch INTEGER NOT NULL,
            score REAL NOT NULL,
            PRIMARY KEY (location_id, image_key, epoch)
        )
        """
    )
    connection.execute(
        """
        INSERT INTO location_scores_window (location_id, image_key, epoch, score)
        SELECT location_id, '', epoch, score FROM location_scores
        """
    )
    connection.execute("DROP TABLE location_scores")
    connection.execute("ALTER TABLE location_scores_window RENAME TO location_scores")
    _ensure_superseding(connection)
