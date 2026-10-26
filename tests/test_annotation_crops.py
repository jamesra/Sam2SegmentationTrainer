"""Leaf formats: ignore.json, mask names, location_scores, and COCO RLE."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from annotation_crops.ignore import load_ignore_ids, save_ignore_ids
from annotation_crops.maskname import MaskName
from annotation_crops.rle import bbox_area_from_rle, decode_coco_rle, encode_coco_rle
from annotation_crops.scores import ensure_location_scores_schema


def test_ignore_round_trip_and_bad_json(tmp_path: Path, capsys) -> None:
    save_ignore_ids(tmp_path, [3, 1, 3])
    assert json.loads((tmp_path / "ignore.json").read_text(encoding="utf-8")) == [1, 3]
    assert load_ignore_ids(tmp_path) == {1, 3}
    (tmp_path / "ignore.json").write_text("{", encoding="utf-8")
    assert load_ignore_ids(tmp_path) == set()
    assert "unreadable ignore list" in capsys.readouterr().out


def test_mask_name_format_and_parse() -> None:
    name = MaskName("RC2_17_D1_X0_Y0", 42)
    assert name.format() == "RC2_17_D1_X0_Y0_42.png"
    assert MaskName.parse(name.format()) == name
    assert MaskName.parse("not-a-mask.jpg") is None


def test_location_scores_legacy_rows_keep_empty_image_key(tmp_path: Path) -> None:
    path = tmp_path / "annotation_crops.sqlite"
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE location_scores (
            location_id INTEGER NOT NULL,
            epoch INTEGER NOT NULL,
            score REAL NOT NULL,
            PRIMARY KEY (location_id, epoch)
        )
        """
    )
    connection.execute("INSERT INTO location_scores VALUES (7, 1, 0.5)")
    ensure_location_scores_schema(connection)
    row = connection.execute(
        "SELECT location_id, image_key, epoch, score FROM location_scores"
    ).fetchone()
    assert row == (7, "", 1, 0.5)
    connection.close()


def _scores(connection: sqlite3.Connection, table: str = "location_scores") -> list[tuple]:
    return connection.execute(
        f"SELECT location_id, image_key, epoch, score FROM {table} ORDER BY location_id, image_key, epoch"
    ).fetchall()


def test_a_new_run_supersedes_an_older_runs_later_epochs(tmp_path: Path) -> None:
    connection = sqlite3.connect(tmp_path / "annotation_crops.sqlite")
    ensure_location_scores_schema(connection)
    # An older run reached epoch 11 on window K of location 7, and epoch 3 on window L.
    for epoch in range(12):
        connection.execute("INSERT INTO location_scores VALUES (7, 'K', ?, ?)", [epoch, 1.0 + epoch])
    connection.execute("INSERT INTO location_scores VALUES (7, 'L', 3, 9.0)")

    # A new run writes epoch 0 then 1 for window K.
    connection.execute("INSERT OR REPLACE INTO location_scores VALUES (7, 'K', 0, 0.5)")
    connection.execute("INSERT OR REPLACE INTO location_scores VALUES (7, 'K', 1, 0.4)")

    assert _scores(connection) == [(7, "K", 0, 0.5), (7, "K", 1, 0.4), (7, "L", 3, 9.0)], "only window K changes"
    moved = _scores(connection, "location_scores_superseded")
    assert [row[2] for row in moved] == list(range(1, 12)), "the older run's epochs 1-11 are kept, not lost"
    assert all(row[1] == "K" for row in moved)
    stamps = connection.execute("SELECT DISTINCT superseded_at FROM location_scores_superseded").fetchall()
    assert stamps and all(stamp[0].endswith("Z") for stamp in stamps)
    connection.close()


def test_one_run_writing_epochs_in_order_supersedes_nothing(tmp_path: Path) -> None:
    connection = sqlite3.connect(tmp_path / "annotation_crops.sqlite")
    ensure_location_scores_schema(connection)
    for epoch in range(4):
        connection.execute("INSERT OR REPLACE INTO location_scores VALUES (7, 'K', ?, ?)", [epoch, float(epoch)])
    # The same window twice in one epoch keeps the later score.
    connection.execute("INSERT OR REPLACE INTO location_scores VALUES (7, 'K', 3, 0.25)")

    assert _scores(connection) == [(7, "K", 0, 0.0), (7, "K", 1, 1.0), (7, "K", 2, 2.0), (7, "K", 3, 0.25)]
    assert _scores(connection, "location_scores_superseded") == []
    connection.close()


def test_the_schema_can_be_ensured_again_and_after_the_legacy_migration(tmp_path: Path) -> None:
    connection = sqlite3.connect(tmp_path / "annotation_crops.sqlite")
    connection.execute(
        "CREATE TABLE location_scores (location_id INTEGER NOT NULL, epoch INTEGER NOT NULL, score REAL NOT NULL, "
        "PRIMARY KEY (location_id, epoch))"
    )
    connection.execute("INSERT INTO location_scores VALUES (7, 5, 0.5)")
    ensure_location_scores_schema(connection)
    ensure_location_scores_schema(connection)

    connection.execute("INSERT INTO location_scores VALUES (7, '', 0, 0.1)")
    assert _scores(connection) == [(7, "", 0, 0.1)], "the migrated table has the trigger too"
    assert _scores(connection, "location_scores_superseded") == [(7, "", 5, 0.5)]
    triggers = connection.execute("SELECT COUNT(*) FROM sqlite_master WHERE type = 'trigger'").fetchone()[0]
    assert triggers == 1
    connection.close()


def test_rle_round_trip_and_bbox() -> None:
    mask = np.zeros((3, 4), dtype=np.uint8)
    mask[1, 2] = 5
    encoded = encode_coco_rle(mask)
    decoded = decode_coco_rle(encoded)
    assert np.array_equal(decoded, (mask > 0).astype(np.uint8))
    bbox, area = bbox_area_from_rle(encoded)
    assert bbox == [2, 1, 1, 1]
    assert area == 1
    empty = encode_coco_rle(np.zeros((0, 0), dtype=np.uint8))
    assert empty["counts"] == []
    foreground = np.zeros((2, 2), dtype=np.uint8)
    foreground[0, 0] = 1
    leading = encode_coco_rle(foreground)
    assert leading["counts"][0] == 0


def test_bbox_area_from_rle_matches_pixel_scan_for_multi_column_runs() -> None:
    mask = np.zeros((5, 6), dtype=np.uint8)
    mask.ravel(order="F")[8:19] = 1  # starts mid-column 1, spans columns 1-3
    mask[4, 5] = 1
    bbox, area = bbox_area_from_rle(encode_coco_rle(mask))
    rows, cols = np.nonzero(mask)
    assert bbox == [
        int(cols.min()),
        int(rows.min()),
        int(cols.max() - cols.min() + 1),
        int(rows.max() - rows.min() + 1),
    ]
    assert area == int(mask.sum())
    assert bbox_area_from_rle({"counts": [0, 4], "size": [0, 0]}) == ([0, 0, 0, 0], 0)
