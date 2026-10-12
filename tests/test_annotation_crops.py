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
