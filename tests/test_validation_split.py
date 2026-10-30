"""validation_split.json: written by a run (and publish_split) beside each volume's crops, read by the gallery."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

from annotation_crops.validation_split import FILE_NAME, ids_by_volume, read_split_file, write_split_file


def test_round_trip_sorts_and_dedupes(tmp_path: Path) -> None:
    write_split_file(tmp_path, run="sam2_em_v5", mode="zslab", validation=[5, 3, 3], training=[9, 1])
    payload = read_split_file(tmp_path)
    assert payload["run"] == "sam2_em_v5" and payload["mode"] == "zslab" and payload["writtenAt"].endswith("Z")
    assert payload["validation"] == [3, 5] and payload["training"] == [1, 9]
    assert not list(tmp_path.glob("*.tmp")), "the temp file is replaced into place"


def test_missing_or_corrupt_file_reads_as_none(tmp_path: Path) -> None:
    assert read_split_file(tmp_path) is None
    (tmp_path / FILE_NAME).write_text("{not json", encoding="utf-8")
    assert read_split_file(tmp_path) is None
    (tmp_path / FILE_NAME).write_text("[1, 2]", encoding="utf-8")
    assert read_split_file(tmp_path) is None
    (tmp_path / FILE_NAME).write_text(json.dumps({"validation": ["x"]}), encoding="utf-8")
    assert read_split_file(tmp_path) is None


def test_split_keys_group_by_volume() -> None:
    assert ids_by_volume(["RC1:12", "RC1:7", "RPC2:3", "junk", "RC2:x"]) == {"RC1": {12, 7}, "RPC2": {3}}


def test_publish_writes_each_volume_from_the_run_split(tmp_path: Path, monkeypatch) -> None:
    from sam2_segmentation_trainer import publish_split

    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "split.json").write_text(json.dumps({
        "mode": "zslab", "train_keys": ["RC1:1", "RC1:2", "RC2:9"], "val_keys": ["RC1:3"],
    }), encoding="utf-8")
    root = tmp_path / "data"
    OmegaConf.save(OmegaConf.create({"data": {"root": str(root)}}), run_dir / "config.yaml")
    for volume in ("RC1", "RC2"):
        (root / volume).mkdir(parents=True)
    monkeypatch.setattr(publish_split, "run_dir_for", lambda run: run_dir)
    monkeypatch.setattr(publish_split, "annotation_crops_root", lambda volume, root_: Path(root_) / volume)
    assert publish_split.publish("r1", root) == {"RC1": (1, 2), "RC2": (0, 1)}
    assert read_split_file(root / "RC1")["validation"] == [3]
    assert read_split_file(root / "RC2") == {**read_split_file(root / "RC2"), "run": "r1", "validation": [], "training": [9]}


def test_a_run_publishes_its_split_and_a_failure_only_warns(tmp_path: Path, monkeypatch, capsys) -> None:
    from sam2_segmentation_trainer import train

    monkeypatch.setattr(train, "annotation_crops_root", lambda volume, data_dir: Path(data_dir) / volume)
    (tmp_path / "RC1").mkdir()
    ex = lambda volume, loc: SimpleNamespace(volume=volume, location_id=loc)  # noqa: E731
    cfg = OmegaConf.create({"data": {"split_mode": "zslab"}})
    train._publish_split(tmp_path / "sam2_em_v9", tmp_path, ["RC1", "RC2"],
                         [ex("RC1", 1), ex("RC1", 2), ex("RC2", 5)], [ex("RC1", 3)], cfg)
    payload = read_split_file(tmp_path / "RC1")
    assert (payload["run"], payload["mode"], payload["validation"], payload["training"]) == ("sam2_em_v9", "zslab", [3], [1, 2])
    assert "could not publish the validation split of RC2" in capsys.readouterr().out  # no RC2 folder: warns only
