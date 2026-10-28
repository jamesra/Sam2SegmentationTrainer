"""Per-card scores: the catalog table, its view, the history, the re-score queue, the checkpoint lookup and the
training-pass scorer."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from annotation_crops import card_scores as store


def _record(location_id: int, image_key: str, *, iou=0.9, core_hit=570, core_gt=600, spill_px=30, gt_area=1000):
    return {
        "key": ("RC1", image_key, location_id),
        "iou": iou,
        "raw_iou": iou - 0.01,
        "recall": core_hit / core_gt if core_gt else None,
        "core_hit": core_hit,
        "core_gt": core_gt,
        "spill_px": spill_px,
        "gt_area": gt_area,
        "bf10": 0.8,
    }


def _connection() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:", isolation_level=None)
    store.ensure_card_scores_schema(connection)
    return connection


def _score(location_id, image_key, *, family=store.DEFAULT_FAMILY, at="2026-10-26T10:00:00Z", epoch=4, **kwargs):
    return store.CardScore.from_record(_record(location_id, image_key, **kwargs), family=family, model=f"v5.{epoch}",
                                       scored_at=at, epoch=epoch)


def test_from_record_derives_spill_share_and_mismatch():
    score = _score(7, "k")
    assert score.location_id == 7 and score.image_key == "k" and score.epoch == 4
    assert score.spill == pytest.approx(0.03)
    assert score.core_recall == pytest.approx(0.95)
    assert score.mismatch == pytest.approx(0.05)  # missing 5% of the core is worse than 3% spill
    assert score.kind == "validation"


def test_pooled_variants_pool_pixel_counts_and_average_iou():
    records = [_record(1, "a", iou=0.8, core_hit=500, core_gt=600, spill_px=0),
               _record(1, "a", iou=1.0, core_hit=600, core_gt=600, spill_px=100)]
    score = store.CardScore.pooled(records, family=store.TRAIN_FAMILY, model="v5.6", scored_at="t", epoch=6,
                                   prompt="circle_mask,point")
    assert score.core_recall == pytest.approx(1100 / 1200)
    assert score.spill == pytest.approx(100 / 2000)
    assert score.iou == pytest.approx(0.9)
    assert score.kind == "train" and score.prompt == "circle_mask,point"


def test_no_core_is_no_recall_and_no_miss():
    score = _score(1, "a", core_hit=0, core_gt=0, spill_px=20)
    assert score.core_recall is None
    assert score.mismatch == pytest.approx(0.02)


def test_mismatch_takes_the_worse_problem():
    assert store.mismatch(0.99, 0.20) == pytest.approx(0.20)
    assert store.mismatch(0.70, 0.01) == pytest.approx(0.30)
    assert store.mismatch(None, None) == 0.0


def test_latest_prefers_validation_and_pairs_within_a_kind():
    connection = _connection()
    store.write_card_scores(connection, [
        _score(1, "a", iou=0.7, at="2026-10-26T10:00:00Z", epoch=3),
        _score(1, "a", iou=0.9, at="2026-10-26T11:00:00Z", epoch=4),
        _score(1, "a", family=store.TRAIN_FAMILY, iou=0.5, at="2026-10-26T12:00:00Z", epoch=6),
        _score(2, "b", family=store.TRAIN_FAMILY, iou=0.6, at="2026-10-26T09:00:00Z", epoch=5),
        _score(2, "b", family=store.TRAIN_FAMILY, iou=0.8, at="2026-10-26T12:00:00Z", epoch=6),
        _score(3, "c", iou=0.95),
    ])
    latest = store.latest_card_scores(connection)
    assert latest[(1, "a")]["kind"] == "validation"
    assert latest[(1, "a")]["new"]["iou"] == pytest.approx(0.9)  # validation wins over a newer training-pass score
    assert latest[(1, "a")]["old"]["iou"] == pytest.approx(0.7)
    assert latest[(2, "b")]["kind"] == "train"
    assert latest[(2, "b")]["new"]["iou"] == pytest.approx(0.8)
    assert latest[(2, "b")]["old"]["epoch"] == 5
    assert latest[(3, "c")]["old"] is None
    assert set(store.latest_card_scores(connection, [2])) == {(2, "b")}
    assert store.latest_card_scores(connection, []) == {}
    current = {(r[0], r[1]): r for r in connection.execute(
        f"SELECT location_id, image_key, kind, iou, mismatch FROM {store.CURRENT_VIEW}")}
    assert current[(1, "a")][2] == "validation" and current[(1, "a")][3] == pytest.approx(0.9)
    assert current[(2, "b")][2] == "train" and current[(2, "b")][3] == pytest.approx(0.8)
    assert current[(3, "c")][4] == pytest.approx(store.mismatch(0.95, 0.03))


def test_history_lists_both_kinds_and_the_loss():
    connection = _connection()
    connection.execute("CREATE TABLE location_scores (location_id INTEGER, image_key TEXT, epoch INTEGER, score REAL)")
    connection.executemany("INSERT INTO location_scores VALUES (1, 'a', ?, ?)", [(4, 0.05), (5, 0.04)])
    store.write_card_scores(connection, [
        _score(1, "a", at="2026-10-26T10:00:00Z", epoch=4),
        _score(1, "a", family=store.TRAIN_FAMILY, at="2026-10-26T11:00:00Z", epoch=5),
        _score(2, "b"),
    ])
    history = store.card_score_history(connection, 1, "a")
    assert [p["epoch"] for p in history["validation"]] == [4]
    assert [p["epoch"] for p in history["train"]] == [5]
    assert history["loss"] == [{"epoch": 4, "score": 0.05}, {"epoch": 5, "score": 0.04}]


def test_an_older_table_gains_the_new_columns():
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE card_scores (location_id INTEGER NOT NULL, image_key TEXT NOT NULL, family TEXT NOT NULL, "
        "model TEXT NOT NULL, scored_at TEXT NOT NULL, iou REAL, raw_iou REAL, core_recall REAL, spill REAL, "
        "bf10 REAL, core_gt INTEGER, gt_area INTEGER, PRIMARY KEY (location_id, image_key, family, scored_at, model))"
    )
    store.ensure_card_scores_schema(connection)
    columns = {row[1] for row in connection.execute("PRAGMA table_info(card_scores)")}
    assert {"epoch", "prompt"} <= columns
    store.write_card_scores(connection, [_score(1, "a")])
    assert store.latest_card_scores(connection)[(1, "a")]["new"]["epoch"] == 4


def test_mismatch_sql_matches_python():
    connection = _connection()
    rows = [(0.99, 0.2), (0.7, 0.01), (None, 0.02)]
    for i, (recall, spill) in enumerate(rows):
        connection.execute(
            "INSERT INTO card_scores (location_id, image_key, family, model, scored_at, core_recall, spill) "
            "VALUES (?, 'k', 'f', 'm', 't', ?, ?)",
            (i, recall, spill),
        )
    sql = f"SELECT location_id, {store.MISMATCH_SQL.format(p='')} FROM card_scores ORDER BY location_id"
    for (_location_id, value), (recall, spill) in zip(connection.execute(sql), rows):
        assert value == pytest.approx(store.mismatch(recall, spill))


def test_queue_claims_once_and_finishes():
    connection = _connection()
    first = store.queue_location(connection, 42)
    assert store.queue_location(connection, 42) == first  # a waiting location is not queued twice
    assert store.request_status(connection, 42)["state"] == "pending"
    assert store.claim_requests(connection) == [(first, 42)]
    assert store.claim_requests(connection) == []
    assert store.request_status(connection, 42)["state"] == "running"
    store.finish_request(connection, first, model="v5.4", cards=3)
    status = store.request_status(connection, 42)
    assert status["state"] == "done" and status["cards"] == 3 and status["model"] == "v5.4"
    assert store.queue_location(connection, 42) != first  # done: a new request
    assert store.request_status(connection, 7) is None


def test_failed_request_keeps_the_error_and_running_requests_are_requeued():
    connection = _connection()
    a = store.queue_location(connection, 1)
    store.queue_location(connection, 2)
    store.claim_requests(connection)
    store.finish_request(connection, a, model=None, cards=0, error="RuntimeError: boom")
    assert store.request_status(connection, 1)["state"] == "failed"
    assert store.request_status(connection, 1)["error"] == "RuntimeError: boom"
    assert store.requeue_running(connection) == 1
    assert store.request_status(connection, 2)["state"] == "pending"


def _run(tmp_path: Path, best_epoch: int, epochs=(0, 1, 2)) -> Path:
    run = tmp_path / "runs" / "sam2_em_v9"
    serve = run / "serve"
    serve.mkdir(parents=True)
    for epoch in epochs:
        (serve / f"serve_epoch{epoch}.pt").write_bytes(b"x")
        (serve / f"serve_epoch{epoch}.model.json").write_text(json.dumps({"model": {"model_version": f"v9.{epoch}"}}))
    (run / "best_model.json").write_text(json.dumps({"epoch": best_epoch}))
    (run / "config.yaml").write_text("training:\n  seed: 1\n")
    return run


def test_resolve_checkpoint_best_latest_and_path(tmp_path):
    from sam2_segmentation_trainer.card_scores import model_label, resolve_checkpoint, run_dir_of

    run = _run(tmp_path, best_epoch=1)
    best = resolve_checkpoint("best:sam2_em_v9", ckpt_root=tmp_path)
    assert best.name == "serve_epoch1.pt" and model_label(best) == "v9.1"
    assert resolve_checkpoint("latest:sam2_em_v9", ckpt_root=tmp_path).name == "serve_epoch2.pt"
    assert resolve_checkpoint(str(best)) == best
    assert run_dir_of(best) == run
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(str(tmp_path / "missing.pt"))


def test_best_that_is_not_exported_yet_is_an_error(tmp_path):
    from sam2_segmentation_trainer.card_scores import resolve_checkpoint

    _run(tmp_path, best_epoch=5)
    with pytest.raises(FileNotFoundError, match="not exported"):
        resolve_checkpoint("best:sam2_em_v9", ckpt_root=tmp_path)


def test_model_labels(tmp_path):
    from sam2_segmentation_trainer.card_scores import model_label, run_model_label

    path = tmp_path / "custom.pt"
    path.write_bytes(b"x")
    assert model_label(path) == "custom.pt"
    assert run_model_label("sam2_em_v5", 6) == "v5.6"


def _catalog(tmp_path: Path, volume: str = "RC1") -> Path:
    folder = tmp_path / volume
    folder.mkdir(parents=True)
    path = folder / "annotation_crops.sqlite"
    sqlite3.connect(str(path)).close()
    return path


def test_training_pass_scores_pool_a_cards_variants_and_write_them(tmp_path):
    import torch

    from sam2_segmentation_trainer.card_scores import TrainPassScores

    catalog = _catalog(tmp_path)
    gt = torch.zeros(1, 96, 96, dtype=torch.uint8)
    gt[0, 20:76, 20:76] = 1
    good = torch.full((96, 96), -8.0)
    good[20:76, 20:76] = 8.0
    small = torch.full((96, 96), -8.0)
    small[30:66, 30:66] = 8.0
    meta = {"volume": "RC1", "location_id": 5, "image_key": "k"}
    batch = {
        "mask": gt,
        "variant_index": torch.tensor([0, 0]),
        "meta": [meta, meta],
        "prompt_type": ["point", "circle_mask"],
    }
    scores = TrainPassScores(tmp_path, run_name="sam2_em_v5", threshold=0.5, band=4.0, threads=1)
    scores.submit(torch.stack([good, small]), batch, epoch=6)
    scores.close()
    assert scores.written == 1
    connection = sqlite3.connect(str(catalog))
    latest = store.latest_card_scores(connection)
    new = latest[(5, "k")]["new"]
    assert latest[(5, "k")]["kind"] == "train"
    assert new["model"] == "v5.6" and new["epoch"] == 6 and new["prompt"] == "circle_mask,point"
    assert 0.0 < new["core_recall"] < 1.0  # the small prediction misses part of the core
    assert new["spill"] == pytest.approx(0.0)


def test_training_pass_scores_skip_steps_when_behind(tmp_path):
    import torch

    from sam2_segmentation_trainer.card_scores import TrainPassScores

    _catalog(tmp_path)
    batch = {"mask": torch.ones(1, 8, 8), "variant_index": torch.tensor([0]),
             "meta": [{"volume": "RC1", "location_id": 1, "image_key": "a"}], "prompt_type": ["point"]}
    scores = TrainPassScores(tmp_path, run_name="sam2_em_v5", threshold=0.5, band=1.0, threads=1, max_backlog=1)
    scores.submit(torch.ones(1, 8, 8), batch, epoch=0)
    scores.submit(torch.ones(1, 8, 8), batch, epoch=0)
    assert scores.skipped == 1
    scores.close()


def test_validation_records_become_validation_scores(tmp_path):
    from sam2_segmentation_trainer.card_scores import record_validation_card_scores

    catalog = _catalog(tmp_path)
    records = {"circle_annotator": [_record(1, "a"), _record(2, "b")], "point": [_record(3, "c")]}
    written = record_validation_card_scores(tmp_path, records, family="circle_annotator", model="v5.6", epoch=6)
    assert written == 2
    latest = store.latest_card_scores(sqlite3.connect(str(catalog)))
    assert set(latest) == {(1, "a"), (2, "b")}
    assert latest[(1, "a")]["kind"] == "validation" and latest[(1, "a")]["new"]["epoch"] == 6


def test_the_heartbeat_says_whether_a_scorer_serves_the_catalog():
    connection = _connection()
    assert store.scorer_alive(connection) is None
    store.beat(connection, spec="best:sam2_em_v5", model="v5.4")
    alive = store.scorer_alive(connection)
    assert alive["spec"] == "best:sam2_em_v5" and alive["model"] == "v5.4"
    connection.execute("UPDATE card_scorer_heartbeat SET seen_at = '2026-01-01T00:00:00Z'")
    assert store.scorer_alive(connection) is None


def test_an_idle_scorer_parks_its_model_in_cpu_memory():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("parking frees GPU memory, so it needs CUDA")
    from sam2_segmentation_trainer.card_scores import CardScorer

    scorer = CardScorer(spec="unused")
    scorer.device = torch.device("cuda")
    scorer.model = torch.nn.Linear(4, 4).to(scorer.device)
    scorer.unload(park=True)
    assert scorer.model is None
    assert next(scorer.parked.parameters()).device.type == "cpu"
    scorer.unload()
    assert scorer.parked is None
