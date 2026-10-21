"""Frozen validation splits and gold sets (data/valsets.py, the sam2-em-valsets CLI, the gold validation row)."""

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from omegaconf import OmegaConf

from sam2_segmentation_trainer import valsets as cli
from sam2_segmentation_trainer.data.manifest import CropExample
from sam2_segmentation_trainer.data.splits import load_split, make_zslab_split
from sam2_segmentation_trainer.data.valsets import (
    GoldSet,
    classify_candidates,
    freeze_split,
    frozen_info,
    gold_from_payload,
    load_gold,
    new_gold_payload,
    partition_with_gold,
    read_gold_payload,
    resolve_gold_path,
    resolve_split_path,
    write_gold_payload,
)
from sam2_segmentation_trainer.records import RunInputs
from sam2_segmentation_trainer.validation import (
    log_results,
    mark_gold_records,
    select_validation_subset,
    summarize_mode,
)


def _example(volume: str, z: int, loc: int, key: str = "") -> CropExample:
    return CropExample(
        volume=volume, z=z, downsample=1, image_key=key or f"{volume}_{z}_k", location_id=loc,
        image_path=Path("i.png"), json_path=Path("i.json"), mask_path=Path("m.png"), split_key=f"{volume}:{loc}",
    )


def _examples(volume: str, zs, per_section: int = 10, start_id: int = 0) -> list[CropExample]:
    out, loc = [], start_id
    for z in zs:
        for _ in range(per_section):
            out.append(_example(volume, z, loc))
            loc += 1
    return out


def _record(volume: str, loc: int, iou: float = 0.9, hit: int = 95) -> dict:
    return {
        "volume": volume, "category": "MC", "key": (volume, f"{volume}_k", loc), "iou": iou, "raw_iou": iou,
        "bf10": 0.8, "bf20": 0.9, "gt_depth": 40.0, "core_gt": 100, "core_hit": hit, "recall": hit / 100,
        "pred_area": 1000, "leak_px": 0, "gt_area": 1000, "spill_px": 0, "merge": False, "frag": False,
    }


class Writer:
    def __init__(self) -> None:
        self.tags: dict[str, float] = {}

    def add_scalar(self, tag, value, step) -> None:
        self.tags[tag] = value


class FrozenSplitTests(unittest.TestCase):
    def test_freeze_writes_a_readable_split_and_never_overwrites(self) -> None:
        split = make_zslab_split(_examples("V", range(100)), val_fraction=0.1, slab_count=2, guard_sections=1)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "splits" / "val_v1.json"
            freeze_split(split, path, name="val_v1", source="sam2_em_v5")
            again = load_split(path)
            self.assertEqual(again.val_keys, split.val_keys)
            self.assertEqual(again.val_z, split.val_z)
            self.assertEqual(again.guard_z, split.guard_z)
            info = frozen_info(path)
            self.assertEqual((info["name"], info["source"]), ("val_v1", "sam2_em_v5"))
            self.assertEqual(info["val_locations"], {"V": len(split.val_keys)})
            with self.assertRaises(FileExistsError):
                freeze_split(split, path, name="val_v1", source="other")

    def test_a_new_run_places_later_locations_by_section(self) -> None:
        from sam2_segmentation_trainer.train import start_from_frozen_split

        examples = _examples("V", range(100))
        split = make_zslab_split(examples, val_fraction=0.1, slab_count=2, guard_sections=1)
        val_z, guard_z = split.val_z["V"][0], split.guard_z["V"][0]
        train_z = next(z for z in range(100) if z not in split.val_z["V"] and z not in split.guard_z["V"])
        fresh = [_example("V", val_z, 9001), _example("V", guard_z, 9002), _example("V", train_z, 9003)]
        cfg = OmegaConf.create({"data": {"train_split": 0.9}, "training": {"seed": 1}})
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "val_v1.json"
            freeze_split(split, path, name="val_v1", source="test")
            with redirect_stdout(io.StringIO()) as out:
                grown = start_from_frozen_split(path, examples + fresh, cfg, ["V", "W"])
        self.assertIn("V:9001", grown.val_keys)
        self.assertNotIn("V:9002", grown.val_keys + grown.train_keys)  # guard sections are used by neither side
        self.assertIn("V:9003", grown.train_keys)
        self.assertEqual(set(split.val_keys) - set(grown.val_keys), set())  # frozen membership stays put
        self.assertIn("no validation sections for ['W']", out.getvalue())

    def test_names_resolve_under_the_valsets_root_and_paths_pass_through(self) -> None:
        root = Path("/vs")
        self.assertEqual(resolve_split_path("val_v1", root), root / "splits" / "val_v1.json")
        self.assertEqual(resolve_gold_path("gold_v1.1", root), root / "gold" / "gold_v1.1.json")
        self.assertEqual(resolve_split_path("/runs/x/split.json", root), Path("/runs/x/split.json"))
        with self.assertRaises(ValueError):
            resolve_gold_path("bad name", root)


class GoldFileTests(unittest.TestCase):
    def test_draft_round_trip_and_final_is_immutable(self) -> None:
        payload = new_gold_payload(name="gold_v1", split="val_v1", candidates={"RC1": [3, 1, 2]}, per_volume=3, seed=0)
        self.assertEqual(payload["volumes"]["RC1"]["pending"], [1, 2, 3])
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "gold_v1.json"
            write_gold_payload(path, payload, create=True)
            with self.assertRaises(FileExistsError):
                write_gold_payload(path, payload, create=True)
            payload["volumes"]["RC1"].update(gold=[1, 2], pending=[3])
            write_gold_payload(path, payload)
            gold = load_gold(path)
            self.assertEqual(gold.locations, frozenset({("RC1", 1), ("RC1", 2)}))
            self.assertEqual((gold.status, gold.pending), ("draft", 1))
            payload["status"] = "final"
            write_gold_payload(path, payload)
            with self.assertRaises(PermissionError):
                write_gold_payload(path, payload)

    def test_a_non_gold_file_is_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.json"
            path.write_text(json.dumps({"kind": "something_else"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                read_gold_payload(path)

    def test_candidates_are_sorted_by_their_review(self) -> None:
        out = classify_candidates(
            [1, 2, 3, 4, 5, 6],
            windows={1: {"a"}, 2: {"a", "b"}, 3: {"a", "b"}, 4: {"a"}, 5: {"a"}},  # 6 has no windows left
            approved_windows={(2, "a"), (2, "b"), (3, "a")},
            approved_whole={1},
            ignored={4},
        )
        self.assertEqual(out, {"gold": [1, 2], "rejected": [4, 6], "pending": [3, 5]})


class GoldRowTests(unittest.TestCase):
    GOLD = GoldSet(name="g", status="final", locations=frozenset({("RC1", 1), ("RC2", 1)}))

    def test_gold_locations_never_train_and_unscored_ones_are_reported(self) -> None:
        train = [_example("RC1", 0, 1), _example("RC1", 0, 2)]
        val = [_example("RC1", 5, 3)]
        lines: list[str] = []
        train_out, val_out = partition_with_gold(self.GOLD, train, val, log=lines.append)
        self.assertEqual([e.location_id for e in train_out], [2])
        self.assertEqual(val_out, val)
        self.assertTrue(any("in the training split" in line for line in lines))
        self.assertTrue(any("not in the validation split" in line for line in lines))  # RC1:1 and RC2:1

    def test_no_gold_changes_nothing(self) -> None:
        train, val = [_example("RC1", 0, 1)], [_example("RC1", 5, 3)]
        self.assertEqual(partition_with_gold(None, train, val), (train, val))

    def test_the_subset_always_holds_every_window_of_a_gold_location(self) -> None:
        examples = [_example("RC1", z, loc, key=f"k{loc}_{z}") for loc in range(200) for z in (0, 1)]
        subset = select_validation_subset(examples, 0.01, seed=0, gold=self.GOLD)
        gold_windows = [e for e in subset if e.volume == "RC1" and e.location_id == 1]
        self.assertEqual(len(gold_windows), 2)

    def test_gold_records_get_their_own_row(self) -> None:
        records = {"circle_annotator": [_record("RC1", 1, 0.5, 80), _record("RC2", 1, 0.7, 90), _record("RC1", 9)]}
        mark_gold_records(records, self.GOLD)
        result = summarize_mode(records["circle_annotator"])
        self.assertEqual(result.gold_n, 2)
        self.assertAlmostEqual(result.gold_iou, 0.6)
        self.assertEqual(result.gold_by_volume, {"RC1": 1, "RC2": 1})
        self.assertAlmostEqual(result.gold.core_recall, 170 / 200)
        writer = Writer()
        text = log_results(writer, 0, {"circle_annotator": result}, "circle_annotator", root="val_subset")
        self.assertEqual(writer.tags["val_subset_gold/n/circle_annotator"], 2)
        self.assertAlmostEqual(writer.tags["val_subset_gold/iou/circle_annotator"], 0.6)
        self.assertIn("val_subset_gold/core_recall/circle_annotator", writer.tags)
        self.assertEqual(writer.tags["val_subset_gold_n/circle_annotator/RC2"], 1)
        self.assertIn("gold (n=2)", text)

    def test_without_gold_no_gold_tags_are_written(self) -> None:
        writer = Writer()
        log_results(writer, 0, {"circle_annotator": summarize_mode([_record("RC1", 9)])}, "circle_annotator")
        self.assertFalse([tag for tag in writer.tags if "_gold" in tag])


class RunInputsTests(unittest.TestCase):
    def _inputs(self, **extra) -> RunInputs:
        return RunInputs(
            run_name="r", data_dir=Path("/d"), volumes=["RC1"], even_per_volume=False, split_path=Path("/s"),
            usable_counts={"RC1": 2}, train_counts={"RC1": 1}, val_counts={"RC1": 1}, seed=0, train_fraction=0.9,
            image_size=1024, batch_size=1, num_epochs=1, model_config="m", pretrained=Path("/p"), **extra,
        )

    def test_valset_fields_appear_only_when_set(self) -> None:
        self.assertNotIn("frozen_split", self._inputs().as_json(created_utc="t"))
        doc = self._inputs(frozen_split="/vs/splits/val_v1.json", gold_set="/vs/gold/gold_v1.json").as_json(created_utc="t")
        self.assertEqual((doc["frozen_split"], doc["gold_set"]), ("/vs/splits/val_v1.json", "/vs/gold/gold_v1.json"))


class CliTests(unittest.TestCase):
    """freeze -> sample -> build -> finalize -> check, with the data index replaced by a small fixture."""

    def test_workflow(self) -> None:
        examples = _examples("RC1", range(60), per_section=4) + _examples("RC2", range(60), per_section=4, start_id=1000)
        split = make_zslab_split(examples, val_fraction=0.2, slab_count=2, guard_sections=1)
        windows = {(e.volume, e.location_id): {e.image_key} for e in examples}
        val = {}
        for key in split.val_keys:
            volume, _, loc = key.partition(":")
            val.setdefault(volume, []).append(int(loc))
        current = ({v: sorted(ids) for v, ids in val.items()}, windows)
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            runs, valsets, data = tmp / "runs", tmp / "valsets", tmp / "data"
            (runs / "sam2_em_v5").mkdir(parents=True)
            from sam2_segmentation_trainer.data.splits import save_split

            save_split(split, runs / "sam2_em_v5" / "split.json")
            for volume in ("RC1", "RC2"):
                (data / volume).mkdir(parents=True)
                (data / volume / "manifest.jsonl").write_text("", encoding="utf-8")  # flat crops layout

            def run(*argv: str) -> tuple[int, str]:
                out, err = io.StringIO(), io.StringIO()
                with redirect_stdout(out), redirect_stderr(err), mock.patch.object(
                    cli, "_current_val_locations", return_value=current
                ), mock.patch.object(cli, "run_dir_for", lambda name: runs / name):
                    code = cli.main(["--valsets", str(valsets), *argv])
                return code, out.getvalue() + err.getvalue()

            self.assertEqual(run("freeze", "--run", "sam2_em_v5", "--name", "val_v1")[0], 0)
            self.assertEqual(run("freeze", "--run", "sam2_em_v5", "--name", "val_v1")[0], 2)  # never rewritten
            code, text = run("sample", "--split", "val_v1", "--name", "gold_v1", "--per-volume", "5", "--root", str(data))
            self.assertEqual(code, 0, text)
            payload = read_gold_payload(valsets / "gold" / "gold_v1.json")
            rc1 = payload["volumes"]["RC1"]["candidates"]
            self.assertEqual(len(rc1), 5)
            self.assertTrue(set(rc1) <= set(val["RC1"]))
            review = (valsets / "gold" / "gold_v1.review" / "RC1.txt").read_text(encoding="utf-8")
            self.assertEqual(review.strip(), ", ".join(str(i) for i in rc1))
            # Same seed, same draw.
            run("sample", "--split", "val_v1", "--name", "gold_v1b", "--per-volume", "5", "--root", str(data))
            self.assertEqual(read_gold_payload(valsets / "gold" / "gold_v1b.json")["volumes"]["RC1"]["candidates"], rc1)

            # Review: approve RC1's first four, reject the fifth; approve all of RC2 as whole locations.
            rc2 = payload["volumes"]["RC2"]["candidates"]
            approved = [{"location_id": i, "image_key": next(iter(windows[("RC1", i)]))} for i in rc1[:4]]
            (data / "RC1" / "approved.json").write_text(json.dumps(approved), encoding="utf-8")
            (data / "RC1" / "ignore.json").write_text(json.dumps([rc1[4]]), encoding="utf-8")
            code, text = run("build", "--name", "gold_v1", "--finalize")
            self.assertEqual(code, 1, text)  # RC2 not reviewed yet
            self.assertIn("not reviewed", text)
            (data / "RC2" / "approved.json").write_text(
                json.dumps([{"location_id": i, "image_key": ""} for i in rc2]), encoding="utf-8"
            )
            code, text = run("build", "--name", "gold_v1", "--finalize")
            self.assertEqual(code, 0, text)
            gold = load_gold(valsets / "gold" / "gold_v1.json")
            self.assertEqual(gold.status, "final")
            self.assertEqual(gold.by_volume(), {"RC1": 4, "RC2": 5})
            self.assertEqual(run("build", "--name", "gold_v1")[0], 2)  # final never changes
            code, text = run("check", "--name", "gold_v1")
            self.assertEqual(code, 0, text)
            code, text = run("list")
            self.assertIn("split val_v1", text)
            self.assertIn("gold gold_v1: final", text)

    def test_check_flags_a_gold_location_that_is_a_training_location(self) -> None:
        split = make_zslab_split(_examples("RC1", range(40)), val_fraction=0.2, slab_count=1, guard_sections=1)
        train_loc = int(split.train_keys[0].split(":")[1])
        with TemporaryDirectory() as tmp:
            valsets = Path(tmp)
            freeze_split(split, valsets / "splits" / "val_v1.json", name="val_v1", source="t")
            payload = new_gold_payload(name="g", split="val_v1", candidates={"RC1": [train_loc]}, per_volume=1, seed=0)
            payload["volumes"]["RC1"].update(gold=[train_loc], pending=[])
            write_gold_payload(valsets / "gold" / "g.json", payload, create=True)
            with redirect_stdout(io.StringIO()) as out:
                code = cli.main(["--valsets", str(valsets), "check", "--name", "g"])
        self.assertEqual(code, 1)
        self.assertIn("is a training location", out.getvalue())
        self.assertEqual(gold_from_payload(payload).by_volume(), {"RC1": 1})


if __name__ == "__main__":
    unittest.main()
