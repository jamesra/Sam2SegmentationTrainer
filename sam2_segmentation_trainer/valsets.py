"""``sam2-em-valsets``: make frozen validation splits and gold sets (see ``docs/validation_sets.md``).

Workflow::

    sam2-em-valsets freeze --run sam2_em_v5 --name v5              # the run's split becomes validation set v5
    sam2-em-valsets sample --split v5 --name gold_v1              # 100 random validation locations per volume
    # review each volume's list in the gallery (paste the ids into its location filter; approve or reject)
    sam2-em-valsets build --name gold_v1                          # read the gallery's decisions
    sam2-em-valsets build --name gold_v1 --finalize               # once nothing is pending: never changes again
    sam2-em-valsets check --name gold_v1                          # gold locations are all validation locations
    sam2-em-valsets list

New runs use them with ``data.frozen_split=v5 data.gold_set=gold_v1``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from sam2_segmentation_trainer.data.approved import APPROVED_FILE, parse_approved
from sam2_segmentation_trainer.data.manifest import index_volumes, load_ignore_ids, take_model_sized
from sam2_segmentation_trainer.data.splits import SplitLists, include_new_keys, load_split
from sam2_segmentation_trainer.data.valsets import (
    GOLD_LISTS,
    check_name,
    classify_candidates,
    freeze_split,
    frozen_info,
    gold_from_payload,
    load_frozen_split,
    new_gold_payload,
    now_utc,
    parse_split_key,
    read_gold_payload,
    resolve_gold_path,
    resolve_split_path,
    val_locations_by_volume,
    valsets_root,
    write_gold_payload,
)
from sam2_segmentation_trainer.paths import annotation_crops_root, data_root, run_dir_for


def _current_val_locations(
    split: SplitLists, volumes: list[str], root: Path, image_size: int
) -> tuple[dict[str, list[int]], dict[tuple[str, int], set[str]]]:
    """Validation locations per volume that exist in the data now (ignored and wrong-size tiles left out), and the
    current crop windows (image keys) of every location."""
    examples = index_volumes(volumes=volumes, root=root, skip_missing=True)
    examples, _rejected = take_model_sized(examples, image_size)
    grown = include_new_keys(split, examples, train_fraction=0.9, seed=0) if split.val_z else split
    val_keys = set(grown.val_keys)
    windows: dict[tuple[str, int], set[str]] = {}
    by_volume: dict[str, set[int]] = {}
    for ex in examples:
        windows.setdefault((ex.volume, int(ex.location_id)), set()).add(ex.image_key)
        if ex.split_key in val_keys:
            by_volume.setdefault(ex.volume, set()).add(int(ex.location_id))
    return {v: sorted(ids) for v, ids in sorted(by_volume.items())}, windows


def _split_volumes(split: SplitLists) -> list[str]:
    return sorted({parse_split_key(k)[0] for k in split.train_keys + split.val_keys})


def cmd_freeze(args) -> int:
    name = check_name(args.name)
    source_path = Path(args.split_file) if args.split_file else run_dir_for(args.run) / "split.json"
    if not source_path.is_file():
        print(f"no split at {source_path}", file=sys.stderr)
        return 2
    split = load_split(source_path)
    if split.mode != "zslab":
        print(
            f"WARNING: {source_path} is a {split.mode} split: validation locations sit beside training locations in "
            "adjacent sections, so its scores run high. A zslab split is the better benchmark."
        )
    target = resolve_split_path(name, valsets_root(args.valsets))
    info = freeze_split(
        split, target, name=name, source=args.run or str(source_path), description=args.description or ""
    )
    print(f"froze {source_path} as {name}: {target}")
    print(f"  mode {split.mode}; validation locations per volume {info['val_locations']}")
    print(f"  new runs: data.frozen_split={name}")
    return 0


def cmd_sample(args) -> int:
    name = check_name(args.name)
    root = valsets_root(args.valsets)
    split_path = resolve_split_path(args.split, root)
    split = load_frozen_split(split_path)
    volumes = [v.strip() for v in args.volumes.split(",")] if args.volumes else _split_volumes(split)
    gold_path = resolve_gold_path(name, root)
    if gold_path.exists():
        print(f"{gold_path} exists; choose a new name", file=sys.stderr)
        return 2
    data = data_root(args.root)
    current, _windows = _current_val_locations(split, volumes, data, args.image_size)
    candidates: dict[str, list[int]] = {}
    for volume in volumes:
        pool = current.get(volume, [])
        if not pool:
            print(f"WARNING: {volume} has no validation locations in {split_path.stem} under {data}")
            continue
        rng = random.Random(f"{args.seed}|{volume}")
        candidates[volume] = sorted(rng.sample(pool, min(args.per_volume, len(pool))))
        print(f"{volume}: {len(candidates[volume])} of {len(pool)} validation locations drawn")
    payload = new_gold_payload(
        name=name,
        split=frozen_info(split_path).get("name") or split_path.stem,
        candidates=candidates,
        per_volume=args.per_volume,
        seed=args.seed,
        description=args.description or "",
    )
    payload["data_root"] = str(data)
    write_gold_payload(gold_path, payload, create=True)
    review_dir = gold_path.parent / f"{name}.review"
    review_dir.mkdir(parents=True, exist_ok=True)
    for volume, ids in candidates.items():
        (review_dir / f"{volume}.txt").write_text(", ".join(str(i) for i in ids) + "\n", encoding="utf-8")
    print(f"wrote draft gold set {gold_path}")
    print(f"review lists (paste into the gallery's location filter, one volume at a time): {review_dir}")
    print(f"then: sam2-em-valsets build --name {name}")
    return 0


def _review_state(volume: str, data: Path) -> tuple[set[tuple[int, str]], set[int], set[int]]:
    crops = annotation_crops_root(volume, data)
    approved_windows: set[tuple[int, str]] = set()
    approved_whole: set[int] = set()
    path = crops / APPROVED_FILE
    if path.is_file():
        approved_windows, approved_whole = parse_approved(json.loads(path.read_text(encoding="utf-8")))
    return approved_windows, approved_whole, load_ignore_ids(crops)


def cmd_build(args) -> int:
    root = valsets_root(args.valsets)
    gold_path = resolve_gold_path(args.name, root)
    payload = read_gold_payload(gold_path)
    if payload.get("status") == "final":
        print(f"{gold_path} is final and never changes", file=sys.stderr)
        return 2
    data = data_root(args.root or payload.get("data_root"))
    split_path = resolve_split_path(payload["split"], root)
    split = load_frozen_split(split_path)
    volumes = sorted(payload.get("volumes") or {})
    _current, windows = _current_val_locations(split, volumes, data, args.image_size)
    total = {k: 0 for k in ("gold", "rejected", "pending")}
    for volume in volumes:
        lists = payload["volumes"][volume]
        approved_windows, approved_whole, ignored = _review_state(volume, data)
        sorted_lists = classify_candidates(
            lists.get("candidates") or (),
            windows={loc: keys for (v, loc), keys in windows.items() if v == volume},
            approved_windows=approved_windows,
            approved_whole=approved_whole,
            ignored=ignored,
        )
        lists.update(sorted_lists)
        for key in total:
            total[key] += len(sorted_lists[key])
        print(
            f"{volume}: gold {len(sorted_lists['gold'])}, rejected {len(sorted_lists['rejected'])}, "
            f"pending {len(sorted_lists['pending'])} of {len(lists.get('candidates') or ())}"
        )
    payload["updated_at"] = now_utc()
    if args.finalize:
        if total["pending"] and not args.allow_pending:
            print(
                f"{total['pending']} candidates are not reviewed; review them or pass --allow-pending "
                "(they are left out of the gold set)",
                file=sys.stderr,
            )
            write_gold_payload(gold_path, payload)
            return 1
        if not total["gold"]:
            print("no accepted locations; nothing to finalize", file=sys.stderr)
            return 1
        payload["status"] = "final"
        payload["finalized_at"] = payload["updated_at"]
    write_gold_payload(gold_path, payload)
    gold = gold_from_payload(payload)
    print(f"{gold_path}: {len(gold)} gold locations, status {gold.status}; per volume {gold.by_volume()}")
    return 0


def cmd_check(args) -> int:
    root = valsets_root(args.valsets)
    gold_path = resolve_gold_path(args.name, root)
    payload = read_gold_payload(gold_path)
    gold = gold_from_payload(payload)
    split_path = resolve_split_path(args.split or payload["split"], root)
    split = load_split(split_path)
    val_keys, train_keys = set(split.val_keys), set(split.train_keys)
    problems = 0
    for volume, location in sorted(gold.locations):
        key = f"{volume}:{location}"
        if key in train_keys:
            print(f"ERROR: {key} is a training location in {split_path.stem}")
            problems += 1
        elif key not in val_keys:
            print(f"note: {key} is not in {split_path.stem} (added since it was frozen, or in a guard section)")
    counts = {list_name: 0 for list_name in GOLD_LISTS}
    for lists in (payload.get("volumes") or {}).values():
        for list_name in GOLD_LISTS:
            counts[list_name] += len(lists.get(list_name) or ())
    print(f"{gold.name}: status {gold.status}, lists {counts}; per volume {gold.by_volume()}")
    print("ok" if not problems else f"{problems} gold location(s) are training locations")
    return 0 if not problems else 1


def cmd_list(args) -> int:
    root = valsets_root(args.valsets)
    print(f"valsets root: {root}")
    for path in sorted((root / "splits").glob("*.json")):
        info = frozen_info(path)
        split = load_split(path)
        print(
            f"split {path.stem}: mode {split.mode}, from {info.get('source', '?')}, frozen {info.get('created_at', '?')}; "
            f"validation locations {({v: len(ids) for v, ids in val_locations_by_volume(split).items()})}"
        )
    for path in sorted((root / "gold").glob("*.json")):
        try:
            payload = read_gold_payload(path)
        except ValueError as exc:
            print(f"gold {path.stem}: unreadable ({exc})")
            continue
        gold = gold_from_payload(payload)
        print(
            f"gold {path.stem}: {gold.status}, split {gold.split}, {len(gold)} locations, {gold.pending} pending; "
            f"per volume {gold.by_volume()}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sam2-em-valsets", description=__doc__.split("\n\n")[0])
    parser.add_argument("--valsets", default=None, help="valsets folder (default {SAM2_CHECKPOINT_ROOT}/valsets)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("freeze", help="save a run's split.json as a named frozen split")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", help="run name (its split.json under {SAM2_CHECKPOINT_ROOT}/runs)")
    source.add_argument("--split-file", help="a split.json path")
    p.add_argument("--name", required=True, help="validation set (frozen split) name, versioned like training sets, for example v5")
    p.add_argument("--description", default="")
    p.set_defaults(func=cmd_freeze)

    p = sub.add_parser("sample", help="draw gold candidates at random from a frozen split's validation locations")
    p.add_argument("--split", required=True, help="frozen split name or path")
    p.add_argument("--name", required=True, help="gold set name, for example gold_v1")
    p.add_argument("--per-volume", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--volumes", default=None, help="comma-separated (default: the split's volumes)")
    p.add_argument("--root", default=None, help="data root (default SAM2_DATA_ROOT)")
    p.add_argument("--image-size", type=int, default=1024)
    p.add_argument("--description", default="")
    p.set_defaults(func=cmd_sample)

    p = sub.add_parser("build", help="sort a gold set's candidates by their gallery review")
    p.add_argument("--name", required=True)
    p.add_argument("--root", default=None, help="data root (default: the one the set was sampled from)")
    p.add_argument("--image-size", type=int, default=1024)
    p.add_argument("--finalize", action="store_true", help="mark final: the set never changes again")
    p.add_argument("--allow-pending", action="store_true", help="finalize with unreviewed candidates left out")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("check", help="verify a gold set against its frozen split")
    p.add_argument("--name", required=True)
    p.add_argument("--split", default=None, help="split name or path (default: the gold set's own)")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("list", help="list frozen splits and gold sets")
    p.set_defaults(func=cmd_list)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (FileExistsError, FileNotFoundError, PermissionError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
