"""Report AnnotationCrops coverage before training."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from sam2_segmentation_trainer.data.manifest import index_volumes, summarize_index
from sam2_segmentation_trainer.paths import DEFAULT_VOLUMES, data_root


def _read_json_stats(examples, *, limit: int | None) -> dict[str, object]:
    empty = 0
    missing_ann = 0
    categories: Counter[str] = Counter()
    scanned = 0
    for ex in examples:
        if limit is not None and scanned >= limit:
            break
        scanned += 1
        try:
            payload = json.loads(ex.json_path.read_text(encoding="utf-8"))
        except OSError:
            missing_ann += 1
            continue
        match = None
        for ann in payload.get("annotations") or []:
            if int(ann.get("id", -1)) == ex.location_id:
                match = ann
                break
        if match is None:
            missing_ann += 1
            continue
        name = str(match.get("category_name") or match.get("structure_label") or "")
        categories[name] += 1
        area = match.get("area")
        if area is not None and float(area) <= 0:
            empty += 1
    return {
        "json_scanned": scanned,
        "empty_area": empty,
        "missing_annotation_id": missing_ann,
        "by_category": dict(categories.most_common()),
    }


def build_report(
    *,
    root: Path,
    volumes: list[str],
    check_json: bool,
    json_limit: int | None,
) -> dict[str, object]:
    examples = index_volumes(volumes=volumes, root=root, skip_missing=False)
    summary = summarize_index(examples)
    present = index_volumes(volumes=volumes, root=root, skip_missing=True)
    report: dict[str, object] = {
        "root": str(root),
        "volumes": volumes,
        "indexed_including_missing": summary,
        "usable_examples": len(present),
        "skipped_vs_manifest_rows": summary["n_examples"] - len(present),
    }
    if check_json:
        report["json_stats"] = _read_json_stats(present, limit=json_limit)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate AnnotationCrops SAM2 training data")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument(
        "--volumes",
        default=",".join(DEFAULT_VOLUMES),
        help="Comma-separated volume names",
    )
    parser.add_argument("--json-report", type=Path, default=None)
    parser.add_argument(
        "--check-json",
        action="store_true",
        help="Open sidecar JSON for area/category stats (slow on CIFS)",
    )
    parser.add_argument("--json-limit", type=int, default=None)
    args = parser.parse_args(argv)

    root = args.root if args.root is not None else data_root()
    volumes = [v.strip() for v in args.volumes.split(",") if v.strip()]
    report = build_report(
        root=root,
        volumes=volumes,
        check_json=args.check_json,
        json_limit=args.json_limit,
    )
    text = json.dumps(report, indent=2)
    print(text)
    if args.json_report is not None:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(text + "\n", encoding="utf-8")
    usable = int(report["usable_examples"])
    missing = report["indexed_including_missing"]
    if usable == 0:
        return 2
    if missing["missing_image"] or missing["missing_json"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
