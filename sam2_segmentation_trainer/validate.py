"""Report AnnotationCrops coverage before training."""

from __future__ import annotations

import argparse
import json
from collections import Counter
import numpy as np
from pathlib import Path

from sam2_segmentation_trainer.data.manifest import (
    examples_usable,
    index_volumes,
    summarize_index,
)
from annotation_crops.negatives import DEFAULT_CIRCLE_DIAMETER_SCALE, load_circle_scale
from sam2_segmentation_trainer.data.tiles import build_tile_context, crops_dir_of
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


def _circle_stats(examples) -> dict[str, object]:
    """How much negative context the mixed-prompt trainer will find.

    Counts circle entries from each volume's ``circle_annotations.json`` by origin, and the tiles whose target
    has another annotation (a neighbor mask or a circle) to use as negative context. A volume with no circle
    file simply contributes no circles.
    """
    context = build_tile_context(examples, circle_scale=DEFAULT_CIRCLE_DIAMETER_SCALE)
    by_source: Counter[str] = Counter()
    for circles in context.circles.values():
        for circle in circles:
            by_source[circle.source] += 1
    scales: dict[str, float | None] = {}
    for example in examples:
        if example.volume not in scales:  # setdefault would parse the whole circle file for every example
            scales[example.volume] = load_circle_scale(crops_dir_of(example))
    with_context = 0
    for key, ids in context.neighbors.items():
        if len(ids) > 1 or any(c.location_id not in ids for c in context.circles_in(key)):
            with_context += 1
    return {
        "tiles_with_circle_entries": len(context.circles),
        "circle_diameter_scale_by_volume": scales,
        "circle_entries_by_source": dict(by_source),
        "tiles_with_negative_context": with_context,
        "tiles": len(context.neighbors),
    }


def circle_overlap_stats(examples, *, sample: int, seed: int = 0) -> dict[str, object]:
    """How often a neighbor's circle reaches into the target mask, on a fixed random sample of examples.

    Circle negatives are written as background; one that overlaps the target contradicts the ground truth
    (assessment 1.6). Only examples whose tile holds a circle that is not the target's own are counted.
    """
    import random

    from PIL import Image

    from annotation_crops.negatives import KIND_CIRCLE, circles_over_target, collect_negative_regions

    context = build_tile_context(examples, circle_scale=DEFAULT_CIRCLE_DIAMETER_SCALE)
    pool = list(examples)
    random.Random(seed).shuffle(pool)
    seen = with_circles = overlapping = 0
    over_circles = total_circles = 0
    by_volume: dict[str, list[int]] = {}
    for example in pool:
        if seen >= sample:
            break
        key = (example.volume, example.image_key)
        circles = [c for c in context.circles_in(key) if c.location_id != example.location_id]
        if not circles:
            continue
        seen += 1
        try:
            with Image.open(example.mask_path) as handle:
                target = np.asarray(handle.convert("L")) > 127
        except OSError:
            continue
        regions = collect_negative_regions(
            example.image_key,
            example.location_id,
            context.neighbor_ids(key),
            circles,
            example.mask_path.parent,
            target.shape,
        )
        circle_regions = [r for r in regions if r.kind == KIND_CIRCLE]
        if not circle_regions:
            continue
        with_circles += 1
        hit = circles_over_target(regions, target)
        total_circles += len(circle_regions)
        over_circles += len(hit)
        counts = by_volume.setdefault(example.volume, [0, 0])
        counts[0] += 1
        if hit:
            overlapping += 1
            counts[1] += 1
    return {
        "examples_sampled": seen,
        "examples_with_circle_negatives": with_circles,
        "examples_where_a_circle_overlaps_the_target": overlapping,
        "share_of_examples": (overlapping / with_circles) if with_circles else 0.0,
        "circle_negatives": total_circles,
        "circle_negatives_overlapping": over_circles,
        "by_volume": {v: {"with_circles": c[0], "overlapping": c[1]} for v, c in sorted(by_volume.items())},
    }


def build_report(
    *,
    root: Path,
    volumes: list[str],
    check_json: bool,
    json_limit: int | None,
    circle_overlap_sample: int = 0,
) -> dict[str, object]:
    examples = index_volumes(volumes=volumes, root=root, skip_missing=False)
    summary = summarize_index(examples)
    present = examples_usable(examples)
    report: dict[str, object] = {
        "root": str(root),
        "volumes": volumes,
        "indexed_including_missing": summary,
        "usable_examples": len(present),
        "skipped_vs_manifest_rows": summary["n_examples"] - len(present),
    }
    report["circle_context"] = _circle_stats(present)
    if check_json:
        report["json_stats"] = _read_json_stats(present, limit=json_limit)
    if circle_overlap_sample > 0:
        report["circle_overlap"] = circle_overlap_stats(present, sample=circle_overlap_sample)
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
    parser.add_argument(
        "--circle-overlap",
        type=int,
        default=0,
        metavar="N",
        help="Measure, on N sampled examples, how often a neighbor's circle overlaps the target mask (reads masks; slow on CIFS)",
    )
    args = parser.parse_args(argv)

    root = args.root if args.root is not None else data_root()
    volumes = [v.strip() for v in args.volumes.split(",") if v.strip()]
    report = build_report(
        root=root,
        volumes=volumes,
        check_json=args.check_json,
        json_limit=args.json_limit,
        circle_overlap_sample=args.circle_overlap,
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
