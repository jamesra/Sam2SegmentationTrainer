"""How large must the per-epoch validation subset be? Measure it from one full validation pass.

A fixed checkpoint scores every validation sample the same way each time (prompts are seeded per sample), so one
full pass that keeps the per-sample records (``sam2-em-eval --goal --records``) is enough: this script draws many
random subsets of those records with the trainer's own subset rule (``validation.select_validation_subset``:
stratified per volume, hash-seeded, Approved samples always in) and pools each draw with the trainer's own summary
code, exactly as an epoch's validation would. Comparing each draw with the full split shows how far a subset of
each size can be off.

    python scripts/valset_size_study.py --records /outputs/valset_size_v5_e0/records.jsonl.gz \\
        --out /outputs/valset_size_v5_e0

Writes ``size_study.csv`` (every family x metric x fraction) and ``size_study.md`` (the summary).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import namedtuple
from pathlib import Path
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sam2_segmentation_trainer.data.approved import ApprovedSet  # noqa: E402
from sam2_segmentation_trainer.evaluate import RECORDS_FILE, read_records  # noqa: E402
from sam2_segmentation_trainer.goal_metrics import CORE_RECALL_TARGET, summarize_goal  # noqa: E402
from sam2_segmentation_trainer.validation import select_validation_subset  # noqa: E402

Sample = namedtuple("Sample", "volume image_key location_id")
"""Stand-in for a validation example: all ``select_validation_subset`` reads."""

DEFAULT_FRACTIONS = (0.01, 0.02, 0.05, 0.10, 0.15, 0.25, 0.50, 1.0)
DEFAULT_FAMILIES = ("circle_annotator", "circle_mask", "point", "correction", "near_correct")
METRICS = ("iou", "raw_iou", "core_recall", "core_recall_ok", "core_recall_p10", "merge_rate", "spill", "leak", "frag")
TOLERANCE = {
    # The trainer's early-stopping thresholds where they exist (goal_metrics: 0.002 merge rate / spill, 0.0005 core
    # recall); otherwise a difference too small to act on.
    "iou": 0.005,
    "raw_iou": 0.005,
    "core_recall": 0.0005,
    "core_recall_ok": 0.01,
    "core_recall_p10": 0.01,
    "merge_rate": 0.002,
    "spill": 0.002,
    "leak": 0.002,
    "frag": 0.01,
}
"""Largest acceptable error of a subset estimate, per metric (``--tolerance metric=value`` overrides)."""


def score(records: Sequence[dict[str, Any]], target: float) -> dict[str, float]:
    """The headline numbers of ``validation.summarize_mode`` for ``records`` (fully masked tiles left out, unless
    they are all there is), without its per-volume and per-category breakdowns."""
    normal = [r for r in records if not r.get("full_tile")]
    scored = normal or list(records)
    if not scored:
        return {metric: float("nan") for metric in METRICS} | {"n": 0}
    goal = summarize_goal(scored, target=target)
    return {
        "n": len(scored),
        "iou": float(np.mean([r["iou"] for r in scored])),
        "raw_iou": float(np.mean([r.get("raw_iou", r["iou"]) for r in scored])),
        "core_recall": goal.core_recall,
        "core_recall_ok": goal.core_recall_ok,
        "core_recall_p10": goal.core_recall_p10,
        "merge_rate": goal.merge_rate,
        "spill": goal.spill,
        "leak": goal.leak,
        "frag": goal.frag,
    }


def study(
    records: dict[str, list[dict[str, Any]]],
    *,
    families: Sequence[str],
    fractions: Sequence[float],
    draws: int,
    target: float = CORE_RECALL_TARGET,
    keep_approved: bool = True,
) -> list[dict[str, Any]]:
    """One row per family x metric x fraction: the full-split value and how far ``draws`` subsets fall from it.

    The full split (fraction 1) is always included, as the reference row.
    """
    fractions = sorted({float(f) for f in fractions} | {1.0})
    samples = {}
    approved_windows = set()
    for rows in records.values():
        for record in rows:
            key = tuple(record["key"])
            samples.setdefault(key, Sample(*key))
            if record.get("approved"):
                volume, image_key, location_id = key
                approved_windows.add((volume, int(location_id), image_key))
    approved = ApprovedSet(windows=frozenset(approved_windows))
    examples = list(samples.values())
    by_key = {family: {} for family in families}
    for family in families:
        for record in records.get(family, []):
            by_key[family].setdefault(tuple(record["key"]), []).append(record)
    full = {family: score(records.get(family, []), target) for family in families}
    rows: list[dict[str, Any]] = []
    for fraction in fractions:
        values = {family: {metric: [] for metric in METRICS} for family in families}
        sizes = {family: [] for family in families}
        for seed in range(draws if fraction < 1.0 else 1):
            chosen = select_validation_subset(examples, fraction, seed, approved if keep_approved else None)
            keys = [(c.volume, c.image_key, c.location_id) for c in chosen]
            for family in families:
                subset = [r for key in keys for r in by_key[family].get(key, ())]
                result = score(subset, target)
                sizes[family].append(result["n"])
                for metric in METRICS:
                    values[family][metric].append(result[metric])
        for family in families:
            for metric in METRICS:
                vals = np.asarray(values[family][metric], dtype=np.float64)
                errors = np.abs(vals - full[family][metric])
                rows.append(
                    {
                        "family": family,
                        "metric": metric,
                        "fraction": fraction,
                        "n_samples": int(round(float(np.mean(sizes[family])))),
                        "draws": len(vals),
                        "full": full[family][metric],
                        "mean": float(np.nanmean(vals)),
                        "sd": float(np.nanstd(vals)),
                        "p95_abs_err": float(np.nanpercentile(errors, 95)),
                        "max_abs_err": float(np.nanmax(errors)),
                    }
                )
    return rows


def converged_fraction(rows: Sequence[dict[str, Any]], family: str, metric: str, tolerance: float) -> float | None:
    """The smallest fraction whose 95th-percentile error is within ``tolerance`` (None if no fraction below 1 is)."""
    for row in sorted((r for r in rows if r["family"] == family and r["metric"] == metric), key=lambda r: r["fraction"]):
        if row["fraction"] < 1.0 and row["p95_abs_err"] <= tolerance:
            return row["fraction"]
    return None


def report(rows: Sequence[dict[str, Any]], families: Sequence[str], tolerance: dict[str, float], meta: dict) -> str:
    """The markdown summary: per family, the 95th-percentile error of each metric at each fraction, and where it
    falls within tolerance."""
    lines = [
        "# Validation subset size study",
        "",
        f"Records: {meta.get('records')}; model: {meta.get('model', '?')}; split: {meta.get('split', '?')}; "
        f"{meta.get('draws')} random draws per fraction with the trainer's subset rule (per volume, Approved always "
        "in). Each cell is the 95th percentile of |subset value - full-split value| over the draws.",
        "",
        "Tolerances: " + ", ".join(f"{m} {tolerance[m]:g}" for m in METRICS) + ".",
        "",
        "Checkpoints are compared on one fixed subset (a paired comparison), which cancels part of this sampling "
        "noise, so these errors overstate what changes a checkpoint choice.",
        "",
    ]
    for family in families:
        family_rows = [r for r in rows if r["family"] == family]
        fractions = sorted({r["fraction"] for r in family_rows})
        lines += [f"## {family}", "", "| fraction | samples | " + " | ".join(METRICS) + " |"]
        lines.append("|---|---|" + "---|" * len(METRICS))
        for fraction in fractions:
            cells = {r["metric"]: r for r in family_rows if r["fraction"] == fraction}
            n = cells[METRICS[0]]["n_samples"]
            values = []
            for metric in METRICS:
                value = cells[metric]["p95_abs_err"] if fraction < 1.0 else cells[metric]["full"]
                within = fraction < 1.0 and value <= tolerance[metric]
                values.append(f"{value:.4f}{' ok' if within else ''}")
            label = f"{fraction:g}" if fraction < 1.0 else "1 (full value)"
            lines.append(f"| {label} | {n} | " + " | ".join(values) + " |")
        lines.append("")
        lines.append(
            "Within tolerance from: "
            + ", ".join(
                f"{m} {converged_fraction(rows, family, m, tolerance[m]) or 'never below 1.0'}" for m in METRICS
            )
        )
        merge = next((r for r in family_rows if r["metric"] == "merge_rate" and r["fraction"] == 1.0), None)
        if merge is not None:
            lines.append(
                f"Expected merges per pass at the full-split rate {merge['full']:.4%}: "
                + ", ".join(
                    f"{r['fraction']:g}: {merge['full'] * r['n_samples']:.1f}"
                    for r in sorted(family_rows, key=lambda r: r["fraction"])
                    if r["metric"] == "merge_rate"
                )
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--records", type=Path, required=True, help=f"{RECORDS_FILE} of a --goal --records run")
    parser.add_argument("--out", type=Path, default=None, help="Output folder (default: the records' folder)")
    parser.add_argument("--fractions", default=",".join(f"{f:g}" for f in DEFAULT_FRACTIONS))
    parser.add_argument("--families", default=",".join(DEFAULT_FAMILIES))
    parser.add_argument("--draws", type=int, default=200, help="Random subsets per fraction")
    parser.add_argument("--no-approved", action="store_true", help="Do not force Approved samples into every draw")
    parser.add_argument("--tolerance", action="append", default=[], metavar="METRIC=VALUE")
    args = parser.parse_args(argv)

    out = args.out or args.records.parent
    tolerance = dict(TOLERANCE)
    for item in args.tolerance:
        metric, value = item.split("=", 1)
        if metric not in tolerance:
            parser.error(f"unknown metric {metric!r}; choose from {list(tolerance)}")
        tolerance[metric] = float(value)
    fractions = sorted({float(f) for f in args.fractions.split(",") if f.strip()} | {1.0})
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    meta: dict[str, Any] = {"records": str(args.records), "draws": args.draws}
    summary_path = args.records.parent / "goal_metrics.json"
    target = CORE_RECALL_TARGET
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        meta.update({k: summary.get(k) for k in ("model", "split", "subset_seed", "epoch_subset")})
        target = float(summary.get("core_recall_target", target))

    records = read_records(args.records)
    missing = [f for f in families if f not in records]
    if missing:
        parser.error(f"families {missing} are not in {args.records}; it has {sorted(records)}")
    rows = study(
        records, families=families, fractions=fractions, draws=args.draws, target=target,
        keep_approved=not args.no_approved,
    )
    out.mkdir(parents=True, exist_ok=True)
    with (out / "size_study.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    text = report(rows, families, tolerance, meta)
    (out / "size_study.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {out / 'size_study.csv'} and {out / 'size_study.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
