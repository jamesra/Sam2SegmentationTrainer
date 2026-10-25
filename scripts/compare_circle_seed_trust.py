#!/usr/bin/env python3
"""Compare two ``sam2-em-eval --goal`` summaries family by family: the circle-seed taper off and on.

Prints, for each family, the goal metrics of both runs and the difference (on minus off), with ``+`` where the
second run is better and ``-`` where it is worse (recall up, leak, merge, spill and fragmentation down), for all
samples and for the ``with_neg`` / ``no_neg`` and gold rows when both runs have them. See
docs/circle_seed_trust_eval.md. Needs only the standard library.

    python scripts/compare_circle_seed_trust.py /outputs/eval_v5.4_edge/goal_metrics.json \\
        /outputs/eval_v5.4_taper/goal_metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_FAMILIES = ("circle_mask", "circle_annotator", "circle_oversize")

HIGHER_IS_BETTER = {
    "core_recall": True,
    "core_recall_p10": True,
    "core_recall_ok": True,
    "leak": False,
    "merge_rate": False,
    "spill": False,
    "frag": False,
}
"""The goal metrics (``goal_metrics.GoalSummary``) and which way is better."""

ROWS = (("all", "goal"), ("with_neg", "with_neg"), ("no_neg", "no_neg"), ("gold", "gold"))
"""Row label and the ``ModeResult`` field holding its ``GoalSummary``."""

TIE = 1e-4
"""Differences smaller than this are not marked."""


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def describe(summary: dict) -> str:
    seed = summary.get("circle_seed") or {}
    taper = seed.get("taper")
    encoding = (
        "edge confidence" if taper is None else f"taper full_frac {taper['full_frac']} zero_frac {taper['zero_frac']}"
    )
    oversize = seed.get("oversize_range")
    extra = "" if oversize is None else f", oversize range {oversize[0]}-{oversize[1]}"
    return f"{summary.get('model')} on {summary.get('samples')} samples ({encoding}{extra})"


def mark(metric: str, delta: float) -> str:
    if abs(delta) < TIE:
        return " "
    return "+" if (delta > 0) == HIGHER_IS_BETTER[metric] else "-"


def compare(off: dict, on: dict, families) -> str:
    lines = [f"off: {describe(off)}", f"on:  {describe(on)}"]
    if off.get("split") != on.get("split") or off.get("samples") != on.get("samples"):
        lines.append(f"WARNING: different splits or sample counts ({off.get('split')} vs {on.get('split')})")
    header = f"{'':10s} {'metric':16s} {'off':>8s} {'on':>8s} {'on-off':>8s}"
    for family in families:
        a, b = (off.get("families") or {}).get(family), (on.get("families") or {}).get(family)
        lines.append("")
        if a is None or b is None:
            lines.append(f"{family}: missing from {'the first' if a is None else 'the second'} run")
            continue
        lines.append(f"{family} (n={a.get('n')} / {b.get('n')}, iou {a.get('iou', 0.0):.4f} / {b.get('iou', 0.0):.4f})")
        lines.append(header)
        for label, field in ROWS:
            x, y = a.get(field), b.get(field)
            if not x or not y:
                continue
            for metric in HIGHER_IS_BETTER:
                delta = float(y[metric]) - float(x[metric])
                row = f"{label} n={x.get('n')}"
                lines.append(
                    f"{row:10s} {metric:16s} {float(x[metric]):8.4f} {float(y[metric]):8.4f} {delta:+8.4f} "
                    f"{mark(metric, delta)}"
                )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("off", type=Path, help="goal_metrics.json of the run without --circle-seed-taper")
    parser.add_argument("on", type=Path, help="goal_metrics.json of the run with it")
    parser.add_argument("--families", default=",".join(DEFAULT_FAMILIES), help="Comma-separated families to compare")
    args = parser.parse_args(argv)
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    print(compare(load(args.off), load(args.on), families))
    return 0


if __name__ == "__main__":
    sys.exit(main())
