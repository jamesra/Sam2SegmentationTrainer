#!/usr/bin/env python3
"""Summarize the runs of ``scripts/compare_throughput.sh``.

For every ``<name>.txt`` (the output of ``sam2-em-train --benchmark --benchmark-no-val``) with its
``<name>.smi.csv`` (``nvidia-smi --query-gpu=... -l 5``) and ``<name>.procs_before.txt`` (the other GPU processes,
listed before the run started), print seconds per sample, the GPU clocks, temperature, power and throttle reasons
during the busy part of the run, and the other GPU processes. Needs only the standard library.

    python scripts/summarize_throughput.py benchmarks/throughput
"""

from __future__ import annotations

import csv
import io
import re
import statistics
import sys
from pathlib import Path

BUSY_UTILIZATION = 50.0
"""GPU utilization (%) from which a sample counts as part of the timed training steps (not data loading or start-up)."""

THROTTLE_BITS = {
    0x0000000000000001: "idle",
    0x0000000000000002: "applications clocks setting",
    0x0000000000000004: "SW power cap",
    0x0000000000000008: "HW slowdown",
    0x0000000000000010: "sync boost",
    0x0000000000000020: "SW thermal slowdown",
    0x0000000000000040: "HW thermal slowdown",
    0x0000000000000080: "HW power brake slowdown",
    0x0000000000000100: "display clocks setting",
}
LIMITING = {"SW power cap", "HW slowdown", "SW thermal slowdown", "HW thermal slowdown", "HW power brake slowdown"}
"""The reasons that mean the GPU ran slower than it could; idle, sync boost and the display setting do not."""


def parse_benchmark(text: str) -> dict[str, float]:
    """Seconds per iteration and per sample, batch size and peak memory from a benchmark's printed summary."""
    out: dict[str, float] = {}
    match = re.search(r"batch_size=(\d+)", text)
    if match:
        out["batch_size"] = float(match.group(1))
    match = re.search(r"^\s*train ([\d.]+) s/it \(median ([\d.]+), min ([\d.]+), max ([\d.]+), n=(\d+)\)", text, re.M)
    if match:
        mean, median, low, high, count = (float(v) for v in match.groups())
        out.update(mean_s=mean, median_s=median, min_s=low, max_s=high, steps=count)
        if "batch_size" in out:
            out["s_per_sample"] = mean / out["batch_size"]
            out["median_s_per_sample"] = median / out["batch_size"]
    match = re.search(r"peak CUDA memory[^:]*: ([\d.]+)G allocated", text)
    if match:
        out["peak_gb"] = float(match.group(1))
    match = re.search(r"data wait ([\d.]+) s/it", text)
    if match:
        out["data_wait_s"] = float(match.group(1))
    return out


def _number(cell: str) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", cell)
    return float(match.group(0)) if match else None


def parse_smi(text: str) -> list[dict[str, float | int | None]]:
    """Rows of an ``nvidia-smi --query-gpu=... --format=csv -l`` log (header lines, which repeat, are skipped)."""
    rows: list[dict[str, float | int | None]] = []
    header: list[str] | None = None
    for cells in csv.reader(io.StringIO(text), skipinitialspace=True):
        if not cells:
            continue
        cells = [cell.strip() for cell in cells]
        if cells[0].lower().startswith("timestamp"):
            header = [cell.lower() for cell in cells]
            continue
        if header is None or len(cells) != len(header):
            continue

        def column(*needles: str) -> str:
            for name, cell in zip(header, cells):
                if all(needle in name for needle in needles):
                    return cell
            return ""

        reasons = column("reasons") or column("throttle")
        try:
            reason_bits = int(reasons, 16) if reasons.lower().startswith("0x") else int(float(reasons))
        except ValueError:
            reason_bits = None
        rows.append(
            {
                "sm": _number(column("clocks.current.sm")) or _number(column("clocks.sm")),
                "sm_max": _number(column("clocks.max.sm")),
                "temp": _number(column("temperature")),
                "power": _number(column("power.draw")),
                "power_limit": _number(column("power.limit")),
                "util": _number(column("utilization.gpu")),
                "reasons": reason_bits,
            }
        )
    return rows


def decode_reasons(bits: int) -> list[str]:
    return [name for bit, name in THROTTLE_BITS.items() if bits & bit]


def summarize_smi(rows: list[dict[str, float | int | None]]) -> dict[str, object]:
    """Clocks, temperature, power and throttle reasons over the busy samples of a run."""
    busy = [r for r in rows if (r["util"] or 0.0) >= BUSY_UTILIZATION]
    if not busy:
        return {"samples": len(rows), "busy": 0}

    def values(key: str) -> list[float]:
        return [float(r[key]) for r in busy if r[key] is not None]

    clocks, peak_clock = values("sm"), values("sm_max")
    top = max(peak_clock) if peak_clock else None
    reasons: dict[str, int] = {}
    for row in busy:
        if row["reasons"] is None:
            continue
        for name in decode_reasons(int(row["reasons"])):
            reasons[name] = reasons.get(name, 0) + 1
    out: dict[str, object] = {
        "samples": len(rows),
        "busy": len(busy),
        "clock_mean": statistics.fmean(clocks) if clocks else None,
        "clock_min": min(clocks) if clocks else None,
        "clock_max": top,
        "slow_share": (sum(c < 0.95 * top for c in clocks) / len(clocks)) if clocks and top else None,
        "temp_mean": statistics.fmean(values("temp")) if values("temp") else None,
        "temp_max": max(values("temp")) if values("temp") else None,
        "power_mean": statistics.fmean(values("power")) if values("power") else None,
        "power_limit": max(values("power_limit")) if values("power_limit") else None,
        "util_mean": statistics.fmean(values("util")),
        "reasons": reasons,
    }
    out["throttled"] = {name: n for name, n in reasons.items() if name in LIMITING}
    return out


def read_processes(path: Path) -> list[str]:
    """The GPU processes listed in a ``--query-compute-apps`` file (empty when none or the file is missing)."""
    if not path.is_file():
        return []
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines()]
    return [line for line in lines if line and not line.lower().startswith("pid") and "no running" not in line.lower()]


def report(folder: Path) -> str:
    skipped = (".procs_before.txt", ".procs_after.txt")
    runs = sorted(
        p
        for p in folder.glob("*.txt")
        if not p.name.endswith(skipped) and p.name != "summary.txt" and not p.name.startswith("nvidia-smi")
    )
    lines = [f"{'run':14s} {'config':32s} {'s/it':>7s} {'s/sample':>9s} {'median':>7s} {'peak GB':>8s} "
             f"{'busy':>5s} {'SM MHz mean (min / max)':>26s} {'slow':>6s} {'temp':>10s} {'power W':>12s}"]
    per_config: dict[str, list[float]] = {}
    notes: list[str] = []
    for path in runs:
        bench = parse_benchmark(path.read_text(encoding="utf-8", errors="replace"))
        if "s_per_sample" not in bench:
            lines.append(f"{path.stem:14s} (no benchmark summary: the run failed or was cut short)")
            continue
        smi_path = path.with_suffix(".smi.csv")
        smi = summarize_smi(parse_smi(smi_path.read_text(encoding="utf-8", errors="replace"))) if smi_path.is_file() else {}
        name = path.stem
        config = f"batch {int(bench['batch_size'])}"
        per_config.setdefault(name.split("_")[-1], []).append(bench["s_per_sample"])
        if smi.get("busy"):
            clock = f"{smi['clock_mean']:.0f} ({smi['clock_min']:.0f} / {smi['clock_max']:.0f})"
            slow = f"{100 * (smi['slow_share'] or 0):.0f}%"
            temp = f"{smi['temp_mean']:.0f} (max {smi['temp_max']:.0f})"
            power = f"{smi['power_mean']:.0f} / {smi['power_limit']:.0f}"
            busy = str(smi["busy"])
        else:
            clock = slow = temp = power = busy = "n/a"
        lines.append(
            f"{name:14s} {config:32s} {bench['mean_s']:7.3f} {bench['s_per_sample']:9.4f} "
            f"{bench['median_s_per_sample']:7.4f} {bench.get('peak_gb', float('nan')):8.2f} {busy:>5s} "
            f"{clock:>26s} {slow:>6s} {temp:>10s} {power:>12s}"
        )
        if smi.get("throttled"):
            notes.append(f"{name}: throttle reasons fired in {smi['throttled']} of {smi['busy']} busy samples")
        elif smi.get("busy"):
            notes.append(f"{name}: no throttle reasons during {smi['busy']} busy samples")
        others = read_processes(path.with_name(f"{name}.procs_before.txt"))
        if others:
            notes.append(f"{name}: other GPU processes before the run: {'; '.join(others)}")
    lines.append("")
    lines.extend(notes)
    lines.append("")
    means = {key: statistics.fmean(values) for key, values in per_config.items()}
    for key, values in sorted(per_config.items()):
        spread = (max(values) - min(values)) / statistics.fmean(values) * 100 if len(values) > 1 else 0.0
        lines.append(
            f"{key}: {', '.join(f'{v:.4f}' for v in values)} s/sample  mean {means[key]:.4f}"
            f"  (run to run spread {spread:.1f}%)"
        )
    if len(means) >= 2:
        fastest = min(means, key=means.get)
        slowest = max(means, key=means.get)
        gain = (means[slowest] - means[fastest]) / means[slowest] * 100
        lines.append(f"fastest: {fastest} by {gain:.1f}% per sample over {slowest}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    folder = Path(argv[1]) if len(argv) > 1 else Path("benchmarks/throughput")
    if not folder.is_dir():
        print(f"no such folder: {folder}", file=sys.stderr)
        return 1
    print(report(folder))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
