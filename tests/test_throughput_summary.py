from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "summarize_throughput.py"
_spec = importlib.util.spec_from_file_location("summarize_throughput", SCRIPT)
summarize = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(summarize)

HEADER = (
    "timestamp, clocks.current.sm [MHz], clocks.max.sm [MHz], temperature.gpu, power.draw [W], power.limit [W], "
    "utilization.gpu [%], clocks_throttle_reasons.active"
)


def _smi(rows: list[tuple]) -> str:
    out = [HEADER]
    for index, (clock, temp, power, util, reasons) in enumerate(rows):
        out.append(f"2026/10/14 01:00:{index * 5:02d}.000, {clock} MHz, 2625 MHz, {temp}, {power} W, 130.00 W, {util} %, {reasons}")
        if index == 2:
            out.append(HEADER)  # nvidia-smi -l can repeat the header
    return "\n".join(out) + "\n"


def _bench(batch: int, mean: float) -> str:
    return (
        f"  train=167300 val=19397 batch_size={batch} image_size=1024 batches/epoch=100 epochs=15\n"
        f"  train {mean:.3f} s/it (median {mean - 0.01:.3f}, min {mean - 0.1:.3f}, max {mean + 0.2:.3f}, n=200)\n"
        "  data wait 0.000 s/it (median 0.000, min 0.000, max 0.001, n=200); 0% of the step\n"
        "  peak CUDA memory in the timed train steps: 14.79G allocated, 15.41G reserved, 22.49G total\n"
    )


class ParsingTests(unittest.TestCase):
    def test_benchmark_summary(self) -> None:
        parsed = summarize.parse_benchmark(_bench(2, 0.64))
        self.assertEqual(parsed["batch_size"], 2.0)
        self.assertAlmostEqual(parsed["s_per_sample"], 0.32)
        self.assertAlmostEqual(parsed["median_s_per_sample"], 0.315)
        self.assertAlmostEqual(parsed["peak_gb"], 14.79)
        self.assertEqual(parsed["steps"], 200.0)
        self.assertEqual(summarize.parse_benchmark("nothing here"), {})

    def test_smi_log_with_units_and_repeated_headers(self) -> None:
        rows = summarize.parse_smi(_smi([(2500, 60, 118.5, 99, "0x0000000000000000"), (2490, 61, 119.0, 98, "0x0000000000000004")]))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["sm"], 2500.0)
        self.assertEqual(rows[0]["sm_max"], 2625.0)
        self.assertEqual(rows[0]["power_limit"], 130.0)
        self.assertEqual(rows[1]["reasons"], 4)
        self.assertEqual(len(summarize.parse_smi(_smi([(1, 1, 1, 1, "0x0")] * 5))), 5)  # the header line is skipped

    def test_the_newer_column_name_is_read_too(self) -> None:
        text = _smi([(2500, 60, 118.5, 99, "0x0000000000000020")]).replace("clocks_throttle_reasons.active", "clocks_event_reasons.active")
        self.assertEqual(summarize.parse_smi(text)[0]["reasons"], 0x20)

    def test_reasons_are_decoded(self) -> None:
        self.assertEqual(summarize.decode_reasons(0), [])
        self.assertEqual(summarize.decode_reasons(0x4 | 0x20), ["SW power cap", "SW thermal slowdown"])
        self.assertEqual(summarize.decode_reasons(0x1), ["idle"])


class SummaryTests(unittest.TestCase):
    def test_only_busy_samples_count_and_throttling_is_detected(self) -> None:
        rows = summarize.parse_smi(
            _smi(
                [
                    (300, 35, 20, 3, "0x0000000000000001"),    # start-up: idle, ignored
                    (2600, 62, 120, 99, "0x0000000000000000"),
                    (2000, 78, 130, 99, "0x0000000000000004"),  # power capped, clock down
                    (2000, 79, 130, 98, "0x0000000000000024"),  # power cap and thermal
                    (2600, 80, 125, 97, "0x0000000000000000"),
                ]
            )
        )
        summary = summarize.summarize_smi(rows)
        self.assertEqual(summary["busy"], 4)
        self.assertEqual(summary["samples"], 5)
        self.assertAlmostEqual(summary["clock_mean"], (2600 + 2000 + 2000 + 2600) / 4)
        self.assertEqual(summary["clock_min"], 2000.0)
        self.assertAlmostEqual(summary["slow_share"], 0.5)
        self.assertEqual(summary["temp_max"], 80.0)
        self.assertEqual(summary["throttled"], {"SW power cap": 2, "SW thermal slowdown": 1})
        self.assertNotIn("idle", summary["reasons"])

    def test_a_steady_gpu_reports_no_throttling(self) -> None:
        rows = summarize.parse_smi(_smi([(2610, 60, 118, 99, "0x0000000000000000")] * 6))
        summary = summarize.summarize_smi(rows)
        self.assertEqual(summary["throttled"], {})
        self.assertEqual(summary["slow_share"], 0.0)
        self.assertEqual(summarize.summarize_smi([])["busy"], 0)

    def test_report_for_a_folder_of_runs(self) -> None:
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            plan = [("run1_A", 2, 0.572), ("run2_B", 4, 1.297), ("run3_A", 2, 0.640), ("run4_B", 4, 1.300)]
            for name, batch, mean in plan:
                (folder / f"{name}.txt").write_text(_bench(batch, mean), encoding="utf-8")
                (folder / f"{name}.smi.csv").write_text(_smi([(2600, 62, 120, 99, "0x0")] * 4), encoding="utf-8")
                (folder / f"{name}.procs_before.txt").write_text("pid, process_name, used_gpu_memory [MiB]\n", encoding="utf-8")
            (folder / "run3_A.procs_before.txt").write_text(
                "pid, process_name, used_gpu_memory [MiB]\n4242, /usr/bin/python3, 3100 MiB\n", encoding="utf-8"
            )
            (folder / "summary.txt").write_text("old", encoding="utf-8")
            (folder / "run5_A.txt").write_text("Traceback: the run failed", encoding="utf-8")
            text = summarize.report(folder)
        self.assertIn("run1_A", text)
        self.assertIn("0.2860", text)  # 0.572 / 2
        self.assertIn("0.3242", text)  # 1.297 / 4
        self.assertRegex(text, r"run5_A\s+\(no benchmark summary")
        self.assertIn("other GPU processes before the run: 4242, /usr/bin/python3, 3100 MiB", text)
        self.assertNotIn("old", text)
        self.assertIn("A: 0.2860, 0.3200", text)
        self.assertIn("fastest: A by", text)
        self.assertIn("no throttle reasons during 4 busy samples", text)


if __name__ == "__main__":
    unittest.main()
