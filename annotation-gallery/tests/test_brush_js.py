"""static/brush_ui.js BrushCore: the painting maths and the undo history, in node without a DOM."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

BRUSH_JS = Path(__file__).resolve().parents[1] / "static" / "brush_ui.js"
NODE = shutil.which("node")

needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")


def _run(body: str):
    script = f"const {{ BrushCore: B }} = require({json.dumps(str(BRUSH_JS))});\nconsole.log(JSON.stringify((() => {{\n{body}\n}})()));"
    result = subprocess.run([NODE, "-e", script], capture_output=True, text=True, encoding="utf-8", timeout=60, check=False)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@needs_node
def test_a_stamp_sets_the_pixels_whose_centres_are_inside_the_disc() -> None:
    got = _run(
        "const m = new Uint8Array(40 * 40); B.stamp(m, 40, 40, 20, 20, 5, 1);"
        "return [B.count(m), m[19 * 40 + 19], m[0], m[20 * 40 + 26]];"
    )
    # A disc of radius 5 centred on a pixel corner holds 80 or 81 centres; never off by more than a couple.
    assert 76 <= got[0] <= 84 and got[1] == 1 and got[2] == 0 and got[3] == 0


@needs_node
def test_stamping_zero_erases_and_the_edge_is_clipped() -> None:
    got = _run(
        "const m = new Uint8Array(20 * 20).fill(1); B.stamp(m, 20, 20, 10, 10, 3, 0);"
        "const hole = 400 - B.count(m); const e = new Uint8Array(20 * 20); B.stamp(e, 20, 20, 0, 0, 4, 1);"
        "return [hole, B.count(e)];"
    )
    assert 24 <= got[0] <= 32 and 10 <= got[1] <= 16


@needs_node
def test_a_stroke_has_no_gaps_even_when_the_pointer_jumps() -> None:
    got = _run(
        "const m = new Uint8Array(100 * 20); B.line(m, 100, 20, 5, 10, 95, 10, 2, 1);"
        "let gaps = 0; for (let x = 6; x < 94; x++) { if (!m[10 * 100 + x] && !m[9 * 100 + x]) gaps++; }"
        "return [gaps, m[10 * 100 + 50], m[0]];"
    )
    assert got == [0, 1, 0]


@needs_node
def test_a_single_point_stroke_paints_one_disc() -> None:
    got = _run("const m = new Uint8Array(30 * 30); B.line(m, 30, 30, 15, 15, 15, 15, 3, 1); return B.count(m);")
    assert 24 <= got <= 32


@needs_node
def test_changed_counts_differing_pixels_and_from_predicate_builds_the_mask() -> None:
    got = _run(
        "const a = B.fromPredicate(10, 10, (x, y) => x < 5); const b = a.slice(); b[0] = 0; b[99] = 1;"
        "return [B.count(a), B.changed(a, b), B.changed(a, a)];"
    )
    assert got == [50, 2, 0]


@needs_node
def test_history_is_bounded_and_returns_the_latest_snapshot_first() -> None:
    got = _run(
        "const h = B.history(3); const m = new Uint8Array(4);"
        "for (let i = 1; i <= 5; i++) { m[0] = i; h.push(m); }"
        "return [h.size, h.pop()[0], h.pop()[0], h.pop()[0], h.pop()];"
    )
    assert got == [3, 5, 4, 3, None]
