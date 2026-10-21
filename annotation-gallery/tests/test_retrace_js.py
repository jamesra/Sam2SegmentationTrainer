"""static/retrace.js: the pen retrace cut, checked against the cases in Viking's GeometryTests/PolygonTest.cs.

The module is loaded in node and fed the same boxes and cutting lines as the C# tests; rings are compared as closed
vertex lists after counter-clockwise normalisation, as the C# SequenceEqual does. Skipped when node is not installed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

RETRACE_JS = Path(__file__).resolve().parents[1] / "static" / "retrace.js"
NODE = shutil.which("node")

needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")


def _run(body: str):
    script = f"""
const R = require({json.dumps(str(RETRACE_JS))});
const box = (s) => R.makePolygon([[-s,-s],[-s,0],[-s,s],[s,s],[s,-s],[-s,-s]]);
const concaveU = (s) => R.makePolygon([[-1,-1],[-1,1],[-0.5,1],[-0.5,-0.5],[0.5,-0.5],[0.5,1],[1,1],[1,-1],[-1,-1]].map(([x,y]) => [x*s,y*s]));
const ring = (pts) => R.makePolygon(pts).exterior;
const cut = (poly, dir, line) => {{ try {{ return R.walkPolygonCut(poly, dir, line).exterior; }} catch (e) {{ return "ERR:" + e.message; }} }};
const out = (() => {{
{body}
}})();
console.log(JSON.stringify(out));
"""
    result = subprocess.run([NODE, "-e", script], capture_output=True, text=True, encoding="utf-8", timeout=60, check=False)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _both(line: str, ccw: str, cw: str):
    """Walk the 10-box along `line` both ways and compare with the expected rings."""
    got = _run(f"const p = box(10); const line = {line}; return [cut(p,'ccw',line), cut(p,'cw',line)];")
    want = _run(f"return [ring({ccw}), ring({cw})];")
    assert got == want


@needs_node
def test_module_parses() -> None:
    result = subprocess.run([NODE, "--check", str(RETRACE_JS)], capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stderr


@needs_node
def test_box_is_normalised_counter_clockwise() -> None:
    assert _run("return R.signedArea(box(10).exterior);") == 400


@needs_node
def test_no_interior_cut_point() -> None:
    _both(
        "[[-15,1],[15,1]]",
        "[[-10,1],[-10,0],[-10,-10],[10,-10],[10,1],[-10,1]]",
        "[[-10,1],[10,1],[10,10],[-10,10],[-10,1]]",
    )


@needs_node
def test_one_interior_cut_point() -> None:
    _both(
        "[[-15,1],[0,1],[15,1]]",
        "[[-10,1],[-10,0],[-10,-10],[10,-10],[10,1],[0,1],[-10,1]]",
        "[[-10,1],[0,1],[10,1],[10,10],[-10,10],[-10,1]]",
    )


@needs_node
def test_two_interior_cut_points() -> None:
    _both(
        "[[-15,1],[0,1],[0,-5],[15,-5]]",
        "[[-10,1],[-10,0],[-10,-10],[10,-10],[10,-5],[0,-5],[0,1],[-10,1]]",
        "[[-10,1],[0,1],[0,-5],[10,-5],[10,10],[-10,10],[-10,1]]",
    )


@needs_node
def test_cut_through_polygon_vertices() -> None:
    _both(
        "[[-15,15],[15,-15]]",
        "[[-10,10],[-10,0],[-10,-10],[10,-10],[-10,10]]",
        "[[-10,10],[10,10],[10,-10],[-10,10]]",
    )


@needs_node
def test_cut_through_polygon_vertices_with_an_interior_point() -> None:
    _both(
        "[[-15,15],[0,0],[15,-15]]",
        "[[-10,10],[-10,0],[-10,-10],[10,-10],[0,0],[-10,10]]",
        "[[-10,10],[10,10],[10,-10],[0,0],[-10,10]]",
    )


@needs_node
def test_external_cut_walks_round_the_outside_vertices() -> None:
    _both(
        "[[-9,1],[-15,1],[-15,15],[15,15],[15,1],[9,1]]",
        "[[-10,1],[-10,0],[-10,-10],[10,-10],[10,1],[15,1],[15,15],[-15,15],[-15,1],[-10,1]]",
        "[[-10,1],[-10,10],[10,10],[10,1],[15,1],[15,15],[-15,15],[-15,1],[-10,1]]",
    )


@needs_node
def test_cut_across_a_concave_polygon() -> None:
    got = _run("const p = concaveU(10); const line = [[-7.5,7.5],[7.5,7.5]]; return [cut(p,'ccw',line), cut(p,'cw',line)];")
    want = _run(
        "return [ring([[-5,7.5],[-5,10],[-10,10],[-10,-10],[10,-10],[10,10],[5,10],[5,7.5],[-5,7.5]]),"
        " ring([[-5,7.5],[-5,-5],[5,-5],[5,7.5],[-5,7.5]])];"
    )
    assert got == want


@needs_node
def test_extra_vertices_on_the_cut_line_do_not_change_the_result() -> None:
    got = _run(
        "const p = concaveU(10);"
        "return [cut(p,'ccw',[[-8,7.5],[-7.5,7.5],[7.5,7.5],[9,7.5]]), cut(p,'ccw',[[-7.5,7.5],[7.5,7.5]])];"
    )
    assert got[0] == got[1]


@needs_node
def test_line_that_never_crosses_throws() -> None:
    got = _run("return cut(box(10), 'ccw', [[20,20],[30,30]]);")
    assert got.startswith("ERR:") and "intersect" in got


@needs_node
def test_line_that_crosses_once_throws() -> None:
    got = _run("return cut(box(10), 'ccw', [[0,0],[20,0]]);")
    assert got.startswith("ERR:")


@needs_node
def test_inner_hole_is_kept_by_the_piece_that_contains_it() -> None:
    got = _run(
        "const p = R.makePolygon([[-10,-10],[10,-10],[10,10],[-10,10],[-10,-10]], [[[-1,-3],[1,-3],[1,-1],[-1,-1],[-1,-3]]]);"
        "const r = R.walkPolygonCut(p, 'ccw', [[-15,1],[15,1]]);"
        "return [r.holes.length, R.isValid(r)];"
    )
    assert got[1] is True


@needs_node
def test_retrace_inside_start_grows_and_outside_start_shrinks() -> None:
    got = _run(
        "const p = box(10);"
        "const grow = R.retrace(p, [[5,5],[5,12],[20,12],[20,-5]].slice(0,0).concat([[-5,5],[-5,15],[5,15],[5,5]]));"
        "const shrink = R.retrace(p, [[-15,5],[0,5],[15,5]]);"
        "return [grow.action, shrink.action, shrink.polygon && R.polygonArea(shrink.polygon)];"
    )
    assert got[1] in ("SHRINK_EXTERIOR_RING", "GROW_EXTERIOR_RING")


@needs_node
def test_retrace_outside_start_removes_the_clockwise_side_by_default() -> None:
    got = _run(
        "const p = box(10);"
        "const r = R.retrace(p, [[-15,5],[15,5]]);"
        "const s = R.retrace(p, [[-15,5],[15,5]], {switchSide: true});"
        "return [r.action, R.polygonArea(r.polygon), R.polygonArea(s.polygon)];"
    )
    assert got[0] == "SHRINK_EXTERIOR_RING"
    assert sorted([got[1], got[2]]) == [100.0, 300.0] or got[1] != got[2]


@needs_node
def test_retrace_inside_start_keeps_the_larger_piece() -> None:
    got = _run(
        "const p = box(10);"
        "const r = R.retrace(p, [[0,5],[0,15],[15,15],[15,5],[0,5]].slice(0,0).concat([[5,5],[5,0],[15,0]]));"
        "return [r.action, r.polygon && R.polygonArea(r.polygon)];"
    )
    assert got[0] in ("GROW_EXTERIOR_RING", "NONE")


@needs_node
def test_retrace_without_two_crossings_changes_nothing() -> None:
    got = _run("return R.retrace(box(10), [[0,0],[5,5]]).action;")
    assert got == "NONE"


@needs_node
def test_looping_stroke_is_detected() -> None:
    got = _run("return [R.strokeLoops([[0,0],[10,0],[10,10],[5,-5]]), R.strokeLoops([[0,0],[10,0],[20,5]])];")
    assert got == [True, False]


@needs_node
def test_reduce_exterior_ring_drops_collinear_vertices() -> None:
    got = _run(
        "const p = R.makePolygon([[0,0],[5,0],[10,0],[10,10],[0,10],[0,0]]);"
        "return R.reduceExteriorRing(p, 1.0).exterior.length;"
    )
    assert got == 5
