"""static/converter.js: the helpers the page uses in converter mode, run in node without a DOM."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

CONVERTER_JS = Path(__file__).resolve().parents[1] / "static" / "converter.js"
NODE = shutil.which("node")

needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")


def _run(body: str):
    script = f"const C = require({json.dumps(str(CONVERTER_JS))});\nconsole.log(JSON.stringify((() => {{\n{body}\n}})()));"
    result = subprocess.run([NODE, "-e", script], capture_output=True, text=True, encoding="utf-8", timeout=60, check=False)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@needs_node
def test_active_needs_both_the_path_and_the_server_switch() -> None:
    got = _run(
        "return ["
        "C.active('/converter/', {converter: true}), C.active('/converter', {converter: true}),"
        "C.active('/', {converter: true}), C.active('/converter/', {converter: false}), C.active('/converter/x', {converter: true}),"
        "C.onConverterPath('/converter/'), C.onConverterPath('/')];"
    )
    assert got == [True, True, False, False, False, True, False]


@needs_node
def test_cards_are_attached_by_location_and_set_the_change_as_loss() -> None:
    rows = _run(
        "const rows = [{location_id: 11}, {location_id: '12'}, {location_id: 13}];"
        "C.attachCards(rows, [{location_id: 11, iou: 0.75, state: 'built'}, {location_id: 12, iou: null, state: 'built'}]);"
        "return rows;"
    )
    assert rows[0]["loss"] == 0.25 and rows[0]["conv"]["iou"] == 0.75
    assert "loss" not in rows[1] and rows[1]["conv"]["state"] == "built"
    assert "conv" not in rows[2]


@needs_node
def test_edge_pixels_outline_a_square_and_leave_the_inside_empty() -> None:
    got = _run(
        "const w = 20, h = 20;"
        "const edge = C.edgePixels(w, h, (x, y) => x >= 5 && x < 15 && y >= 5 && y < 15, 1);"
        "const at = (x, y) => edge[y * w + x];"
        "return [at(5, 5), at(14, 9), at(9, 5), at(9, 9), at(2, 2), Array.from(edge).reduce((a, b) => a + b, 0)];"
    )
    assert got == [1, 1, 1, 0, 0, 36]


@needs_node
def test_edge_pixels_of_a_mask_touching_the_border_still_mark_the_border() -> None:
    got = _run("const edge = C.edgePixels(6, 6, () => true, 1); return Array.from(edge).reduce((a, b) => a + b, 0);")
    assert got == 20


@needs_node
def test_info_lines_and_batch_link() -> None:
    got = _run(
        "return [C.infoLines({iou: 0.934, model_id: 'v5', state: 'built'}), C.infoLines({state: 'failed', reason: 'too_large: <b>'}),"
        "C.infoLines(null), C.batchHref('structure-8889', 'RC1')];"
    )
    assert got[0] == ["overlap with current outline 93%", "model v5"]
    assert got[1] == ["not built: too_large: b"]
    assert got[2] == []
    assert got[3] == "/api/volumes/RC1/converter/batch?set=structure-8889"
