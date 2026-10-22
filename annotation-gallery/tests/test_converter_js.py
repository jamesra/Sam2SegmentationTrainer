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


@needs_node
def test_failure_reasons_are_put_in_words() -> None:
    got = _run(
        "return ["
        "C.humanReason('too_large: the shape is 9000 mosaic pixels across'),"
        "C.humanReason('RESOURCE_EXHAUSTED: tiles over limit 64'),"
        "C.humanReason('UNAVAILABLE: gpu lost'),"
        "C.humanReason('error: boom'),"
        "C.humanReason('failed: circle radius is below one pixel at this downsample'),"
        "C.humanReason(''), C.humanReason(null)];"
    )
    assert got[0] == "Too big to fit one card at any zoom level"
    assert got[1] == "Beyond the segmentation server's size limits" and got[2] == "The segmentation server was not available"
    assert got[3] == "Unexpected error: boom"
    assert got[4] == "circle radius is below one pixel at this downsample"
    assert got[5] == "Unknown reason" and got[6] == "Unknown reason"


@needs_node
def test_build_labels_and_counts() -> None:
    got = _run(
        "const b = {volume: 'RC2', structureId: 1968, built: 42, approved: 10, rejected: 3, failed: 3, running: false};"
        "return [C.buildLabel(b), C.buildLabel({...b, running: true}), C.countsLine(b), C.countsLine({built: 5, approved: 0, rejected: 0, failed: 0}), C.countsLine(null)];"
    )
    assert got[0] == "RC2 · structure 1968 — 29 of 42 to review"
    assert got[1].endswith("(building…)")
    assert got[2] == "42 built · 10 approved · 3 rejected · 29 to review · 3 failed"
    assert got[3] == "5 built · 0 approved · 5 to review" and got[4] == ""


@needs_node
def test_the_preview_picks_the_volume_that_has_the_structure_and_warns_about_ambiguity() -> None:
    got = _run(
        "const v = {RC1: {count: 0}, RC2: {count: 42}};"
        "return [C.pickVolume(v), C.pickVolume({RC1: {count: 0}}), C.pickVolume({RC1: {error: 'x'}}), C.pickVolume(null),"
        "C.previewLine(v, 'RC2'), C.previewLine({RC1: {count: 1}}, 'RC1'),"
        "C.previewLine({RC1: {count: 0}, RC2: {count: 0}}, ''), C.previewLine({RC1: {error: 'x'}, RC2: {count: 0}}, ''),"
        "C.previewLine({RC1: {count: 3}, RC2: {count: 9}}, 'RC1'), C.previewLine({}, '')];"
    )
    assert got[0] == "RC2" and got[1] == "" and got[2] == "" and got[3] == ""
    assert got[4] == "42 annotations to convert in RC2."
    assert got[5] == "1 annotation to convert in RC1."
    assert "No matching annotations in RC1, RC2" in got[6]
    assert "Could not check RC1" in got[7]
    assert got[8].startswith("3 annotations to convert in RC1.") and "also exists in RC2 (9)" in got[8]
    assert "review rights" in got[9]
