"""static/gold.js: the helpers the page uses in gold mode, run in node without a DOM."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

GOLD_JS = Path(__file__).resolve().parents[1] / "static" / "gold.js"
NODE = shutil.which("node")

needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")


def _run(body: str):
    script = f"const G = require({json.dumps(str(GOLD_JS))});\nconsole.log(JSON.stringify((() => {{\n{body}\n}})()));"
    result = subprocess.run([NODE, "-e", script], capture_output=True, text=True, encoding="utf-8", timeout=60, check=False)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@needs_node
def test_active_needs_both_the_path_and_the_server_switch() -> None:
    got = _run(
        "return [G.active('/gold/', {gold: true}), G.active('/gold', {gold: true}), G.active('/', {gold: true}),"
        "G.active('/gold/', {gold: false}), G.active('/converter/', {gold: true}), G.onGoldPath('/gold/'), G.onGoldPath('/x')];"
    )
    assert got == [True, True, False, False, False, True, False]


@needs_node
def test_cards_attach_by_location_id() -> None:
    rows = _run(
        "const rows = [{location_id: 5}, {location_id: '6'}, {location_id: 7}];"
        "G.attachCards(rows, [{location_id: 5, edited: 1}, {location_id: 6, edited: 0}]); return rows;"
    )
    assert rows[0]["gold"]["edited"] == 1 and rows[1]["gold"]["edited"] == 0 and "gold" not in rows[2]


@needs_node
def test_progress_line_and_replacement_need() -> None:
    got = _run(
        "const p = {target: 100, cards: 98, approved: 37, edited: 12, rejected: 2, remaining: 61};"
        "return [G.progressLine('RC1', p), G.progressLine('RC1', {target: 100, cards: 100, approved: 0, edited: 0, rejected: 0, remaining: 100}),"
        "G.progressLine('RC1', null), G.needsReplacement(p), G.needsReplacement({target: 5, cards: 5}), G.needsReplacement(null)];"
    )
    assert got[0] == "RC1: 37 of 100 approved · 12 edited · 2 rejected · 61 left"
    assert got[1] == "RC1: 0 of 100 approved · 100 left" and got[2] == ""
    assert got[3:] == [True, False, False]


@needs_node
def test_info_lines_and_export_link() -> None:
    got = _run(
        "return [G.infoLines({edited: 1, changed_pixels: 420, approved_by: 'Pat <b>'}), G.infoLines({edited: 0}), G.infoLines(null),"
        "G.exportHref('gold-2026', 'RC1')];"
    )
    assert got[0] == ["edited: 420 px changed from the source mask", "approved by Pat b"]
    assert got[1] == ["not edited"] and got[2] == []
    assert got[3] == "/api/gold/sets/gold-2026/export?volume=RC1"
