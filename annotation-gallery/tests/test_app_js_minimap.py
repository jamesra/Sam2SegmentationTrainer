"""app.js minimapGeometry: the Process / branch mini-map's size and the screen's box on it.

The function is cut out of app.js between its markers and run in node's vm. The tests skip when node is not installed.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / "static"
NODE = shutil.which("node")

needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")

# A 256px card with a 4px gap, rows 276px apart, on a 1600x900 scroller whose grid starts 16px in.
BASE = {
    "columns": 10, "rows": 6, "cardPx": 256, "colPitch": 260, "rowPitch": 276,
    "viewW": 1600, "viewH": 900, "scrollLeft": 16, "scrollTop": 0, "gridLeft": 16,
}


def _geometry(**overrides: float) -> dict:
    source = (STATIC / "app.js").read_text(encoding="utf-8")
    match = re.search(r"// minimap-begin\n(.*?)// minimap-end", source, re.S)
    assert match, "minimap block markers are missing from app.js"
    args = json.dumps({**BASE, **overrides})
    script = f"{match.group(1)}\nprocess.stdout.write(JSON.stringify(minimapGeometry({args})));"
    result = subprocess.run([NODE, "-e", script], capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@needs_node
def test_small_layout_uses_4px_squares_with_1px_gaps() -> None:
    geo = _geometry()
    assert (geo["width"], geo["height"]) == (49, 29)
    assert (geo["pitch"], geo["tile"]) == (5, 4)


@needs_node
def test_large_layout_shrinks_to_one_card_area_and_keeps_its_shape() -> None:
    geo = _geometry(columns=200, rows=100)
    assert geo["width"] * geo["height"] <= 256 * 256 + 256 + 256
    assert geo["pitch"] < 5
    assert abs(geo["width"] / geo["height"] - 999 / 499) < 0.05


@needs_node
def test_view_box_follows_the_scroll() -> None:
    # Three columns and two rows in, the box starts at their squares.
    geo = _geometry(columns=40, rows=20, scrollLeft=16 + 3 * 260, scrollTop=2 * 276, viewW=4 * 260, viewH=2 * 276)
    assert geo["pitch"] == 5
    assert geo["view"] == {"x": 15, "y": 10, "w": 20, "h": 10}


@needs_node
def test_view_box_is_clamped_to_the_map() -> None:
    geo = _geometry(columns=4, rows=2, scrollLeft=0, viewW=1600, viewH=900)
    view = geo["view"]
    assert view["x"] == 0 and view["y"] == 0
    assert view["x"] + view["w"] <= geo["width"]
    assert view["y"] + view["h"] <= geo["height"]
