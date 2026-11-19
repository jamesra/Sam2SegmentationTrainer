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
    "columns": 10, "rows": 6, "colPitch": 260, "rowPitch": 276,
    "viewW": 1600, "viewH": 900, "scrollLeft": 16, "scrollTop": 0, "gridLeft": 16,
}


def _run(call: str) -> dict:
    source = (STATIC / "app.js").read_text(encoding="utf-8")
    match = re.search(r"// minimap-begin\n(.*?)// minimap-end", source, re.S)
    assert match, "minimap block markers are missing from app.js"
    script = f"{match.group(1)}\nprocess.stdout.write(JSON.stringify({call}));"
    result = subprocess.run([NODE, "-e", script], capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _geometry(**overrides: float) -> dict:
    return _run(f"minimapGeometry({json.dumps({**BASE, **overrides})})")


@needs_node
def test_small_layout_uses_8px_squares_with_1px_gaps() -> None:
    geo = _geometry()
    assert (geo["width"], geo["height"]) == (89, 53)
    assert (geo["pitch"], geo["tile"]) == (9, 8)


@needs_node
def test_large_layout_shrinks_to_stay_on_screen_and_keeps_its_shape() -> None:
    geo = _geometry(columns=400, rows=100)
    assert geo["width"] <= 1600 - 16 and geo["height"] <= 900 - 16
    assert geo["pitch"] < 9
    assert abs(geo["width"] / geo["height"] - 3599 / 899) < 0.05


@needs_node
def test_layout_bigger_than_a_card_is_not_shrunk_while_it_fits_the_screen() -> None:
    geo = _geometry(columns=60, rows=40)
    assert geo["pitch"] == 9


@needs_node
def test_view_box_follows_the_scroll() -> None:
    geo = _geometry(columns=40, rows=20, scrollLeft=16 + 3 * 260, scrollTop=2 * 276, viewW=4 * 260, viewH=2 * 276)
    assert geo["pitch"] == 9
    assert geo["view"] == {"x": 27, "y": 18, "w": 36, "h": 18}


@needs_node
def test_view_box_is_clamped_to_the_map() -> None:
    geo = _geometry(columns=4, rows=2, scrollLeft=0, viewW=1600, viewH=900)
    view = geo["view"]
    assert view["x"] == 0 and view["y"] == 0
    assert view["x"] + view["w"] <= geo["width"]
    assert view["y"] + view["h"] <= geo["height"]


@needs_node
def test_click_centres_the_screen_on_that_cell() -> None:
    # Column 10, row 4 of the map, centre of the square: the screen centres on that card.
    g = {"colPitch": 260, "rowPitch": 276, "viewW": 1040, "viewH": 552, "gridLeft": 16}
    out = _run(f"minimapTarget({{pitch: 9}}, 10 * 9 + 4.5, 4 * 9 + 4.5, {json.dumps(g)})")
    assert out["scrollLeft"] == 16 + 10.5 * 260 - 520
    assert out["scrollTop"] == 4.5 * 276 - 276


@needs_node
def test_click_near_the_origin_does_not_scroll_negative() -> None:
    g = {"colPitch": 260, "rowPitch": 276, "viewW": 1040, "viewH": 552, "gridLeft": 16}
    out = _run(f"minimapTarget({{pitch: 9}}, 1, 1, {json.dumps(g)})")
    assert out == {"scrollLeft": 0, "scrollTop": 0}
