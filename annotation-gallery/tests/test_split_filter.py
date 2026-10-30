"""The Split filter: the catalog carries the run's validation_split.json, and the page sorts cards by it.

The JavaScript is cut out of app.js between its split-begin / split-end markers and run in node's vm (skipped when
node is not installed); the markup and server checks run everywhere.
"""

from __future__ import annotations

import gzip
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

GALLERY_ROOT = Path(__file__).resolve().parents[1]
if str(GALLERY_ROOT) not in sys.path:
    sys.path.insert(0, str(GALLERY_ROOT))

import server  # noqa: E402
from annotation_crops.validation_split import write_split_file  # noqa: E402

STATIC = GALLERY_ROOT / "static"
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")


def test_catalog_carries_the_split_and_a_new_split_file_refreshes_the_cache(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(server, "list_catalog_rows", lambda crops: [{"location_id": 1}])
    body, packed = server.catalog_body(tmp_path, "RC1", "review", 1024)
    assert json.loads(body)["split"] is None
    write_split_file(tmp_path, run="sam2_em_v5", mode="zslab", validation=[1], training=[2, 3])
    body, packed = server.catalog_body(tmp_path, "RC1", "review", 1024)
    split = json.loads(gzip.decompress(packed))["split"]
    assert split["run"] == "sam2_em_v5" and split["validation"] == [1] and split["training"] == [2, 3]


def test_the_split_control_offers_all_validation_training_neither() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    select = re.search(r'<select id="split"[^>]*>(.*?)</select>', html, re.S)
    assert select, "no Split select"
    assert re.findall(r'<option value="([^"]*)"', select.group(1)) == ["", "validation", "training", "neither"]
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert 'split: document.getElementById("split")' in js
    assert "splitWanted && splitOf(state.split, row) !== splitWanted" in js
    assert 'els.split.addEventListener("change", render)' in js


@needs_node
def test_split_of_each_card() -> None:
    source = (STATIC / "app.js").read_text(encoding="utf-8")
    block = re.search(r"// split-begin\n(.*?)// split-end", source, re.S)
    assert block, "split-begin / split-end markers are missing from app.js"
    script = f"""
const vm = require("vm");
const context = {{}};
vm.createContext(context);
vm.runInContext({json.dumps(block.group(1))} + "\\nthis.indexSplit = indexSplit; this.splitOf = splitOf;", context);
const split = context.indexSplit({{run: "r", validation: [1, "2"], training: [3]}});
const out = [1, 2, 3, 4].map((id) => context.splitOf(split, {{location_id: String(id)}}));
out.push(context.splitOf(context.indexSplit(null), {{location_id: 1}}));
out.push(context.indexSplit({{run: "r"}}));
console.log(JSON.stringify(out));
"""
    result = subprocess.run([NODE, "-e", script], capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == ["validation", "validation", "training", "neither", None, None]
