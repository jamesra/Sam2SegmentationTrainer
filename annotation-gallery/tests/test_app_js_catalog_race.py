"""app.js loadCatalog: a slow response for a previous volume or training set must not overwrite the newer one.

The script has no module system, so the tests cut the pieces they need out of the source text and run them in
node's vm with a stubbed fetch and state. They skip when node is not installed.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

APP_JS = Path(__file__).resolve().parents[1] / "static" / "app.js"
NODE = shutil.which("node")

needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")


def _source() -> str:
    return APP_JS.read_text(encoding="utf-8")


def _helper_source() -> str:
    match = re.search(r"// catalog-latest-begin\n(.*?)// catalog-latest-end", _source(), re.S)
    assert match, "makeLatest block markers are missing from app.js"
    return match.group(1)


def _load_catalog_source() -> str:
    source = _source()
    match = re.search(r"  async function loadCatalog\(.*?\n  \}\n", source, re.S)
    assert match, "loadCatalog not found in app.js"
    latest = re.search(r"  const catalogLatest = makeLatest\(\);\n", source)
    assert latest, "catalogLatest is not created from makeLatest in app.js"
    return latest.group(0) + match.group(0)


def _run_node(script: str) -> dict:
    result = subprocess.run([NODE, "-e", script], capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@needs_node
def test_app_js_parses() -> None:
    result = subprocess.run([NODE, "--check", str(APP_JS)], capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stderr


@needs_node
def test_slow_response_for_previous_volume_is_dropped() -> None:
    # Volume A's catalog is requested first and answers last; B was selected in between.
    script = _harness("""
      const slowA = deferred();
      const fastB = deferred();
      state.volume = "A";
      pending["A"] = slowA;
      pending["B"] = fastB;
      const loadA = loadCatalog(true);
      state.volume = "B";
      const loadB = loadCatalog(true);
      fastB.resolve({ rows: [{ location_id: 2 }], cropSize: 512, permission: "review" });
      await loadB;
      slowA.resolve({ rows: [{ location_id: 1 }], cropSize: 2048, permission: "read" });
      await loadA;
    """)
    out = _run_node(script)
    assert out["rows"] == [2]
    assert out["cropSize"] == 512
    assert out["permission"] == "review"
    assert out["renders"] == 1


@needs_node
def test_response_is_dropped_after_volume_changes_before_next_load_starts() -> None:
    # The change handler sets state.volume, then awaits loadSettings; no newer loadCatalog exists yet.
    script = _harness("""
      const slowA = deferred();
      state.volume = "A";
      pending["A"] = slowA;
      const loadA = loadCatalog(true);
      state.volume = "B";
      slowA.resolve({ rows: [{ location_id: 1 }], cropSize: 2048, permission: "review" });
      await loadA;
    """)
    out = _run_node(script)
    assert out["rows"] == []
    assert out["renders"] == 0


@needs_node
def test_response_is_dropped_after_training_set_changes() -> None:
    script = _harness("""
      const slow = deferred();
      state.volume = "A";
      pending["A"] = slow;
      const load = loadCatalog(true);
      state.trainingSet = "v2.0";
      slow.resolve({ rows: [{ location_id: 1 }], cropSize: 2048, permission: "review" });
      await load;
    """)
    out = _run_node(script)
    assert out["rows"] == []
    assert out["renders"] == 0


@needs_node
def test_newest_request_for_same_volume_wins() -> None:
    # Two reloads of one volume (Update All poll, then a locate import): the older answer must not replace the newer.
    script = _harness("""
      const first = deferred();
      const second = deferred();
      state.volume = "A";
      pending["A"] = first;
      const loadFirst = loadCatalog(false, true);
      pending["A"] = second;
      const loadSecond = loadCatalog(false, true);
      second.resolve({ rows: [{ location_id: 1 }, { location_id: 2 }], cropSize: 1024, permission: "review" });
      await loadSecond;
      first.resolve({ rows: [{ location_id: 1 }], cropSize: 1024, permission: "review" });
      await loadFirst;
    """)
    out = _run_node(script)
    assert out["rows"] == [1, 2]


@needs_node
def test_stale_failure_is_swallowed_and_current_failure_still_throws() -> None:
    script = _harness("""
      const slowA = deferred();
      const fastB = deferred();
      state.volume = "A";
      pending["A"] = slowA;
      pending["B"] = fastB;
      const loadA = loadCatalog(true);
      state.volume = "B";
      const loadB = loadCatalog(true);
      slowA.reject(new Error("A failed"));
      try { await loadA; } catch (error) { thrown.push("A:" + error.message); }
      fastB.reject(new Error("B failed"));
      try { await loadB; } catch (error) { thrown.push("B:" + error.message); }
    """)
    out = _run_node(script)
    assert out["thrown"] == ["B:B failed"]


@needs_node
def test_no_volume_clears_rows_and_supersedes_inflight_load() -> None:
    script = _harness("""
      const slowA = deferred();
      state.volume = "A";
      pending["A"] = slowA;
      const loadA = loadCatalog(true);
      state.volume = "";
      await loadCatalog(true);
      slowA.resolve({ rows: [{ location_id: 1 }], cropSize: 2048, permission: "review" });
      await loadA;
    """)
    out = _run_node(script)
    assert out["rows"] == []


def _harness(body: str) -> str:
    """Node script that defines loadCatalog from app.js over stubs, runs body, and prints the resulting state."""
    return f"""
const vm = require("vm");
const code = {json.dumps(_helper_source() + _load_catalog_source())};
const pending = {{}};
const thrown = [];
let renders = 0;
const state = {{ volume: "", trainingSet: "current", rows: [], cropSize: 1024, permission: "read",
                hasSam2: false, sorts: [{{ key: "z", desc: false }}] }};
const viewer = {{ locationId: "" }};
function deferred() {{
  let resolve, reject;
  const promise = new Promise((res, rej) => {{ resolve = res; reject = rej; }});
  return {{ promise, resolve, reject }};
}}
const context = {{
  state, viewer,
  getJson: (url) => pending[decodeURIComponent(url.split("/")[3])].promise,
  withSet: (url) => url,
  indexCatalog: () => {{}},
  visibleClasses: () => new Set(["pending"]),
  reviewClass: () => "pending",
  closeViewer: () => {{}},
  renderSortStack: () => {{}},
  watchCatalogRescores: () => {{}},
  indexSplit: () => null,
  applySplitControl: () => {{}},
  CARD_SORT_WORST_DESC: new Map(),
  render: () => {{ renders += 1; }},
  document: {{ body: {{ classList: {{ toggle: () => {{}} }} }} }},
}};
vm.createContext(context);
vm.runInContext(code + "; this.loadCatalog = loadCatalog;", context);
(async () => {{
  const loadCatalog = context.loadCatalog;
{body}
  console.log(JSON.stringify({{
    rows: state.rows.map((row) => row.location_id),
    cropSize: state.cropSize, permission: state.permission, renders, thrown,
  }}));
}})();
"""
