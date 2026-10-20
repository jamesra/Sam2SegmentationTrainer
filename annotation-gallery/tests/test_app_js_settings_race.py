"""app.js loadSettings, saveSettings and loadVolumes: a response that was overtaken must not overwrite newer state.

Same approach as test_app_js_catalog_race.py: the functions are cut out of the source text and run in node's vm
over stubs, with a fetch whose responses the test resolves in any order. They skip when node is not installed.
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


def _function(source: str, name: str) -> str:
    """The text of one top-level function plus the makeLatest guard created just before it, if any."""
    match = re.search(rf"(?:  const \w+Latest = makeLatest\(\);\n\n)?  (?:async )?function {name}\(.*?\n  \}}\n", source, re.S)
    assert match, f"{name} not found in app.js"
    return match.group(0)


def _code() -> str:
    source = _source()
    helper = re.search(r"// catalog-latest-begin\n(.*?)// catalog-latest-end", source, re.S)
    assert helper, "makeLatest block markers are missing from app.js"
    return helper.group(1) + "".join(_function(source, name) for name in ("applyRefreshVisibility", "loadVolumes", "loadSettings", "saveSettings"))


def _run_node(script: str) -> dict:
    result = subprocess.run([NODE, "-e", script], capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _harness(body: str) -> str:
    """Node script that defines the three functions over stubs, runs body, and prints what the page would show."""
    return f"""
const vm = require("vm");
const code = {json.dumps(_code())};
const requests = [];
const thrown = [];
const events = [];
const state = {{ volume: "", trainingSet: "v1", settings: null, me: null, config: {{ identityMode: "local" }} }};
const dialog = {{ hidden: false }};
const fields = {{
  "settings-odata": {{ value: "" }}, "settings-exporter": {{ value: "tiled" }}, "settings-dialog": dialog,
}};
const els = {{
  who: {{ textContent: "" }}, logout: {{ hidden: true }}, status: {{ textContent: "" }},
  volume: {{ innerHTML: "", value: "", style: {{}}, closest: () => null }},
}};
function deferred() {{
  let resolve, reject;
  const promise = new Promise((res, rej) => {{ resolve = res; reject = rej; }});
  return {{ promise, resolve, reject }};
}}
// Every getJson call is parked until the test answers it: respond / fail pick the nth request whose url has a fragment.
function request(fragment, nth = 0) {{
  const hits = requests.filter((item) => item.url.includes(fragment));
  if (!hits[nth]) throw new Error("no request " + fragment + " #" + nth + " in " + requests.map((i) => i.url));
  return hits[nth];
}}
const respond = (fragment, value, nth = 0) => request(fragment, nth).d.resolve(value);
const fail = (fragment, message, nth = 0) => request(fragment, nth).d.reject(new Error(message));
const tick = () => new Promise((resolve) => setTimeout(resolve, 0));
const context = {{
  state, els,
  getJson: (url, options) => {{ const d = deferred(); requests.push({{ url, options, d }}); return d.promise; }},
  withSet: (url) => url + (url.includes("?") ? "&" : "?") + "set=" + state.trainingSet,
  readView: () => ({{ volumes: {{}} }}),
  rememberView: () => {{}},
  render: () => events.push("render"),
  loadCatalog: async () => {{ events.push("catalog:" + state.volume + "@" + state.trainingSet); }},
  document: {{
    body: {{ classList: {{ toggle: (name, on) => events.push("refresh:" + Boolean(on)) }} }},
    getElementById: (id) => fields[id],
  }},
}};
vm.createContext(context);
vm.runInContext(
  code + "; this.loadVolumes = loadVolumes; this.loadSettings = loadSettings; this.saveSettings = saveSettings;",
  context);
(async () => {{
  const {{ loadVolumes, loadSettings, saveSettings }} = context;
{body}
  console.log(JSON.stringify({{
    settings: state.settings, volume: state.volume, me: state.me && state.me.displayName,
    options: els.volume.innerHTML, status: els.status.textContent, dialogHidden: dialog.hidden, thrown, events,
    posts: requests.filter((item) => item.options).map((item) => JSON.parse(item.options.body)),
  }}));
}})();
"""


@needs_node
def test_app_js_parses() -> None:
    result = subprocess.run([NODE, "--check", str(APP_JS)], capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stderr


@needs_node
def test_slow_settings_for_previous_volume_are_dropped() -> None:
    # Volume A's settings are requested first and answer last; B was selected in between.
    out = _run_node(_harness("""
      state.volume = "A";
      const loadA = loadSettings();
      state.volume = "B";
      const loadB = loadSettings();
      respond("/volumes/B/settings", { odata: "http://b" });
      await loadB;
      respond("/volumes/A/settings", { odata: "http://a" });
      await loadA;
    """))
    assert out["settings"] == {"odata": "http://b"}
    assert out["events"] == ["refresh:true"]


@needs_node
def test_settings_are_dropped_after_volume_changes_before_next_load_starts() -> None:
    out = _run_node(_harness("""
      state.volume = "A";
      const loadA = loadSettings();
      state.volume = "B";
      respond("/volumes/A/settings", { odata: "http://a" });
      await loadA;
    """))
    assert out["settings"] is None
    assert out["events"] == []


@needs_node
def test_stale_settings_failure_is_swallowed_and_current_failure_still_throws() -> None:
    out = _run_node(_harness("""
      state.volume = "A";
      const loadA = loadSettings();
      state.volume = "B";
      const loadB = loadSettings();
      fail("/volumes/A/settings", "A failed");
      try { await loadA; } catch (error) { thrown.push("A:" + error.message); }
      fail("/volumes/B/settings", "B failed");
      try { await loadB; } catch (error) { thrown.push("B:" + error.message); }
    """))
    assert out["thrown"] == ["B:B failed"]


@needs_node
def test_clearing_the_volume_supersedes_an_inflight_settings_load() -> None:
    out = _run_node(_harness("""
      state.settings = { odata: "http://old" };
      state.volume = "A";
      const loadA = loadSettings();
      state.volume = "";
      await loadSettings();
      respond("/volumes/A/settings", { odata: "http://a" });
      await loadA;
    """))
    assert out["settings"] is None


@needs_node
def test_overtaken_save_response_does_not_replace_the_newest_one() -> None:
    # Save is clicked twice while the dialog is open; the first answer arrives after the second.
    out = _run_node(_harness("""
      state.volume = "A";
      fields["settings-odata"].value = " http://one ";
      const first = saveSettings();
      fields["settings-odata"].value = "http://two";
      const second = saveSettings();
      respond("/volumes/A/settings", { odata: "http://two" }, 1);
      await second;
      respond("/volumes/A/settings", { odata: "http://one" }, 0);
      await first;
    """))
    assert out["settings"] == {"odata": "http://two"}
    assert out["status"] == "Saved the endpoint for this volume."
    assert out["dialogHidden"] is True
    # Both saves were sent: dropping a response never drops the write itself.
    assert [post["odata"] for post in out["posts"]] == ["http://one", "http://two"]


@needs_node
def test_older_save_answering_first_leaves_the_dialog_for_the_newest() -> None:
    out = _run_node(_harness("""
      state.volume = "A";
      const first = saveSettings();
      const second = saveSettings();
      respond("/volumes/A/settings", { odata: "http://one" }, 0);
      await first;
      events.push("dialog:" + dialog.hidden);
      respond("/volumes/A/settings", { odata: "http://two" }, 1);
      await second;
    """))
    assert out["events"] == ["dialog:false", "refresh:true"]
    assert out["settings"] == {"odata": "http://two"}
    assert out["dialogHidden"] is True


@needs_node
def test_save_failures_only_surface_for_the_newest_save() -> None:
    out = _run_node(_harness("""
      state.volume = "A";
      state.settings = { odata: "http://before" };
      const first = saveSettings();
      const second = saveSettings();
      fail("/volumes/A/settings", "first failed", 0);
      try { await first; } catch (error) { thrown.push("1:" + error.message); }
      fail("/volumes/A/settings", "second failed", 1);
      try { await second; } catch (error) { thrown.push("2:" + error.message); }
    """))
    assert out["thrown"] == ["2:second failed"]
    assert out["dialogHidden"] is False
    assert out["settings"] == {"odata": "http://before"}


@needs_node
def test_save_response_for_a_previous_volume_does_not_become_the_new_volumes_settings() -> None:
    out = _run_node(_harness("""
      state.volume = "A";
      state.settings = { odata: "http://a-before" };
      const saving = saveSettings();
      state.volume = "B";
      state.settings = { odata: "http://b" };
      respond("/volumes/A/settings", { odata: "http://a-saved" });
      await saving;
    """))
    assert out["settings"] == {"odata": "http://b"}
    assert out["status"] == ""


@needs_node
def test_overtaken_volume_list_is_dropped() -> None:
    # Training set v1 is requested first and answers last; v2 was selected in between.
    out = _run_node(_harness("""
      state.trainingSet = "v1";
      const loadV1 = loadVolumes();
      state.trainingSet = "v2";
      const loadV2 = loadVolumes();
      respond("/api/me?set=v2", { displayName: "two", volumes: [{ name: "B2", permission: "read" }] });
      await tick();
      respond("/volumes/B2/settings", { odata: "http://b2" });
      await loadV2;
      respond("/api/me?set=v1", { displayName: "one", volumes: [{ name: "A1", permission: "read" }] });
      await loadV1;
    """))
    assert out["me"] == "two"
    assert out["volume"] == "B2"
    assert "A1" not in out["options"]
    assert out["settings"] == {"odata": "http://b2"}
    assert out["events"] == ["refresh:true", "catalog:B2@v2"]


@needs_node
def test_volume_list_overtaken_while_its_settings_load_does_not_load_the_catalog() -> None:
    out = _run_node(_harness("""
      state.trainingSet = "v1";
      const loadV1 = loadVolumes();
      respond("/api/me?set=v1", { displayName: "one", volumes: [{ name: "A1", permission: "read" }] });
      await tick();
      state.trainingSet = "v2";
      const loadV2 = loadVolumes();
      respond("/volumes/A1/settings", { odata: "http://a1" });
      await loadV1;
      respond("/api/me?set=v2", { displayName: "two", volumes: [{ name: "B2", permission: "read" }] });
      await tick();
      respond("/volumes/B2/settings", { odata: "http://b2" });
      await loadV2;
    """))
    assert out["volume"] == "B2"
    assert [event for event in out["events"] if event.startswith("catalog")] == ["catalog:B2@v2"]


@needs_node
def test_overtaken_volume_list_failure_is_swallowed_and_current_failure_still_throws() -> None:
    out = _run_node(_harness("""
      state.trainingSet = "v1";
      const loadV1 = loadVolumes();
      state.trainingSet = "v2";
      const loadV2 = loadVolumes();
      fail("/api/me?set=v1", "v1 failed");
      try { await loadV1; } catch (error) { thrown.push("v1:" + error.message); }
      fail("/api/me?set=v2", "v2 failed");
      try { await loadV2; } catch (error) { thrown.push("v2:" + error.message); }
    """))
    assert out["thrown"] == ["v2:v2 failed"]
