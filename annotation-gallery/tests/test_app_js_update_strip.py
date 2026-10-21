"""Update all: progress, Cancel and the result live in a slim strip under the cards, and the dialog never covers a run.

Same approach as test_app_js_settings_race.py: the functions are cut out of app.js and run in node's vm over stubs.
They skip when node is not installed. The markup checks run everywhere.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / "static"
APP_JS = STATIC / "app.js"
NODE = shutil.which("node")

needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")


# ------------------------------------------------------------------------------------------------- the markup

class _Tree(HTMLParser):
    """id -> (tag, classes, ids of its ancestors), enough to say what an element sits inside."""

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[tuple[str, str]] = []
        self.found: dict[str, tuple[str, set[str], list[str]]] = {}
        self.order: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        element_id = values.get("id") or ""
        if element_id:
            self.found[element_id] = (tag, set((values.get("class") or "").split()), [i for _t, i in self.stack if i])
            self.order.append(element_id)
        if tag not in {"input", "br", "meta", "link", "img"}:
            self.stack.append((tag, element_id))

    def handle_endtag(self, tag: str) -> None:
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                return


def _page() -> _Tree:
    tree = _Tree()
    tree.feed((STATIC / "index.html").read_text(encoding="utf-8"))
    return tree


def test_progress_and_cancel_are_not_inside_the_modal_dialog() -> None:
    page = _page()
    for element in ("update-bar", "update-line", "update-cancel", "update-details-toggle", "update-detail"):
        assert "update-dialog" not in page.found[element][2], f"{element} would be covered by the dialog's backdrop"
        assert "update-strip" in page.found[element][2], f"{element} belongs in the strip"
    assert "update-dialog" in page.found["update-start"][2], "the dialog is still where a run is asked for"


def test_the_strip_follows_the_cards_in_the_page_flow_and_starts_hidden() -> None:
    page = _page()
    assert page.found["update-strip"][0] == "footer"
    assert page.order.index("update-strip") == page.order.index("grid") + 1 or page.order.index("update-strip") > page.order.index("scroller")
    assert "update-strip" not in page.found["scroller"][2]
    assert re.search(r'<footer id="update-strip"[^>]*\bhidden\b', (STATIC / "index.html").read_text(encoding="utf-8"))


def test_the_strip_is_in_flow_not_an_overlay() -> None:
    css = (STATIC / "app.css").read_text(encoding="utf-8")
    block = re.search(r"\.update-strip \{(.*?)\}", css, re.S)
    assert block, "no .update-strip rule"
    assert "position: fixed" not in block.group(1) and "position: absolute" not in block.group(1)
    assert "[hidden] { display: none; }" in css.split(".update-strip[hidden]")[1][:40] or ".update-strip[hidden]" in css


# ------------------------------------------------------------------------------------------------ the strip's words

def _view_source() -> str:
    source = APP_JS.read_text(encoding="utf-8")
    match = re.search(r"// update-strip-view-begin\n(.*?)// update-strip-view-end", source, re.S)
    assert match, "update-strip-view markers are missing from app.js"
    return match.group(1)


def _node(script: str) -> dict:
    result = subprocess.run([NODE, "-e", script], capture_output=True, text=True, encoding="utf-8", timeout=60, check=False)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _views(statuses: dict, cancelling: bool = False) -> dict:
    script = f"""
const vm = require("vm");
const context = {{}};
vm.createContext(context);
vm.runInContext({json.dumps(_view_source())} + "\\nthis.updateStripView = updateStripView;", context);
const statuses = {json.dumps(statuses)};
const out = {{}};
for (const [name, status] of Object.entries(statuses)) out[name] = context.updateStripView(status, "RC1", {json.dumps(cancelling)});
console.log(JSON.stringify(out));
"""
    return _node(script)


RUNNING = {
    "running": True, "phase": "updating", "page": 3, "pages": 41, "refreshed": 1204, "added": 4, "skipped": 12,
    "failed": 0, "circles": 0, "watermarkBefore": "2026-10-20T02:41:07.046666Z",
}


@needs_node
def test_a_running_update_says_where_it_is_and_offers_cancel_but_not_dismiss() -> None:
    view = _views({"s": RUNNING})["s"]
    assert view["title"] == "Updating RC1"
    assert view["line"] == (
        "Checking changes since 2026-10-20 02:41:07Z · page 3 of 41 · "
        "1,204 refreshed, 4 added, 12 skipped, 0 failed"
    )
    assert (view["showBar"], view["indeterminate"], view["max"], view["value"]) == (True, False, 41, 3)
    assert (view["showCancel"], view["showDismiss"], view["tone"]) == (True, False, "running")


@needs_node
def test_before_the_page_count_is_known_the_bar_is_indeterminate_and_says_what_it_is_doing() -> None:
    views = _views({
        "catalog": {"running": True, "phase": "reading catalog"},
        "retry": {"running": True, "phase": "retrying skipped", "skipped": 3},
    })
    assert views["catalog"]["indeterminate"] is True and views["catalog"]["line"].startswith("Reading the catalog")
    assert views["retry"]["line"].startswith("Retrying locations skipped last time")
    assert "page" not in views["catalog"]["line"]


@needs_node
def test_cancelling_hides_the_button_and_says_the_run_is_finishing() -> None:
    view = _views({"s": RUNNING}, cancelling=True)["s"]
    assert view["showCancel"] is False and view["line"].startswith("Cancelling")


@needs_node
def test_a_finished_run_shows_its_result_with_a_tone_and_can_be_dismissed() -> None:
    views = _views({
        "done": {**RUNNING, "running": False, "phase": "done", "page": 41},
        "cancelled": {**RUNNING, "running": False, "phase": "cancelled"},
        "failed": {**RUNNING, "running": False, "phase": "failed", "failed": 2, "errors": ["7: boom"]},
    })
    assert views["done"]["tone"] == "ok" and views["done"]["line"].startswith("Finished · page 41 of 41")
    assert views["cancelled"]["tone"] == "warn" and views["cancelled"]["line"].startswith("Cancelled")
    assert views["failed"]["tone"] == "bad" and views["failed"]["line"].startswith("Finished with failures")
    assert all(view["showDismiss"] and not view["showCancel"] for view in views.values())
    assert views["failed"]["detail"] == "7: boom\n... 12 skipped in all; the full list is in update_all.log"


@needs_node
def test_details_list_notes_errors_and_a_capped_skip_list_with_the_total() -> None:
    skip_lines = [f"skip {i} z=1: no mask geometry" for i in range(1, 31)]
    view = _views({"s": {**RUNNING, "running": False, "phase": "done", "skipped": 45, "notes": ["note"],
                         "skipLines": skip_lines}})["s"]
    lines = view["detail"].split("\n")
    assert lines[0] == "note" and lines[1] == skip_lines[0]
    assert lines[-1] == "... 45 skipped in all; the full list is in update_all.log"
    assert len(lines) == 1 + 20 + 1, "only the first 20 skip lines are listed"


# ------------------------------------------------------------------------------------ what the page does with it

def _functions(*names: str) -> str:
    source = APP_JS.read_text(encoding="utf-8")
    parts = [_view_source()]
    for name in names:
        match = re.search(rf"  (?:async )?function {name}\(.*?\n  \}}\n", source, re.S)
        assert match, f"{name} not found in app.js"
        parts.append(match.group(0))
    return "\n".join(parts)


def _page_script(body: str) -> str:
    code = _functions("describeWatermark", "showUpdateStrip", "paintUpdate", "readUpdateStatus", "pollUpdate",
                      "watchUpdate", "openUpdateAll", "startUpdateAll")
    return f"""
const vm = require("vm");
const element = () => ({{ hidden: false, textContent: "", title: "", dataset: {{}}, disabled: false,
  value: 0, max: 1, attrs: {{}}, removeAttribute(n) {{ delete this.attrs[n]; this.valueRemoved = true; }},
  setAttribute(n, v) {{ this.attrs[n] = v; }} }});
const updateUi = {{ timer: 0, wasRunning: false, cancelling: false }};
for (const name of ["dialog", "since", "last", "start", "close", "strip", "title", "bar", "line", "detail",
                    "detailToggle", "cancel", "dismiss"]) updateUi[name] = element();
updateUi.dialog.hidden = true; updateUi.strip.hidden = true;
const state = {{ volume: "RC1" }};
const els = {{ status: {{ textContent: "" }} }};
let statusAnswer = null;
const calls = [];
const renders = [];
const timers = [];
const context = {{
  updateUi, state, els,
  updateStripShown: false,
  withSet: (url) => url,
  updateUrl: (suffix = "") => "/api/volumes/" + state.volume + "/update-all" + suffix,
  getJson: async (url, options) => {{ calls.push({{ url, method: options && options.method || "GET" }}); return statusAnswer; }},
  render: (flag) => renders.push(flag),
  window: {{ setTimeout: (fn, ms) => {{ timers.push(ms); return timers.length; }}, clearTimeout: () => {{}} }},
  locationGenerations: new Map(), locationGen: () => 0, resetVisibleCards: () => {{}}, loadCatalog: async () => {{}},
  layerEpoch: 0,
}};
vm.createContext(context);
vm.runInContext({json.dumps(_functions_prefix())} + {json.dumps(code)} + `
this.api = {{ showUpdateStrip, paintUpdate, pollUpdate, watchUpdate, openUpdateAll, startUpdateAll }};`, context);
const api = context.api;
(async () => {{
{body}
}})().catch((e) => {{ console.error(e); process.exit(1); }});
"""


def _functions_prefix() -> str:
    # updateStripShown is a `let` in the page; the harness keeps it as a context global the functions read and write.
    return "var updateStripShown = false;\n"


@needs_node
def test_clicking_update_all_while_a_run_is_going_opens_no_dialog_and_shows_the_strip() -> None:
    result = _node(_page_script(f"""
statusAnswer = {json.dumps(RUNNING)};
await api.openUpdateAll();
console.log(JSON.stringify({{ dialogHidden: updateUi.dialog.hidden, stripHidden: updateUi.strip.hidden,
  polling: timers.length > 0, cancelShown: !updateUi.cancel.hidden, renders }}));
"""))
    assert result["dialogHidden"] is True, "a running update must not put a window over the cards"
    assert result["stripHidden"] is False and result["polling"] and result["cancelShown"]
    assert result["renders"] == [True], "the cards are laid out once, when the strip appears"


@needs_node
def test_clicking_update_all_when_idle_asks_in_the_dialog_without_a_strip() -> None:
    result = _node(_page_script("""
statusAnswer = { running: false, phase: "idle", watermark: "2026-10-20T02:41:07.046666Z" };
await api.openUpdateAll();
console.log(JSON.stringify({ dialogHidden: updateUi.dialog.hidden, stripHidden: updateUi.strip.hidden,
  since: updateUi.since.textContent, renders }));
"""))
    assert result["dialogHidden"] is False and result["stripHidden"] is True
    assert result["since"] == "Changes since 2026-10-20 02:41:07Z will be checked."
    assert result["renders"] == []


@needs_node
def test_start_closes_the_dialog_and_hands_over_to_the_strip() -> None:
    result = _node(_page_script(f"""
updateUi.dialog.hidden = false;
statusAnswer = {json.dumps({**RUNNING, "page": 0, "pages": 0})};
await api.startUpdateAll();
console.log(JSON.stringify({{ dialogHidden: updateUi.dialog.hidden, stripHidden: updateUi.strip.hidden,
  indeterminate: updateUi.bar.valueRemoved === true, post: calls.filter((c) => c.method === "POST").length,
  polling: timers.length > 0 }}));
"""))
    assert result["dialogHidden"] is True and result["stripHidden"] is False
    assert result["indeterminate"] is True and result["post"] == 1 and result["polling"]


@needs_node
def test_opening_a_volume_picks_up_a_job_that_is_already_running_and_ignores_an_idle_one() -> None:
    result = _node(_page_script(f"""
statusAnswer = {json.dumps(RUNNING)};
await api.watchUpdate();
const running = {{ stripHidden: updateUi.strip.hidden, polling: timers.length > 0 }};
statusAnswer = {{ running: false, phase: "done", refreshed: 5 }};
await api.watchUpdate();
console.log(JSON.stringify({{ running, afterFinished: {{ stripHidden: updateUi.strip.hidden }}, renders }}));
"""))
    assert result["running"] == {"stripHidden": False, "polling": True}
    assert result["afterFinished"]["stripHidden"] is True, "an old result is not shown when a volume is opened"
    assert result["renders"] == [True, True], "laid out when the strip appeared and again when it went"


@needs_node
def test_polling_does_not_relayout_the_cards_on_every_tick() -> None:
    result = _node(_page_script(f"""
statusAnswer = {json.dumps(RUNNING)};
await api.pollUpdate(); await api.pollUpdate(); await api.pollUpdate();
console.log(JSON.stringify({{ renders }}));
"""))
    assert result["renders"] == [True]
