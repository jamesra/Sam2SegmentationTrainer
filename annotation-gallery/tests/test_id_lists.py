"""Id boxes follow the Viking / nornir id-list convention: commas separate ids, a hyphen between two ids is a range.

The JavaScript side cuts parseIdList and isIdListText out of app.js and runs them in node's vm (skipped when node is
not installed); the gold list upload is checked through gold.draw.parse_id_list.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

GALLERY = Path(__file__).resolve().parents[1]
APP_JS = GALLERY / "static" / "app.js"
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")

sys.path.insert(0, str(GALLERY))
from gold import draw  # noqa: E402


def _function(source: str, name: str) -> str:
    match = re.search(rf"  function {name}\(.*?\n  \}}\n", source, re.S)
    assert match, f"{name} not found in app.js"
    return match.group(0)


def _js(cases: dict[str, str], probe: list[int]) -> dict:
    source = APP_JS.read_text(encoding="utf-8")
    code = "".join(_function(source, name) for name in ("parseIdList", "isIdListText"))
    script = f"""
const vm = require("vm");
const context = {{}};
vm.createContext(context);
vm.runInContext({json.dumps(code)} + "\\nthis.parseIdList = parseIdList; this.isIdListText = isIdListText;", context);
const out = {{}};
for (const [name, text] of Object.entries({json.dumps(cases)})) {{
  const ids = context.parseIdList(text);
  out[name] = {{
    isList: context.isIdListText(text),
    parsed: ids === null ? null : {{
      empty: ids.empty, listed: [...ids.listed], ranges: ids.ranges,
      hits: {json.dumps(probe)}.filter((id) => ids.has(id)),
    }},
  }};
}}
console.log(JSON.stringify(out));
"""
    result = subprocess.run([NODE, "-e", script], capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


PROBE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 120, 487, 193664, 193665, 193680, 193697, 193698]


@needs_node
def test_commas_and_inclusive_ranges_like_nornir_integer_list() -> None:
    out = _js({"mixed": "1, 3, 5-8, 11", "spaced": "5 - 8", "range": "193665-193697"}, PROBE)
    assert out["mixed"]["parsed"]["hits"] == [1, 3, 5, 6, 7, 8, 11]
    assert out["mixed"]["parsed"]["listed"] == [1, 3, 11] and out["mixed"]["parsed"]["ranges"] == [[5, 8]]
    assert out["spaced"]["parsed"]["hits"] == [5, 6, 7, 8]
    assert out["range"]["parsed"]["hits"] == [193665, 193680, 193697]
    assert all(case["isList"] for case in out.values())


@needs_node
def test_a_pasted_column_and_other_separators_still_work() -> None:
    out = _js({"column": "487\n120\n", "semicolons": "487; 120", "trailing": "487, 120,"}, PROBE)
    for name in ("column", "semicolons", "trailing"):
        assert out[name]["parsed"]["hits"] == [120, 487], name
        assert out[name]["isList"], name


@needs_node
def test_junk_backwards_ranges_and_empty_text() -> None:
    out = _js({"junk": "abc", "backwards": "8-5", "negative": "0 -4", "empty": "  ", "regex": "^M", "label": "5-8a"}, PROBE)
    assert out["junk"]["parsed"]["empty"] and not out["junk"]["isList"]
    assert out["backwards"]["parsed"]["empty"]
    assert out["negative"]["parsed"]["listed"] == [0] and out["negative"]["parsed"]["ranges"] == []
    assert not out["negative"]["isList"]
    assert out["empty"]["parsed"] is None
    assert not out["regex"]["isList"] and not out["label"]["isList"]


def test_the_filter_reads_type_and_structure_boxes_as_id_lists() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    assert "const typeIds = isIdListText(type) ? parseIdList(type) : null;" in source
    assert "const structureIds = isIdListText(label) ? parseIdList(label) : null;" in source
    # A range is never expanded into one server lookup per id.
    assert "[...ids.listed].filter(" in source


def test_gold_lists_take_ranges() -> None:
    ids, bad = draw.parse_id_list("# list\n12, 5-8\n193665 - 193667; 6\n8-5 0-3\n")
    assert ids == [12, 5, 6, 7, 8, 193665, 193666, 193667]
    assert bad == ["8-5", "0-3"]


def test_gold_list_limit_counts_every_id_in_a_range() -> None:
    assert draw.parse_id_list("1-3", limit=3)[0] == [1, 2, 3]
    with pytest.raises(ValueError):
        draw.parse_id_list("1-4", limit=3)
    with pytest.raises(ValueError):
        draw.parse_id_list("1-1000000000", limit=100000)
