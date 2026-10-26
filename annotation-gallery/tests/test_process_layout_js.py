"""static/process_layout.js: the converter's process / branch layout, run in node without a DOM."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

LAYOUT_JS = Path(__file__).resolve().parents[1] / "static" / "process_layout.js"
NODE = shutil.which("node")

needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")


def _layout(cards, links, **extra):
    """cards: {id: (z, area)} or {id: dict}; links: [(a, b)]. Returns {id: (row, col)} plus the whole result."""
    payload = {
        "cards": [{"id": i, **(c if isinstance(c, dict) else {"z": c[0], "area": c[1]})} for i, c in cards.items()],
        "links": links,
        **extra,
    }
    script = (
        f"const L = require({json.dumps(str(LAYOUT_JS))});"
        f"console.log(JSON.stringify(L.layoutProcesses({json.dumps(payload)})));"
    )
    result = subprocess.run([NODE, "-e", script], capture_output=True, text=True, encoding="utf-8", timeout=60, check=False)
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout)
    out["at"] = {c["id"]: (c["row"], c["col"]) for c in out["cells"]}
    return out


def _chain(prefix, count, z0, step=1):
    ids = [f"{prefix}{i}" for i in range(count)]
    return {i: (z0 + step * k, 10) for k, i in enumerate(ids)}, [(ids[k], ids[k + 1]) for k in range(count - 1)], ids


def _star(root_z=100, lengths=(4, 3, 2), below=True):
    """A root R (largest) with one process per length, each linked to R at its first card."""
    cards = {"R": (root_z, 1000)}
    links = []
    for n, length in enumerate(lengths):
        z0 = root_z + 1 if below else root_z - length
        c, l, ids = _chain(f"P{n}_", length, z0)
        cards.update(c)
        links += l
        links.append(("R", ids[0] if below else ids[-1]))
    return cards, links


@needs_node
def test_the_root_ends_its_row_and_its_processes_are_columns_longest_leftmost() -> None:
    cards, links = _star(below=True)
    out = _layout(cards, links)["at"]
    # Columns left to right: 4, 3, 2 long; root last, on row 0, nothing to its right on that row.
    assert out["R"] == (0, 3)
    assert [out[f"P0_{k}"] for k in range(4)] == [(k, 0) for k in range(4)]
    assert [out[f"P1_{k}"] for k in range(3)] == [(k, 1) for k in range(3)]
    assert [out[f"P2_{k}"] for k in range(2)] == [(k, 2) for k in range(2)]


@needs_node
def test_processes_above_the_root_in_z_run_up_with_the_root_row_below_them() -> None:
    cards, links = _star(below=False)
    out = _layout(cards, links)["at"]
    rows = [r for r, _ in out.values()]
    assert min(rows) == 0 and out["R"][0] == 3  # the longest process takes 4 rows, ending on the root's row
    # The card touching the root is on the root's row in every column.
    assert out["P0_3"] == (3, 0) and out["P1_2"] == (3, 1) and out["P2_1"] == (3, 2)
    assert out["P0_0"] == (0, 0)


@needs_node
def test_mixed_directions_put_the_root_row_in_the_middle() -> None:
    cards = {"R": (100, 1000)}
    links = []
    c, l, up = _chain("U", 2, 97)
    cards.update(c); links += l + [("R", "U1")]
    c, l, down = _chain("D", 3, 101)
    cards.update(c); links += l + [("R", "D0")]
    c, l, ids = _chain("E", 2, 102)
    cards.update(c); links += l + [("R", "E0")]
    out = _layout(cards, links)["at"]
    assert out["R"][0] == 1
    # Down columns (3 and 2 long) come before the 2-long up column; every string touches the root's row.
    assert out["D0"] == (1, 0) and out["D2"] == (3, 0)
    assert out["E0"] == (1, 1) and out["E1"] == (2, 1)
    assert out["U1"] == (1, 2) and out["U0"] == (0, 2)


@needs_node
def test_a_branch_ends_a_process_column_and_its_other_processes_start_right_of_it() -> None:
    # R -- A0 -- A1 -- B (branch, smaller than R) with two more processes C and D below B.
    cards = {"R": (10, 1000), "A0": (11, 5), "A1": (12, 5), "B": (13, 500)}
    links = [("R", "A0"), ("A0", "A1"), ("A1", "B")]
    for name in ("S", "T"):
        c, l, ids = _chain(name, 1, 11)
        cards.update(c); links += l + [("R", ids[0])]
    for name in ("C", "D"):
        c, l, ids = _chain(name, 2, 14)
        cards.update(c); links += l + [("B", ids[0])]
    out = _layout(cards, links)["at"]
    assert out["R"] == (0, 3)
    assert out["A0"] == (0, 0) and out["A1"] == (1, 0) and out["B"] == (2, 0)
    # B's processes run below B, at B's column or to its right, never on B's row and never left of it.
    for card in ("C0", "C1", "D0", "D1"):
        assert out[card][0] > out["B"][0] and out[card][1] >= out["B"][1]
    assert len(set(out.values())) == len(out)  # no two cards share a cell


@needs_node
def test_nothing_sits_right_of_a_branch_on_its_row() -> None:
    cards = {"R": (10, 1000), "A0": (11, 5), "A1": (12, 5), "B": (13, 500)}
    links = [("R", "A0"), ("A0", "A1"), ("A1", "B")]
    c, l, ids = _chain("C", 3, 14)
    cards.update(c); links += l + [("B", ids[0])]
    for name in ("S", "T"):
        c, l, ids = _chain(name, 2, 11)
        cards.update(c); links += l + [("R", ids[0])]
    out = _layout(cards, links)
    assert out["at"]["R"][1] > out["at"]["B"][1]
    for branch in ("R", "B"):
        row, col = out["at"][branch]
        assert all(r != row or c <= col for r, c in out["at"].values()), branch


@needs_node
def test_a_ring_is_laid_out_once() -> None:
    # R has three processes; two of them end in the same branch B (a cycle through R and B).
    cards = {"R": (10, 1000), "B": (20, 500), "X": (11, 5), "Y": (12, 5), "Z": (13, 5), "T": (14, 5), "U": (15, 5)}
    links = [("R", "X"), ("X", "B"), ("R", "Y"), ("Y", "B"), ("R", "Z"), ("B", "T"), ("B", "U")]
    out = _layout(cards, links)
    assert sorted(out["at"]) == sorted(cards)
    assert len(out["cells"]) == len(cards) and len(set(out["at"].values())) == len(cards)


@needs_node
def test_components_stack_then_strings_then_unlinked_cards_by_z() -> None:
    cards, links = _star(lengths=(2, 2, 2))
    cards["S0"], cards["S1"] = (5, 1), (6, 1)
    links.append(("S0", "S1"))
    cards["U1"], cards["U2"] = (9, 1), (3, 1)
    out = _layout(cards, links, maxColumns=4)
    at = out["at"]
    tree_rows = max(r for k, (r, _) in at.items() if k.startswith(("R", "P")))
    assert at["S0"][0] > tree_rows and at["S1"] == (at["S0"][0] + 1, at["S0"][1])
    assert at["U2"][0] > at["S1"][0] and at["U1"] == (at["U2"][0], at["U2"][1] + 1)


@needs_node
def test_done_processes_and_dangling_done_branches_are_removed() -> None:
    cards, links = _star(lengths=(3, 3, 3))
    for k in range(3):
        cards[f"P1_{k}"] = {"z": 101 + k, "area": 10, "done": True}
    # A leaf branch B hanging off P2 whose own card is done and has no other process.
    cards["B"] = {"z": 150, "area": 500, "done": True}
    links.append(("P2_2", "B"))
    cards["B2"], cards["B3"] = (151, 1), (152, 1)
    links += [("B", "B2")]
    out = _layout(cards, links)["at"]
    assert not any(k.startswith("P1_") for k in out)
    assert "R" in out and "P0_0" in out and "P2_0" in out


@needs_node
def test_a_done_branch_with_other_processes_is_kept_and_an_unfinished_leaf_branch_is_kept() -> None:
    cards = {"R": (10, 1000), "A": (11, 5), "B": {"z": 12, "area": 500, "done": False}, "C": (13, 5)}
    links = [("R", "A"), ("A", "B"), ("B", "C")]
    # Three links on R make it a branch; B has only two so it is just a card of the string. Nothing is removed.
    cards["R2"], cards["R3"] = (11, 5), (11, 5)
    links += [("R", "R2"), ("R", "R3")]
    out = _layout(cards, links)["at"]
    assert sorted(out) == sorted(cards)


@needs_node
def test_a_card_that_spans_columns_widens_its_whole_column() -> None:
    cards, links = _star(lengths=(3, 2, 2))
    cards["P0_1"] = {"z": 102, "area": 10, "w": 2}
    out = _layout(cards, links)
    assert out["at"]["P1_0"][1] == 2  # the second column starts after the two-wide first one
    assert out["at"]["R"][1] == 4


@needs_node
def test_omitted_types_inside_a_string_are_skipped_not_cut() -> None:
    # R has three strings; the middle card of one (X) is an omitted type: it has no card but is in the links.
    cards = {"R": (10, 1000), "A0": (11, 5), "A2": (13, 5), "B0": (11, 5), "B1": (12, 5), "C0": (11, 5)}
    links = [("R", "A0"), ("A0", "X"), ("X", "A2"), ("R", "B0"), ("B0", "B1"), ("R", "C0")]
    out = _layout(cards, links)
    assert sorted(out["at"]) == sorted(cards)
    # A0 and A2 are one process (a column of two) joined through the skipped X: A2 is directly below A0.
    assert out["at"]["A2"] == (out["at"]["A0"][0] + 1, out["at"]["A0"][1])


@needs_node
def test_a_string_of_only_omitted_types_between_branches_leaves_them_linked() -> None:
    cards = {"R": (10, 1000), "B": (20, 500), "S": (11, 5), "T": (12, 5), "U": (21, 5), "V": (22, 5)}
    links = [("R", "S"), ("R", "T"), ("R", "X1"), ("X1", "X2"), ("X2", "B"), ("B", "U"), ("B", "V")]
    out = _layout(cards, links)
    assert sorted(out["at"]) == sorted(cards)
    # R and B are joined directly (no cards between them), so B sits in a column of its own and U, V hang from it.
    assert out["at"]["B"][1] >= 0 and out["at"]["U"][1] >= out["at"]["B"][1] and out["at"]["V"][1] >= out["at"]["B"][1]
