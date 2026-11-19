"""static/hv_layout.js: the converter's HV process / branch layout, run in node without a DOM."""

from __future__ import annotations

import json
import random
import shutil
import subprocess
from collections import deque
from pathlib import Path

import pytest

HV_JS = Path(__file__).resolve().parents[1] / "static" / "hv_layout.js"
NODE = shutil.which("node")

needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")

STEP = {"n": (-1, 0), "e": (0, 1), "s": (1, 0), "w": (0, -1)}
OPPOSITE = {"n": "s", "s": "n", "e": "w", "w": "e"}


def _layout(cards, links, **extra):
    """cards: {id: (z, area)}; links: [(a, b)]."""
    payload = {
        "cards": [{"id": i, "z": z, "area": a} for i, (z, a) in cards.items()],
        "links": links,
        **extra,
    }
    script = (
        f"const L = require({json.dumps(str(HV_JS))});"
        f"console.log(JSON.stringify(L.layoutHv({json.dumps(payload)})));"
    )
    result = subprocess.run([NODE, "-e", script], capture_output=True, text=True, encoding="utf-8", timeout=120, check=False)
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout)
    out["at"] = {c["id"]: (c["row"], c["col"]) for c in out["cells"]}
    return out


def _chain(prefix, count, z0, step=1):
    ids = [f"{prefix}{i}" for i in range(count)]
    return {i: (z0 + step * k, 10) for k, i in enumerate(ids)}, [(ids[k], ids[k + 1]) for k in range(count - 1)], ids


def _random_tree(seed, size):
    rng = random.Random(seed)
    cards = {"n0": (100, 50)}
    links = []
    for i in range(1, size):
        parent = f"n{rng.randrange(i)}" if rng.random() < 0.3 else f"n{i - 1}"
        cards[f"n{i}"] = (cards[parent][0] + rng.choice((-1, 1)), rng.uniform(1, 100))
        links.append((parent, f"n{i}"))
    return cards, links


def _check_drawing(out, links):
    """No shared cells, every pipe arm meets a matching arm or a card, every link drawn whole."""
    cards = {(c["row"], c["col"]): c["id"] for c in out["cells"]}
    assert len(cards) == len(out["cells"]), "two cards share a cell"
    pipes = {(p["row"], p["col"]): p for p in out["pipes"]}
    assert len(pipes) == len(out["pipes"])
    assert not set(cards) & set(pipes), "a connector sits on a card"
    for (r, c), pipe in pipes.items():
        for side, (dr, dc) in STEP.items():
            if pipe[side]:
                there = (r + dr, c + dc)
                assert there in cards or (there in pipes and pipes[there][OPPOSITE[side]]), ((r, c), side)
    at = out["at"]
    for a, b in links:
        start, goal = at[a], at[b]
        if abs(start[0] - goal[0]) + abs(start[1] - goal[1]) == 1:
            continue
        # Walk the connectors out of `a`: a pipe whose arm points at the card, then along matching arms.
        seen = set()
        queue = deque()
        for side, (dr, dc) in STEP.items():
            cell = (start[0] + dr, start[1] + dc)
            if cell in pipes and pipes[cell][OPPOSITE[side]]:
                queue.append(cell)
        reached = False
        while queue and not reached:
            cell = queue.popleft()
            if cell in seen:
                continue
            seen.add(cell)
            for side, (dr, dc) in STEP.items():
                if not pipes[cell][side]:
                    continue
                nxt = (cell[0] + dr, cell[1] + dc)
                if nxt == goal:
                    reached = True
                elif nxt in pipes:
                    queue.append(nxt)
        assert reached, f"link {a}-{b} is neither touching nor joined by connectors"


@needs_node
@pytest.mark.parametrize("seed", range(8))
def test_random_trees_are_drawn_whole_without_overlaps(seed) -> None:
    cards, links = _random_tree(seed, 120)
    out = _layout(cards, links, maxColumns=6)
    assert set(out["at"]) == set(cards)
    _check_drawing(out, links)


@needs_node
def test_processes_keep_their_z_side_of_the_root() -> None:
    cards = {"R": (100, 1000)}
    links = []
    for name, length, step in (("UP", 5, -1), ("DN", 7, 1), ("U2", 3, -1), ("D2", 4, 1), ("D3", 2, 1)):
        c, l, ids = _chain(name, length, 100 + step, step)
        cards.update(c)
        links += l + [("R", ids[0])]
    out = _layout(cards, links)
    at = out["at"]
    root_row = at["R"][0]
    for card, (row, _col) in at.items():
        if card.startswith("U"):
            assert row < root_row, card
        elif card.startswith("D"):
            assert row > root_row, card
    _check_drawing(out, links)


@needs_node
def test_the_largest_process_on_each_side_hangs_straight_off_the_root() -> None:
    cards = {"R": (100, 1000)}
    links = []
    for name, length in (("A", 6), ("B", 3), ("C", 2)):
        c, l, ids = _chain(name, length, 101)
        cards.update(c)
        links += l + [("R", ids[0])]
    out = _layout(cards, links, aspect=0.2)  # tall screen: nothing winds
    at = out["at"]
    r, c = at["R"]
    assert at["A0"] == (r + 1, c)
    # The others hang below the bus on the root's row, so they touch a connector, not the root.
    assert at["B0"][0] == r + 1 and at["B0"][1] > c
    assert {(p["row"], p["col"]) for p in out["pipes"] if p["row"] == r} >= {(r, k) for k in range(c + 1, at["C0"][1] + 1)}
    _check_drawing(out, links)


@needs_node
def test_a_long_process_winds_in_layers_two_columns_apart() -> None:
    cards = {"R": (100, 1000)}
    c, l, ids = _chain("A", 80, 101)
    cards.update(c)
    links = l + [("R", ids[0])]
    for name in ("B", "C"):
        c2, l2, ids2 = _chain(name, 2, 101)
        cards.update(c2)
        links += l2 + [("R", ids2[0])]
    out = _layout(cards, links, aspect=16 / 9)
    assert out["windHeight"] != float("inf") and out["windHeight"] is not None
    at = out["at"]
    for a, b in zip(ids, ids[1:]):
        (ra, ca), (rb, cb) = at[a], at[b]
        assert abs(ra - rb) + abs(ca - cb) == 1
    columns = sorted({at[i][1] for i in ids})
    layers = [col for col in columns if sum(1 for i in ids if at[i][1] == col) > 1]
    assert all(b - a == 2 for a, b in zip(layers, layers[1:]))
    _check_drawing(out, links)


@needs_node
def test_the_wind_height_follows_the_screen_shape() -> None:
    cards, links = _random_tree(3, 300)
    wide = _layout(cards, links, aspect=4.0)
    tall = _layout(cards, links, aspect=0.25)
    assert wide["columns"] / wide["totalRows"] > tall["columns"] / tall["totalRows"]
