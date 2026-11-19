/*
 * The converter's HV process / branch layout ("Process / branch (HV)"): the cards of one structure placed as an
 * orthogonal tree drawing whose subtrees occupy boxes that never overlap. A pure function so the node tests can load
 * this file on its own. Links, processes, branches and the pruning of done cards come from static/process_layout.js
 * (`analyseLinks`), so both layouts read the links the same way.
 *
 * Words as in process_layout.js: a *process* is a string of linked locations with one or two links, a *branch* a
 * location with three or more.
 *   - Z is vertical: a process runs up from its branch when its cards' mean Z is below the branch's, down otherwise.
 *   - At a branch, the largest process on each free side (not the side the branch was reached from) hangs straight
 *     off it. The others hang from a *bus*, a row of connector cells running right from the branch along its row:
 *     up-processes above it, down-processes below, each side packed left to right, largest first.
 *   - A process with its far branch and everything beyond is one box; boxes sit side by side, so nothing collides.
 *     When a box reaches back past its first card (a far branch's processes running back toward this one), it hangs
 *     lower (or higher) on a vertical connector so it clears the bus.
 *   - A process longer than the wind height H winds: it runs H cards, steps two columns sideways (one turn card in
 *     the empty column between), runs back H, and so on (Chan, Goodrich, Kosaraju & Tamassia's "winding", Comput.
 *     Geom. 23, 2002). A process that ends in a branch winds an odd number of layers and runs on straight to it.
 *   - H is chosen from WIND_HEIGHTS so the whole drawing's shape is closest to the screen's.
 * Where linked cards cannot touch (a grid cell has four sides; deciding whether a tree fits with every link touching
 * is NP-complete, Bhatt & Cosmadakis 1987), the result lists connector cells (`pipes`) so every link is drawn whole.
 * Trees stack top to bottom (most cards first), then unbranched strings side by side, then unlinked cards in Z order.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory(require("./process_layout.js"));
  } else {
    root.HvLayout = factory(root.ProcessLayout);
  }
})(typeof self !== "undefined" ? self : this, function (ProcessLayout) {
  "use strict";

  const WIND_HEIGHTS = [4, 6, 8, 12, 16, 24, Infinity];

  const byId = (a, b) => (a < b ? -1 : a > b ? 1 : 0);

  /** A drawing in progress: cards, connector cells keyed "row,col", and its bounds. */
  function newShape() {
    return { cells: [], pipes: new Map(), top: Infinity, bottom: -Infinity, right: -Infinity };
  }

  function grow(shape, row, col) {
    shape.top = Math.min(shape.top, row);
    shape.bottom = Math.max(shape.bottom, row);
    shape.right = Math.max(shape.right, col);
  }

  function addCell(shape, cell) {
    shape.cells.push(cell);
    grow(shape, cell.row, cell.col);
  }

  /** Adds a connector cell, or more arms to one already there. `arms` holds n/e/s/w flags. */
  function addPipe(shape, row, col, arms, group) {
    const key = `${row},${col}`;
    const pipe = shape.pipes.get(key) || { row, col, n: false, e: false, s: false, w: false, group };
    for (const side of ["n", "e", "s", "w"]) {
      pipe[side] = pipe[side] || Boolean(arms[side]);
    }
    if (pipe.group < 0 && group >= 0) {
      pipe.group = group;
    }
    shape.pipes.set(key, pipe);
    grow(shape, row, col);
  }

  /** Copies `from` into `into`, shifted by (dr, dc). */
  function merge(into, from, dr, dc) {
    for (const cell of from.cells) {
      addCell(into, { ...cell, row: cell.row + dr, col: cell.col + dc });
    }
    for (const pipe of from.pipes.values()) {
      addPipe(into, pipe.row + dr, pipe.col + dc, pipe, pipe.group);
    }
  }

  /** Vertical connector cells from row `from` to row `to` (inclusive) in column `col`. */
  function verticalPipes(shape, from, to, col, group) {
    for (let row = Math.min(from, to); row <= Math.max(from, to); row++) {
      addPipe(shape, row, col, { n: true, s: true }, group);
    }
  }

  /** Where the i-th card of a winding process goes: layers of `height` cards two columns apart, one turn card between. */
  function windSpot(i, height) {
    const layer = Math.floor(i / (height + 1));
    const along = i % (height + 1);
    if (along === height) {
      return { row: layer % 2 === 0 ? height - 1 : 0, col: 2 * layer + 1 };
    }
    return { row: layer % 2 === 0 ? along : height - 1 - along, col: 2 * layer };
  }

  /** Positions of a process's `n` cards from its first card, running in `dir` (+1 down, -1 up). */
  function processSpots(n, height, endsInBranch, dir) {
    const spots = [];
    let wound = 0;
    if (Number.isFinite(height) && n > height) {
      if (!endsInBranch) {
        wound = n;
      } else {
        // An odd number of layers ends on the outer row, from where the rest runs straight on to the branch.
        let layers = Math.floor((n + 1) / (height + 1));
        if (layers % 2 === 0) {
          layers -= 1;
        }
        wound = layers >= 3 ? layers * height + (layers - 1) : 0;
      }
    }
    for (let i = 0; i < n; i++) {
      if (i < wound) {
        const spot = windSpot(i, height);
        spots.push({ row: dir * spot.row, col: spot.col });
      } else if (wound > 0) {
        const last = windSpot(wound - 1, height);
        spots.push({ row: dir * (last.row + 1 + i - wound), col: last.col });
      } else {
        spots.push({ row: dir * i, col: 0 });
      }
    }
    return spots;
  }

  /** One drawing of every block for wind height `height`. */
  function drawAll(analysis, height, maxColumns) {
    const { cardById, trees, strings, leftovers } = analysis;
    const z = (id) => cardById.get(id).z;
    const area = (id) => cardById.get(id).area;
    const meanZ = (ids) => ids.reduce((sum, id) => sum + z(id), 0) / ids.length;

    const blocks = [];

    for (const tree of trees) {
      const taken = new Set();
      const placed = new Set();

      // The arms of a branch not drawn yet: each a process (cards ordered away from the branch) and the branch it
      // ends in, if that branch is not drawn yet. Claiming marks them, so a ring is drawn once.
      const claim = (branchId) => {
        const arms = [];
        for (const p of tree.processes) {
          if (taken.has(p.index) || !p.ends.includes(branchId)) {
            continue;
          }
          taken.add(p.index);
          const nodes = p.ends[0] === branchId ? p.nodes.slice() : p.nodes.slice().reverse();
          const far = p.ends[0] === branchId && p.ends[1] !== branchId ? p.ends[1] : p.ends[0] !== branchId ? p.ends[0] : null;
          const end = far !== null && !placed.has(far) ? far : null;
          if (end !== null) {
            placed.add(end);
          }
          arms.push({ group: p.index, nodes, end });
        }
        for (const [a, b] of tree.bridges) {
          const other = a === branchId ? b : b === branchId ? a : null;
          if (other !== null && !placed.has(other)) {
            placed.add(other);
            arms.push({ group: -1, nodes: [], end: other });
          }
        }
        return arms;
      };

      // An arm drawn from its first card at (0, 0), running in `dir`.
      const drawArm = (arm, dir) => {
        if (!arm.nodes.length) {
          return drawBranch(arm.end, -dir);
        }
        const shape = newShape();
        const spots = processSpots(arm.nodes.length, height, arm.end !== null, dir);
        arm.nodes.forEach((id, i) => {
          addCell(shape, { id, row: spots[i].row, col: spots[i].col, w: 1, h: 1, group: arm.group, kind: "process" });
        });
        if (arm.end !== null) {
          const last = spots[spots.length - 1];
          merge(shape, drawBranch(arm.end, -dir), last.row + dir, last.col);
        }
        return shape;
      };

      // A branch at (0, 0), reached from `inSide` (-1 from above, +1 from below, 0 for the root).
      const drawBranch = (branchId, inSide) => {
        const shape = newShape();
        addCell(shape, { id: branchId, row: 0, col: 0, w: 1, h: 1, group: -1, kind: "branch" });
        const arms = claim(branchId).map((arm) => {
          const ids = arm.nodes.length ? arm.nodes : [arm.end];
          const dir = meanZ(ids) < z(branchId) ? -1 : 1;
          return { ...arm, dir };
        });
        const junctions = [];
        for (const dir of [-1, 1]) {
          const drawn = arms
            .filter((arm) => arm.dir === dir)
            .map((arm) => ({ arm, shape: drawArm(arm, dir) }))
            .sort((a, b) => b.shape.cells.length - a.shape.cells.length || byId(a.shape.cells[0].id, b.shape.cells[0].id));
          let cursor = 1;
          drawn.forEach(({ arm, shape: armShape }, index) => {
            // How far the arm reaches back past its first card toward this branch's row.
            const reach = Math.max(0, dir > 0 ? -armShape.top : armShape.bottom);
            const straight = index === 0 && inSide !== dir;
            const col = straight ? 0 : cursor;
            if (reach > 0) {
              verticalPipes(shape, dir, dir * reach, col, arm.group);
            }
            merge(shape, armShape, dir * (1 + reach), col);
            if (!straight) {
              junctions.push({ col, dir, group: arm.group });
            }
            cursor = col + armShape.right + 1;
          });
        }
        if (junctions.length) {
          const last = Math.max(...junctions.map((j) => j.col));
          for (let col = 1; col <= last; col++) {
            const here = junctions.filter((j) => j.col === col);
            const next = junctions.filter((j) => j.col >= col).sort((a, b) => a.col - b.col)[0];
            addPipe(shape, 0, col, {
              w: true,
              e: col < last,
              n: here.some((j) => j.dir < 0),
              s: here.some((j) => j.dir > 0),
            }, next.group);
          }
        }
        return shape;
      };

      const rootId = [...tree.branches].sort((a, b) => area(b) - area(a) || byId(a, b))[0];
      placed.add(rootId);
      const shape = drawBranch(rootId, 0);
      shape.cells.find((cell) => cell.id === rootId).root = true;
      blocks.push(shape);
    }

    // Unbranched strings side by side, each running down from its lower-Z end.
    if (strings.length) {
      const shape = newShape();
      let cursor = 0;
      for (const component of strings) {
        const p = component.processes[0];
        const nodes = z(p.nodes[0]) > z(p.nodes[p.nodes.length - 1]) ? p.nodes.slice().reverse() : p.nodes.slice();
        const spots = processSpots(nodes.length, height, false, 1);
        const one = newShape();
        nodes.forEach((id, i) => {
          addCell(one, { id, row: spots[i].row, col: spots[i].col, w: 1, h: 1, group: p.index, kind: "process" });
        });
        merge(shape, one, 0, cursor);
        cursor += one.right + 1;
      }
      blocks.push(shape);
    }

    // Unlinked cards in Z order, wrapping at maxColumns.
    if (leftovers.length) {
      const shape = newShape();
      leftovers.forEach((id, i) => {
        addCell(shape, {
          id, row: Math.floor(i / maxColumns), col: i % maxColumns, w: 1, h: 1, group: -1, kind: "single",
        });
      });
      blocks.push(shape);
    }

    const cells = [];
    const pipes = [];
    let bottom = 0;
    let columns = 1;
    for (const block of blocks) {
      const dr = bottom - block.top;
      for (const cell of block.cells) {
        cells.push({ ...cell, row: cell.row + dr });
      }
      for (const pipe of block.pipes.values()) {
        pipes.push({ ...pipe, row: pipe.row + dr });
      }
      bottom += block.bottom - block.top + 1;
      columns = Math.max(columns, block.right + 1);
    }
    cells.sort((a, b) => a.row - b.row || a.col - b.col);
    pipes.sort((a, b) => a.row - b.row || a.col - b.col);
    return { cells, pipes, columns, totalRows: bottom, windHeight: height };
  }

  /**
   * @param {{cards: {id: string|number, z: number, area?: number, done?: boolean}[],
   *          links: [string|number, string|number][], maxColumns?: number, aspect?: number}} input
   *   As for ProcessLayout.layoutProcesses; cards are one cell each. `aspect` is the screen's width over its height in
   *   grid cells (default 16/9).
   * @returns {{cells: {id: string, row: number, col: number, w: number, h: number, group: number, kind: string,
   *                    root?: boolean}[],
   *            pipes: {row: number, col: number, n: boolean, e: boolean, s: boolean, w: boolean, group: number}[],
   *            columns: number, totalRows: number, windHeight: number}}
   *   `pipes` are connector cells; `group` is the process a pipe leads to (-1 for a branch-to-branch link).
   */
  function layoutHv(input) {
    const maxColumns = Math.max(1, Number(input.maxColumns) || 1);
    const aspect = Number(input.aspect) > 0 ? Number(input.aspect) : 16 / 9;
    const analysis = ProcessLayout.analyseLinks(input);
    let best = null;
    for (const height of WIND_HEIGHTS) {
      const drawn = drawAll(analysis, height, maxColumns);
      const miss = Math.abs(Math.log(drawn.columns / Math.max(1, drawn.totalRows) / aspect));
      const size = drawn.columns * drawn.totalRows;
      if (!best || miss < best.miss - 1e-9 || (Math.abs(miss - best.miss) <= 1e-9 && size < best.size)) {
        best = { drawn, miss, size };
      }
    }
    return best.drawn;
  }

  return { layoutHv, windSpot };
});
