/*
 * The converter's process / branch layout: the cards of one structure placed by how their locations are linked.
 * A pure function so the node tests can load this file on its own.
 *
 * Words: a *process* is a string of linked locations that each have one or two links; a *branch* is a location with
 * three or more. Each tree of processes and branches is laid out from its largest branch (the root):
 *   - the root ends its row (nothing to its right on that row);
 *   - the processes attached to it are columns to its left, longest leftmost, the end that touches the root on the root's
 *     row, the rest running up when the process lies above the root in Z (smaller Z), down otherwise;
 *   - a process that ends in another branch puts that branch at the end of its column; the branch's remaining processes
 *     become columns at the branch's column or to its right (never left of it), above or below the branch's row, and
 *     nothing is placed right of the branch on its row;
 *   - a column is as wide as its widest card.
 * Trees stack top to bottom (most cards first), then unbranched strings as columns, then unlinked cards in Z order.
 * Locations that are approved or rejected drop out: a process whose cards are all done, and a done branch left with
 * no process other than the one that reaches it.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.ProcessLayout = factory();
  }
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  /**
   * @param {{cards: {id: string|number, z: number, area?: number, w?: number, h?: number, done?: boolean}[],
   *          links: [string|number, string|number][], maxColumns?: number}} input
   *   `w` and `h` are the card's size in grid cells (default 1); `done` is true once the card is approved or rejected;
   *   `maxColumns` is how wide unlinked cards may wrap.
   * @returns {{cells: {id: string, row: number, col: number, w: number, h: number, group: number, kind: string}[],
   *            columns: number, totalRows: number}}
   *   `group` is the process number (-1 for a branch or an unlinked card); `cells` are in reading order.
   */
  function layoutProcesses(input) {
    const maxColumns = Math.max(1, Number(input.maxColumns) || 1);
    const cardById = new Map();
    for (const card of input.cards || []) {
      cardById.set(String(card.id), {
        id: String(card.id),
        z: Number(card.z) || 0,
        area: Number(card.area) || 0,
        w: Math.max(1, Number(card.w) || 1),
        h: Math.max(1, Number(card.h) || 1),
        done: Boolean(card.done),
      });
    }

    // Links among the cards, each once. A location that has no card (an annotation type that is not converted) is
    // skipped over: the cards on either side of it are linked to each other, so a string of mixed types is still
    // one process of the types that are shown. An omitted junction links its neighbours in a chain, in id order.
    const everyone = new Map();
    const touch = (id) => {
      if (!everyone.has(id)) {
        everyone.set(id, new Set());
      }
      return everyone.get(id);
    };
    for (const id of cardById.keys()) {
      touch(id);
    }
    for (const [a, b] of input.links || []) {
      const x = String(a);
      const y = String(b);
      if (x !== y) {
        touch(x).add(y);
        touch(y).add(x);
      }
    }
    const omittedIds = [...everyone.keys()].filter((id) => !cardById.has(id)).sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
    for (const id of omittedIds) {
      const around = [...everyone.get(id)].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
      for (const other of around) {
        everyone.get(other).delete(id);
      }
      for (let i = 0; i + 1 < around.length; i++) {
        everyone.get(around[i]).add(around[i + 1]);
        everyone.get(around[i + 1]).add(around[i]);
      }
      everyone.delete(id);
    }
    const neighbours = everyone;

    const isBranch = (id) => neighbours.get(id).size >= 3;
    const byId = (a, b) => (a < b ? -1 : a > b ? 1 : 0);

    // Processes: maximal strings of non-branch cards, in order along the string.
    const processes = [];
    const seen = new Set();
    const ids = [...cardById.keys()].sort(byId);
    for (const start of ids) {
      if (isBranch(start) || seen.has(start)) {
        continue;
      }
      const members = new Set();
      const stack = [start];
      while (stack.length) {
        const id = stack.pop();
        if (members.has(id)) {
          continue;
        }
        members.add(id);
        for (const next of neighbours.get(id)) {
          if (!isBranch(next) && !members.has(next)) {
            stack.push(next);
          }
        }
      }
      for (const id of members) {
        seen.add(id);
      }
      const inside = (id) => [...neighbours.get(id)].filter((next) => members.has(next));
      // A string starts at a card with at most one neighbour inside it; a ring starts anywhere.
      const first = [...members].sort(byId).find((id) => inside(id).length <= 1) || [...members].sort(byId)[0];
      const nodes = [first];
      const used = new Set(nodes);
      for (;;) {
        const next = inside(nodes[nodes.length - 1]).filter((id) => !used.has(id)).sort(byId)[0];
        if (next === undefined) {
          break;
        }
        nodes.push(next);
        used.add(next);
      }
      const branchNeighbours = (id) => [...neighbours.get(id)].filter(isBranch).sort(byId);
      let endA = null;
      let endB = null;
      if (nodes.length === 1) {
        [endA = null, endB = null] = branchNeighbours(nodes[0]);
      } else {
        endA = branchNeighbours(nodes[0])[0] ?? null;
        endB = branchNeighbours(nodes[nodes.length - 1])[0] ?? null;
      }
      processes.push({ index: processes.length, nodes, ends: [endA, endB] });
    }

    // Branches linked straight to branches have no cards between them.
    const bridges = [];
    for (const id of ids) {
      if (!isBranch(id)) {
        continue;
      }
      for (const next of [...neighbours.get(id)].sort(byId)) {
        if (isBranch(next) && id < next) {
          bridges.push([id, next]);
        }
      }
    }

    // Prune what is done.
    const live = processes.filter((p) => !p.nodes.every((id) => cardById.get(id).done));
    const liveBridges = bridges.slice();
    const removedBranches = new Set();
    const attachments = (branch) =>
      live.filter((p) => p.ends.includes(branch)).length + liveBridges.filter((pair) => pair.includes(branch)).length;
    for (let changed = true; changed; ) {
      changed = false;
      for (const id of ids) {
        if (!isBranch(id) || removedBranches.has(id) || !cardById.get(id).done || attachments(id) > 1) {
          continue;
        }
        removedBranches.add(id);
        for (const p of live) {
          p.ends = p.ends.map((end) => (end === id ? null : end));
        }
        for (let i = liveBridges.length - 1; i >= 0; i--) {
          if (liveBridges[i].includes(id)) {
            liveBridges.splice(i, 1);
          }
        }
        changed = true;
      }
    }
    const branches = ids.filter((id) => isBranch(id) && !removedBranches.has(id));

    // Items (live processes, branches, bridges) joined into components.
    const parent = new Map();
    const find = (x) => {
      while (parent.get(x) !== x) {
        parent.set(x, parent.get(parent.get(x)));
        x = parent.get(x);
      }
      return x;
    };
    const union = (a, b) => parent.set(find(a), find(b));
    for (const id of branches) {
      parent.set(`b:${id}`, `b:${id}`);
    }
    for (const p of live) {
      parent.set(`p:${p.index}`, `p:${p.index}`);
    }
    for (const p of live) {
      for (const end of p.ends) {
        if (end !== null) {
          union(`p:${p.index}`, `b:${end}`);
        }
      }
    }
    for (const [a, b] of liveBridges) {
      union(`b:${a}`, `b:${b}`);
    }
    const components = new Map();
    const componentOf = (key) => {
      const rootKey = find(key);
      if (!components.has(rootKey)) {
        components.set(rootKey, { branches: [], processes: [], bridges: [] });
      }
      return components.get(rootKey);
    };
    for (const id of branches) {
      componentOf(`b:${id}`).branches.push(id);
    }
    for (const p of live) {
      componentOf(`p:${p.index}`).processes.push(p);
    }
    for (const pair of liveBridges) {
      componentOf(`b:${pair[0]}`).bridges.push(pair);
    }

    const cells = [];
    const sizeOf = (component) =>
      component.branches.length + component.processes.reduce((sum, p) => sum + p.nodes.length, 0);
    const ordered = [...components.values()].sort((a, b) => sizeOf(b) - sizeOf(a));
    const trees = ordered.filter((c) => c.branches.length > 0);
    const strings = ordered
      .filter((c) => c.branches.length === 0 && sizeOf(c) > 1)
      .sort((a, b) => sizeOf(b) - sizeOf(a));
    const singles = ordered.filter((c) => c.branches.length === 0 && sizeOf(c) === 1).map((c) => c.processes[0].nodes[0]);
    // A done-but-kept branch whose attachments all vanished is a single too (it has no component entry above).
    let bottom = 0;
    let columns = 1;

    const place = (list) => {
      const top = Math.min(...list.map((cell) => cell.row));
      for (const cell of list) {
        cell.row += bottom - top;
        cells.push(cell);
        columns = Math.max(columns, cell.col + cell.w);
      }
      bottom = Math.max(...list.map((cell) => cell.row + cell.h));
    };

    for (const tree of trees) {
      place(layoutTree(tree, cardById));
    }

    // Unbranched strings: columns, longest first, each running down from its lower-Z end.
    if (strings.length) {
      let col = 0;
      const list = [];
      for (const component of strings) {
        const p = component.processes[0];
        const nodes = p.nodes.slice();
        if (cardById.get(nodes[0]).z > cardById.get(nodes[nodes.length - 1]).z) {
          nodes.reverse();
        }
        const width = Math.max(...nodes.map((id) => cardById.get(id).w));
        let row = 0;
        for (const id of nodes) {
          const card = cardById.get(id);
          list.push({ id, row, col, w: card.w, h: card.h, group: p.index, kind: "process" });
          row += card.h;
        }
        col += width;
      }
      place(list);
    }

    // Unlinked cards, and branches left with nothing attached, in Z order wrapping at maxColumns.
    const placedBranches = new Set(trees.flatMap((c) => c.branches));
    const leftovers = [
      ...singles,
      ...branches.filter((id) => !placedBranches.has(id)),
    ].sort((a, b) => cardById.get(a).z - cardById.get(b).z || byId(a, b));
    if (leftovers.length) {
      const list = [];
      let row = 0;
      let col = 0;
      let rowHeight = 1;
      for (const id of leftovers) {
        const card = cardById.get(id);
        if (col > 0 && col + card.w > maxColumns) {
          row += rowHeight;
          col = 0;
          rowHeight = 1;
        }
        list.push({ id, row, col, w: card.w, h: card.h, group: -1, kind: "single" });
        col += card.w;
        rowHeight = Math.max(rowHeight, card.h);
      }
      place(list);
    }

    cells.sort((a, b) => a.row - b.row || a.col - b.col);
    return { cells, columns, totalRows: bottom };
  }

  /** One tree of processes and branches, rows relative (the caller shifts them), columns from 0. */
  function layoutTree(tree, cardById) {
    const cells = [];
    const occupied = new Set();
    const rightLimit = new Map(); // row -> first column that must stay empty (right of a branch)
    const key = (row, col) => `${row},${col}`;
    const area = (id) => cardById.get(id).area;
    const byId = (a, b) => (a < b ? -1 : a > b ? 1 : 0);

    const rootId = [...tree.branches].sort((a, b) => area(b) - area(a) || byId(a, b))[0];
    const placed = new Set([rootId]);
    const where = new Map(); // branch id -> {row, col, width}
    const taken = new Set(); // process indices already laid out

    /** The cards a string will occupy from the branch outward: its nodes, then the branch it ends in if not placed yet. */
    const sequenceOf = (p, from) => {
      const nodes = p.nodes[0] !== undefined && p.ends[0] !== from && p.ends[1] === from ? p.nodes.slice().reverse() : p.nodes.slice();
      const far = p.ends[0] === from ? p.ends[1] : p.ends[0];
      const endBranch = far !== null && far !== from && !placed.has(far) ? far : null;
      return { nodes, endBranch };
    };
    const averageZ = (nodes, fallback) =>
      nodes.length ? nodes.reduce((sum, id) => sum + cardById.get(id).z, 0) / nodes.length : fallback;

    /** Rows each card of the sequence takes when the string runs from `startRow` in `dir` (-1 up, +1 down). */
    const rowsFor = (ids, startRow, dir) => {
      const out = [];
      let cursor = startRow;
      for (const id of ids) {
        const h = cardById.get(id).h;
        if (dir > 0) {
          out.push(cursor);
          cursor += h;
        } else {
          out.push(cursor - h + 1);
          cursor -= h;
        }
      }
      return out;
    };

    const fits = (ids, rows, col, width, endsInBranch, relaxed) => {
      for (let i = 0; i < ids.length; i++) {
        const h = cardById.get(ids[i]).h;
        for (let r = rows[i]; r < rows[i] + h; r++) {
          for (let c = col; c < col + width; c++) {
            if (occupied.has(key(r, c))) {
              return false;
            }
            if (!relaxed && rightLimit.has(r) && c >= rightLimit.get(r)) {
              return false;
            }
          }
        }
      }
      if (endsInBranch) {
        // The branch ends its row: nothing already placed to its right on that row.
        const last = ids.length - 1;
        for (let r = rows[last]; r < rows[last] + cardById.get(ids[last]).h; r++) {
          for (const cell of cells) {
            if (r >= cell.row && r < cell.row + cell.h && cell.col >= col + width) {
              return false;
            }
          }
        }
      }
      return true;
    };

    const put = (ids, rows, col, width, group, endBranch) => {
      ids.forEach((id, i) => {
        const card = cardById.get(id);
        const isEnd = endBranch !== null && i === ids.length - 1;
        cells.push({ id, row: rows[i], col, w: card.w, h: card.h, group: isEnd ? -1 : group, kind: isEnd ? "branch" : "process" });
        for (let r = rows[i]; r < rows[i] + card.h; r++) {
          for (let c = col; c < col + width; c++) {
            occupied.add(key(r, c));
          }
        }
      });
      if (endBranch !== null) {
        const last = ids.length - 1;
        const branchRow = rows[last];
        where.set(endBranch, { row: branchRow, col, width });
        for (let r = branchRow; r < branchRow + cardById.get(endBranch).h; r++) {
          rightLimit.set(r, Math.min(rightLimit.has(r) ? rightLimit.get(r) : Infinity, col + width));
        }
      }
    };

    // The strings attached to a branch that are not laid out yet, longest first.
    const attached = (branchId) => {
      const strings = [];
      for (const p of tree.processes) {
        if (!taken.has(p.index) && p.ends.includes(branchId)) {
          strings.push({ process: p, ...sequenceOf(p, branchId) });
        }
      }
      for (const pair of tree.bridges) {
        const other = pair[0] === branchId ? pair[1] : pair[1] === branchId ? pair[0] : null;
        if (other !== null && !placed.has(other)) {
          strings.push({ process: null, nodes: [], endBranch: other });
        }
      }
      const length = (s) => s.nodes.length + (s.endBranch !== null ? 1 : 0);
      return strings.sort((a, b) => length(b) - length(a) || byId(a.nodes[0] ?? a.endBranch, b.nodes[0] ?? b.endBranch));
    };

    const ids = (s) => (s.endBranch !== null ? [...s.nodes, s.endBranch] : s.nodes);

    // The root: its strings are columns to its left, side by side from column 0, touching the root's row (row 0).
    const rootStrings = attached(rootId);
    const rootRow = 0;
    let col = 0;
    for (const s of rootStrings) {
      if (s.process !== null && taken.has(s.process.index)) {
        continue;
      }
      if (s.endBranch !== null && placed.has(s.endBranch)) {
        s.endBranch = null; // reached already by another string: a ring, so this string just ends
      }
      const sequence = ids(s);
      if (!sequence.length) {
        continue;
      }
      if (s.process !== null) {
        taken.add(s.process.index);
      }
      const dir = averageZ(s.nodes.length ? s.nodes : [s.endBranch], cardById.get(rootId).z) < cardById.get(rootId).z ? -1 : 1;
      const width = Math.max(...sequence.map((id) => cardById.get(id).w));
      const rows = rowsFor(sequence, rootRow, dir);
      if (s.endBranch !== null) {
        placed.add(s.endBranch);
      }
      put(sequence, rows, col, width, s.process === null ? -1 : s.process.index, s.endBranch);
      col += width;
    }
    const rootCard = cardById.get(rootId);
    cells.push({ id: rootId, row: rootRow, col, w: rootCard.w, h: rootCard.h, group: -1, kind: "branch" });
    for (let r = rootRow; r < rootRow + rootCard.h; r++) {
      occupied.add(key(r, col));
      rightLimit.set(r, col + rootCard.w);
    }
    where.set(rootId, { row: rootRow, col, width: rootCard.w });

    // Every other branch in the order it was reached: its strings go at its column or to the right, above or below it.
    const queue = cells.filter((cell) => cell.kind === "branch" && cell.id !== rootId).map((cell) => cell.id);
    while (queue.length) {
      const branchId = queue.shift();
      const spot = where.get(branchId);
      const branch = cardById.get(branchId);
      for (const s of attached(branchId)) {
        if (s.process !== null && taken.has(s.process.index)) {
          continue;
        }
        if (s.endBranch !== null && placed.has(s.endBranch)) {
          s.endBranch = null;
        }
        const sequence = ids(s);
        if (!sequence.length) {
          continue;
        }
        const preferred = averageZ(s.nodes.length ? s.nodes : [s.endBranch], branch.z) < branch.z ? -1 : 1;
        const width = Math.max(...sequence.map((id) => cardById.get(id).w));
        const reach = Math.max(...cells.map((cell) => cell.col + cell.w), spot.col) + 1;
        // The closest free spot: start next to the branch and slide away from it a row at a time, in the preferred
        // direction first, trying every column from the branch's column rightwards (past the edge of what is placed
        // if need be) before sliding further. The row rule holds throughout.
        let found = null;
        const extent = (pick) => pick(...cells.map((cell) => cell.row + cell.h), ...cells.map((cell) => cell.row), spot.row);
        const tallest = extent(Math.max) - extent(Math.min) + sequence.length + 2;
        for (let slide = 0; slide <= tallest && !found; slide++) {
          for (const dir of [preferred, -preferred]) {
            const startRow = dir > 0 ? spot.row + branch.h + slide : spot.row - 1 - slide;
            const rows = rowsFor(sequence, startRow, dir);
            for (let at = spot.col; at <= reach; at++) {
              if (fits(sequence, rows, at, width, s.endBranch !== null, false)) {
                found = { rows, at };
                break;
              }
            }
            if (found) {
              break;
            }
          }
        }
        if (!found) {
          const dir = preferred;
          const rows = rowsFor(sequence, dir > 0 ? extent(Math.max) : extent(Math.min) - 1, dir);
          found = { rows, at: reach };
        }
        const { rows, at } = found;
        if (s.process !== null) {
          taken.add(s.process.index);
        }
        if (s.endBranch !== null) {
          placed.add(s.endBranch);
          queue.push(s.endBranch);
        }
        put(sequence, rows, at, width, s.process === null ? -1 : s.process.index, s.endBranch);
      }
    }
    return cells;
  }

  return { layoutProcesses };
});
