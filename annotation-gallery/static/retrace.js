/*
 * Pen retrace geometry: a port of Viking's RetraceAndReplacePathCommand cut (Geometry.Core Polygon.WalkPolygonCut,
 * AddPointsAtIntersections, IntersectingSegmentIndices, PolygonCurveExtensions.ReduceExteriorRing,
 * CurveExtensions.DouglasPeuckerReduction). Pure functions on plain arrays, no DOM, so the browser and the node tests
 * run the same file.
 *
 * A polygon is { exterior: ring, holes: [ring, ...] }. A ring is a closed array of [x, y] (last point equals the
 * first). The exterior is kept counter-clockwise in ordinary maths axes (positive signed area), as Polygon does;
 * "ccw" and "cw" walks below refer to walking that ring with Next and Previous. The caller chooses the axes; the
 * geometry does not care whether Y points up or down on screen.
 *
 * Sizes are in whatever units the points use (the gallery uses crop pixels). EPS matches Geometry Tolerance.Epsilon.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.Retrace = factory();
  }
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const EPS = 1e-3;
  const EPS2 = 1e-6;

  const ACTION = Object.freeze({
    NONE: "NONE",
    GROW_EXTERIOR_RING: "GROW_EXTERIOR_RING",
    SHRINK_EXTERIOR_RING: "SHRINK_EXTERIOR_RING",
    GROW_INTERNAL_RING: "GROW_INTERNAL_RING",
    SHRINK_INTERNAL_RING: "SHRINK_INTERNAL_RING",
  });

  class RetraceError extends Error {
    constructor(message) {
      super(message);
      this.name = "RetraceError";
    }
  }

  // ------------------------------------------------------------------------------------------------ vectors

  const dist2 = (a, b) => (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]);
  const dist = (a, b) => Math.sqrt(dist2(a, b));
  const same = (a, b) => a[0] === b[0] && a[1] === b[1];
  const copy = (p) => [p[0], p[1]];

  function pointSegmentDistance(p, a, b) {
    const dx = b[0] - a[0];
    const dy = b[1] - a[1];
    const len2 = dx * dx + dy * dy;
    if (len2 === 0) {
      return dist(p, a);
    }
    let t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2;
    t = Math.max(0, Math.min(1, t));
    return dist(p, [a[0] + t * dx, a[1] + t * dy]);
  }

  /**
   * Where segments a-b and c-d meet, as { point, t } (t along a-b), or null. Touching at an end counts, as in
   * LineSegment.Intersects. A meeting point within EPS of an end is snapped to that end (ring ends first), so the same
   * vertex is reported identically from both ring segments that share it. Parallel segments report a shared end only.
   */
  function segmentIntersection(a, b, c, d) {
    const rx = b[0] - a[0];
    const ry = b[1] - a[1];
    const sx = d[0] - c[0];
    const sy = d[1] - c[1];
    const rl = Math.hypot(rx, ry);
    const sl = Math.hypot(sx, sy);
    if (rl === 0 || sl === 0) {
      return null;
    }
    const cross = rx * sy - ry * sx;
    if (Math.abs(cross) < EPS2) {
      for (const p of [a, b, c, d]) {
        const onFirst = pointSegmentDistance(p, a, b) < EPS;
        const onSecond = pointSegmentDistance(p, c, d) < EPS;
        if (onFirst && onSecond) {
          const t = ((p[0] - a[0]) * rx + (p[1] - a[1]) * ry) / (rl * rl);
          return { point: copy(p), t };
        }
      }
      return null;
    }
    const qx = c[0] - a[0];
    const qy = c[1] - a[1];
    const t = (qx * sy - qy * sx) / cross;
    const u = (qx * ry - qy * rx) / cross;
    if (t < -EPS / rl || t > 1 + EPS / rl || u < -EPS / sl || u > 1 + EPS / sl) {
      return null;
    }
    let point = [a[0] + t * rx, a[1] + t * ry];
    for (const e of [a, b, c, d]) {
      if (dist2(point, e) <= EPS2) {
        point = copy(e);
        break;
      }
    }
    return { point, t };
  }

  // ------------------------------------------------------------------------------------------------ rings

  function signedArea(ring) {
    let sum = 0;
    for (let i = 0; i < ring.length - 1; i++) {
      sum += ring[i][0] * ring[i + 1][1] - ring[i + 1][0] * ring[i][1];
    }
    return sum / 2;
  }

  function closeRing(points) {
    const ring = points.map(copy);
    if (ring.length && !same(ring[0], ring[ring.length - 1])) {
      ring.push(copy(ring[0]));
    }
    return ring;
  }

  /** Closed ring, counter-clockwise: a clockwise ring is reversed, so it still starts at its first point. */
  function normalizeCcw(points) {
    const ring = closeRing(points);
    return signedArea(ring) < 0 ? ring.reverse() : ring;
  }

  function makePolygon(exterior, holes = []) {
    return { exterior: normalizeCcw(exterior), holes: holes.map(closeRing) };
  }

  function clonePolygon(polygon) {
    return { exterior: polygon.exterior.map(copy), holes: polygon.holes.map((h) => h.map(copy)) };
  }

  const polygonArea = (polygon) => Math.abs(signedArea(polygon.exterior));
  const totalVertices = (polygon) =>
    polygon.exterior.length - 1 + polygon.holes.reduce((n, h) => n + h.length - 1, 0);

  function pointInRing(ring, p) {
    for (let i = 0; i < ring.length - 1; i++) {
      if (pointSegmentDistance(p, ring[i], ring[i + 1]) < EPS) {
        return true;
      }
    }
    let inside = false;
    for (let i = 0, j = ring.length - 2; i < ring.length - 1; j = i++) {
      const [xi, yi] = ring[i];
      const [xj, yj] = ring[j];
      if (yi > p[1] !== yj > p[1] && p[0] < ((xj - xi) * (p[1] - yi)) / (yj - yi) + xi) {
        inside = !inside;
      }
    }
    return inside;
  }

  /** True when p lies in the exterior (edge included) and not inside a hole. */
  function covers(polygon, p) {
    if (!pointInRing(polygon.exterior, p)) {
      return false;
    }
    for (const hole of polygon.holes) {
      if (pointInRing(hole, p) && !hole.slice(0, -1).some((v, i) => pointSegmentDistance(p, v, hole[i + 1]) < EPS)) {
        return false;
      }
    }
    return true;
  }

  function segmentsTouch(a, b, c, d) {
    return segmentIntersection(a, b, c, d) !== null;
  }

  function ringSelfIntersects(ring) {
    const n = ring.length - 1;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const adjacent = j === i + 1 || (i === 0 && j === n - 1);
        if (adjacent) {
          continue;
        }
        if (segmentsTouch(ring[i], ring[i + 1], ring[j], ring[j + 1])) {
          return true;
        }
      }
    }
    return false;
  }

  function ringsIntersect(ringA, ringB) {
    for (let i = 0; i < ringA.length - 1; i++) {
      for (let j = 0; j < ringB.length - 1; j++) {
        if (segmentsTouch(ringA[i], ringA[i + 1], ringB[j], ringB[j + 1])) {
          return true;
        }
      }
    }
    return false;
  }

  /** Polygon.IsValid: no repeated vertex, no self-intersection, holes inside the exterior and apart, and area. */
  function isValid(polygon) {
    const ring = polygon.exterior;
    const distinct = new Set(ring.slice(0, -1).map((p) => `${p[0]},${p[1]}`));
    if (distinct.size !== ring.length - 1 || ring.length < 4) {
      return false;
    }
    if (ringSelfIntersects(ring)) {
      return false;
    }
    for (let i = 0; i < polygon.holes.length; i++) {
      const hole = polygon.holes[i];
      if (ringSelfIntersects(hole) || ringsIntersect(ring, hole) || !pointInRing(ring, hole[0])) {
        return false;
      }
      for (let j = i + 1; j < polygon.holes.length; j++) {
        if (ringsIntersect(hole, polygon.holes[j])) {
          return false;
        }
      }
    }
    return polygonArea(polygon) >= EPS;
  }

  // ------------------------------------------------------------------------------------------------ cutting

  /** Polygon.AddPointsAtIntersections for one ring: a vertex wherever a cut segment meets it. */
  function addPointsAtIntersections(ring, cutSegments) {
    const out = [];
    for (let i = 0; i < ring.length - 1; i++) {
      const a = ring[i];
      const b = ring[i + 1];
      if (!out.length || dist2(out[out.length - 1], a) > EPS2) {
        out.push(copy(a));
      }
      const hits = [];
      for (const [c, d] of cutSegments) {
        const hit = segmentIntersection(a, b, c, d);
        if (hit) {
          hits.push(hit);
        }
      }
      hits.sort((x, y) => x.t - y.t);
      for (const hit of hits) {
        if (!out.length || dist2(out[out.length - 1], hit.point) > EPS2) {
          out.push(copy(hit.point));
        }
      }
    }
    // Close on the first vertex exactly, even when a crossing landed within EPS of it.
    if (dist2(out[out.length - 1], ring[0]) <= EPS2) {
      out[out.length - 1] = copy(ring[0]);
    } else {
      out.push(copy(ring[0]));
    }
    return out;
  }

  /**
   * Polygon.IntersectingSegmentIndices for one cut segment: the ring vertex each crossing resolves to, nearest to
   * `from` first. A crossing on a vertex is reported once, whichever of the two ring segments sharing it finds it.
   * `ring` 0 is the exterior and k is hole k - 1.
   */
  function crossingsOnSegment(rings, from, to) {
    const found = [];
    const seen = new Set();
    rings.forEach((ring, ringIndex) => {
      const n = ring.length - 1;
      for (let i = 0; i < n; i++) {
        const a = ring[i];
        const b = ring[i + 1];
        const hit = segmentIntersection(a, b, from, to);
        if (!hit) {
          continue;
        }
        let index = i;
        if (same(hit.point, b)) {
          index = (i + 1) % n;
        }
        const key = `${ringIndex}:${index}`;
        if (seen.has(key)) {
          continue;
        }
        seen.add(key);
        found.push({ ring: ringIndex, index, distance: dist(from, hit.point) });
      }
    });
    found.sort((x, y) => x.distance - y.distance);
    return found;
  }

  function toSegments(points) {
    const segments = [];
    for (let i = 0; i < points.length - 1; i++) {
      segments.push([points[i], points[i + 1]]);
    }
    return segments;
  }

  /** Crossings of a whole path with the polygon's rings, ordered by distance along the path. */
  function intersectingSegmentIndices(polygon, path) {
    const rings = [polygon.exterior, ...polygon.holes];
    const all = [];
    let travelled = 0;
    for (let i = 0; i < path.length - 1; i++) {
      for (const hit of crossingsOnSegment(rings, path[i], path[i + 1])) {
        all.push({ ring: hit.ring, index: hit.index, distance: hit.distance + travelled });
      }
      travelled += dist(path[i], path[i + 1]);
    }
    return all.sort((x, y) => x.distance - y.distance);
  }

  function walkRing(ring, first, last, direction, cutVertices, source) {
    const n = ring.length - 1;
    const walked = [];
    let current = first.index;
    do {
      walked.push(ring[current]);
      current = direction === "ccw" ? (current + 1) % n : (current - 1 + n) % n;
    } while (current !== last.index);
    walked.push(ring[last.index]);

    // The walk began at the first crossing, so the cut comes back to it in reverse order.
    const startPoint = ring[first.index];
    for (const cutPoint of [...cutVertices].reverse()) {
      if (dist2(cutPoint, walked[walked.length - 1]) <= EPS2) {
        continue;
      }
      if (dist2(cutPoint, startPoint) <= EPS2) {
        continue;
      }
      if (walked.some((w) => same(w, cutPoint))) {
        throw new RetraceError(
          "Cut line coincides with a vertex already on the walked polygon arc. (Does the cutting line graze a corner?)",
        );
      }
      walked.push(copy(cutPoint));
    }
    walked.push(copy(startPoint));

    const output = makePolygon(walked);
    for (const hole of source.holes) {
      if (pointInRing(output.exterior, hole[0])) {
        output.holes.push(hole.map(copy));
      }
    }
    if (!isValid(output)) {
      throw new RetraceError("Invalid polygon created by cut. (Does the cutting line have loops?)");
    }
    return output;
  }

  /**
   * Polygon.WalkPolygonCut: cut `polygon` along `cutLine`. The line must meet the same ring twice. `direction` is
   * "ccw" or "cw" (walk the ring with Next or Previous from the first crossing to the second). Returns the polygon made
   * by that walk closed with the part of the line between the crossings. Throws RetraceError when no cut is possible.
   */
  function walkPolygonCut(polygon, direction, cutLine) {
    const cutSegments = toSegments(cutLine);
    const exterior = addPointsAtIntersections(polygon.exterior, cutSegments);
    const holes = polygon.holes.map((hole) => addPointsAtIntersections(hole, cutSegments));
    const rings = [exterior, ...holes];

    const cutVertices = [];
    const crossings = [];
    let firstFound = false;
    for (let i = 0; i < cutLine.length - 1; i++) {
      const hits = crossingsOnSegment(rings, cutLine[i], cutLine[i + 1]);
      if (firstFound) {
        if (hits.length === 0) {
          cutVertices.push(cutLine[i + 1]);
        }
      } else if (hits.length === 1) {
        firstFound = true;
        cutVertices.push(cutLine[i + 1]);
      }
      crossings.push(...hits);
      if (crossings.length >= 2) {
        break;
      }
    }

    if (crossings.length === 0) {
      throw new RetraceError("cutLine must intersect a polygon ring");
    }
    if (crossings.length === 1) {
      throw new RetraceError("cutline must intersect a polygon ring a second time.");
    }
    const first = crossings[crossings.length - 2];
    const last = crossings[crossings.length - 1];
    if (first.ring !== last.ring) {
      throw new RetraceError("Cut line must cross segments on the same ring of the polygon");
    }
    if (first.index === last.index) {
      throw new RetraceError(`Start and End index must be different to cut polygon. Both are ${first.index}`);
    }
    return walkRing(rings[first.ring], first, last, direction, cutVertices, { holes: polygon.holes.length && first.ring === 0 ? holes : [] });
  }

  // ------------------------------------------------------------------------------------- reduction (Douglas-Peucker)

  function douglasPeuckerKeep(points, first, last, tolerance, keep) {
    let maxDistance = 0;
    let farthest = 0;
    for (let i = first + 1; i < last; i++) {
      const d = pointSegmentDistance(points[i], points[first], points[last]);
      if (d > maxDistance) {
        maxDistance = d;
        farthest = i;
      }
    }
    if (maxDistance > tolerance && farthest !== 0) {
      keep.add(farthest);
      douglasPeuckerKeep(points, first, farthest, tolerance, keep);
      douglasPeuckerKeep(points, farthest, last, tolerance, keep);
    }
  }

  /** CurveExtensions.DouglasPeuckerReduction: surviving points keep their coordinates. For a closed ring the closing point is kept. */
  function douglasPeucker(points, tolerance) {
    if (points.length < 3) {
      return points.map(copy);
    }
    const keep = new Set([0, points.length - 1]);
    let last = points.length - 1;
    while (last > 0 && same(points[0], points[last])) {
      last--;
    }
    keep.add(last);
    douglasPeuckerKeep(points, 0, last, tolerance, keep);
    return [...keep].sort((a, b) => a - b).map((i) => copy(points[i]));
  }

  /** PolygonCurveExtensions.ReduceExteriorRing: drop exterior vertices within `tolerance` of the chord; holes are copied. */
  function reduceExteriorRing(polygon, tolerance) {
    if (tolerance <= 0 || polygon.exterior.length < 5) {
      return polygon;
    }
    const reduced = douglasPeucker(polygon.exterior, tolerance);
    if (reduced.length < 4 || reduced.length >= polygon.exterior.length) {
      return polygon;
    }
    const output = makePolygon(reduced, polygon.holes);
    return isValid(output) ? output : polygon;
  }

  /** Open-path Douglas-Peucker (both ends kept) after dropping repeated samples: the stroke simplification. */
  function simplifyStroke(points, tolerance) {
    const unique = [];
    for (const p of points) {
      if (!unique.length || dist2(unique[unique.length - 1], p) > EPS2) {
        unique.push(copy(p));
      }
    }
    if (unique.length < 3 || tolerance <= 0) {
      return unique;
    }
    const keep = new Set([0, unique.length - 1]);
    douglasPeuckerKeep(unique, 0, unique.length - 1, tolerance, keep);
    return [...keep].sort((a, b) => a - b).map((i) => unique[i]);
  }

  // ------------------------------------------------------------------------------------------- the retrace edit

  /**
   * The whole edit, as RetraceAndReplacePathCommand does it for one stroke.
   *
   * Starting inside the shape grows it (the larger of the two cut polygons is kept); starting outside shrinks it (the
   * counter-clockwise piece is kept, or the clockwise one with `switchSide`). The first ring the stroke meets decides
   * whether the exterior or a hole is edited. Nothing is changed when the stroke does not cross a ring twice, cannot be
   * spliced, or loops (see `loopHole` for the loop case).
   *
   * Returns { action, polygon, clockwise, counterClockwise }; polygon is null when action is NONE.
   */
  function retrace(polygon, rawStroke, options = {}) {
    const { strokeTolerance = 1.0, reduceTolerance = 1.0, switchSide = false } = options;
    const none = { action: ACTION.NONE, polygon: null, clockwise: null, counterClockwise: null };
    const stroke = simplifyStroke(rawStroke, strokeTolerance);
    if (stroke.length <= 1 || totalVertices(polygon) <= 3) {
      return none;
    }
    const hits = intersectingSegmentIndices(polygon, stroke);
    if (hits.length < 2) {
      return none;
    }
    const ringIndex = hits[0].ring;
    const toCut = ringIndex === 0 ? polygon : makePolygon(polygon.holes[ringIndex - 1]);

    let clockwise;
    let counterClockwise;
    try {
      clockwise = reduceExteriorRing(walkPolygonCut(toCut, "cw", stroke), reduceTolerance);
      counterClockwise = reduceExteriorRing(walkPolygonCut(toCut, "ccw", stroke), reduceTolerance);
    } catch (error) {
      if (error instanceof RetraceError) {
        return none;
      }
      throw error;
    }

    const expandsArea = covers(polygon, stroke[0]);
    const larger = polygonArea(counterClockwise) > polygonArea(clockwise) ? counterClockwise : clockwise;
    const shrunk = switchSide ? clockwise : counterClockwise;
    let action;
    let replacement;
    if (ringIndex > 0) {
      action = expandsArea ? ACTION.SHRINK_INTERNAL_RING : ACTION.GROW_INTERNAL_RING;
      replacement = expandsArea ? shrunk : larger;
    } else {
      action = expandsArea ? ACTION.GROW_EXTERIOR_RING : ACTION.SHRINK_EXTERIOR_RING;
      replacement = expandsArea ? larger : shrunk;
    }

    let result;
    if (ringIndex > 0) {
      result = clonePolygon(polygon);
      result.holes[ringIndex - 1] = replacement.exterior.map(copy);
    } else {
      result = replacement;
    }
    return { action, polygon: result, clockwise, counterClockwise };
  }

  /**
   * A stroke that crosses itself (a loop) is not a retrace: Viking adds the first loop as a new hole instead. Returns
   * the polygon with the loop's ring added, or null when the stroke has no loop or the ring is not valid inside.
   */
  function loopHole(polygon, rawStroke, options = {}) {
    const stroke = simplifyStroke(rawStroke, options.strokeTolerance ?? 1.0);
    for (let i = 0; i < stroke.length - 1; i++) {
      for (let j = i + 2; j < stroke.length - 1; j++) {
        const hit = segmentIntersection(stroke[i], stroke[i + 1], stroke[j], stroke[j + 1]);
        if (!hit) {
          continue;
        }
        const ring = closeRing([hit.point, ...stroke.slice(i + 1, j + 1)]);
        if (ring.length < 4 || Math.abs(signedArea(ring)) < EPS || ringSelfIntersects(ring)) {
          return null;
        }
        const candidate = clonePolygon(polygon);
        candidate.holes.push(ring);
        return isValid(candidate) ? candidate : null;
      }
    }
    return null;
  }

  /** True when the stroke crosses itself, which makes it a loop and not a retrace. */
  function strokeLoops(rawStroke, strokeTolerance = 1.0) {
    const stroke = simplifyStroke(rawStroke, strokeTolerance);
    for (let i = 0; i < stroke.length - 1; i++) {
      for (let j = i + 2; j < stroke.length - 1; j++) {
        if (segmentIntersection(stroke[i], stroke[i + 1], stroke[j], stroke[j + 1])) {
          return true;
        }
      }
    }
    return false;
  }

  /**
   * Outline of the largest blob in a mask: `inside(x, y)` is true for mask pixels. Walks the pixel edges of the 4-connected
   * largest component (so the result follows pixel corners), then drops vertices within `tolerance` of the chord. Holes
   * are not returned. Returns a closed ring in pixel-corner coordinates, or null when the mask is empty.
   */
  function maskToPolygon(width, height, inside, tolerance = 1.0) {
    const label = new Int32Array(width * height);
    let best = { id: 0, size: 0, start: -1 };
    let nextId = 0;
    const stack = [];
    for (let start = 0; start < width * height; start++) {
      if (label[start] || !inside(start % width, Math.floor(start / width))) {
        continue;
      }
      nextId += 1;
      let size = 0;
      label[start] = nextId;
      stack.push(start);
      while (stack.length) {
        const at = stack.pop();
        size += 1;
        const x = at % width;
        const y = (at - x) / width;
        for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
          const nx = x + dx;
          const ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= width || ny >= height) {
            continue;
          }
          const n = ny * width + nx;
          if (!label[n] && inside(nx, ny)) {
            label[n] = nextId;
            stack.push(n);
          }
        }
      }
      if (size > best.size) {
        best = { id: nextId, size, start };
      }
    }
    if (!best.id) {
      return null;
    }
    const mine = (x, y) => x >= 0 && y >= 0 && x < width && y < height && label[y * width + x] === best.id;
    const outgoing = new Map();
    const add = (ax, ay, bx, by) => {
      const key = ay * (width + 1) + ax;
      if (!outgoing.has(key)) {
        outgoing.set(key, []);
      }
      outgoing.get(key).push([bx, by]);
    };
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (!mine(x, y)) {
          continue;
        }
        if (!mine(x, y - 1)) add(x, y, x + 1, y);
        if (!mine(x + 1, y)) add(x + 1, y, x + 1, y + 1);
        if (!mine(x, y + 1)) add(x + 1, y + 1, x, y + 1);
        if (!mine(x - 1, y)) add(x, y + 1, x, y);
      }
    }
    // The first pixel in scan order has a free top edge: the walk starts on the outer boundary.
    const sx = best.start % width;
    const sy = (best.start - sx) / width;
    const ring = [[sx, sy]];
    let [cx, cy] = [sx + 1, sy];
    let guard = 0;
    while ((cx !== sx || cy !== sy) && guard++ < 4 * width * height + 8) {
      ring.push([cx, cy]);
      const options = outgoing.get(cy * (width + 1) + cx);
      if (!options || !options.length) {
        return null;
      }
      [cx, cy] = options.pop();
    }
    ring.push([sx, sy]);
    const corners = [];
    for (let i = 0; i < ring.length - 1; i++) {
      const a = ring[(i + ring.length - 2) % (ring.length - 1)];
      const b = ring[i];
      const c = ring[i + 1];
      if ((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) !== 0) {
        corners.push(b);
      }
    }
    if (corners.length < 3) {
      return null;
    }
    corners.push(corners[0]);
    return douglasPeucker(corners, tolerance);
  }

  return {
    maskToPolygon,
    ACTION,
    EPS,
    RetraceError,
    signedArea,
    makePolygon,
    clonePolygon,
    polygonArea,
    covers,
    isValid,
    addPointsAtIntersections,
    intersectingSegmentIndices,
    walkPolygonCut,
    douglasPeucker,
    reduceExteriorRing,
    simplifyStroke,
    retrace,
    loopHole,
    strokeLoops,
  };
});
