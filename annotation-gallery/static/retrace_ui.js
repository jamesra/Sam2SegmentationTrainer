/*
 * Pen retrace in the viewer: draw a stroke across the mask outline and the outline is re-cut the way Viking's retrace
 * command does it (geometry in retrace.js). Starting inside the shape and leaving it grows the shape; starting outside
 * and crossing it twice removes the side counter-clockwise from the first crossing (hold Ctrl to remove the other).
 *
 * The gallery cannot write annotations yet, so a finished retrace is only shown, with a notice saying so.
 */
(function (root) {
  "use strict";

  const NOT_SAVED = "Saving retraced shapes to the database is not completed yet; your change is only shown here.";

  /**
   * deps: frame (the element holding the canvases), maskCanvas (the viewer's mask canvas; its size is the pixel
   * space), getMask() -> { width, height, inside(x, y) } | null, getHue() -> "#rrggbb", setStatus(text), onChange().
   */
  function create(deps) {
    const Retrace = root.Retrace;
    const canvas = document.createElement("canvas");
    canvas.id = "viewer-retrace";
    canvas.className = "viewer-retrace";
    deps.frame.appendChild(canvas);
    const dialog = document.getElementById("retrace-notice");
    const state = { active: false, polygon: null, original: null, stroke: null, source: "" };

    function sync() {
      const { width, height } = deps.maskCanvas;
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
    }

    function drawRing(context, ring, style) {
      context.beginPath();
      ring.forEach(([x, y], index) => (index ? context.lineTo(x, y) : context.moveTo(x, y)));
      context.closePath();
      if (style.fill) {
        context.fillStyle = style.fill;
        context.fill("evenodd");
      }
      context.strokeStyle = style.stroke;
      context.lineWidth = style.width;
      context.stroke();
    }

    function paint() {
      sync();
      const context = canvas.getContext("2d");
      context.clearRect(0, 0, canvas.width, canvas.height);
      const edited = state.polygon && state.polygon !== state.original;
      deps.maskCanvas.style.visibility = edited ? "hidden" : "";
      if (state.polygon) {
        const hue = deps.getHue();
        const lineWidth = Math.max(1.5, canvas.width / 400);
        context.save();
        context.beginPath();
        for (const ring of [state.polygon.exterior, ...state.polygon.holes]) {
          ring.forEach(([x, y], index) => (index ? context.lineTo(x, y) : context.moveTo(x, y)));
          context.closePath();
        }
        context.fillStyle = `${hue}59`;
        context.fill("evenodd");
        context.restore();
        drawRing(context, state.polygon.exterior, { stroke: hue, width: lineWidth });
        for (const hole of state.polygon.holes) {
          drawRing(context, hole, { stroke: hue, width: lineWidth });
        }
      }
      if (state.stroke && state.stroke.length > 1) {
        context.beginPath();
        state.stroke.forEach(([x, y], index) => (index ? context.lineTo(x, y) : context.moveTo(x, y)));
        context.strokeStyle = "#ffffff";
        context.lineWidth = Math.max(1.5, canvas.width / 500);
        context.setLineDash([6, 4]);
        context.stroke();
        context.setLineDash([]);
      }
    }

    function toPixels(event) {
      const box = canvas.getBoundingClientRect();
      return [
        ((event.clientX - box.left) / box.width) * canvas.width,
        ((event.clientY - box.top) / box.height) * canvas.height,
      ];
    }

    function loadSource() {
      const mask = deps.getMask();
      if (!mask) {
        return false;
      }
      const ring = Retrace.maskToPolygon(mask.width, mask.height, mask.inside, 1.0);
      if (!ring) {
        return false;
      }
      state.original = Retrace.makePolygon(ring);
      state.polygon = state.original;
      state.source = "mask";
      return true;
    }

    function finish(event) {
      const stroke = state.stroke;
      state.stroke = null;
      if (!stroke || stroke.length < 2) {
        paint();
        return;
      }
      const result = Retrace.retrace(state.polygon, stroke, { switchSide: Boolean(event && event.ctrlKey) });
      if (result.action === Retrace.ACTION.NONE) {
        deps.setStatus("Retrace needs a stroke that crosses the outline twice without looping or grazing a corner.");
        paint();
        return;
      }
      state.polygon = result.polygon;
      paint();
      deps.setStatus(`Retrace: ${result.action.toLowerCase().replace(/_/g, " ")}. ${NOT_SAVED}`);
      if (dialog) {
        dialog.hidden = false;
      }
      deps.onChange && deps.onChange(state.polygon);
    }

    canvas.addEventListener("pointerdown", (event) => {
      if (!state.active || event.button !== 0) {
        return;
      }
      event.preventDefault();
      canvas.setPointerCapture(event.pointerId);
      state.stroke = [toPixels(event)];
      paint();
    });
    canvas.addEventListener("pointermove", (event) => {
      if (!state.active || !state.stroke) {
        return;
      }
      state.stroke.push(toPixels(event));
      paint();
    });
    canvas.addEventListener("pointerup", (event) => {
      if (state.stroke) {
        canvas.releasePointerCapture(event.pointerId);
        finish(event);
      }
    });
    canvas.addEventListener("pointercancel", () => {
      state.stroke = null;
      paint();
    });

    function setActive(on) {
      if (on && !state.polygon && !loadSource()) {
        deps.setStatus("There is no mask outline to retrace here.");
        return false;
      }
      state.active = Boolean(on);
      canvas.classList.toggle("active", state.active);
      paint();
      return state.active;
    }

    function reset() {
      state.active = false;
      state.polygon = null;
      state.original = null;
      state.stroke = null;
      canvas.classList.remove("active");
      deps.maskCanvas.style.visibility = "";
      const context = canvas.getContext("2d");
      context.clearRect(0, 0, canvas.width, canvas.height);
    }

    function revert() {
      if (state.original) {
        state.polygon = state.original;
        paint();
      }
    }

    return {
      get active() {
        return state.active;
      },
      get polygon() {
        return state.polygon;
      },
      setActive,
      toggle: () => setActive(!state.active),
      reset,
      revert,
      repaint: paint,
    };
  }

  root.RetraceUI = { create, NOT_SAVED };
})(typeof self !== "undefined" ? self : this);
