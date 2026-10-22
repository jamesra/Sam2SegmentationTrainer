/*
 * Brush and eraser for touching up a gold card's mask in the viewer. Paint with the left button, erase with the eraser tool
 * (or hold Shift while painting). Edits stay on a working copy until Save; Undo steps back one stroke (Ctrl+Z while the brush
 * is on). BrushCore is the pure part (no DOM) and is what the node tests exercise.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    const api = factory();
    root.BrushCore = api.BrushCore;
    root.BrushUI = api.BrushUI;
  }
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const MAX_UNDO = 30;

  const BrushCore = {
    /** Working mask from a predicate: Uint8Array of 0/1, row-major. */
    fromPredicate(width, height, inside) {
      const mask = new Uint8Array(width * height);
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          if (inside(x, y)) {
            mask[y * width + x] = 1;
          }
        }
      }
      return mask;
    },

    /** Set (value 1) or clear (value 0) every pixel whose centre is within `radius` of (cx, cy), pixel-edge coordinates. */
    stamp(mask, width, height, cx, cy, radius, value) {
      const r = Math.max(0.5, radius);
      const x0 = Math.max(0, Math.floor(cx - r));
      const x1 = Math.min(width - 1, Math.ceil(cx + r));
      const y0 = Math.max(0, Math.floor(cy - r));
      const y1 = Math.min(height - 1, Math.ceil(cy + r));
      const r2 = r * r;
      for (let y = y0; y <= y1; y++) {
        const dy = y + 0.5 - cy;
        for (let x = x0; x <= x1; x++) {
          const dx = x + 0.5 - cx;
          if (dx * dx + dy * dy <= r2) {
            mask[y * width + x] = value;
          }
        }
      }
    },

    /** A stroke from (x0, y0) to (x1, y1): discs spaced no more than a quarter radius apart, so there are no gaps. */
    line(mask, width, height, x0, y0, x1, y1, radius, value) {
      const step = Math.max(0.5, radius / 4);
      const length = Math.hypot(x1 - x0, y1 - y0);
      const count = Math.max(1, Math.ceil(length / step));
      for (let i = 0; i <= count; i++) {
        const t = i / count;
        BrushCore.stamp(mask, width, height, x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, radius, value);
      }
    },

    count(mask) {
      let total = 0;
      for (let i = 0; i < mask.length; i++) {
        total += mask[i];
      }
      return total;
    },

    /** Pixels that differ between two masks. */
    changed(a, b) {
      let total = 0;
      for (let i = 0; i < a.length; i++) {
        if (a[i] !== b[i]) {
          total += 1;
        }
      }
      return total;
    },

    /** A bounded undo history of mask snapshots. */
    history(limit = MAX_UNDO) {
      const stack = [];
      return {
        push(mask) {
          stack.push(mask.slice());
          if (stack.length > limit) {
            stack.shift();
          }
        },
        pop() {
          return stack.pop() || null;
        },
        get size() {
          return stack.length;
        },
        clear() {
          stack.length = 0;
        },
      };
    },
  };

  const BrushUI = {
    /**
     * deps: frame, maskCanvas (pixel space), getMask() -> { width, height, inside(x, y) } | null, getHue(), setStatus(text),
     * onSave(dataUrl) -> Promise (the host stores the mask and reloads the card).
     */
    create(deps) {
      const canvas = document.createElement("canvas");
      canvas.id = "viewer-brush";
      canvas.className = "viewer-brush";
      deps.frame.appendChild(canvas);
      const bar = document.getElementById("brush-bar");
      const sizeInput = document.getElementById("brush-size");
      const eraserInput = document.getElementById("brush-eraser");
      const undoButton = document.getElementById("brush-undo");
      const saveButton = document.getElementById("brush-save");
      const state = { active: false, mask: null, original: null, width: 0, height: 0, history: BrushCore.history(), stroke: null, cursor: null };

      const radius = () => Math.max(1, Number(sizeInput ? sizeInput.value : 6) / 2);
      const erasing = (event) => Boolean(eraserInput && eraserInput.checked) || Boolean(event && event.shiftKey);

      function paint() {
        if (!state.mask) {
          return;
        }
        if (canvas.width !== state.width || canvas.height !== state.height) {
          canvas.width = state.width;
          canvas.height = state.height;
        }
        const context = canvas.getContext("2d");
        const image = context.createImageData(state.width, state.height);
        const hue = deps.getHue();
        const red = parseInt(hue.slice(1, 3), 16);
        const green = parseInt(hue.slice(3, 5), 16);
        const blue = parseInt(hue.slice(5, 7), 16);
        for (let i = 0; i < state.mask.length; i++) {
          if (state.mask[i]) {
            image.data[i * 4] = red;
            image.data[i * 4 + 1] = green;
            image.data[i * 4 + 2] = blue;
            image.data[i * 4 + 3] = 115;
          }
        }
        context.putImageData(image, 0, 0);
        if (state.cursor) {
          context.beginPath();
          context.arc(state.cursor[0], state.cursor[1], radius(), 0, Math.PI * 2);
          context.strokeStyle = "#ffffff";
          context.lineWidth = Math.max(1, state.width / 600);
          context.stroke();
        }
        const dirty = BrushCore.changed(state.mask, state.original) > 0;
        if (undoButton) {
          undoButton.disabled = state.history.size === 0;
        }
        if (saveButton) {
          saveButton.disabled = !dirty;
        }
        deps.maskCanvas.style.visibility = "hidden";
      }

      function toPixels(event) {
        const box = canvas.getBoundingClientRect();
        return [((event.clientX - box.left) / box.width) * canvas.width, ((event.clientY - box.top) / box.height) * canvas.height];
      }

      function load() {
        const source = deps.getMask();
        if (!source) {
          return false;
        }
        state.width = source.width;
        state.height = source.height;
        state.mask = BrushCore.fromPredicate(source.width, source.height, source.inside);
        state.original = state.mask.slice();
        state.history.clear();
        return true;
      }

      canvas.addEventListener("pointerdown", (event) => {
        if (!state.active || event.button !== 0) {
          return;
        }
        event.preventDefault();
        canvas.setPointerCapture(event.pointerId);
        state.history.push(state.mask);
        const [x, y] = toPixels(event);
        state.stroke = { value: erasing(event) ? 0 : 1, last: [x, y] };
        BrushCore.stamp(state.mask, state.width, state.height, x, y, radius(), state.stroke.value);
        state.cursor = [x, y];
        paint();
      });
      canvas.addEventListener("pointermove", (event) => {
        if (!state.active) {
          return;
        }
        const [x, y] = toPixels(event);
        state.cursor = [x, y];
        if (state.stroke) {
          BrushCore.line(state.mask, state.width, state.height, state.stroke.last[0], state.stroke.last[1], x, y, radius(), state.stroke.value);
          state.stroke.last = [x, y];
        }
        paint();
      });
      canvas.addEventListener("pointerup", (event) => {
        if (state.stroke) {
          canvas.releasePointerCapture(event.pointerId);
          state.stroke = null;
          paint();
        }
      });
      canvas.addEventListener("pointerleave", () => {
        state.cursor = null;
        if (state.active) {
          paint();
        }
      });

      function undo() {
        const previous = state.history.pop();
        if (previous) {
          state.mask = previous;
          paint();
        }
      }

      function reset() {
        state.active = false;
        state.mask = null;
        state.original = null;
        state.stroke = null;
        state.history.clear();
        canvas.classList.remove("active");
        if (bar) {
          bar.hidden = true;
        }
        deps.maskCanvas.style.visibility = "";
        canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
      }

      async function save() {
        if (!state.mask || BrushCore.changed(state.mask, state.original) === 0) {
          return;
        }
        const out = document.createElement("canvas");
        out.width = state.width;
        out.height = state.height;
        const context = out.getContext("2d");
        const image = context.createImageData(state.width, state.height);
        for (let i = 0; i < state.mask.length; i++) {
          const value = state.mask[i] ? 255 : 0;
          image.data[i * 4] = value;
          image.data[i * 4 + 1] = value;
          image.data[i * 4 + 2] = value;
          image.data[i * 4 + 3] = 255;
        }
        context.putImageData(image, 0, 0);
        saveButton.disabled = true;
        try {
          await deps.onSave(out.toDataURL("image/png"));
        } catch (error) {
          deps.setStatus(error.message);
          paint();
        }
      }

      function setActive(on) {
        if (on && !state.mask && !load()) {
          deps.setStatus("There is no mask to paint on here.");
          return false;
        }
        state.active = Boolean(on);
        canvas.classList.toggle("active", state.active);
        if (bar) {
          bar.hidden = !state.active;
        }
        if (state.active) {
          paint();
        } else {
          reset();
        }
        return state.active;
      }

      if (undoButton) {
        undoButton.addEventListener("click", undo);
      }
      if (saveButton) {
        saveButton.addEventListener("click", () => {
          save();
        });
      }
      // Ctrl+Z while painting steps back one stroke instead of undoing the last review click.
      window.addEventListener("keydown", (event) => {
        if (state.active && (event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
          event.preventDefault();
          event.stopImmediatePropagation();
          undo();
        }
      }, true);

      return {
        get active() {
          return state.active;
        },
        get dirty() {
          return Boolean(state.mask) && BrushCore.changed(state.mask, state.original) > 0;
        },
        setActive,
        toggle: () => setActive(!state.active),
        reset,
        undo,
        repaint: paint,
      };
    },
  };

  return { BrushCore, BrushUI };
});
