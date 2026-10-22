/*
 * Converter mode of the gallery page (/converter/): the helpers app.js calls when the page runs as the converter.
 * Pure functions plus a little DOM, so the node tests can load this file on its own.
 *
 * A "training set" in this mode is one built structure (structure-<id>); the card's mask is the proposed polygon and the
 * original outline (the annotation as it is in the database) is drawn over it so a reviewer sees what would change.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.ConverterMode = factory();
  }
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const PATH = /^\/converter\/?$/;

  /** True when this page is the converter: the path says so and the server has it switched on. */
  function active(pathname, config) {
    return Boolean(config && config.converter) && PATH.test(pathname || "");
  }

  function onConverterPath(pathname) {
    return PATH.test(pathname || "");
  }

  /**
   * Adds each card's converter facts to its catalog rows (by location id). `change` is how far the proposal moved from the
   * current outline, 1 - IoU; it goes in `loss` so the gallery's existing "biggest first" sort puts the largest changes up.
   */
  function attachCards(rows, cards) {
    const byLocation = new Map();
    for (const card of cards || []) {
      byLocation.set(String(card.location_id), card);
    }
    for (const row of rows) {
      const card = byLocation.get(String(row.location_id));
      if (!card) {
        continue;
      }
      row.conv = card;
      if (card.iou != null) {
        row.loss = 1 - Number(card.iou);
      }
    }
    return rows;
  }

  /** Edge pixels of a mask: inside pixels with an outside 4-neighbour, `thickness` pixels deep. Returns a Uint8Array of 0/1. */
  function edgePixels(width, height, inside, thickness = 2) {
    const edge = new Uint8Array(width * height);
    const isInside = (x, y) => x >= 0 && y >= 0 && x < width && y < height && inside(x, y);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (!isInside(x, y)) {
          continue;
        }
        let near = false;
        for (let d = 1; d <= thickness && !near; d++) {
          near = !isInside(x + d, y) || !isInside(x - d, y) || !isInside(x, y + d) || !isInside(x, y - d);
        }
        if (near) {
          edge[y * width + x] = 1;
        }
      }
    }
    return edge;
  }

  /** A canvas holding only the outline of a mask image, in `color` ("#rrggbb"). */
  function outlineCanvas(maskImage, color) {
    const width = maskImage.naturalWidth || maskImage.width;
    const height = maskImage.naturalHeight || maskImage.height;
    const probe = document.createElement("canvas");
    probe.width = width;
    probe.height = height;
    const probeContext = probe.getContext("2d", { willReadFrequently: true });
    probeContext.drawImage(maskImage, 0, 0, width, height);
    const pixels = probeContext.getImageData(0, 0, width, height).data;
    const edge = edgePixels(width, height, (x, y) => pixels[(y * width + x) * 4] > 127);
    const out = document.createElement("canvas");
    out.width = width;
    out.height = height;
    const context = out.getContext("2d");
    const image = context.createImageData(width, height);
    const red = parseInt(color.slice(1, 3), 16);
    const green = parseInt(color.slice(3, 5), 16);
    const blue = parseInt(color.slice(5, 7), 16);
    for (let index = 0; index < edge.length; index++) {
      if (edge[index]) {
        image.data[index * 4] = red;
        image.data[index * 4 + 1] = green;
        image.data[index * 4 + 2] = blue;
        image.data[index * 4 + 3] = 255;
      }
    }
    context.putImageData(image, 0, 0);
    return out;
  }

  /** Lines for the viewer's info panel about one card (HTML-safe text only). */
  function infoLines(card) {
    if (!card) {
      return [];
    }
    const lines = [];
    if (card.iou != null) {
      const percent = Math.round(Number(card.iou) * 100);
      lines.push(`overlap with current outline ${percent}%`);
    }
    if (card.model_id) {
      lines.push(`model ${String(card.model_id).replace(/[<>&]/g, "")}`);
    }
    if (card.state === "failed" && card.reason) {
      lines.push(`not built: ${String(card.reason).replace(/[<>&]/g, "")}`);
    }
    return lines;
  }

  /** Words for a build's failure reason, which the server records as "code: detail". */
  function humanReason(reason) {
    const text = String(reason || "");
    const code = text.split(":", 1)[0].trim();
    const detail = text.includes(":") ? text.slice(text.indexOf(":") + 1).trim() : "";
    const known = {
      too_large: "Too big to fit one card at any zoom level",
      no_object: "The segmentation server found no object here",
      empty_seed: "There was nothing to start the segmentation from",
      tile_unavailable: "A tile image was not available",
      RESOURCE_EXHAUSTED: "Beyond the segmentation server's size limits",
      UNAVAILABLE: "The segmentation server was not available",
      DEADLINE_EXCEEDED: "The segmentation server took too long",
      NOT_FOUND: "The segmentation server did not have a tile it needed",
      FAILED_PRECONDITION: "The segmentation server could not use this request",
      INVALID_ARGUMENT: "The segmentation server rejected the request",
    };
    if (known[code]) {
      return known[code];
    }
    if (code === "error") {
      return `Unexpected error: ${detail}`;
    }
    return text.replace(/^failed:\s*/, "") || "Unknown reason";
  }

  /** "RC2 · structure 1968 — 29 of 42 to review" for the build picker. */
  function buildLabel(build) {
    const toReview = Math.max(0, build.built - build.approved - build.rejected);
    const state = build.running ? " (building…)" : "";
    return `${build.volume} · structure ${build.structureId} — ${toReview} of ${build.built} to review${state}`;
  }

  /** "42 built · 10 approved · 3 rejected · 29 to review · 3 failed" */
  function countsLine(build) {
    if (!build) {
      return "";
    }
    const toReview = Math.max(0, build.built - build.approved - build.rejected);
    const parts = [`${build.built} built`, `${build.approved} approved`];
    if (build.rejected) {
      parts.push(`${build.rejected} rejected`);
    }
    parts.push(`${toReview} to review`);
    if (build.failed) {
      parts.push(`${build.failed} failed`);
    }
    return parts.join(" · ");
  }

  /** The volume a preview recommends: the one with annotations (the first when several), else "" . */
  function pickVolume(volumes) {
    const found = Object.entries(volumes || {}).filter(([, info]) => info && info.count > 0);
    return found.length ? found[0][0] : "";
  }

  /** One line under the Structure ID field about what was found where. */
  function previewLine(volumes, chosen) {
    const entries = Object.entries(volumes || {});
    if (!entries.length) {
      return "You do not have review rights on a volume to build in.";
    }
    const found = entries.filter(([, info]) => info && info.count > 0);
    if (!found.length) {
      const failed = entries.filter(([, info]) => info && info.error).map(([name]) => name);
      const where = entries.map(([name]) => name).join(", ");
      return failed.length
        ? `No annotations found. Could not check ${failed.join(", ")}.`
        : `No matching annotations in ${where}. Check the structure ID and the filters below.`;
    }
    const pick = found.find(([name]) => name === chosen) || found[0];
    const others = found.filter(([name]) => name !== pick[0]).map(([name, info]) => `${name} (${info.count})`);
    const base = `${pick[1].count} annotation${pick[1].count === 1 ? "" : "s"} to convert in ${pick[0]}.`;
    return others.length ? `${base} The same id also exists in ${others.join(", ")}: check the volume.` : base;
  }

  function batchHref(set, volume) {
    return `/api/volumes/${encodeURIComponent(volume)}/converter/batch?set=${encodeURIComponent(set)}`;
  }

  /** The labels that differ from the gallery's. */
  function applyChrome(doc) {
    const heading = doc.querySelector("header.bar h1");
    if (heading) {
      heading.textContent = "Annotation converter";
    }
    doc.title = "Annotation converter";
    // The build picker in the bar replaces the training set and volume controls, and the gallery-only filters go.
    for (const id of ["training-set", "volume", "type", "label", "tags", "radius", "export-ignore", "update-all"]) {
      const element = doc.getElementById(id);
      const block = element && element.closest(".filter");
      if (block) {
        block.hidden = true;
      }
    }
    const names = { approved: "Approved", rejected: "Rejected", unknown: "To review" };
    for (const input of doc.querySelectorAll("#view-classes input[data-view]")) {
      const label = input.closest("label");
      if (label && names[input.dataset.view]) {
        label.lastChild.textContent = ` ${names[input.dataset.view]}`;
      }
    }
    const show = doc.querySelector("#view-classes") && doc.querySelector("#view-classes").closest(".filter").querySelector("span");
    if (show) {
      show.textContent = "Show";
      show.title = "To review: not approved or rejected yet. Approved cards go into the batch. Rejected cards are left alone.";
    }
    const bar = doc.getElementById("converter-bar");
    if (bar) {
      bar.hidden = false;
    }
    doc.body.classList.add("converter");
  }

  return {
    active, onConverterPath, attachCards, edgePixels, outlineCanvas, infoLines, batchHref, applyChrome,
    humanReason, buildLabel, countsLine, pickVolume, previewLine,
  };
});
