/*
 * Gold-standard mode of the gallery page (/gold/): helpers app.js calls when the page runs as the gold review.
 * A "training set" in this mode is one gold set (gold-<name>): a seeded random sample of locations per volume whose masks a
 * person touches up (retrace, brush) and approves. Approving pins the mask's hash; any later edit withdraws the approval.
 * Pure functions plus a little DOM, so the node tests can load this file alone.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.GoldMode = factory();
  }
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const PATH = /^\/gold\/?$/;

  function onGoldPath(pathname) {
    return PATH.test(pathname || "");
  }

  function active(pathname, config) {
    return Boolean(config && config.gold) && PATH.test(pathname || "");
  }

  /** Adds each card's gold facts (edited, changed pixels, approval) to its catalog rows by location id. */
  function attachCards(rows, cards) {
    const byLocation = new Map();
    for (const card of cards || []) {
      byLocation.set(String(card.location_id), card);
    }
    for (const row of rows) {
      const card = byLocation.get(String(row.location_id));
      if (card) {
        row.gold = card;
      }
    }
    return rows;
  }

  /** One line of progress for a volume: "RC1: 37 of 100 approved · 12 edited · 3 rejected · 60 left". */
  function progressLine(volume, progress) {
    if (!progress) {
      return "";
    }
    const parts = [`${volume}: ${progress.approved} of ${progress.target} approved`];
    if (progress.edited) {
      parts.push(`${progress.edited} edited`);
    }
    if (progress.rejected) {
      parts.push(`${progress.rejected} rejected`);
    }
    parts.push(`${progress.remaining} left`);
    return parts.join(" · ");
  }

  /** True when the set has fewer usable cards than its target (a rejection needs a replacement). */
  function needsReplacement(progress) {
    return Boolean(progress) && progress.cards < progress.target;
  }

  function infoLines(card) {
    if (!card) {
      return [];
    }
    const lines = [];
    if (card.edited) {
      lines.push(`edited: ${card.changed_pixels} px changed from the source mask`);
    } else {
      lines.push("not edited");
    }
    if (card.approved_by) {
      lines.push(`approved by ${String(card.approved_by).replace(/[<>&]/g, "")}`);
    }
    return lines;
  }

  function exportHref(set, volume) {
    return `/api/gold/sets/${encodeURIComponent(set)}/export?volume=${encodeURIComponent(volume)}`;
  }

  function applyChrome(doc) {
    const heading = doc.querySelector("header.bar h1");
    if (heading) {
      heading.textContent = "Gold standard review";
    }
    doc.title = "Gold standard review";
    const label = doc.querySelector('label[for="training-set"]');
    if (label) {
      label.textContent = "Gold set";
    }
    const select = doc.getElementById("training-set");
    if (select) {
      select.title = "A gold set: a seeded random sample of locations per volume. Touch up each mask, then approve it.";
    }
    for (const id of ["update-all", "update-visible", "volume-settings"]) {
      const element = doc.getElementById(id);
      if (element) {
        element.hidden = true;
      }
    }
    doc.body.classList.add("gold");
  }

  return { active, onGoldPath, attachCards, progressLine, needsReplacement, infoLines, exportHref, applyChrome };
});
