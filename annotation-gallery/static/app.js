(() => {
  const RGB_INDEX_MAP = [
    [0, 1, 2],
    [1, 0, 2],
    [2, 0, 1],
    [2, 1, 0],
    [1, 2, 0],
    [0, 2, 1],
  ];
  const COMPONENT_LUMA_WEIGHTS = [
    [0.3, 0.59, 0.11],
    [0.59, 0.3, 0.11],
    [0.59, 0.11, 0.3],
    [0.11, 0.59, 0.3],
    [0.11, 0.3, 0.59],
    [0.3, 0.11, 0.59],
  ];
  const HUE_KEY = "galleryOverlayHue";
  const VIS_KEY = "galleryOverlayVisibility";
  // Other annotations in the tile, which the trainer treats as background: shown red in the viewer, yellow where
  // they overlap the target (the trainer resolves that overlap to "unknown").
  const NEG_KEY = "galleryNegativeOverlay";
  const NEG_HEX = "#ef4444";
  const NEG_OVERLAP_HEX = "#f5d90a";
  const NEG_VISIBILITY = 55;
  const SIZE_KEY = "galleryCardSize";
  const COLLAPSE_KEY = "galleryFiltersCollapsed";
  // Last training set, and the volume last opened inside each set.
  const VIEW_KEY = "galleryView";
  // The converter page (/converter/) remembers its own view, so it never overwrites the gallery's.
  const VIEW_STORAGE_KEY = window.ConverterMode && window.ConverterMode.onConverterPath(location.pathname)
    ? "converterView"
    : (window.GoldMode && window.GoldMode.onGoldPath(location.pathname) ? "goldView" : VIEW_KEY);
  const CARD_SIZES = { huge: 1024, large: 512, medium: 384, small: 256 };
  const CACHE_MAX = 80;

  const state = {
    me: null,
    config: { vikingUrl: "", identityMode: "stub" },
    volume: "",
    trainingSet: "current",
    // True on /converter/ when the server has the converter switched on (see converter.js).
    converter: false,
    // True on /gold/ when the server has gold sets switched on (see gold.js).
    gold: false,
    goldProgress: null,
    noStructures: false,
    rows: [],
    permission: "read",
    hasSam2: false,
    // True when some card has a card score (the scorer service's IoU, core recall and spill; see cardScoreHtml).
    hasCardScores: false,
    sorts: [{ key: "z", desc: false }],
    // The converter's location links ([a, b] pairs) and a counter that changes when the layout must be redone.
    convLinks: [],
    layoutEpoch: 0,
    tagIds: null,
    filterNote: "",
    cropSize: 1024,
    cardSize: "large",
    settings: null,
  };

  const locationGenerations = new Map();

  const els = {
    who: document.getElementById("who"),
    logout: document.getElementById("logout"),
    trainingSet: document.getElementById("training-set"),
    volume: document.getElementById("volume"),
    type: document.getElementById("type"),
    label: document.getElementById("label"),
    tags: document.getElementById("tags"),
    z: document.getElementById("z"),
    radius: document.getElementById("radius"),
    locate: document.getElementById("locate"),
    sortStack: document.getElementById("sort-stack"),
    hue: document.getElementById("hue"),
    visibility: document.getElementById("visibility"),
    visibilityValue: document.getElementById("visibility-value"),
    status: document.getElementById("status"),
    scroller: document.getElementById("scroller"),
    spacer: document.getElementById("spacer"),
    grid: document.getElementById("grid"),
    filters: document.getElementById("filters"),
    filtersToggle: document.getElementById("filters-toggle"),
    filterSummary: document.getElementById("filter-summary"),
    viewer: document.getElementById("viewer"),
    viewerStage: document.getElementById("viewer-stage"),
    viewerFrame: document.getElementById("viewer-frame"),
    viewerImage: document.getElementById("viewer-image"),
    viewerMask: document.getElementById("viewer-mask"),
    viewerGrid: document.getElementById("viewer-grid"),
    viewerBack: document.getElementById("viewer-back"),
    viewerClose: document.getElementById("viewer-close"),
    viewerCues: document.getElementById("viewer-cues"),
  };

  const GRID_GAP = 4;
  // Dark, low-saturation tones. Same location shares one; neighbors take another.
  const LINK_TONES = [
    [215, 18, 11],
    [198, 20, 12],
    [172, 16, 11],
    [142, 14, 11],
    [88, 12, 12],
    [42, 16, 12],
    [18, 18, 12],
    [350, 14, 12],
    [322, 14, 11],
    [276, 16, 12],
    [246, 14, 11],
    [230, 10, 14],
  ];

  function linkToneIndex(locationId, used) {
    const palette = LINK_TONES.length;
    const hashed = Math.abs(Math.imul(Number(locationId) | 0, 2654435761) >>> 0) % palette;
    if (!used.has(hashed)) {
      return hashed;
    }
    for (let step = 1; step < palette; step += 1) {
      const candidate = (hashed + step) % palette;
      if (!used.has(candidate)) {
        return candidate;
      }
    }
    return hashed;
  }

  // What a cell's ring and tone belong to: its process in the process layout, else its location.
  function cellGroup(cell) {
    return cell.group != null && cell.group >= 0 ? 1e12 + cell.group : Number(cell.row.location_id);
  }

  function assignLinkHues(cells) {
    const at = new Map();
    for (const cell of cells) {
      at.set(`${cell.gr},${cell.gc}`, cellGroup(cell));
    }
    const neighbors = new Map();
    const dirs = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]];
    const order = [];
    const seen = new Set();
    for (const cell of cells) {
      const id = cellGroup(cell);
      if (!neighbors.has(id)) {
        neighbors.set(id, new Set());
      }
      if (!seen.has(id)) {
        seen.add(id);
        order.push(id);
      }
      for (const [dr, dc] of dirs) {
        const other = at.get(`${cell.gr + dr},${cell.gc + dc}`);
        if (other != null && other !== id) {
          neighbors.get(id).add(other);
        }
      }
    }
    const assigned = new Map();
    for (const id of order) {
      const used = new Set();
      for (const other of neighbors.get(id) || []) {
        if (assigned.has(other)) {
          used.add(assigned.get(other));
        }
      }
      assigned.set(id, linkToneIndex(id, used));
    }
    for (const cell of cells) {
      const tone = LINK_TONES[assigned.get(cellGroup(cell))];
      cell.linkH = tone[0];
      cell.linkS = tone[1];
      cell.linkL = tone[2];
    }
  }

  // A shared grid edge of one location stays open so the ring is the group's perimeter.
  function assignJoins(cells) {
    const at = new Map();
    for (const cell of cells) {
      at.set(`${cell.gr},${cell.gc}`, cellGroup(cell));
    }
    for (const cell of cells) {
      const id = cellGroup(cell);
      cell.joinN = at.get(`${cell.gr - 1},${cell.gc}`) === id;
      cell.joinE = at.get(`${cell.gr},${cell.gc + 1}`) === id;
      cell.joinS = at.get(`${cell.gr + 1},${cell.gc}`) === id;
      cell.joinW = at.get(`${cell.gr},${cell.gc - 1}`) === id;
    }
  }

  function joinClasses(cell) {
    const names = [];
    if (cell.joinN) {
      names.push("join-n");
    }
    if (cell.joinE) {
      names.push("join-e");
    }
    if (cell.joinS) {
      names.push("join-s");
    }
    if (cell.joinW) {
      names.push("join-w");
    }
    if (!names.length) {
      names.push("solo");
    }
    return names.join(" ");
  }
  let rowStride = 0;
  let windowTop = -1;
  let packMemo = { packed: null, repackFrom: null };
  let catalogEpoch = 0;
  let tagEpoch = 0;
  const catalogIndex = {
    byLocation: new Map(),
    visibleCards: [],
    cardAt: new Map(),
    stamp: "",
  };
  let windowBottom = -1;
  let measuringRow = false;
  const overlayCache = new Map();
  let paintToken = 0;
  let renderTimer = 0;
  const viewer = {
    token: 0,
    image: null,
    overlay: null,
    partIndex: null,
    parts: null,
    layoutWidth: 0,
    layoutHeight: 0,
    hue: "#3478cb",
    visibility: 30,
    // { count, base, overlap } for the open location: `base` is the tile with the negatives tinted, `overlap`
    // marks where they cover the target. Null until loaded, or when the tile has no other annotation.
    negatives: null,
    // The open location's mask PNG, kept so the pen retrace can trace its outline.
    mask: null,
    // Converter: the current (database) outline of the open location as an outline-only canvas, drawn over the proposal.
    original: null,
  };

  // The open card's mask as pixels: { width, height, inside(x, y) }, or null (no mask, or the split-parts view).
  function currentMask() {
    if (!viewer.mask || viewer.parts) {
      return null;
    }
    const width = viewer.mask.naturalWidth || viewer.mask.width;
    const height = viewer.mask.naturalHeight || viewer.mask.height;
    const probe = document.createElement("canvas");
    probe.width = width;
    probe.height = height;
    const context = probe.getContext("2d", { willReadFrequently: true });
    context.drawImage(viewer.mask, 0, 0, width, height);
    const pixels = context.getImageData(0, 0, width, height).data;
    return { width, height, inside: (x, y) => pixels[(y * width + x) * 4] > 127 };
  }

  // Pen retrace (retrace.js / retrace_ui.js). It works on the whole-tile view only, never on the split parts.
  const retraceButton = document.getElementById("viewer-retrace-btn");
  const retraceUi = window.RetraceUI
    ? window.RetraceUI.create({
        frame: els.viewerFrame,
        maskCanvas: els.viewerMask,
        getHue: () => viewer.hue,
        persists: () => state.converter || state.gold,
        onChange: (polygon) => {
          if (state.converter || state.gold) {
            saveRetrace(polygon).catch((error) => {
              els.status.textContent = error.message;
            });
          }
        },
        setStatus: (text) => {
          els.status.textContent = text;
        },
        getMask: () => currentMask(),
      })
    : null;

  // Converter: the retraced outline replaces the card's proposal (and so the batch); the database is not written.
  function saveRetrace(polygon) {
    return saveEdit({ rings: [polygon.exterior, ...polygon.holes] }, "Retrace");
  }

  // Gold: the painted mask replaces the card's mask.
  function saveBrush(dataUrl) {
    return saveEdit({ png: dataUrl }, "Painted mask");
  }

  async function saveEdit(payload, label) {
    const locationId = viewer.locationId;
    const imageKey = viewer.imageKey;
    const volume = state.volume;
    const editPath = state.gold ? "gold/edit" : "converter/edit";
    const result = await getJson(withSet(`/api/volumes/${encodeURIComponent(volume)}/${editPath}`), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ location_id: Number(locationId), image_key: imageKey, ...payload }),
    });
    const row = viewerRow();
    if (row && state.gold) {
      row.gold = { ...(row.gold || {}), edited: result.edited ? 1 : 0, changed_pixels: result.changed_pixels, approved_by: null };
      if (result.unapproved) {
        row.approved = 0;
      }
    } else if (row) {
      row.loss = 1 - Number(result.iou);
      row.conv = { ...(row.conv || {}), iou: result.iou, reason: "retraced" };
    }
    if (locationId !== viewer.locationId || volume !== state.volume) {
      return;
    }
    // Show the stored mask: the outline just drawn, now as the card's mask.
    const mask = await loadImage(`${fileUrl(volume, `masks/${imageKey}_${locationId}.png`)}&t=${Date.now()}`);
    if (locationId !== viewer.locationId) {
      return;
    }
    retraceUi.reset();
    if (brushUi) {
      brushUi.reset();
    }
    syncRetraceButton();
    viewer.mask = mask;
    viewer.overlay = composeOverlay(viewer.image, mask, viewer.hue, viewer.visibility);
    layerEpoch += 1;
    resetVisibleCards();
    paintViewer();
    placeViewerInfo();
    els.status.textContent = state.gold
      ? `${label} saved to the gold card (${result.changed_pixels} px differ from the source mask).${result.unapproved ? " Its approval was withdrawn: approve it again." : ""}`
      : `${label} saved to the converter batch (${Math.round(Number(result.iou) * 100)}% overlap with the current outline). The database is not changed.`;
    if (state.gold) {
      refreshGold(volume, state.trainingSet, () => true).catch(() => {});
    }
  }

  const brushButton = document.getElementById("viewer-brush-btn");
  const brushUi = window.BrushUI && document.getElementById("brush-bar")
    ? window.BrushUI.create({
        frame: els.viewerFrame,
        maskCanvas: els.viewerMask,
        getHue: () => viewer.hue,
        setStatus: (text) => {
          els.status.textContent = text;
        },
        getMask: () => currentMask(),
        onSave: (dataUrl) => saveBrush(dataUrl),
      })
    : null;

  function syncBrushButton() {
    if (brushButton && brushUi) {
      brushButton.classList.toggle("on", brushUi.active);
      brushButton.setAttribute("aria-pressed", String(brushUi.active));
    }
  }

  function toggleBrush() {
    if (!brushUi || !state.gold) {
      return;
    }
    if (!brushUi.active && retraceUi && retraceUi.active) {
      retraceUi.setActive(false);
      syncRetraceButton();
    }
    brushUi.toggle();
    syncBrushButton();
  }

  // Gold: card facts, progress, the export link and whether rejected cards need a replacement.
  async function refreshGold(volume, trainingSet, isLatest) {
    try {
      const [cards, progress] = await Promise.all([
        getJson(withSet(`/api/volumes/${encodeURIComponent(volume)}/gold/cards`)),
        getJson(`/api/gold/sets/${encodeURIComponent(trainingSet)}/progress`),
      ]);
      if (!isLatest()) {
        return;
      }
      window.GoldMode.attachCards(state.rows, cards.cards);
      state.goldProgress = (progress.progress || {})[volume] || null;
    } catch {
      state.goldProgress = null;
    }
    document.getElementById("gold-progress").textContent = window.GoldMode.progressLine(volume, state.goldProgress);
    const link = document.getElementById("gold-export");
    link.href = window.GoldMode.exportHref(trainingSet, volume);
    link.hidden = false;
    document.getElementById("gold-replace").hidden = !window.GoldMode.needsReplacement(state.goldProgress);
  }

  // Gold dialogs: draw a sample, open a list of ids, draw replacements.
  const goldUi = {
    dialog: document.getElementById("gold-dialog"),
    listDialog: document.getElementById("gold-list-dialog"),
    file: document.getElementById("gold-list-file"),
    listText: "",
  };

  async function goldCreate() {
    const files = [...document.getElementById("gold-lists").files];
    const lists = {};
    for (const file of files) {
      lists[file.name.replace(/\.[^.]*$/, "")] = await file.text();
    }
    const body = {
      ...(files.length ? { lists } : {}),
      name: document.getElementById("gold-name").value.trim(),
      sourceSet: document.getElementById("gold-source").value || "current",
      n: document.getElementById("gold-n").value,
      seed: document.getElementById("gold-seed").value,
      stratify: document.getElementById("gold-stratify").value,
    };
    const button = document.getElementById("gold-start");
    button.disabled = true;
    try {
      const created = await getJson("/api/gold/sets", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      goldUi.dialog.hidden = true;
      const sets = await getJson("/api/training-sets?app=gold");
      els.trainingSet.innerHTML = (sets.sets || []).map((name) => `<option value="${name}">${name}</option>`).join("");
      state.trainingSet = created.set;
      els.trainingSet.value = created.set;
      document.getElementById("filters").hidden = false;
      rememberView(true);
      els.status.textContent = files.length
        ? `Made ${created.set} from ${files.length} list file(s).${skippedSummary(created.meta)}`
        : `Drew ${created.set} with seed ${created.meta.seed}.`;
      await loadVolumes();
    } catch (error) {
      els.status.textContent = error.message;
    } finally {
      button.disabled = false;
    }
  }

  // A short note of ids a list could not use ("RC1: 2 not in source"), or nothing.
  function skippedSummary(meta) {
    const parts = [];
    for (const [volume, info] of Object.entries(meta.volumes || {})) {
      for (const [why, ids] of Object.entries(info.skipped || {})) {
        parts.push(`${volume}: ${ids.length} ${why}`);
      }
    }
    return parts.length ? ` Not used: ${parts.join("; ")}.` : "";
  }

  async function goldApplyList() {
    const mode = document.getElementById("gold-list-mode").value;
    const send = (force) => getJson(`/api/gold/sets/${encodeURIComponent(state.trainingSet)}/list`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ volume: state.volume, text: goldUi.listText, mode, force }),
    });
    let result;
    try {
      try {
        result = await send(false);
      } catch (error) {
        if (!/approved/.test(error.message) || !window.confirm(`${error.message}. Replace anyway?`)) {
          throw error;
        }
        result = await send(true);
      }
    } catch (error) {
      els.status.textContent = error.message;
      return;
    }
    goldUi.listDialog.hidden = true;
    const skipped = Object.entries(result.skipped || {}).map(([why, ids]) => `${ids.length} ${why}`);
    els.status.textContent = `Added ${result.added.length} location(s); now ${result.cards} cards.${skipped.length ? ` Not used: ${skipped.join(", ")}.` : ""}${(result.notIds || []).length ? ` Ignored text: ${result.notIds.join(" ")}.` : ""}`;
    await loadCatalog(false, true);
  }

  async function goldReplacements() {
    const missing = state.goldProgress ? state.goldProgress.target - state.goldProgress.cards : 1;
    try {
      const result = await getJson(`/api/gold/sets/${encodeURIComponent(state.trainingSet)}/replacements`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ volume: state.volume, count: Math.max(1, missing) }),
      });
      els.status.textContent = `Added ${result.added.length} replacement location(s) from the seeded order.`;
      await loadCatalog(false, true);
    } catch (error) {
      els.status.textContent = error.message;
    }
  }

  if (document.getElementById("gold-draw")) {
    document.getElementById("gold-draw").addEventListener("click", async () => {
      try {
        const sets = await getJson("/api/training-sets");
        document.getElementById("gold-source").innerHTML = (sets.sets || ["current"])
          .map((name) => `<option value="${name}"${name === "current" ? " selected" : ""}>${name}</option>`)
          .join("");
      } catch {
        document.getElementById("gold-source").innerHTML = '<option value="current">current</option>';
      }
      goldUi.dialog.hidden = false;
      document.getElementById("gold-name").focus();
    });
    document.getElementById("gold-start").addEventListener("click", () => {
      goldCreate();
    });
    document.getElementById("gold-cancel").addEventListener("click", () => {
      goldUi.dialog.hidden = true;
    });
    document.getElementById("gold-list").addEventListener("click", () => {
      if (!state.volume) {
        els.status.textContent = "Open a gold set and pick a volume first.";
        return;
      }
      goldUi.file.value = "";
      goldUi.file.click();
    });
    goldUi.file.addEventListener("change", async () => {
      const file = goldUi.file.files && goldUi.file.files[0];
      if (!file) {
        return;
      }
      goldUi.listText = await file.text();
      document.getElementById("gold-list-file-name").textContent = `${file.name} for ${state.volume}`;
      goldUi.listDialog.hidden = false;
    });
    document.getElementById("gold-list-start").addEventListener("click", () => {
      goldApplyList();
    });
    document.getElementById("gold-list-cancel").addEventListener("click", () => {
      goldUi.listDialog.hidden = true;
    });
    document.getElementById("gold-replace").addEventListener("click", () => {
      goldReplacements();
    });
  }

  function syncRetraceButton() {
    if (retraceButton && retraceUi) {
      retraceButton.classList.toggle("on", retraceUi.active);
      retraceButton.setAttribute("aria-pressed", String(retraceUi.active));
    }
  }

  function toggleRetrace() {
    if (retraceUi) {
      if (!retraceUi.active && brushUi && brushUi.active) {
        brushUi.setActive(false);
        syncBrushButton();
      }
      retraceUi.toggle();
      syncRetraceButton();
    }
  }

  function rgbToHcl(red255, green255, blue255) {
    const red = red255 / 255;
    const green = green255 / 255;
    const blue = blue255 / 255;
    const maximum = Math.max(red, green, blue);
    const minimum = Math.min(red, green, blue);
    const chroma = maximum - minimum;
    const luma = 0.3 * red + 0.59 * green + 0.11 * blue;
    let hue = 0;
    if (chroma > 0) {
      if (maximum === red) {
        hue = (green - blue) / chroma / 6;
        if (hue < 0) {
          hue += 1;
        }
      } else if (maximum === green) {
        hue = ((blue - red) / chroma + 2) / 6;
      } else {
        hue = ((red - green) / chroma + 4) / 6;
      }
    }
    return { hue, chroma, luma };
  }

  function hueHextant(hue) {
    const hPrime = (hue % 1) * 6;
    const hextant = Math.trunc(hPrime) % 6;
    const fDescend = hPrime % 2;
    const slopeCoeff = 1 - Math.abs(fDescend - 1);
    return { hextant, slopeCoeff };
  }

  function maxChromaAtLuma(hextant, slopeCoeff, luma) {
    if (luma <= 0.001 || luma >= 0.999) {
      return 0;
    }
    const weights = COMPONENT_LUMA_WEIGHTS[hextant];
    const chromaLumaCoeff = weights[0] + weights[1] * slopeCoeff;
    const upper = (1 - luma) / Math.max(1 - chromaLumaCoeff, 1e-6);
    const lower = luma / Math.max(chromaLumaCoeff, 1e-6);
    return Math.max(Math.min(upper, lower), 0);
  }

  function correctLuma(hextant, c0, c1, c2, luma) {
    const weights = COMPONENT_LUMA_WEIGHTS[hextant];
    const invRg = 1 / (weights[0] + weights[1]);
    const dot = (a, b, c) => a * weights[0] + b * weights[1] + c * weights[2];
    const delta = luma - dot(c0, c1, c2);
    const f0 = c0 + delta;
    const f1 = c1 + delta;
    const f2 = c2 + delta;
    if (f0 >= 0 && f0 <= 1) {
      return [f0, f1, f2];
    }
    const s0 = Math.min(1, Math.max(0, f0));
    let s1 = Math.min(1, Math.max(0, f1));
    let s2 = f2;
    const spill = (luma - dot(s0, s1, s2)) * invRg;
    s1 += spill;
    s2 += spill;
    if (s1 >= 0 && s1 <= 1) {
      return [s0, s1, s2];
    }
    const t1 = Math.min(1, Math.max(0, s1));
    let t2 = Math.min(1, Math.max(0, s2));
    t2 += (luma - dot(s0, t1, t2)) * invRg;
    return [s0, t1, t2];
  }

  function hclToRgb(hextant, slopeCoeff, chroma, luma) {
    const corrected = correctLuma(hextant, chroma, chroma * slopeCoeff, 0, luma);
    const [redIndex, greenIndex, blueIndex] = RGB_INDEX_MAP[hextant];
    return [corrected[redIndex], corrected[greenIndex], corrected[blueIndex]];
  }

  function hexToRgb(hex) {
    const value = hex.replace("#", "");
    return {
      r: parseInt(value.slice(0, 2), 16),
      g: parseInt(value.slice(2, 4), 16),
      b: parseInt(value.slice(4, 6), 16),
    };
  }

  function imageSize(image) {
    return {
      width: image.naturalWidth || image.width,
      height: image.naturalHeight || image.height,
    };
  }

  function greyPlate(width, height) {
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d");
    context.fillStyle = "#808080";
    context.fillRect(0, 0, width, height);
    return canvas;
  }

  function imageHasSignal(image) {
    const probe = document.createElement("canvas");
    probe.width = 16;
    probe.height = 16;
    const context = probe.getContext("2d", { willReadFrequently: true });
    context.drawImage(image, 0, 0, 16, 16);
    const pixels = context.getImageData(0, 0, 16, 16).data;
    for (let index = 0; index < pixels.length; index += 4) {
      if (pixels[index] > 0 || pixels[index + 1] > 0 || pixels[index + 2] > 0) {
        return true;
      }
    }
    return false;
  }

  function composeOverlay(image, mask, hex, visibility) {
    const { width, height } = imageSize(image);
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    const maskCanvas = document.createElement("canvas");
    maskCanvas.width = width;
    maskCanvas.height = height;
    const maskContext = maskCanvas.getContext("2d", { willReadFrequently: true });
    maskContext.drawImage(mask, 0, 0, width, height);
    const maskPixels = maskContext.getImageData(0, 0, width, height).data;
    context.drawImage(image, 0, 0, width, height);
    const imageData = context.getImageData(0, 0, width, height);
    const pixels = imageData.data;
    tintMaskedPixels(pixels, maskPixels, hex, visibility, false);
    context.putImageData(imageData, 0, 0);
    return canvas;
  }

  function tintMaskedPixels(pixels, maskPixels, hex, visibility, keepOutside) {
    const color = hexToRgb(hex);
    const { hue, chroma } = rgbToHcl(color.r, color.g, color.b);
    const { hextant, slopeCoeff } = hueHextant(hue);
    const alpha = Math.min(1, Math.max(0, visibility / 100));
    for (let index = 0; index < pixels.length; index += 4) {
      if (maskPixels[index] <= 127) {
        if (!keepOutside) {
          pixels[index + 3] = 0;
        }
        continue;
      }
      const red = pixels[index] / 255;
      const green = pixels[index + 1] / 255;
      const blue = pixels[index + 2] / 255;
      const luma = 0.3 * red + 0.59 * green + 0.11 * blue;
      const limited = Math.min(chroma, maxChromaAtLuma(hextant, slopeCoeff, luma));
      const tinted = hclToRgb(hextant, slopeCoeff, limited, luma);
      pixels[index] = Math.round(((1 - alpha) * red + alpha * tinted[0]) * 255);
      pixels[index + 1] = Math.round(((1 - alpha) * green + alpha * tinted[1]) * 255);
      pixels[index + 2] = Math.round(((1 - alpha) * blue + alpha * tinted[2]) * 255);
      pixels[index + 3] = 255;
    }
  }

  function glyphMask(width, height, text) {
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d");
    const cssSize = cardPixelSize();
    context.fillStyle = "#fff";
    context.font = `700 ${Math.round(112 * (width / cssSize))}px "Segoe UI", sans-serif`;
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillText(text, width / 2, height / 2);
    return canvas;
  }

  function stampStatus(image, status) {
    const { width, height } = imageSize(image);
    const mark = status === "rejected" ? "×" : "✓";
    const hex = status === "rejected" ? "#ef4444" : "#3dd68c";
    const colored = composeOverlay(greyPlate(width, height), glyphMask(width, height, mark), hex, 100);
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d");
    context.drawImage(image, 0, 0, width, height);
    context.globalAlpha = 0.5;
    context.drawImage(colored, 0, 0);
    return canvas;
  }

  function cacheGet(key) {
    const hit = overlayCache.get(key);
    if (!hit) {
      return null;
    }
    overlayCache.delete(key);
    overlayCache.set(key, hit);
    return hit;
  }

  function cacheSet(key, canvas) {
    if (overlayCache.has(key)) {
      overlayCache.delete(key);
    }
    overlayCache.set(key, canvas);
    while (overlayCache.size > CACHE_MAX) {
      overlayCache.delete(overlayCache.keys().next().value);
    }
  }

  function blit(target, source) {
    target.width = source.width;
    target.height = source.height;
    target.getContext("2d").drawImage(source, 0, 0);
  }

  function loadImage(url) {
    return new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve(image);
      image.onerror = () => reject(new Error(url));
      image.src = url;
    });
  }

  // Bumped whenever a review or refresh can change what the other cards in a tile show as negatives, so cached
  // composites built from older card layers are not reused.
  let layerEpoch = 0;

  // Every card image (photo, mask, card layer) goes through one bounded queue, so the browser is never handed
  // dozens of requests in DOM order. Pending jobs are ordered by where their card is: the cards on screen first,
  // top row to bottom and left to right, then the pre-rendered rows nearest the screen. The order is recomputed
  // whenever the view changes (new render, sort, filter, scroll), so a card that scrolled away waits and one that
  // scrolled into view jumps ahead. A job whose card has gone (or whose paint was superseded) is dropped unrun.
  const LOAD_CONCURRENCY = 8;
  const loadQueue = [];
  let loadActive = 0;
  let loadSeq = 0;
  let reprioritizeTimer = 0;

  function cardPriority(node, scrollerRect) {
    const rect = node.getBoundingClientRect();
    const top = rect.top - scrollerRect.top;
    const left = Math.round(rect.left);
    if (top + rect.height > 0 && top < scrollerRect.height) {
      return [0, Math.round(Math.max(top, 0) / 8), left];
    }
    const away = top < 0 ? -(top + rect.height) : top - scrollerRect.height;
    return [1, Math.round(away / 8), left];
  }

  function priorityBefore(a, b) {
    for (let index = 0; index < 3; index += 1) {
      if (a[index] !== b[index]) {
        return a[index] - b[index];
      }
    }
    return 0;
  }

  function reprioritize() {
    window.clearTimeout(reprioritizeTimer);
    for (let index = loadQueue.length - 1; index >= 0; index -= 1) {
      if (!loadQueue[index].isLive()) {
        loadQueue.splice(index, 1)[0].reject(new Error("stale"));
      }
    }
    if (loadQueue.length < 2) {
      return;
    }
    const scrollerRect = els.scroller.getBoundingClientRect();
    const byNode = new Map();
    for (const job of loadQueue) {
      if (!byNode.has(job.node)) {
        byNode.set(job.node, job.node.isConnected ? cardPriority(job.node, scrollerRect) : [2, 0, 0]);
      }
      job.priority = byNode.get(job.node);
    }
    loadQueue.sort((a, b) => priorityBefore(a.priority, b.priority) || a.seq - b.seq);
  }

  function scheduleReprioritize() {
    window.clearTimeout(reprioritizeTimer);
    reprioritizeTimer = window.setTimeout(reprioritize, 60);
  }

  function queuedImage(url, node, isLive) {
    return new Promise((resolve, reject) => {
      loadQueue.push({ url, node, isLive, resolve, reject, seq: loadSeq += 1, priority: [1, 0, 0] });
      pumpLoads();
    });
  }

  function pumpLoads() {
    while (loadActive < LOAD_CONCURRENCY && loadQueue.length) {
      const job = loadQueue.shift();
      if (!job.isLive()) {
        job.reject(new Error("stale"));
        continue;
      }
      loadActive += 1;
      loadImage(job.url).then(job.resolve, job.reject).finally(() => {
        loadActive -= 1;
        pumpLoads();
      });
    }
  }

  // The <img> under the canvas gets its source only when the card's turn comes, not when the card is built.
  function showPhoto(node, photo) {
    if (photo && photo.getAttribute("src") !== node.dataset.image) {
      photo.src = node.dataset.image;
    }
  }

  // Blends a pixel toward `hex` the way tintMaskedPixels does and writes the result (0-1 floats) to `out`.
  function makeTinter(hex, visibility) {
    const color = hexToRgb(hex);
    const { hue, chroma } = rgbToHcl(color.r, color.g, color.b);
    const { hextant, slopeCoeff } = hueHextant(hue);
    const alpha = Math.min(1, Math.max(0, visibility / 100));
    return (red, green, blue, out) => {
      const luma = 0.3 * red + 0.59 * green + 0.11 * blue;
      const limited = Math.min(chroma, maxChromaAtLuma(hextant, slopeCoeff, luma));
      const tinted = hclToRgb(hextant, slopeCoeff, limited, luma);
      out[0] = (1 - alpha) * red + alpha * tinted[0];
      out[1] = (1 - alpha) * green + alpha * tinted[1];
      out[2] = (1 - alpha) * blue + alpha * tinted[2];
    };
  }

  // Draws the tissue with the card layer on top: R is the target (hue), G the other annotations (red), B where they
  // overlap (yellow). Channel values are soft alpha, so a downsampled edge stays smooth. Pixels no channel covers
  // are transparent, so the photo underneath shows through, as with the full-size mask path.
  function composeLayered(image, layer, hex, visibility, withNegatives) {
    const { width, height } = imageSize(image);
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    const layerCanvas = document.createElement("canvas");
    layerCanvas.width = width;
    layerCanvas.height = height;
    const layerContext = layerCanvas.getContext("2d", { willReadFrequently: true });
    layerContext.drawImage(layer, 0, 0, width, height);
    const planes = layerContext.getImageData(0, 0, width, height).data;
    context.drawImage(image, 0, 0, width, height);
    const imageData = context.getImageData(0, 0, width, height);
    const pixels = imageData.data;
    const tintTarget = makeTinter(hex, visibility);
    const tintNegative = makeTinter(NEG_HEX, NEG_VISIBILITY);
    const tintOverlap = makeTinter(NEG_OVERLAP_HEX, NEG_VISIBILITY);
    const out = [0, 0, 0];
    for (let index = 0; index < pixels.length; index += 4) {
      const own = planes[index] / 255;
      const other = withNegatives ? planes[index + 1] / 255 : 0;
      const both = withNegatives ? planes[index + 2] / 255 : 0;
      const aOverlap = both;
      const aTarget = withNegatives ? Math.max(own - both, 0) : own;
      const aNegative = withNegatives ? Math.max(other - both, 0) : 0;
      const total = aOverlap + aTarget + aNegative;
      if (total <= 0.004) {
        pixels[index + 3] = 0;
        continue;
      }
      const red = pixels[index] / 255;
      const green = pixels[index + 1] / 255;
      const blue = pixels[index + 2] / 255;
      let r = 0;
      let g = 0;
      let b = 0;
      if (aTarget > 0) {
        tintTarget(red, green, blue, out);
        r += aTarget * out[0];
        g += aTarget * out[1];
        b += aTarget * out[2];
      }
      if (aNegative > 0) {
        tintNegative(red, green, blue, out);
        r += aNegative * out[0];
        g += aNegative * out[1];
        b += aNegative * out[2];
      }
      if (aOverlap > 0) {
        tintOverlap(red, green, blue, out);
        r += aOverlap * out[0];
        g += aOverlap * out[1];
        b += aOverlap * out[2];
      }
      pixels[index] = Math.round((r / total) * 255);
      pixels[index + 1] = Math.round((g / total) * 255);
      pixels[index + 2] = Math.round((b / total) * 255);
      pixels[index + 3] = Math.round(Math.min(1, total) * 255);
    }
    context.putImageData(imageData, 0, 0);
    return canvas;
  }

  // Cards below Huge: one request for the target mask and its negatives at the thumbnail's size.
  async function paintLayered(node, token, key, canvas, photo) {
    const hue = node.dataset.hue;
    const visibility = Number(node.dataset.visibility);
    const withNegatives = negativesOn();
    let image = null;
    const live = () => node.isConnected && token === paintToken;
    const imageTask = queuedImage(node.dataset.image, node, live).then((loaded) => {
      image = loaded;
      showPhoto(node, photo);
    }).catch(() => null);
    let layer;
    try {
      layer = await queuedImage(node.dataset.layer, node, live);
    } catch {
      await imageTask;
      if (!live()) {
        return;
      }
      setMediaState(node, image);
      if (image && imageHasSignal(image) && photo) {
        photo.classList.remove("pending");
      }
      return;
    }
    if (!live()) {
      return;
    }
    const plate = greyPlate(imageSize(layer).width, imageSize(layer).height);
    blit(canvas, composeLayered(plate, layer, hue, visibility, withNegatives));
    await imageTask;
    if (!live()) {
      return;
    }
    const tissue = image && imageHasSignal(image) ? image : plate;
    let composed = composeLayered(tissue, layer, hue, visibility, withNegatives);
    if (node.dataset.status === "rejected" || node.dataset.status === "approved") {
      composed = stampStatus(composed, node.dataset.status);
    }
    composed.photoVisible = tissue !== plate;
    composed.missing = !image;
    setMediaState(node, image);
    if (photo) {
      photo.classList.toggle("pending", !composed.photoVisible);
    }
    if (tissue !== plate) {
      blit(canvas, composed);
    }
    cacheSet(key, composed);
  }

  async function paintComposite(node, token) {
    const layered = Boolean(node.dataset.layer);
    const layerPart = layered ? `|L${layerEpoch}${negativesOn() ? "n" : ""}` : "";
    const key = `${node.dataset.loc}|${node.dataset.key}|${node.dataset.size}|${node.dataset.hue}|${node.dataset.visibility}|${node.dataset.status || ""}|${node.dataset.gen || ""}${layerPart}`;
    const canvas = node.querySelector("canvas.tint");
    const photo = node.querySelector("img.jpeg");
    const cached = cacheGet(key);
    if (cached) {
      if (node.isConnected && token === paintToken) {
        blit(canvas, cached);
        if (photo) {
          showPhoto(node, photo);
          photo.classList.toggle("pending", !cached.photoVisible);
        }
        setMediaState(node, cached.missing ? null : true);
      }
      return;
    }
    if (layered) {
      await paintLayered(node, token, key, canvas, photo);
      return;
    }
    const hue = node.dataset.hue;
    const visibility = Number(node.dataset.visibility);
    let image = null;
    const live = () => node.isConnected && token === paintToken;
    const imageTask = queuedImage(node.dataset.image, node, live).then((loaded) => {
      image = loaded;
      showPhoto(node, photo);
    }).catch(() => null);
    let mask;
    try {
      mask = await queuedImage(node.dataset.mask, node, live);
    } catch {
      await imageTask;
      if (!node.isConnected || token !== paintToken) {
        return;
      }
      setMediaState(node, image);
      if (image && imageHasSignal(image) && photo) {
        photo.classList.remove("pending");
      }
      return;
    }
    if (!node.isConnected || token !== paintToken) {
      return;
    }
    const plate = greyPlate(imageSize(mask).width, imageSize(mask).height);
    const preview = composeOverlay(plate, mask, hue, visibility);
    blit(canvas, preview);
    await imageTask;
    if (!node.isConnected || token !== paintToken) {
      return;
    }
    const tissue = image && imageHasSignal(image) ? image : plate;
    let composed = tissue === plate ? preview : composeOverlay(tissue, mask, hue, visibility);
    if (node.dataset.status === "rejected" || node.dataset.status === "approved") {
      composed = stampStatus(composed, node.dataset.status);
    }
    composed.photoVisible = tissue !== plate;
    composed.missing = !image;
    setMediaState(node, image);
    if (photo) {
      photo.classList.toggle("pending", !composed.photoVisible);
    }
    if (tissue !== plate) {
      blit(canvas, composed);
    }
    cacheSet(key, composed);
  }

  function setMediaState(node, image) {
    node.classList.remove("waiting", "missing");
    if (!image) {
      node.classList.add("missing");
    }
  }

  function paintVisible() {
    const token = ++paintToken;
    els.grid.querySelectorAll(".composite").forEach((node) => {
      paintComposite(node, token);
    });
    reprioritize();
  }

  async function getJson(url, options) {
    const response = await fetch(url, options);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      const error = new Error(payload.error || response.statusText);
      error.status = response.status;
      throw error;
    }
    return payload;
  }

  function withSet(url) {
    const set = state.trainingSet || "current";
    const join = url.includes("?") ? "&" : "?";
    return `${url}${join}set=${encodeURIComponent(set)}`;
  }

  function fileUrl(volume, rel) {
    return withSet(`/api/volumes/${encodeURIComponent(volume)}/file?path=${encodeURIComponent(rel)}`);
  }

  function locationGen(locationId) {
    return locationGenerations.get(Number(locationId)) || 0;
  }

  function mediaUrl(volume, rel, locationId) {
    const url = fileUrl(volume, rel);
    const gen = locationGen(locationId);
    return gen ? `${url}&v=${gen}` : url;
  }

  function vikingHref(row) {
    let template = state.config.vikingUrl || "";
    if (!template || template.includes("connectomes.utah.edu/connectome")) {
      template = "viking://open?volumeName={volume}&location={id}";
    }
    return template
      .replaceAll("{volume}", encodeURIComponent(identityNameOf(state.volume)))
      .replaceAll("{id}", encodeURIComponent(String(row.location_id)));
  }

  // The name Identity (and so Viking) knows the volume by: NM for the NasalMonkey folder.
  function identityNameOf(volume) {
    const item = (state.me?.volumes || []).find((entry) => entry.name === volume);
    return item?.identityName || volume;
  }

  // A click on a location link asks the gallery for a viking:// link with a one-use launch code, so Viking opens
  // signed in (or an already open Viking for the same volume jumps to the location). The plain href is the fallback.
  async function openInViking(link) {
    const match = /[?&]location=([^&]+)/.exec(link.getAttribute("href") || "");
    if (!match || !state.volume) {
      return false;
    }
    const location = decodeURIComponent(match[1]);
    try {
      const response = await fetch(withSet(`/api/volumes/${encodeURIComponent(state.volume)}/viking-link`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ location }),
      });
      if (response.status === 401) {
        window.location.href = "/login";
        return true;
      }
      if (!response.ok) {
        return false;
      }
      const payload = await response.json();
      if (payload.url) {
        window.location.href = payload.url;
        return true;
      }
    } catch (error) {
      console.warn("Viking launch code failed; using the plain link", error);
    }
    return false;
  }

  document.addEventListener("click", (event) => {
    const link = event.target instanceof Element ? event.target.closest("a.viking-link") : null;
    if (!link || event.ctrlKey || event.metaKey || event.shiftKey || event.button !== 0) {
      return;
    }
    if (state.config.identityMode !== "oidc") {
      return;
    }
    event.preventDefault();
    openInViking(link).then((handled) => {
      if (!handled) {
        window.location.href = link.getAttribute("href");
      }
    });
  });

  // "250" is one radius, "100-500" a range, "100-" and "-500" are open ended. Anything else is `invalid`, which
  // matches nothing (and says so), instead of being ignored and leaving every card on screen.
  function parseRadius(text) {
    const trimmed = text.trim();
    if (!trimmed) {
      return null;
    }
    const range = /^(\d+(?:\.\d+)?)?\s*-\s*(\d+(?:\.\d+)?)?$/.exec(trimmed);
    if (range && (range[1] !== undefined || range[2] !== undefined)) {
      return {
        lo: range[1] === undefined ? -Infinity : Number(range[1]),
        hi: range[2] === undefined ? Infinity : Number(range[2]),
      };
    }
    const value = Number(trimmed);
    return Number.isFinite(value) ? { lo: value, hi: value } : { invalid: true };
  }

  // "1529372" or "1529372, 1529388": the location ids to show. Text with no usable id matches nothing.
  function parseLocationIds(text) {
    const trimmed = (text || "").trim();
    if (!trimmed) {
      return null;
    }
    const ids = new Set();
    for (const piece of trimmed.split(/[\s,;]+/)) {
      if (/^\d+$/.test(piece)) {
        ids.add(Number(piece));
      }
    }
    return ids;
  }

  function overlayHue() {
    const value = els.hue.value;
    return /^#[0-9a-fA-F]{6}$/.test(value) ? value : "#3478cb";
  }

  function overlayVisibility() {
    const value = Number(els.visibility.value);
    return Number.isFinite(value) ? Math.min(100, Math.max(0, value)) : 30;
  }

  function syncVisibilityLabel() {
    els.visibilityValue.textContent = `${overlayVisibility()}%`;
  }

  function cardPixelSize() {
    return CARD_SIZES[state.cardSize] || CARD_SIZES.large;
  }

  function tissueRelpath(row) {
    if (state.cardSize === "huge") {
      return row.image_relpath || row.jpeg_relpath;
    }
    return `overlays/thumb/${cardPixelSize()}/${row.image_key}.jpg`;
  }

  function syncCardSizeButtons() {
    document.body.dataset.cardSize = state.cardSize;
    document.querySelectorAll("[data-card-size]").forEach((button) => {
      const on = button.dataset.cardSize === state.cardSize;
      button.classList.toggle("active", on);
      button.setAttribute("aria-pressed", on ? "true" : "false");
    });
  }

  function readView() {
    try {
      const parsed = JSON.parse(localStorage.getItem(VIEW_STORAGE_KEY) || "");
      if (!parsed || typeof parsed !== "object") {
        return { set: "", volumes: {} };
      }
      const volumes = {};
      if (parsed.volumes && typeof parsed.volumes === "object") {
        for (const [key, value] of Object.entries(parsed.volumes)) {
          if (typeof value === "string") {
            volumes[key] = value;
          }
        }
      }
      return { set: typeof parsed.set === "string" ? parsed.set : "", volumes };
    } catch {
      return { set: "", volumes: {} };
    }
  }

  function rememberView(keepVolume) {
    const view = readView();
    view.set = state.trainingSet || "current";
    if (keepVolume && state.volume) {
      view.volumes[view.set] = state.volume;
    }
    localStorage.setItem(VIEW_STORAGE_KEY, JSON.stringify(view));
  }

  function loadCardSizePref() {
    const stored = localStorage.getItem(SIZE_KEY);
    state.cardSize = Object.prototype.hasOwnProperty.call(CARD_SIZES, stored) ? stored : "large";
    syncCardSizeButtons();
  }

  let resizeToken = 0;

  // A new size re-lays out every card and asks for new images, which takes a while. Show the choice at once: mark the
  // button, clear the old cards (dropping their queued loads) and say so, then lay out the new size only after the
  // browser has painted that. Quick repeated clicks lay out only the last size.
  function selectCardSize(name) {
    if (!Object.prototype.hasOwnProperty.call(CARD_SIZES, name) || name === state.cardSize) {
      return;
    }
    state.cardSize = name;
    localStorage.setItem(SIZE_KEY, name);
    syncCardSizeButtons();
    rowStride = 0;
    windowTop = -1;
    els.grid.innerHTML = "";
    reprioritize();
    els.status.textContent = `Loading ${name} cards…`;
    const mine = ++resizeToken;
    window.requestAnimationFrame(() => window.requestAnimationFrame(() => {
      if (mine === resizeToken) {
        render(true);
      }
    }));
  }

  function rememberOverlay() {
    localStorage.setItem(HUE_KEY, overlayHue());
    localStorage.setItem(VIS_KEY, String(overlayVisibility()));
  }

  function loadOverlayPrefs() {
    const hue = localStorage.getItem(HUE_KEY);
    const storedVisibility = localStorage.getItem(VIS_KEY);
    if (/^#[0-9a-fA-F]{6}$/.test(hue || "") && hue.toLowerCase() !== "#ff0000") {
      els.hue.value = hue;
    }
    if (storedVisibility != null && storedVisibility !== "25") {
      const visibility = Number(storedVisibility);
      if (Number.isFinite(visibility) && visibility >= 0 && visibility <= 100) {
        els.visibility.value = String(visibility);
      }
    }
    syncVisibilityLabel();
  }

  const VIEW_LABELS = { approved: "Approved", rejected: "Rejected", unknown: "Unknown" };

  // What the collapsed bar says about the cards. Pure, so it reads the same wherever the values come from.
  // `values` holds the control values (trainingSet, volume, classes as a Set, type, label, tags, z, radius, sorts);
  // `sortLabel` names a sort key. Returns { scope, criteria }: where the cards come from, and what narrows or orders them.
  function describeFilters(values, sortLabel) {
    const scope = [];
    if (values.trainingSet && values.trainingSet !== "current") {
      scope.push(`Training set ${values.trainingSet}`);
    }
    if (values.volume) {
      scope.push(`Volume ${values.volume}`);
    }
    const criteria = [];
    const classes = values.classes;
    if (!(classes.size === 1 && classes.has("unknown"))) {
      const names = Object.keys(VIEW_LABELS).filter((name) => classes.has(name)).map((name) => VIEW_LABELS[name]);
      criteria.push(names.length ? `View ${names.join(" + ")}` : "View nothing");
    }
    for (const [label, text] of [
      ["Type", values.type],
      ["Structure", values.label],
      ["Tags", values.tags],
      ["Z", values.z],
      ["Radius", values.radius],
      ["Location", values.location],
    ]) {
      const trimmed = (text || "").trim();
      if (trimmed) {
        criteria.push(`${label} ${trimmed}`);
      }
    }
    const sorts = values.sorts || [];
    const defaultSort = sorts.length === 1 && sorts[0].key === "z" && !sorts[0].desc;
    if (sorts.length && !defaultSort) {
      criteria.push(`Sorted by ${sorts.map((item) => `${sortLabel(item.key)} ${item.desc ? "↓" : "↑"}`).join(", ")}`);
    }
    return { scope, criteria };
  }

  function updateFilterSummary() {
    const labels = new Map(sortFields());
    const { scope, criteria } = describeFilters({
      trainingSet: state.trainingSet,
      volume: state.volume,
      classes: visibleClasses(),
      type: els.type.value,
      label: els.label.value,
      tags: els.tags.value,
      z: els.z.value,
      radius: els.radius.value,
      location: els.locate.value,
      sorts: state.sorts,
    }, (key) => labels.get(key) || key);
    const text = [...scope, ...(criteria.length ? criteria : ["No filters"])].join(" · ");
    els.filterSummary.textContent = text;
    els.filterSummary.title = text;
  }

  function filtersCollapsed() {
    return document.body.classList.contains("filters-collapsed");
  }

  // Hides the filter controls and shows their criteria as one line in the top bar, or brings the controls back.
  // The filters stay in effect either way. `remember` keeps the choice for the next visit.
  function setFiltersCollapsed(collapsed, remember) {
    document.body.classList.toggle("filters-collapsed", collapsed);
    els.filterSummary.hidden = !collapsed;
    els.filtersToggle.textContent = collapsed ? "Show filters" : "Hide filters";
    els.filtersToggle.setAttribute("aria-expanded", collapsed ? "false" : "true");
    els.filtersToggle.title = collapsed
      ? "Show the filter controls again."
      : "Hide the filter controls to make room for cards. The filters stay in effect.";
    if (collapsed) {
      updateFilterSummary();
    }
    if (remember) {
      localStorage.setItem(COLLAPSE_KEY, collapsed ? "1" : "0");
    }
  }

  function loadFiltersCollapsedPref() {
    setFiltersCollapsed(localStorage.getItem(COLLAPSE_KEY) === "1", false);
  }

  const SORT_FIELDS = [
    ["z", "Z"],
    ["type_id", "Type"],
    ["structure_id", "Structure ID"],
    ["structure_label", "Label"],
    ["radius", "Radius"],
    ["location_id", "Location ID"],
    ["loss", "Loss"],
  ];

  // The converter's default order: cards laid out by how their locations are linked (static/process_layout.js).
  // It is exclusive: it replaces every other sort, and every other sort replaces it.
  const LAYOUT_SORT = "layout";

  function layoutActive() {
    return Boolean(state.converter) && state.sorts.length > 0 && state.sorts[0].key === LAYOUT_SORT;
  }

  function sortFields() {
    const fields = SORT_FIELDS.slice();
    if (state.converter) {
      fields.unshift([LAYOUT_SORT, "Process / branch"]);
      // In the converter the "loss" of a card is how far the proposal moved from the current outline (1 - overlap).
      fields[fields.findIndex(([key]) => key === "loss")] = ["loss", "Change from current outline"];
    }
    if (state.hasSam2) {
      fields.push(["sam2GtIou", "SAM2 GT IoU"]);
    }
    if (state.hasCardScores && !state.converter) {
      fields.push(...CARD_SORT_FIELDS);
    }
    return fields;
  }

  // Card-score sorts. Picking one sets its "worst first" direction (low IoU and core recall, high spill and mismatch);
  // the arrow still flips it. Unscored cards sort last either way (compareValues puts missing values last).
  const CARD_SORT_FIELDS = [
    ["cardIou", "IoU"],
    ["cardCore", "Core recall"],
    ["cardSpill", "Spill"],
    ["cardMismatch", "Model mismatch"],
  ];
  const CARD_SORT_WORST_DESC = new Map([["cardIou", false], ["cardCore", false], ["cardSpill", true], ["cardMismatch", true]]);

  function compareValues(left, right, desc) {
    if (left == null && right == null) {
      return 0;
    }
    if (left == null) {
      return 1;
    }
    if (right == null) {
      return -1;
    }
    const direction = desc ? -1 : 1;
    if (typeof left === "string" || typeof right === "string") {
      return String(left).localeCompare(String(right)) * direction;
    }
    return (left - right) * direction;
  }

  function compareSort(left, right) {
    for (const criterion of state.sorts) {
      const result = compareValues(left[criterion.key], right[criterion.key], criterion.desc);
      if (result !== 0) {
        return result;
      }
    }
    return left.location_id - right.location_id;
  }

  // Five sorts share a column with the add button. A sixth sort opens the next column.
  const SORTS_PER_COLUMN = 5;

  // The filters do not apply to the process layout, which always lays out every card: switch them off while it is on.
  function syncLayoutLock() {
    const locked = layoutActive();
    els.filters.querySelectorAll("input, select, button").forEach((control) => {
      if (control.closest("#sort-filter")) {
        return;
      }
      if (locked) {
        if (!control.disabled) {
          control.disabled = true;
          control.dataset.layoutLocked = "1";
        }
      } else if (control.dataset.layoutLocked) {
        control.disabled = false;
        delete control.dataset.layoutLocked;
      }
    });
    els.filters.classList.toggle("layout-locked", locked);
  }

  function renderSortStack() {
    const fields = sortFields();
    syncLayoutLock();
    if (layoutActive()) {
      const options = fields.map(([value, label]) => `<option value="${value}"${value === LAYOUT_SORT ? " selected" : ""}>${label}</option>`).join("");
      els.sortStack.innerHTML = `<div class="sort-column"><div class="sort-row"><select data-index="0" aria-label="Sort by">${options}</select></div></div>`;
      return;
    }
    const used = new Set(state.sorts.map((item) => item.key));
    const canAdd = state.sorts.length < fields.filter(([value]) => value !== LAYOUT_SORT).length;
    const columns = [];
    for (let start = 0; start < state.sorts.length; start += SORTS_PER_COLUMN) {
      columns.push(state.sorts.slice(start, start + SORTS_PER_COLUMN));
    }
    if (!columns.length) {
      columns.push([]);
    }
    const html = columns.map((group, columnIndex) => {
      const base = columnIndex * SORTS_PER_COLUMN;
      const rows = group.map((criterion, offset) => {
        const index = base + offset;
        const options = fields.map(([value, label]) => {
          const taken = value !== LAYOUT_SORT && used.has(value) && value !== criterion.key;
          return `<option value="${value}"${value === criterion.key ? " selected" : ""}${taken ? " disabled" : ""}>${label}</option>`;
        }).join("");
        const remove = state.sorts.length > 1
          ? `<button type="button" class="sort-remove" data-index="${index}" title="Remove sort" aria-label="Remove sort">×</button>`
          : "";
        const direction = criterion.desc ? "Descending" : "Ascending";
        return `
          <div class="sort-row">
            <select data-index="${index}" aria-label="Sort by">${options}</select>
            <button type="button" class="sort-dir${criterion.desc ? " active" : ""}" data-index="${index}" title="${direction}" aria-label="${direction}" aria-pressed="${criterion.desc ? "true" : "false"}">${criterion.desc ? "↓" : "↑"}</button>
            ${remove}
          </div>`;
      }).join("");
      const add = columnIndex === columns.length - 1 && canAdd
        ? `<button type="button" id="sort-add" title="Add a sort">+</button>`
        : "";
      return `<div class="sort-column">${rows}${add}</div>`;
    }).join("");
    els.sortStack.innerHTML = html;
  }

  function syncReverseButtons() {
    const primary = state.sorts[0] || { key: "z", desc: false };
    const label = primary.desc ? "Descending" : "Ascending";
    els.filters.querySelectorAll("button.reverse").forEach((button) => {
      const selected = button.dataset.sort === primary.key;
      button.classList.toggle("active", selected);
      button.textContent = selected ? label : "Reverse";
      button.setAttribute("aria-pressed", selected && primary.desc ? "true" : "false");
    });
  }

  function reviewClass(row) {
    if (row.ignored) {
      return "rejected";
    }
    if (row.approved) {
      return "approved";
    }
    return "unknown";
  }

  function visibleClasses() {
    const selected = new Set();
    document.querySelectorAll("#view-classes input:checked").forEach((input) => {
      selected.add(input.dataset.view);
    });
    return selected;
  }

  function filtered() {
    if (layoutActive()) {
      state.filterNote = "";
      return state.rows.slice().sort(compareSort);
    }
    state.filterNote = state.filterNote.startsWith("tag") ? state.filterNote : "";
    const type = els.type.value.trim().toLowerCase();
    const label = els.label.value.trim().toLowerCase();
    const zText = els.z.value.trim();
    const radius = parseRadius(els.radius.value);
    const locations = parseLocationIds(els.locate.value);
    const classes = visibleClasses();
    if (radius && radius.invalid) {
      state.filterNote = "radius: use a number or a range such as 100-500";
    }
    return state.rows.filter((row) => {
      if (locations) {
        // Asking for a location by id shows it whatever its review state, so View does not hide it.
        if (!locations.has(Number(row.location_id))) {
          return false;
        }
      } else if (!classes.has(reviewClass(row))) {
        return false;
      }
      if (zText && String(row.z) !== zText) {
        return false;
      }
      if (type) {
        const hay = `${row.type_id || ""} ${row.type_name || ""}`.toLowerCase();
        if (!hay.includes(type)) {
          return false;
        }
      }
      if (label) {
        const idText = row.structure_id == null ? "" : String(row.structure_id);
        const name = `${row.structure_label || ""}`;
        const pieces = label.split(",").map((item) => item.trim()).filter(Boolean);
        if (pieces.length > 1 && pieces.every((item) => /^\d+$/.test(item))) {
          if (!pieces.includes(idText)) {
            return false;
          }
        } else if (/^\d+$/.test(label)) {
          if (idText !== label) {
            return false;
          }
        } else {
          let pattern;
          try {
            pattern = new RegExp(label, "i");
          } catch (error) {
            state.filterNote = error.message;
            return false;
          }
          if (!pattern.test(name)) {
            return false;
          }
        }
      }
      if (state.tagIds && !state.tagIds.has(Number(row.structure_id))) {
        return false;
      }
      if (radius) {
        const value = Number(row.radius);
        if (radius.invalid || !Number.isFinite(value) || value < radius.lo || value > radius.hi) {
          return false;
        }
      }
      return true;
    }).sort(compareSort);
  }

  function sbfsemHref(row) {
    if (row.structure_id == null) {
      return "";
    }
    const params = new URLSearchParams({
      volume: state.volume,
      cells: String(row.structure_id),
      location: String(row.location_id),
    });
    return `https://sbfsem-tools.com/open?${params}`;
  }

  function windowActions(row) {
    const key = row.image_key || "";
    const approved = Boolean(row.approved) && !row.ignored;
    const ignored = Boolean(row.ignored);
    const trashIcon = `<svg viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M9 3h6l1 2h5v2H3V5h5l1-2zm1 6h2v9h-2V9zm4 0h2v9h-2V9zM7 9h2v9H7V9zm-1 12h12l1-13H5l1 13z"/></svg>`;
    const approveAct = approved ? "unapprove" : "approve";
    const approveLabel = approved ? "Approved" : "Approve";
    const check = `<button class="icon check${approved ? " on" : ""}" data-act="${approveAct}" data-id="${row.location_id}" data-key="${key}" title="${approveLabel}" aria-label="${approveLabel}" aria-pressed="${approved ? "true" : "false"}">✓</button>`;
    const reject = ignored
      ? `<button class="icon plus" data-act="restore" data-id="${row.location_id}" data-key="${key}" title="Undo reject" aria-label="Undo reject">+</button>`
      : `<button class="icon trash" data-act="ignore" data-id="${row.location_id}" data-key="${key}" title="Reject" aria-label="Reject">${trashIcon}</button>`;
    return check + reject;
  }

  // The card-size layer (target mask + negatives) for sizes below Huge; Huge keeps the full-size mask.
  function layerUrl(row) {
    if (state.cardSize === "huge" || !row.image_key) {
      return "";
    }
    const url = withSet(
      `/api/volumes/${encodeURIComponent(state.volume)}/cardmask?size=${cardPixelSize()}`
      + `&key=${encodeURIComponent(row.image_key)}&loc=${encodeURIComponent(row.location_id)}`,
    );
    const gen = locationGen(row.location_id);
    return gen ? `${url}&v=${gen}` : url;
  }

  function windowHtml(row) {
    const image = mediaUrl(state.volume, tissueRelpath(row), row.location_id);
    const mask = mediaUrl(state.volume, row.mask_relpath, row.location_id);
    const layer = layerUrl(row);
    const hue = overlayHue();
    const visibility = overlayVisibility();
    const status = row.ignored ? "rejected" : (row.approved ? "approved" : "");
    return `
      <div class="window">
        <div class="composite waiting" data-loc="${row.location_id}" data-key="${row.image_key || ""}" data-size="${state.cardSize}" data-status="${status}" data-gen="${locationGen(row.location_id)}" data-image="${image}" data-mask="${mask}" data-layer="${layer}" data-hue="${hue}" data-visibility="${visibility}">
          <img class="jpeg pending" alt="">
          <canvas class="tint" aria-hidden="true"></canvas>
          <span class="media-state" aria-hidden="true"></span>
        </div>
      </div>`;
  }

  const WINDOW_ORIGIN = /_X(\d+)(?:-(\d+))?_Y(\d+)(?:-(\d+))?$/;

  function windowOrigin(row) {
    const match = WINDOW_ORIGIN.exec(String(row.image_key || ""));
    if (!match) {
      return null;
    }
    return { x: Number(match[1]), y: Number(match[3]) };
  }

  function shapeOf(windows) {
    const cropSize = state.cropSize || 1024;
    const cells = [];
    for (const row of windows) {
      const origin = windowOrigin(row);
      if (!origin) {
        return null;
      }
      cells.push({
        row,
        x: Math.round(origin.x / cropSize),
        y: Math.round(origin.y / cropSize),
      });
    }
    if (cells.length < 2) {
      return null;
    }
    const minX = Math.min(...cells.map((cell) => cell.x));
    const minY = Math.min(...cells.map((cell) => cell.y));
    const cols = Math.max(...cells.map((cell) => cell.x)) - minX + 1;
    const rows = Math.max(...cells.map((cell) => cell.y)) - minY + 1;
    return { cells, minX, minY, cols, rows };
  }

  function captionHtml(row) {
    // score.py stores "v5.4 at D16" when a large object had to be scored whole at a coarser level than its crops.
    const coarse = /\bat D(\d+)$/.exec(row.sam2Checkpoint || "");
    const sam2Title = [
      "SAM2 from a seed circle: IoU with this mask",
      row.sam2Checkpoint ? `model ${row.sam2Checkpoint}` : "",
      coarse ? `too large for one card at its own level, so scored whole at D${coarse[1]}, coarser than the model `
        + "trained on: large objects score lower here than in training" : "",
      row.sam2PredIou != null ? `model's own estimate ${Number(row.sam2PredIou).toFixed(2)}` : "",
      row.sam2ScoredAt ? `scored ${row.sam2ScoredAt}` : "",
    ].filter(Boolean).join("; ");
    const sam2 = row.sam2GtIou == null
      ? ""
      : `<span class="badge" title="${sam2Title}">SAM2 ${Number(row.sam2GtIou).toFixed(2)}${coarse ? ` · D${coarse[1]}` : ""}</span>`;
    // In the gallery the card's score is its card score (cardScoreHtml, beside the caption); the training loss is only
    // in that badge's tooltip. The converter's "change" stays here.
    const loss = row.loss == null || !state.converter ? "" : `<span class="badge">change ${Number(row.loss).toFixed(2)}</span>`;
    const structureName = row.type_name || "Structure";
    const structureText = row.structure_id == null ? structureName : `${structureName} ${row.structure_id}`;
    const structureHref = sbfsemHref(row);
    const structure = structureHref
      ? `<a href="${structureHref}" target="_blank" rel="noreferrer" title="Open in sbfsem-tools">${structureText}</a>`
      : structureText;
    const label = row.structure_label ? ` · ${row.structure_label}` : "";
    return `<a class="viking-link" href="${vikingHref(row)}" title="Open in Viking">#${row.location_id}</a> · Z ${row.z} · ${structure}${label} ${loss}${sam2}`;
  }

  // ---- card scores --------------------------------------------------------------------------------------------------
  // The scorer service (sam2-em-score-cards --serve) scores each card the way validation does: IoU, core recall (null
  // when the annotation is too thin to have a core) and spill (can exceed 1). The catalog carries the newest score as
  // cardIou, cardCore, cardSpill, cardMismatch (= max(1 - core, spill)), cardModel, cardScoredAt, and the one before it
  // as cardOld after a re-score. A location whose re-score request is pending or running shows "re-scoring…".
  // cardKind is "validation" (exact: eval mode, fixed prompt) or "train" (measured while the card was trained:
  // augmented image, random prompt, weights mid-epoch; shown with a "train" tag, never compared with validation).
  const CARD_SCORE_KEYS = ["cardIou", "cardCore", "cardSpill", "cardMismatch", "cardModel", "cardScoredAt", "cardOld",
    "cardKind", "cardEpoch"];
  const TRAIN_SCORE_NOTE = "Measured while training (augmented image, random prompt, weights mid-epoch): a rough guide, not comparable with validation scores.";
  const RESCORE_POLL_MS = 2000;
  const RESCORE_GIVE_UP_MS = 5 * 60 * 1000;
  // location id -> { state: "pending" | "running" | "failed", error, since (ms) }
  const rescoreMarks = new Map();
  const rescoreWatch = { timer: 0, volume: "", trainingSet: "" };

  function attrText(text) {
    return String(text).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function scoreText(value) {
    return value == null ? "–" : Number(value).toFixed(2);
  }

  // "0.91", or "0.82→0.91" after a re-score.
  function scorePair(oldValue, newValue, hasOld) {
    return hasOld ? `${scoreText(oldValue)}→${scoreText(newValue)}` : scoreText(newValue);
  }

  function cardScoreTitle(row, mark) {
    const old = row.cardOld || null;
    const lines = [];
    if (mark && (mark.state === "pending" || mark.state === "running")) {
      lines.push(mark.state === "running" ? "Re-scoring this location's cards now" : "Re-score queued; waiting for the scorer");
    }
    if (mark && mark.state === "failed") {
      lines.push(`Re-score failed: ${mark.error || "no reason given"}`);
    }
    if (row.cardIou != null || row.cardMismatch != null) {
      const kind = row.cardKind === "train" ? "Training-pass score" : "Validation score";
      lines.push(`${kind}${row.cardModel ? ` (model ${row.cardModel}` : " ("}${row.cardScoredAt ? `, ${row.cardScoredAt})` : ")"}`);
      if (row.cardKind === "train") {
        lines.push(TRAIN_SCORE_NOTE);
      }
      lines.push(`IoU ${scorePair(old && old.iou, row.cardIou, Boolean(old))}`);
      lines.push(row.cardCore == null
        ? "core recall: none (the annotation is too thin to have a core)"
        : `core recall ${scorePair(old && old.core, row.cardCore, Boolean(old))}`);
      lines.push(`spill ${scorePair(old && old.spill, row.cardSpill, Boolean(old))} (predicted area outside the mask, as a share of it)`);
      lines.push(`model mismatch ${scorePair(old && old.mismatch, row.cardMismatch, Boolean(old))} = max(1 − core recall, spill)`);
      if (old) {
        lines.push(`before: ${old.model ? `model ${old.model}` : "earlier score"}${old.scoredAt ? `, ${old.scoredAt}` : ""}`);
      }
    } else if (!mark || mark.state === "failed") {
      lines.push("No card score yet");
    }
    if (row.loss != null) {
      lines.push(`training loss ${Number(row.loss).toFixed(3)}`);
    }
    if (state.permission === "review") {
      lines.push("Click to re-score this location's cards with the current model (no redraw)");
    }
    return lines.join("\n");
  }

  // The badge. Large and Huge cards show IoU, core recall and spill; Medium and Small have room for IoU only (app.css
  // hides .cs-full there). Reviewers can click it to queue a re-score without redrawing.
  function cardScoreBadge(row) {
    if (state.converter) {
      return "";
    }
    const mark = rescoreMarks.get(Number(row.location_id));
    const busy = Boolean(mark) && (mark.state === "pending" || mark.state === "running");
    const scored = row.cardIou != null || row.cardMismatch != null;
    if (!busy && !scored && row.loss == null && !mark) {
      return "";
    }
    let full;
    let short;
    if (busy) {
      full = "re-scoring…";
      short = full;
    } else if (scored) {
      const old = row.cardOld || null;
      const hasOld = Boolean(old);
      short = `IoU ${scorePair(old && old.iou, row.cardIou, hasOld)}`;
      full = `${short} · core ${scorePair(old && old.core, row.cardCore, hasOld)} · spill ${scorePair(old && old.spill, row.cardSpill, hasOld)}`;
    } else {
      full = "–";
      short = full;
    }
    const train = scored && !busy && row.cardKind === "train";
    const classes = ["badge", "card-score", busy ? "rescoring" : "", mark && mark.state === "failed" ? "failed" : "",
      scored ? "" : "unscored", train ? "train" : ""]
      .filter(Boolean).join(" ");
    const title = attrText(cardScoreTitle(row, mark));
    const tag = train ? `<span class="cs-tag">train</span>` : "";
    const inner = `${tag}<span class="cs-full">${full}</span><span class="cs-short">${short}</span>`;
    return state.permission === "review"
      ? `<button type="button" class="${classes}" data-act="rescore" data-id="${row.location_id}" title="${title}">${inner}</button>`
      : `<span class="${classes}" title="${title}">${inner}</span>`;
  }

  function cardScoreHtml(row) {
    return `<span class="card-score-slot" data-id="${row.location_id}" data-key="${row.image_key || ""}">${cardScoreBadge(row)}${trendButton(row)}</span>`;
  }

  // ---- score trend --------------------------------------------------------------------------------------------------
  // A small chart of one card's IoU, core recall and spill across epochs (validation solid, training pass dashed).
  const TREND_ICON = `<svg viewBox="0 0 16 16" width="12" height="12" aria-hidden="true"><polyline points="1,13 5,8 9,10 15,3" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
  const TREND_SERIES = [
    { key: "iou", label: "IoU", color: "#7dd3fc" },
    { key: "core_recall", label: "core", color: "#4ade80" },
    { key: "spill", label: "spill", color: "#fb923c" },
  ];

  function trendButton(row) {
    if (state.converter || (row.cardIou == null && row.cardMismatch == null)) {
      return "";
    }
    return `<button type="button" class="trend-btn" data-act="trend" data-id="${row.location_id}" data-key="${attrText(row.image_key || "")}" title="Score trend across epochs" aria-label="Score trend">${TREND_ICON}</button>`;
  }

  function closeTrend() {
    const open = document.querySelector(".trend-pop");
    if (open) {
      open.remove();
    }
  }

  // Points of one kind as [{x, point}], x the epoch (or the order scored when a score has no epoch).
  function trendPoints(points) {
    return points.map((point, index) => ({ x: point.epoch == null ? index : Number(point.epoch), point }));
  }

  // The chart's SVG: one line per measure and kind, y from 0 to 1 (spill above 1 is drawn at the top edge).
  function trendSvg(history) {
    const width = 260;
    const height = 120;
    const pad = { l: 28, r: 8, t: 8, b: 20 };
    const kinds = [
      { name: "validation", points: trendPoints(history.validation || []), dash: "", hollow: false },
      { name: "train", points: trendPoints(history.train || []), dash: "3 3", hollow: true },
    ];
    const xs = kinds.flatMap((kind) => kind.points.map((p) => p.x));
    if (!xs.length) {
      return `<p class="trend-empty">No scores yet.</p>`;
    }
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const sx = (x) => pad.l + (maxX === minX ? (width - pad.l - pad.r) / 2 : ((x - minX) / (maxX - minX)) * (width - pad.l - pad.r));
    const sy = (y) => pad.t + (1 - Math.max(0, Math.min(1, y))) * (height - pad.t - pad.b);
    const parts = [];
    for (const tick of [0, 0.5, 1]) {
      parts.push(`<line x1="${pad.l}" x2="${width - pad.r}" y1="${sy(tick)}" y2="${sy(tick)}" class="trend-grid"/>`);
      parts.push(`<text x="${pad.l - 4}" y="${sy(tick) + 3}" text-anchor="end" class="trend-axis">${tick}</text>`);
    }
    const epochs = [...new Set(xs)].sort((a, b) => a - b);
    const step = Math.max(1, Math.ceil(epochs.length / 8));
    epochs.filter((_, i) => i % step === 0).forEach((x) => {
      parts.push(`<text x="${sx(x)}" y="${height - 6}" text-anchor="middle" class="trend-axis">${x}</text>`);
    });
    for (const kind of kinds) {
      for (const series of TREND_SERIES) {
        const pts = kind.points.filter((p) => p.point[series.key] != null);
        if (!pts.length) {
          continue;
        }
        const path = pts.map((p) => `${sx(p.x).toFixed(1)},${sy(Number(p.point[series.key])).toFixed(1)}`).join(" ");
        parts.push(`<polyline points="${path}" fill="none" stroke="${series.color}" stroke-width="1.6"${kind.dash ? ` stroke-dasharray="${kind.dash}"` : ""}/>`);
        for (const p of pts) {
          const value = Number(p.point[series.key]);
          const tip = `${series.label} ${value.toFixed(3)} · ${kind.name} · ${p.point.model || ""}${p.point.epoch != null ? ` (epoch ${p.point.epoch})` : ""}`;
          parts.push(`<circle cx="${sx(p.x).toFixed(1)}" cy="${sy(value).toFixed(1)}" r="2.6" fill="${kind.hollow ? "var(--panel)" : series.color}" stroke="${series.color}" stroke-width="1.2"><title>${attrText(tip)}</title></circle>`);
        }
      }
    }
    return `<svg class="trend-chart" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" role="img" aria-label="Score trend">${parts.join("")}</svg>`;
  }

  function trendLegend(history) {
    const series = TREND_SERIES.map((s) => `<span><i style="background:${s.color}"></i>${s.label}</span>`).join("");
    const kinds = [
      (history.validation || []).length ? `<span class="trend-kind">── validation</span>` : "",
      (history.train || []).length ? `<span class="trend-kind" title="${attrText(TRAIN_SCORE_NOTE)}">- - training pass</span>` : "",
    ].join("");
    return `<div class="trend-legend">${series}${kinds}</div>`;
  }

  async function openTrend(button) {
    closeTrend();
    const id = Number(button.dataset.id);
    const key = button.dataset.key || "";
    const pop = document.createElement("div");
    pop.className = "trend-pop";
    pop.setAttribute("role", "dialog");
    pop.innerHTML = `<div class="trend-head">#${id} · score trend <button type="button" class="trend-close" aria-label="Close">×</button></div><p class="trend-empty">Loading…</p>`;
    document.body.appendChild(pop);
    const box = button.getBoundingClientRect();
    const left = Math.max(8, Math.min(window.innerWidth - 290, box.right - 280));
    const top = box.bottom + 190 < window.innerHeight ? box.bottom + 4 : Math.max(8, box.top - 194);
    pop.style.left = `${left + window.scrollX}px`;
    pop.style.top = `${top + window.scrollY}px`;
    pop.querySelector(".trend-close").addEventListener("click", closeTrend);
    try {
      const url = withSet(`/api/volumes/${encodeURIComponent(state.volume)}/card-history?id=${id}&key=${encodeURIComponent(key)}`);
      const history = await getJson(url);
      if (!document.body.contains(pop)) {
        return;
      }
      pop.querySelector(".trend-empty").outerHTML = trendSvg(history) + trendLegend(history);
    } catch (error) {
      const empty = pop.querySelector(".trend-empty");
      if (empty) {
        empty.textContent = `Could not load the trend: ${error.message}`;
      }
    }
  }

  // Redraw one location's card-score badges in place (after its scores or its re-score state changed).
  function paintCardScores(locationId) {
    const id = String(locationId);
    els.grid.querySelectorAll(`.card-score-slot[data-id="${id}"]`).forEach((slot) => {
      const row = windowsForLocation(locationId).find((item) => (item.image_key || "") === (slot.dataset.key || ""));
      slot.innerHTML = row ? cardScoreBadge(row) + trendButton(row) : "";
    });
  }

  // Put a location's newest card scores (from /card-scores) on its rows; its cards without one lose theirs.
  function applyCardScores(locationIds, cards) {
    const wanted = new Set(locationIds.map(String));
    for (const row of state.rows) {
      if (wanted.has(String(row.location_id))) {
        for (const key of CARD_SCORE_KEYS) {
          delete row[key];
        }
        delete row.cardRescore;
      }
    }
    for (const card of cards) {
      const row = windowsForLocation(card.location_id).find((item) => (item.image_key || "") === (card.image_key || ""));
      if (!row) {
        continue;
      }
      for (const key of CARD_SCORE_KEYS) {
        if (key in card) {
          row[key] = card[key];
        }
      }
    }
    if (!state.hasCardScores && cards.length) {
      state.hasCardScores = true;
      renderSortStack();
    }
  }

  function rescoreUrl(volume, suffix, ids) {
    return withSet(`/api/volumes/${encodeURIComponent(volume)}/${suffix}?ids=${ids.join(",")}`);
  }

  // Show "re-scoring…" on a location's cards and poll until its request is done or failed (or 5 minutes pass).
  function watchRescore(locationId, request) {
    const id = Number(locationId);
    if (rescoreWatch.volume !== state.volume || rescoreWatch.trainingSet !== state.trainingSet) {
      resetRescoreWatch();
    }
    const known = request && request.state;
    if (known === "done" || known === "failed") {
      rescoreMarks.set(id, { state: "running", error: "", since: Date.now() });
    } else {
      rescoreMarks.set(id, { state: known || "pending", error: "", since: Date.now() });
    }
    paintCardScores(id);
    scheduleRescorePoll();
  }

  function resetRescoreWatch() {
    window.clearTimeout(rescoreWatch.timer);
    rescoreWatch.timer = 0;
    rescoreMarks.clear();
    rescoreWatch.volume = state.volume;
    rescoreWatch.trainingSet = state.trainingSet;
  }

  function watchedRescores() {
    return [...rescoreMarks].filter(([, mark]) => mark.state === "pending" || mark.state === "running").map(([id]) => id);
  }

  function scheduleRescorePoll() {
    if (!rescoreWatch.timer && watchedRescores().length) {
      rescoreWatch.timer = window.setTimeout(pollRescores, RESCORE_POLL_MS);
    }
  }

  async function pollRescores() {
    rescoreWatch.timer = 0;
    const { volume, trainingSet } = rescoreWatch;
    const ids = watchedRescores();
    if (!ids.length || !volume) {
      return;
    }
    let statuses = null;
    try {
      statuses = (await getJson(rescoreUrl(volume, "rescore-status", ids))).status || {};
    } catch {
      // A failed poll is retried at the next tick, until the give-up time.
    }
    if (rescoreWatch.volume !== volume || rescoreWatch.trainingSet !== trainingSet) {
      return;
    }
    const finished = [];
    const now = Date.now();
    for (const id of ids) {
      const mark = rescoreMarks.get(id);
      if (!mark) {
        continue;
      }
      const status = statuses ? statuses[String(id)] : null;
      if (status && (status.state === "done" || status.state === "failed")) {
        finished.push({ id, status });
      } else if (now - mark.since > RESCORE_GIVE_UP_MS) {
        finished.push({ id, status: null });
      } else if (status && status.state !== mark.state) {
        mark.state = status.state;
        paintCardScores(id);
      }
    }
    if (finished.length) {
      await finishRescores(volume, trainingSet, finished);
    }
    scheduleRescorePoll();
  }

  async function finishRescores(volume, trainingSet, finished) {
    const ids = finished.map((item) => item.id);
    let cards = null;
    try {
      cards = (await getJson(rescoreUrl(volume, "card-scores", ids))).cards || [];
    } catch {
      // Keep the scores shown; a reload picks the new ones up.
    }
    if (rescoreWatch.volume !== volume || rescoreWatch.trainingSet !== trainingSet) {
      return;
    }
    if (cards) {
      applyCardScores(ids, cards);
    }
    const notes = [];
    for (const { id, status } of finished) {
      if (status && status.state === "failed") {
        rescoreMarks.set(id, { state: "failed", error: status.error || "", since: Date.now() });
        notes.push(`Re-score of #${id} failed${status.error ? `: ${status.error}` : ""}.`);
      } else if (!status) {
        rescoreMarks.delete(id);
        notes.push(`Re-score of #${id} has not finished after ${RESCORE_GIVE_UP_MS / 60000} minutes; reload later to see it.`);
      } else {
        rescoreMarks.delete(id);
        notes.push(`Re-scored #${id}${status.cards != null ? ` (${status.cards} card${status.cards === 1 ? "" : "s"})` : ""}.`);
      }
      paintCardScores(id);
    }
    els.status.textContent = notes.join(" ");
  }

  async function queueRescore(locationId) {
    const id = Number(locationId);
    const result = await getJson(withSet(`/api/volumes/${encodeURIComponent(state.volume)}/rescore`), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ location_id: id }),
    });
    watchRescore(id, result.request);
    els.status.textContent = `Queued #${id} for re-scoring.`;
  }

  // After a catalog load: watch the locations the catalog says are still being re-scored (cardRescore). A reload of the
  // same volume keeps the marks it has (a failed re-score keeps its tooltip); another volume or set starts afresh.
  function watchCatalogRescores() {
    if (rescoreWatch.volume !== state.volume || rescoreWatch.trainingSet !== state.trainingSet) {
      resetRescoreWatch();
    }
    for (const row of state.rows) {
      const id = Number(row.location_id);
      const mark = rescoreMarks.get(id);
      if (row.cardRescore && (!mark || mark.state === "failed")) {
        rescoreMarks.set(id, { state: row.cardRescore.state || "pending", error: "", since: Date.now() });
      }
    }
    scheduleRescorePoll();
  }

  function shapeOffsets(windows, columns) {
    const shape = shapeOf(windows);
    if (!shape || columns < 1) {
      return null;
    }
    const buckets = new Map();
    for (const cell of shape.cells) {
      const dr = cell.y - shape.minY;
      const dc = cell.x - shape.minX;
      const key = `${dr},${dc}`;
      const list = buckets.get(key);
      if (list) {
        list.push(cell.row);
      } else {
        buckets.set(key, [cell.row]);
      }
    }
    // A shape wider than the screen is cut into bands of `columns` and stacked. Each band keeps only the rows it
    // uses, so a sparse shape does not leave empty rows above it (the grid draws only the rows near the viewport,
    // so a card that starts rows down the page would show as a blank grid) or between its bands.
    const bandRows = new Map();
    for (const key of buckets.keys()) {
      const [dr, dc] = key.split(",").map(Number);
      const band = Math.floor(dc / columns);
      const range = bandRows.get(band);
      if (range) {
        range.lo = Math.min(range.lo, dr);
        range.hi = Math.max(range.hi, dr);
      } else {
        bandRows.set(band, { lo: dr, hi: dr });
      }
    }
    let layerHeight = 0;
    for (const band of [...bandRows.keys()].sort((a, b) => a - b)) {
      const range = bandRows.get(band);
      range.start = layerHeight;
      layerHeight += range.hi - range.lo + 1;
    }
    const offsets = [];
    for (const [key, rows] of buckets) {
      const [dr, dc] = key.split(",").map(Number);
      const range = bandRows.get(Math.floor(dc / columns));
      rows.forEach((row, copy) => {
        offsets.push({
          row,
          dr: copy * layerHeight + range.start + dr - range.lo,
          dc: dc % columns,
        });
      });
    }
    return offsets;
  }

  function cellHtml(cell, rowOffset) {
    const row = cell.row;
    return `
      <article class="card ${joinClasses(cell)}" data-id="${row.location_id}" data-key="${row.image_key || ""}" style="--link-h:${cell.linkH};--link-s:${cell.linkS}%;--link-l:${cell.linkL}%;grid-column:${cell.gc + 1};grid-row:${cell.gr - rowOffset + 1}">
        ${windowHtml(row)}
        ${windowActions(row)}
        <div class="meta"><div class="meta-row"><div>${captionHtml(row)}</div>${cardScoreHtml(row)}${refreshButton(row)}</div></div>
      </article>`;
  }

  const refreshMarks = new Map();

  function refreshMarkClass(locationId) {
    const mark = refreshMarks.get(Number(locationId));
    return mark ? ` ${mark}` : "";
  }

  function refreshButton(row) {
    const icon = `<svg class="arrow" viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M17.65 6.35A7.95 7.95 0 0 0 12 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08a5.99 5.99 0 0 1-5.65 4c-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"/></svg><svg class="mark" viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M9 16.2 4.8 12l-1.4 1.4L9 19 21 7l-1.4-1.4z"/></svg>`;
    return `<button type="button" class="icon refresh review-only${refreshMarkClass(row.location_id)}" data-act="refresh" data-id="${row.location_id}" title="Refresh mask from OData" aria-label="Refresh mask">${icon}</button>`;
  }

  function cardsFrom(rows) {
    const byId = new Map();
    for (const row of rows) {
      const list = byId.get(row.location_id) || [];
      list.push(row);
      byId.set(row.location_id, list);
    }
    const seen = new Set();
    const cards = [];
    for (const row of rows) {
      if (seen.has(row.location_id)) {
        continue;
      }
      seen.add(row.location_id);
      cards.push(byId.get(row.location_id));
    }
    return cards;
  }

  function windowsForLocation(locationId) {
    return catalogIndex.byLocation.get(locationId)
      || catalogIndex.byLocation.get(Number(locationId))
      || [];
  }

  function indexCatalog() {
    const byLocation = new Map();
    for (const row of state.rows) {
      const list = byLocation.get(row.location_id);
      if (list) {
        list.push(row);
      } else {
        byLocation.set(row.location_id, [row]);
      }
    }
    catalogEpoch += 1;
    catalogIndex.byLocation = byLocation;
    catalogIndex.stamp = "";
    packMemo.packed = null;
    packMemo.repackFrom = null;
  }

  function viewStamp() {
    const views = [];
    document.querySelectorAll("#view-classes input").forEach((input) => {
      views.push(`${input.dataset.view}${input.checked ? 1 : 0}`);
    });
    return [
      catalogEpoch,
      tagEpoch,
      els.type.value,
      els.label.value,
      els.z.value,
      els.radius.value,
      els.locate.value,
      views.join(""),
      state.sorts.map((item) => `${item.key}:${item.desc ? 1 : 0}`).join(","),
      layoutActive() ? state.layoutEpoch : "",
      state.cropSize || 0,
    ].join("\n");
  }

  function ensureVisibleCards() {
    const stamp = viewStamp();
    if (catalogIndex.stamp === stamp) {
      return catalogIndex.visibleCards;
    }
    catalogIndex.visibleCards = cardsFrom(filtered());
    catalogIndex.cardAt = new Map();
    catalogIndex.visibleCards.forEach((windows, index) => {
      catalogIndex.cardAt.set(windows[0].location_id, index);
    });
    catalogIndex.stamp = stamp;
    packMemo.packed = null;
    packMemo.repackFrom = null;
    return catalogIndex.visibleCards;
  }

  function forgetVisibleLocation(locationId) {
    const index = catalogIndex.cardAt.get(locationId) ?? catalogIndex.cardAt.get(Number(locationId));
    if (index == null) {
      return null;
    }
    catalogIndex.visibleCards.splice(index, 1);
    catalogIndex.cardAt.delete(locationId);
    catalogIndex.cardAt.delete(Number(locationId));
    for (let cursor = index; cursor < catalogIndex.visibleCards.length; cursor += 1) {
      catalogIndex.cardAt.set(catalogIndex.visibleCards[cursor][0].location_id, cursor);
    }
    packMemo.repackFrom = index;
    return index;
  }

  function resetVisibleCards() {
    catalogIndex.stamp = "";
    packMemo.packed = null;
    packMemo.repackFrom = null;
  }

  function packGrid(cards, columns, resume) {
    const byLocation = catalogIndex.byLocation;
    const occupied = new Set();
    const cells = [];
    const anchors = [];
    let scanFrom = 0;
    let maxRow = 0;
    let starts = [0];
    let anchorStarts = [0];
    let startCard = 0;

    function isFree(row, col) {
      return col >= 0 && col < columns && row >= 0 && !occupied.has(row * columns + col);
    }

    function reserve(row, col) {
      const slot = row * columns + col;
      occupied.add(slot);
      anchors.push(slot);
      if (row + 1 > maxRow) {
        maxRow = row + 1;
      }
      while (occupied.has(scanFrom)) {
        scanFrom += 1;
      }
    }

    function take(row, col, source) {
      reserve(row, col);
      cells.push({ row: source, gr: row, gc: col });
    }

    function placeSingles(windows) {
      for (const source of windows) {
        const index = scanFrom;
        take(Math.floor(index / columns), index % columns, source);
      }
    }

    if (
      resume
      && resume.columns === columns
      && resume.startCard > 0
      && resume.anchors
      && resume.anchorStarts
      && resume.starts
      && resume.cells
    ) {
      startCard = resume.startCard;
      const keepAnchors = resume.anchorStarts[startCard] || 0;
      for (let index = 0; index < keepAnchors; index += 1) {
        const slot = resume.anchors[index];
        anchors.push(slot);
        occupied.add(slot);
        const row = Math.floor(slot / columns);
        if (row + 1 > maxRow) {
          maxRow = row + 1;
        }
      }
      scanFrom = 0;
      while (occupied.has(scanFrom)) {
        scanFrom += 1;
      }
      const keepCells = resume.starts[startCard] || 0;
      for (let index = 0; index < keepCells; index += 1) {
        cells.push(resume.cells[index]);
      }
      starts = resume.starts.slice(0, startCard + 1);
      anchorStarts = resume.anchorStarts.slice(0, startCard + 1);
    }

    for (let index = startCard; index < cards.length; index += 1) {
      const visible = cards[index];
      const all = byLocation.get(visible[0].location_id) || visible;
      const offsets = shapeOffsets(all.length ? all : visible, columns);
      if (!offsets) {
        placeSingles(visible);
      } else {
        // Hide one window by leaving its cell empty. Sibling windows keep the
        // shape, including a pair that would otherwise collapse into one tile.
        const visibleKeys = new Set(visible.map((row) => row.image_key || ""));
        const height = Math.max(...offsets.map((cell) => cell.dr)) + 1;
        const limit = scanFrom + columns * (maxRow + height + 2);
        let placed = false;
        for (let anchor = scanFrom; anchor < limit; anchor += 1) {
          const row = Math.floor(anchor / columns);
          const col = anchor % columns;
          if (!offsets.every((cell) => isFree(row + cell.dr, col + cell.dc))) {
            continue;
          }
          for (const cell of offsets) {
            const atRow = row + cell.dr;
            const atCol = col + cell.dc;
            if (visibleKeys.has(cell.row.image_key || "")) {
              take(atRow, atCol, cell.row);
            } else {
              reserve(atRow, atCol);
            }
          }
          placed = true;
          break;
        }
        if (!placed) {
          placeSingles(visible);
        }
      }
      starts.push(cells.length);
      anchorStarts.push(anchors.length);
    }
    const packed = { columns, totalRows: Math.max(maxRow, 1), cells, starts, anchors, anchorStarts };
    assignLinkHues(packed.cells);
    assignJoins(packed.cells);
    return packed;
  }

  function layoutCards(cards) {
    const cardPx = cardPixelSize();
    const columns = viewColumnCount();
    const packed = packMemo.packed;
    const from = packMemo.repackFrom;
    if (layoutActive()) {
      if (!packed || packed.viewColumns !== columns || from != null) {
        packMemo.packed = packProcesses(cards, columns);
      }
    } else if (!packed || packed.columns !== columns || from === 0) {
      packMemo.packed = packGrid(cards, columns, null);
    } else if (from != null) {
      packMemo.packed = packGrid(cards, columns, {
        columns,
        cells: packed.cells,
        starts: packed.starts,
        anchors: packed.anchors,
        anchorStarts: packed.anchorStarts,
        startCard: from,
      });
    }
    packMemo.repackFrom = null;
    const laid = packMemo.packed;
    const baseHeight = rowStride || (cardPx + 20);
    const rowOffsets = [0];
    for (let index = 0; index < laid.totalRows; index += 1) {
      rowOffsets.push((index + 1) * baseHeight);
    }
    return { cardPx, baseHeight, rowOffsets, ...laid };
  }

  // The process / branch layout: every card at the row and column static/process_layout.js gives it. Cards that are
  // approved or rejected count as done; the layout drops finished processes. The grid may be wider than the screen.
  function packProcesses(cards, viewColumns) {
    const input = cards.map((windows) => {
      const row = windows[0];
      const radius = Number(row.radius);
      return {
        id: String(row.location_id),
        z: Number(row.z) || 0,
        area: Number.isFinite(radius) ? Math.PI * radius * radius : 0,
        done: reviewClass(row) !== "unknown",
      };
    });
    const laid = window.ProcessLayout.layoutProcesses({ cards: input, links: state.convLinks, maxColumns: viewColumns });
    const byId = new Map(cards.map((windows) => [String(windows[0].location_id), windows[0]]));
    const cells = laid.cells.map((cell) => ({ row: byId.get(cell.id), gr: cell.row, gc: cell.col, group: cell.group }));
    const packed = {
      columns: Math.max(laid.columns, viewColumns),
      viewColumns,
      totalRows: Math.max(laid.totalRows, 1),
      cells,
      starts: [0],
      anchors: [],
      anchorStarts: [0],
    };
    assignLinkHues(packed.cells);
    assignJoins(packed.cells);
    return packed;
  }

  // Why the grid is empty when locations were asked for: not in this volume, or hidden by another filter.
  function locationMissNote() {
    const text = els.locate.value.trim();
    const ids = parseLocationIds(text);
    if (!ids) {
      return "";
    }
    if (!ids.size) {
      return `"${text}" has no location id`;
    }
    const known = state.rows.some((row) => ids.has(Number(row.location_id)));
    return known
      ? `location ${text} is hidden by the other filters`
      : `location ${text} is not in this volume`;
  }

  // The Location box is a filter: only the listed locations stay. Start at the top of the (short) list.
  function showLocations() {
    els.scroller.scrollTop = 0;
    render(true);
  }

  // How many cards fit across the screen: the scroller's width less the grid's 16px margins. The grid itself can be
  // wider than that (the process layout), so its own width says nothing about the screen.
  function viewColumnCount() {
    return columnCount(Math.max(0, els.scroller.clientWidth - 32));
  }

  function columnCount(width) {
    const card = cardPixelSize();
    if (width <= 0) {
      return 1;
    }
    return Math.max(1, Math.floor((width + GRID_GAP) / (card + GRID_GAP)));
  }

  function cardKey(row) {
    return `${row.location_id}|${row.image_key || ""}`;
  }

  function reconcileGrid(visible, rowOffset) {
    const wanted = new Map();
    for (const cell of visible) {
      wanted.set(cardKey(cell.row), cell);
    }
    const existing = new Map();
    els.grid.querySelectorAll(".card").forEach((node) => {
      existing.set(`${node.dataset.id}|${node.dataset.key || ""}`, node);
    });
    for (const [key, node] of existing) {
      if (!wanted.has(key)) {
        node.remove();
      }
    }
    const fresh = [];
    for (const cell of visible) {
      const key = cardKey(cell.row);
      const column = String(cell.gc + 1);
      const row = String(cell.gr - rowOffset + 1);
      let node = existing.get(key);
      if (!node) {
        const holder = document.createElement("div");
        holder.innerHTML = cellHtml(cell, rowOffset);
        node = holder.firstElementChild;
        els.grid.appendChild(node);
        fresh.push(node);
        continue;
      }
      node.hidden = false;
      node.classList.toggle("join-n", Boolean(cell.joinN));
      node.classList.toggle("join-e", Boolean(cell.joinE));
      node.classList.toggle("join-s", Boolean(cell.joinS));
      node.classList.toggle("join-w", Boolean(cell.joinW));
      node.classList.toggle("solo", !(cell.joinN || cell.joinE || cell.joinS || cell.joinW));
      node.style.gridColumn = column;
      node.style.gridRow = row;
      node.style.setProperty("--link-h", String(cell.linkH));
      node.style.setProperty("--link-s", `${cell.linkS}%`);
      node.style.setProperty("--link-l", `${cell.linkL}%`);
    }
    applyLinkHot(linkHotId);
    if (!fresh.length) {
      return;
    }
    const token = ++paintToken;
    fresh.forEach((node) => {
      const composite = node.querySelector(".composite");
      if (composite) {
        paintComposite(composite, token);
      }
    });
    reprioritize();
  }

  function render(force = true, reuse = false) {
    if (force && filtersCollapsed()) {
      updateFilterSummary();
    }
    const scroll = els.scroller.scrollTop;
    const viewH = els.scroller.clientHeight || 1;
    const page = viewH;
    const covers = !force
      && windowTop >= 0
      && (windowTop === 0 || scroll - windowTop >= page * 0.5)
      && windowBottom - (scroll + viewH) >= page * 0.5;
    if (covers) {
      return;
    }
    const cards = ensureVisibleCards();
    const layout = layoutCards(cards);
    const { cardPx, columns, baseHeight, totalRows, rowOffsets, cells } = layout;
    els.grid.style.setProperty("--grid-gap", `${GRID_GAP}px`);
    els.grid.style.gridTemplateColumns = `repeat(${columns}, ${cardPx}px)`;
    // The process layout can be wider than the screen: give the grid its full width so the scroller gets a
    // horizontal scrollbar. Other layouts fit the screen.
    const fullWidth = columns * cardPx + (columns - 1) * GRID_GAP;
    const wide = layoutActive() && columns > viewColumnCount();
    els.grid.style.right = wide ? "auto" : "";
    els.grid.style.width = wide ? `${fullWidth}px` : "";
    els.spacer.style.width = wide ? `${fullWidth + 32}px` : "";
    els.grid.style.gridAutoRows = `${Math.max(1, baseHeight - GRID_GAP)}px`;
    els.spacer.style.height = `${rowOffsets[rowOffsets.length - 1] || baseHeight}px`;
    const missNote = cards.length ? "" : locationMissNote();
    const notes = [state.filterNote, missNote].filter(Boolean).join(" · ");
    els.status.textContent = `${cards.length} location(s) · ${state.permission}${notes ? ` · ${notes}` : ""}`;
    syncVisibilityLabel();
    syncReverseButtons();
    const rangeTop = Math.max(0, scroll - page);
    const rangeBottom = scroll + viewH + page;
    let first = 0;
    while (first < totalRows && rowOffsets[first + 1] <= rangeTop) {
      first += 1;
    }
    let last = first;
    while (last < totalRows && rowOffsets[last] < rangeBottom) {
      last += 1;
    }
    windowTop = rowOffsets[first] || 0;
    windowBottom = rowOffsets[last] || windowTop + baseHeight;
    els.grid.style.top = `${windowTop}px`;
    const visible = cells.filter((cell) => cell.gr >= first && cell.gr < last);
    if (reuse) {
      reconcileGrid(visible, first);
    } else {
      els.grid.innerHTML = visible.map((cell) => cellHtml(cell, first)).join("");
      paintVisible();
    }
    applyLinkHot(linkHotId);
    const card = els.grid.querySelector(".card");
    if (!card) {
      return;
    }
    const measured = card.offsetHeight + GRID_GAP;
    if (!measuringRow && measured > 1 && Math.abs(measured - rowStride) > 1) {
      measuringRow = true;
      rowStride = measured;
      windowTop = -1;
      render(true, reuse);
      measuringRow = false;
    }
  }

  function scheduleRender() {
    window.clearTimeout(renderTimer);
    renderTimer = window.setTimeout(render, 120);
  }

  // catalog-latest-begin
  // Hands out one "still current?" check per request: only the newest request, while its own key still holds,
  // may apply its response. An older response for another volume or training set is dropped.
  function makeLatest() {
    let newest = 0;
    return (keyHolds) => {
      const mine = ++newest;
      return () => mine === newest && keyHolds();
    };
  }
  // catalog-latest-end

  const catalogLatest = makeLatest();

  async function loadCatalog(applyDefaultSort = false, preserveViewer = false) {
    const { volume, trainingSet } = state;
    const isLatest = catalogLatest(() => state.volume === volume && state.trainingSet === trainingSet);
    if (!volume) {
      state.rows = [];
      indexCatalog();
      render();
      return;
    }
    let payload;
    try {
      payload = await getJson(withSet(`/api/volumes/${encodeURIComponent(volume)}/catalog`));
    } catch (error) {
      if (!isLatest()) {
        return;
      }
      throw error;
    }
    if (!isLatest()) {
      return;
    }
    state.rows = payload.rows || [];
    if (state.converter) {
      try {
        const cards = await getJson(withSet(`/api/volumes/${encodeURIComponent(volume)}/converter/cards`));
        if (!isLatest()) {
          return;
        }
        window.ConverterMode.attachCards(state.rows, cards.cards);
        state.convLinks = cards.links || [];
        state.layoutEpoch += 1;
      } catch {
        // The cards are extra detail; the grid works without them.
      }
      const batchLink = document.getElementById("converter-batch");
      if (batchLink) {
        batchLink.href = window.ConverterMode.batchHref(trainingSet, volume);
        batchLink.hidden = false;
      }
      refreshBuilds(trainingSet, volume).catch(() => {});
      state.convCards = [];
      try {
        const all = await getJson(withSet(`/api/volumes/${encodeURIComponent(volume)}/converter/cards`));
        state.convCards = all.cards || [];
      } catch {
        // counts are extra detail
      }
      const failed = state.convCards.filter((card) => card.state === "failed");
      const failedButton = document.getElementById("converter-failed");
      failedButton.hidden = failed.length === 0;
      failedButton.textContent = `Failed (${failed.length})`;
    }
    if (state.gold) {
      await refreshGold(volume, trainingSet, isLatest);
      if (!isLatest()) {
        return;
      }
    }
    indexCatalog();
    if (!preserveViewer && viewer.locationId && !state.rows.some((row) => String(row.location_id) === viewer.locationId && visibleClasses().has(reviewClass(row)))) {
      closeViewer();
    }
    state.cropSize = Number(payload.cropSize) || 1024;
    state.permission = payload.permission || "read";
    document.body.classList.toggle("no-review", state.permission !== "review");
    state.hasSam2 = state.rows.some((row) =>
      row.sam2PredIou != null || row.sam2ObjectScore != null || row.sam2Stability != null || row.sam2GtIou != null
    );
    if (!state.hasSam2) {
      state.sorts = state.sorts.filter((item) => item.key !== "sam2GtIou");
      if (!state.sorts.length) {
        state.sorts = [{ key: "z", desc: false }];
      }
    }
    state.hasCardScores = !state.converter && state.rows.some((row) => row.cardIou != null || row.cardMismatch != null);
    if (!state.hasCardScores) {
      state.sorts = state.sorts.filter((item) => !CARD_SORT_WORST_DESC.has(item.key));
      if (!state.sorts.length) {
        state.sorts = [{ key: "z", desc: false }];
      }
    }
    if (applyDefaultSort) {
      // Worst model mismatch first when the cards have card scores, else the highest training loss, else by section.
      const hasLoss = state.rows.some((row) => row.loss != null);
      state.sorts = state.converter
        ? [{ key: LAYOUT_SORT, desc: false }]
        : (state.hasCardScores
          ? [{ key: "cardMismatch", desc: true }]
          : [{ key: hasLoss ? "loss" : "z", desc: hasLoss }]);
    }
    watchCatalogRescores();
    renderSortStack();
    render();
  }

  const undoStack = [];

  function findStatusRow(locationId, imageKey) {
    return windowsForLocation(locationId).find((row) => (row.image_key || "") === (imageKey || ""));
  }

  function applyLocalReview(row, act) {
    if (!row) {
      return;
    }
    if (act === "approve") {
      row.approved = 1;
      row.ignored = 0;
    } else if (act === "unapprove") {
      row.approved = 0;
    } else if (act === "ignore") {
      row.ignored = 1;
      row.approved = 0;
    } else if (act === "restore") {
      row.ignored = 0;
    }
  }

  async function postReview(act, locationId, imageKey) {
    await getJson(withSet(`/api/volumes/${encodeURIComponent(state.volume)}/${act}`), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ location_id: locationId, image_key: imageKey || undefined }),
    });
  }

  function rowsForLocation(locationId) {
    return windowsForLocation(locationId);
  }

  function rememberLocation(locationId) {
    const rows = rowsForLocation(locationId);
    if (!rows.length) {
      return;
    }
    undoStack.push({
      locationId: Number(locationId),
      windows: rows.map((row) => ({
        imageKey: row.image_key || "",
        ignored: Boolean(row.ignored),
        approved: Boolean(row.approved),
      })),
    });
    if (undoStack.length > 50) {
      undoStack.shift();
    }
  }

  function paintReviewNow(locationId) {
    const id = String(locationId);
    const classes = visibleClasses();
    els.grid.querySelectorAll(".card").forEach((node) => {
      if (node.dataset.id !== id) {
        return;
      }
      const row = findStatusRow(locationId, node.dataset.key || "");
      if (!row || !classes.has(reviewClass(row))) {
        node.remove();
        return;
      }
      const approved = Boolean(row.approved) && !row.ignored;
      const check = node.querySelector("button.icon.check");
      if (!check) {
        return;
      }
      check.classList.toggle("on", approved);
      check.dataset.act = approved ? "unapprove" : "approve";
      const label = approved ? "Approved" : "Approve";
      check.title = label;
      check.setAttribute("aria-label", label);
      check.setAttribute("aria-pressed", approved ? "true" : "false");
    });
  }

  function afterPaint() {
    return new Promise((resolve) => {
      requestAnimationFrame(() => requestAnimationFrame(resolve));
    });
  }

  async function mutate(act, locationId) {
    if (act === "ignore" || act === "restore") {
      layerEpoch += 1;
    }
    const rows = rowsForLocation(locationId);
    rememberLocation(locationId);
    const snapshot = rows.map((row) => ({
      imageKey: row.image_key || "",
      approved: row.approved,
      ignored: row.ignored,
    }));
    for (const row of rows) {
      applyLocalReview(row, act);
    }
    const classes = visibleClasses();
    const stays = !layoutActive() && rows.some((row) => classes.has(reviewClass(row)));
    if (layoutActive()) {
      // A finished process leaves the layout and the rest moves up, so the whole layout is redone.
      state.layoutEpoch += 1;
      resetVisibleCards();
    } else if (!stays) {
      forgetVisibleLocation(locationId);
    }
    paintReviewNow(locationId);
    const save = postReview(act, locationId, "");
    if (!stays) {
      await afterPaint();
      render(true, true);
    }
    try {
      await save;
    } catch (error) {
      if (rows.length) {
        undoStack.pop();
      }
      for (const item of snapshot) {
        const row = findStatusRow(locationId, item.imageKey);
        if (row) {
          row.approved = item.approved;
          row.ignored = item.ignored;
        }
      }
      resetVisibleCards();
      render(true, true);
      throw error;
    }
  }

  // Approve or reject every card that is visible now, after one confirmation that lists the count. Cards that
  // are hidden by a filter, including locations already in the other state, are never touched.
  async function reviewVisible(act) {
    const approving = act === "approve";
    const wanted = approving ? "approved" : "rejected";
    const ids = ensureVisibleCards()
      // The process layout shows every card, so a batch there only touches cards nobody has decided yet.
      .filter((windows) => (layoutActive()
        ? windows.every((row) => reviewClass(row) === "unknown")
        : windows.some((row) => reviewClass(row) !== wanted)))
      .map((windows) => windows[0].location_id);
    if (!ids.length) {
      els.status.textContent = `Nothing to ${approving ? "accept" : "reject"}: every visible card is already ${wanted}.`;
      return;
    }
    const verb = approving ? "Accept" : "Reject";
    const noun = ids.length === 1 ? "annotation" : "annotations";
    if (!window.confirm(`${verb} ${ids.length} visible ${noun}? Hidden annotations are not changed.`)) {
      return;
    }
    els.status.textContent = `${approving ? "Accepting" : "Rejecting"} ${ids.length} ${noun}...`;
    const payload = await getJson(withSet(`/api/volumes/${encodeURIComponent(state.volume)}/review-batch`), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: approving ? "approve" : "ignore", location_ids: ids }),
    });
    for (const locationId of ids) {
      for (const row of windowsForLocation(locationId)) {
        applyLocalReview(row, act);
      }
    }
    undoStack.length = 0;
    layerEpoch += 1;
    resetVisibleCards();
    render(true, true);
    els.status.textContent = `${approving ? "Accepted" : "Rejected"} ${payload.count} ${noun}.`;
  }

  async function applyRememberedStatus(entry) {
    const windows = entry.windows || [{
      imageKey: entry.imageKey || "",
      ignored: Boolean(entry.ignored),
      approved: Boolean(entry.approved),
    }];
    const locationId = entry.locationId;
    const first = windows[0];
    const uniform = windows.every((item) => item.ignored === first.ignored && item.approved === first.approved);
    if (uniform && first) {
      if (first.ignored) {
        await postReview("ignore", locationId, "");
      } else {
        await postReview("restore", locationId, "");
        await postReview(first.approved ? "approve" : "unapprove", locationId, "");
      }
    } else {
      for (const item of windows) {
        if (item.ignored) {
          await postReview("ignore", locationId, item.imageKey);
        } else {
          await postReview("restore", locationId, item.imageKey);
          await postReview(item.approved ? "approve" : "unapprove", locationId, item.imageKey);
        }
      }
    }
    for (const item of windows) {
      const row = findStatusRow(locationId, item.imageKey);
      if (row) {
        row.ignored = item.ignored ? 1 : 0;
        row.approved = item.approved ? 1 : 0;
      }
    }
    resetVisibleCards();
    render(true, true);
  }

  async function undoLastStatus() {
    const entry = undoStack.pop();
    if (!entry) {
      els.status.textContent = "Nothing to undo.";
      return;
    }
    await applyRememberedStatus(entry);
    els.status.textContent = `Undid the last status change for location ${entry.locationId}.`;
  }

  function layoutViewerFrame(pixelW, pixelH) {
    const closeWidth = els.viewerClose.offsetWidth || 56;
    const approveWidth = document.getElementById("viewer-approve").offsetWidth || 56;
    const rejectWidth = document.getElementById("viewer-reject").offsetWidth || 56;
    const maxW = Math.max(1, els.viewerStage.clientWidth - closeWidth - approveWidth - rejectWidth - 96 - 32);
    const maxH = els.viewerStage.clientHeight;
    const scale = Math.min(maxW / pixelW, maxH / pixelH);
    els.viewerFrame.style.width = `${Math.max(1, Math.floor(pixelW * scale))}px`;
    els.viewerFrame.style.height = `${Math.max(1, Math.floor(pixelH * scale))}px`;
    placeViewerInfo();
  }

  function infoHtml(row) {
    const structureName = row.type_name || "Structure";
    const structureText = row.structure_id == null ? structureName : `${structureName} ${row.structure_id}`;
    const structureHref = sbfsemHref(row);
    const structure = structureHref
      ? `<a href="${structureHref}" target="_blank" rel="noreferrer">${structureText}</a>`
      : structureText;
    const lines = [
      `<a class="viking-link" href="${vikingHref(row)}">#${row.location_id}</a>`,
      `Z ${row.z}`,
      structure,
    ];
    if (row.structure_label) {
      lines.push(row.structure_label);
    }
    if (row.loss != null) {
      lines.push(state.converter ? `change ${Number(row.loss).toFixed(2)}` : `loss ${Number(row.loss).toFixed(3)}`);
    }
    if (row.sam2GtIou != null) {
      lines.push(`SAM2 ${Number(row.sam2GtIou).toFixed(2)}`);
    }
    if (!state.converter && (row.cardIou != null || row.cardMismatch != null)) {
      const old = row.cardOld || null;
      const hasOld = Boolean(old);
      lines.push(`IoU ${scorePair(old && old.iou, row.cardIou, hasOld)} · core ${scorePair(old && old.core, row.cardCore, hasOld)}`
        + ` · spill ${scorePair(old && old.spill, row.cardSpill, hasOld)}`);
    }
    if (state.converter && row.conv) {
      lines.push(...window.ConverterMode.infoLines(row.conv));
    }
    if (state.gold && row.gold) {
      lines.push(...window.GoldMode.infoLines(row.gold));
    }
    if (viewer.negatives) {
      // In the converter the existing annotation is one of the regions: it is red, the proposal's hue, both yellow.
      const existing = state.converter ? 1 : 0;
      const count = viewer.negatives.count - existing;
      const parts = [];
      if (existing) {
        parts.push("existing annotation in red");
      }
      if (count > 0) {
        parts.push(`${count} negative annotation${count === 1 ? "" : "s"}`);
      }
      lines.push(`${parts.join(", ")} <span class="neg-state">(N: ${negativesOn() ? "hide" : "show"})</span>`);
    }
    return lines.map((line) => `<div>${line}</div>`).join("");
  }

  function placeViewerInfo() {
    const panel = document.getElementById("viewer-info");
    const row = viewerRow();
    if (!panel || !row) {
      if (panel) {
        panel.hidden = true;
      }
      return;
    }
    panel.innerHTML = infoHtml(row);
    const approveWidth = document.getElementById("viewer-approve").offsetWidth || 112;
    const rejectWidth = document.getElementById("viewer-reject").offsetWidth || 112;
    const closeWidth = els.viewerClose.offsetWidth || 56;
    const used = els.viewerFrame.offsetWidth + approveWidth + rejectWidth + closeWidth + 96;
    panel.hidden = els.viewerStage.clientWidth - used < 220;
  }

  function flashViewerAct(id) {
    const button = document.getElementById(id);
    if (!button) {
      return Promise.resolve();
    }
    button.classList.remove("flash");
    void button.offsetWidth;
    button.classList.add("flash");
    return new Promise((resolve) => {
      window.setTimeout(() => {
        button.classList.remove("flash");
        resolve();
      }, 250);
    });
  }

  function viewerCells() {
    return layoutCards(ensureVisibleCards()).cells;
  }

  function viewerCellIndex(cells) {
    return cells.findIndex((cell) =>
      String(cell.row.location_id) === String(viewer.locationId)
      && (cell.row.image_key || "") === (viewer.imageKey || "")
    );
  }

  function stepViewer(delta) {
    if (els.viewer.hidden) {
      return;
    }
    const cells = viewerCells();
    const index = viewerCellIndex(cells);
    const next = index >= 0 ? cells[index + delta] : null;
    if (!next) {
      return;
    }
    openViewerFromRow(next.row);
  }

  // On unless the viewer turned it off. Remembered in the browser when storage works; the variable keeps the
  // toggle working when it does not.
  let negativesEnabled = (() => {
    try {
      return localStorage.getItem(NEG_KEY) !== "off";
    } catch {
      return true;
    }
  })();

  function negativesOn() {
    return negativesEnabled;
  }

  function negativesShown() {
    return Boolean(viewer.negatives) && negativesOn();
  }

  // The negatives layer for one location: the tile with the other annotations tinted red, and a yellow patch where
  // they cover the target mask. Reuses the foreground tint so red and blue read as one system. Pure of viewer state.
  // `negMask` is the server's 0/255 bitmask, `target` the location's own mask image.
  function composeNegatives(image, negMask, target, count) {
    const { width, height } = imageSize(image);
    const readMask = (source) => {
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      const context = canvas.getContext("2d", { willReadFrequently: true });
      context.drawImage(source, 0, 0, width, height);
      return context.getImageData(0, 0, width, height).data;
    };
    const tinted = (maskPixels, hex, visibility, keepOutside) => {
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      const context = canvas.getContext("2d", { willReadFrequently: true });
      context.drawImage(image, 0, 0, width, height);
      const imageData = context.getImageData(0, 0, width, height);
      tintMaskedPixels(imageData.data, maskPixels, hex, visibility, keepOutside);
      context.putImageData(imageData, 0, 0);
      return canvas;
    };
    const negative = readMask(negMask);
    const base = tinted(negative, NEG_HEX, NEG_VISIBILITY, true);
    let overlap = null;
    if (target) {
      const own = readMask(target);
      const both = new Uint8ClampedArray(negative.length);
      let covered = false;
      for (let index = 0; index < both.length; index += 4) {
        if (negative[index] > 127 && own[index] > 127) {
          both[index] = 255;
          covered = true;
        }
      }
      overlap = covered ? tinted(both, NEG_OVERLAP_HEX, NEG_VISIBILITY, false) : null;
    }
    return { count, base, overlap };
  }

  // Fetches the other annotations in the open location's tile and shows them. Nothing is drawn, and no badge is
  // added, when the tile has none or the request fails. Safe to drop on the floor if another location opens first.
  async function loadViewerNegatives(token, image, target) {
    const imageKey = viewer.imageKey;
    if (!imageKey) {
      return;
    }
    let negMask;
    let count;
    try {
      const response = await fetch(withSet(
        `/api/volumes/${encodeURIComponent(state.volume)}/negatives?key=${encodeURIComponent(imageKey)}&loc=${encodeURIComponent(viewer.locationId)}`,
      ));
      if (!response.ok) {
        return;
      }
      count = Number(response.headers.get("X-Negative-Count") || "0");
      if (!count) {
        return;
      }
      const url = URL.createObjectURL(await response.blob());
      try {
        negMask = await loadImage(url);
      } finally {
        URL.revokeObjectURL(url);
      }
    } catch {
      return;
    }
    if (token !== viewer.token) {
      return;
    }
    viewer.negatives = composeNegatives(image, negMask, target, count);
    paintViewer();
    placeViewerInfo();
  }

  function toggleNegatives() {
    negativesEnabled = !negativesEnabled;
    try {
      localStorage.setItem(NEG_KEY, negativesEnabled ? "on" : "off");
    } catch {
      // Not remembered across sessions; the toggle still works for this page.
    }
    paintViewer();
    placeViewerInfo();
    paintVisible();
  }

  function drawFull(image, overlay) {
    const width = image.naturalWidth;
    const height = image.naturalHeight;
    els.viewerImage.width = width;
    els.viewerImage.height = height;
    els.viewerMask.width = width;
    els.viewerMask.height = height;
    // Negatives only belong to the whole-tile view, not to the parts of a split location.
    const negatives = negativesShown() && image === viewer.image ? viewer.negatives : null;
    els.viewerImage.getContext("2d").drawImage(negatives ? negatives.base : image, 0, 0);
    const maskContext = els.viewerMask.getContext("2d");
    maskContext.clearRect(0, 0, width, height);
    if (overlay) {
      maskContext.drawImage(overlay, 0, 0);
    }
    if (negatives && negatives.overlap) {
      maskContext.drawImage(negatives.overlap, 0, 0);
    }
    if (viewer.original && image === viewer.image) {
      maskContext.drawImage(viewer.original, 0, 0);
    }
    layoutViewerFrame(width, height);
    if (retraceUi && retraceUi.polygon) {
      retraceUi.repaint();
    }
  }

  function paintViewer() {
    if (els.viewer.hidden) {
      return;
    }
    const parts = viewer.parts;
    const showParts = parts && parts.length > 1 && viewer.partIndex == null;
    els.viewerBack.hidden = viewer.partIndex == null;
    els.viewerGrid.hidden = !showParts;
    els.viewerCues.hidden = showParts;
    if (viewer.partIndex != null && parts) {
      const part = parts[viewer.partIndex];
      drawFull(part.image, part.overlay);
      return;
    }
    if (!showParts) {
      if (!viewer.image) {
        return;
      }
      drawFull(viewer.image, viewer.overlay);
      return;
    }
    const width = viewer.layoutWidth;
    const height = viewer.layoutHeight;
    els.viewerImage.width = width;
    els.viewerImage.height = height;
    els.viewerMask.width = width;
    els.viewerMask.height = height;
    const imageContext = els.viewerImage.getContext("2d");
    const maskContext = els.viewerMask.getContext("2d");
    imageContext.fillStyle = "#000";
    imageContext.fillRect(0, 0, width, height);
    maskContext.clearRect(0, 0, width, height);
    parts.forEach((part) => {
      imageContext.drawImage(part.image, part.x, part.y, part.width, part.height);
      if (part.overlay) {
        maskContext.drawImage(part.overlay, part.x, part.y, part.width, part.height);
      }
    });
    layoutViewerFrame(width, height);
    els.viewerGrid.replaceChildren();
    parts.forEach((part, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.index = String(index);
      button.style.left = `${(part.x / width) * 100}%`;
      button.style.top = `${(part.y / height) * 100}%`;
      button.style.width = `${(part.width / width) * 100}%`;
      button.style.height = `${(part.height / height) * 100}%`;
      button.setAttribute("aria-label", `Part ${index + 1}`);
      els.viewerGrid.appendChild(button);
    });
  }

  function closeViewer() {
    if (brushUi && brushUi.dirty && !window.confirm("Discard the unsaved painting on this mask?")) {
      return;
    }
    if (brushUi) {
      brushUi.reset();
      syncBrushButton();
    }
    if (retraceUi) {
      retraceUi.reset();
      syncRetraceButton();
    }
    viewer.token += 1;
    viewer.locationId = "";
    viewer.imageKey = "";
    viewer.partIndex = null;
    viewer.parts = null;
    viewer.image = null;
    viewer.overlay = null;
    setMaskHidden(false);
    els.viewer.hidden = true;
  }

  async function loadOptionalImage(url) {
    try {
      return await loadImage(url);
    } catch {
      return null;
    }
  }

  function blankViewer() {
    viewer.partIndex = null;
    viewer.parts = null;
    viewer.image = null;
    viewer.overlay = null;
    viewer.negatives = null;
    viewer.mask = null;
    viewer.original = null;
    if (brushUi) {
      brushUi.reset();
      syncBrushButton();
    }
    if (retraceUi) {
      retraceUi.reset();
      syncRetraceButton();
    }
    for (const canvas of [els.viewerImage, els.viewerMask]) {
      canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
    }
  }

  async function loadPart(part, locationId, includeIgnored) {
    const image = await loadImage(fileUrl(state.volume, `images/${part.imageKey}.png`));
    let mask = await loadOptionalImage(fileUrl(state.volume, `masks/${part.imageKey}_${locationId}.png`));
    if (!mask && includeIgnored) {
      mask = await loadOptionalImage(fileUrl(state.volume, `ignored/${part.imageKey}_${locationId}.png`));
    }
    if (!mask) {
      return null;
    }
    return {
      imageKey: part.imageKey,
      x: Number(part.x),
      y: Number(part.y),
      width: Number(part.width),
      height: Number(part.height),
      image,
      overlay: composeOverlay(image, mask, viewer.hue, viewer.visibility),
    };
  }

  async function loadParts(locationId) {
    let response;
    try {
      response = await fetch(fileUrl(state.volume, `overlays/parts/${locationId}.json`));
    } catch {
      return null;
    }
    if (!response.ok) {
      return null;
    }
    const payload = await response.json();
    const parts = Array.isArray(payload.parts) ? payload.parts : [];
    if (parts.length < 2) {
      return null;
    }
    const includeRejected = document.querySelector('#view-classes input[data-view="rejected"]').checked;
    const loaded = [];
    for (const part of parts) {
      try {
        const item = await loadPart(part, locationId, includeRejected);
        if (item) {
          loaded.push(item);
        }
      } catch {
        return null;
      }
    }
    if (loaded.length < 2) {
      return null;
    }
    return {
      width: Number(payload.width),
      height: Number(payload.height),
      parts: loaded,
    };
  }

  async function openViewer(node) {
    const token = viewer.token + 1;
    const locationId = node.dataset.loc;
    const imageKey = node.dataset.key || "";
    const maskUrl = node.dataset.mask;
    viewer.token = token;
    viewer.locationId = locationId;
    viewer.imageKey = imageKey;
    viewer.hue = node.dataset.hue;
    viewer.visibility = Number(node.dataset.visibility);
    blankViewer();
    els.viewer.hidden = false;
    if (document.activeElement && document.activeElement !== document.body) {
      document.activeElement.blur();
    }
    els.viewerBack.hidden = true;
    els.viewerGrid.hidden = true;
    let image;
    let mask;
    try {
      const zoomImage = imageKey
        ? fileUrl(state.volume, `images/${imageKey}.png`)
        : node.dataset.image;
      [image, mask] = await Promise.all([
        loadImage(zoomImage),
        loadImage(maskUrl),
      ]);
    } catch {
      if (token === viewer.token) {
        closeViewer();
        els.status.textContent = `Image missing for location ${locationId}.`;
      }
      return;
    }
    if (token !== viewer.token) {
      return;
    }
    viewer.image = image;
    viewer.mask = mask;
    viewer.overlay = composeOverlay(image, mask, viewer.hue, viewer.visibility);
    viewer.parts = null;
    viewer.partIndex = null;
    if (state.converter && imageKey) {
      // The annotation as it is in the database, drawn as a thin outline over the proposed mask.
      const original = await loadOptionalImage(fileUrl(state.volume, `original/${imageKey}_${locationId}.png`));
      if (token !== viewer.token) {
        return;
      }
      viewer.original = original ? window.ConverterMode.outlineCanvas(original, "#ffe14d") : null;
    }
    paintViewer();
    syncViewerActs();
    void loadViewerNegatives(token, image, mask);
  }

  function viewerRow() {
    return state.rows.find((row) =>
      String(row.location_id) === String(viewer.locationId)
      && (row.image_key || "") === (viewer.imageKey || "")
    );
  }

  function syncViewerActs() {
    const row = viewerRow();
    const approve = document.getElementById("viewer-approve");
    const reject = document.getElementById("viewer-reject");
    if (!row || !approve || !reject) {
      return;
    }
    const approved = Boolean(row.approved);
    approve.classList.toggle("on", approved);
    approve.dataset.act = approved ? "unapprove" : "approve";
    approve.title = approved ? "Approved" : "Approve";
    approve.setAttribute("aria-label", approve.title);
    const rejected = Boolean(row.ignored);
    reject.dataset.act = rejected ? "restore" : "ignore";
    reject.title = rejected ? "Undo reject" : "Reject";
    reject.setAttribute("aria-label", reject.title);
  }

  async function reviewFromViewer(act) {
    const cells = viewerCells();
    const index = viewerCellIndex(cells);
    const locationId = Number(viewer.locationId);
    let next = null;
    for (let cursor = index + 1; cursor < cells.length; cursor += 1) {
      if (Number(cells[cursor].row.location_id) !== locationId) {
        next = cells[cursor].row;
        break;
      }
    }
    await mutate(act, locationId);
    if (!next) {
      closeViewer();
      return;
    }
    const row = state.rows.find((item) =>
      String(item.location_id) === String(next.location_id)
      && (item.image_key || "") === (next.image_key || "")
    );
    if (!row || !visibleClasses().has(reviewClass(row))) {
      closeViewer();
      return;
    }
    await openViewerFromRow(row);
  }

  function openViewerFromRow(row) {
    const node = {
      dataset: {
        loc: String(row.location_id),
        key: row.image_key || "",
        mask: mediaUrl(state.volume, row.mask_relpath, row.location_id),
        hue: overlayHue(),
        visibility: String(overlayVisibility()),
        image: mediaUrl(state.volume, tissueRelpath(row), row.location_id),
      },
    };
    return openViewer(node);
  }

  let linkHotId = "";

  function applyLinkHot(locationId) {
    if (!locationId) {
      return;
    }
    linkHotId = String(locationId);
    els.grid.querySelectorAll(".card").forEach((node) => {
      node.classList.toggle("link-hot", node.dataset.id === linkHotId);
    });
  }

  function clearLinkHot() {
    linkHotId = "";
    els.grid.querySelectorAll(".card.link-hot").forEach((node) => node.classList.remove("link-hot"));
  }

  els.grid.addEventListener("pointerover", (event) => {
    const card = event.target.closest(".card");
    if (!card || card.dataset.id === linkHotId) {
      return;
    }
    applyLinkHot(card.dataset.id);
  });

  els.grid.addEventListener("pointerout", (event) => {
    const card = event.target.closest(".card");
    if (!card) {
      return;
    }
    const next = event.relatedTarget;
    const nextCard = next && next.closest ? next.closest(".card") : null;
    if (nextCard && nextCard.dataset.id === card.dataset.id && els.grid.contains(nextCard)) {
      return;
    }
    clearLinkHot();
  });

  // The trend popover closes on a click outside it, on Escape, and when the grid scrolls.
  document.addEventListener("click", (event) => {
    if (!event.target.closest(".trend-pop") && !event.target.closest(".trend-btn")) {
      closeTrend();
    }
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeTrend();
    }
  });
  els.grid.addEventListener("scroll", closeTrend, { passive: true });

  els.grid.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-act]");
    if (button) {
      if (button.dataset.act === "refresh") {
        event.preventDefault();
        event.stopPropagation();
        refreshLocation(Number(button.dataset.id)).catch((error) => {
          els.status.textContent = error.message;
        });
        return;
      }
      if (button.dataset.act === "trend") {
        event.preventDefault();
        event.stopPropagation();
        openTrend(button);
        return;
      }
      if (button.dataset.act === "rescore") {
        event.preventDefault();
        event.stopPropagation();
        queueRescore(Number(button.dataset.id)).catch((error) => {
          els.status.textContent = `Could not queue a re-score: ${error.message}`;
        });
        return;
      }
      mutate(button.dataset.act, Number(button.dataset.id)).catch((error) => {
        els.status.textContent = error.message;
      });
      return;
    }
    if (event.target.closest("a")) {
      return;
    }
    const composite = event.target.closest(".composite")
      || event.target.closest(".card")?.querySelector(".composite");
    if (composite) {
      openViewer(composite);
    }
  });

  document.getElementById("viewer-prev").addEventListener("click", () => stepViewer(-1));
  document.getElementById("viewer-next").addEventListener("click", () => stepViewer(1));
  document.addEventListener("click", (event) => {
    const link = event.target.closest("a.viking-link");
    if (!link) {
      return;
    }
    event.preventDefault();
    const frame = document.createElement("iframe");
    frame.hidden = true;
    frame.src = link.href;
    document.body.appendChild(frame);
    window.setTimeout(() => frame.remove(), 1500);
  });
  document.getElementById("viewer-refresh").addEventListener("click", () => {
    refreshLocation(Number(viewer.locationId)).catch((error) => {
      els.status.textContent = error.message;
    });
  });
  document.getElementById("viewer-approve").addEventListener("click", () => {
    const act = document.getElementById("viewer-approve").dataset.act || "approve";
    reviewFromViewer(act).catch((error) => {
      els.status.textContent = error.message;
    });
  });
  document.getElementById("viewer-reject").addEventListener("click", () => {
    const act = document.getElementById("viewer-reject").dataset.act || "ignore";
    reviewFromViewer(act).catch((error) => {
      els.status.textContent = error.message;
    });
  });

  els.viewerClose.addEventListener("click", closeViewer);
  if (retraceButton) {
    retraceButton.addEventListener("click", toggleRetrace);
  }
  if (brushButton) {
    brushButton.addEventListener("click", toggleBrush);
  }
  const retraceOk = document.getElementById("retrace-notice-ok");
  if (retraceOk) {
    retraceOk.addEventListener("click", () => {
      document.getElementById("retrace-notice").hidden = true;
    });
  }
  els.viewerBack.addEventListener("click", () => {
    viewer.partIndex = null;
    paintViewer();
  });
  els.viewerGrid.addEventListener("click", (event) => {
    const button = event.target.closest("button");
    if (!button || !viewer.parts) {
      return;
    }
    viewer.partIndex = Number(button.dataset.index);
    paintViewer();
  });
  // Whether the viewer hides the mask. Space toggles it; the Hide and Show buttons on the image set it. The pointer's
  // position on the image does nothing.
  let maskHidden = false;

  function applyMaskVisibility() {
    els.viewerMask.style.opacity = maskHidden ? "0" : "1";
    els.viewerCues.querySelector(".show").classList.toggle("active", !maskHidden);
    els.viewerCues.querySelector(".hide").classList.toggle("active", maskHidden);
  }

  function setMaskHidden(hidden) {
    maskHidden = hidden;
    applyMaskVisibility();
  }

  els.viewerCues.addEventListener("click", (event) => {
    const cue = event.target.closest(".viewer-cue");
    if (cue) {
      setMaskHidden(cue.dataset.side === "hide");
    }
  });
  function zoomKeyTarget(event) {
    const tag = event.target && event.target.tagName;
    return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
  }

  function isSpaceKey(event) {
    return event.code === "Space" || event.key === " ";
  }

  let gridSpaceHidesMask = false;

  function setGridMasksHidden(hidden) {
    gridSpaceHidesMask = hidden;
    document.body.classList.toggle("grid-hide-masks", hidden);
  }

  window.addEventListener("keydown", (event) => {
    if (!els.viewer.hidden && (event.key === "Escape" || event.key === "Esc" || event.code === "Escape")) {
      event.preventDefault();
      closeViewer();
      return;
    }
    if (
      !els.viewer.hidden
      && (event.key === "n" || event.key === "N")
      && !event.ctrlKey && !event.metaKey && !event.altKey
      && !zoomKeyTarget(event)
    ) {
      event.preventDefault();
      toggleNegatives();
      return;
    }
    if (
      !els.viewer.hidden
      && (event.key === "r" || event.key === "R")
      && !event.ctrlKey && !event.metaKey && !event.altKey
      && !zoomKeyTarget(event)
    ) {
      event.preventDefault();
      toggleRetrace();
      return;
    }
    if (
      !els.viewer.hidden
      && (event.key === "b" || event.key === "B")
      && !event.ctrlKey && !event.metaKey && !event.altKey
      && !zoomKeyTarget(event)
    ) {
      event.preventDefault();
      toggleBrush();
      return;
    }
    if (!isSpaceKey(event) || zoomKeyTarget(event)) {
      return;
    }
    if (els.viewer.hidden) {
      event.preventDefault();
      if (!event.repeat) {
        setGridMasksHidden(true);
      }
      return;
    }
    event.preventDefault();
    if (event.repeat) {
      return;
    }
    // A tap used to hide the mask only until keyup, which fired before the
    // change was visible. Space toggles and stays until the next press.
    setMaskHidden(!maskHidden);
  }, true);
  window.addEventListener("keyup", (event) => {
    if (!isSpaceKey(event)) {
      return;
    }
    if (gridSpaceHidesMask) {
      event.preventDefault();
      setGridMasksHidden(false);
      return;
    }
    if (els.viewer.hidden) {
      return;
    }
    event.preventDefault();
  }, true);
  window.addEventListener("blur", () => {
    setGridMasksHidden(false);
  });
  window.addEventListener("keydown", (event) => {
    if (!els.viewer.hidden && (event.key === "ArrowLeft" || event.key === "ArrowRight" || event.key === "ArrowUp" || event.key === "ArrowDown")) {
      const tag = event.target && event.target.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") {
        return;
      }
      event.preventDefault();
      if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
        stepViewer(event.key === "ArrowRight" ? 1 : -1);
        return;
      }
      const approve = event.key === "ArrowUp";
      flashViewerAct(approve ? "viewer-approve" : "viewer-reject")
        .then(() => reviewFromViewer(approve ? "approve" : "ignore"))
        .catch((error) => {
          els.status.textContent = error.message;
        });
    }
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z" && !event.shiftKey) {
      const tag = event.target && event.target.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || event.target.isContentEditable) {
        return;
      }
      event.preventDefault();
      undoLastStatus().catch((error) => {
        els.status.textContent = error.message;
      });
    }
  });

  els.filters.addEventListener("click", (event) => {
    const sizeButton = event.target.closest("[data-card-size]");
    if (sizeButton) {
      selectCardSize(sizeButton.dataset.cardSize);
      return;
    }
    const button = event.target.closest("button.reverse");
    if (!button) {
      return;
    }
    const key = button.dataset.sort;
    const primary = state.sorts[0];
    if (primary && primary.key === key) {
      primary.desc = !primary.desc;
    } else {
      state.sorts[0] = { key, desc: true };
    }
    renderSortStack();
    render();
  });

  els.sortStack.addEventListener("click", (event) => {
    const add = event.target.closest("#sort-add");
    if (add) {
      const used = new Set(state.sorts.map((item) => item.key));
      const next = sortFields().find(([value]) => value !== LAYOUT_SORT && !used.has(value));
      if (next) {
        state.sorts.push({ key: next[0], desc: CARD_SORT_WORST_DESC.get(next[0]) || false });
        renderSortStack();
        render();
      }
      return;
    }
    const remove = event.target.closest(".sort-remove");
    if (remove) {
      state.sorts.splice(Number(remove.dataset.index), 1);
      renderSortStack();
      render();
      return;
    }
    const direction = event.target.closest(".sort-dir");
    if (direction) {
      const criterion = state.sorts[Number(direction.dataset.index)];
      criterion.desc = !criterion.desc;
      renderSortStack();
      render();
    }
  });
  els.sortStack.addEventListener("change", (event) => {
    const select = event.target.closest("select");
    if (!select) {
      return;
    }
    if (select.value === LAYOUT_SORT) {
      state.sorts = [{ key: LAYOUT_SORT, desc: false }];
    } else if (layoutActive()) {
      state.sorts = [{ key: select.value, desc: false }];
    } else {
      const criterion = state.sorts[Number(select.dataset.index)];
      criterion.key = select.value;
      if (CARD_SORT_WORST_DESC.has(select.value)) {
        criterion.desc = CARD_SORT_WORST_DESC.get(select.value);
      }
    }
    renderSortStack();
    render();
  });
  // A searched id the volume does not have is looked up on the server. If OData has it and the crop images it
  // needs are on disk, the server imports it and the catalog is reloaded.
  const locateCheck = { timer: 0, active: new Set(), hideTimer: 0 };

  function showLocateStatus(text, kind) {
    window.clearTimeout(locateCheck.hideTimer);
    const bar = document.getElementById("locate-status");
    bar.hidden = !text;
    bar.textContent = text;
    bar.className = `locate-status ${kind || ""}`.trim();
    if (text && kind !== "busy" && kind !== "error") {
      locateCheck.hideTimer = window.setTimeout(() => showLocateStatus(""), 5000);
    }
  }

  function scheduleLocateCheck() {
    window.clearTimeout(locateCheck.timer);
    const ids = parseLocationIds(els.locate.value);
    const missing = ids ? [...ids].filter((id) => !state.rows.some((row) => Number(row.location_id) === id)) : [];
    if (!missing.length || state.permission !== "review") {
      if (!locateCheck.active.size) {
        showLocateStatus("");
      }
      return;
    }
    locateCheck.timer = window.setTimeout(() => checkMissingLocations(missing), 600);
  }

  async function checkMissingLocations(missing) {
    const todo = missing.filter((id) => !locateCheck.active.has(id));
    if (!todo.length) {
      return;
    }
    showLocateStatus("Checking server", "busy");
    const notes = [];
    let imported = false;
    for (const id of todo) {
      locateCheck.active.add(id);
      try {
        const result = await getJson(`/api/volumes/${encodeURIComponent(state.volume)}/import-location`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ location_id: id }),
        });
        if (result.status === "imported") {
          imported = true;
          notes.push(`${id} imported`);
        } else if (result.status === "not_found") {
          notes.push(`${id} is not on the server`);
        } else if (result.status === "no_tiles") {
          notes.push(`${id} found, but missing image tiles (${result.reason})`);
        } else if (result.status === "circle") {
          notes.push(`${id} is a circle; circle file updated`);
        } else if (result.status === "busy") {
          notes.push(`${id}: ${result.reason}`);
        }
      } catch (error) {
        notes.push(`${id}: ${error.message}`);
      } finally {
        locateCheck.active.delete(id);
      }
    }
    if (imported) {
      await loadCatalog(false, true);
    }
    const failed = notes.some((note) => !note.endsWith("imported"));
    showLocateStatus(notes.join("; "), failed ? "error" : "");
  }

  els.locate.addEventListener("input", () => {
    showLocations();
    scheduleLocateCheck();
  });
  els.locate.addEventListener("change", showLocations);
  els.locate.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      showLocations();
      scheduleLocateCheck();
    }
  });

  ["input", "change"].forEach((name) => {
    els.type.addEventListener(name, render);
    els.label.addEventListener(name, render);
    els.z.addEventListener(name, render);
    els.radius.addEventListener(name, render);
  });
  document.querySelectorAll("#view-classes input").forEach((input) => {
    input.addEventListener("change", render);
  });
  ["input", "change"].forEach((name) => {
    els.tags.addEventListener(name, () => {
      state.filterNote = "";
      scheduleTagLookup();
    });
  });
  let tagTimer = 0;

  function scheduleTagLookup() {
    window.clearTimeout(tagTimer);
    tagTimer = window.setTimeout(() => {
      lookupTags().catch((error) => {
        state.filterNote = error.message.startsWith("tag") ? error.message : `tag lookup failed: ${error.message}`;
        tagEpoch += 1;
        state.tagIds = new Set();
        render();
      });
    }, 300);
  }

  async function lookupTags() {
    const query = els.tags.value.trim();
    if (!query || !state.volume) {
      tagEpoch += 1;
      state.tagIds = null;
      if (state.filterNote.startsWith("tag")) {
        state.filterNote = "";
      }
      render();
      return;
    }
    const payload = await getJson(
      withSet(`/api/volumes/${encodeURIComponent(state.volume)}/tags?q=${encodeURIComponent(query)}`)
    );
    if (els.tags.value.trim() !== query) {
      return;
    }
    state.filterNote = "";
    tagEpoch += 1;
    state.tagIds = new Set(payload.structureIds || []);
    render();
  }

  ["input", "change"].forEach((name) => {
    els.hue.addEventListener(name, () => {
      rememberOverlay();
      scheduleRender();
    });
    els.visibility.addEventListener(name, () => {
      rememberOverlay();
      syncVisibilityLabel();
      scheduleRender();
    });
  });
  els.trainingSet.addEventListener("change", () => {
    state.trainingSet = els.trainingSet.value || "current";
    rememberView(false);
    undoStack.length = 0;
    closeViewer();
    loadVolumes().catch((error) => {
      els.status.textContent = error.message;
    });
  });
  els.volume.addEventListener("change", () => {
    undoStack.length = 0;
    state.volume = els.volume.value;
    rememberView(true);
    watchUpdate().catch(() => {});
    tagEpoch += 1;
    state.tagIds = null;
    if (els.tags) {
      els.tags.value = "";
    }
    loadSettings()
      .then(() => loadCatalog(true))
      .catch((error) => {
        els.status.textContent = error.message;
      });
  });

  [["accept-all", "approve"], ["reject-all", "ignore"]].forEach(([id, act]) => {
    document.getElementById(id).addEventListener("click", () => {
      reviewVisible(act).catch((error) => {
        els.status.textContent = error.message;
      });
    });
  });
  document.getElementById("export-ignore").addEventListener("click", () => {
    exportIgnore().catch((error) => {
      els.status.textContent = error.message;
    });
  });
  document.getElementById("repair-pack").addEventListener("click", () => {
    downloadRepairPack().catch((error) => {
      els.status.textContent = error.message;
    });
  });
  document.getElementById("import-ignore").addEventListener("click", () => {
    document.getElementById("import-ignore-file").click();
  });
  document.getElementById("import-ignore-file").addEventListener("change", (event) => {
    const file = event.target.files && event.target.files[0];
    event.target.value = "";
    if (!file) {
      return;
    }
    importIgnore(file).catch((error) => {
      els.status.textContent = error.message;
    });
  });

  async function exportIgnore() {
    const payload = await getJson(withSet("/api/review-lists"));
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    const setName = payload.trainingSet || state.trainingSet || "current";
    link.download = `${setName}-review.json`;
    link.click();
    URL.revokeObjectURL(url);
    const volumes = payload.volumes || [];
    els.status.textContent = `Exported review lists for ${volumes.length} volume(s).`;
  }

  async function downloadRepairPack() {
    const response = await fetch(withSet(`/api/volumes/${encodeURIComponent(state.volume)}/repair-pack`));
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.error || response.statusText);
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${state.volume}-repair.zip`;
    link.click();
    URL.revokeObjectURL(url);
    els.status.textContent = "Downloaded the rejected-mask repair pack.";
  }

  async function importIgnore(file) {
    const text = await file.text();
    let payload;
    try {
      payload = JSON.parse(text);
    } catch {
      throw new Error("review file is not JSON");
    }
    const result = await getJson(withSet("/api/review-lists"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    await loadCatalog();
    const applied = (result.applied || []).length;
    const held = (result.held || []).length;
    els.status.textContent = held
      ? `Imported ${applied} volume(s). Kept ${held} that are not in this training set.`
      : `Imported review lists for ${applied} volume(s).`;
  }

  els.filtersToggle.addEventListener("click", () => {
    setFiltersCollapsed(!filtersCollapsed(), true);
    // The scroller gained or lost the controls' height, so lay the cards out for the new size.
    render(true);
  });

  els.scroller.addEventListener("scroll", () => {
    render(false);
    scheduleReprioritize();
  });
  window.addEventListener("resize", () => {
    renderSortStack();
    render();
    if (!els.viewer.hidden) {
      paintViewer();
    }
  });

  // A quick training-set change starts a second loadVolumes while the first /api/me is still in flight; only the
  // newest call may rebuild the volume list, so the dropdown and state.volume never come from the older set.
  // Every training-set change starts a call, so being the newest is the whole check.
  const volumesLatest = makeLatest();

  async function loadVolumes(forceVolume) {
    const isLatest = volumesLatest(() => true);
    let me;
    try {
      me = await getJson(withSet("/api/me"));
    } catch (error) {
      if (!isLatest()) {
        return;
      }
      throw error;
    }
    if (!isLatest()) {
      return;
    }
    state.me = me;
    const volumes = state.me.volumes || [];
    els.who.textContent = `${state.me.displayName} (${state.config.identityMode})`;
    els.logout.hidden = state.config.identityMode !== "oidc";
    els.volume.innerHTML = volumes
      .map((item) => `<option value="${item.name}">${item.name} · ${item.permission}</option>`)
      .join("");
    const names = volumes.map((item) => item.name);
    const preferred = readView().volumes[state.trainingSet];
    state.volume = forceVolume && names.includes(forceVolume)
      ? forceVolume
      : (names.includes(preferred) ? preferred : (names[0] || ""));
    els.volume.value = state.volume;
    rememberView(true);
    const volumeWrap = els.volume.closest(".filter");
    if (volumes.length <= 1) {
      els.volume.style.display = volumes.length ? "none" : "";
      const volumeLabel = volumeWrap ? volumeWrap.querySelector("label") : null;
      if (volumeLabel) {
        volumeLabel.style.display = volumes.length ? "none" : "";
      }
    }
    if (!volumes.length) {
      state.rows = [];
      els.status.textContent = "No volumes with Read access in this training set.";
      render();
      return;
    }
    watchUpdate().catch(() => {});
    await loadSettings();
    if (!isLatest()) {
      return;
    }
    await loadCatalog(true);
  }

  function applyRefreshVisibility() {
    document.body.classList.toggle("can-refresh", Boolean(state.settings && state.settings.odata));
  }

  // Switching volumes quickly leaves the older volume's settings request in flight; its answer must not replace
  // state.settings, which saveSettings and the refresh buttons read for the volume now selected.
  const settingsLatest = makeLatest();

  async function loadSettings() {
    const { volume, trainingSet } = state;
    const isLatest = settingsLatest(() => state.volume === volume && state.trainingSet === trainingSet);
    if (!volume) {
      state.settings = null;
      applyRefreshVisibility();
      return;
    }
    let settings;
    try {
      settings = await getJson(withSet(`/api/volumes/${encodeURIComponent(volume)}/settings`));
    } catch (error) {
      if (!isLatest()) {
        return;
      }
      throw error;
    }
    if (!isLatest()) {
      return;
    }
    state.settings = settings;
    applyRefreshVisibility();
  }

  function openSettings() {
    const dialog = document.getElementById("settings-dialog");
    const endpoint = document.getElementById("settings-odata");
    const exporter = document.getElementById("settings-exporter");
    const current = state.settings || {};
    const fallback = current.placeholder || `https://websvc.codepharm.net/${state.volume}/OData`;
    endpoint.value = (current.odata || fallback).trim();
    endpoint.placeholder = fallback;
    exporter.value = current.exporter === "legacy" ? "legacy" : "tiled";
    dialog.hidden = false;
  }

  // The dialog stays open until the response arrives, so Save can be clicked twice. Every POST is still sent (a
  // dropped response is never a dropped save); only the newest one's answer updates state.settings and closes the dialog.
  const saveLatest = makeLatest();

  async function saveSettings() {
    const endpoint = document.getElementById("settings-odata").value.trim();
    const exporter = document.getElementById("settings-exporter").value;
    const { volume, trainingSet } = state;
    const isLatest = saveLatest(() => state.volume === volume && state.trainingSet === trainingSet);
    let saved;
    try {
      saved = await getJson(withSet(`/api/volumes/${encodeURIComponent(volume)}/settings`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ odata: endpoint, exporter }),
      });
    } catch (error) {
      if (!isLatest()) {
        return;
      }
      throw error;
    }
    if (!isLatest()) {
      return;
    }
    state.settings = saved;
    applyRefreshVisibility();
    document.getElementById("settings-dialog").hidden = true;
    els.status.textContent = state.settings.odata
      ? "Saved the endpoint for this volume."
      : "Cleared the endpoint for this volume.";
  }

  function paintRefreshMark(locationId) {
    const mark = refreshMarks.get(Number(locationId)) || "";
    const id = String(locationId);
    document.querySelectorAll(`button.refresh[data-id="${id}"]`).forEach((button) => {
      button.classList.toggle("busy", mark === "busy");
      button.classList.toggle("done", mark === "done");
      button.disabled = mark === "busy";
      button.title = mark === "busy" ? "Refreshing mask" : (mark === "done" ? "Mask refreshed" : "Refresh mask from OData");
    });
    const viewerButton = document.getElementById("viewer-refresh");
    if (viewerButton && String(viewer.locationId) === id) {
      viewerButton.classList.toggle("busy", mark === "busy");
      viewerButton.classList.toggle("done", mark === "done");
      viewerButton.disabled = mark === "busy";
    }
  }

  function setRefreshMark(locationId, mark) {
    const id = Number(locationId);
    if (mark) {
      refreshMarks.set(id, mark);
    } else {
      refreshMarks.delete(id);
    }
    paintRefreshMark(id);
  }

  function repaintLocation(locationId) {
    const id = String(locationId);
    const gen = String(locationGen(locationId));
    const rows = state.rows.filter((row) => String(row.location_id) === id);
    els.grid.querySelectorAll(`.composite[data-loc="${id}"]`).forEach((node) => {
      const row = rows.find((item) => (item.image_key || "") === (node.dataset.key || ""));
      if (!row) {
        return;
      }
      const image = mediaUrl(state.volume, tissueRelpath(row), row.location_id);
      const mask = mediaUrl(state.volume, row.mask_relpath, row.location_id);
      node.dataset.gen = gen;
      node.dataset.image = image;
      node.dataset.mask = mask;
      node.dataset.layer = layerUrl(row);
      node.classList.add("waiting");
      const photo = node.querySelector("img.jpeg");
      if (photo) {
        photo.classList.add("pending");
        photo.src = image;
      }
      paintComposite(node, paintToken);
    });
  }

  // The refresh also scores the location with the current segmentation model (score.py on the server). Its numbers go
  // on the location's rows and captions at once, without reloading the catalog.
  function applyRefreshScore(locationId, score) {
    if (!score || score.error || score.sam2GtIou == null) {
      return;
    }
    const id = String(locationId);
    for (const row of state.rows) {
      if (String(row.location_id) === id) {
        row.sam2GtIou = score.sam2GtIou;
        row.sam2PredIou = score.sam2PredIou;
        row.sam2Checkpoint = score.sam2Checkpoint;
        row.sam2ScoredAt = score.sam2ScoredAt;
      }
    }
    state.hasSam2 = true;
    els.grid.querySelectorAll(`article.card[data-id="${id}"]`).forEach((card) => {
      const row = state.rows.find(
        (item) => String(item.location_id) === id && (item.image_key || "") === (card.dataset.key || "")
      );
      const caption = card.querySelector(".meta-row > div");
      if (row && caption) {
        caption.innerHTML = captionHtml(row);
      }
    });
  }

  function refreshScoreText(score) {
    if (!score) {
      return "";
    }
    if (score.error) {
      return ` Not scored: ${score.error}`;
    }
    const model = score.sam2Checkpoint ? ` (${score.sam2Checkpoint})` : "";
    return ` SAM2 ${Number(score.sam2GtIou).toFixed(2)}${model}.`;
  }

  async function refreshLocation(locationId) {
    const id = Number(locationId);
    if (refreshMarks.get(id) === "busy") {
      return;
    }
    const openId = viewer.locationId;
    const openKey = viewer.imageKey;
    setRefreshMark(id, "busy");
    try {
      const result = await getJson(withSet(`/api/volumes/${encodeURIComponent(state.volume)}/refresh-location`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ location_id: locationId }),
      });
      locationGenerations.set(id, locationGen(id) + 1);
      layerEpoch += 1;
      setRefreshMark(id, "done");
      repaintLocation(id);
      applyRefreshScore(id, result.score);
      // The server also queued the location for the card scorer: its cards say "re-scoring…" until it is done.
      const rescore = result.rescore || null;
      if (rescore && rescore.request) {
        watchRescore(id, rescore.request);
      }
      const count = (result.imageKeys || []).length;
      els.status.textContent = (count
        ? `Refreshed ${count} crop mask${count === 1 ? "" : "s"}.`
        : "No existing crop for that location.") + refreshScoreText(result.score)
        + (rescore && rescore.request ? " Re-scoring its cards." : "")
        + (rescore && rescore.error ? ` Not queued for re-scoring: ${rescore.error}` : "");
      if (!els.viewer.hidden && String(openId) === String(locationId)) {
        const row = state.rows.find((item) =>
          String(item.location_id) === String(openId) && (item.image_key || "") === (openKey || "")
        );
        if (row) {
          await openViewerFromRow(row);
        }
      }
      window.setTimeout(() => {
        if (refreshMarks.get(id) === "done") {
          setRefreshMark(id, "");
        }
      }, 2500);
    } catch (error) {
      setRefreshMark(id, "");
      throw error;
    }
  }

  // Update all: one background job per volume on the server. The dialog only asks (what will be checked, Start); the
  // progress, Cancel and the result live in a slim strip under the cards, so the view is never covered while it runs.
  // The strip is also picked up again after a reload or a volume switch, because the job outlives the page.
  const updateUi = {
    timer: 0,
    wasRunning: false,
    cancelling: false,
    // A converter build shows its cards as they are made: the built count last shown and when (ms), to throttle reloads.
    shownBuilt: 0,
    shownAt: 0,
    dialog: document.getElementById("update-dialog"),
    since: document.getElementById("update-since"),
    last: document.getElementById("update-last"),
    start: document.getElementById("update-start"),
    close: document.getElementById("update-close"),
    strip: document.getElementById("update-strip"),
    title: document.getElementById("update-strip-title"),
    bar: document.getElementById("update-bar"),
    line: document.getElementById("update-line"),
    detail: document.getElementById("update-detail"),
    detailToggle: document.getElementById("update-details-toggle"),
    cancel: document.getElementById("update-cancel"),
    dismiss: document.getElementById("update-dismiss"),
  };

  function updateUrl(suffix = "") {
    return withSet(`/api/volumes/${encodeURIComponent(state.volume)}/update-all${suffix}`);
  }

  function describeWatermark(status) {
    const since = status.watermark || status.watermarkAfter;
    if (since) {
      return `Changes since ${since.replace("T", " ").replace(/\.\d+Z$/, "Z")} will be checked.`;
    }
    return "First run: starts from the age of the oldest masks, so it may redraw more than later runs.";
  }

  // update-strip-view-begin
  // What the strip shows for one status from the server: its words, the bar, and which buttons apply. No DOM here.
  function updateStripView(status, volume, cancelling) {
    const running = Boolean(status.running);
    const phase = status.phase || "idle";
    const count = (value) => Number(value || 0).toLocaleString("en-US");
    const circles = status.circles ? `, ${count(status.circles)} circles` : "";
    const counts = `${count(status.refreshed)} refreshed, ${count(status.added)} added${circles}, ${count(status.skipped)} skipped, ${count(status.failed)} failed`;
    const since = status.watermarkBefore
      ? status.watermarkBefore.replace("T", " ").replace(/\.\d+Z$/, "Z")
      : "";
    const unit = status.unit || "page";
    const place = status.pages ? `${unit} ${count(status.page)} of ${count(status.pages)}` : "";
    let headline;
    if (running && cancelling) {
      headline = "Cancelling, finishing the row in progress";
    } else if (running && phase === "reading catalog") {
      headline = "Reading the catalog";
    } else if (running && phase === "retrying skipped") {
      headline = "Retrying locations skipped last time";
    } else if (running) {
      headline = since ? `Checking changes since ${since}` : "Checking changes";
    } else {
      headline = { done: "Finished", cancelled: "Cancelled", failed: "Finished with failures" }[phase] || "";
    }
    const parts = [headline, running && !status.pages ? "" : place, phase === "idle" ? "" : counts];
    const lines = [...(status.notes || []), ...(status.errors || []), ...(status.skipLines || []).slice(0, 20)];
    if (status.skipped > (status.skipLines || []).length) {
      lines.push(`... ${status.skipped} skipped in all; the full list is in update_all.log`);
    }
    if (status.set && running && !cancelling && status.headline) {
      headline = status.headline;
    }
    if (status.set && !running && phase === "failed" && (status.errors || []).length) {
      // A build that could not run says why on the strip itself, not only under Details.
      return {
        title: `Build ${volume}`,
        line: status.errors[0],
        tone: "bad",
        indeterminate: false,
        max: 1,
        value: 0,
        showBar: false,
        showCancel: false,
        showDismiss: true,
        detail: "",
      };
    }
    const max = Math.max(1, Number(status.pages) || 1);
    return {
      title: status.set ? (running ? `Building ${volume}` : `Build ${volume}`) : (running ? `Updating ${volume}` : `Update ${volume}`),
      line: parts.filter(Boolean).join(" · "),
      tone: running ? "running" : ({ done: "ok", cancelled: "warn" }[phase] || "bad"),
      indeterminate: running && !status.pages,
      max,
      value: Math.min(Number(status.page) || 0, max),
      showBar: running || Boolean(status.pages),
      showCancel: running && !cancelling,
      showDismiss: !running,
      detail: lines.join("\n"),
    };
  }
  // update-strip-view-end

  let updateStripShown = false;

  function showUpdateStrip(visible) {
    updateUi.strip.hidden = !visible;
    if (visible !== updateStripShown) {
      updateStripShown = visible;
      // The cards' area just lost or regained the strip's height, so lay them out for the new size.
      render(true);
    }
  }

  function paintUpdate(status) {
    const phase = status.phase || "idle";
    if (phase === "idle" && !status.running) {
      showUpdateStrip(false);
      updateUi.since.textContent = describeWatermark(status);
      const last = status.lastCounts || {};
      updateUi.last.textContent = status.lastRun
        ? `Last run ${status.lastRun}: ${last.refreshed || 0} refreshed, ${last.added || 0} added, ${last.skipped || 0} skipped.`
        : "";
      return;
    }
    if (!status.running) {
      updateUi.cancelling = false;
    }
    const view = updateStripView(status, state.volume, updateUi.cancelling);
    updateUi.strip.dataset.tone = view.tone;
    updateUi.title.textContent = view.title;
    updateUi.line.textContent = view.line;
    updateUi.line.title = view.line;
    updateUi.bar.hidden = !view.showBar;
    if (view.indeterminate) {
      updateUi.bar.removeAttribute("value");
    } else {
      updateUi.bar.max = view.max;
      updateUi.bar.value = view.value;
    }
    updateUi.cancel.hidden = !view.showCancel;
    updateUi.dismiss.hidden = !view.showDismiss;
    updateUi.detail.textContent = view.detail;
    updateUi.detailToggle.hidden = !view.detail;
    if (!view.detail) {
      updateUi.detail.hidden = true;
      updateUi.detailToggle.setAttribute("aria-expanded", "false");
    }
    showUpdateStrip(true);
  }

  // The status of this volume's job, or null when it cannot be read (a viewer without review rights, a lost link).
  async function readUpdateStatus() {
    const volume = state.volume;
    try {
      const status = await getJson(updateUrl());
      return volume === state.volume ? status : null;
    } catch (error) {
      return null;
    }
  }

  async function pollUpdate() {
    window.clearTimeout(updateUi.timer);
    const status = await readUpdateStatus();
    if (!status) {
      return;
    }
    paintUpdate(status);
    if (status.running) {
      updateUi.wasRunning = true;
      const built = status.refreshed || 0;
      if (state.converter && built > updateUi.shownBuilt && Date.now() - updateUi.shownAt > 5000) {
        // Cards made so far appear while the build goes on, not only when it ends.
        updateUi.shownBuilt = built;
        updateUi.shownAt = Date.now();
        await loadCatalog(false, true).catch(() => {});
      }
      updateUi.timer = window.setTimeout(pollUpdate, 1000);
      return;
    }
    const justFinished = updateUi.wasRunning;
    updateUi.wasRunning = false;
    updateUi.shownBuilt = 0;
    updateUi.shownAt = 0;
    if (justFinished && (status.refreshed || status.added || status.circles)) {
      for (const id of status.refreshedIds || []) {
        locationGenerations.set(Number(id), locationGen(id) + 1);
      }
      layerEpoch += 1;
      resetVisibleCards();
      await loadCatalog(false, true);
    }
  }

  // A volume was opened: follow its job if one is running (started by this page earlier, or by another), else show none.
  async function watchUpdate() {
    window.clearTimeout(updateUi.timer);
    updateUi.wasRunning = false;
    updateUi.cancelling = false;
    showUpdateStrip(false);
    if (!state.volume) {
      return;
    }
    const status = await readUpdateStatus();
    if (status && status.running) {
      await pollUpdate();
    }
  }

  async function openUpdateAll() {
    if (!state.volume) {
      return;
    }
    const status = await readUpdateStatus();
    if (status && status.running) {
      // Already going: the strip is the whole interface, there is nothing to ask.
      await pollUpdate();
      return;
    }
    if (status) {
      updateUi.since.textContent = describeWatermark(status);
      const last = status.lastCounts || {};
      updateUi.last.textContent = status.lastRun
        ? `Last run ${status.lastRun}: ${last.refreshed || 0} refreshed, ${last.added || 0} added, ${last.skipped || 0} skipped.`
        : "";
    }
    updateUi.start.disabled = false;
    updateUi.dialog.hidden = false;
  }

  async function startUpdateAll() {
    updateUi.start.disabled = true;
    try {
      const status = await getJson(updateUrl(), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{}",
      });
      updateUi.dialog.hidden = true;
      updateUi.wasRunning = true;
      paintUpdate(status);
    } catch (error) {
      updateUi.start.disabled = false;
      els.status.textContent = error.message;
      return;
    }
    updateUi.timer = window.setTimeout(pollUpdate, 500);
  }

  // Converter: build a structure. The server makes the set, starts the job, and the strip follows it like Update all.
  const buildUi = {
    dialog: document.getElementById("build-dialog"),
    structure: document.getElementById("build-structure"),
    types: document.getElementById("build-types"),
    zMin: document.getElementById("build-zmin"),
    zMax: document.getElementById("build-zmax"),
    start: document.getElementById("build-start"),
  };

  const buildKey = (build) => `${build.set}|${build.volume}`;

  // The builds the user may see, in the picker in the converter bar. Returns them; refreshes the counts line.
  async function refreshBuilds(currentSet, currentVolume) {
    const payload = await getJson("/api/converter/builds");
    const builds = payload.builds || [];
    state.builds = builds;
    const select = document.getElementById("converter-build-select");
    const keep = select.value;
    select.innerHTML = builds
      .map((build) => `<option value="${buildKey(build)}">${window.ConverterMode.buildLabel(build)}</option>`)
      .join("");
    const sets = [...new Set(builds.map((build) => build.set))];
    els.trainingSet.innerHTML = sets.map((name) => `<option value="${name}">${name}</option>`).join("");
    const wanted = currentSet ? `${currentSet}|${currentVolume}` : keep;
    if (builds.some((build) => buildKey(build) === wanted)) {
      select.value = wanted;
    }
    const current = builds.find((build) => buildKey(build) === select.value);
    document.getElementById("converter-counts").textContent = window.ConverterMode.countsLine(current);
    return builds;
  }

  function buildSettings(structureId) {
    return {
      structureId,
      typeCodes: [...buildUi.types.querySelectorAll("input:checked")].map((box) => Number(box.value)),
      zMin: buildUi.zMin.value,
      zMax: buildUi.zMax.value,
    };
  }

  // The Convert dialog. Its volume list comes from the preview, so before an id is typed it lists the volumes the user can review.
  function openBuildDialog() {
    const picker = document.getElementById("build-volume");
    const reviewable = ((state.me && state.me.volumes) || []).filter((item) => item.permission === "review");
    picker.innerHTML = reviewable.map((item) => `<option value="${item.name}">${item.name}</option>`).join("");
    buildUi.start.disabled = true;
    buildUi.dialog.hidden = false;
    buildUi.structure.focus();
  }

  async function startBuild() {
    const structureId = Number(buildUi.structure.value);
    if (!Number.isInteger(structureId) || structureId <= 0) {
      els.status.textContent = "Enter the structure ID to build.";
      return;
    }
    const body = {
      structureId,
      volume: document.getElementById("build-volume").value,
      typeCodes: [...buildUi.types.querySelectorAll("input:checked")].map((box) => Number(box.value)),
      zMin: buildUi.zMin.value,
      zMax: buildUi.zMax.value,
    };
    buildUi.start.disabled = true;
    try {
      const started = await getJson("/api/converter/structures", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      buildUi.dialog.hidden = true;
      document.getElementById("filters").hidden = false;
      await refreshBuilds();
      state.trainingSet = started.set;
      els.trainingSet.value = started.set;
      document.getElementById("converter-build-select").value = `${started.set}|${body.volume}`;
      undoStack.length = 0;
      closeViewer();
      rememberView(false);
      await loadVolumes(body.volume);
      updateUi.wasRunning = true;
      paintUpdate(started);
      updateUi.timer = window.setTimeout(pollUpdate, 500);
    } catch (error) {
      els.status.textContent = error.message;
    } finally {
      buildUi.start.disabled = false;
    }
  }

  const buildButton = document.getElementById("converter-build");
  if (buildButton) {
    buildButton.addEventListener("click", () => openBuildDialog());
  }
  if (document.getElementById("converter-new")) {
    document.getElementById("converter-new").addEventListener("click", () => openBuildDialog());
    buildUi.start.addEventListener("click", () => {
      startBuild();
    });
    document.getElementById("build-cancel").addEventListener("click", () => {
      buildUi.dialog.hidden = true;
    });
    // Looking up the structure as the id is typed shows which volume it is in and how many annotations match.
    let previewTimer = 0;
    let previewSerial = 0;
    const schedulePreview = () => {
      window.clearTimeout(previewTimer);
      buildUi.start.disabled = true;
      previewTimer = window.setTimeout(async () => {
        const serial = (previewSerial += 1);
        const structureId = Number(buildUi.structure.value);
        const note = document.getElementById("build-preview");
        if (!Number.isInteger(structureId) || structureId <= 0) {
          note.textContent = "Enter a structure ID. The volumes you can build in are checked for it.";
          return;
        }
        note.textContent = "Looking for it…";
        try {
          const result = await getJson("/api/converter/preview", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(buildSettings(structureId)),
          });
          if (serial !== previewSerial) {
            return;
          }
          const picker = document.getElementById("build-volume");
          const keep = picker.value;
          picker.innerHTML = Object.entries(result.volumes).map(([name, info]) => {
            const text = info.error ? "could not check" : `${info.count} annotation${info.count === 1 ? "" : "s"}`;
            return `<option value="${name}">${name} — ${text}</option>`;
          }).join("");
          const chosen = (result.volumes[keep] && result.volumes[keep].count > 0) ? keep : window.ConverterMode.pickVolume(result.volumes);
          if (chosen) {
            picker.value = chosen;
          }
          note.textContent = window.ConverterMode.previewLine(result.volumes, picker.value);
          const info = result.volumes[picker.value];
          buildUi.start.disabled = !(info && info.count > 0);
        } catch (error) {
          if (serial === previewSerial) {
            document.getElementById("build-preview").textContent = `Could not check: ${error.message}`;
          }
        }
      }, 450);
    };
    buildUi.structure.addEventListener("input", schedulePreview);
    buildUi.types.addEventListener("change", schedulePreview);
    buildUi.zMin.addEventListener("input", schedulePreview);
    buildUi.zMax.addEventListener("input", schedulePreview);
    document.getElementById("build-volume").addEventListener("change", () => {
      const picked = document.getElementById("build-volume");
      const match = /(\d+) annotation/.exec(picked.options[picked.selectedIndex] ? picked.options[picked.selectedIndex].text : "");
      buildUi.start.disabled = !(match && Number(match[1]) > 0);
    });

    // The bar: pick a build, continue it, see what failed.
    document.getElementById("converter-build-select").addEventListener("change", (event) => {
      const [set, volume] = event.target.value.split("|");
      state.trainingSet = set;
      els.trainingSet.value = set;
      undoStack.length = 0;
      closeViewer();
      rememberView(false);
      loadVolumes(volume).catch((error) => {
        els.status.textContent = error.message;
      });
    });
    document.getElementById("converter-continue").addEventListener("click", () => {
      document.getElementById("update-all").click();
    });
    // Delete the open build, after a warning that names what is lost. A running build must be cancelled first.
    const currentBuild = () => {
      const [set, volume] = document.getElementById("converter-build-select").value.split("|");
      return (state.builds || []).find((build) => build.set === set && build.volume === volume) || null;
    };
    document.getElementById("converter-delete").addEventListener("click", () => {
      const build = currentBuild();
      if (!build) {
        return;
      }
      document.getElementById("delete-warning").textContent = build.running
        ? `Structure ${build.structureId} of ${build.volume} is still building. Cancel the build first.`
        : window.ConverterMode.deleteWarning(build);
      document.getElementById("delete-confirm").disabled = Boolean(build.running);
      document.getElementById("delete-error").hidden = true;
      document.getElementById("delete-dialog").hidden = false;
      document.getElementById("delete-cancel").focus();
    });
    document.getElementById("delete-cancel").addEventListener("click", () => {
      document.getElementById("delete-dialog").hidden = true;
    });
    document.getElementById("delete-confirm").addEventListener("click", async () => {
      const build = currentBuild();
      if (!build) {
        return;
      }
      const confirm = document.getElementById("delete-confirm");
      confirm.disabled = true;
      try {
        await getJson("/api/converter/delete", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ set: build.set, volume: build.volume }),
        });
        // The page was built around this build; start over from what is left (or the empty page).
        window.location.reload();
      } catch (error) {
        const message = document.getElementById("delete-error");
        message.textContent = `Not deleted: ${error.message}`;
        message.hidden = false;
        confirm.disabled = false;
      }
    });
    document.getElementById("converter-failed").addEventListener("click", () => {
      const failed = (state.convCards || []).filter((card) => card.state === "failed");
      const list = document.getElementById("failed-list");
      list.replaceChildren(...failed.map((card) => {
        const line = document.createElement("div");
        const link = document.createElement("a");
        link.href = vikingHref({ location_id: card.location_id });
        link.textContent = `#${card.location_id}`;
        const reason = document.createElement("small");
        reason.textContent = window.ConverterMode.humanReason(card.reason);
        reason.title = card.reason || "";
        line.append(link, document.createTextNode(` · Z ${card.z}`), reason);
        return line;
      }));
      document.getElementById("failed-dialog").hidden = false;
    });
    document.getElementById("failed-close").addEventListener("click", () => {
      document.getElementById("failed-dialog").hidden = true;
    });
    document.getElementById("failed-retry").addEventListener("click", () => {
      document.getElementById("failed-dialog").hidden = true;
      document.getElementById("update-all").click();
    });
  }

  document.getElementById("update-all").addEventListener("click", () => {
    openUpdateAll().catch((error) => {
      els.status.textContent = error.message;
    });
  });
  updateUi.start.addEventListener("click", () => {
    startUpdateAll();
  });
  updateUi.close.addEventListener("click", () => {
    updateUi.dialog.hidden = true;
  });
  updateUi.cancel.addEventListener("click", () => {
    updateUi.cancelling = true;
    updateUi.cancel.hidden = true;
    updateUi.line.textContent = "Cancelling, finishing the row in progress";
    getJson(updateUrl("/cancel"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    }).catch((error) => {
      updateUi.cancelling = false;
      updateUi.cancel.hidden = false;
      els.status.textContent = error.message;
    });
  });
  updateUi.detailToggle.addEventListener("click", () => {
    const open = updateUi.detail.hidden;
    updateUi.detail.hidden = !open;
    updateUi.detailToggle.setAttribute("aria-expanded", String(open));
    render(true);
  });
  updateUi.dismiss.addEventListener("click", () => {
    window.clearTimeout(updateUi.timer);
    updateUi.detail.hidden = true;
    showUpdateStrip(false);
  });

  document.getElementById("volume-settings").addEventListener("click", () => {
    openSettings();
  });
  document.getElementById("settings-cancel").addEventListener("click", () => {
    document.getElementById("settings-dialog").hidden = true;
  });
  document.getElementById("settings-save").addEventListener("click", () => {
    saveSettings().catch((error) => {
      els.status.textContent = error.message;
    });
  });

  /** Shows the header links to the gallery's other pages: each one the server has switched on, except this one. */
  function showPageLinks() {
    const current = state.converter ? "converter" : (state.gold ? "gold" : "gallery");
    const enabled = { gallery: true, converter: Boolean(state.config.converter), gold: Boolean(state.config.gold) };
    for (const link of document.querySelectorAll("#page-links a[data-page]")) {
      link.hidden = !enabled[link.dataset.page] || link.dataset.page === current;
    }
  }

  async function boot() {
    loadOverlayPrefs();
    loadCardSizePref();
    loadFiltersCollapsedPref();
    renderSortStack();
    state.config = await getJson("/api/config");
    state.converter = Boolean(window.ConverterMode && window.ConverterMode.active(location.pathname, state.config));
    if (state.converter) {
      window.ConverterMode.applyChrome(document);
      document.getElementById("converter-build").hidden = false;
    }
    state.gold = Boolean(window.GoldMode && window.GoldMode.active(location.pathname, state.config));
    if (state.gold) {
      window.GoldMode.applyChrome(document);
      document.getElementById("gold-draw").hidden = false;
      document.getElementById("gold-list").hidden = false;
      if (brushButton) {
        brushButton.hidden = false;
      }
    }
    showPageLinks();
    const setsPayload = await getJson(
      state.converter ? "/api/training-sets?app=converter" : (state.gold ? "/api/training-sets?app=gold" : "/api/training-sets"),
    );
    if (state.gold && !(setsPayload.sets && setsPayload.sets.length)) {
      els.status.textContent = "No gold set yet. Draw a sample, or start from list files.";
      document.getElementById("filters").hidden = true;
      document.getElementById("gold-draw").click();
      return;
    }
    if (state.converter) {
      const me = await getJson("/api/me");
      els.who.textContent = `${me.displayName} (${state.config.identityMode})`;
      els.logout.hidden = state.config.identityMode !== "oidc";
      state.me = me;
      const builds = await refreshBuilds();
      if (!builds.length) {
        // Nothing built yet: the first thing to do is choose a structure, so ask for it straight away.
        els.status.textContent = "No structure has been converted yet.";
        document.getElementById("filters").hidden = true;
        openBuildDialog();
        return;
      }
      const view = readView();
      const wanted = builds.find((item) => item.set === view.set && item.volume === (view.volumes || {})[view.set]) || builds[0];
      state.trainingSet = wanted.set;
      els.trainingSet.value = wanted.set;
      document.getElementById("converter-build-select").value = buildKey(wanted);
      rememberView(false);
      await loadVolumes(wanted.volume);
      return;
    }
    const sets = setsPayload.sets && setsPayload.sets.length ? setsPayload.sets : ["current"];
    const rememberedSet = readView().set;
    state.trainingSet = sets.includes(rememberedSet)
      ? rememberedSet
      : ((state.converter || state.gold) ? setsPayload.default || sets[0] : (sets.includes("current") ? "current" : sets[0]));
    els.trainingSet.innerHTML = sets
      .map((name) => `<option value="${name}">${name}</option>`)
      .join("");
    els.trainingSet.value = state.trainingSet;
    await loadVolumes();
  }

  boot().catch((error) => {
    els.status.textContent = error.message;
  });
})();
