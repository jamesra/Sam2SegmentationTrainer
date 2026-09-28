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
  const SIZE_KEY = "galleryCardSize";
  const CARD_SIZES = { huge: 1024, large: 512, medium: 384, small: 256 };
  const CACHE_MAX = 80;

  const state = {
    me: null,
    config: { vikingUrl: "", identityMode: "stub" },
    volume: "",
    trainingSet: "current",
    rows: [],
    permission: "read",
    hasSam2: false,
    sorts: [{ key: "z", desc: false }],
    tagIds: null,
    filterNote: "",
    cropSize: 1024,
    cardSize: "large",
    settings: null,
    maskGeneration: 0,
  };

  const els = {
    who: document.getElementById("who"),
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

  function assignLinkHues(cells) {
    const at = new Map();
    for (const cell of cells) {
      at.set(`${cell.gr},${cell.gc}`, Number(cell.row.location_id));
    }
    const neighbors = new Map();
    const dirs = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]];
    const order = [];
    const seen = new Set();
    for (const cell of cells) {
      const id = Number(cell.row.location_id);
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
      const tone = LINK_TONES[assigned.get(Number(cell.row.location_id))];
      cell.linkH = tone[0];
      cell.linkS = tone[1];
      cell.linkL = tone[2];
    }
  }

  // A shared grid edge of one location stays open so the ring is the group's perimeter.
  function assignJoins(cells) {
    const at = new Map();
    for (const cell of cells) {
      at.set(`${cell.gr},${cell.gc}`, Number(cell.row.location_id));
    }
    for (const cell of cells) {
      const id = Number(cell.row.location_id);
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
  };

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

  async function paintComposite(node, token) {
    const key = `${node.dataset.loc}|${node.dataset.key}|${node.dataset.size}|${node.dataset.hue}|${node.dataset.visibility}|${node.dataset.status || ""}|${node.dataset.gen || ""}`;
    const canvas = node.querySelector("canvas.tint");
    const photo = node.querySelector("img.jpeg");
    const cached = cacheGet(key);
    if (cached) {
      if (node.isConnected && token === paintToken) {
        blit(canvas, cached);
        if (photo) {
          photo.classList.toggle("pending", !cached.photoVisible);
        }
        setMediaState(node, cached.missing ? null : true);
      }
      return;
    }
    const hue = node.dataset.hue;
    const visibility = Number(node.dataset.visibility);
    let image = null;
    const imageTask = loadImage(node.dataset.image).then((loaded) => {
      image = loaded;
    }).catch(() => null);
    let mask;
    try {
      mask = await loadImage(node.dataset.mask);
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

  function vikingHref(row) {
    let template = state.config.vikingUrl || "";
    if (!template || template.includes("connectomes.utah.edu/connectome")) {
      template = "viking://open?volumeName={volume}&location={id}";
    }
    return template
      .replaceAll("{volume}", encodeURIComponent(state.volume))
      .replaceAll("{id}", encodeURIComponent(String(row.location_id)));
  }

  function parseRadius(text) {
    const trimmed = text.trim();
    if (!trimmed) {
      return null;
    }
    if (trimmed.includes("-")) {
      const [lo, hi] = trimmed.split("-", 2).map(Number);
      return { lo, hi };
    }
    const value = Number(trimmed);
    return Number.isFinite(value) ? { lo: value, hi: value } : null;
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

  function loadCardSizePref() {
    const stored = localStorage.getItem(SIZE_KEY);
    state.cardSize = Object.prototype.hasOwnProperty.call(CARD_SIZES, stored) ? stored : "large";
    syncCardSizeButtons();
  }

  function selectCardSize(name) {
    if (!Object.prototype.hasOwnProperty.call(CARD_SIZES, name) || name === state.cardSize) {
      return;
    }
    state.cardSize = name;
    localStorage.setItem(SIZE_KEY, name);
    syncCardSizeButtons();
    rowStride = 0;
    windowTop = -1;
    render(true);
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

  const SORT_FIELDS = [
    ["z", "Z"],
    ["type_id", "Type"],
    ["structure_id", "Structure ID"],
    ["structure_label", "Label"],
    ["radius", "Radius"],
    ["location_id", "Location ID"],
    ["loss", "Loss"],
  ];

  function sortFields() {
    const fields = SORT_FIELDS.slice();
    if (state.hasSam2) {
      fields.push(["sam2GtIou", "SAM2 GT IoU"]);
    }
    return fields;
  }

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

  function renderSortStack() {
    const fields = sortFields();
    const used = new Set(state.sorts.map((item) => item.key));
    const canAdd = state.sorts.length < fields.length;
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
          const taken = used.has(value) && value !== criterion.key;
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
    state.filterNote = state.filterNote.startsWith("tag") ? state.filterNote : "";
    const type = els.type.value.trim().toLowerCase();
    const label = els.label.value.trim().toLowerCase();
    const zText = els.z.value.trim();
    const radius = parseRadius(els.radius.value);
    const classes = visibleClasses();
    return state.rows.filter((row) => {
      if (!classes.has(reviewClass(row))) {
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
          if (idText !== label && !idText.startsWith(label)) {
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
        if (!Number.isFinite(value) || value < radius.lo || value > radius.hi) {
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

  function windowHtml(row) {
    const image = fileUrl(state.volume, tissueRelpath(row));
    const mask = fileUrl(state.volume, row.mask_relpath);
    const hue = overlayHue();
    const visibility = overlayVisibility();
    const status = row.ignored ? "rejected" : (row.approved ? "approved" : "");
    return `
      <div class="window">
        <div class="composite waiting" data-loc="${row.location_id}" data-key="${row.image_key || ""}" data-size="${state.cardSize}" data-status="${status}" data-gen="${state.maskGeneration}" data-image="${image}" data-mask="${mask}" data-hue="${hue}" data-visibility="${visibility}">
          <img class="jpeg pending" alt="" src="${image}">
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
    const sam2 = row.sam2GtIou == null ? "" : `<span class="badge">SAM2 ${Number(row.sam2GtIou).toFixed(2)}</span>`;
    const loss = row.loss == null ? "" : `<span class="badge">loss ${Number(row.loss).toFixed(3)}</span>`;
    const structureName = row.type_name || "Structure";
    const structureText = row.structure_id == null ? structureName : `${structureName} ${row.structure_id}`;
    const structureHref = sbfsemHref(row);
    const structure = structureHref
      ? `<a href="${structureHref}" target="_blank" rel="noreferrer" title="Open in sbfsem-tools">${structureText}</a>`
      : structureText;
    const label = row.structure_label ? ` · ${row.structure_label}` : "";
    return `<a class="viking-link" href="${vikingHref(row)}" title="Open in Viking">#${row.location_id}</a> · Z ${row.z} · ${structure}${label} ${loss}${sam2}`;
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
    const span = shape.rows;
    // A shape wider than the screen is cut into bands of `columns` and stacked.
    const bandCount = Math.max(1, Math.ceil(shape.cols / columns));
    const offsets = [];
    for (const [key, rows] of buckets) {
      const [dr, dc] = key.split(",").map(Number);
      const band = Math.floor(dc / columns);
      rows.forEach((row, copy) => {
        offsets.push({
          row,
          dr: (band + copy * bandCount) * span + dr,
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
        <div class="meta"><div class="meta-row"><div>${captionHtml(row)}</div>${refreshButton(row)}</div></div>
      </article>`;
  }

  function refreshButton(row) {
    const icon = `<svg viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M17.65 6.35A7.95 7.95 0 0 0 12 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08a5.99 5.99 0 0 1-5.65 4c-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"/></svg>`;
    return `<button type="button" class="icon refresh review-only" data-act="refresh" data-id="${row.location_id}" title="Refresh mask from OData" aria-label="Refresh mask">${icon}</button>`;
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
      views.join(""),
      state.sorts.map((item) => `${item.key}:${item.desc ? 1 : 0}`).join(","),
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
    const columns = columnCount(els.grid.clientWidth);
    const packed = packMemo.packed;
    const from = packMemo.repackFrom;
    if (!packed || packed.columns !== columns || from === 0) {
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

  function scrollToLocation() {
    const text = els.locate.value.trim();
    if (!text) {
      return;
    }
    const cards = ensureVisibleCards();
    const layout = layoutCards(cards);
    const cell = layout.cells.find((item) => String(item.row.location_id) === text);
    if (!cell) {
      render(true);
      els.status.textContent = `Location ${text} is not in this view.`;
      return;
    }
    els.scroller.scrollTop = layout.rowOffsets[cell.gr] || 0;
    render(true);
    els.status.textContent = `Location ${text}.`;
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
      node.style.gridColumn = column;
      node.style.gridRow = row;
      node.style.setProperty("--link-h", String(cell.linkH));
      node.style.setProperty("--link-s", `${cell.linkS}%`);
      node.style.setProperty("--link-l", `${cell.linkL}%`);
    }
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
  }

  function render(force = true, reuse = false) {
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
    els.grid.style.gridAutoRows = `${Math.max(1, baseHeight - GRID_GAP)}px`;
    els.spacer.style.height = `${rowOffsets[rowOffsets.length - 1] || baseHeight}px`;
    els.status.textContent = `${cards.length} location(s) · ${state.permission}${state.filterNote ? ` · ${state.filterNote}` : ""}`;
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

  async function loadCatalog(applyDefaultSort = false, preserveViewer = false) {
    if (!state.volume) {
      state.rows = [];
      indexCatalog();
      render();
      return;
    }
    const payload = await getJson(withSet(`/api/volumes/${encodeURIComponent(state.volume)}/catalog`));
    state.rows = payload.rows || [];
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
    if (applyDefaultSort) {
      const hasLoss = state.rows.some((row) => row.loss != null);
      state.sorts = [{ key: hasLoss ? "loss" : "z", desc: hasLoss }];
    }
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
    const stays = rows.some((row) => classes.has(reviewClass(row)));
    if (!stays) {
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
      lines.push(`loss ${Number(row.loss).toFixed(3)}`);
    }
    if (row.sam2GtIou != null) {
      lines.push(`SAM2 ${Number(row.sam2GtIou).toFixed(2)}`);
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

  function drawFull(image, overlay) {
    const width = image.naturalWidth;
    const height = image.naturalHeight;
    els.viewerImage.width = width;
    els.viewerImage.height = height;
    els.viewerMask.width = width;
    els.viewerMask.height = height;
    els.viewerImage.getContext("2d").drawImage(image, 0, 0);
    const maskContext = els.viewerMask.getContext("2d");
    maskContext.clearRect(0, 0, width, height);
    if (overlay) {
      maskContext.drawImage(overlay, 0, 0);
    }
    layoutViewerFrame(width, height);
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
    viewer.token += 1;
    viewer.locationId = "";
    viewer.imageKey = "";
    viewer.partIndex = null;
    viewer.parts = null;
    viewer.image = null;
    viewer.overlay = null;
    spaceHidesMask = false;
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
    viewer.overlay = composeOverlay(image, mask, viewer.hue, viewer.visibility);
    viewer.parts = null;
    viewer.partIndex = null;
    paintViewer();
    syncViewerActs();
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
        mask: fileUrl(state.volume, row.mask_relpath),
        hue: overlayHue(),
        visibility: String(overlayVisibility()),
        image: fileUrl(state.volume, tissueRelpath(row)),
      },
    };
    return openViewer(node);
  }

  els.grid.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-act]");
    if (button) {
      if (button.dataset.act === "refresh") {
        refreshLocation(Number(button.dataset.id)).catch((error) => {
          els.status.textContent = error.message;
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
  let pointerShowsMask = true;
  let spaceHidesMask = false;

  function applyMaskVisibility() {
    const show = spaceHidesMask ? !pointerShowsMask : pointerShowsMask;
    els.viewerMask.style.opacity = show ? "1" : "0";
    els.viewerCues.querySelector(".show").classList.toggle("active", show && pointerShowsMask);
    els.viewerCues.querySelector(".hide").classList.toggle("active", !show && !pointerShowsMask);
  }

  function setPointerShowsMask(show) {
    pointerShowsMask = show;
    applyMaskVisibility();
  }

  els.viewer.addEventListener("mousemove", (event) => {
    const frame = els.viewerFrame.getBoundingClientRect();
    const overImage = event.clientX >= frame.left && event.clientX <= frame.right
      && event.clientY >= frame.top && event.clientY <= frame.bottom;
    if (!overImage) {
      setPointerShowsMask(true);
      return;
    }
    setPointerShowsMask(event.clientX >= frame.left + frame.width / 2);
  });
  els.viewer.addEventListener("mouseleave", () => setPointerShowsMask(true));
  function zoomKeyTarget(event) {
    const tag = event.target && event.target.tagName;
    return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
  }

  function isSpaceKey(event) {
    return event.code === "Space" || event.key === " ";
  }

  window.addEventListener("keydown", (event) => {
    if (!els.viewer.hidden && (event.key === "Escape" || event.key === "Esc" || event.code === "Escape")) {
      event.preventDefault();
      closeViewer();
      return;
    }
    if (els.viewer.hidden || !isSpaceKey(event) || zoomKeyTarget(event)) {
      return;
    }
    event.preventDefault();
    if (event.repeat || spaceHidesMask) {
      return;
    }
    spaceHidesMask = true;
    applyMaskVisibility();
  }, true);
  window.addEventListener("keyup", (event) => {
    if (!isSpaceKey(event)) {
      return;
    }
    if (!els.viewer.hidden) {
      event.preventDefault();
    }
    spaceHidesMask = false;
    if (!els.viewer.hidden) {
      applyMaskVisibility();
    }
  }, true);
  window.addEventListener("blur", () => {
    spaceHidesMask = false;
    if (!els.viewer.hidden) {
      applyMaskVisibility();
    }
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
      const next = sortFields().find(([value]) => !used.has(value));
      if (next) {
        state.sorts.push({ key: next[0], desc: false });
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
    state.sorts[Number(select.dataset.index)].key = select.value;
    renderSortStack();
    render();
  });
  els.locate.addEventListener("change", () => {
    scrollToLocation();
  });
  els.locate.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      scrollToLocation();
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
    undoStack.length = 0;
    closeViewer();
    loadVolumes().catch((error) => {
      els.status.textContent = error.message;
    });
  });
  els.volume.addEventListener("change", () => {
    undoStack.length = 0;
    state.volume = els.volume.value;
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

  els.scroller.addEventListener("scroll", () => render(false));
  window.addEventListener("resize", () => {
    renderSortStack();
    render();
    if (!els.viewer.hidden) {
      paintViewer();
    }
  });

  async function loadVolumes() {
    state.me = await getJson(withSet("/api/me"));
    const volumes = state.me.volumes || [];
    els.who.textContent = `${state.me.displayName} (${state.config.identityMode})`;
    els.volume.innerHTML = volumes
      .map((item) => `<option value="${item.name}">${item.name} · ${item.permission}</option>`)
      .join("");
    const volumeWrap = els.volume.closest(".filter");
    if (volumes.length <= 1) {
      els.volume.style.display = volumes.length ? "none" : "";
      const volumeLabel = volumeWrap ? volumeWrap.querySelector("label") : null;
      if (volumeLabel) {
        volumeLabel.style.display = volumes.length ? "none" : "";
      }
    }
    state.volume = volumes[0] ? volumes[0].name : "";
    if (!volumes.length) {
      state.rows = [];
      els.status.textContent = "No volumes with Read access in this training set.";
      render();
      return;
    }
    await loadSettings();
    await loadCatalog(true);
  }

  function applyRefreshVisibility() {
    document.body.classList.toggle("can-refresh", Boolean(state.settings && state.settings.odata));
  }

  async function loadSettings() {
    if (!state.volume) {
      state.settings = null;
      applyRefreshVisibility();
      return;
    }
    state.settings = await getJson(withSet(`/api/volumes/${encodeURIComponent(state.volume)}/settings`));
    applyRefreshVisibility();
  }

  function openSettings() {
    const dialog = document.getElementById("settings-dialog");
    const endpoint = document.getElementById("settings-odata");
    const exporter = document.getElementById("settings-exporter");
    const current = state.settings || {};
    endpoint.value = current.odata || "";
    endpoint.placeholder = current.placeholder || `https://websvc.codepharm.net/${state.volume}/OData`;
    exporter.value = current.exporter === "legacy" ? "legacy" : "tiled";
    dialog.hidden = false;
  }

  async function saveSettings() {
    const endpoint = document.getElementById("settings-odata").value.trim();
    const exporter = document.getElementById("settings-exporter").value;
    state.settings = await getJson(withSet(`/api/volumes/${encodeURIComponent(state.volume)}/settings`), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ odata: endpoint, exporter }),
    });
    applyRefreshVisibility();
    document.getElementById("settings-dialog").hidden = true;
    els.status.textContent = state.settings.odata
      ? "Saved the endpoint for this volume."
      : "Cleared the endpoint for this volume.";
  }

  async function refreshLocation(locationId) {
    const openId = viewer.locationId;
    const openKey = viewer.imageKey;
    const result = await getJson(withSet(`/api/volumes/${encodeURIComponent(state.volume)}/refresh-location`), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ location_id: locationId }),
    });
    state.maskGeneration += 1;
    await loadCatalog(false, true);
    const count = (result.imageKeys || []).length;
    els.status.textContent = count
      ? `Refreshed ${count} crop mask${count === 1 ? "" : "s"}.`
      : "No existing crop for that location.";
    if (!els.viewer.hidden && String(openId) === String(locationId)) {
      const row = state.rows.find((item) =>
        String(item.location_id) === String(openId) && (item.image_key || "") === (openKey || "")
      );
      if (row) {
        await openViewerFromRow(row);
      }
    }
  }

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

  async function boot() {
    loadOverlayPrefs();
    loadCardSizePref();
    renderSortStack();
    state.config = await getJson("/api/config");
    const setsPayload = await getJson("/api/training-sets");
    const sets = setsPayload.sets && setsPayload.sets.length ? setsPayload.sets : ["current"];
    state.trainingSet = sets.includes("current") ? "current" : sets[0];
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
