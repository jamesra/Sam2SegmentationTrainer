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

  const GRID_GAP = 12;
  let rowStride = 0;
  let windowTop = -1;
  let packMemo = { key: null, packed: null };
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
    if (row.origin_x != null && row.origin_y != null && row.origin_x !== "" && row.origin_y !== "") {
      return { x: Number(row.origin_x), y: Number(row.origin_y) };
    }
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
      <article class="card" data-id="${row.location_id}" data-key="${row.image_key || ""}" style="grid-column:${cell.gc + 1};grid-row:${cell.gr - rowOffset + 1}">
        ${windowHtml(row)}
        ${windowActions(row)}
        <div class="meta"><div class="meta-row"><div>${captionHtml(row)}</div></div></div>
        ${refreshButton(row)}
      </article>`;
  }

  function refreshButton(row) {
    const icon = `<svg viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M12 6V3L8 7l4 4V8a4 4 0 1 1-4 4H6a6 6 0 1 0 6-6z"/></svg>`;
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

  function packGrid(cards, columns) {
    const occupied = new Set();
    const cells = [];
    let scanFrom = 0;
    let maxRow = 0;

    function isFree(row, col) {
      return col >= 0 && col < columns && row >= 0 && !occupied.has(row * columns + col);
    }

    function take(row, col, source) {
      occupied.add(row * columns + col);
      if (row + 1 > maxRow) {
        maxRow = row + 1;
      }
      while (occupied.has(scanFrom)) {
        scanFrom += 1;
      }
      cells.push({ row: source, gr: row, gc: col });
    }

    for (const windows of cards) {
      const offsets = shapeOffsets(windows, columns);
      if (!offsets) {
        const row = Math.floor(scanFrom / columns);
        take(row, scanFrom % columns, windows[0]);
        continue;
      }
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
          take(row + cell.dr, col + cell.dc, cell.row);
        }
        placed = true;
        break;
      }
      if (!placed) {
        for (const row of windows) {
          const index = scanFrom;
          take(Math.floor(index / columns), index % columns, row);
        }
      }
    }
    return { columns, totalRows: Math.max(maxRow, 1), cells };
  }

  function packKey(cards, columns) {
    let hash = 2166136261;
    const mix = (value) => {
      hash ^= value;
      hash = Math.imul(hash, 16777619);
    };
    mix(columns);
    mix(state.cropSize || 0);
    mix(cards.length);
    for (const windows of cards) {
      mix(Number(windows[0].location_id));
      for (const row of windows) {
        const origin = windowOrigin(row);
        mix(origin ? origin.x : 0);
        mix(origin ? origin.y : 0);
      }
    }
    return hash >>> 0;
  }

  function layoutCards(cards) {
    const cardPx = cardPixelSize();
    const columns = columnCount(els.grid.clientWidth);
    const key = packKey(cards, columns);
    if (!packMemo.packed || packMemo.key !== key) {
      packMemo = { key, packed: packGrid(cards, columns) };
    }
    const packed = packMemo.packed;
    const baseHeight = rowStride || (cardPx + 72);
    const rowOffsets = [0];
    for (let index = 0; index < packed.totalRows; index += 1) {
      rowOffsets.push((index + 1) * baseHeight);
    }
    return { cardPx, baseHeight, rowOffsets, ...packed };
  }

  function scrollToLocation() {
    const text = els.locate.value.trim();
    if (!text) {
      return;
    }
    const cards = cardsFrom(filtered());
    const layout = layoutCards(cards);
    const cell = layout.cells.find((item) => String(item.row.location_id) === text);
    if (!cell) {
      render(true);
      els.status.textContent = `Location ${text} is not in this view.`;
      return;
    }
    els.scroller.scrollTop = layout.rowOffsets[cell.gr] || 0;
    render(true);
    els.grid.querySelectorAll(".card.located").forEach((node) => node.classList.remove("located"));
    els.grid.querySelectorAll(".card").forEach((node) => {
      if (node.dataset.id === text) {
        node.classList.add("located");
      }
    });
    els.status.textContent = `Location ${text}.`;
  }

  function columnCount(width) {
    const card = cardPixelSize();
    if (width <= 0) {
      return 1;
    }
    return Math.max(1, Math.floor((width + GRID_GAP) / (card + GRID_GAP)));
  }

  function render(force = true) {
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
    const rows = filtered();
    const cards = cardsFrom(rows);
    const layout = layoutCards(cards);
    const { cardPx, columns, baseHeight, totalRows, rowOffsets, cells } = layout;
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
    els.grid.innerHTML = visible.map((cell) => cellHtml(cell, first)).join("");
    paintVisible();
    const card = els.grid.querySelector(".card");
    if (!card) {
      return;
    }
    const measured = card.offsetHeight + GRID_GAP;
    if (!measuringRow && measured > 1 && Math.abs(measured - rowStride) > 1) {
      measuringRow = true;
      rowStride = measured;
      windowTop = -1;
      render(true);
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
      render();
      return;
    }
    const payload = await getJson(withSet(`/api/volumes/${encodeURIComponent(state.volume)}/catalog`));
    state.rows = payload.rows || [];
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

  function rememberStatus(row) {
    if (!row) {
      return;
    }
    undoStack.push({
      locationId: Number(row.location_id),
      imageKey: row.image_key || "",
      ignored: Boolean(row.ignored),
      approved: Boolean(row.approved),
    });
    if (undoStack.length > 50) {
      undoStack.shift();
    }
  }

  function findStatusRow(locationId, imageKey) {
    return state.rows.find((row) =>
      Number(row.location_id) === Number(locationId) && (row.image_key || "") === (imageKey || "")
    );
  }

  async function postReview(act, locationId, imageKey) {
    await getJson(withSet(`/api/volumes/${encodeURIComponent(state.volume)}/${act}`), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ location_id: locationId, image_key: imageKey || undefined }),
    });
  }

  async function mutate(act, locationId, imageKey) {
    rememberStatus(findStatusRow(locationId, imageKey));
    await postReview(act, locationId, imageKey);
    await loadCatalog();
  }

  async function applyRememberedStatus(entry) {
    let row = findStatusRow(entry.locationId, entry.imageKey);
    if (entry.ignored) {
      if (!row || !row.ignored) {
        await postReview("ignore", entry.locationId, entry.imageKey);
        await loadCatalog(false, true);
        row = findStatusRow(entry.locationId, entry.imageKey);
      }
    } else if (row && row.ignored) {
      await postReview("restore", entry.locationId, entry.imageKey);
      await loadCatalog(false, true);
      row = findStatusRow(entry.locationId, entry.imageKey);
    }
    if (entry.approved) {
      if (!row || !row.approved) {
        await postReview("approve", entry.locationId, entry.imageKey);
      }
    } else if (row && row.approved) {
      await postReview("unapprove", entry.locationId, entry.imageKey);
    }
    await loadCatalog(false, true);
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
    return layoutCards(cardsFrom(filtered())).cells;
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
    const next = index >= 0 && cells[index + 1] ? cells[index + 1].row : null;
    rememberStatus(viewerRow());
    await postReview(act, Number(viewer.locationId), viewer.imageKey || "");
    await loadCatalog(false, true);
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
      mutate(button.dataset.act, Number(button.dataset.id), button.dataset.key).catch((error) => {
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

  const keysButton = document.getElementById("viewer-keys");
  const keysHelp = document.getElementById("viewer-keys-help");
  keysButton.addEventListener("click", (event) => {
    event.stopPropagation();
    const open = keysHelp.hidden;
    keysHelp.hidden = !open;
    keysButton.setAttribute("aria-expanded", open ? "true" : "false");
  });
  document.addEventListener("click", (event) => {
    if (keysHelp.hidden || keysButton.contains(event.target) || keysHelp.contains(event.target)) {
      return;
    }
    keysHelp.hidden = true;
    keysButton.setAttribute("aria-expanded", "false");
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
    if (event.key === "Escape" && !els.viewer.hidden) {
      closeViewer();
      return;
    }
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
        state.tagIds = new Set();
        render();
      });
    }, 300);
  }

  async function lookupTags() {
    const query = els.tags.value.trim();
    if (!query || !state.volume) {
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
