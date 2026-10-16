# SAM2 v4 Training — Mixed Prompt & Negative Mask Plan

Oct 11, 2026 · @James · Revised Oct 13, 2026 to match the implementation in `Sam2Trainer` and the nornir exporter; Goals added Oct 13; circle-annotation trust added Oct 15

## Goals

**What success means.** The goal is to reliably identify cells and the connectivity between cells, not to trace every membrane perfectly.
- Some noise at the boundary is acceptable. A perfect membrane trace is a shoreline problem with little scientific value.
- **Target:** capture at least 97.5% of each cell's interior volume without merging into a neighbor. Meeting that reliably is an excellent result.
- **Merges and splits matter more than edge accuracy.** A prediction that leaks into a neighboring cell, or breaks one cell into pieces, corrupts connectivity. A boundary a few tens of nanometres off does not.

**What the annotations are like.**
- Good traces average about ±20 nm (about 9 px at 2.18 nm/px) from the membrane.
- Where the membrane wobbles, traces make excursions of 30–50 px (65–110 nm) before coming back.
- The training targets carry this noise. Losses, prompt perturbations and metrics are scaled to it (*claude/Mask refinement and edge tolerance plan*).
- **Reference set:** masks marked Approved in the annotation gallery (`approved.json`) are the highest-quality examples. They are reported separately in validation.

**Roadmap.**
1. **Now:** interactive segmentation of cell profiles from circle, adjacent-section and mask prompts (this plan and the refinement plan).
2. **Near future:** section-to-section propagation, treating the z-stack as video, so a cell traced on one section is carried to the next.
3. **Later:** sorting cells into broad classes from intracellular texture. RGB overlays of small-molecule signals registered to the EM (GABA, glycine, glutamate and others) are also available for this but have not been added to the export yet.
4. **Then:** subcellular features, and SEM as a secondary modality.

## Revision note (Oct 13)

This plan now describes what the code does. The changes:
- **Prompt mix:** circle-mask probability is 0.45; the table summed to 0.95 before.
- **Boxes:** passed to the encoder as `boxes=`; the box containment loss and box clamp are removed.
- **Dense masks:** fed as ±10 logits at 256×256.
- **`mask_downscaling` size:** ≈4.7K parameters, not 50K.
- **Circles:** for polygon targets they are inscribed circles, never clipped; a `source` field was added. The exporter writes true radii (`diameter_scale` fixed at 1.0, nornir commit `3b6fe2f`, which dropped `-CircleDiameterScale`); the trainer applies `prompt.circle_diameter_scale` (0.95) to inscribed (seed) circles when it loads the file. Since Oct 15 (encoding `edge-confidence-v6`), annotated circles load at their true radius and are trusted by `prompt.circle_annotation_trust` instead.
- **Caching and decoding:** neighbor masks are cached per worker, and samples are decoded in groups by prompt shape.
- **Negative-only prompt removed from the plan (decided Oct 13; code not yet changed).** The prompt type with no foreground cue is dropped, together with its optional circle targets. Other annotations stay as −1 context and background points on every remaining type. Its 5% share is proposed to go to circle-mask (45% → 50%).

Recommendations are kept separately in *claude/Training plan assessment*; the SAM 3 question is in *claude/SAM3 evaluation*.

## Architecture Decisions

Three design choices govern every component below.

**Decision 1 — JSON index for circle annotations, not pre-generated masks.**
- Rasterizing a filled circle on a 1024×1024 canvas takes < 0.5 ms.
- The composite negative mask also depends on which *other* positive masks share the grid cell, which is only known at training time.
- A JSON file is the source of truth: you can change radii or add annotations without re-exporting raster files.
- The trainer composites lazily per sample and keeps a small per-worker LRU of decoded neighbor masks (Part 2, Step 3).
- The gallery rasterizes on demand with a disk cache.

**Decision 2 — Grid cell index built from the manifest.**
- Mask filenames already encode the crop window through `imageKey` (e.g. `RC2_73_D1_X44032-45056_Y42496-43520`). "Grid cell" in this plan means that window. With the tiled exporter, windows are not cells of one fixed grid and can overlap (Part 1, *Crop window placement*).
- The manifest's `locationIds` array lists every annotation in each tile.
- At trainer startup, `build_tile_context()` builds `dict[(volume, imageKey)] → location ids` from the manifest, an O(1) lookup for "what else shares my tile". No second index file is needed.

**Decision 3 — Signed mask convention: −1 / 0 / +1 (int8).**
- −1 = definite background (another annotation's territory); 0 = unknown (no information); +1 = definite foreground (the target annotation).
- Where a foreground region (+1) overlaps a background region (−1), the sum is 0 = unknown. The conflict becomes uncertainty, not a training signal.
- **Encoding for SAM2:** the canvas is nearest-neighbor resized to SAM2's 256×256 `mask_input` and multiplied by `MASK_PROMPT_LOGIT = 10`, giving +10 / 0 / −10 logits.
  - SAM2's `mask_downscaling` CNN was trained on low-resolution *logits* of earlier predictions, and SAM2 itself converts binary masks to ±10 logits.
  - **Inference must use the same ±10 scale**, or the trained mask network sees inputs on a different scale.
- `mask_downscaling` is unfrozen (`model.unfreeze_mask_cnn: true`) so it learns the three-valued semantic. The rest of `sam_prompt_encoder` stays frozen.

## Part 1 — Nornir Exporter

Existing positive mask and image tile exports are unchanged. The exporter (`nornir_buildmanager/operations/segmentationtraining/circles.py`) adds one output: a JSON file per volume listing circles by grid cell. The file is rebuilt from current data on every export, including `-Update`, `-Repair` and `-Cleanup`.

### Crop window placement (clarification, Oct 13)

The exporter has two placement paths, chosen with `-Exporter` (`planning.py` → `geometry.py`). This section records what the code does. No change has been decided.

**`tiled` (the default).** The crop stays at `-Downsample` and `place_mask_windows()` positions the windows.
- **A mask that fits in one 1024 window** gets the single window, from a **512-px half-step lattice**, whose center is closest to the mask centroid. The lattice guarantees that any object up to 512 px across fits whole in some window.
- **A larger mask** is split into **non-overlapping** 1024 windows. The grid is shifted by one of four phases (0 or 512 in x and in y), chosen so the smallest clipped piece is as large as possible, which avoids slivers.
- **Shared windows:** identical windows chosen for different locations become one shared image (`group_windows`).
- **Result:**
  - one location's windows never overlap each other;
  - windows of *different* locations sit on the shared 512 lattice and can overlap by 50% or 25%;
  - a window may extend past the mosaic edge, giving keys with negative origins such as `X-512-512`.

**`legacy`.** Coarsens the downsample until the mask fits one crop, and never splits. It does not use the half-step lattice.

**The v2 export is tiled** (all tiles D1). Measured on the v2 manifests, Oct 13:

| Volume | Windows | Off the 1024 grid | Overlap another window | Single-window objects a fixed 1024 grid would cut |
| --- | --- | --- | --- | --- |
| RC1 | 53,057 | 75% | 49.2% | 56% (33/59 sampled) |
| RC2 | 9,706 | 75% | 48.5% | 60% (37/62) |
| RPC1 | 71,826 | 75% | 42.3% | 63% (17/27) |
| RPC2 | 52,317 | 74% | 17.2% | 69% (20/29) |

**Decision.**
- **Option 1, a fixed non-overlapping 1024 grid** in the tiled path, would remove the overlap. But most objects that now fit one window would be cut into 2–4 pieces.
- **Option 2, keep the lattice** and handle leakage in the split.
- **Resolved Oct 13: keep the lattice.** The z-slab split (Part 3) puts all windows of a section on one side, so overlap no longer crosses the split. See *claude/Training plan assessment* 1.2.

**Negative window origins (fixed).** `circles.py` once rejected keys with negative origins such as `X-512-512` (148 crops in RPC1, 2 in RPC2). nornir commit `87a7b89` accepts them.

**Output file:** `{AnnotationCrops}/circle_annotations.json`

| Field | Type | Description |
| --- | --- | --- |
| version | int | Schema version (1) |
| diameter_scale | float | Always 1.0: radii are written unscaled. A file from an older exporter may carry another value, which the trainer divides out |
| grid_cells | object | Map from imageKey to list of circle entries |
| grid_cells[imageKey][].location_id | int | Annotation location id |
| grid_cells[imageKey][].center_x | float | Circle center x in tile-local pixels (may lie outside the tile) |
| grid_cells[imageKey][].center_y | float | Circle center y in tile-local pixels (may lie outside the tile) |
| grid_cells[imageKey][].radius | float | True circle radius in pixels: the annotated circle as drawn, or the full inscribed circle |
| grid_cells[imageKey][].structure_label | string | Category name if available |
| grid_cells[imageKey][].has_polygon_mask | bool | Whether a polygon mask exists for this annotation in the manifest |
| grid_cells[imageKey][].source | string | annotated (a Viking circle location) or "inscribed" (largest circle inside a polygon location) |

**Example JSON structure:**

```json
{
  "version": 1,
  "diameter_scale": 1.0,
  "grid_cells": {
    "RC2_73_D1_X44032-45056_Y42496-43520": [
      {
        "location_id": 455300,
        "center_x": 640.0,
        "center_y": 512.0,
        "radius": 120.0,
        "structure_label": "mitochondria",
        "has_polygon_mask": false,
        "source": "annotated"
      }
    ]
  }
}
```

**Which circles are exported.** In Viking a location is *either* a circle *or* a polygon, so the file holds two kinds of entry:

- **`annotated`:** every Viking circle location (`TypeCode == 1`). The center comes from its `CURVEPOLYGON(CIRCULARSTRING …)` MosaicShape and the radius from the stored `Radius`. These have `has_polygon_mask = false`. They are negative context in other tiles.
- **`inscribed`:** every polygon location that has a mask. The circle is the largest circle inside the polygon's exterior ring, found with a distance transform and refined against the exact boundary; holes are ignored. These have `has_polygon_mask = true` and power the circle-mask prompt type (the circle is the prompt, the polygon mask is the ground truth).
  - The centroid and equal-area radius are deliberately not used, because a C-shaped polygon's centroid can lie outside it.
  - Inscribed circles are cached per section in `_work/circle_inscribed_{z}.json`, keyed by location id and LastModified.

**Exporter logic:**

1. For each section, load the annotated circles (separate OData query: `TypeCode eq 1`, same parent filter as the polygon query; it also selects `Username`) and compute inscribed circles for the manifest's masked locations.
2. Write every radius unscaled and record `diameter_scale: 1.0`. Shrinking circles is the trainer's job.
3. Convert each circle to crop pixels at the crop's downsample. Write an entry to every crop it overlaps.
   - Centers and radii are **never clipped**: a circle spanning several tiles keeps its true center in each tile's coordinates, even when that center lies outside the tile.
   - The rasterizer clips at draw time.
4. Only manifest rows whose crop image still exists contribute. An unchanged file is not rewritten.

**Circle scale belongs to the trainer.** The exporter has no scale option (`-CircleDiameterScale` was removed in nornir `3b6fe2f`). The exporter never scales: file radii are true radii (`diameter_scale` 1.0). The trainer's `prompt.circle_diameter_scale` (default 0.95) applies to **inscribed** circles only (the seed circles), so the scale can change without a re-export. **Annotated** circles load at their true radius, since their radii are unreliable and they are only negatives; they are trusted by `prompt.circle_annotation_trust` (`full_frac` 0.5, `zero_frac` 0.95, encoding `edge-confidence-v6`). An older file's baked-in scale is divided out for annotated circles and rescaled to the configured value for inscribed ones.

## Part 2 — Trainer Data Pipeline

Negative context is built per sample in the DataLoader workers, from indices built once at startup.

### Step 1: Build the grid cell index

`build_tile_context()` (`data/tiles.py`) builds a `TileContext` from the training inventory:

```
neighbors: dict[(volume, imageKey), tuple[location_id, ...]]       # locations with a mask PNG in the tile
circles:   dict[(volume, imageKey), tuple[CircleAnnotation, ...]]  # from circle_annotations.json
```

Both are built once and held in memory. They are small: one dict entry per tile, and a few numbers per annotation. A missing or unreadable circle file yields no circles; training still runs, with negatives from masks only.

### Step 2: Composite the negative mask

The shared code is `annotation_crops/negatives.py`. For a training sample targeting `location_id = L` in tile `K`, `collect_negative_regions()` returns each other annotation as its own boolean raster:

1. **Other polygon masks:** every neighbor id ≠ L whose `masks/{K}_{id}.png` exists, thresholded to binary.
2. **Circle annotations:** every circle in the tile whose location is neither L nor a neighbor already loaded as a mask. So a polygon is never drawn twice, and its inscribed circle is not stacked on top of it. An annotated circle claims only the pixels within `zero_frac` of its true radius: full strength inside `full_frac`, tapering to 0 at `zero_frac`. Its core, used for the leak weight, the merge metric and its background points, is the part inside `full_frac` (*claude/Mask refinement and edge tolerance plan*, sections 2 and 3.6).

`stamp_negative_regions()` writes −1 wherever any region is set (0 elsewhere). `combine_signed(foreground, negative)` adds the +1 foreground and resolves overlaps to 0 (unknown). The result is a signed int8 array: −1 background, 0 unknown, +1 foreground.

The target, every negative region and the target's circle all go through the same geometric augmentation as the image, so prompts stay registered to it.

### Step 3: Caching strategy

Compositing reads one to three neighbor mask PNGs per sample from CIFS.
- Pre-compositing every sample (~180K × 1024² bytes ≈ 185 GB) is too large, so compositing is **lazy per sample**.
- Each DataLoader worker keeps a `MaskCache`: an LRU of *decoded neighbor masks* keyed by file path, `negative.cache_size` entries (default 64).
- Most tiles have zero or one neighbor, so most composites are trivial.
- When a sample has no dense content (no negatives and no foreground mask), the dense prompt is `None`, so the prompt encoder uses its learned no-mask embedding. That is what the model sees for most samples.

**Shared compositing function.** The trainer and the gallery call the same code, so what an annotator sees as background context is exactly what training uses:

```python
def composite_negative_mask(
    image_key: str,
    target_location_id: int,
    neighbor_location_ids: Iterable[int],
    circles: Sequence[CircleAnnotation],
    mask_dir: Path,
    shape: tuple[int, int],
    cache: MaskCache | None = None,
) -> np.ndarray:
    """int8 array: -1 at every other annotation in the tile, 0 elsewhere.
    Caller adds +1 foreground with combine_signed() and decides rendering vs training use."""
```

The trainer uses `collect_negative_regions()` directly, because it needs the regions individually for background points and augmentation. The gallery uses the same function and caches its result as a bitmask (Part 6).

### Step 4: Background point sampling from negatives

Independent of the dense mask prompt, negative regions can also produce background *points* (sparse prompt, `point_label=0`). This teaches the model to use both the sparse and the dense negative channels.

- **Circle annotations → background points:** with probability `bg_point_prob_circle` (default 0.25), each circle region becomes 1–9 background points (for an annotated circle, sampled only from its core). The sampling algorithm is the same as for foreground points: distance transform, farthest-point selection with line-fraction placement at 40–90%, and a Benford's-law count.
- **Other polygon masks → background points:** with probability `bg_point_prob_polygon` (default 0.15), each other polygon mask becomes 1–9 background points, sampled the same way.
- Background points are appended after the foreground points. SAM2 handles mixed `point_labels` natively (1 = foreground, 0 = background).
- Each negative region is rolled independently, so a sample can have zero, one or several sets of background points.

The polygon rate (0.15, raised from 0.10) reflects that polygon masks have precise boundaries, so their background points are higher quality than circles'.

## Part 3 — Mixed Prompt Strategy

Each training sample draws one prompt type from the table below. The negative mask and background points (Part 2, Step 4) are added on top of whichever type is drawn, whenever they exist.

**Availability:**
- `circle_mask` needs an inscribed circle for the target whose center lies in the tile.
- When a sample cannot support it, that type's probability moves to the remaining types in proportion.
- If augmentation empties the circle, the sample falls back to `full_mask`.

**Validation** always uses the original single centroid point, so IoU stays comparable across runs.

**Train/validation split.**
- **Since Oct 13 (`data.split_mode: zslab`; commit `ba05ea3` on branch `zslab-split`):** split by z-slab within each volume (`make_zslab_split` in `data/splits.py`).
  - A validation fraction X (default 10%) of each volume goes to 5 contiguous z-slabs spaced evenly through the volume, centered at 10/30/50/70/90% of its z-range. Each slab holds about X/5 of the volume's samples.
  - Slabs grow from their centers one section at a time, weighted by crop-window samples, and stop at the size nearest X/5.
  - A guard band of 1 section on each side of each slab (`data.zslab.guard_sections`) is used for neither split.
  - Every window of a section lands on one side, so neither adjacent sections nor overlapping windows leak across the split.
  - Keys that appear after the split (an inventory refresh) are placed by their section.
  - `split.json` records `mode`, `val_z` and `guard_z`.
  - **Switching:** `data.split_mode: location` restores the old random 90/10 over locations. An existing run keeps its own `split.json`; adopting the new split needs `-refresh`.
  - See *claude/Training plan assessment* 1.2.

| Prompt type | Probability | Foreground sparse | Dense mask input | Rationale |
| --- | --- | --- | --- | --- |
| Points only | 10% | 1–9 foreground points (Benford count, distance-transform + line-fraction 40–90% placement) | −1 context mask (−1 at other annotations, 0 elsewhere), or None if there are none | Legacy manual workflow; kept for backward compatibility and quick corrections |
| Box only | 5% | Jittered bounding box, passed as `boxes=` | −1 context mask or None | Keeps box-prompt coverage; the auto-seg pipeline prefers circle masks |
| Box + points | 5% | Box + 1–3 foreground points inside (uniform count) | −1 context mask or None | Matches the overlap-resolution path in the inference pipeline |
| Circle mask | 50% (code: 45% until the change is implemented) | 1 foreground point at the circle's center (label 1) | Circle rasterized as +1, negatives as −1, 0 elsewhere. Loss compares to the full ground-truth polygon mask | Matches the production prompt; trains grow-outward from the circle to the full boundary |
| Full mask prompt | 30% | 1 foreground point inside the polygon (distance-transform placement, label 1) | Full signed mask: +1 target, −1 negatives, 0 unknown | Covers the dense prompt path; closest proxy for brush-painted masks in manual segmentation |

All probabilities are configurable (Part 7). `PromptConfig.validate()` rejects a set that does not sum to 1.0.

**Removed: negative-only prompt (decided Oct 13; code not yet changed).**
- The trainer still has a sixth type, `negative_only` (5%): background points and the −1 mask with no foreground cue, plus optional circle-disc targets (`negative.allow_circle_targets`, off by default).
- It is removed because, with no positive cue, nothing identifies which unannotated profile is the target. It trains guessing and miscalibrates the IoU head (*claude/Training plan assessment* 1.5).
- **Pending code change:**
  - remove `PromptType.NEGATIVE_ONLY` and `prompt.prob_negative_only`;
  - remove `negative.allow_circle_targets`, `negative.circle_target_fraction` and `tiles.circle_target_examples`;
  - set `prompt.prob_circle_mask` to 0.50;
  - update the tests.

### Box generation details

The bounding box is derived from the target's (augmented) ground-truth mask:

1. Compute the tight bounding box of the binary mask.
2. Push each edge out by `uniform(0, jitter_frac × edge length)`, with `jitter_frac` defaulting to 0.15. The object therefore always stays inside the box; the jitter only makes the box imprecise.
3. Clamp to the image bounds.

A box is only a hint that the object is most likely inside it (`.claude/rules/sam2-prompt-semantics.md`). No loss forces predictions into the box, and nothing clamps them at inference.

In the prompt encoder, boxes go in as `boxes=` in xyxy pixels scaled to `image_size`. SAM2 adds its own corner embeddings, so this is equivalent to corner points with labels 2 and 3.

### Point generation details

Foreground points (`sample_points_in_mask()`):

1. Compute `scipy.ndimage.distance_transform_edt` of the binary mask.
2. Discard pixels within `min_edge_px` (default 15) of the boundary. A mask too thin for that uses half its own maximum depth instead.
3. Draw the point count from Benford's law: P(d) = log10(1 + 1/d) for d = 1…9, so a single point is 30.1% of samples.
4. The first point is the pixel with the maximum distance-transform value (deepest interior).
5. Each later point: find the candidate farthest from all selected points, then place the new point a random fraction (40–90%) of the way from the nearest selected point toward it. If that spot falls outside the mask, the far pixel itself is used.

For the "box + points" type, use 1–3 points (uniform) rather than 1–9. The box already gives a strong spatial signal, so fewer extra points better match the inference use case.

### Circle mask generation

This prompt type uses the target's circle from `circle_annotations.json` as the mask prompt. The model must recover the full polygon boundary from it.

1. Look up the circle for the target `location_id`. For a polygon target this is its **inscribed** circle (`source = "inscribed"`), scaled on load by the trainer's `prompt.circle_diameter_scale` (default 0.95). The circle's center must lie in the tile.
2. Rasterize the circle → +1.
3. Composite the negatives as usual: other polygon masks and circles in the tile → −1. An overlap of +1 and −1 resolves to 0.
4. The sparse prompt is one foreground point at the circle's center (`point_label=1`).
5. The loss compares the output to the **full ground-truth polygon mask**, not the circle. The model must learn to expand from the circle to the true cell boundary.

**Relation to production.** Production starts from the annotator's Viking circle. Annotators are instructed to draw circles inscribed in the cell, so the inscribed circle of the finished polygon is the training stand-in for that prompt. Circles that break the rule are listed for review in `Y:\SAM2TrainingData\circle_review\` (from `scripts/circle_review.py`).

### What always accompanies the prompt

Regardless of prompt type:

1. **Negative mask (dense):**
   - If any other annotation shares the tile, its −1 regions are always in the dense prompt.
   - For "circle mask" and "full mask prompt" they share one signed mask with the foreground. For the points and box types the dense prompt holds only the −1 context.
   - When the dense canvas is empty, the dense prompt is `None` (learned no-mask embedding).
2. **Background points (sparse):** rolled independently per negative region, per the probabilities in Part 2, Step 4, and appended with `point_label=0`.

## Part 4 — Model Code Changes

### 4a. Prompt encoding and decoding in `model.py`

The image encoder and the prompt path are split so one batch can mix prompt shapes:

- **`encode_images(model, images)`** runs the image encoder once and returns `(image_embed, high_res_features)`.
- **`decode_prompts(model, image_embed, high_res, point_coords, point_labels, *, image_size, orig_hw, boxes=None, dense_masks=None)`** runs the prompt encoder and the mask decoder for samples that share one prompt structure.
  - Points and boxes are scaled from `orig_hw` to `image_size`.
  - `boxes` is `[B, 4]` xyxy, passed as `boxes=`.
  - `dense_masks` is `[B, 1, 256, 256]` float `mask_input` logits.
  - None selects the encoder's learned no-box / no-mask embedding, which an all-zero tensor would not.
  - At least one point or box is required.
- **`predict_masks_mixed(…, boxes: list, dense_masks: list)`** encodes once, groups samples by (has box, has dense mask), decodes each group, and scatters results back to batch order. The prompt encoder takes one `boxes` and one `masks` tensor per call, which is why samples are grouped.
- **`predict_masks()`** stays the points-only path used by validation, evaluation and benchmarks.

`multimask_output=False` throughout.

### 4b. `configure_trainable_params(model, freeze_mode, unfreeze_mask_cnn)`

- `freeze_mode="none"` trains everything except `sam_prompt_encoder`.
- With `unfreeze_mask_cnn=True` (config `model.unfreeze_mask_cnn`, default true), `sam_prompt_encoder.mask_downscaling` is unfrozen and joins the non-encoder optimizer group. The point and box embedding tables stay frozen.

```
for param in model.sam_prompt_encoder.mask_downscaling.parameters():
    param.requires_grad = True
```

- `mask_downscaling` is three convolutions (1→4→16→256 channels, with LayerNorm): about **4.7K parameters**, so the training cost is negligible.
- Changing this setting changes the optimizer's parameter list, so a run cannot resume across it; use `-refresh`.

### 4c. `EMSegDataset.__getitem__()`

With a `TileContext` and a `PromptConfig`, each sample returns:

```
{
    "image": image,
    "mask": ground_truth_mask,       # target for loss, (H, W) int64
    "point": point_coords,           # (N, 2) variable-length; foreground first, then background
    "point_label": point_labels,     # (N,) with 1=fg, 0=bg
    "boxes": box_tensor_or_none,     # (4,) xyxy or None
    "dense_mask": low_res_logits,    # (256, 256) float logits (±10 / 0) or None
    "prompt_fg": foreground_or_none, # (H, W) uint8: the +1 region of the dense mask, for mask containment
    "prompt_type": prompt_type_str,  # for logging
    "orig_hw": (H, W),
    "sample_meta": meta,
}
```

- Prompt draws are seeded from `(seed, epoch, index)`, with the epoch shared with persistent workers through `EpochCounter`. A resumed epoch therefore replays the same prompts.
- Without a `TileContext` (validation, evaluation), a sample is the single centroid point.

### 4d. `collate_em()`

- Points are padded to the largest count in the batch, with label −1 for padding (SAM2 treats −1 as "not a point").
- `boxes` and `dense_mask` stay lists with None entries; `predict_masks_mixed` groups samples by which prompts they have.
- `prompt_fg` is stacked, with zeros where a sample has no foreground mask.

```
def collate_em(batch):
    most = max(item["point"].shape[0] for item in batch)
    points = zeros(B, most, 2); labels = full((B, most), -1)
    # copy each item's points/labels into the first n slots
    return {
        "image": stack(images), "mask": stack(masks),
        "point": points, "point_label": labels,
        "boxes": [item.get("boxes") for item in batch],
        "dense_mask": [item.get("dense_mask") for item in batch],
        "prompt_fg": stack(prompt_fg or zeros),
        "prompt_type": [...], "orig_hw": [...], "meta": [...],
    }
```

## Part 5 — Containment Loss

One one-sided loss enforces the meaning of a dense mask prompt. It is zero when the model already respects the constraint, so it acts as a guardrail rather than a primary training signal.

### Mask containment loss (grow-outward)

When a dense mask prompt has a +1 region, every pixel in it should be foreground in the output. `mask_containment_loss_per_sample()` (`losses.py`) penalizes predictions that shrink from the mask:

```
def mask_containment_loss_per_sample(pred_logits, foreground):
    """pred_logits: [B, H, W]; foreground: [B, H, W], non-zero where the dense prompt was +1."""
    fg = (foreground != 0).to(pred_logits.dtype)
    violation = F.relu(-pred_logits) * fg
    return violation.flatten(1).sum(1) / fg.flatten(1).sum(1).clamp_min(1.0)
```

It is zero when `pred_logits > 0` everywhere the prompt is +1; the model is free to grow outward beyond the mask. A sample with no foreground region contributes 0.

**There is deliberately no box containment loss.** A box is only a hint that the object is most likely inside it.

### Integration into `combined_loss()`

```
loss = mean_over_batch(
    focal_weight * loss_focal
    + dice_weight * loss_dice
    + iou_weight * loss_iou
    + mask_contain_weight * warmup * loss_mask_contain
    + containment_weight * loss_point_containment   # optional, default 0 (off)
)
```

- The default `mask_contain_weight` is 0.5, light relative to focal (20.0) and dice (1.0).
- `loss_mask_contain` is logged to TensorBoard and should approach zero within the first 2–3 epochs.
- The optional point-containment term (`loss.containment_weight`, `loss.containment_radius`) pushes logits toward foreground inside a small disk around each positive point. It is off by default.

**Warmup schedule.** The mask containment weight ramps linearly from 0 to its target over the first `containment_warmup_epochs` (default 2).
- `containment_warmup_scale()` uses the fractional epoch, `(epoch + step / steps_per_epoch) / warmup_epochs`, capped at 1, so the first epoch is not entirely at zero.
- This gives focal and dice losses time to establish a baseline before containment activates.

### Inference-time enforcement (planned; not in `infer.py`)

At inference, hard-clamp logits after the decoder where a mask prompt was given:

- **Mask prompt +1 region:** `pred_logits = max(pred_logits, +6.0)` there. sigmoid(6) ≈ 0.998, effectively guaranteed foreground.

The clamp is applied after the forward pass and before thresholding, as a safety net. Boxes are not clamped.

## Part 6 — Gallery Changes

The gallery shows negative masks with browser-side HSL blending, the same rendering pipeline used for foreground masks.
- Masks are stored as 1-bit bitmasks and colorized at render time, so annotators can verify exactly what the trainer sees as background context.
- Colors and opacity are adjustable without regenerating cached files.

**Data source.** `annotation-gallery/negatives_view.py` calls `annotation_crops.negatives.collect_negative_regions()`, the function the trainer calls, so the two cannot drift. It reads `circle_annotations.json` and the tile's other mask PNGs (the same `imageKey` → `locationIds` mapping as the manifest).

**Rendering.** When displaying a tile for annotation `location_id = L`:

1. Collect the other annotations' regions and stamp them to a −1 canvas (Part 2, Step 2).
2. Save the result as an 8-bit grayscale PNG (0 = no mask, 255 = mask region). Typical sparse masks compress to ~5–20 KB, against ~100–400 KB for a pre-rendered RGBA overlay. The PNG carries the number of negative annotations in a text field.
3. The browser loads each binary mask (negative and foreground) as a texture and colorizes it through the HSL blending pipeline. That pipeline was written from scratch, because neither Viking nor Nornir Pyre exposes reusable blending code. Negative regions get a red hue; the target foreground gets green/blue.
4. Where positive and negative masks overlap (the unknown zone), the pipeline assigns a distinct hue (e.g. yellow). All colors and opacity are configurable at render time.

**Caching.**
- **Cache path:** `{AnnotationCrops}/cache/neg_{imageKey}_{locationId}.png`.
- **Invalidation:** the cache is reused until the sidecar, `circle_annotations.json` or any neighbor mask is newer than it.
- **Read-only shares:** a cache that cannot be written is ignored.

**Gallery UI:** a toggle for the negative overlay (on by default when negatives exist) and a count badge, e.g. "3 negative annotations", in the tile's info panel.

## Part 7 — Configuration Parameters

All parameters live in the trainer's YAML config (`config/em_finetune.yaml`).

| Parameter | Default | Description |
| --- | --- | --- |
| `prompt.enabled` | true | false: every training sample is the original single centroid point |
| `prompt.prob_points_only` | 0.10 | Probability of the points-only prompt type |
| `prompt.prob_box_only` | 0.05 | Probability of the box-only prompt type |
| `prompt.prob_box_points` | 0.05 | Probability of the box + points prompt type |
| `prompt.prob_circle_mask` | 0.50 (code: 0.45) | Probability of the circle-mask prompt type |
| `prompt.prob_full_mask` | 0.30 | Probability of the full-mask prompt type |
| `prompt.box_jitter_frac` | 0.15 | Max fractional outward push on each box edge |
| `prompt.box_points_max` | 3 | Max foreground points in box + points mode |
| `prompt.min_edge_px` | 15 | Points stay at least this far (pixels) from the mask boundary |
| `prompt.line_frac_min` | 0.40 | Minimum line fraction for point placement |
| `prompt.line_frac_max` | 0.90 | Maximum line fraction for point placement |
| `negative.bg_point_prob_circle` | 0.25 | Probability a circle region yields background points |
| `negative.bg_point_prob_polygon` | 0.15 | Probability a polygon mask yields background points |
| `negative.cache_size` | 64 | Decoded neighbor masks kept per DataLoader worker (LRU) |
| `data.train_split` | 0.9 | Training fraction; validation gets the rest (about 10%) |
| `data.split_mode` | zslab | `zslab`: evenly spaced z-slabs per volume; `location`: random 90/10 over locations |
| `data.zslab.slab_count` | 5 | Validation slabs per volume, centered at 10/30/50/70/90% of the z-range |
| `data.zslab.guard_sections` | 1 | Sections on each side of every slab used by neither side |
| `data.split_name` | zslab_v1 | Name of a legacy shared split file (a run's own `split.json` takes precedence) |
| `loss.mask_contain_weight` | 0.5 | Weight for mask containment loss (0 disables) |
| `loss.containment_warmup_epochs` | 2 | Epochs over which mask containment ramps linearly from 0 to its target weight |
| `loss.containment_weight` | 0.0 | Optional point-containment loss weight (0 disables) |
| `loss.containment_radius` | 8.0 | Disk radius (pixels) around positive points for point containment |
| `loss.focal_weight` | 20.0 | Weight for focal loss (unchanged) |
| `loss.dice_weight` | 1.0 | Weight for dice loss (unchanged) |
| `loss.iou_weight` | 1.0 | Weight for IoU prediction loss (unchanged) |
| `model.unfreeze_mask_cnn` | true | Unfreeze `mask_downscaling`; changing it needs `-refresh`, not resume |

Circle scale: `prompt.circle_diameter_scale` (default 0.95), applied by the trainer to inscribed (seed) circles only. Annotated circles are trusted by `prompt.circle_annotation_trust` (`full_frac` 0.5, `zero_frac` 0.95). The exporter never scales; it writes true radii.

Prompt probabilities must sum to 1.0; the config loader validates this.

## Feedback & Recommended Adjustments

See *claude/Training plan assessment* (Oct 12, updated Oct 13, 2026), *claude/SAM3 evaluation* and *claude/Mask refinement and edge tolerance plan*. The refinement plan replaces the full-mask prompt (Part 3), the ±10 binary encoding (Decision 3) and IoU-only validation with goal-driven metrics (core recall, leak, fragmentation).
