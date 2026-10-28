# Mask Refinement, Edge Tolerance and Multi-Prompt Training Plan

Oct 13, 2026 · Phase A, its follow-ups and Phase C item 2 (section 9) implemented in the trainer (Oct 18: Phase B items 1 and 3 done, the exporter writes `shapes.jsonl` and the trainer draws adjacent-section seeds from it; Phases C–F open: model-made correction masks, server and Viking, section-to-section propagation, cell typing) · Revised Oct 14 for the project goals: cells and connectivity, not perfect membranes · Extends *Overall training plan* (Goals, Parts 3–5) and *claude/Training plan assessment* (1.1, 1.4, 1.5), both in the Sam2 claude.ai project. This plan is self-contained for implementation.

## What changed in this revision (Oct 14)

The goal is to identify cells and their connectivity, with at least 97.5% of each cell's interior captured and no merges (*Overall training plan*, Goals). The annotations themselves average about ±9 px (20 nm) from the membrane, with excursions of 30–50 px. The plan is rescaled to match:
- **Edge band:** τ goes from 2.5 px to 8 px, capped for thin processes (section 1).
- **Perturbations:** near-correct masks now wobble 5–12 px on average with excursions up to 30 px; masks to correct reach 50 px (section 3).
- **Loss:** a leak penalty on neighboring cells is added (section 4).
- **Metrics:** core recall, leak, merge rate, spill and fragmentation replace tolerant IoU as the headline numbers, and checkpoint selection follows them (section 6).
- **Reference set:** Approved masks are reported separately (section 6).
- **New phases:** E (section-to-section propagation) and F (cell typing from texture and small-molecule overlays) (section 9).
- **Shelved:** excursion truncation and SEAL/STEAL-style label realignment (section 10).
- **Phase A follow-ups** from the code review are listed in section 9.

**Later on Oct 14 (negative context and overlaps):**
- **Overlaps:** a pixel covered by both a positive and a negative region is 0 (unknown), whatever the confidences; positives take the larger value and negatives the strongest, never a sum. Encoding `edge-confidence-v5` (section 2). Done, `816374e`.
- **Overlapping circle negatives** are dropped (`negative.drop_overlapping_circles: true`), measured at 2.4% of samples (section 3.6). Done, `cdd4b92`.
- **Negative-context dropout** and a `circle_mask_noneg` validation family, so the model does not depend on neighbors always being sent (sections 3.6 and 6). Done.
- **Overlap scoring:** pixels inside both the target and a neighbor's trace count as target, in the loss and in validation (sections 4 and 6). This is the current behaviour, now stated.
- **Circle radii:** the exporter writes true radii; the trainer applies `prompt.circle_diameter_scale` to inscribed (seed) circles (sections 3 and 8).

**Circle annotation trust (encoding `edge-confidence-v6`):**
- Circle *annotations* have unreliable radii (about 2× in the radius-bug cases), so they are only negatives, and their trust is a fraction of the **true** radius, `prompt.circle_annotation_trust` (`full_frac` 0.5, `zero_frac` 0.95), not the polygon edge curve (sections 2, 3.6 and 4). `circle_diameter_scale` now applies to inscribed (seed) circles only. Seed circles, polygon negatives and the containment loss are unchanged. The exporter never scales.

## Summary

This plan changes how the trainer uses masks, both as prompts and as targets.

1. **Edge confidence.** Human-traced masks are trusted in their interior and only loosely at their edge. The same curve, c(d) = 1 − exp(−d/τ), is used in two places:
   - mask prompts are fed as c(d)-scaled logits;
   - the loss is down-weighted by w(d) near the traced edge, with hard 0/1 labels, so the image decides where the membrane is.
2. **Mask refinement replaces the full-mask prompt.** The full-mask prompt teaches copying because the prompt is the answer. In its place come four families, all trained against the true polygon:
   - seeds from the adjacent section, scaled to fit inside the cell (the Viking workflow);
   - shrunk or cut seeds;
   - masks to correct (too large, scaled, warped, or merged with a neighbor);
   - near-correct masks, as good as a typical human trace, that the model must not make worse.
3. **Several prompts per encoded image.** Each sample is encoded once and decoded with four prompt variants (three until Oct 18, 2026). The image encoder is most of the cost, so prompt variety multiplies for little extra compute.
4. **Goal-driven validation.** Core recall (target ≥ 97.5%), leak and merges into neighbors, spill, and fragmentation, reported per prompt family and for the Approved masks.

The negative-only prompt is removed (assessment 1.5); this plan assumes it is gone.

## 1. Edge confidence curve

**Definition.**
- d is the distance in full-resolution pixels from the polygon edge, from a Euclidean distance transform.
- The confidence is c(d) = 1 − exp(−d / τ_eff). It is 0 at the edge, about 0.63 at τ, 0.95 at 3τ, and effectively 1 beyond.
- **Default τ = 8 px.** Traces average about 9 px from the membrane, where c ≈ 0.68; confidence reaches 0.95 about 24 px (52 nm) inside the line.
- **Thin-process cap:** τ_eff = min(τ, 0.25 × D), where D is the mask's maximum depth (the largest distance-transform value). A process 20 px wide has D = 10 and gets τ_eff = 2.5, so its whole profile is not treated as edge. Each mask gets its own cap: the target, each seed and each neighbor.

**What this covers and what it does not.**
- The band absorbs the routine ±9 px wobble.
- **Excursions of 30–50 px are not covered.** No symmetric band that wide is possible without erasing small processes. They stay as sparse label noise; the loss on the rest of the mask outweighs them, and the core-recall metric ignores the outer 10 px.
- **Systematic bias is not covered either.** An annotator who always traces on one side of the membrane teaches the model that side. This matters little for the goal, because the error is a few pixels at the boundary.

## 2. Mask prompts with edge confidence

**Encoding.** The signed dense prompt is continuous:

| Region | Logit fed to `mask_input` |
| --- | --- |
| Target seed (trusted, inside the cell) | +M_seed × c(d), with M_seed = 10 |
| Mask to correct, or near-correct mask (may be wrong) | +M_corr × c(d), with M_corr = 3 |
| Other annotations (negative context) | −10 × c(d_neighbor), using each neighbor's own edge distance and cap |
| Circle annotations (negative context, radius unreliable) | −10 × w(ρ), ρ the distance from the circle's center over its **true** radius; w = 1 for ρ ≤ `full_frac`, 0 for ρ ≥ `zero_frac`, ½(1 + cos(π(ρ − full_frac)/(zero_frac − full_frac))) between |
| Everywhere else | 0 (unknown) |
| Overlap of two positive layers (seed and mask to correct) | the larger value, not the sum |
| Overlap of several negatives | the strongest one (−10 × c at best), never a stack |
| Overlap of + and − | 0 (unknown), whatever the confidences: decided by the masks, not the values |

- **Overlaps are decided by masks.** A pixel covered by a positive region and a negative region is a contradiction, so it is unknown (0) however deep either is; summing would let a deep seed outvote a shallow neighbor, or the reverse, in a way the annotations do not support. The encoder takes the union of the positive masks and the union of the negative masks, marks their intersection unknown, and elsewhere uses the larger positive value or the strongest negative (no sum, no clip). A negative region's mask travels with its canvas, because c(d) is 0 on its outermost pixels. The encoding version string is `edge-confidence-v6` (v5 plus the circle-annotation trust below; the overlap rule is unchanged). Any number of overlapping regions follows the same rule: three neighbors on one pixel read −10, not −30, and a seed under any of them reads 0. This applies to masks to correct too, so where a grown or merged mask to correct covers a neighbor, the prompt is unknown and the image and the leak penalty decide.
- **Circle annotations (v6).** An annotated circle is a negative whose radius cannot be trusted, so it is encoded by ρ rather than by distance to a raster edge: full strength in the inner `full_frac` of the true radius, a half-cosine fall to 0 at `zero_frac`, nothing beyond. The region it *claims* (the mask used by the overlap rule and `drop_overlapping_circles`) is ρ ≤ `zero_frac`, which at the defaults is exactly the 0.95-scaled disc v5 rasterized. It is computed analytically inside the circle's bounding box (no rasterize, no distance transform; about 5× faster, `scripts/benchmark_circle_negatives.py`), and its trust map goes through the same flips, rotations and elastic warps as the image. Validated at config load: `0 ≤ full_frac < zero_frac`. A polygon (and an inscribed circle used as a negative) keeps the c(d) curve above.
- **Where the negatives come from** is section 3.6.
- **Two strengths.** M_corr is lower so the mask network can learn that a mask to correct is a suggestion, while a seed is trusted. SAM 2 itself was trained to correct its own soft prediction logits.
- **Downsampling.** Downsample to the 256×256 `mask_input` by area averaging, not nearest-neighbor.
- **Neighbor-merge samples (section 3.3).** Leave out the merged neighbor's negative context; otherwise the error is labelled for the model.
- **Containment loss.** It applies only where the prompt is a seed, weighted by c(d), so it is effectively the trusted core. It is never applied to masks to correct or near-correct masks.
- **Production must use the identical function.** The server must build prompts with the same c(d), τ, cap, M_seed and M_corr. The formula and constants live in `annotation_crops/prompt_encoding.py`, to be mirrored in the server.

## 3. Prompt families

The full-mask prompt is replaced. Starting mix, configurable:

| Family | Share | Input | Encoding | Containment |
| --- | --- | --- | --- | --- |
| Points only | 10% | 1–9 positive points | none (plus negative context) | no |
| Box only | 5% | jittered box | none (plus negative context) | no |
| Box + points | 5% | box + 1–3 points | none (plus negative context) | no |
| Circle seed | 10% | the tile's circle or the inscribed circle × `circle_diameter_scale` (0.95) + center point | seed | core only |
| Circle annotator | 20% | annotator-like circle, off-centre and smaller (`prompt.circle_seed_jitter`) + center point | seed | core only, inside the target |
| Circle pull-through | 5% | circle across the cell edge, ≥ 60% inside, ≥ 5% outside, not clipped (`prompt.circle_pull_through`) + center point | seed | core only, inside the target |
| Adjacent-section seed | 15% | linked polygon from z ± 1–2, scaled to fit inside | seed | core only |
| Shrunk or cut seed | 15% | subset of the true mask | seed | core only |
| Mask to correct | 10% | grown, eroded, scaled, warped or merged mask | correction | no |
| Near-correct mask | 5% | the true mask with human-scale wobble | correction | no |

- Every family is trained against the true polygon of the target section. Background points and negative context from other annotations ride along as now.
- **Effective mix.** Every variant, the first included, is drawn from the mix without replacement (section 5), so a family appears at most once per sample and its realised share differs from the table's: rare families gain. The trainer logs the per-family counts every epoch (of the family built; `train/prompt_fallback/*` counts circles and adjacent seeds that fell back to the plain circle seed). At the defaults, without adjacent shapes, circle_seed gets 14.5% of the variants, circle_annotator 20.4% and circle_pull_through 8.6% (README, "Prompts in training").
- **Correction clicks.** Where a family adds a foreground click, the click is placed from the *input* mask, not from the ground truth; otherwise the click leaks where the true cell is.
- **Circle scale.** The exporter's circle file holds true radii (`diameter_scale` 1.0, nornir `3b6fe2f`). The trainer applies `prompt.circle_diameter_scale` (0.95) to circle seeds (inscribed circles) only; the exporter never scales. Annotated circles load at their true radius and are trusted by `prompt.circle_annotation_trust` (section 2).

### 3.1 Adjacent-section seed (the Viking workflow)

Annotators bring the polygon from the adjacent section and scale it with the mouse wheel until it fits inside the cell. To train on that:

1. Pick a location of the same structure in section z ± 1 (or ± 2) that has a polygon.
2. Take its shape relative to its own centroid, and place it at the target's inscribed-circle center.
3. Find by bisection the largest scale s* at which at least 98% of the seed lies inside the **target polygon grown by 10 px**. The annotator fits the seed to the visible membrane, which can sit about 9 px outside the trace, and the 2% allows for excursions.
4. Apply s = s* × U(0.7, 1.0), since people will not fit it exactly, plus a small random offset that keeps it inside.
5. Encode it as a seed with the edge curve.

This is a better seed than the inscribed circle for long, thin or curved profiles, where a circle covers only a small part of the cell.

**Data needed.** The exporter must provide, per location:
- `structure_id`;
- the linked location ids in adjacent sections (Viking `LocationLinks`);
- each location's polygon rings relative to its centroid, in D1 pixels.

Put the shapes in a per-volume `shapes.jsonl`, so the trainer does not reassemble shapes from tiled mask PNGs. `structure_id` is also needed for propagation (Phase E) and cell typing (Phase F).

**Status (Oct 18, 2026).** Implemented. The exporter writes `{volume}/shapes.jsonl` (one line per location: `structure_id`, `linked` `[location_id, z]` pairs within two sections, `centroid` and `area` in D1 volume pixels, polygon rings relative to the centroid). `build_tile_context` loads it per volume (`data/shapes.py`: only targets with a linked neighbour and those neighbours' outlines, packed in numpy arrays; a missing file prints a warning and the family is not drawn for that volume). `TileContext.adjacent_shape` rasterizes ONE linked outline from a section 1–2 away into the target tile's pixels (`X / d - x0`, `Y / d - y0` from the image key `_D{d}_X{x0}-{x1}_Y{y0}-{y1}`), picked uniformly by the generator `[seed, epoch, index, 0, 4]` when there are several; the family is available for a sample exactly when that raster exists. The exporter's coordinates turned out to be each section's own mosaic pixels, not registered volume pixels (a linked neighbour's centroid lies a median 23,000 px from its target's in RC2, 9,000 in RPC1; under 1% of links are within 256 px), so an outline at its raw position almost never reaches the target's tile.

**Projected placement (Oct 18, 2026; the default, `prompt.adjacent_poly.placement: projected`).** The outline is carried where Viking puts a carried-forward polygon, and is used there instead of being scaled to fit (steps 2-4 above apply only to the fallback):

1. neighbour mosaic -> volume through the neighbour section's stos in the volume's slice-to-volume group (the group the correction service publishes for: RC2, RPC1, RPC2 `SliceToVolume1`), as `Viking.VolumeModel.VolumeTransformProvider` and the correction builder (`BuildCells`) do;
2. plus the published residual correction, `v + f_neighbour(v) - f_target(v)` (nm, divided by the volume's nm per pixel, sampled as `SectionVectorField.Sample`), the carry used by `JointRegistrationTests` in Viking;
3. volume -> target mosaic through the inverse of the target section's stos;
4. eroded by U(`prompt.adjacent_poly.erode_px`) = U(0, 10) px (capped at 0.6 of its depth);
5. kept only when at least `prompt.adjacent_poly.overlap_min` = 60% of its area in the tile lies on the target mask, the pull-through circle's rule; otherwise that variant is drawn again from the other families (generator `[seed, epoch, index, k, 5]`; the adjacent share moves to the circle peers as when no outline exists) and counted in `train/prompt_fallback/adjacent_poly_seed`. `train/adjacent_poly/pass_rate` and `train/adjacent_poly/share` show the rule and the realised share.

`data/projection.py` reads the VikingXML and grid stos files under `data.registration_root` (`<root>/<VOL>/`; the trainer container mounts the volume share at `/storage4`) and the residual fields under `data.corrections_root` (`<root>/<VOL>/<stos group>/manifest.json` and `<z>.vfield.npz`; the correction service's folder, which the trainer compose file bind-mounts read-only at `/corrections` from `/srv/viking/section-correction/corrections` in WSL; `SAM2_CORRECTIONS_HOST` overrides the host path. A container created before that mount existed has no `/corrections` and falls back to `centroid` with a warning). The manifest names the stos group and the VikingXML (`volume_url`). Every neighbour outline is projected once, in the main process (RC2: 6,953 of 6,973 links in about 6 s; RPC1, RPC2 in 1-2 s) and packed in a few numpy arrays (6 MB for RC2), so DataLoader workers never touch a transform; the stos files are parsed once per process. `training_data.json` records, per volume, the placement, the VikingXML and stos group, and the correction set's directory, manifest provenance (`built_utc`, `annotation_watermark`, method, joint fit) and manifest SHA-256 (`adjacent_projection`). A null root, a missing file, or a chained stos group (RC1 `grid16`: each section registered to the one before) keeps that volume on the centroid placement with one warning.

How the convention was chosen (set v3, 800 random linked pairs per volume, share of the projected neighbour outline's area in the target's tile that lies on the target's mask; "pass" is the 60% rule; no erosion unless noted):

| Placement | RC2 median / pass | RPC1 median / pass | RPC2 median / pass |
|-----------|-------------------|--------------------|--------------------|
| centroid on the target's centroid (before) | 0.79 / 77.0% | 0.48 / 37.2% | 0.42 / 33.1% |
| stos only | **0.85 / 80.0%** | 0.81 / 75.6% | 0.88 / 86.6% |
| stos + residual `+n` / `-n` | 0.57 / 47.9%, 0.53 / 44.0% | 0.81 / 77.0%, 0.79 / 73.0% | 0.88 / 87.0%, 0.87 / 86.5% |
| stos + residual `+t` / `-t` | 0.52 / 43.8%, 0.56 / 46.9% | 0.79 / 72.0%, 0.83 / 77.2% | 0.87 / 86.2%, 0.88 / 87.4% |
| stos + residual `+n-t` (used) | 0.55 / 45.6% | **0.83 / 78.0%** | **0.89 / 87.2%** |
| stos + residual `-n+t` | 0.49 / 39.8% | 0.76 / 69.6% | 0.87 / 86.1% |
| stos only, eroded 10 px | 0.89 / 81.4% | 0.86 / 78.4% | 0.92 / 88.2% |
| `+n-t`, eroded 10 px | 0.55 / 47.0% | 0.89 / 80.1% | 0.93 / 88.0% |

The target's own outline scores 0.996 by the same measure on every volume, so the frames are right. The stos alone already brings a neighbour's centroid within a median 49-58 px of its target's. `+n-t` (forward through the neighbour's residual, back through the target's) is the best convention on RPC1 and ties on RPC2, and it is the carry Viking's joint-registration tests use. On RC2 every residual convention is far worse than the stos alone: the published RC2 field (built 2026-10-18 02:12) moves points a median 178 px while the links' own carry error is 51 px, and it is uncorrelated with that error (r about 0.05). So `data.adjacent_residual: auto` (the default) applies the residual to a volume only when it lowers the median centroid carry error of the volume's own links at load time: on for RPC1 (57 -> 55 px) and RPC2 (52 -> 51 px), off for RC2 (52 -> 197 px). Why the RC2 field disagrees with the stos is an open question for the correction service.

Through the production dataset path (`split=train`, augmentation on, the default config with these roots; 3,000 training samples that have a linked outline, four variants each): 1,704 variants drew `adjacent_poly_seed` with a projected outline, 90.5% passed the rule (RC2 89.8%, RPC1 89.8%, RPC2 94.5%; median share on the cell 0.93), and `adjacent_poly_seed` was built for 12.8% of those samples' variants. Over 3,000 uniformly drawn training samples it is 0.7% of the variants, because only 14,748 of 279,281 usable samples (5%; RC2 3,779 locations, RPC1 619, RPC2 581, RC1 3) have a linked outline.

### 3.2 Shrunk or cut seed (a subset of the truth)

- **Erosion:** shrink the true mask by U(10, 40) px. The minimum of 10 px keeps the seed inside the real cell where the trace runs outside the membrane.
- **Straight cut:** a random line through the eroded mask keeps 20–80% of its area. This mimics a mask that stopped at a tile edge.
- **Shape-following cut:** keep the pixels within a random distance from an interior anchor point, measured along the inside of the cell. This mimics a mask that stopped partway along a process or at a branch.
- Keep only the connected piece containing the anchor.

### 3.3 Mask to correct (not a subset)

- **Grow:** dilate by U(5, 50) px. This models tracing outside the membrane, up to the largest excursions seen.
- **Wheel-style scale:** U(0.8, 1.25) about the centroid.
- **Outline wobble:** an elastic warp with mean displacement U(10, 20) px and local excursions up to 50 px, giving over- and under-shoot in different places.
- **CascadePSP-style:** simplify the outline, then randomly dilate or erode by up to 30 px.
- **Neighbor merge:** union with part of an annotated neighbor's mask, teaching the model to remove a merge. The neighbor is chosen among those within 20 px of the target (any neighbor when none is that close). Background points are sampled *after* the merge and never inside the merged region, and the positive click is taken from the mask minus the merged neighbor (the deepest point of the union is often inside the neighbor).

### 3.4 Near-correct mask ("do no harm")

The input looks like a typical good human trace, encoded as a correction mask:
- **Wobble:** a smooth warp with **mean** displacement U(5, 12) px along the outline (the amplitude parameter is the mean, not the peak).
- **Excursions:** 1–3 local bulges or dents of U(15, 30) px, each spanning 30–150 px of outline.

It teaches the model to leave a good mask at least as good. Without it, a refinement model learns that every input mask needs large changes. The model may improve the input; it must not make it worse (section 6).

### 3.5 Exclusions

- Tiles where the target covers at least 99% of the tile (assessment 1.9) get only point or circle prompts. They have no edge to refine.
- If a derived seed or mask is empty after augmentation, the sample falls back to a circle seed.

### 3.6 Negative context

**Sources.** Every other annotation in the tile is negative context, as the Viking client does in production (it always sends neighboring annotations as background):
- neighbor polygon masks;
- inscribed circles of polygon cells that appear in the tile without their own mask there (about 90% of circle entries);
- annotated circles, Viking TypeCode 1 (`source: "annotated"`, about 10%). These are used only as negatives, background points, leak cores and validation neighbor cores, never as targets. Their radii are unreliable, so each is loaded at its true radius and trusted by fractions of it (`prompt.circle_annotation_trust`, section 2): its *core* is ρ ≤ `full_frac` (0.5) instead of 10 px inside an edge, and its background points are sampled from that core.

All of them are cells or cell subtypes: the exporter's default filter is `(TypeCode eq 4 or TypeCode eq 6) and (Parent/TypeID eq 1 or Parent/Type/ParentID eq 1)`, and the server already sends only those as negatives. Nothing to change there.

**Overlapping circles.** `negative.drop_overlapping_circles: true` removes a circle negative that overlaps the target. Measured on 2,000 samples: 82% have circle negatives, and 2.4% of samples (43 of 2,396 circles) have one overlapping the target, above the 1% threshold. The v5 overlap rule (section 2) would also make those pixels unknown; the drop removes the conflict at its source. The overlap test uses the claimed region (ρ ≤ `zero_frac`), identical to the old 0.95-scaled raster at the defaults.

**Dropout (done).** Negative context is present in 72–82% of samples, so the model may learn to rely on it. Two dropouts, drawn from each variant's own generator (`rng([seed, epoch, index, k])`):
- `negative.context_dropout` (0.2): the variant's prompt carries no negative regions and no background points;
- `negative.region_dropout` (0.2): after that draw, each remaining negative region is dropped independently, with its background points.

**Every training variant** can drop (Oct 18, 2026: no variant is a forced production prompt any more; before, variant 0 was exempt). About 20% of prompts lose all negative context.

What dropout does not change:
- the leak weight and neighbor cores in the loss (section 4), which use every neighbor, since the ground truth has not changed;
- the containment weight, still zero under every real neighbor;
- merged neighbors, still left out of the negatives (section 3.3); a mask to correct can still merge a neighbor that was dropped from its context;
- validation, which never drops anything (the `circle_mask_noneg` family in section 6 removes negatives on purpose).

The encoding rule does not change, so there is no version bump. A rate of 0 draws nothing from the generator, so a run without dropout reproduces its earlier prompts.

## 4. Edge-tolerant loss with a leak penalty

**Labels stay hard (0/1). The loss is weighted per pixel:**

w(d) = max(w_min, 1 − exp(−d / τ_eff)) × (λ_leak where the pixel is in a neighbor's core, else 1)

- d is the distance to the ground-truth polygon edge, on both sides; τ_eff is the target's capped τ (section 1).
- w_min = 0.05 keeps a little gradient at the edge.
- **Overlapping traces:** a pixel inside both the target's trace and a neighbor's trace is labelled target (1). The prompt calls it unknown (section 2), but the label is the target's polygon, so the model is not taught to cut the target where traces overlap.
- **Leak penalty:** pixels inside another annotation's core (at least 10 px inside that neighbor, and outside the target) get λ_leak = 3. For a polygon neighbor the core is 10 px inside its edge (`loss.leak_core_px`); for an annotated circle it is ρ ≤ `full_frac` of its true radius. A merge into a neighbor is the error that breaks connectivity, so it costs more than an equal area of edge error. It also applies to neighbor-merge samples, where removing the merged region is the lesson.

**How it applies to each term:**
- **Focal:** per-pixel weight, normalised by the weight sum.
- **Dice:** weighted soft dice, with sums of w·p·g, w·p and w·g.
- **IoU-head target:** IoU of the thresholded prediction over pixels with edge weight ≥ 0.5, which matches tolerant validation.
- **Mask containment:** unchanged apart from the c(d) weighting in section 2.

**Why weight the loss rather than soften the target.**
- Soft targets near the edge teach the model to output that uncertainty. Edges come out blurred and low-confidence, and the 0.5 threshold still lands on the traced line on average, so the model gets no freedom to find the membrane.
- Down-weighting makes disagreement with the trace nearly free near the edge, so image evidence decides where the edge goes. Supervision stays at full strength elsewhere.
- For a two-class mask, boundary label relaxation (Zhu et al., CVPR 2019) reduces to this.

**Cost.** Distance transforms are computed inside each mask's bounding box grown by 3τ (or by the perturbation radius), not over the full 1024² tile. On a full tile this is 3–7× faster. Edge weights can also be computed on the GPU from the target batch.

## 5. Several prompts per encoded image

- **Per sample:** four prompt variants against the same target, with `prompt.variants_per_sample = 4`.
  - No family is forced (Oct 18, 2026; before, variant 0 was always a circle or adjacent seed). Every variant is drawn from the mix without replacement, variant k from `rng([seed, epoch, index, k, 1])`, renormalised over the families still available.
  - A fully masked tile can draw only points, circle_seed and circle_annotator (a pull-through circle has no edge to cross there), so it gets exactly those three. A sample that runs out of families with a positive share gets one points variant and stops.
- **One encoder pass.** `predict_masks_mixed` encodes once, decodes in groups, and reuses the embeddings through a variant→sample index.
- **Loss:** averaged over a sample's variants, so samples with more variants do not weigh more on the encoder. No `.item()` host syncs in the loss path.
- **Seeding:** variant k uses `rng([seed, epoch, index, k])`, so resumed epochs still replay exactly.
- **Cost:** decoder activations grow by about 3×, still small next to the Hiera-L encoder. Check with `--benchmark` before a full run.
- **Run length:** prompts seen per epoch triple. Reconsider `num_epochs`.
- **Precedent:** SAM trains this way, with one image-encoder pass and many prompts and correction rounds through the light decoder.

## 6. Validation

Use a fixed seed and the z-slab validation split.

**Definitions.** b = 10 px is the core band (`val.core_band_px`). The ground truth is the target's trace alone, with no neighbor subtracted: pixels where the target and a neighbor overlap count as target, and neighbor cores exclude the target. A polygon neighbor's core is b px inside its edge; an annotated circle's core is ρ ≤ `full_frac` of its true radius (`prompt.circle_annotation_trust`), because its radius is not reliable enough for a band.
- **Core recall:** the share of ground-truth pixels at least b inside the traced edge that are predicted foreground. Reported pooled over all pixels (the "interior volume" figure), and per sample as the 10th percentile and the share of samples at or above 97.5%.
- **Leak:** the share of predicted pixels that fall in annotated neighbors' cores (at least b inside a neighbor).
- **Merge rate:** the share of samples whose prediction covers at least 5% of any one neighbor's visible core and at least `val.merge_min_px` (500) px of it. The floor keeps a stray blob into a neighbor clipped by the tile edge from counting as a merge, since the merge rate selects the checkpoint.
- **Spill:** predicted area more than b outside the ground truth, as a share of the ground-truth area. It catches leaks into cells nobody annotated, which leak cannot see.
- **Fragmentation:** the share of samples whose prediction has more connected components (of at least 200 px) than the ground truth.
- **VI (split and merge parts):** on tiles where every annotated object is predicted, as in connectomics benchmarks (Meilă 2007; Arganda-Carreras et al. 2015). Optional in Phase A; required in Phase E.
- **Diagnostics only:** tolerant IoU (ignoring pixels with edge weight < 0.5) and boundary F1 at 10 and 20 px.

**Per family:**

| Check | Metrics |
| --- | --- |
| Circle seed (production) | Core recall, leak, merge rate, spill, fragmentation. Also split by `with_neg` and `no_neg` (whether the sample has any negative context): `val_*_neg/<metric>/<family>/<with_neg or no_neg>`, counts in `val_*_neg/n/...`, and the same two figures in the epoch summary line |
| Circle seed without negatives (`circle_mask_noneg`) | The same circle seed with every negative region and background point removed; same target and metrics. In the per-epoch subset, the full pass and `--goal`. Compare with the `with_neg` subset of circle seeds to see how much the model leans on neighbors being sent (on `no_neg` samples the two prompts are identical); the same split is reported for this family. Not used for selection |
| Adjacent-section seed | Same |
| Mask to correct | Same, and the change from the input mask on each |
| Near-correct ("do no harm") | Share of samples where the output is no worse than the input: core recall down by no more than 0.5 points, and spill up by no more than 0.5% of the ground-truth area |
| Centroid point (legacy) | Raw IoU, kept for continuity with earlier runs |
| Approved masks | All of the above, for Approved masks in the validation sections |

**Breakdowns (full passes only).** Core recall (pooled and p10), merge rate and spill are also logged per category (`val_full_cat/<metric>/<family>/<category>`) and per volume (`val_full_vol/...`). Core recall is stratified by the ground truth's maximum depth, under 20 px, 20–60 px and 60 px and over (`val_full_depth/...`, with the per-sample mean next to the pooled figure): thin processes weigh little in the pooled number, and a profile under about 20 px wide has no core at all. Selection does not use them.

**Approved masks as the reference set.**
- The gallery writes `approved.json` (location id and image key entries; an empty image key approves every window of that location) into each volume's crops folder. With `data.approved_path` null the trainer reads `{data_root}/{volume}/approved.json` for every configured volume, in the flat or `AnnotationCrops` layout; it may also be a file, a directory or a list of those. Location ids are only unique within a volume, so matching always includes the volume. The counts per volume are logged at start-up, and a total of 0 warns loudly.
- Approved masks stay on their own side of the z-slab split. Count how many fall in the validation sections before relying on the Approved row; if there are too few, report a confidence interval. Every Approved sample is always in the per-epoch subset.
- Optionally, weight Approved masks 2× in training (`data.approved_weight`).

**Per-epoch cost: a fixed subset, a full pass every third epoch.**
- A full pass (all families, about 60 minutes on the RTX 4500 Ada) is too slow to run each epoch. Every epoch validates a fixed subset instead: `val.epoch_subset` (0.25) of each volume, drawn once from the run seed by a hash of volume, image key and location id, always including the Approved samples, without the legacy centroid family. It takes about 15 minutes.
- The full split, centroid included, runs every `val.full_every_n_epochs` (3) epochs, after the last epoch and before an early stop. TensorBoard tags are `val_subset/...` and `val_full/...`; the old `val/iou` follows the subset.
- Each sample is loaded once and encoded once, and every family decodes from that encoding. The ground-truth terms are computed in the DataLoader workers and the metrics run in a thread pool on the cropped bounding box, so the pass is GPU-bound. Validation runs in fp32, like `infer.py` and the server, so there is no `val.mixed_precision`.
- Corrective-click evaluation (`sam2-em-eval --noc`) samples its clicks from goal errors and reports two stopping rules: IoU at the target (NoC@90) and core recall at the target with no merge.
- `sam2-em-eval --goal` runs one full pass for any model (zero-shot, an earlier checkpoint) and prints every family's goal metrics, for baselines.

**Checkpoint selection.** Five metrics of the primary family (circle_annotator, on the per-epoch subset) decide: core recall, tolerant IoU, spill, merge rate and fragmentation. An epoch replaces the best so far when at least one of them is better by its noise floor and none is worse by its floor. A trade (one better, another worse, both beyond their floors) does not count. The 97.5% core-recall target is reported but is not a gate, because the validation masks are traced annotations, not a checked validation set. (Until 2026-10-21 the target was a gate and, below it, only core recall counted; on v5 that ignored an epoch whose IoU, spill and fragmentation all improved well beyond their noise.)

**Early stopping and the noise floor.** The floors (`training.early_stopping_floors`) are the 95th-percentile error of a 10% subset of validation set v5 (`scripts/valset_size_study.py`): core recall 0.004, IoU 0.006, spill 0.006, merge rate 0.002, fragmentation 0.012. A change inside a floor neither counts nor blocks, so it does not reset the patience (2 epochs). Metrics are rounded to 4 decimals before they are compared, and the log line `selection:` names the metrics that moved beyond their floor.

## 7. Production requirements (server and Viking)

- **Shared encoding:** the server builds `mask_input` with the shared function and constants from section 2. **A v4 model must not be served until it does.** `sam2-em-export-serve` writes the constants (`EDGE_TAU_PX`, `EDGE_TAU_DEPTH_FRAC`, `SEED_LOGIT`, `CORRECTION_LOGIT`, `LOGIT_CLIP`, `LOW_RES`, `circle_diameter_scale`) and a `prompt_encoding` version string (now `edge-confidence-v6`) and `circle_annotation_trust` (`full_frac`, `zero_frac`, `ramp: cosine`) into the exported file next to the weights, so the server can check them, and writes the same dict as `<weights stem>.prompt_encoding.json` beside it. Training writes `prompt_encoding.json` into the run folder at start, so a raw `best_model.pt` has its record too; `sam2-em-eval` and `sam2-em-validate` read it back instead of the defaults, and log any command-line override. The server must implement the v5 overlap rule (section 2), not the old sum and clip, and the v6 circle trust: it reads `full_frac` and `zero_frac` from the metadata and must not assume the defaults. The exporter never scales circle radii; the server applies the recorded values to the true radii it holds.
- **Open: exporting runs trained before v6.** `export_serve` builds the record from the run's `config.yaml` and stamps the current version; it does not read the run's own `prompt_encoding.json`. A run trained at v5 would therefore be exported as `edge-confidence-v6` with the default trust, and a server would build its circle negatives the wrong way. Fix: use `run_dir/prompt_encoding.json` when present, and otherwise refuse, or label the model `edge-confidence-v5`, unless overridden. Until then, export only runs trained at v6.
- **Adjacent-section polygon:** when an annotator brings a polygon from the adjacent section and scales it to fit inside the cell, Viking sends it as a **seed**.
- **New command, "Resegment this polygon":** sends an existing polygon as a **mask to correct** (M_corr), optionally with correction clicks.
- **Clamp:** no hard clamp on masks to correct. If the +6 clamp is adopted at all, it applies only to a seed's trusted core.

## 8. Configuration

| Parameter | Default | Description |
| --- | --- | --- |
| `prompt.edge_tau_px` | 8.0 | τ of the prompt confidence curve |
| `prompt.edge_tau_depth_frac` | 0.25 | Thin-process cap: τ_eff = min(τ, this × the mask's maximum depth) |
| `prompt.seed_logit` | 10.0 | M_seed |
| `prompt.correction_logit` | 3.0 | M_corr |
| `prompt.variants_per_sample` | 4 | Prompt variants decoded per encoded sample, all drawn from the mix without replacement |
| `prompt.prob_circle_seed` | 0.10 | Plain circle (replaces `prob_circle_mask`) |
| `prompt.prob_circle_annotator` | 0.20 | Annotator-like circle (`circle_seed_jitter` shapes it) |
| `prompt.prob_circle_pull_through` | 0.05 | Circle across the cell edge (`circle_pull_through` shapes it) |
| `prompt.circle_seed_jitter.depth_range`, `scale_range`, `fit_grow_px`, `min_radius_px` | [0.3, 1.0], [0.45, 1.0], 10, 6 | Annotator-like disc (docs/circle_annotator_spec.md); a retired `prob` key in an old config is ignored |
| `prompt.circle_pull_through.overlap_min` | 0.60 | Share of the pull-through disc inside the target, at least |
| `prompt.circle_pull_through.outside_min` | 0.05 | Share of it outside the target, at least (`overlap_min + outside_min <= 1`) |
| `prompt.circle_pull_through.radius_range` | [0.4, 0.9] | Radius as a fraction of the target's maximum depth |
| `prompt.circle_pull_through.min_radius_px` | 6 | Floor on the radius |
| `prompt.circle_pull_through.max_tries` | 20 | Draws before falling back to the plain circle |
| `prompt.prob_adjacent_poly_seed` | 0.15 | Drawn from the exporter's `shapes.jsonl` (3.1) on samples with a linked outline 1–2 sections away; elsewhere (and on a fully masked tile) its share moves to the three circle peers in proportion |
| `prompt.prob_subset_seed` | 0.15 | Shrunk or cut seed |
| `prompt.prob_correction` | 0.10 | Mask to correct |
| `prompt.prob_near_correct` | 0.05 | Near-correct mask |
| `prompt.prob_points_only`, `prob_box_only`, `prob_box_points` | 0.10, 0.05, 0.05 | Unchanged |
| `prompt.circle_diameter_scale` | 0.95 | Applied by the trainer to the exporter's true radii of inscribed (seed) circles only |
| `prompt.circle_annotation_trust.full_frac` | 0.5 | Circle annotations: full −`seed_logit` inside this fraction of the true radius; also the circle's core |
| `prompt.circle_annotation_trust.zero_frac` | 0.95 | Circle annotations: the cosine taper reaches 0 here and the circle claims nothing beyond (`0 ≤ full_frac < zero_frac`) |
| `negative.drop_overlapping_circles` | true | Drop a circle negative that overlaps the target (measured 2.4% of samples) |
| `negative.context_dropout` | 0.2 | Chance a training variant carries no negatives or background points |
| `negative.region_dropout` | 0.2 | Per-region drop of the remaining negatives; dropped regions still count for leak and metrics |
| `prompt.adjacent_poly.placement` | projected | `projected` (3.1: through the registration, used where it lands), `centroid` (shape only, fitted inside) or `raw` |
| `prompt.adjacent_poly.overlap_min` | 0.60 | A projected outline counts only when this share of its area in the tile lies on the target; else the variant is drawn again |
| `prompt.adjacent_poly.erode_px` | [0, 10] | Erosion of the projected outline before the overlap test |
| `data.registration_root`, `data.corrections_root` | `/storage4`, `/corrections` | VikingXML + stos per volume, and the published residual fields (3.1); null or a missing folder falls back to `centroid` |
| `data.adjacent_residual` | auto | Apply the residual where it lowers the links' carry error (`on`, `off` force it) |
| `prompt.adjacent_fit_inside` | 0.98 | Centroid fallback only: minimum share of the scaled seed inside the target grown by `adjacent_fit_grow_px` |
| `prompt.adjacent_fit_grow_px` | 10 | Growth of the target for the fit test |
| `prompt.adjacent_scale_range` | [0.7, 1.0] | Factor applied to the fitted scale |
| `prompt.erode_px` | [10, 40] | Shrink range for subset seeds |
| `prompt.dilate_px` | [5, 50] | Grow range for masks to correct |
| `prompt.shrink_px` | [5, 50] | Erode range for masks to correct (the mirror of `dilate_px`; capped at half the mask depth) |
| `prompt.scale_range` | [0.8, 1.25] | Wheel-style scale for masks to correct |
| `prompt.correction_wobble_px` | [10, 20] | Mean displacement of the correction wobble; excursions up to 50 px |
| `prompt.near_correct_wobble_px` | [5, 12] | Mean displacement of the near-correct wobble |
| `prompt.near_correct_excursion_px` | [15, 30] | Amplitude of each of 1–3 near-correct excursions |
| `loss.edge_tau_px` | 8.0 | τ of the loss weight (same cap); 0 disables edge weighting |
| `loss.edge_weight_min` | 0.05 | w_min |
| `loss.leak_weight` | 3.0 | λ_leak on neighbor cores; 1 disables |
| `iterative.batch_fraction`, `rounds_min`, `rounds_max`, `prev_mask_prob` | 0.2, 2, 6, 0.5 | Corrective-click training: the share of batches, rounds per batch, and the share of samples that feed back their previous logits. Clicks target goal errors (a miss in the core, a spill beyond the band) |
| `val.core_band_px` | 10 | Band ignored by core recall, leak and spill, and by corrective-click sampling |
| `val.core_recall_target` | 0.975 | Checkpoint gate |
| `val.merge_min_px` | 500 | A merge covers at least this many px of one neighbor's visible core, as well as 5% of it |
| `val.epoch_subset` | 0.25 | Share of each volume validated every epoch (fixed, includes Approved) |
| `val.full_every_n_epochs` | 3 | Full validation pass interval (and after the last epoch) |
| `val.batch_size`, `val.metric_threads` | 4, 4 | Validation batch and metric threads |
| `training.early_stopping_patience`, `early_stopping_floors` | 2; core recall 0.004, IoU 0.006, spill 0.006, merge rate 0.002, frag 0.012 | Epochs without a better checkpoint before stopping; the noise floor of each selection metric |
| `data.approved_path` | null | null: each volume's `approved.json`. Or a file, a directory or a list. Empty image key = every window of the location |
| `data.approved_weight` | 1.0 | Training weight for Approved masks (2.0 to try) |
| `training.batch_size`, `gradient_accumulation_steps` | 1, 16 | Fits beside the segmentation server, see below |
| `model.activation_checkpointing` | false | Faster than checkpointing at batch 4 (see below) |
| `training.cudnn_benchmark` | false | No speed benefit; one run had a 21 GB workspace spike |

**Throughput defaults.** Since Oct 18 2026 the default is batch 1 × 16 accumulation (the same effective batch and optimizer-step cadence): the GPU is shared with the segmentation server (about 8 GB), and under a 14 GiB cap with 4 prompt variants batch 2 without checkpointing runs out of memory at the first optimizer step, while batch 1 takes 0.352 s/sample at 9.42 GiB allocated / 9.56 GiB reserved (batch 2 with a quarter of the blocks checkpointed is no faster and leaves under 1.5 GB). The figures below were measured with 3 variants and the GPU to itself. An A/B/A/B comparison of 200 training steps each, with the GPU clocks logged: batch 2 × 8 accumulation without activation checkpointing took 0.2765 and 0.2755 s/sample at 14.8 GB peak; batch 4 × 4 with checkpointing took 0.314 and 0.339 s/sample at 7.0 GB. The first is 15.5% faster, so it is the default. Clocks were steady (only the software power cap fired, in both), and no other process used the GPU. Checkpointing off at batch 4 does not fit in 22.5 GB, and a quarter of the blocks thrashed at 20.5 GB. An earlier whole-benchmark run showed 0.321 s/sample for the first configuration; a single v4 epoch will show which holds.

Removed: `prompt.prob_full_mask`, `prompt.prob_negative_only`, `negative.allow_circle_targets`, `negative.circle_target_fraction`.

## 9. Implementation order

**Phase A: trainer only, no new export. Implemented Oct 13.**
1. Shared `prompt_encoding` module: c(d), seed and correction encodings, area downsampling.
2. Edge-weighted focal, dice and IoU-head target; tolerant IoU in validation.
3. Families 3.2–3.4; full-mask and negative-only types removed.
4. Multiple variants per sample in the dataset, collate and `predict_masks_mixed`.
5. Per-family validation.

**Phase A follow-ups (from the Oct 13 review and this revision).**
1. Correction and near-correct clicks come from the input mask, not the ground truth.
2. Background points are sampled after a neighbor merge, never inside the merged region.
3. The adjacent polygon seed is reachable as variant 0 when a sample has a linked shape. (Superseded Oct 18, 2026: no variant is forced; it is drawn from the mix at any position.)
4. Rescale to section 1 and section 3: τ = 8 with the thin-process cap; near-correct wobble with the mean (not peak) as its amplitude, plus excursions; correction ranges to 50 px; erosion from 10 px; the adjacent fit against the grown target.
5. The leak penalty (section 4).
6. The section 6 metrics, the Approved row and the new checkpoint rule.
7. Per-family counts logged each epoch, and the effective mix documented.
8. Bounding-box distance transforms (3–7× faster on full tiles); GPU edge weights if loading is still the bottleneck.
9. No `.item()` syncs in the loss path; stale comments (the "50K" mask CNN size) fixed.
10. Tests: shrunk and cut seeds stay inside the truth; masks to correct differ from it; near-correct masks have mean displacement 5–12 px and peak excursion no more than 30 px; leak weights appear only in neighbor cores.
11. Second review (Oct 14): merge corrections click outside the merged neighbor; Approved discovery per volume with whole-location entries; a merge floor of 500 px (`val.merge_min_px`); corrective clicks aimed at goal errors, and NoC reported by IoU and by core recall with no merge; encoding constants written into the exported model; per-category, per-volume and per-depth goal metrics on full passes; losses in fp32; merges drawn from neighbors within 20 px; `sam2-em-eval --goal` for baselines.
12. Negative context (Oct 14, evening):
    - the v5 overlap rule (section 2). **Done**, `816374e`;
    - `negative.drop_overlapping_circles: true`. **Done**, `cdd4b92`;
    - negative-context and per-region dropout (section 3.6). **Done**;
    - the `circle_mask_noneg` family and the `with_neg`/`no_neg` split of circle-seed results (section 6). **Done**.

**Acceptance for Phase A**, on the z-slab validation split, against the current model:
- pooled core recall on circle seeds does not drop, and merge rate does not rise;
- masks to correct improve on their input in core recall and merge rate on average;
- at least 95% of near-correct samples pass "do no harm".

**Phase B: exporter.**
1. Add `structure_id` and linked location ids to the manifest, and write `shapes.jsonl`. **Done** (nornir `segmentationtraining/shapes.py`; written for every v3 volume).
2. Fix the `circles.py` key regex for negative window origins.
3. Add the adjacent-section seed family (3.1). **Done** Oct 18, 2026: the trainer loads `shapes.jsonl` and draws the family. The neighbour's outline is projected through the volume's registration and residual correction to where Viking carries it, eroded 0-10 px and used only when 60% of it lies on the target (3.1, "Projected placement"); a volume without registration data places it on the target's centroid and fits it inside (shape only).
4. Copy `approved.json` next to the export.

**Phase C: correction training.**
1. Correction masks from the model's own predictions, used for about half of the correction samples (as in μSAM). Open.
2. Correction clicks sampled from the error region (as in SAM, SAM 2 and RITM). **Done** (`iterative.*`): 20% of batches run 2–6 rounds of predict, click, feed back the previous logits, with only the last pass carrying gradients. Clicks target goal errors (a positive click in the part of the core the prediction misses, a negative one beyond the core band), and with no goal error the click is a positive one at the deepest target pixel (never a click on edge-band noise).

**Phase D: production.**
1. The server uses the shared encoding.
2. Viking sends adjacent-section polygons as seeds and adds the "Resegment" command.

**Phase E: section-to-section propagation (near future).**
1. **Chained seeds, no new training.** Segment a cell on section z, map the prediction to z + 1 through the nornir section transforms, erode it by about 15 px to absorb registration jitter, and use it as an adjacent-section seed. Stop when core area changes abruptly or the prediction merges.
2. **SAM 2 video mode.** Treat a registered run of sections as a video, so memory attention carries the cell forward. SAM 2's released training code supports video fine-tuning; train on short clips (about 8 sections) cut from the training sections, with residual jitter as augmentation.
3. **Data needed:** crops in volume coordinates through the nornir transforms, and `structure_id` plus `LocationLinks` as the ground truth for which profiles are the same cell.
4. **Validation:** the z-slabs are contiguous runs of sections, so they serve as test clips. Report sections tracked before the first merge or loss (core recall below 50%), 3D VI split and merge, and agreement with Viking `LocationLinks`.

**Phase F: cell typing from texture and small-molecule signals (later).**
1. **Labels:** the cell class of each `structure_id` from Viking, grouped first into broad classes so rare types are not tiny classes.
2. **Features:** encoder features pooled over the cell's profile on each section, then pooled over sections per cell. Add the RGB small-molecule overlays (GABA, glycine, glutamate and others) as per-channel statistics within the profile, on the sections where overlays exist. Small-molecule signatures are an established way to separate retinal cell classes (Marc et al. 1995).
3. **Export work:** register the overlays to the EM crops and record which channel holds which molecule.
4. **Models:** a linear probe or small MLP first. Compare texture alone, molecules alone and both.
5. **Split by cell, not by section.** A cell spans many sections, so the z-slab split would put the same cell on both sides. Hold out whole cells (grouped by `structure_id`), and test across volumes where possible.
6. **Class imbalance:** class-balanced sampling or loss; report macro-F1, per-class recall and the confusion matrix.

## 10. Shelved

- **Excursion truncation:** detecting where a trace strays far from the membrane and removing those pixels from the loss. Shelved because the goal tolerates boundary noise and the core metrics already ignore the edge. Revisit only if core recall stalls below 97.5% because of edge errors.
- **SEAL/STEAL-style label realignment:** learning to move noisy boundary labels onto image edges during training. Shelved for the same reason.

## 11. Open questions

- **τ.** 8 px follows the reported ±20 nm average. It can be measured: compare traced edges with the nearest membrane (image intensity minimum) along edge normals, separately for Approved and other masks.
- **λ_leak and the 5% merge threshold.** Starting values; tune after the first per-family validation.
- **The mix.** The shares in section 3 are a starting point. Adjust after the first per-family validation.
- **M_corr = 3.** It could be tuned by validating with the input mask at a few strengths.
- **Radius-relative seed trust (proposed, not decided; do not implement).** Not to be confused with the circle-*annotation* trust above, which is implemented for negatives only. Circle radii in production are untrusted (some carry a 2× radius bug). One option: full seed trust within the inner half of a seed's depth, a cosine fall to 0 at its edge, about 10% of training circle seeds scaled 1.2–2×, circle-neighbor cores limited to their inner half, and a `circle_oversize` validation family. It would need its own encoding version bump.

## References

- Kirillov A. et al. Segment Anything. *ICCV* 2023.
- Ravi N. et al. SAM 2: Segment Anything in Images and Videos. *ICLR* 2025.
- Ke L. et al. Segment Anything in High Quality (HQ-SAM). *NeurIPS* 2023. Mask prompts made by adding noise to ground-truth boundaries.
- Cheng H.K. et al. CascadePSP: Toward Class-Agnostic and Very High-Resolution Segmentation via Global and Local Refinement. *CVPR* 2020.
- Sofiiuk K., Petrov I., Konushin A. Reviving Iterative Training with Mask Guidance for Interactive Segmentation. *ICIP* 2022.
- Archit A. et al. Segment Anything for Microscopy. *Nat. Methods* 22, 579–591 (2025).
- Zhu Y. et al. Improving Semantic Segmentation via Video Propagation and Label Relaxation. *CVPR* 2019.
- Arganda-Carreras I. et al. Crowdsourcing the creation of image segmentation algorithms for connectomics. *Front. Neuroanat.* 9:142 (2015).
- Meilă M. Comparing clusterings: an information based distance. *J. Multivariate Anal.* 98, 873–895 (2007).
- Marc R.E., Murry R.F., Basinger S.F. Pattern recognition of amino acid signatures in retinal neurons. *J. Neurosci.* 15, 5106–5129 (1995).
- Schubert P.J. et al. Learning cellular morphology with neural networks. *Nat. Commun.* 10, 2736 (2019).
- Dorkenwald S. et al. Multi-layered maps of neuropil with segmentation-guided contrastive learning (SegCLR). *Nat. Methods* 20, 2011–2020 (2023).
