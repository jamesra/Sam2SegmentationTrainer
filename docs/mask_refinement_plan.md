# Mask Refinement, Edge Tolerance and Multi-Prompt Training Plan

Oct 13, 2026 · Phase A, its follow-ups and Phase C item 2 (section 9) implemented in the trainer (Phases B–F open: exporter `shapes.jsonl` and adjacent-section seeds, model-made correction masks, server and Viking, section-to-section propagation, cell typing) · Revised Oct 14 for the project goals: cells and connectivity, not perfect membranes · Extends *Overall training plan* (Goals, Parts 3–5) and *claude/Training plan assessment* (1.1, 1.4, 1.5), both in the Sam2 claude.ai project. This plan is self-contained for implementation.

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
3. **Several prompts per encoded image.** Each sample is encoded once and decoded with three prompt variants. The image encoder is most of the cost, so prompt variety triples for little extra compute.
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
| Everywhere else | 0 (unknown) |
| Overlap of + and − | sum, clipped to ±10 |

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
| Circle seed | 40% | inscribed circle × 0.95 + center point | seed | core only |
| Adjacent-section seed | 15% | linked polygon from z ± 1–2, scaled to fit inside | seed | core only |
| Shrunk or cut seed | 10% | subset of the true mask | seed | core only |
| Mask to correct | 10% | grown, scaled, warped or merged mask | correction | no |
| Near-correct mask | 5% | the true mask with human-scale wobble | correction | no |

- Every family is trained against the true polygon of the target section. Background points and negative context from other annotations ride along as now.
- **Effective mix.** Variant 0 of each sample is always the production prompt (section 5), and the shares above steer variants 1 onwards. The share each family actually receives is therefore not the table's. The trainer logs the per-family counts every epoch, and the config comment states the effective mix.
- **Correction clicks.** Where a family adds a foreground click, the click is placed from the *input* mask, not from the ground truth; otherwise the click leaks where the true cell is.

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

**Faithful variant (later).** Place the linked polygon through the section-to-section volume transforms instead of re-centering it. That also reproduces real placement error, but needs nornir's transforms in the trainer path. Phase E needs them anyway.

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

## 4. Edge-tolerant loss with a leak penalty

**Labels stay hard (0/1). The loss is weighted per pixel:**

w(d) = max(w_min, 1 − exp(−d / τ_eff)) × (λ_leak where the pixel is in a neighbor's core, else 1)

- d is the distance to the ground-truth polygon edge, on both sides; τ_eff is the target's capped τ (section 1).
- w_min = 0.05 keeps a little gradient at the edge.
- **Leak penalty:** pixels inside another annotation's core (at least 10 px inside that neighbor, and outside the target) get λ_leak = 3. A merge into a neighbor is the error that breaks connectivity, so it costs more than an equal area of edge error. It also applies to neighbor-merge samples, where removing the merged region is the lesson.

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

- **Per sample:** three prompt variants against the same target, with `prompt.variants_per_sample = 3`.
  - Variant 0 is always the production prompt: a circle seed, or an adjacent-section seed when one is available.
  - Variants 1 and 2 are drawn from the mix without replacement.
- **One encoder pass.** `predict_masks_mixed` encodes once, decodes in groups, and reuses the embeddings through a variant→sample index.
- **Loss:** averaged over a sample's variants, so samples with more variants do not weigh more on the encoder. No `.item()` host syncs in the loss path.
- **Seeding:** variant k uses `rng([seed, epoch, index, k])`, so resumed epochs still replay exactly.
- **Cost:** decoder activations grow by about 3×, still small next to the Hiera-L encoder. Check with `--benchmark` before a full run.
- **Run length:** prompts seen per epoch triple. Reconsider `num_epochs`.
- **Precedent:** SAM trains this way, with one image-encoder pass and many prompts and correction rounds through the light decoder.

## 6. Validation

Use a fixed seed and the z-slab validation split.

**Definitions.** b = 10 px is the core band (`val.core_band_px`).
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
| Circle seed (production) | Core recall, leak, merge rate, spill, fragmentation |
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

**Checkpoint selection.** Among checkpoints whose pooled core recall on circle seeds (on the subset) is at least 97.5%, keep the one with the lowest merge rate, with spill breaking ties. If none reaches 97.5%, keep the highest core recall. This replaces IoU-based selection (assessment 1.1).

**Early stopping and the noise floor.** A new best must beat the best so far by `training.early_stopping_min_delta` (0.002) in merge rate or spill, or 0.0005 in core recall, so noise on a 25% subset does not reset the patience (2 epochs); crossing the core-recall gate always counts. Metrics are rounded to 4 decimals before they are compared.

## 7. Production requirements (server and Viking)

- **Shared encoding:** the server builds `mask_input` with the shared function and constants from section 2. **A v4 model must not be served until it does.** `sam2-em-export-serve` writes the constants (`EDGE_TAU_PX`, `EDGE_TAU_DEPTH_FRAC`, `SEED_LOGIT`, `CORRECTION_LOGIT`, `LOGIT_CLIP`, `LOW_RES`, `circle_diameter_scale`) and a `prompt_encoding` version string into the exported file next to the weights, so the server can check them.
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
| `prompt.variants_per_sample` | 3 | Prompt variants decoded per encoded sample |
| `prompt.prob_circle_seed` | 0.40 | Replaces `prob_circle_mask` |
| `prompt.prob_adjacent_seed` | 0.15 | Needs `shapes.jsonl` (3.1); its share moves to circle seed until then |
| `prompt.prob_subset_seed` | 0.10 | Shrunk or cut seed |
| `prompt.prob_correction` | 0.10 | Mask to correct |
| `prompt.prob_near_correct` | 0.05 | Near-correct mask |
| `prompt.prob_points_only`, `prob_box_only`, `prob_box_points` | 0.10, 0.05, 0.05 | Unchanged |
| `prompt.adjacent_fit_inside` | 0.98 | Minimum share of the scaled seed inside the target grown by `adjacent_fit_grow_px` |
| `prompt.adjacent_fit_grow_px` | 10 | Growth of the target for the fit test |
| `prompt.adjacent_scale_range` | [0.7, 1.0] | Factor applied to the fitted scale |
| `prompt.erode_px` | [10, 40] | Shrink range for subset seeds |
| `prompt.dilate_px` | [5, 50] | Grow range for masks to correct |
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
| `training.early_stopping_patience`, `early_stopping_min_delta` | 2, 0.002 | Epochs without a better checkpoint before stopping; the noise floor of "better" (core recall uses 0.0005) |
| `data.approved_path` | null | null: each volume's `approved.json`. Or a file, a directory or a list. Empty image key = every window of the location |
| `data.approved_weight` | 1.0 | Training weight for Approved masks (2.0 to try) |
| `training.batch_size`, `gradient_accumulation_steps` | 2, 8 | Throughput default, see below |
| `model.activation_checkpointing` | false | Faster than checkpointing at batch 4 (see below) |
| `training.cudnn_benchmark` | false | No speed benefit; one run had a 21 GB workspace spike |

**Throughput defaults.** An A/B/A/B comparison of 200 training steps each, with the GPU clocks logged: batch 2 × 8 accumulation without activation checkpointing took 0.2765 and 0.2755 s/sample at 14.8 GB peak; batch 4 × 4 with checkpointing took 0.314 and 0.339 s/sample at 7.0 GB. The first is 15.5% faster, so it is the default. Clocks were steady (only the software power cap fired, in both), and no other process used the GPU. Checkpointing off at batch 4 does not fit in 22.5 GB, and a quarter of the blocks thrashed at 20.5 GB. An earlier whole-benchmark run showed 0.321 s/sample for the first configuration; a single v4 epoch will show which holds.

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
3. The adjacent seed is reachable as variant 0 when a sample has a linked shape.
4. Rescale to section 1 and section 3: τ = 8 with the thin-process cap; near-correct wobble with the mean (not peak) as its amplitude, plus excursions; correction ranges to 50 px; erosion from 10 px; the adjacent fit against the grown target.
5. The leak penalty (section 4).
6. The section 6 metrics, the Approved row and the new checkpoint rule.
7. Per-family counts logged each epoch, and the effective mix documented.
8. Bounding-box distance transforms (3–7× faster on full tiles); GPU edge weights if loading is still the bottleneck.
9. No `.item()` syncs in the loss path; stale comments (the "50K" mask CNN size) fixed.
10. Tests: shrunk and cut seeds stay inside the truth; masks to correct differ from it; near-correct masks have mean displacement 5–12 px and peak excursion no more than 30 px; leak weights appear only in neighbor cores.
11. Second review (Oct 14): merge corrections click outside the merged neighbor; Approved discovery per volume with whole-location entries; a merge floor of 500 px (`val.merge_min_px`); corrective clicks aimed at goal errors, and NoC reported by IoU and by core recall with no merge; encoding constants written into the exported model; per-category, per-volume and per-depth goal metrics on full passes; losses in fp32; merges drawn from neighbors within 20 px; `sam2-em-eval --goal` for baselines.

**Acceptance for Phase A**, on the z-slab validation split, against the current model:
- pooled core recall on circle seeds does not drop, and merge rate does not rise;
- masks to correct improve on their input in core recall and merge rate on average;
- at least 95% of near-correct samples pass "do no harm".

**Phase B: exporter.**
1. Add `structure_id` and linked location ids to the manifest, and write `shapes.jsonl`.
2. Fix the `circles.py` key regex for negative window origins.
3. Add the adjacent-section seed family (3.1).
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
