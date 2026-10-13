# Mask Refinement, Edge Tolerance and Multi-Prompt Training Plan

Oct 13, 2026 · Proposed, not yet implemented · Extends *Overall training plan* (Parts 3–5) and *claude/Training plan assessment* (1.1, 1.4, 1.5), both in the Sam2 claude.ai project. This plan is self-contained for implementation.

## Summary

This plan changes how the trainer uses masks, both as prompts and as targets.

1. **Edge confidence.** Human-traced masks are trusted in their interior and only loosely at their edge, where hand tremor moves the line by a few pixels. The same fast-rising curve, c(d) = 1 − exp(−d/τ), is used in two places:
   - mask prompts are fed as c(d)-scaled logits;
   - the loss is down-weighted by w(d) near the traced edge, with hard 0/1 labels, so the image decides where the membrane is.
2. **Mask refinement replaces the full-mask prompt.** The full-mask prompt teaches copying because the prompt is the answer. In its place come four families, all trained against the true polygon:
   - seeds from the adjacent section, scaled to fit inside the cell (the Viking workflow);
   - shrunk or cut seeds;
   - masks to correct (too large, scaled, warped, or merged with a neighbor);
   - near-correct masks the model must leave alone.
3. **Several prompts per encoded image.** Each sample is encoded once and decoded with three prompt variants. The image encoder is most of the cost, so prompt variety triples for little extra compute.
4. **Tolerant validation.** IoU that ignores the edge band, boundary F1 with a few pixels of tolerance, and per-family checks, including "does no harm" on near-correct masks.

The negative-only prompt is already removed from the plan (assessment 1.5); this plan assumes it is gone.

## 1. Edge confidence curve

**Definition.**
- d is the distance in full-resolution pixels from the polygon edge, from a Euclidean distance transform.
- The confidence is c(d) = 1 − exp(−d / τ). It is 0 at the edge, about 0.63 at τ, 0.95 at 3τ, and effectively 1 beyond.
- Default τ = 2.5 px: 95% confidence about 7.5 px (16 nm) inside the traced line. Membranes here are about 3–5 px thick, so this band covers tremor without blurring which side of the membrane the cell is on.

**Why this shape.** You trust the centers and expect errors only at the traced edge, so confidence should reach 1 within a few pixels. A curve set as a percentage of object size would give thin processes a 2-px band and large cells a 60-px one.

**Systematic bias is not covered.** A symmetric band absorbs random wobble. It does not correct an annotator who consistently traces on one side of the membrane; the model would learn that side.

## 2. Mask prompts with edge confidence

**Encoding.** The signed dense prompt becomes continuous instead of −1 / 0 / +1:

| Region | Logit fed to `mask_input` |
| --- | --- |
| Target seed (trusted, inside the cell) | +M_seed × c(d), with M_seed = 10 |
| Mask to correct (may be wrong) | +M_corr × c(d), with M_corr = 3 |
| Other annotations (negative context) | −10 × c(d_neighbor), using each neighbor's own edge distance |
| Everywhere else | 0 (unknown) |
| Overlap of + and − | sum, clipped to ±10 |

- **Two strengths.** M_corr is lower so the mask network can learn that a mask to correct is a suggestion, while a seed is trusted. SAM 2 itself was trained to correct its own soft prediction logits.
- **Downsampling.** The values are now continuous, so downsample to the 256×256 `mask_input` by area averaging, not nearest-neighbor.
- **Neighbor-merge samples (section 3.3).** Leave out the merged neighbor's −1 context; otherwise the error is labelled for the model.
- **Containment loss.** It applies only where the prompt is a seed, weighted by c(d), so it is effectively the trusted core. It is never applied to masks to correct or near-correct masks.
- **Production must use the identical function.** The server must build prompts with the same c(d), τ, M_seed and M_corr. Put the formula and constants in one shared module (`annotation_crops/prompt_encoding.py`) and mirror it in the server.

## 3. Prompt families

The full-mask prompt (30%) is replaced. Proposed starting mix, configurable:

| Family | Share | Input | Encoding | Containment |
| --- | --- | --- | --- | --- |
| Points only | 10% | 1–9 positive points | none (plus −1 context) | no |
| Box only | 5% | jittered box | none (plus −1 context) | no |
| Box + points | 5% | box + 1–3 points | none (plus −1 context) | no |
| Circle seed | 40% | inscribed circle × 0.95 + center point | seed | core only |
| Adjacent-section seed | 15% | linked polygon from z ± 1–2, scaled to fit inside | seed | core only |
| Shrunk or cut seed | 10% | subset of the true mask | seed | core only |
| Mask to correct | 10% | grown, scaled, warped or merged mask | correction | no |
| Near-correct mask | 5% | true mask with tiny perturbation | correction | no |

Every family is trained against the true polygon of the target section. Background points and −1 context from other annotations ride along as now.

### 3.1 Adjacent-section seed (the Viking workflow)

Annotators bring the polygon from the adjacent section and scale it with the mouse wheel until it fits inside the cell. To train on that:

1. Pick a location of the same structure in section z ± 1 (or ± 2) that has a polygon.
2. Take its shape relative to its own centroid, and place it at the target's inscribed-circle center.
3. Find by bisection the largest scale s* at which at least 98% of the seed lies inside the target polygon. The 2% allows for tremor.
4. Apply s = s* × U(0.7, 1.0), since people will not fit it exactly, plus a small random offset that keeps it inside.
5. Encode it as a seed with the edge curve.

This is a better seed than the inscribed circle for long, thin or curved profiles, where a circle covers only a small part of the cell.

**Data needed.** The exporter must provide, per location:
- `structure_id`;
- the linked location ids in adjacent sections (Viking `LocationLinks`);
- each location's polygon rings relative to its centroid, in D1 pixels.

Put the shapes in a per-volume `shapes.jsonl`, so the trainer does not reassemble shapes from tiled mask PNGs. `structure_id` also enables structure-level analysis later.

**Faithful variant (later).** Place the linked polygon through the section-to-section volume transforms instead of re-centering it. That reproduces real placement error as well, but needs nornir's transforms in the trainer path.

### 3.2 Shrunk or cut seed (a subset of the truth)

- **Erosion:** shrink the true mask by U(3, 30) px.
- **Straight cut:** a random line through the mask keeps 20–80% of its area. This mimics a mask that stopped at a tile edge.
- **Shape-following cut:** keep the pixels within a random distance from an interior anchor point, measured along the inside of the cell. This mimics a mask that stopped partway along a process or at a branch.
- Keep only the connected piece containing the anchor.

### 3.3 Mask to correct (not a subset)

- **Grow:** dilate by U(3, 40) px. This models tracing outside the membrane.
- **Wheel-style scale:** U(0.8, 1.25) about the centroid.
- **Outline wobble:** an elastic warp of the mask, giving over- and under-shoot in different places.
- **CascadePSP-style:** simplify the outline, then randomly dilate or erode.
- **Neighbor merge:** union with part of an annotated neighbor's mask, teaching the model to remove a merge.

### 3.4 Near-correct mask ("do no harm")

The true mask with a ±1–3 px boundary perturbation, encoded as a correction mask. It teaches the model to leave a good mask alone. Without it, a refinement model learns that every input mask needs changing.

### 3.5 Exclusions

- Tiles where the target covers at least 99% of the tile (assessment 1.9) get only point or circle prompts. They have no edge to refine.
- If a derived seed or mask is empty after augmentation, the sample falls back to a circle seed.

## 4. Edge-tolerant loss

**Labels stay hard (0/1). The loss is weighted per pixel instead:**

w(d) = max(w_min, 1 − exp(−d / τ_t))

- d is the distance to the ground-truth polygon edge, on both sides.
- τ_t = 2.5 px, matching the prompt curve.
- w_min = 0.05 keeps a little gradient at the edge.

**How it applies to each term:**
- **Focal:** per-pixel weight, normalised by the weight sum.
- **Dice:** weighted soft dice, with sums of w·p·g, w·p and w·g.
- **IoU-head target:** IoU of the thresholded prediction over pixels with w ≥ 0.5, which matches tolerant validation.
- **Mask containment:** unchanged apart from the c(d) weighting in section 2.

**Why weight the loss rather than soften the target.**
- Soft targets near the edge (0.8, 0.6, …) teach the model to output that uncertainty. Edges come out blurred and low-confidence, and the 0.5 threshold still lands on the traced line on average, so the model gets no freedom to find the membrane.
- Down-weighting makes a few pixels' disagreement with the trace nearly free near the edge, so image evidence decides where the edge goes. Supervision stays at full strength elsewhere.
- For a two-class mask, boundary label relaxation (Zhu et al., CVPR 2019) reduces to this.

## 5. Several prompts per encoded image

- **Per sample:** three prompt variants against the same target, with `prompt.variants_per_sample = 3`.
  - Variant 1 is always the production prompt (circle seed, or adjacent-section seed when available).
  - Variants 2 and 3 are drawn from the mix without replacement.
- **One encoder pass.** `predict_masks_mixed` already encodes once and decodes in groups. Extend it to take a variant→sample index and reuse the embeddings with `index_select`.
- **Loss:** average over a sample's variants, so samples with more variants do not weigh more on the encoder.
- **Seeding:** variant k uses `rng([seed, epoch, index, k])`, so resumed epochs still replay exactly.
- **Cost:** decoder activations grow by about 3×, still small next to the Hiera-L encoder. Check with `--benchmark` before a full run.
- **Run length:** prompts seen per epoch triple. Reconsider `num_epochs` (the run is already about 6× the original design; see the run overview).
- **Precedent:** SAM trains this way, with one image-encoder pass and many prompts and correction rounds through the light decoder.

## 6. Validation

Use a fixed seed and the z-slab validation split. Report per family:

| Check | Metric |
| --- | --- |
| Circle seed (production) | Tolerant IoU, boundary F1 at 2 and 4 px |
| Adjacent-section seed | Tolerant IoU, boundary F1 |
| Mask to correct | Tolerant IoU of the output, and its gain over the input mask's IoU |
| Near-correct ("do no harm") | Share of samples where output IoU ≥ input IoU − 0.01 |
| Centroid point (legacy) | Raw IoU, kept for continuity with earlier runs |

- **Tolerant IoU** ignores pixels with w < 0.5.
- **Boundary F1** follows the tolerance-based boundary metrics used for EM segmentation (Arganda-Carreras et al. 2015).
- **Checkpoint selection** moves to tolerant IoU on the circle-seed family (assessment 1.1).

## 7. Production requirements (server and Viking)

- **Shared encoding:** the server builds `mask_input` with the shared function and constants from section 2.
- **Adjacent-section polygon:** when an annotator brings a polygon from the adjacent section and scales it to fit inside the cell, Viking sends it as a **seed**.
- **New command, "Resegment this polygon":** sends an existing polygon as a **mask to correct** (M_corr), optionally with correction clicks.
- **Clamp:** no hard clamp on masks to correct. If the +6 clamp is adopted at all, it applies only to a seed's trusted core.

## 8. Configuration

| Parameter | Default | Description |
| --- | --- | --- |
| `prompt.edge_tau_px` | 2.5 | τ of the prompt confidence curve |
| `prompt.seed_logit` | 10.0 | M_seed |
| `prompt.correction_logit` | 3.0 | M_corr |
| `prompt.variants_per_sample` | 3 | Prompt variants decoded per encoded sample |
| `prompt.prob_circle_seed` | 0.40 | Replaces `prob_circle_mask` |
| `prompt.prob_adjacent_seed` | 0.15 | Needs the exporter changes in 3.1; its share moves to circle seed until then |
| `prompt.prob_subset_seed` | 0.10 | Shrunk or cut seed |
| `prompt.prob_correction` | 0.10 | Mask to correct |
| `prompt.prob_near_correct` | 0.05 | Near-correct mask |
| `prompt.prob_points_only`, `prob_box_only`, `prob_box_points` | 0.10, 0.05, 0.05 | Unchanged |
| `prompt.adjacent_fit_inside` | 0.98 | Minimum share of the scaled seed inside the target |
| `prompt.adjacent_scale_range` | [0.7, 1.0] | Factor applied to the fitted scale |
| `prompt.erode_px` / `dilate_px` | [3, 30] / [3, 40] | Shrink and grow ranges |
| `prompt.scale_range` | [0.8, 1.25] | Wheel-style scale for masks to correct |
| `loss.edge_tau_px` | 2.5 | τ of the loss weight; 0 disables edge weighting |
| `loss.edge_weight_min` | 0.05 | w_min |

Removed: `prompt.prob_full_mask`, `prompt.prob_negative_only`, `negative.allow_circle_targets`, `negative.circle_target_fraction`.

## 9. Implementation order

**Phase A: trainer only, no new export.**
1. Shared `prompt_encoding` module: c(d), seed and correction encodings, area downsampling. Unit tests.
2. Edge-weighted focal, dice and IoU-head target; tolerant IoU and boundary F1 in validation. Unit tests on synthetic masks.
3. Families 3.2–3.4; remove the full-mask and negative-only types. Unit tests per family:
   - shrunk and cut seeds stay inside the truth;
   - masks to correct differ from it;
   - near-correct masks stay within 3 px.
4. Multiple variants per sample in the dataset, collate and `predict_masks_mixed`. Test that one encoder pass serves all variants.
5. Per-family validation. Short `--benchmark` run to check memory and speed.

**Phase B: exporter.**
1. Add `structure_id` and linked location ids to the manifest, and write `shapes.jsonl`.
2. Fix the `circles.py` key regex for negative window origins.
3. Add the adjacent-section seed family (3.1).

**Phase C: later.**
1. Correction masks from the model's own predictions, used for about half of the correction samples (as in μSAM).
2. Correction clicks sampled from the error region of masks to correct (as in SAM, SAM 2 and RITM).

**Phase D: production.**
1. The server uses the shared encoding.
2. Viking sends adjacent-section polygons as seeds and adds the "Resegment" command.

**Acceptance for Phase A**, on the z-slab validation split, against the current model:
- tolerant IoU on circle seeds does not drop;
- masks to correct improve on their input on average;
- at least 95% of near-correct samples pass "do no harm".

## 10. Open questions

- **τ.** 2.5 px is an estimate. It can be measured from data: compare traced edges with the nearest membrane (image intensity minimum) along edge normals.
- **The mix.** The shares in section 3 are a starting point. Adjust after the first per-family validation.
- **M_corr = 3.** It could be tuned by validating with the input mask at a few strengths.

## References

- Kirillov A. et al. Segment Anything. *ICCV* 2023.
- Ravi N. et al. SAM 2: Segment Anything in Images and Videos. *ICLR* 2025.
- Ke L. et al. Segment Anything in High Quality (HQ-SAM). *NeurIPS* 2023. Mask prompts made by adding noise to ground-truth boundaries.
- Cheng H.K. et al. CascadePSP: Toward Class-Agnostic and Very High-Resolution Segmentation via Global and Local Refinement. *CVPR* 2020. Coarse inputs made by simplifying the outline, then random dilation and erosion.
- Sofiiuk K., Petrov I., Konushin A. Reviving Iterative Training with Mask Guidance for Interactive Segmentation. *ICIP* 2022.
- Archit A. et al. Segment Anything for Microscopy. *Nat. Methods* 22, 579–591 (2025).
- Zhu Y. et al. Improving Semantic Segmentation via Video Propagation and Label Relaxation. *CVPR* 2019.
- Arganda-Carreras I. et al. Crowdsourcing the creation of image segmentation algorithms for connectomics. *Front. Neuroanat.* 9:142 (2015).
