# Assessment: SAM2 v4 Training Plan

Oct 12, 2026 · Updated Oct 13, 2026 · Review of "Overall training plan", checked against `Sam2Trainer`, the nornir circle exporter and the v2 export on `Y:\SAM2TrainingData`.

The SAM 3 question is now its own document: *claude/SAM3 evaluation*.

## Summary

The plan is well reasoned. The document drift found in the first review was fixed in the plan on Oct 13 (see 1.12). Three problems still matter more than any of the prompt-mix details:

1. **Validation measures the wrong thing, and it leaks.** Checkpoint selection and early stopping use one centroid point, but production uses a circle. The split is by location, so adjacent sections of the same cell, and overlapping windows of different locations, fall on both sides of the split.
2. **The full-mask prompt (30%) is almost an identity task.** The ground truth is the prompt. SAM 2 itself never trains the decoder this way.
3. **The negative-only prompt was ill-posed.** With no positive cue, nothing identifies *which* unannotated cell is the target. It is now removed from the plan (1.5); the code change is pending.

**Decisions made since the first review**

- **Circle scale 0.95** for all volumes (1.3). Annotators are instructed to keep circles inscribed, and measurement confirms about 1% exceed their cell. Suspect circles are listed for review rather than handled by a per-annotator rule.
- **Validation split by z-slab per volume** (1.2; implemented). This replaces the location split and closes both the adjacent-section leak and the window-overlap leak. Whole-volume holdout was rejected.
- **Fully masked tiles are capped, not removed** (1.9). Keep about a quarter of the tiles where the target covers ≥99% of the tile, give them only point or circle prompts, and keep them out of `val/iou`.

## 1. Weaknesses and incorrect uses of SAM 2

### 1.1 Validation does not measure the production task (high)

- `EMSegDataset` validation always uses one point at the mask centroid. `val/iou` drives best-checkpoint selection and early stopping.
- The circle prompt is 45% of training and *the* production prompt, but it is never evaluated. So there is no evidence that mixed-prompt training helps the real workflow.
- The centroid is not always inside the target. In 250 sampled masks per volume, it fell outside the mask in **6.4% (RC1), 2.8% (RC2), 4.4% (RPC1) and 1.6% (RPC2)** of cases, mostly Müller cells and other irregular shapes, where the prompt then points at a different structure.
- **Recommendation:** run a fixed-seed validation per prompt type, with circle-mask as the primary metric.
  - Add an interaction-cost metric: clicks needed to reach 90% IoU (NoC@90), the standard interactive-segmentation measure (e.g. Sofiiuk et al., ICIP 2022; SAM reports the related mIoU-vs-clicks curve). It maps directly onto "reduce annotator labour".
  - Report a boundary metric (boundary F1 or Rand/VI, as in Arganda-Carreras et al., *Front. Neuroanat.* 2015), because split and merge errors cost connectomics more than interior pixels do.

### 1.2 Train/validation leakage inflates `val/iou` (resolved Oct 13: z-slab split implemented)

- `splits.py` splits 90/10 by `(volume, locationId)`. Two things cross that split:
  - **Same cell, adjacent sections.** A Viking structure has a separate location in every section, and adjacent serial sections are tens of nm apart and nearly identical. So a cell's section *z* can be in training and *z*+1 in validation.
  - **Overlapping windows of different locations.** This applies to the tiled exporter (the default, used for v2). The legacy path and the segmentation server are separate.
    - A location's own windows never overlap. But each small location picks its window from a 512-px half-step lattice, and large locations are tiled at one of four 512-px phases, so windows of *different* locations overlap by 50% or 25%.
    - Overlap measured on v2: 49% of windows in RC1 and RC2, 42% in RPC1, 17% in RPC2.
    - Validation cells therefore appear unlabeled in training images. They also appear as −1 regions, because their inscribed circles are written into every window they overlap.
    - See *Overall training plan*, Part 1, *Crop window placement*.
- **Decision (Oct 13): split by z-slab within each volume.**
  - Every window in a section goes to the same side of the split, and windows only overlap within a section. So this one change closes **both** leaks: adjacent sections, and overlapping windows of different locations.
  - The tiled exporter's half-step lattice can therefore stay, and objects stay whole in their windows.
  - **Implemented (Oct 13; commit `ba05ea3` on branch `zslab-split`)** in `data/splits.py` (`make_zslab_split`), `train.py` and `config/em_finetune.yaml`. It is the default as `data.split_mode: zslab`, with unit tests in `tests/test_zslab_split.py`.
    - Validation is X = 1 − `train_split` (10%) of each volume's samples, in **5 contiguous z-slabs** centered at 10/30/50/70/90% of the z-range.
    - Each slab grows from its center, weighted by crop-window samples, to the whole-section size nearest X/5.
    - **1 guard section** on each side of every slab is used by neither side.
    - New locations found on an inventory refresh are placed by their section.
    - Existing runs keep their own `split.json`; a new split needs `-refresh`.
  - Result on the v2 manifests: see the table below.

**z-slab split on the v2 manifests**

| Volume | Sections | Train | Val | Guard | Slabs (z, data sections) |
| --- | --- | --- | --- | --- | --- |
| RC1 | 247 | 86.5% | 9.7% | 3.7% | 30–33 (3), 83–91 (7), 140–147 (8), 198–202 (5), 254–257 (4) |
| RC2 | 490 | 89.0% | 9.8% | 1.1% | 140–147 (6), 319–645 (33), 675–748 (61), 829–1119 (65), 1268–1291 (22) |
| RPC1 | 383 | 87.3% | 10.0% | 2.8% | 363–368 (6), 447–451 (4), 521–543 (21), 603–631 (27), 698–702 (5) |
| RPC2 | 872 | 88.3% | 9.9% | 1.8% | 288–308 (20), 461–508 (45), 632–708 (74), 839–874 (34), 1042–1044 (3) |

Slab widths vary because annotation density varies with depth. Where sections are sparse (RC2), a slab spans a wide z-range.

- **What it measures:** new sections of cells that may also appear, traced, in other sections. That matches the production task.
- **What it does not measure:** generalization to a new specimen. Cover that with the production measure below.
- **Alternatives considered:**
  - **Whole-volume holdout:** rejected; it costs a volume of training data.
  - **3D blocks (z-slab × XY region):** not needed, because z alone closes the overlap leak.
  - **Structure-level holdout:** needs `structure_id` and still leaks through overlapping windows.
  - **Fixed 1024 grid in the exporter:** not needed for leakage; it would cut 56–69% of single-window objects.
- **Production measure (recommended alongside the split):** log the server's proposals against the polygons annotators finally save in Viking (accepted unchanged, vertices edited, clicks added). This is the direct measure of labor saved, and it covers new volumes as they arrive.

### 1.3 Circle-mask prompt and circle scale (resolved: 0.95)

- For polygon targets the training circle is the *inscribed* circle, the largest circle inside the polygon. In Viking a location is either a circle or a polygon, so no polygon target has an annotator-drawn circle.
- Annotators are instructed to keep circles inscribed in the cell, so the inscribed circle is a fair stand-in for the production prompt.
- **Measured (Viking OData, Oct 13).** Each circle's radius was compared with the equal-area radius of the same cell's linked polygon within two sections. That compares sizes only, so misregistration between sections doesn't matter.
  - The median ratio is about 0.45 in RPC1 and RPC2, and the same for rpfeiffer, JHY and the other annotators.
  - About 1% of circles exceed their cell.
  - No time window for the Viking radius bug is visible.
- **Decision:** circle scale **0.95**, with no per-annotator rule. It is applied by the trainer (`prompt.circle_diameter_scale`); the exporter writes true radii (nornir `3b6fe2f` removed `-CircleDiameterScale`). Since Oct 15 the scale applies to inscribed (seed) circles only. Annotated circles, used only as negatives, load at their true radius and are trusted by `prompt.circle_annotation_trust`: full strength inside 0.5 of the radius, tapering to 0 at 0.95 (encoding `edge-confidence-v6`).
- **Suspect circles** are listed for review in `Y:\SAM2TrainingData\circle_review\` (`circle_review.xlsx`, from `scripts/circle_review.py`). There are 52,355 of about 2.0M circles:
  - 335 are larger than their linked polygon, almost certainly wrong;
  - 38,998 are ~2× jumps between linked circles, which need a human look;
  - 13,022 are copies of a flagged radius.
- **Still open:** confirmed-bad circles should be skipped by the exporter, and the planned +6 clamp at inference should not be applied to circles flagged as oversized.

### 1.4 Full-mask prompt teaches copying (high)

- In this type the dense prompt is the ground-truth mask itself (nearest-downsampled to 256²), and the loss target is the same mask. The model learns to copy and upsample its input. That teaches little about finding membranes, while still lowering the training loss.
- In SAM 2's own training, a ground-truth mask prompt is a 50% initial prompt for video, but on that frame the decoder is bypassed (`use_mask_input_as_output_without_sam`). The mask only seeds memory for *other* frames.
- Where SAM 2 does feed a mask into the decoder, it is the **model's own previous low-res logits** during corrective-click rounds: 7 corrections, with clicks sampled from the error region.
- μSAM (Archit et al., *Nat. Methods* 2025) feeds the previous mask only 50% of the time, because always feeding it caused over-reliance. RITM showed that mask-guided iterative training helps.
- **Recommendation:** replace the full-mask type with:
  - **iterative corrective training:** predict, sample a click from the error region, then re-run with the previous logits plus the new click, for 3–8 rounds;
  - **plus a small share of deliberately corrupted masks** (random erosion or dilation, partial strokes, coarse polygonization) to stand in for brush-painted input.

### 1.5 The negative-only prompt is ill-posed (decided Oct 13: remove)

- With only background points and −1 regions, the target is not identifiable whenever the tile holds more than one unannotated profile, which in retinal neuropil is always. This trains guessing and miscalibrates the IoU head.
- The optional circle targets (`allow_circle_targets`) are worse: they teach the model to output discs.
- **Decision (Oct 13):** remove the negative-only type from the plan, together with its circle-disc targets. Negatives stay as −1 context and background points on the other types.
  - Its 5% is proposed to go to circle-mask (45% → 50%).
  - The trainer code is not yet changed; the plan lists the pending edits.

### 1.6 Circle negatives can contradict the ground truth (medium)

- `combine_signed` cancels −1 only against a +1 foreground canvas. Points, box and box+points prompts have no canvas.
- So a neighbor's circle that overlaps the target's true extent stamps −1 on pixels the loss calls foreground, and the model is told two opposite things.
- Clipping negatives with the ground truth would leak the target boundary into the prompt.
- **Recommendation:** now that circles are exported, measure the overlap rate, then drop or shrink circle negatives that intersect the target. With circles inscribed by instruction, this should be rare; the circle review list covers the exceptions.
- **Measured Oct 14:** 2.4% of examples have an overlapping circle negative (see the baselines appendix). `negative.drop_overlapping_circles` is on for v4.

### 1.7 Circle files and the realized prompt mix (check)

- On Oct 12 the v2 export had no `circle_annotations.json`. Without it, circle-mask is never available, and the effective mix becomes about 60% full-mask.
- **As of Oct 13** all four v2 volumes have the file (RC1, RC2 and RPC2 written Oct 12; RPC1 at 01:32 Oct 13).
  - Since nornir `3b6fe2f` every file records `diameter_scale` 1.0 (true radii); the trainer rescales older files that baked a scale in.
  - The volume layout also changed: files now sit directly in `v2/{volume}/` with no `AnnotationCrops/` subfolder. Check the trainer's data path.
  - An empty `v3` folder exists.
  - Crop keys with negative origins (e.g. `X-512-512`: 148 in RPC1, 2 in RPC2) are not parsed by `circles.py` (`_IMAGE_KEY_WINDOW` expects non-negative numbers). Those crops get no circle entries.
- **Recommendation:** have the loader log the realized prompt-type mix each epoch, and fail loudly when circles are expected but missing.

### 1.8 Field of view and target size (medium)

- All tiles are D1: 1024 px × 2.18 nm ≈ 2.2 µm. Large cells span up to 13 windows. In about 2,500 sampled annotations:
  - 5–12% of targets cover at least 90% of the tile;
  - 14–25% cover more than half;
  - some are thin slivers at the tile edge.
- In these cases the boundary is mostly out of view.
- **Recommendation:**
  - Export D2 and D4 tiles as well, and train across scales.
  - Have the inference server choose a scale from the circle radius, so the object fills roughly 10–60% of the tile.
  - This is the long-term fix for 1.9.

### 1.9 Fully masked interior tiles (decided: cap, don't remove)

**Size.** Share of targets by fraction of the tile they cover, from the re-organized v2 export (Oct 13; about 700 sampled annotations per volume):

| Volume | ≥90% | ≥99% | 100% | Mostly |
| --- | --- | --- | --- | --- |
| RC1 | 4.2% | 3.0% | 2.4% | Müller cells |
| RC2 | 0.5% | 0.1% | 0% | — |
| RPC1 | 10.9% | 7.2% | 5.7% | Horizontal cells |
| RPC2 | 8.4% | 5.5% | 4.5% | Horizontal and bipolar cells |

**Why cap them**
- A tile with no boundary gives focal and dice loss only one lesson: predict foreground everywhere. That nudges the model toward over-filling, which is the failure that matters most at membranes.
- With a full-mask prompt the sample is a pure copy.
- In validation, a centroid point in a fully masked tile scores IoU ≈ 1, which inflates `val/iou` and early stopping by a few percent in RPC1 and RPC2.

**Why not remove them**
- The segmentation server walks large cells across tiles, so production does meet tiles that lie entirely inside a cell.
- A model that has never seen one may invent a boundary there, for example cutting out a horizontal cell's nucleus or stopping at a large organelle.
- These tiles are the only place it learns that interior organelles belong to the cell.

**Decision**
1. **Training:** keep about a quarter of the tiles where the target covers ≥99% (roughly 1% of samples), chosen deterministically per location.
2. **Prompts:** give those tiles only point or circle prompts, never full-mask.
3. **Keep 90–99% tiles unchanged.** They hold the boundary fragments that matter.
4. **Validation:** exclude ≥99% tiles from `val/iou`, and log them as a separate metric.

**Implementation (pending)**
- Two settings, applied when the manifest is indexed, using the `area` already in each sidecar:
  - `data.full_tile_threshold: 0.99`
  - `data.full_tile_keep_fraction: 0.25`
- Revisit once D2/D4 tiles exist (1.8). At coarser scales these cells fit in the tile with their boundary visible.

### 1.10 Class and volume imbalance (medium)

- One cell class dominates each volume: Müller cells are about 77% of sampled RC1 annotations, horizontal cells about 48% of RPC1 and bipolar cells about 55% of RPC2.
- RC2 also contains vessels, which are not cells.
- **Recommendation:** report metrics per class and per volume, and consider class-balanced sampling so amacrine and ganglion processes are not under-trained.

### 1.11 Minor SAM 2 conventions

- **IoU-head loss.** The code uses MSE. SAM 2's paper and configs use L1 (`iou_use_l1_loss: true`). This is minor; L1 is less dominated by outliers.
- **Ambiguity.** `multimask_output=False` everywhere. SAM, SAM 2 and μSAM output 3 masks for a single-click prompt and back-propagate the best one.
  - This matters little for whole-cell targets today.
  - It matters for phase 2, where a click inside a mitochondrion inside a dendrite is genuinely ambiguous.
- **Object-score head.** It is ignored. That is fine, but keep it out of any inference gating, because it is untrained here.
- **Pre-trained encoder.** Full fine-tuning at encoder LR × 0.1 is reasonable. Keep a zero-shot SAM2.1-L baseline on the held-out set to quantify the gain.
- **SEM (secondary goal).** There is no SEM in the training data and no plan for it. Expect a domain gap; see *claude/SAM3 evaluation*.

### 1.12 Document drift (resolved Oct 13)

The plan was updated to match the code. The corrections:
- **Prompt mix:** probabilities sum to 1.0, with circle-mask at 0.45.
- **Boxes:** passed as `boxes=`, with no box containment loss or box clamp.
- **Mask prompts:** fed as ±10 logits at 256×256, and the doc states that inference must use the same scale.
- **Mask network size:** `mask_downscaling` is about 4.7K parameters.
- **Exporter:** inscribed circles for polygon targets, a `source` field, no clipping, and true radii (the trainer applies the 0.95 scale).
- **Caching and decoding:** neighbor masks are cached per worker, and samples are decoded in groups by prompt shape.

## References

- Kirillov A. et al. Segment Anything. *ICCV* 2023.
- Ravi N. et al. SAM 2: Segment Anything in Images and Videos. *ICLR* 2025. arXiv:2408.00714. Training code: `facebookresearch/sam2/training`.
- Archit A. et al. Segment Anything for Microscopy. *Nat. Methods* 22, 579–591 (2025).
- Sofiiuk K., Petrov I., Konushin A. Reviving Iterative Training with Mask Guidance for Interactive Segmentation. *ICIP* 2022.
- Arganda-Carreras I. et al. Crowdsourcing the creation of image segmentation algorithms for connectomics. *Front. Neuroanat.* 9:142 (2015).

## Appendix: measurements behind this review

**v2 export as of Oct 12** (`AnnotationCrops/` layout):

| Volume | Tiles | Locations | Tiles with ≥2 annotations | Targets ≥90% of tile | Centroid outside mask |
| --- | --- | --- | --- | --- | --- |
| RC1 | 52,950 | 15,912 | 7.8% | 5% | 6.4% |
| RC2 | 12,827 | 3,977 | 5.1% | 6% | 2.8% |
| RPC1 | 71,920 | 16,384 | 5.9% | 12% | 4.4% |
| RPC2 | 52,315 | 13,038 | 1.8% | 9% | 1.6% |

- Tile and location counts come from the full manifests.
- Target-area fractions come from about 600 sampled annotations per volume; centroid checks from 250 sampled masks per volume.
- All tiles were downsample 1.
- The Oct 13 re-organized export gives somewhat different area fractions (see the table in 1.9), most noticeably for RC2.

**Window overlap (v2 manifests, Oct 13; tiled exporter):**

| Volume | Windows | Off the 1024 grid | Overlap another window | Single-window objects a fixed grid would cut |
| --- | --- | --- | --- | --- |
| RC1 | 53,057 | 75% | 49.2% | 56% (33/59 sampled) |
| RC2 | 9,706 | 75% | 48.5% | 60% (37/62) |
| RPC1 | 71,826 | 75% | 42.3% | 63% (17/27) |
| RPC2 | 52,317 | 74% | 17.2% | 69% (20/29) |

- Overlap is counted between distinct windows in the same section.
- "Would cut" means the object's bounding box crosses a multiple of 1024 in x or y. These come from about 500 sampled sidecars per volume, so the counts are small.

**Circle measurements (Viking OData, Oct 13):** circle radius ÷ linked polygon's equal-area radius, by the circle's annotator.

| Volume | Annotator | Pairs | Median | 90th percentile | Larger than cell |
| --- | --- | --- | --- | --- | --- |
| RPC1 | rpfeiffer | 6,365 | 0.46 | 0.76 | 1.1% |
| RPC1 | JHY | 5,399 | 0.43 | 0.67 | 0.9% |
| RPC2 | JHY | 4,042 | 0.45 | 0.72 | 0.9% |
| RPC2 | rpfeiffer | 843 | 0.49 | 0.76 | 1.3% |

**Baselines on the v4 z-slab validation split (Oct 14, `sam2-em-eval --goal`).** 19,176 validation samples; 828 fully masked tiles scored apart, leaving 18,348. Core recall is pooled over pixels; "at target" is the share of samples with core recall ≥ 97.5%.

| Prompt family | Model | Core recall | At target | Merge rate | Spill | Fragmented | Do no harm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Circle seed | Zero-shot SAM2.1-L | 0.534 | 12% | 2.1% | 0.105 | 45% | — |
| Circle seed | v3 | 0.753 | 16% | 1.5% | 0.034 | 10% | — |
| Deepest point | Zero-shot | 0.663 | 29% | 6.9% | 0.319 | 44% | — |
| Deepest point | v3 | 0.774 | 17% | 1.1% | 0.035 | 9% | — |
| Centroid point | Zero-shot | 0.654 | 29% | 6.9% | 0.346 | 45% | — |
| Centroid point | v3 | 0.986 | 76% | 0.4% | 0.015 | 3% | — |
| Mask to correct | Input mask | 0.965 | 66% | 2.5% | 0.051 | 3% | — |
| Mask to correct | Zero-shot | 0.507 | 23% | 1.6% | 0.087 | 41% | — |
| Mask to correct | v3 | 0.251 | 9% | 1.3% | 0.021 | 8% | — |
| Near-correct | Input mask | 0.994 | 80% | 0.1% | 0.006 | 2% | — |
| Near-correct | Zero-shot | 0.488 | 22% | 1.3% | 0.075 | 41% | 6% |
| Near-correct | v3 | 0.106 | 6% | 1.2% | 0.015 | 8% | 2% |

- **v3's centroid result is most likely leakage.** v3 predates the z-slab split, so it was probably trained on the old location split, in which case many v4 validation sections were in its training set, prompted with the same centroid point. Its `split.json` would confirm this. Moving the click to the deepest interior point, a geometrically better prompt, drops core recall from 0.986 to 0.774. Read v3's centroid row as memorisation and its other rows as optimistic.
- **The honest bar for v4** is v3's circle-seed row: core recall 0.753, merge rate 1.5%, spill 0.034, 10% fragmented. The goal (≥ 97.5% of the interior, no merges) is far from met; only about one sample in six reaches it.
- **Neither model can use a mask prompt yet.** Given masks that already meet the goal, both destroy them (near-correct core recall 0.994 → 0.11–0.49; do no harm 2–6%). This is the capability the refinement families train; v4 must reach ≥ 95% do no harm.
- **Approved masks:** 777 entries were found, but only 23 validation samples are Approved, too few to use alone.
- **Circle overlap (1.6), measured on 2,000 sampled examples:** 1,637 have at least one circle as negative context (2,396 circles). In 40 examples (2.4%) a circle negative overlaps the target; 43 of the 2,396 circles (1.8%) do. By volume: RPC1 25 of 713 examples with circles, RC1 10 of 640, RPC2 5 of 167, RC2 0 of 117. Of all circle entries, 158,492 are inscribed circles of polygons and 17,792 (10%) are annotated circles.
  - This is above the 1% bar set in 1.6, so `negative.drop_overlapping_circles` is turned on for v4.
  - The measurement does not say which source the overlapping circles come from, or how deep the overlap is. Splitting it by source (annotated or inscribed) and by whether the overlap reaches the inner half of the circle would show whether the annotated radii are the cause, and whether the radius-relative taper alone would be enough.
