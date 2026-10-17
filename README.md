# SAM2 Segmentation Trainer

GPU Dev Container for fine-tuning Meta SAM2 on TEM/SEM data. The trainer package is bind-mounted at `/workspace` and installed editable. Training data is the NAS share already used by Nornir (`/storage4` via in-container CIFS). The container installs `nornir_shared` for MQTT telemetry only; training code does not use the rest of Nornir.

## Prerequisites

- Docker Desktop on Windows with WSL2 backend and GPU passthrough (host driver CUDA 13.2 or newer; image is CUDA 13.2 + PyTorch `cu132`)
- `NORNIR_DOCKER_USER_ROOT=D:\Docker` in the Windows user environment (restart Cursor after setting)
- CIFS loaded on the WSL2 kernel (`grep cifs /proc/filesystems`)
- Existing Nornir net-mounts: `D:\Docker\Run\nornir-dev\net-mounts` and `...\secrets\net-creds` (do not copy `.cred` files into this repo)

## One-time machine layout

```powershell
# From this repo
Copy-Item docker\example.cursor-dev.run.env D:\Docker\Run\sam2-dev\.env
Copy-Item docker\example.compose.net-mounts.override.yaml D:\Docker\Run\sam2-dev\compose.net-mounts.override.yaml
New-Item -ItemType Directory -Force D:\Docker\Builds\sam2-trainer | Out-Null
New-Item -ItemType Directory -Force D:\Docker\mounted-configs\sam2-trainer | Out-Null
# Compose substitution: hardlink (same volume) or copy. Junctions cannot target files.
New-Item -ItemType HardLink -Path docker\.env -Target D:\Docker\Run\sam2-dev\.env
```

Edit `D:\Docker\Run\sam2-dev\.env` if your net-mount paths differ. Defaults match Nornir:

- `NORNIR_NET_MOUNTS_DIR_HOST=D:\Docker\Run\nornir-dev\net-mounts`
- `NORNIR_NET_CREDS_DIR_HOST=D:\Docker\Run\nornir-dev\secrets\net-creds`

## Build

```powershell
Set-Location D:\Docker\Builds\sam2-trainer
./rebuild-and-start.ps1 SEM      # or TEM
```

The first argument is the modality (`TEM` or `SEM`, required unless `-SkipRecreate`). It rebuilds `sam2-trainer:dev`, then brings `cursor-dev` to that modality on the new image and starts both galleries and `gallery-router`. `cursor-dev` is started if not running and recreated only when needed (new image, or a different modality), so a rebuild that changes nothing does not interrupt a training run. If the other modality is running it asks whether to replace it. `docker build` alone would leave the old container running.

Options: `-SkipRecreate` tags only, `-NoCache`. If Cursor is attached, wait for reconnect or **Reopen in Container**. The wrapper is machine-local and calls `docker/docker-build.ps1`.

Optional `--build-arg` files in that folder (Nornir names; the `example.*` template is not merged):

- `build.env`
- `.build.sam2-trainer-dev.env` (tag `sam2-trainer:dev`)

```powershell
Copy-Item D:\src\git\Sam2Trainer\docker\example.sam2-trainer-dev.build.env `
    D:\Docker\Builds\sam2-trainer\.build.sam2-trainer-dev.env
```

## Run / Dev Container

The same container serves TEM and SEM data, one at a time (they share the ports). What differs is in `TEM.env` / `SEM.env` next to the shared `.env` in `D:\Docker\Run\sam2-dev\` (templates: `docker/example.TEM.env`, `docker/example.SEM.env`):

| Key | TEM | SEM |
| --- | --- | --- |
| `SAM2_LOCAL_DATA_HOST` (mounted at `/data-local`) | `\wsl.localhost\Ubuntu\home\jamesan\sam2-training` | `...\sam2-training-sem` |
| `SAM2_OUTPUT_HOST` (`/outputs`) | `D:/Docker/mounted-configs/sam2-trainer` | `...-sem` |
| `SAM2_CHECKPOINT_ROOT` | `/storage4/Sam2Trainer` | `/storage4/Sam2Trainer/SEM` |

Both scripts require `-Modality TEM|SEM` and pass `--env-file docker/.env --env-file <Modality>.env` to Compose. If a trainer of the other modality is running they say so and ask whether to replace it (a non-interactive session stops instead). Compose itself has no defaults for these keys, so a launch without a modality env file fails with a message.

```powershell
# Mirror the data first (WSL): scripts/WSL/sync_training_data.sh   or   sem-sync_training_data.sh v1
& D:\src\git\Sam2Trainer\docker\run-cursor-dev.ps1 -Modality SEM
# Rebuild the image and recreate a running cursor-dev onto it:
& D:\Docker\Builds\sam2-trainer\rebuild-and-start.ps1 SEM      # or -SkipRecreate to only build
# The gallery shares /data-local; after switching modality recreate it (the script prints the command):
docker compose --env-file docker\.env --env-file D:\Docker\Run\sam2-dev\SEM.env -f docker\compose.cursor-dev.yaml up -d annotation-gallery-tem annotation-gallery-sem gallery-router
```

**Dev Container ("Reopen in Container").** Cursor starts Compose with `docker/.env` only, so it cannot pick a modality; the Compose file then stops with the "set SAM2_MODALITY via TEM.env or SEM.env" error. Start the container with `run-cursor-dev.ps1 -Modality ...` (or Compose with both `--env-file`s) and use **Attach to Running Container**. `COMPOSE_ENV_FILES` would do it, but Compose only reads it from the process environment (not from `.env`), and setting it globally would also change the default `.env` of every other Compose project on the machine.

## Paths inside the container

| Variable | Typical value | Role |
| --- | --- | --- |
| `SAM2_DATA_ROOT` | `/data-local/current` (symlink to a version; fallback `/storage4`) | Local NVMe or NAS volumes |
| `SAM2_OUTPUT_ROOT` | `/outputs` | Splits, pretrained download, TB pid/log |
| `SAM2_CHECKPOINT_ROOT` | `/storage4/Sam2Trainer` (SEM: `/storage4/Sam2Trainer/SEM`) | Run dir: `last.pt`, `best_model.pt`, epoch snaps, TB events (CIFS) |
| workspace | `/workspace` | This git checkout (editable install) |
| SAM2 source | `/opt/sam2` | Pinned `facebookresearch/sam2` |
| TensorBoard | host `8060` → container `6006` | Auto-started; reads `$SAM2_CHECKPOINT_ROOT/runs` |

If the 32k dataset is not under `/storage4`, add a row to `D:\Docker\Run\nornir-dev\net-mounts\nas-mounts.tsv` (no image rebuild).

## AnnotationCrops layout

Training reads RC1, RC2, RPC1, and RPC2 under `$SAM2_DATA_ROOT` (default `/data-local/current`). Each volume folder holds the export directly (flat: `{volume}/manifest.jsonl`). Older mirrors and the NAS export nest it under `{volume}/AnnotationCrops`; the trainer and gallery accept both, and the flat layout wins when a volume has a `manifest.jsonl` or `annotation_crops.sqlite`. To flatten an old version run `version-flatten <version>` (NAS scripts) or, on WSL, `TRAIN_ROOT=~/sam2-training scripts/NAS/version-flatten <version>`. Contents:

- `manifest.jsonl` — index (`image` or `jpeg`, `json`, `imageKey`, `locationIds`, `downsample`, `volume`, `z`)
- `images/` — tile PNG/JPEG plus COCO-RLE sidecar JSON
- `masks/` — required raster `{imageKey}_{locationId}.png` (examples without a mask PNG are skipped; no RLE fallback)
- `overlays/` — viewing only, not used for training
- `circle_annotations.json` — optional. Circle annotations keyed by `imageKey`, in that crop's pixel space: `location_id`, `center_x`, `center_y`, `radius` (the true radius, since the exporter records `diameter_scale` 1.0; a file from an older exporter has its scale baked in), `structure_label`, `has_polygon_mask`, and `source` (`annotated` for a location that is still a circle, `inscribed` for the largest circle inside a polygon). Written by the nornir exporter, which no longer takes `-CircleDiameterScale`. The exporter never scales; all scaling is the trainer's job. `prompt.circle_diameter_scale` (0.95) applies to *inscribed* circles only (the seed circles), rescaled over the file's own `diameter_scale`, so changing it needs no re-export. *Annotated* circles load at their true radius (only divided by the file's `diameter_scale` for an older file): their radii are unreliable (about 2x in the radius-bug cases), so they serve only as negatives, trusted by fractions of the true radius (`prompt.circle_annotation_trust`: full strength inside `full_frac` 0.5, a cosine taper to 0 at `zero_frac` 0.95). `sam2-em-eval` has `--circle-scale`, `--circle-full-frac` and `--circle-zero-frac` for the same values; without them it reads the model's own record. Training needs it while circle-mask prompts are on: with `prompt.prob_circle_seed > 0` a volume without a usable file stops the run (`data.require_circle_file`, set `false` to warn only). Crops whose key has a negative origin (`X-512-512`) are read correctly by the exporter since the key decoder accepts them.

One tile image can hold several `locationId`s (disk-efficient shared crops). The loader expands those to one training example per annotation. Every annotation is kept at its native downsample (D1–D128). Each new run stores its own `split.json` and `inputs.json` (volumes, per-volume train/val counts, data root) under the run directory. Every usable sample is kept. Validation is split by section (`data.split_mode: zslab`): within each volume, five contiguous runs of sections spaced evenly through the volume hold about 10% of its samples, and one section on each side of every run (`data.zslab.guard_sections`) is used by neither side. All crop windows of a section land on one side, so adjacent sections and overlapping windows never cross the split. `data.split_mode=location` restores the original random 90/10 over locations. An existing run keeps the split in its `split.json`; use `-refresh` to get the new one.

This is **not** Pascal VOC (`JPEGImages` / `Annotations` / `ImageSets`). Do not reshape the NAS.

## Local version mirrors

On the WSL host, `~/sam2-training` (TEM) or `~/sam2-training-sem` (SEM, via the `sem-` scripts) is mounted at `/data-local`, whichever `-Modality` you launch with. Source trees live under `/mnt/d/TrainingData/<ver>` (`v1`, `v1b`, `v2`, …). Sync from that host, not inside the container:

```bash
# Mirror the highest source version and point current at it.
scripts/WSL/sync_training_data.sh
# Mirror one version and select it.
scripts/WSL/sync_training_data.sh v1b
# Mirror without changing current.
scripts/WSL/sync_training_data.sh -noselect v1
# Preview what a mirror would change (nothing is written).
scripts/WSL/sync_training_data.sh -n -noselect v2
# Mirror a new version and carry the review lists and location_scores over from v2.
scripts/WSL/sync_training_data.sh -clone-from v2 v3
# Mirror from the NAS instead of D: (CIFS mount of //192.168.0.199/Data at /mnt/y; setup in scripts/WSL/wsl-training.README).
TRAINING_DATA_SRC=/mnt/y/SAM2TrainingData scripts/WSL/sync_training_data.sh -noselect v2   # reads .../TEM; use sem-sync_training_data.sh for SEM
# Only retarget current (no copy). No arg picks the highest local version.
scripts/WSL/select_training_data.sh v2
```

`current` is a relative symlink (`v2`). Restart training after it changes.

The mirror copies only what training and the gallery read: `images/`, `masks/`, the manifest, catalog and JSON files, and `overlays/thumb` and `overlays/parts`. `_work/` and the full-size overlay renders are skipped (tens of GB per volume). It runs one rsync per volume and folder in parallel; `-j N` sets the job count (default: CPUs, at most 8). The NAS `version-sync` skips the same folders.

A mirror keeps each volume's `approved.json`, `ignore.json`, and `ignored/` (rejected masks are moved back to `ignored/` after the copy) and the `location_scores` table in `annotation_crops.sqlite` (saved before the copy and merged back after). A new version starts without them, so build it with `-clone-from <older version>`: it hardlinks the older version (unchanged files take no extra disk space), copies the catalog and review files for real, and rsyncs only the changes. `-seed-from` carries the review state over without cloning.

`sync_training_data.sh` refuses to mirror when a volume is flat on one side and nested (`{volume}/AnnotationCrops`) on the other, because `rsync --delete` would swap the destination's layout. Flatten the source mirror first (the NAS `version-sync` output is flat), or pass `-force-layout` to take the source's layout on purpose.

If `~/sam2-training` still has flat `RC1` `RC2` `RPC1` `RPC2` and no version directories yet, move them aside once before the first sync:

```bash
mkdir -p ~/sam2-training/v1
mv ~/sam2-training/RC1 ~/sam2-training/RC2 \
   ~/sam2-training/RPC1 ~/sam2-training/RPC2 ~/sam2-training/v1/
```

SAM2 is trained class-agnostic. Viking labels such as `MC` / `ConePR` are kept only for split reports and eval tables.

### Prompts in training

Each training sample is encoded once and decoded with `prompt.variants_per_sample` (3) prompt variants against the same ground-truth polygon (`prompt:` in `em_finetune.yaml`; the shares must sum to 1.0). Variant 1 is always the production prompt, a circle seed (once adjacent-section shapes exist, an adjacent seed with probability `prob_adjacent_seed / (prob_adjacent_seed + prob_circle_seed)`); the others are drawn from the mix without replacement. **The shares below therefore steer variants 2 and 3 only**: the realised mix has more circle seeds than the table says. TensorBoard logs it per epoch as `train/prompt_mix/<family>` (share) and `train/prompt_count/<family>` (count). The loss averages a sample's variants first. Variant `k` is seeded from `(seed, epoch, index, k)`, so a resumed epoch replays exactly. `prompt.enabled: false` goes back to one centroid point. The full ground-truth mask is never a prompt: that is a copy task. The design is in `docs/mask_refinement_plan.md`.

| Family | Share | What it is |
| --- | --- | --- |
| Points only | 10% | 1-9 positive points |
| Box only | 5% | jittered box |
| Box and points | 5% | box plus 1-3 points |
| Circle seed | 40% | inscribed circle x `circle_diameter_scale`, plus its center point (production; circle annotations are never seeds) |
| Adjacent-section seed | 15% | linked polygon from z +-1-2 scaled to fit inside the cell grown by 10 px (the annotator fits it to the visible membrane; needs the exporter's `shapes.jsonl`, not written yet; its share moves to the circle seed) |
| Shrunk or cut seed | 10% | a subset of the truth: eroded 10-40 px, then optionally cut by a straight line or along the cell from an interior anchor |
| Mask to correct | 10% | the truth grown 5-50 px, scaled 0.8-1.25, wobbled (mean 10-20 px, local excursions up to 50 px), simplified then grown or shrunk by up to 30 px, or merged with part of a neighbor |
| Near-correct mask | 5% | the truth as a good human trace would draw it: outline wobble of 5-12 px on average plus 1-3 local bulges or dents of 15-30 px (never more than 30 px off); the model must leave it at least as good. Its click comes from this mask, as in production |

- **Seeds and masks to correct:** a seed is trusted inside the cell and the engine grows beyond it (`data/refine.py`); a mask to correct is a suggestion that may be wrong, fed at a lower strength. Only seeds feed the mask-containment loss.
- **Edge confidence:** traces are about 9 px (20 nm) from the membrane on average, interiors are trusted. Every dense value is scaled by `c(d) = 1 - exp(-d / tau_eff)`, `d` the pixel distance to the mask's edge, `tau_eff = min(prompt.edge_tau_px, prompt.edge_tau_depth_frac * D)` with `D` the mask's maximum depth, computed per mask (target, each seed, each neighbor). With `edge_tau_px` 8 that is 95% about 24 px inside the line, and a process 20 px wide gets `tau_eff` about 2.5 so it is not all edge. The array border is not an edge, so a mask cut by the tile stays confident there. Excursions of 30-50 px are not covered by any symmetric band; they stay as sparse label noise.
- **Fully masked tiles:** where the target covers at least `data.full_tile_threshold` (0.99) of the tile, training keeps a deterministic `data.full_tile_keep_fraction` (0.25) of the tiles per location and gives them only point or circle-mask prompts. Production still meets such tiles, so they are not removed.
- **Corrective clicks (`iterative:`):** `iterative.batch_fraction` (0.2) of the batches predict, sample a click from where the prediction is wrong (a missed area gives a positive click, a spurious one a negative click), and re-run with the previous low-resolution logits plus the click, `rounds_min` to `rounds_max` times. Only the last pass has gradients, and the previous logits are fed back for `prev_mask_prob` of the samples (`sam2_segmentation_trainer/interactive.py`). A ground-truth mask is never the prompt.
- **Circle annotations as negatives (`edge-confidence-v6`):** a circle annotation is encoded by its radius-relative trust, not by the polygon edge curve: with `rho = |p - c| / r_true`, `-seed_logit * w(rho)`, `w = 1` for `rho <= full_frac`, `0` for `rho >= zero_frac`, a half cosine between (`prompt.circle_annotation_trust`, validated `0 <= full_frac < zero_frac`). It is computed analytically inside the circle's bounding box (about 5x faster than rasterizing and running a distance transform; `scripts/benchmark_circle_negatives.py`). The region it claims, used by the overlap rule, is `rho <= zero_frac`, the same pixels as the old 0.95-scaled raster. Its *core* is `rho <= full_frac` (a polygon's stays 10 px inside its edge): the leak weight, the metric cores and the circle's background points use it. Polygon negatives and seed circles are unchanged.
- **Circle negatives over the target:** a neighbor's circle that reaches into the target contradicts the ground truth. `sam2-em-validate --circle-overlap 2000` measures how often; `negative.drop_overlapping_circles` leaves such circles out. The measurement on the four volumes was 2.4% of examples with circle negatives (40 of 1637), so it is on by default.
- **IoU head:** trained with absolute (L1) error as in SAM 2. It is the only form: the old `loss.iou_use_l1` switch and the squared-error path are gone.
- **Prompt mix log:** each epoch prints the realized mix and writes `train/prompt_mix/<type>`, so a missing circle file or a dropped type shows at once.

### Validation

The goal is to identify cells and their connectivity: at least 97.5% of each cell's interior captured, no merges into neighbors. The validation split (see the layout section) is scored once per prompt in `eval.val_prompts`, each with a fixed, reproducible prompt, and every family reports the goal metrics. All of them ignore a band of `val.core_band_px` (10) around the traced edge, because the trace itself is about 9 px from the membrane:

| Metric | What it is |
| --- | --- |
| Core recall | the share of ground-truth pixels at least 10 px inside the edge that are predicted. Reported pooled, as the per-sample 10th percentile, and as the share of samples at or above 97.5% |
| Leak | the share of predicted pixels inside a neighbor's core (at least 10 px inside that neighbor) |
| Merge rate | the share of samples whose prediction covers at least 5% of any one neighbor's core |
| Spill | predicted area more than 10 px outside the ground truth, divided by the ground-truth area (leaks into cells nobody annotated) |
| Fragmentation | the share of samples whose prediction has more components (of at least 200 px) than the ground truth |

| Prompt | What it checks |
| --- | --- |
| `circle_mask` | the inscribed circle as a seed (what production starts from); the primary family |
| `correction` | a mask to correct; the input mask's metrics and the change from them are reported too |
| `near_correct` | a good human trace; "do no harm" is the share of samples whose core recall drops at most 0.5 points and whose spill rises at most 0.5% of the ground truth |
| `point` | one point at the deepest interior pixel |
| `centroid` | the original prompt (falls outside C-shaped cells in 2-6% of masks): raw IoU, for continuity with earlier runs |
| `circle_mask_noneg` | the circle seed with every negative region and background point removed: how much the model leans on neighbors being sent. A diagnostic, never used for selection. `circle_mask` and this family are also reported split by `with_neg` / `no_neg` (whether the sample's tile has other annotations) under `val_*_neg/...` |
| `circle_annotator` | an annotator-like circle seed: a random centre depth and radius (`prompt.circle_seed_jitter`: `depth_range`, `scale_range`, `fit_grow_px`, `min_radius_px`; `prob` is ignored here) instead of the inscribed circle x 0.95, with negatives kept and the same `with_neg` / `no_neg` split. Production seeds are drawn by annotators, so this is the harder, truer circle. A diagnostic; it never selects a checkpoint. Scored in the per-epoch subset, the full pass and `sam2-em-eval --goal` (not by the one-prompt-at-a-time evaluation). See `docs/circle_annotator_spec.md`. |

Tolerant IoU (ignoring pixels whose loss weight is below 0.5) and boundary F1 at 10 and 20 px are kept as diagnostics only; their values are not comparable with `val/iou` of earlier runs. TensorBoard gets `val/<metric>/<prompt>` for `core_recall`, `core_recall_p10`, `core_recall_ok`, `leak`, `merge_rate`, `spill` and `frag`; for the refinement prompts also `val/input/<metric>/<prompt>` and `val/delta_<metric>/<prompt>`; `val/no_harm/near_correct`; and the diagnostics `val/iou/<prompt>`, `val/raw_iou/<prompt>`, `val/bf10/<prompt>`, `val/bf20/<prompt>`, `val_iou/<prompt>/<volume>` and `val_iou_cat/<prompt>/<category>`. Fully masked tiles are scored apart as `val/iou_fulltile/<prompt>` and left out of the headline.

- **Approved masks:** `data.approved_path` points at the gallery's `approved.json` (a list of `{"location_id", "image_key"}` entries). Those masks are reported as their own row (`val_approved/<metric>/<prompt>`, count in `val_approved/n/<prompt>`), how many there are per volume is printed at start-up, and `data.approved_weight` (1.0) is their training weight. A missing file warns and skips the row. Keep a copy next to the export so runs are reproducible.
- **Checkpoint and early stopping:** among epochs whose pooled core recall on the primary prompt is at least `val.core_recall_target` (0.975), `best_model.pt` is the one with the lowest merge rate, spill breaking ties; if none reaches it, the highest core recall. A change smaller than the noise floor does not count as better: at the target the merge rate must fall by at least `training.early_stopping_min_delta` (0.002), or, if it moved by less than that, the spill must; below the target the core recall must rise by at least 0.0005; crossing the target always counts. The comparison is against the best epoch so far, so many tiny gains add up. `training.early_stopping_patience` (2) epochs without a better one stop the run. The numbers that decide are the per-epoch subset's.
- **Cost and the subset:** one validation pass loads each sample once, runs the image encoder once and decodes every family from that embedding (batches of `val.batch_size` images, so `batch_size x families` prompts), with the metrics in `val.metric_threads` threads while the GPU does the next batch. Each family's prompt is seeded from the run seed, the sample (volume, image key, location id) and the family, so results do not depend on batch composition, family order or which samples are in the pass. Every epoch validates a fixed subset (`val.epoch_subset`, 25% of each volume, drawn once from the run seed, stratified by volume, always holding every Approved sample) without the legacy `centroid` family. The full split, centroid included, runs every `val.full_every_n_epochs` (3) epochs and after the last (and before an early stop). TensorBoard keeps them apart: `val_subset/...` and `val_full/...` (the old `val/iou` follows the subset). `sam2-em-train -benchmark` times both passes and projects the run.

- **Box:** a hint that the object is most likely inside (each edge jittered outward by up to `box_jitter_frac`). Nothing penalizes predictions that leave the box.
- **Negative context:** every other annotation in the tile (neighbor masks, plus circles that have no mask) is a negative area of the dense mask, scaled by that neighbor's own edge confidence, and, with probability `negative.bg_point_prob_*`, becomes 1-9 background points. A neighbor merged into a mask to correct is left out, otherwise the prompt would label the error. With no negatives and no dense foreground the dense prompt is omitted so the encoder uses its learned no-mask embedding, not an all-zero mask. Training variants after the first sometimes lose their negative context (`negative.context_dropout` 0.2: none at all; `negative.region_dropout` 0.2: each neighbor on its own), so the model does not depend on neighbors always being sent; variant 0 and validation never drop, and the loss weights still see every neighbor.
- **Dense mask scale:** `annotation_crops/prompt_encoding.py` builds the 256x256 `mask_input`: seeds at `+prompt.seed_logit` (10) x c(d), masks to correct at `+prompt.correction_logit` (3) x c(d), negatives at -10 x c(d), unknown 0; where a positive and a negative region overlap the pixel is 0 (decided by the masks, not the values), overlapping positives take the larger value and overlapping negatives the strongest (no sum, no clip); the canvas is reduced by area averaging. The segmentation server does not yet use this module; when it builds `mask_input` it must use the same function and constants.
- **Losses:** labels stay hard 0/1. `loss.edge_tau_px` (8) weights each pixel by `w(d) = max(loss.edge_weight_min, 1 - exp(-d / tau_eff))`, `d` the distance to the ground-truth edge on both sides and `tau_eff` capped by the target's own depth, in the focal, dice and IoU-head terms (the IoU target ignores pixels with weight below 0.5); 0 turns the edge weighting off. **Leak penalty:** `loss.leak_weight` (3, 1 turns it off) multiplies the focal and dice weight of pixels at least `val.core_band_px` inside another annotation and outside the target, neighbor-merge samples included, because a merge breaks connectivity. `loss.mask_contain_weight` penalizes predicting background where a seed said foreground, weighted by the seed's confidence, so it acts on the trusted core (zero when the model respects it; ramps up over `loss.containment_warmup_epochs`). It counts only pixels inside the ground-truth target: where an annotator-like circle seed reaches past the outline the weight there is zero (the seed in the prompt is not clipped). It never applies to masks to correct. There is no box counterpart, and no hard clamp on box exteriors.
- **Prompt encoder:** still frozen except its `mask_downscaling` network (`model.unfreeze_mask_cnn`), which has to learn the mask semantics. A run cannot resume across changing this; use `-refresh`.
- **Removed:** `prompt.prob_full_mask`, `prob_negative_only`, `prob_circle_mask`, `prob_corrupted_mask`, `negative.allow_circle_targets` and `negative.circle_target_fraction`. A run started before this change should be restarted with `-refresh`: its mask network was trained on the old +-1 encoding.

## Train / eval / infer

After `pip install -e .` (the Dev Container postAttach already does this, and installs pytest; `pip install -e ".[dev]"` does too):

```bash
# Index check (add --check-json for area/category stats; slow on CIFS)
sam2-em-validate --json-report /outputs/validate.json

python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

# Fine-tune sam2.1_hiera_large (prompt encoder frozen except its mask network).
# Restart continues the run with the newest last.pt (for example sam2_em_rpc1).
# -refresh starts the next sam2_em_vN from pretrained weights.
# Each volume keeps every usable sample; ~10% of it (5 evenly spaced z-slabs) is validation.
sam2-em-train
# Pin a version: sam2-em-train data.root=/data-local/v1b
# Overrides: sam2-em-train training.num_epochs=8 training.batch_size=2

# Crash / Ctrl+C: rerun the same command. It loads last.pt (or last.pt.bak) and continues.
# Snapshots every 200 optimizer steps plus epoch end. Fresh start from pretrained:
sam2-em-train -refresh

# Time ~50 train batches + 20 val forwards (plus warmup), then exit. Does not load or write last.pt.
# It prints the step time, the data-loader wait (data-bound or GPU-bound), the peak CUDA memory of the timed steps
# and the seconds per sample of a validation subset pass and of a full pass, with the projected run.
# Keep the outputs in benchmarks/ (git-ignored): sam2-em-train -benchmark 2>&1 | tee benchmarks/v3_bs4.txt
sam2-em-train -benchmark
sam2-em-train -benchmark --benchmark-steps 80 --benchmark-val-steps 30
# Training steps only (no validation timing): sam2-em-train -benchmark --benchmark-no-val
# Compare two configurations A, B, A, B with the GPU clocks, temperature, power and throttle reasons logged
# (nvidia-smi every 5 s) and the other GPU processes noted; prints a table when done:
#   bash scripts/compare_throughput.sh          # results in benchmarks/throughput/
#   python scripts/summarize_throughput.py benchmarks/throughput
# Try a setting without editing the config: prompt variants, batch size, activation checkpointing, cuDNN autotuner.
sam2-em-train -benchmark prompt.variants_per_sample=2
# The defaults are batch 2 x 8 accumulation with activation checkpointing off (see em_finetune.yaml for the
# measurements). On a smaller card: checkpointing on, or only a share of the blocks, at batch 4.
sam2-em-train -benchmark model.activation_checkpointing=true training.batch_size=4 training.gradient_accumulation_steps=4
sam2-em-train -benchmark model.activation_checkpointing=true model.activation_checkpointing_fraction=0.5 training.batch_size=4 training.gradient_accumulation_steps=4

# TensorBoard starts with the container (cursor-dev-entry.sh) on :6006.
# Compose maps host 8060 -> 6006. Events under /storage4/Sam2Trainer/runs/<run>/tb/.
# Open http://localhost:8060  (no need to start tensorboard by hand)
# PYTORCH_PROFILER reads torch.profiler traces from that same tb/ folder.
# Training records one short window per process (training.profiler in em_finetune.yaml).
# The trace file shows up after skip_first+wait+warmup+active batches (default 10).
# sam2-em-train training.profiler.enabled=false

# Annotation gallery is the other Compose service. It reads the same
# /data-local/current tree. Trash moves a mask from masks/ to ignored/.
# A running sam2-em-train rebuilds its inventory at the next epoch when
# ignore.json is newer than the last read, or when a listed tile is missing.
# TEM gallery http://127.0.0.1:8081, SEM gallery http://127.0.0.1:8082 (plain HTTP, debugging; sign-in needs the HTTPS names).
# Public: https://tem.gallery.codepharm.net, https://sem.gallery.codepharm.net; https://gallery.codepharm.net redirects to
# the modality the trainer runs. See "Gallery hostnames, certificate and access" below.
# Reopen in Container starts it. From the CLI:
#   docker compose --env-file docker/.env -f docker/compose.cursor-dev.yaml up -d annotation-gallery-tem annotation-gallery-sem gallery-router

# Live progress is also published to the Nornir build dashboard when its Mosquitto
# broker is up (nornir-docker/start-dashboard.ps1). The container uses
# NORNIR_MQTT_HOST=host.docker.internal and NORNIR_MQTT_PORT=1883. Open
# http://127.0.0.1:8087. Each run_name (sam2_em_vN) is one dashboard row:
# epoch and step bars, loss and learning rate on the step bar, val IoU in the log.
# Ctrl+C is status stopped. NORNIR_MQTT_ENABLE=0 leaves progress on the console.

sam2-em-eval --checkpoint /storage4/Sam2Trainer/runs/sam2_em_v1/best_model.pt --out /outputs/eval
# Scores every validation prompt; add --noc for clicks to reach 90% IoU, and --split <run>/split.json to
# score the run's own validation split. The zero-shot baseline to beat (base weights only):
sam2-em-eval --zero-shot --split /storage4/Sam2Trainer/runs/sam2_em_v1/split.json --out /outputs/eval_zero_shot
# Goal metrics of every prompt family on the FULL validation split (what a run logs as val_full), for any model: the
# baselines a new run must beat. --split may name a run's split.json that does not exist yet; it is then created from
# the config, and the run reuses it.
sam2-em-eval --goal --zero-shot --split /storage4/Sam2Trainer/runs/sam2_em_v4/split.json --out /outputs/goal_zero_shot
sam2-em-eval --goal --checkpoint /storage4/Sam2Trainer/runs/sam2_em_v3/best_model.pt --split /storage4/Sam2Trainer/runs/sam2_em_v4/split.json --out /outputs/goal_v3
sam2-em-infer --image /path/to/crop.png --checkpoint /storage4/Sam2Trainer/runs/sam2_em_v1/best_model.pt --point 512,400 --out /outputs/pred.png

# SegmentationServer loads Meta {"model": state_dict}, not the raw trainer file.
# Training does this itself after every epoch (training.export_serve_each_epoch, default true): it starts a separate
# detached CPU process that writes <run>/serve/serve_epoch<N>.pt and its .prompt_encoding.json (log: serve/export.log).
# Copy that file to the segmentation server, together with its <stem>.model.json (the model manifest: model id and
# version, the training data it saw, tiled or whole-screen, the mask-prompt encoding, and what a client must do to its
# masks; see docs/model_manifest.md). The server reads the manifest next to the checkpoint and reports it to clients.
# training_data.json in the run folder records which data folder (v3, ...) and which file fingerprints the run saw;
# the manifest carries it as training_data, separate from the model's own version. v2 and v3 predate the exporter and
# need hand-written manifests (examples in docs/model_manifest_examples/, checked with
# `python -m sam2_segmentation_trainer.model_manifest check <file>`).
# The command below exports best_model.pt by hand. Name the file for the model: v<VERSION>.<EPOCH>.pt, here v1.6.pt
# for run sam2_em_v1 and epoch 6 (VERSION comes from the run name, EPOCH from best_model.json that training writes
# beside best_model.pt; for an older run pass --epoch N, and --model-id v<N> when the run name has no _vN).
# --description "..." sets the operator's description; leave it out to keep the one already in the manifest.
# best_model.pt is weights only; this writes the file the server expects.
sam2-em-export-serve \
  --checkpoint /storage4/Sam2Trainer/runs/sam2_em_v1/best_model.pt \
  --out /path/to/v1.6.pt
# The file is {"model": state_dict, "prompt_encoding": {...}}: build_sam2 reads only "model"; the other key records
# EDGE_TAU_PX, EDGE_TAU_DEPTH_FRAC, SEED_LOGIT, CORRECTION_LOGIT, LOGIT_CLIP, LOW_RES, circle_diameter_scale,
# circle_annotation_trust {full_frac, zero_frac, ramp: cosine} and a prompt_encoding version string
# (edge-confidence-v6). The same dict is written as <weights stem>.prompt_encoding.json next to the file, and
# training writes prompt_encoding.json into the run folder at start, so a raw best_model.pt has its record too;
# sam2-em-eval and sam2-em-validate read it back (a command-line override is logged). The server must read
# full_frac and zero_frac from it, not assume the defaults. The values come from the run's own saved config.yaml (next to --checkpoint, or
# --run-dir); without one the library defaults are written with a warning. Training prints a warning at start when
# a prompt.* value differs from the annotation_crops.prompt_encoding constants the server implements.
```

**Do not serve a v4 model with a segmentation server that builds `mask_input` the old way.** The server in the Viking
repo (`Servers/SegmentationServer`) reads the model manifest and builds `mask_input` with
`annotation_crops/prompt_encoding.py` and the constants in the exported file when the manifest names
`edge-confidence-v6`; an older build does not. The mask network was trained on edge-confidence canvases (a seed at +10 times the
edge confidence, a mask to correct at +3, other annotations at -10, area-averaged to 256 x 256), not on the mask
encoding of earlier models. A server that still builds `mask_input` the old way gives the model a differently scaled
input than it saw in training. Check the `prompt_encoding` version and constants against the server's before switching.

Unit tests (no GPU, no NAS):

```bash
python -m unittest discover -s tests -t /workspace -v
```

Historical implementer brief: `docs/SAM2 Fine-Tuning Plan for TEM SEM Electron Microscopy.md`. The trainer follows that brief with the confirmed API/data fixes (2.1 config, ImageNet normalize, prompt coords via SAM2 scale, validation prompts are fixed per mode (circle, interior point, centroid), custom image loop rather than the MOSE video trainer).

## Verify

```bash
python -c "import torch, sam2; print(torch.__version__, torch.cuda.is_available()); print(sam2.__file__)"
echo "$NORNIR_NET_MOUNTS"          # expect 1
findmnt -no FSTYPE,SOURCE /storage4
ls /storage4 | head
pip show sam2-segmentation-trainer
```

## Gallery hostnames, certificate and access

- `tem.gallery.codepharm.net` and `sem.gallery.codepharm.net` are two gallery containers (`annotation-gallery-tem`, `-sem`) on the
  compose network. `gallery-router` (nginx, `docker/gallery-router/gallery.conf`) terminates TLS for both and for
  `gallery.codepharm.net`, which 302-redirects to the modality the trainer runs. `run-cursor-dev.ps1` / `docker-build.ps1`
  write that choice (`active.conf` in `SAM2_GALLERY_ROUTER_DIR`) and reload the router. Ports: router HTTP 8080 / HTTPS 8442,
  TEM gallery 8081, SEM gallery 8082. The reverse proxy's stream map sends all three names to 8442; plain HTTP for the tem. and sem.
  names is a redirect in the proxy's `custom/http.conf`.
- One certificate (lineage `gallery.codepharm.net`, SANs for all three names) from `annotation-gallery-certbot-renewer`, which restarts
  `gallery-router` after a renewal. The renewer only requests extra names for a new lineage. To widen an existing one, run once:
  `docker exec annotation-gallery-certbot-renewer certbot certonly --non-interactive --agree-tos --dns-cloudflare --dns-cloudflare-credentials /tmp/cloudflare.ini --cert-name gallery.codepharm.net -d gallery.codepharm.net -d tem.gallery.codepharm.net -d sem.gallery.codepharm.net --expand --email <address>`.
- Sign-in: `GALLERY_IDENTITY_MODE=oidc` (default). Anonymous visitors are sent to Identity Server. Register one OIDC client per host
  (redirect `https://<tem|sem>.gallery.codepharm.net/callback`) and set `GALLERY_TEM_OIDC_CLIENT_ID/_SECRET` and
  `GALLERY_SEM_OIDC_CLIENT_ID/_SECRET` in the shared `.env`. Read permission on a volume's annotation database lets a user view
  it; the permission named by `GALLERY_REVIEW_PERMISSION` (default `Annotate`) allows refresh, accept, reject and the other writes.
- Grants are matched by annotation database name (`annotationDb` in `source.json`; else the OData URL's `<db>`; else the folder name),
  because SEM folders are not named after their database. `scripts/NAS/backfill-annotation-db [--write]` adds the field to
  existing datasets without a re-export. The exporter should write it from now on.
- AI crawlers: `robots.txt` disallows everything, every response carries `X-Robots-Tag: noindex, noai`, and known AI user agents
  get 403 (in the gallery and in the router). The login is what actually keeps scrapers out.
