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
./build.ps1
```

A successful build retags `sam2-trainer:dev` and **recreates** a running Compose `cursor-dev` (the Dev Container) onto that new image ID. `docker build` alone would leave the old container running. Use `./build.ps1 -SkipRecreate` to tag only, or `./build.ps1 -RecreateOnly` to replace the running container without rebuilding. If Cursor is attached, wait for reconnect or **Reopen in Container**.

Optional `--build-arg` files in that folder (Nornir names; the `example.*` template is not merged):

- `build.env`
- `.build.sam2-trainer-dev.env` (tag `sam2-trainer:dev`)

```powershell
Copy-Item D:\src\git\Sam2SegmentationTrainer\docker\example.sam2-trainer-dev.build.env `
    D:\Docker\Builds\sam2-trainer\.build.sam2-trainer-dev.env
```

## Run / Dev Container

Open **this folder** (not a multi-root workspace) and **Reopen in Container**. Compose loads `docker/compose.cursor-dev.yaml` plus `D:\Docker\Run\sam2-dev\compose.net-mounts.override.yaml`.

CLI:

```powershell
& D:\src\git\Sam2SegmentationTrainer\docker\run-cursor-dev.ps1
```

## Paths inside the container

| Variable | Typical value | Role |
| --- | --- | --- |
| `SAM2_DATA_ROOT` | `/data-local/current` (symlink to a version; fallback `/storage4`) | Local NVMe or NAS volumes |
| `SAM2_OUTPUT_ROOT` | `/outputs` | Splits, pretrained download, TB pid/log |
| `SAM2_CHECKPOINT_ROOT` | `/storage4/Sam2Trainer` | Run dir: `last.pt`, `best_model.pt`, epoch snaps, TB events (CIFS) |
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
- `circle_annotations.json` — optional. Circle annotations keyed by `imageKey`, in that crop's pixel space: `location_id`, `center_x`, `center_y`, `radius` (the true radius, since the exporter records `diameter_scale` 1.0; a file from an older exporter has its scale baked in), `structure_label`, `has_polygon_mask`, and `source` (`annotated` for a location that is still a circle, `inscribed` for the largest circle inside a polygon). Written by the nornir exporter, which no longer takes `-CircleDiameterScale`. The trainer shrinks the radii to `prompt.circle_diameter_scale` (0.95) when it loads them, rescaling by that value over the file's own `diameter_scale`, so changing the scale needs no re-export. `sam2-em-eval` has `--circle-scale` for the same value. Training needs it while circle-mask prompts are on: with `prompt.prob_circle_mask > 0` a volume without a usable file stops the run (`data.require_circle_file`, set `false` to warn only). Crops whose key has a negative origin (`X-512-512`) are read correctly by the exporter since the key decoder accepts them.

One tile image can hold several `locationId`s (disk-efficient shared crops). The loader expands those to one training example per annotation. Every annotation is kept at its native downsample (D1–D128). Each new run stores its own `split.json` and `inputs.json` (volumes, per-volume train/val counts, data root) under the run directory. Every usable sample is kept. Validation is split by section (`data.split_mode: zslab`): within each volume, five contiguous runs of sections spaced evenly through the volume hold about 10% of its samples, and one section on each side of every run (`data.zslab.guard_sections`) is used by neither side. All crop windows of a section land on one side, so adjacent sections and overlapping windows never cross the split. `data.split_mode=location` restores the original random 90/10 over locations. An existing run keeps the split in its `split.json`; use `-refresh` to get the new one.

This is **not** Pascal VOC (`JPEGImages` / `Annotations` / `ImageSets`). Do not reshape the NAS.

## Local version mirrors

On the WSL host, `~/sam2-training` is mounted at `/data-local`. Source trees live under `/mnt/d/TrainingData/<ver>` (`v1`, `v1b`, `v2`, …). Sync from that host, not inside the container:

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
TRAINING_DATA_SRC=/mnt/y/SAM2TrainingData scripts/WSL/sync_training_data.sh -noselect v2
# Only retarget current (no copy). No arg picks the highest local version.
scripts/WSL/select_training_data.sh v2
```

`current` is a relative symlink (`v2`). Restart training after it changes.

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

Each training sample draws one prompt type (`prompt:` in `em_finetune.yaml`; they must sum to 1.0): points only (15%), box only (5%), box and points (10%), circle mask (45%), corrupted mask (25%). Full mask and negative only are available but set to 0. `prompt.enabled: false` goes back to one centroid point.

- **Circle mask:** the annotation circle is the dense mask prompt (+1), one foreground point sits at its center, and the loss compares to the full polygon. The model learns to grow from the circle to the boundary.
- **Corrupted mask:** the ground-truth mask degraded the way hand painting is (grown or shrunk by up to `corrupt_max_shift_px`, partly painted with strokes, or traced as a coarse polygon; `data/corrupt.py`), plus one interior point. It does not feed the mask-containment loss because it is deliberately wrong in places.
- **Full mask (off):** the whole ground-truth mask as the prompt is a copy task: the model can lower the loss by echoing its input. Kept for experiments.
- **Negative only (off):** no foreground prompt; background points and the negative mask only. With more than one unannotated cell in a tile nothing says which one is the target, so it trains guessing. `negative.allow_circle_targets` stays false.
- **Fully masked tiles:** where the target covers at least `data.full_tile_threshold` (0.99) of the tile, training keeps a deterministic `data.full_tile_keep_fraction` (0.25) of the tiles per location and gives them only point or circle-mask prompts. Production still meets such tiles, so they are not removed.
- **Corrective clicks (`iterative:`):** `iterative.batch_fraction` (0.2) of the batches predict, sample a click from where the prediction is wrong (a missed area gives a positive click, a spurious one a negative click), and re-run with the previous low-resolution logits plus the click, `rounds_min` to `rounds_max` times. Only the last pass has gradients, and the previous logits are fed back for `prev_mask_prob` of the samples (`sam2_segmentation_trainer/interactive.py`). A ground-truth mask is never the prompt.
- **Circle negatives over the target:** a neighbor's circle that reaches into the target contradicts the ground truth. `sam2-em-validate --circle-overlap 2000` measures how often; `negative.drop_overlapping_circles` leaves such circles out (off by default).
- **IoU head:** trained with L1 as in SAM 2 (`loss.iou_use_l1`, false for the old squared error).
- **Prompt mix log:** each epoch prints the realized mix and writes `train/prompt_mix/<type>`, so a missing circle file or a dropped type shows at once.

### Validation

The validation split (see the layout section) is scored once per prompt in `eval.val_prompts`, each with a fixed, reproducible prompt: `circle_mask` (the inscribed circle, what production starts from), `point` (the deepest interior pixel) and `centroid` (the original prompt; it falls outside C-shaped cells in 2-6% of masks). `eval.primary_val_prompt` (`circle_mask`) picks `best_model.pt` and drives early stopping; `val/iou` follows it. TensorBoard also gets `val/iou/<prompt>`, `val/bf/<prompt>` (boundary F1), `val_iou/<prompt>/<volume>` and `val_iou_cat/<prompt>/<category>`. Fully masked tiles are scored apart as `val/iou_fulltile/<prompt>` and left out of the headline number.
- **Box:** a hint that the object is most likely inside (each edge jittered outward by up to `box_jitter_frac`). Nothing penalizes predictions that leave the box.
- **Negative context:** every other annotation in the tile (neighbor masks, plus circles that have no mask) is a -1 area of the dense mask and, with probability `negative.bg_point_prob_*`, becomes 1-9 background points. Where a +1 area and a -1 area overlap the value is 0 (unknown). With no negatives and no foreground mask the dense prompt is omitted so the encoder uses its learned no-mask embedding, not an all-zero mask.
- **Dense mask scale:** the signed mask is downsampled (nearest) to the 256x256 `mask_input` and fed as logits of +/-10 (`data/prompts.py` `MASK_PROMPT_LOGIT`). The segmentation server does not yet turn a 1-bit mask Feature into `mask_input`; when it does it must use the same values.
- **Losses:** `loss.mask_contain_weight` penalizes predicting background where a mask prompt said foreground (zero when the model respects it; ramps up over `loss.containment_warmup_epochs`). There is no box counterpart, and no hard clamp on box exteriors.
- **Prompt encoder:** still frozen except its `mask_downscaling` network (`model.unfreeze_mask_cnn`), which has to learn the mask semantics. A run cannot resume across changing this; use `-refresh`.
- **Circle-only targets:** `negative.allow_circle_targets` also trains the negative-only prompt on circles that have no mask (the disc is the target), capped at `negative.circle_target_fraction` of the mask-bearing examples.

## Train / eval / infer

After `pip install -e .` (the Dev Container postAttach already does this):

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
sam2-em-train -benchmark
sam2-em-train -benchmark --benchmark-steps 80 --benchmark-val-steps 30

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
# HTTP http://127.0.0.1:8080
# HTTPS https://127.0.0.1:8443 when SSL_CERT_PATH and SSL_KEY_PATH exist.
# Reopen in Container starts it. From the CLI:
#   docker compose --env-file docker/.env -f docker/compose.cursor-dev.yaml up -d annotation-gallery

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
sam2-em-infer --image /path/to/crop.png --checkpoint /storage4/Sam2Trainer/runs/sam2_em_v1/best_model.pt --point 512,400 --out /outputs/pred.png

# SegmentationServer loads Meta {"model": state_dict}, not the raw trainer file.
# best_model.pt is weights only; this writes the file the server expects.
sam2-em-export-serve \
  --checkpoint /storage4/Sam2Trainer/runs/sam2_em_v1/best_model.pt \
  --out /path/to/best_TEM_model.pt
```

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
