# SAM2 Segmentation Trainer

GPU Dev Container for fine-tuning Meta SAM2 on TEM/SEM data. The trainer package is bind-mounted at `/workspace` and installed editable. Training data is the NAS share already used by Nornir (`/storage4` via in-container CIFS). Nornir Python packages are **not** installed.

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
| `SAM2_DATA_ROOT` | `/data-local` (fallback `/storage4`) | Local NVMe or NAS volumes |
| `SAM2_OUTPUT_ROOT` | `/outputs` | Splits, pretrained download, TB pid/log |
| `SAM2_CHECKPOINT_ROOT` | `/storage4/Sam2Trainer` | Run dir: `last.pt`, `best_model.pt`, epoch snaps, TB events (CIFS) |
| workspace | `/workspace` | This git checkout (editable install) |
| SAM2 source | `/opt/sam2` | Pinned `facebookresearch/sam2` |
| TensorBoard | host `8060` → container `6006` | Auto-started; reads `$SAM2_CHECKPOINT_ROOT/runs` |

If the 32k dataset is not under `/storage4`, add a row to `D:\Docker\Run\nornir-dev\net-mounts\nas-mounts.tsv` (no image rebuild).

## AnnotationCrops layout

Training reads RC1, RC2, RPC1, and RPC2 under `$SAM2_DATA_ROOT` (default `/storage4`). Each volume uses `{volume}/AnnotationCrops`:

- `manifest.jsonl` — index (`image` or `jpeg`, `json`, `imageKey`, `locationIds`, `downsample`, `volume`, `z`)
- `images/` — tile PNG/JPEG plus COCO-RLE sidecar JSON
- `masks/` — optional raster `{imageKey}_{locationId}.png` (used when present; otherwise RLE in the JSON)
- `overlays/` — viewing only, not used for training

One tile image can hold several `locationId`s (disk-efficient shared crops). The loader expands those to one training example per annotation. Every annotation is kept at its native downsample (D1–D128). Each new run stores its own `split.json` and `inputs.json` (volumes, per-volume train/val counts, data root) under the run directory. New runs draw the same number of train and val samples from each volume.

This is **not** Pascal VOC (`JPEGImages` / `Annotations` / `ImageSets`). Do not reshape the NAS.

SAM2 is trained class-agnostic (binary mask + point prompt). Viking labels such as `MC` / `ConePR` are kept only for split reports and eval tables.

## Train / eval / infer

After `pip install -e .` (the Dev Container postAttach already does this):

```bash
# Index check (add --check-json for area/category stats; slow on CIFS)
sam2-em-validate --json-report /outputs/validate.json

python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

# Fine-tune sam2.1_hiera_large (prompt encoder frozen).
# A new launch picks the next run folder (sam2_em_v2, v3, ...) and writes
# /storage4/Sam2Trainer/runs/<name>/ with split.json and inputs.json.
# Train/val counts are equal across RC1, RC2, RPC1, and RPC2.
# Resume an older run by name: sam2-em-train training.run_name=sam2_em_v1
sam2-em-train data.root=/data-local
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

sam2-em-eval --checkpoint /storage4/Sam2Trainer/runs/sam2_em_v1/best_model.pt --out /outputs/eval
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

Historical implementer brief: `docs/SAM2 Fine-Tuning Plan for TEM SEM Electron Microscopy.md`. The trainer follows that brief with the confirmed API/data fixes (2.1 config, ImageNet normalize, prompt coords via SAM2 scale, val uses GT centroid, custom image loop rather than the MOSE video trainer).

## Verify

```bash
python -c "import torch, sam2; print(torch.__version__, torch.cuda.is_available()); print(sam2.__file__)"
echo "$NORNIR_NET_MOUNTS"          # expect 1
findmnt -no FSTYPE,SOURCE /storage4
ls /storage4 | head
pip show sam2-segmentation-trainer
```
