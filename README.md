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
| `SAM2_DATA_ROOT` | `/storage4` | NAS volumes (`//192.168.0.199/Data/Volumes`) |
| `SAM2_OUTPUT_ROOT` | `/outputs` | Checkpoints, TensorBoard, run logs |
| workspace | `/workspace` | This git checkout (editable install) |
| SAM2 source | `/opt/sam2` | Pinned `facebookresearch/sam2` |

If the 32k dataset is not under `/storage4`, add a row to `D:\Docker\Run\nornir-dev\net-mounts\nas-mounts.tsv` (no image rebuild).

## Verify

```bash
python -c "import torch, sam2; print(torch.__version__, torch.cuda.is_available()); print(sam2.__file__)"
echo "$NORNIR_NET_MOUNTS"          # expect 1
findmnt -no FSTYPE,SOURCE /storage4
ls /storage4 | head
pip show sam2-segmentation-trainer
```
