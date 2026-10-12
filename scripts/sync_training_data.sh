#!/usr/bin/env bash
# Mirror one TrainingData version onto the local SSD used by the trainer.
#
# Run on the WSL Ubuntu host (not inside the cursor-dev container): /mnt/d is
# not mounted there. ~/sam2-training is bind-mounted at /data-local.
#
# Usage:
#   scripts/sync_training_data.sh [-noselect] [v1|v1b|v2|...]
#
# No version: sync the highest /mnt/d/TrainingData/v* directory.
# Default: after a successful mirror, point ~/sam2-training/current at that version.
# -noselect: mirror only; leave current unchanged.
#
# approved.json and ignore.json are written on this SSD by the gallery. They are
# excluded from the mirror, and rejected masks are moved back to ignored/ after
# the copy so --delete cannot clear a volume's review lists.
#
# Env:
#   TRAINING_DATA_SRC=/mnt/d/TrainingData
#   SAM2_TRAINING_DEST=$HOME/sam2-training
#
# One-time move of a flat layout (RC1, RC2, RPC1, RPC2 next to version dirs):
#   mkdir -p ~/sam2-training/v1
#   mv ~/sam2-training/RC1 ~/sam2-training/RC2 \
#      ~/sam2-training/RPC1 ~/sam2-training/RPC2 ~/sam2-training/v1/
#   scripts/sync_training_data.sh v1b
#   scripts/sync_training_data.sh          # latest source, and select it
#   readlink ~/sam2-training/current       # expect the highest synced version
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=training_data_common.sh
source "${SCRIPT_DIR}/training_data_common.sh"

TRAINING_DATA_SRC="${TRAINING_DATA_SRC:-/mnt/d/TrainingData}"
SAM2_TRAINING_DEST="${SAM2_TRAINING_DEST:-${HOME}/sam2-training}"

noselect=0
ver=""
for arg in "$@"; do
  case "$arg" in
    -noselect) noselect=1 ;;
    -h|--help)
      sed -n '2,22p' "$0"
      exit 0
      ;;
    -*)
      echo "unknown flag: ${arg}" >&2
      exit 2
      ;;
    *)
      if [[ -n "$ver" ]]; then
        echo "extra argument: ${arg}" >&2
        exit 2
      fi
      ver="$arg"
      ;;
  esac
done

if [[ -n "$ver" ]]; then
  if ! version_name_ok "$ver"; then
    echo "version must look like v1, v1b, or v2 (got ${ver})" >&2
    exit 2
  fi
else
  if ! ver="$(highest_version_dir "$TRAINING_DATA_SRC")"; then
    echo "no version directories under ${TRAINING_DATA_SRC}" >&2
    exit 1
  fi
  echo "selected highest source version ${ver}"
fi

src="${TRAINING_DATA_SRC}/${ver}"
dest="${SAM2_TRAINING_DEST}/${ver}"
if [[ ! -d "$src" ]]; then
  echo "missing source ${src}" >&2
  exit 1
fi

mkdir -p "$dest"
echo "rsync ${src}/ -> ${dest}/"
if [[ "$noselect" -eq 1 ]]; then
  echo "current unchanged (-noselect)"
else
  echo "will select ${ver} as current"
fi
echo "keeping AnnotationCrops approved.json, ignore.json, and ignored/"

rsync -a --delete --info=stats2 \
  --exclude Thumbs.db --exclude .DS_Store \
  --exclude '**/AnnotationCrops/approved.json' \
  --exclude '**/AnnotationCrops/ignore.json' \
  --exclude '**/AnnotationCrops/ignored/***' \
  --exclude '_review/***' \
  "${src}/" "${dest}/"

reapply_rejected_masks "$dest"

if [[ "$noselect" -eq 0 ]]; then
  link_current "$SAM2_TRAINING_DEST" "$ver"
fi
