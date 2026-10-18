#!/usr/bin/env bash
# Point ~/sam2-training/current at a local version directory. Does not copy files.
#
# Run on the WSL Ubuntu host. The container reads that link as /data-local/current.
#
# Usage:
#   scripts/WSL/select_training_data.sh [v1|v1b|v2|...]
#
# No version: link current to the highest version directory already under the dest.
#
# Env:
#   MODALITY=TEM                            TEM (default) or SEM; sem-select_training_data.sh sets SEM
#   SAM2_TRAINING_DEST=$HOME/sam2-training  TEM mirror
#   SAM2_TRAINING_DEST_SEM=$HOME/sam2-training-sem   SEM mirror
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
# shellcheck source=training_data_common.sh
source "${SCRIPT_DIR}/training_data_common.sh"

MODALITY="${MODALITY:-TEM}"
check_modality "$MODALITY" || exit 2
SAM2_TRAINING_DEST="$(resolve_training_dest "$MODALITY")"

ver=""
for arg in "$@"; do
  case "$arg" in
    -h|--help)
      sed -n '2,/^set -euo/p' "$0" | sed '$d'
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

if [[ -z "$ver" ]]; then
  if ! ver="$(highest_version_dir "$SAM2_TRAINING_DEST")"; then
    echo "no version directories under ${SAM2_TRAINING_DEST}" >&2
    exit 1
  fi
  echo "selected highest local version ${ver}"
elif ! version_name_ok "$ver"; then
  echo "version must look like v2 or v2.1, or the old v1b (got ${ver})" >&2
  exit 2
fi

mkdir -p "$SAM2_TRAINING_DEST"
link_current "$SAM2_TRAINING_DEST" "$ver"
