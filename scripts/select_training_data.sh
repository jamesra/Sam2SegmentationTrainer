#!/usr/bin/env bash
# Point ~/sam2-training/current at a local version directory. Does not copy files.
#
# Run on the WSL Ubuntu host. The container reads that link as /data-local/current.
#
# Usage:
#   scripts/select_training_data.sh [v1|v1b|v2|...]
#
# No version: link current to the highest version directory already under the dest.
#
# Env:
#   SAM2_TRAINING_DEST=$HOME/sam2-training
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=training_data_common.sh
source "${SCRIPT_DIR}/training_data_common.sh"

SAM2_TRAINING_DEST="${SAM2_TRAINING_DEST:-${HOME}/sam2-training}"

ver=""
for arg in "$@"; do
  case "$arg" in
    -h|--help)
      sed -n '2,14p' "$0"
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
  echo "version must look like v1, v1b, or v2 (got ${ver})" >&2
  exit 2
fi

mkdir -p "$SAM2_TRAINING_DEST"
link_current "$SAM2_TRAINING_DEST" "$ver"
