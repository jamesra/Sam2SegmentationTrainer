#!/usr/bin/env bash
# Mirror one TrainingData version onto the local SSD used by the trainer.
#
# Run on the WSL Ubuntu host (not inside the cursor-dev container): /mnt/d is
# not mounted there. ~/sam2-training is bind-mounted at /data-local.
#
# Usage:
#   scripts/WSL/sync_training_data.sh [-n] [-j N] [-noselect] [-force-layout] [-seed-from vX] [v1|v1b|v2|...]
#
# No version: sync the highest source version directory.
# Default: after a successful mirror, point ~/sam2-training/current at that version.
# -j N: parallel rsync jobs, one per volume and folder (default: CPUs, at most 8; -j 1 = one at a time).
# Only what training and the gallery read is copied: _work/ and the full-size overlay renders are
# left out (overlays/thumb and overlays/parts are kept). See SYNC_NEVER_COPY in training_data_common.sh.
# The _review/ and _work_cache/ folders in the root of a version are not volumes and are not copied.
# -n: dry run. Report the layout check, seeding, and rsync changes; change nothing.
# -noselect: mirror only; leave current unchanged.
# -force-layout: mirror even when a volume is flat on one side and nested
#   (<volume>/AnnotationCrops/) on the other. The destination takes the source's layout.
# -seed-from vX: before mirroring, copy approved.json, ignore.json, and ignored/ from the
#   local version vX into volumes of the destination that do not have them yet. Use it
#   when mirroring into a new version so the review lists and location_scores carry over.
#   Never overwrites existing lists; this version's own scores win on a clash.
# -clone-from vX: build the new version as a hardlink clone of the local version vX, then
#   rsync only the changes. Unchanged files stay one copy on disk shared by both versions.
#   The catalog, approved.json, ignore.json, and ignored/ are real copies. Implies the
#   review-state carry-over of -seed-from. The destination version must not exist yet.
#
# Kept across a mirror: approved.json, ignore.json, and ignored/ (excluded from rsync, with
# rejected masks moved back to ignored/ afterwards), and the location_scores table in
# annotation_crops.sqlite (saved before the copy and merged back after).
#
# Env:
#   TRAINING_DATA_SRC=/mnt/d/TrainingData   (e.g. /mnt/y/SAM2TrainingData with the NAS
#                                            share CIFS-mounted at /mnt/y; see
#                                            scripts/WSL/wsl-training.README).
#                                            Versions are read from $TRAINING_DATA_SRC/$MODALITY
#                                            when that folder exists; TEM falls back to the
#                                            old layout with the versions directly in it.
#   MODALITY=TEM                            TEM (default) or SEM. sem-sync_training_data.sh sets SEM.
#   SAM2_TRAINING_DEST=$HOME/sam2-training  TEM mirror (the one the containers mount)
#   SAM2_TRAINING_DEST_SEM=$HOME/sam2-training-sem   SEM mirror
#
# One-time move of a flat layout (RC1, RC2, RPC1, RPC2 next to version dirs):
#   mkdir -p ~/sam2-training/v1
#   mv ~/sam2-training/RC1 ~/sam2-training/RC2 \
#      ~/sam2-training/RPC1 ~/sam2-training/RPC2 ~/sam2-training/v1/
#   scripts/WSL/sync_training_data.sh v1b
#   scripts/WSL/sync_training_data.sh          # latest source, and select it
#   readlink ~/sam2-training/current       # expect the highest synced version
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
# shellcheck source=training_data_common.sh
source "${SCRIPT_DIR}/training_data_common.sh"

# Fail before any clone or copy when the repo (needed for location_scores) cannot be found.
find_trainer_repo >/dev/null || exit 1

MODALITY="${MODALITY:-TEM}"
check_modality "$MODALITY" || exit 2
TRAINING_DATA_SRC="$(resolve_training_src "${TRAINING_DATA_SRC:-/mnt/d/TrainingData}" "$MODALITY")" || exit 1
SAM2_TRAINING_DEST="$(resolve_training_dest "$MODALITY")"
echo "${MODALITY}: ${TRAINING_DATA_SRC} -> ${SAM2_TRAINING_DEST}"

dry=0
jobs=""
noselect=0
force_layout=0
seed_from=""
clone_from=""
ver=""
while [[ $# -gt 0 ]]; do
  arg="$1"
  shift
  case "$arg" in
    -n) dry=1 ;;
    -j)
      if [[ $# -eq 0 || ! "$1" =~ ^[1-9][0-9]*$ ]]; then
        echo "-j needs a positive number" >&2
        exit 2
      fi
      jobs="$1"
      shift
      ;;
    -noselect) noselect=1 ;;
    -force-layout) force_layout=1 ;;
    -seed-from)
      if [[ $# -eq 0 ]]; then
        echo "-seed-from needs a version" >&2
        exit 2
      fi
      seed_from="$1"
      shift
      ;;
    -clone-from)
      if [[ $# -eq 0 ]]; then
        echo "-clone-from needs a version" >&2
        exit 2
      fi
      clone_from="$1"
      shift
      ;;
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

if [[ -n "$ver" ]]; then
  if ! version_name_ok "$ver"; then
    echo "version must look like v2 or v2.1, or the old v1b (got ${ver})" >&2
    exit 2
  fi
else
  if ! ver="$(highest_version_dir "$TRAINING_DATA_SRC")"; then
    echo "no version directories under ${TRAINING_DATA_SRC}" >&2
    exit 1
  fi
  echo "selected highest source version ${ver}"
fi

if [[ -n "$clone_from" ]]; then
  if [[ -n "$seed_from" && "$seed_from" != "$clone_from" ]]; then
    echo "-clone-from already carries the review state; drop -seed-from or use the same version" >&2
    exit 2
  fi
  # A clone has the review lists and scores of its parent, so it doubles as the seed.
  seed_from="$clone_from"
fi

if [[ -n "$seed_from" ]]; then
  if ! version_name_ok "$seed_from"; then
    echo "-seed-from version must look like v2 or v2.1, or the old v1b (got ${seed_from})" >&2
    exit 2
  fi
  if [[ "$seed_from" == "$ver" ]]; then
    echo "-seed-from must differ from the version being mirrored" >&2
    exit 2
  fi
  if [[ ! -d "${SAM2_TRAINING_DEST}/${seed_from}" ]]; then
    echo "missing seed version ${SAM2_TRAINING_DEST}/${seed_from}" >&2
    exit 1
  fi
fi

src="${TRAINING_DATA_SRC}/${ver}"
dest="${SAM2_TRAINING_DEST}/${ver}"
if [[ ! -d "$src" ]]; then
  echo "missing source ${src}" >&2
  exit 1
fi

echo "rsync ${src}/ -> ${dest}/$([[ "$dry" -eq 1 ]] && echo ' (dry run)')"
if [[ "$dry" -eq 1 ]]; then
  echo "dry run: nothing will be changed"
elif [[ "$noselect" -eq 1 ]]; then
  echo "current unchanged (-noselect)"
else
  echo "will select ${ver} as current"
fi
echo "keeping approved.json, ignore.json, ignored/, and location_scores in each volume"

# With -clone-from the destination will be a copy of that version, so compare against it.
layout_ref="$dest"
[[ -z "$clone_from" ]] || layout_ref="${SAM2_TRAINING_DEST}/${clone_from}"
if [[ "$force_layout" -eq 0 ]] && ! check_layouts_match "$src" "$layout_ref"; then
  echo "refusing to mirror: rsync --delete would swap the destination's layout." >&2
  echo "Flatten the source (the NAS version-sync output) or re-run with -force-layout." >&2
  exit 1
fi

if [[ "$dry" -eq 1 ]]; then
  [[ -z "$clone_from" ]] || clone_version_hardlinked "${SAM2_TRAINING_DEST}/${clone_from}" "$dest" dry
  [[ -n "$clone_from" || -z "$seed_from" ]] || seed_review_lists "${SAM2_TRAINING_DEST}/${seed_from}" "$dest" "$src" dry
else
  if [[ -n "$clone_from" ]]; then
    clone_version_hardlinked "${SAM2_TRAINING_DEST}/${clone_from}" "$dest"
  fi
  mkdir -p "$dest"
  [[ -z "$seed_from" ]] || seed_review_lists "${SAM2_TRAINING_DEST}/${seed_from}" "$dest" "$src"
  stash="$(mktemp -d)"
  seed_stash="$(mktemp -d)"
  trap '[[ -d "${stash:-}" ]] && echo "location_scores copy kept at ${stash} and ${seed_stash}" >&2' EXIT
  preserve_location_scores save "$dest" "$stash"
  [[ -z "$seed_from" ]] || preserve_location_scores save "${SAM2_TRAINING_DEST}/${seed_from}" "$seed_stash"
fi

# Flat layout: <volume>/approved.json etc. Legacy layout: <volume>/AnnotationCrops/...
rsync_dest="$dest"
# A dry run with -clone-from has no clone yet, so compare against the version it would clone.
[[ "$dry" -eq 1 && -n "$clone_from" ]] && rsync_dest="$layout_ref"
if [[ -z "$jobs" ]]; then
  jobs="$(nproc 2>/dev/null || echo 4)"
  [[ "$jobs" -le 8 ]] || jobs=8
fi
mirror_volumes_parallel "$src" "$rsync_dest" "$dry" "$jobs"

if [[ "$dry" -eq 1 ]]; then
  echo "dry run complete: nothing changed"
  exit 0
fi

# Seed scores first so this version's own scores win on a key clash.
preserve_location_scores restore "$dest" "$seed_stash"
preserve_location_scores restore "$dest" "$stash"
reapply_rejected_masks "$dest"
rm -rf "$stash" "$seed_stash"

if [[ "$noselect" -eq 0 ]]; then
  link_current "$SAM2_TRAINING_DEST" "$ver"
fi
