# Shared helpers for versioned AnnotationCrops mirrors.
# Source this file; do not execute it.

# A training SET name: v<SET> or v<SET>.<REVISION> (v2, v2.1). The revision starts at 1, so v2.0 is just v2.
# A single trailing letter (v1b) is the old style: still accepted, but do not create new ones.
# These name sets only; a model is v<VERSION>.<EPOCH>, a different number (.cursor/rules/naming-conventions.mdc).
# Order is numeric: v1 < v1b < v2 < v2.1 < v2.2 < v2.10 < v3 < v10.
# Groups: 1 = set number, 3 = revision, 4 = old-style letter.
_VERSION_NAME_RE='^v([0-9]+)(\.([1-9][0-9]*)|([a-zA-Z]))?$'

version_name_ok() {
  [[ "$1" =~ $_VERSION_NAME_RE ]]
}

version_sort_key() {
  local name="$1"
  [[ "$name" =~ $_VERSION_NAME_RE ]] || return 1
  local letter="${BASH_REMATCH[4]:-}"
  letter="${letter,,}"
  # 10# reads the numbers as decimal, so a name such as v08 is not taken for octal.
  printf '%05d.%05d%s\n' "$((10#${BASH_REMATCH[1]}))" "$((10#${BASH_REMATCH[3]:-0}))" "${letter}"
}

# Modality of the data: TEM (default) or SEM. They keep separate version numbers, sources and
# local copies. $1 = the value of MODALITY.
check_modality() {
  case "$1" in
    TEM|SEM) ;;
    *) echo "MODALITY must be TEM or SEM (got '$1')" >&2; return 1 ;;
  esac
}

# Print the NAS source tree: $1 = TRAINING_DATA_SRC, $2 = modality. The NAS keeps
# SAM2TrainingData/TEM/<ver> and SAM2TrainingData/SEM/<ver>. TEM falls back to $1 itself
# for the old layout (SAM2TrainingData/<ver>, or a local /mnt/d/TrainingData).
resolve_training_src() {
  local base="$1" modality="$2"
  if [[ -d "$base/$modality" ]]; then
    echo "$base/$modality"
  elif [[ "$modality" == TEM ]]; then
    echo "$base"
  else
    echo "no $modality folder under $base" >&2
    return 1
  fi
}

# Print the local mirror: TEM uses SAM2_TRAINING_DEST (default ~/sam2-training, the folder the
# containers mount); SEM uses SAM2_TRAINING_DEST_SEM (default ~/sam2-training-sem), never the
# TEM one, so a SEM version cannot land next to TEM versions or under their `current` link.
resolve_training_dest() {
  case "$1" in
    SEM) echo "${SAM2_TRAINING_DEST_SEM:-${HOME}/sam2-training-sem}" ;;
    *) echo "${SAM2_TRAINING_DEST:-${HOME}/sam2-training}" ;;
  esac
}

# Print the highest matching version directory name under $1.
highest_version_dir() {
  local root="$1"
  local best="" best_key="" name key
  [[ -d "$root" ]] || return 1
  shopt -s nullglob
  for path in "$root"/*; do
    [[ -d "$path" && ! -L "$path" ]] || continue
    name="$(basename "$path")"
    version_name_ok "$name" || continue
    key="$(version_sort_key "$name")" || continue
    if [[ -z "$best_key" || "$key" > "$best_key" ]]; then
      best="$name"
      best_key="$key"
    fi
  done
  shopt -u nullglob
  [[ -n "$best" ]] || return 1
  printf '%s\n' "$best"
}

# Print the layout of one volume folder: flat (manifest/catalog directly inside),
# nested (legacy <volume>/AnnotationCrops/), or empty.
volume_layout() {
  local dir="$1"
  if [[ -f "$dir/manifest.jsonl" || -f "$dir/annotation_crops.sqlite" ]]; then
    echo flat
  elif [[ -d "$dir/AnnotationCrops" ]]; then
    echo nested
  else
    echo empty
  fi
}

# Folders in the root of a version that are not volumes and are never mirrored: _review (review
# artifacts) and _work_cache (the exporter's cache). Every loop over a version's volumes skips them.
SYNC_ROOT_NEVER_COPY=(_review _work_cache)

# True when $1, a folder name in the root of a version, is not a volume.
is_version_root_extra() {
  local name="$1" n
  for n in "${SYNC_ROOT_NEVER_COPY[@]}"; do
    [[ "$name" == "$n" ]] && return 0
  done
  return 1
}

# Fail when a volume is flat on one side and nested on the other. rsync --delete
# would otherwise replace the destination's layout with the source's.
check_layouts_match() {
  local src="$1" dest="$2" path vol sl dl bad=0
  shopt -s nullglob
  for path in "$src"/*/; do
    vol="$(basename "$path")"
    is_version_root_extra "$vol" && continue
    [[ -d "$dest/$vol" ]] || continue
    sl="$(volume_layout "$src/$vol")"
    dl="$(volume_layout "$dest/$vol")"
    if [[ "$sl" != empty && "$dl" != empty && "$sl" != "$dl" ]]; then
      echo "layout mismatch for ${vol}: source is ${sl}, destination is ${dl}" >&2
      bad=1
    fi
  done
  shopt -u nullglob
  return "$bad"
}

# Crops folder of one volume inside a version tree, whichever layout it uses.
# Prints nothing when the volume has no export.
volume_crops_dir() {
  local dir="$1"
  case "$(volume_layout "$dir")" in
    flat) echo "$dir" ;;
    nested) echo "$dir/AnnotationCrops" ;;
  esac
}

# Copy approved.json, ignore.json, and ignored/ from an earlier version dir ($1) into the
# destination version dir ($2) for every volume in the source tree ($3). The destination
# volume takes the source's layout. Existing destination files are never overwritten.
# With $4 = dry, only report.
seed_review_lists() {
  local seed="$1" dest="$2" src="$3" mode="${4:-}" path vol sc dc item
  shopt -s nullglob
  for path in "$src"/*/; do
    vol="$(basename "$path")"
    is_version_root_extra "$vol" && continue
    sc="$(volume_crops_dir "$seed/$vol")"
    [[ -n "$sc" ]] || continue
    if [[ "$(volume_layout "$src/$vol")" == nested ]]; then
      dc="$dest/$vol/AnnotationCrops"
    else
      dc="$dest/$vol"
    fi
    for item in approved.json ignore.json ignored; do
      [[ -e "$sc/$item" && ! -e "$dc/$item" ]] || continue
      echo "seed ${vol}/${item} from ${seed##*/}$([[ "$mode" == dry ]] && echo ' (dry run)')"
      [[ "$mode" == dry ]] && continue
      mkdir -p "$dc"
      cp -a "$sc/$item" "$dc/$item"
    done
  done
  shopt -u nullglob
}

# Build version dir $2 from version dir $1 on the same filesystem, sharing disk space:
# every file is a hardlink to the earlier version, except the files that get modified in
# place (the SQLite catalog, approved.json, ignore.json, ignored/), which are real copies.
# rsync writes a temp file and renames it, so a changed file breaks its link and the
# earlier version keeps the old content. With $3 = dry, only report.
clone_version_hardlinked() {
  local from="$1" to="$2" mode="${3:-}" path vol crops rel item
  if [[ -e "$to" ]]; then
    echo "clone: ${to} already exists; -clone-from builds a new version" >&2
    return 1
  fi
  if [[ "$mode" == dry ]]; then
    echo "clone ${from##*/} -> ${to##*/} with hardlinks (dry run)"
    return 0
  fi
  echo "clone ${from##*/} -> ${to##*/} with hardlinks"
  if ! cp -al "$from" "$to"; then
    echo "clone failed; removing ${to}" >&2
    rm -rf "$to"
    return 1
  fi
  shopt -s nullglob
  for path in "$to"/*/; do
    vol="$(basename "$path")"
    crops="$(volume_crops_dir "$to/$vol")"
    [[ -n "$crops" ]] || continue
    rel="${crops#"$to"/}"
    for item in annotation_crops.sqlite annotation_crops.sqlite-wal annotation_crops.sqlite-shm \
        approved.json ignore.json ignored; do
      [[ -e "$crops/$item" ]] || continue
      rm -rf "${crops:?}/$item"
      cp -a "$from/$rel/$item" "$crops/$item"
    done
  done
  shopt -u nullglob
}

# Print the trainer repo root (the folder that holds annotation_crops/scores.py).
# $SAM2_TRAINER_REPO wins; else search upward from this file's real path (symlinks followed);
# else the usual WSL path of the D: checkout.
find_trainer_repo() {
  local marker="annotation_crops/scores.py" dir
  if [[ -n "${SAM2_TRAINER_REPO:-}" ]]; then
    if [[ -f "${SAM2_TRAINER_REPO}/${marker}" ]]; then
      echo "$SAM2_TRAINER_REPO"
      return 0
    fi
    echo "SAM2_TRAINER_REPO=${SAM2_TRAINER_REPO} has no ${marker}" >&2
    return 1
  fi
  dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
  while [[ "$dir" != "/" ]]; do
    if [[ -f "${dir}/${marker}" ]]; then
      echo "$dir"
      return 0
    fi
    dir="$(dirname "$dir")"
  done
  if [[ -f "/mnt/d/src/git/Sam2Trainer/${marker}" ]]; then
    echo "/mnt/d/src/git/Sam2Trainer"
    return 0
  fi
  echo "cannot find the Sam2Trainer checkout (no ${marker})." >&2
  echo "Set SAM2_TRAINER_REPO=/mnt/d/src/git/Sam2Trainer and re-run." >&2
  return 1
}

# Keep the trainer's location_scores table across a mirror. The source catalog replaces
# annotation_crops.sqlite, which would drop the scores written on this machine.
#   save:    copy each volume's scores into $3 (a scratch dir)
#   restore: merge them back into the new catalogs (saved rows win on a key clash)
preserve_location_scores() {
  local mode="$1" dest="$2" stash="$3" repo
  repo="$(find_trainer_repo)" || return 1
  if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 missing; cannot preserve location_scores. Install it or stop." >&2
    return 1
  fi
  python3 - "$mode" "$dest" "$stash" "$repo" <<'PY'
import importlib.util
import sqlite3
import sys
from pathlib import Path

mode, dest, stash, repo = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
# Load scores.py by path: importing the annotation_crops package would need numpy.
spec = importlib.util.spec_from_file_location("location_scores_schema", Path(repo) / "annotation_crops" / "scores.py")
scores = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scores)
ensure_location_scores_schema = scores.ensure_location_scores_schema

TARGET = ["location_id", "image_key", "epoch", "score"]


def catalogs():
    dbs = set(dest.glob("*/annotation_crops.sqlite")) | set(dest.glob("*/AnnotationCrops/annotation_crops.sqlite"))
    for db in sorted(dbs):
        crops = db.parent
        yield (crops.parent.name if crops.name == "AnnotationCrops" else crops.name), db


stash.mkdir(parents=True, exist_ok=True)
for volume, db in catalogs():
    saved = stash / f"{volume}.scores.sqlite"
    connection = sqlite3.connect(db)
    try:
        if mode == "save":
            saved.unlink(missing_ok=True)
            has = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='location_scores'"
            ).fetchone()
            if not has:
                continue
            connection.execute("ATTACH DATABASE ? AS stash", (str(saved),))
            connection.execute("CREATE TABLE stash.location_scores AS SELECT * FROM main.location_scores")
            count = connection.execute("SELECT COUNT(*) FROM stash.location_scores").fetchone()[0]
            connection.commit()
            connection.execute("DETACH DATABASE stash")
            print(f"saved {count} location_scores row(s) for {volume}")
        else:
            if not saved.is_file():
                continue
            ensure_location_scores_schema(connection)
            connection.execute("ATTACH DATABASE ? AS stash", (str(saved),))
            have = {row[1] for row in connection.execute("PRAGMA stash.table_info(location_scores)")}
            select = ", ".join(name if name in have else "''" for name in TARGET)
            connection.execute(
                f"INSERT OR REPLACE INTO main.location_scores ({', '.join(TARGET)}) "
                f"SELECT {select} FROM stash.location_scores"
            )
            count = connection.execute("SELECT COUNT(*) FROM main.location_scores").fetchone()[0]
            connection.commit()
            connection.execute("DETACH DATABASE stash")
            print(f"restored location_scores for {volume}: {count} row(s) now")
    except sqlite3.Error as exc:
        print(f"location_scores {mode} failed for {volume}: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        connection.close()
PY
}

# Move masks named in each volume's ignore.json back to ignored/, and mark those
# catalog rows ignored. approved.json is left as the gallery wrote it.
# The source mirror does not contain these lists; rsync --delete must not be
# the thing that drops them, and it must not put rejected masks back into masks/.
reapply_rejected_masks() {
  local dest="$1"
  if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 missing; review lists were kept, rejected masks were not moved back" >&2
    return 0
  fi
  python3 - "$dest" <<'PY'
import json
import sqlite3
import sys
from pathlib import Path

root = Path(sys.argv[1])
# Flat layout: <volume>/ignore.json. Legacy layout: <volume>/AnnotationCrops/ignore.json.
ignore_paths = sorted(set(root.glob("*/ignore.json")) | set(root.glob("*/AnnotationCrops/ignore.json")))
for ignore_path in ignore_paths:
    crops = ignore_path.parent
    volume = crops.parent.name if crops.name == "AnnotationCrops" else crops.name
    try:
        payload = json.loads(ignore_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"reject list unreadable for {volume}: {exc}", file=sys.stderr)
        continue
    if not isinstance(payload, list):
        print(f"reject list for {volume} is not a JSON array", file=sys.stderr)
        continue
    ids = []
    for item in payload:
        try:
            ids.append(int(item))
        except (TypeError, ValueError):
            continue
    masks = crops / "masks"
    ignored = crops / "ignored"
    moved = 0
    if ids and masks.is_dir():
        ignored.mkdir(parents=True, exist_ok=True)
        for location_id in ids:
            for source in masks.glob(f"*_{location_id}.png"):
                target = ignored / source.name
                if target.is_file():
                    target.unlink()
                source.replace(target)
                moved += 1
    db = crops / "annotation_crops.sqlite"
    if ids and db.is_file():
        connection = sqlite3.connect(db)
        try:
            connection.executemany(
                "UPDATE locations SET ignored = 1 WHERE location_id = ?",
                [(location_id,) for location_id in ids],
            )
            connection.commit()
        except sqlite3.OperationalError as exc:
            print(f"could not mark {volume} catalog ignored: {exc}", file=sys.stderr)
        finally:
            connection.close()
    print(f"kept reject list {volume}: {len(ids)} location(s), {moved} mask(s) moved")
PY
}

# Point $dest/current at a relative version directory that already exists.
link_current() {
  local dest="$1" ver="$2"
  if [[ ! -d "$dest/$ver" ]]; then
    echo "select: missing version dir ${dest}/${ver}" >&2
    return 1
  fi
  ln -sfn "$ver" "$dest/current"
  echo "current -> ${ver} (${dest}/current)"
}

# ---- parallel mirror of a version tree --------------------------------------------------------------
# What training and the annotation gallery read from a volume's crops folder:
#   training: images/ masks/ ignored/ manifest.jsonl annotation_crops.sqlite and the *.json files
#   gallery:  the same, plus overlays/thumb/{512,384,256}/*.jpg and overlays/parts/<id>.json.
#             overlays/cardmask is rebuilt on demand and is not copied.
# Never copied: _work/ (the exporter's cache), _review/, ignored/ and approved.json / ignore.json
# (review state kept on this machine), and everything else under overlays/ (full-size renders, viewing only).
# In the root of a version, _review/ and _work_cache/ are not volumes and are skipped (SYNC_ROOT_NEVER_COPY).
SYNC_NEVER_COPY=(_work _review ignored)
SYNC_OVERLAY_KEEP=(thumb parts)

# Print one NUL-terminated task per unit of work: "<kind>\t<rel>\t<src>\t<dest>".
#   files: the loose files directly in a volume's crops folder (no subfolders)
#   dir:   one subfolder (images, masks, overlays/thumb, ...)
list_mirror_tasks() {
  local src="$1" dest="$2" vol_path vol rel s d sub name skip n k
  shopt -s nullglob
  for vol_path in "$src"/*/; do
    vol="$(basename "$vol_path")"
    is_version_root_extra "$vol" && continue
    case "$(volume_layout "$src/$vol")" in
      nested) rel="$vol/AnnotationCrops" ;;
      *) rel="$vol" ;;
    esac
    s="$src/$rel"
    d="$dest/$rel"
    printf 'files\t%s\t%s\t%s\0' "$rel" "$s" "$d"
    for sub in "$s"/*/ "$s"/.[!.]*/; do
      [[ -d "$sub" ]] || continue
      name="$(basename "$sub")"
      skip=0
      for n in "${SYNC_NEVER_COPY[@]}"; do [[ "$name" == "$n" ]] && skip=1; done
      [[ "$skip" -eq 0 ]] || continue
      if [[ "$name" == overlays ]]; then
        for k in "${SYNC_OVERLAY_KEEP[@]}"; do
          [[ -d "$s/overlays/$k" ]] && printf 'dir\t%s\t%s\t%s\0' "$rel/overlays/$k" "$s/overlays/$k" "$d/overlays/$k"
        done
        continue
      fi
      printf 'dir\t%s\t%s\t%s\0' "$rel/$name" "$s/$name" "$d/$name"
    done
  done
  shopt -u nullglob
}

# Run one task (argument: "<kind>\t<rel>\t<src>\t<dest>"). MIRROR_DRY=1 lists changes only.
_mirror_one() {
  local kind rel s d args out n
  IFS=$'\t' read -r kind rel s d <<<"$1"
  args=(-a --delete --exclude Thumbs.db --exclude .DS_Store)
  if [[ "${MIRROR_DRY:-0}" == 1 ]]; then
    args+=(-n --itemize-changes)
  else
    args+=(--info=stats2)
    mkdir -p "$d"
  fi
  # Loose files only: subfolders are their own tasks. The review lists are never overwritten.
  [[ "$kind" == files ]] && args+=(--exclude '/*/' --exclude '/approved.json' --exclude '/ignore.json')
  if ! out="$(rsync "${args[@]}" "$s/" "$d/" 2>&1)"; then
    printf 'FAILED %s:\n%s\n' "$rel" "$out" >&2
    return 1
  fi
  if [[ "${MIRROR_DRY:-0}" == 1 ]]; then
    out="$(grep -E '^(>|<|c|\*)' <<<"$out" | sed "s|^|[${rel}] |" || true)"
    [[ -z "$out" ]] || printf '%s\n' "$out"
  else
    n="$(awk -F': ' '/Number of regular files transferred/ {gsub(",", "", $2); print $2}' <<<"$out")"
    printf 'synced %s: %s file(s) copied\n' "$rel" "${n:-0}"
  fi
}
export -f _mirror_one

# mirror_volumes_parallel <src> <dest> <dry 0|1> <jobs>
# One rsync per volume and folder, up to <jobs> at a time (CIFS is latency bound, so this scales).
# Folders deleted from the source are no longer removed from the destination.
mirror_volumes_parallel() {
  local src="$1" dest="$2" dry="$3" jobs="$4" count rc=0
  count="$(list_mirror_tasks "$src" "$dest" | tr -cd '\0' | wc -c)"
  if [[ "$count" -eq 0 ]]; then
    echo "nothing to mirror under ${src}" >&2
    return 1
  fi
  [[ "$jobs" -le "$count" ]] || jobs="$count"
  echo "${count} task(s), ${jobs} parallel job(s)$([[ "$dry" -eq 1 ]] && echo ' (dry run)')"
  export MIRROR_DRY="$dry"
  list_mirror_tasks "$src" "$dest" | xargs -0 -P "$jobs" -I{} bash -c '_mirror_one "$1"' _ {} || rc=$?
  return "$rc"
}
