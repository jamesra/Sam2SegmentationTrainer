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

# Fail when a volume is flat on one side and nested on the other. rsync --delete
# would otherwise replace the destination's layout with the source's.
check_layouts_match() {
  local src="$1" dest="$2" path vol sl dl bad=0
  shopt -s nullglob
  for path in "$src"/*/; do
    vol="$(basename "$path")"
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
