# Shared helpers for versioned AnnotationCrops mirrors.
# Source this file; do not execute it.

# v1 < v1b < v2 < v10. Optional single letter after the number.
version_name_ok() {
  [[ "$1" =~ ^v[0-9]+[a-zA-Z]?$ ]]
}

version_sort_key() {
  local name="$1"
  if [[ ! "$name" =~ ^v([0-9]+)([a-zA-Z])?$ ]]; then
    return 1
  fi
  local letter="${BASH_REMATCH[2]:-}"
  letter="${letter,,}"
  printf '%05d%s\n' "${BASH_REMATCH[1]}" "${letter}"
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
for ignore_path in sorted(root.glob("*/AnnotationCrops/ignore.json")):
    crops = ignore_path.parent
    volume = crops.parent.name
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
