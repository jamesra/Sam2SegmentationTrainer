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
