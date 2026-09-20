#!/usr/bin/env bash
# Prepare the SAM2 trainer cursor-dev shell: optional path-B CIFS, editable pip install, checkpoints.
# SAM2_CURSOR_DEV_SETUP_ONLY=1: pip install -e only (Dev Container postAttach; skip remount).
# NORNIR_NET_MOUNTS=1: apply /etc/nornir-net-mounts/nas-mounts.tsv via mount-network-shares.sh.
# After successful mounts, CAP_SYS_ADMIN is dropped before the final exec when setpriv/capsh exist.
set -euo pipefail

ulimit -n 65536 2>/dev/null || true

SAM2_NET_MOUNTS_APPLIED=0
SAM2_CHECKPOINT_URL="${SAM2_CHECKPOINT_URL:-https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt}"
SAM2_CHECKPOINT_NAME="${SAM2_CHECKPOINT_NAME:-sam2.1_hiera_large.pt}"

apply_network_shares() {
  local script="/usr/local/bin/mount-network-shares.sh"
  if [[ "${NORNIR_NET_MOUNTS:-}" != "1" ]]; then
    return 0
  fi
  if [[ ! -f "${script}" ]]; then
    echo "cursor-dev-entry: NORNIR_NET_MOUNTS=1 but mount-network-shares.sh not found" >&2
    exit 1
  fi
  bash "${script}"
  SAM2_NET_MOUNTS_APPLIED=1
}

exec_after_mounts() {
  local helper="/usr/local/bin/drop-sys-admin-after-mounts.sh"
  if [[ "${SAM2_NET_MOUNTS_APPLIED}" -eq 1 && -f "${helper}" ]]; then
    # shellcheck source=/dev/null
    source "${helper}"
    drop_sys_admin_after_mounts -- "$@"
  fi
  exec "$@"
}

install_editables() {
  if [[ ! -f /workspace/pyproject.toml ]]; then
    echo "cursor-dev-entry: missing /workspace/pyproject.toml (is the trainer repo bind-mounted?)" >&2
    exit 1
  fi
  pip install --no-cache-dir --no-deps -e /workspace
}

ensure_checkpoint() {
  local dest="${SAM2_OUTPUT_ROOT:-/outputs}/checkpoints"
  local file="${dest}/${SAM2_CHECKPOINT_NAME}"
  mkdir -p "${dest}"
  if [[ -f "${file}" ]]; then
    echo "cursor-dev-entry: checkpoint already present ${file}"
    return 0
  fi
  echo "cursor-dev-entry: downloading ${SAM2_CHECKPOINT_NAME} -> ${file}"
  curl -L --fail --retry 3 -o "${file}.partial" "${SAM2_CHECKPOINT_URL}"
  mv "${file}.partial" "${file}"
}

if [[ "${SAM2_CURSOR_DEV_SETUP_ONLY:-}" == "1" ]]; then
  install_editables
  if [[ $# -gt 0 ]]; then
    exec "$@"
  fi
  exit 0
fi

apply_network_shares
install_editables
ensure_checkpoint

if [[ $# -eq 0 ]]; then
  set -- bash
fi
exec_after_mounts "$@"
