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

# Telemetry only. --no-deps avoids nornir_shared's numpy/matplotlib pins on the CUDA torch stack.
install_nornir_mqtt() {
  local src="/opt/nornir-shared"
  if [[ ! -f "${src}/pyproject.toml" ]]; then
    echo "cursor-dev-entry: /opt/nornir-shared not mounted; training progress stays on the console" >&2
    return 0
  fi
  if ! pip install --no-cache-dir 'paho-mqtt>=2.1.0'; then
    echo "cursor-dev-entry: paho-mqtt install failed; training progress stays on the console" >&2
    return 0
  fi
  # The bind is read-only. setuptools writes nornir_shared.egg-info and then
  # utimes that directory, which fails on a read-only mount.
  local build
  build="$(mktemp -d /tmp/nornir-shared.XXXXXX)"
  cp -a "${src}/." "${build}/"
  if ! pip install --no-cache-dir --no-deps "${build}"; then
    rm -rf "${build}"
    echo "cursor-dev-entry: nornir_shared install failed; training progress stays on the console" >&2
    return 0
  fi
  rm -rf "${build}"
  if ! python -c "import paho.mqtt.client; from nornir_shared.mqtt_telemetry import publish_run_meta" >/dev/null 2>&1; then
    echo "cursor-dev-entry: nornir_shared MQTT import failed; training progress stays on the console" >&2
    return 0
  fi
  echo "cursor-dev-entry: nornir_shared MQTT telemetry available (dashboard via NORNIR_MQTT_HOST)"
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

# Training writes SummaryWriter events to $SAM2_CHECKPOINT_ROOT/runs/<run_name>/tb/
# (CIFS). Prefer that over the Windows /outputs bind for large last.pt writes.
ensure_tensorboard() {
  local out_root="${SAM2_OUTPUT_ROOT:-/outputs}"
  local ckpt_root="${SAM2_CHECKPOINT_ROOT:-/storage4/Sam2Trainer}"
  local logdir="${ckpt_root}/runs"
  local port="${SAM2_TENSORBOARD_PORT:-6006}"
  local host_port="${SAM2_TENSORBOARD_HOST_PORT:-8060}"
  local pidfile="${out_root}/tensorboard.pid"
  local logfile="${out_root}/tensorboard.log"
  local pid=""

  mkdir -p "${logdir}"

  if [[ -f "${pidfile}" ]]; then
    pid="$(cat "${pidfile}" 2>/dev/null || true)"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      echo "cursor-dev-entry: tensorboard already running pid=${pid} logdir=${logdir} :${port} (http://localhost:${host_port})"
      return 0
    fi
    rm -f "${pidfile}"
  fi

  if ! command -v tensorboard >/dev/null 2>&1; then
    echo "cursor-dev-entry: tensorboard not on PATH; skip" >&2
    return 0
  fi

  nohup tensorboard \
    --logdir "${logdir}" \
    --bind_all \
    --port "${port}" \
    --reload_interval 30 \
    >"${logfile}" 2>&1 &
  echo $! >"${pidfile}"
  echo "cursor-dev-entry: tensorboard started pid=$(cat "${pidfile}") logdir=${logdir} :${port} -> http://localhost:${host_port}"
}

if [[ "${SAM2_CURSOR_DEV_SETUP_ONLY:-}" == "1" ]]; then
  install_editables
  install_nornir_mqtt
  ensure_tensorboard
  if [[ $# -gt 0 ]]; then
    exec "$@"
  fi
  exit 0
fi

apply_network_shares
install_editables
install_nornir_mqtt
ensure_checkpoint
ensure_tensorboard

if [[ $# -eq 0 ]]; then
  set -- bash
fi
exec_after_mounts "$@"
