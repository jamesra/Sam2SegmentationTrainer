#!/usr/bin/env bash
# sync_training_data.sh for SEM data: reads SAM2TrainingData/SEM, mirrors to ~/sam2-training-sem.
# Same arguments as sync_training_data.sh. Run on the WSL Ubuntu host.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
MODALITY=SEM exec "${SCRIPT_DIR}/sync_training_data.sh" "$@"
