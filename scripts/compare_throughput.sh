#!/usr/bin/env bash
# Compare two training configurations back to back on the GPU, alternating A, B, A, B, with the GPU clocks,
# temperature, power and throttle reasons logged during every run, and the other GPU processes noted.
#
#   A: model.activation_checkpointing=false  training.batch_size=2  training.gradient_accumulation_steps=8
#   B: model.activation_checkpointing=true   training.batch_size=4  training.gradient_accumulation_steps=4
#
# Each run is the training-only benchmark (no validation timing). Outputs go to $OUT (default
# benchmarks/throughput, which is git-ignored), then scripts/summarize_throughput.py prints the table.
#
#   bash scripts/compare_throughput.sh
#   STEPS=300 OUT=benchmarks/throughput2 bash scripts/compare_throughput.sh
#   COOLDOWN=120 bash scripts/compare_throughput.sh      # idle seconds between runs (default 0)
#
# Make sure nothing else is using the GPU while it runs; the processes that are get listed in the report.
set -u

OUT="${1:-${OUT:-benchmarks/throughput}}"
STEPS="${STEPS:-200}"
COOLDOWN="${COOLDOWN:-0}"
mkdir -p "$OUT"

FIELDS="timestamp,clocks.sm,clocks.max.sm,temperature.gpu,power.draw,power.limit,utilization.gpu,clocks_throttle_reasons.active"
# Newer drivers renamed the last field.
if ! nvidia-smi --query-gpu="$FIELDS" --format=csv >/dev/null 2>&1; then
  FIELDS="${FIELDS/clocks_throttle_reasons.active/clocks_event_reasons.active}"
fi
PROCS="pid,name,used_memory"

nvidia-smi > "$OUT/nvidia-smi.start.txt" 2>&1

run() {
  local label="$1" name="$2"
  shift 2
  echo "=== $name (config $label): $*"
  nvidia-smi --query-compute-apps="$PROCS" --format=csv > "$OUT/$name.procs_before.txt" 2>&1
  nvidia-smi --query-gpu="$FIELDS" --format=csv -l 5 > "$OUT/$name.smi.csv" 2>&1 &
  local smi=$!
  sam2-em-train --benchmark --benchmark-no-val --benchmark-steps "$STEPS" "$@" 2>&1 | tee "$OUT/$name.txt"
  kill "$smi" 2>/dev/null
  wait "$smi" 2>/dev/null
  nvidia-smi --query-compute-apps="$PROCS" --format=csv > "$OUT/$name.procs_after.txt" 2>&1
  if [ "$COOLDOWN" -gt 0 ]; then sleep "$COOLDOWN"; fi
}

A=(model.activation_checkpointing=false training.batch_size=2 training.gradient_accumulation_steps=8)
B=(model.activation_checkpointing=true training.batch_size=4 training.gradient_accumulation_steps=4)

run A run1_A "${A[@]}"
run B run2_B "${B[@]}"
run A run3_A "${A[@]}"
run B run4_B "${B[@]}"

echo
python "$(dirname "$0")/summarize_throughput.py" "$OUT" | tee "$OUT/summary.txt"
