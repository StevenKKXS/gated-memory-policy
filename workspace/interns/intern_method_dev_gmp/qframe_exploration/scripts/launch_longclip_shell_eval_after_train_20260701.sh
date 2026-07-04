#!/usr/bin/env bash
set -euo pipefail

ICRA=/mnt/3fs1/data/tingwen.du/icra_method_dev
CODE=$ICRA/experiments/qframe_evidence_memory_20260630/code/imitation-learning-policies
EVAL=$ICRA/experiments/memory_method_dev/eval_mikasa_checkpoint.sh
TRAIN_RUN=$ICRA/runs/qframe_evidence_memory_20260630/20260701_longclip_shell_formal/qframe_shell_c8_h2_l2_cand8_h2_l2_seed0
RUNLOG=$ICRA/logs/mikasa_method_dev/longclip_shell_5seed_20260701
SEEDS=(1000 1001 1002 1003 1004)
EPISODES=100
ENV_NUM=50

mkdir -p "$RUNLOG"

wait_for_ckpt() {
  local ckpt=""
  while true; do
    ckpt=$(find "$TRAIN_RUN/checkpoints" -maxdepth 1 -type f -name 'epoch_*_train_mean_loss_*.ckpt' 2>/dev/null | sort -V | tail -1 || true)
    if [[ -n "$ckpt" && -f "$ckpt" ]]; then
      printf '%s\n' "$ckpt"
      return 0
    fi
    echo "[wait] no shell longclip ckpt yet utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$RUNLOG/master.log" >&2
    sleep 300
  done
}

run_eval() {
  local gpu=$1
  local mode=$2
  local seed=$3
  local ckpt=$4
  local label="longclip_${mode}_shell_s${seed}_5seed_20260701"
  local log="$RUNLOG/${label}.launcher.log"
  local overrides=(MIKASA_EVAL_QFRAME_QUERY_MODE=image_only)
  if [[ "$mode" == "fused" ]]; then
    overrides=(MIKASA_EVAL_QFRAME_QUERY_MODE=image_text_fused MIKASA_EVAL_QFRAME_TEXT_ALPHA=0.5)
  fi
  echo "[start] gpu=$gpu mode=$mode seed=$seed label=$label utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$RUNLOG/master.log"
  env ICRA_IMITATION_DIR="$CODE" bash "$EVAL" --launch ShellGameTouch-v0 "$ckpt" "$label" "$EPISODES" "$ENV_NUM" "$gpu" "$seed" "${overrides[@]}" > "$log" 2>&1
  local status=$?
  echo "[done] status=$status gpu=$gpu mode=$mode seed=$seed label=$label utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$RUNLOG/master.log"
  return "$status"
}

main() {
  echo "[master-start] utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname)" | tee -a "$RUNLOG/master.log"
  local ckpt
  ckpt=$(wait_for_ckpt)
  echo "[ckpt] $ckpt" | tee -a "$RUNLOG/master.log"

  for seed in "${SEEDS[@]}"; do
    run_eval 2 image "$seed" "$ckpt" &
    pid_image=$!
    run_eval 3 fused "$seed" "$ckpt" &
    pid_fused=$!
    wait "$pid_image" "$pid_fused"
  done
  echo "[master-done] utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$RUNLOG/master.log"
}

main "$@"
