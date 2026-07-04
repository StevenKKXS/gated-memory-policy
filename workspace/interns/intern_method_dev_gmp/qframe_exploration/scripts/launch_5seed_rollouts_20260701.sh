#!/usr/bin/env bash
set -euo pipefail

ICRA=/mnt/3fs1/data/tingwen.du/icra_method_dev
CODE=$ICRA/experiments/qframe_evidence_memory_20260630/code/imitation-learning-policies
EVAL=$ICRA/experiments/memory_method_dev/eval_mikasa_checkpoint.sh
RUNLOG=$ICRA/logs/mikasa_method_dev/qframe_5seed_20260701
EPISODES=100
ENV_NUM=50
SEEDS=(1000 1001 1002 1003 1004)

mkdir -p "$RUNLOG"

QFRAME_RC3=$ICRA/runs/qframe_evidence_memory_20260630/20260630_1754_qframe_formal_train/qframe_rc3_c8_h2_l2_cand8_h2_l2_seed0/checkpoints/epoch_0_train_mean_loss_0_013.ckpt
QFRAME_RC5=$ICRA/runs/qframe_evidence_memory_20260630/20260630_1754_qframe_formal_train/qframe_rc5_c8_h2_l2_cand8_h2_l2_seed0/checkpoints/epoch_0_train_mean_loss_0_004.ckpt
QFRAME_RC9=$ICRA/runs/qframe_evidence_memory_20260630/20260630_1800_qframe_formal_train_rc9_retry/qframe_rc9_c8_h2_l2_cand8_h2_l2_seed0/checkpoints/epoch_0_train_mean_loss_0_003.ckpt
QFRAME_INTERCEPT=$ICRA/runs/qframe_evidence_memory_20260630/20260630_1754_qframe_formal_train/qframe_intercept_c12_h2_l2_cand12_h2_l2_seed0/checkpoints/epoch_0_train_mean_loss_0_002.ckpt
QFRAME_SHELL=$ICRA/runs/qframe_evidence_memory_20260630/20260630_1754_qframe_formal_train/qframe_shell_c8_h2_l2_cand8_h2_l2_seed0/checkpoints/epoch_0_train_mean_loss_0_041.ckpt

LONGCLIP_RC3=$ICRA/runs/qframe_evidence_memory_20260630/20260630_1853_qframe_longclip_formal_nonshell/qframe_rc3_c8_h2_l2_cand8_h2_l2_seed0/checkpoints/epoch_0_train_mean_loss_0_008.ckpt
LONGCLIP_RC5=$ICRA/runs/qframe_evidence_memory_20260630/20260630_1853_qframe_longclip_formal_nonshell/qframe_rc5_c8_h2_l2_cand8_h2_l2_seed0/checkpoints/epoch_0_train_mean_loss_0_004.ckpt
LONGCLIP_RC9=$ICRA/runs/qframe_evidence_memory_20260630/20260630_1853_qframe_longclip_formal_nonshell/qframe_rc9_c8_h2_l2_cand8_h2_l2_seed0/checkpoints/epoch_0_train_mean_loss_0_003.ckpt
LONGCLIP_INTERCEPT=$ICRA/runs/qframe_evidence_memory_20260630/20260630_1853_qframe_longclip_formal_nonshell/qframe_intercept_c12_h2_l2_cand12_h2_l2_seed0/checkpoints/epoch_0_train_mean_loss_0_002.ckpt

for ckpt in \
  "$QFRAME_RC3" "$QFRAME_RC5" "$QFRAME_RC9" "$QFRAME_INTERCEPT" "$QFRAME_SHELL" \
  "$LONGCLIP_RC3" "$LONGCLIP_RC5" "$LONGCLIP_RC9" "$LONGCLIP_INTERCEPT"; do
  test -f "$ckpt"
done

run_eval() {
  local gpu=$1
  local method=$2
  local env_id=$3
  local ckpt=$4
  local label=$5
  local seed=$6
  shift 6
  local log="$RUNLOG/${label}.launcher.log"
  echo "[start] gpu=$gpu method=$method env=$env_id seed=$seed label=$label utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$RUNLOG/master.log"
  env ICRA_IMITATION_DIR="$CODE" bash "$EVAL" --launch "$env_id" "$ckpt" "$label" "$EPISODES" "$ENV_NUM" "$gpu" "$seed" "$@" > "$log" 2>&1
  local status=$?
  echo "[done] status=$status gpu=$gpu method=$method env=$env_id seed=$seed label=$label utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$RUNLOG/master.log"
  return "$status"
}

lane0_shell() {
  for seed in "${SEEDS[@]}"; do
    run_eval 0 qframe_v1 ShellGameTouch-v0 "$QFRAME_SHELL" "qframe_v1_shell_s${seed}_5seed_20260701" "$seed"
  done
}

lane1_qframe_rc() {
  for seed in "${SEEDS[@]}"; do
    run_eval 1 qframe_v1 RememberColor3-v0 "$QFRAME_RC3" "qframe_v1_rc3_s${seed}_5seed_20260701" "$seed"
    run_eval 1 qframe_v1 RememberColor5-v0 "$QFRAME_RC5" "qframe_v1_rc5_s${seed}_5seed_20260701" "$seed"
  done
}

lane2_qframe_rc_intercept() {
  for seed in "${SEEDS[@]}"; do
    run_eval 2 qframe_v1 RememberColor9-v0 "$QFRAME_RC9" "qframe_v1_rc9_s${seed}_5seed_20260701" "$seed"
    run_eval 2 qframe_v1 InterceptMedium-v0 "$QFRAME_INTERCEPT" "qframe_v1_intercept_s${seed}_5seed_20260701" "$seed"
  done
}

lane3_longclip_image_a() {
  for seed in "${SEEDS[@]}"; do
    run_eval 3 longclip_image RememberColor3-v0 "$LONGCLIP_RC3" "longclip_image_rc3_s${seed}_5seed_20260701" "$seed" MIKASA_EVAL_QFRAME_QUERY_MODE=image_only
    run_eval 3 longclip_image RememberColor5-v0 "$LONGCLIP_RC5" "longclip_image_rc5_s${seed}_5seed_20260701" "$seed" MIKASA_EVAL_QFRAME_QUERY_MODE=image_only
  done
}

lane4_longclip_image_b() {
  for seed in "${SEEDS[@]}"; do
    run_eval 4 longclip_image RememberColor9-v0 "$LONGCLIP_RC9" "longclip_image_rc9_s${seed}_5seed_20260701" "$seed" MIKASA_EVAL_QFRAME_QUERY_MODE=image_only
    run_eval 4 longclip_image InterceptMedium-v0 "$LONGCLIP_INTERCEPT" "longclip_image_intercept_s${seed}_5seed_20260701" "$seed" MIKASA_EVAL_QFRAME_QUERY_MODE=image_only
  done
}

lane5_longclip_fused_a() {
  for seed in "${SEEDS[@]}"; do
    run_eval 5 longclip_fused RememberColor3-v0 "$LONGCLIP_RC3" "longclip_fused_rc3_s${seed}_5seed_20260701" "$seed" MIKASA_EVAL_QFRAME_QUERY_MODE=image_text_fused MIKASA_EVAL_QFRAME_TEXT_ALPHA=0.5
    run_eval 5 longclip_fused RememberColor5-v0 "$LONGCLIP_RC5" "longclip_fused_rc5_s${seed}_5seed_20260701" "$seed" MIKASA_EVAL_QFRAME_QUERY_MODE=image_text_fused MIKASA_EVAL_QFRAME_TEXT_ALPHA=0.5
  done
}

lane6_longclip_fused_b() {
  for seed in "${SEEDS[@]}"; do
    run_eval 6 longclip_fused RememberColor9-v0 "$LONGCLIP_RC9" "longclip_fused_rc9_s${seed}_5seed_20260701" "$seed" MIKASA_EVAL_QFRAME_QUERY_MODE=image_text_fused MIKASA_EVAL_QFRAME_TEXT_ALPHA=0.5
    run_eval 6 longclip_fused InterceptMedium-v0 "$LONGCLIP_INTERCEPT" "longclip_fused_intercept_s${seed}_5seed_20260701" "$seed" MIKASA_EVAL_QFRAME_QUERY_MODE=image_text_fused MIKASA_EVAL_QFRAME_TEXT_ALPHA=0.5
  done
}

lane7_spare_qframe_rc5_ablation_guard() {
  echo "[lane7] spare lane intentionally idle for manual reruns utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$RUNLOG/master.log"
}

main() {
  echo "[master-start] utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname)" | tee -a "$RUNLOG/master.log"
  lane0_shell &
  pid0=$!
  lane1_qframe_rc &
  pid1=$!
  lane2_qframe_rc_intercept &
  pid2=$!
  lane3_longclip_image_a &
  pid3=$!
  lane4_longclip_image_b &
  pid4=$!
  lane5_longclip_fused_a &
  pid5=$!
  lane6_longclip_fused_b &
  pid6=$!
  lane7_spare_qframe_rc5_ablation_guard &
  pid7=$!
  printf '%s\n' "$pid0" "$pid1" "$pid2" "$pid3" "$pid4" "$pid5" "$pid6" "$pid7" > "$RUNLOG/lane_pids.txt"
  wait "$pid0" "$pid1" "$pid2" "$pid3" "$pid4" "$pid5" "$pid6" "$pid7"
  echo "[master-done] utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$RUNLOG/master.log"
}

main "$@"
