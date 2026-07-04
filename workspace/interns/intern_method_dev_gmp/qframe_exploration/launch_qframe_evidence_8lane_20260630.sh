#!/usr/bin/env bash
set -euo pipefail

MODE=${MODE:---dry-run}
case "$MODE" in
  --dry-run|--launch) ;;
  *)
    echo "MODE must be --dry-run or --launch, got: $MODE" >&2
    exit 1
    ;;
esac

RUN_SET=${RUN_SET:-formal}
case "$RUN_SET" in
  smoke|formal|main|ablation) ;;
  *)
    echo "RUN_SET must be smoke, formal, main, or ablation, got: $RUN_SET" >&2
    exit 1
    ;;
esac

ICRA_BASE=${ICRA_BASE:-/mnt/3fs1/data/tingwen.du/icra_method_dev}
EXP_ROOT=$ICRA_BASE/experiments/qframe_evidence_memory_20260630
CODE_DIR=${CODE_DIR:-$EXP_ROOT/code/imitation-learning-policies}
PY=${PY:-$ICRA_BASE/envs/imitation-py310-h200-headless/bin/python}
EVAL_SCRIPT=${EVAL_SCRIPT:-$ICRA_BASE/experiments/memory_method_dev/eval_mikasa_checkpoint.sh}
OPENCV_DIR=$ICRA_BASE/deps/opencv_headless_py310
GLVND_LIB=$ICRA_BASE/deps/glvnd_ubuntu24.04_amd64/root/usr/lib/x86_64-linux-gnu
SAPIEN_BUILTIN_VULKAN=${SAPIEN_BUILTIN_VULKAN:-/mnt/3fs1/data/tingwen.du/gated-memory-policy-envs/task003_robomimic_memory_gate_repro/mikasa-py311/lib/python3.11/site-packages/sapien/vulkan_library/libvulkan.so.1.3.224}

STAMP=${STAMP:-$(date -u +%Y%m%d_%H%M%S_qframe_evidence)}
POLICY_NAME=${POLICY_NAME:-diffusion_mikasa_qframe_evidence_memory_transformer}
TRAIN_SEED=${TRAIN_SEED:-0}
EVAL_SEED=${EVAL_SEED:-1000}
NUM_EPOCHS=${NUM_EPOCHS:-1}
LR=${LR:-3e-5}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
SPLIT_BATCH_SIZE=${SPLIT_BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-0}
PERSISTENT_WORKERS=${PERSISTENT_WORKERS:-false}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-null}
INDEX_POOL_SIZE_PER_EPISODE=${INDEX_POOL_SIZE_PER_EPISODE:-20}
INCLUDE_EPISODE_INDICES=${INCLUDE_EPISODE_INDICES:-'${range:0,250}'}
TRAINER_DEBUG=${TRAINER_DEBUG:-false}
SKIP_EVAL=${SKIP_EVAL:-false}
EPISODES=${EPISODES:-100}
ENV_NUM=${ENV_NUM:-50}
SETTLE_SECONDS=${SETTLE_SECONDS:-0}
LANE_FILTER=${LANE_FILTER:-}

if [[ "$RUN_SET" == "smoke" ]]; then
  TRAINER_DEBUG=true
  EPISODES=1
  ENV_NUM=1
  LANE_FILTER=${LANE_FILTER:-rc5_c8_h2_l2}
fi

RC3_CKPT=$ICRA_BASE/runs/mikasa_method_dev/20260609_2221visual_k4_v2/rc3_start_anchor_v3fbest_state_token_visualwriter_k4_b4_read001_gate2_s02_lr1e4_6ep_idx100_seed0_v2/checkpoints/epoch_3_train_mean_loss_0_000.ckpt
RC5_CKPT=$ICRA_BASE/runs/mikasa_method_dev/20260607_064830/rc5_dense_idx4_stepmatch_seed0/checkpoints/epoch_32_train_mean_loss_0_003.ckpt
RC9_CKPT=$ICRA_BASE/runs/mikasa_method_dev/20260607_064937/rc9_top1_idx4_seed2_retry/checkpoints/epoch_37_train_mean_loss_0_002.ckpt
INTERCEPT_CKPT=$ICRA_BASE/runs/mikasa_method_dev/20260607_053019/intercept_top1_seed0/checkpoints/epoch_8_train_mean_loss_0_003.ckpt
SHELL_CKPT=$ICRA_BASE/runs/mikasa_method_dev/20260607_064831/shell_dense_idx4_stepmatch_seed0/checkpoints/epoch_7_train_mean_loss_0_001.ckpt

test -d "$CODE_DIR"
test -x "$PY"
test -f "$EVAL_SCRIPT"
test -d "$OPENCV_DIR/cv2"
test -d "$GLVND_LIB"
test -f "$SAPIEN_BUILTIN_VULKAN"
for ckpt in "$RC3_CKPT" "$RC5_CKPT" "$RC9_CKPT" "$INTERCEPT_CKPT" "$SHELL_CKPT"; do
  test -f "$ckpt"
done

RUN_ROOT=$ICRA_BASE/runs/qframe_evidence_memory_20260630/$STAMP
LOG_ROOT=$ICRA_BASE/logs/qframe_evidence_memory_20260630/$STAMP
MANIFEST=$LOG_ROOT/launch_manifest.tsv
RESULTS_TSV=$LOG_ROOT/results.tsv
mkdir -p "$RUN_ROOT" "$LOG_ROOT"

common_env() {
  export HF_HOME=$ICRA_BASE/cache/hf-home
  export TRANSFORMERS_CACHE=$HF_HOME/hub
  export TRANSFORMERS_OFFLINE=1
  export HF_HUB_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  export TORCH_HOME=$ICRA_BASE/cache/torch
  export XDG_CACHE_HOME=$ICRA_BASE/cache/xdg
  export MPLCONFIGDIR=$ICRA_BASE/cache/matplotlib
  export LD_LIBRARY_PATH="$GLVND_LIB:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
  export ICRA_IMITATION_DIR="$CODE_DIR"
  export PYTHONPATH="$OPENCV_DIR:$CODE_DIR:${PYTHONPATH:-}"
}

action_indices_for_length() {
  local action_len=$1
  local values=()
  local idx
  for ((idx=0; idx<action_len; idx++)); do
    values+=("$idx")
  done
  local joined
  joined=$(IFS=,; echo "${values[*]}")
  printf '[%s]\n' "$joined"
}

should_run_lane() {
  local lane=$1
  local group=$2
  if [[ "$RUN_SET" == "main" && "$group" != "main" ]]; then
    return 1
  fi
  if [[ "$RUN_SET" == "ablation" && "$group" != "ablation" ]]; then
    return 1
  fi
  if [[ -n "$LANE_FILTER" && ",$LANE_FILTER," != *",$lane,"* ]]; then
    return 1
  fi
  return 0
}

run_lane() {
  local lane=$1
  local group=$2
  local gpu=$3
  local port=$4
  local task_name=$5
  local env_id=$6
  local base_ckpt=$7
  local max_history_len=$8
  local action_len=$9
  local traj_interval=${10}
  local candidate_max=${11}
  local high_topk=${12}
  local low_topk=${13}
  local ignore_history_time_embedding=${14}

  local train_traj_num=$((max_history_len + 1))
  local action_indices
  action_indices=$(action_indices_for_length "$action_len")
  local run_name="qframe_${lane}_cand${candidate_max}_h${high_topk}_l${low_topk}_seed${TRAIN_SEED}"
  local run_dir="$RUN_ROOT/$run_name"
  local train_log="$LOG_ROOT/$run_name.train.log"
  local eval_label="${run_name}_eval_seed${EVAL_SEED}"
  local eval_log="$LOG_ROOT/$run_name.eval_wrapper.log"
  local status_path="$LOG_ROOT/$run_name.status.tsv"

  local cmd=(
    "$PY" -m accelerate.commands.launch
    --gpu_ids "$gpu"
    --num_processes 1
    --main_process_port "$port"
    scripts/train_policy.py
    +policy_name="$POLICY_NAME"
    +task_name="$task_name"
    +logger_project_name=icra_qframe_evidence
    +project_name=icra_method_dev
    +run_name="$run_name"
    +train_server_name=10.100.0.3
    +seed="$TRAIN_SEED"
    +base_ckpt_path="$base_ckpt"
    +workspace.logging_cfg.mode=offline
    +workspace.trainer.output_dir="$run_dir"
    +workspace.trainer.num_epochs="$NUM_EPOCHS"
    +workspace.trainer.debug="$TRAINER_DEBUG"
    +workspace.trainer.rollout_every=0
    +workspace.trainer.val_every=0
    +workspace.trainer.sample_every=0
    +workspace.trainer.optimizer_partial.lr="$LR"
    +workspace.train_dataset.compressed_dir="$ICRA_BASE/datasets/mikasa/zarr"
    +workspace.train_dataset.root_dir="$ICRA_BASE/datasets/mikasa/zarr"
    +workspace.train_dataset.normalizer_dir="$ICRA_BASE/datasets/mikasa/zarr"
    +workspace.train_dataset.name="$env_id"
    "+workspace.train_dataset.include_episode_indices=$INCLUDE_EPISODE_INDICES"
    +workspace.train_dataset.include_episode_num=-1
    +workspace.train_dataset.starting_percentile_max=1.0
    +workspace.train_dataset.index_pool_size_per_episode="$INDEX_POOL_SIZE_PER_EPISODE"
    +workspace.train_dataset.dataloader_cfg.batch_size="$TRAIN_BATCH_SIZE"
    +workspace.train_dataset.dataloader_cfg.num_workers="$NUM_WORKERS"
    +workspace.train_dataset.dataloader_cfg.persistent_workers="$PERSISTENT_WORKERS"
    +workspace.train_dataset.split_dataloader_cfg.batch_size="$SPLIT_BATCH_SIZE"
    +workspace.train_dataset.split_dataloader_cfg.num_workers="$NUM_WORKERS"
    +workspace.train_dataset.split_dataloader_cfg.persistent_workers="$PERSISTENT_WORKERS"
    ++workspace.model.trainable_param_keywords="[evidence_memory,denoising_network.blocks.14,denoising_network.blocks.15]"
    +workspace.model.denoising_network_partial.include_action_history=false
    ++workspace.train_dataset.traj_num="$train_traj_num"
    ++workspace.train_dataset.traj_interval_min="$traj_interval"
    ++workspace.train_dataset.traj_interval_max="$traj_interval"
    ++workspace.model.action_length="$action_len"
    "++workspace.model.action_indices=$action_indices"
    ++workspace.model.history_action_num_per_chunk="$action_len"
    ++workspace.model.denoising_network_partial.max_history_len="$max_history_len"
    ++workspace.model.denoising_network_partial.evidence_candidate_max_num="$candidate_max"
    ++workspace.model.denoising_network_partial.evidence_high_topk="$high_topk"
    ++workspace.model.denoising_network_partial.evidence_low_topk="$low_topk"
  )

  if [[ "$ignore_history_time_embedding" == "true" ]]; then
    cmd+=(+base_ckpt_ignore_keywords="[history_time_embedding]")
  fi
  cmd+=(+workspace.train_dataset.dataloader_cfg.prefetch_factor="$PREFETCH_FACTOR")
  cmd+=(+workspace.train_dataset.split_dataloader_cfg.prefetch_factor="$PREFETCH_FACTOR")

  {
    printf 'lane\tgroup\tgpu\tpolicy\ttask_name\tenv_id\tbase_ckpt\tmax_history_len\tcandidate_max\thigh_topk\tlow_topk\trun_dir\ttrain_log\teval_label\n'
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$lane" "$group" "$gpu" "$POLICY_NAME" "$task_name" "$env_id" "$base_ckpt" \
      "$max_history_len" "$candidate_max" "$high_topk" "$low_topk" "$run_dir" "$train_log" "$eval_label"
  } > "$status_path"

  if [[ "$MODE" == "--dry-run" ]]; then
    printf '[dry-run:%s] gpu=%s env=%s run_dir=%s\n' "$lane" "$gpu" "$env_id" "$run_dir"
    printf '  '
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return 0
  fi

  mkdir -p "$run_dir"
  common_env
  cd "$CODE_DIR"
  printf '[lane:%s] train_start gpu=%s env=%s cand=%s h=%s l=%s\n' "$lane" "$gpu" "$env_id" "$candidate_max" "$high_topk" "$low_topk" | tee -a "$status_path"
  set +e
  "${cmd[@]}" > "$train_log" 2>&1
  local train_status=$?
  set -e
  printf 'train_status\t%s\n' "$train_status" >> "$status_path"
  if [[ "$train_status" != "0" ]]; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$lane" "$group" "$env_id" "$candidate_max" "$high_topk" "$low_topk" "$train_status" "not_started" "missing" >> "$RESULTS_TSV"
    return "$train_status"
  fi

  local ckpt
  ckpt=$(find "$run_dir/checkpoints" -maxdepth 1 -type f -name 'epoch_*_train_mean_loss_*.ckpt' | sort -V | tail -1)
  if [[ -z "$ckpt" ]]; then
    printf 'eval_status\tmissing_ckpt\n' >> "$status_path"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$lane" "$group" "$env_id" "$candidate_max" "$high_topk" "$low_topk" "$train_status" "missing_ckpt" "missing" >> "$RESULTS_TSV"
    return 3
  fi
  printf 'trained_ckpt\t%s\n' "$ckpt" >> "$status_path"

  if [[ "$SKIP_EVAL" == "true" ]]; then
    printf 'eval_status\tskipped\n' >> "$status_path"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$lane" "$group" "$env_id" "$candidate_max" "$high_topk" "$low_topk" "$train_status" "skipped" "$ckpt" >> "$RESULTS_TSV"
    return 0
  fi

  if [[ "$SETTLE_SECONDS" != "0" ]]; then
    sleep "$SETTLE_SECONDS"
  fi
  printf '[lane:%s] eval_start gpu=%s ckpt=%s\n' "$lane" "$gpu" "$ckpt" | tee -a "$status_path"
  set +e
  SAPIEN_VULKAN_LIBRARY_PATH="$SAPIEN_BUILTIN_VULKAN" \
  ICRA_IMITATION_DIR="$CODE_DIR" \
    bash "$EVAL_SCRIPT" --launch "$env_id" "$ckpt" "$eval_label" "$EPISODES" "$ENV_NUM" "$gpu" "$EVAL_SEED" \
    > "$eval_log" 2>&1
  local eval_status=$?
  set -e
  local summary
  summary=$(awk -F= '$1=="SUMMARY" {print $2; exit}' "$eval_log" 2>/dev/null || true)
  if [[ -z "$summary" || ! -f "$summary" ]]; then
    summary=missing
  fi
  printf 'eval_status\t%s\n' "$eval_status" >> "$status_path"
  printf 'eval_log\t%s\n' "$eval_log" >> "$status_path"
  printf 'summary\t%s\n' "$summary" >> "$status_path"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$lane" "$group" "$env_id" "$candidate_max" "$high_topk" "$low_topk" "$train_status" "$eval_status" "$summary" >> "$RESULTS_TSV"
  return "$eval_status"
}

lane_specs=(
  "rc3_c8_h2_l2 main 0 33000 mikasa_remember_color_3 RememberColor3-v0 $RC3_CKPT 8 10 10 8 2 2 true"
  "rc5_c8_h2_l2 main 1 33001 mikasa_remember_color_5 RememberColor5-v0 $RC5_CKPT 8 10 10 8 2 2 true"
  "rc9_c8_h2_l2 main 2 33002 mikasa_remember_color_9 RememberColor9-v0 $RC9_CKPT 8 10 10 8 2 2 true"
  "intercept_c12_h2_l2 main 3 33003 mikasa_intercept_medium InterceptMedium-v0 $INTERCEPT_CKPT 12 8 8 12 2 2 false"
  "shell_c8_h2_l2 main 4 33004 mikasa_shell_game_touch ShellGameTouch-v0 $SHELL_CKPT 45 2 10 8 2 2 false"
  "rc5_c4_h1_l1 ablation 5 33005 mikasa_remember_color_5 RememberColor5-v0 $RC5_CKPT 8 10 10 4 1 1 true"
  "rc5_c8_h1_l1 ablation 6 33006 mikasa_remember_color_5 RememberColor5-v0 $RC5_CKPT 8 10 10 8 1 1 true"
  "rc5_c8_h1_l2 ablation 7 33007 mikasa_remember_color_5 RememberColor5-v0 $RC5_CKPT 8 10 10 8 1 2 true"
)

{
  printf 'lane\tgroup\tgpu\tpolicy\ttask_name\tenv_id\tcandidate_max\thigh_topk\tlow_topk\n'
  for spec in "${lane_specs[@]}"; do
    read -r lane group gpu _port task_name env_id _base _max_hist _action_len _traj_interval candidate_max high_topk low_topk _ignore <<< "$spec"
    if should_run_lane "$lane" "$group"; then
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$lane" "$group" "$gpu" "$POLICY_NAME" "$task_name" "$env_id" "$candidate_max" "$high_topk" "$low_topk"
    fi
  done
} > "$MANIFEST"
printf 'lane\tgroup\tenv_id\tcandidate_max\thigh_topk\tlow_topk\ttrain_status\teval_status\tsummary\n' > "$RESULTS_TSV"

echo "MODE=$MODE RUN_SET=$RUN_SET STAMP=$STAMP"
echo "CODE_DIR=$CODE_DIR"
echo "RUN_ROOT=$RUN_ROOT"
echo "LOG_ROOT=$LOG_ROOT"
echo "RESULTS_TSV=$RESULTS_TSV"
echo "NUM_EPOCHS=$NUM_EPOCHS TRAINER_DEBUG=$TRAINER_DEBUG LR=$LR TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE"
echo "SKIP_EVAL=$SKIP_EVAL EPISODES=$EPISODES ENV_NUM=$ENV_NUM EVAL_SEED=$EVAL_SEED"
echo "LANE_FILTER=$LANE_FILTER"

pids=()
for spec in "${lane_specs[@]}"; do
  read -r lane group _rest <<< "$spec"
  if ! should_run_lane "$lane" "$group"; then
    continue
  fi
  if [[ "$MODE" == "--dry-run" ]]; then
    # shellcheck disable=SC2086
    run_lane $spec
  else
    # shellcheck disable=SC2086
    ( run_lane $spec ) &
    pids+=("$!")
  fi
done

if [[ "$MODE" == "--dry-run" ]]; then
  echo "Dry run only. Re-run with MODE=--launch."
  exit 0
fi

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
echo "[qframe-evidence] completed status=$status log_root=$LOG_ROOT"
exit "$status"
