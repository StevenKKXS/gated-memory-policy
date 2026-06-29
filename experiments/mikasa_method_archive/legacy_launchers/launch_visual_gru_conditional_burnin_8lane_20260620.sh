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

ICRA_BASE=/mnt/3fs1/data/tingwen.du/icra_method_dev
MEM_DEV=$ICRA_BASE/experiments/memory_method_dev
CODE_DIR=${CODE_DIR:-$MEM_DEV/code/imitation-learning-policies_visual_gru_carrier_clean_burnin_20260620}
PY=${PY:-$ICRA_BASE/envs/imitation-py310-h200-headless/bin/python}
EVAL_SCRIPT=${EVAL_SCRIPT:-$MEM_DEV/eval_mikasa_checkpoint.sh}
OPENCV_DIR=$ICRA_BASE/deps/opencv_headless_py310
GLVND_LIB=$ICRA_BASE/deps/glvnd_ubuntu24.04_amd64/root/usr/lib/x86_64-linux-gnu
SAPIEN_BUILTIN_VULKAN=${SAPIEN_BUILTIN_VULKAN:-/mnt/3fs1/data/tingwen.du/gated-memory-policy-envs/task003_robomimic_memory_gate_repro/mikasa-py311/lib/python3.11/site-packages/sapien/vulkan_library/libvulkan.so.1.3.224}

METHOD_LABEL=visual_gru_conditional_burnin
POLICY_NAME=diffusion_visual_gru_late_anchor_memory
STAMP=${STAMP:-$(date -u +%Y%m%d_${METHOD_LABEL}_8lane)}
TRAIN_SEED=${TRAIN_SEED:-0}
EVAL_SEED=${EVAL_SEED:-42}
NUM_EPOCHS=${NUM_EPOCHS:-1}
TRAINER_DEBUG=${TRAINER_DEBUG:-false}
LR=${LR:-3e-5}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
SPLIT_BATCH_SIZE=${SPLIT_BATCH_SIZE:-64}
MAX_TRAINING_TRAJ_NUM=${MAX_TRAINING_TRAJ_NUM:-7}
INCLUDE_EPISODE_INDICES=${INCLUDE_EPISODE_INDICES:-'${range:0,250}'}
INDEX_POOL_SIZE_PER_EPISODE=${INDEX_POOL_SIZE_PER_EPISODE:-20}
NUM_WORKERS=${NUM_WORKERS:-0}
PERSISTENT_WORKERS=${PERSISTENT_WORKERS:-false}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-null}
EPISODES=${EPISODES:-100}
ENV_NUM=${ENV_NUM:-20}
SETTLE_SECONDS=${SETTLE_SECONDS:-0}
SKIP_EVAL=${SKIP_EVAL:-false}
LANE_FILTER=${LANE_FILTER:-}
GPU_BASE=${GPU_BASE:-0}

ANCHOR_ACTION_DELTA_LOSS_WEIGHT=${ANCHOR_ACTION_DELTA_LOSS_WEIGHT:-0.05}
ANCHOR_ACTION_DELTA_COSINE_LOSS_WEIGHT=${ANCHOR_ACTION_DELTA_COSINE_LOSS_WEIGHT:-0.0}
ANCHOR_ACTION_DELTA_ACTION_DELTA_QUANTILE=${ANCHOR_ACTION_DELTA_ACTION_DELTA_QUANTILE:-0.60}
ANCHOR_ACTION_DELTA_ACTION_DELTA_MIN=${ANCHOR_ACTION_DELTA_ACTION_DELTA_MIN:-0.0}
ANCHOR_ACTION_DELTA_MIN_TRAJ_INDEX=${ANCHOR_ACTION_DELTA_MIN_TRAJ_INDEX:-1}
ANCHOR_ACTION_DELTA_INCLUDE_CRITICAL=${ANCHOR_ACTION_DELTA_INCLUDE_CRITICAL:-false}
BASE_ANCHOR_LOSS_WEIGHT=${BASE_ANCHOR_LOSS_WEIGHT:-0.10}
BASE_ANCHOR_KEYWORDS=${BASE_ANCHOR_KEYWORDS:-'[denoising_network.blocks.14,denoising_network.blocks.15]'}
TRAINABLE_PARAM_KEYWORDS=${TRAINABLE_PARAM_KEYWORDS:-'[visual_memory_carrier,late_cue_action_prehead_adapter,denoising_network.blocks.14,denoising_network.blocks.15]'}

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

RUN_ROOT=$ICRA_BASE/runs/mikasa_method_dev/$STAMP
LOG_ROOT=$ICRA_BASE/logs/mikasa_method_dev/${METHOD_LABEL}_$STAMP
MANIFEST=$LOG_ROOT/launch_manifest.tsv
RESULTS_TSV=$LOG_ROOT/results.tsv

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

gpu_for_slot() {
  local slot=$1
  printf '%s\n' "$((GPU_BASE + slot))"
}

conditional_enabled() {
  local burn_in_loss_traj_num=$1
  if (( burn_in_loss_traj_num > 0 )); then
    printf 'true\n'
  else
    printf 'false\n'
  fi
}

run_lane() {
  local lane=$1
  local gpu=$2
  local port=$3
  local task_name=$4
  local env_id=$5
  local base_ckpt=$6
  local traj_num=$7
  local max_history_len=$8
  local action_len=$9
  local traj_interval=${10}
  local burn_in_start_id=${11}
  local burn_in_loss_traj_num=${12}
  local retrieval_topk=${13}
  local training_traj_sampling_strategy=${14}

  local history_action_num_per_chunk="$action_len"
  local conditional_burn_in_enabled
  conditional_burn_in_enabled=$(conditional_enabled "$burn_in_loss_traj_num")
  local action_indices
  action_indices=$(action_indices_for_length "$action_len")
  local run_name="${METHOD_LABEL}_${lane}_seed${TRAIN_SEED}"
  local run_dir="$RUN_ROOT/$run_name"
  local train_log="$LOG_ROOT/$run_name.train.log"
  local eval_label="${run_name}_seed${EVAL_SEED}"
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
    +logger_project_name=icra_mikasa_method_dev
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
    +workspace.train_dataset.compressed_dir=$ICRA_BASE/datasets/mikasa/zarr
    +workspace.train_dataset.root_dir=$ICRA_BASE/datasets/mikasa/zarr
    +workspace.train_dataset.normalizer_dir=$ICRA_BASE/datasets/mikasa/zarr
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
    ++workspace.model.trainable_param_keywords="$TRAINABLE_PARAM_KEYWORDS"
    +workspace.model.denoising_network_partial.include_action_history=false
    ++workspace.model.anchor_action_delta_loss_weight="$ANCHOR_ACTION_DELTA_LOSS_WEIGHT"
    ++workspace.model.anchor_action_delta_cosine_loss_weight="$ANCHOR_ACTION_DELTA_COSINE_LOSS_WEIGHT"
    ++workspace.model.anchor_action_delta_action_delta_quantile="$ANCHOR_ACTION_DELTA_ACTION_DELTA_QUANTILE"
    ++workspace.model.anchor_action_delta_action_delta_min="$ANCHOR_ACTION_DELTA_ACTION_DELTA_MIN"
    ++workspace.model.anchor_action_delta_min_traj_index="$ANCHOR_ACTION_DELTA_MIN_TRAJ_INDEX"
    ++workspace.model.anchor_action_delta_include_critical="$ANCHOR_ACTION_DELTA_INCLUDE_CRITICAL"
    ++workspace.model.base_anchor_loss_weight="$BASE_ANCHOR_LOSS_WEIGHT"
    ++workspace.model.base_anchor_keywords="$BASE_ANCHOR_KEYWORDS"
    ++workspace.train_dataset.traj_num="$traj_num"
    +workspace.model.max_training_traj_num="$MAX_TRAINING_TRAJ_NUM"
    ++workspace.model.training_traj_sampling_strategy="$training_traj_sampling_strategy"
    ++workspace.train_dataset.traj_interval_min="$traj_interval"
    ++workspace.train_dataset.traj_interval_max="$traj_interval"
    ++workspace.model.action_length="$action_len"
    "++workspace.model.action_indices=$action_indices"
    ++workspace.model.history_action_num_per_chunk="$history_action_num_per_chunk"
    ++workspace.model.denoising_network_partial.max_history_len="$max_history_len"
    ++workspace.model.burn_in_start_id="$burn_in_start_id"
    ++workspace.model.burn_in_loss_traj_num="$burn_in_loss_traj_num"
  )

  if [[ "$retrieval_topk" != "none" ]]; then
    cmd+=(++workspace.model.denoising_network_partial.history_retrieval_topk="$retrieval_topk")
  fi
  if [[ -n "$PREFETCH_FACTOR" ]]; then
    cmd+=(+workspace.train_dataset.dataloader_cfg.prefetch_factor="$PREFETCH_FACTOR")
    cmd+=(+workspace.train_dataset.split_dataloader_cfg.prefetch_factor="$PREFETCH_FACTOR")
  fi

  if [[ "$MODE" == "--dry-run" ]]; then
    printf '[dry-run:%s] gpu=%s env=%s traj_num=%s max_history_len=%s burn_start=%s burn_loss=%s conditional=%s run_dir=%s\n' \
      "$lane" "$gpu" "$env_id" "$traj_num" "$max_history_len" "$burn_in_start_id" "$burn_in_loss_traj_num" "$conditional_burn_in_enabled" "$run_dir"
    printf '  '
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return 0
  fi

  mkdir -p "$run_dir" "$LOG_ROOT"
  {
    printf 'key\tvalue\n'
    printf 'method\t%s\n' "$METHOD_LABEL"
    printf 'lane\t%s\n' "$lane"
    printf 'gpu\t%s\n' "$gpu"
    printf 'port\t%s\n' "$port"
    printf 'task_name\t%s\n' "$task_name"
    printf 'env_id\t%s\n' "$env_id"
    printf 'policy_name\t%s\n' "$POLICY_NAME"
    printf 'base_ckpt\t%s\n' "$base_ckpt"
    printf 'traj_num\t%s\n' "$traj_num"
    printf 'max_history_len\t%s\n' "$max_history_len"
    printf 'traj_interval\t%s\n' "$traj_interval"
    printf 'action_len\t%s\n' "$action_len"
    printf 'history_action_num_per_chunk\t%s\n' "$history_action_num_per_chunk"
    printf 'burn_in_start_id\t%s\n' "$burn_in_start_id"
    printf 'burn_in_loss_traj_num\t%s\n' "$burn_in_loss_traj_num"
    printf 'conditional_burn_in_enabled\t%s\n' "$conditional_burn_in_enabled"
    printf 'max_training_traj_num\t%s\n' "$MAX_TRAINING_TRAJ_NUM"
    printf 'training_traj_sampling_strategy\t%s\n' "$training_traj_sampling_strategy"
    printf 'retrieval_topk\t%s\n' "$retrieval_topk"
    printf 'run_dir\t%s\n' "$run_dir"
    printf 'train_log\t%s\n' "$train_log"
    printf 'eval_label\t%s\n' "$eval_label"
    printf 'started_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$status_path"

  common_env
  cd "$CODE_DIR"
  printf '[lane:%s] training on gpu=%s env=%s conditional=%s\n' "$lane" "$gpu" "$env_id" "$conditional_burn_in_enabled" | tee -a "$status_path"
  set +e
  "${cmd[@]}" > "$train_log" 2>&1
  local train_status=$?
  set -e
  printf 'train_status\t%s\n' "$train_status" >> "$status_path"
  if [[ "$train_status" != "0" ]]; then
    printf '[lane:%s] train failed status=%s log=%s\n' "$lane" "$train_status" "$train_log" | tee -a "$status_path"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$lane" "$env_id" "$POLICY_NAME" "$train_status" "not_started" "missing" "missing" "false" >> "$RESULTS_TSV"
    return "$train_status"
  fi

  local ckpt
  ckpt=$(find "$run_dir/checkpoints" -maxdepth 1 -type f -name 'epoch_*_train_mean_loss_*.ckpt' | sort -V | tail -1)
  if [[ -z "$ckpt" ]]; then
    printf 'eval_status\tmissing_ckpt\n' >> "$status_path"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$lane" "$env_id" "$POLICY_NAME" "$train_status" "missing_ckpt" "missing" "missing" "false" >> "$RESULTS_TSV"
    return 3
  fi
  printf 'trained_ckpt\t%s\n' "$ckpt" >> "$status_path"

  if [[ "$SKIP_EVAL" == "true" ]]; then
    printf 'eval_status\tskipped\n' >> "$status_path"
    printf 'ended_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$status_path"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$lane" "$env_id" "$POLICY_NAME" "$train_status" "skipped" "$ckpt" "missing" "train_only" >> "$RESULTS_TSV"
    return 0
  fi

  if [[ "$SETTLE_SECONDS" != "0" ]]; then
    sleep "$SETTLE_SECONDS"
  fi
  printf '[lane:%s] eval on gpu=%s ckpt=%s\n' "$lane" "$gpu" "$ckpt" | tee -a "$status_path"
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
  local success=false
  if [[ "$eval_status" == "0" && "$summary" != "missing" ]]; then
    success=true
  fi
  printf 'eval_status\t%s\n' "$eval_status" >> "$status_path"
  printf 'eval_log\t%s\n' "$eval_log" >> "$status_path"
  printf 'summary\t%s\n' "$summary" >> "$status_path"
  printf 'success\t%s\n' "$success" >> "$status_path"
  printf 'ended_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$status_path"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$lane" "$env_id" "$POLICY_NAME" "$train_status" "$eval_status" "$ckpt" "$summary" "$success" >> "$RESULTS_TSV"
  return "$eval_status"
}

mkdir -p "$LOG_ROOT"
{
  printf 'lane\tgpu\tpolicy\ttask_name\tenv_id\tbase_ckpt\ttraj_num\tmax_history_len\ttraj_interval\taction_len\tburn_in_start_id\tburn_in_loss_traj_num\tconditional_burn_in_enabled\tmax_training_traj_num\ttraining_traj_sampling_strategy\n'
} > "$MANIFEST"
printf 'lane\tenv_id\tpolicy\ttrain_status\teval_status\tckpt\tsummary\tsuccess\n' > "$RESULTS_TSV"

echo "MODE=$MODE"
echo "POLICY_NAME=$POLICY_NAME"
echo "STAMP=$STAMP"
echo "CODE_DIR=$CODE_DIR"
echo "RUN_ROOT=$RUN_ROOT"
echo "LOG_ROOT=$LOG_ROOT"
echo "MANIFEST=$MANIFEST"
echo "RESULTS_TSV=$RESULTS_TSV"
echo "GPU_BASE=$GPU_BASE"
echo "EPISODES=$EPISODES ENV_NUM=$ENV_NUM EVAL_SEED=$EVAL_SEED SKIP_EVAL=$SKIP_EVAL"
echo "NUM_EPOCHS=$NUM_EPOCHS LR=$LR TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE MAX_TRAINING_TRAJ_NUM=$MAX_TRAINING_TRAJ_NUM"
echo "TRAINER_DEBUG=$TRAINER_DEBUG LANE_FILTER=$LANE_FILTER"

launch_specs=(
  "rc3_nonburn $(gpu_for_slot 0) 31900 mikasa_remember_color_3 RememberColor3-v0 $RC3_CKPT 7 6 10 10 8 0 none random"
  "rc3_burn $(gpu_for_slot 1) 31901 mikasa_remember_color_3 RememberColor3-v0 $RC3_CKPT 7 6 10 10 8 2 none tail"
  "rc5_nonburn $(gpu_for_slot 2) 31902 mikasa_remember_color_5 RememberColor5-v0 $RC5_CKPT 7 6 10 10 8 0 none random"
  "rc5_burn $(gpu_for_slot 3) 31903 mikasa_remember_color_5 RememberColor5-v0 $RC5_CKPT 7 6 10 10 8 2 none tail"
  "rc9_nonburn $(gpu_for_slot 4) 31904 mikasa_remember_color_9 RememberColor9-v0 $RC9_CKPT 7 6 10 10 8 0 1 random"
  "rc9_burn $(gpu_for_slot 5) 31905 mikasa_remember_color_9 RememberColor9-v0 $RC9_CKPT 7 6 10 10 8 2 1 tail"
  "intercept_burn $(gpu_for_slot 6) 31906 mikasa_intercept_medium InterceptMedium-v0 $INTERCEPT_CKPT 13 12 8 8 16 4 1 tail"
  "shell_burn $(gpu_for_slot 7) 31907 mikasa_shell_game_touch ShellGameTouch-v0 $SHELL_CKPT 46 45 2 10 8 8 none tail"
)

pids=()
for spec in "${launch_specs[@]}"; do
  lane=${spec%% *}
  # shellcheck disable=SC2086
  set -- $spec
  lane=$1
  gpu=$2
  port=$3
  task_name=$4
  env_id=$5
  base_ckpt=$6
  traj_num=$7
  max_history_len=$8
  action_len=$9
  traj_interval=${10}
  burn_in_start_id=${11}
  burn_in_loss_traj_num=${12}
  retrieval_topk=${13}
  training_traj_sampling_strategy=${14}
  conditional_burn_in_enabled=$(conditional_enabled "$burn_in_loss_traj_num")
  if [[ -n "$LANE_FILTER" && ",$LANE_FILTER," != *",$lane,"* ]]; then
    continue
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$lane" "$gpu" "$POLICY_NAME" "$task_name" "$env_id" "$base_ckpt" "$traj_num" "$max_history_len" "$traj_interval" "$action_len" \
    "$burn_in_start_id" "$burn_in_loss_traj_num" "$conditional_burn_in_enabled" "$MAX_TRAINING_TRAJ_NUM" "$training_traj_sampling_strategy" >> "$MANIFEST"

  if [[ "$MODE" == "--dry-run" ]]; then
    run_lane "$lane" "$gpu" "$port" "$task_name" "$env_id" "$base_ckpt" "$traj_num" "$max_history_len" "$action_len" "$traj_interval" "$burn_in_start_id" "$burn_in_loss_traj_num" "$retrieval_topk" "$training_traj_sampling_strategy"
  else
    ( run_lane "$lane" "$gpu" "$port" "$task_name" "$env_id" "$base_ckpt" "$traj_num" "$max_history_len" "$action_len" "$traj_interval" "$burn_in_start_id" "$burn_in_loss_traj_num" "$retrieval_topk" "$training_traj_sampling_strategy" ) &
    pids+=("$!")
  fi
done

if [[ "$MODE" == "--dry-run" ]]; then
  echo "Dry run only. Re-run with MODE=--launch on the GPU host."
  exit 0
fi

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
echo "[visual-gru-conditional-burnin-8lane] completed status=$status log_root=$LOG_ROOT"
exit "$status"
