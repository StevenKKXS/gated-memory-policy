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
CODE_DIR=${CODE_DIR:-$MEM_DEV/code/imitation-learning-policies_anchor_carrier_clear_20260619}
PY=${PY:-$ICRA_BASE/envs/imitation-py310-h200-headless/bin/python}
EVAL_SCRIPT=${EVAL_SCRIPT:-$MEM_DEV/eval_mikasa_checkpoint.sh}
OPENCV_DIR=$ICRA_BASE/deps/opencv_headless_py310
GLVND_LIB=$ICRA_BASE/deps/glvnd_ubuntu24.04_amd64/root/usr/lib/x86_64-linux-gnu
SAPIEN_BUILTIN_VULKAN=${SAPIEN_BUILTIN_VULKAN:-/mnt/3fs1/data/tingwen.du/gated-memory-policy-envs/task003_robomimic_memory_gate_repro/mikasa-py311/lib/python3.11/site-packages/sapien/vulkan_library/libvulkan.so.1.3.224}

STAMP=${STAMP:-$(date -u +%Y%m%d_late_direct_anchor_5task)}
TRAIN_SEED=${TRAIN_SEED:-0}
EVAL_SEED=${EVAL_SEED:-42}
NUM_EPOCHS=${NUM_EPOCHS:-1}
TRAINER_DEBUG=${TRAINER_DEBUG:-false}
LR=${LR:-3e-5}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
SPLIT_BATCH_SIZE=${SPLIT_BATCH_SIZE:-64}
TRAIN_TRAJ_NUM=${TRAIN_TRAJ_NUM:-7}
TRAIN_EFFECTIVE_TRAJ_NUM=${TRAIN_EFFECTIVE_TRAJ_NUM:-$TRAIN_TRAJ_NUM}
INCLUDE_EPISODE_INDICES=${INCLUDE_EPISODE_INDICES:-'${range:0,250}'}
INDEX_POOL_SIZE_PER_EPISODE=${INDEX_POOL_SIZE_PER_EPISODE:-20}
NUM_WORKERS=${NUM_WORKERS:-0}
PERSISTENT_WORKERS=${PERSISTENT_WORKERS:-false}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-null}
EPISODES=${EPISODES:-100}
ENV_NUM=${ENV_NUM:-50}
SETTLE_SECONDS=${SETTLE_SECONDS:-0}
SKIP_EVAL=${SKIP_EVAL:-false}
LANE_FILTER=${LANE_FILTER:-}

LATE_CUE_ANCHOR_LEN=${LATE_CUE_ANCHOR_LEN:-1}
LATE_CUE_ANCHOR_CAUSAL_MASK=${LATE_CUE_ANCHOR_CAUSAL_MASK:-false}
LATE_CUE_ACTION_PREHEAD_ADAPTER_GATE_INIT_BIAS=${LATE_CUE_ACTION_PREHEAD_ADAPTER_GATE_INIT_BIAS:-0.0}
LATE_CUE_ACTION_PREHEAD_ADAPTER_OUT_INIT_GAIN=${LATE_CUE_ACTION_PREHEAD_ADAPTER_OUT_INIT_GAIN:-0.005}
LATE_CUE_ACTION_PREHEAD_ADAPTER_RESIDUAL_SCALE=${LATE_CUE_ACTION_PREHEAD_ADAPTER_RESIDUAL_SCALE:-0.5}
ANCHOR_ACTION_DELTA_LOSS_WEIGHT=${ANCHOR_ACTION_DELTA_LOSS_WEIGHT:-0.05}
ANCHOR_ACTION_DELTA_COSINE_LOSS_WEIGHT=${ANCHOR_ACTION_DELTA_COSINE_LOSS_WEIGHT:-0.0}
ANCHOR_ACTION_DELTA_ACTION_DELTA_QUANTILE=${ANCHOR_ACTION_DELTA_ACTION_DELTA_QUANTILE:-0.60}
ANCHOR_ACTION_DELTA_ACTION_DELTA_MIN=${ANCHOR_ACTION_DELTA_ACTION_DELTA_MIN:-0.0}
ANCHOR_ACTION_DELTA_MIN_TRAJ_INDEX=${ANCHOR_ACTION_DELTA_MIN_TRAJ_INDEX:-1}
ANCHOR_ACTION_DELTA_INCLUDE_CRITICAL=${ANCHOR_ACTION_DELTA_INCLUDE_CRITICAL:-false}
BASE_ANCHOR_LOSS_WEIGHT=${BASE_ANCHOR_LOSS_WEIGHT:-0.10}
BASE_ANCHOR_KEYWORDS=${BASE_ANCHOR_KEYWORDS:-'[denoising_network.blocks.14,denoising_network.blocks.15]'}
TRAINABLE_PARAM_KEYWORDS=${TRAINABLE_PARAM_KEYWORDS:-'[late_cue_action_prehead_adapter,denoising_network.blocks.14,denoising_network.blocks.15]'}
RECORD_DATA_ENTRIES=${RECORD_DATA_ENTRIES:-'[late_cue_action_prehead_adapter_gate,late_cue_action_prehead_adapter_valid_ratio,late_cue_action_prehead_adapter_delta_norm]'}

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
LOG_ROOT=$ICRA_BASE/logs/mikasa_method_dev/late_direct_anchor_5task_$STAMP
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

run_lane() {
  local lane=$1
  local gpu=$2
  local port=$3
  local task_name=$4
  local env_id=$5
  local policy_name=$6
  local base_ckpt=$7
  local max_history_len=$8
  local action_len=$9
  local history_action_num_per_chunk=${10}
  local traj_interval=${11}
  local retrieval_topk=${12}

  local lane_train_traj_num="$TRAIN_TRAJ_NUM"
  local min_traj_num=$((max_history_len + 1))
  if (( lane_train_traj_num < min_traj_num )); then
    lane_train_traj_num="$min_traj_num"
  fi

  local action_indices
  action_indices=$(action_indices_for_length "$action_len")
  local run_name="late_direct_anchor_${lane}_seed${TRAIN_SEED}"
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
    +policy_name="$policy_name"
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
    ++workspace.model.denoising_network_partial.late_cue_adapter_enabled=false
    ++workspace.model.denoising_network_partial.late_cue_anchor_enabled=true
    ++workspace.model.denoising_network_partial.late_cue_anchor_len="$LATE_CUE_ANCHOR_LEN"
    ++workspace.model.denoising_network_partial.late_cue_anchor_causal_mask="$LATE_CUE_ANCHOR_CAUSAL_MASK"
    ++workspace.model.denoising_network_partial.late_cue_action_prehead_adapter_enabled=true
    ++workspace.model.denoising_network_partial.late_cue_action_prehead_adapter_gate_init_bias="$LATE_CUE_ACTION_PREHEAD_ADAPTER_GATE_INIT_BIAS"
    ++workspace.model.denoising_network_partial.late_cue_action_prehead_adapter_out_init_gain="$LATE_CUE_ACTION_PREHEAD_ADAPTER_OUT_INIT_GAIN"
    ++workspace.model.denoising_network_partial.late_cue_action_prehead_adapter_residual_scale="$LATE_CUE_ACTION_PREHEAD_ADAPTER_RESIDUAL_SCALE"
    ++workspace.model.anchor_action_delta_loss_weight="$ANCHOR_ACTION_DELTA_LOSS_WEIGHT"
    ++workspace.model.anchor_action_delta_cosine_loss_weight="$ANCHOR_ACTION_DELTA_COSINE_LOSS_WEIGHT"
    ++workspace.model.anchor_action_delta_action_delta_quantile="$ANCHOR_ACTION_DELTA_ACTION_DELTA_QUANTILE"
    ++workspace.model.anchor_action_delta_action_delta_min="$ANCHOR_ACTION_DELTA_ACTION_DELTA_MIN"
    ++workspace.model.anchor_action_delta_min_traj_index="$ANCHOR_ACTION_DELTA_MIN_TRAJ_INDEX"
    ++workspace.model.anchor_action_delta_include_critical="$ANCHOR_ACTION_DELTA_INCLUDE_CRITICAL"
    ++workspace.model.base_anchor_loss_weight="$BASE_ANCHOR_LOSS_WEIGHT"
    ++workspace.model.base_anchor_keywords="$BASE_ANCHOR_KEYWORDS"
    ++workspace.model.denoising_network_partial.record_data_entries="$RECORD_DATA_ENTRIES"
    ++workspace.train_dataset.episode_starting_idx_max=1
    ++workspace.train_dataset.traj_num="$lane_train_traj_num"
    +workspace.model.max_training_traj_num="$TRAIN_EFFECTIVE_TRAJ_NUM"
    ++workspace.train_dataset.traj_interval_min="$traj_interval"
    ++workspace.train_dataset.traj_interval_max="$traj_interval"
    ++workspace.model.action_length="$action_len"
    "++workspace.model.action_indices=$action_indices"
    ++workspace.model.history_action_num_per_chunk="$history_action_num_per_chunk"
    ++workspace.model.denoising_network_partial.max_history_len="$max_history_len"
  )

  if [[ "$retrieval_topk" != "none" ]]; then
    cmd+=(++workspace.model.denoising_network_partial.history_retrieval_topk="$retrieval_topk")
  fi
  if [[ -n "$PREFETCH_FACTOR" ]]; then
    cmd+=(+workspace.train_dataset.dataloader_cfg.prefetch_factor="$PREFETCH_FACTOR")
    cmd+=(+workspace.train_dataset.split_dataloader_cfg.prefetch_factor="$PREFETCH_FACTOR")
  fi

  if [[ "$MODE" == "--dry-run" ]]; then
    printf '[dry-run:%s] gpu=%s env=%s policy=%s run_dir=%s\n' "$lane" "$gpu" "$env_id" "$policy_name" "$run_dir"
    printf '  '
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return 0
  fi

  mkdir -p "$run_dir" "$LOG_ROOT"
  {
    printf 'key\tvalue\n'
    printf 'lane\t%s\n' "$lane"
    printf 'gpu\t%s\n' "$gpu"
    printf 'port\t%s\n' "$port"
    printf 'task_name\t%s\n' "$task_name"
    printf 'env_id\t%s\n' "$env_id"
    printf 'policy_name\t%s\n' "$policy_name"
    printf 'base_ckpt\t%s\n' "$base_ckpt"
    printf 'max_history_len\t%s\n' "$max_history_len"
    printf 'train_traj_num\t%s\n' "$lane_train_traj_num"
    printf 'action_len\t%s\n' "$action_len"
    printf 'history_action_num_per_chunk\t%s\n' "$history_action_num_per_chunk"
    printf 'traj_interval\t%s\n' "$traj_interval"
    printf 'retrieval_topk\t%s\n' "$retrieval_topk"
    printf 'run_dir\t%s\n' "$run_dir"
    printf 'train_log\t%s\n' "$train_log"
    printf 'eval_label\t%s\n' "$eval_label"
    printf 'started_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$status_path"

  common_env
  cd "$CODE_DIR"
  printf '[lane:%s] training on gpu=%s env=%s\n' "$lane" "$gpu" "$env_id" | tee -a "$status_path"
  set +e
  "${cmd[@]}" > "$train_log" 2>&1
  local train_status=$?
  set -e
  printf 'train_status\t%s\n' "$train_status" >> "$status_path"
  if [[ "$train_status" != "0" ]]; then
    printf '[lane:%s] train failed status=%s log=%s\n' "$lane" "$train_status" "$train_log" | tee -a "$status_path"
    return "$train_status"
  fi

  local ckpt
  ckpt=$(find "$run_dir/checkpoints" -maxdepth 1 -type f -name 'epoch_*_train_mean_loss_*.ckpt' | sort -V | tail -1)
  if [[ -z "$ckpt" ]]; then
    printf 'eval_status\tmissing_ckpt\n' >> "$status_path"
    printf '[lane:%s] no checkpoint found under %s\n' "$lane" "$run_dir/checkpoints" | tee -a "$status_path"
    return 3
  fi
  printf 'trained_ckpt\t%s\n' "$ckpt" >> "$status_path"

  if [[ "$SKIP_EVAL" == "true" ]]; then
    printf 'eval_status\tskipped\n' >> "$status_path"
    printf 'ended_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$status_path"
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
  printf 'eval_status\t%s\n' "$eval_status" >> "$status_path"
  printf 'eval_log\t%s\n' "$eval_log" >> "$status_path"
  printf 'summary\t%s\n' "$summary" >> "$status_path"
  printf 'ended_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$status_path"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$lane" "$env_id" "$policy_name" "$train_status" "$eval_status" "$ckpt" "$summary" >> "$RESULTS_TSV"
  printf '[lane:%s] done train=%s eval=%s status=%s\n' "$lane" "$train_status" "$eval_status" "$status_path"
  return "$eval_status"
}

mkdir -p "$LOG_ROOT"
{
  printf 'lane\tgpu\tpolicy\ttask_name\tenv_id\tbase_ckpt\n'
  printf 'rc3\t0\tdiffusion_start_anchored_direct_anchor_memory\tmikasa_remember_color_3\tRememberColor3-v0\t%s\n' "$RC3_CKPT"
  printf 'rc5\t1\tdiffusion_start_anchored_direct_anchor_memory\tmikasa_remember_color_5\tRememberColor5-v0\t%s\n' "$RC5_CKPT"
  printf 'rc9\t2\tdiffusion_start_anchored_direct_anchor_memory\tmikasa_remember_color_9\tRememberColor9-v0\t%s\n' "$RC9_CKPT"
  printf 'intercept\t3\tdiffusion_start_anchored_direct_anchor_memory\tmikasa_intercept_medium\tInterceptMedium-v0\t%s\n' "$INTERCEPT_CKPT"
  printf 'shell\t4\tdiffusion_start_anchored_direct_anchor_memory\tmikasa_shell_game_touch\tShellGameTouch-v0\t%s\n' "$SHELL_CKPT"
} > "$MANIFEST"
printf 'lane\tenv_id\tpolicy\ttrain_status\teval_status\tckpt\tsummary\n' > "$RESULTS_TSV"

echo "MODE=$MODE"
echo "STAMP=$STAMP"
echo "CODE_DIR=$CODE_DIR"
echo "RUN_ROOT=$RUN_ROOT"
echo "LOG_ROOT=$LOG_ROOT"
echo "MANIFEST=$MANIFEST"
echo "RESULTS_TSV=$RESULTS_TSV"
echo "EPISODES=$EPISODES ENV_NUM=$ENV_NUM EVAL_SEED=$EVAL_SEED"
echo "TRAINER_DEBUG=$TRAINER_DEBUG SKIP_EVAL=$SKIP_EVAL LANE_FILTER=$LANE_FILTER"

launch_specs=(
  "rc3 0 31410 mikasa_remember_color_3 RememberColor3-v0 diffusion_start_anchored_direct_anchor_memory $RC3_CKPT 6 10 10 10 none"
  "rc5 1 31411 mikasa_remember_color_5 RememberColor5-v0 diffusion_start_anchored_direct_anchor_memory $RC5_CKPT 6 10 10 10 none"
  "rc9 2 31412 mikasa_remember_color_9 RememberColor9-v0 diffusion_start_anchored_direct_anchor_memory $RC9_CKPT 6 10 10 10 1"
  "intercept 3 31413 mikasa_intercept_medium InterceptMedium-v0 diffusion_start_anchored_direct_anchor_memory $INTERCEPT_CKPT 12 8 8 8 1"
  "shell 4 31414 mikasa_shell_game_touch ShellGameTouch-v0 diffusion_start_anchored_direct_anchor_memory $SHELL_CKPT 45 2 2 10 none"
)

pids=()
for spec in "${launch_specs[@]}"; do
  lane=${spec%% *}
  if [[ -n "$LANE_FILTER" && ",$LANE_FILTER," != *",$lane,"* ]]; then
    continue
  fi
  # shellcheck disable=SC2086
  if [[ "$MODE" == "--dry-run" ]]; then
    run_lane $spec
  else
    ( run_lane $spec ) &
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
echo "[late-direct-anchor-5task] completed status=$status log_root=$LOG_ROOT"
exit "$status"
