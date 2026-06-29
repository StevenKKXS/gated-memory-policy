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

STAMP=${STAMP:-$(date -u +%Y%m%d_no_memory_diffusion_5task)}
TRAIN_SEED=${TRAIN_SEED:-0}
EVAL_SEED=${EVAL_SEED:-42}
NUM_EPOCHS=${NUM_EPOCHS:-21}
TRAINER_DEBUG=${TRAINER_DEBUG:-false}
LR=${LR:-3e-4}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
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

test -d "$CODE_DIR"
test -x "$PY"
test -f "$EVAL_SCRIPT"
test -d "$OPENCV_DIR/cv2"
test -d "$GLVND_LIB"
test -f "$SAPIEN_BUILTIN_VULKAN"

RUN_ROOT=$ICRA_BASE/runs/mikasa_method_dev/$STAMP
LOG_ROOT=$ICRA_BASE/logs/mikasa_method_dev/no_memory_diffusion_5task_$STAMP
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
  local action_len=$6

  local action_indices
  action_indices=$(action_indices_for_length "$action_len")
  local run_name="no_memory_diffusion_${lane}_seed${TRAIN_SEED}"
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
    +policy_name=diffusion_transformer
    +task_name="$task_name"
    +logger_project_name=icra_mikasa_method_dev
    +project_name=icra_method_dev
    +run_name="$run_name"
    +train_server_name=10.100.0.3
    +seed="$TRAIN_SEED"
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
    ++workspace.train_dataset.episode_starting_idx_max=1
    ++workspace.model.action_length="$action_len"
    "++workspace.model.action_indices=$action_indices"
    +workspace.model.denoising_network_partial.include_action_history=false
  )

  if [[ -n "$PREFETCH_FACTOR" ]]; then
    cmd+=(+workspace.train_dataset.dataloader_cfg.prefetch_factor="$PREFETCH_FACTOR")
  fi

  if [[ "$MODE" == "--dry-run" ]]; then
    printf '[dry-run:%s] gpu=%s env=%s policy=diffusion_transformer run_dir=%s\n' "$lane" "$gpu" "$env_id" "$run_dir"
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
    printf 'policy_name\tdiffusion_transformer\n'
    printf 'action_len\t%s\n' "$action_len"
    printf 'run_dir\t%s\n' "$run_dir"
    printf 'train_log\t%s\n' "$train_log"
    printf 'eval_label\t%s\n' "$eval_label"
    printf 'started_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$status_path"

  common_env
  cd "$CODE_DIR"
  printf '[lane:%s] training no-history diffusion on gpu=%s env=%s\n' "$lane" "$gpu" "$env_id" | tee -a "$status_path"
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
  printf '%s\t%s\tdiffusion_transformer\t%s\t%s\t%s\t%s\n' "$lane" "$env_id" "$train_status" "$eval_status" "$ckpt" "$summary" >> "$RESULTS_TSV"
  printf '[lane:%s] done train=%s eval=%s status=%s\n' "$lane" "$train_status" "$eval_status" "$status_path"
  return "$eval_status"
}

mkdir -p "$LOG_ROOT"
{
  printf 'lane\tgpu\tpolicy\ttask_name\tenv_id\taction_len\n'
  printf 'rc3\t0\tdiffusion_transformer\tmikasa_remember_color_3\tRememberColor3-v0\t10\n'
  printf 'rc5\t1\tdiffusion_transformer\tmikasa_remember_color_5\tRememberColor5-v0\t10\n'
  printf 'rc9\t2\tdiffusion_transformer\tmikasa_remember_color_9\tRememberColor9-v0\t10\n'
  printf 'intercept\t5\tdiffusion_transformer\tmikasa_intercept_medium\tInterceptMedium-v0\t8\n'
  printf 'shell\t6\tdiffusion_transformer\tmikasa_shell_game_touch\tShellGameTouch-v0\t2\n'
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
echo "NUM_EPOCHS=$NUM_EPOCHS LR=$LR TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE"
echo "TRAINER_DEBUG=$TRAINER_DEBUG SKIP_EVAL=$SKIP_EVAL LANE_FILTER=$LANE_FILTER"

launch_specs=(
  "rc3 0 31510 mikasa_remember_color_3 RememberColor3-v0 10"
  "rc5 1 31511 mikasa_remember_color_5 RememberColor5-v0 10"
  "rc9 2 31512 mikasa_remember_color_9 RememberColor9-v0 10"
  "intercept 5 31513 mikasa_intercept_medium InterceptMedium-v0 8"
  "shell 6 31514 mikasa_shell_game_touch ShellGameTouch-v0 2"
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
echo "[no-memory-diffusion-5task] completed status=$status log_root=$LOG_ROOT"
exit "$status"
