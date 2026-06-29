#!/usr/bin/env bash
set -euo pipefail

MODE=${1:---dry-run}
GPU_ID=${2:-0}
TASK_NAME=${3:-mikasa_remember_color_3}
ENV_ID=${4:-RememberColor3-v0}
RUN_LABEL=${5:-mikasa_task_sparse_retrieval_top1_seed0}
NUM_EPOCHS=${6:-51}
NUM_WORKERS=${7:-24}
POLICY_NAME=${POLICY_NAME:-diffusion_retrieval_memory_transformer}
EXTRA_ARGS=("${@:8}")

case "$MODE" in
  --dry-run|--launch) ;;
  *)
    echo "Usage: $0 [--dry-run|--launch] <gpu_id> <task_name> <env_id> <run_label> [num_epochs] [num_workers] [hydra_overrides...]"
    exit 1
    ;;
esac

case "$GPU_ID" in
  *[!0-9]*|"")
    echo "GPU id must be a single numeric id, got: $GPU_ID"
    exit 1
    ;;
esac

case "$ENV_ID" in
  RememberColor3-v0|RememberColor5-v0|RememberColor9-v0|InterceptMedium-v0|ShellGameTouch-v0) ;;
  *)
    echo "Unsupported MIKASA env id: $ENV_ID"
    exit 1
    ;;
esac

case "$RUN_LABEL" in
  *[!A-Za-z0-9_.-]*|"")
    echo "Run label may only contain letters, numbers, underscore, dash, or dot: $RUN_LABEL"
    exit 1
    ;;
esac

ICRA_BASE=/mnt/3fs1/data/tingwen.du/icra_method_dev
CODE_DIR=$ICRA_BASE/experiments/memory_method_dev/code/imitation-learning-policies_remdp
IMITATION_PY=/mnt/3fs1/data/tingwen.du/gated-memory-policy-envs/task003_robomimic_memory_gate_repro/imitation-py310/bin/python
ZARR_BASE=$ICRA_BASE/datasets/mikasa/zarr
OPENCV_DIR=$ICRA_BASE/deps/opencv_headless_py310
GLVND_LIB=$ICRA_BASE/deps/glvnd_ubuntu24.04_amd64/root/usr/lib/x86_64-linux-gnu

test -d "$CODE_DIR"
test -x "$IMITATION_PY"
test -d "$ZARR_BASE/$ENV_ID/episode_data.zarr"
test -f "$ZARR_BASE/${ENV_ID}_normalizer.json"
test -d "$ICRA_BASE/cache/hf-home/hub/models--google--siglip2-base-patch16-256"
test -d "$OPENCV_DIR/cv2"
test -d "$GLVND_LIB"

STAMP=$(date -u +%Y%m%d_%H%M%S)
OUT_DIR=$ICRA_BASE/runs/mikasa_method_dev/$STAMP/$RUN_LABEL
LOG_ROOT=$ICRA_BASE/logs/mikasa_method_dev/train_$STAMP
LOG_PATH=$LOG_ROOT/${RUN_LABEL}.train.log
MANIFEST=$LOG_ROOT/launch_manifest.tsv

export HF_HOME=$ICRA_BASE/cache/hf-home
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TORCH_HOME=$ICRA_BASE/cache/torch
export XDG_CACHE_HOME=$ICRA_BASE/cache/xdg
export MPLCONFIGDIR=$ICRA_BASE/cache/matplotlib
export LD_LIBRARY_PATH="$GLVND_LIB:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$OPENCV_DIR:$CODE_DIR:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

ARGS=(
  scripts/train_policy.py
  "+policy_name=$POLICY_NAME"
  "+task_name=$TASK_NAME"
  "+logger_project_name=icra_mikasa_method_dev"
  "+project_name=icra_method_dev"
  "+run_name=$RUN_LABEL"
  "+train_server_name=10.100.0.3"
  "+workspace.logging_cfg.mode=offline"
  "+workspace.trainer.output_dir=$OUT_DIR"
  "+workspace.trainer.num_epochs=$NUM_EPOCHS"
  "+workspace.trainer.rollout_every=0"
  "+workspace.train_dataset.compressed_dir=$ZARR_BASE"
  "+workspace.train_dataset.root_dir=$ZARR_BASE"
  "+workspace.train_dataset.normalizer_dir=$ZARR_BASE"
  "+workspace.train_dataset.name=$ENV_ID"
  "+workspace.train_dataset.starting_percentile_max=1.0"
  "+workspace.train_dataset.index_pool_size_per_episode=4"
  "+workspace.train_dataset.dataloader_cfg.num_workers=$NUM_WORKERS"
  "+workspace.train_dataset.dataloader_cfg.persistent_workers=true"
  "+workspace.train_dataset.split_dataloader_cfg.num_workers=$NUM_WORKERS"
  "+workspace.train_dataset.split_dataloader_cfg.persistent_workers=true"
  "+workspace.model.denoising_network_partial.include_action_history=false"
  "+workspace.model.denoising_network_partial.history_retrieval_topk=1"
)

cd "$CODE_DIR"

echo "MODE=$MODE"
echo "GPU_ID=$GPU_ID"
echo "TASK_NAME=$TASK_NAME"
echo "ENV_ID=$ENV_ID"
echo "RUN_LABEL=$RUN_LABEL"
echo "POLICY_NAME=$POLICY_NAME"
echo "NUM_EPOCHS=$NUM_EPOCHS"
echo "NUM_WORKERS=$NUM_WORKERS"
if (( ${#EXTRA_ARGS[@]} > 0 )); then
  printf "EXTRA_ARGS:"
  printf " %q" "${EXTRA_ARGS[@]}"
  printf "\n"
fi
echo "OUT_DIR=$OUT_DIR"
echo "LOG_PATH=$LOG_PATH"
printf "Command:"
printf " %q" "$IMITATION_PY" -u "${ARGS[@]}" "${EXTRA_ARGS[@]}"
printf "\n"

if [[ "$MODE" == "--dry-run" ]]; then
  echo "Dry run only."
  exit 0
fi

mkdir -p "$OUT_DIR" "$LOG_ROOT"
printf "task\tenv_id\tpolicy\trun_label\tgpu_id\tpid\tout_dir\tlog_path\tnum_epochs\tnum_workers\tcode_dir\n" > "$MANIFEST"

(
  exec "$IMITATION_PY" -u "${ARGS[@]}" "${EXTRA_ARGS[@]}"
) > "$LOG_PATH" 2>&1 &
PID=$!

printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
  "$TASK_NAME" "$ENV_ID" "$POLICY_NAME" "$RUN_LABEL" "$GPU_ID" "$PID" "$OUT_DIR" "$LOG_PATH" "$NUM_EPOCHS" "$NUM_WORKERS" "$CODE_DIR" >> "$MANIFEST"

echo "[train] pid=$PID gpu=$GPU_ID task=$TASK_NAME env=$ENV_ID out_dir=$OUT_DIR log=$LOG_PATH manifest=$MANIFEST"
