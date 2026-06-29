#!/usr/bin/env bash
set -euo pipefail

ICRA_BASE="${ICRA_BASE:-/mnt/3fs1/data/tingwen.du/icra_method_dev}"
PY="${PY:-$ICRA_BASE/envs/imitation-py310-h200-headless/bin/python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CODE_DIR="$REPO_ROOT/imitation-learning-policies"
export PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"

RUN_ROOT="$ICRA_BASE/runs/mikasa_method_archive"
LOG_ROOT="$ICRA_BASE/logs/mikasa_method_archive"
mkdir -p "$RUN_ROOT" "$LOG_ROOT"

METHOD_LABEL=top1_retrieval
POLICY_NAME=diffusion_mikasa_retrieval_memory_transformer
HISTORY_RETRIEVAL_TOPK=1
INCLUDE_ACTION_HISTORY=false
INDEX_POOL_SIZE_PER_EPISODE=4
LANES=(
  $'rc3\t0\t32410\tmikasa_remember_color_3\tRememberColor3-v0\tnone\t6\t5\t10\t10\t0\t0\trandom\t1'
  $'rc5\t1\t32411\tmikasa_remember_color_5\tRememberColor5-v0\tnone\t6\t5\t10\t10\t0\t0\trandom\t1'
  $'rc9\t2\t32412\tmikasa_remember_color_9\tRememberColor9-v0\tnone\t6\t5\t10\t10\t0\t0\trandom\t1'
  $'intercept\t3\t32413\tmikasa_intercept_medium\tInterceptMedium-v0\tnone\t12\t11\t8\t8\t0\t0\trandom\t1'
  $'shell\t4\t32414\tmikasa_shell_game_touch\tShellGameTouch-v0\tnone\t45\t44\t2\t10\t0\t0\trandom\t1'
)

source "$REPO_ROOT/experiments/mikasa_method_archive/launchers/_run_5task_common.sh"
run_archive_lanes "${1:-}"
