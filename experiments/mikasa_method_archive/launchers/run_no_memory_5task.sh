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

METHOD_LABEL=no_memory
POLICY_NAME=diffusion_transformer
INCLUDE_ACTION_HISTORY=false
EPISODE_STARTING_IDX_MAX=1
LANES=(
  $'rc3\t0\t32310\tmikasa_remember_color_3\tRememberColor3-v0\tnone\t1\t0\t10\t10\t0\t0\trandom\tnone'
  $'rc5\t1\t32311\tmikasa_remember_color_5\tRememberColor5-v0\tnone\t1\t0\t10\t10\t0\t0\trandom\tnone'
  $'rc9\t2\t32312\tmikasa_remember_color_9\tRememberColor9-v0\tnone\t1\t0\t10\t10\t0\t0\trandom\tnone'
  $'intercept\t3\t32313\tmikasa_intercept_medium\tInterceptMedium-v0\tnone\t1\t0\t8\t8\t0\t0\trandom\tnone'
  $'shell\t4\t32314\tmikasa_shell_game_touch\tShellGameTouch-v0\tnone\t1\t0\t2\t10\t0\t0\trandom\tnone'
)

source "$REPO_ROOT/experiments/mikasa_method_archive/launchers/_run_5task_common.sh"
run_archive_lanes "${1:-}"
