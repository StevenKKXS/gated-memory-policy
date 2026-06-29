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

METHOD_LABEL=selector
POLICY_NAME=diffusion_mikasa_visual_selector_late_anchor_memory
VISUAL_MEMORY_CARRIER_TYPE=selector
BASE_RC3="$ICRA_BASE/runs/mikasa_method_dev/20260609_2221visual_k4_v2/rc3_start_anchor_v3fbest_state_token_visualwriter_k4_b4_read001_gate2_s02_lr1e4_6ep_idx100_seed0_v2/checkpoints/epoch_3_train_mean_loss_0_000.ckpt"
BASE_RC5="$ICRA_BASE/runs/mikasa_method_dev/20260607_064830/rc5_dense_idx4_stepmatch_seed0/checkpoints/epoch_32_train_mean_loss_0_003.ckpt"
BASE_RC9="$ICRA_BASE/runs/mikasa_method_dev/20260607_064937/rc9_top1_idx4_seed2_retry/checkpoints/epoch_37_train_mean_loss_0_002.ckpt"
BASE_INTERCEPT="$ICRA_BASE/runs/mikasa_method_dev/20260607_053019/intercept_top1_seed0/checkpoints/epoch_8_train_mean_loss_0_003.ckpt"
BASE_SHELL="$ICRA_BASE/runs/mikasa_method_dev/20260607_064831/shell_dense_idx4_stepmatch_seed0/checkpoints/epoch_7_train_mean_loss_0_001.ckpt"
LANES=(
  $'rc3\t0\t32510\tmikasa_remember_color_3\tRememberColor3-v0\t'"$BASE_RC3"$'\t6\t5\t10\t10\t0\t0\trandom\tnone'
  $'rc5\t1\t32511\tmikasa_remember_color_5\tRememberColor5-v0\t'"$BASE_RC5"$'\t6\t5\t10\t10\t0\t0\trandom\tnone'
  $'rc9\t2\t32512\tmikasa_remember_color_9\tRememberColor9-v0\t'"$BASE_RC9"$'\t6\t5\t10\t10\t0\t0\trandom\tnone'
  $'intercept\t3\t32513\tmikasa_intercept_medium\tInterceptMedium-v0\t'"$BASE_INTERCEPT"$'\t12\t11\t8\t8\t0\t0\trandom\tnone'
  $'shell\t4\t32514\tmikasa_shell_game_touch\tShellGameTouch-v0\t'"$BASE_SHELL"$'\t45\t44\t2\t10\t0\t0\trandom\tnone'
)

source "$REPO_ROOT/experiments/mikasa_method_archive/launchers/_run_5task_common.sh"
run_archive_lanes "${1:-}"
