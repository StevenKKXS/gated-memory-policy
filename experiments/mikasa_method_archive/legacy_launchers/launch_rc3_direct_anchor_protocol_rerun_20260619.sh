#!/usr/bin/env bash
set -euo pipefail

ICRA_BASE=/mnt/3fs1/data/tingwen.du/icra_method_dev
MEM_DEV=$ICRA_BASE/experiments/memory_method_dev
WRAPPER=$MEM_DEV/eval_mikasa_checkpoint.sh

RUN_STAMP=${RUN_STAMP:-$(date -u +%Y%m%d_%H%M%S)}
RUN_DIR=${RUN_DIR:-$MEM_DEV/diagnostics/rc3_direct_anchor_protocol_rerun_${RUN_STAMP}}
LOG_DIR=$RUN_DIR/logs
RESULTS_TSV=$RUN_DIR/results.tsv

DIRECT_ANCHOR_CODE=${DIRECT_ANCHOR_CODE:-$MEM_DEV/code/imitation-learning-policies_remdp_direct_anchor_20260611}
DIRECT_ANCHOR_CKPT=${DIRECT_ANCHOR_CKPT:-$ICRA_BASE/runs/mikasa_method_dev/20260611_direct_anchor_stability_axis_idx20_fix1/rc3_da_stab_blocks_w05_ba01_idx20_ep250_seed0_v1/checkpoints/epoch_0_train_mean_loss_0_008.ckpt}
SAPIEN_VULKAN_LIBRARY_PATH_VALUE=${SAPIEN_VULKAN_LIBRARY_PATH_VALUE:-/mnt/3fs1/data/tingwen.du/gated-memory-policy-envs/task003_robomimic_memory_gate_repro/mikasa-py311/lib/python3.11/site-packages/sapien/vulkan_library/libvulkan.so.1.3.224}

EPISODES=${EPISODES:-100}
SEED=${SEED:-42}
GLOBAL_WATCHDOG_SECONDS=${GLOBAL_WATCHDOG_SECONDS:-3300}
PER_EVAL_TIMEOUT_SECONDS=${PER_EVAL_TIMEOUT_SECONDS:-3300}
GPU_ENV50=${GPU_ENV50:-4}
GPU_ENV10=${GPU_ENV10:-7}

mkdir -p "$LOG_DIR"
test -x "$WRAPPER"
test -d "$DIRECT_ANCHOR_CODE"
test -f "$DIRECT_ANCHOR_CKPT"
test -f "$SAPIEN_VULKAN_LIBRARY_PATH_VALUE"

{
  printf 'run_dir\t%s\n' "$RUN_DIR"
  printf 'hostname\t%s\n' "$(hostname)"
  printf 'started_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'run_stamp\t%s\n' "$RUN_STAMP"
  printf 'episodes\t%s\n' "$EPISODES"
  printf 'seed\t%s\n' "$SEED"
  printf 'global_watchdog_seconds\t%s\n' "$GLOBAL_WATCHDOG_SECONDS"
  printf 'per_eval_timeout_seconds\t%s\n' "$PER_EVAL_TIMEOUT_SECONDS"
  printf 'direct_anchor_code\t%s\n' "$DIRECT_ANCHOR_CODE"
  printf 'direct_anchor_ckpt\t%s\n' "$DIRECT_ANCHOR_CKPT"
  printf 'sapien_vulkan_library_path\t%s\n' "$SAPIEN_VULKAN_LIBRARY_PATH_VALUE"
  printf 'gpu_env50\t%s\n' "$GPU_ENV50"
  printf 'gpu_env10\t%s\n' "$GPU_ENV10"
} > "$RUN_DIR/invocation.tsv"

printf 'kind\tlabel\tgpu\tenv_num\tstatus\twall_seconds\tsummary\tlog_path\tstarted_utc\tended_utc\n' > "$RESULTS_TSV"

kill_tree() {
  local root="$1"
  local child
  for child in $(pgrep -P "$root" 2>/dev/null || true); do
    kill_tree "$child"
  done
  kill -TERM "$root" 2>/dev/null || true
}

kill_tree_hard() {
  local root="$1"
  local child
  for child in $(pgrep -P "$root" 2>/dev/null || true); do
    kill_tree_hard "$child"
  done
  kill -KILL "$root" 2>/dev/null || true
}

run_one() {
  local kind="$1"
  local env_num="$2"
  local gpu="$3"
  shift 3
  local extra_overrides=("$@")
  local label="rc3_direct_anchor_protocol_${kind}_seed${SEED}_ep${EPISODES}_env${env_num}_gpu${gpu}"
  local log_path="$LOG_DIR/${label}.outer.log"
  local started start_s status end_s ended summary
  started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  start_s=$(date +%s)
  set +e
  SAPIEN_VULKAN_LIBRARY_PATH="$SAPIEN_VULKAN_LIBRARY_PATH_VALUE" \
  ICRA_IMITATION_DIR="$DIRECT_ANCHOR_CODE" \
  PYTHONPATH="$DIRECT_ANCHOR_CODE:${PYTHONPATH:-}" \
    timeout -k 60s "${PER_EVAL_TIMEOUT_SECONDS}s" \
    bash "$WRAPPER" --launch RememberColor3-v0 "$DIRECT_ANCHOR_CKPT" "$label" "$EPISODES" "$env_num" "$gpu" "$SEED" \
      "${extra_overrides[@]}" \
    > "$log_path" 2>&1
  status=$?
  set -e
  end_s=$(date +%s)
  ended=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  summary=$(awk -F= '$1=="SUMMARY" {print $2; exit}' "$log_path" 2>/dev/null || true)
  if [[ -z "$summary" || ! -f "$summary" ]]; then
    summary=missing
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$kind" "$label" "$gpu" "$env_num" "$status" "$((end_s - start_s))" \
    "$summary" "$log_path" "$started" "$ended" \
    > "$RUN_DIR/${label}.meta"
  return "$status"
}

run_queue_env50() {
  set +e
  run_one env50_full 50 "$GPU_ENV50"
  run_one env50_no_anchor 50 "$GPU_ENV50" MIKASA_EVAL_ABLATION=disable_late_cue_anchor
  exit 0
}

run_queue_env10() {
  set +e
  run_one env10_full 10 "$GPU_ENV10"
  run_one env10_no_anchor 10 "$GPU_ENV10" MIKASA_EVAL_ABLATION=disable_late_cue_anchor
  exit 0
}

run_queue_env50 &
pid_env50=$!
run_queue_env10 &
pid_env10=$!
all_pids=("$pid_env50" "$pid_env10")

(
  sleep "$GLOBAL_WATCHDOG_SECONDS"
  printf '[watchdog]\t%s\tstopping direct-anchor protocol rerun after %s seconds\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$GLOBAL_WATCHDOG_SECONDS" >> "$RUN_DIR/watchdog.log"
  for pid in "${all_pids[@]}"; do
    kill_tree "$pid"
  done
  sleep 60
  for pid in "${all_pids[@]}"; do
    kill_tree_hard "$pid"
  done
) &
watchdog_pid=$!

set +e
for pid in "${all_pids[@]}"; do
  wait "$pid"
done
set -e

kill "$watchdog_pid" 2>/dev/null || true
wait "$watchdog_pid" 2>/dev/null || true

for meta in "$RUN_DIR"/*.meta; do
  [[ -f "$meta" ]] || continue
  cat "$meta" >> "$RESULTS_TSV"
  rm -f "$meta"
done

python - <<'PY' "$RUN_DIR"
import json
import math
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
rows = []
for line in (run_dir / "results.tsv").read_text().splitlines()[1:]:
    parts = line.split("\t")
    if len(parts) != 10:
        continue
    kind, label, gpu, env_num, status, wall, summary, log_path, started, ended = parts
    row = {
        "kind": kind,
        "label": label,
        "gpu": gpu,
        "env_num": env_num,
        "status": status,
        "wall_seconds": wall,
        "num_episodes": "missing",
        "success_once": "nan",
        "success_at_end": "nan",
        "summary": summary,
        "log_path": log_path,
    }
    path = Path(summary)
    if path.exists():
        data = json.loads(path.read_text())
        row["num_episodes"] = str(data.get("num_episodes", "missing"))
        row["success_once"] = f"{float(data.get('success_once', math.nan)):.8f}"
        row["success_at_end"] = f"{float(data.get('success_at_end', math.nan)):.8f}"
    rows.append(row)

with (run_dir / "metric_summary.tsv").open("w", encoding="utf-8") as f:
    f.write("kind\tgpu\tenv_num\tstatus\twall_seconds\tnum_episodes\tsuccess_once\tsuccess_at_end\tsummary\tlog_path\n")
    for row in rows:
        f.write("\t".join(row[k] for k in [
            "kind", "gpu", "env_num", "status", "wall_seconds", "num_episodes",
            "success_once", "success_at_end", "summary", "log_path",
        ]) + "\n")
PY

printf 'ended_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RUN_DIR/invocation.tsv"
