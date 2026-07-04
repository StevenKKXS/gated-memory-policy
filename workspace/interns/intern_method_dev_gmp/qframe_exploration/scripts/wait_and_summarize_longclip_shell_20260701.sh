#!/usr/bin/env bash
set -euo pipefail

ICRA=/mnt/3fs1/data/tingwen.du/icra_method_dev
SHELL_RUNLOG=$ICRA/logs/mikasa_method_dev/longclip_shell_5seed_20260701
SUMMARY_SCRIPT=$ICRA/experiments/qframe_evidence_memory_20260630/scripts/summarize_5seed_rollouts_with_mean_20260701.py
PYTHON=$ICRA/envs/imitation-py310-h200-headless/bin/python
OUTLOG=$SHELL_RUNLOG/summary_watcher.log

mkdir -p "$SHELL_RUNLOG"
echo "[summary-watch-start] utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname)" >> "$OUTLOG"

while true; do
  if grep -q "^\[master-done\]" "$SHELL_RUNLOG/master.log" 2>/dev/null; then
    echo "[summary-watch] shell eval master done utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUTLOG"
    "$PYTHON" "$SUMMARY_SCRIPT" >> "$OUTLOG" 2>&1
    echo "[summary-watch-done] utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUTLOG"
    exit 0
  fi
  echo "[summary-watch] waiting utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUTLOG"
  sleep 300
done
