#!/usr/bin/env bash
set -euo pipefail

WATCH_SESSION="${WATCH_SESSION:-train}"
LAUNCH_SESSION="${LAUNCH_SESSION:-coco_scratch}"
CHECK_SECONDS="${CHECK_SECONDS:-600}"
WORKDIR="${WORKDIR:-/home/setupishe/bel_conf}"
CONFIG="${CONFIG:-configs/random_coco_s_scratch.yaml}"
CONDA_BIN="${CONDA_BIN:-/home/setupishe/miniconda3/bin/conda}"

echo "[$(date)] watching tmux session '$WATCH_SESSION'"
echo "[$(date)] will launch '$LAUNCH_SESSION' with $CONFIG"

if tmux has-session -t "$LAUNCH_SESSION" 2>/dev/null; then
  echo "[$(date)] launch session '$LAUNCH_SESSION' already exists, aborting"
  exit 1
fi

while true; do
  current_cmd="$(tmux list-panes -t "$WATCH_SESSION" -F '#{pane_current_command}' 2>/dev/null | head -n 1 || true)"
  if [[ -z "$current_cmd" ]]; then
    echo "[$(date)] watch session '$WATCH_SESSION' not found, aborting"
    exit 1
  fi

  echo "[$(date)] '$WATCH_SESSION' current command: $current_cmd"

  # nvtop is not script-friendly, but keep a periodic snapshot attempt for manual inspection.
  timeout 2s nvtop -C -p -d 10 >/tmp/coco_scratch_nvtop.out 2>&1 || true

  if [[ "$current_cmd" == "bash" ]]; then
    echo "[$(date)] '$WATCH_SESSION' is idle, launching '$LAUNCH_SESSION'"
    tmux new-session -d -s "$LAUNCH_SESSION" "cd \"$WORKDIR\" && \"$CONDA_BIN\" run -n lbc python run_chain.py \"$CONFIG\""
    echo "[$(date)] launch submitted"
    exit 0
  fi

  sleep "$CHECK_SECONDS"
done
