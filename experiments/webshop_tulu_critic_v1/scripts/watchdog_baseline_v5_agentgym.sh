#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/u2021110842/super-auto-rubric
RUN_DIR="$ROOT/artifacts/webshop_tulu/formal_v5_agentgym"
LOG_DIR="$RUN_DIR/logs"
LAUNCH="$RUN_DIR/launch_baseline_v5_agentgym.sh"
PID_FILE="$LOG_DIR/baseline_v5.pid"
LOG_FILE="$LOG_DIR/baseline_v5.log"

mkdir -p "$LOG_DIR"

while true; do
  if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    sleep 120
    continue
  fi

  if grep -q "finished training" "$LOG_FILE" 2>/dev/null; then
    exit 0
  fi

  nohup bash "$LAUNCH" >>"$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"
  sleep 120
done
