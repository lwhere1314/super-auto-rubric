#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/u2021110842/super-auto-rubric
RUN_DIR="$ROOT/artifacts/webshop_tulu/formal_v5_agentgym"
LOG_DIR="$RUN_DIR/logs"
BASELINE_WATCHDOG="$RUN_DIR/watchdog_baseline_v5_agentgym.sh"
CRITIC_WATCHDOG="$RUN_DIR/watchdog_critic_after_baseline_v5_agentgym.sh"
WEBSHOP_BASE_URL="${WEBSHOP_BASE_URL:-http://127.0.0.1:36002}"
BASELINE_LOG="$LOG_DIR/baseline_v5.log"
CRITIC_LOG="$LOG_DIR/critic_v5.log"
BASELINE_WATCHDOG_PID="$LOG_DIR/baseline_v5_watchdog.pid"
CRITIC_WATCHDOG_PID="$LOG_DIR/critic_after_baseline_watchdog.pid"
MONITOR_LOG="$LOG_DIR/monitor_v5.log"
STATUS_JSONL="$LOG_DIR/monitor_v5_status.jsonl"
MONITOR_PID="$LOG_DIR/monitor_v5.pid"
BASELINE_DISABLED="$LOG_DIR/baseline_v5.disabled"
INTERVAL="${MONITOR_INTERVAL_SECONDS:-300}"
STALE_SECONDS="${MONITOR_STALE_SECONDS:-1800}"

mkdir -p "$LOG_DIR"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$MONITOR_LOG"
}

pid_alive() {
  local pid_file="$1"
  [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null
}

pattern_alive() {
  local pattern="$1"
  pgrep -f "$pattern" >/dev/null 2>&1
}

log_finished() {
  local path="$1"
  [ -f "$path" ] && grep -q "finished training" "$path"
}

start_baseline_watchdog() {
  nohup bash "$BASELINE_WATCHDOG" > "$LOG_DIR/baseline_v5_watchdog.log" 2>&1 &
  echo $! > "$BASELINE_WATCHDOG_PID"
  log "started baseline watchdog pid=$(cat "$BASELINE_WATCHDOG_PID")"
}

start_critic_watchdog() {
  nohup bash "$CRITIC_WATCHDOG" > "$LOG_DIR/critic_after_baseline_watchdog.log" 2>&1 &
  echo $! > "$CRITIC_WATCHDOG_PID"
  log "started critic-after-baseline watchdog pid=$(cat "$CRITIC_WATCHDOG_PID")"
}

write_status() {
  ROOT="$ROOT" RUN_DIR="$RUN_DIR" LOG_DIR="$LOG_DIR" STALE_SECONDS="$STALE_SECONDS" WEBSHOP_BASE_URL="$WEBSHOP_BASE_URL" python - <<'PY'
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

root = Path(os.environ["ROOT"])
run_dir = Path(os.environ["RUN_DIR"])
log_dir = Path(os.environ["LOG_DIR"])
stale_seconds = int(os.environ.get("STALE_SECONDS", "1800"))
webshop_base_url = os.environ.get("WEBSHOP_BASE_URL", "http://127.0.0.1:36002").rstrip("/")
now = time.time()

def cmd(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""

def pid_file(path):
    p = Path(path)
    if not p.exists():
        return None
    try:
        return int(p.read_text().strip())
    except Exception:
        return None

def pid_alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False

def pgrep(pattern):
    out = cmd(["pgrep", "-f", pattern])
    return [int(x) for x in out.splitlines() if x.strip().isdigit()]

def trace_summary(name):
    path = run_dir / name
    if not path.exists():
        return {"exists": False, "rows": 0, "latest_step": None, "positive": 0, "mean_score": 0.0, "stale_seconds": None}
    rows = []
    total = 0.0
    positive = 0
    latest = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            rows.append(row)
            score = float(row.get("score", 0.0) or 0.0)
            total += score
            positive += int(score > 0)
            step = row.get("training_step")
            if isinstance(step, int):
                latest = step if latest is None else max(latest, step)
    mtime = path.stat().st_mtime
    return {
        "exists": True,
        "rows": len(rows),
        "latest_step": latest,
        "positive": positive,
        "mean_score": total / len(rows) if rows else 0.0,
        "mtime": int(mtime),
        "stale_seconds": int(now - mtime),
        "stale": (now - mtime) > stale_seconds,
    }

def log_summary(name):
    path = log_dir / name
    if not path.exists():
        return {"exists": False}
    text_tail = ""
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 4096))
            text_tail = f.read().decode("utf-8", errors="replace")
    except Exception:
        pass
    return {
        "exists": True,
        "size": path.stat().st_size,
        "mtime": int(path.stat().st_mtime),
        "stale_seconds": int(now - path.stat().st_mtime),
        "finished": "finished training" in text_tail,
        "last_generation_seen": "Generation time" in text_tail,
        "last_skip_seen": "Skipping optimization" in text_tail,
    }

def webshop_health():
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    started = time.time()
    try:
        with opener.open(webshop_base_url + "/", timeout=5) as resp:
            body = resp.read(64).decode("utf-8", errors="replace")
            return {
                "url": webshop_base_url,
                "ok": 200 <= resp.status < 300,
                "status": resp.status,
                "body": body,
                "latency_sec": round(time.time() - started, 3),
            }
    except urllib.error.HTTPError as e:
        return {"url": webshop_base_url, "ok": False, "status": e.code, "error": str(e)}
    except Exception as e:
        return {"url": webshop_base_url, "ok": False, "status": None, "error": f"{type(e).__name__}: {e}"}

gpu_raw = cmd(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"])
gpus = []
for line in gpu_raw.splitlines():
    parts = [p.strip() for p in line.split(",")]
    if len(parts) == 3:
        gpus.append({"index": int(parts[0]), "memory_used_mib": int(parts[1]), "utilization_gpu": int(parts[2])})

status = {
    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    "time_epoch": int(now),
    "baseline": {
        "main_pids": pgrep("webshop-agentgym-baseline-v5-6144"),
        "watchdog_pid": pid_file(log_dir / "baseline_v5_watchdog.pid"),
        "watchdog_alive": pid_alive(pid_file(log_dir / "baseline_v5_watchdog.pid")),
        "trace": trace_summary("baseline_v5_rollout_trace.jsonl"),
        "log": log_summary("baseline_v5.log"),
    },
    "critic": {
        "main_pids": pgrep("webshop-agentgym-critic-v5-8192"),
        "watchdog_pid": pid_file(log_dir / "critic_after_baseline_watchdog.pid"),
        "watchdog_alive": pid_alive(pid_file(log_dir / "critic_after_baseline_watchdog.pid")),
        "trace": trace_summary("critic_v5_rollout_trace.jsonl"),
        "log": log_summary("critic_v5.log"),
    },
    "webshop": webshop_health(),
    "gpu": gpus,
}
print(json.dumps(status, ensure_ascii=False, sort_keys=True))
PY
}

echo $$ > "$MONITOR_PID"
log "monitor started pid=$$ interval=${INTERVAL}s stale=${STALE_SECONDS}s"

while true; do
  baseline_finished=false
  critic_finished=false
  log_finished "$BASELINE_LOG" && baseline_finished=true
  log_finished "$CRITIC_LOG" && critic_finished=true

  if [ "$baseline_finished" = false ] && [ ! -f "$BASELINE_DISABLED" ] && ! pid_alive "$BASELINE_WATCHDOG_PID" && ! pattern_alive "watchdog_baseline_v5_agentgym.sh"; then
    start_baseline_watchdog
  fi

  if [ "$critic_finished" = false ] && ! pid_alive "$CRITIC_WATCHDOG_PID" && ! pattern_alive "watchdog_critic_after_baseline_v5_agentgym.sh"; then
    start_critic_watchdog
  fi

  status="$(write_status)"
  printf '%s\n' "$status" >> "$STATUS_JSONL"
  log "$status"

  sleep "$INTERVAL"
done
