#!/usr/bin/env bash
set -euo pipefail

SSD_ROOT="${SSD_ROOT:-/Volumes/SSD}"
VENV_DIR="${VENV_DIR:-${SSD_ROOT}/venvs/agentenv-webshop}"
JDK_HOME="${JDK_HOME:-/opt/homebrew/Cellar/openjdk/25.0.2/libexec/openjdk.jdk/Contents/Home}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-36001}"

export JAVA_HOME="${JDK_HOME}"
export PATH="${JAVA_HOME}/bin:${PATH}"
export PYTHONPATH="external/AgentGym/agentenv-webshop:external/AgentGym/agentenv-webshop/webshop"

exec "${VENV_DIR}/bin/python" -m uvicorn agentenv_webshop:app \
  --host "${HOST}" \
  --port "${PORT}" \
  --workers 1
