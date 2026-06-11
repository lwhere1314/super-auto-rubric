#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL_DIR="${ROOT_DIR}/external"

AGENTGYM_URL="https://github.com/WooooDyy/AgentGym.git"
AGENTGYM_DIR="${EXTERNAL_DIR}/AgentGym"
AGENTGYM_COMMIT="3ef9235"

DR_TULU_URL="https://github.com/rlresearch/dr-tulu.git"
DR_TULU_DIR="${EXTERNAL_DIR}/dr-tulu"
DR_TULU_COMMIT="9d7b037"

clone_or_update() {
  local url="$1"
  local dir="$2"
  local commit="$3"

  if [ ! -d "${dir}/.git" ]; then
    git clone "${url}" "${dir}"
  fi

  if ! git -C "${dir}" rev-parse --verify --quiet "${commit}^{commit}" >/dev/null; then
    git -C "${dir}" fetch --all --tags
  fi
  git -C "${dir}" -c advice.detachedHead=false checkout --detach "${commit}"
}

mkdir -p "${EXTERNAL_DIR}"
clone_or_update "${AGENTGYM_URL}" "${AGENTGYM_DIR}" "${AGENTGYM_COMMIT}"
clone_or_update "${DR_TULU_URL}" "${DR_TULU_DIR}" "${DR_TULU_COMMIT}"

echo "External repositories are ready:"
git -C "${AGENTGYM_DIR}" rev-parse --short HEAD
git -C "${DR_TULU_DIR}" rev-parse --short HEAD
