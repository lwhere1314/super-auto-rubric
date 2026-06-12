#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DR_TULU_OPEN_INSTRUCT="${DR_TULU_OPEN_INSTRUCT:-$REPO_ROOT/external/dr-tulu/rl/open-instruct}"

if [ ! -d "$DR_TULU_OPEN_INSTRUCT/open_instruct" ]; then
  echo "DR-Tulu Open-Instruct checkout not found: $DR_TULU_OPEN_INSTRUCT" >&2
  echo "Set DR_TULU_OPEN_INSTRUCT=/path/to/dr-tulu/rl/open-instruct and retry." >&2
  exit 1
fi

install -m 0644 \
  "$REPO_ROOT/experiments/webshop_tulu_critic_v1/patches/open_instruct/grpo_fast.py" \
  "$DR_TULU_OPEN_INSTRUCT/open_instruct/grpo_fast.py"

mkdir -p "$DR_TULU_OPEN_INSTRUCT/open_instruct/search_rewards"
install -m 0644 \
  "$REPO_ROOT/experiments/webshop_tulu_critic_v1/patches/open_instruct/search_rewards/webshop_critic_error.py" \
  "$DR_TULU_OPEN_INSTRUCT/open_instruct/search_rewards/webshop_critic_error.py"

echo "Applied WebShop Tulu critic-error v1 patches to $DR_TULU_OPEN_INSTRUCT"
