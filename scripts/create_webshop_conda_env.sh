#!/usr/bin/env bash
set -euo pipefail

SSD_ROOT="${SSD_ROOT:-/Volumes/SSD}"
ENV_PREFIX="${ENV_PREFIX:-${SSD_ROOT}/conda-envs/agentenv-webshop}"
PKGS_DIR="${CONDA_PKGS_DIRS:-${SSD_ROOT}/conda-pkgs}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${SSD_ROOT}/pip-cache}"

mkdir -p "$(dirname "${ENV_PREFIX}")" "${PKGS_DIR}" "${PIP_CACHE_DIR}"

export CONDA_PKGS_DIRS="${PKGS_DIR}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR}"

conda create -y \
  -p "${ENV_PREFIX}" \
  -c conda-forge \
  python=3.8 \
  faiss-cpu=1.7 \
  openjdk=11

conda run -p "${ENV_PREFIX}" python -m pip install --upgrade pip
conda run -p "${ENV_PREFIX}" python -m pip install \
  -r requirements/agentgym-webshop.txt \
  -e . \
  -e external/AgentGym/agentenv-webshop \
  -e external/AgentGym/agentenv

echo "WebShop conda env ready at ${ENV_PREFIX}"
