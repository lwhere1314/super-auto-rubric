#!/usr/bin/env bash
set -euo pipefail

SSD_ROOT="${SSD_ROOT:-/Volumes/SSD}"
PY38="${PY38:-/opt/miniconda3/envs/TAPE/bin/python}"
VENV_DIR="${VENV_DIR:-${SSD_ROOT}/venvs/agentenv-webshop}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${SSD_ROOT}/pip-cache}"

mkdir -p "$(dirname "${VENV_DIR}")" "${PIP_CACHE_DIR}"

"${PY38}" -m venv --system-site-packages "${VENV_DIR}"
export PIP_CACHE_DIR

"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV_DIR}/bin/python" -m pip install -r requirements/agentgym-webshop-mac-runtime.txt

echo "WebShop SSD venv ready at ${VENV_DIR}"
echo "This venv reuses system-site packages from ${PY38}; on this Mac that provides torch/numpy/pandas/sklearn."
