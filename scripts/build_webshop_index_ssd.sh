#!/usr/bin/env bash
set -euo pipefail

SSD_ROOT="${SSD_ROOT:-/Volumes/SSD}"
VENV_DIR="${VENV_DIR:-${SSD_ROOT}/venvs/agentenv-webshop}"
JDK_HOME="${JDK_HOME:-/opt/homebrew/Cellar/openjdk/25.0.2/libexec/openjdk.jdk/Contents/Home}"
SEARCH_ROOT="${SEARCH_ROOT:-${SSD_ROOT}/webshop-search}"
SEARCH_DIR="external/AgentGym/agentenv-webshop/webshop/search_engine"

export JAVA_HOME="${JDK_HOME}"
export PATH="${JAVA_HOME}/bin:${PATH}"
export PYTHONPATH="../"

mkdir -p \
  "${SEARCH_ROOT}/resources_100" \
  "${SEARCH_ROOT}/resources" \
  "${SEARCH_ROOT}/resources_1k" \
  "${SEARCH_ROOT}/resources_100k"

cd "${SEARCH_DIR}"

for dir in resources_100 resources resources_1k resources_100k; do
  if [ -e "${dir}" ] && [ ! -L "${dir}" ]; then
    echo "${SEARCH_DIR}/${dir} exists and is not a symlink; move it before rebuilding on SSD." >&2
    exit 1
  fi
  ln -sfn "${SEARCH_ROOT}/${dir}" "${dir}"
done

"${VENV_DIR}/bin/python" convert_product_file_format.py

rm -rf "${SEARCH_ROOT}/indexes_1k"
"${VENV_DIR}/bin/python" -m pyserini.index.lucene \
  --collection JsonCollection \
  --input "${SEARCH_ROOT}/resources_1k" \
  --index "${SEARCH_ROOT}/indexes_1k" \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw

ln -sfn "${SEARCH_ROOT}/indexes_1k" indexes_1k
echo "WebShop 1k Lucene index ready at ${SEARCH_ROOT}/indexes_1k"
