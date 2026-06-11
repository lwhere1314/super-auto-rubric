# Mac WebShop Setup On SSD

This machine can run WebShop environment smoke tests and trajectory collection
locally, but it should not run the CUDA training stack. The local environment is
kept on the external SSD to avoid filling the system disk.

## Paths

- SSD root: `/Volumes/SSD`
- SSD venv: `/Volumes/SSD/venvs/agentenv-webshop`
- Pip cache: `/Volumes/SSD/pip-cache`
- WebShop resources and Lucene index: `/Volumes/SSD/webshop-search`
- Homebrew JDK used locally: `/opt/homebrew/Cellar/openjdk/25.0.2/libexec/openjdk.jdk/Contents/Home`

## Current Working Setup

```sh
scripts/create_webshop_ssd_venv.sh
scripts/build_webshop_index_ssd.sh
```

This uses the existing Python 3.8 interpreter at
`/opt/miniconda3/envs/TAPE/bin/python` to create an SSD-backed venv with
`--system-site-packages`. On this Mac that reuses the already installed
`torch 2.3.0`, `numpy`, `pandas`, and `sklearn`, while new WebShop runtime
packages are installed into `/Volumes/SSD/venvs/agentenv-webshop`.

The index script symlinks WebShop `resources_*` and `indexes_1k` from AgentGym's
expected paths to `/Volumes/SSD/webshop-search`, then builds the 1k Lucene index
used by `num_products=1000`.

## Launch WebShop

```sh
scripts/run_agentgym_webshop_server.sh
```

In another terminal, run a real smoke episode:

```sh
PYTHONPATH=src /Volumes/SSD/venvs/agentenv-webshop/bin/python \
  scripts/run_webshop_smoke.py \
  --base-url http://127.0.0.1:36001 \
  --output-dir artifacts/trajectories/smoke-real \
  --max-steps 8
```

Run a small real baseline batch:

```sh
PYTHONPATH=src /Volumes/SSD/venvs/agentenv-webshop/bin/python \
  scripts/run_webshop_baseline_batch.py \
  --base-url http://127.0.0.1:36001 \
  --output-dir artifacts/trajectories/baseline-real \
  --episodes 3 \
  --max-steps 8
```

Then mine weaknesses and update the adaptive rubric pool:

```sh
PYTHONPATH=src /Volumes/SSD/venvs/agentenv-webshop/bin/python \
  scripts/mine_webshop_weaknesses.py \
  artifacts/trajectories/baseline-real \
  --output artifacts/annotations/baseline-real-weaknesses.jsonl

PYTHONPATH=src /Volumes/SSD/venvs/agentenv-webshop/bin/python \
  scripts/update_webshop_rubric_pool.py \
  artifacts/annotations/baseline-real-weaknesses.jsonl \
  --active artifacts/rubrics/baseline-real-active.jsonl \
  --retired artifacts/rubrics/baseline-real-retired.jsonl \
  --batch-id 0
```

## Conda Alternative

`scripts/create_webshop_conda_env.sh` records the closer AgentGym setup
(`python=3.8`, `faiss-cpu=1.7`, `openjdk=11`) and redirects conda/pip caches to
SSD. On this Mac, conda metadata downloads repeatedly failed with network-level
SSL EOF / connection reset errors from both default channels and a Tsinghua
mirror, so the working local setup uses the SSD venv route above.

For real training, prefer the Linux/CUDA 4090 machine rather than this Mac.

## Verification

Commands verified locally:

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src /Volumes/SSD/venvs/agentenv-webshop/bin/python \
  -m unittest discover -s tests -v
/Volumes/SSD/venvs/agentenv-webshop/bin/python \
  -m py_compile $(find src scripts -name '*.py' -print)
```

## Offline Wiring Check

The local trajectory, weakness mining, and rubric pool code can be checked
without the real WebShop server:

```sh
python3 scripts/run_webshop_smoke.py --synthetic
python3 scripts/mine_webshop_weaknesses.py artifacts/trajectories/smoke
python3 scripts/update_webshop_rubric_pool.py \
  artifacts/annotations/weakness_candidates.jsonl
```

Run unit tests with:

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
```
