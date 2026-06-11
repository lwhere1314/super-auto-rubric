# Mac WebShop Setup On SSD

This machine can run WebShop environment smoke tests and trajectory collection
locally, but it should not run the CUDA training stack. The local environment is
kept on the external SSD to avoid filling the system disk.

## Paths

- SSD root: `/Volumes/SSD`
- Conda env prefix: `/Volumes/SSD/conda-envs/agentenv-webshop`
- Conda package cache: `/Volumes/SSD/conda-pkgs`
- Pip cache: `/Volumes/SSD/pip-cache`

## Create The Environment

```sh
scripts/create_webshop_conda_env.sh
```

The script exports:

```sh
CONDA_PKGS_DIRS=/Volumes/SSD/conda-pkgs
PIP_CACHE_DIR=/Volumes/SSD/pip-cache
```

and creates the environment with:

```sh
conda create -p /Volumes/SSD/conda-envs/agentenv-webshop \
  -c conda-forge python=3.8 faiss-cpu=1.7 openjdk=11
```

## Current Install Status

The SSD directories have been created, but the conda environment has not been
installed yet because conda package metadata downloads repeatedly failed with
network-level SSL EOF / connection reset errors from both the default channels
and a Tsinghua mirror.

No partial environment currently exists under
`/Volumes/SSD/conda-envs/agentenv-webshop`.

## Launch WebShop

After the environment is installed:

```sh
conda run -p /Volumes/SSD/conda-envs/agentenv-webshop \
  webshop --host 0.0.0.0 --port 36001
```

In another terminal:

```sh
python3 scripts/run_webshop_smoke.py \
  --base-url http://127.0.0.1:36001 \
  --output-dir artifacts/trajectories/smoke
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
