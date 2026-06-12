# Super Auto Rubric

This repository explores automatic rubric evolution from model trajectories.

The first target environment is WebShop. The shared hypothesis is that a model's
training or evaluation trajectories expose recurring weaknesses that static
rubrics miss. Those weaknesses can be clustered, converted into supervision
signals, and fed back into later runs.

## Shared Research Loop

1. Run a benchmark batch and collect trajectories.
2. Extract failure modes and model-specific weaknesses from those trajectories.
3. Cluster related weaknesses into a managed weakness pool.
4. Convert active weaknesses into finer-grained rubrics.
5. Use the updated rubrics in the next batch.
6. Retire weaknesses that stop triggering in later batches.

## Initial Infrastructure

- Environment base: AgentGym, with native WebShop support.
- Rubric evolution reference: RLER / dr-tulu, especially evolved rubric buffer
  management and negative rubrics.
- Failure taxonomy reference: AgentDebug / Harness-TrajecDebug style trajectory
  diagnosis.
- Reward hacking mitigation references: RLER negative rubrics and CHERRL/RHDA
  style detection.

## Repository Layout

```text
.
├── configs/                         # Reproducible experiment/config snapshots.
├── docs/
│   ├── plans/                       # Design notes, todo lists, and experiment plans.
│   └── setup/                       # Machine-specific setup notes.
├── experiments/
│   └── webshop_tulu_critic_v1/      # Current 4x4090 WebShop Tulu critic-error RL snapshot.
│       ├── data/                    # Train/eval prompt JSONL files used by the v1 run.
│       ├── logs/                    # Baseline/critic logs, summaries, rubric pool, metrics.
│       ├── patches/                 # Patched DR-Tulu/Open-Instruct source files.
│       ├── scripts/                 # Launch/watchdog/monitor scripts for the GPU server.
│       └── traces/                  # Complete rollout trace snapshots.
├── requirements/                    # Python dependency pins for local WebShop tooling.
├── scripts/                         # Local data prep, smoke tests, mining, and run helpers.
├── src/super_auto_rubric/           # Library code for trajectories, rubrics, clients, policies.
├── tests/                           # Unit tests for local, non-CUDA components.
└── external/                        # Ignored local checkouts of AgentGym / DR-Tulu.
```

Ignored runtime directories:

```text
artifacts/       # Generated local/remote experiment artifacts.
outputs/         # Model outputs and checkpoint directories.
runs/            # Ad-hoc run outputs.
wandb/           # WandB local cache.
external/        # Large third-party checkouts.
```

## Current Experiment Snapshot

The most important current artifact is:

```text
experiments/webshop_tulu_critic_v1/
```

Start there if you want to reproduce or extend the first WebShop RL experiment.
Its README documents:

- the exact reward implementation entry point;
- how the patched DR-Tulu files are applied;
- how the 4-card baseline and critic-error runs are launched;
- where the full rollout traces live;
- which logs show zero-advantage baseline behavior versus critic-error
  advantage separation;
- how the dynamic rubric pool is represented.

The key reward code for v1 is:

```text
experiments/webshop_tulu_critic_v1/patches/open_instruct/search_rewards/webshop_critic_error.py
```

The GRPO integration point is:

```text
experiments/webshop_tulu_critic_v1/patches/open_instruct/grpo_fast.py
```

## Branches

- `train/webshop-rubric-evolution`: training-based rubric evolution for WebShop.
- `training-free/webshop-trajecdebug`: training-free trajectory diagnosis and
  rubric rewriting for WebShop.

## Training Branch Setup

On `train/webshop-rubric-evolution`, run:

```sh
scripts/bootstrap_train_branch.sh
```

The training branch keeps AgentGym and dr-tulu as untracked local checkouts in
`external/` and records the replicated settings under `configs/`.

The active WebShop-first execution list is in
`docs/plans/webshop-todo-list.md`.

For local Mac setup on the external SSD, see
`docs/setup/mac-webshop-ssd.md`.

Local non-CUDA checks can be run with:

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The current Mac path is intentionally environment-only: it can run WebShop
smoke tests, trajectory collection, weakness mining, and rubric-buffer updates.
Actual model training should move to the Linux/CUDA 4090 machine.

## Collaboration Notes

- Keep third-party repos under `external/`; do not commit those checkouts.
- Put reusable experiment snapshots under `experiments/<name>/` with a local
  README, scripts, data manifest, and evidence logs.
- Put reusable library code under `src/super_auto_rubric/`; keep one-off remote
  run outputs in ignored `artifacts/` or `outputs/`.
- Do not commit API keys, model weights, checkpoints, or private `.env` files.
- For local checks, prefer:

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

- For WebShop/Tulu GPU experiments, update the experiment README with the reward
  entry point, launch command, data files, trace files, and log evidence so other
  collaborators can audit the result without needing access to the original GPU
  session.
