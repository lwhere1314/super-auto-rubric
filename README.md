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
