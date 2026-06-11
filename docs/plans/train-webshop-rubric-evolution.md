# Training Branch: WebShop Rubric Evolution

## Branch Purpose

Build the training-based version of trajectory-derived rubric supervision for
WebShop. This branch should answer whether evolved rubrics improve subsequent
training batches beyond a static rubric or vanilla reward signal.

## Method Sketch

1. Run WebShop benchmark batches through AgentGym.
2. Persist trajectories and task metadata for every episode.
3. Diagnose failures into weakness candidates.
4. Cluster weakness candidates and update the active weakness pool.
5. Convert active weaknesses into positive and negative rubrics.
6. Add those rubrics as auxiliary supervision for the next training batch.
7. Evaluate on held-out WebShop tasks and track reward hacking indicators.

## Initial Modules

- `envs/agentgym_webshop`: AgentGym WebShop adapter and run config.
- `trajectory_store`: immutable trajectory persistence and batch indexing.
- `weakness_miner`: extraction, failure taxonomy tagging, and clustering.
- `rubric_buffer`: RLER-style evolved rubric buffer with add, merge, decay, and
  retire operations.
- `training_loop`: batch runner that mixes original supervision with evolved
  rubrics.
- `evals`: held-out success, rubric agreement, and reward hacking checks.

## Settings Entry Points

- `configs/external_repos.yaml`: pinned AgentGym and dr-tulu checkouts.
- `configs/agentgym_webshop.yaml`: replicated WebShop runtime, server, data, and
  action settings.
- `configs/dr_tulu_rubric.yaml`: replicated judge, reward, GRPO, clustering, and
  adaptive rubric settings.
- `configs/train_webshop_rubric_evolution.yaml`: top-level experiment loop.

## Rubric Buffer Entry

Each active rubric should keep:

- `rubric_id`
- `source_weakness_ids`
- `natural_language_rule`
- `polarity`: positive or negative
- `severity`
- `support_count`
- `last_triggered_batch`
- `decay_score`
- `examples`: trajectory ids and minimal evidence spans

## WebShop-Specific Weakness Seeds

- Query drift: search terms stop matching the user constraints.
- Attribute neglect: price, size, brand, rating, or compatibility constraints are
  ignored.
- Premature purchase: agent buys before comparing enough candidate products.
- Observation over-trust: agent accepts product text without checking details.
- Navigation loop: agent repeats search or product pages without new evidence.
- Proxy hacking: agent optimizes rubric-like wording instead of task success.

## First Milestone

Produce a small offline loop:

1. Run one baseline batch.
2. Mine weaknesses from failed trajectories.
3. Generate rubrics.
4. Run a second batch using those rubrics as extra supervision.
5. Compare against a static-rubric baseline.

The concrete WebShop-first task list is maintained in
`docs/plans/webshop-todo-list.md`.

## Success Criteria

- WebShop task success improves on held-out tasks.
- Rubric-triggered behavior changes are supported by trajectory evidence.
- Negative rubrics reduce shortcut behavior without suppressing valid strategies.
- Retired weaknesses do not keep influencing later batches.
