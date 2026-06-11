# WebShop-First TODO List

We will use WebShop as the primary benchmark because it is closest to the target
shopping-agent domain. WebGen-Bench and Terminal-Bench stay as later-stage
stress tests after the WebShop loop works end to end.

## M0: Branch And Settings

- [x] Create `train/webshop-rubric-evolution`.
- [x] Clone and pin AgentGym at `3ef9235`.
- [x] Clone and pin dr-tulu at `9d7b037`.
- [x] Replicate AgentGym WebShop settings into `configs/agentgym_webshop.yaml`.
- [x] Replicate dr-tulu rubric/reward settings into `configs/dr_tulu_rubric.yaml`.
- [x] Add `scripts/bootstrap_train_branch.sh` for reproducible external checkout.

## M1: WebShop Environment Smoke Test

- [x] Create or document the Python 3.8 WebShop environment.
- [ ] Run AgentGym WebShop setup and search index generation.
- [ ] Launch the WebShop server on port `36001`.
- [ ] Run one scripted or random-policy episode.
- [ ] Confirm observations, available actions, rewards, and done flags are emitted.
- [ ] Save the smoke-test trace under `artifacts/trajectories/smoke/`.

Status: SSD install paths and script are ready in
`docs/setup/mac-webshop-ssd.md` and `scripts/create_webshop_conda_env.sh`, but
the actual conda create step is currently blocked by network reset/SSL EOF
errors while downloading package metadata.

## M2: Trajectory Store

- [x] Define a JSONL trajectory schema for WebShop episodes.
- [x] Capture instruction text, observations, actions, available actions, rewards,
  done flags, and environment state.
- [x] Preserve raw traces separately from derived annotations.
- [x] Add batch metadata: model, prompt version, rubric version, seed, session id,
  and timestamp.
- [x] Add a loader that can stream failed trajectories for weakness mining.

## M3: Baseline Batch Runner

- [x] Implement a small `bench_size` runner using AgentGym WebShop.
- [x] Start with a cheap deterministic slice before full randomized batches.
- [x] Record success rate, average reward, step count, invalid-action rate, and
  loop rate.
- [x] Save per-episode traces and aggregate metrics.
- [x] Produce a static-rubric baseline for later comparison.

Status: runner is implemented for both AgentGym HTTP WebShop and synthetic
offline checks. Static persistent rubrics are defined in
`configs/static_webshop_rubrics.yaml`, and baseline trajectories default to
`rubric_version=static-webshop-v0`.

## M4: Weakness Mining

- [x] Segment trajectories into search, inspect, compare, decide, and purchase
  phases.
- [x] Identify the earliest decisive mistake in failed trajectories.
- [x] Tag failures with the initial taxonomy: query drift, attribute neglect,
  premature purchase, observation over-trust, navigation loop, and proxy hacking.
- [x] Extract evidence spans from observations and actions.
- [x] Cluster similar weakness candidates before adding them to the pool.

Status: initial heuristic miner is implemented and covered by synthetic tests.
The taxonomy should be tightened after real WebShop traces arrive.

## M5: Rubric Pool And Buffer

- [x] Implement RLER-style active rubric entries with source weakness ids,
  evidence trajectory ids, polarity, severity, support count, and last triggered
  batch.
- [x] Add merge, decay, retire, and quarantine operations.
- [x] Generate both positive rubrics and negative rubrics.
- [x] Limit active adaptive rubrics to the configured `max_active_rubrics`.
- [x] Persist active and retired rubrics as JSONL artifacts.

Status: positive and negative rubric generation are implemented and covered by
synthetic tests. Real WebShop traces will be used to tune wording and weights.

## M6: Training Integration

- [ ] Convert active weaknesses into supervision that can be mixed with original
  WebShop rewards.
- [ ] Run batch 0 without adaptive rubrics.
- [ ] Mine weaknesses and generate rubrics after batch 0.
- [ ] Run batch 1 with adaptive rubric supervision.
- [ ] Compare against the static-rubric baseline on held-out WebShop tasks.

## M7: Reward Hacking Checks

- [ ] Detect cases where rubric score improves while task success drops.
- [ ] Flag purchases missing required user constraints.
- [ ] Flag repeated template rationales without supporting observation evidence.
- [ ] Quarantine suspicious rubrics until held-out confirmation.
- [ ] Log negative-rubric triggers separately from task reward.

## M8: Experiment Report

- [ ] Summarize success rate, reward, invalid actions, loops, and repeated failure
  reduction.
- [ ] Include before/after examples for each recurring weakness.
- [ ] Report which adaptive rubrics were added, merged, decayed, retired, or
  quarantined.
- [ ] Decide whether to scale next to larger WebShop batches, WebGen-Bench, or
  Terminal-Bench.

## Minimum Viable Loop

The first runnable target is intentionally small:

1. Run one WebShop smoke episode.
2. Save the trajectory.
3. Manually label one failure or near-failure.
4. Convert it into one negative rubric.
5. Re-run the same task slice with the rubric active.
6. Check whether the original mistake disappears without blocking valid actions.
