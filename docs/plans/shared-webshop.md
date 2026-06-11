# Shared WebShop Experiment Notes

## Goal

Validate whether trajectory-derived weaknesses can improve WebShop agents by
turning observed failures into reusable rubric supervision.

## Core Objects

- Trajectory: one complete WebShop interaction trace.
- Weakness: a recurring behavior pattern that explains avoidable failure.
- Rubric: a supervision or scoring rule derived from one or more weaknesses.
- Weakness pool: the active set of weaknesses retained across batches.

## Baseline Batch

- Use AgentGym's WebShop environment as the first execution target.
- Store trajectories, actions, observations, rewards, final answers, and any
  model self-rationales if available.
- Keep raw traces immutable and put derived annotations in separate artifacts.

## Weakness Pool Policy

- Add: introduce a weakness when it appears across enough trajectories or has
  high severity.
- Merge: combine semantically equivalent weaknesses after clustering.
- Decay: lower priority when a weakness is absent in later batches.
- Retire: remove or archive a weakness after repeated non-triggering batches.

## Reward Hacking Checks

- Track rubrics that improve proxy scores while degrading task success.
- Maintain negative rubrics for unsafe shortcuts and suspicious compliance.
- Compare rubric-triggered improvements against held-out WebShop tasks.
