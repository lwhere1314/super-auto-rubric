# Training-Free Branch: WebShop Trajectory Debugging

## Branch Purpose

Build a training-free version of the trajectory-to-rubric idea for WebShop. This
branch should improve agent behavior without weight updates by diagnosing
trajectories and feeding the discovered weaknesses into prompts, rubrics,
validators, or decoding-time checks.

## Method Sketch

1. Run WebShop tasks and collect trajectories.
2. Diagnose failed or suspicious trajectories with a failure taxonomy.
3. Convert recurring weaknesses into rubric patches and prompt constraints.
4. Re-run or replay tasks with the updated guidance.
5. Keep a dynamic weakness pool, but apply it only at inference or evaluation
   time.

## Harness-TrajecDebug-Inspired Loop

- Segment each trajectory into search, inspect, compare, decide, and purchase
  phases.
- Identify the earliest decisive mistake instead of only labeling the final
  failure.
- Attach evidence spans from observations and actions to every weakness.
- Generate targeted feedback that can be injected before the next attempt.
- Verify whether the injected feedback prevents the same failure mode.

## Initial Modules

- `envs/agentgym_webshop`: shared AgentGym WebShop adapter and run config.
- `trajectory_store`: shared trace persistence and replay support.
- `failure_taxonomy`: WebShop-specific failure labels and evidence schemas.
- `weakness_pool`: dynamic active weakness set with trigger and retire logic.
- `rubric_patcher`: converts weaknesses into rubric, prompt, or validator
  patches.
- `debug_runner`: run, diagnose, patch, and rerun loop for inference-time tests.

## Patch Types

- Prompt patch: concise instruction added before the next WebShop attempt.
- Rubric patch: extra scoring or critique rule used by a judge or self-checker.
- Validator patch: deterministic or model-assisted check before final purchase.
- Negative rubric: explicit rule against shortcut or reward-hacking behavior.

## WebShop-Specific Failure Labels

- Constraint omission: final item misses a user-specified attribute.
- Bad comparison: agent chooses without contrasting plausible alternatives.
- Search narrowing error: agent over-constrains or under-constrains the query.
- Evidence mismatch: final rationale cites evidence not present in observations.
- Tool loop: repeated actions do not add information.
- Suspicious proxy success: rubric score improves while task constraints fail.

## First Milestone

Produce a no-training debug harness:

1. Run a baseline WebShop batch.
2. Diagnose failures and create active weakness patches.
3. Re-run the same task slice with patches applied.
4. Measure pass-rate delta, repeated-failure reduction, and false-positive
   patch interference.

## Success Criteria

- Same-task reruns avoid the original decisive mistake more often than baseline.
- Patches are short, evidence-grounded, and reusable across similar tasks.
- Validators catch unsafe final purchases without blocking correct ones.
- The weakness pool retires stale patches after they stop triggering.
