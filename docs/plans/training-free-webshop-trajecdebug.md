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

## Step3: Critic Error ICL Injection

The first implemented training-free path injects active critic rubrics directly
into the action-selection prompt. It does not update model weights.

Runtime configuration:

- Actor model: `doubao-seed-2.0-mini`.
- Critic model: `kimi-k2.6`.
- API base URL: `https://ark.cn-beijing.volces.com/api/plan/v3`.
- API key environment variable: `SEED_AGENT_PLAN_API_KEY`.
- Active rubrics: `artifacts/rubrics/baseline-real-active.jsonl`.

Reward used for reporting:

```text
R = WebShop task reward
  + format/tool validity reward
  + sum(active critic rubric judged scores)
```

Implementation details:

- `InContextRubricPolicy` prompts the actor with instruction, observation,
  available actions, and active rubrics. The actor must return JSON with one
  WebShop action.
- The format/tool validity reward is the per-episode average of per-step
  validity scores: `0.5` for parseable JSON action plus `0.5` for a valid
  WebShop `search[...]` or `click[...]` action.
- `CriticRubricJudge` asks `kimi-k2.6` to score every active rubric on the
  completed trajectory in `[-1, 1]`; each contribution is
  `score * abs(rubric_weight)`.
- If a model returns invalid action JSON, the runner falls back to the scripted
  policy and records the fallback in the trajectory.

Verified command:

```sh
zsh -lc 'source ~/.bashrc 2>/dev/null || true; source ~/.zshrc 2>/dev/null || true; \
  PYTHONPATH=src SEED_AGENT_PLAN_API_KEY="$SEED_AGENT_PLAN_API_KEY" \
  /Volumes/SSD/venvs/agentenv-webshop/bin/python \
  scripts/run_webshop_training_free_icl.py \
  --base-url http://127.0.0.1:36001 \
  --active-rubrics artifacts/rubrics/baseline-real-active.jsonl \
  --output-dir artifacts/trajectories/training-free-icl-real \
  --episodes 1 \
  --max-steps 6 \
  --seed 300 \
  --actor-model doubao-seed-2.0-mini \
  --critic-model kimi-k2.6 \
  --api-base-url https://ark.cn-beijing.volces.com/api/plan/v3 \
  --api-concurrency 5'
```

Observed result from the first real WebShop run:

- WebShop task reward: `0.0`.
- Format/tool validity reward: `1.0`.
- Active critic rubric judged score sum: `-2.5`.
- Combined reward: `-1.5`.
- Invalid action rate: `0.0`.
- Loop rate: `0.3333`.

The actor produced valid tool calls, but the critic identified a critical error:
the trajectory kept searching/paging without a purchase and repeated nearly
identical progress-free actions. This is the intended training-free signal for
the next ICL patch or validator pass.

### Batched API Run

The runner now supports episode-level API batching via `--api-concurrency`.
Each episode remains sequential internally because WebShop observations depend
on prior actions, but independent episodes run concurrently and overlap actor
and critic API calls. The runner saves partial results after each episode
finishes.

Verified 10-episode batch:

```sh
zsh -lc 'source ~/.bashrc 2>/dev/null || true; source ~/.zshrc 2>/dev/null || true; \
  PYTHONPATH=src SEED_AGENT_PLAN_API_KEY="$SEED_AGENT_PLAN_API_KEY" \
  /Volumes/SSD/venvs/agentenv-webshop/bin/python \
  scripts/run_webshop_training_free_icl.py \
  --base-url http://127.0.0.1:36001 \
  --active-rubrics artifacts/rubrics/baseline-real-active.jsonl \
  --output-dir artifacts/trajectories/training-free-icl-real-n10-batched \
  --episodes 10 \
  --max-steps 6 \
  --seed 700 \
  --actor-model doubao-seed-2.0-mini \
  --critic-model kimi-k2.6 \
  --api-base-url https://ark.cn-beijing.volces.com/api/plan/v3 \
  --api-concurrency 5'
```

N=10 result:

- WebShop task reward mean: `0.0`; standard deviation: `0.0`.
- Format/tool validity reward mean: `1.0`; invalid action rate: `0.0`.
- Critic rubric score sum mean: `-2.3`; standard deviation: `0.6`.
- Combined reward mean: `-1.3`; standard deviation: `0.6`.
- Combined reward distribution: `-1.5` for 9 trajectories, `0.5` for 1
  trajectory.
- Negative critic triggers: navigation loop `10/10`, no satisfying purchase
  `9/10`.

Concrete benefit observed: the environment task reward collapsed all ten
attempts to the same value, `0.0`, so it provided no ranking signal. The
critic-augmented reward separated the failed trajectories by error severity and
attached actionable labels. This makes the signal usable for filtering,
reranking, prompt patching, or future training supervision even before task
success improves.

### Meta-Harness Feedback Pass

The next pass converts critic errors into reusable in-context feedback. This is
the meta-harness behavior we want: after a critic says a trajectory received low
reward because of repeated searches, paging, or no satisfying purchase, the next
actor prompt gets an explicit warning that those action patterns are bad and a
short alternative strategy.

Build feedback memory from the N=10 critic-scored batch:

```sh
PYTHONPATH=src /Volumes/SSD/venvs/agentenv-webshop/bin/python \
  scripts/build_webshop_meta_feedback.py \
  artifacts/trajectories/training-free-icl-real-n10-batched \
  --output artifacts/feedback/meta-harness-feedback-n10.jsonl \
  --max-hints 12
```

The generated feedback entries include:

- `trigger`: what the critic penalized.
- `avoid_actions`: concrete action patterns from failed trajectories.
- `lesson`: why those actions got low reward.
- `suggested_strategy`: what to do instead.

Run with active rubrics plus feedback memory:

```sh
zsh -lc 'source ~/.bashrc 2>/dev/null || true; source ~/.zshrc 2>/dev/null || true; \
  PYTHONPATH=src SEED_AGENT_PLAN_API_KEY="$SEED_AGENT_PLAN_API_KEY" \
  /Volumes/SSD/venvs/agentenv-webshop/bin/python \
  scripts/run_webshop_training_free_icl.py \
  --base-url http://127.0.0.1:36001 \
  --active-rubrics artifacts/rubrics/baseline-real-active.jsonl \
  --feedback-memory artifacts/feedback/meta-harness-feedback-n10.jsonl \
  --feedback-limit 8 \
  --output-dir artifacts/trajectories/training-free-icl-real-n10-feedback-history \
  --episodes 10 \
  --max-steps 6 \
  --seed 700 \
  --actor-model doubao-seed-2.0-mini \
  --critic-model kimi-k2.6 \
  --api-base-url https://ark.cn-beijing.volces.com/api/plan/v3 \
  --api-concurrency 5'
```

This pass also injects recent actions into the actor prompt. That mattered:
feedback memory alone changed search behavior but did not improve success,
because the actor still repeated product-page options or detail clicks. Recent
actions let the prompt say, concretely, "you just did this; choose a different
action that gathers evidence or completes purchase."

Comparison on the same 10 WebShop sessions:

| Condition | Success Rate | Avg Task Reward | Avg Critic Sum | Avg Combined Reward | Loop Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rubrics only | `0.0` | `0.0` | `-2.3` | `-1.3` | `0.2667` |
| Rubrics + feedback memory | `0.0` | `0.0` | `-2.4` | `-1.4` | `0.2167` |
| Rubrics + feedback memory + recent actions | `0.4` | `0.4` | `-0.51` | `0.89` | `0.0714` |

Behavioral shift:

- Rubrics only: mostly repeated `search`, `next`, and `back to search`; no
  purchases.
- Feedback memory only: more product clicks and detail checks, but still no
  purchases.
- Feedback memory plus recent actions: four trajectories completed with
  `Buy Now`, including successful option selection such as
  `search -> product -> color -> size -> buy now`.

The useful conclusion is not just that the scalar reward improved. The more
important result is causal: critic feedback needed to be grounded in the
actor's current action history before the small actor changed its search path
and progressed to purchase.
