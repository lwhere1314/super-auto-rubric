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

### AgentDebug WebShop Labels

The local AgentDebug AgentErrorBench download contains 50 WebShop labels,
50 GAIA labels, and 100 ALFWorld labels. The WebShop labels are useful as a
human annotated seed for critic-error taxonomy and feedback memory.

Generated assets:

```sh
/Volumes/SSD/venvs/agentenv-webshop/bin/python \
  scripts/build_agentdebug_webshop_assets.py \
  --labels-zip /Users/hugo/Downloads/Label-20260611T171105Z-3-001.zip \
  --summary-output artifacts/agentdebug/webshop_label_summary.json \
  --feedback-output artifacts/feedback/agentdebug-webshop-feedback.jsonl \
  --rubrics-output artifacts/rubrics/agentdebug-webshop-active.jsonl \
  --merge-feedback artifacts/feedback/meta-harness-feedback-n10.jsonl \
  --merged-feedback-output artifacts/feedback/webshop-agentdebug-plus-meta-feedback.jsonl \
  --merge-rubrics artifacts/rubrics/baseline-real-active.jsonl \
  --merged-rubrics-output artifacts/rubrics/webshop-agentdebug-plus-baseline-active.jsonl
```

WebShop label distribution:

| Module | Count |
| --- | ---: |
| plan | `19` |
| memory | `15` |
| reflection | `9` |
| system | `5` |
| action | `2` |

Most frequent WebShop failure types:

| Failure Type | Count |
| --- | ---: |
| `plan.inefficient_plan` | `14` |
| `memory.hallucination` | `8` |
| `reflection.progress_misjudge` | `7` |
| `memory.over_simplification` | `6` |
| `plan.constraint_ignorance` | `5` |

This supports the current WebShop focus: tool-format errors are not the main
bottleneck. The dominant failures are repeated or near-duplicate searches,
forgetting visible candidates, judging that no relevant products exist when
there are plausible options, and over-exploring after a plausible product page
is already open.

### Held-Out 100-AgentDebug Run

The 100-sample evaluation used sessions `100..199`, `max_steps=6`,
`doubao-seed-2.0-mini` as actor, `kimi-k2.6` as critic, and the same combined
AgentDebug plus mined active rubrics for both conditions. The only main
condition difference is whether the actor receives the merged AgentDebug plus
meta-harness feedback memory.

Run directories:

- Baseline: `artifacts/trajectories/heldout100-agentdebug-rubrics-recent`
- Feedback: `artifacts/trajectories/heldout100-agentdebug-feedback-history`
- Analysis: `artifacts/analysis/heldout100-agentdebug-feedback-comparison.json`

Summary:

| Condition | Success | Avg Task Reward | Avg Critic Sum | Avg Combined | Buy Episode Rate | Partial Reward Rate | Repeated Action Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rubrics + recent actions | `0.17` | `0.222` | `-2.641` | `-1.419` | `0.25` | `0.08` | `0.078` |
| Rubrics + feedback + recent actions | `0.19` | `0.280` | `-1.424` | `-0.144` | `0.35` | `0.15` | `0.031` |

Paired comparison over the same 100 sessions:

- Average task reward delta: `+0.058`; bootstrap 95% CI
  `[-0.014, +0.132]`.
- Success delta: `+0.02`; bootstrap 95% CI `[-0.04, +0.09]`.
- Critic sum delta: `+1.217`.
- Repeated action rate delta: `-0.045`.
- Task reward improved on `17` sessions, worsened on `8`, unchanged on `75`.
- Success was gained on `7` sessions and lost on `5`.

Action distribution moved in the intended direction:

| Condition | Search | Navigation | Product/Option | Details | Buy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | `155` | `177` | `181` | `29` | `25` |
| Feedback | `133` | `122` | `219` | `44` | `35` |

Interpretation:

- The benefit is real but still directional at `N=100`, not statistically
  decisive. The confidence intervals cross zero.
- Feedback memory increases buying and partial completion, reduces loops, and
  shifts behavior from repeated search/navigation toward product and option
  interactions.
- The biggest positive cases are direct rescues from `0.0` to `1.0`, usually
  with short chains such as `search -> product -> option -> option -> buy now`.
- Regressions often show over-checking `features` or using `< prev>` after the
  right options were already selected. This suggests the next prompt patch
  should make "Buy Now after required options are selected" higher priority
  than optional details inspection.

Recommended next change:

- Keep AgentDebug labels as taxonomy and rubric seeds.
- Reduce feedback prompt crowding by selecting feedback top-k based on current
  state: search page gets plan/memory feedback; product page gets
  progress-misjudge and buy-now feedback.
- Add a lightweight product-page validator: if required options are selected
  and `Buy Now` is visible, prefer buying unless a hard constraint is still
  explicitly unknown.

### State-Aware Feedback Ablation

The next ablation tested whether feedback should be selected by the current
WebShop page state instead of injecting the full feedback memory every step.
The policy class now supports:

- `state_aware_feedback`: rank feedback hints by detected page state.
- `state_feedback_limit`: cap injected hints per step.
- `purchase_priority`: optional product-page guidance that prefers `Buy Now`
  after required options appear selected.

Two extra held-out runs used the same sessions `100..199`:

- State-aware only:
  `artifacts/trajectories/heldout100-agentdebug-state-aware-only`
- State-aware plus purchase priority:
  `artifacts/trajectories/heldout100-agentdebug-state-aware-feedback`

Four-condition summary:

| Condition | Success | Avg Task Reward | Avg Critic Sum | Avg Combined | Buy Episode Rate | Partial Reward Rate | Repeated Action Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rubrics + recent actions | `0.17` | `0.222` | `-2.641` | `-1.419` | `0.25` | `0.08` | `0.078` |
| Full feedback + recent actions | `0.19` | `0.280` | `-1.424` | `-0.144` | `0.35` | `0.15` | `0.031` |
| State-aware feedback | `0.21` | `0.350` | `-0.537` | `0.813` | `0.44` | `0.22` | `0.047` |
| State-aware + purchase priority | `0.17` | `0.334` | `-1.419` | `-0.085` | `0.44` | `0.26` | `0.055` |

Paired against the rubrics-only baseline, state-aware feedback produced:

- Average task reward delta: `+0.128`; bootstrap 95% CI
  `[+0.045, +0.210]`.
- Success delta: `+0.04`; bootstrap 95% CI `[-0.03, +0.11]`.
- Critic sum delta: `+2.104`.
- Repeated action rate delta: `-0.032`.
- Task reward improved on `23` sessions, worsened on `8`, unchanged on `69`.
- Success was gained on `9` sessions and lost on `5`.

Interpretation:

- The reward-feedback design now has stronger evidence for average task reward:
  the paired bootstrap interval against baseline is positive for state-aware
  feedback.
- Success rate is still not statistically decisive, but it moved in the right
  direction for state-aware feedback.
- The main gain comes from selecting relevant feedback by page state, not from
  globally pushing `Buy Now`.
- `purchase_priority` increased buy and partial rates, but did not improve
  success. It likely pushes some trajectories into premature or partially
  correct purchases.
- State-aware feedback is the current best training-free setting: it reduces
  repeated actions, increases product/option/buy behavior, and gives the
  highest average task and combined reward among the four conditions.

Recommended next change:

- Keep `state_aware_feedback=true` and `purchase_priority=false` as the default
  training-free setting.
- Replace the soft purchase-priority prompt with a stricter verifier that only
  encourages `Buy Now` after the current product page exposes selected options
  matching instruction constraints.
- Mine regressions where state-aware feedback still opens `features`, `< prev`,
  or `description` after selecting options, then add a targeted
  product-page-completion feedback type rather than a broad Buy Now rule.
