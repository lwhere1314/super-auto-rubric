# WebShop Tulu Critic-Error RL v1

This directory is the reproducible v1 snapshot used for the 4x RTX 4090 WebShop
experiment with Qwen3-0.6B and dynamic critic-error rubrics.

It contains implementation files, launch scripts, prompt data, full rollout trace
snapshots, and the first-run evidence logs. Checkpoints, API keys, and model
weights are not included.

## Contents

- `patches/open_instruct/grpo_fast.py`
  - DR-Tulu/Open-Instruct GRPO runner with interactive WebShop rollouts.
  - Adds action-prefix generation, observation masking, no-gradient batch skip
    handling, and `num_evals <= 0` train-time eval disablement.
- `patches/open_instruct/search_rewards/webshop_critic_error.py`
  - WebShop reward with true environment reward logging, proxy/progress signals,
    and dynamic critic-error rubric mining/scoring.
- `scripts/launch_baseline_v5_agentgym.sh`
  - Baseline 4-card run using the same WebShop rollout setup, with
    `critic_error_enabled=False`.
- `scripts/launch_critic_v5_agentgym.sh`
  - Critic-error v1 run with Kimi judge, dynamic rubric pool, and WebShop on
    `http://127.0.0.1:36002`.
- `scripts/watchdog_*.sh`
  - Minimal restart/watch scripts used on the GPU server.
- `scripts/monitor_v5_agentgym.sh`
  - Non-destructive monitor for process liveness, trace freshness, GPU status,
    and WebShop health.
- `data/*.jsonl`
  - WebShop train/eval prompt files used by the v1 runs.
- `traces/*.jsonl`
  - Complete rollout trace snapshots from the diagnostic baseline and the first
    critic-error v1 run.
- `logs/*`
  - Baseline/critic logs, step metrics, active rubric snapshots, and summaries
    used to verify the advantage signal.

## Reward Computation

The key reward implementation is:

```text
patches/open_instruct/search_rewards/webshop_critic_error.py
```

The main entry point is:

```text
compute_webshop_critic_error_reward(...)
```

For each rollout it:

1. Extracts executable WebShop actions with `extract_webshop_actions`.
2. Replays those actions against the WebShop environment with
   `replay_webshop_actions`.
3. Logs the true WebShop environment reward as `webshop_env_reward`.
4. Loads active critic-error rubrics from the dynamic weakness pool.
5. Uses the configured critic judge (`kimi-k2.6` in v1) to score whether each
   active rubric is triggered.
6. Computes the scalar critic penalty:

```text
critic_error_reward = - weighted_mean(trigger_score_by_rubric)
```

The final scalar reward used by GRPO is:

```text
reward = webshop_env_reward + critic_error_reward_weight * critic_error_reward
```

In the checked-in v1 launch script, `critic_error_reward_weight=0.35`.

The proxy/progress terms are diagnostic only in this v1 implementation:

```text
webshop_proxy_reward
webshop_format_reward
webshop_action_reward
webshop_attribute_reward
```

They are logged to understand behavior, but they are not the main scalar task
reward. This keeps the RL objective aligned with the original WebShop/AgentGym
environment reward while adding critic-error supervision.

Dynamic rubric handling lives in the same file:

```text
mine_and_update_critic_error_pool(...)
update_weakness_pool(...)
inject_active_critic_rubrics_into_ground_truths(...)
update_critic_error_pool_from_logs(...)
```

The GRPO integration is in:

```text
patches/open_instruct/grpo_fast.py
```

The relevant area is the `apply_verifiable_reward(...)` block in the data
preparation thread. That code applies the reward, logs scalar metrics, updates
the weakness pool, and now writes per-step raw rubric scoring to:

```text
<critic_error_pool_path>.step_scores.jsonl
```

The first already-running v1 process did not persist this raw per-rollout
per-rubric score file; it only printed aggregate metrics and updated the pool.
The checked-in code includes the logger for subsequent runs.

## Evidence Logs And Traces

The first evidence bundle is under `logs/` and `traces/`.

Key files:

```text
logs/REWARD_EVIDENCE.md
logs/baseline_pre_fix_36001_zero_advantage.log
logs/baseline_summary.json
logs/critic_v1_36002_advantage.log
logs/critic_v1_summary.json
logs/critic_v1_step_metrics.jsonl
logs/critic_v1_active_rubrics_by_step.jsonl
logs/critic_error_pool_8192.step64.jsonl
traces/baseline_pre_fix_36001_rollout_trace.jsonl
traces/critic_v1_36002_rollout_trace.snapshot.jsonl
```

Important interpretation:

- `baseline_pre_fix_36001_zero_advantage.log` is a diagnostic pre-fix baseline.
  It used the stale `127.0.0.1:36001` WebShop endpoint and contains many HTTP
  502 environment errors. It is included to document the all-zero reward /
  skipped-optimization failure mode, not as the final fair baseline.
- `critic_v1_36002_advantage.log` is the corrected critic-error run on
  `127.0.0.1:36002`. It has non-zero GRPO advantages and does not skip
  optimization in the logged metric steps.
- `critic_error_pool_8192.step64.jsonl` shows the active dynamic rubrics,
  including search-loop, hallucinated observation, failure to click product
  results, and premature purchase weaknesses.
- The complete rollout traces are committed so every action trajectory can be
  inspected directly.

## Apply Patch

From the repository root, after checking out DR-Tulu under `external/dr-tulu`:

```bash
bash experiments/webshop_tulu_critic_v1/apply_to_dr_tulu.sh
```

The script copies the patched files into:

```text
external/dr-tulu/rl/open-instruct/open_instruct/grpo_fast.py
external/dr-tulu/rl/open-instruct/open_instruct/search_rewards/webshop_critic_error.py
```

## Runtime Assumptions

The launch scripts assume:

```text
conda env: verl-new
model path: /home/u2021110842/models/Qwen3-0.6B
WebShop server: http://127.0.0.1:36002
data dir: /home/u2021110842/super-auto-rubric/artifacts/webshop_tulu/formal_v5_agentgym
DR-Tulu root: /home/u2021110842/super-auto-rubric/external/dr-tulu/rl/open-instruct
```

For critic-error runs, set `SEED_AGENT_PLAN_API_KEY` in the shell environment.
Do not commit the value.

The v1 judge configuration in `launch_critic_v5_agentgym.sh` uses:

```text
critic_error_judge_model: kimi-k2.6
critic_error_openai_base_url: https://ark.cn-beijing.volces.com/api/plan/v3
critic_error_max_active_weaknesses: 8
critic_error_update_every_steps: 64
critic_error_inject_every_steps: 32
```

## Data Placement

On the GPU server, copy the data files to the run directory expected by the
launch scripts:

```bash
mkdir -p /home/u2021110842/super-auto-rubric/artifacts/webshop_tulu/formal_v5_agentgym
cp experiments/webshop_tulu_critic_v1/data/*.jsonl \
  /home/u2021110842/super-auto-rubric/artifacts/webshop_tulu/formal_v5_agentgym/
```

## Current v1 Notes

This version demonstrates that dynamic critic-error rubrics create non-zero
advantage signals for GRPO on WebShop. The main performance bottleneck is online
Kimi judge scoring during reward calculation, not GPU training. A v2 should move
toward structured failure clustering, rubric compilation into local detectors,
and sampled LLM audit rather than full online LLM scoring for every rollout.
