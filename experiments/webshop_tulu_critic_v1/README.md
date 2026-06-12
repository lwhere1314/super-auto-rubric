# WebShop Tulu Critic-Error RL v1

This directory is the reproducible v1 snapshot used for the 4x RTX 4090 WebShop
experiment with Qwen3-0.6B and dynamic critic-error rubrics.

It contains only implementation files, launch scripts, and prompt data. Runtime
outputs, rollout traces, logs, checkpoints, API keys, and model weights are not
included.

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

For critic-error runs, set the API key in the shell environment:

```bash
export SEED_AGENT_PLAN_API_KEY=...
```

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
