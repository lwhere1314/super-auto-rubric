# Reward Evidence Snapshot

Generated from the 4090 server logs on 2026-06-12.

## Reward Implementation Entry Points

- Main scalar reward: `patches/open_instruct/search_rewards/webshop_critic_error.py::compute_webshop_critic_error_reward`
- Final scalar formula in v1: `reward = webshop_env_reward + critic_error_reward_weight * critic_error_reward`
- Diagnostic proxy signals only: `webshop_proxy_reward`, `webshop_format_reward`, `webshop_action_reward`, `webshop_attribute_reward`
- Dynamic rubric mining/update: `mine_and_update_critic_error_pool`, `update_weakness_pool`
- Active rubric injection: `inject_active_critic_rubrics_into_ground_truths`
- GRPO-side reward application and metric logging: `patches/open_instruct/grpo_fast.py` around `apply_verifiable_reward(...)`
- Detailed rubric score log for future runs: `<critic_error_pool_path>.step_scores.jsonl`

## Complete Rollout Traces

- Baseline diagnostic trace: `../traces/baseline_pre_fix_36001_rollout_trace.jsonl`
- Critic v1 trace snapshot: `../traces/critic_v1_36002_rollout_trace.snapshot.jsonl`

## Diagnostic Baseline Log

File: `baseline_pre_fix_36001_zero_advantage.log`

This baseline run is **not** a final performance baseline because it used the old WebShop endpoint `127.0.0.1:36001`; many rollouts hit HTTP 502. It documents the zero-advantage failure mode before the corrected critic-error run.

- rollout rows: 2816
- latest step: 88
- score mean: 0.0
- positive rewards: 0
- optimization skips: 88
- metric boxes with advantages: 0

## Critic-Error v1 Log

File: `critic_v1_36002_advantage.log`

This run uses the corrected WebShop endpoint `127.0.0.1:36002` and dynamic Kimi-generated critic-error rubrics. It demonstrates non-zero advantage separation, so GRPO performs updates instead of skipping optimization.

- rollout rows in trace snapshot: 2112
- latest trace step: 66
- logged metric steps: 64
- latest metric step: 64
- score mean: -2.257732
- recent score mean: -2.251727
- optimization skips: 0
- latest critic_error_reward: -0.61
- latest advantages_min/max: -1.73 / 1.73
- latest good_outputs_rate: 0.94

## Rubric Evidence

- Current active rubric pool: `critic_error_pool_8192.step64.jsonl`
- Step-level active rubric and aggregate reward snapshot: `critic_v1_active_rubrics_by_step.jsonl`
- Step-level scalar reward and advantage metrics: `critic_v1_step_metrics.jsonl`

The first running v1 process did not persist per-rollout per-rubric raw scores; those were used in memory and only aggregate metrics were printed. The checked-in `grpo_fast.py` patch now persists those raw scores and evidence to `<critic_error_pool_path>.step_scores.jsonl` for subsequent runs.
