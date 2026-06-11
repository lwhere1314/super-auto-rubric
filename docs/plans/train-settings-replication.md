# Training Settings Replication

This branch keeps external repositories as untracked local checkouts under
`external/` and records the settings needed for WebShop rubric-evolution
training in this repository.

## External Pins

- AgentGym: `https://github.com/WooooDyy/AgentGym.git` at `3ef9235`
- dr-tulu: `https://github.com/rlresearch/dr-tulu.git` at `9d7b037`

Run `scripts/bootstrap_train_branch.sh` to recreate these checkouts.

## Replicated AgentGym Settings

- Conda environment: Python 3.8, `faiss-cpu=1.7`, `openjdk=11`.
- WebShop package: `agentenv_webshop` with FastAPI launcher `webshop`.
- Server default: `webshop --host 0.0.0.0 --port 36001`.
- Gym env: `WebAgentTextEnv-v0`, text observations, 1000 products.
- Default data: `items_shuffle_1000.json` and `items_ins_v2_1000.json`.
- Action surface: `search[keywords]` and `click[value]`.

## Replicated dr-tulu Settings

- Rubric judge default: `RUBRIC_JUDGE_MODEL=gpt-4.1`.
- LLM judge defaults: `azure/gpt-4o-mini-standard`, 2048 max tokens,
  temperature 1.0, timeout 60 seconds.
- Weighted reward components mirror `longform_rubric_rewards.py`:
  rubric 0.5, citation 0.2, format 0.2, search-turn 0.1.
- No-citation variant: rubric 0.6, format 0.2, search-turn 0.2.
- Adaptive rubric defaults mirror `grpo_fast.py`: rubric buffer on, maximum
  five active rubrics, persistent static rubrics, refresh every 10 steps.

## Local Config Entry Points

- `configs/external_repos.yaml`
- `configs/agentgym_webshop.yaml`
- `configs/dr_tulu_rubric.yaml`
- `configs/train_webshop_rubric_evolution.yaml`
