#!/usr/bin/env bash
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate verl-new
cd /home/u2021110842/super-auto-rubric/external/dr-tulu/rl/open-instruct

export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export NO_PROXY="127.0.0.1,localhost,::1,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,::1,${no_proxy:-}"

python -u open_instruct/grpo_fast.py \
  --exp_name webshop-agentgym-baseline-v5-6144 \
  --wandb_project_name rl-rag \
  --with_tracking False \
  --push_to_hub False \
  --try_auto_save_to_beaker False \
  --try_launch_beaker_eval_jobs_on_weka False \
  --model_name_or_path /home/u2021110842/models/Qwen3-0.6B \
  --dataset_mixer_list /home/u2021110842/super-auto-rubric/artifacts/webshop_tulu/formal_v5_agentgym/webshop_tulu_train_5000.jsonl 1.0 \
  --dataset_mixer_list_splits train \
  --dataset_mixer_eval_list /home/u2021110842/super-auto-rubric/artifacts/webshop_tulu/formal_v5_agentgym/webshop_tulu_eval_128.jsonl 1.0 \
  --dataset_mixer_eval_list_splits train \
  --dataset_skip_cache True \
  --dataset_local_cache_dir /home/u2021110842/super-auto-rubric/artifacts/webshop_tulu/formal_v5_agentgym/cache_baseline_6144 \
  --ground_truths_key ground_truth \
  --sft_messages_key messages \
  --overwrite_reward_fn_tag webshop_critic_error \
  --webshop_base_url http://127.0.0.1:36002 \
  --webshop_interactive_rollout True \
  --webshop_max_replay_steps 8 \
  --webshop_action_max_tokens 80 \
  --webshop_observation_token_budget 192 \
  --critic_error_enabled False \
  --apply_verifiable_reward True \
  --verification_reward 10.0 \
  --beta 0.001 \
  --learning_rate 5e-7 \
  --lr_scheduler_type constant \
  --num_unique_prompts_rollout 8 \
  --num_samples_per_prompt_rollout 4 \
  --num_mini_batches 1 \
  --num_epochs 1 \
  --per_device_train_batch_size 1 \
  --total_episodes 6144 \
  --response_length 1024 \
  --max_prompt_token_length 1024 \
  --max_token_length 2304 \
  --pack_length 2304 \
  --temperature 0.8 \
  --webshop_rollout_trace_path /home/u2021110842/super-auto-rubric/artifacts/webshop_tulu/formal_v5_agentgym/baseline_v5_rollout_trace.jsonl \
  --num_evals 0 \
  --save_freq 96 \
  --async_mode False \
  --deepspeed_stage 3 \
  --num_learners_per_node 1 \
  --vllm_num_engines 3 \
  --vllm_tensor_parallel_size 1 \
  --vllm_gpu_memory_utilization 0.45 \
  --vllm_sync_backend gloo \
  --vllm_enforce_eager True \
  --single_gpu_mode False \
  --allow_world_padding True \
  --gradient_checkpointing \
  --output_dir /home/u2021110842/super-auto-rubric/outputs/formal_v5_agentgym/webshop-baseline-6144 \
  --seed 5
