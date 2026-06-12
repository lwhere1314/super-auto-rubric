#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from super_auto_rubric.webshop.client import AgentGymWebShopClient, SyntheticWebShopClient
from super_auto_rubric.webshop.seed_plan import SeedPlanChatClient
from super_auto_rubric.webshop.training_free_icl import (
    CriticRubricJudge,
    InContextRubricPolicy,
    load_feedback_hints,
    load_active_rubrics,
    run_training_free_icl_episode,
    save_training_free_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run training-free WebShop ICL with active critic rubrics."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:36001")
    parser.add_argument("--api-base-url", default="https://ark.cn-beijing.volces.com/api/plan/v3")
    parser.add_argument("--api-key-env", default="SEED_AGENT_PLAN_API_KEY")
    parser.add_argument("--actor-model", default="doubao-seed-2.0-mini")
    parser.add_argument("--critic-model", default="kimi-k2.6")
    parser.add_argument("--active-rubrics", default="artifacts/rubrics/baseline-real-active.jsonl")
    parser.add_argument("--feedback-memory", default=None)
    parser.add_argument("--output-dir", default="artifacts/trajectories/training-free-icl")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--session-offset",
        type=int,
        default=0,
        help="Start WebShop session ids from this offset for held-out task slices.",
    )
    parser.add_argument("--rubric-limit", type=int, default=5)
    parser.add_argument("--feedback-limit", type=int, default=8)
    parser.add_argument(
        "--state-aware-feedback",
        action="store_true",
        help="Select a smaller feedback subset based on the current WebShop page state.",
    )
    parser.add_argument(
        "--state-feedback-limit",
        type=int,
        default=8,
        help="Maximum feedback hints injected per step when --state-aware-feedback is enabled.",
    )
    parser.add_argument(
        "--purchase-priority",
        action="store_true",
        help="Add product-page guidance that prefers Buy Now after required options are selected.",
    )
    parser.add_argument(
        "--disable-recent-actions",
        action="store_true",
        help="Ablation: do not inject recent action history into the actor prompt.",
    )
    parser.add_argument(
        "--api-concurrency",
        type=int,
        default=1,
        help="Run independent episodes concurrently to overlap actor/critic API calls.",
    )
    parser.add_argument("--synthetic", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rubrics = load_active_rubrics(Path(args.active_rubrics), limit=args.rubric_limit)
    if not rubrics:
        print(f"No active rubrics found at {args.active_rubrics}.", file=sys.stderr)
        return 2
    feedback_hints = (
        load_feedback_hints(Path(args.feedback_memory), limit=args.feedback_limit)
        if args.feedback_memory
        else []
    )

    chat_client = SeedPlanChatClient.from_env(
        api_key_env=args.api_key_env,
        base_url=args.api_base_url,
    )
    if not args.synthetic:
        health_client = AgentGymWebShopClient(base_url=args.base_url)
        if not health_client.health():
            print(f"WebShop server is not healthy at {args.base_url}.", file=sys.stderr)
            return 3

    output_dir = Path(args.output_dir)

    def run_one_episode(episode_id: int):
        client = SyntheticWebShopClient() if args.synthetic else AgentGymWebShopClient(base_url=args.base_url)
        policy = InContextRubricPolicy(
            chat_client=chat_client,
            actor_model=args.actor_model,
            rubrics=rubrics,
            feedback_hints=feedback_hints,
            include_recent_actions=not args.disable_recent_actions,
            state_aware_feedback=args.state_aware_feedback,
            state_feedback_limit=args.state_feedback_limit,
            purchase_priority=args.purchase_priority,
        )
        judge = CriticRubricJudge(
            chat_client=chat_client,
            critic_model=args.critic_model,
            rubrics=rubrics,
        )
        return run_training_free_icl_episode(
            client,
            policy,
            judge,
            split="training-free-icl",
            max_steps=args.max_steps,
            session_id=args.session_offset + episode_id,
            seed=args.seed + episode_id,
            actor_model=args.actor_model,
            critic_model=args.critic_model,
            rubric_version=Path(args.active_rubrics).stem,
        )

    completed = {}
    max_workers = max(1, args.api_concurrency)
    if max_workers == 1:
        for episode_id in range(args.episodes):
            completed[episode_id] = run_one_episode(episode_id)
            results = [completed[idx] for idx in sorted(completed)]
            save_training_free_results(output_dir, results)
            breakdown = completed[episode_id][1]
            print(
                "episode="
                f"{episode_id + 1}/{args.episodes} "
                f"task_reward={breakdown.task_reward:.3f} "
                f"format_tool={breakdown.format_tool_validity_reward:.3f} "
                f"critic_sum={breakdown.critic_rubric_judged_score_sum:.3f} "
                f"combined={breakdown.combined_reward:.3f}",
                file=sys.stderr,
                flush=True,
            )
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_one_episode, episode_id): episode_id
                for episode_id in range(args.episodes)
            }
            for future in as_completed(futures):
                episode_id = futures[future]
                completed[episode_id] = future.result()
                results = [completed[idx] for idx in sorted(completed)]
                save_training_free_results(output_dir, results)
                breakdown = completed[episode_id][1]
                print(
                    "episode="
                    f"{episode_id + 1}/{args.episodes} "
                    f"finished={len(completed)}/{args.episodes} "
                    f"task_reward={breakdown.task_reward:.3f} "
                    f"format_tool={breakdown.format_tool_validity_reward:.3f} "
                    f"critic_sum={breakdown.critic_rubric_judged_score_sum:.3f} "
                    f"combined={breakdown.combined_reward:.3f}",
                    file=sys.stderr,
                    flush=True,
                )

    results = [completed[idx] for idx in sorted(completed)]
    metrics = save_training_free_results(output_dir, results)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"saved={Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
