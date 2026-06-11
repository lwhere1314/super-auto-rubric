#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from super_auto_rubric.webshop.client import AgentGymWebShopClient, SyntheticWebShopClient
from super_auto_rubric.webshop.seed_plan import SeedPlanChatClient
from super_auto_rubric.webshop.training_free_icl import (
    CriticRubricJudge,
    InContextRubricPolicy,
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
    parser.add_argument("--output-dir", default="artifacts/trajectories/training-free-icl")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rubric-limit", type=int, default=5)
    parser.add_argument("--synthetic", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rubrics = load_active_rubrics(Path(args.active_rubrics), limit=args.rubric_limit)
    if not rubrics:
        print(f"No active rubrics found at {args.active_rubrics}.", file=sys.stderr)
        return 2

    chat_client = SeedPlanChatClient.from_env(
        api_key_env=args.api_key_env,
        base_url=args.api_base_url,
    )
    policy = InContextRubricPolicy(
        chat_client=chat_client,
        actor_model=args.actor_model,
        rubrics=rubrics,
    )
    judge = CriticRubricJudge(
        chat_client=chat_client,
        critic_model=args.critic_model,
        rubrics=rubrics,
    )

    if not args.synthetic:
        health_client = AgentGymWebShopClient(base_url=args.base_url)
        if not health_client.health():
            print(f"WebShop server is not healthy at {args.base_url}.", file=sys.stderr)
            return 3

    results = []
    for episode_id in range(args.episodes):
        client = SyntheticWebShopClient() if args.synthetic else AgentGymWebShopClient(base_url=args.base_url)
        result = run_training_free_icl_episode(
            client,
            policy,
            judge,
            split="training-free-icl",
            max_steps=args.max_steps,
            session_id=episode_id,
            seed=args.seed + episode_id,
            actor_model=args.actor_model,
            critic_model=args.critic_model,
            rubric_version=Path(args.active_rubrics).stem,
        )
        results.append(result)

    metrics = save_training_free_results(Path(args.output_dir), results)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"saved={Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
