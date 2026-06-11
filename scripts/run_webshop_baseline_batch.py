#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from super_auto_rubric.webshop.baseline import run_episode, save_batch_results
from super_auto_rubric.webshop.client import AgentGymWebShopClient, SyntheticWebShopClient
from super_auto_rubric.webshop.policies import ScriptedWebShopPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small scripted WebShop baseline batch.")
    parser.add_argument("--base-url", default="http://127.0.0.1:36001")
    parser.add_argument("--output-dir", default="artifacts/trajectories/baseline")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rubric-version", default="static-webshop-v0")
    parser.add_argument("--synthetic", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = SyntheticWebShopClient() if args.synthetic else AgentGymWebShopClient(base_url=args.base_url)
    if not args.synthetic and not client.health():
        print(f"WebShop server is not healthy at {args.base_url}.", file=sys.stderr)
        return 2

    results = []
    for episode_id in range(args.episodes):
        if not args.synthetic:
            client = AgentGymWebShopClient(base_url=args.base_url)
        result = run_episode(
            client,
            ScriptedWebShopPolicy(),
            split="baseline",
            max_steps=args.max_steps,
            session_id=episode_id,
            seed=args.seed + episode_id,
            rubric_version=args.rubric_version,
        )
        results.append(result)

    metrics = save_batch_results(Path(args.output_dir), results)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
