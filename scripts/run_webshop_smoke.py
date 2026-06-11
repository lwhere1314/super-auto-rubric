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
    parser = argparse.ArgumentParser(description="Run one WebShop smoke episode.")
    parser.add_argument("--base-url", default="http://127.0.0.1:36001")
    parser.add_argument("--output-dir", default="artifacts/trajectories/smoke")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--session-id", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use the built-in synthetic client instead of a running AgentGym server.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.synthetic:
        client = SyntheticWebShopClient()
    else:
        client = AgentGymWebShopClient(base_url=args.base_url)
        if not client.health():
            print(
                f"WebShop server is not healthy at {args.base_url}. "
                "Start it first or use --synthetic for local wiring checks.",
                file=sys.stderr,
            )
            return 2

    result = run_episode(
        client,
        ScriptedWebShopPolicy(),
        split="smoke",
        max_steps=args.max_steps,
        session_id=args.session_id,
        seed=args.seed,
    )
    metrics = save_batch_results(Path(args.output_dir), [result])
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"saved={Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
