#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from super_auto_rubric.webshop.training_free_icl import (
    build_feedback_hints_from_trajectories,
    save_feedback_hints,
)
from super_auto_rubric.webshop.trajectory import load_trajectories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build meta-harness feedback hints from critic-scored WebShop trajectories."
    )
    parser.add_argument("trajectories", help="Trajectory file or directory.")
    parser.add_argument("--output", default="artifacts/feedback/meta-harness-feedback.jsonl")
    parser.add_argument("--max-hints", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trajectories = load_trajectories(Path(args.trajectories))
    hints = build_feedback_hints_from_trajectories(trajectories, max_hints=args.max_hints)
    save_feedback_hints(Path(args.output), hints)
    print(
        json.dumps(
            {
                "trajectories": len(trajectories),
                "feedback_hints": len(hints),
                "output": args.output,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
