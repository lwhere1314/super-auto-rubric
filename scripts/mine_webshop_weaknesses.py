#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from super_auto_rubric.webshop.trajectory import load_trajectories, write_jsonl
from super_auto_rubric.webshop.weakness import mine_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine weakness candidates from WebShop trajectories.")
    parser.add_argument("trajectory_path", help="A trajectory JSONL file or directory.")
    parser.add_argument("--output", default="artifacts/annotations/weakness_candidates.jsonl")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trajectories = load_trajectories(Path(args.trajectory_path))
    weaknesses = mine_batch(trajectories)
    write_jsonl(Path(args.output), [weakness.to_dict() for weakness in weaknesses])
    print(f"trajectories={len(trajectories)} weaknesses={len(weaknesses)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
