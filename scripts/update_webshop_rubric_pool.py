#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from super_auto_rubric.webshop.rubric_pool import RubricPool
from super_auto_rubric.webshop.weakness import WeaknessCandidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update the active WebShop rubric pool.")
    parser.add_argument("weakness_path")
    parser.add_argument("--active", default="artifacts/weakness_pool/active_rubrics.jsonl")
    parser.add_argument("--retired", default="artifacts/weakness_pool/retired_rubrics.jsonl")
    parser.add_argument("--batch-id", type=int, default=0)
    parser.add_argument("--max-active-rubrics", type=int, default=5)
    return parser.parse_args()


def load_weaknesses(path: Path) -> list[WeaknessCandidate]:
    weaknesses = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                weaknesses.append(WeaknessCandidate.from_dict(json.loads(line)))
    return weaknesses


def main() -> int:
    args = parse_args()
    active_path = Path(args.active)
    pool = RubricPool.load(active_path, max_active_rubrics=args.max_active_rubrics)
    weaknesses = load_weaknesses(Path(args.weakness_path))
    triggered = {item.cluster_key or item.label for item in weaknesses}
    pool.upsert_from_weaknesses(weaknesses, batch_id=args.batch_id)
    pool.decay_and_retire(triggered, batch_id=args.batch_id)
    pool.save(active_path, Path(args.retired))
    print(
        f"weaknesses={len(weaknesses)} active={len(pool.active)} "
        f"retired={len(pool.retired)} active_path={active_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
