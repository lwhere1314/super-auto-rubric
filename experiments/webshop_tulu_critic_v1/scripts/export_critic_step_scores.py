#!/usr/bin/env python3
"""Export per-rollout, per-rubric critic-error scores from saved WebShop traces.

This is a post-hoc companion for the online logger in the v1 GRPO patch. Use it
when a run was started before `<critic_error_pool_path>.step_scores.jsonl` was
persisted, or when you want to re-score a trace with a newer rubric pool.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _unwrap_ground_truth(ground_truth: Any) -> Dict[str, Any]:
    if isinstance(ground_truth, list) and ground_truth:
        ground_truth = ground_truth[0]
    if isinstance(ground_truth, str):
        return json.loads(ground_truth)
    if isinstance(ground_truth, dict):
        return ground_truth
    raise TypeError(f"Unsupported ground_truth type: {type(ground_truth).__name__}")


def _active_rubrics_at_step(pool_rows: Iterable[Dict[str, Any]], step: int, max_active: int) -> List[Dict[str, Any]]:
    active = []
    for row in pool_rows:
        first_seen = int(row.get("first_seen_step") or 0)
        retired_step = row.get("retired_step")
        if first_seen > step:
            continue
        if retired_step is not None and step >= int(retired_step):
            continue
        if not row.get("active", True) and retired_step is None:
            continue
        active.append(row)
    active.sort(key=lambda row: (float(row.get("severity", 0.0)), int(row.get("support_count", 0))), reverse=True)
    return active[:max_active]


def _select_steps(all_steps: List[int], steps_arg: Optional[str], min_step: Optional[int], max_step: Optional[int]) -> List[int]:
    if steps_arg:
        selected = set()
        for part in steps_arg.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                lo, hi = part.split("-", 1)
                selected.update(range(int(lo), int(hi) + 1))
            else:
                selected.add(int(part))
        all_steps = [step for step in all_steps if step in selected]
    if min_step is not None:
        all_steps = [step for step in all_steps if step >= min_step]
    if max_step is not None:
        all_steps = [step for step in all_steps if step <= max_step]
    return all_steps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path, help="Rollout trace JSONL produced by --webshop_rollout_trace_path.")
    parser.add_argument("--rubric-pool", required=True, type=Path, help="Critic error pool JSONL.")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL with one row per training step.")
    parser.add_argument(
        "--open-instruct-root",
        type=Path,
        default=None,
        help="Path to dr-tulu/rl/open-instruct. Required if open_instruct is not already importable.",
    )
    parser.add_argument("--model", default="kimi-k2.6")
    parser.add_argument("--base-url", default="https://ark.cn-beijing.volces.com/api/plan/v3")
    parser.add_argument("--api-key-env", default="SEED_AGENT_PLAN_API_KEY")
    parser.add_argument("--max-active-rubrics", type=int, default=8)
    parser.add_argument("--trigger-threshold", type=float, default=0.55)
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--steps", default=None, help="Comma-separated steps/ranges, for example '1,64-66'.")
    parser.add_argument("--min-step", type=int, default=None)
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--max-rollouts-per-step", type=int, default=None)
    args = parser.parse_args()

    if args.open_instruct_root:
        sys.path.insert(0, str(args.open_instruct_root))

    from open_instruct.search_rewards.webshop_critic_error import (  # pylint: disable=import-error
        WebShopCriticErrorConfig,
        judge_critic_rubric,
    )

    trace_rows = _load_jsonl(args.trace)
    pool_rows = _load_jsonl(args.rubric_pool)
    by_step: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in trace_rows:
        step = row.get("training_step")
        if isinstance(step, int):
            by_step[step].append(row)

    steps = _select_steps(sorted(by_step), args.steps, args.min_step, args.max_step)
    config = WebShopCriticErrorConfig(
        critic_error_enabled=True,
        critic_error_judge_model=args.model,
        critic_error_openai_base_url=args.base_url,
        critic_error_api_key_env=args.api_key_env,
        critic_error_max_active_weaknesses=args.max_active_rubrics,
        critic_error_trigger_threshold=args.trigger_threshold,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for step in steps:
            rollouts = by_step[step]
            if args.max_rollouts_per_step is not None:
                rollouts = rollouts[: args.max_rollouts_per_step]
            active = _active_rubrics_at_step(pool_rows, step, args.max_active_rubrics)

            jobs = {}
            with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
                for rollout_index, rollout in enumerate(rollouts):
                    ground_truth = _unwrap_ground_truth(rollout.get("ground_truth"))
                    response = str(rollout.get("response") or "")
                    for rubric in active:
                        weakness_id = str(rubric.get("weakness_id") or rubric.get("id"))
                        fut = executor.submit(judge_critic_rubric, response, ground_truth, rubric, config)
                        jobs[fut] = (rollout_index, weakness_id)

                scored: List[Dict[str, Any]] = [
                    {
                        "index": int(rollout.get("index", idx)),
                        "trace_score": float(rollout.get("score", 0.0) or 0.0),
                        "finish_reason": rollout.get("finish_reason"),
                        "num_calls": int(rollout.get("num_calls", 0) or 0),
                        "tool_error": str(rollout.get("tool_error") or ""),
                        "critic_scores_by_id": {},
                        "critic_evidence_by_id": {},
                    }
                    for idx, rollout in enumerate(rollouts)
                ]
                for fut in as_completed(jobs):
                    rollout_index, weakness_id = jobs[fut]
                    judged = fut.result()
                    scored[rollout_index]["critic_scores_by_id"][weakness_id] = float(judged.get("score", 0.0) or 0.0)
                    scored[rollout_index]["critic_evidence_by_id"][weakness_id] = str(judged.get("evidence", ""))

            rubric_means: Dict[str, float] = {}
            for rubric in active:
                weakness_id = str(rubric.get("weakness_id") or rubric.get("id"))
                vals = [float(row["critic_scores_by_id"].get(weakness_id, 0.0)) for row in scored]
                rubric_means[weakness_id] = statistics.mean(vals) if vals else 0.0

            out.write(
                json.dumps(
                    {
                        "training_step": step,
                        "active_rubrics": active,
                        "rubric_score_means": rubric_means,
                        "rollouts": scored,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            print(f"wrote step {step}: rollouts={len(scored)} active_rubrics={len(active)}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
