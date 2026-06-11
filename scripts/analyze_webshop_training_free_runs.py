#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from super_auto_rubric.webshop.trajectory import Trajectory, load_trajectories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze one or more training-free WebShop run dirs.")
    parser.add_argument("runs", nargs="+", help="Run directories containing trajectory JSONL files.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_summaries = []
    trajectories_by_run = []
    for run in args.runs:
        path = Path(run)
        trajectories = load_trajectories(path)
        trajectories_by_run.append((path, trajectories))
        run_summaries.append(_summarize_run(path, trajectories))

    result: dict[str, Any] = {"runs": run_summaries}
    if len(trajectories_by_run) == 2:
        result["paired_comparison"] = _paired_comparison(
            trajectories_by_run[0][0],
            trajectories_by_run[0][1],
            trajectories_by_run[1][0],
            trajectories_by_run[1][1],
            bootstrap_samples=args.bootstrap,
        )

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    return 0


def _summarize_run(path: Path, trajectories: list[Trajectory]) -> dict[str, Any]:
    rewards = [_task_reward(item) for item in trajectories]
    critic_sums = [_critic_sum(item) for item in trajectories]
    combined = [_combined_reward(item) for item in trajectories]
    format_scores = [_format_reward(item) for item in trajectories]
    steps = [item.step_count for item in trajectories]
    action_counts: Counter[str] = Counter()
    buy_episodes = 0
    repeated_action_steps = 0
    total_steps = 0
    task_zero_critic_positive = 0
    task_positive_critic_negative = 0

    for trajectory in trajectories:
        actions = [step.action for step in trajectory.steps]
        seen: set[str] = set()
        for action in actions:
            action_counts[_action_type(action)] += 1
            total_steps += 1
            if action in seen:
                repeated_action_steps += 1
            seen.add(action)
        if any(_action_type(action) == "buy" for action in actions):
            buy_episodes += 1
        task_reward = _task_reward(trajectory)
        critic_sum = _critic_sum(trajectory)
        if task_reward == 0 and critic_sum > 0:
            task_zero_critic_positive += 1
        if task_reward > 0 and critic_sum < 0:
            task_positive_critic_negative += 1

    return {
        "run": str(path),
        "episodes": len(trajectories),
        "success_rate": _fraction(item.success for item in trajectories),
        "average_task_reward": _safe_mean(rewards),
        "average_critic_sum": _safe_mean(critic_sums),
        "average_combined_reward": _safe_mean(combined),
        "average_format_tool_reward": _safe_mean(format_scores),
        "average_steps": _safe_mean(steps),
        "partial_reward_rate": _fraction(0 < reward < 1 for reward in rewards),
        "buy_episode_rate": buy_episodes / len(trajectories) if trajectories else 0.0,
        "repeated_action_step_rate": repeated_action_steps / total_steps if total_steps else 0.0,
        "task_zero_critic_positive_count": task_zero_critic_positive,
        "task_positive_critic_negative_count": task_positive_critic_negative,
        "action_counts": dict(sorted(action_counts.items())),
    }


def _paired_comparison(
    left_path: Path,
    left: list[Trajectory],
    right_path: Path,
    right: list[Trajectory],
    *,
    bootstrap_samples: int,
) -> dict[str, Any]:
    left_by_session = {_session_id(item): item for item in left if _session_id(item) is not None}
    right_by_session = {_session_id(item): item for item in right if _session_id(item) is not None}
    sessions = sorted(set(left_by_session) & set(right_by_session))
    task_deltas = [_task_reward(right_by_session[item]) - _task_reward(left_by_session[item]) for item in sessions]
    success_deltas = [
        float(right_by_session[item].success) - float(left_by_session[item].success)
        for item in sessions
    ]
    critic_deltas = [_critic_sum(right_by_session[item]) - _critic_sum(left_by_session[item]) for item in sessions]
    loop_deltas = [
        _repeated_rate(right_by_session[item]) - _repeated_rate(left_by_session[item])
        for item in sessions
    ]
    improved = [
        {
            "session_id": session,
            "left_reward": _task_reward(left_by_session[session]),
            "right_reward": _task_reward(right_by_session[session]),
            "delta": _task_reward(right_by_session[session]) - _task_reward(left_by_session[session]),
            "right_actions": [step.action for step in right_by_session[session].steps],
        }
        for session in sessions
        if _task_reward(right_by_session[session]) > _task_reward(left_by_session[session])
    ]
    worsened = [
        {
            "session_id": session,
            "left_reward": _task_reward(left_by_session[session]),
            "right_reward": _task_reward(right_by_session[session]),
            "delta": _task_reward(right_by_session[session]) - _task_reward(left_by_session[session]),
            "right_actions": [step.action for step in right_by_session[session].steps],
        }
        for session in sessions
        if _task_reward(right_by_session[session]) < _task_reward(left_by_session[session])
    ]

    return {
        "left_run": str(left_path),
        "right_run": str(right_path),
        "paired_sessions": len(sessions),
        "average_task_reward_delta": _safe_mean(task_deltas),
        "task_reward_delta_bootstrap_95ci": _bootstrap_ci(task_deltas, bootstrap_samples),
        "success_rate_delta": _safe_mean(success_deltas),
        "success_delta_bootstrap_95ci": _bootstrap_ci(success_deltas, bootstrap_samples),
        "average_critic_sum_delta": _safe_mean(critic_deltas),
        "average_repeated_action_rate_delta": _safe_mean(loop_deltas),
        "task_reward_improved_count": len(improved),
        "task_reward_worsened_count": len(worsened),
        "task_reward_unchanged_count": len(sessions) - len(improved) - len(worsened),
        "success_gained_count": sum(
            bool(right_by_session[item].success) and not bool(left_by_session[item].success)
            for item in sessions
        ),
        "success_lost_count": sum(
            bool(left_by_session[item].success) and not bool(right_by_session[item].success)
            for item in sessions
        ),
        "top_improvements": sorted(improved, key=lambda item: item["delta"], reverse=True)[:8],
        "top_regressions": sorted(worsened, key=lambda item: item["delta"])[:8],
    }


def _task_reward(trajectory: Trajectory) -> float:
    return float(trajectory.final_reward)


def _critic_sum(trajectory: Trajectory) -> float:
    reward = trajectory.metadata.get("training_free_reward", {})
    return float(reward.get("critic_rubric_judged_score_sum", 0.0))


def _combined_reward(trajectory: Trajectory) -> float:
    reward = trajectory.metadata.get("training_free_reward", {})
    return float(reward.get("combined_reward", trajectory.final_reward))


def _format_reward(trajectory: Trajectory) -> float:
    reward = trajectory.metadata.get("training_free_reward", {})
    return float(reward.get("format_tool_validity_reward", 0.0))


def _session_id(trajectory: Trajectory) -> int | None:
    session_id = trajectory.session_id
    return int(session_id) if session_id is not None else None


def _repeated_rate(trajectory: Trajectory) -> float:
    actions = [step.action for step in trajectory.steps]
    if not actions:
        return 0.0
    seen: set[str] = set()
    repeats = 0
    for action in actions:
        if action in seen:
            repeats += 1
        seen.add(action)
    return repeats / len(actions)


def _action_type(action: str) -> str:
    normalized = action.strip().lower()
    if normalized.startswith("search["):
        return "search"
    match = re.fullmatch(r"click\[(.*)\]", normalized)
    if not match:
        return "invalid_or_other"
    label = match.group(1).strip()
    if label == "buy now":
        return "buy"
    if label in {"next >", "< prev", "prev", "back to search"}:
        return "navigation"
    if label in {"description", "features", "reviews", "details"}:
        return "details"
    return "product_or_option"


def _fraction(values: Any) -> float:
    items = list(values)
    return sum(bool(item) for item in items) / len(items) if items else 0.0


def _safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _bootstrap_ci(values: list[float], samples: int) -> list[float]:
    if not values or samples <= 0:
        return [0.0, 0.0]
    rng = random.Random(0)
    means = []
    for _ in range(samples):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(mean(sample))
    means.sort()
    lower = means[int(0.025 * (len(means) - 1))]
    upper = means[int(0.975 * (len(means) - 1))]
    return [lower, upper]


if __name__ == "__main__":
    raise SystemExit(main())
