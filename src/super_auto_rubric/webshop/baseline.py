from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .client import WebShopClientProtocol
from .policies import ScriptedWebShopPolicy
from .trajectory import Trajectory, TrajectoryStep, save_trajectory, write_jsonl


@dataclass
class EpisodeResult:
    trajectory: Trajectory
    invalid_action_count: int
    repeated_action_count: int


def _is_invalid_step(step_body: dict[str, Any]) -> bool:
    state = str(step_body.get("state", "")).lower()
    return "invalid action" in state


def run_episode(
    client: WebShopClientProtocol,
    policy: ScriptedWebShopPolicy,
    *,
    split: str,
    max_steps: int,
    session_id: int | None,
    seed: int | None,
    model: str = "scripted-policy",
    prompt_version: str = "scripted-v0",
    rubric_version: str = "none",
) -> EpisodeResult:
    policy.reset()
    client.create()
    client.reset(session_id=session_id)
    instruction_text = client.instruction_text()
    trajectory = Trajectory.start(
        instruction_text=instruction_text,
        split=split,
        model=model,
        policy=policy.__class__.__name__,
        prompt_version=prompt_version,
        rubric_version=rubric_version,
        seed=seed,
        session_id=session_id,
        metadata={"env_idx": client.env_idx},
    )

    invalid_action_count = 0
    repeated_action_count = 0
    seen_actions: set[str] = set()

    for step_index in range(max_steps):
        observation_before = client.observation()
        available_actions = client.available_actions()
        state_before = client.state()
        action = policy.choose_action(observation_before, available_actions, instruction_text)
        if action in seen_actions:
            repeated_action_count += 1
        seen_actions.add(action)

        body = client.step(action)
        if _is_invalid_step(body):
            invalid_action_count += 1
        observation_after = str(body.get("state", client.observation()))
        step = TrajectoryStep(
            step_index=step_index,
            observation_before=observation_before,
            available_actions=available_actions,
            action=action,
            reward=float(body.get("reward", 0.0)),
            done=bool(body.get("done", False)),
            observation_after=observation_after,
            state=state_before,
            info=body.get("info"),
        )
        trajectory.add_step(step)
        if step.done:
            break

    return EpisodeResult(
        trajectory=trajectory,
        invalid_action_count=invalid_action_count,
        repeated_action_count=repeated_action_count,
    )


def summarize_episode_results(results: list[EpisodeResult]) -> dict[str, Any]:
    if not results:
        return {
            "episodes": 0,
            "success_rate": 0.0,
            "average_final_reward": 0.0,
            "average_steps": 0.0,
            "invalid_action_rate": 0.0,
            "loop_rate": 0.0,
        }

    total_steps = sum(result.trajectory.step_count for result in results)
    total_invalid = sum(result.invalid_action_count for result in results)
    total_repeated = sum(result.repeated_action_count for result in results)
    return {
        "episodes": len(results),
        "success_rate": sum(1 for result in results if result.trajectory.success) / len(results),
        "average_final_reward": sum(result.trajectory.final_reward for result in results) / len(results),
        "average_steps": total_steps / len(results),
        "invalid_action_rate": total_invalid / max(total_steps, 1),
        "loop_rate": total_repeated / max(total_steps, 1),
    }


def save_batch_results(output_dir: Path, results: list[EpisodeResult]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        save_trajectory(output_dir / f"{result.trajectory.trajectory_id}.jsonl", result.trajectory)
    metrics = summarize_episode_results(results)
    write_jsonl(output_dir / "metrics.jsonl", [metrics])
    return metrics
