from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4


SCHEMA_VERSION = "webshop-trajectory-v1"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TrajectoryStep:
    step_index: int
    observation_before: str
    available_actions: dict[str, Any]
    action: str
    reward: float
    done: bool
    observation_after: str
    state: dict[str, Any] | None = None
    info: dict[str, Any] | None = None


@dataclass
class Trajectory:
    trajectory_id: str
    benchmark: str
    env_name: str
    split: str
    instruction_text: str
    model: str
    policy: str
    prompt_version: str
    rubric_version: str
    seed: int | None
    session_id: int | None
    started_at: str
    ended_at: str | None = None
    final_reward: float = 0.0
    success: bool = False
    steps: list[TrajectoryStep] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    @classmethod
    def start(
        cls,
        *,
        instruction_text: str,
        split: str = "smoke",
        model: str = "scripted-policy",
        policy: str = "scripted-webshop-v0",
        prompt_version: str = "none",
        rubric_version: str = "none",
        seed: int | None = None,
        session_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "Trajectory":
        return cls(
            trajectory_id=f"traj_{uuid4().hex}",
            benchmark="webshop",
            env_name="AgentGym-WebShop",
            split=split,
            instruction_text=instruction_text,
            model=model,
            policy=policy,
            prompt_version=prompt_version,
            rubric_version=rubric_version,
            seed=seed,
            session_id=session_id,
            started_at=utc_now_iso(),
            metadata=metadata or {},
        )

    def add_step(self, step: TrajectoryStep) -> None:
        self.steps.append(step)
        self.final_reward = float(step.reward)
        self.success = self.final_reward >= 1.0
        if step.done:
            self.ended_at = utc_now_iso()

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Trajectory":
        steps = [TrajectoryStep(**step) for step in data.pop("steps", [])]
        trajectory = cls(**data)
        trajectory.steps = steps
        return trajectory


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def save_trajectory(path: Path, trajectory: Trajectory) -> None:
    write_jsonl(path, [trajectory.to_dict()])


def load_trajectories(path: Path) -> list[Trajectory]:
    paths: list[Path]
    if path.is_dir():
        paths = sorted(path.rglob("*.jsonl"))
    else:
        paths = [path]

    trajectories: list[Trajectory] = []
    for file_path in paths:
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                data = json.loads(line)
                if data.get("schema_version") == SCHEMA_VERSION:
                    trajectories.append(Trajectory.from_dict(data))
    return trajectories
