from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4

from .trajectory import Trajectory, TrajectoryStep


ATTRIBUTE_TERMS = {
    "black",
    "blue",
    "green",
    "red",
    "white",
    "pink",
    "small",
    "medium",
    "large",
    "x-large",
    "washable",
    "wireless",
    "cotton",
    "leather",
    "stainless",
    "steel",
}


@dataclass
class EvidenceSpan:
    step_index: int
    field: str
    text: str


@dataclass
class WeaknessCandidate:
    weakness_id: str
    trajectory_id: str
    label: str
    phase: str
    description: str
    severity: float
    evidence: list[EvidenceSpan]
    missing_terms: list[str] = field(default_factory=list)
    cluster_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WeaknessCandidate":
        evidence = [EvidenceSpan(**item) for item in data.pop("evidence", [])]
        return cls(**data, evidence=evidence)


def extract_instruction_terms(instruction: str) -> set[str]:
    words = set(re.findall(r"[a-z0-9-]+", instruction.lower()))
    terms = {word for word in words if word in ATTRIBUTE_TERMS}
    price_match = re.search(r"(?:under|lower than|less than)\s+([0-9]+(?:\.[0-9]+)?)", instruction.lower())
    if price_match:
        terms.add(f"price<{price_match.group(1)}")
    return terms


def classify_phase(step: TrajectoryStep) -> str:
    action = step.action.lower()
    observation = step.observation_before.lower()
    if action.startswith("search["):
        return "search"
    if "buy now" in action or "done" in observation:
        return "purchase"
    if "item page" in observation:
        return "inspect"
    if "results" in observation:
        return "compare"
    return "decide"


def _short_evidence(step: TrajectoryStep, field: str, text: str, limit: int = 240) -> EvidenceSpan:
    compact = " ".join(text.split())
    return EvidenceSpan(step_index=step.step_index, field=field, text=compact[:limit])


def _find_repeated_actions(trajectory: Trajectory) -> list[WeaknessCandidate]:
    candidates: list[WeaknessCandidate] = []
    seen: dict[str, TrajectoryStep] = {}
    for step in trajectory.steps:
        if step.action in seen:
            first = seen[step.action]
            candidates.append(
                WeaknessCandidate(
                    weakness_id=f"weak_{uuid4().hex}",
                    trajectory_id=trajectory.trajectory_id,
                    label="navigation_loop",
                    phase=classify_phase(step),
                    description=f"Repeated action `{step.action}` without clear new evidence.",
                    severity=0.6,
                    evidence=[
                        _short_evidence(first, "action", first.action),
                        _short_evidence(step, "action", step.action),
                        _short_evidence(step, "observation_before", step.observation_before),
                    ],
                )
            )
            break
        seen[step.action] = step
    return candidates


def _find_premature_purchase(trajectory: Trajectory) -> list[WeaknessCandidate]:
    for step in trajectory.steps:
        if "buy now" not in step.action.lower():
            continue
        prior_inspections = [
            item for item in trajectory.steps[: step.step_index] if classify_phase(item) in {"inspect", "compare"}
        ]
        if len(prior_inspections) < 2 and trajectory.final_reward < 1.0:
            return [
                WeaknessCandidate(
                    weakness_id=f"weak_{uuid4().hex}",
                    trajectory_id=trajectory.trajectory_id,
                    label="premature_purchase",
                    phase="purchase",
                    description="Purchased before collecting enough comparison or attribute evidence.",
                    severity=0.8,
                    evidence=[
                        _short_evidence(step, "action", step.action),
                        _short_evidence(step, "observation_before", step.observation_before),
                    ],
                )
            ]
    return []


def _find_attribute_neglect(trajectory: Trajectory) -> list[WeaknessCandidate]:
    if trajectory.final_reward >= 1.0:
        return []
    required = extract_instruction_terms(trajectory.instruction_text)
    if not required:
        return []
    final_text = " ".join(
        [
            trajectory.steps[-1].observation_after if trajectory.steps else "",
            str(trajectory.steps[-1].info or {}) if trajectory.steps else "",
        ]
    ).lower()
    missing = sorted(term for term in required if term not in final_text)
    if not missing:
        return []
    step = trajectory.steps[-1]
    return [
        WeaknessCandidate(
            weakness_id=f"weak_{uuid4().hex}",
            trajectory_id=trajectory.trajectory_id,
            label="attribute_neglect",
            phase=classify_phase(step),
            description="Final state appears to miss required instruction constraints: " + ", ".join(missing),
            severity=0.75,
            evidence=[
                EvidenceSpan(0, "instruction_text", trajectory.instruction_text),
                _short_evidence(step, "observation_after", step.observation_after),
            ],
            missing_terms=missing,
        )
    ]


def _find_query_drift(trajectory: Trajectory) -> list[WeaknessCandidate]:
    required = extract_instruction_terms(trajectory.instruction_text)
    if not required:
        return []
    for step in trajectory.steps:
        if not step.action.startswith("search["):
            continue
        query = step.action.removeprefix("search[").removesuffix("]").lower()
        covered = {term for term in required if term.split("<", 1)[0] in query}
        if len(covered) < max(1, len(required) // 3):
            return [
                WeaknessCandidate(
                    weakness_id=f"weak_{uuid4().hex}",
                    trajectory_id=trajectory.trajectory_id,
                    label="query_drift",
                    phase="search",
                    description="Search query omitted most required instruction constraints.",
                    severity=0.55,
                    evidence=[
                        EvidenceSpan(0, "instruction_text", trajectory.instruction_text),
                        _short_evidence(step, "action", step.action),
                    ],
                    missing_terms=sorted(required - covered),
                )
            ]
    return []


def _find_observation_overtrust(trajectory: Trajectory) -> list[WeaknessCandidate]:
    if trajectory.final_reward >= 1.0 or not trajectory.steps:
        return []
    for step in trajectory.steps:
        if classify_phase(step) == "inspect" and "details" not in step.action.lower():
            if any("buy now" in item.action.lower() for item in trajectory.steps[step.step_index + 1 :]):
                return [
                    WeaknessCandidate(
                        weakness_id=f"weak_{uuid4().hex}",
                        trajectory_id=trajectory.trajectory_id,
                        label="observation_overtrust",
                        phase="inspect",
                        description="Moved from a product page toward purchase without checking additional details.",
                        severity=0.5,
                        evidence=[
                            _short_evidence(step, "observation_before", step.observation_before),
                        ],
                    )
                ]
    return []


def mine_weaknesses(trajectory: Trajectory) -> list[WeaknessCandidate]:
    candidates: list[WeaknessCandidate] = []
    candidates.extend(_find_repeated_actions(trajectory))
    candidates.extend(_find_premature_purchase(trajectory))
    candidates.extend(_find_attribute_neglect(trajectory))
    candidates.extend(_find_query_drift(trajectory))
    candidates.extend(_find_observation_overtrust(trajectory))
    for candidate in candidates:
        terms = "-".join(candidate.missing_terms[:4]) if candidate.missing_terms else "general"
        candidate.cluster_key = f"{candidate.label}:{terms}"
    return candidates


def mine_batch(trajectories: list[Trajectory]) -> list[WeaknessCandidate]:
    candidates: list[WeaknessCandidate] = []
    for trajectory in trajectories:
        if trajectory.success:
            continue
        candidates.extend(mine_weaknesses(trajectory))
    return candidates
