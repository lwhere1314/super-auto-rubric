from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from .baseline import EpisodeResult, summarize_episode_results
from .client import WebShopClientProtocol
from .policies import ScriptedWebShopPolicy
from .rubric_pool import RubricEntry
from .trajectory import Trajectory, TrajectoryStep, save_trajectory, write_jsonl


class ChatClientProtocol(Protocol):
    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 512,
        response_format: dict[str, Any] | None = None,
    ) -> Any:
        ...


@dataclass
class PolicyDecision:
    action: str
    raw_response: str
    parse_ok: bool
    tool_valid: bool
    rationale: str = ""
    fallback_used: bool = False
    rubric_focus: list[str] = field(default_factory=list)


@dataclass
class StepValidityScore:
    step_index: int
    format_ok: bool
    tool_valid: bool
    reward: float
    action: str


@dataclass
class CriticRubricScore:
    rubric_id: str
    score: float
    weight: float
    contribution: float
    explanation: str = ""


@dataclass
class TrainingFreeRewardBreakdown:
    task_reward: float
    format_tool_validity_reward: float
    critic_rubric_judged_score_sum: float
    combined_reward: float
    critic_scores: list[CriticRubricScore] = field(default_factory=list)
    step_validity_scores: list[StepValidityScore] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FeedbackHint:
    feedback_id: str
    source_trajectory_id: str
    source_rubric_ids: list[str]
    trigger: str
    avoid_actions: list[str]
    lesson: str
    suggested_strategy: str
    severity: float
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeedbackHint":
        return cls(**data)


def load_active_rubrics(path: Path, *, limit: int | None = None) -> list[RubricEntry]:
    rubrics: list[RubricEntry] = []
    if not path.exists():
        return rubrics
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rubrics.append(RubricEntry.from_dict(json.loads(line)))
    ranked = sorted(
        rubrics,
        key=lambda item: (abs(item.weight), item.severity, item.support_count),
        reverse=True,
    )
    return ranked[:limit] if limit is not None else ranked


def load_feedback_hints(path: Path, *, limit: int | None = None) -> list[FeedbackHint]:
    hints: list[FeedbackHint] = []
    if not path.exists():
        return hints
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            hints.append(FeedbackHint.from_dict(json.loads(line)))
    ranked = sorted(hints, key=lambda item: item.severity, reverse=True)
    return ranked[:limit] if limit is not None else ranked


def save_feedback_hints(path: Path, hints: list[FeedbackHint]) -> None:
    write_jsonl(path, [hint.to_dict() for hint in hints])


def build_feedback_hints_from_trajectories(
    trajectories: list[Trajectory],
    *,
    max_hints: int = 20,
) -> list[FeedbackHint]:
    hints: list[FeedbackHint] = []
    seen_keys: set[tuple[str, str]] = set()
    for trajectory in trajectories:
        reward = trajectory.metadata.get("training_free_reward", {})
        critic_scores = reward.get("critic_scores", [])
        negative_scores = [score for score in critic_scores if float(score.get("contribution", 0.0)) < 0]
        if not negative_scores:
            continue

        actions = [step.action for step in trajectory.steps]
        repeated_actions = _repeated_items(actions)
        negative_text = " ".join(str(score.get("explanation", "")) for score in negative_scores).lower()
        source_rubric_ids = [str(score.get("rubric_id", "")) for score in negative_scores if score.get("rubric_id")]
        evidence = [
            f"{step.step_index}: {step.action}"
            for step in trajectory.steps[-4:]
        ]

        if repeated_actions or any(term in negative_text for term in ("repeat", "loop", "same result", "new evidence")):
            key = ("navigation_loop", _canonical_action_pattern(repeated_actions or actions[-3:]))
            if key not in seen_keys:
                seen_keys.add(key)
                hints.append(
                    FeedbackHint(
                        feedback_id=f"feedback_{uuid4().hex}",
                        source_trajectory_id=trajectory.trajectory_id,
                        source_rubric_ids=source_rubric_ids,
                        trigger="Prior critic penalized repeated search, paging, or returning to search without new evidence.",
                        avoid_actions=(repeated_actions or actions[-3:])[:4],
                        lesson="These actions received low critic reward because they did not gather new evidence or move toward a purchase.",
                        suggested_strategy=(
                            "Change the search path: keep the core product type and hard constraints, "
                            "drop brittle adjectives or punctuation-heavy terms, then inspect a plausible item "
                            "instead of repeatedly paging or repeating the same query."
                        ),
                        severity=max(abs(float(score.get("contribution", 0.0))) for score in negative_scores),
                        evidence=evidence,
                    )
                )

        if any(term in negative_text for term in ("no purchase", "constraints not satisfied", "not satisfied")):
            key = ("no_satisfying_purchase", _canonical_action_pattern(actions[-3:]))
            if key not in seen_keys:
                seen_keys.add(key)
                hints.append(
                    FeedbackHint(
                        feedback_id=f"feedback_{uuid4().hex}",
                        source_trajectory_id=trajectory.trajectory_id,
                        source_rubric_ids=source_rubric_ids,
                        trigger="Prior critic penalized ending the attempt without a satisfying purchase.",
                        avoid_actions=actions[-4:],
                        lesson="Only searching or paging until the budget ends gets poor reward even when every tool call is valid.",
                        suggested_strategy=(
                            "After one or two searches, inspect candidate product pages, verify attributes/options/price, "
                            "and choose the best available item if it satisfies the hard constraints."
                        ),
                        severity=max(abs(float(score.get("contribution", 0.0))) for score in negative_scores),
                        evidence=evidence,
                    )
                )

    ranked = sorted(hints, key=lambda item: item.severity, reverse=True)
    return ranked[:max_hints]


def is_action_tool_valid(action: str, available_actions: dict[str, Any]) -> bool:
    action = action.strip()
    search_match = re.fullmatch(r"search\[(.+)\]", action, flags=re.IGNORECASE | re.DOTALL)
    if search_match:
        return bool(available_actions.get("has_search_bar")) and bool(search_match.group(1).strip())

    click_match = re.fullmatch(r"click\[(.+)\]", action, flags=re.IGNORECASE | re.DOTALL)
    if not click_match:
        return False
    click_arg = click_match.group(1).strip().lower()
    clickables = {str(item).strip().lower() for item in available_actions.get("clickables", [])}
    return click_arg in clickables


def step_validity_reward(parse_ok: bool, tool_valid: bool) -> float:
    return 0.5 * float(parse_ok) + 0.5 * float(tool_valid)


class InContextRubricPolicy:
    """Training-free actor that injects active critic rubrics into each action prompt."""

    def __init__(
        self,
        *,
        chat_client: ChatClientProtocol,
        actor_model: str,
        rubrics: list[RubricEntry],
        feedback_hints: list[FeedbackHint] | None = None,
        include_recent_actions: bool = True,
        max_observation_chars: int = 4000,
    ) -> None:
        self.chat_client = chat_client
        self.actor_model = actor_model
        self.rubrics = rubrics
        self.feedback_hints = feedback_hints or []
        self.include_recent_actions = include_recent_actions
        self.max_observation_chars = max_observation_chars
        self.fallback = ScriptedWebShopPolicy()
        self.recent_actions: list[str] = []

    def reset(self) -> None:
        self.fallback.reset()
        self.recent_actions.clear()

    def choose_decision(
        self,
        observation: str,
        available_actions: dict[str, Any],
        instruction_text: str,
    ) -> PolicyDecision:
        messages = self._messages(observation, available_actions, instruction_text)
        try:
            completion = self.chat_client.complete(
                model=self.actor_model,
                messages=messages,
                temperature=0.0,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            raw_response = str(completion.content)
            parsed = _parse_json_object(raw_response)
            action = str(parsed.get("action", "")).strip()
            rationale = str(parsed.get("rationale", ""))
            rubric_focus = [str(item) for item in parsed.get("rubric_focus", []) if str(item).strip()]
            rubric_focus.extend(
                str(item) for item in parsed.get("feedback_focus", []) if str(item).strip()
            )
            parse_ok = bool(action)
        except Exception as exc:
            raw_response = f"{type(exc).__name__}: {exc}"
            action = ""
            rationale = ""
            rubric_focus = []
            parse_ok = False

        tool_valid = is_action_tool_valid(action, available_actions) if parse_ok else False
        if not tool_valid:
            action = self.fallback.choose_action(observation, available_actions, instruction_text)
            tool_valid = is_action_tool_valid(action, available_actions)
            decision = PolicyDecision(
                action=action,
                raw_response=raw_response,
                parse_ok=parse_ok,
                tool_valid=tool_valid,
                rationale=rationale,
                fallback_used=True,
                rubric_focus=rubric_focus,
            )
            self._record_action(decision.action)
            return decision

        decision = PolicyDecision(
            action=action,
            raw_response=raw_response,
            parse_ok=parse_ok,
            tool_valid=tool_valid,
            rationale=rationale,
            fallback_used=False,
            rubric_focus=rubric_focus,
        )
        self._record_action(decision.action)
        return decision

    def _record_action(self, action: str) -> None:
        self.recent_actions.append(action)
        self.recent_actions = self.recent_actions[-8:]

    def _messages(
        self,
        observation: str,
        available_actions: dict[str, Any],
        instruction_text: str,
    ) -> list[dict[str, str]]:
        rubrics_text = "\n".join(
            f"- {rubric.rubric_id} | {rubric.polarity} | weight={rubric.weight}: "
            f"{rubric.natural_language_rule}"
            for rubric in self.rubrics
        ) or "- No active critic rubrics."
        feedback_text = "\n".join(
            f"- {hint.feedback_id}: trigger={hint.trigger} "
            f"avoid={hint.avoid_actions} lesson={hint.lesson} try_instead={hint.suggested_strategy}"
            for hint in self.feedback_hints
        ) or "- No prior meta-harness feedback."
        clickables = [str(item) for item in available_actions.get("clickables", [])]
        allowed_actions = {
            "has_search_bar": bool(available_actions.get("has_search_bar")),
            "clickables": clickables[:80],
        }
        recent_actions = self.recent_actions[-6:] if self.include_recent_actions else []
        recent_action_guidance = (
            "Avoid repeating recent actions unless the observation clearly changed. If you have inspected a "
            "plausible product and Buy Now is available, choose required options and move "
            "toward purchase instead of going back or opening the same details page again."
            if self.include_recent_actions
            else "Recent action history is not provided in this ablation."
        )
        feedback_guidance = (
            "If a recent action appears in the allowed clickables again, prefer a different action "
            "that gathers new evidence or completes the purchase.\n\n"
            if self.include_recent_actions
            else "\n\n"
        )
        return [
            {
                "role": "system",
                "content": (
                    "You are a WebShop task agent. Choose exactly one next environment action. "
                    "Optimize this training-free reward: WebShop task reward + format/tool validity "
                    "reward + active critic rubric judged scores. Return JSON only with keys "
                    "`action`, `rationale`, and `rubric_focus`. Valid actions are `search[query]` "
                    "when a search bar is available, or `click[exact clickable text]` using one of "
                    f"the provided clickables. Do not invent tools or extra text. {recent_action_guidance}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Instruction:\n{instruction_text}\n\n"
                    f"Observation:\n{_truncate(observation, self.max_observation_chars)}\n\n"
                    f"Available actions JSON:\n{json.dumps(allowed_actions, ensure_ascii=True)}\n\n"
                    f"Recent actions JSON:\n{json.dumps(recent_actions, ensure_ascii=True)}\n\n"
                    f"Active critic rubrics:\n{rubrics_text}\n\n"
                    f"Meta-harness feedback from prior low-reward attempts:\n{feedback_text}\n\n"
                    "Use the feedback as counterfactual guidance: if an action pattern was penalized before, "
                    "choose a different search path or inspect a plausible product instead of repeating it. "
                    + feedback_guidance
                    +
                    "Return JSON like: "
                    '{"action":"search[green mug]","rationale":"...","rubric_focus":["rubric_id"],'
                    '"feedback_focus":["feedback_id"]}'
                ),
            },
        ]


class CriticRubricJudge:
    """Model-assisted judge for active critic rubrics over a completed trajectory."""

    def __init__(
        self,
        *,
        chat_client: ChatClientProtocol,
        critic_model: str,
        rubrics: list[RubricEntry],
        max_trajectory_chars: int = 9000,
    ) -> None:
        self.chat_client = chat_client
        self.critic_model = critic_model
        self.rubrics = rubrics
        self.max_trajectory_chars = max_trajectory_chars

    def judge(self, trajectory: Trajectory) -> list[CriticRubricScore]:
        if not self.rubrics:
            return []
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict WebShop critic. Score how well a completed trajectory "
                    "satisfies each active critic rubric. Return JSON only. For every rubric, "
                    "use score in [-1, 1]: 1 means the trajectory clearly satisfied or avoided "
                    "the weakness, 0 means unclear/not applicable, -1 means it violated the rubric."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Trajectory summary:\n{_truncate(_trajectory_for_prompt(trajectory), self.max_trajectory_chars)}\n\n"
                    f"Active rubrics JSON:\n{json.dumps([_rubric_for_prompt(item) for item in self.rubrics], ensure_ascii=True)}\n\n"
                    'Return JSON: {"scores":[{"rubric_id":"...","score":0.0,"explanation":"..."}]}'
                ),
            },
        ]
        try:
            completion = self.chat_client.complete(
                model=self.critic_model,
                messages=messages,
                temperature=0.0,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
            parsed = _parse_json_object(str(completion.content))
        except Exception:
            return []

        by_id = {rubric.rubric_id: rubric for rubric in self.rubrics}
        scores: list[CriticRubricScore] = []
        for item in parsed.get("scores", []):
            rubric_id = str(item.get("rubric_id", ""))
            if rubric_id not in by_id:
                continue
            rubric = by_id[rubric_id]
            score = _clamp(float(item.get("score", 0.0)), -1.0, 1.0)
            weight = abs(float(rubric.weight))
            scores.append(
                CriticRubricScore(
                    rubric_id=rubric_id,
                    score=score,
                    weight=weight,
                    contribution=score * weight,
                    explanation=str(item.get("explanation", "")),
                )
            )
        return scores


def run_training_free_icl_episode(
    client: WebShopClientProtocol,
    policy: InContextRubricPolicy,
    judge: CriticRubricJudge,
    *,
    split: str,
    max_steps: int,
    session_id: int | None,
    seed: int | None,
    actor_model: str,
    critic_model: str,
    rubric_version: str,
) -> tuple[EpisodeResult, TrainingFreeRewardBreakdown]:
    policy.reset()
    client.create()
    client.reset(session_id=session_id)
    instruction_text = client.instruction_text()
    trajectory = Trajectory.start(
        instruction_text=instruction_text,
        split=split,
        model=actor_model,
        policy=policy.__class__.__name__,
        prompt_version="training-free-icl-v0",
        rubric_version=rubric_version,
        seed=seed,
        session_id=session_id,
        metadata={
            "env_idx": client.env_idx,
            "training_free_mode": "in_context_critic_rubric_injection",
            "critic_model": critic_model,
            "active_rubric_ids": [rubric.rubric_id for rubric in policy.rubrics],
            "active_feedback_ids": [hint.feedback_id for hint in policy.feedback_hints],
        },
    )

    invalid_action_count = 0
    repeated_action_count = 0
    seen_actions: set[str] = set()
    step_scores: list[StepValidityScore] = []
    decisions: list[dict[str, Any]] = []

    for step_index in range(max_steps):
        observation_before = client.observation()
        available_actions = client.available_actions()
        state_before = client.state()
        decision = policy.choose_decision(observation_before, available_actions, instruction_text)
        if not decision.tool_valid:
            invalid_action_count += 1
        if decision.action in seen_actions:
            repeated_action_count += 1
        seen_actions.add(decision.action)

        body = client.step(decision.action)
        observation_after = str(body.get("state", client.observation()))
        step = TrajectoryStep(
            step_index=step_index,
            observation_before=observation_before,
            available_actions=available_actions,
            action=decision.action,
            reward=float(body.get("reward", 0.0)),
            done=bool(body.get("done", False)),
            observation_after=observation_after,
            state=state_before,
            info={
                "env_info": body.get("info"),
                "icl_decision": asdict(decision),
            },
        )
        trajectory.add_step(step)
        step_scores.append(
            StepValidityScore(
                step_index=step_index,
                format_ok=decision.parse_ok,
                tool_valid=decision.tool_valid,
                reward=step_validity_reward(decision.parse_ok, decision.tool_valid),
                action=decision.action,
            )
        )
        decisions.append(asdict(decision))
        if step.done:
            break

    critic_scores = judge.judge(trajectory)
    format_tool_reward = (
        sum(item.reward for item in step_scores) / len(step_scores) if step_scores else 0.0
    )
    critic_sum = sum(item.contribution for item in critic_scores)
    breakdown = TrainingFreeRewardBreakdown(
        task_reward=trajectory.final_reward,
        format_tool_validity_reward=format_tool_reward,
        critic_rubric_judged_score_sum=critic_sum,
        combined_reward=trajectory.final_reward + format_tool_reward + critic_sum,
        critic_scores=critic_scores,
        step_validity_scores=step_scores,
    )
    trajectory.metadata["training_free_reward"] = breakdown.to_dict()
    trajectory.metadata["icl_decisions"] = decisions
    return (
        EpisodeResult(
            trajectory=trajectory,
            invalid_action_count=invalid_action_count,
            repeated_action_count=repeated_action_count,
        ),
        breakdown,
    )


def save_training_free_results(
    output_dir: Path,
    results: list[tuple[EpisodeResult, TrainingFreeRewardBreakdown]],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_results = [item[0] for item in results]
    for result, _ in results:
        save_trajectory(output_dir / f"{result.trajectory.trajectory_id}.jsonl", result.trajectory)
    metrics = summarize_episode_results(episode_results)
    if results:
        metrics.update(
            {
                "average_format_tool_validity_reward": sum(
                    item[1].format_tool_validity_reward for item in results
                )
                / len(results),
                "average_critic_rubric_judged_score_sum": sum(
                    item[1].critic_rubric_judged_score_sum for item in results
                )
                / len(results),
                "average_combined_reward": sum(item[1].combined_reward for item in results)
                / len(results),
            }
        )
    write_jsonl(output_dir / "metrics.jsonl", [metrics])
    write_jsonl(output_dir / "reward_breakdowns.jsonl", [item[1].to_dict() for item in results])
    return metrics


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    parsed = json.loads(match.group(0))
    return parsed if isinstance(parsed, dict) else {}


def _repeated_items(items: list[str]) -> list[str]:
    seen: set[str] = set()
    repeated: list[str] = []
    for item in items:
        if item in seen and item not in repeated:
            repeated.append(item)
        seen.add(item)
    return repeated


def _canonical_action_pattern(actions: list[str]) -> str:
    normalized = []
    for action in actions:
        compact = re.sub(r"\s+", " ", action.lower()).strip()
        compact = re.sub(r"\[[^\]]+\]", "[...]", compact) if compact.startswith("search[") else compact
        normalized.append(compact)
    return " -> ".join(normalized)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n[TRUNCATED]"


def _rubric_for_prompt(rubric: RubricEntry) -> dict[str, Any]:
    return {
        "rubric_id": rubric.rubric_id,
        "title": rubric.title,
        "polarity": rubric.polarity,
        "weight": rubric.weight,
        "rule": rubric.natural_language_rule,
    }


def _trajectory_for_prompt(trajectory: Trajectory) -> str:
    lines = [
        f"Instruction: {trajectory.instruction_text}",
        f"Final task reward: {trajectory.final_reward}",
        f"Success: {trajectory.success}",
    ]
    for step in trajectory.steps:
        lines.append(
            "\n".join(
                [
                    f"Step {step.step_index}",
                    f"Observation before: {_truncate(step.observation_before, 900)}",
                    f"Available actions: {json.dumps(step.available_actions, ensure_ascii=True)[:1200]}",
                    f"Action: {step.action}",
                    f"Reward: {step.reward}",
                    f"Done: {step.done}",
                    f"Observation after: {_truncate(step.observation_after, 900)}",
                ]
            )
        )
    return "\n\n".join(lines)


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))
