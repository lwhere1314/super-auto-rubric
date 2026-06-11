from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from .weakness import WeaknessCandidate


@dataclass
class RubricEntry:
    rubric_id: str
    title: str
    natural_language_rule: str
    polarity: str
    weight: float
    severity: float
    support_count: int
    source_weakness_ids: list[str]
    evidence_trajectory_ids: list[str]
    cluster_key: str
    source_cluster_key: str
    last_triggered_batch: int
    decay_score: float = 0.0
    status: str = "active"
    examples: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RubricEntry":
        data.setdefault("source_cluster_key", data["cluster_key"].rsplit(":", 1)[0])
        return cls(**data)


@dataclass
class RubricPool:
    max_active_rubrics: int = 5
    retire_after_non_triggered_batches: int = 3
    decay_per_non_triggered_batch: float = 0.2
    active: dict[str, RubricEntry] = field(default_factory=dict)
    retired: list[RubricEntry] = field(default_factory=list)
    quarantined: list[RubricEntry] = field(default_factory=list)

    def upsert_from_weaknesses(self, weaknesses: list[WeaknessCandidate], batch_id: int) -> None:
        grouped: dict[str, list[WeaknessCandidate]] = {}
        for weakness in weaknesses:
            grouped.setdefault(weakness.cluster_key or weakness.label, []).append(weakness)

        for cluster_key, group in grouped.items():
            for polarity in ("negative", "positive"):
                entry_key = f"{cluster_key}:{polarity}"
                if entry_key in self.active:
                    entry = self.active[entry_key]
                    entry.support_count += len(group)
                    entry.severity = max(entry.severity, max(item.severity for item in group))
                    entry.last_triggered_batch = batch_id
                    entry.decay_score = 0.0
                    entry.source_weakness_ids.extend(item.weakness_id for item in group)
                    entry.evidence_trajectory_ids.extend(item.trajectory_id for item in group)
                    entry.examples.extend(_examples_from_weaknesses(group))
                    continue

                representative = group[0]
                entry = RubricEntry(
                    rubric_id=f"rubric_{uuid4().hex}",
                    title=f"{_title_for_label(representative.label)} ({polarity})",
                    natural_language_rule=_rule_for_weakness(representative, polarity),
                    polarity=polarity,
                    weight=_weight_for_weakness(representative, polarity),
                    severity=max(item.severity for item in group),
                    support_count=len(group),
                    source_weakness_ids=[item.weakness_id for item in group],
                    evidence_trajectory_ids=[item.trajectory_id for item in group],
                    cluster_key=entry_key,
                    source_cluster_key=cluster_key,
                    last_triggered_batch=batch_id,
                    examples=_examples_from_weaknesses(group),
                )
                self.active[entry_key] = entry

        self._trim_active()

    def decay_and_retire(self, triggered_cluster_keys: set[str], batch_id: int) -> None:
        for cluster_key, entry in list(self.active.items()):
            if entry.source_cluster_key in triggered_cluster_keys:
                entry.decay_score = 0.0
                entry.last_triggered_batch = batch_id
                continue
            inactive_batches = batch_id - entry.last_triggered_batch
            entry.decay_score = min(1.0, inactive_batches * self.decay_per_non_triggered_batch)
            if inactive_batches >= self.retire_after_non_triggered_batches:
                entry.status = "retired"
                self.retired.append(entry)
                del self.active[cluster_key]

    def quarantine(self, cluster_key: str, reason: str) -> None:
        entry = self.active.pop(cluster_key)
        entry.status = "quarantined"
        entry.examples.append({"quarantine_reason": reason})
        self.quarantined.append(entry)

    def _trim_active(self) -> None:
        if len(self.active) <= self.max_active_rubrics:
            return
        ranked = sorted(
            self.active.items(),
            key=lambda item: (item[1].severity, item[1].support_count, -item[1].decay_score),
            reverse=True,
        )
        keep = {key for key, _ in ranked[: self.max_active_rubrics]}
        for key in list(self.active):
            if key not in keep:
                entry = self.active.pop(key)
                entry.status = "retired"
                self.retired.append(entry)

    def save(self, active_path: Path, retired_path: Path | None = None) -> None:
        _write_jsonl(active_path, [entry.to_dict() for entry in self.active.values()])
        if retired_path is not None:
            _write_jsonl(retired_path, [entry.to_dict() for entry in self.retired])

    @classmethod
    def load(cls, active_path: Path, **kwargs: Any) -> "RubricPool":
        pool = cls(**kwargs)
        if not active_path.exists():
            return pool
        with active_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                entry = RubricEntry.from_dict(json.loads(line))
                pool.active[entry.cluster_key] = entry
        return pool


def _title_for_label(label: str) -> str:
    return label.replace("_", " ").title()


def _weight_for_weakness(weakness: WeaknessCandidate, polarity: str) -> float:
    if polarity == "positive":
        return 1.0
    return -0.5 if weakness.severity < 0.8 else -1.0


def _rule_for_weakness(weakness: WeaknessCandidate, polarity: str) -> str:
    if polarity == "positive":
        return _positive_rule_for_weakness(weakness)
    return _negative_rule_for_weakness(weakness)


def _positive_rule_for_weakness(weakness: WeaknessCandidate) -> str:
    if weakness.label == "attribute_neglect":
        return "Reward final purchases that explicitly satisfy every required user constraint."
    if weakness.label == "query_drift":
        return "Reward search queries that preserve the important user constraints from the instruction."
    if weakness.label == "premature_purchase":
        return "Reward checking product details and plausible alternatives before buying."
    if weakness.label == "navigation_loop":
        return "Reward actions that gather new evidence or make clear progress toward purchase."
    if weakness.label == "observation_overtrust":
        return "Reward verifying relevant product details before trusting a product page."
    if weakness.label == "proxy_hacking":
        return "Reward behavior that improves task success rather than rubric-like wording alone."
    return f"Reward behavior that avoids recurring WebShop weakness: {weakness.description}"


def _negative_rule_for_weakness(weakness: WeaknessCandidate) -> str:
    if weakness.label == "attribute_neglect":
        return "Penalize final purchases that omit required user constraints before buying."
    if weakness.label == "query_drift":
        return "Penalize search queries that drop important user constraints from the instruction."
    if weakness.label == "premature_purchase":
        return "Penalize buying before enough product details and alternatives have been checked."
    if weakness.label == "navigation_loop":
        return "Penalize repeated actions that do not gather new evidence."
    if weakness.label == "observation_overtrust":
        return "Penalize trusting a product page without verifying relevant details."
    if weakness.label == "proxy_hacking":
        return "Penalize behavior that improves rubric-like text while missing task success."
    return f"Penalize recurring WebShop weakness: {weakness.description}"


def _examples_from_weaknesses(weaknesses: list[WeaknessCandidate]) -> list[dict[str, Any]]:
    examples = []
    for weakness in weaknesses:
        examples.append(
            {
                "weakness_id": weakness.weakness_id,
                "trajectory_id": weakness.trajectory_id,
                "evidence": [asdict(item) for item in weakness.evidence[:3]],
            }
        )
    return examples


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
