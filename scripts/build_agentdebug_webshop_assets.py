#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from super_auto_rubric.webshop.rubric_pool import RubricEntry
from super_auto_rubric.webshop.training_free_icl import FeedbackHint
from super_auto_rubric.webshop.trajectory import write_jsonl


FEEDBACK_TEMPLATES: dict[tuple[str, str], dict[str, Any]] = {
    ("plan", "inefficient_plan"): {
        "trigger": "AgentDebug WebShop labels mark near-duplicate searches, repeated paging, or returning to the initial search as critical planning failures.",
        "avoid_actions": ["search[near-duplicate query]", "click[next >] loops", "click[back to search] after finding a plausible item"],
        "lesson": "Repeating a query or paging pattern without new evidence usually burns the step budget and prevents a purchase.",
        "suggested_strategy": "Keep the product type and hard constraints, simplify brittle adjectives, then inspect a plausible product instead of repeating the same search path.",
    },
    ("plan", "constraint_ignorance"): {
        "trigger": "AgentDebug WebShop labels mark search plans that drop hard user constraints or switch product scope as critical failures.",
        "avoid_actions": ["search[query missing color/size/price/product type]", "search[changed product category]"],
        "lesson": "A new query can look reasonable but still fail if it loses mandatory constraints from the instruction.",
        "suggested_strategy": "Preserve product type plus hard constraints such as gender, color, size, material, washable, and price; only remove low-value descriptive words.",
    },
    ("memory", "hallucination"): {
        "trigger": "AgentDebug WebShop labels mark cases where the agent claims no relevant products exist even when the observation contains plausible candidates.",
        "avoid_actions": ["dismiss relevant results", "search[new query] before checking obvious candidate"],
        "lesson": "Misremembering visible candidate products as irrelevant causes unnecessary search resets.",
        "suggested_strategy": "When results contain plausible titles, click one candidate and verify options/details before concluding the search failed.",
    },
    ("memory", "over_simplification"): {
        "trigger": "AgentDebug WebShop labels mark summaries that collapse partial evidence into one shallow criterion and forget unknown attributes.",
        "avoid_actions": ["forget unchecked color/size/material", "treat partial match as fully rejected"],
        "lesson": "Losing partial evidence makes later steps repeat search instead of using known candidates.",
        "suggested_strategy": "Track which constraints are satisfied, unknown, or missing; use details/options pages to resolve unknown attributes before abandoning a product.",
    },
    ("reflection", "progress_misjudge"): {
        "trigger": "AgentDebug WebShop labels mark reflections that misjudge progress, especially when the agent has already found a plausible item.",
        "avoid_actions": ["search[more options] after plausible item", "click[back to search] when Buy Now/options are available"],
        "lesson": "Over-exploring after finding a plausible product can block completion.",
        "suggested_strategy": "If a plausible product page is open, choose required options, inspect details only if needed, and move toward Buy Now.",
    },
    ("reflection", "causal_misattribution"): {
        "trigger": "AgentDebug WebShop labels mark reflections that blame the wrong cause of failure and choose the wrong correction.",
        "avoid_actions": ["repeat failed correction", "change unrelated search terms"],
        "lesson": "Wrong failure diagnosis leads to repeated low-reward search paths.",
        "suggested_strategy": "Diagnose the missing constraint directly from the instruction and observation, then change only the action needed to verify or satisfy it.",
    },
    ("action", "parameter_error"): {
        "trigger": "AgentDebug WebShop labels mark wrong action parameters as critical failures.",
        "avoid_actions": ["click[text not in clickables]", "search[malformed or irrelevant parameter]"],
        "lesson": "A syntactically valid tool call can still fail if its argument does not match the current page.",
        "suggested_strategy": "Click exact visible clickable text; for search, use a concise query anchored in the product type and hard constraints.",
    },
    ("action", "misalignment"): {
        "trigger": "AgentDebug WebShop labels mark actions that are valid but misaligned with the task objective as critical failures.",
        "avoid_actions": ["click[irrelevant product]", "click[option conflicting with instruction]"],
        "lesson": "Valid clicks can still reduce reward when they move away from the requested product constraints.",
        "suggested_strategy": "Prefer actions that verify or satisfy the most restrictive remaining constraint.",
    },
    ("system", "environment_error"): {
        "trigger": "AgentDebug WebShop labels include environment/task issues such as contradictory or low-yield instructions.",
        "avoid_actions": ["overfit conflicting wording", "loop after impossible exact match"],
        "lesson": "When the task wording is noisy or contradictory, exact matching can cause loops.",
        "suggested_strategy": "Preserve the core product category and hard constraints, then choose the best observable candidate rather than looping on impossible wording.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert AgentDebug AgentErrorBench WebShop labels into rubrics and ICL feedback hints."
    )
    parser.add_argument("--labels-zip", required=True, help="Path to Label-*.zip from AgentDebug.")
    parser.add_argument("--summary-output", default="artifacts/agentdebug/webshop_label_summary.json")
    parser.add_argument("--feedback-output", default="artifacts/feedback/agentdebug-webshop-feedback.jsonl")
    parser.add_argument("--rubrics-output", default="artifacts/rubrics/agentdebug-webshop-active.jsonl")
    parser.add_argument(
        "--merge-feedback",
        action="append",
        default=[],
        help="Existing feedback JSONL file to merge with AgentDebug hints. May be repeated.",
    )
    parser.add_argument("--merged-feedback-output", default=None)
    parser.add_argument(
        "--merge-rubrics",
        action="append",
        default=[],
        help="Existing rubric JSONL file to merge with AgentDebug rubrics. May be repeated.",
    )
    parser.add_argument("--merged-rubrics-output", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    labels = _read_webshop_labels(Path(args.labels_zip))
    annotations = _extract_annotations(labels)
    grouped = _group_annotations(annotations)
    feedback_hints = _build_feedback_hints(grouped)
    rubrics = _build_rubrics(grouped)
    summary = _build_summary(labels, annotations, feedback_hints, rubrics)

    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_jsonl(Path(args.feedback_output), [hint.to_dict() for hint in feedback_hints])
    write_jsonl(Path(args.rubrics_output), [rubric.to_dict() for rubric in rubrics])

    merged_output = args.merged_feedback_output
    if merged_output:
        merged_hints = _load_feedback_files([Path(item) for item in args.merge_feedback])
        merged_hints.extend(feedback_hints)
        merged_hints = _dedupe_hints(merged_hints)
        write_jsonl(Path(merged_output), [hint.to_dict() for hint in merged_hints])

    merged_rubrics_output = args.merged_rubrics_output
    if merged_rubrics_output:
        merged_rubrics = _load_rubric_files([Path(item) for item in args.merge_rubrics])
        merged_rubrics.extend(rubrics)
        merged_rubrics = _dedupe_rubrics(merged_rubrics)
        write_jsonl(Path(merged_rubrics_output), [rubric.to_dict() for rubric in merged_rubrics])

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _read_webshop_labels(path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        with archive.open("Label/webshop_labels.json") as handle:
            return json.loads(handle.read().decode("utf-8"))


def _extract_annotations(labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    annotations: list[dict[str, Any]] = []
    for label in labels:
        for step_annotation in label.get("step_annotations", []):
            step = int(step_annotation.get("step", label.get("critical_failure_step", -1)))
            for module, value in step_annotation.items():
                if module == "step" or not isinstance(value, dict):
                    continue
                failure_type = str(value.get("failure_type", "")).strip()
                module, failure_type = _normalize_failure_type(module, failure_type)
                annotations.append(
                    {
                        "trajectory_id": label["trajectory_id"],
                        "llm": label.get("LLM", ""),
                        "module": module,
                        "failure_type": failure_type,
                        "failure_key": f"{module}.{failure_type}",
                        "critical_failure_step": int(label.get("critical_failure_step", step)),
                        "annotation_step": step,
                        "reasoning": str(value.get("reasoning", "")).strip(),
                    }
                )
    return annotations


def _normalize_failure_type(module: str, failure_type: str) -> tuple[str, str]:
    if module == "plan" and failure_type == "plan_inefficient":
        return module, "inefficient_plan"
    if not failure_type:
        return module, "unspecified"
    return module, failure_type


def _group_annotations(annotations: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        grouped[(annotation["module"], annotation["failure_type"])].append(annotation)
    return dict(grouped)


def _build_feedback_hints(
    grouped: dict[tuple[str, str], list[dict[str, Any]]]
) -> list[FeedbackHint]:
    hints = []
    for key, rows in grouped.items():
        template = FEEDBACK_TEMPLATES.get(key) or _generic_template(key)
        support = len(rows)
        severity = _severity_for_group(key, support)
        evidence = [
            _evidence_line(row)
            for row in sorted(rows, key=lambda item: (item["llm"], item["trajectory_id"]))[:5]
        ]
        hints.append(
            FeedbackHint(
                feedback_id=f"agentdebug_{key[0]}_{key[1]}",
                source_trajectory_id=",".join(row["trajectory_id"] for row in rows[:5]),
                source_rubric_ids=[f"agentdebug:{key[0]}.{key[1]}"],
                trigger=f"{template['trigger']} support={support}/50.",
                avoid_actions=list(template["avoid_actions"]),
                lesson=template["lesson"],
                suggested_strategy=template["suggested_strategy"],
                severity=severity,
                evidence=evidence,
            )
        )
    return sorted(hints, key=lambda item: (item.severity, item.feedback_id), reverse=True)


def _build_rubrics(grouped: dict[tuple[str, str], list[dict[str, Any]]]) -> list[RubricEntry]:
    rubrics = []
    for key, rows in grouped.items():
        template = FEEDBACK_TEMPLATES.get(key) or _generic_template(key)
        support = len(rows)
        severity = _severity_for_group(key, support)
        rubrics.append(
            RubricEntry(
                rubric_id=f"agentdebug_{key[0]}_{key[1]}",
                title=f"AgentDebug {key[0].title()} {key[1].replace('_', ' ').title()}",
                natural_language_rule=(
                    f"Reward trajectories that avoid this AgentDebug WebShop critical failure: "
                    f"{template['lesson']} Try instead: {template['suggested_strategy']}"
                ),
                polarity="negative",
                weight=1.0 if severity >= 0.8 else 0.75,
                severity=severity,
                support_count=support,
                source_weakness_ids=[f"agentdebug:{row['trajectory_id']}:{row['annotation_step']}" for row in rows],
                evidence_trajectory_ids=[row["trajectory_id"] for row in rows],
                cluster_key=f"agentdebug:{key[0]}.{key[1]}:negative",
                source_cluster_key=f"agentdebug:{key[0]}.{key[1]}",
                last_triggered_batch=0,
                examples=[
                    {
                        "trajectory_id": row["trajectory_id"],
                        "critical_failure_step": row["critical_failure_step"],
                        "evidence": [{"field": "agentdebug_reasoning", "step_index": row["annotation_step"], "text": row["reasoning"]}],
                    }
                    for row in rows[:3]
                ],
            )
        )
    return sorted(rubrics, key=lambda item: (item.severity, item.support_count), reverse=True)


def _build_summary(
    labels: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    feedback_hints: list[FeedbackHint],
    rubrics: list[RubricEntry],
) -> dict[str, Any]:
    return {
        "label_count": len(labels),
        "annotation_count": len(annotations),
        "model_counts": dict(sorted(Counter(label.get("LLM", "") for label in labels).items())),
        "module_counts": dict(sorted(Counter(item["module"] for item in annotations).items())),
        "failure_type_counts": dict(sorted(Counter(item["failure_key"] for item in annotations).items())),
        "feedback_hint_count": len(feedback_hints),
        "rubric_count": len(rubrics),
        "top_feedback_ids": [hint.feedback_id for hint in feedback_hints[:8]],
    }


def _load_feedback_files(paths: list[Path]) -> list[FeedbackHint]:
    hints: list[FeedbackHint] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    hints.append(FeedbackHint.from_dict(json.loads(line)))
    return hints


def _load_rubric_files(paths: list[Path]) -> list[RubricEntry]:
    rubrics: list[RubricEntry] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rubrics.append(RubricEntry.from_dict(json.loads(line)))
    return rubrics


def _dedupe_hints(hints: list[FeedbackHint]) -> list[FeedbackHint]:
    by_id: dict[str, FeedbackHint] = {}
    for hint in hints:
        by_id[hint.feedback_id] = hint
    return sorted(by_id.values(), key=lambda item: (item.severity, item.feedback_id), reverse=True)


def _dedupe_rubrics(rubrics: list[RubricEntry]) -> list[RubricEntry]:
    by_id: dict[str, RubricEntry] = {}
    for rubric in rubrics:
        by_id[rubric.rubric_id] = rubric
    return sorted(
        by_id.values(),
        key=lambda item: (abs(item.weight), item.severity, item.support_count, item.rubric_id),
        reverse=True,
    )


def _generic_template(key: tuple[str, str]) -> dict[str, Any]:
    module, failure_type = key
    return {
        "trigger": f"AgentDebug WebShop labels mark {module}.{failure_type} as a critical failure.",
        "avoid_actions": [f"repeat behavior associated with {module}.{failure_type}"],
        "lesson": f"This pattern was annotated as a critical {module} failure.",
        "suggested_strategy": "Choose the action that gathers new evidence, preserves task constraints, or completes a verified purchase.",
    }


def _severity_for_group(key: tuple[str, str], support: int) -> float:
    if key == ("system", "environment_error"):
        return min(0.7, 0.4 + 0.03 * support)
    return min(1.0, 0.6 + 0.04 * support)


def _evidence_line(row: dict[str, Any]) -> str:
    reasoning = row["reasoning"] or "(no reasoning text in label)"
    if len(reasoning) > 300:
        reasoning = reasoning[:300] + "..."
    return (
        f"{row['trajectory_id']} step={row['critical_failure_step']} "
        f"type={row['failure_key']}: {reasoning}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
