from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "can",
    "dollars",
    "find",
    "for",
    "i",
    "in",
    "is",
    "it",
    "looking",
    "me",
    "please",
    "price",
    "should",
    "than",
    "the",
    "to",
    "under",
    "want",
    "with",
}


def action_from_clickable(clickable: str) -> str:
    return f"click[{clickable}]"


@dataclass
class ScriptedWebShopPolicy:
    """A deterministic policy for smoke tests and baseline trace collection."""

    max_query_terms: int = 8
    seen_actions: set[str] = field(default_factory=set)

    def reset(self) -> None:
        self.seen_actions.clear()

    def _query_from_instruction(self, instruction_text: str) -> str:
        tokens = re.findall(r"[a-zA-Z0-9.]+", instruction_text.lower())
        selected = [token for token in tokens if token not in STOPWORDS]
        return " ".join(selected[: self.max_query_terms]) or instruction_text[:80]

    def choose_action(
        self,
        observation: str,
        available_actions: dict[str, Any],
        instruction_text: str,
    ) -> str:
        if available_actions.get("has_search_bar"):
            return f"search[{self._query_from_instruction(instruction_text)}]"

        clickables = [str(item) for item in available_actions.get("clickables", [])]
        if not clickables:
            return "click[back to search]"

        normalized_instruction = instruction_text.lower()
        preferred_terms = [
            "medium",
            "large",
            "small",
            "green",
            "blue",
            "black",
            "white",
            "red",
            "buy now",
        ]
        for term in preferred_terms:
            if term in normalized_instruction or term == "buy now":
                for clickable in clickables:
                    candidate = action_from_clickable(clickable)
                    if term in clickable.lower() and candidate not in self.seen_actions:
                        self.seen_actions.add(candidate)
                        return candidate

        for clickable in clickables:
            candidate = action_from_clickable(clickable)
            if candidate not in self.seen_actions:
                self.seen_actions.add(candidate)
                return candidate

        return action_from_clickable(clickables[0])
