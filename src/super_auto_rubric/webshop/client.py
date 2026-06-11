from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import requests


class WebShopClientProtocol(Protocol):
    """Minimum environment interface needed by the local benchmark runner."""

    env_idx: int | None

    def create(self) -> int:
        ...

    def reset(self, session_id: int | None = None) -> str:
        ...

    def observation(self) -> str:
        ...

    def available_actions(self) -> dict[str, Any]:
        ...

    def instruction_text(self) -> str:
        ...

    def state(self) -> dict[str, Any]:
        ...

    def step(self, action: str) -> dict[str, Any]:
        ...


@dataclass
class AgentGymWebShopClient:
    """Thin HTTP client for AgentGym's WebShop FastAPI server."""

    base_url: str = "http://127.0.0.1:36001"
    timeout_seconds: int = 300
    env_idx: int | None = None

    def _url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"

    def health(self) -> bool:
        try:
            response = requests.get(self._url("/"), timeout=5)
        except requests.RequestException:
            return False
        return response.status_code == 200 and response.json() == "ok"

    def create(self) -> int:
        response = requests.post(self._url("/create"), timeout=self.timeout_seconds)
        response.raise_for_status()
        self.env_idx = int(response.json())
        return self.env_idx

    def _require_env(self) -> int:
        if self.env_idx is None:
            return self.create()
        return self.env_idx

    def reset(self, session_id: int | None = None) -> str:
        env_idx = self._require_env()
        payload = {"env_idx": env_idx, "session_id": session_id}
        response = requests.post(self._url("/reset"), json=payload, timeout=self.timeout_seconds)
        response.raise_for_status()
        body = response.json()
        if isinstance(body, list) and body:
            return str(body[0])
        if isinstance(body, tuple) and body:
            return str(body[0])
        return self.observation()

    def observation(self) -> str:
        env_idx = self._require_env()
        response = requests.get(
            self._url("/observation"),
            params={"env_idx": env_idx},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return str(response.json())

    def available_actions(self) -> dict[str, Any]:
        env_idx = self._require_env()
        response = requests.get(
            self._url("/available_actions"),
            params={"env_idx": env_idx},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return dict(response.json())

    def instruction_text(self) -> str:
        env_idx = self._require_env()
        response = requests.get(
            self._url("/instruction_text"),
            params={"env_idx": env_idx},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return str(response.json())

    def state(self) -> dict[str, Any]:
        env_idx = self._require_env()
        response = requests.get(
            self._url("/state"),
            params={"env_idx": env_idx},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        body = dict(response.json())
        if (
            body.get("url") == "url"
            and body.get("html") == "html"
            and body.get("instruction_text") == "instruction_text"
        ):
            body = {
                "url": f"{self.base_url.rstrip('/')}/env/{env_idx}",
                "html": self.observation(),
                "instruction_text": self.instruction_text(),
                "source": "agentgym-state-fallback",
            }
        return body

    def step(self, action: str) -> dict[str, Any]:
        env_idx = self._require_env()
        payload = {"env_idx": env_idx, "action": action}
        response = requests.post(self._url("/step"), json=payload, timeout=self.timeout_seconds)
        response.raise_for_status()
        body = dict(response.json())
        body.setdefault("info", None)
        return body


class SyntheticWebShopClient:
    """Small deterministic client for tests and offline wiring checks."""

    env_idx: int | None

    def __init__(self) -> None:
        self.env_idx = None
        self.step_index = 0
        self.instruction = (
            "Find a medium green machine washable jumpsuit under 60 dollars."
        )
        self.observations = [
            "WebShop [SEP] Instruction: Find a medium green machine washable jumpsuit under 60 dollars [SEP] Search",
            "Results [SEP] Blue jumpsuit [SEP] Green jumpsuit [SEP] next",
            "Item page [SEP] Green jumpsuit [SEP] color green [SEP] size small [SEP] buy now",
            "Done [SEP] purchased green jumpsuit size small",
        ]

    def create(self) -> int:
        self.env_idx = 1
        return self.env_idx

    def reset(self, session_id: int | None = None) -> str:
        self.create()
        self.step_index = 0
        return self.observation()

    def observation(self) -> str:
        return self.observations[min(self.step_index, len(self.observations) - 1)]

    def available_actions(self) -> dict[str, Any]:
        if self.step_index == 0:
            return {"has_search_bar": True, "clickables": ["search"]}
        if self.step_index == 1:
            return {"has_search_bar": False, "clickables": ["blue jumpsuit", "green jumpsuit", "next"]}
        if self.step_index == 2:
            return {"has_search_bar": False, "clickables": ["small", "medium", "buy now"]}
        return {"has_search_bar": False, "clickables": []}

    def instruction_text(self) -> str:
        return self.instruction

    def state(self) -> dict[str, Any]:
        return {
            "url": f"synthetic://webshop/{self.step_index}",
            "html": self.observation(),
            "instruction_text": self.instruction,
        }

    def step(self, action: str) -> dict[str, Any]:
        self.step_index += 1
        done = self.step_index >= len(self.observations) - 1 or action == "click[buy now]"
        reward = 0.25 if done else 0.0
        return {
            "state": self.observation(),
            "reward": reward,
            "done": done,
            "info": {
                "synthetic": True,
                "score": reward,
                "action": action,
            },
        }
