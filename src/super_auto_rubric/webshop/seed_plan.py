from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class ChatCompletion:
    content: str
    raw: dict[str, Any]


@dataclass
class SeedPlanChatClient:
    """Small OpenAI-compatible chat client for Ark Plan v3."""

    api_key: str
    base_url: str = "https://ark.cn-beijing.volces.com/api/plan/v3"
    timeout_seconds: int = 120

    @classmethod
    def from_env(
        cls,
        *,
        api_key_env: str = "SEED_AGENT_PLAN_API_KEY",
        base_url: str = "https://ark.cn-beijing.volces.com/api/plan/v3",
        timeout_seconds: int = 120,
    ) -> "SeedPlanChatClient":
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing required API key environment variable: {api_key_env}")
        return cls(api_key=api_key, base_url=base_url, timeout_seconds=timeout_seconds)

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 512,
        response_format: dict[str, Any] | None = None,
    ) -> ChatCompletion:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        response = self._post(payload)
        if response_format is not None and _is_unsupported_response_format(response):
            payload.pop("response_format", None)
            response = self._post(payload)
        response.raise_for_status()
        body = response.json()
        content = str(body["choices"][0]["message"].get("content", ""))
        return ChatCompletion(content=content, raw=body)

    def _post(self, payload: dict[str, Any]) -> requests.Response:
        return requests.post(
            self.base_url.rstrip("/") + "/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout_seconds,
        )


def _is_unsupported_response_format(response: requests.Response) -> bool:
    if response.status_code != 400:
        return False
    try:
        body = response.json()
    except ValueError:
        return False
    message = str(body.get("error", {}).get("message", ""))
    return "response_format" in message and "not supported" in message.lower()
