import asyncio
import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

from open_instruct.search_rewards.utils.run_utils import extract_json_from_response


DEFAULT_OPENAI_BASE_URL = "https://ark.cn-beijing.volces.com/api/plan/v3"
DEFAULT_CRITIC_MODEL = "kimi-k2.6"

ACTION_RE = re.compile(r"\b(search|click)\[([^\]\n]{1,300})\]", re.IGNORECASE)
MALFORMED_ACTION_RE = re.compile(r"\b(search|click)\s*[\(\{<]", re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
UNMATCHED_THINK_RE = re.compile(r"<think\b[^>]*>.*", re.IGNORECASE | re.DOTALL)

_JUDGE_CACHE: Dict[str, Dict[str, Any]] = {}
_ENV_CACHE: Dict[str, int] = {}
_JUDGE_MAX_WORKERS = int(os.environ.get("WEBSHOP_CRITIC_JUDGE_MAX_WORKERS", "32"))
_JUDGE_MAX_CONCURRENCY = int(os.environ.get("WEBSHOP_CRITIC_JUDGE_MAX_CONCURRENCY", str(_JUDGE_MAX_WORKERS)))
_JUDGE_EXECUTOR = ThreadPoolExecutor(max_workers=max(1, _JUDGE_MAX_WORKERS))
_JUDGE_SEMAPHORE = threading.BoundedSemaphore(max(1, _JUDGE_MAX_CONCURRENCY))


def _get_replay_env_idx(base_url: str, timeout: float) -> int:
    base = base_url.rstrip("/")
    env_idx = _ENV_CACHE.get(base)
    if env_idx is not None:
        return env_idx
    create_resp = requests.post(f"{base}/create", timeout=timeout)
    create_resp.raise_for_status()
    env_idx = int(create_resp.json())
    _ENV_CACHE[base] = env_idx
    return env_idx


@dataclass
class WebShopCriticErrorConfig:
    webshop_base_url: str = "http://127.0.0.1:36001"
    webshop_replay_enabled: bool = True
    webshop_replay_timeout: float = 15.0
    webshop_max_replay_steps: int = 8
    critic_error_enabled: bool = False
    critic_error_pool_path: Optional[str] = None
    critic_error_judge_model: str = DEFAULT_CRITIC_MODEL
    critic_error_openai_base_url: str = DEFAULT_OPENAI_BASE_URL
    critic_error_api_key_env: str = "SEED_AGENT_PLAN_API_KEY"
    critic_error_max_active_weaknesses: int = 8
    critic_error_trigger_threshold: float = 0.55
    # Keep these legacy weights only for diagnostic proxy logging. The scalar
    # reward follows WebShop/AgentGym: environment reward is emitted only after
    # a valid purchase trajectory reaches click[Buy Now].
    webshop_task_reward_weight: float = 0.45
    webshop_format_reward_weight: float = 0.15
    webshop_action_reward_weight: float = 0.15
    webshop_attribute_reward_weight: float = 0.10
    critic_error_reward_weight: float = 0.35


def _unwrap_ground_truth(ground_truth: Any) -> Dict[str, Any]:
    if isinstance(ground_truth, list) and ground_truth:
        ground_truth = ground_truth[0]
    if isinstance(ground_truth, str):
        return json.loads(ground_truth)
    if isinstance(ground_truth, dict):
        return ground_truth
    raise TypeError(f"Unsupported ground truth type: {type(ground_truth).__name__}")


def _wrap_like(original: Any, value: Dict[str, Any]) -> Any:
    encoded = json.dumps(value, ensure_ascii=False)
    if isinstance(original, list):
        return [encoded]
    return encoded


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _strip_thinking_text(response: str) -> str:
    """Remove reasoning text so rewards only see executable/final content."""
    text = THINK_BLOCK_RE.sub(" ", response or "")
    return UNMATCHED_THINK_RE.sub(" ", text)


def _action_region(response: str) -> str:
    # WebShop actions are executable even when small reasoning models wrap the
    # whole turn in <think>. Do not strip thinking text before action parsing.
    return ANSWER_RE.sub(" ", response or "")


def _weakness_id(error_type: str, title: str, rubric: str) -> str:
    raw = f"{error_type}|{_normalize_text(title)}|{_normalize_text(rubric)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _pool_paths(pool_path: str) -> Tuple[Path, Path]:
    path = Path(pool_path)
    return path, path.with_suffix(path.suffix + ".retired")


def load_weakness_pool(pool_path: Optional[str]) -> List[Dict[str, Any]]:
    if not pool_path:
        return []
    path = Path(pool_path)
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_weakness_pool(pool_path: str, rows: Iterable[Dict[str, Any]]) -> None:
    path, _ = _pool_paths(pool_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp, path)


def append_retired_weaknesses(pool_path: str, rows: Iterable[Dict[str, Any]]) -> None:
    _, retired_path = _pool_paths(pool_path)
    retired_path.parent.mkdir(parents=True, exist_ok=True)
    with retired_path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def active_weaknesses(pool_path: Optional[str], limit: int) -> List[Dict[str, Any]]:
    rows = [row for row in load_weakness_pool(pool_path) if row.get("active", True)]
    rows.sort(key=lambda row: (float(row.get("severity", 0.0)), int(row.get("support_count", 0))), reverse=True)
    return rows[: max(0, limit)]


def extract_webshop_actions(response: str) -> List[str]:
    actions = []
    for match in ACTION_RE.finditer(_action_region(response)):
        fn = match.group(1).lower()
        arg = " ".join(match.group(2).strip().split())
        actions.append(f"{fn}[{arg}]")
    return actions


def extract_answer(response: str) -> Optional[str]:
    matches = list(ANSWER_RE.finditer(_strip_thinking_text(response)))
    if not matches:
        return None
    answer = matches[-1].group(1).strip()
    return answer or None


def attribute_match_reward(response: str, ground_truth: Dict[str, Any]) -> float:
    attrs = ground_truth.get("target_attributes") or []
    if not attrs:
        return 0.0
    haystack = _normalize_text(_strip_thinking_text(response))
    hits = 0
    for attr in attrs:
        attr_text = _normalize_text(str(attr))
        if attr_text and attr_text in haystack:
            hits += 1
    return hits / max(1, len(attrs))


def action_format_reward(response: str, actions: List[str]) -> float:
    if not actions:
        return 0.0
    malformed = len(MALFORMED_ACTION_RE.findall(_action_region(response)))
    has_search = any(action.startswith("search[") for action in actions)
    if malformed:
        return 0.25 if has_search else 0.0
    if has_search:
        return 1.0
    return 0.5


def replay_webshop_actions(
    actions: List[str],
    ground_truth: Dict[str, Any],
    base_url: str,
    timeout: float,
    max_steps: int,
) -> Dict[str, Any]:
    result = {
        "reward": 0.0,
        "done": False,
        "num_steps": 0,
        "error": None,
        "last_observation": "",
    }
    if not actions:
        result["error"] = "no_actions"
        return result

    session_id = ground_truth.get("webshop_session_id")
    try:
        base = base_url.rstrip("/")
        env_idx = _get_replay_env_idx(base, timeout)
        if session_id is not None:
            reset_resp = requests.post(
                f"{base}/reset",
                json={"env_idx": env_idx, "session_id": int(session_id)},
                timeout=timeout,
            )
            if not reset_resp.ok:
                _ENV_CACHE.pop(base, None)
                env_idx = _get_replay_env_idx(base, timeout)
                reset_resp = requests.post(
                    f"{base}/reset",
                    json={"env_idx": env_idx, "session_id": int(session_id)},
                    timeout=timeout,
                )
            reset_resp.raise_for_status()

        for action in actions[:max_steps]:
            step_resp = requests.post(
                f"{base}/step",
                json={"env_idx": env_idx, "action": action},
                timeout=timeout,
            )
            step_resp.raise_for_status()
            payload = step_resp.json()
            result["reward"] = float(payload.get("reward", 0.0) or 0.0)
            result["done"] = bool(payload.get("done", False))
            result["num_steps"] += 1
            result["last_observation"] = str(payload.get("state", ""))[:2000]
            if result["done"]:
                break
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    return result


def _openai_chat_json(
    *,
    model: str,
    base_url: str,
    api_key_env: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1200,
) -> Dict[str, Any]:
    api_key = os.environ.get(api_key_env) or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(f"Missing API key env var: {api_key_env}")

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content or ""
    obj = extract_json_from_response(content)
    if not isinstance(obj, dict):
        raise RuntimeError(f"Judge did not return JSON: {content[:500]}")
    return obj


def judge_critic_rubric(
    response: str,
    ground_truth: Dict[str, Any],
    rubric: Dict[str, Any],
    config: WebShopCriticErrorConfig,
) -> Dict[str, Any]:
    rubric_id = str(rubric.get("weakness_id") or rubric.get("id") or _weakness_id(
        str(rubric.get("error_type", "error")),
        str(rubric.get("title", "critic_error")),
        str(rubric.get("rubric") or rubric.get("description", "")),
    ))
    cache_key = hashlib.sha1(
        json.dumps(
            {
                "response": response,
                "query": ground_truth.get("query"),
                "rubric_id": rubric_id,
                "rubric": rubric.get("rubric") or rubric.get("description"),
                "model": config.critic_error_judge_model,
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    if cache_key in _JUDGE_CACHE:
        return _JUDGE_CACHE[cache_key]

    system_prompt = (
        "You are a strict WebShop RL critic. Decide whether the model trajectory "
        "exhibits the specified critical error. Output only JSON."
    )
    user_prompt = f"""
Instruction:
{ground_truth.get("query") or ground_truth.get("instruction_text") or ""}

Model trajectory:
{response}

Critical error criterion:
{rubric.get("rubric") or rubric.get("description") or ""}

Return JSON exactly with:
{{
  "triggered": true or false,
  "score": a number from 0.0 to 1.0 where 1.0 means the critical error is clearly present,
  "evidence": "short quote or explanation"
}}
""".strip()
    try:
        obj = _openai_chat_json(
            model=config.critic_error_judge_model,
            base_url=config.critic_error_openai_base_url,
            api_key_env=config.critic_error_api_key_env,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=600,
        )
        score = max(0.0, min(1.0, float(obj.get("score", 0.0))))
        judged = {
            "weakness_id": rubric_id,
            "triggered": bool(obj.get("triggered", score >= config.critic_error_trigger_threshold)),
            "score": score,
            "evidence": str(obj.get("evidence", ""))[:500],
        }
    except Exception as exc:
        judged = {
            "weakness_id": rubric_id,
            "triggered": False,
            "score": 0.0,
            "evidence": f"judge_error:{type(exc).__name__}:{str(exc)[:200]}",
        }
    _JUDGE_CACHE[cache_key] = judged
    return judged


def _judge_critic_rubric_limited(
    response: str,
    ground_truth: Dict[str, Any],
    rubric: Dict[str, Any],
    config: WebShopCriticErrorConfig,
) -> Dict[str, Any]:
    with _JUDGE_SEMAPHORE:
        return judge_critic_rubric(response, ground_truth, rubric, config)


def _critic_rubrics_from_ground_truth(
    ground_truth: Dict[str, Any],
    config: WebShopCriticErrorConfig,
) -> List[Dict[str, Any]]:
    rubrics = list(ground_truth.get("critic_rubrics") or [])
    if ground_truth.get("disable_critic_error_pool"):
        return rubrics[: config.critic_error_max_active_weaknesses]
    if config.critic_error_enabled and config.critic_error_pool_path:
        existing = {str(r.get("weakness_id") or r.get("id")) for r in rubrics}
        for weakness in active_weaknesses(config.critic_error_pool_path, config.critic_error_max_active_weaknesses):
            weakness_id = str(weakness.get("weakness_id"))
            if weakness_id in existing:
                continue
            rubrics.append(
                {
                    "weakness_id": weakness_id,
                    "title": weakness.get("title", "Critic Error"),
                    "description": weakness.get("description", ""),
                    "rubric": weakness.get("rubric") or weakness.get("description", ""),
                    "error_type": weakness.get("error_type", "error"),
                    "severity": float(weakness.get("severity", 0.7)),
                    "weight": -abs(float(weakness.get("severity", 0.7))),
                }
            )
    return rubrics[: config.critic_error_max_active_weaknesses]


def compute_webshop_critic_error_reward(
    prediction: str,
    ground_truth: Dict[str, Any],
    config: WebShopCriticErrorConfig,
) -> Dict[str, Any]:
    actions = extract_webshop_actions(prediction)
    answer = extract_answer(prediction)
    format_score = 1.0 if answer else 0.0
    action_score = action_format_reward(prediction, actions)
    attr_score = attribute_match_reward(prediction, ground_truth)

    replay = {"reward": 0.0, "done": False, "num_steps": 0, "error": "disabled", "last_observation": ""}
    if config.webshop_replay_enabled:
        replay = replay_webshop_actions(
            actions=actions,
            ground_truth=ground_truth,
            base_url=config.webshop_base_url,
            timeout=config.webshop_replay_timeout,
            max_steps=config.webshop_max_replay_steps,
        )
    task_reward = max(0.0, min(1.0, float(replay.get("reward", 0.0))))

    critic_rubrics = _critic_rubrics_from_ground_truth(ground_truth, config)
    critic_scores_by_id: Dict[str, float] = {}
    critic_evidence_by_id: Dict[str, str] = {}
    critic_penalty = 0.0
    critic_weight_total = 0.0
    if len(critic_rubrics) > 1:
        futures = [
            _JUDGE_EXECUTOR.submit(_judge_critic_rubric_limited, prediction, ground_truth, rubric, config)
            for rubric in critic_rubrics
        ]
        judged_rubrics = [future.result() for future in futures]
    else:
        judged_rubrics = [
            _judge_critic_rubric_limited(prediction, ground_truth, rubric, config)
            for rubric in critic_rubrics
        ]
    for rubric, judged in zip(critic_rubrics, judged_rubrics):
        severity = abs(float(rubric.get("severity", abs(float(rubric.get("weight", -0.7)))) or 0.7))
        weakness_id = judged["weakness_id"]
        score = float(judged.get("score", 0.0))
        critic_scores_by_id[weakness_id] = score
        critic_evidence_by_id[weakness_id] = str(judged.get("evidence", ""))
        critic_penalty += severity * score
        critic_weight_total += severity
    critic_error_reward = -(critic_penalty / max(critic_weight_total, 1.0))

    proxy_reward = (
        config.webshop_task_reward_weight * task_reward
        + config.webshop_format_reward_weight * format_score
        + config.webshop_action_reward_weight * action_score
        + config.webshop_attribute_reward_weight * attr_score
    )
    reward = (
        task_reward
        + config.critic_error_reward_weight * critic_error_reward
    )

    log_values = {
        "webshop_task_reward": task_reward,
        "webshop_env_reward": task_reward,
        "webshop_proxy_reward": proxy_reward,
        "webshop_format_reward": format_score,
        "webshop_action_reward": action_score,
        "webshop_attribute_reward": attr_score,
        "webshop_replay_done": 1.0 if replay.get("done") else 0.0,
        "webshop_replay_steps": float(replay.get("num_steps", 0)),
        "critic_error_reward": critic_error_reward,
        "critic_error_num_active": float(len(critic_rubrics)),
        "critic_error_scores_by_id": critic_scores_by_id,
        "critic_error_evidence_by_id": critic_evidence_by_id,
    }
    if replay.get("error"):
        log_values["webshop_replay_error"] = str(replay["error"])

    return {
        "reward": reward,
        "log_values": log_values,
        "actions": actions,
        "answer": answer,
        "replay": replay,
    }


def inject_active_critic_rubrics_into_ground_truths(
    ground_truths: List[Any],
    pool_path: Optional[str],
    max_active: int,
) -> List[Any]:
    active = active_weaknesses(pool_path, max_active)
    if not active:
        return ground_truths

    injected: List[Any] = []
    for original in ground_truths:
        gt = _unwrap_ground_truth(original)
        existing_ids = {str(r.get("weakness_id") or r.get("id")) for r in gt.get("critic_rubrics", [])}
        critic_rubrics = list(gt.get("critic_rubrics") or [])
        longform_rubrics = list(gt.get("rubrics") or [])
        rubrics_types = list(gt.get("rubrics_types") or ["persistent"] * len(longform_rubrics))
        for weakness in active:
            weakness_id = str(weakness.get("weakness_id"))
            if weakness_id in existing_ids:
                continue
            rubric = {
                "weakness_id": weakness_id,
                "title": weakness.get("title", "Critic Error"),
                "description": weakness.get("description", ""),
                "rubric": weakness.get("rubric") or weakness.get("description", ""),
                "error_type": weakness.get("error_type", "error"),
                "severity": float(weakness.get("severity", 0.7)),
                "weight": -abs(float(weakness.get("severity", 0.7))),
            }
            critic_rubrics.append(rubric)
            longform_rubrics.append(
                {
                    "title": f"Critic Error: {rubric['title']}",
                    "description": rubric["rubric"],
                    "weight": rubric["weight"],
                    "source": "critic_error_pool",
                    "weakness_id": weakness_id,
                }
            )
            rubrics_types.append("critic_error")
        gt["critic_rubrics"] = critic_rubrics
        gt["rubrics"] = longform_rubrics
        gt["rubrics_types"] = rubrics_types
        injected.append(_wrap_like(original, gt))
    return injected


def disable_critic_error_pool_for_ground_truths(ground_truths: List[Any]) -> List[Any]:
    disabled: List[Any] = []
    for original in ground_truths:
        gt = _unwrap_ground_truth(original)
        gt["disable_critic_error_pool"] = True
        gt["critic_rubrics"] = []
        disabled.append(_wrap_like(original, gt))
    return disabled


def update_weakness_pool(
    *,
    pool_path: str,
    weaknesses: List[Dict[str, Any]],
    training_step: int,
    model_name: str,
    max_active: int,
) -> Dict[str, int]:
    rows = {str(row.get("weakness_id")): row for row in load_weakness_pool(pool_path)}
    added = 0
    updated = 0
    for item in weaknesses:
        rubric = str(item.get("rubric") or item.get("description") or "").strip()
        title = str(item.get("title") or item.get("weakness") or "Critic Error").strip()
        error_type = str(item.get("error_type") or "error").strip().lower()
        if not rubric:
            continue
        weakness_id = str(item.get("weakness_id") or _weakness_id(error_type, title, rubric))
        severity = max(0.0, min(1.0, float(item.get("severity", 0.7) or 0.7)))
        if weakness_id in rows:
            row = rows[weakness_id]
            row["support_count"] = int(row.get("support_count", 0)) + 1
            row["last_triggered_step"] = training_step
            row["absent_count"] = 0
            row["active"] = True
            row["severity"] = max(float(row.get("severity", 0.0)), severity)
            evidence = str(item.get("evidence", ""))[:500]
            if evidence:
                row.setdefault("examples", []).append({"step": training_step, "evidence": evidence})
                row["examples"] = row["examples"][-5:]
            updated += 1
        else:
            rows[weakness_id] = {
                "weakness_id": weakness_id,
                "title": title,
                "description": str(item.get("description") or rubric),
                "rubric": rubric,
                "error_type": error_type if error_type in {"error", "hacking", "unsafe"} else "error",
                "severity": severity,
                "support_count": 1,
                "first_seen_step": training_step,
                "last_triggered_step": training_step,
                "absent_count": 0,
                "active": True,
                "source_model": model_name,
                "examples": [{"step": training_step, "evidence": str(item.get("evidence", ""))[:500]}],
            }
            added += 1

    active_rows = [row for row in rows.values() if row.get("active", True)]
    active_rows.sort(key=lambda row: (float(row.get("severity", 0.0)), int(row.get("support_count", 0))), reverse=True)
    save_weakness_pool(pool_path, active_rows[:max_active])
    return {"added": added, "updated": updated, "active": min(len(active_rows), max_active)}


async def mine_and_update_critic_error_pool(
    *,
    decoded_responses: List[str],
    ground_truths: List[Any],
    training_step: int,
    pool_path: Optional[str],
    model_name: str,
    base_url: str,
    api_key_env: str,
    max_active: int,
    max_examples: int = 12,
) -> Dict[str, int]:
    if not pool_path:
        return {"added": 0, "updated": 0, "active": 0}

    examples = []
    for idx, (response, ground_truth) in enumerate(zip(decoded_responses[:max_examples], ground_truths[:max_examples])):
        try:
            gt = _unwrap_ground_truth(ground_truth)
        except Exception:
            continue
        examples.append(
            {
                "idx": idx,
                "instruction": gt.get("query") or gt.get("instruction_text") or "",
                "session_id": gt.get("webshop_session_id"),
                "response": response[:4000],
            }
        )
    if not examples:
        return {"added": 0, "updated": 0, "active": len(active_weaknesses(pool_path, max_active))}

    system_prompt = (
        "You are a WebShop RL supervisor. Mine critical model weaknesses from trajectories. "
        "Focus on critical errors, reward hacking, unsafe/hacking behavior, invalid tool use, "
        "premature purchase, query drift, and ignoring required product attributes. Output only JSON."
    )
    user_prompt = f"""
Analyze these Qwen3-0.6B WebShop rollouts from training step {training_step}.

Return JSON:
{{
  "weaknesses": [
    {{
      "title": "short general weakness name",
      "description": "what the model is doing wrong",
      "error_type": "error|hacking|unsafe",
      "rubric": "A judge should assign score 1 if the trajectory exhibits this critical error, otherwise 0. Be concrete and observable.",
      "severity": 0.0 to 1.0,
      "evidence": "brief evidence from a rollout"
    }}
  ]
}}

Rules:
- Only include weaknesses visible in the rollouts.
- Prefer reusable rubrics that can supervise future batches.
- Do not include generic product-quality advice unless it is tied to an actual error.
- Include hacking/unsafe behavior if the model tries to exploit reward, fake observations, skip required actions, or claim success without evidence.

Rollouts:
{json.dumps(examples, ensure_ascii=False, indent=2)}
""".strip()

    def call() -> Dict[str, Any]:
        return _openai_chat_json(
            model=model_name,
            base_url=base_url,
            api_key_env=api_key_env,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=1800,
        )

    try:
        obj = await asyncio.to_thread(call)
        weaknesses = obj.get("weaknesses", [])
        if not isinstance(weaknesses, list):
            weaknesses = []
    except Exception as exc:
        print(f"[CriticError] weakness mining failed at step {training_step}: {type(exc).__name__}: {exc}")
        weaknesses = []

    stats = update_weakness_pool(
        pool_path=pool_path,
        weaknesses=weaknesses,
        training_step=training_step,
        model_name=model_name,
        max_active=max_active,
    )
    print(f"[CriticError] pool update at step {training_step}: {stats}")
    return stats


def update_critic_error_pool_from_logs(
    *,
    pool_path: Optional[str],
    log_values: Dict[str, Any],
    training_step: int,
    trigger_threshold: float,
    retire_after_absent_batches: int,
) -> Dict[str, int]:
    if not pool_path:
        return {"retired": 0, "active": 0}
    rows = load_weakness_pool(pool_path)
    if not rows:
        return {"retired": 0, "active": 0}
    batch_scores: Dict[str, float] = {}
    for item in log_values.get("critic_error_scores_by_id", []) or []:
        if not isinstance(item, dict):
            continue
        for weakness_id, score in item.items():
            batch_scores[str(weakness_id)] = max(batch_scores.get(str(weakness_id), 0.0), float(score or 0.0))

    kept = []
    retired = []
    for row in rows:
        weakness_id = str(row.get("weakness_id"))
        score = batch_scores.get(weakness_id, 0.0)
        if score >= trigger_threshold:
            row["last_triggered_step"] = training_step
            row["absent_count"] = 0
            row["trigger_count"] = int(row.get("trigger_count", 0)) + 1
            kept.append(row)
        else:
            row["absent_count"] = int(row.get("absent_count", 0)) + 1
            if row["absent_count"] >= retire_after_absent_batches:
                row["active"] = False
                row["retired_step"] = training_step
                retired.append(row)
            else:
                kept.append(row)

    save_weakness_pool(pool_path, kept)
    if retired:
        append_retired_weaknesses(pool_path, retired)
        print(f"[CriticError] retired {len(retired)} weaknesses at step {training_step}")
    return {"retired": len(retired), "active": len(kept)}
