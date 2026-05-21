from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class ChatbotCall:
    item_id: str
    query: str
    system_prompt: str
    payload: dict[str, object]


def build_chatbot_payload(*, item: dict[str, object], system_prompt: str) -> dict[str, object]:
    item_id = str(item["item_id"])
    query = str(item["query_text"])
    payload = {
        "item_id": item_id,
        "query": query,
        "system_prompt": system_prompt,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        "metadata": {
            "signature": str(item.get("signature", "")),
            "target_facet": str(item.get("target_facet", item.get("facet", ""))),
            "target_facets": list(item.get("target_facets", [])),
        },
    }
    return payload


def invoke_http_chatbot(
    *,
    endpoint: str,
    item: dict[str, object],
    system_prompt: str,
    response_json_key: str,
    post: Callable[..., Any] | None = None,
    timeout: int = 60,
    headers: dict[str, str] | None = None,
) -> str:
    if not endpoint.strip():
        raise ValueError("HTTP chatbot endpoint must be non-empty")
    if not response_json_key.strip():
        raise ValueError("response_json_key must be non-empty")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if post is None:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError("requests is required for HTTP chatbot adapters") from exc
        post = requests.post

    response = post(
        url=endpoint,
        json=build_chatbot_payload(item=item, system_prompt=system_prompt),
        headers=headers or {},
        timeout=timeout,
    )
    status_code = int(response.status_code)
    if status_code < 200 or status_code >= 300:
        raise RuntimeError(
            f"HTTP chatbot endpoint returned status {status_code} for item {item['item_id']}: {response.text}"
        )
    try:
        body = response.json()
    except ValueError as exc:
        raise RuntimeError(f"HTTP chatbot endpoint returned non-JSON body for item {item['item_id']}") from exc
    if not isinstance(body, dict):
        raise RuntimeError(f"HTTP chatbot endpoint JSON body must be an object for item {item['item_id']}")
    if response_json_key not in body:
        raise KeyError(
            f"HTTP chatbot response for item {item['item_id']} is missing required key: {response_json_key}"
        )
    return _require_non_empty_response_text(body[response_json_key], context=f"HTTP item {item['item_id']}")


def invoke_command_chatbot(
    *,
    command: list[str],
    item: dict[str, object],
    system_prompt: str,
    output_mode: str,
    response_json_key: str,
    timeout: int = 60,
) -> str:
    if not command:
        raise ValueError("command adapter requires at least one argv element")
    if output_mode not in {"json", "text"}:
        raise ValueError("output_mode must be either 'json' or 'text'")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    payload = build_chatbot_payload(item=item, system_prompt=system_prompt)
    completed = subprocess.run(
        command,
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command chatbot exited with code {completed.returncode} for item {item['item_id']}: "
            f"stderr={completed.stderr.strip()}"
        )
    stdout = completed.stdout.strip()
    if output_mode == "text":
        return _require_non_empty_response_text(stdout, context=f"command item {item['item_id']}")
    try:
        body = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Command chatbot stdout was not JSON for item {item['item_id']}: {stdout}") from exc
    if not isinstance(body, dict):
        raise RuntimeError(f"Command chatbot JSON stdout must be an object for item {item['item_id']}")
    if response_json_key not in body:
        raise KeyError(
            f"Command chatbot response for item {item['item_id']} is missing required key: {response_json_key}"
        )
    return _require_non_empty_response_text(body[response_json_key], context=f"command item {item['item_id']}")


def _require_non_empty_response_text(value: object, *, context: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{context} response text must be a string")
    response_text = value.strip()
    if not response_text:
        raise ValueError(f"{context} response text must be non-empty")
    return response_text
