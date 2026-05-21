from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from copal.io import ensure_directory
from copal.llm import LLMClient, LLMJsonError, LLMMessage, LLMProviderError, parse_strict_json_payload


class LiveSchemaError(ValueError):
    """Raised when a live model response parses as JSON but violates the requested schema."""


def complete_live_json_object(
    *,
    client: LLMClient,
    model: str,
    messages: list[LLMMessage],
    stage_dir: Path,
    stage_name: str,
    target_id: str,
    required_fields: tuple[str, ...] = (),
    validator: Callable[[dict[str, Any]], None] | None = None,
    max_attempts: int | None = None,
) -> dict[str, Any]:
    if max_attempts is None:
        max_attempts = int(os.getenv("COPAL_LIVE_JSON_MAX_ATTEMPTS", "3"))
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")

    last_error: LLMProviderError | LLMJsonError | LiveSchemaError | None = None
    retry_nonce_base = _count_existing_live_errors(
        stage_dir=stage_dir,
        stage_name=stage_name,
        target_id=target_id,
        model=model,
    )
    for attempt_index in range(max_attempts):
        response_text = ""
        try:
            response = client.complete(
                model=model,
                messages=_messages_for_attempt(
                    messages=messages,
                    stage_name=stage_name,
                    target_id=target_id,
                    attempt_index=attempt_index,
                    max_attempts=max_attempts,
                    previous_error=last_error,
                    retry_nonce=retry_nonce_base + attempt_index,
                ),
            )
            response_text = response.text
            payload = parse_strict_json_payload(response.text)
            if not isinstance(payload, dict):
                raise LiveSchemaError(f"{stage_name} {target_id} expected JSON object")
            require_fields(payload, required_fields, context=f"{stage_name} {target_id}")
            if validator is not None:
                validator(payload)
            return payload
        except (LLMProviderError, LLMJsonError, LiveSchemaError) as exc:
            last_error = exc
            record_live_error(
                stage_dir=stage_dir,
                stage_name=stage_name,
                target_id=target_id,
                model=model,
                error=exc,
                raw_response=response_text or getattr(exc, "response_text", ""),
                attempt=attempt_index + 1,
                max_attempts=max_attempts,
            )
            if isinstance(exc, LLMProviderError) and _is_provider_safety_block(exc):
                raise
            if attempt_index + 1 == max_attempts:
                raise

    raise RuntimeError("unreachable live JSON retry state")


def _is_provider_safety_block(exc: LLMProviderError) -> bool:
    if getattr(exc, "status_code", None) != 400:
        return False
    message = str(exc).lower()
    return (
        "content_filter" in message
        or "content management policy" in message
        or "cyber_policy" in message
        or "safety check" in message
        or "safety filter" in message
    )


def _messages_for_attempt(
    *,
    messages: list[LLMMessage],
    stage_name: str,
    target_id: str,
    attempt_index: int,
    max_attempts: int,
    previous_error: BaseException | None,
    retry_nonce: int,
) -> list[LLMMessage]:
    if attempt_index == 0:
        if retry_nonce > 0:
            return [
                *messages,
                LLMMessage(
                    role="user",
                    content=(
                        f"Previous live JSON run already failed for {stage_name} {target_id}. "
                        f"Make a fresh attempt and return only a strict JSON object matching the requested schema. "
                        f"The first byte must be {{ and the last byte must be }}. "
                        f"The entire response must be accepted by Python json.loads(response_text). "
                        f"Do not include markdown fences, XML/thinking tags, prose, self-review, or chain-of-thought. "
                        f"Initial cache bypass nonce: {retry_nonce}."
                    ),
                ),
            ]
        return list(messages)
    if previous_error is None:
        raise ValueError("previous_error is required for retry attempts")
    return [
        *messages,
        LLMMessage(
            role="user",
            content=(
                f"Previous live JSON attempt failed for {stage_name} {target_id}: {previous_error}. "
                f"Return only a corrected strict JSON object matching the requested schema. "
                f"The first byte must be {{ and the last byte must be }}. "
                f"The entire response must be accepted by Python json.loads(response_text). "
                f"Do not include markdown fences, XML/thinking tags, prose, self-review, or chain-of-thought. "
                f"This is retry attempt {attempt_index + 1} of {max_attempts}. "
                f"Retry cache bypass nonce: {retry_nonce}."
            ),
        ),
    ]


def _count_existing_live_errors(*, stage_dir: Path, stage_name: str, target_id: str, model: str) -> int:
    error_path = stage_dir / "live_errors.jsonl"
    if not error_path.exists():
        return 0
    count = 0
    with error_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                row.get("stage_name") == stage_name
                and row.get("target_id") == target_id
                and row.get("model") == model
            ):
                count += 1
    return count


def record_live_error(
    *,
    stage_dir: Path,
    stage_name: str,
    target_id: str,
    model: str,
    error: BaseException,
    raw_response: str,
    attempt: int,
    max_attempts: int,
) -> None:
    ensure_directory(stage_dir)
    row = {
        "stage_name": stage_name,
        "target_id": target_id,
        "model": model,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "raw_response": raw_response,
        "attempt": attempt,
        "max_attempts": max_attempts,
    }
    status_code = getattr(error, "status_code", None)
    if status_code is not None:
        row["status_code"] = int(status_code)
    with (stage_dir / "live_errors.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def require_fields(payload: dict[str, Any], fields: tuple[str, ...], *, context: str) -> None:
    for field in fields:
        if field not in payload:
            raise LiveSchemaError(f"{context} missing required field: {field}")


def require_object(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LiveSchemaError(f"{context} must be an object")
    return value


def require_object_list(value: Any, *, context: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise LiveSchemaError(f"{context} must be a list")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise LiveSchemaError(f"{context}[{index}] must be an object")
        rows.append(item)
    return rows


def require_bool(value: Any, *, context: str) -> bool:
    if not isinstance(value, bool):
        raise LiveSchemaError(f"{context} must be a bool")
    return value


def require_str(value: Any, *, context: str) -> str:
    if not isinstance(value, str):
        raise LiveSchemaError(f"{context} must be a string")
    if not value.strip():
        raise LiveSchemaError(f"{context} must be a non-empty string")
    return value


def require_str_allow_empty(value: Any, *, context: str) -> str:
    if not isinstance(value, str):
        raise LiveSchemaError(f"{context} must be a string")
    return value


def require_str_list(value: Any, *, context: str) -> list[str]:
    if not isinstance(value, list):
        raise LiveSchemaError(f"{context} must be a list")
    labels: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise LiveSchemaError(f"{context}[{index}] must be a non-empty string")
        labels.append(item)
    return labels


def require_number(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise LiveSchemaError(f"{context} must be a number")
    return float(value)
