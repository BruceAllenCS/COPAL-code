from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from copal.config import require_execution_mode
from copal.io import append_jsonl, ensure_directory, read_jsonl, write_json, write_jsonl
from copal.llm import LLMClient, LLMProviderError
from copal.prompts import build_downstream_chat_messages


def run_downstream_chatbot_stage(
    *,
    evaluation_dir: Path,
    benchmark_items: list[dict[str, object]],
    system_prompt: str,
    execution_mode: str,
    downstream_client: LLMClient | None = None,
    downstream_model: str = "",
    downstream_models: tuple[str, ...] | list[str] = (),
    live_max_workers: int = 1,
) -> dict[str, object]:
    require_execution_mode(execution_mode)
    if live_max_workers < 1:
        raise ValueError("live_max_workers must be positive")
    ensure_directory(evaluation_dir)
    model_roster = tuple(str(model).strip() for model in downstream_models if str(model).strip())
    if not model_roster:
        model_roster = (downstream_model,) if downstream_model else ("deterministic",)
    expected_response_ids = {
        f"{item['item_id']}::{model}"
        for item in benchmark_items
        for model in model_roster
    }
    requests = [
        {
            "item_id": item["item_id"],
            "response_id": f"{item['item_id']}::{model}",
            "query_text": item["query_text"],
            "system_prompt": system_prompt,
        }
        for item in benchmark_items
        for model in model_roster
    ]
    write_jsonl(evaluation_dir / "chatbot_requests.jsonl", requests)
    existing_responses: dict[str, dict[str, object]] = {}
    responses_path = evaluation_dir / "chatbot_responses.jsonl"
    if responses_path.exists():
        for row in read_jsonl(responses_path):
            response_id = str(row.get("response_id", ""))
            if response_id not in expected_response_ids:
                raise ValueError(f"Existing downstream response is not part of this run: {response_id}")
            if response_id in existing_responses:
                raise ValueError(f"Duplicate downstream response in checkpoint: {response_id}")
            existing_responses[response_id] = row
    missing_jobs = [
        (item, model)
        for item in benchmark_items
        for model in model_roster
        if f"{item['item_id']}::{model}" not in existing_responses
    ]

    def build_response_row(item: dict[str, object], model: str) -> dict[str, object]:
        response_id = f"{item['item_id']}::{model}"
        if execution_mode == "live":
            if downstream_client is None:
                raise ValueError("Live downstream chatbot requires downstream_client")
            try:
                response = downstream_client.complete(
                    model=model,
                    messages=build_downstream_chat_messages(
                        system_prompt=system_prompt,
                        query_text=str(item["query_text"]),
                    ),
                )
                response_text = response.text
                provider_error = None
            except LLMProviderError as exc:
                if not _is_provider_safety_block(exc):
                    raise
                response_text = (
                    "The provider-side safety filter blocked this model request before generation. "
                    f"Recorded provider error: {exc}"
                )
                provider_error = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                status_code = getattr(exc, "status_code", None)
                if status_code is not None:
                    provider_error["status_code"] = int(status_code)
        else:
            response_text = (
                f"{system_prompt[:80]}... Handling pattern: {item['signature']} / {item['target_facet']}. "
                f"Downstream model: {model}. User request: {item['query_text']}"
            )
            provider_error = None
        row = {
            "response_id": response_id,
            "item_id": item["item_id"],
            "response_text": response_text,
            "response_model": model,
        }
        if provider_error is not None:
            row["provider_error"] = provider_error
        return row

    if execution_mode == "live" and live_max_workers > 1 and missing_jobs:
        with ThreadPoolExecutor(max_workers=live_max_workers) as executor:
            futures = [executor.submit(build_response_row, item, model) for item, model in missing_jobs]
            for future in as_completed(futures):
                row = future.result()
                append_jsonl(responses_path, row)
                existing_responses[str(row["response_id"])] = row
    else:
        for item, model in missing_jobs:
            row = build_response_row(item, model)
            append_jsonl(responses_path, row)
            existing_responses[str(row["response_id"])] = row

    responses = [
        _canonical_response_row(existing_responses[response_id])
        for item in benchmark_items
        for model in model_roster
        for response_id in (f"{item['item_id']}::{model}",)
    ]
    summary = {
        "item_count": len(benchmark_items),
        "model_count": len(model_roster),
        "downstream_models": list(model_roster),
        "response_count": len(responses),
        "execution_mode": execution_mode,
    }
    write_jsonl(evaluation_dir / "chatbot_responses.jsonl", responses)
    write_json(evaluation_dir / "chatbot_summary.json", summary)
    return summary


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


def _canonical_response_row(row: dict[str, object]) -> dict[str, object]:
    canonical = {
        "response_id": row["response_id"],
        "item_id": row["item_id"],
        "response_text": row["response_text"],
        "response_model": row["response_model"],
    }
    if "provider_error" in row:
        canonical["provider_error"] = row["provider_error"]
    return canonical
