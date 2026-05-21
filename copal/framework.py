from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from copal.chatbot_adapters import invoke_command_chatbot, invoke_http_chatbot
from copal.io import append_jsonl, ensure_directory, read_json, read_jsonl, write_json, write_jsonl


def load_benchmark_items(run_dir: Path) -> list[dict[str, object]]:
    path = run_dir / "selection" / "benchmark_items_final.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Selected benchmark items do not exist: {path}")
    items = read_jsonl(path)
    if not items:
        raise ValueError(f"Selected benchmark item file is empty: {path}")
    seen: set[str] = set()
    for item in items:
        item_id = str(item["item_id"])
        if item_id in seen:
            raise ValueError(f"Duplicate benchmark item_id: {item_id}")
        seen.add(item_id)
        if not str(item["query_text"]).strip():
            raise ValueError(f"Benchmark item has empty query_text: {item_id}")
    return items


def load_system_prompt(run_dir: Path) -> str:
    path = run_dir / "inputs" / "selected_system_prompt.json"
    if not path.exists():
        raise FileNotFoundError(f"Selected system prompt does not exist: {path}")
    payload = read_json(path)
    system_prompt = str(payload["system_prompt"]).strip()
    if not system_prompt:
        raise ValueError(f"Selected system prompt is empty: {path}")
    return system_prompt


def write_imported_responses(
    *,
    evaluation_dir: Path,
    benchmark_items: list[dict[str, object]],
    responses_path: Path,
    bot_id: str,
) -> dict[str, object]:
    if not responses_path.exists():
        raise FileNotFoundError(f"Imported chatbot responses do not exist: {responses_path}")
    ensure_directory(evaluation_dir)
    expected_ids = {str(item["item_id"]) for item in benchmark_items}
    imported_rows = read_jsonl(responses_path)
    rows_by_item_id: dict[str, dict[str, object]] = {}
    for row in imported_rows:
        item_id = str(row["item_id"])
        if item_id in rows_by_item_id:
            raise ValueError(f"Duplicate imported chatbot response for item_id: {item_id}")
        rows_by_item_id[item_id] = row
    imported_ids = set(rows_by_item_id)
    if imported_ids != expected_ids:
        missing = sorted(expected_ids - imported_ids)
        extra = sorted(imported_ids - expected_ids)
        raise ValueError(
            "Imported chatbot response set does not match selected benchmark items: "
            f"missing={missing}, extra={extra}"
        )
    responses = [
        _canonical_response_row(
            item_id=str(item["item_id"]),
            response_text=rows_by_item_id[str(item["item_id"])]["response_text"],
            bot_id=bot_id,
            response_id=str(rows_by_item_id[str(item["item_id"])].get("response_id", "")),
        )
        for item in benchmark_items
    ]
    write_jsonl(evaluation_dir / "chatbot_responses.jsonl", responses)
    write_jsonl(
        evaluation_dir / "chatbot_requests.jsonl",
        [
            {
                "item_id": item["item_id"],
                "response_id": f"{item['item_id']}::{bot_id}",
                "query_text": item["query_text"],
                "adapter": "import",
            }
            for item in benchmark_items
        ],
    )
    summary = {
        "adapter": "import",
        "bot_id": bot_id,
        "item_count": len(benchmark_items),
        "response_count": len(responses),
    }
    write_json(evaluation_dir / "chatbot_summary.json", summary)
    return summary


def run_http_chatbot_probe(
    *,
    evaluation_dir: Path,
    benchmark_items: list[dict[str, object]],
    system_prompt: str,
    endpoint: str,
    response_json_key: str,
    bot_id: str,
    live_max_workers: int,
    timeout: int,
    headers: dict[str, str] | None = None,
    post: Callable[..., object] | None = None,
) -> dict[str, object]:
    return _run_probe(
        evaluation_dir=evaluation_dir,
        benchmark_items=benchmark_items,
        bot_id=bot_id,
        adapter_name="http",
        live_max_workers=live_max_workers,
        request_builder=lambda item: {
            "item_id": item["item_id"],
            "response_id": f"{item['item_id']}::{bot_id}",
            "query_text": item["query_text"],
            "adapter": "http",
            "endpoint": endpoint,
        },
        response_builder=lambda item: invoke_http_chatbot(
            endpoint=endpoint,
            item=item,
            system_prompt=system_prompt,
            response_json_key=response_json_key,
            headers=headers,
            timeout=timeout,
            post=post,
        ),
    )


def run_command_chatbot_probe(
    *,
    evaluation_dir: Path,
    benchmark_items: list[dict[str, object]],
    system_prompt: str,
    command: list[str],
    output_mode: str,
    response_json_key: str,
    bot_id: str,
    live_max_workers: int,
    timeout: int,
) -> dict[str, object]:
    return _run_probe(
        evaluation_dir=evaluation_dir,
        benchmark_items=benchmark_items,
        bot_id=bot_id,
        adapter_name="command",
        live_max_workers=live_max_workers,
        request_builder=lambda item: {
            "item_id": item["item_id"],
            "response_id": f"{item['item_id']}::{bot_id}",
            "query_text": item["query_text"],
            "adapter": "command",
            "command": list(command),
        },
        response_builder=lambda item: invoke_command_chatbot(
            command=command,
            item=item,
            system_prompt=system_prompt,
            output_mode=output_mode,
            response_json_key=response_json_key,
            timeout=timeout,
        ),
    )


def _run_probe(
    *,
    evaluation_dir: Path,
    benchmark_items: list[dict[str, object]],
    bot_id: str,
    adapter_name: str,
    live_max_workers: int,
    request_builder: Callable[[dict[str, object]], dict[str, object]],
    response_builder: Callable[[dict[str, object]], str],
) -> dict[str, object]:
    if live_max_workers < 1:
        raise ValueError("live_max_workers must be positive")
    ensure_directory(evaluation_dir)
    requests = [request_builder(item) for item in benchmark_items]
    write_jsonl(evaluation_dir / "chatbot_requests.jsonl", requests)
    responses_path = evaluation_dir / "chatbot_responses.jsonl"
    if responses_path.exists():
        responses_path.unlink()

    def build_row(item: dict[str, object]) -> dict[str, object]:
        item_id = str(item["item_id"])
        return _canonical_response_row(
            item_id=item_id,
            response_text=response_builder(item),
            bot_id=bot_id,
            response_id=f"{item_id}::{bot_id}",
        )

    if live_max_workers == 1:
        for item in benchmark_items:
            append_jsonl(responses_path, build_row(item))
    else:
        with ThreadPoolExecutor(max_workers=live_max_workers) as executor:
            futures = [executor.submit(build_row, item) for item in benchmark_items]
            for future in as_completed(futures):
                append_jsonl(responses_path, future.result())
    responses_by_id = {str(row["item_id"]): row for row in read_jsonl(responses_path)}
    responses = [responses_by_id[str(item["item_id"])] for item in benchmark_items]
    write_jsonl(responses_path, responses)
    summary = {
        "adapter": adapter_name,
        "bot_id": bot_id,
        "item_count": len(benchmark_items),
        "response_count": len(responses),
    }
    write_json(evaluation_dir / "chatbot_summary.json", summary)
    return summary


def _canonical_response_row(
    *,
    item_id: str,
    response_text: object,
    bot_id: str,
    response_id: str,
) -> dict[str, object]:
    if not bot_id.strip():
        raise ValueError("bot_id must be non-empty")
    if not isinstance(response_text, str) or not response_text.strip():
        raise ValueError(f"response_text must be a non-empty string for item_id: {item_id}")
    final_response_id = response_id.strip() or f"{item_id}::{bot_id}"
    return {
        "response_id": final_response_id,
        "item_id": item_id,
        "response_text": response_text.strip(),
        "response_model": bot_id,
    }
