from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from copal.config import require_execution_mode
from copal.io import append_jsonl, ensure_directory, read_jsonl, write_json, write_jsonl
from copal.llm import LLMClient
from copal.live_validation import complete_live_json_object, require_bool, require_str, require_str_list
from copal.prompts import build_response_judge_messages

RESPONSE_JUDGMENT_REQUIRED_FIELDS = (
    "handling_pattern_match",
    "response_mode_correct",
    "boundary_correct",
    "required_actions_correct",
    "mandatory_content_correct",
    "overall_correct",
    "under_enforcement",
    "over_enforcement",
    "observed_handling",
    "observed_facets",
    "response_outcome",
    "scoring_rationale",
    "evidence_span",
)


def _build_response_judgment(benchmark_item: dict[str, object]) -> dict[str, object]:
    row = {
        "response_id": benchmark_item["response_id"],
        "item_id": benchmark_item["item_id"],
        "response_model": benchmark_item["response_model"],
        "signature": benchmark_item["signature"],
        "facet": benchmark_item["facet"],
        "target_facet": benchmark_item.get("target_facet", benchmark_item["facet"]),
        "target_facets": list(benchmark_item.get("target_facets", [benchmark_item.get("target_facet", benchmark_item["facet"])])),
        "handling_pattern_match": True,
        "response_mode_correct": True,
        "boundary_correct": True,
        "required_actions_correct": True,
        "mandatory_content_correct": True,
        "overall_correct": True,
        "under_enforcement": False,
        "over_enforcement": False,
        "observed_handling": ["deterministic_expected_handling"],
        "observed_facets": [],
        "response_outcome": "acceptable",
        "scoring_rationale": "Deterministic execution mode marked the expected handling pattern as matched.",
        "evidence_span": "deterministic execution mode",
    }
    _attach_pairing_metadata(row=row, benchmark_item=benchmark_item)
    return row


def _normalize_response_judgment(
    *,
    benchmark_item: dict[str, object],
    payload: dict[str, object],
    response_judge_model: str,
) -> dict[str, object]:
    row = {
        "response_id": benchmark_item["response_id"],
        "item_id": benchmark_item["item_id"],
        "response_model": benchmark_item["response_model"],
        "signature": benchmark_item["signature"],
        "facet": benchmark_item.get("facet", benchmark_item.get("target_facet", "")),
        "target_facet": benchmark_item.get("target_facet", benchmark_item.get("facet", "")),
        "target_facets": list(benchmark_item.get("target_facets", [benchmark_item.get("target_facet", benchmark_item.get("facet", ""))])),
        "handling_pattern_match": require_bool(payload["handling_pattern_match"], context=f"response_judgment {benchmark_item['item_id']}.handling_pattern_match"),
        "response_mode_correct": require_bool(payload["response_mode_correct"], context=f"response_judgment {benchmark_item['item_id']}.response_mode_correct"),
        "boundary_correct": require_bool(payload["boundary_correct"], context=f"response_judgment {benchmark_item['item_id']}.boundary_correct"),
        "required_actions_correct": require_bool(payload["required_actions_correct"], context=f"response_judgment {benchmark_item['item_id']}.required_actions_correct"),
        "mandatory_content_correct": require_bool(payload["mandatory_content_correct"], context=f"response_judgment {benchmark_item['item_id']}.mandatory_content_correct"),
        "overall_correct": require_bool(payload["overall_correct"], context=f"response_judgment {benchmark_item['item_id']}.overall_correct"),
        "under_enforcement": require_bool(payload["under_enforcement"], context=f"response_judgment {benchmark_item['item_id']}.under_enforcement"),
        "over_enforcement": require_bool(payload["over_enforcement"], context=f"response_judgment {benchmark_item['item_id']}.over_enforcement"),
        "observed_handling": require_str_list(payload["observed_handling"], context=f"response_judgment {benchmark_item['item_id']}.observed_handling"),
        "observed_facets": require_str_list(payload["observed_facets"], context=f"response_judgment {benchmark_item['item_id']}.observed_facets"),
        "response_outcome": require_str(payload["response_outcome"], context=f"response_judgment {benchmark_item['item_id']}.response_outcome"),
        "scoring_rationale": require_str(payload["scoring_rationale"], context=f"response_judgment {benchmark_item['item_id']}.scoring_rationale"),
        "evidence_span": require_str(payload["evidence_span"], context=f"response_judgment {benchmark_item['item_id']}.evidence_span"),
        "response_judge_model": response_judge_model,
    }
    _attach_pairing_metadata(row=row, benchmark_item=benchmark_item)
    return row


def _attach_pairing_metadata(*, row: dict[str, object], benchmark_item: dict[str, object]) -> None:
    for key in (
        "item_type",
        "paired_composed_item_id",
        "paired_composed_query_id",
        "projection_clause_id",
        "projection_index",
    ):
        if key in benchmark_item:
            row[key] = benchmark_item[key]


def _validate_response_judgment_payload(*, benchmark_item: dict[str, object], payload: dict[str, object]) -> None:
    _normalize_response_judgment(
        benchmark_item=benchmark_item,
        payload=payload,
        response_judge_model="schema-check",
    )


def _summarize_signature_scores(rows: list[dict[str, object]]) -> dict[str, object]:
    totals: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "count": 0})
    total_correct = 0
    for row in rows:
        signature = str(row["signature"])
        totals[signature]["count"] += 1
        if bool(row["overall_correct"]):
            totals[signature]["correct"] += 1
            total_correct += 1
    return {
        "overall_accuracy": total_correct / len(rows) if rows else 0.0,
        "accuracy_per_signature": {
            key: value["correct"] / value["count"] for key, value in totals.items()
        },
    }


def run_response_judgment_stage(
    *,
    evaluation_dir: Path,
    benchmark_items: list[dict[str, object]],
    execution_mode: str,
    response_judge_client: LLMClient | None = None,
    response_judge_model: str = "",
    live_max_workers: int = 1,
) -> dict[str, object]:
    require_execution_mode(execution_mode)
    if live_max_workers < 1:
        raise ValueError("live_max_workers must be positive")
    ensure_directory(evaluation_dir)
    responses = read_jsonl(evaluation_dir / "chatbot_responses.jsonl")
    item_by_id = {str(item["item_id"]): item for item in benchmark_items}
    judge_inputs = []
    response_items: list[tuple[dict[str, object], dict[str, object]]] = []
    for response in responses:
        item_id = str(response["item_id"])
        if item_id not in item_by_id:
            raise ValueError(f"Chatbot response does not match a benchmark item: {item_id}")
        benchmark_item = {
            **item_by_id[item_id],
            "response_id": response.get("response_id", item_id),
            "response_model": response.get("response_model", ""),
        }
        judge_inputs.append(
            {
                "response_id": benchmark_item["response_id"],
                "item_id": item_id,
                "response_model": benchmark_item["response_model"],
                "benchmark_item": benchmark_item,
                "response_text": response["response_text"],
            }
        )
        response_items.append((benchmark_item, response))
    write_jsonl(evaluation_dir / "response_judge_inputs.jsonl", judge_inputs)
    expected_response_ids = {str(item["response_id"]) for item, _response in response_items}
    judgments_path = evaluation_dir / "response_judgments.jsonl"
    existing_judgments: dict[str, dict[str, object]] = {}
    if judgments_path.exists():
        for row in read_jsonl(judgments_path):
            response_id = str(row.get("response_id", ""))
            if response_id not in expected_response_ids:
                raise ValueError(f"Existing response judgment is not part of this run: {response_id}")
            if response_id in existing_judgments:
                raise ValueError(f"Duplicate response judgment in checkpoint: {response_id}")
            existing_judgments[response_id] = row
    missing_items = [
        (item, response)
        for item, response in response_items
        if str(item["response_id"]) not in existing_judgments
    ]

    def build_judgment_row(item: dict[str, object], response: dict[str, object]) -> dict[str, object]:
        response_id = str(item["response_id"])
        response_text = str(response["response_text"])
        if execution_mode == "live":
            if response_judge_client is None or not response_judge_model:
                raise ValueError("Live response judgment requires response_judge_client and response_judge_model")
            payload = complete_live_json_object(
                client=response_judge_client,
                model=response_judge_model,
                messages=build_response_judge_messages(
                    benchmark_item=item,
                    response_text=response_text,
                ),
                stage_dir=evaluation_dir,
                stage_name="response_judgment",
                target_id=response_id,
                required_fields=RESPONSE_JUDGMENT_REQUIRED_FIELDS,
                validator=lambda payload: _validate_response_judgment_payload(
                    benchmark_item=item,
                    payload=payload,
                ),
            )
            return _normalize_response_judgment(
                benchmark_item=item,
                payload=dict(payload),
                response_judge_model=response_judge_model,
            )
        return _build_response_judgment(item)

    if execution_mode == "live" and live_max_workers > 1 and missing_items:
        with ThreadPoolExecutor(max_workers=live_max_workers) as executor:
            futures = [executor.submit(build_judgment_row, item, response) for item, response in missing_items]
            for future in as_completed(futures):
                row = future.result()
                append_jsonl(judgments_path, row)
                existing_judgments[str(row["response_id"])] = row
    else:
        for item, response in missing_items:
            row = build_judgment_row(item, response)
            append_jsonl(judgments_path, row)
            existing_judgments[str(row["response_id"])] = row

    judgments = []
    for item, _response in response_items:
        response_id = str(item["response_id"])
        if response_id in existing_judgments:
            row = existing_judgments[response_id]
        else:
            raise ValueError(f"Missing response judgment after live execution: {response_id}")
        judgments.append(row)
    per_signature = _summarize_signature_scores(judgments)

    facet_totals: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "count": 0})
    for row in judgments:
        facet = str(row["facet"])
        facet_totals[facet]["count"] += 1
        if bool(row["overall_correct"]):
            facet_totals[facet]["correct"] += 1
    per_facet = {
        facet: {"accuracy": values["correct"] / values["count"]}
        for facet, values in facet_totals.items()
    }
    observed_facets = sorted(
        {
            str(observed_facet)
            for row in judgments
            for observed_facet in row.get("observed_facets", [])
        }
    )
    summary = {
        "judgment_count": len(judgments),
        "response_count": len(responses),
        "item_count": len(benchmark_items),
        "model_count": len({str(row.get("response_model", "")) for row in responses}),
        "overall_accuracy": per_signature["overall_accuracy"],
        "signature_count": len(per_signature["accuracy_per_signature"]),
        "facet_count": len(per_facet),
        "observed_facet_count": len(observed_facets),
        "observed_facets": observed_facets,
        "execution_mode": execution_mode,
    }
    write_jsonl(evaluation_dir / "response_judgments.jsonl", judgments)
    write_jsonl(evaluation_dir / "per_item_scores.jsonl", judgments)
    write_json(evaluation_dir / "per_signature_scores.json", per_signature)
    write_json(evaluation_dir / "per_facet_scores.json", per_facet)
    write_json(evaluation_dir / "evaluation_summary.json", summary)
    return summary
