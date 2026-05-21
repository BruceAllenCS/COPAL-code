from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from copal.config import require_execution_mode
from copal.io import ensure_directory, write_json, write_jsonl
from copal.llm import LLMClient, complete_json
from copal.prompts import build_downstream_chat_messages, build_response_judge_messages


def build_response_judgment(
    *,
    item_id: str,
    signature: str,
    facet: str,
    handling_pattern_match: bool,
) -> dict[str, object]:
    return {
        "item_id": item_id,
        "signature": signature,
        "facet": facet,
        "target_facet": facet,
        "target_facets": [facet],
        "handling_pattern_match": handling_pattern_match,
        "response_mode_correct": handling_pattern_match,
        "boundary_correct": handling_pattern_match,
        "required_actions_correct": handling_pattern_match,
        "mandatory_content_correct": handling_pattern_match,
        "overall_correct": handling_pattern_match,
        "under_enforcement": False,
        "over_enforcement": False,
        "observed_handling": ["deterministic_expected_handling"],
        "observed_facets": [],
        "response_outcome": "acceptable" if handling_pattern_match else "error",
        "scoring_rationale": "",
        "evidence_span": "deterministic execution mode",
    }


def summarize_scores(rows: list[dict[str, object]]) -> dict[str, object]:
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
        "policy_handling_score": total_correct / len(rows) if rows else 0.0,
        "policy_handling_error_rate": (len(rows) - total_correct) / len(rows) if rows else 0.0,
        "accuracy_per_signature": {
            key: value["correct"] / value["count"] for key, value in totals.items()
        },
        "policy_handling_score_per_signature": {
            key: value["correct"] / value["count"] for key, value in totals.items()
        },
    }


def _build_chatbot_response(system_prompt: str, item: dict[str, object]) -> str:
    signature = str(item["signature"])
    facet = str(item["target_facet"])
    query_text = str(item["query_text"])
    return (
        f"{system_prompt[:80]}... "
        f"Handling pattern: {signature} / {facet}. "
        f"User request: {query_text}"
    )


def run_evaluation_stage(
    *,
    evaluation_dir: Path,
    benchmark_items: list[dict[str, object]],
    system_prompt: str,
    execution_mode: str,
    downstream_client: LLMClient | None = None,
    downstream_model: str = "",
    response_judge_client: LLMClient | None = None,
    response_judge_model: str = "",
) -> dict[str, object]:
    require_execution_mode(execution_mode)
    ensure_directory(evaluation_dir)
    requests = [
        {
            "item_id": item["item_id"],
            "query_text": item["query_text"],
            "system_prompt": system_prompt,
        }
        for item in benchmark_items
    ]
    responses = []
    per_item_scores = []
    for item in benchmark_items:
        if execution_mode == "live":
            if downstream_client is None or response_judge_client is None:
                raise ValueError("Live evaluation requires downstream_client and response_judge_client")
            if not downstream_model or not response_judge_model:
                raise ValueError("Live evaluation requires downstream_model and response_judge_model")
            response = downstream_client.complete(
                model=downstream_model,
                messages=build_downstream_chat_messages(
                    system_prompt=system_prompt,
                    query_text=str(item["query_text"]),
                ),
            )
            response_text = response.text
            judgment_payload = complete_json(
                client=response_judge_client,
                model=response_judge_model,
                messages=build_response_judge_messages(
                    benchmark_item=item,
                    response_text=response_text,
                ),
            )
            score_row = {
                "item_id": str(item["item_id"]),
                "signature": str(item["signature"]),
                "facet": str(item["target_facet"]),
                "target_facet": str(item["target_facet"]),
                "target_facets": [str(facet) for facet in item.get("target_facets", [item["target_facet"]])],
                "handling_pattern_match": bool(judgment_payload["handling_pattern_match"]),
                "response_mode_correct": bool(judgment_payload["response_mode_correct"]),
                "boundary_correct": bool(judgment_payload["boundary_correct"]),
                "required_actions_correct": bool(judgment_payload["required_actions_correct"]),
                "mandatory_content_correct": bool(judgment_payload["mandatory_content_correct"]),
                "overall_correct": bool(judgment_payload["overall_correct"]),
                "under_enforcement": bool(judgment_payload["under_enforcement"]),
                "over_enforcement": bool(judgment_payload["over_enforcement"]),
                "observed_handling": [str(value) for value in judgment_payload["observed_handling"]],
                "observed_facets": [str(facet) for facet in judgment_payload["observed_facets"]],
                "response_outcome": str(judgment_payload["response_outcome"]),
                "scoring_rationale": str(judgment_payload["scoring_rationale"]),
                "evidence_span": str(judgment_payload["evidence_span"]),
            }
        else:
            response_text = _build_chatbot_response(system_prompt, item)
            score_row = build_response_judgment(
                item_id=str(item["item_id"]),
                signature=str(item["signature"]),
                facet=str(item["target_facet"]),
                handling_pattern_match=True,
            )
        responses.append(
            {
                "item_id": item["item_id"],
                "response_text": response_text,
            }
        )
        per_item_scores.append(score_row)
    per_signature_scores = summarize_scores(per_item_scores)
    per_facet_scores: dict[str, dict[str, float]] = {}
    facet_totals: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "count": 0})
    for row in per_item_scores:
        facet = str(row["facet"])
        facet_totals[facet]["count"] += 1
        if bool(row["overall_correct"]):
            facet_totals[facet]["correct"] += 1
    per_facet_scores = {
        facet: {"accuracy": values["correct"] / values["count"]}
        for facet, values in facet_totals.items()
    }
    summary = {
        "response_count": len(responses),
        "overall_accuracy": per_signature_scores["overall_accuracy"],
        "policy_handling_score": per_signature_scores["policy_handling_score"],
        "policy_handling_error_rate": per_signature_scores["policy_handling_error_rate"],
        "signature_count": len(per_signature_scores["accuracy_per_signature"]),
        "facet_count": len(per_facet_scores),
        "execution_mode": execution_mode,
    }

    write_jsonl(evaluation_dir / "chatbot_requests.jsonl", requests)
    write_jsonl(evaluation_dir / "chatbot_responses.jsonl", responses)
    write_jsonl(evaluation_dir / "evaluation_judge_results.jsonl", per_item_scores)
    write_jsonl(evaluation_dir / "per_item_scores.jsonl", per_item_scores)
    write_json(evaluation_dir / "per_signature_scores.json", per_signature_scores)
    write_json(evaluation_dir / "per_facet_scores.json", per_facet_scores)
    write_json(evaluation_dir / "evaluation_summary.json", summary)
    return summary
