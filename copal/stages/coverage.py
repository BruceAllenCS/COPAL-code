from __future__ import annotations

from pathlib import Path

from copal.config import require_execution_mode
from copal.io import ensure_directory, write_json, write_jsonl
from copal.llm import LLMClient, complete_json
from copal.prompts import build_coverage_messages


def normalize_coverage_result(
    query_id: str,
    signature_label: str,
    facet_labels: list[str],
    coverage_rationale: str = "",
    coverage_judge_model: str = "",
) -> dict[str, object]:
    return {
        "query_id": query_id,
        "signature_label": signature_label,
        "facet_labels": list(facet_labels),
        "coverage_set": list(facet_labels),
        "coverage_rationale": coverage_rationale,
        "coverage_judge_model": coverage_judge_model,
    }


def run_coverage_stage(
    *,
    coverage_dir: Path,
    accepted_queries: list[dict[str, object]],
    execution_mode: str,
    coverage_client: LLMClient | None = None,
    coverage_model: str = "",
) -> dict[str, object]:
    require_execution_mode(execution_mode)
    ensure_directory(coverage_dir)
    coverage_rows = []
    for query in accepted_queries:
        if execution_mode == "live":
            if coverage_client is None or not coverage_model:
                raise ValueError("Live coverage stage requires coverage_client and coverage_model")
            payload = complete_json(
                client=coverage_client,
                model=coverage_model,
                messages=build_coverage_messages(query_row=query),
            )
            coverage_rows.append(
                normalize_coverage_result(
                    query_id=str(query["query_id"]),
                    signature_label=str(payload["signature_label"]),
                    facet_labels=[str(label) for label in payload["facet_labels"]],
                    coverage_rationale=str(payload.get("coverage_rationale", "")),
                    coverage_judge_model=coverage_model,
                )
            )
        else:
            coverage_rows.append(
                normalize_coverage_result(
                    query_id=str(query["query_id"]),
                    signature_label=str(query["signature_proposal"]),
                    facet_labels=[str(query["target_facet"])],
                )
            )
    summary = {
        "coverage_result_count": len(coverage_rows),
        "signature_count": len({row["signature_label"] for row in coverage_rows}),
        "execution_mode": execution_mode,
    }
    write_jsonl(coverage_dir / "coverage_results.jsonl", coverage_rows)
    write_json(coverage_dir / "coverage_summary.json", summary)
    return summary
