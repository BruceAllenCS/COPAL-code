from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from copal.config import require_execution_mode
from copal.io import ensure_directory, write_json, write_jsonl
from copal.llm import LLMClient
from copal.live_validation import LiveSchemaError, complete_live_json_object, require_str, require_str_list
from copal.prompts import build_coverage_messages
from copal.stages.coverage import normalize_coverage_result


def run_coverage_judge_stage(
    *,
    coverage_dir: Path,
    accepted_queries: list[dict[str, object]],
    facet_library: dict[str, tuple[str, ...] | list[str]],
    execution_mode: str,
    coverage_client: LLMClient | None = None,
    coverage_model: str = "",
    live_max_workers: int = 1,
) -> dict[str, object]:
    require_execution_mode(execution_mode)
    if live_max_workers < 1:
        raise ValueError("live_max_workers must be positive")
    ensure_directory(coverage_dir)
    universes: dict[str, dict[str, object]] = {}
    results: list[dict[str, object]] = []
    accepted_coverages: list[dict[str, object]] = []

    def run_job(
        index: int,
        query: dict[str, object],
    ) -> tuple[int, str, str, list[str], dict[str, object], dict[str, object]]:
        signature = str(query["signature_proposal"])
        composition_id = str(query["composition_id"])
        facet_universe = list(facet_library.get(signature, ()))
        if execution_mode == "live":
            if coverage_client is None or not coverage_model:
                raise ValueError("Live coverage judge requires coverage_client and coverage_model")
            target_id = str(query["query_id"])
            payload = complete_live_json_object(
                client=coverage_client,
                model=coverage_model,
                messages=build_coverage_messages(query_row=query, facet_library=facet_library),
                stage_dir=coverage_dir,
                stage_name="coverage_judge",
                target_id=target_id,
                required_fields=("signature_label", "facet_labels", "coverage_rationale"),
                validator=lambda payload: _validate_coverage_payload(
                    payload=payload,
                    facet_library=facet_library,
                    target_id=target_id,
                ),
            )
            signature_label = payload["signature_label"]
            facet_labels = payload["facet_labels"]
            facet_universe = list(facet_library.get(signature_label, facet_universe))
            result = normalize_coverage_result(
                query_id=str(query["query_id"]),
                signature_label=signature_label,
                facet_labels=facet_labels,
                coverage_rationale=payload["coverage_rationale"],
                coverage_judge_model=coverage_model,
            )
        else:
            signature_label = signature
            result = normalize_coverage_result(
                query_id=str(query["query_id"]),
                signature_label=signature,
                facet_labels=[str(query["target_facet"])],
                coverage_rationale="Deterministic coverage label aligned to proposal facet.",
                coverage_judge_model="external_coverage_judge",
            )
        accepted_coverage = {**query, **result, "facet_universe": facet_universe}
        return index, composition_id, str(signature_label), facet_universe, result, accepted_coverage

    job_results: dict[int, tuple[str, str, list[str], dict[str, object], dict[str, object]]] = {}
    jobs = list(enumerate(accepted_queries))
    if execution_mode == "live" and live_max_workers > 1:
        with ThreadPoolExecutor(max_workers=live_max_workers) as executor:
            futures = {executor.submit(run_job, index, query): index for index, query in jobs}
            for future in as_completed(futures):
                index, composition_id, signature_label, facet_universe, result, accepted_coverage = future.result()
                job_results[index] = (composition_id, signature_label, facet_universe, result, accepted_coverage)
    else:
        for index, query in jobs:
            index, composition_id, signature_label, facet_universe, result, accepted_coverage = run_job(index, query)
            job_results[index] = (composition_id, signature_label, facet_universe, result, accepted_coverage)

    for index in sorted(job_results):
        composition_id, signature_label, facet_universe, result, accepted_coverage = job_results[index]
        universes[composition_id] = {
            "composition_id": composition_id,
            "signature": signature_label,
            "facet_universe": facet_universe,
        }
        results.append(result)
        accepted_coverages.append(accepted_coverage)

    summary = {
        "coverage_result_count": len(results),
        "composition_universe_count": len(universes),
        "live_max_workers": live_max_workers,
        "execution_mode": execution_mode,
    }
    write_jsonl(coverage_dir / "composition_facet_universes.jsonl", universes.values())
    write_jsonl(coverage_dir / "coverage_judge_results.jsonl", results)
    write_jsonl(coverage_dir / "accepted_query_coverages.jsonl", accepted_coverages)
    write_json(coverage_dir / "coverage_summary.json", summary)
    return summary


def _validate_coverage_payload(
    *,
    payload: dict[str, object],
    facet_library: dict[str, tuple[str, ...] | list[str]],
    target_id: str,
) -> None:
    signature_label = require_str(payload["signature_label"], context=f"coverage_judge {target_id}.signature_label")
    facet_labels = require_str_list(payload["facet_labels"], context=f"coverage_judge {target_id}.facet_labels")
    require_str(payload["coverage_rationale"], context=f"coverage_judge {target_id}.coverage_rationale")
    if signature_label not in facet_library:
        raise LiveSchemaError(f"coverage_judge {target_id}.signature_label has unsupported label: {signature_label}")
    universe = {str(facet) for facet in facet_library[signature_label]}
    for facet_label in facet_labels:
        if facet_label not in universe:
            raise LiveSchemaError(f"coverage_judge {target_id}.facet_labels has unsupported label: {facet_label}")
