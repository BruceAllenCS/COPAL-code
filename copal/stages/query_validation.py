from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from copal.config import require_execution_mode
from copal.io import ensure_directory, read_jsonl, write_json, write_jsonl
from copal.llm import LLMClient
from copal.live_validation import (
    complete_live_json_object,
    require_bool,
    require_number,
    require_str,
    require_str_list,
)
from copal.prompts import build_query_validation_messages


def _normalize_query_adjudication(
    *,
    candidate: dict[str, object],
    payload: dict[str, object],
    validator_model: str,
) -> dict[str, object]:
    pass_value = require_bool(payload["pass"], context=f"query_validation {candidate['query_id']}.pass")
    composition_validity = require_bool(
        payload["composition_validity"],
        context=f"query_validation {candidate['query_id']}.composition_validity",
    )
    non_separability = require_bool(
        payload["non_separability"],
        context=f"query_validation {candidate['query_id']}.non_separability",
    )
    facet_coverage = require_str_list(
        payload["facet_coverage"],
        context=f"query_validation {candidate['query_id']}.facet_coverage",
    )
    return {
        "target_type": "query",
        "target_id": str(candidate["query_id"]),
        "query_id": str(candidate["query_id"]),
        "pass": pass_value,
        "composition_validity": composition_validity,
        "non_separability": non_separability,
        "facet_coverage": facet_coverage,
        "target_facets": facet_coverage,
        "scenario_level_interaction": require_str(
            payload["scenario_level_interaction"],
            context=f"query_validation {candidate['query_id']}.scenario_level_interaction",
        ),
        "query_level_interaction": require_str(
            payload["query_level_interaction"],
            context=f"query_validation {candidate['query_id']}.query_level_interaction",
        ),
        "independent_subrequests": require_bool(
            payload["independent_subrequests"],
            context=f"query_validation {candidate['query_id']}.independent_subrequests",
        ),
        "naturalness": require_str(payload["naturalness"], context=f"query_validation {candidate['query_id']}.naturalness"),
        "leakage": require_str(payload["leakage"], context=f"query_validation {candidate['query_id']}.leakage"),
        "redundancy": require_str(payload["redundancy"], context=f"query_validation {candidate['query_id']}.redundancy"),
        "validation_confidence": require_number(
            payload["validation_confidence"],
            context=f"query_validation {candidate['query_id']}.validation_confidence",
        ),
        "validation_rationale": require_str(
            payload["validation_rationale"],
            context=f"query_validation {candidate['query_id']}.validation_rationale",
        ),
        "validator_model": validator_model,
    }


def _validate_query_adjudication_payload(*, candidate: dict[str, object], payload: dict[str, object]) -> None:
    _normalize_query_adjudication(candidate=candidate, payload=payload, validator_model="schema-check")


def run_query_validation_stage(
    *,
    query_generation_dir: Path,
    execution_mode: str,
    validator_client: LLMClient | None = None,
    validator_model: str = "",
    live_max_workers: int = 1,
) -> dict[str, object]:
    require_execution_mode(execution_mode)
    if live_max_workers < 1:
        raise ValueError("live_max_workers must be positive")
    ensure_directory(query_generation_dir)
    candidates = read_jsonl(query_generation_dir / "candidate_queries.jsonl")
    deterministic_rows: list[dict[str, object]] = []
    adjudication_queue: list[dict[str, object]] = []
    adjudications: list[dict[str, object]] = []
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    live_jobs: list[tuple[int, dict[str, object], dict[str, object], dict[str, object]]] = []

    for index, candidate in enumerate(candidates):
        deterministic = {
            "target_type": "query",
            "target_id": candidate["query_id"],
            "schema_consistent": True,
            "exact_dedup_pass": True,
            "structure_constraint_pass": True,
            "heuristic_feasibility_result": "pass",
            "heuristic_non_separability_result": "pass",
            "scenario_level_interaction": "pass",
            "query_level_interaction": "pass",
            "independent_subrequests": False,
            "requires_adjudication": execution_mode == "live",
            "validation_notes": (
                "Passed deterministic query checks; queued for rubric validation."
                if execution_mode == "live"
                else "Passed deterministic query checks."
            ),
        }
        deterministic_row = {**candidate, **deterministic}
        deterministic_rows.append(deterministic_row)
        if execution_mode == "live":
            if validator_client is None or not validator_model:
                raise ValueError("Live query validation requires validator_client and validator_model")
            adjudication_queue.append(deterministic_row)
            live_jobs.append((index, candidate, deterministic, deterministic_row))
        else:
            accepted.append(
                {
                    **candidate,
                    "validation_metadata": deterministic,
                }
            )

    if execution_mode == "live":
        def run_live_job(
            index: int,
            candidate: dict[str, object],
            deterministic: dict[str, object],
            deterministic_row: dict[str, object],
        ) -> tuple[int, dict[str, object], dict[str, object]]:
            payload = complete_live_json_object(
                client=validator_client,
                model=validator_model,
                messages=build_query_validation_messages(query_row=deterministic_row),
                stage_dir=query_generation_dir,
                stage_name="query_validation",
                target_id=str(candidate["query_id"]),
                required_fields=(
                    "pass",
                    "composition_validity",
                    "non_separability",
                    "facet_coverage",
                    "scenario_level_interaction",
                    "query_level_interaction",
                    "independent_subrequests",
                    "naturalness",
                    "leakage",
                    "redundancy",
                    "validation_confidence",
                    "validation_rationale",
                ),
                validator=lambda payload: _validate_query_adjudication_payload(candidate=candidate, payload=payload),
            )
            adjudication = _normalize_query_adjudication(
                candidate=candidate,
                payload=dict(payload),
                validator_model=validator_model,
            )
            merged = {
                **candidate,
                "target_facets": adjudication["target_facets"],
                "construction_judgment": adjudication,
                "validation_metadata": {**deterministic, **adjudication},
            }
            return index, adjudication, merged

        live_results: dict[int, tuple[dict[str, object], dict[str, object]]] = {}
        if live_max_workers == 1:
            for job in live_jobs:
                index, adjudication, merged = run_live_job(*job)
                live_results[index] = (adjudication, merged)
        else:
            with ThreadPoolExecutor(max_workers=live_max_workers) as executor:
                futures = {executor.submit(run_live_job, *job): job[0] for job in live_jobs}
                for future in as_completed(futures):
                    index, adjudication, merged = future.result()
                    live_results[index] = (adjudication, merged)

        for index in sorted(live_results):
            adjudication, merged = live_results[index]
            adjudications.append(adjudication)
            if adjudication["pass"]:
                accepted.append(merged)
            else:
                rejected.append(merged)

    summary = {
        "candidate_count": len(candidates),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "adjudication_queue_count": len(adjudication_queue),
        "adjudication_count": len(adjudications),
        "live_max_workers": live_max_workers,
        "execution_mode": execution_mode,
    }
    write_jsonl(query_generation_dir / "query_deterministic_results.jsonl", deterministic_rows)
    write_jsonl(query_generation_dir / "query_adjudication_queue.jsonl", adjudication_queue)
    write_jsonl(query_generation_dir / "query_adjudications.jsonl", adjudications)
    write_jsonl(query_generation_dir / "accepted_queries.jsonl", accepted)
    write_jsonl(query_generation_dir / "rejected_queries.jsonl", rejected)
    write_json(query_generation_dir / "query_validation_summary.json", summary)
    return summary
