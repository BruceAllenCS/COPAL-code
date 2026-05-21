from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from copal.config import require_execution_mode
from copal.io import ensure_directory, write_json, write_jsonl
from copal.llm import LLMClient
from copal.live_validation import LiveSchemaError, complete_live_json_object, require_fields, require_object, require_str, require_str_list
from copal.prompts import build_query_verbalization_messages
from copal.stages.query_generation import build_candidate_query_row


def _build_scenario(
    composition: dict[str, object],
    target_facet: str,
    *,
    query_variant_index: int = 0,
    include_variant_suffix: bool = False,
) -> dict[str, object]:
    scenario_id = f"{composition['composition_id']}::{target_facet}::scenario-0"
    if include_variant_suffix:
        scenario_id = f"{composition['composition_id']}::{target_facet}::v{query_variant_index}::scenario-0"
    relation_pattern = str(composition.get("relation_pattern") or composition["signature_proposal"])
    return {
        "scenario_id": scenario_id,
        "shared_user_scenario": f"A user request activates {', '.join(str(scope) for scope in composition['scope_pair'])}.",
        "jointly_satisfied_triggers": [str(trigger) for trigger in composition.get("trigger_set", [])],
        "coupled_scope_or_path": "; ".join(str(scope) for scope in composition["scope_pair"]),
        "non_decomposability_rationale": (
            "The clauses jointly shape one response boundary or handling path; independent answers cannot be recombined."
        ),
        "required_state_assumptions": [],
        "leakage_naturalness_self_check": "Natural wording without policy labels.",
        "signature": relation_pattern,
        "relation_pattern": relation_pattern,
        "relation_patterns": list(composition.get("relation_patterns", [relation_pattern])),
        "target_facet": target_facet,
        "target_facets": [target_facet],
        "query_variant_index": query_variant_index,
        "clause_ids": composition["clause_ids"],
        "scope_pair": composition["scope_pair"],
        "interaction_filter": {
            "level": "scenario",
            "status": "pass",
            "conditions": list(dict(composition.get("interaction_filter", {})).get("conditions", [])),
        },
    }


def _render_query_text(composition: dict[str, object], target_facet: str) -> str:
    scopes = " and ".join(str(scope) for scope in composition["scope_pair"])
    signature = str(composition.get("relation_pattern") or composition["signature_proposal"])
    return (
        f"I need help with {scopes}. "
        f"Please handle this as a {signature} case and avoid the {target_facet} failure mode."
    )


def _normalize_query_proposal_payload(*, payload: dict[str, object], context: str) -> dict[str, object]:
    require_fields(payload, ("scenario",), context=f"query_proposal {context}")
    scenario = require_object(payload["scenario"], context=f"query_proposal {context}.scenario")
    query_text_source = "query_text"
    if "query_text" in payload:
        query_text = require_str(payload["query_text"], context=f"query_proposal {context}.query_text")
    elif "query_text" in scenario:
        query_text = require_str(
            scenario["query_text"],
            context=f"query_proposal {context}.scenario.query_text",
        )
        query_text_source = "scenario.query_text"
        scenario = {key: value for key, value in scenario.items() if key != "query_text"}
    else:
        raise LiveSchemaError(f"query_proposal {context} missing required field: query_text")

    scenario_field_aliases: list[str] = []
    if "leakage_naturalness_self_check" not in scenario and "leakage_naturality_self_check" in scenario:
        scenario = {
            **{key: value for key, value in scenario.items() if key != "leakage_naturality_self_check"},
            "leakage_naturalness_self_check": scenario["leakage_naturality_self_check"],
        }
        scenario_field_aliases.append("leakage_naturality_self_check->leakage_naturalness_self_check")
    require_fields(
        scenario,
        (
            "shared_user_scenario",
            "jointly_satisfied_triggers",
            "coupled_scope_or_path",
            "non_decomposability_rationale",
            "required_state_assumptions",
            "leakage_naturalness_self_check",
        ),
        context=f"query_proposal {context}.scenario",
    )
    require_str(scenario["shared_user_scenario"], context=f"query_proposal {context}.scenario.shared_user_scenario")
    require_str_list(
        scenario["jointly_satisfied_triggers"],
        context=f"query_proposal {context}.scenario.jointly_satisfied_triggers",
    )
    require_str(scenario["coupled_scope_or_path"], context=f"query_proposal {context}.scenario.coupled_scope_or_path")
    require_str(
        scenario["non_decomposability_rationale"],
        context=f"query_proposal {context}.scenario.non_decomposability_rationale",
    )
    require_str_list(
        scenario["required_state_assumptions"],
        context=f"query_proposal {context}.scenario.required_state_assumptions",
    )
    require_str(
        scenario["leakage_naturalness_self_check"],
        context=f"query_proposal {context}.scenario.leakage_naturalness_self_check",
    )
    return {
        "query_text": query_text,
        "scenario": scenario,
        "query_text_source": query_text_source,
        "scenario_field_aliases": scenario_field_aliases,
    }


def _validate_scenario_payload(*, payload: dict[str, object], context: str) -> None:
    _normalize_query_proposal_payload(payload=payload, context=context)


def run_query_proposal_stage(
    *,
    query_generation_dir: Path,
    accepted_compositions: list[dict[str, object]],
    facet_library: dict[str, tuple[str, ...] | list[str]],
    execution_mode: str,
    proposal_client: LLMClient | None = None,
    proposal_model: str = "",
    query_variants_per_facet: int = 1,
    live_max_workers: int = 1,
) -> dict[str, object]:
    require_execution_mode(execution_mode)
    if query_variants_per_facet < 1:
        raise ValueError("query_variants_per_facet must be positive")
    if live_max_workers < 1:
        raise ValueError("live_max_workers must be positive")
    ensure_directory(query_generation_dir)
    include_variant_suffix = query_variants_per_facet > 1
    jobs: list[tuple[int, dict[str, object], str, str, int]] = []
    job_index = 0
    for composition in accepted_compositions:
        signature = str(composition.get("relation_pattern") or composition["signature_proposal"])
        for target_facet in facet_library.get(signature, ()):
            for query_variant_index in range(query_variants_per_facet):
                jobs.append((job_index, composition, signature, str(target_facet), query_variant_index))
                job_index += 1

    def run_job(
        index: int,
        composition: dict[str, object],
        signature: str,
        target_facet: str,
        query_variant_index: int,
    ) -> tuple[int, dict[str, object], dict[str, object]]:
        scenario = _build_scenario(
            composition,
            target_facet,
            query_variant_index=query_variant_index,
            include_variant_suffix=include_variant_suffix,
        )
        target_id = f"{composition['composition_id']}::{target_facet}"
        if include_variant_suffix:
            target_id = f"{target_id}::v{query_variant_index}"
        if execution_mode == "live":
            if proposal_client is None or not proposal_model:
                raise ValueError("Live query proposal requires proposal_client and proposal_model")
            payload = complete_live_json_object(
                client=proposal_client,
                model=proposal_model,
                messages=build_query_verbalization_messages(
                    composition=composition,
                    target_facet=target_facet,
                    scenario=scenario,
                    query_variant_index=query_variant_index,
                    query_variants_per_facet=query_variants_per_facet,
                ),
                stage_dir=query_generation_dir,
                stage_name="query_proposal",
                target_id=target_id,
                required_fields=("scenario",),
                validator=lambda payload, context=target_id: _validate_scenario_payload(
                    payload=payload,
                    context=context,
                ),
            )
            normalized_payload = _normalize_query_proposal_payload(payload=dict(payload), context=target_id)
            query_text = normalized_payload["query_text"]
            scenario = {**scenario, **dict(normalized_payload["scenario"])}
            proposal_meta = {
                "execution_mode": "live",
                "proposal_model": proposal_model,
                "query_variants_per_facet": query_variants_per_facet,
                "query_text_source": normalized_payload["query_text_source"],
                "scenario_field_aliases": normalized_payload["scenario_field_aliases"],
            }
        else:
            query_text = _render_query_text(composition, target_facet)
            proposal_meta = {
                "execution_mode": "deterministic",
                "query_variants_per_facet": query_variants_per_facet,
            }
        query = build_candidate_query_row(
            composition_id=str(composition["composition_id"]),
            signature=signature,
            target_facet=target_facet,
            scenario=scenario,
            query_text=query_text,
            query_variant_index=query_variant_index,
            include_variant_suffix=include_variant_suffix,
        )
        query["proposal_meta"] = proposal_meta
        scenario = {
            "query_id": query["query_id"],
            "composition_id": query["composition_id"],
            "scenario": scenario,
        }
        return index, scenario, query

    results: dict[int, tuple[dict[str, object], dict[str, object]]] = {}
    if execution_mode == "live" and live_max_workers > 1:
        with ThreadPoolExecutor(max_workers=live_max_workers) as executor:
            futures = {executor.submit(run_job, *job): job[0] for job in jobs}
            for future in as_completed(futures):
                index, scenario, query = future.result()
                results[index] = (scenario, query)
    else:
        for job in jobs:
            index, scenario, query = run_job(*job)
            results[index] = (scenario, query)

    scenarios: list[dict[str, object]] = []
    candidate_queries: list[dict[str, object]] = []
    for index in sorted(results):
        scenario, query = results[index]
        scenarios.append(scenario)
        candidate_queries.append(query)
    summary = {
        "scenario_count": len(scenarios),
        "candidate_query_count": len(candidate_queries),
        "query_variants_per_facet": query_variants_per_facet,
        "live_max_workers": live_max_workers,
        "execution_mode": execution_mode,
    }
    write_jsonl(query_generation_dir / "intermediate_scenarios.jsonl", scenarios)
    write_jsonl(query_generation_dir / "candidate_queries.jsonl", candidate_queries)
    write_json(query_generation_dir / "query_proposal_summary.json", summary)
    return summary
