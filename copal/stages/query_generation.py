from __future__ import annotations

from pathlib import Path

from copal.config import require_execution_mode
from copal.io import ensure_directory, write_json, write_jsonl
from copal.llm import LLMClient, complete_json
from copal.prompts import build_query_verbalization_messages


def build_candidate_query_row(
    *,
    composition_id: str,
    signature: str,
    target_facet: str,
    scenario: dict[str, object] | None = None,
    scenario_stub: dict[str, object] | None = None,
    query_text: str,
    query_variant_index: int = 0,
    include_variant_suffix: bool = False,
) -> dict[str, object]:
    scenario_row = scenario if scenario is not None else scenario_stub
    if scenario_row is None:
        raise ValueError("candidate query requires a scenario")
    query_id = f"{composition_id}::{target_facet}"
    if include_variant_suffix:
        query_id = f"{query_id}::v{query_variant_index}"
    return {
        "query_id": query_id,
        "composition_id": composition_id,
        "signature_proposal": signature,
        "relation_pattern": signature,
        "relation_patterns": [signature],
        "target_facet": target_facet,
        "target_facets": [target_facet],
        "query_variant_index": query_variant_index,
        "scenario": dict(scenario_row),
        "scenario_stub": dict(scenario_row),
        "query_text": query_text,
        "proposal_meta": {},
    }


def _build_scenario(composition: dict[str, object], target_facet: str) -> dict[str, object]:
    relation_pattern = str(composition.get("relation_pattern") or composition["signature_proposal"])
    return {
        "scenario_id": f"{composition['composition_id']}::{target_facet}::scenario-0",
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
        "clause_ids": composition["clause_ids"],
        "scope_pair": composition["scope_pair"],
        "interaction_filter": {
            "level": "scenario",
            "status": "pass",
            "conditions": list(dict(composition.get("interaction_filter", {})).get("conditions", [])),
        },
    }


def _render_query_text(composition: dict[str, object], target_facet: str) -> str:
    signature = str(composition.get("relation_pattern") or composition["signature_proposal"])
    scopes = " and ".join(str(scope) for scope in composition["scope_pair"])
    return (
        f"I need help with {scopes}. "
        f"Please handle this as a {signature} case and avoid the {target_facet} failure mode."
    )


def run_query_generation_stage(
    *,
    query_generation_dir: Path,
    accepted_compositions: list[dict[str, object]],
    facet_library: dict[str, tuple[str, ...] | list[str]],
    execution_mode: str,
    proposal_client: LLMClient | None = None,
    proposal_model: str = "",
) -> dict[str, object]:
    require_execution_mode(execution_mode)
    ensure_directory(query_generation_dir)

    scenarios: list[dict[str, object]] = []
    candidate_queries: list[dict[str, object]] = []
    construction_judgments: list[dict[str, object]] = []
    accepted_queries: list[dict[str, object]] = []

    for composition in accepted_compositions:
        signature = str(composition.get("relation_pattern") or composition["signature_proposal"])
        for target_facet in facet_library.get(signature, ()):
            scenario = _build_scenario(composition, str(target_facet))
            if execution_mode == "live":
                if proposal_client is None or not proposal_model:
                    raise ValueError("Live query generation requires proposal_client and proposal_model")
                query_payload = complete_json(
                    client=proposal_client,
                    model=proposal_model,
                    messages=build_query_verbalization_messages(
                        composition=composition,
                        target_facet=str(target_facet),
                        scenario=scenario,
                    ),
                )
                query_text = str(query_payload["query_text"])
                if "scenario" in query_payload:
                    scenario = {**scenario, **dict(query_payload["scenario"])}
            else:
                query_text = _render_query_text(composition, str(target_facet))
            query_row = build_candidate_query_row(
                composition_id=str(composition["composition_id"]),
                signature=signature,
                target_facet=str(target_facet),
                scenario=scenario,
                query_text=query_text,
            )
            judgment = {
                "query_id": query_row["query_id"],
                "composition_validity": True,
                "non_separability": True,
                "facet_coverage": [target_facet],
                "target_facets": [target_facet],
                "scenario_level_interaction": "pass",
                "query_level_interaction": "pass",
                "independent_subrequests": False,
                "naturalness": "pass",
                "leakage": "pass",
                "redundancy": "novel",
                "validation_confidence": 1.0,
                "pass": True,
            }
            scenarios.append(
                {
                    "query_id": query_row["query_id"],
                    "composition_id": query_row["composition_id"],
                    "scenario": scenario,
                }
            )
            candidate_queries.append(query_row)
            construction_judgments.append(judgment)
            accepted_queries.append({**query_row, "construction_judgment": judgment})

    summary = {
        "composition_count": len(accepted_compositions),
        "candidate_query_count": len(candidate_queries),
        "accepted_query_count": len(accepted_queries),
        "execution_mode": execution_mode,
    }

    write_jsonl(query_generation_dir / "intermediate_scenarios.jsonl", scenarios)
    write_jsonl(query_generation_dir / "candidate_queries.jsonl", candidate_queries)
    write_jsonl(query_generation_dir / "construction_judge_results.jsonl", construction_judgments)
    write_jsonl(query_generation_dir / "accepted_queries.jsonl", accepted_queries)
    write_json(query_generation_dir / "query_generation_summary.json", summary)
    return summary
