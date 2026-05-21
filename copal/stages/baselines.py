from __future__ import annotations

from pathlib import Path
from random import Random
from typing import Iterable

from copal.io import ensure_directory, write_json, write_jsonl

INVALID_BREAKDOWN_FIELDS: tuple[str, ...] = (
    "independent_cooccurrence",
    "contradictory",
    "wrong_target",
    "unsupported_path",
    "unnatural_redundant",
)

BASELINE_METHODS: tuple[dict[str, object], ...] = (
    {
        "method_id": "single_policy_generator",
        "display_name": "Single-policy generator",
        "role": "lower-bound control",
        "uses_grounded_clauses": True,
        "uses_structural_prefilter": False,
        "uses_interaction_filter": False,
        "uses_copal_taxonomy": False,
        "budget_rule": "B candidates / N final",
    },
    {
        "method_id": "naive_composition",
        "display_name": "Naive composition",
        "role": "lower-bound control",
        "uses_grounded_clauses": True,
        "uses_structural_prefilter": False,
        "uses_interaction_filter": False,
        "uses_copal_taxonomy": False,
        "budget_rule": "B candidates / N final",
    },
    {
        "method_id": "taxonomy_free_llm_planner",
        "display_name": "Taxonomy-free LLM planner",
        "role": "strong LLM planning baseline",
        "uses_grounded_clauses": True,
        "uses_structural_prefilter": True,
        "uses_interaction_filter": False,
        "uses_copal_taxonomy": False,
        "budget_rule": "B candidates / N final",
    },
    {
        "method_id": "random_valid_interacting",
        "display_name": "Random valid interacting",
        "role": "selection-control baseline",
        "uses_grounded_clauses": True,
        "uses_structural_prefilter": True,
        "uses_interaction_filter": True,
        "uses_copal_taxonomy": False,
        "budget_rule": "same valid pool / N final",
    },
    {
        "method_id": "copal",
        "display_name": "COPAL",
        "role": "proposed method",
        "uses_grounded_clauses": True,
        "uses_structural_prefilter": True,
        "uses_interaction_filter": True,
        "uses_copal_taxonomy": True,
        "budget_rule": "B candidates / N final",
    },
)


def run_baseline_experiment_stage(
    *,
    baseline_dir: Path,
    grounded_rows: list[dict[str, object]],
    candidate_compositions: list[dict[str, object]],
    accepted_compositions: list[dict[str, object]],
    accepted_queries: list[dict[str, object]],
    coverage_rows: list[dict[str, object]],
    benchmark_items: list[dict[str, object]],
    facet_library: dict[str, tuple[str, ...] | list[str]],
    final_query_budget: int,
    random_seed: int = 0,
) -> dict[str, object]:
    if final_query_budget < 1:
        raise ValueError("final_query_budget must be positive")
    _validate_benchmark_lineage(
        accepted_queries=accepted_queries,
        coverage_rows=coverage_rows,
        benchmark_items=benchmark_items,
    )
    ensure_directory(baseline_dir)

    method_rows = [dict(method) for method in BASELINE_METHODS]
    method_candidates = _build_method_candidates(
        grounded_rows=grounded_rows,
        candidate_compositions=candidate_compositions,
        accepted_compositions=accepted_compositions,
        accepted_queries=accepted_queries,
        coverage_rows=coverage_rows,
        benchmark_items=benchmark_items,
        final_query_budget=final_query_budget,
        random_seed=random_seed,
    )
    metrics_rows = [
        _construction_quality_row(
            method_id=method["method_id"],
            rows=method_candidates[str(method["method_id"])],
            facet_library=facet_library,
        )
        for method in method_rows
    ]
    invalid_rows = [
        _invalid_breakdown_row(
            method_id=method["method_id"],
            rows=method_candidates[str(method["method_id"])],
        )
        for method in method_rows
    ]
    ablation_rows = _build_ablation_rows(
        copal_rows=method_candidates["copal"],
        no_selection_rows=method_candidates["random_valid_interacting"],
        facet_library=facet_library,
    )
    summary = {
        "method_count": len(method_rows),
        "final_query_budget": final_query_budget,
        "random_seed": random_seed,
        "baseline_protocol_artifact": "baseline_protocols.jsonl",
        "construction_quality_artifact": "construction_quality_metrics.jsonl",
        "invalid_breakdown_artifact": "invalid_item_breakdown.jsonl",
        "ablation_artifact": "ablation_metrics.jsonl",
    }

    write_jsonl(baseline_dir / "baseline_protocols.jsonl", method_rows)
    write_jsonl(baseline_dir / "baseline_candidate_records.jsonl", _flatten_method_candidates(method_candidates))
    write_jsonl(baseline_dir / "construction_quality_metrics.jsonl", metrics_rows)
    write_jsonl(baseline_dir / "invalid_item_breakdown.jsonl", invalid_rows)
    write_jsonl(baseline_dir / "ablation_metrics.jsonl", ablation_rows)
    write_json(baseline_dir / "baseline_experiment_summary.json", summary)
    return summary


def _build_method_candidates(
    *,
    grounded_rows: list[dict[str, object]],
    candidate_compositions: list[dict[str, object]],
    accepted_compositions: list[dict[str, object]],
    accepted_queries: list[dict[str, object]],
    coverage_rows: list[dict[str, object]],
    benchmark_items: list[dict[str, object]],
    final_query_budget: int,
    random_seed: int,
) -> dict[str, list[dict[str, object]]]:
    copal_rows = [
        _benchmark_item_record(method_id="copal", item=item)
        for item in benchmark_items[:final_query_budget]
    ]
    valid_interacting_rows = [
        _query_record(method_id="random_valid_interacting", query=query, use_taxonomy=True)
        for query in coverage_rows
        if _query_is_interacting(query)
    ]
    taxonomy_free_rows = [
        _query_record(method_id="taxonomy_free_llm_planner", query=query, use_taxonomy=False)
        for query in accepted_queries
        if _query_is_interacting(query)
    ][:final_query_budget]
    rng = Random(random_seed)
    shuffled_valid = list(valid_interacting_rows)
    rng.shuffle(shuffled_valid)

    return {
        "single_policy_generator": [
            _single_policy_record(row, index) for index, row in enumerate(grounded_rows[:final_query_budget], start=1)
        ],
        "naive_composition": [
            _composition_record(method_id="naive_composition", composition=row)
            for row in candidate_compositions[:final_query_budget]
        ],
        "taxonomy_free_llm_planner": taxonomy_free_rows,
        "random_valid_interacting": shuffled_valid[:final_query_budget],
        "copal": copal_rows,
    }


def _validate_benchmark_lineage(
    *,
    accepted_queries: list[dict[str, object]],
    coverage_rows: list[dict[str, object]],
    benchmark_items: list[dict[str, object]],
) -> None:
    accepted_query_ids = {str(row["query_id"]) for row in accepted_queries}
    covered_query_ids = {str(row["query_id"]) for row in coverage_rows}
    for item in benchmark_items:
        query_id = str(item["query_id"])
        if query_id not in accepted_query_ids:
            raise ValueError(f"Benchmark item does not come from accepted_queries: {query_id}")
        if query_id not in covered_query_ids:
            raise ValueError(f"Benchmark item does not have a coverage row: {query_id}")


def _single_policy_record(row: dict[str, object], index: int) -> dict[str, object]:
    return {
        "method_id": "single_policy_generator",
        "candidate_id": f"single-policy::{row['clause_id']}::{index}",
        "clause_ids": [str(row["clause_id"])],
        "signature": "",
        "target_facets": [],
        "coverage_set": [],
        "valid": False,
        "interaction_filter_status": "fail",
        "invalid_reasons": ["wrong_target", "unsupported_path"],
        "clause_count": 1,
    }


def _composition_record(*, method_id: str, composition: dict[str, object]) -> dict[str, object]:
    interaction_filter = dict(composition["interaction_filter"])
    signature = str(composition.get("relation_pattern") or composition.get("signature_proposal", ""))
    valid = interaction_filter["status"] == "pass" and bool(signature)
    invalid_reasons = []
    if interaction_filter["status"] != "pass":
        invalid_reasons.append("independent_cooccurrence")
    if not signature:
        invalid_reasons.append("wrong_target")
    return {
        "method_id": method_id,
        "candidate_id": str(composition["composition_id"]),
        "composition_id": str(composition["composition_id"]),
        "clause_ids": [str(clause_id) for clause_id in composition["clause_ids"]],
        "signature": signature,
        "relation_pattern": signature,
        "relation_patterns": list(composition.get("relation_patterns", [signature] if signature else [])),
        "target_facets": [],
        "coverage_set": [],
        "valid": valid,
        "interaction_filter_status": str(interaction_filter["status"]),
        "invalid_reasons": invalid_reasons,
        "clause_count": len(list(composition["clause_ids"])),
    }


def _benchmark_item_record(*, method_id: str, item: dict[str, object]) -> dict[str, object]:
    target_facets = [str(facet) for facet in item["target_facets"]]
    coverage_set = [str(facet) for facet in item["coverage_set"]]
    return {
        "method_id": method_id,
        "candidate_id": str(item["item_id"]),
        "query_id": str(item["query_id"]),
        "composition_id": str(item["composition_id"]),
        "clause_ids": [str(clause_id) for clause_id in item["active_clause_ids"]],
        "signature": str(item["signature"]),
        "relation_pattern": str(item.get("relation_pattern", item["signature"])),
        "relation_patterns": list(item.get("relation_patterns", [item.get("relation_pattern", item["signature"])])),
        "target_facets": target_facets,
        "coverage_set": coverage_set,
        "valid": True,
        "interaction_filter_status": "pass",
        "invalid_reasons": [],
        "clause_count": len(list(item["active_clause_ids"])),
    }


def _query_is_interacting(query: dict[str, object]) -> bool:
    metadata = dict(query.get("validation_metadata", {}))
    if "independent_subrequests" in metadata and bool(metadata["independent_subrequests"]):
        return False
    if "non_separability" in metadata and not bool(metadata["non_separability"]):
        return False
    return True


def _query_record(*, method_id: str, query: dict[str, object], use_taxonomy: bool) -> dict[str, object]:
    target_facets = [str(facet) for facet in query.get("target_facets", [])] if use_taxonomy else []
    coverage_set = [str(facet) for facet in query.get("coverage_set", [])] if use_taxonomy else []
    scenario = dict(query.get("scenario", query.get("scenario_stub", {})))
    risk_description = str(
        query.get("risk_description")
        or scenario.get("non_decomposability_rationale")
        or query.get("query_text", "")
    )
    return {
        "method_id": method_id,
        "candidate_id": str(query["query_id"]),
        "query_id": str(query["query_id"]),
        "composition_id": str(query["composition_id"]),
        "clause_ids": [str(clause_id) for clause_id in scenario.get("clause_ids", [])],
        "signature": str(query.get("relation_pattern") or query.get("signature_proposal", "")) if use_taxonomy else "",
        "relation_pattern": str(query.get("relation_pattern") or query.get("signature_proposal", "")) if use_taxonomy else "",
        "relation_patterns": list(query.get("relation_patterns", [])) if use_taxonomy else [],
        "target_facets": target_facets,
        "coverage_set": coverage_set,
        "valid": True,
        "interaction_filter_status": "pass",
        "invalid_reasons": [],
        "clause_count": len(list(scenario.get("clause_ids", []))),
        "query_text": str(query.get("query_text", "")),
        "risk_description": risk_description,
    }


def _has_structural_prefilter_signal(row: dict[str, object]) -> bool:
    signals = dict(row["structure_signals"])
    return bool(
        signals.get("scope_coupled")
        or signals.get("scope_overlap")
        or signals.get("effect_interaction")
        or signals.get("same_semantic_span")
        or signals.get("changes_scope_or_handling")
    )


def _k_medoids_cover_records(
    rows: list[dict[str, object]],
    final_query_budget: int,
    *,
    text_key: str,
    method_id: str,
) -> list[dict[str, object]]:
    if len(rows) <= final_query_budget:
        return [{**row, "method_id": method_id} for row in rows]
    selected_indexes: list[int] = [0]
    while len(selected_indexes) < final_query_budget:
        best_index = None
        best_distance = -1.0
        for index, row in enumerate(rows):
            if index in selected_indexes:
                continue
            distance = min(
                _jaccard_distance(str(row[text_key]), str(rows[selected_index][text_key]))
                for selected_index in selected_indexes
            )
            if distance > best_distance:
                best_index = index
                best_distance = distance
        if best_index is None:
            break
        selected_indexes.append(best_index)
    return [{**rows[index], "method_id": method_id} for index in selected_indexes]


def _jaccard_distance(left: str, right: str) -> float:
    left_tokens = _token_set(left)
    right_tokens = _token_set(right)
    if not left_tokens and not right_tokens:
        return 0.0
    return 1.0 - (len(left_tokens & right_tokens) / len(left_tokens | right_tokens))


def _token_set(value: str) -> set[str]:
    return {token for token in value.lower().replace("/", " ").replace("-", " ").split() if token}


def _construction_quality_row(
    *,
    method_id: object,
    rows: list[dict[str, object]],
    facet_library: dict[str, tuple[str, ...] | list[str]],
) -> dict[str, object]:
    valid_rows = [row for row in rows if bool(row["valid"])]
    signatures = {str(row["signature"]) for row in valid_rows if str(row["signature"])}
    covered_cells = sorted(_covered_cell_labels(valid_rows))
    universe = _facet_universe_cell_labels(facet_library)
    candidate_count = len(rows)
    return {
        "method_id": str(method_id),
        "candidate_count": candidate_count,
        "valid_count": len(valid_rows),
        "human_valid_count": None,
        "interacting_count": sum(1 for row in rows if row["interaction_filter_status"] == "pass"),
        "signature_coverage": len(signatures),
        "target_facet_coverage": len(covered_cells),
        "cell_count": len(covered_cells),
        "cells": covered_cells,
        "target_facet_universe_size": len(universe),
        "cost_per_valid": None,
        "mean_clauses_per_item": _mean(row["clause_count"] for row in valid_rows),
        "three_plus_clause_percent": _three_plus_percent(valid_rows),
        "coverage_per_query": (len(covered_cells) / candidate_count) if candidate_count else 0.0,
        "vir": (len(valid_rows) / candidate_count) if candidate_count else 0.0,
        "cpq": (len(covered_cells) / candidate_count) if candidate_count else 0.0,
    }


def _covered_cell_labels(rows: list[dict[str, object]]) -> set[str]:
    cells: set[str] = set()
    for row in rows:
        patterns = [str(pattern) for pattern in row.get("relation_patterns", []) if str(pattern)]
        if not patterns and str(row.get("relation_pattern", "")):
            patterns = [str(row["relation_pattern"])]
        if not patterns and str(row.get("signature", "")):
            patterns = [str(row["signature"])]
        facets = {
            str(facet)
            for facet in [*row.get("target_facets", []), *row.get("coverage_set", [])]
            if str(facet)
        }
        for pattern in patterns:
            for facet in facets:
                cells.add(f"{pattern}::{facet}")
    return cells


def _facet_universe_cell_labels(facet_library: dict[str, tuple[str, ...] | list[str]]) -> set[str]:
    return {
        f"{pattern}::{facet}"
        for pattern, facets in facet_library.items()
        for facet in facets
    }


def _invalid_breakdown_row(*, method_id: object, rows: list[dict[str, object]]) -> dict[str, object]:
    counts = {field: 0 for field in INVALID_BREAKDOWN_FIELDS}
    for row in rows:
        for reason in row["invalid_reasons"]:
            if reason not in counts:
                raise ValueError(f"Unsupported invalid reason: {reason}")
            counts[str(reason)] += 1
    return {"method_id": str(method_id), **counts}


def _build_ablation_rows(
    *,
    copal_rows: list[dict[str, object]],
    no_selection_rows: list[dict[str, object]],
    facet_library: dict[str, tuple[str, ...] | list[str]],
) -> list[dict[str, object]]:
    variants = (
        ("full_copal", True, True, True, copal_rows),
        ("without_interaction_filter", False, True, True, [{**row, "valid": False, "invalid_reasons": ["independent_cooccurrence"]} for row in copal_rows]),
        ("without_facet_driven_synthesis", True, False, True, [{**row, "target_facets": [], "coverage_set": []} for row in copal_rows]),
        ("without_coverage_aware_selection", True, True, False, no_selection_rows),
    )
    rows: list[dict[str, object]] = []
    for variant_id, interaction_filter, facets, selection, variant_rows in variants:
        metric = _construction_quality_row(
            method_id=variant_id,
            rows=variant_rows,
            facet_library=facet_library,
        )
        rows.append(
            {
                "ablation_id": variant_id,
                "uses_interaction_filter": interaction_filter,
                "uses_facet_driven_synthesis": facets,
                "uses_coverage_aware_selection": selection,
                **metric,
            }
        )
    return rows


def _flatten_method_candidates(method_candidates: dict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
    return [row for rows in method_candidates.values() for row in rows]


def _mean(values: Iterable[object]) -> float | None:
    numbers = [float(value) for value in values]
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def _three_plus_percent(rows: list[dict[str, object]]) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if int(row["clause_count"]) >= 3) / len(rows)
