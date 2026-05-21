from __future__ import annotations

from random import Random
from statistics import mean
from typing import Iterable


def summarize_fast_pilot_ablation(
    *,
    candidates: list[dict[str, object]],
    selected: list[dict[str, object]],
    budgets: list[int] | None = None,
    random_seed_count: int = 200,
) -> dict[str, object]:
    if not candidates:
        raise ValueError("candidates must not be empty")
    if not selected:
        raise ValueError("selected must not be empty")
    if random_seed_count < 1:
        raise ValueError("random_seed_count must be positive")

    selected_budget = len(selected)
    if budgets is None:
        budgets = _default_budgets(selected_budget)
    normalized_budgets = sorted({budget for budget in budgets if 0 < budget <= selected_budget})
    if not normalized_budgets:
        raise ValueError("budgets must include at least one positive value within selected size")

    resolved_selected = _resolve_selected_rows(candidates=candidates, selected=selected)
    full_metrics = _coverage_metrics(rows=resolved_selected, denominator=selected_budget)
    random_full_budget_samples = [
        _coverage_metrics(
            rows=_random_take(rows=candidates, budget=selected_budget, seed=seed),
            denominator=selected_budget,
        )
        for seed in range(random_seed_count)
    ]

    return {
        "candidate_count": len(candidates),
        "selected_count": selected_budget,
        "candidate_cell_count": len(_cells(candidates)),
        "candidate_pattern_coverage": len(_patterns(candidates)),
        "candidate_target_facet_coverage": len(_facets(candidates)),
        "candidate_unique_clause_set_count": len(_clause_sets(candidates)),
        "candidate_contract_valid_rate": _contract_valid_rate(candidates),
        "random_seed_count": random_seed_count,
        "variants": [
            {
                "ablation_id": "full_copal",
                "uses_interaction_filter": True,
                "uses_facet_driven_synthesis": True,
                "uses_coverage_aware_selection": True,
                **full_metrics,
            },
            _not_run_variant(
                ablation_id="without_interaction_filter",
                uses_interaction_filter=False,
                uses_facet_driven_synthesis=True,
                uses_coverage_aware_selection=True,
                requires_artifact="independent generation run without trigger/scope interaction filtering, followed by interaction-validity judging",
            ),
            {
                "ablation_id": "without_facet_driven_synthesis",
                "uses_interaction_filter": True,
                "uses_facet_driven_synthesis": False,
                "uses_coverage_aware_selection": True,
                "status": "not_run",
                "reportable": False,
                "requires_artifact": (
                    "independent no-facet generation run with post-hoc relation-pattern and target-facet mapping; "
                    "dropping COPAL facet labels from full outputs is not a valid ablation"
                ),
            },
            {
                "ablation_id": "without_coverage_aware_selection",
                "uses_interaction_filter": True,
                "uses_facet_driven_synthesis": True,
                "uses_coverage_aware_selection": False,
                **_mean_random_metrics(random_full_budget_samples),
            },
        ],
        "coverage_curve": [
            _coverage_curve_row(
                candidates=candidates,
                selected=resolved_selected,
                budget=budget,
                random_seed_count=random_seed_count,
            )
            for budget in normalized_budgets
        ],
    }


def _default_budgets(selected_budget: int) -> list[int]:
    return sorted({budget for budget in (4, 6, 8, 10, 12, selected_budget) if 0 < budget <= selected_budget})


def _coverage_curve_row(
    *,
    candidates: list[dict[str, object]],
    selected: list[dict[str, object]],
    budget: int,
    random_seed_count: int,
) -> dict[str, object]:
    full_rows = selected[:budget]
    random_samples = [
        _coverage_metrics(
            rows=_random_take(rows=candidates, budget=budget, seed=seed),
            denominator=budget,
        )
        for seed in range(random_seed_count)
    ]
    random_cell_counts = [float(sample["cell_count"]) for sample in random_samples]
    random_clause_set_counts = [float(sample["unique_clause_set_count"]) for sample in random_samples]
    full_metrics = _coverage_metrics(rows=full_rows, denominator=budget)
    return {
        "budget": budget,
        "full_cell_count": full_metrics["cell_count"],
        "full_pattern_coverage": full_metrics["pattern_coverage"],
        "full_target_facet_coverage": full_metrics["target_facet_coverage"],
        "full_cpq": full_metrics["cpq"],
        "full_unique_clause_set_count": full_metrics["unique_clause_set_count"],
        "full_clause_set_diversity": full_metrics["clause_set_diversity"],
        "full_contract_valid_rate": full_metrics["contract_valid_rate"],
        "random_mean_cell_count": mean(random_cell_counts),
        "random_min_cell_count": min(random_cell_counts),
        "random_max_cell_count": max(random_cell_counts),
        "random_mean_cpq": mean(float(sample["cpq"]) for sample in random_samples),
        "random_mean_unique_clause_set_count": mean(random_clause_set_counts),
        "random_mean_clause_set_diversity": mean(float(sample["clause_set_diversity"]) for sample in random_samples),
        "random_mean_contract_valid_rate": mean(float(sample["contract_valid_rate"]) for sample in random_samples),
    }


def _coverage_metrics(*, rows: list[dict[str, object]], denominator: int) -> dict[str, object]:
    cell_labels = _cells(rows)
    clause_sets = _clause_sets(rows)
    return {
        "status": "computed",
        "reportable": True,
        "candidate_count": denominator,
        "valid_count": len(rows),
        "vir": 1.0 if denominator else 0.0,
        "contract_valid_rate": _contract_valid_rate(rows),
        "pattern_coverage": len(_patterns(rows)),
        "target_facet_coverage": len(_facets(rows)),
        "cell_count": len(cell_labels),
        "cells": sorted(cell_labels),
        "cpq": (len(cell_labels) / denominator) if denominator else 0.0,
        "unique_clause_set_count": len(clause_sets),
        "clause_set_diversity": (len(clause_sets) / denominator) if denominator else 0.0,
    }


def _mean_random_metrics(samples: list[dict[str, object]]) -> dict[str, object]:
    return {
        "status": "computed",
        "reportable": True,
        "candidate_count": int(samples[0]["candidate_count"]),
        "valid_count": mean(float(sample["valid_count"]) for sample in samples),
        "vir": mean(float(sample["vir"]) for sample in samples),
        "contract_valid_rate": mean(float(sample["contract_valid_rate"]) for sample in samples),
        "pattern_coverage": mean(float(sample["pattern_coverage"]) for sample in samples),
        "target_facet_coverage": mean(float(sample["target_facet_coverage"]) for sample in samples),
        "cell_count": mean(float(sample["cell_count"]) for sample in samples),
        "mean_cell_count": mean(float(sample["cell_count"]) for sample in samples),
        "cpq": mean(float(sample["cpq"]) for sample in samples),
        "unique_clause_set_count": mean(float(sample["unique_clause_set_count"]) for sample in samples),
        "mean_unique_clause_set_count": mean(float(sample["unique_clause_set_count"]) for sample in samples),
        "clause_set_diversity": mean(float(sample["clause_set_diversity"]) for sample in samples),
    }


def _random_take(*, rows: list[dict[str, object]], budget: int, seed: int) -> list[dict[str, object]]:
    shuffled = list(rows)
    Random(seed).shuffle(shuffled)
    return shuffled[:budget]


def _resolve_selected_rows(
    *, candidates: list[dict[str, object]], selected: list[dict[str, object]]
) -> list[dict[str, object]]:
    by_id = {str(row.get("query_id", "")): row for row in candidates if str(row.get("query_id", ""))}
    resolved: list[dict[str, object]] = []
    for selected_row in selected:
        candidate = by_id.get(str(selected_row.get("query_id", "")), {})
        resolved.append({**candidate, **selected_row})
    return resolved


def _not_run_variant(
    *,
    ablation_id: str,
    uses_interaction_filter: bool,
    uses_facet_driven_synthesis: bool,
    uses_coverage_aware_selection: bool,
    requires_artifact: str,
) -> dict[str, object]:
    return {
        "ablation_id": ablation_id,
        "uses_interaction_filter": uses_interaction_filter,
        "uses_facet_driven_synthesis": uses_facet_driven_synthesis,
        "uses_coverage_aware_selection": uses_coverage_aware_selection,
        "status": "not_run",
        "reportable": False,
        "requires_artifact": requires_artifact,
    }


def _cells(rows: Iterable[dict[str, object]]) -> set[str]:
    cells: set[str] = set()
    for row in rows:
        pattern = _primary_pattern(row)
        if not pattern:
            continue
        for facet in _row_facets(row):
            cells.add(f"{pattern}::{facet}")
    return cells


def _patterns(rows: Iterable[dict[str, object]]) -> set[str]:
    return {pattern for row in rows for pattern in [_primary_pattern(row)] if pattern}


def _facets(rows: Iterable[dict[str, object]]) -> set[str]:
    return {facet for row in rows for facet in _row_facets(row)}


def _clause_sets(rows: Iterable[dict[str, object]]) -> set[str]:
    return {clause_set for row in rows for clause_set in [_clause_set_key(row)] if clause_set}


def _clause_set_key(row: dict[str, object]) -> str:
    raw_clause_ids = row.get("active_clause_ids") or row.get("clause_ids")
    if not raw_clause_ids and isinstance(row.get("composition"), dict):
        raw_clause_ids = row["composition"].get("clause_ids")  # type: ignore[index]
    if not raw_clause_ids and isinstance(row.get("scenario"), dict):
        raw_clause_ids = row["scenario"].get("clause_ids")  # type: ignore[index]
    if not isinstance(raw_clause_ids, list):
        return ""
    clause_ids = sorted({str(clause_id) for clause_id in raw_clause_ids if str(clause_id)})
    return "+".join(clause_ids)


def _contract_valid_rate(rows: list[dict[str, object]]) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if _has_strict_response_contract(row)) / len(rows)


def _has_strict_response_contract(row: dict[str, object]) -> bool:
    text_fields = [
        "allowed_answer_anchor",
        "forbidden_outcome",
        "required_gate_or_route",
        "trap_mechanism",
    ]
    for field in text_fields:
        if not str(_row_value(row, field)).strip():
            return False
    expected = _row_value(row, "expected_composed_handling")
    if not isinstance(expected, dict):
        return False
    return bool(expected.get("must_do")) and bool(expected.get("must_not_do"))


def _row_value(row: dict[str, object], key: str) -> object:
    if key in row:
        return row[key]
    scenario = row.get("scenario")
    if isinstance(scenario, dict) and key in scenario:
        return scenario[key]
    return ""


def _primary_pattern(row: dict[str, object]) -> str:
    if str(row.get("relation_pattern", "")):
        return str(row["relation_pattern"])
    raw_patterns = row.get("relation_patterns", [])
    patterns = [str(pattern) for pattern in raw_patterns if str(pattern)]
    return patterns[0] if patterns else ""


def _row_facets(row: dict[str, object]) -> list[str]:
    raw_facets = [
        *list(row.get("target_facets", [])),
        *list(row.get("coverage_set", [])),
    ]
    if not raw_facets and str(row.get("target_facet", "")):
        raw_facets = [str(row["target_facet"])]
    return sorted({str(facet) for facet in raw_facets if str(facet)})
