import pytest

from copal.fast_pilot_ablation import summarize_fast_pilot_ablation
from scripts.run_fast_generation_pilot import aggregate_fast_ablation_summaries


def row(query_id: str, pattern: str, facet: str, clause_ids: list[str] | None = None) -> dict[str, object]:
    return {
        "query_id": query_id,
        "relation_pattern": pattern,
        "relation_patterns": [pattern],
        "target_facet": facet,
        "target_facets": [facet],
        "coverage_set": [facet],
        "active_clause_ids": clause_ids or [f"{query_id}-A", f"{query_id}-B"],
        "allowed_answer_anchor": "answer the permitted portion",
        "forbidden_outcome": "do not provide the prohibited portion",
        "required_gate_or_route": "apply the required gate or route",
        "trap_mechanism": "the composed request tempts a single-policy shortcut",
        "expected_composed_handling": {
            "must_do": ["answer permitted content"],
            "must_not_do": ["provide prohibited content"],
        },
    }


def test_fast_pilot_ablation_reports_full_facet_and_selection_effects() -> None:
    candidates = [
        row("a0", "scope-restriction", "boundary-overreach", ["C1", "C2"]),
        row("a1", "scope-restriction", "boundary-overreach", ["C1", "C2"]),
        row("b0", "scope-restriction", "over-refusal", ["C1", "C3"]),
        row("b1", "scope-restriction", "over-refusal", ["C1", "C3"]),
        row("c0", "workflow-transfer", "wrong-route", ["C4", "C5"]),
        row("c1", "workflow-transfer", "wrong-route", ["C4", "C5"]),
        row("d0", "workflow-transfer", "latent-continuation", ["C4", "C6"]),
        row("d1", "workflow-transfer", "latent-continuation", ["C4", "C6"]),
    ]
    selected = [{"query_id": candidates[index]["query_id"]} for index in [0, 2, 4, 6]]

    summary = summarize_fast_pilot_ablation(
        candidates=candidates,
        selected=selected,
        budgets=[2, 4],
        random_seed_count=50,
    )

    variants = {row["ablation_id"]: row for row in summary["variants"]}
    assert variants["full_copal"]["cell_count"] == 4
    assert variants["full_copal"]["target_facet_coverage"] == 4
    assert variants["full_copal"]["unique_clause_set_count"] == 4
    assert variants["full_copal"]["contract_valid_rate"] == 1.0
    assert variants["without_interaction_filter"]["status"] == "not_run"
    assert variants["without_facet_driven_synthesis"]["status"] == "not_run"
    assert variants["without_facet_driven_synthesis"]["reportable"] is False
    assert "cell_count" not in variants["without_facet_driven_synthesis"]
    assert variants["without_coverage_aware_selection"]["mean_cell_count"] < 4
    assert variants["without_coverage_aware_selection"]["mean_unique_clause_set_count"] < 4

    curve_by_budget = {row["budget"]: row for row in summary["coverage_curve"]}
    assert curve_by_budget[4]["full_cell_count"] == 4
    assert curve_by_budget[4]["full_unique_clause_set_count"] == 4
    assert curve_by_budget[4]["random_mean_cell_count"] < 4


def test_fast_pilot_ablation_rejects_empty_inputs() -> None:
    with pytest.raises(ValueError, match="candidates"):
        summarize_fast_pilot_ablation(candidates=[], selected=[row("a", "scope-restriction", "over-refusal")])

    with pytest.raises(ValueError, match="selected"):
        summarize_fast_pilot_ablation(candidates=[row("a", "scope-restriction", "over-refusal")], selected=[])


def test_fast_pilot_ablation_uses_primary_pattern_for_target_cell() -> None:
    candidate = {
        **row("q1", "prerequisite-gating", "skipped-gate"),
        "relation_patterns": ["prerequisite-gating", "scope-restriction"],
    }

    summary = summarize_fast_pilot_ablation(
        candidates=[candidate],
        selected=[candidate],
        random_seed_count=1,
    )

    assert summary["candidate_cell_count"] == 1
    assert summary["variants"][0]["cells"] == ["prerequisite-gating::skipped-gate"]


def test_aggregate_fast_ablation_summaries_macro_averages_variants() -> None:
    company_summaries = [
        {
            "fast_ablation_summary": {
                "variants": [
                    {"ablation_id": "full_copal", "cell_count": 12, "cpq": 1.0},
                    {"ablation_id": "without_coverage_aware_selection", "cell_count": 8, "cpq": 0.67},
                ]
            }
        },
        {
            "fast_ablation_summary": {
                "variants": [
                    {"ablation_id": "full_copal", "cell_count": 10, "cpq": 0.83},
                    {"ablation_id": "without_coverage_aware_selection", "cell_count": 6, "cpq": 0.50},
                ]
            }
        },
    ]

    aggregate = aggregate_fast_ablation_summaries(company_summaries)

    rows = {row["ablation_id"]: row for row in aggregate["variants"]}
    assert rows["full_copal"]["mean_cell_count"] == 11
    assert rows["without_coverage_aware_selection"]["mean_cell_count"] == 7


def test_aggregate_fast_ablation_summaries_preserves_not_run_variants() -> None:
    company_summaries = [
        {
            "fast_ablation_summary": {
                "variants": [
                    {
                        "ablation_id": "without_facet_driven_synthesis",
                        "status": "not_run",
                        "reportable": False,
                        "requires_artifact": "independent no-facet run",
                    }
                ]
            }
        }
    ]

    aggregate = aggregate_fast_ablation_summaries(company_summaries)

    row = aggregate["variants"][0]
    assert row["ablation_id"] == "without_facet_driven_synthesis"
    assert row["status"] == "not_run"
    assert row["reportable_company_count"] == 0
    assert row["mean_cell_count"] is None
    assert row["requires_artifact"] == "independent no-facet run"
