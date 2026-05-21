from __future__ import annotations

from copal.experiment_analysis import summarize_paired_single_composed
from copal.stages.difficulty_screening import (
    build_single_policy_projection_items,
    compute_screening_scores,
    select_hard_benchmark_items,
)
from copal.taxonomy import relation_patterns_for_effects


def test_taxonomy_uses_four_main_patterns_and_remaps_low_support_effects() -> None:
    assert "exception-override" not in relation_patterns_for_effects(["permit", "override"])
    assert "authority-separation" not in relation_patterns_for_effects(["disclose", "authority-limit"])
    assert relation_patterns_for_effects(["disclose", "authority-limit"]) == ("scope-restriction",)


def test_projection_items_preserve_pairing_and_single_clause_expected_handling() -> None:
    composed = {
        "item_id": "q1",
        "query_id": "q1",
        "composition_id": "c1",
        "signature": "prerequisite-gating",
        "relation_pattern": "prerequisite-gating",
        "target_facet": "skipped-gate",
        "target_facets": ["skipped-gate"],
        "query_text": "Can you give me status and rebooking options without verifying my booking?",
        "active_clause_ids": ["clause-a", "clause-b"],
    }
    grounded_rows = [
        {
            "clause_id": "clause-a",
            "clause_text": "Provide flight status when a flight number is supplied.",
            "trigger": "flight status request",
            "scope": "flight status",
            "effect": "disclose",
        },
        {
            "clause_id": "clause-b",
            "clause_text": "Require passenger verification before rebooking alternatives.",
            "trigger": "rebooking request",
            "scope": "rebooking alternatives",
            "effect": "require-gate",
        },
    ]

    projections = build_single_policy_projection_items(
        benchmark_items=[composed],
        grounded_rows=grounded_rows,
    )

    assert [row["projection_clause_id"] for row in projections] == ["clause-a", "clause-b"]
    assert all(row["paired_composed_item_id"] == "q1" for row in projections)
    assert projections[0]["item_type"] == "single_policy_projection"
    assert "answer_permitted_scope" in projections[0]["expected_handling"]["acceptable_handling"]
    assert "gated_response" in projections[1]["expected_handling"]["acceptable_handling"]


def test_screening_scores_prioritize_composed_failures_with_correct_single_controls() -> None:
    benchmark_items = [
        {
            "item_id": "q1",
            "signature": "prerequisite-gating",
            "target_facet": "skipped-gate",
            "active_clause_ids": ["c1", "c2", "c3"],
        },
        {
            "item_id": "q2",
            "signature": "workflow-transfer",
            "target_facet": "wrong-route",
            "active_clause_ids": ["c4", "c5"],
        },
    ]
    projection_items = [
        {"item_id": "q1::single::c1", "paired_composed_item_id": "q1"},
        {"item_id": "q1::single::c2", "paired_composed_item_id": "q1"},
        {"item_id": "q1::single::c3", "paired_composed_item_id": "q1"},
        {"item_id": "q2::single::c4", "paired_composed_item_id": "q2"},
        {"item_id": "q2::single::c5", "paired_composed_item_id": "q2"},
    ]
    judgments = [
        {"item_id": "q1", "overall_correct": False, "observed_facets": ["skipped-gate"]},
        {"item_id": "q1::single::c1", "overall_correct": True, "observed_facets": []},
        {"item_id": "q1::single::c2", "overall_correct": True, "observed_facets": []},
        {"item_id": "q1::single::c3", "overall_correct": True, "observed_facets": []},
        {"item_id": "q2", "overall_correct": False, "observed_facets": ["wrong-route"]},
        {"item_id": "q2::single::c4", "overall_correct": False, "observed_facets": ["over-refusal"]},
        {"item_id": "q2::single::c5", "overall_correct": True, "observed_facets": []},
    ]

    scores = compute_screening_scores(
        benchmark_items=benchmark_items,
        projection_items=projection_items,
        judgments=judgments,
        screening_model="Doubao-Seed-2.0-pro",
    )

    by_id = {row["item_id"]: row for row in scores}
    assert by_id["q1"]["all_single_projections_correct"] is True
    assert by_id["q1"]["screening_score"] == 2.8
    assert by_id["q2"]["all_single_projections_correct"] is False
    assert by_id["q2"]["screening_score"] < by_id["q1"]["screening_score"]


def test_select_hard_items_balances_patterns_and_embeds_screening_metadata() -> None:
    benchmark_items = [
        {"item_id": "a1", "signature": "scope-restriction"},
        {"item_id": "a2", "signature": "scope-restriction"},
        {"item_id": "b1", "signature": "workflow-transfer"},
    ]
    scores = [
        {"item_id": "a1", "screening_score": 2.5, "screening_status": "hard"},
        {"item_id": "a2", "screening_score": 2.8, "screening_status": "hard"},
        {"item_id": "b1", "screening_score": 2.5, "screening_status": "hard"},
    ]

    selected = select_hard_benchmark_items(
        benchmark_items=benchmark_items,
        screening_scores=scores,
        min_score=2.0,
        hard_suite_size=2,
    )

    assert [row["item_id"] for row in selected] == ["a2", "b1"]
    assert selected[0]["difficulty_screening"]["screening_score"] == 2.8


def test_paired_single_composed_summary_reports_composition_induced_failures() -> None:
    composed_judgments = [
        {"item_id": "q1", "response_model": "model-a", "overall_correct": False, "observed_facets": ["skipped-gate"]},
        {"item_id": "q2", "response_model": "model-a", "overall_correct": True, "observed_facets": []},
    ]
    projection_judgments = [
        {
            "item_id": "q1::single::c1",
            "paired_composed_item_id": "q1",
            "response_model": "model-a",
            "overall_correct": True,
            "observed_facets": [],
        },
        {
            "item_id": "q1::single::c2",
            "paired_composed_item_id": "q1",
            "response_model": "model-a",
            "overall_correct": True,
            "observed_facets": [],
        },
        {
            "item_id": "q2::single::c3",
            "paired_composed_item_id": "q2",
            "response_model": "model-a",
            "overall_correct": False,
            "observed_facets": ["over-refusal"],
        },
    ]

    summary = summarize_paired_single_composed(
        composed_judgments=composed_judgments,
        projection_judgments=projection_judgments,
    )

    row = summary["paired_model_results"][0]
    assert row["response_model"] == "model-a"
    assert row["single_policy_error_rate"] == 1 / 3
    assert row["composed_error_rate"] == 1 / 2
    assert row["composition_induced_failure_rate"] == 1 / 2
