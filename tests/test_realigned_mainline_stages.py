from pathlib import Path

from copal.data_sources import select_company_world
from copal.io import read_json, read_jsonl
from copal.stages.composition_proposal import run_composition_proposal_stage
from copal.stages.composition_validation import run_composition_validation_stage
from copal.stages.coverage_judge import run_coverage_judge_stage
from copal.stages.downstream_chatbot import run_downstream_chatbot_stage
from copal.stages.grounding_proposal import run_grounding_proposal_stage
from copal.stages.grounding_resolution import run_grounding_resolution_stage
from copal.stages.query_proposal import run_query_proposal_stage
from copal.stages.query_validation import run_query_validation_stage
from copal.stages.reference_subset import run_reference_subset_stage
from copal.stages.response_judgment import run_response_judgment_stage
from copal.stages.selection import run_selection_stage


def test_grounding_mainline_writes_proposal_and_resolution_artifacts(tmp_path: Path) -> None:
    world, _ = select_company_world(
        policies_path=Path("data/compass_policies/compass_policies_final.jsonl"),
        prompts_path=Path("data/compass_policies/company_system_prompts.jsonl"),
        company_key="Air transportation||000||Skyline International Airways",
    )
    grounding_dir = tmp_path / "grounding"

    proposal_summary = run_grounding_proposal_stage(
        grounding_dir=grounding_dir,
        world=world,
        execution_mode="deterministic",
    )
    resolution_summary = run_grounding_resolution_stage(
        grounding_dir=grounding_dir,
    )

    assert proposal_summary["raw_clause_count"] >= 1
    assert resolution_summary["grounded_clause_count"] >= 1
    assert (grounding_dir / "raw_clause_extractions.jsonl").exists()
    assert (grounding_dir / "canonicalization_candidates.jsonl").exists()
    assert (grounding_dir / "grounded_clause_library.jsonl").exists()
    grounded_rows = read_jsonl(grounding_dir / "grounded_clause_library.jsonl")
    assert grounded_rows
    assert "provenance" in grounded_rows[0]


def test_composition_and_query_mainline_emit_adjudication_queues(tmp_path: Path) -> None:
    world, _ = select_company_world(
        policies_path=Path("data/compass_policies/compass_policies_final.jsonl"),
        prompts_path=Path("data/compass_policies/company_system_prompts.jsonl"),
        company_key="Air transportation||000||Skyline International Airways",
    )
    grounding_dir = tmp_path / "grounding"
    run_grounding_proposal_stage(grounding_dir=grounding_dir, world=world, execution_mode="deterministic")
    run_grounding_resolution_stage(grounding_dir=grounding_dir)
    grounded_rows = read_jsonl(grounding_dir / "grounded_clause_library.jsonl")

    compositions_dir = tmp_path / "compositions"
    query_dir = tmp_path / "query_generation"
    run_composition_proposal_stage(
        compositions_dir=compositions_dir,
        grounded_rows=grounded_rows,
    )
    composition_summary = run_composition_validation_stage(
        compositions_dir=compositions_dir,
        execution_mode="deterministic",
    )
    accepted_compositions = read_jsonl(compositions_dir / "accepted_compositions.jsonl")
    run_query_proposal_stage(
        query_generation_dir=query_dir,
        accepted_compositions=accepted_compositions,
        facet_library={
            "scope-restriction": ("boundary-overreach", "semantic-leakage"),
            "prerequisite-gating": ("skipped-gate",),
            "workflow-transfer": ("missed-transfer",),
        },
        execution_mode="deterministic",
    )
    query_summary = run_query_validation_stage(
        query_generation_dir=query_dir,
        execution_mode="deterministic",
    )

    assert composition_summary["accepted_count"] >= 1
    assert (compositions_dir / "composition_adjudication_queue.jsonl").exists()
    assert (compositions_dir / "accepted_compositions.jsonl").exists()
    assert (compositions_dir / "rejected_compositions.jsonl").exists()
    assert query_summary["accepted_count"] >= 1
    assert (query_dir / "query_adjudication_queue.jsonl").exists()
    assert (query_dir / "accepted_queries.jsonl").exists()
    assert (query_dir / "rejected_queries.jsonl").exists()


def test_coverage_reference_and_response_mainline_write_split_artifacts(tmp_path: Path) -> None:
    accepted_queries = [
        {
            "query_id": "q1",
            "composition_id": "c1",
            "signature_proposal": "scope-restriction",
            "target_facet": "semantic-leakage",
            "query_text": "Need help with a sensitive product question.",
            "scenario": {
                "scenario_id": "s1",
                "shared_user_scenario": "sensitive product question",
                "jointly_satisfied_triggers": ["product", "safety"],
                "coupled_scope_or_path": "same answer span",
                "non_decomposability_rationale": "One clause narrows the answer path.",
                "required_state_assumptions": [],
                "leakage_naturalness_self_check": "natural",
                "clause_ids": ["a", "b"],
            },
            "validation_metadata": {"stage": "query_validation"},
        },
        {
            "query_id": "q2",
            "composition_id": "c1",
            "signature_proposal": "scope-restriction",
            "target_facet": "boundary-overreach",
            "query_text": "Need help with a product exception.",
            "scenario": {
                "scenario_id": "s2",
                "shared_user_scenario": "product exception",
                "jointly_satisfied_triggers": ["product", "exception"],
                "coupled_scope_or_path": "same answer span",
                "non_decomposability_rationale": "The exception changes the allowed answer path.",
                "required_state_assumptions": [],
                "leakage_naturalness_self_check": "natural",
                "clause_ids": ["a", "b"],
            },
            "validation_metadata": {"stage": "query_validation"},
        },
    ]
    coverage_dir = tmp_path / "coverage"
    selection_dir = tmp_path / "selection"
    reference_dir = tmp_path / "reference_subset"
    evaluation_dir = tmp_path / "evaluation"

    coverage_summary = run_coverage_judge_stage(
        coverage_dir=coverage_dir,
        accepted_queries=accepted_queries,
        facet_library={"scope-restriction": ("semantic-leakage", "boundary-overreach")},
        execution_mode="deterministic",
    )
    coverage_rows = read_jsonl(coverage_dir / "accepted_query_coverages.jsonl")
    selection_summary = run_selection_stage(
        selection_dir=selection_dir,
        accepted_queries=accepted_queries,
        coverage_rows=coverage_rows,
    )
    benchmark_items = read_jsonl(selection_dir / "benchmark_items_final.jsonl")
    reference_summary = run_reference_subset_stage(
        reference_subset_dir=reference_dir,
        accepted_items=benchmark_items,
        rejected_candidates=[
            {
                "item_id": "rej-1",
                "signature": "scope-restriction",
                "target_facet": "boundary-overreach",
                "nonseparability_slice": "borderline",
            }
        ],
        target_size=3,
    )
    chatbot_summary = run_downstream_chatbot_stage(
        evaluation_dir=evaluation_dir,
        benchmark_items=benchmark_items,
        system_prompt="You are the official customer-facing AI assistant.",
        execution_mode="deterministic",
        downstream_models=("model-a", "model-b"),
    )
    response_summary = run_response_judgment_stage(
        evaluation_dir=evaluation_dir,
        benchmark_items=benchmark_items,
        execution_mode="deterministic",
    )

    assert coverage_summary["coverage_result_count"] == 2
    assert selection_summary["final_benchmark_count"] >= 1
    assert reference_summary["reference_count"] >= 2
    assert chatbot_summary["model_count"] == 2
    assert chatbot_summary["response_count"] == selection_summary["final_benchmark_count"] * 2
    assert response_summary["judgment_count"] == chatbot_summary["response_count"]
    assert (coverage_dir / "composition_facet_universes.jsonl").exists()
    assert (coverage_dir / "coverage_judge_results.jsonl").exists()
    assert (evaluation_dir / "response_judge_inputs.jsonl").exists()
    reference_rows = read_jsonl(reference_dir / "reference_subset.jsonl")
    assert "nonseparability_slice" in reference_rows[0]
    evaluation_summary = read_json(evaluation_dir / "evaluation_summary.json")
    assert evaluation_summary["judgment_count"] == response_summary["judgment_count"]


def test_live_capable_stages_reject_unknown_execution_mode(tmp_path: Path) -> None:
    try:
        run_downstream_chatbot_stage(
            evaluation_dir=tmp_path / "evaluation",
            benchmark_items=[],
            system_prompt="system",
            execution_mode="bogus",
        )
    except ValueError as exc:
        assert "Unsupported execution_mode" in str(exc)
    else:
        raise AssertionError("unknown execution_mode must not fall back to deterministic behavior")
