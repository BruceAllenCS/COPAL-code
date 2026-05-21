from __future__ import annotations

from pathlib import Path

from copal import table3_model_eval
from copal.io import read_json, read_jsonl, write_json, write_jsonl


def _candidate_row(query_id: str, challenge_score: float) -> dict[str, object]:
    return {
        "company_key": "company-1",
        "company_name": "Company One",
        "query_id": query_id,
        "composition_id": "COMP-1",
        "relation_pattern": "scope-restriction",
        "relation_patterns": ["scope-restriction"],
        "target_facet": "boundary-overreach",
        "target_facets": ["boundary-overreach"],
        "query_text": f"Query {query_id}",
        "scenario": {"clause_ids": ["CL-1"], "target_facet": "boundary-overreach"},
        "active_clause_ids": ["CL-1"],
        "allowed_answer_anchor": "Allowed anchor",
        "forbidden_outcome": "Forbidden outcome",
        "required_gate_or_route": "Required route",
        "trap_mechanism": "Trap mechanism",
        "expected_composed_handling": {"must_do": ["help"], "must_not_do": ["overreach"]},
        "single_policy_projections": [],
        "challenge_score": challenge_score,
        "screening_rationale": f"rationale {query_id}",
    }


def _benchmark_item(item_id: str) -> dict[str, object]:
    return {
        "item_id": item_id,
        "query_id": item_id,
        "company_key": "company-1",
        "company_name": "Company One",
        "signature": "scope-restriction",
        "relation_pattern": "scope-restriction",
        "target_facet": "boundary-overreach",
        "target_facets": ["boundary-overreach"],
        "query_text": f"Seed query {item_id}",
        "selection_metadata": {"selection_rank": 1, "challenge_score": 0.1},
    }


def _source_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "source_run"
    run_dir.mkdir()
    write_json(
        run_dir / "selected_company.json",
        {"company_key": "company-1", "company_name": "Company One"},
    )
    return run_dir


def test_table3_selection_keeps_existing_final_items_before_candidate_fill(tmp_path: Path) -> None:
    run_dir = _source_run(tmp_path)
    copal_dir = run_dir / "variants" / "copal"
    copal_dir.mkdir(parents=True)
    write_jsonl(
        copal_dir / "candidate_queries_labeled.jsonl",
        [
            _candidate_row("high-1", 9.0),
            _candidate_row("high-2", 8.0),
            _candidate_row("high-3", 7.0),
            _candidate_row("seed-low", 0.1),
        ],
    )
    write_jsonl(copal_dir / "benchmark_items_final.jsonl", [_benchmark_item("seed-low")])

    result = table3_model_eval.load_table3_items_for_run(run_dir=run_dir, max_items=3)

    item_ids = [item["item_id"] for item in result.items]
    assert item_ids[0] == "seed-low"
    assert len(item_ids) == 3
    assert set(item_ids) == {"seed-low", "high-1", "high-2"}
    assert result.source_kind == "candidate_queries_labeled_seeded_by_benchmark_items_final"


def test_prefill_reuses_matching_source_responses_and_judgments(tmp_path: Path) -> None:
    source_run_dir = _source_run(tmp_path)
    source_eval_dir = source_run_dir / "variants" / "copal" / "evaluation"
    source_eval_dir.mkdir(parents=True)
    selected_items = [_benchmark_item("seed-1"), _benchmark_item("fill-1")]
    response_ids = [
        "seed-1::Doubao-Seed-2.0-pro",
        "seed-1::gemini-3.1-pro-preview",
    ]
    write_jsonl(
        source_eval_dir / "chatbot_responses.jsonl",
        [
            {
                "response_id": response_id,
                "item_id": "seed-1",
                "response_model": response_id.split("::", maxsplit=1)[1],
                "response_text": f"response {response_id}",
            }
            for response_id in response_ids
        ],
    )
    write_jsonl(
        source_eval_dir / "response_judgments.jsonl",
        [
            {
                "response_id": response_id,
                "item_id": "seed-1",
                "response_model": response_id.split("::", maxsplit=1)[1],
                "signature": "scope-restriction",
                "target_facet": "boundary-overreach",
                "overall_correct": True,
            }
            for response_id in response_ids
        ],
    )
    output_run_dir = tmp_path / "table3_run"

    summary = table3_model_eval.prefill_table3_from_source_copal_evaluation(
        output_run_dir=output_run_dir,
        source_run_dir=source_run_dir,
        selected_items=selected_items,
        eval_models=["Doubao-Seed-2.0-pro", "gemini-3.1-pro-preview", "gpt-5.5"],
    )

    copied_responses = read_jsonl(output_run_dir / "evaluation" / "chatbot_responses.jsonl")
    copied_judgments = read_jsonl(output_run_dir / "evaluation" / "response_judgments.jsonl")
    assert [row["response_id"] for row in copied_responses] == response_ids
    assert [row["response_id"] for row in copied_judgments] == response_ids
    assert summary["prefilled_response_count"] == 2
    assert summary["prefilled_judgment_count"] == 2
    assert read_json(output_run_dir / "evaluation" / "table2_copal_prefill_summary.json") == summary
