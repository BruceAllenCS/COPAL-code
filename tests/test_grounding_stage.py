from pathlib import Path

from copal.data_sources import select_company_world
from copal.io import read_json, read_jsonl
from copal.stages.grounding import (
    dedupe_exact_clauses,
    normalize_clause_row,
    run_grounding_stage,
)


def test_normalize_clause_row_emits_required_grounded_fields() -> None:
    row = normalize_clause_row(
        company_key="x",
        source_rule_id="A1",
        source_rule_type="allowed",
        clause={
            "clause_text": "Provide status.",
            "trigger": "status request",
            "scope": "booking",
            "effect": "allow",
            "source_span": "Provide status.",
        },
    )
    assert row["company_key"] == "x"
    assert row["source_rule_id"] == "A1"
    assert row["effect"] == "permit"
    assert "provenance" in row


def test_dedupe_exact_clauses_removes_exact_duplicates() -> None:
    rows = [
        {"clause_id": "c1", "clause_text": "Provide status.", "trigger": "t", "scope": "s", "effect": "permit"},
        {"clause_id": "c2", "clause_text": "Provide status.", "trigger": "t", "scope": "s", "effect": "permit"},
    ]
    kept, removed = dedupe_exact_clauses(rows)
    assert len(kept) == 1
    assert len(removed) == 1


def test_run_grounding_stage_writes_aligned_artifacts(tmp_path: Path) -> None:
    world, _ = select_company_world(
        policies_path=Path("data/compass_policies/compass_policies_final.jsonl"),
        prompts_path=Path("data/compass_policies/company_system_prompts.jsonl"),
        company_key="Air transportation||000||Skyline International Airways",
    )
    grounding_dir = tmp_path / "grounding"
    summary = run_grounding_stage(grounding_dir=grounding_dir, world=world, execution_mode="deterministic")

    assert summary["company_key"] == world.company_key
    assert (grounding_dir / "raw_clause_extraction.json").exists()
    assert (grounding_dir / "canonicalized_clauses.jsonl").exists()
    assert (grounding_dir / "grounded_clauses_final.jsonl").exists()
    assert (grounding_dir / "grounding_summary.json").exists()

    grounded_rows = read_jsonl(grounding_dir / "grounded_clauses_final.jsonl")
    assert grounded_rows
    assert any(row["source_rule_type"] == "allowed" for row in grounded_rows)
    assert any(row["source_rule_type"] == "prohibited" for row in grounded_rows)

    raw_payload = read_json(grounding_dir / "raw_clause_extraction.json")
    assert raw_payload["company_key"] == world.company_key
