from pathlib import Path

from copal.io import read_json, read_jsonl
from copal.stages.audit import build_audit_record, run_audit_stage
from copal.stages.reference_subset import build_reference_subset_row, run_reference_subset_stage


def test_build_reference_subset_row_tracks_accept_reject_status() -> None:
    row = build_reference_subset_row(
        target_id="cand-1",
        accepted_or_rejected="rejected",
        signature="scope-restriction",
        facet="semantic-leakage",
    )
    assert row["accepted_or_rejected"] == "rejected"
    assert row["nonseparability_slice"] == "borderline"


def test_build_audit_record_marks_codex_as_auditor() -> None:
    row = build_audit_record(
        target_type="accepted_item",
        target_id="item-1",
        sample_reason="stratified_sample",
        decision="accepted",
        notes="Facet label matches query semantics.",
    )
    assert row["audited_by"] == "codex"


def test_reference_subset_and_audit_stages_write_outputs(tmp_path: Path) -> None:
    accepted_items = [
        {"item_id": "item-1", "signature": "scope-restriction", "target_facet": "semantic-leakage"},
        {"item_id": "item-2", "signature": "prerequisite-gating", "target_facet": "skipped-gate"},
    ]
    rejected_candidates = [
        {"item_id": "cand-1", "signature": "scope-restriction", "target_facet": "boundary-overreach"},
    ]
    reference_summary = run_reference_subset_stage(
        reference_subset_dir=tmp_path / "reference_subset",
        accepted_items=accepted_items,
        rejected_candidates=rejected_candidates,
        target_size=3,
    )
    audit_summary = run_audit_stage(
        audit_dir=tmp_path / "audit",
        grounded_rows=[{"clause_id": "c1"}, {"clause_id": "c2"}],
        benchmark_items=accepted_items,
        sample_size=2,
    )

    assert reference_summary["reference_count"] == 3
    assert audit_summary["audit_record_count"] >= 1
    assert (tmp_path / "reference_subset" / "reference_subset.jsonl").exists()
    assert (tmp_path / "audit" / "human_audit_records.jsonl").exists()
    subset_rows = read_jsonl(tmp_path / "reference_subset" / "reference_subset.jsonl")
    assert {row["accepted_or_rejected"] for row in subset_rows} == {"accepted", "rejected"}
    audit_report = read_json(tmp_path / "audit" / "audit_summary.json")
    assert audit_report["audit_record_count"] == audit_summary["audit_record_count"]
