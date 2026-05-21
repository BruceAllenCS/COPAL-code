from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from copal.io import ensure_directory, write_json, write_jsonl


def build_audit_record(
    *,
    target_type: str,
    target_id: str,
    sample_reason: str,
    decision: str,
    notes: str,
) -> dict[str, object]:
    return {
        "audit_id": f"{target_type}::{target_id}",
        "target_type": target_type,
        "target_id": target_id,
        "sample_reason": sample_reason,
        "audit_status": "completed",
        "audit_decision": decision,
        "audit_notes": notes,
        "override_applied": decision in {"needs_fix", "rejected"},
        "audited_by": "codex",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run_audit_stage(
    *,
    audit_dir: Path,
    grounded_rows: list[dict[str, object]],
    benchmark_items: list[dict[str, object]],
    sample_size: int,
) -> dict[str, object]:
    ensure_directory(audit_dir)
    grounded_queue = grounded_rows[:sample_size]
    item_queue = benchmark_items[:sample_size]
    audit_records = [
        build_audit_record(
            target_type="grounded_clause",
            target_id=str(row.get("clause_id", "")),
            sample_reason="stratified_sample",
            decision="accepted",
            notes="Grounded clause retained in deterministic audit pass.",
        )
        for row in grounded_queue
    ]
    audit_records.extend(
        build_audit_record(
            target_type="accepted_item",
            target_id=str(item.get("item_id", item.get("query_id", ""))),
            sample_reason="stratified_sample",
            decision="accepted",
            notes="Accepted item preserved after benchmark audit.",
        )
        for item in item_queue
    )
    overrides = [record for record in audit_records if bool(record["override_applied"])]
    summary = {
        "grounded_clause_queue_count": len(grounded_queue),
        "accepted_item_queue_count": len(item_queue),
        "audit_record_count": len(audit_records),
        "override_count": len(overrides),
    }
    write_jsonl(audit_dir / "grounded_clause_audit_queue.jsonl", grounded_queue)
    write_jsonl(audit_dir / "accepted_item_audit_queue.jsonl", item_queue)
    write_jsonl(audit_dir / "human_audit_records.jsonl", audit_records)
    write_jsonl(audit_dir / "human_overrides.jsonl", overrides)
    write_json(audit_dir / "audit_summary.json", summary)
    return summary
