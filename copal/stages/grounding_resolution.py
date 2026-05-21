from __future__ import annotations

from pathlib import Path

from copal.io import ensure_directory, read_jsonl, write_json, write_jsonl
from copal.stages.grounding import dedupe_exact_clauses, normalize_clause_row


def run_grounding_resolution_stage(*, grounding_dir: Path) -> dict[str, object]:
    ensure_directory(grounding_dir)
    candidate_rows = read_jsonl(grounding_dir / "canonicalization_candidates.jsonl")
    normalized_rows = [
        normalize_clause_row(
            company_key=str(row["company_key"]),
            source_rule_id=str(row["source_rule_id"]),
            source_rule_type=str(row["source_rule_type"]),
            clause=row,
        )
        for row in candidate_rows
    ]
    kept_rows, removed_rows = dedupe_exact_clauses(normalized_rows)
    semantic_pairs: list[dict[str, object]] = []
    semantic_resolutions: list[dict[str, object]] = []

    summary = {
        "grounded_clause_count": len(kept_rows),
        "exact_duplicate_count": len(removed_rows),
        "semantic_duplicate_candidate_count": len(semantic_pairs),
    }

    write_json(grounding_dir / "exact_dedup_report.json", summary)
    write_jsonl(grounding_dir / "semantic_dedup_pairs.jsonl", semantic_pairs)
    write_jsonl(grounding_dir / "semantic_dedup_resolutions.jsonl", semantic_resolutions)
    write_jsonl(grounding_dir / "grounded_clause_library.jsonl", kept_rows)
    write_json(grounding_dir / "grounding_resolution_summary.json", summary)
    return summary
