from __future__ import annotations

from pathlib import Path

from copal.io import ensure_directory, write_json, write_jsonl
from copal.stages.compositions import propose_grounded_compositions


def run_composition_proposal_stage(
    *,
    compositions_dir: Path,
    grounded_rows: list[dict[str, object]],
) -> dict[str, object]:
    ensure_directory(compositions_dir)
    candidates = propose_grounded_compositions(grounded_rows)
    structure_signal_rows = [
        {
            "composition_id": row["composition_id"],
            "structure_signals": row["structure_signals"],
        }
        for row in candidates
    ]
    signature_rows = [
        {
            "composition_id": row["composition_id"],
            "signature_proposal": row["signature_proposal"],
            "relation_pattern": row["relation_pattern"],
            "relation_patterns": row["relation_patterns"],
            "signature_source": row["signature_source"],
        }
        for row in candidates
    ]
    summary = {
        "candidate_count": len(candidates),
        "signature_candidate_count": sum(1 for row in candidates if row["signature_proposal"]),
    }
    write_jsonl(compositions_dir / "candidate_compositions.jsonl", candidates)
    write_jsonl(compositions_dir / "structure_signal_records.jsonl", structure_signal_rows)
    write_jsonl(compositions_dir / "signature_proposals.jsonl", signature_rows)
    write_json(compositions_dir / "composition_proposal_summary.json", summary)
    return summary
