from __future__ import annotations

from pathlib import Path

from copal.io import ensure_directory, write_json, write_jsonl


def build_reference_subset_row(
    *,
    target_id: str,
    accepted_or_rejected: str,
    signature: str,
    facet: str,
    nonseparability_slice: str | None = None,
) -> dict[str, object]:
    if nonseparability_slice is None:
        nonseparability_slice = "clear_non_separable" if accepted_or_rejected == "accepted" else "borderline"
    return {
        "reference_id": target_id,
        "target_type": "candidate",
        "target_id": target_id,
        "accepted_or_rejected": accepted_or_rejected,
        "signature": signature,
        "facet": facet,
        "nonseparability_slice": nonseparability_slice,
        "extraction_fidelity_label": "",
        "signature_assignment_label": "",
        "non_separability_label": "",
        "facet_labeling_label": "",
        "response_handling_category": "",
        "calibration_notes": "",
    }


def run_reference_subset_stage(
    *,
    reference_subset_dir: Path,
    accepted_items: list[dict[str, object]],
    rejected_candidates: list[dict[str, object]],
    target_size: int,
) -> dict[str, object]:
    ensure_directory(reference_subset_dir)
    rows: list[dict[str, object]] = []
    for item in accepted_items[:target_size]:
        rows.append(
            build_reference_subset_row(
                target_id=str(item.get("item_id", item.get("query_id", "accepted"))),
                accepted_or_rejected="accepted",
                signature=str(item.get("signature", item.get("signature_proposal", ""))),
                facet=str(item.get("target_facet", item.get("facet", ""))),
                nonseparability_slice=str(item.get("nonseparability_slice", "clear_non_separable")),
            )
        )
    remaining = max(target_size - len(rows), 0)
    for item in rejected_candidates[:remaining]:
        rows.append(
            build_reference_subset_row(
                target_id=str(item.get("item_id", item.get("query_id", "rejected"))),
                accepted_or_rejected="rejected",
                signature=str(item.get("signature", item.get("signature_proposal", ""))),
                facet=str(item.get("target_facet", item.get("facet", ""))),
                nonseparability_slice=str(item.get("nonseparability_slice", "borderline")),
            )
        )
    summary = {
        "reference_count": len(rows),
        "accepted_count": sum(1 for row in rows if row["accepted_or_rejected"] == "accepted"),
        "rejected_count": sum(1 for row in rows if row["accepted_or_rejected"] == "rejected"),
        "nonseparability_slice_counts": {
            slice_name: sum(1 for row in rows if row["nonseparability_slice"] == slice_name)
            for slice_name in sorted({str(row["nonseparability_slice"]) for row in rows})
        },
    }
    write_jsonl(reference_subset_dir / "reference_subset.jsonl", rows)
    write_json(reference_subset_dir / "reference_subset_summary.json", summary)
    return summary
