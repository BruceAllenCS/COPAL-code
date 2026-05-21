from __future__ import annotations

from pathlib import Path

from copal.io import ensure_directory


def write_run_reports(*, reports_dir: Path, summary: dict[str, object]) -> None:
    ensure_directory(reports_dir)
    run_id = str(summary.get("run_id", "unknown"))
    company_key = str(summary.get("company_key", "unknown"))
    status = str(summary.get("status", "unknown"))
    grounding = dict(summary.get("grounding", {}))
    composition = dict(summary.get("composition", {}))
    selection = dict(summary.get("selection", {}))
    query_generation = dict(summary.get("query_generation", {}))
    evaluation = dict(summary.get("evaluation", {}))
    reference_subset = dict(summary.get("reference_subset", {}))
    audit = dict(summary.get("audit", {}))

    (reports_dir / "run_report.md").write_text(
        "\n".join(
            [
                f"# COPAL Run Report: {run_id}",
                "",
                f"- company_key: `{company_key}`",
                f"- status: `{status}`",
                f"- grounded clauses: `{grounding.get('grounded_clause_count', 0)}`",
                f"- accepted compositions: `{composition.get('accepted_count', 0)}`",
                f"- accepted queries: `{query_generation.get('accepted_count', 0)}`",
                f"- selected compositions: `{selection.get('selected_composition_count', 0)}`",
                f"- final benchmark items: `{selection.get('final_benchmark_count', 0)}`",
                f"- evaluated responses: `{evaluation.get('response_count', 0)}`",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (reports_dir / "signature_facet_breakdown.md").write_text(
        "\n".join(
            [
                "# Signature And Facet Breakdown",
                "",
                f"- composition signature counts: `{composition.get('signature_counts', {})}`",
                f"- response judgment signature count: `{evaluation.get('signature_count', 0)}`",
                f"- benchmark facet count: `{evaluation.get('facet_count', 0)}`",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (reports_dir / "validation_calibration_report.md").write_text(
        "\n".join(
            [
                "# Validation Calibration Report",
                "",
                f"- reference subset count: `{reference_subset.get('reference_count', 0)}`",
                f"- reference accepted count: `{reference_subset.get('accepted_count', 0)}`",
                f"- reference rejected count: `{reference_subset.get('rejected_count', 0)}`",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (reports_dir / "audit_report.md").write_text(
        "\n".join(
            [
                "# Audit Report",
                "",
                f"- audit record count: `{audit.get('audit_record_count', 0)}`",
                f"- override count: `{audit.get('override_count', 0)}`",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (reports_dir / "qualitative_cases.md").write_text(
        "\n".join(
            [
                "# Qualitative Cases",
                "",
                "This first-pass local implementation records candidate qualitative cases through the staged artifacts.",
                "Use benchmark items, audit records, and evaluation outputs to curate worked examples.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
