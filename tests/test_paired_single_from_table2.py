from __future__ import annotations

import json
from pathlib import Path

from scripts.run_paired_single_composed_from_table2 import (
    discover_ready_table2_copal_runs,
    paired_run_is_complete,
)


def test_discover_ready_table2_copal_runs_requires_completed_copal_variant(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "runs" / "experiments" / "table2"
    ready_run = experiment_dir / "company_runs" / "run-ready"
    incomplete_run = experiment_dir / "company_runs" / "run-incomplete"
    for path in (
        ready_run / "shared_grounding" / "grounded_clauses.jsonl",
        ready_run / "variants" / "copal" / "benchmark_items_final.jsonl",
        ready_run / "variants" / "copal" / "evaluation" / "response_judgments.jsonl",
        ready_run / "variants" / "copal" / "table2_variant_summary.json",
        incomplete_run / "shared_grounding" / "grounded_clauses.jsonl",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    ready = discover_ready_table2_copal_runs([experiment_dir])

    assert ready == [ready_run]


def test_paired_run_completion_uses_summary_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    assert paired_run_is_complete(run_dir, ["model-a"]) is False

    _write_items(run_dir / "variants" / "copal" / "benchmark_items_final.jsonl", ["composed-1"])
    _write_items(run_dir / "paired_single_policy" / "single_policy_projection_items.jsonl", ["single-1"])

    summary_path = run_dir / "paired_single_policy" / "paired_single_composed_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("{}", encoding="utf-8")

    assert paired_run_is_complete(run_dir, ["model-a"]) is False

    _write_judgments(
        run_dir / "variants" / "copal" / "evaluation" / "response_judgments.jsonl",
        [("composed-1::model-a", "model-a")],
    )
    _write_judgments(
        run_dir / "paired_single_policy" / "response_judgments.jsonl",
        [("single-1::model-a", "model-a")],
    )

    assert paired_run_is_complete(run_dir, ["model-a"]) is True
    assert paired_run_is_complete(run_dir, ["model-a", "model-b"]) is False

    _write_judgments(
        run_dir / "variants" / "copal" / "evaluation" / "response_judgments.jsonl",
        [("composed-1::model-a", "model-a"), ("composed-1::model-b", "model-b")],
    )
    _write_judgments(
        run_dir / "paired_single_policy" / "response_judgments.jsonl",
        [("single-1::model-a", "model-a"), ("single-1::model-b", "model-b")],
    )

    assert paired_run_is_complete(run_dir, ["model-a", "model-b"]) is True


def _write_items(path: Path, item_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps({"item_id": item_id}) for item_id in item_ids) + "\n",
        encoding="utf-8",
    )


def _write_judgments(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps({"response_id": response_id, "response_model": model}) for response_id, model in rows)
        + "\n",
        encoding="utf-8",
    )
