from __future__ import annotations

import json
from pathlib import Path

from scripts.run_paired_single_composed_from_table3 import (
    discover_ready_table3_runs,
    paired_table3_run_is_complete,
    prefill_legacy_single_policy_rows,
)


def test_discover_ready_table3_runs_requires_completed_table3_eval(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "runs" / "experiments" / "table3"
    ready_run = experiment_dir / "company_runs" / "run-ready"
    incomplete_run = experiment_dir / "company_runs" / "run-incomplete"
    for path in (
        ready_run / "selected_items.jsonl",
        ready_run / "evaluation" / "response_judgments.jsonl",
        ready_run / "table3_company_manifest.json",
        ready_run / "table3_company_summary.json",
        incomplete_run / "selected_items.jsonl",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    ready = discover_ready_table3_runs([experiment_dir])

    assert ready == [ready_run]


def test_prefill_legacy_single_policy_rows_reuses_only_expected_responses(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    paired = tmp_path / "paired"
    projection_items = [{"item_id": "composed-1::single::c1"}, {"item_id": "composed-2::single::c2"}]
    expected_response_id = "composed-1::single::c1::model-a"
    unexpected_response_id = "old-only::single::c9::model-a"
    _write_rows(
        legacy / "chatbot_responses.jsonl",
        [
            {"response_id": expected_response_id, "response_model": "model-a", "response_text": "ok"},
            {"response_id": unexpected_response_id, "response_model": "model-a", "response_text": "old"},
        ],
    )
    _write_rows(
        legacy / "response_judgments.jsonl",
        [
            {"response_id": expected_response_id, "response_model": "model-a", "overall_correct": True},
            {"response_id": unexpected_response_id, "response_model": "model-a", "overall_correct": False},
        ],
    )

    summary = prefill_legacy_single_policy_rows(
        legacy_paired_dir=legacy,
        paired_dir=paired,
        projection_items=projection_items,
        eval_models=["model-a"],
    )

    assert summary["expected_response_count"] == 2
    assert summary["prefilled_response_count"] == 1
    assert summary["prefilled_judgment_count"] == 1
    assert _read_response_ids(paired / "chatbot_responses.jsonl") == [expected_response_id]
    assert _read_response_ids(paired / "response_judgments.jsonl") == [expected_response_id]


def test_paired_table3_run_completion_uses_table3_selected_items(tmp_path: Path) -> None:
    run_dir = tmp_path / "table3-paired"
    assert paired_table3_run_is_complete(run_dir, ["model-a"]) is False

    _write_items(run_dir / "selected_items.jsonl", ["composed-1", "composed-2"])
    _write_items(
        run_dir / "paired_single_policy" / "single_policy_projection_items.jsonl",
        ["composed-1::single::c1", "composed-2::single::c2"],
    )
    summary_path = run_dir / "paired_single_policy" / "paired_single_composed_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("{}", encoding="utf-8")

    _write_judgments(
        run_dir / "composed" / "response_judgments.jsonl",
        [("composed-1::model-a", "model-a"), ("composed-2::model-a", "model-a")],
    )
    _write_judgments(
        run_dir / "paired_single_policy" / "response_judgments.jsonl",
        [
            ("composed-1::single::c1::model-a", "model-a"),
            ("composed-2::single::c2::model-a", "model-a"),
        ],
    )

    assert paired_table3_run_is_complete(run_dir, ["model-a"]) is True


def _write_items(path: Path, item_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps({"item_id": item_id}) for item_id in item_ids) + "\n",
        encoding="utf-8",
    )


def _write_judgments(path: Path, rows: list[tuple[str, str]]) -> None:
    _write_rows(path, [{"response_id": response_id, "response_model": model} for response_id, model in rows])


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _read_response_ids(path: Path) -> list[str]:
    return [str(json.loads(line)["response_id"]) for line in path.read_text(encoding="utf-8").splitlines() if line]
