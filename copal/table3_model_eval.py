from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from copal.config import DEFAULT_FACETS, DEFAULT_SIGNATURES
from copal.fast_pilot import build_pilot_benchmark_items, run_pilot_evaluation
from copal.io import ensure_directory, read_json, read_jsonl, write_json, write_jsonl
from copal.llm import LLMClient

TABLE3_CANDIDATE_FILENAME = "candidate_queries_labeled.jsonl"
TABLE3_FINAL_FILENAME = "benchmark_items_final.jsonl"
TABLE3_COMPANY_MANIFEST = "table3_company_manifest.json"


@dataclass(frozen=True, slots=True)
class Table3ItemsForRun:
    source_run_dir: Path
    source_kind: str
    source_path: Path
    source_sha256: str
    seed_source_path: Path | None
    seed_source_sha256: str | None
    seed_item_count: int
    fill_item_count: int
    company_key: str
    company_name: str
    items: list[dict[str, object]]


def discover_ready_table2_copal_runs(source_experiment_dirs: list[Path]) -> list[Path]:
    ready: list[Path] = []
    for experiment_dir in source_experiment_dirs:
        company_runs_dir = experiment_dir / "company_runs"
        if not company_runs_dir.exists():
            continue
        for run_dir in sorted(path for path in company_runs_dir.iterdir() if path.is_dir()):
            if not (run_dir / "selected_company.json").exists():
                continue
            if _nonempty_path(_candidate_path(run_dir)) or _nonempty_path(_final_items_path(run_dir)):
                ready.append(run_dir)
    return ready


def load_table3_items_for_run(*, run_dir: Path, max_items: int) -> Table3ItemsForRun:
    if max_items < 1:
        raise ValueError("max_items must be positive")
    selected_company = read_json(run_dir / "selected_company.json")
    company_key = str(selected_company["company_key"])
    company_name = str(selected_company.get("company_name", company_key))

    candidates_path = _candidate_path(run_dir)
    final_items_path = _final_items_path(run_dir)
    if _nonempty_path(candidates_path):
        seed_items = read_jsonl(final_items_path) if _nonempty_path(final_items_path) else []
        source_kind = (
            "candidate_queries_labeled_seeded_by_benchmark_items_final"
            if seed_items
            else "candidate_queries_labeled"
        )
        items, seed_item_count, fill_item_count = _select_table3_items(
            company_key=company_key,
            company_name=company_name,
            source_rows=read_jsonl(candidates_path),
            source_kind=source_kind,
            max_items=max_items,
            seed_items=seed_items,
        )
        return Table3ItemsForRun(
            source_run_dir=run_dir,
            source_kind=source_kind,
            source_path=candidates_path,
            source_sha256=_sha256_path(candidates_path),
            seed_source_path=final_items_path if seed_items else None,
            seed_source_sha256=_sha256_path(final_items_path) if seed_items else None,
            seed_item_count=seed_item_count,
            fill_item_count=fill_item_count,
            company_key=company_key,
            company_name=company_name,
            items=items,
        )

    if _nonempty_path(final_items_path):
        items, seed_item_count, fill_item_count = _select_table3_items(
            company_key=company_key,
            company_name=company_name,
            source_rows=read_jsonl(final_items_path),
            source_kind="benchmark_items_final",
            max_items=max_items,
        )
        return Table3ItemsForRun(
            source_run_dir=run_dir,
            source_kind="benchmark_items_final",
            source_path=final_items_path,
            source_sha256=_sha256_path(final_items_path),
            seed_source_path=None,
            seed_source_sha256=None,
            seed_item_count=seed_item_count,
            fill_item_count=fill_item_count,
            company_key=company_key,
            company_name=company_name,
            items=items,
        )
    raise FileNotFoundError(f"No Table3-ready COPAL source artifact found under: {run_dir}")


def process_ready_table3_runs(
    *,
    output_experiment_dir: Path,
    source_experiment_dirs: list[Path],
    prompts_by_key: dict[str, object],
    downstream_client: LLMClient,
    judge_client: LLMClient,
    judge_model: str,
    eval_models: list[str],
    max_items_per_company: int,
    live_max_workers: int,
    max_companies: int,
) -> int:
    if live_max_workers < 1:
        raise ValueError("live_max_workers must be positive")
    if max_items_per_company < 1:
        raise ValueError("max_items_per_company must be positive")
    if max_companies < 0:
        raise ValueError("max_companies must be non-negative")
    if not eval_models:
        raise ValueError("eval_models must include at least one model")
    ensure_directory(output_experiment_dir / "company_runs")

    processed = 0
    for source_run_dir in discover_ready_table2_copal_runs(source_experiment_dirs):
        table3_input = load_table3_items_for_run(
            run_dir=source_run_dir,
            max_items=max_items_per_company,
        )
        output_run_dir = ensure_directory(output_experiment_dir / "company_runs" / source_run_dir.name)
        validate_or_write_table3_company_manifest(
            run_dir=output_run_dir,
            table3_input=table3_input,
            eval_models=eval_models,
            judge_model=judge_model,
            max_items_per_company=max_items_per_company,
        )
        write_jsonl(output_run_dir / "selected_items.jsonl", table3_input.items)
        prefill_table3_from_source_copal_evaluation(
            output_run_dir=output_run_dir,
            source_run_dir=source_run_dir,
            selected_items=table3_input.items,
            eval_models=eval_models,
        )
        if company_table3_is_complete(
            run_dir=output_run_dir,
            selected_items=table3_input.items,
            eval_models=eval_models,
        ):
            if not (output_run_dir / "table3_company_summary.json").exists():
                _write_completed_company_summary(
                    output_run_dir=output_run_dir,
                    source_run_dir=source_run_dir,
                    table3_input=table3_input,
                    eval_models=eval_models,
                    judge_model=judge_model,
                    evaluation_summary=_load_existing_evaluation_summary(output_run_dir),
                )
            continue
        prompt_record = prompts_by_key.get(table3_input.company_key)
        if prompt_record is None:
            raise KeyError(f"Missing system prompt for company_key={table3_input.company_key}")
        evaluation_summary = run_pilot_evaluation(
            evaluation_dir=output_run_dir / "evaluation",
            benchmark_items=table3_input.items,
            system_prompt=_system_prompt_text(prompt_record),
            eval_models=eval_models,
            downstream_client=downstream_client,
            judge_client=judge_client,
            judge_model=judge_model,
            live_max_workers=live_max_workers,
        )
        _write_completed_company_summary(
            output_run_dir=output_run_dir,
            source_run_dir=source_run_dir,
            table3_input=table3_input,
            eval_models=eval_models,
            judge_model=judge_model,
            evaluation_summary=evaluation_summary,
        )
        processed += 1
        if max_companies and processed >= max_companies:
            break
    return processed


def _write_completed_company_summary(
    *,
    output_run_dir: Path,
    source_run_dir: Path,
    table3_input: Table3ItemsForRun,
    eval_models: list[str],
    judge_model: str,
    evaluation_summary: dict[str, object],
) -> None:
    company_summary = {
        "source_run_id": source_run_dir.name,
        "source_run_dir": str(source_run_dir),
        "source_kind": table3_input.source_kind,
        "source_path": str(table3_input.source_path),
        "source_sha256": table3_input.source_sha256,
        "seed_source_path": str(table3_input.seed_source_path) if table3_input.seed_source_path else None,
        "seed_source_sha256": table3_input.seed_source_sha256,
        "seed_item_count": table3_input.seed_item_count,
        "fill_item_count": table3_input.fill_item_count,
        "company_key": table3_input.company_key,
        "company_name": table3_input.company_name,
        "selected_item_count": len(table3_input.items),
        "selected_item_ids": [str(item["item_id"]) for item in table3_input.items],
        "eval_models": list(eval_models),
        "judge_model": judge_model,
        "evaluation_summary": evaluation_summary,
    }
    write_json(output_run_dir / "table3_company_summary.json", company_summary)


def _load_existing_evaluation_summary(output_run_dir: Path) -> dict[str, object]:
    summary_path = output_run_dir / "evaluation" / "pilot_evaluation_summary.json"
    if summary_path.exists():
        return read_json(summary_path)
    judgments_path = output_run_dir / "evaluation" / "response_judgments.jsonl"
    return aggregate_table3_judgments(read_jsonl(judgments_path))


def company_table3_is_complete(
    *,
    run_dir: Path,
    selected_items: list[dict[str, object]],
    eval_models: list[str],
) -> bool:
    judgments_path = run_dir / "evaluation" / "response_judgments.jsonl"
    if not judgments_path.exists():
        return False
    expected = _expected_response_ids(selected_items=selected_items, eval_models=eval_models)
    observed = {str(row.get("response_id", "")) for row in read_jsonl(judgments_path)}
    unexpected = observed - expected
    if unexpected:
        raise ValueError(f"Table3 checkpoint has unexpected response ids in {judgments_path}: {sorted(unexpected)}")
    return expected <= observed


def validate_or_write_table3_company_manifest(
    *,
    run_dir: Path,
    table3_input: Table3ItemsForRun,
    eval_models: list[str],
    judge_model: str,
    max_items_per_company: int,
) -> dict[str, object]:
    ensure_directory(run_dir)
    manifest_path = run_dir / TABLE3_COMPANY_MANIFEST
    manifest = {
        "source_run_id": table3_input.source_run_dir.name,
        "source_run_dir": str(table3_input.source_run_dir),
        "source_kind": table3_input.source_kind,
        "source_path": str(table3_input.source_path),
        "source_sha256": table3_input.source_sha256,
        "seed_source_path": str(table3_input.seed_source_path) if table3_input.seed_source_path else None,
        "seed_source_sha256": table3_input.seed_source_sha256,
        "seed_item_count": table3_input.seed_item_count,
        "fill_item_count": table3_input.fill_item_count,
        "company_key": table3_input.company_key,
        "company_name": table3_input.company_name,
        "selected_item_ids": [str(item["item_id"]) for item in table3_input.items],
        "eval_models": list(eval_models),
        "judge_model": judge_model,
        "max_items_per_company": max_items_per_company,
    }
    if manifest_path.exists():
        existing = read_json(manifest_path)
        if existing != manifest:
            raise ValueError(
                f"Existing Table3 manifest does not match requested configuration: {manifest_path}"
            )
        return existing
    write_json(manifest_path, manifest)
    return manifest


def aggregate_table3_outputs(*, experiment_id: str, output_experiment_dir: Path) -> dict[str, object]:
    company_runs_dir = output_experiment_dir / "company_runs"
    judgment_rows: list[dict[str, object]] = []
    completed_run_ids: list[str] = []
    if company_runs_dir.exists():
        for run_dir in sorted(path for path in company_runs_dir.iterdir() if path.is_dir()):
            if not (run_dir / "table3_company_summary.json").exists():
                continue
            judgments_path = run_dir / "evaluation" / "response_judgments.jsonl"
            if not judgments_path.exists():
                continue
            completed_run_ids.append(run_dir.name)
            for row in read_jsonl(judgments_path):
                judgment_rows.append({**row, "table3_run_id": run_dir.name})
    return {
        "experiment_id": experiment_id,
        "completed_company_count": len(completed_run_ids),
        "completed_run_ids": completed_run_ids,
        **aggregate_table3_judgments(judgment_rows),
    }


def aggregate_table3_judgments(judgments: list[dict[str, object]]) -> dict[str, object]:
    by_model: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_pattern: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_facet: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in judgments:
        by_model[str(row["response_model"])].append(row)
        by_pattern[str(row["signature"])].append(row)
        by_facet[str(row.get("target_facet", row.get("facet", "")))].append(row)
    return {
        "judgment_count": len(judgments),
        "overall": _aggregate_group(judgments),
        "by_model": {key: _aggregate_group(rows) for key, rows in sorted(by_model.items())},
        "by_pattern": {key: _aggregate_group(rows) for key, rows in sorted(by_pattern.items())},
        "by_facet": {key: _aggregate_group(rows) for key, rows in sorted(by_facet.items())},
    }


def _select_table3_items(
    *,
    company_key: str,
    company_name: str,
    source_rows: list[dict[str, object]],
    source_kind: str,
    max_items: int,
    seed_items: list[dict[str, object]] | None = None,
) -> tuple[list[dict[str, object]], int, int]:
    selected_seed_items = _select_seed_items(seed_items or [], max_items=max_items)
    selected_seed_ids = {_row_id(item) for item in selected_seed_items}
    fill_budget = max_items - len(selected_seed_items)

    if source_kind in {
        "candidate_queries_labeled",
        "candidate_queries_labeled_seeded_by_benchmark_items_final",
    }:
        candidate_fill_rows = [
            row for row in source_rows
            if _row_id(row) not in selected_seed_ids
        ]
        selected_rows = (
            _coverage_aware_select(rows=candidate_fill_rows, max_items=fill_budget)
            if fill_budget > 0 and candidate_fill_rows
            else []
        )
        selected = [
            {
                "query_id": str(row["query_id"]),
                "selection_rank": index,
                "challenge_score": _challenge_score(row),
                "screening_rationale": _screening_rationale(row),
            }
            for index, row in enumerate(selected_rows, start=1)
        ]
        items = build_pilot_benchmark_items(
            company_key=company_key,
            company_name=company_name,
            queries=source_rows,
            selected=selected,
        )
        items = [dict(item) for item in selected_seed_items] + items
    elif source_kind == "benchmark_items_final":
        selected_rows = _coverage_aware_select(rows=source_rows, max_items=max_items)
        items = [dict(row) for row in selected_rows]
    else:
        raise ValueError(f"Unsupported Table3 source kind: {source_kind}")

    normalized_items: list[dict[str, object]] = []
    for index, item in enumerate(items, start=1):
        normalized = dict(item)
        if "item_id" not in normalized:
            normalized["item_id"] = normalized["query_id"]
        metadata = dict(normalized.get("selection_metadata", {}))
        metadata["table3_selection_rank"] = index
        metadata["table3_source_kind"] = source_kind
        metadata["table3_selection_role"] = (
            "seed_final_item"
            if _row_id(normalized) in selected_seed_ids
            else "candidate_fill"
        )
        normalized["selection_metadata"] = metadata
        normalized_items.append(normalized)
    return normalized_items, len(selected_seed_items), len(normalized_items) - len(selected_seed_items)


def prefill_table3_from_source_copal_evaluation(
    *,
    output_run_dir: Path,
    source_run_dir: Path,
    selected_items: list[dict[str, object]],
    eval_models: list[str],
) -> dict[str, object]:
    evaluation_dir = ensure_directory(output_run_dir / "evaluation")
    expected_order = [
        f"{item['item_id']}::{model}"
        for item in selected_items
        for model in eval_models
    ]
    expected_response_ids = set(expected_order)
    responses_path = evaluation_dir / "chatbot_responses.jsonl"
    judgments_path = evaluation_dir / "response_judgments.jsonl"
    source_evaluation_dir = source_run_dir / "variants" / "copal" / "evaluation"
    source_responses_path = source_evaluation_dir / "chatbot_responses.jsonl"
    source_judgments_path = source_evaluation_dir / "response_judgments.jsonl"

    existing_responses = _read_response_id_rows(
        responses_path,
        expected_response_ids=expected_response_ids,
        reject_unexpected=True,
    )
    existing_judgments = _read_response_id_rows(
        judgments_path,
        expected_response_ids=expected_response_ids,
        reject_unexpected=True,
    )
    source_responses = _read_response_id_rows(
        source_responses_path,
        expected_response_ids=expected_response_ids,
        reject_unexpected=False,
    )
    source_judgments = _read_response_id_rows(
        source_judgments_path,
        expected_response_ids=expected_response_ids,
        reject_unexpected=False,
    )
    reusable_response_ids = [
        response_id
        for response_id in expected_order
        if response_id in source_responses and response_id in source_judgments
    ]

    prefilled_response_count = 0
    prefilled_judgment_count = 0
    for response_id in reusable_response_ids:
        if response_id not in existing_responses:
            existing_responses[response_id] = source_responses[response_id]
            prefilled_response_count += 1
        if response_id not in existing_judgments:
            existing_judgments[response_id] = source_judgments[response_id]
            prefilled_judgment_count += 1

    if existing_responses:
        write_jsonl(
            responses_path,
            [
                existing_responses[response_id]
                for response_id in expected_order
                if response_id in existing_responses
            ],
        )
    if existing_judgments:
        write_jsonl(
            judgments_path,
            [
                existing_judgments[response_id]
                for response_id in expected_order
                if response_id in existing_judgments
            ],
        )

    summary = {
        "source_evaluation_dir": str(source_evaluation_dir),
        "expected_response_count": len(expected_order),
        "source_matching_pair_count": len(reusable_response_ids),
        "prefilled_response_count": prefilled_response_count,
        "prefilled_judgment_count": prefilled_judgment_count,
        "existing_response_count": len(existing_responses),
        "existing_judgment_count": len(existing_judgments),
    }
    write_json(evaluation_dir / "table2_copal_prefill_summary.json", summary)
    return summary


def _select_seed_items(
    seed_items: list[dict[str, object]],
    *,
    max_items: int,
) -> list[dict[str, object]]:
    if not seed_items:
        return []
    if len(seed_items) <= max_items:
        return [dict(item) for item in seed_items]
    return [dict(item) for item in _coverage_aware_select(rows=seed_items, max_items=max_items)]


def _read_response_id_rows(
    path: Path,
    *,
    expected_response_ids: set[str],
    reject_unexpected: bool,
) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    rows_by_id: dict[str, dict[str, object]] = {}
    for row in read_jsonl(path):
        response_id = str(row.get("response_id", ""))
        if response_id not in expected_response_ids:
            if reject_unexpected:
                raise ValueError(f"Existing Table3 checkpoint has unexpected response id in {path}: {response_id}")
            continue
        rows_by_id[response_id] = row
    return rows_by_id


def _coverage_aware_select(*, rows: list[dict[str, object]], max_items: int) -> list[dict[str, object]]:
    if max_items < 1:
        raise ValueError("max_items must be positive")
    if not rows:
        raise ValueError("Table3 source artifact contains no rows")

    rows_by_cell: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    rows_by_pattern: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        pattern = _pattern(row)
        rows_by_pattern[pattern].append(row)
        for facet in _facets(row):
            rows_by_cell[(pattern, facet)].append(row)
    for bucket in [*rows_by_cell.values(), *rows_by_pattern.values()]:
        bucket.sort(key=_ranking_key)

    selected: list[dict[str, object]] = []
    selected_ids: set[str] = set()

    def add(row: dict[str, object]) -> None:
        row_id = _row_id(row)
        if row_id in selected_ids or len(selected) >= max_items:
            return
        selected_ids.add(row_id)
        selected.append(row)

    for pattern in DEFAULT_SIGNATURES:
        for facet in DEFAULT_FACETS[pattern]:
            bucket = rows_by_cell.get((pattern, facet), [])
            if bucket:
                add(bucket[0])

    for pattern in DEFAULT_SIGNATURES:
        for row in rows_by_pattern.get(pattern, []):
            add(row)

    for row in sorted(rows, key=_ranking_key):
        add(row)
    return selected


def _aggregate_group(rows: list[dict[str, object]]) -> dict[str, object]:
    error_count = sum(1 for row in rows if not _overall_correct(row))
    count = len(rows)
    return {
        "judgment_count": count,
        "error_count": error_count,
        "error_rate": error_count / count if count else 0.0,
        "policy_handling_score": 1.0 - (error_count / count) if count else 0.0,
    }


def _overall_correct(row: dict[str, object]) -> bool:
    value = row.get("overall_correct")
    if not isinstance(value, bool):
        raise TypeError(f"Table3 judgment overall_correct must be bool for response_id={row.get('response_id')}")
    return value


def _expected_response_ids(*, selected_items: list[dict[str, object]], eval_models: list[str]) -> set[str]:
    return {
        f"{item['item_id']}::{model}"
        for item in selected_items
        for model in eval_models
    }


def _candidate_path(run_dir: Path) -> Path:
    return run_dir / "variants" / "copal" / TABLE3_CANDIDATE_FILENAME


def _final_items_path(run_dir: Path) -> Path:
    return run_dir / "variants" / "copal" / TABLE3_FINAL_FILENAME


def _nonempty_path(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8") as handle:
        return any(line.strip() for line in handle)


def _sha256_path(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _system_prompt_text(prompt_record: object) -> str:
    if isinstance(prompt_record, str):
        return prompt_record
    system_prompt = getattr(prompt_record, "system_prompt", None)
    if not isinstance(system_prompt, str) or not system_prompt:
        raise TypeError("prompt record must be a system prompt string or expose .system_prompt")
    return system_prompt


def _pattern(row: dict[str, object]) -> str:
    for key in ("relation_pattern", "signature"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    raise KeyError(f"Table3 row is missing relation_pattern/signature: {_row_id(row)}")


def _facets(row: dict[str, object]) -> list[str]:
    facets: list[str] = []
    for key in ("target_facets", "coverage_set"):
        value = row.get(key)
        if isinstance(value, list):
            facets.extend(str(item) for item in value if str(item))
    for key in ("target_facet", "facet"):
        value = row.get(key)
        if isinstance(value, str) and value:
            facets.append(value)
    deduped: list[str] = []
    seen: set[str] = set()
    for facet in facets:
        if facet in seen:
            continue
        seen.add(facet)
        deduped.append(facet)
    if not deduped:
        raise KeyError(f"Table3 row is missing target facet metadata: {_row_id(row)}")
    return deduped


def _row_id(row: dict[str, object]) -> str:
    for key in ("query_id", "item_id"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    raise KeyError("Table3 row is missing query_id/item_id")


def _ranking_key(row: dict[str, object]) -> tuple[float, int, str]:
    metadata = row.get("selection_metadata", {})
    selection_rank = 10_000
    if isinstance(metadata, dict) and "selection_rank" in metadata:
        selection_rank = int(metadata["selection_rank"])
    return (-_challenge_score(row), selection_rank, _row_id(row))


def _challenge_score(row: dict[str, object]) -> float:
    metadata = row.get("selection_metadata", {})
    if isinstance(metadata, dict) and "challenge_score" in metadata:
        return float(metadata["challenge_score"])
    value = row.get("challenge_score", 0.0)
    return float(value) if value is not None else 0.0


def _screening_rationale(row: dict[str, object]) -> str:
    metadata = row.get("selection_metadata", {})
    if isinstance(metadata, dict) and "screening_rationale" in metadata:
        return str(metadata["screening_rationale"])
    return str(row.get("screening_rationale", "Selected by deterministic Table3 coverage-aware selection."))
