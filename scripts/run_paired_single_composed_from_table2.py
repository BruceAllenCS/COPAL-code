from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

from copal.config import DEFAULT_PROMPTS_PATH
from copal.data_sources import load_system_prompts
from copal.experiment_analysis import summarize_paired_single_composed
from copal.fast_pilot import run_paired_single_policy_evaluation
from copal.io import ensure_directory, read_json, read_jsonl, write_json
from copal.llm import build_live_client, live_client_runtime_metadata


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def discover_ready_table2_copal_runs(source_experiment_dirs: list[Path]) -> list[Path]:
    ready: list[Path] = []
    for experiment_dir in source_experiment_dirs:
        company_runs_dir = experiment_dir / "company_runs"
        if not company_runs_dir.exists():
            continue
        for run_dir in sorted(path for path in company_runs_dir.iterdir() if path.is_dir()):
            if _table2_copal_run_is_ready(run_dir):
                ready.append(run_dir)
    return ready


def _table2_copal_run_is_ready(run_dir: Path) -> bool:
    required_paths = (
        run_dir / "shared_grounding" / "grounded_clauses.jsonl",
        run_dir / "variants" / "copal" / "benchmark_items_final.jsonl",
        run_dir / "variants" / "copal" / "evaluation" / "response_judgments.jsonl",
        run_dir / "variants" / "copal" / "table2_variant_summary.json",
    )
    return all(path.exists() for path in required_paths)


def paired_run_is_complete(run_dir: Path, eval_models: list[str]) -> bool:
    paired_dir = run_dir / "paired_single_policy"
    return (
        (paired_dir / "paired_single_composed_summary.json").exists()
        and _judgments_cover_expected_responses(
            run_dir / "variants" / "copal" / "evaluation" / "response_judgments.jsonl",
            run_dir / "variants" / "copal" / "benchmark_items_final.jsonl",
            eval_models,
        )
        and _judgments_cover_expected_responses(
            paired_dir / "response_judgments.jsonl",
            paired_dir / "single_policy_projection_items.jsonl",
            eval_models,
        )
    )


def _judgments_cover_expected_responses(
    judgments_path: Path,
    items_path: Path,
    eval_models: list[str],
) -> bool:
    if not judgments_path.exists() or not items_path.exists():
        return False
    expected_ids = {
        f"{item['item_id']}::{model}"
        for item in read_jsonl(items_path)
        for model in eval_models
    }
    if not expected_ids:
        return False
    present_ids = {
        str(row.get("response_id", "")).strip()
        for row in read_jsonl(judgments_path)
        if str(row.get("response_id", "")).strip()
    }
    return expected_ids.issubset(present_ids)


def _filter_judgments_by_models(rows: list[dict[str, object]], eval_models: list[str]) -> list[dict[str, object]]:
    required_models = {model.strip() for model in eval_models if model.strip()}
    return [row for row in rows if str(row.get("response_model", "")).strip() in required_models]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run single-policy vs composed-policy control from completed Table 2 COPAL runs."
    )
    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--source-experiment-ids", required=True)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--prompts-path", type=Path, default=DEFAULT_PROMPTS_PATH)
    parser.add_argument("--eval-models", default="Doubao-Seed-2.0-pro")
    parser.add_argument("--judge-model", default="gemini-3-flash-preview")
    parser.add_argument("--live-max-workers", type=int, default=2)
    parser.add_argument("--max-companies", type=int, default=0)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--target-company-count", type=int, default=30)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-idle-polls", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.live_max_workers < 1:
        raise ValueError("--live-max-workers must be positive")
    if args.max_companies < 0:
        raise ValueError("--max-companies must be non-negative")
    if args.target_company_count < 1:
        raise ValueError("--target-company-count must be positive")
    if args.poll_seconds < 1:
        raise ValueError("--poll-seconds must be positive")
    if args.max_idle_polls < 1:
        raise ValueError("--max-idle-polls must be positive")
    eval_models = parse_csv(args.eval_models)
    if not eval_models:
        raise ValueError("--eval-models must include at least one model")

    experiment_id = args.experiment_id or f"paired_single_composed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_dir = ensure_directory(args.runs_dir / "experiments" / experiment_id)
    cache_dir = ensure_directory(experiment_dir / "llm_cache")
    source_experiment_dirs = [
        args.runs_dir / "experiments" / experiment_id
        for experiment_id in parse_csv(args.source_experiment_ids)
    ]
    prompts_by_key = {prompt.company_key: prompt for prompt in load_system_prompts(args.prompts_path)}
    live_client = build_live_client(cache_dir=cache_dir / "json")
    downstream_client = build_live_client(cache_dir=cache_dir / "chat", response_format_override=None)
    manifest = {
        "experiment_id": experiment_id,
        "source_experiment_ids": parse_csv(args.source_experiment_ids),
        "models": {
            "eval_models": eval_models,
            "judge_model": args.judge_model,
        },
        "budgets": {
            "live_max_workers": args.live_max_workers,
            "max_companies": args.max_companies,
            "watch": args.watch,
            "target_company_count": args.target_company_count,
            "poll_seconds": args.poll_seconds,
            "max_idle_polls": args.max_idle_polls,
        },
        "live_client": live_client_runtime_metadata(),
    }
    write_json(experiment_dir / "paired_manifest.json", manifest)

    processed_total = 0
    idle_polls = 0
    while True:
        processed_now = process_ready_runs(
            source_experiment_dirs=source_experiment_dirs,
            prompts_by_key=prompts_by_key,
            downstream_client=downstream_client,
            judge_client=live_client,
            judge_model=args.judge_model,
            eval_models=eval_models,
            live_max_workers=args.live_max_workers,
            max_companies=args.max_companies,
        )
        processed_total += processed_now
        aggregate = aggregate_paired_outputs(
            experiment_id=experiment_id,
            source_experiment_dirs=source_experiment_dirs,
            eval_models=eval_models,
        )
        write_json(experiment_dir / "paired_single_composed_summary.json", aggregate)

        completed_company_count = int(aggregate["completed_company_count"])
        if not args.watch:
            break
        if completed_company_count >= args.target_company_count:
            break
        if args.max_companies and processed_total >= args.max_companies:
            break
        if processed_now == 0:
            idle_polls += 1
        else:
            idle_polls = 0
        if idle_polls >= args.max_idle_polls:
            break
        time.sleep(args.poll_seconds)
    return 0


def process_ready_runs(
    *,
    source_experiment_dirs: list[Path],
    prompts_by_key: dict[str, object],
    downstream_client: object,
    judge_client: object,
    judge_model: str,
    eval_models: list[str],
    live_max_workers: int,
    max_companies: int,
) -> int:
    processed = 0
    for run_dir in discover_ready_table2_copal_runs(source_experiment_dirs):
        if not _judgments_cover_expected_responses(
            run_dir / "variants" / "copal" / "evaluation" / "response_judgments.jsonl",
            run_dir / "variants" / "copal" / "benchmark_items_final.jsonl",
            eval_models,
        ):
            continue
        if paired_run_is_complete(run_dir, eval_models):
            continue
        selected_company = read_json(run_dir / "selected_company.json")
        company_key = str(selected_company["company_key"])
        prompt_record = prompts_by_key.get(company_key)
        if prompt_record is None:
            raise KeyError(f"Missing system prompt for company_key={company_key}")
        benchmark_items = read_jsonl(run_dir / "variants" / "copal" / "benchmark_items_final.jsonl")
        grounded_rows = read_jsonl(run_dir / "shared_grounding" / "grounded_clauses.jsonl")
        composed_judgments = read_jsonl(
            run_dir / "variants" / "copal" / "evaluation" / "response_judgments.jsonl"
        )
        run_paired_single_policy_evaluation(
            paired_dir=run_dir / "paired_single_policy",
            benchmark_items=benchmark_items,
            grounded_rows=grounded_rows,
            composed_judgments=composed_judgments,
            system_prompt=str(prompt_record.system_prompt),
            eval_models=eval_models,
            downstream_client=downstream_client,
            judge_client=judge_client,
            judge_model=judge_model,
            live_max_workers=live_max_workers,
        )
        processed += 1
        if max_companies and processed >= max_companies:
            break
    return processed


def aggregate_paired_outputs(
    *,
    experiment_id: str,
    source_experiment_dirs: list[Path],
    eval_models: list[str],
) -> dict[str, object]:
    composed_judgments: list[dict[str, object]] = []
    projection_judgments: list[dict[str, object]] = []
    completed_run_ids: list[str] = []
    ready_count = 0
    for run_dir in discover_ready_table2_copal_runs(source_experiment_dirs):
        ready_count += 1
        paired_dir = run_dir / "paired_single_policy"
        if not paired_run_is_complete(run_dir, eval_models):
            continue
        completed_run_ids.append(run_dir.name)
        composed_judgments.extend(
            _filter_judgments_by_models(
                read_jsonl(run_dir / "variants" / "copal" / "evaluation" / "response_judgments.jsonl"),
                eval_models,
            )
        )
        projection_judgments.extend(
            _filter_judgments_by_models(read_jsonl(paired_dir / "response_judgments.jsonl"), eval_models)
        )
    paired_summary = (
        summarize_paired_single_composed(
            composed_judgments=composed_judgments,
            projection_judgments=projection_judgments,
        )
        if projection_judgments
        else {"paired_model_results": []}
    )
    return {
        "experiment_id": experiment_id,
        "ready_company_count": ready_count,
        "completed_company_count": len(completed_run_ids),
        "completed_run_ids": completed_run_ids,
        "composed_judgment_count": len(composed_judgments),
        "single_policy_projection_judgment_count": len(projection_judgments),
        **paired_summary,
    }


if __name__ == "__main__":
    raise SystemExit(main())
