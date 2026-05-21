from __future__ import annotations

import argparse
import time
from pathlib import Path

from copal.config import DEFAULT_PROMPTS_PATH, load_model_names
from copal.data_sources import load_system_prompts
from copal.io import ensure_directory, write_json
from copal.llm import build_live_client, live_client_runtime_metadata
from copal.table3_model_eval import aggregate_table3_outputs, process_ready_table3_runs

DEFAULT_TABLE3_EXPERIMENT_ID = "table3_model_eval_30c_10model_seed12_20260514"
DEFAULT_TABLE2_SOURCE_EXPERIMENT_IDS = (
    "table2_ablation_30c_shard0_20260513",
    "table2_ablation_30c_shard1_20260513",
    "table2_ablation_30c_shard2_20260513",
)
DEFAULT_TABLE3_JUDGE_MODEL = "gemini-3-flash-preview"


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run resumable Table 3 multi-model evaluation from completed Table 2 COPAL artifacts."
    )
    parser.add_argument("--experiment-id", default=DEFAULT_TABLE3_EXPERIMENT_ID)
    parser.add_argument("--source-experiment-ids", default=",".join(DEFAULT_TABLE2_SOURCE_EXPERIMENT_IDS))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--prompts-path", type=Path, default=DEFAULT_PROMPTS_PATH)
    parser.add_argument("--model-name-path", type=Path, default=Path("model_name.json"))
    parser.add_argument("--eval-models", default="")
    parser.add_argument("--judge-model", default=DEFAULT_TABLE3_JUDGE_MODEL)
    parser.add_argument("--max-items-per-company", type=int, default=30)
    parser.add_argument("--live-max-workers", type=int, default=1)
    parser.add_argument("--max-companies", type=int, default=0)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--target-company-count", type=int, default=30)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-idle-polls", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_items_per_company < 1:
        raise ValueError("--max-items-per-company must be positive")
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

    source_experiment_ids = parse_csv(args.source_experiment_ids)
    if not source_experiment_ids:
        raise ValueError("--source-experiment-ids must include at least one experiment id")
    eval_models = parse_csv(args.eval_models) if args.eval_models else list(load_model_names(args.model_name_path))
    if not eval_models:
        raise ValueError("No evaluation models configured")

    output_experiment_dir = ensure_directory(args.runs_dir / "experiments" / args.experiment_id)
    cache_dir = ensure_directory(output_experiment_dir / "llm_cache")
    source_experiment_dirs = [
        args.runs_dir / "experiments" / experiment_id
        for experiment_id in source_experiment_ids
    ]
    manifest = {
        "experiment_id": args.experiment_id,
        "source_experiment_ids": source_experiment_ids,
        "source_experiment_dirs": [str(path) for path in source_experiment_dirs],
        "models": {
            "eval_models": eval_models,
            "judge_model": args.judge_model,
        },
        "budgets": {
            "max_items_per_company": args.max_items_per_company,
            "live_max_workers": args.live_max_workers,
            "max_companies": args.max_companies,
            "watch": args.watch,
            "target_company_count": args.target_company_count,
            "poll_seconds": args.poll_seconds,
            "max_idle_polls": args.max_idle_polls,
        },
        "live_client": live_client_runtime_metadata(),
    }
    write_json(output_experiment_dir / "table3_manifest.json", manifest)

    prompts_by_key = {prompt.company_key: prompt for prompt in load_system_prompts(args.prompts_path)}
    judge_client = build_live_client(cache_dir=cache_dir / "json")
    downstream_client = build_live_client(cache_dir=cache_dir / "chat", response_format_override=None)

    processed_total = 0
    idle_polls = 0
    while True:
        processed_now = process_ready_table3_runs(
            output_experiment_dir=output_experiment_dir,
            source_experiment_dirs=source_experiment_dirs,
            prompts_by_key=prompts_by_key,
            downstream_client=downstream_client,
            judge_client=judge_client,
            judge_model=args.judge_model,
            eval_models=eval_models,
            max_items_per_company=args.max_items_per_company,
            live_max_workers=args.live_max_workers,
            max_companies=args.max_companies,
        )
        processed_total += processed_now
        aggregate = aggregate_table3_outputs(
            experiment_id=args.experiment_id,
            output_experiment_dir=output_experiment_dir,
        )
        write_json(output_experiment_dir / "table3_summary.json", aggregate)

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


if __name__ == "__main__":
    raise SystemExit(main())
