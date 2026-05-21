from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from copal.config import DEFAULT_POLICIES_PATH, DEFAULT_PROMPTS_PATH
from copal.data_sources import load_company_worlds, load_system_prompts
from copal.fast_pilot_ablation import summarize_fast_pilot_ablation
from copal.fast_pilot import (
    build_pilot_benchmark_items,
    run_batch_composition,
    run_batch_grounding,
    run_batch_query_generation,
    run_paired_single_policy_evaluation,
    run_pilot_evaluation,
    run_query_screening,
    summarize_pilot_judgments,
)
from copal.io import ensure_directory, read_jsonl, write_json, write_jsonl
from copal.llm import build_live_client, live_client_runtime_metadata
from copal.models import CompanyWorld


DEFAULT_EVAL_MODELS = (
    "Doubao-Seed-2.0-pro",
    "gemini-3.1-pro-preview",
)


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def select_worlds(worlds: list[CompanyWorld], *, company_limit: int, sample_strategy: str) -> list[CompanyWorld]:
    if company_limit < 1:
        raise ValueError("company_limit must be positive")
    if sample_strategy == "first":
        return worlds[:company_limit]
    if sample_strategy == "one-per-industry":
        selected: list[CompanyWorld] = []
        seen: set[str] = set()
        for world in worlds:
            if world.industry in seen:
                continue
            selected.append(world)
            seen.add(world.industry)
            if len(selected) == company_limit:
                return selected
        raise ValueError(f"Could only select {len(selected)} one-per-industry companies")
    raise ValueError(f"Unsupported sample strategy: {sample_strategy}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a small COPAL fast-generation pilot.")
    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--company-limit", type=int, default=2)
    parser.add_argument("--sample-strategy", choices=("first", "one-per-industry"), default="one-per-industry")
    parser.add_argument("--policies-path", type=Path, default=DEFAULT_POLICIES_PATH)
    parser.add_argument("--prompts-path", type=Path, default=DEFAULT_PROMPTS_PATH)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--grounding-model", default="gemini-3-flash-preview")
    parser.add_argument("--composition-model", default="gemini-3-flash-preview")
    parser.add_argument("--query-model", default="gpt-5.5")
    parser.add_argument("--screening-model", default="gemini-3-flash-preview")
    parser.add_argument("--judge-model", default="gemini-3-flash-preview")
    parser.add_argument("--eval-models", default=",".join(DEFAULT_EVAL_MODELS))
    parser.add_argument("--max-compositions-per-company", type=int, default=8)
    parser.add_argument("--query-variants-per-facet", type=int, default=2)
    parser.add_argument("--selected-per-company", type=int, default=30)
    parser.add_argument("--live-max-workers", type=int, default=6)
    parser.add_argument("--stop-after", choices=("screening", "evaluation"), default="evaluation")
    parser.add_argument("--run-fast-ablation", action="store_true")
    parser.add_argument("--run-paired-single-policy", action="store_true")
    return parser


def aggregate_fast_ablation_summaries(company_summaries: list[dict[str, object]]) -> dict[str, object]:
    rows_by_id: dict[str, list[dict[str, object]]] = {}
    for company_summary in company_summaries:
        summary = company_summary.get("fast_ablation_summary")
        if not isinstance(summary, dict):
            continue
        for row in summary.get("variants", []):
            if not isinstance(row, dict):
                continue
            ablation_id = str(row["ablation_id"])
            rows_by_id.setdefault(ablation_id, []).append(row)
    return {
        "company_count": sum(1 for summary in company_summaries if isinstance(summary.get("fast_ablation_summary"), dict)),
        "variants": [
            _aggregate_fast_ablation_variant(ablation_id=ablation_id, rows=rows)
            for ablation_id, rows in rows_by_id.items()
        ],
    }


def _aggregate_fast_ablation_variant(*, ablation_id: str, rows: list[dict[str, object]]) -> dict[str, object]:
    reportable_rows = [row for row in rows if row.get("reportable", True) is not False and row.get("status") != "not_run"]
    aggregate: dict[str, object] = {
        "ablation_id": ablation_id,
        "company_count": len(rows),
        "reportable_company_count": len(reportable_rows),
        "status": "computed" if reportable_rows else "not_run",
        "mean_cell_count": _mean_numeric(reportable_rows, "cell_count"),
        "mean_target_facet_coverage": _mean_numeric(reportable_rows, "target_facet_coverage"),
        "mean_pattern_coverage": _mean_numeric(reportable_rows, "pattern_coverage"),
        "mean_cpq": _mean_numeric(reportable_rows, "cpq"),
        "mean_vir": _mean_numeric(reportable_rows, "vir"),
        "mean_contract_valid_rate": _mean_numeric(reportable_rows, "contract_valid_rate"),
        "mean_unique_clause_set_count": _mean_numeric(reportable_rows, "unique_clause_set_count"),
        "mean_clause_set_diversity": _mean_numeric(reportable_rows, "clause_set_diversity"),
    }
    if not reportable_rows:
        required = sorted({str(row.get("requires_artifact", "")) for row in rows if str(row.get("requires_artifact", ""))})
        aggregate["requires_artifact"] = "; ".join(required)
    return aggregate


def _mean_numeric(rows: list[dict[str, object]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if key in row and row[key] is not None]
    if not values:
        return None
    return sum(values) / len(values)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_compositions_per_company < 1:
        raise ValueError("--max-compositions-per-company must be positive")
    if args.query_variants_per_facet < 1:
        raise ValueError("--query-variants-per-facet must be positive")
    if args.selected_per_company < 1:
        raise ValueError("--selected-per-company must be positive")
    if args.live_max_workers < 1:
        raise ValueError("--live-max-workers must be positive")
    eval_models = parse_csv(args.eval_models)
    if not eval_models:
        raise ValueError("--eval-models must include at least one model")

    experiment_id = args.experiment_id or f"fast_generation_pilot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_dir = ensure_directory(args.runs_dir / "experiments" / experiment_id)
    company_runs_dir = ensure_directory(experiment_dir / "company_runs")
    cache_dir = ensure_directory(experiment_dir / "llm_cache")

    worlds = select_worlds(
        load_company_worlds(args.policies_path),
        company_limit=args.company_limit,
        sample_strategy=args.sample_strategy,
    )
    prompts_by_key = {prompt.company_key: prompt for prompt in load_system_prompts(args.prompts_path)}
    missing_prompts = [world.company_key for world in worlds if world.company_key not in prompts_by_key]
    if missing_prompts:
        raise KeyError(f"Missing system prompts for selected companies: {missing_prompts}")

    live_client = build_live_client(cache_dir=cache_dir / "json")
    downstream_client = build_live_client(cache_dir=cache_dir / "chat", response_format_override=None)
    manifest = {
        "experiment_id": experiment_id,
        "company_limit": args.company_limit,
        "sample_strategy": args.sample_strategy,
        "selected_companies": [world.company_key for world in worlds],
        "models": {
            "grounding_model": args.grounding_model,
            "composition_model": args.composition_model,
            "query_model": args.query_model,
            "screening_model": args.screening_model,
            "judge_model": args.judge_model,
            "eval_models": eval_models,
        },
        "budgets": {
            "max_compositions_per_company": args.max_compositions_per_company,
            "query_variants_per_facet": args.query_variants_per_facet,
            "selected_per_company": args.selected_per_company,
            "live_max_workers": args.live_max_workers,
            "stop_after": args.stop_after,
            "run_fast_ablation": args.run_fast_ablation,
            "run_paired_single_policy": args.run_paired_single_policy,
        },
        "live_client": live_client_runtime_metadata(),
        "pipeline_note": (
            "Fast pilot uses construction labels for relation-pattern/facet coverage; "
            "no full coverage_judge stage is run."
        ),
    }
    write_json(experiment_dir / "pilot_manifest.json", manifest)

    all_judgments: list[dict[str, object]] = []
    company_summaries: list[dict[str, object]] = []
    for index, world in enumerate(worlds):
        run_id = f"{experiment_id}__{index:03d}"
        run_dir = ensure_directory(company_runs_dir / run_id)
        write_json(
            run_dir / "selected_company.json",
            {
                "company_key": world.company_key,
                "industry": world.industry,
                "company_name": world.company_name,
                "company_index": world.company_index,
            },
        )

        clauses = run_batch_grounding(
            stage_dir=run_dir / "grounding",
            world=world,
            client=live_client,
            model=args.grounding_model,
        )
        compositions = run_batch_composition(
            stage_dir=run_dir / "compositions",
            world=world,
            clauses=clauses,
            client=live_client,
            model=args.composition_model,
            max_compositions=args.max_compositions_per_company,
        )
        queries = run_batch_query_generation(
            stage_dir=run_dir / "query_generation",
            world=world,
            compositions=compositions,
            client=live_client,
            model=args.query_model,
            query_variants_per_facet=args.query_variants_per_facet,
            max_workers=args.live_max_workers,
        )
        selected = run_query_screening(
            stage_dir=run_dir / "query_screening",
            world=world,
            candidates=queries,
            client=live_client,
            model=args.screening_model,
            max_selected=args.selected_per_company,
        )
        benchmark_items = build_pilot_benchmark_items(
            company_key=world.company_key,
            company_name=world.company_name,
            queries=queries,
            selected=selected,
        )
        write_jsonl(run_dir / "benchmark_items_final.jsonl", benchmark_items)
        fast_ablation_summary = None
        if args.run_fast_ablation:
            fast_ablation_summary = summarize_fast_pilot_ablation(candidates=queries, selected=selected)
            fast_ablation_dir = ensure_directory(run_dir / "fast_ablation")
            write_json(fast_ablation_dir / "fast_ablation_summary.json", fast_ablation_summary)
            write_jsonl(fast_ablation_dir / "fast_ablation_variants.jsonl", list(fast_ablation_summary["variants"]))
            write_jsonl(
                fast_ablation_dir / "fast_ablation_coverage_curve.jsonl",
                list(fast_ablation_summary["coverage_curve"]),
            )
        if args.stop_after == "screening":
            company_summary = {
                "run_id": run_id,
                "company_key": world.company_key,
                "industry": world.industry,
                "company_name": world.company_name,
                "clause_count": len(clauses),
                "composition_count": len(compositions),
                "candidate_query_count": len(queries),
                "selected_item_count": len(benchmark_items),
                "evaluation_summary": None,
            }
            if fast_ablation_summary is not None:
                company_summary["fast_ablation_summary"] = fast_ablation_summary
            write_json(run_dir / "pilot_company_summary.json", company_summary)
            company_summaries.append(company_summary)
            continue
        evaluation_summary = run_pilot_evaluation(
            evaluation_dir=run_dir / "evaluation",
            benchmark_items=benchmark_items,
            system_prompt=prompts_by_key[world.company_key].system_prompt,
            eval_models=eval_models,
            downstream_client=downstream_client,
            judge_client=live_client,
            judge_model=args.judge_model,
            live_max_workers=args.live_max_workers,
        )
        judgments = read_jsonl(run_dir / "evaluation" / "response_judgments.jsonl")
        all_judgments.extend(judgments)
        paired_summary = None
        if args.run_paired_single_policy:
            paired_summary = run_paired_single_policy_evaluation(
                paired_dir=run_dir / "paired_single_policy",
                benchmark_items=benchmark_items,
                grounded_rows=clauses,
                composed_judgments=judgments,
                system_prompt=prompts_by_key[world.company_key].system_prompt,
                eval_models=eval_models,
                downstream_client=downstream_client,
                judge_client=live_client,
                judge_model=args.judge_model,
                live_max_workers=args.live_max_workers,
            )
        company_summary = {
            "run_id": run_id,
            "company_key": world.company_key,
            "industry": world.industry,
            "company_name": world.company_name,
            "clause_count": len(clauses),
            "composition_count": len(compositions),
            "candidate_query_count": len(queries),
            "selected_item_count": len(benchmark_items),
            "evaluation_summary": evaluation_summary,
        }
        if fast_ablation_summary is not None:
            company_summary["fast_ablation_summary"] = fast_ablation_summary
        if paired_summary is not None:
            company_summary["paired_single_policy_summary"] = paired_summary
        write_json(run_dir / "pilot_company_summary.json", company_summary)
        company_summaries.append(company_summary)

    aggregate = {
        "experiment_id": experiment_id,
        "company_count": len(company_summaries),
        "candidate_query_count": sum(int(row["candidate_query_count"]) for row in company_summaries),
        "selected_item_count": sum(int(row["selected_item_count"]) for row in company_summaries),
        **summarize_pilot_judgments(all_judgments),
        "company_summaries": company_summaries,
    }
    fast_ablation_aggregate = aggregate_fast_ablation_summaries(company_summaries)
    if fast_ablation_aggregate["company_count"]:
        aggregate["fast_ablation_summary"] = fast_ablation_aggregate
    paired_model_rows = [
        row
        for company_summary in company_summaries
        for paired_summary in (company_summary.get("paired_single_policy_summary"),)
        if isinstance(paired_summary, dict)
        for row in paired_summary.get("paired_model_results", [])
    ]
    if paired_model_rows:
        aggregate["paired_single_policy_model_results"] = paired_model_rows
    write_json(experiment_dir / "pilot_summary.json", aggregate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
