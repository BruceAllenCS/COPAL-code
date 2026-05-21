from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

from copal.config import DEFAULT_POLICIES_PATH, DEFAULT_PROMPTS_PATH
from copal.data_sources import load_company_worlds, load_system_prompts
from copal.fast_pilot import summarize_pilot_judgments
from copal.io import ensure_directory, read_jsonl, write_json
from copal.llm import build_live_client, live_client_runtime_metadata
from copal.models import CompanyWorld
from copal.table2_ablation import aggregate_table2_variant_summaries, run_table2_company_ablation


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def select_worlds(
    worlds: list[CompanyWorld],
    *,
    company_limit: int,
    sample_strategy: str,
    company_offset: int = 0,
) -> list[CompanyWorld]:
    if company_limit < 1:
        raise ValueError("company_limit must be positive")
    if company_offset < 0:
        raise ValueError("company_offset must be non-negative")
    if sample_strategy == "first":
        return worlds[company_offset : company_offset + company_limit]
    if sample_strategy == "one-per-industry":
        selected: list[CompanyWorld] = []
        seen: set[str] = set()
        for world in worlds:
            if world.industry in seen:
                continue
            selected.append(world)
            seen.add(world.industry)
            if len(selected) == company_offset + company_limit:
                break
        shard = selected[company_offset : company_offset + company_limit]
        if len(shard) == company_limit:
            return shard
        raise ValueError(
            f"Could only select {len(shard)} one-per-industry companies "
            f"after offset {company_offset}"
        )
    raise ValueError(f"Unsupported sample strategy: {sample_strategy}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the COPAL Table 2 component ablation.")
    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--company-limit", type=int, default=1)
    parser.add_argument("--company-offset", type=int, default=0)
    parser.add_argument("--sample-strategy", choices=("first", "one-per-industry"), default="one-per-industry")
    parser.add_argument("--policies-path", type=Path, default=DEFAULT_POLICIES_PATH)
    parser.add_argument("--prompts-path", type=Path, default=DEFAULT_PROMPTS_PATH)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--grounding-model", default="gemini-3-flash-preview")
    parser.add_argument("--composition-model", default="gemini-3-flash-preview")
    parser.add_argument("--query-model", default="gpt-5.5")
    parser.add_argument("--mapping-model", default="gemini-3-flash-preview")
    parser.add_argument("--screening-model", default="gemini-3-flash-preview")
    parser.add_argument("--judge-model", default="gemini-3-flash-preview")
    parser.add_argument("--eval-models", default="Doubao-Seed-2.0-pro,gemini-3.1-pro-preview")
    parser.add_argument("--max-compositions-per-company", type=int, default=8)
    parser.add_argument("--direct-candidate-count", type=int, default=36)
    parser.add_argument("--query-variants-per-composition", type=int, default=4)
    parser.add_argument("--query-variants-per-facet", type=int, default=2)
    parser.add_argument("--selected-per-company", type=int, default=12)
    parser.add_argument("--live-max-workers", type=int, default=6)
    parser.add_argument("--stop-after", choices=("screening", "evaluation"), default="evaluation")
    return parser


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("COPAL_LIVE_JSON_MAX_ATTEMPTS", "5")
    args = build_parser().parse_args(argv)
    if args.max_compositions_per_company < 1:
        raise ValueError("--max-compositions-per-company must be positive")
    if args.direct_candidate_count < 1:
        raise ValueError("--direct-candidate-count must be positive")
    if args.query_variants_per_composition < 1:
        raise ValueError("--query-variants-per-composition must be positive")
    if args.query_variants_per_facet < 1:
        raise ValueError("--query-variants-per-facet must be positive")
    if args.selected_per_company < 1:
        raise ValueError("--selected-per-company must be positive")
    if args.live_max_workers < 1:
        raise ValueError("--live-max-workers must be positive")
    eval_models = parse_csv(args.eval_models)
    if not eval_models:
        raise ValueError("--eval-models must include at least one model")

    experiment_id = args.experiment_id or f"table2_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_dir = ensure_directory(args.runs_dir / "experiments" / experiment_id)
    company_runs_dir = ensure_directory(experiment_dir / "company_runs")
    cache_dir = ensure_directory(experiment_dir / "llm_cache")

    worlds = select_worlds(
        load_company_worlds(args.policies_path),
        company_limit=args.company_limit,
        sample_strategy=args.sample_strategy,
        company_offset=args.company_offset,
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
        "company_offset": args.company_offset,
        "sample_strategy": args.sample_strategy,
        "selected_companies": [world.company_key for world in worlds],
        "models": {
            "grounding_model": args.grounding_model,
            "composition_model": args.composition_model,
            "query_model": args.query_model,
            "mapping_model": args.mapping_model,
            "screening_model": args.screening_model,
            "judge_model": args.judge_model,
            "eval_models": eval_models,
        },
        "budgets": {
            "max_compositions_per_company": args.max_compositions_per_company,
            "direct_candidate_count": args.direct_candidate_count,
            "query_variants_per_composition": args.query_variants_per_composition,
            "query_variants_per_facet": args.query_variants_per_facet,
            "selected_per_company": args.selected_per_company,
            "live_max_workers": args.live_max_workers,
            "stop_after": args.stop_after,
        },
        "live_client": live_client_runtime_metadata(),
    }
    write_json(experiment_dir / "table2_manifest.json", manifest)

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
        company_summary = run_table2_company_ablation(
            run_dir=run_dir,
            world=world,
            system_prompt=prompts_by_key[world.company_key].system_prompt,
            live_client=live_client,
            downstream_client=downstream_client,
            grounding_model=args.grounding_model,
            composition_model=args.composition_model,
            query_model=args.query_model,
            mapping_model=args.mapping_model,
            screening_model=args.screening_model,
            judge_model=args.judge_model,
            eval_models=eval_models,
            max_compositions=args.max_compositions_per_company,
            direct_candidate_count=args.direct_candidate_count,
            query_variants_per_composition=args.query_variants_per_composition,
            query_variants_per_facet=args.query_variants_per_facet,
            selected_per_company=args.selected_per_company,
            live_max_workers=args.live_max_workers,
            stop_after=args.stop_after,
        )
        company_summary["run_id"] = run_id
        company_summaries.append(company_summary)

    all_judgments = []
    if args.stop_after == "evaluation":
        for company_dir in company_runs_dir.iterdir():
            for path in sorted(company_dir.glob("variants/*/evaluation/response_judgments.jsonl")):
                all_judgments.extend(read_jsonl(path))

    aggregate = {
        "experiment_id": experiment_id,
        "company_count": len(company_summaries),
        "company_summaries": company_summaries,
        "table2": aggregate_table2_variant_summaries(company_summaries),
    }
    if all_judgments:
        aggregate["overall_probe_summary"] = summarize_pilot_judgments(all_judgments)
    write_json(experiment_dir / "table2_summary.json", aggregate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
