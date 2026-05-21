from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from statistics import mean, median
from pathlib import Path
from typing import Sequence

from copal.checkpoints import run_checkpointed_stage
from copal.config import (
    DEFAULT_POLICIES_PATH,
    DEFAULT_PROMPTS_PATH,
    STOP_AFTER_STAGES,
    RoleConfig,
    RunConfig,
    default_role_config,
    load_model_names,
    require_stop_after,
)
from copal.data_sources import load_company_worlds, select_company_world
from copal.experiment_analysis import (
    summarize_experiment_baselines,
    summarize_experiment_evaluation,
    summarize_experiment_mitigation,
    summarize_experiment_taxonomy,
)
from copal.io import ensure_directory, read_json, read_jsonl, write_json, write_jsonl
from copal.llm import build_live_client, live_client_runtime_metadata
from copal.manifest_compat import manifests_match_for_resume
from copal.models import CompanyWorld
from copal.stages.audit import run_audit_stage
from copal.stages.baselines import run_baseline_experiment_stage
from copal.stages.composition_proposal import run_composition_proposal_stage
from copal.stages.composition_validation import run_composition_validation_stage
from copal.stages.coverage_judge import run_coverage_judge_stage
from copal.stages.difficulty_screening import (
    build_single_policy_projection_items,
    run_difficulty_screening_stage,
    run_paired_projection_evaluation_stage,
)
from copal.stages.downstream_chatbot import run_downstream_chatbot_stage
from copal.stages.grounding_proposal import run_grounding_proposal_stage
from copal.stages.grounding_resolution import run_grounding_resolution_stage
from copal.stages.mitigation import DEFAULT_MITIGATION_SETTINGS, run_mitigation_stage
from copal.stages.query_proposal import run_query_proposal_stage
from copal.stages.query_validation import run_query_validation_stage
from copal.stages.reference_subset import run_reference_subset_stage
from copal.stages.reporting import write_run_reports
from copal.stages.response_judgment import run_response_judgment_stage
from copal.stages.run_setup import create_run_id, initialize_run
from copal.stages.selection import run_selection_stage

COMPLETED_STATUSES_FOR_MITIGATION = {
    "selection_completed",
    "baselines_completed",
    "audit_completed",
    "evaluation_completed",
}

ROLE_MODEL_FIELDS: tuple[str, ...] = (
    "proposal_model",
    "query_proposal_model",
    "canonicalization_model",
    "validator_model",
    "coverage_judge_model",
    "downstream_chatbot_model",
    "response_judge_model",
)


def limit_world_for_live_smoke(world: object, *, rule_limit_per_side: int) -> object:
    if rule_limit_per_side < 1:
        raise ValueError("rule_limit_per_side must be positive")
    try:
        from dataclasses import replace

        return replace(
            world,
            allowed_behaviors=list(world.allowed_behaviors[:rule_limit_per_side]),
            prohibited_behaviors=list(world.prohibited_behaviors[:rule_limit_per_side]),
        )
    except AttributeError as exc:
        raise TypeError("limit_world_for_live_smoke requires a CompanyWorld-like object") from exc


def limit_facets_for_live_smoke(
    facet_library: dict[str, tuple[str, ...] | list[str]],
    *,
    facet_limit_per_signature: int,
) -> dict[str, tuple[str, ...]]:
    if facet_limit_per_signature < 1:
        raise ValueError("facet_limit_per_signature must be positive")
    return {
        signature: tuple(str(facet) for facet in facets[:facet_limit_per_signature])
        for signature, facets in facet_library.items()
    }


def select_experiment_worlds(
    worlds: list[CompanyWorld],
    *,
    company_limit: int,
    sample_strategy: str,
) -> list[CompanyWorld]:
    if company_limit < 1:
        raise ValueError("--company-limit must be positive")
    if sample_strategy == "first":
        return list(worlds[:company_limit])
    if sample_strategy == "one-per-industry":
        selected: list[CompanyWorld] = []
        seen_industries: set[str] = set()
        for world in worlds:
            if world.industry in seen_industries:
                continue
            selected.append(world)
            seen_industries.add(world.industry)
            if len(selected) == company_limit:
                return selected
        raise ValueError(
            f"--sample-strategy one-per-industry could only select {len(selected)} companies; "
            f"requested {company_limit}"
        )
    raise ValueError(f"Unsupported sample_strategy: {sample_strategy}")


def completion_status_for_stop_after(stop_after: str) -> str:
    require_stop_after(stop_after)
    return f"{stop_after}_completed"


def build_uniform_role_config(model: str) -> RoleConfig:
    if not model.strip():
        raise ValueError("model must be non-empty")
    return RoleConfig(
        proposal_model=model,
        canonicalization_model=model,
        validator_model=model,
        coverage_judge_model=model,
        downstream_chatbot_model=model,
        response_judge_model=model,
    )


def _role_config_record(role_config: RoleConfig) -> dict[str, str]:
    return {
        "proposal_model": role_config.proposal_model,
        "query_proposal_model": role_config.query_proposal_model,
        "canonicalization_model": role_config.canonicalization_model,
        "validator_model": role_config.validator_model,
        "coverage_judge_model": role_config.coverage_judge_model,
        "downstream_chatbot_model": role_config.downstream_chatbot_model,
        "response_judge_model": role_config.response_judge_model,
    }


def _explicit_role_model_values(args: argparse.Namespace) -> dict[str, str]:
    values: dict[str, str] = {}
    for field_name in ROLE_MODEL_FIELDS:
        raw_value = getattr(args, field_name, None)
        if raw_value is None:
            continue
        value = str(raw_value).strip()
        if not value:
            raise ValueError(f"--{field_name.replace('_', '-')} must be non-empty")
        values[field_name] = value
    return values


def build_role_config_from_args(args: argparse.Namespace) -> RoleConfig | None:
    uniform_model = getattr(args, "live_smoke_model", None) or getattr(args, "all_roles_model", None)
    if uniform_model:
        return build_uniform_role_config(str(uniform_model))
    explicit_values = _explicit_role_model_values(args)
    if not explicit_values:
        return None
    role_config = default_role_config()
    for field_name, value in explicit_values.items():
        setattr(role_config, field_name, value)
    return role_config


def _append_role_model_arguments(run_args: list[str], args: argparse.Namespace) -> None:
    for field_name, value in _explicit_role_model_values(args).items():
        run_args.extend([f"--{field_name.replace('_', '-')}", value])


def _add_role_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--proposal-model")
    parser.add_argument("--query-proposal-model")
    parser.add_argument("--canonicalization-model")
    parser.add_argument("--validator-model")
    parser.add_argument("--coverage-judge-model")
    parser.add_argument("--downstream-chatbot-model")
    parser.add_argument("--response-judge-model")


def _validate_live_model_arguments(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    explicit_values = _explicit_role_model_values(args)
    explicit_flags = ", ".join(f"--{field.replace('_', '-')}" for field in explicit_values)
    if explicit_values and args.execution_mode != "live":
        parser.error(f"{explicit_flags} require --execution-mode live")
    if args.all_roles_model and explicit_values:
        parser.error("--all-roles-model cannot be combined with explicit role model flags")
    if args.live_smoke_model and explicit_values:
        parser.error("--live-smoke-model cannot be combined with explicit role model flags")


def optional_positive_limit(value: int, *, name: str) -> int | None:
    if value == 0:
        return None
    if value < 0:
        raise ValueError(f"{name} must be 0 for unlimited or a positive integer")
    return value


def optional_non_negative_limit(value: int, *, name: str) -> int | None:
    if value < 0:
        return None
    return value


def build_live_stage_kwargs(
    *,
    config: RunConfig,
    live_client: object | None,
    downstream_live_client: object | None = None,
) -> dict[str, dict[str, object]]:
    empty = {
        "grounding_proposal": {"execution_mode": config.execution_mode},
        "composition_validation": {"execution_mode": config.execution_mode},
        "query_proposal": {"execution_mode": config.execution_mode},
        "query_validation": {"execution_mode": config.execution_mode},
        "coverage_judge": {"execution_mode": config.execution_mode},
        "downstream_chatbot": {"execution_mode": config.execution_mode},
        "response_judgment": {"execution_mode": config.execution_mode},
    }
    if config.execution_mode != "live":
        return empty
    if live_client is None:
        raise ValueError("Live execution requires a live_client")
    downstream_client = downstream_live_client if downstream_live_client is not None else live_client
    roles = config.role_config
    return {
        "grounding_proposal": {
            "execution_mode": "live",
            "proposal_client": live_client,
            "canonicalization_client": live_client,
            "proposal_model": roles.proposal_model,
            "canonicalization_model": roles.canonicalization_model,
            "live_max_workers": config.live_max_workers,
        },
        "composition_validation": {
            "execution_mode": "live",
            "validator_client": live_client,
            "validator_model": roles.validator_model,
            "live_max_workers": config.live_max_workers,
        },
        "query_proposal": {
            "execution_mode": "live",
            "proposal_client": live_client,
            "proposal_model": roles.query_proposal_model or roles.proposal_model,
            "live_max_workers": config.live_max_workers,
        },
        "query_validation": {
            "execution_mode": "live",
            "validator_client": live_client,
            "validator_model": roles.validator_model,
            "live_max_workers": config.live_max_workers,
        },
        "coverage_judge": {
            "execution_mode": "live",
            "coverage_client": live_client,
            "coverage_model": roles.coverage_judge_model,
            "live_max_workers": config.live_max_workers,
        },
        "downstream_chatbot": {
            "execution_mode": "live",
            "downstream_client": downstream_client,
            "downstream_models": config.model_names or (roles.downstream_chatbot_model,),
            "live_max_workers": config.live_max_workers,
        },
        "response_judgment": {
            "execution_mode": "live",
            "response_judge_client": live_client,
            "response_judge_model": roles.response_judge_model,
            "live_max_workers": config.live_max_workers,
        },
    }


def record_live_usage_summary(*, summary: dict[str, object], live_client: object) -> None:
    try:
        usage_summary = live_client.usage_summary
    except AttributeError as exc:
        raise TypeError("Live client must expose a usage_summary method for run metering") from exc
    if not callable(usage_summary):
        raise TypeError("Live client usage_summary must be callable for run metering")
    usage = usage_summary()
    if not isinstance(usage, dict):
        raise TypeError("Live client usage_summary must return a dict for run metering")
    existing_usage = summary.get("llm_usage")
    if (
        isinstance(existing_usage, dict)
        and int(usage.get("total_tokens", 0) or 0) == 0
        and int(usage.get("cache_hits", 0) or 0) == 0
        and int(usage.get("cache_misses", 0) or 0) == 0
    ):
        summary["llm_usage"] = dict(existing_usage)
        return
    summary["llm_usage"] = dict(usage)


def build_experiment_manifest(
    *,
    args: argparse.Namespace,
    company_runs_dir: Path,
    composition_limit_per_signature: int | None,
    composition_adjudication_limit: int | None,
    role_config: RoleConfig | None,
    model_names: Sequence[str],
) -> dict[str, object]:
    manifest: dict[str, object] = {
        "experiment_id": args.experiment_id,
        "company_limit": args.company_limit,
        "sample_strategy": args.sample_strategy,
        "stop_after": args.stop_after,
        "company_workers": args.company_workers,
        "live_max_workers": args.live_max_workers,
        "composition_limit_per_signature": composition_limit_per_signature,
        "composition_adjudication_limit": composition_adjudication_limit,
        "query_variants_per_facet": args.query_variants_per_facet,
        "selection_variants_per_facet": args.selection_variants_per_facet,
        "screening_model": args.screening_model,
        "screening_min_score": args.screening_min_score,
        "screening_hard_suite_size": args.screening_hard_suite_size,
        "screening_use_hard_suite": args.screening_use_hard_suite,
        "execution_mode": args.execution_mode,
        "policies_path": args.policies_path,
        "model_name_path": args.model_name_path,
        "company_runs_dir": str(company_runs_dir),
        "model_roster": list(model_names),
        "model_count": len(model_names),
    }
    if role_config is not None:
        manifest["role_config"] = _role_config_record(role_config)
    if args.execution_mode == "live":
        manifest["live_client"] = live_client_runtime_metadata()
    if args.all_roles_model:
        manifest["all_roles_model"] = args.all_roles_model
    if args.prompts_path != str(DEFAULT_PROMPTS_PATH):
        manifest["prompts_path"] = args.prompts_path
    if args.live_smoke:
        manifest["live_smoke"] = True
        manifest["smoke_rule_limit_per_side"] = args.smoke_rule_limit_per_side
        manifest["smoke_facet_limit_per_signature"] = args.smoke_facet_limit_per_signature
        if args.live_smoke_model:
            manifest["live_smoke_model"] = args.live_smoke_model
    return manifest


def finalize_run_summary(
    *,
    run_dir: Path,
    status: str,
    summary_sections: dict[str, dict[str, object]],
    checkpoint_records: dict[str, dict[str, object]],
    live_client: object | None,
) -> dict[str, object]:
    summary = read_json(run_dir / "reports" / "summary.json")
    summary["status"] = status
    for section_name, section_summary in summary_sections.items():
        summary[section_name] = section_summary
    summary["checkpoints"] = checkpoint_records
    if live_client is not None:
        record_live_usage_summary(summary=summary, live_client=live_client)
    write_json(run_dir / "reports" / "summary.json", summary)
    write_run_reports(reports_dir=run_dir / "reports", summary=summary)
    return summary


def build_construction_yield_row(
    *,
    world: CompanyWorld,
    run_id: str,
    run_dir: Path,
    summary: dict[str, object],
) -> dict[str, object]:
    grounding = dict(summary.get("grounding", {}))
    composition = dict(summary.get("composition", {}))
    query_generation = dict(summary.get("query_generation", {}))
    selection = dict(summary.get("selection", {}))
    return {
        "run_id": run_id,
        "company_key": world.company_key,
        "industry": world.industry,
        "company_name": world.company_name,
        "run_dir": str(run_dir),
        "policy_rule_count": len(world.allowed_behaviors) + len(world.prohibited_behaviors),
        "allowed_rule_count": len(world.allowed_behaviors),
        "prohibited_rule_count": len(world.prohibited_behaviors),
        "grounded_clause_count": int(grounding.get("grounded_clause_count", 0)),
        "candidate_composition_count": int(composition.get("candidate_count", 0)),
        "accepted_composition_count": int(composition.get("accepted_count", 0)),
        "candidate_query_count": int(query_generation.get("candidate_query_count", 0)),
        "accepted_query_count": int(query_generation.get("accepted_count", 0)),
        "final_benchmark_count": int(selection.get("final_benchmark_count", 0)),
        "status": str(summary.get("status", "unknown")),
    }


def build_experiment_company_run_args(
    *,
    args: argparse.Namespace,
    world: CompanyWorld,
    index: int,
    company_runs_dir: Path,
) -> tuple[str, list[str]]:
    run_id = f"{args.experiment_id}__{index:03d}"
    run_args = [
        "run",
        "--company-key",
        world.company_key,
        "--execution-mode",
        args.execution_mode,
        "--run-id",
        run_id,
        "--runs-dir",
        str(company_runs_dir),
        "--cache-dir",
        str(Path(args.cache_dir)),
        "--model-name-path",
        args.model_name_path,
        "--policies-path",
        args.policies_path,
        "--prompts-path",
        args.prompts_path,
        "--stop-after",
        args.stop_after,
        "--live-max-workers",
        str(args.live_max_workers),
        "--composition-limit-per-signature",
        str(args.composition_limit_per_signature),
        "--composition-adjudication-limit",
        str(args.composition_adjudication_limit),
        "--query-variants-per-facet",
        str(args.query_variants_per_facet),
        "--selection-variants-per-facet",
        str(args.selection_variants_per_facet),
        "--screening-model",
        args.screening_model,
        "--screening-min-score",
        str(args.screening_min_score),
        "--screening-hard-suite-size",
        str(args.screening_hard_suite_size),
    ]
    if args.screening_use_hard_suite:
        run_args.append("--screening-use-hard-suite")
    if args.all_roles_model:
        run_args.extend(["--all-roles-model", args.all_roles_model])
    _append_role_model_arguments(run_args, args)
    if args.live_smoke:
        run_args.extend(
            [
                "--live-smoke",
                "--smoke-rule-limit-per-side",
                str(args.smoke_rule_limit_per_side),
                "--smoke-facet-limit-per-signature",
                str(args.smoke_facet_limit_per_signature),
            ]
        )
        if args.live_smoke_model:
            run_args.extend(["--live-smoke-model", args.live_smoke_model])
    return run_id, run_args


def run_experiment_company(
    *,
    args: argparse.Namespace,
    world: CompanyWorld,
    index: int,
    company_runs_dir: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    run_id, run_args = build_experiment_company_run_args(
        args=args,
        world=world,
        index=index,
        company_runs_dir=company_runs_dir,
    )
    main(run_args)
    run_dir = company_runs_dir / run_id
    run_summary = read_json(run_dir / "reports" / "summary.json")
    status_row = {
        "run_id": run_id,
        "company_key": world.company_key,
        "status": run_summary["status"],
        "run_dir": str(run_dir),
    }
    yield_row = build_construction_yield_row(world=world, run_id=run_id, run_dir=run_dir, summary=run_summary)
    return status_row, yield_row


def run_experiment_companies(
    *,
    args: argparse.Namespace,
    worlds: Sequence[CompanyWorld],
    company_runs_dir: Path,
) -> list[tuple[dict[str, object], dict[str, object]]]:
    indexed_worlds = list(enumerate(worlds))
    if args.company_workers == 1:
        return [
            run_experiment_company(args=args, world=world, index=index, company_runs_dir=company_runs_dir)
            for index, world in indexed_worlds
        ]
    with ThreadPoolExecutor(max_workers=args.company_workers) as executor:
        return list(
            executor.map(
                lambda item: run_experiment_company(
                    args=args,
                    world=item[1],
                    index=item[0],
                    company_runs_dir=company_runs_dir,
                ),
                indexed_worlds,
            )
        )


def summarize_construction_yields(rows: list[dict[str, object]]) -> dict[str, object]:
    numeric_fields = (
        "policy_rule_count",
        "grounded_clause_count",
        "candidate_composition_count",
        "accepted_composition_count",
        "candidate_query_count",
        "accepted_query_count",
        "final_benchmark_count",
    )
    field_stats: dict[str, dict[str, object]] = {}
    for field in numeric_fields:
        values = [int(row[field]) for row in rows]
        field_stats[field] = {
            "total": sum(values),
            "mean": mean(values) if values else 0.0,
            "median": median(values) if values else 0.0,
            "min": min(values) if values else 0,
            "max": max(values) if values else 0,
        }
    return {
        "company_count": len(rows),
        "industry_count": len({str(row["industry"]) for row in rows}),
        "totals": {field: field_stats[field]["total"] for field in numeric_fields},
        "fields": field_stats,
    }


def run_stage_with_checkpoint(
    *,
    checkpoint_records: dict[str, dict[str, object]],
    stage_name: str,
    stage_dir: Path,
    input_paths: Sequence[Path],
    config: dict[str, object],
    output_files: Sequence[str],
    runner: object,
) -> dict[str, object]:
    if not callable(runner):
        raise TypeError("runner must be callable")
    result = run_checkpointed_stage(
        stage_name=stage_name,
        stage_dir=stage_dir,
        input_paths=input_paths,
        config=config,
        output_files=output_files,
        runner=runner,
        manifest_file=f"{stage_name}_stage_manifest.json",
    )
    checkpoint_records[stage_name] = {
        "checkpoint_reused": bool(result["checkpoint_reused"]),
        "manifest_path": str(result["manifest_path"]),
    }
    return dict(result["summary"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the COPAL pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the staged COPAL pipeline.")
    run_parser.add_argument("--company-key", required=True)
    run_parser.add_argument("--run-id")
    run_parser.add_argument("--runs-dir", default="runs")
    run_parser.add_argument("--cache-dir", default="cache")
    run_parser.add_argument("--execution-mode", choices=("deterministic", "live"), required=True)
    run_parser.add_argument("--model-name-path", default="model_name.json")
    run_parser.add_argument("--policies-path", default=str(DEFAULT_POLICIES_PATH))
    run_parser.add_argument("--prompts-path", default=str(DEFAULT_PROMPTS_PATH))
    run_parser.add_argument("--stop-after", choices=STOP_AFTER_STAGES, default="evaluation")
    run_parser.add_argument("--all-roles-model")
    _add_role_model_arguments(run_parser)
    run_parser.add_argument("--live-max-workers", type=int, default=1)
    run_parser.add_argument("--composition-limit-per-signature", type=int, default=0)
    run_parser.add_argument("--composition-adjudication-limit", type=int, default=-1)
    run_parser.add_argument("--query-variants-per-facet", type=int, default=1)
    run_parser.add_argument("--selection-variants-per-facet", type=int, default=1)
    run_parser.add_argument("--screening-model", default="")
    run_parser.add_argument("--screening-min-score", type=float, default=2.0)
    run_parser.add_argument("--screening-hard-suite-size", type=int, default=0)
    run_parser.add_argument("--screening-use-hard-suite", action="store_true")
    run_parser.add_argument("--live-smoke", action="store_true")
    run_parser.add_argument("--smoke-rule-limit-per-side", type=int, default=1)
    run_parser.add_argument("--smoke-facet-limit-per-signature", type=int, default=1)
    run_parser.add_argument("--live-smoke-model")

    experiment_parser = subparsers.add_parser("experiment", help="Run or inspect a multi-company experiment.")
    experiment_subparsers = experiment_parser.add_subparsers(dest="experiment_command", required=True)
    experiment_run_parser = experiment_subparsers.add_parser("run", help="Run a resumable COPAL experiment.")
    experiment_run_parser.add_argument("--experiment-id", required=True)
    experiment_run_parser.add_argument("--company-limit", type=int, required=True)
    experiment_run_parser.add_argument("--runs-dir", default="runs")
    experiment_run_parser.add_argument("--cache-dir", default="cache")
    experiment_run_parser.add_argument("--execution-mode", choices=("deterministic", "live"), required=True)
    experiment_run_parser.add_argument("--model-name-path", default="model_name.json")
    experiment_run_parser.add_argument("--policies-path", default=str(DEFAULT_POLICIES_PATH))
    experiment_run_parser.add_argument("--prompts-path", default=str(DEFAULT_PROMPTS_PATH))
    experiment_run_parser.add_argument("--sample-strategy", choices=("first", "one-per-industry"), default="first")
    experiment_run_parser.add_argument("--stop-after", choices=STOP_AFTER_STAGES, default="evaluation")
    experiment_run_parser.add_argument("--company-workers", type=int, default=1)
    experiment_run_parser.add_argument("--all-roles-model")
    _add_role_model_arguments(experiment_run_parser)
    experiment_run_parser.add_argument("--live-max-workers", type=int, default=1)
    experiment_run_parser.add_argument("--composition-limit-per-signature", type=int, default=0)
    experiment_run_parser.add_argument("--composition-adjudication-limit", type=int, default=-1)
    experiment_run_parser.add_argument("--query-variants-per-facet", type=int, default=1)
    experiment_run_parser.add_argument("--selection-variants-per-facet", type=int, default=1)
    experiment_run_parser.add_argument("--screening-model", default="")
    experiment_run_parser.add_argument("--screening-min-score", type=float, default=2.0)
    experiment_run_parser.add_argument("--screening-hard-suite-size", type=int, default=0)
    experiment_run_parser.add_argument("--screening-use-hard-suite", action="store_true")
    experiment_run_parser.add_argument("--live-smoke", action="store_true")
    experiment_run_parser.add_argument("--smoke-rule-limit-per-side", type=int, default=1)
    experiment_run_parser.add_argument("--smoke-facet-limit-per-signature", type=int, default=1)
    experiment_run_parser.add_argument("--live-smoke-model")
    experiment_mitigation_run_parser = experiment_subparsers.add_parser(
        "run-mitigation",
        help="Run mitigation settings over an existing experiment selection.",
    )
    experiment_mitigation_run_parser.add_argument("--experiment-id", required=True)
    experiment_mitigation_run_parser.add_argument("--runs-dir", default="runs")
    experiment_mitigation_run_parser.add_argument("--cache-dir", default="cache")
    experiment_mitigation_run_parser.add_argument("--execution-mode", choices=("deterministic", "live"), default="deterministic")
    experiment_mitigation_run_parser.add_argument("--base-model", default="glm-5.1")
    experiment_mitigation_run_parser.add_argument("--response-judge-model", default="glm-5.1")
    experiment_mitigation_run_parser.add_argument("--company-limit", type=int)
    experiment_mitigation_run_parser.add_argument("--company-workers", type=int, default=1)
    experiment_mitigation_run_parser.add_argument(
        "--settings",
        nargs="+",
        choices=DEFAULT_MITIGATION_SETTINGS,
        default=list(DEFAULT_MITIGATION_SETTINGS),
    )
    experiment_summary_parser = experiment_subparsers.add_parser(
        "summarize-taxonomy",
        help="Summarize effect and relation-pattern distributions for a completed experiment.",
    )
    experiment_summary_parser.add_argument("--experiment-id", required=True)
    experiment_summary_parser.add_argument("--runs-dir", default="runs")
    experiment_baseline_summary_parser = experiment_subparsers.add_parser(
        "summarize-baselines",
        help="Summarize construction-quality and invalid-item baseline metrics for a completed experiment.",
    )
    experiment_baseline_summary_parser.add_argument("--experiment-id", required=True)
    experiment_baseline_summary_parser.add_argument("--runs-dir", default="runs")
    experiment_evaluation_summary_parser = experiment_subparsers.add_parser(
        "summarize-evaluation",
        help="Summarize downstream model error rates for a completed evaluation experiment.",
    )
    experiment_evaluation_summary_parser.add_argument("--experiment-id", required=True)
    experiment_evaluation_summary_parser.add_argument("--runs-dir", default="runs")
    experiment_mitigation_summary_parser = experiment_subparsers.add_parser(
        "summarize-mitigation",
        help="Summarize mitigation setting error rates for an experiment with mitigation outputs.",
    )
    experiment_mitigation_summary_parser.add_argument("--experiment-id", required=True)
    experiment_mitigation_summary_parser.add_argument("--runs-dir", default="runs")

    return parser


def run_experiment_command(args: argparse.Namespace) -> int:
    if args.company_limit < 1:
        raise ValueError("--company-limit must be positive")
    require_stop_after(args.stop_after)
    if args.company_workers < 1:
        raise ValueError("--company-workers must be positive")
    if args.live_max_workers < 1:
        raise ValueError("--live-max-workers must be positive")
    if args.query_variants_per_facet < 1:
        raise ValueError("--query-variants-per-facet must be positive")
    if args.selection_variants_per_facet < 1:
        raise ValueError("--selection-variants-per-facet must be positive")
    if args.screening_model and args.execution_mode != "live":
        raise ValueError("--screening-model requires --execution-mode live")
    if args.screening_min_score < 0:
        raise ValueError("--screening-min-score must be non-negative")
    if args.screening_hard_suite_size < 0:
        raise ValueError("--screening-hard-suite-size must be zero for unlimited or a positive integer")
    composition_limit_per_signature = optional_positive_limit(
        args.composition_limit_per_signature,
        name="--composition-limit-per-signature",
    )
    composition_adjudication_limit = optional_non_negative_limit(
        args.composition_adjudication_limit,
        name="--composition-adjudication-limit",
    )
    model_name_path = Path(args.model_name_path)
    model_names = load_model_names(model_name_path) if args.execution_mode == "live" else ()
    role_config = build_role_config_from_args(args)
    runs_dir = Path(args.runs_dir)
    experiment_dir = ensure_directory(runs_dir / "experiments" / args.experiment_id)
    company_runs_dir = ensure_directory(experiment_dir / "company_runs")
    manifest = build_experiment_manifest(
        args=args,
        company_runs_dir=company_runs_dir,
        composition_limit_per_signature=composition_limit_per_signature,
        composition_adjudication_limit=composition_adjudication_limit,
        role_config=role_config,
        model_names=model_names,
    )
    manifest_path = experiment_dir / "experiment_manifest.json"
    if manifest_path.exists():
        existing_manifest = read_json(manifest_path)
        if not manifests_match_for_resume(existing_manifest, manifest):
            raise ValueError(f"Existing experiment manifest does not match requested config: {experiment_dir}")
    else:
        write_json(manifest_path, manifest)

    worlds = select_experiment_worlds(
        load_company_worlds(Path(args.policies_path)),
        company_limit=args.company_limit,
        sample_strategy=args.sample_strategy,
    )
    company_results = run_experiment_companies(args=args, worlds=worlds, company_runs_dir=company_runs_dir)
    status_rows = [status_row for status_row, _yield_row in company_results]
    yield_rows = [yield_row for _status_row, yield_row in company_results]

    expected_status = completion_status_for_stop_after(args.stop_after)
    summary = {
        "experiment_id": args.experiment_id,
        "company_count": len(status_rows),
        "expected_status": expected_status,
        "completed_count": sum(1 for row in status_rows if row["status"] == expected_status),
        "failed_count": sum(1 for row in status_rows if row["status"] != expected_status),
        "construction_yield": summarize_construction_yields(yield_rows),
    }
    write_jsonl(experiment_dir / "company_status.jsonl", status_rows)
    write_jsonl(experiment_dir / "construction_yield.jsonl", yield_rows)
    write_json(experiment_dir / "construction_yield_summary.json", summary["construction_yield"])
    write_json(experiment_dir / "experiment_summary.json", summary)
    return 0


def run_experiment_taxonomy_summary_command(args: argparse.Namespace) -> int:
    experiment_dir = Path(args.runs_dir) / "experiments" / args.experiment_id
    summary = summarize_experiment_taxonomy(experiment_dir=experiment_dir)
    write_json(experiment_dir / "taxonomy_distribution_summary.json", summary)
    return 0


def run_experiment_baseline_summary_command(args: argparse.Namespace) -> int:
    experiment_dir = Path(args.runs_dir) / "experiments" / args.experiment_id
    summary = summarize_experiment_baselines(experiment_dir=experiment_dir)
    write_json(experiment_dir / "baseline_comparison_summary.json", summary)
    return 0


def run_experiment_evaluation_summary_command(args: argparse.Namespace) -> int:
    experiment_dir = Path(args.runs_dir) / "experiments" / args.experiment_id
    summary = summarize_experiment_evaluation(experiment_dir=experiment_dir)
    write_json(experiment_dir / "downstream_evaluation_summary.json", summary)
    return 0


def run_experiment_mitigation_command(args: argparse.Namespace) -> int:
    if args.company_workers < 1:
        raise ValueError("--company-workers must be positive")
    if args.company_limit is not None and args.company_limit < 1:
        raise ValueError("--company-limit must be positive when provided")
    experiment_dir = Path(args.runs_dir) / "experiments" / args.experiment_id
    company_runs_dir = experiment_dir / "company_runs"
    if not company_runs_dir.exists():
        raise FileNotFoundError(f"Experiment company_runs directory does not exist: {company_runs_dir}")
    run_dirs = sorted(path for path in company_runs_dir.iterdir() if path.is_dir())
    if args.company_limit is not None:
        run_dirs = run_dirs[: args.company_limit]
    if not run_dirs:
        raise ValueError(f"Experiment has no company run directories: {experiment_dir}")

    if args.company_workers == 1:
        rows = [
            run_experiment_mitigation_company(args=args, experiment_dir=experiment_dir, run_dir=run_dir)
            for run_dir in run_dirs
        ]
    else:
        with ThreadPoolExecutor(max_workers=args.company_workers) as executor:
            rows = list(
                executor.map(
                    lambda run_dir: run_experiment_mitigation_company(
                        args=args,
                        experiment_dir=experiment_dir,
                        run_dir=run_dir,
                    ),
                    run_dirs,
                )
            )
    summary = {
        "experiment_id": args.experiment_id,
        "company_count": len(rows),
        "completed_count": sum(1 for row in rows if row["status"] == "mitigation_completed"),
        "failed_count": sum(1 for row in rows if row["status"] != "mitigation_completed"),
        "settings": list(args.settings),
        "base_model": args.base_model,
        "execution_mode": args.execution_mode,
        "company_results": rows,
    }
    write_json(experiment_dir / "mitigation_run_summary.json", summary)
    mitigation_summary = summarize_experiment_mitigation(experiment_dir=experiment_dir, run_dirs=run_dirs)
    write_json(experiment_dir / "mitigation_comparison_summary.json", mitigation_summary)
    return 0


def run_experiment_mitigation_company(
    *,
    args: argparse.Namespace,
    experiment_dir: Path,
    run_dir: Path,
) -> dict[str, object]:
    run_summary = read_json(run_dir / "reports" / "summary.json")
    run_id = str(run_summary["run_id"])
    status = str(run_summary["status"])
    if status not in COMPLETED_STATUSES_FOR_MITIGATION:
        raise ValueError(f"Company run is not ready for mitigation: {run_id} status={status}")
    benchmark_items = read_jsonl(run_dir / "selection" / "benchmark_items_final.jsonl")
    prompt_record = read_json(run_dir / "inputs" / "selected_system_prompt.json")
    downstream_client = None
    response_judge_client = None
    if args.execution_mode == "live":
        cache_dir = Path(args.cache_dir) / "mitigation" / args.experiment_id / run_id / "llm"
        downstream_client = build_live_client(
            cache_dir=cache_dir,
            response_format_override=None,
        )
        response_judge_client = build_live_client(cache_dir=cache_dir)
    summary = run_mitigation_stage(
        mitigation_dir=run_dir / "mitigation",
        benchmark_items=benchmark_items,
        system_prompt=str(prompt_record["system_prompt"]),
        execution_mode=args.execution_mode,
        base_model=args.base_model,
        settings=tuple(args.settings),
        downstream_client=downstream_client,
        response_judge_client=response_judge_client,
        response_judge_model=args.response_judge_model,
    )
    return {
        "run_id": run_id,
        "company_key": run_summary["company_key"],
        "status": "mitigation_completed",
        "run_dir": str(run_dir),
        "response_count": summary["response_count"],
        "judgment_count": summary["judgment_count"],
    }


def run_experiment_mitigation_summary_command(args: argparse.Namespace) -> int:
    experiment_dir = Path(args.runs_dir) / "experiments" / args.experiment_id
    summary = summarize_experiment_mitigation(experiment_dir=experiment_dir)
    write_json(experiment_dir / "mitigation_comparison_summary.json", summary)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "experiment":
        if args.experiment_command == "run":
            if args.live_smoke and args.execution_mode != "live":
                parser.error("--live-smoke requires --execution-mode live")
            if args.live_smoke_model and not args.live_smoke:
                parser.error("--live-smoke-model requires --live-smoke")
            if args.all_roles_model and args.execution_mode != "live":
                parser.error("--all-roles-model requires --execution-mode live")
            if args.all_roles_model and args.live_smoke_model:
                parser.error("--all-roles-model and --live-smoke-model cannot both be set")
            _validate_live_model_arguments(args, parser)
            if args.company_workers < 1:
                parser.error("--company-workers must be positive")
            if args.live_max_workers < 1:
                parser.error("--live-max-workers must be positive")
            if args.query_variants_per_facet < 1:
                parser.error("--query-variants-per-facet must be positive")
            if args.selection_variants_per_facet < 1:
                parser.error("--selection-variants-per-facet must be positive")
            if args.screening_model and args.execution_mode != "live":
                parser.error("--screening-model requires --execution-mode live")
            if args.screening_min_score < 0:
                parser.error("--screening-min-score must be non-negative")
            if args.screening_hard_suite_size < 0:
                parser.error("--screening-hard-suite-size must be zero for unlimited or a positive integer")
            if args.composition_limit_per_signature < 0:
                parser.error("--composition-limit-per-signature must be 0 for unlimited or a positive integer")
            return run_experiment_command(args)
        if args.experiment_command == "run-mitigation":
            if args.execution_mode == "live" and not args.response_judge_model:
                parser.error("--response-judge-model is required for live mitigation")
            if args.company_workers < 1:
                parser.error("--company-workers must be positive")
            if args.company_limit is not None and args.company_limit < 1:
                parser.error("--company-limit must be positive when provided")
            return run_experiment_mitigation_command(args)
        if args.experiment_command == "summarize-taxonomy":
            return run_experiment_taxonomy_summary_command(args)
        if args.experiment_command == "summarize-baselines":
            return run_experiment_baseline_summary_command(args)
        if args.experiment_command == "summarize-evaluation":
            return run_experiment_evaluation_summary_command(args)
        if args.experiment_command == "summarize-mitigation":
            return run_experiment_mitigation_summary_command(args)
    if args.command == "run":
        if args.live_smoke and args.execution_mode != "live":
            parser.error("--live-smoke requires --execution-mode live")
        if args.live_smoke_model and not args.live_smoke:
            parser.error("--live-smoke-model requires --live-smoke")
        if args.all_roles_model and args.execution_mode != "live":
            parser.error("--all-roles-model requires --execution-mode live")
        if args.all_roles_model and args.live_smoke_model:
            parser.error("--all-roles-model and --live-smoke-model cannot both be set")
        _validate_live_model_arguments(args, parser)
        if args.live_max_workers < 1:
            parser.error("--live-max-workers must be positive")
        if args.query_variants_per_facet < 1:
            parser.error("--query-variants-per-facet must be positive")
        if args.selection_variants_per_facet < 1:
            parser.error("--selection-variants-per-facet must be positive")
        if args.screening_model and args.execution_mode != "live":
            parser.error("--screening-model requires --execution-mode live")
        if args.screening_min_score < 0:
            parser.error("--screening-min-score must be non-negative")
        if args.screening_hard_suite_size < 0:
            parser.error("--screening-hard-suite-size must be zero for unlimited or a positive integer")
        require_stop_after(args.stop_after)
        if args.composition_limit_per_signature < 0:
            parser.error("--composition-limit-per-signature must be 0 for unlimited or a positive integer")
        composition_limit_per_signature = optional_positive_limit(
            args.composition_limit_per_signature,
            name="--composition-limit-per-signature",
        )
        composition_adjudication_limit = optional_non_negative_limit(
            args.composition_adjudication_limit,
            name="--composition-adjudication-limit",
        )
        model_name_path = Path(args.model_name_path)
        model_names = load_model_names(model_name_path) if args.execution_mode == "live" else ()
        role_config = build_role_config_from_args(args)
        config = RunConfig(
            run_id=args.run_id or create_run_id(args.company_key),
            company_key=args.company_key,
            execution_mode=args.execution_mode,
            policies_path=Path(args.policies_path),
            prompts_path=Path(args.prompts_path),
            model_name_path=model_name_path,
            runs_dir=Path(args.runs_dir),
            cache_dir=Path(args.cache_dir),
            model_names=model_names,
            stop_after=args.stop_after,
            live_max_workers=args.live_max_workers,
            composition_limit_per_signature=composition_limit_per_signature,
            composition_adjudication_limit=composition_adjudication_limit,
            query_variants_per_facet=args.query_variants_per_facet,
            selection_variants_per_facet=args.selection_variants_per_facet,
            screening_model=args.screening_model,
            screening_min_score=args.screening_min_score,
            screening_hard_suite_size=args.screening_hard_suite_size,
            screening_use_hard_suite=args.screening_use_hard_suite,
            live_smoke=bool(args.live_smoke),
            smoke_rule_limit_per_side=args.smoke_rule_limit_per_side if args.live_smoke else None,
            smoke_facet_limit_per_signature=args.smoke_facet_limit_per_signature if args.live_smoke else None,
        )
        if role_config is not None:
            config.role_config = role_config
        if config.live_smoke:
            config.facet_library = limit_facets_for_live_smoke(
                config.facet_library,
                facet_limit_per_signature=args.smoke_facet_limit_per_signature,
            )
        run_dir = initialize_run(config)
        world, prompt = select_company_world(
            policies_path=config.policies_path,
            prompts_path=config.prompts_path,
            company_key=config.company_key,
        )
        if config.live_smoke:
            world = limit_world_for_live_smoke(
                world,
                rule_limit_per_side=args.smoke_rule_limit_per_side,
            )
        live_client = None
        downstream_live_client = None
        if config.execution_mode == "live":
            live_client = build_live_client(
                cache_dir=config.cache_dir / "llm",
            )
            downstream_live_client = build_live_client(
                cache_dir=config.cache_dir / "llm",
                response_format_override=None,
            )
        live_stage_kwargs = build_live_stage_kwargs(
            config=config,
            live_client=live_client,
            downstream_live_client=downstream_live_client,
        )
        checkpoint_records: dict[str, dict[str, object]] = {}
        role_config_record = _role_config_record(config.role_config)
        facet_library_record = {key: list(value) for key, value in config.facet_library.items()}
        summary_sections: dict[str, dict[str, object]] = {}

        grounding_proposal_summary = run_stage_with_checkpoint(
            checkpoint_records=checkpoint_records,
            stage_name="grounding_proposal",
            stage_dir=run_dir / "grounding",
            input_paths=[run_dir / "inputs" / "selected_company.json"],
            config={
                "execution_mode": config.execution_mode,
                "role_config": role_config_record,
                "live_smoke": config.live_smoke,
                "smoke_rule_limit_per_side": config.smoke_rule_limit_per_side,
                "live_max_workers": config.live_max_workers,
            },
            output_files=[
                "raw_clause_extractions.jsonl",
                "canonicalization_candidates.jsonl",
                "grounding_proposal_summary.json",
            ],
            runner=lambda: run_grounding_proposal_stage(
                grounding_dir=run_dir / "grounding",
                world=world,
                **live_stage_kwargs["grounding_proposal"],
            ),
        )
        grounding_resolution_summary = run_stage_with_checkpoint(
            checkpoint_records=checkpoint_records,
            stage_name="grounding_resolution",
            stage_dir=run_dir / "grounding",
            input_paths=[run_dir / "grounding" / "canonicalization_candidates.jsonl"],
            config={"execution_mode": "deterministic"},
            output_files=[
                "exact_dedup_report.json",
                "semantic_dedup_pairs.jsonl",
                "semantic_dedup_resolutions.jsonl",
                "grounded_clause_library.jsonl",
                "grounding_resolution_summary.json",
            ],
            runner=lambda: run_grounding_resolution_stage(
                grounding_dir=run_dir / "grounding",
            ),
        )
        summary_sections["grounding"] = {
            **grounding_proposal_summary,
            **grounding_resolution_summary,
        }
        composition_proposal_summary = run_stage_with_checkpoint(
            checkpoint_records=checkpoint_records,
            stage_name="composition_proposal",
            stage_dir=run_dir / "compositions",
            input_paths=[run_dir / "grounding" / "grounded_clause_library.jsonl"],
            config={"execution_mode": "deterministic"},
            output_files=[
                "candidate_compositions.jsonl",
                "structure_signal_records.jsonl",
                "signature_proposals.jsonl",
                "composition_proposal_summary.json",
            ],
            runner=lambda: run_composition_proposal_stage(
                compositions_dir=run_dir / "compositions",
                grounded_rows=read_jsonl(run_dir / "grounding" / "grounded_clause_library.jsonl"),
            ),
        )
        composition_validation_summary = run_stage_with_checkpoint(
            checkpoint_records=checkpoint_records,
            stage_name="composition_validation",
            stage_dir=run_dir / "compositions",
            input_paths=[run_dir / "compositions" / "candidate_compositions.jsonl"],
            config={
                "execution_mode": config.execution_mode,
                "role_config": role_config_record,
                "allowed_signatures": list(config.facet_library.keys()),
                "composition_limit_per_signature": config.composition_limit_per_signature,
                "composition_adjudication_limit": config.composition_adjudication_limit,
                "live_max_workers": config.live_max_workers,
            },
            output_files=[
                "composition_deterministic_results.jsonl",
                "composition_adjudication_queue.jsonl",
                "composition_adjudications.jsonl",
                "accepted_compositions.jsonl",
                "rejected_compositions.jsonl",
                "composition_validation_summary.json",
            ],
            runner=lambda: run_composition_validation_stage(
                compositions_dir=run_dir / "compositions",
                composition_limit_per_signature=config.composition_limit_per_signature,
                composition_adjudication_limit=config.composition_adjudication_limit,
                **live_stage_kwargs["composition_validation"],
            ),
        )
        summary_sections["composition"] = {
            **composition_proposal_summary,
            **composition_validation_summary,
        }
        query_proposal_summary = run_stage_with_checkpoint(
            checkpoint_records=checkpoint_records,
            stage_name="query_proposal",
            stage_dir=run_dir / "query_generation",
            input_paths=[run_dir / "compositions" / "accepted_compositions.jsonl"],
            config={
                "execution_mode": config.execution_mode,
                "role_config": role_config_record,
                "facet_library": facet_library_record,
                "query_variants_per_facet": config.query_variants_per_facet,
                "selection_variants_per_facet": config.selection_variants_per_facet,
                "live_max_workers": config.live_max_workers,
            },
            output_files=[
                "intermediate_scenarios.jsonl",
                "candidate_queries.jsonl",
                "query_proposal_summary.json",
            ],
            runner=lambda: run_query_proposal_stage(
                query_generation_dir=run_dir / "query_generation",
                accepted_compositions=read_jsonl(run_dir / "compositions" / "accepted_compositions.jsonl"),
                facet_library=config.facet_library,
                query_variants_per_facet=config.query_variants_per_facet,
                **live_stage_kwargs["query_proposal"],
            ),
        )
        query_validation_summary = run_stage_with_checkpoint(
            checkpoint_records=checkpoint_records,
            stage_name="query_validation",
            stage_dir=run_dir / "query_generation",
            input_paths=[run_dir / "query_generation" / "candidate_queries.jsonl"],
            config={
                "execution_mode": config.execution_mode,
                "role_config": role_config_record,
                "live_max_workers": config.live_max_workers,
            },
            output_files=[
                "query_deterministic_results.jsonl",
                "query_adjudication_queue.jsonl",
                "query_adjudications.jsonl",
                "accepted_queries.jsonl",
                "rejected_queries.jsonl",
                "query_validation_summary.json",
            ],
            runner=lambda: run_query_validation_stage(
                query_generation_dir=run_dir / "query_generation",
                **live_stage_kwargs["query_validation"],
            ),
        )
        summary_sections["query_generation"] = {
            **query_proposal_summary,
            **query_validation_summary,
        }
        coverage_summary = run_stage_with_checkpoint(
            checkpoint_records=checkpoint_records,
            stage_name="coverage_judge",
            stage_dir=run_dir / "coverage",
            input_paths=[run_dir / "query_generation" / "accepted_queries.jsonl"],
            config={
                "execution_mode": config.execution_mode,
                "role_config": role_config_record,
                "facet_library": facet_library_record,
                "live_max_workers": config.live_max_workers,
            },
            output_files=[
                "composition_facet_universes.jsonl",
                "coverage_judge_results.jsonl",
                "accepted_query_coverages.jsonl",
                "coverage_summary.json",
            ],
            runner=lambda: run_coverage_judge_stage(
                coverage_dir=run_dir / "coverage",
                accepted_queries=read_jsonl(run_dir / "query_generation" / "accepted_queries.jsonl"),
                facet_library=config.facet_library,
                **live_stage_kwargs["coverage_judge"],
            ),
        )
        summary_sections["coverage"] = coverage_summary
        accepted_queries = read_jsonl(run_dir / "query_generation" / "accepted_queries.jsonl")
        coverage_rows = read_jsonl(run_dir / "coverage" / "accepted_query_coverages.jsonl")
        selection_summary = run_stage_with_checkpoint(
            checkpoint_records=checkpoint_records,
            stage_name="selection",
            stage_dir=run_dir / "selection",
            input_paths=[
                run_dir / "query_generation" / "accepted_queries.jsonl",
                run_dir / "coverage" / "accepted_query_coverages.jsonl",
            ],
            config={
                "random_seed": config.random_seed,
                "max_query_variants_per_facet": config.query_variants_per_facet,
                "selection_variants_per_facet": config.selection_variants_per_facet,
            },
            output_files=[
                "benchmark_items_pre_audit.jsonl",
                "benchmark_items_final.jsonl",
                "coverage_matrix.json",
                "greedy_selection_trace.json",
                "selection_summary.json",
            ],
            runner=lambda: run_selection_stage(
                selection_dir=run_dir / "selection",
                accepted_queries=accepted_queries,
                coverage_rows=coverage_rows,
                max_query_variants_per_facet=config.selection_variants_per_facet,
            ),
        )
        summary_sections["selection"] = selection_summary
        if config.stop_after == "selection":
            finalize_run_summary(
                run_dir=run_dir,
                status="selection_completed",
                summary_sections=summary_sections,
                checkpoint_records=checkpoint_records,
                live_client=live_client,
            )
            return 0
        benchmark_items = read_jsonl(run_dir / "selection" / "benchmark_items_final.jsonl")
        projection_items: list[dict[str, object]] = []
        if config.screening_model:
            screening_summary = run_stage_with_checkpoint(
                checkpoint_records=checkpoint_records,
                stage_name="difficulty_screening",
                stage_dir=run_dir / "difficulty_screening",
                input_paths=[
                    run_dir / "selection" / "benchmark_items_final.jsonl",
                    run_dir / "grounding" / "grounded_clause_library.jsonl",
                    run_dir / "inputs" / "selected_system_prompt.json",
                ],
                config={
                    "execution_mode": config.execution_mode,
                    "screening_model": config.screening_model,
                    "screening_min_score": config.screening_min_score,
                    "screening_hard_suite_size": config.screening_hard_suite_size,
                    "screening_use_hard_suite": config.screening_use_hard_suite,
                    "response_judge_model": config.role_config.response_judge_model,
                    "live_max_workers": config.live_max_workers,
                },
                output_files=[
                    "single_policy_projection_items.jsonl",
                    "screening_items.jsonl",
                    "chatbot_requests.jsonl",
                    "chatbot_responses.jsonl",
                    "chatbot_summary.json",
                    "response_judge_inputs.jsonl",
                    "response_judgments.jsonl",
                    "difficulty_scores.jsonl",
                    "hard_benchmark_items_final.jsonl",
                    "difficulty_screening_summary.json",
                ],
                runner=lambda: run_difficulty_screening_stage(
                    screening_dir=run_dir / "difficulty_screening",
                    benchmark_items=benchmark_items,
                    grounded_rows=read_jsonl(run_dir / "grounding" / "grounded_clause_library.jsonl"),
                    system_prompt=prompt.system_prompt,
                    execution_mode=config.execution_mode,
                    screening_model=config.screening_model,
                    min_score=config.screening_min_score,
                    hard_suite_size=config.screening_hard_suite_size,
                    downstream_client=downstream_live_client,
                    response_judge_client=live_client,
                    response_judge_model=config.role_config.response_judge_model,
                    live_max_workers=config.live_max_workers,
                ),
            )
            summary_sections["difficulty_screening"] = screening_summary
            projection_items = read_jsonl(run_dir / "difficulty_screening" / "single_policy_projection_items.jsonl")
            if config.screening_use_hard_suite:
                benchmark_items = read_jsonl(run_dir / "difficulty_screening" / "hard_benchmark_items_final.jsonl")
        elif config.stop_after == "screening":
            raise ValueError("--stop-after screening requires --screening-model")
        if config.stop_after == "screening":
            finalize_run_summary(
                run_dir=run_dir,
                status="screening_completed",
                summary_sections=summary_sections,
                checkpoint_records=checkpoint_records,
                live_client=live_client,
            )
            return 0
        baseline_summary = run_stage_with_checkpoint(
            checkpoint_records=checkpoint_records,
            stage_name="baselines",
            stage_dir=run_dir / "baselines",
            input_paths=[
                run_dir / "grounding" / "grounded_clause_library.jsonl",
                run_dir / "compositions" / "candidate_compositions.jsonl",
                run_dir / "compositions" / "accepted_compositions.jsonl",
                run_dir / "query_generation" / "accepted_queries.jsonl",
                run_dir / "coverage" / "accepted_query_coverages.jsonl",
                run_dir / "selection" / "benchmark_items_final.jsonl",
            ],
            config={
                "facet_library": facet_library_record,
                "final_query_budget": len(benchmark_items),
                "random_seed": config.random_seed,
            },
            output_files=[
                "baseline_protocols.jsonl",
                "baseline_candidate_records.jsonl",
                "construction_quality_metrics.jsonl",
                "invalid_item_breakdown.jsonl",
                "ablation_metrics.jsonl",
                "baseline_experiment_summary.json",
            ],
            runner=lambda: run_baseline_experiment_stage(
                baseline_dir=run_dir / "baselines",
                grounded_rows=read_jsonl(run_dir / "grounding" / "grounded_clause_library.jsonl"),
                candidate_compositions=read_jsonl(run_dir / "compositions" / "candidate_compositions.jsonl"),
                accepted_compositions=read_jsonl(run_dir / "compositions" / "accepted_compositions.jsonl"),
                accepted_queries=accepted_queries,
                coverage_rows=coverage_rows,
                benchmark_items=benchmark_items,
                facet_library=config.facet_library,
                final_query_budget=len(benchmark_items),
                random_seed=config.random_seed,
            ),
        )
        summary_sections["baselines"] = baseline_summary
        if config.stop_after == "baselines":
            finalize_run_summary(
                run_dir=run_dir,
                status="baselines_completed",
                summary_sections=summary_sections,
                checkpoint_records=checkpoint_records,
                live_client=live_client,
            )
            return 0
        selected_item_ids = {str(item["item_id"]) for item in benchmark_items}
        rejected_candidates = [
            {
                "item_id": str(query["query_id"]),
                "signature": str(query["signature_proposal"]),
                "target_facet": str(query["target_facet"]),
            }
            for query in accepted_queries
            if str(query["query_id"]) not in selected_item_ids
        ]
        reference_subset_summary = run_stage_with_checkpoint(
            checkpoint_records=checkpoint_records,
            stage_name="reference_subset",
            stage_dir=run_dir / "reference_subset",
            input_paths=[
                run_dir / "selection" / "benchmark_items_final.jsonl",
                run_dir / "query_generation" / "accepted_queries.jsonl",
            ],
            config={"target_size": config.reference_subset_size},
            output_files=["reference_subset.jsonl", "reference_subset_summary.json"],
            runner=lambda: run_reference_subset_stage(
                reference_subset_dir=run_dir / "reference_subset",
                accepted_items=benchmark_items,
                rejected_candidates=rejected_candidates,
                target_size=config.reference_subset_size,
            ),
        )
        summary_sections["reference_subset"] = reference_subset_summary
        audit_summary = run_stage_with_checkpoint(
            checkpoint_records=checkpoint_records,
            stage_name="audit",
            stage_dir=run_dir / "audit",
            input_paths=[
                run_dir / "grounding" / "grounded_clause_library.jsonl",
                run_dir / "selection" / "benchmark_items_final.jsonl",
            ],
            config={"sample_size": config.audit_sample_size},
            output_files=[
                "grounded_clause_audit_queue.jsonl",
                "accepted_item_audit_queue.jsonl",
                "human_audit_records.jsonl",
                "human_overrides.jsonl",
                "audit_summary.json",
            ],
            runner=lambda: run_audit_stage(
                audit_dir=run_dir / "audit",
                grounded_rows=read_jsonl(run_dir / "grounding" / "grounded_clause_library.jsonl"),
                benchmark_items=benchmark_items,
                sample_size=config.audit_sample_size,
            ),
        )
        summary_sections["audit"] = audit_summary
        if config.stop_after == "audit":
            finalize_run_summary(
                run_dir=run_dir,
                status="audit_completed",
                summary_sections=summary_sections,
                checkpoint_records=checkpoint_records,
                live_client=live_client,
            )
            return 0
        downstream_chatbot_summary = run_stage_with_checkpoint(
            checkpoint_records=checkpoint_records,
            stage_name="downstream_chatbot",
            stage_dir=run_dir / "evaluation",
            input_paths=[
                run_dir / "selection" / "benchmark_items_final.jsonl",
                run_dir / "inputs" / "selected_system_prompt.json",
            ],
            config={"execution_mode": config.execution_mode, "role_config": role_config_record},
            output_files=["chatbot_requests.jsonl", "chatbot_responses.jsonl", "chatbot_summary.json"],
            runner=lambda: run_downstream_chatbot_stage(
                evaluation_dir=run_dir / "evaluation",
                benchmark_items=benchmark_items,
                system_prompt=prompt.system_prompt,
                **live_stage_kwargs["downstream_chatbot"],
            ),
        )
        response_judgment_summary = run_stage_with_checkpoint(
            checkpoint_records=checkpoint_records,
            stage_name="response_judgment",
            stage_dir=run_dir / "evaluation",
            input_paths=[
                run_dir / "selection" / "benchmark_items_final.jsonl",
                run_dir / "evaluation" / "chatbot_responses.jsonl",
            ],
            config={"execution_mode": config.execution_mode, "role_config": role_config_record},
            output_files=[
                "response_judge_inputs.jsonl",
                "response_judgments.jsonl",
                "per_item_scores.jsonl",
                "per_signature_scores.json",
                "per_facet_scores.json",
                "evaluation_summary.json",
            ],
            runner=lambda: run_response_judgment_stage(
                evaluation_dir=run_dir / "evaluation",
                benchmark_items=benchmark_items,
                **live_stage_kwargs["response_judgment"],
            ),
        )
        summary_sections["evaluation"] = {
            **downstream_chatbot_summary,
            **response_judgment_summary,
        }
        if not projection_items:
            projection_items = build_single_policy_projection_items(
                benchmark_items=benchmark_items,
                grounded_rows=read_jsonl(run_dir / "grounding" / "grounded_clause_library.jsonl"),
            )
        paired_single_policy_summary = run_stage_with_checkpoint(
            checkpoint_records=checkpoint_records,
            stage_name="paired_single_policy",
            stage_dir=run_dir / "paired_single_policy",
            input_paths=[
                run_dir / "evaluation" / "response_judgments.jsonl",
                run_dir / "grounding" / "grounded_clause_library.jsonl",
            ],
            config={
                "execution_mode": config.execution_mode,
                "role_config": role_config_record,
                "model_roster": list(config.model_names or (config.role_config.downstream_chatbot_model,)),
                "screening_model": config.screening_model,
                "screening_use_hard_suite": config.screening_use_hard_suite,
                "live_max_workers": config.live_max_workers,
            },
            output_files=[
                "single_policy_projection_items.jsonl",
                "chatbot_requests.jsonl",
                "chatbot_responses.jsonl",
                "chatbot_summary.json",
                "response_judge_inputs.jsonl",
                "response_judgments.jsonl",
                "paired_single_composed_summary.json",
            ],
            runner=lambda: run_paired_projection_evaluation_stage(
                paired_dir=run_dir / "paired_single_policy",
                benchmark_items=benchmark_items,
                projection_items=projection_items,
                composed_judgments=read_jsonl(run_dir / "evaluation" / "response_judgments.jsonl"),
                system_prompt=prompt.system_prompt,
                execution_mode=config.execution_mode,
                downstream_client=downstream_live_client,
                downstream_models=config.model_names or (config.role_config.downstream_chatbot_model,),
                response_judge_client=live_client,
                response_judge_model=config.role_config.response_judge_model,
                live_max_workers=config.live_max_workers,
            ),
        )
        summary_sections["paired_single_policy"] = paired_single_policy_summary
        finalize_run_summary(
            run_dir=run_dir,
            status="evaluation_completed",
            summary_sections=summary_sections,
            checkpoint_records=checkpoint_records,
            live_client=live_client,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
