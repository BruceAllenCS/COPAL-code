from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path

from copal.config import RunConfig
from copal.data_sources import select_company_world
from copal.io import ensure_directory, read_json, write_json
from copal.manifest_compat import manifests_match_for_resume
from copal.llm import live_client_runtime_metadata


def build_run_manifest(config: RunConfig) -> dict[str, object]:
    manifest = {
        "run_id": config.run_id,
        "company_key": config.company_key,
        "execution_mode": config.execution_mode,
        "inputs": {
            "policies_path": str(config.policies_path),
            "prompts_path": str(config.prompts_path),
            "model_name_path": str(config.model_name_path),
        },
        "directories": {
            "runs_dir": str(config.runs_dir),
            "cache_dir": str(config.cache_dir),
        },
        "random_seed": config.random_seed,
        "reference_subset_size": config.reference_subset_size,
        "audit_sample_size": config.audit_sample_size,
        "live_smoke": config.live_smoke,
        "smoke_rule_limit_per_side": config.smoke_rule_limit_per_side,
        "smoke_facet_limit_per_signature": config.smoke_facet_limit_per_signature,
        "models": asdict(config.role_config),
        "model_roster": list(config.model_names),
        "target_signatures": list(config.signatures),
        "facet_library": {signature: list(facets) for signature, facets in config.facet_library.items()},
        "difficulty_screening": {
            "selection_variants_per_facet": config.selection_variants_per_facet,
            "screening_model": config.screening_model,
            "screening_min_score": config.screening_min_score,
            "screening_hard_suite_size": config.screening_hard_suite_size,
            "screening_use_hard_suite": config.screening_use_hard_suite,
        },
    }
    if config.execution_mode == "live":
        manifest["live_client"] = live_client_runtime_metadata()
    if config.stop_after != "evaluation":
        manifest["stop_after"] = config.stop_after
    if config.live_max_workers != 1:
        manifest["live_max_workers"] = config.live_max_workers
    if config.composition_limit_per_signature is not None:
        manifest["composition_limit_per_signature"] = config.composition_limit_per_signature
    if config.composition_adjudication_limit is not None:
        manifest["composition_adjudication_limit"] = config.composition_adjudication_limit
    return manifest


def create_run_id(company_key: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    company_name = company_key.split("||")[-1].lower().replace(" ", "_")
    return f"copal_e2e_{timestamp}_{company_name}"


def initialize_run(config: RunConfig) -> Path:
    run_dir = ensure_directory(config.runs_dir / config.run_id)
    ensure_directory(config.cache_dir)
    manifest = build_run_manifest(config)

    stage_dirs = [
        "inputs",
        "grounding",
        "compositions",
        "query_generation",
        "validation",
        "coverage",
        "difficulty_screening",
        "reference_subset",
        "selection",
        "baselines",
        "audit",
        "evaluation",
        "reports",
        "logs",
    ]
    for stage_dir in stage_dirs:
        ensure_directory(run_dir / stage_dir)

    world, prompt = select_company_world(
        policies_path=config.policies_path,
        prompts_path=config.prompts_path,
        company_key=config.company_key,
    )
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        existing_manifest = read_json(manifest_path)
        if not manifests_match_for_resume(existing_manifest, manifest):
            raise ValueError(f"Existing run manifest does not match requested config: {run_dir}")
    else:
        write_json(manifest_path, manifest)
    runtime_errors_path = run_dir / "runtime_errors.jsonl"
    if not runtime_errors_path.exists():
        runtime_errors_path.write_text("", encoding="utf-8")
    selected_company_path = run_dir / "inputs" / "selected_company.json"
    selected_prompt_path = run_dir / "inputs" / "selected_system_prompt.json"
    if not selected_company_path.exists():
        write_json(selected_company_path, world.raw)
    if not selected_prompt_path.exists():
        write_json(selected_prompt_path, prompt.raw)
    summary_path = run_dir / "reports" / "summary.json"
    if not summary_path.exists():
        write_json(
            summary_path,
            {
                "run_id": config.run_id,
                "company_key": config.company_key,
                "status": "initialized",
            },
        )
    return run_dir
