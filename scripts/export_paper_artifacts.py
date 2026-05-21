from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

from copal.io import ensure_directory, iter_jsonl, read_json, read_jsonl, write_json, write_jsonl


DATASET_VERSION = "copal-paper-v1"
ARTIFACT_RELEASE_DATE = "2026-05-21"
VARIANT_IDS = (
    "raw_policy_planning",
    "clause_only_planning",
    "without_facet_query_generation",
    "copal",
)
VALIDATION_COPY_SPECS = (
    (
        "construction_quality",
        "runs/experiments/construction_quality_validation_20260515",
        (
            "annotations.jsonl",
            "construction_quality_samples.jsonl",
            "construction_quality_summary.json",
            "manifest.json",
        ),
    ),
    (
        "llm_annotation_response_judge_v11",
        "runs/experiments/llm_annotation_response_judge_v11_gpt55_claude240",
        (
            "annotation_samples.jsonl",
            "annotations.jsonl",
            "llm_human_validation_summary.json",
            "manifest.json",
        ),
    ),
    (
        "llm_annotation_construction_v2",
        "runs/experiments/llm_human_validation_20260514_v2",
        (
            "annotation_samples.jsonl",
            "annotations.jsonl",
            "llm_human_validation_summary.json",
            "manifest.json",
        ),
    ),
    (
        "manual_construction_quality_adjudication",
        "paper_final/annotation_adjudication/construction_quality_20260515",
        (
            "manual_adjudication_export.json",
            "manual_adjudication_summary.json",
            "manual_adjudication_detailed_results.md",
        ),
    ),
    (
        "manual_response_judge_adjudication",
        "paper_final/annotation_adjudication/response_judge_v11",
        (
            "manual_adjudication_export.json",
            "manual_adjudication_summary.json",
            "reliability_detailed_results.md",
        ),
    ),
)
RUN_MANIFEST_COPY_SPECS = (
    (
        "table2_ablation",
        "runs/experiments/table2_ablation_30c_hiddenfix_20260519",
        (
            "table2_summary.json",
            "hidden_info_repair_manifest.json",
            "hidden_info_repair_applied.jsonl",
        ),
    ),
    (
        "table3_model_eval",
        "runs/experiments/table3_model_eval_30c_10model_seed12_20260514_merged",
        (
            "table3_manifest.json",
            "table3_summary.json",
        ),
    ),
    (
        "paired_single_composed",
        "runs/experiments/paired_single_composed_table3_30c_5model_seed12_20260520_merged",
        (
            "paired_single_composed_summary.json",
        ),
    ),
    (
        "judge_sensitivity",
        "runs/experiments/table3_judge_sensitivity_300case_9model_5judge_20260521",
        (
            "judge_sensitivity_manifest.json",
            "judge_sensitivity_summary.json",
            "pairwise_judge_agreement.json",
            "judge_disagreement_analysis.json",
            "judge_vote_distribution.json",
            "5_judge_majority_error_summary.json",
        ),
    ),
    (
        "paper_final",
        "paper_final/manifests",
        (
            "experiment_manifest.md",
            "final_result_snapshot.md",
            "judge_sensitivity_writeup.md",
            "table2_ablation_diagnosis.md",
            "final_artifacts.json",
        ),
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact source file is missing: {path}")
    return path


def require_dir(path: Path) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"Required artifact source directory is missing: {path}")
    return path


def source_label(copal_root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(copal_root.resolve()))


def count_jsonl(path: Path) -> int:
    return sum(1 for _ in iter_jsonl(path))


def item_company_key(item: dict[str, Any]) -> str:
    if "company_key" in item:
        return str(item["company_key"])
    benchmark_item = item["benchmark_item"]
    return str(benchmark_item["company_key"])


def company_run_dirs(table2_dir: Path) -> list[Path]:
    run_root = require_dir(table2_dir / "company_runs")
    dirs = sorted(path for path in run_root.iterdir() if path.is_dir())
    if len(dirs) != 30:
        raise ValueError(f"Expected 30 company run directories in {run_root}, found {len(dirs)}")
    return dirs


def load_selected_companies(table2_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in company_run_dirs(table2_dir):
        selected = read_json(require_file(run_dir / "selected_company.json"))
        rows.append(
            {
                **selected,
                "company_run_id": run_dir.name,
                "artifact_source": source_label(table2_dir.parents[2], run_dir / "selected_company.json"),
            }
        )
    return rows


def build_company_world_specs(
    *,
    output_dir: Path,
    table2_dir: Path,
    policy_worlds_path: Path,
    companies_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_companies = load_selected_companies(table2_dir)
    selected_keys = [str(row["company_key"]) for row in selected_companies]
    policy_worlds = read_jsonl(policy_worlds_path)
    company_rows = read_jsonl(companies_path)
    if len(policy_worlds) != len(company_rows):
        raise ValueError(
            f"Policy-world/company row count mismatch: {len(policy_worlds)} worlds, {len(company_rows)} companies"
        )
    worlds_by_key: dict[str, dict[str, Any]] = {}
    for world, company in zip(policy_worlds, company_rows, strict=True):
        normalized_world = dict(world)
        normalized_world["company_key"] = company["company_key"]
        worlds_by_key[str(company["company_key"])] = normalized_world
    missing = [key for key in selected_keys if key not in worlds_by_key]
    if missing:
        raise KeyError(f"Selected companies missing from policy worlds: {missing}")

    world_rows = []
    inventory_rows = []
    for selected in selected_companies:
        key = str(selected["company_key"])
        world = dict(worlds_by_key[key])
        world["paper_company_index"] = len(world_rows)
        world["company_run_id"] = selected["company_run_id"]
        world_rows.append(world)

        policies = world["policies"]
        rules: list[dict[str, Any]] = []
        for rule_type, field_name in (("allowed", "allowed_behaviors"), ("prohibited", "prohibited_behaviors")):
            for rule in policies[field_name]:
                rules.append(
                    {
                        "rule_type": rule_type,
                        "rule_id": rule["rule_id"],
                        "category": rule["category"],
                        "severity": rule.get("severity", rule.get("severe", rule.get("severeity"))),
                        "rule_text": rule["rule_text"],
                    }
                )
        inventory_rows.append(
            {
                "company_key": key,
                "company_name": selected["company_name"],
                "industry": selected["industry"],
                "company_run_id": selected["company_run_id"],
                "allowed_rule_count": len(policies["allowed_behaviors"]),
                "prohibited_rule_count": len(policies["prohibited_behaviors"]),
                "total_rule_count": len(rules),
                "rules": rules,
            }
        )

    write_jsonl(output_dir / "company_world_specs.jsonl", world_rows)
    write_jsonl(output_dir / "policy_inventories.jsonl", inventory_rows)
    return world_rows, inventory_rows


def rows_from_company_file(
    *,
    copal_root: Path,
    run_dir: Path,
    selected: dict[str, Any],
    relative_path: Path,
) -> Iterable[dict[str, Any]]:
    source_path = require_file(run_dir / relative_path)
    for row in iter_jsonl(source_path):
        yield {
            **row,
            "company_run_id": run_dir.name,
            "artifact_source": source_label(copal_root, source_path),
            "company_key": row.get("company_key", selected["company_key"]),
            "company_name": row.get("company_name", selected["company_name"]),
        }


def export_grounding_and_compositions(*, copal_root: Path, output_dir: Path, table2_dir: Path) -> tuple[int, int]:
    grounded_rows: list[dict[str, Any]] = []
    composition_rows: list[dict[str, Any]] = []
    for run_dir in company_run_dirs(table2_dir):
        selected = read_json(require_file(run_dir / "selected_company.json"))
        grounded_rows.extend(
            rows_from_company_file(
                copal_root=copal_root,
                run_dir=run_dir,
                selected=selected,
                relative_path=Path("shared_grounding/grounded_clauses.jsonl"),
            )
        )
        composition_rows.extend(
            rows_from_company_file(
                copal_root=copal_root,
                run_dir=run_dir,
                selected=selected,
                relative_path=Path("shared_compositions/accepted_compositions.jsonl"),
            )
        )

    write_jsonl(output_dir / "grounded_clauses.jsonl", grounded_rows)
    write_jsonl(output_dir / "composition_records.jsonl", composition_rows)
    return len(grounded_rows), len(composition_rows)


def variant_source_file(variant_dir: Path, *candidates: str) -> Path:
    existing = [variant_dir / candidate for candidate in candidates if (variant_dir / candidate).is_file()]
    if len(existing) != 1:
        candidate_text = ", ".join(candidates)
        raise FileNotFoundError(f"Expected exactly one of [{candidate_text}] in {variant_dir}, found {existing}")
    return existing[0]


def export_candidate_and_screening_logs(*, copal_root: Path, output_dir: Path, table2_dir: Path) -> tuple[int, int, int]:
    generated_rows: list[dict[str, Any]] = []
    screening_rows: list[dict[str, Any]] = []
    ablation_pool_rows: list[dict[str, Any]] = []

    for run_dir in company_run_dirs(table2_dir):
        selected = read_json(require_file(run_dir / "selected_company.json"))
        for variant_id in VARIANT_IDS:
            variant_dir = require_dir(run_dir / "variants" / variant_id)
            generation_file = variant_source_file(
                variant_dir,
                "generation/candidate_queries.jsonl",
                "generation/candidate_queries_unmapped.jsonl",
            )
            labeled_file = require_file(variant_dir / "candidate_queries_labeled.jsonl")

            for row in iter_jsonl(generation_file):
                generated_rows.append(
                    {
                        **row,
                        "company_run_id": run_dir.name,
                        "variant_id": variant_id,
                        "artifact_source": source_label(copal_root, generation_file),
                        "company_key": row.get("company_key", selected["company_key"]),
                        "company_name": row.get("company_name", selected["company_name"]),
                    }
                )
            for row in iter_jsonl(labeled_file):
                ablation_pool_rows.append(
                    {
                        **row,
                        "company_run_id": run_dir.name,
                        "variant_id": variant_id,
                        "candidate_pool_source": source_label(copal_root, labeled_file),
                        "company_key": row.get("company_key", selected["company_key"]),
                        "company_name": row.get("company_name", selected["company_name"]),
                    }
                )

            for jsonl_path, record_type in (
                (variant_dir / "query_screening/selected_queries.jsonl", "query_screening_selected_query"),
                (variant_dir / "posthoc_mapping/posthoc_labels.jsonl", "posthoc_mapping_label"),
            ):
                if jsonl_path.is_file():
                    for row in iter_jsonl(jsonl_path):
                        screening_rows.append(
                            {
                                "record_type": record_type,
                                "company_key": selected["company_key"],
                                "company_name": selected["company_name"],
                                "company_run_id": run_dir.name,
                                "variant_id": variant_id,
                                "artifact_source": source_label(copal_root, jsonl_path),
                                "payload": row,
                            }
                        )

            for json_path, record_type in (
                (variant_dir / "generation/query_generation_summary.json", "query_generation_summary"),
                (variant_dir / "generation/direct_planning_summary.json", "direct_planning_summary"),
                (variant_dir / "posthoc_mapping/posthoc_mapping_summary.json", "posthoc_mapping_summary"),
                (variant_dir / "query_screening/query_screening_summary.json", "query_screening_summary"),
                (variant_dir / "table2_variant_summary.json", "table2_variant_summary"),
            ):
                if json_path.is_file():
                    screening_rows.append(
                        {
                            "record_type": record_type,
                            "company_key": selected["company_key"],
                            "company_name": selected["company_name"],
                            "company_run_id": run_dir.name,
                            "variant_id": variant_id,
                            "artifact_source": source_label(copal_root, json_path),
                            "payload": read_json(json_path),
                        }
                    )

    write_jsonl(output_dir / "generated_candidate_queries.jsonl", generated_rows)
    write_jsonl(output_dir / "screening_mapping_logs.jsonl", screening_rows)
    write_jsonl(output_dir / "ablation_candidate_pools.jsonl", ablation_pool_rows)
    return len(generated_rows), len(screening_rows), len(ablation_pool_rows)


def table3_company_runs(*, copal_root: Path, table3_merged_dir: Path) -> list[tuple[str, Path]]:
    summary = read_json(require_file(table3_merged_dir / "table3_summary.json"))
    runs: list[tuple[str, Path]] = []
    for completed in summary["completed_runs"]:
        shard_id = str(completed["shard_id"])
        run_dir = copal_root / str(completed["run_dir"])
        require_dir(run_dir)
        runs.append((shard_id, run_dir))
    if len(runs) != 30:
        raise ValueError(f"Expected 30 Table3 company runs, found {len(runs)}")
    return runs


def export_selected_suites_and_contracts(
    *,
    copal_root: Path,
    output_dir: Path,
    table2_dir: Path,
    table3_merged_dir: Path,
) -> tuple[int, int]:
    selected_suite_rows: list[dict[str, Any]] = []
    contract_rows: list[dict[str, Any]] = []

    for shard_id, run_dir in table3_company_runs(copal_root=copal_root, table3_merged_dir=table3_merged_dir):
        selected_file = require_file(run_dir / "selected_items.jsonl")
        for row in iter_jsonl(selected_file):
            item = {
                **row,
                "suite_family": "table3_model_eval_composed",
                "suite_id": "table3_30_companies_30_cases",
                "table3_shard_id": shard_id,
                "company_run_id": run_dir.name,
                "artifact_source": source_label(copal_root, selected_file),
            }
            selected_suite_rows.append(item)
            contract_rows.append(build_contract_record(item))

    for run_dir in company_run_dirs(table2_dir):
        selected = read_json(require_file(run_dir / "selected_company.json"))
        for variant_id in VARIANT_IDS:
            suite_file = require_file(run_dir / "variants" / variant_id / "benchmark_items_final.jsonl")
            for row in iter_jsonl(suite_file):
                item = {
                    **row,
                    "company_key": row.get("company_key", selected["company_key"]),
                    "company_name": row.get("company_name", selected["company_name"]),
                    "suite_family": "table2_ablation_variant",
                    "suite_id": f"table2_{variant_id}_30_companies_12_cases",
                    "variant_id": variant_id,
                    "company_run_id": run_dir.name,
                    "artifact_source": source_label(copal_root, suite_file),
                }
                selected_suite_rows.append(item)
                contract_rows.append(build_contract_record(item))

    write_jsonl(output_dir / "final_selected_suites.jsonl", selected_suite_rows)
    write_jsonl(output_dir / "handling_contracts.jsonl", contract_rows)
    return len(selected_suite_rows), len(contract_rows)


def build_contract_record(item: dict[str, Any]) -> dict[str, Any]:
    expected = item["expected_handling"]
    strict = expected["strict_response_contract"]
    generated_contract = strict["generated_case_contract"]
    return {
        "contract_id": f"{item['suite_family']}::{item.get('variant_id', 'copal')}::{item['item_id']}",
        "item_id": item["item_id"],
        "query_id": item["query_id"],
        "company_key": item_company_key(item),
        "suite_family": item["suite_family"],
        "suite_id": item["suite_id"],
        "variant_id": item.get("variant_id"),
        "relation_pattern": item["relation_pattern"],
        "target_facets": item["target_facets"],
        "active_clause_ids": item["active_clause_ids"],
        "acceptable_handling": expected["acceptable_handling"],
        "disallowed_handling": item["disallowed_handling"],
        "must_satisfy": strict["must_satisfy"],
        "must_avoid": strict["must_avoid"],
        "incorrect_shortcuts": strict["incorrect_shortcuts"],
        "generated_case_contract": generated_contract,
        "artifact_source": item["artifact_source"],
    }


def export_model_artifacts(
    *,
    copal_root: Path,
    output_dir: Path,
    table3_merged_dir: Path,
) -> tuple[int, int, int]:
    prompt_rows_by_company: dict[str, dict[str, Any]] = {}
    model_output_rows: list[dict[str, Any]] = []

    for shard_id, run_dir in table3_company_runs(copal_root=copal_root, table3_merged_dir=table3_merged_dir):
        request_file = require_file(run_dir / "evaluation/chatbot_requests.jsonl")
        response_file = require_file(run_dir / "evaluation/chatbot_responses.jsonl")
        for row in iter_jsonl(request_file):
            company_key = str(row["item_id"]).split("::", 1)[0]
            if company_key not in prompt_rows_by_company:
                prompt_rows_by_company[company_key] = {
                    "company_key": company_key,
                    "system_prompt": row["system_prompt"],
                    "first_query_text": row["query_text"],
                    "first_item_id": row["item_id"],
                    "first_response_model": row["response_id"].rsplit("::", 1)[-1],
                    "table3_shard_id": shard_id,
                    "company_run_id": run_dir.name,
                    "artifact_source": source_label(copal_root, request_file),
                }
        for row in iter_jsonl(response_file):
            model_output_rows.append(
                {
                    **row,
                    "table3_shard_id": shard_id,
                    "company_run_id": run_dir.name,
                    "artifact_source": source_label(copal_root, response_file),
                }
            )

    if len(prompt_rows_by_company) != 30:
        raise ValueError(f"Expected 30 reconstructed chatbot prompts, found {len(prompt_rows_by_company)}")

    label_source = require_file(table3_merged_dir / "response_judgments.jsonl")
    judge_label_rows = [
        {
            **row,
            "artifact_source": source_label(copal_root, label_source),
        }
        for row in iter_jsonl(label_source)
    ]

    write_jsonl(output_dir / "reconstructed_chatbot_prompts.jsonl", prompt_rows_by_company.values())
    write_jsonl(output_dir / "model_outputs.jsonl", model_output_rows)
    write_jsonl(output_dir / "automatic_judge_labels.jsonl", judge_label_rows)
    return len(prompt_rows_by_company), len(model_output_rows), len(judge_label_rows)


def copy_curated_files(
    *,
    copal_root: Path,
    output_dir: Path,
    subdir: str,
    copy_specs: Iterable[tuple[str, str, tuple[str, ...]]],
) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    base_output = ensure_directory(output_dir / subdir)
    for group_name, source_dir_rel, filenames in copy_specs:
        source_dir = require_dir(copal_root / source_dir_rel)
        group_output = ensure_directory(base_output / group_name)
        for filename in filenames:
            source_path = require_file(source_dir / filename)
            destination = group_output / filename
            shutil.copy2(source_path, destination)
            if destination.suffix == ".md":
                lines = destination.read_text(encoding="utf-8").splitlines()
                destination.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")
            copied.append(
                {
                    "group": group_name,
                    "path": str(destination.relative_to(output_dir)),
                    "source_path": source_label(copal_root, source_path),
                    "bytes": destination.stat().st_size,
                    "sha256": sha256_file(destination),
                }
            )
    return copied


def list_artifact_files(output_dir: Path) -> list[dict[str, Any]]:
    files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    return [
        {
            "path": str(path.relative_to(output_dir)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
        if path.name != "manifest.json"
    ]


def export_paper_artifacts(
    *,
    dataset_dir: Path,
    copal_root: Path,
    update_dataset_manifest: bool,
) -> dict[str, Any]:
    output_dir = ensure_directory(dataset_dir / "artifacts")
    table2_dir = require_dir(copal_root / "runs/experiments/table2_ablation_30c_hiddenfix_20260519")
    table3_merged_dir = require_dir(copal_root / "runs/experiments/table3_model_eval_30c_10model_seed12_20260514_merged")
    policy_worlds_path = require_file(dataset_dir / "policies/policy_worlds.jsonl")
    companies_path = require_file(dataset_dir / "companies/companies.jsonl")

    world_rows, inventory_rows = build_company_world_specs(
        output_dir=output_dir,
        table2_dir=table2_dir,
        policy_worlds_path=policy_worlds_path,
        companies_path=companies_path,
    )
    grounded_count, composition_count = export_grounding_and_compositions(
        copal_root=copal_root,
        output_dir=output_dir,
        table2_dir=table2_dir,
    )
    generated_count, screening_count, ablation_pool_count = export_candidate_and_screening_logs(
        copal_root=copal_root,
        output_dir=output_dir,
        table2_dir=table2_dir,
    )
    selected_suite_count, contract_count = export_selected_suites_and_contracts(
        copal_root=copal_root,
        output_dir=output_dir,
        table2_dir=table2_dir,
        table3_merged_dir=table3_merged_dir,
    )
    prompt_count, model_output_count, judge_label_count = export_model_artifacts(
        copal_root=copal_root,
        output_dir=output_dir,
        table3_merged_dir=table3_merged_dir,
    )
    validation_files = copy_curated_files(
        copal_root=copal_root,
        output_dir=output_dir,
        subdir="validation_records",
        copy_specs=VALIDATION_COPY_SPECS,
    )
    run_manifest_files = copy_curated_files(
        copal_root=copal_root,
        output_dir=output_dir,
        subdir="run_manifests",
        copy_specs=RUN_MANIFEST_COPY_SPECS,
    )

    manifest = {
        "dataset_version": DATASET_VERSION,
        "artifact_release_date": ARTIFACT_RELEASE_DATE,
        "description": (
            "Curated COPAL paper experiment artifacts: the 30-company paper slice, "
            "grounded clauses, policy-composition records, generated/screened queries, "
            "final suites, handling contracts, model outputs, judge labels, validation "
            "records, and run manifests."
        ),
        "source_root": "<COPAL_WORKSPACE_ROOT>",
        "required_artifact_families": {
            "30_company_world_specs": "company_world_specs.jsonl",
            "policy_inventories": "policy_inventories.jsonl",
            "grounded_clauses": "grounded_clauses.jsonl",
            "composition_records": "composition_records.jsonl",
            "generated_candidate_queries": "generated_candidate_queries.jsonl",
            "screening_mapping_logs": "screening_mapping_logs.jsonl",
            "final_selected_suites": "final_selected_suites.jsonl",
            "handling_contracts": "handling_contracts.jsonl",
            "reconstructed_chatbot_prompts": "reconstructed_chatbot_prompts.jsonl",
            "construction_and_judge_prompt_templates": "../prompts/copal_prompt_templates.json",
            "model_outputs": "model_outputs.jsonl",
            "automatic_judge_labels": "automatic_judge_labels.jsonl",
            "ablation_candidate_pools": "ablation_candidate_pools.jsonl",
            "validation_records": "validation_records/",
            "run_manifests": "run_manifests/",
        },
        "counts": {
            "company_world_specs": len(world_rows),
            "policy_inventories": len(inventory_rows),
            "grounded_clauses": grounded_count,
            "composition_records": composition_count,
            "generated_candidate_queries": generated_count,
            "screening_mapping_logs": screening_count,
            "final_selected_suite_items": selected_suite_count,
            "handling_contracts": contract_count,
            "reconstructed_chatbot_prompts": prompt_count,
            "model_outputs": model_output_count,
            "automatic_judge_labels": judge_label_count,
            "ablation_candidate_pool_items": ablation_pool_count,
            "validation_record_files": len(validation_files),
            "run_manifest_files": len(run_manifest_files),
        },
        "validation_record_files": validation_files,
        "run_manifest_files": run_manifest_files,
        "files": list_artifact_files(output_dir),
        "excluded_artifacts": [
            "llm_cache directories",
            "provider transport logs beyond curated live error summaries",
            "private/internal real-bot deployment probes",
        ],
    }
    write_json(output_dir / "manifest.json", manifest)

    if update_dataset_manifest:
        dataset_manifest_path = require_file(dataset_dir / "manifest.json")
        dataset_manifest = read_json(dataset_manifest_path)
        dataset_manifest["paper_artifacts"] = {
            "manifest_path": "artifacts/manifest.json",
            "counts": manifest["counts"],
            "required_artifact_families": manifest["required_artifact_families"],
        }
        write_json(dataset_manifest_path, dataset_manifest)

    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export curated COPAL paper experiment artifacts.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/copal-paper-v1"))
    parser.add_argument(
        "--copal-root",
        type=Path,
        default=Path("../.."),
        help="Root of the full COPAL workspace containing runs/experiments and paper_final.",
    )
    parser.add_argument(
        "--no-update-dataset-manifest",
        action="store_true",
        help="Do not add the artifact manifest summary to datasets/copal-paper-v1/manifest.json.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    export_paper_artifacts(
        dataset_dir=args.dataset_dir,
        copal_root=args.copal_root.resolve(),
        update_dataset_manifest=not args.no_update_dataset_manifest,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
