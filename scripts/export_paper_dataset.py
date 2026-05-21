from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

from copal import prompts
from copal.data_sources import compose_company_key
from copal.io import ensure_directory, read_json, read_jsonl, write_json, write_jsonl


DATASET_VERSION = "copal-paper-v1"
DATASET_RELEASE_DATE = "2026-05-21"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_company_rows(policy_worlds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    industry_counts: dict[str, int] = {}
    for world in policy_worlds:
        enterprise = dict(world["enterprise_config"])
        policies = dict(world["policies"])
        allowed = list(policies["allowed_behaviors"])
        prohibited = list(policies["prohibited_behaviors"])
        industry = str(world["industry"])
        company_index = industry_counts.get(industry, 0)
        industry_counts[industry] = company_index + 1
        company_key = str(world.get("company_key", "")).strip()
        if not company_key:
            company_key = compose_company_key(industry, company_index, str(enterprise["company_name"]))
        rows.append(
            {
                "company_key": company_key,
                "industry": industry,
                "company_name": enterprise["company_name"],
                "subtype": enterprise.get("subtype", ""),
                "company_size": enterprise.get("company_size", ""),
                "geographic_focus": enterprise.get("geographic_focus", ""),
                "customer_segment": enterprise.get("customer_segment", ""),
                "primary_offering": enterprise.get("primary_offering", ""),
                "chatbot_use_case": enterprise.get("chatbot_use_case", ""),
                "regulatory_constraints": list(enterprise.get("regulatory_constraints", [])),
                "allowed_rule_count": len(allowed),
                "prohibited_rule_count": len(prohibited),
                "total_rule_count": len(allowed) + len(prohibited),
                "quality_scores": dict(world.get("quality_scores", {})),
            }
        )
    return rows


def build_policy_rule_rows(policy_worlds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    industry_counts: dict[str, int] = {}
    for world in policy_worlds:
        enterprise = dict(world["enterprise_config"])
        industry = str(world["industry"])
        company_index = industry_counts.get(industry, 0)
        industry_counts[industry] = company_index + 1
        company_key = str(world.get("company_key", "")).strip()
        if not company_key:
            company_key = compose_company_key(industry, company_index, str(enterprise["company_name"]))
        policies = dict(world["policies"])
        for rule_type, field_name in (("allowed", "allowed_behaviors"), ("prohibited", "prohibited_behaviors")):
            for rule in policies[field_name]:
                rows.append(
                    {
                        "company_key": company_key,
                        "industry": industry,
                        "company_name": enterprise["company_name"],
                        "rule_type": rule_type,
                        "rule_id": rule["rule_id"],
                        "rule_text": rule["rule_text"],
                        "category": rule["category"],
                        "severity": first_present(rule, "severity", "severe", "severeity"),
                        "rationale": rule["rationale"],
                        "verifiable": rule["verifiable"],
                        "verifiability_confidence": rule["verifiability_confidence"],
                        "raw_rule": dict(rule),
                    }
                )
    return rows


def first_present(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    joined = ", ".join(keys)
    raise KeyError(f"Expected one of [{joined}] in payload")


def build_prompt_template_record() -> dict[str, Any]:
    return {
        "dataset_version": DATASET_VERSION,
        "template_source_file": "prompts/copal_prompt_templates.py",
        "system_prompts": {
            "grounding": prompts.GROUNDING_SYSTEM_PROMPT,
            "validation": prompts.VALIDATION_SYSTEM_PROMPT,
            "coverage": prompts.COVERAGE_SYSTEM_PROMPT,
            "response_judge": prompts.RESPONSE_JUDGE_SYSTEM_PROMPT,
        },
        "message_builders": [
            {
                "name": "build_clause_extraction_messages",
                "stage": "clause_grounding",
                "purpose": "Extract trigger, scope, effect, and source evidence from each source policy rule.",
            },
            {
                "name": "build_clause_canonicalization_messages",
                "stage": "clause_grounding",
                "purpose": "Canonicalize extracted clauses into COPAL's grounded-clause schema.",
            },
            {
                "name": "build_composition_adjudication_messages",
                "stage": "composition_validation",
                "purpose": "Validate unresolved clause compositions and assign one relation pattern.",
            },
            {
                "name": "build_query_verbalization_messages",
                "stage": "query_generation",
                "purpose": "Generate realistic user queries for accepted composed-policy interactions.",
            },
            {
                "name": "build_query_validation_messages",
                "stage": "query_validation",
                "purpose": "Validate query naturalness, non-separability, and target-facet coverage.",
            },
            {
                "name": "build_coverage_messages",
                "stage": "coverage_selection",
                "purpose": "Assign relation-pattern and target-facet coverage labels.",
            },
            {
                "name": "build_downstream_chat_messages",
                "stage": "chatbot_evaluation",
                "purpose": "Wrap each selected query for a target chatbot.",
            },
            {
                "name": "build_response_judge_messages",
                "stage": "response_judgment",
                "purpose": "Judge chatbot responses against expected and forbidden handling contracts.",
            },
        ],
    }


def build_manifest(
    *,
    output_dir: Path,
    policy_worlds: list[dict[str, Any]],
    system_prompts: list[dict[str, Any]],
    company_rows: list[dict[str, Any]],
    policy_rule_rows: list[dict[str, Any]],
    source_policy_path: Path,
    source_system_prompt_path: Path,
    source_summary_path: Path,
) -> dict[str, Any]:
    files = [
        output_dir / "companies" / "companies.jsonl",
        output_dir / "policies" / "policy_worlds.jsonl",
        output_dir / "policies" / "policy_rules.jsonl",
        output_dir / "prompts" / "system_prompts.jsonl",
        output_dir / "prompts" / "copal_prompt_templates.json",
        output_dir / "prompts" / "copal_prompt_templates.py",
        output_dir / "metadata" / "dataset_summary.json",
    ]
    industries = sorted({str(world["industry"]) for world in policy_worlds})
    return {
        "dataset_name": "COPAL Paper Reproducibility Dataset",
        "dataset_version": DATASET_VERSION,
        "release_date": DATASET_RELEASE_DATE,
        "description": (
            "Synthetic organizational-policy chatbot dataset used by the COPAL paper. "
            "Includes company metadata, source policies, deployment system prompts, "
            "and COPAL construction/evaluation prompt templates."
        ),
        "source_files": {
            "policy_worlds": str(source_policy_path),
            "system_prompts": str(source_system_prompt_path),
            "dataset_summary": str(source_summary_path),
            "copal_prompt_source": "copal/prompts.py",
        },
        "counts": {
            "industries": len(industries),
            "companies": len(company_rows),
            "policy_worlds": len(policy_worlds),
            "system_prompts": len(system_prompts),
            "policy_rules": len(policy_rule_rows),
            "allowed_policy_rules": sum(1 for row in policy_rule_rows if row["rule_type"] == "allowed"),
            "prohibited_policy_rules": sum(1 for row in policy_rule_rows if row["rule_type"] == "prohibited"),
        },
        "industries": industries,
        "files": [
            {
                "path": str(path.relative_to(output_dir)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in files
        ],
        "license": "MIT for code and dataset files in this repository unless otherwise noted.",
    }


def export_dataset(
    *,
    output_dir: Path,
    source_policy_path: Path,
    source_system_prompt_path: Path,
    source_summary_path: Path,
) -> dict[str, Any]:
    policy_worlds = read_jsonl(source_policy_path)
    system_prompts = read_jsonl(source_system_prompt_path)
    summary = read_json(source_summary_path)
    company_rows = build_company_rows(policy_worlds)
    policy_rule_rows = build_policy_rule_rows(policy_worlds)

    ensure_directory(output_dir / "companies")
    ensure_directory(output_dir / "policies")
    ensure_directory(output_dir / "prompts")
    ensure_directory(output_dir / "metadata")

    write_jsonl(output_dir / "companies" / "companies.jsonl", company_rows)
    write_jsonl(output_dir / "policies" / "policy_worlds.jsonl", policy_worlds)
    write_jsonl(output_dir / "policies" / "policy_rules.jsonl", policy_rule_rows)
    write_jsonl(output_dir / "prompts" / "system_prompts.jsonl", system_prompts)
    write_json(output_dir / "prompts" / "copal_prompt_templates.json", build_prompt_template_record())
    (output_dir / "prompts" / "copal_prompt_templates.py").write_text(
        inspect.getsource(prompts),
        encoding="utf-8",
    )
    write_json(output_dir / "metadata" / "dataset_summary.json", summary)

    manifest = build_manifest(
        output_dir=output_dir,
        policy_worlds=policy_worlds,
        system_prompts=system_prompts,
        company_rows=company_rows,
        policy_rule_rows=policy_rule_rows,
        source_policy_path=source_policy_path,
        source_system_prompt_path=source_system_prompt_path,
        source_summary_path=source_summary_path,
    )
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export the COPAL paper reproducibility dataset.")
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/copal-paper-v1"))
    parser.add_argument("--source-policy-path", type=Path, default=Path("data/compass_policies/compass_policies_final.jsonl"))
    parser.add_argument("--source-system-prompt-path", type=Path, default=Path("data/compass_policies/company_system_prompts.jsonl"))
    parser.add_argument("--source-summary-path", type=Path, default=Path("data/compass_policies/dataset_summary.json"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    export_dataset(
        output_dir=args.output_dir,
        source_policy_path=args.source_policy_path,
        source_system_prompt_path=args.source_system_prompt_path,
        source_summary_path=args.source_summary_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
