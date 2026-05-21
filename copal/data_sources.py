from __future__ import annotations

from pathlib import Path

from copal.io import iter_jsonl
from copal.models import CompanyWorld, PolicyRule, SystemPromptRecord


def compose_company_key(industry: str, company_index: int, company_name: str) -> str:
    return f"{industry}||{company_index:03d}||{company_name}"


def _first_present(raw: dict[str, object], *keys: str) -> object:
    for key in keys:
        if key in raw:
            return raw[key]
    joined = ", ".join(keys)
    raise KeyError(f"Expected one of [{joined}] in policy rule payload")


def _build_policy_rule(raw: dict[str, object]) -> PolicyRule:
    return PolicyRule(
        rule_id=str(raw["rule_id"]),
        rule_text=str(raw["rule_text"]),
        category=str(raw["category"]),
        severity=str(_first_present(raw, "severity", "severe", "severeity")),
        rationale=str(raw["rationale"]),
        verifiable=bool(raw["verifiable"]),
        verifiability_confidence=str(raw["verifiability_confidence"]),
        raw=dict(raw),
    )


def load_company_worlds(policies_path: Path) -> list[CompanyWorld]:
    worlds: list[CompanyWorld] = []
    industry_counts: dict[str, int] = {}
    for record in iter_jsonl(policies_path):
        industry = str(record["industry"])
        company_index = industry_counts.get(industry, 0)
        industry_counts[industry] = company_index + 1
        enterprise_config = dict(record["enterprise_config"])
        policies = dict(record["policies"])
        company_name = str(enterprise_config["company_name"])
        company_key = compose_company_key(industry, company_index, company_name)
        allowed = [_build_policy_rule(rule) for rule in policies["allowed_behaviors"]]
        prohibited = [_build_policy_rule(rule) for rule in policies["prohibited_behaviors"]]
        worlds.append(
            CompanyWorld(
                company_key=company_key,
                industry=industry,
                company_name=company_name,
                company_index=company_index,
                enterprise_config=enterprise_config,
                allowed_behaviors=allowed,
                prohibited_behaviors=prohibited,
                quality_scores=dict(record["quality_scores"]),
                raw=dict(record),
            )
        )
    return worlds


def load_system_prompts(prompts_path: Path) -> list[SystemPromptRecord]:
    prompts: list[SystemPromptRecord] = []
    for record in iter_jsonl(prompts_path):
        prompts.append(
            SystemPromptRecord(
                company_key=str(record["company_key"]),
                industry=str(record["industry"]),
                company_name=str(record["company_name"]),
                company_index=int(record["company_index"]),
                system_prompt=str(record["system_prompt"]),
                raw=dict(record),
            )
        )
    return prompts


def select_company_world(
    policies_path: Path,
    prompts_path: Path,
    company_key: str,
) -> tuple[CompanyWorld, SystemPromptRecord]:
    worlds_by_key = {world.company_key: world for world in load_company_worlds(policies_path)}
    prompts_by_key = {prompt.company_key: prompt for prompt in load_system_prompts(prompts_path)}

    if company_key not in worlds_by_key:
        raise KeyError(f"Unknown company_key in policies dataset: {company_key}")
    if company_key not in prompts_by_key:
        raise KeyError(f"Unknown company_key in system prompt dataset: {company_key}")

    return worlds_by_key[company_key], prompts_by_key[company_key]
