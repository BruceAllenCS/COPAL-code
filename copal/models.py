from __future__ import annotations

from dataclasses import dataclass
from typing import Any


JsonDict = dict[str, Any]


@dataclass(slots=True)
class PolicyRule:
    rule_id: str
    rule_text: str
    category: str
    severity: str
    rationale: str
    verifiable: bool
    verifiability_confidence: str
    raw: JsonDict


@dataclass(slots=True)
class CompanyWorld:
    company_key: str
    industry: str
    company_name: str
    company_index: int
    enterprise_config: JsonDict
    allowed_behaviors: list[PolicyRule]
    prohibited_behaviors: list[PolicyRule]
    quality_scores: JsonDict
    raw: JsonDict


@dataclass(slots=True)
class SystemPromptRecord:
    company_key: str
    industry: str
    company_name: str
    company_index: int
    system_prompt: str
    raw: JsonDict
