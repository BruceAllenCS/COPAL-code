from __future__ import annotations

from hashlib import sha1
from pathlib import Path
from typing import Iterable

from copal.config import require_execution_mode
from copal.io import ensure_directory, write_json, write_jsonl
from copal.llm import LLMClient, complete_json
from copal.models import CompanyWorld, PolicyRule
from copal.prompts import build_clause_canonicalization_messages, build_clause_extraction_messages
from copal.taxonomy import normalize_effect_label as _canonicalize_effect_label

SCOPE_HINTS: tuple[tuple[str, str], ...] = (
    ("flight status", "flight_status"),
    ("gate", "flight_status"),
    ("boarding", "flight_status"),
    ("refund", "refund_processing"),
    ("rebook", "rebooking"),
    ("booking", "booking_management"),
    ("itinerary", "itinerary_details"),
    ("baggage", "baggage_support"),
    ("compensation", "compensation_handling"),
    ("medical", "medical_or_liability"),
    ("health", "medical_or_liability"),
    ("injury", "medical_or_liability"),
    ("passport", "travel_documents"),
    ("identity verification", "identity_verification"),
    ("verify", "identity_verification"),
    ("payment", "financial_transactions"),
    ("credit card", "financial_transactions"),
    ("loyalty", "loyalty_account"),
)

TRIGGER_ONTOLOGY_FIELDS: tuple[str, ...] = (
    "request_intent",
    "user_account_state",
    "dialogue_history",
    "entity_type",
    "external_action_state",
)

TRIGGER_MARKERS: tuple[str, ...] = (
    " when ",
    " during ",
    " if ",
    " for ",
    " involving ",
    " regarding ",
    " after ",
)


def normalize_clause_row(
    *,
    company_key: str,
    source_rule_id: str,
    source_rule_type: str,
    clause: dict[str, object],
) -> dict[str, object]:
    text = str(clause["clause_text"]).strip()
    trigger_text, trigger_ontology = _normalize_trigger(clause["trigger"])
    scope_text, scope_semantic_type, scope_entity_types = _normalize_scope(clause["scope"])
    effect = _normalize_effect_label(str(clause["effect"]).strip())
    source_span = _normalize_source_span(clause["source_span"])
    digest = sha1(
        f"{company_key}|{source_rule_id}|{source_rule_type}|{text}|"
        f"{trigger_text}|{scope_text}|{scope_semantic_type}|{effect}|{source_span}".encode("utf-8")
    )
    clause_id = digest.hexdigest()[:12]
    audit_metadata = dict(clause.get("audit_metadata", {}))
    if "priority_notes" in clause:
        audit_metadata["priority_notes"] = _normalize_priority_notes(str(clause["priority_notes"]).strip())
    if "exceptions" in clause:
        audit_metadata["exceptions"] = list(clause["exceptions"])
    return {
        "company_key": company_key,
        "source_rule_id": source_rule_id,
        "source_rule_type": source_rule_type,
        "clause_id": clause_id,
        "clause_text": text,
        "source_span": source_span,
        "trigger": trigger_text,
        "trigger_ontology": trigger_ontology,
        "scope": scope_text,
        "scope_description": scope_text,
        "scope_semantic_type": scope_semantic_type,
        "scope_entity_types": scope_entity_types,
        "effect": effect,
        "provenance": {
            "source_rule_id": source_rule_id,
            "source_rule_type": source_rule_type,
            "source_span": source_span,
        },
        "audit_metadata": audit_metadata,
        "grounding_meta": dict(clause.get("grounding_meta", {})),
    }


def _normalize_trigger(value: object) -> tuple[str, dict[str, str]]:
    if isinstance(value, dict):
        ontology = {field: str(value.get(field, "")).strip() for field in TRIGGER_ONTOLOGY_FIELDS}
        trigger_text = str(value.get("source_text", "") or ontology["request_intent"]).strip()
        if not trigger_text:
            raise ValueError("Grounded clause trigger must include source_text or request_intent")
        return trigger_text, ontology
    trigger_text = str(value).strip()
    if not trigger_text:
        raise ValueError("Grounded clause trigger must be non-empty")
    return trigger_text, {
        "request_intent": trigger_text,
        "user_account_state": "",
        "dialogue_history": "",
        "entity_type": "",
        "external_action_state": "",
    }


def _normalize_scope(value: object) -> tuple[str, str, list[str]]:
    if isinstance(value, dict):
        description = str(value["description"]).strip()
        semantic_type = str(value["semantic_type"]).strip()
        entity_types = [str(entity).strip() for entity in value["entity_types"]]
        if not description or not semantic_type:
            raise ValueError("Grounded clause scope must include description and semantic_type")
        return description, semantic_type, entity_types
    description = str(value).strip()
    if not description:
        raise ValueError("Grounded clause scope must be non-empty")
    return description, _normalize_fragment(description), []


def _normalize_effect_label(effect: str) -> str:
    return normalize_effect_label(effect)


def normalize_effect_label(effect: object) -> str:
    return _canonicalize_effect_label(effect)


def _normalize_source_span(value: object) -> str:
    source_span = str(value).strip()
    if not source_span:
        raise ValueError("Grounded clause source_span must be non-empty")
    return source_span


def _normalize_priority_notes(value: str) -> str:
    lowered = value.lower()
    explicit_markers = ("exception", "emergency", "escalat", "precedence", "priority", "override")
    if any(marker in lowered for marker in explicit_markers):
        return value
    return ""


def dedupe_exact_clauses(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    kept: list[dict[str, object]] = []
    removed: list[dict[str, object]] = []
    seen: dict[tuple[str, str, str, str], dict[str, object]] = {}
    for row in rows:
        scope_description = str(row.get("scope_description", row["scope"])).strip()
        scope_semantic_type = str(row.get("scope_semantic_type", _normalize_fragment(scope_description))).strip()
        fingerprint = (
            str(row["clause_text"]).strip(),
            str(row["trigger"]).strip(),
            scope_description,
            scope_semantic_type,
            str(row["effect"]).strip(),
        )
        if fingerprint in seen:
            removed.append(row)
            continue
        seen[fingerprint] = row
        kept.append(row)
    return kept, removed


def _normalize_fragment(value: str) -> str:
    return value.strip(" .,:;").lower().replace("/", " ").replace("-", " ").replace(" ", "_")


def _extract_trigger(rule_text: str, category: str) -> str:
    lowered = f" {rule_text.lower()} "
    for marker in TRIGGER_MARKERS:
        start = lowered.find(marker)
        if start == -1:
            continue
        raw_fragment = rule_text[start + len(marker) - 1 :]
        fragment = raw_fragment.split(".")[0].split(",")[0].strip()
        normalized = _normalize_fragment(fragment)
        if normalized:
            return normalized
    return _normalize_fragment(category)


def _infer_scope(rule: PolicyRule) -> str:
    lowered = rule.rule_text.lower()
    for needle, scope in SCOPE_HINTS:
        if needle in lowered:
            return scope
    return _normalize_fragment(rule.category)


def _infer_effect(rule: PolicyRule, source_rule_type: str) -> str:
    lowered = rule.rule_text.lower()
    authority_markers = (
        "authority",
        "authorized",
        "authorization",
        "supervisor",
        "manager approval",
        "human agent",
        "legal advice",
        "medical advice",
        "cannot commit",
        "may not promise",
    )
    if any(marker in lowered for marker in authority_markers):
        return "authority-limit"
    if "override" in lowered or "exception" in lowered or "precedence" in lowered or "priority" in lowered:
        return "override"
    if "escalat" in lowered or rule.category == "escalation":
        return "route"
    if "redirect" in lowered:
        return "route"
    if "disclose" in lowered:
        return "disclose"
    if "withhold" in lowered:
        return "withhold"
    if source_rule_type == "prohibited":
        return "prohibit"
    gating_markers = (
        "request explicit confirmation",
        "before finalizing",
        "before processing",
        "identity-verified",
        "verified passenger",
        "authenticated",
    )
    if any(marker in lowered for marker in gating_markers):
        return "require-gate"
    return "permit"


def _priority_notes(rule: PolicyRule) -> str:
    lowered = rule.rule_text.lower()
    explicit_markers = ("exception", "emergency", "escalat", "precedence", "priority", "override")
    if any(marker in lowered for marker in explicit_markers):
        return rule.category
    return ""


def build_clause_candidate(
    *,
    company_key: str,
    rule: PolicyRule,
    source_rule_type: str,
) -> dict[str, object]:
    trigger = _extract_trigger(rule.rule_text, rule.category)
    scope = _infer_scope(rule)
    return {
        "clause_text": rule.rule_text,
        "trigger": {
            "source_text": trigger,
            "request_intent": trigger,
            "user_account_state": "",
            "dialogue_history": "",
            "entity_type": "",
            "external_action_state": "",
        },
        "scope": {
            "description": scope,
            "semantic_type": scope,
            "entity_types": [],
        },
        "effect": _infer_effect(rule, source_rule_type),
        "source_span": rule.rule_text,
        "audit_metadata": {
            "priority_notes": _priority_notes(rule),
            "exceptions": [],
        },
        "grounding_meta": {
            "category": rule.category,
            "severity": rule.severity,
            "verifiable": rule.verifiable,
            "verifiability_confidence": rule.verifiability_confidence,
        },
    }


def propose_grounded_clauses(world: CompanyWorld) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    source_groups: Iterable[tuple[str, list[PolicyRule]]] = (
        ("allowed", world.allowed_behaviors),
        ("prohibited", world.prohibited_behaviors),
    )
    for source_rule_type, rules in source_groups:
        for rule in rules:
            rows.append(
                normalize_clause_row(
                    company_key=world.company_key,
                    source_rule_id=rule.rule_id,
                    source_rule_type=source_rule_type,
                    clause=build_clause_candidate(
                        company_key=world.company_key,
                        rule=rule,
                        source_rule_type=source_rule_type,
                    ),
                )
            )
    return rows


def _live_extract_rule_clauses(
    *,
    world: CompanyWorld,
    rule: PolicyRule,
    source_rule_type: str,
    proposal_client: LLMClient,
    canonicalization_client: LLMClient,
    proposal_model: str,
    canonicalization_model: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    extraction_payload = complete_json(
        client=proposal_client,
        model=proposal_model,
        messages=build_clause_extraction_messages(
            company_key=world.company_key,
            rule=rule,
            source_rule_type=source_rule_type,
        ),
    )
    raw_clauses = extraction_payload["clauses"] if isinstance(extraction_payload, dict) else extraction_payload
    normalized_rows: list[dict[str, object]] = []
    for clause in raw_clauses:
        canonicalization_payload = complete_json(
            client=canonicalization_client,
            model=canonicalization_model,
            messages=build_clause_canonicalization_messages(
                company_key=world.company_key,
                source_rule_id=rule.rule_id,
                source_rule_type=source_rule_type,
                clause=dict(clause),
            ),
        )
        canonical_clause = canonicalization_payload.get("clause", canonicalization_payload)
        canonical_clause["grounding_meta"] = {
            "category": rule.category,
            "severity": rule.severity,
            "verifiable": rule.verifiable,
            "verifiability_confidence": rule.verifiability_confidence,
            "llm_extracted": True,
        }
        normalized_rows.append(
            normalize_clause_row(
                company_key=world.company_key,
                source_rule_id=rule.rule_id,
                source_rule_type=source_rule_type,
                clause=canonical_clause,
            )
        )
    return raw_clauses, normalized_rows


def run_grounding_stage(
    *,
    grounding_dir: Path,
    world: CompanyWorld,
    execution_mode: str,
    proposal_client: LLMClient | None = None,
    canonicalization_client: LLMClient | None = None,
    proposal_model: str = "",
    canonicalization_model: str = "",
) -> dict[str, object]:
    require_execution_mode(execution_mode)
    ensure_directory(grounding_dir)
    raw_rule_records: list[dict[str, object]] = []
    if execution_mode == "live":
        if proposal_client is None or canonicalization_client is None:
            raise ValueError("Live grounding requires both proposal_client and canonicalization_client")
        if not proposal_model or not canonicalization_model:
            raise ValueError("Live grounding requires proposal_model and canonicalization_model")
        canonicalized_rows: list[dict[str, object]] = []
        for source_rule_type, rules in (
            ("allowed", world.allowed_behaviors),
            ("prohibited", world.prohibited_behaviors),
        ):
            for rule in rules:
                raw_clauses, normalized_rows = _live_extract_rule_clauses(
                    world=world,
                    rule=rule,
                    source_rule_type=source_rule_type,
                    proposal_client=proposal_client,
                    canonicalization_client=canonicalization_client,
                    proposal_model=proposal_model,
                    canonicalization_model=canonicalization_model,
                )
                raw_rule_records.append(
                    {
                        "source_rule_id": rule.rule_id,
                        "source_rule_type": source_rule_type,
                        "rule_text": rule.rule_text,
                        "raw_clauses": raw_clauses,
                    }
                )
                canonicalized_rows.extend(normalized_rows)
    else:
        canonicalized_rows = propose_grounded_clauses(world)
        raw_rule_records = [
            {
                "source_rule_id": row["source_rule_id"],
                "source_rule_type": row["source_rule_type"],
                "rule_text": row["clause_text"],
                "raw_clauses": [
                    {
                        "clause_text": row["clause_text"],
                        "trigger": row["trigger"],
                        "scope": row["scope"],
                        "effect": row["effect"],
                        "source_span": row["source_span"],
                    }
                ],
            }
            for row in canonicalized_rows
        ]
    kept_rows, removed_rows = dedupe_exact_clauses(canonicalized_rows)

    raw_payload = {
        "company_key": world.company_key,
        "source_company_name": world.company_name,
        "proposed_clause_count": len(canonicalized_rows),
        "execution_mode": execution_mode,
        "rule_records": raw_rule_records,
        "extracted_clauses": canonicalized_rows,
    }
    exact_dedup_report = {
        "input_count": len(canonicalized_rows),
        "kept_count": len(kept_rows),
        "removed_count": len(removed_rows),
        "removed_clause_ids": [row["clause_id"] for row in removed_rows],
    }
    effect_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for row in kept_rows:
        effect = str(row["effect"])
        effect_counts[effect] = effect_counts.get(effect, 0) + 1
        source_rule_type = str(row["source_rule_type"])
        source_counts[source_rule_type] = source_counts.get(source_rule_type, 0) + 1
    summary = {
        "company_key": world.company_key,
        "raw_clause_count": len(canonicalized_rows),
        "grounded_clause_count": len(kept_rows),
        "exact_duplicate_count": len(removed_rows),
        "semantic_duplicate_candidate_count": 0,
        "execution_mode": execution_mode,
        "effect_counts": effect_counts,
        "source_rule_type_counts": source_counts,
    }

    write_json(grounding_dir / "raw_clause_extraction.json", raw_payload)
    write_jsonl(grounding_dir / "canonicalized_clauses.jsonl", canonicalized_rows)
    write_json(grounding_dir / "exact_dedup_report.json", exact_dedup_report)
    write_jsonl(grounding_dir / "semantic_dedup_candidates.jsonl", [])
    write_jsonl(grounding_dir / "semantic_dedup_decisions.jsonl", [])
    write_jsonl(grounding_dir / "grounded_clauses_final.jsonl", kept_rows)
    write_json(grounding_dir / "grounding_summary.json", summary)
    return summary
