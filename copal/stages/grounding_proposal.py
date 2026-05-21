from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from copal.config import require_execution_mode
from copal.io import ensure_directory, write_json, write_jsonl
from copal.llm import LLMClient
from copal.live_validation import (
    LiveSchemaError,
    complete_live_json_object,
    require_fields,
    require_object,
    require_object_list,
    require_str,
    require_str_allow_empty,
    require_str_list,
)
from copal.models import CompanyWorld, PolicyRule
from copal.prompts import build_clause_canonicalization_messages, build_clause_extraction_messages
from copal.stages.grounding import propose_grounded_clauses
from copal.taxonomy import CANONICAL_EFFECTS, normalize_effect_label

ALLOWED_EFFECTS: tuple[str, ...] = CANONICAL_EFFECTS


def _iter_policy_rules(world: CompanyWorld) -> list[tuple[str, PolicyRule]]:
    return [("allowed", rule) for rule in world.allowed_behaviors] + [
        ("prohibited", rule) for rule in world.prohibited_behaviors
    ]


def _validate_clause_payload(*, clause: dict[str, object], context: str) -> None:
    require_fields(clause, ("clause_text", "trigger", "scope", "effect", "source_span"), context=context)
    require_str(clause["clause_text"], context=f"{context}.clause_text")
    trigger = require_object(clause["trigger"], context=f"{context}.trigger")
    require_fields(
        trigger,
        ("source_text", "request_intent", "user_account_state", "dialogue_history", "entity_type", "external_action_state"),
        context=f"{context}.trigger",
    )
    require_str(trigger["source_text"], context=f"{context}.trigger.source_text")
    for field in ("request_intent", "user_account_state", "dialogue_history", "entity_type", "external_action_state"):
        require_str_allow_empty(trigger[field], context=f"{context}.trigger.{field}")
    scope = require_object(clause["scope"], context=f"{context}.scope")
    require_fields(scope, ("description", "semantic_type", "entity_types"), context=f"{context}.scope")
    require_str(scope["description"], context=f"{context}.scope.description")
    require_str(scope["semantic_type"], context=f"{context}.scope.semantic_type")
    require_str_list(scope["entity_types"], context=f"{context}.scope.entity_types")
    effect = require_str(clause["effect"], context=f"{context}.effect")
    try:
        normalize_effect_label(effect)
    except ValueError as exc:
        raise LiveSchemaError(f"{context}.effect has unsupported label: {effect}") from exc
    require_str(clause["source_span"], context=f"{context}.source_span")


def _validate_extraction_payload(*, payload: dict[str, object], context: str) -> None:
    clauses = require_object_list(payload["clauses"], context=f"{context}.clauses")
    for index, clause in enumerate(clauses):
        _validate_clause_payload(clause=clause, context=f"{context}.clauses[{index}]")


def _validate_canonicalization_payload(*, payload: dict[str, object], context: str) -> None:
    clause = require_object(payload["clause"], context=f"{context}.clause")
    _validate_clause_payload(clause=clause, context=f"{context}.clause")


def _live_grounding_candidates(
    *,
    grounding_dir: Path,
    world: CompanyWorld,
    proposal_client: LLMClient,
    canonicalization_client: LLMClient,
    proposal_model: str,
    canonicalization_model: str,
    live_max_workers: int = 1,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if live_max_workers < 1:
        raise ValueError("live_max_workers must be positive")
    policy_rules = _iter_policy_rules(world)
    if live_max_workers == 1:
        rows = [
            _live_grounding_candidates_for_rule(
                grounding_dir=grounding_dir,
                world=world,
                source_rule_type=source_rule_type,
                rule=rule,
                proposal_client=proposal_client,
                canonicalization_client=canonicalization_client,
                proposal_model=proposal_model,
                canonicalization_model=canonicalization_model,
            )
            for source_rule_type, rule in policy_rules
        ]
    else:
        with ThreadPoolExecutor(max_workers=live_max_workers) as executor:
            futures = [
                executor.submit(
                    _live_grounding_candidates_for_rule,
                    grounding_dir=grounding_dir,
                    world=world,
                    source_rule_type=source_rule_type,
                    rule=rule,
                    proposal_client=proposal_client,
                    canonicalization_client=canonicalization_client,
                    proposal_model=proposal_model,
                    canonicalization_model=canonicalization_model,
                )
                for source_rule_type, rule in policy_rules
            ]
            rows = [future.result() for future in futures]

    raw_rows: list[dict[str, object]] = []
    canonicalized_rows: list[dict[str, object]] = []
    for rule_raw_rows, rule_canonicalized_rows in rows:
        raw_rows.extend(rule_raw_rows)
        canonicalized_rows.extend(rule_canonicalized_rows)
    return raw_rows, canonicalized_rows


def _live_grounding_candidates_for_rule(
    *,
    grounding_dir: Path,
    world: CompanyWorld,
    source_rule_type: str,
    rule: PolicyRule,
    proposal_client: LLMClient,
    canonicalization_client: LLMClient,
    proposal_model: str,
    canonicalization_model: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    raw_rows: list[dict[str, object]] = []
    canonicalized_rows: list[dict[str, object]] = []
    extraction_context = f"grounding_proposal extraction {rule.rule_id}"
    extraction_payload = complete_live_json_object(
        client=proposal_client,
        model=proposal_model,
        messages=build_clause_extraction_messages(
            company_key=world.company_key,
            rule=rule,
            source_rule_type=source_rule_type,
        ),
        stage_dir=grounding_dir,
        stage_name="grounding_proposal",
        target_id=f"{rule.rule_id}::extraction",
        required_fields=("clauses",),
        validator=lambda payload: _validate_extraction_payload(payload=payload, context=extraction_context),
    )
    extracted_clauses = require_object_list(extraction_payload["clauses"], context=f"{extraction_context}.clauses")
    for clause in extracted_clauses:
        raw_rows.append(
            {
                "company_key": world.company_key,
                "source_rule_id": rule.rule_id,
                "source_rule_type": source_rule_type,
                "proposal_stage": "grounding_proposal",
                "candidate_clause": dict(clause),
            }
        )
        canonicalization_context = f"grounding_proposal canonicalization {rule.rule_id}"
        canonicalization_payload = complete_live_json_object(
            client=canonicalization_client,
            model=canonicalization_model,
            messages=build_clause_canonicalization_messages(
                company_key=world.company_key,
                source_rule_id=rule.rule_id,
                source_rule_type=source_rule_type,
                clause=dict(clause),
            ),
            stage_dir=grounding_dir,
            stage_name="grounding_proposal",
            target_id=f"{rule.rule_id}::canonicalization",
            required_fields=("clause",),
            validator=lambda payload: _validate_canonicalization_payload(
                payload=payload,
                context=canonicalization_context,
            ),
        )
        canonical_clause = dict(canonicalization_payload["clause"])
        canonical_clause["grounding_meta"] = {
            "category": rule.category,
            "severity": rule.severity,
            "verifiable": rule.verifiable,
            "verifiability_confidence": rule.verifiability_confidence,
            "llm_extracted": True,
            "proposal_model": proposal_model,
            "canonicalization_model": canonicalization_model,
        }
        canonicalized_rows.append(
            {
                **canonical_clause,
                "company_key": world.company_key,
                "source_rule_id": rule.rule_id,
                "source_rule_type": source_rule_type,
            }
        )
    return raw_rows, canonicalized_rows


def run_grounding_proposal_stage(
    *,
    grounding_dir: Path,
    world: CompanyWorld,
    execution_mode: str,
    proposal_client: LLMClient | None = None,
    canonicalization_client: LLMClient | None = None,
    proposal_model: str = "",
    canonicalization_model: str = "",
    live_max_workers: int = 1,
) -> dict[str, object]:
    require_execution_mode(execution_mode)
    ensure_directory(grounding_dir)
    if execution_mode == "live":
        if proposal_client is None or canonicalization_client is None:
            raise ValueError("Live grounding proposal requires proposal_client and canonicalization_client")
        if not proposal_model or not canonicalization_model:
            raise ValueError("Live grounding proposal requires proposal_model and canonicalization_model")
        raw_rows, canonicalization_candidates = _live_grounding_candidates(
            grounding_dir=grounding_dir,
            world=world,
            proposal_client=proposal_client,
            canonicalization_client=canonicalization_client,
            proposal_model=proposal_model,
            canonicalization_model=canonicalization_model,
            live_max_workers=live_max_workers,
        )
    else:
        proposed_rows = propose_grounded_clauses(world)
        raw_rows = [
            {
                "company_key": row["company_key"],
                "source_rule_id": row["source_rule_id"],
                "source_rule_type": row["source_rule_type"],
                "proposal_stage": "grounding_proposal",
                "candidate_clause": {
                    "clause_text": row["clause_text"],
                    "trigger": row["trigger"],
                    "scope": row["scope"],
                    "effect": row["effect"],
                    "source_span": row["source_span"],
                    "grounding_meta": row["grounding_meta"],
                    "audit_metadata": row["audit_metadata"],
                },
            }
            for row in proposed_rows
        ]
        canonicalization_candidates = [
            row["candidate_clause"]
            | {
                "company_key": row["company_key"],
                "source_rule_id": row["source_rule_id"],
                "source_rule_type": row["source_rule_type"],
            }
            for row in raw_rows
        ]

    summary = {
        "company_key": world.company_key,
        "raw_clause_count": len(raw_rows),
        "canonicalization_candidate_count": len(raw_rows),
        "execution_mode": execution_mode,
    }

    write_jsonl(grounding_dir / "raw_clause_extractions.jsonl", raw_rows)
    write_jsonl(grounding_dir / "canonicalization_candidates.jsonl", canonicalization_candidates)
    write_json(grounding_dir / "grounding_proposal_summary.json", summary)
    return summary
