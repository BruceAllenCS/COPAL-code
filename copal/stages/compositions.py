from __future__ import annotations

from hashlib import sha1
from itertools import combinations
from pathlib import Path

from copal.io import ensure_directory, write_json, write_jsonl
from copal.stages.validation import validate_structure_constraints
from copal.taxonomy import normalize_effect_label, primary_relation_pattern, relation_patterns_for_effects


def _effect(row: dict[str, object]) -> str:
    return normalize_effect_label(row["effect"])


def _scope_description(row: dict[str, object]) -> str:
    return str(row.get("scope_description", row["scope"])).strip()


def _scope_semantic_type(row: dict[str, object]) -> str:
    return str(row.get("scope_semantic_type", row.get("scope", ""))).strip()


def _trigger_ontology(row: dict[str, object]) -> dict[str, str]:
    trigger = row.get("trigger_ontology")
    if isinstance(trigger, dict):
        return {str(key): str(value).strip() for key, value in trigger.items()}
    return {"request_intent": str(row.get("trigger", "")).strip()}


def _trigger_description(row: dict[str, object]) -> str:
    return str(row.get("trigger", "")).strip()


def _token_set(value: str) -> set[str]:
    return {token for token in value.lower().replace("/", " ").replace("-", " ").replace("_", " ").split() if token}


def _scopes_overlap(left: dict[str, object], right: dict[str, object]) -> bool:
    left_semantic = _scope_semantic_type(left)
    right_semantic = _scope_semantic_type(right)
    if left_semantic and right_semantic and left_semantic == right_semantic:
        return True
    left_tokens = _token_set(_scope_description(left))
    right_tokens = _token_set(_scope_description(right))
    return bool(left_tokens & right_tokens)


def _joint_trigger_satisfiable(left: dict[str, object], right: dict[str, object]) -> bool:
    left_trigger = _trigger_ontology(left)
    right_trigger = _trigger_ontology(right)
    for field in ("request_intent", "user_account_state", "dialogue_history", "entity_type", "external_action_state"):
        left_value = left_trigger.get(field, "")
        right_value = right_trigger.get(field, "")
        if not left_value or not right_value:
            continue
        if _trigger_values_contradict(left_value, right_value):
            return False
    return True


def _trigger_values_contradict(left_value: str, right_value: str) -> bool:
    left_tokens = _token_set(left_value)
    right_tokens = _token_set(right_value)
    negation_tokens = {"no", "not", "without", "unauthenticated", "unverified", "absent"}
    if left_tokens == right_tokens:
        return False
    shared_content = (left_tokens - negation_tokens) & (right_tokens - negation_tokens)
    if not shared_content:
        return False
    return bool((left_tokens & negation_tokens) ^ (right_tokens & negation_tokens))


def _pairwise_connected(clauses: list[dict[str, object]], relation: str) -> bool:
    if len(clauses) < 2:
        return False
    connected_indexes = {0}
    changed = True
    while changed:
        changed = False
        for left_index, right_index in combinations(range(len(clauses)), 2):
            if left_index not in connected_indexes and right_index not in connected_indexes:
                continue
            left = clauses[left_index]
            right = clauses[right_index]
            if relation == "scope" and not _scopes_overlap(left, right):
                continue
            if relation == "trigger" and not _joint_trigger_satisfiable(left, right):
                continue
            before = len(connected_indexes)
            connected_indexes.update((left_index, right_index))
            changed = changed or len(connected_indexes) > before
    return len(connected_indexes) == len(clauses)


def _effect_interacts(effects: set[str]) -> bool:
    return bool(relation_patterns_for_effects(effects))


def derive_clause_set_structure_signals(clauses: list[dict[str, object]]) -> dict[str, object]:
    if len(clauses) < 2:
        raise ValueError("composition signals require at least two clauses")
    effects = tuple(_effect(row) for row in clauses)
    effect_set = set(effects)
    trigger_compatible = all(_joint_trigger_satisfiable(left, right) for left, right in combinations(clauses, 2))
    scope_coupled = _pairwise_connected(clauses, "scope")
    effect_interaction = _effect_interacts(effect_set)
    relation_patterns = relation_patterns_for_effects(effect_set)
    conditions = []
    if trigger_compatible:
        conditions.append("trigger_compatibility")
    if scope_coupled:
        conditions.append("scope_coupling")
    if effect_interaction:
        conditions.append("effect_interaction")
    interaction_status = "pass" if trigger_compatible and scope_coupled and effect_interaction else "fail"
    trigger_values = {str(row.get("trigger", "")).strip() for row in clauses}
    return {
        "effect_pair": effects,
        "effect_set": sorted(effect_set),
        "relation_patterns": list(relation_patterns),
        "relation_pattern": relation_patterns[0] if relation_patterns else "",
        "trigger_compatible": trigger_compatible,
        "scope_coupled": scope_coupled,
        "effect_interaction": effect_interaction,
        "scope_overlap": scope_coupled,
        "same_semantic_span": scope_coupled,
        "priority_present": False,
        "trigger_overlap": len(trigger_values) == 1,
        "joint_trigger_satisfiable": trigger_compatible,
        "changes_scope_or_handling": bool(effect_set & {"require-gate", "route", "override", "authority-limit"}),
        "recombine_loses_expected_handling": effect_interaction and scope_coupled,
        "independently_resolvable": interaction_status == "fail",
        "interaction_filter": {
            "level": "clause_set",
            "status": interaction_status,
            "conditions": conditions,
        },
    }


def derive_structure_signals(left: dict[str, object], right: dict[str, object]) -> dict[str, object]:
    return derive_clause_set_structure_signals([left, right])


def propose_signature(signals: dict[str, object]) -> str:
    return primary_relation_pattern(signals["effect_pair"])


def _build_composition_id(clause_ids: list[str]) -> str:
    ordered = "|".join(sorted(clause_ids))
    return sha1(ordered.encode("utf-8")).hexdigest()[:12]


def propose_grounded_compositions(
    grounded_rows: list[dict[str, object]],
    *,
    max_clause_set_size: int = 3,
) -> list[dict[str, object]]:
    if max_clause_set_size < 2:
        raise ValueError("max_clause_set_size must be at least 2")
    candidates: list[dict[str, object]] = []
    upper = min(max_clause_set_size, len(grounded_rows))
    for clause_count in range(2, upper + 1):
        for clause_tuple in combinations(grounded_rows, clause_count):
            clauses = list(clause_tuple)
            structure_signals = derive_clause_set_structure_signals(clauses)
            signature = propose_signature(structure_signals)
            relation_patterns = list(structure_signals["relation_patterns"])
            if not signature and structure_signals["interaction_filter"]["status"] == "fail":
                continue
            clause_ids = [str(row["clause_id"]) for row in clauses]
            candidates.append(
                {
                    "composition_id": _build_composition_id(clause_ids),
                    "company_key": clauses[0]["company_key"],
                    "clause_ids": clause_ids,
                    "clause_count": clause_count,
                    "source_rule_ids": [row["source_rule_id"] for row in clauses],
                    "effect_pair": list(structure_signals["effect_pair"]),
                    "effect_set": list(structure_signals["effect_set"]),
                    "trigger_set": [_trigger_description(row) for row in clauses],
                    "scope_pair": [_scope_description(row) for row in clauses],
                    "scope_set": [_scope_description(row) for row in clauses],
                    "structure_signals": structure_signals,
                    "interaction_filter": structure_signals["interaction_filter"],
                    "signature_proposal": signature,
                    "relation_pattern": signature,
                    "relation_patterns": relation_patterns,
                    "signature_source": "structure_derived" if signature else "unresolved",
                    "feasibility_status": "proposed",
                    "non_separability_status": "proposed",
                }
            )
    return candidates


def run_composition_stage(
    *,
    compositions_dir: Path,
    validation_dir: Path,
    grounded_rows: list[dict[str, object]],
) -> dict[str, object]:
    ensure_directory(compositions_dir)
    ensure_directory(validation_dir)

    candidates = propose_grounded_compositions(grounded_rows)
    feasibility_rows: list[dict[str, object]] = []
    non_separability_rows: list[dict[str, object]] = []
    signature_rows: list[dict[str, object]] = []
    accepted_rows: list[dict[str, object]] = []
    unresolved_rows: list[dict[str, object]] = []

    for candidate in candidates:
        validation = validate_structure_constraints(candidate)
        signature_row = {
            "composition_id": candidate["composition_id"],
            "signature_proposal": candidate["signature_proposal"],
            "relation_pattern": candidate["relation_pattern"],
            "relation_patterns": candidate["relation_patterns"],
            "structure_signals": candidate["structure_signals"],
            "requires_adjudication": validation["requires_adjudication"],
        }
        feasibility_rows.append(
            {
                "composition_id": candidate["composition_id"],
                "heuristic_feasibility_result": validation["heuristic_feasibility_result"],
            }
        )
        non_separability_rows.append(
            {
                "composition_id": candidate["composition_id"],
                "heuristic_non_separability_result": validation["heuristic_non_separability_result"],
            }
        )
        signature_rows.append(signature_row)

        merged = {
            **candidate,
            **validation,
            "feasibility_status": validation["heuristic_feasibility_result"],
            "non_separability_status": validation["heuristic_non_separability_result"],
        }
        if validation["requires_adjudication"]:
            unresolved_rows.append(merged)
            continue
        if (
            candidate["signature_proposal"]
            and validation["heuristic_feasibility_result"] == "pass"
            and validation["heuristic_non_separability_result"] == "pass"
        ):
            accepted_rows.append(merged)
        else:
            unresolved_rows.append(merged)

    signature_counts: dict[str, int] = {}
    relation_pattern_counts: dict[str, int] = {}
    for row in accepted_rows:
        signature = str(row["signature_proposal"])
        signature_counts[signature] = signature_counts.get(signature, 0) + 1
        relation_pattern = str(row.get("relation_pattern", signature))
        relation_pattern_counts[relation_pattern] = relation_pattern_counts.get(relation_pattern, 0) + 1

    summary = {
        "candidate_count": len(candidates),
        "accepted_count": len(accepted_rows),
        "unresolved_count": len(unresolved_rows),
        "signature_counts": signature_counts,
        "relation_pattern_counts": relation_pattern_counts,
    }

    write_jsonl(compositions_dir / "candidate_compositions.jsonl", candidates)
    write_jsonl(compositions_dir / "accepted_compositions.jsonl", accepted_rows)
    write_json(compositions_dir / "composition_summary.json", summary)
    write_jsonl(validation_dir / "feasibility_judgments.jsonl", feasibility_rows)
    write_jsonl(validation_dir / "non_separability_judgments.jsonl", non_separability_rows)
    write_jsonl(validation_dir / "signature_assignments.jsonl", signature_rows)
    write_jsonl(validation_dir / "unresolved_composition_candidates.jsonl", unresolved_rows)
    write_json(validation_dir / "validation_summary.json", summary)

    return summary
