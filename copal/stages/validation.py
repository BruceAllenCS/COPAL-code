from __future__ import annotations

from copal.taxonomy import relation_patterns_for_effects


def validate_structure_constraints(row: dict[str, object]) -> dict[str, object]:
    signature = str(row.get("signature_proposal", "")).strip()
    signals = dict(row.get("structure_signals", {}))
    interaction_filter = dict(signals.get("interaction_filter", row.get("interaction_filter", {})))
    interaction_status = str(interaction_filter.get("status", "")).strip()
    trigger_compatible = bool(signals.get("trigger_compatible", signals.get("joint_trigger_satisfiable", False)))
    scope_coupled = bool(signals.get("scope_coupled", signals.get("scope_overlap", False)))
    effect_interaction = bool(signals.get("effect_interaction", _effect_pair_interacts(row, signals)))

    if signature and trigger_compatible and scope_coupled and effect_interaction and interaction_status == "pass":
        return {
            "target_type": "composition",
            "target_id": str(row.get("composition_id", "")),
            "schema_consistent": True,
            "exact_dedup_pass": True,
            "structure_constraint_pass": True,
            "heuristic_feasibility_result": "pass",
            "heuristic_non_separability_result": "pass",
            "interaction_filter_status": "pass",
            "requires_adjudication": False,
            "validation_notes": "Passed deterministic structure checks.",
        }
    if interaction_status == "fail" and not (scope_coupled or effect_interaction):
        return {
            "target_type": "composition",
            "target_id": str(row.get("composition_id", "")),
            "schema_consistent": True,
            "exact_dedup_pass": True,
            "structure_constraint_pass": False,
            "heuristic_feasibility_result": "fail" if not trigger_compatible else "pass",
            "heuristic_non_separability_result": "fail",
            "interaction_filter_status": "fail",
            "requires_adjudication": False,
            "validation_notes": "Rejected as independently resolvable co-occurrence.",
        }
    return {
        "target_type": "composition",
        "target_id": str(row.get("composition_id", "")),
        "schema_consistent": True,
        "exact_dedup_pass": True,
        "structure_constraint_pass": False,
        "heuristic_feasibility_result": "unresolved",
        "heuristic_non_separability_result": "unresolved",
        "interaction_filter_status": interaction_status or "unresolved",
        "requires_adjudication": True,
        "validation_notes": "Requires semantic adjudication after deterministic screening.",
    }


def _effect_pair_interacts(row: dict[str, object], signals: dict[str, object]) -> bool:
    raw_effects = signals.get("effect_pair", row.get("effect_pair", []))
    if not isinstance(raw_effects, list | tuple | set):
        return bool(signals.get("changes_scope_or_handling", False))
    return bool(relation_patterns_for_effects(raw_effects))
