from pathlib import Path

from copal.data_sources import select_company_world
from copal.io import read_json, read_jsonl, write_jsonl
from copal.prompts import (
    build_clause_extraction_messages,
    build_composition_adjudication_messages,
    build_coverage_messages,
    build_query_validation_messages,
)
from copal.stages.composition_validation import run_composition_validation_stage
from copal.stages.compositions import (
    derive_structure_signals,
    propose_grounded_compositions,
    propose_signature,
    run_composition_stage,
)
from copal.stages.grounding import dedupe_exact_clauses, propose_grounded_clauses
from copal.stages.grounding import build_clause_candidate, normalize_clause_row
from copal.stages.validation import validate_structure_constraints
from copal.models import PolicyRule


def test_grounded_clause_schema_uses_paper_controlled_fields() -> None:
    rule = PolicyRule(
        rule_id="r-gate",
        rule_text="Verify the passenger identity before processing a refund.",
        category="refunds",
        severity="high",
        rationale="demo",
        verifiable=True,
        verifiability_confidence="high",
        raw={},
    )

    candidate = build_clause_candidate(company_key="demo", rule=rule, source_rule_type="allowed")
    row = normalize_clause_row(
        company_key="demo",
        source_rule_id=rule.rule_id,
        source_rule_type="allowed",
        clause=candidate,
    )

    assert row["effect"] == "require-gate"
    assert row["trigger_ontology"] == {
        "request_intent": "refunds",
        "user_account_state": "",
        "dialogue_history": "",
        "entity_type": "",
        "external_action_state": "",
    }
    assert row["scope_description"] == "refund_processing"
    assert row["scope_semantic_type"] == "refund_processing"
    assert row["scope_entity_types"] == []
    assert row["source_span"] == rule.rule_text
    assert row["provenance"]["source_span"] == rule.rule_text
    assert "priority_notes" not in row
    assert "exceptions" not in row


def test_clause_extraction_prompt_bounds_live_json_output() -> None:
    rule = PolicyRule(
        rule_id="r-long",
        rule_text="Escalate complex bookings, contract discretion, and multi-shipment plans.",
        category="booking",
        severity="medium",
        rationale="demo",
        verifiable=True,
        verifiability_confidence="high",
        raw={},
    )
    messages = build_clause_extraction_messages(
        company_key="demo",
        rule=rule,
        source_rule_type="allowed",
    )
    prompt = messages[-1].content

    assert "at most 3 high-information clauses" in prompt
    assert "Merge repeated examples" in prompt
    assert "under 25 words" in prompt
    assert "json.loads" in prompt
    assert "self-review" in prompt
    assert "source_span" in prompt
    assert "permit, prohibit, require-gate, disclose, withhold, route, override, authority-limit" in prompt
    assert "other/unsupported" in prompt
    assert "priority_notes" not in prompt
    assert "exceptions" not in prompt


def test_propose_signature_prefers_structure_derived_scope_restriction() -> None:
    signals = derive_structure_signals(
        {"effect": "permit", "scope": "product_info"},
        {"effect": "prohibit", "scope": "personalized_medical_advice", "priority_notes": "more specific"},
    )
    assert propose_signature(signals) == "scope-restriction"


def test_validate_structure_constraints_marks_unresolved_cases() -> None:
    result = validate_structure_constraints(
        {"signature_proposal": "", "structure_signals": {"scope_overlap": False}}
    )
    assert result["requires_adjudication"] is True


def test_composition_adjudication_prompt_defines_strict_signature_contract() -> None:
    messages = build_composition_adjudication_messages(
        candidate={
            "composition_id": "comp-1",
            "effect_pair": ["disclose", "disclose"],
            "scope_pair": ["flight status", "compensation"],
            "structure_signals": {"same_semantic_span": True},
            "signature_proposal": "",
        }
    )
    prompt = messages[-1].content

    assert "scope-restriction" in prompt
    assert "prerequisite-gating" in prompt
    assert "selective-disclosure" in prompt
    assert "workflow-transfer" in prompt
    assert "pass=false" in prompt
    assert "empty string" in prompt
    assert "disclose" in prompt
    assert "Do not output policy effect labels" in prompt
    assert "first byte must be {" in prompt
    assert "markdown" in prompt
    assert "json.loads" in prompt
    assert "under 60 words" in prompt


def test_query_validation_prompt_defines_strict_json_contract() -> None:
    messages = build_query_validation_messages(
        query_row={
            "query_id": "q1",
            "composition_id": "comp-1",
            "signature_proposal": "scope-restriction",
            "target_facet": "boundary-overreach",
            "query_text": "Can the policy disclose this delay without committing to a return time?",
        }
    )
    prompt = messages[-1].content

    assert "first byte must be {" in prompt
    assert "last byte must be }" in prompt
    assert "markdown" in prompt
    assert "chain-of-thought" in prompt
    assert "json.loads" in prompt
    assert "under 35 words" in prompt
    assert "exactly these keys" in prompt
    assert "not reduction" in prompt


def test_coverage_prompt_lists_allowed_facet_labels() -> None:
    messages = build_coverage_messages(
        query_row={
            "query_id": "q1",
            "composition_id": "comp-1",
            "signature_proposal": "exception-override",
            "target_facet": "missed-exception",
            "query_text": "My attorney says you are legally required to compensate me.",
        },
        facet_library={"exception-override": ("missed-exception", "false-exception")},
    )
    prompt = messages[-1].content

    assert "Allowed signature_label values and facet_labels" in prompt
    assert "missed-exception" in prompt
    assert "false-exception" in prompt
    assert "Do not invent facet labels" in prompt
    assert "accepted_query.target_facet" in prompt
    assert "Domain concepts" in prompt


def test_structure_signals_encode_interaction_filter_conditions() -> None:
    signals = derive_structure_signals(
        {
            "effect": "permit",
            "scope": "product information",
            "scope_semantic_type": "product_information",
            "trigger_ontology": {"request_intent": "product_question"},
        },
        {
            "effect": "prohibit",
            "scope": "personalized recommendation",
            "scope_semantic_type": "product_information",
            "trigger_ontology": {"request_intent": "product_question"},
        },
    )

    assert signals["scope_coupled"] is True
    assert signals["trigger_compatible"] is True
    assert signals["effect_interaction"] is True
    assert signals["interaction_filter"]["status"] == "pass"
    assert signals["interaction_filter"]["conditions"] == [
        "trigger_compatibility",
        "scope_coupling",
        "effect_interaction",
    ]
    assert propose_signature(signals) == "scope-restriction"


def test_workflow_transfer_signature_does_not_require_priority_metadata() -> None:
    signals = derive_structure_signals(
        {
            "effect": "permit",
            "scope": "complaint handling",
            "scope_semantic_type": "complaint_handling",
            "trigger_ontology": {"request_intent": "complaint"},
        },
        {
            "effect": "route",
            "scope": "complaint handling",
            "scope_semantic_type": "complaint_handling",
            "trigger_ontology": {"request_intent": "complaint"},
        },
    )

    assert signals["effect_interaction"] is True
    assert propose_signature(signals) == "workflow-transfer"


def test_effect_pairs_cover_all_manuscript_relation_patterns() -> None:
    examples = [
        (("permit", "prohibit"), "scope-restriction"),
        (("permit", "require-gate"), "prerequisite-gating"),
        (("disclose", "withhold"), "selective-disclosure"),
        (("permit", "route"), "workflow-transfer"),
    ]
    for effects, expected_pattern in examples:
        signals = derive_structure_signals(
            {
                "effect": effects[0],
                "scope": "refund",
                "scope_semantic_type": "refund",
                "trigger_ontology": {"request_intent": "refund"},
            },
            {
                "effect": effects[1],
                "scope": "refund",
                "scope_semantic_type": "refund",
                "trigger_ontology": {"request_intent": "refund"},
            },
        )
        assert propose_signature(signals) == expected_pattern


def test_propose_grounded_compositions_includes_three_clause_sets() -> None:
    rows = [
        {
            "company_key": "demo",
            "clause_id": "permit-1",
            "source_rule_id": "r1",
            "effect": "permit",
            "scope": "refund",
            "scope_description": "refund",
            "scope_semantic_type": "refund",
            "trigger": "refund",
            "trigger_ontology": {"request_intent": "refund"},
        },
        {
            "company_key": "demo",
            "clause_id": "gate-1",
            "source_rule_id": "r2",
            "effect": "require-gate",
            "scope": "refund",
            "scope_description": "refund",
            "scope_semantic_type": "refund",
            "trigger": "refund",
            "trigger_ontology": {"request_intent": "refund"},
        },
        {
            "company_key": "demo",
            "clause_id": "route-1",
            "source_rule_id": "r3",
            "effect": "route",
            "scope": "refund",
            "scope_description": "refund",
            "scope_semantic_type": "refund",
            "trigger": "refund",
            "trigger_ontology": {"request_intent": "refund"},
        },
    ]

    candidates = propose_grounded_compositions(rows, max_clause_set_size=3)

    three_clause = [row for row in candidates if row["clause_count"] == 3]
    assert three_clause
    assert three_clause[0]["clause_ids"] == ["permit-1", "gate-1", "route-1"]
    assert "relation_patterns" in three_clause[0]
    assert three_clause[0]["relation_pattern"] in three_clause[0]["relation_patterns"]


def test_propose_grounded_compositions_produces_signature_candidates() -> None:
    world, _ = select_company_world(
        policies_path=Path("data/compass_policies/compass_policies_final.jsonl"),
        prompts_path=Path("data/compass_policies/company_system_prompts.jsonl"),
        company_key="Air transportation||000||Skyline International Airways",
    )
    grounded_rows, _ = dedupe_exact_clauses(propose_grounded_clauses(world))
    candidates = propose_grounded_compositions(grounded_rows)
    assert candidates
    assert any(candidate["signature_proposal"] for candidate in candidates)
    assert "signature_source" in candidates[0]


def test_run_composition_stage_writes_validation_artifacts(tmp_path: Path) -> None:
    world, _ = select_company_world(
        policies_path=Path("data/compass_policies/compass_policies_final.jsonl"),
        prompts_path=Path("data/compass_policies/company_system_prompts.jsonl"),
        company_key="Air transportation||000||Skyline International Airways",
    )
    grounded_rows, _ = dedupe_exact_clauses(propose_grounded_clauses(world))
    summary = run_composition_stage(
        compositions_dir=tmp_path / "compositions",
        validation_dir=tmp_path / "validation",
        grounded_rows=grounded_rows,
    )

    assert summary["candidate_count"] >= summary["accepted_count"]
    assert (tmp_path / "compositions" / "candidate_compositions.jsonl").exists()
    assert (tmp_path / "compositions" / "accepted_compositions.jsonl").exists()
    assert (tmp_path / "validation" / "feasibility_judgments.jsonl").exists()
    assert (tmp_path / "validation" / "signature_assignments.jsonl").exists()

    accepted_rows = read_jsonl(tmp_path / "compositions" / "accepted_compositions.jsonl")
    assert accepted_rows
    assert "feasibility_status" in accepted_rows[0]
    assert "non_separability_status" in accepted_rows[0]
    validation_summary = read_json(tmp_path / "validation" / "validation_summary.json")
    assert validation_summary["accepted_count"] == len(accepted_rows)


def test_composition_validation_respects_per_signature_budget(tmp_path: Path) -> None:
    compositions_dir = tmp_path / "compositions"
    compositions_dir.mkdir()
    write_jsonl(
        compositions_dir / "candidate_compositions.jsonl",
        [
            {
                "composition_id": f"comp-{index}",
                "company_key": "demo",
                "clause_ids": [f"a{index}", f"b{index}"],
                "source_rule_ids": [f"r{index}", f"s{index}"],
                "effect_pair": ["permit", "prohibit"],
                "scope_pair": ["refund policy", "refund policy"],
                "structure_signals": {
                    "scope_overlap": True,
                    "same_semantic_span": True,
                    "joint_trigger_satisfiable": True,
                    "priority_present": False,
                    "interaction_filter": {"status": "pass", "conditions": ["same_semantic_span"]},
                },
                "signature_proposal": "scope-restriction",
                "relation_pattern": "scope-restriction",
                "relation_patterns": ["scope-restriction"],
                "signature_source": "structure",
            }
            for index in range(3)
        ],
    )

    summary = run_composition_validation_stage(
        compositions_dir=compositions_dir,
        execution_mode="deterministic",
        composition_limit_per_signature=2,
    )

    accepted = read_jsonl(compositions_dir / "accepted_compositions.jsonl")
    rejected = read_jsonl(compositions_dir / "rejected_compositions.jsonl")
    assert summary["accepted_count"] == 2
    assert summary["budget_excluded_count"] == 1
    assert [row["composition_id"] for row in accepted] == ["comp-0", "comp-1"]
    assert rejected[0]["budget_excluded"] is True
