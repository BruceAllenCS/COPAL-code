from pathlib import Path
from threading import Lock
from time import monotonic, sleep

from copal.io import read_jsonl, write_jsonl
from copal.live_validation import LiveSchemaError
from copal.llm import LLMMessage, LLMResponse
from copal.models import CompanyWorld, PolicyRule
from copal.config import RoleConfig, RunConfig
from copal.cli import build_live_stage_kwargs, record_live_usage_summary
from copal.stages.composition_validation import run_composition_validation_stage
from copal.stages.coverage_judge import run_coverage_judge_stage
from copal.stages.downstream_chatbot import run_downstream_chatbot_stage
from copal.stages.grounding_proposal import run_grounding_proposal_stage
from copal.stages.grounding_resolution import run_grounding_resolution_stage
from copal.stages.query_proposal import run_query_proposal_stage
from copal.stages.query_validation import run_query_validation_stage
from copal.stages.response_judgment import run_response_judgment_stage


class QueueLLMClient:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def complete(self, *, model: str, messages: list[LLMMessage]) -> LLMResponse:
        self.calls.append({"model": model, "messages": messages})
        return LLMResponse(text=self._responses.pop(0), model=model)


def test_build_live_stage_kwargs_routes_separate_model_roles() -> None:
    client = QueueLLMClient([])
    config = RunConfig(
        run_id="run-1",
        company_key="demo",
        execution_mode="live",
        role_config=RoleConfig(
            proposal_model="proposal-x",
            query_proposal_model="query-proposal-x",
            canonicalization_model="canon-x",
            validator_model="validator-x",
            coverage_judge_model="coverage-x",
            downstream_chatbot_model="chat-x",
            response_judge_model="judge-x",
        ),
    )

    kwargs = build_live_stage_kwargs(config=config, live_client=client)

    assert kwargs["grounding_proposal"]["proposal_model"] == "proposal-x"
    assert kwargs["grounding_proposal"]["canonicalization_model"] == "canon-x"
    assert kwargs["query_proposal"]["proposal_model"] == "query-proposal-x"
    assert kwargs["composition_validation"]["validator_model"] == "validator-x"
    assert kwargs["composition_validation"]["live_max_workers"] == 1
    assert kwargs["query_validation"]["validator_client"] is client
    assert kwargs["coverage_judge"]["coverage_model"] == "coverage-x"
    assert kwargs["downstream_chatbot"]["downstream_models"] == ("chat-x",)
    assert kwargs["response_judgment"]["response_judge_model"] == "judge-x"


def test_build_live_stage_kwargs_uses_freeform_client_only_for_downstream() -> None:
    structured_client = QueueLLMClient([])
    freeform_client = QueueLLMClient([])
    config = RunConfig(
        run_id="run-1",
        company_key="demo",
        execution_mode="live",
        role_config=RoleConfig(
            proposal_model="proposal-x",
            canonicalization_model="canon-x",
            validator_model="validator-x",
            coverage_judge_model="coverage-x",
            downstream_chatbot_model="chat-x",
            response_judge_model="judge-x",
        ),
    )

    kwargs = build_live_stage_kwargs(
        config=config,
        live_client=structured_client,
        downstream_live_client=freeform_client,
    )

    assert kwargs["downstream_chatbot"]["downstream_client"] is freeform_client
    assert kwargs["response_judgment"]["response_judge_client"] is structured_client
    assert kwargs["query_proposal"]["proposal_client"] is structured_client


def test_record_live_usage_summary_requires_usage_summary_method() -> None:
    class UsageClient:
        def usage_summary(self) -> dict[str, object]:
            return {
                "provider": "friday",
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
                "cache_hits": 2,
                "cache_misses": 3,
            }

    summary: dict[str, object] = {}
    record_live_usage_summary(summary=summary, live_client=UsageClient())
    assert summary["llm_usage"] == {
        "provider": "friday",
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
        "cache_hits": 2,
        "cache_misses": 3,
    }

    try:
        record_live_usage_summary(summary={}, live_client=object())
    except TypeError as exc:
        assert "usage_summary" in str(exc)
    else:
        raise AssertionError("TypeError was not raised")


def test_record_live_usage_summary_preserves_existing_usage_when_checkpoint_only() -> None:
    class EmptyUsageClient:
        def usage_summary(self) -> dict[str, object]:
            return {
                "provider": "friday",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cache_hits": 0,
                "cache_misses": 0,
            }

    summary: dict[str, object] = {
        "llm_usage": {
            "provider": "friday",
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
            "cache_hits": 2,
            "cache_misses": 3,
        }
    }

    record_live_usage_summary(summary=summary, live_client=EmptyUsageClient())

    assert summary["llm_usage"] == {
        "provider": "friday",
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
        "cache_hits": 2,
        "cache_misses": 3,
    }


def test_grounding_proposal_live_writes_extraction_and_canonicalization_artifacts(tmp_path: Path) -> None:
    world = CompanyWorld(
        company_key="demo||000||Demo Co",
        industry="demo",
        company_name="Demo Co",
        company_index=0,
        enterprise_config={"company_name": "Demo Co"},
        allowed_behaviors=[
            PolicyRule(
                rule_id="A1",
                rule_text="The chatbot SHALL provide refund status on request.",
                category="refunds",
                severity="high",
                rationale="demo",
                verifiable=True,
                verifiability_confidence="high",
                raw={},
            )
        ],
        prohibited_behaviors=[],
        quality_scores={},
        raw={},
    )
    proposal_client = QueueLLMClient(
        [
            """{"clauses": [{"clause_text": "Provide refund status on request.", "trigger": {"source_text": "refund request", "request_intent": "refund_status", "user_account_state": "", "dialogue_history": "", "entity_type": "", "external_action_state": ""}, "scope": {"description": "refund processing", "semantic_type": "refund_processing", "entity_types": []}, "effect": "permit", "source_span": "The chatbot SHALL provide refund status on request."}]}""",
        ]
    )
    canonicalization_client = QueueLLMClient(
        [
            """{"clause": {"clause_text": "Provide refund status on request.", "trigger": {"source_text": "refund request", "request_intent": "refund_status", "user_account_state": "", "dialogue_history": "", "entity_type": "", "external_action_state": ""}, "scope": {"description": "refund processing", "semantic_type": "refund_processing", "entity_types": []}, "effect": "permit", "source_span": "The chatbot SHALL provide refund status on request."}}""",
        ]
    )

    run_grounding_proposal_stage(
        grounding_dir=tmp_path / "grounding",
        world=world,
        execution_mode="live",
        proposal_client=proposal_client,
        canonicalization_client=canonicalization_client,
        proposal_model="proposal-x",
        canonicalization_model="canon-x",
    )
    resolution_summary = run_grounding_resolution_stage(grounding_dir=tmp_path / "grounding")

    candidates = read_jsonl(tmp_path / "grounding" / "canonicalization_candidates.jsonl")
    grounded = read_jsonl(tmp_path / "grounding" / "grounded_clause_library.jsonl")
    assert candidates[0]["trigger"]["request_intent"] == "refund_status"
    assert grounded[0]["grounding_meta"]["llm_extracted"] is True
    assert resolution_summary["grounded_clause_count"] == 1
    assert len(proposal_client.calls) == 1
    assert len(canonicalization_client.calls) == 1


def test_grounding_proposal_live_can_run_policy_rules_with_bounded_concurrency(tmp_path: Path) -> None:
    world = CompanyWorld(
        company_key="demo||000||Demo Co",
        industry="demo",
        company_name="Demo Co",
        company_index=0,
        enterprise_config={"company_name": "Demo Co"},
        allowed_behaviors=[
            PolicyRule(
                rule_id=f"A{index}",
                rule_text=f"The chatbot SHALL provide status {index} on request.",
                category="status",
                severity="high",
                rationale="demo",
                verifiable=True,
                verifiability_confidence="high",
                raw={},
            )
            for index in range(3)
        ],
        prohibited_behaviors=[],
        quality_scores={},
        raw={},
    )

    class DelayedGroundingClient:
        def __init__(self) -> None:
            self.calls: list[float] = []
            self._lock = Lock()

        def complete(self, *, model: str, messages: list[LLMMessage]) -> LLMResponse:
            with self._lock:
                self.calls.append(monotonic())
            sleep(0.05)
            prompt = messages[-1].content
            if "Extract one or more operational clauses" in prompt:
                payload = {
                    "clauses": [
                        {
                            "clause_text": "Provide status on request.",
                            "trigger": {
                                "source_text": "status request",
                                "request_intent": "status",
                                "user_account_state": "",
                                "dialogue_history": "",
                                "entity_type": "",
                                "external_action_state": "",
                            },
                            "scope": {
                                "description": "status",
                                "semantic_type": "status",
                                "entity_types": [],
                            },
                            "effect": "permit",
                            "source_span": "The chatbot SHALL provide status on request.",
                        }
                    ]
                }
            else:
                payload = {
                    "clause": {
                        "clause_text": "Provide status on request.",
                        "trigger": {
                            "source_text": "status request",
                            "request_intent": "status",
                            "user_account_state": "",
                            "dialogue_history": "",
                            "entity_type": "",
                            "external_action_state": "",
                        },
                        "scope": {
                            "description": "status",
                            "semantic_type": "status",
                            "entity_types": [],
                        },
                        "effect": "permit",
                        "source_span": "The chatbot SHALL provide status on request.",
                    }
                }
            import json

            return LLMResponse(text=json.dumps(payload), model=model)

    client = DelayedGroundingClient()
    started = monotonic()
    summary = run_grounding_proposal_stage(
        grounding_dir=tmp_path / "grounding",
        world=world,
        execution_mode="live",
        proposal_client=client,
        canonicalization_client=client,
        proposal_model="proposal-x",
        canonicalization_model="canon-x",
        live_max_workers=3,
    )
    elapsed = monotonic() - started

    assert summary["raw_clause_count"] == 3
    assert len(client.calls) == 6
    assert elapsed < 0.2


def test_composition_validation_live_adjudicates_unresolved_candidates(tmp_path: Path) -> None:
    compositions_dir = tmp_path / "compositions"
    compositions_dir.mkdir()
    write_jsonl(
        compositions_dir / "candidate_compositions.jsonl",
        [
            {
                "composition_id": "comp-1",
                "company_key": "demo",
                "clause_ids": ["a", "b"],
                "source_rule_ids": ["r1", "r2"],
                "effect_pair": ["permit", "route"],
                "scope_pair": ["refunds", "refunds"],
                "structure_signals": {"scope_overlap": False, "priority_present": False},
                "signature_proposal": "",
                "signature_source": "unresolved",
            }
        ],
    )
    validator = QueueLLMClient(
        [
            """{"pass": true, "signature": "workflow-transfer", "feasibility_status": "pass", "non_separability_status": "pass", "nonseparability_slice": "borderline", "adjudication_rationale": "Routing modifies the handling path."}""",
        ]
    )

    summary = run_composition_validation_stage(
        compositions_dir=compositions_dir,
        execution_mode="live",
        validator_client=validator,
        validator_model="validator-x",
    )

    accepted = read_jsonl(compositions_dir / "accepted_compositions.jsonl")
    adjudications = read_jsonl(compositions_dir / "composition_adjudications.jsonl")
    assert summary["accepted_count"] == 1
    assert accepted[0]["signature_proposal"] == "workflow-transfer"
    assert accepted[0]["signature_source"] == "rubric_adjudicated"
    assert accepted[0]["nonseparability_slice"] == "borderline"
    assert adjudications[0]["validator_model"] == "validator-x"
    assert len(validator.calls) == 1


def test_composition_validation_live_retries_invalid_signature_taxonomy(tmp_path: Path) -> None:
    compositions_dir = tmp_path / "compositions"
    compositions_dir.mkdir()
    write_jsonl(
        compositions_dir / "candidate_compositions.jsonl",
        [
            {
                "composition_id": "comp-1",
                "company_key": "demo",
                "clause_ids": ["a", "b"],
                "source_rule_ids": ["r1", "r2"],
                "effect_pair": ["disclose", "prohibit"],
                "scope_pair": ["flight status", "flight status"],
                "structure_signals": {
                    "scope_overlap": True,
                    "same_semantic_span": True,
                    "priority_present": False,
                },
                "signature_proposal": "",
                "signature_source": "unresolved",
            }
        ],
    )
    validator = QueueLLMClient(
        [
            """{"pass": true, "signature": "prohibit", "feasibility_status": "pass", "non_separability_status": "pass", "nonseparability_slice": "same span", "adjudication_rationale": "Invalid method taxonomy label."}""",
            """{"pass": true, "signature": "selective-disclosure", "feasibility_status": "pass", "non_separability_status": "pass", "nonseparability_slice": "same span", "adjudication_rationale": "The prohibition changes disclosure handling."}""",
        ]
    )

    summary = run_composition_validation_stage(
        compositions_dir=compositions_dir,
        execution_mode="live",
        validator_client=validator,
        validator_model="validator-x",
    )

    accepted = read_jsonl(compositions_dir / "accepted_compositions.jsonl")
    error_rows = read_jsonl(compositions_dir / "live_errors.jsonl")
    assert summary["accepted_count"] == 1
    assert accepted[0]["signature_proposal"] == "selective-disclosure"
    assert len(validator.calls) == 2
    assert len(error_rows) == 1
    assert error_rows[0]["error_type"] == "LiveSchemaError"
    assert "unsupported signature" in error_rows[0]["error_message"]


def test_composition_validation_live_allows_no_signature_for_rejected_adjudication(tmp_path: Path) -> None:
    compositions_dir = tmp_path / "compositions"
    compositions_dir.mkdir()
    write_jsonl(
        compositions_dir / "candidate_compositions.jsonl",
        [
            {
                "composition_id": "comp-1",
                "company_key": "demo",
                "clause_ids": ["a", "b"],
                "source_rule_ids": ["r1", "r2"],
                "effect_pair": ["disclose", "disclose"],
                "scope_pair": ["flight status", "gate change notification"],
                "structure_signals": {
                    "scope_overlap": True,
                    "same_semantic_span": False,
                    "priority_present": False,
                },
                "signature_proposal": "",
                "signature_source": "unresolved",
            }
        ],
    )
    validator = QueueLLMClient(
        [
            """{"pass": false, "signature": "none", "feasibility_status": "feasible", "non_separability_status": "separable", "nonseparability_slice": "", "adjudication_rationale": "The clauses are independently resolvable."}""",
        ]
    )

    summary = run_composition_validation_stage(
        compositions_dir=compositions_dir,
        execution_mode="live",
        validator_client=validator,
        validator_model="validator-x",
    )

    rejected = read_jsonl(compositions_dir / "rejected_compositions.jsonl")
    assert summary["accepted_count"] == 0
    assert summary["rejected_count"] == 1
    assert summary["adjudication_count"] == 1
    assert len(validator.calls) == 1
    assert rejected[0]["adjudication"]["signature"] == ""
    assert rejected[0]["adjudication"]["raw_signature"] == "none"
    assert rejected[0]["adjudication"]["nonseparability_slice"] == ""


def test_composition_validation_live_can_adjudicate_with_bounded_concurrency(tmp_path: Path) -> None:
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
                "effect_pair": ["disclose", "disclose"],
                "scope_pair": ["flight status", "compensation"],
                "structure_signals": {
                    "scope_overlap": True,
                    "same_semantic_span": True,
                    "priority_present": False,
                },
                "signature_proposal": "",
                "signature_source": "unresolved",
            }
            for index in range(4)
        ],
    )

    class ConcurrentValidator:
        def __init__(self) -> None:
            self.calls = 0
            self.active = 0
            self.max_active = 0
            self.lock = Lock()

        def complete(self, *, model: str, messages: list[LLMMessage]) -> LLMResponse:
            with self.lock:
                self.calls += 1
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                sleep(0.05)
                return LLMResponse(
                    text=(
                        '{"pass": false, "signature": "", "feasibility_status": "feasible", '
                        '"non_separability_status": "separable", "nonseparability_slice": "", '
                        '"adjudication_rationale": "The clauses are independently resolvable."}'
                    ),
                    model=model,
                )
            finally:
                with self.lock:
                    self.active -= 1

    validator = ConcurrentValidator()

    summary = run_composition_validation_stage(
        compositions_dir=compositions_dir,
        execution_mode="live",
        validator_client=validator,
        validator_model="validator-x",
        live_max_workers=4,
    )

    assert summary["adjudication_queue_count"] == 4
    assert summary["adjudication_count"] == 4
    assert summary["accepted_count"] == 0
    assert summary["live_max_workers"] == 4
    assert validator.calls == 4
    assert validator.max_active > 1


def test_composition_validation_adjudication_limit_prioritizes_interaction_candidates(tmp_path: Path) -> None:
    compositions_dir = tmp_path / "compositions"
    compositions_dir.mkdir()
    write_jsonl(
        compositions_dir / "candidate_compositions.jsonl",
        [
            {
                "composition_id": "low-1",
                "company_key": "demo",
                "clause_ids": ["a", "b"],
                "source_rule_ids": ["r1", "r2"],
                "effect_pair": ["disclose", "disclose"],
                "scope_pair": ["flight status", "compensation"],
                "structure_signals": {
                    "scope_overlap": True,
                    "same_semantic_span": True,
                    "joint_trigger_satisfiable": False,
                    "independently_resolvable": True,
                    "priority_present": False,
                    "interaction_filter": {"status": "fail", "conditions": []},
                },
                "signature_proposal": "",
                "signature_source": "unresolved",
            },
            {
                "composition_id": "low-2",
                "company_key": "demo",
                "clause_ids": ["c", "d"],
                "source_rule_ids": ["r3", "r4"],
                "effect_pair": ["disclose", "disclose"],
                "scope_pair": ["gate change", "refunds"],
                "structure_signals": {
                    "scope_overlap": True,
                    "same_semantic_span": True,
                    "joint_trigger_satisfiable": False,
                    "independently_resolvable": True,
                    "priority_present": False,
                    "interaction_filter": {"status": "fail", "conditions": []},
                },
                "signature_proposal": "",
                "signature_source": "unresolved",
            },
            {
                "composition_id": "high-late",
                "company_key": "demo",
                "clause_ids": ["e", "f"],
                "source_rule_ids": ["r5", "r6"],
                "effect_pair": ["disclose", "prohibit"],
                "scope_pair": ["tarmac delay notice", "tarmac notification commitment"],
                "structure_signals": {
                    "scope_overlap": True,
                    "same_semantic_span": True,
                    "joint_trigger_satisfiable": True,
                    "independently_resolvable": False,
                    "priority_present": False,
                    "interaction_filter": {"status": "pass", "conditions": ["same_semantic_span"]},
                },
                "signature_proposal": "",
                "signature_source": "unresolved",
            },
        ],
    )
    validator = QueueLLMClient(
        [
            """{"pass": false, "signature": "", "feasibility_status": "feasible", "non_separability_status": "separable", "nonseparability_slice": "", "adjudication_rationale": "The clauses are independently resolvable."}""",
            """{"pass": true, "signature": "selective-disclosure", "feasibility_status": "pass", "non_separability_status": "pass", "nonseparability_slice": "same tarmac notification scope", "adjudication_rationale": "The prohibition changes the disclosure obligation."}""",
        ]
    )

    summary = run_composition_validation_stage(
        compositions_dir=compositions_dir,
        execution_mode="live",
        validator_client=validator,
        validator_model="validator-x",
        composition_adjudication_limit=2,
    )

    queue = read_jsonl(compositions_dir / "composition_adjudication_queue.jsonl")
    queued_ids = [row["composition_id"] for row in queue]
    assert summary["adjudication_queue_count"] == 2
    assert "high-late" in queued_ids
    assert "low-2" not in queued_ids
    assert len(validator.calls) == 2


def test_composition_validation_skips_live_adjudication_when_signature_budget_is_full(tmp_path: Path) -> None:
    compositions_dir = tmp_path / "compositions"
    compositions_dir.mkdir()
    write_jsonl(
        compositions_dir / "candidate_compositions.jsonl",
        [
            {
                "composition_id": "accepted-1",
                "company_key": "demo",
                "clause_ids": ["a", "b"],
                "source_rule_ids": ["r1", "r2"],
                "effect_pair": ["permit", "prohibit"],
                "scope_pair": ["status", "status"],
                "structure_signals": {
                    "scope_overlap": True,
                    "same_semantic_span": True,
                    "joint_trigger_satisfiable": True,
                    "priority_present": False,
                    "interaction_filter": {"status": "pass", "conditions": ["same_semantic_span"]},
                },
                "signature_proposal": "scope-restriction",
                "signature_source": "structure",
            },
            {
                "composition_id": "skip-1",
                "company_key": "demo",
                "clause_ids": ["c", "d"],
                "source_rule_ids": ["r3", "r4"],
                "effect_pair": ["permit", "prohibit"],
                "scope_pair": ["status", "status"],
                "structure_signals": {
                    "scope_overlap": True,
                    "same_semantic_span": True,
                    "joint_trigger_satisfiable": False,
                    "priority_present": False,
                    "interaction_filter": {"status": "unresolved", "conditions": []},
                },
                "signature_proposal": "scope-restriction",
                "signature_source": "structure",
            },
        ],
    )
    validator = QueueLLMClient(
        [
            """{"pass": true, "signature": "scope-restriction", "feasibility_status": "pass", "non_separability_status": "pass", "nonseparability_slice": "same span", "adjudication_rationale": "Would be wasteful if called."}""",
        ]
    )

    summary = run_composition_validation_stage(
        compositions_dir=compositions_dir,
        execution_mode="live",
        validator_client=validator,
        validator_model="validator-x",
        composition_limit_per_signature=1,
        composition_adjudication_limit=5,
    )

    rejected = read_jsonl(compositions_dir / "rejected_compositions.jsonl")
    assert summary["accepted_count"] == 1
    assert summary["adjudication_count"] == 0
    assert len(validator.calls) == 0
    assert rejected[0]["budget_excluded_reason"] == "composition_limit_per_signature"


def test_query_proposal_and_validation_live_use_separate_clients(tmp_path: Path) -> None:
    query_dir = tmp_path / "query_generation"
    proposal_client = QueueLLMClient(
        [
            """{"scenario": {"shared_user_scenario": "A refund request also raises account-holder verification requirements.", "jointly_satisfied_triggers": ["refund request", "unverified account holder"], "coupled_scope_or_path": "refunds; verification", "non_decomposability_rationale": "The refund action and verification gate determine one handling path.", "required_state_assumptions": ["The account holder is not verified."], "leakage_naturalness_self_check": "Natural customer wording without policy labels."}, "query_text": "Can you refund this booking even though the account holder is not verified?"}""",
        ]
    )
    validator_client = QueueLLMClient(
        [
            """{"pass": true, "composition_validity": true, "non_separability": true, "facet_coverage": ["skipped-gate"], "scenario_level_interaction": "pass", "query_level_interaction": "pass", "independent_subrequests": false, "naturalness": "pass", "leakage": "pass", "redundancy": "novel", "validation_confidence": 0.97, "validation_rationale": "The query tests gating before fulfillment."}""",
        ]
    )

    run_query_proposal_stage(
        query_generation_dir=query_dir,
        accepted_compositions=[
            {
                "composition_id": "comp-1",
                "signature_proposal": "prerequisite-gating",
                "clause_ids": ["a", "b"],
                "scope_pair": ["refunds", "verification"],
            }
        ],
        facet_library={"prerequisite-gating": ("skipped-gate",)},
        execution_mode="live",
        proposal_client=proposal_client,
        proposal_model="proposal-x",
    )
    summary = run_query_validation_stage(
        query_generation_dir=query_dir,
        execution_mode="live",
        validator_client=validator_client,
        validator_model="validator-x",
    )

    accepted = read_jsonl(query_dir / "accepted_queries.jsonl")
    adjudications = read_jsonl(query_dir / "query_adjudications.jsonl")
    assert summary["accepted_count"] == 1
    assert accepted[0]["query_text"].startswith("Can you refund")
    assert accepted[0]["validation_metadata"]["validator_model"] == "validator-x"
    assert adjudications[0]["facet_coverage"] == ["skipped-gate"]
    assert len(proposal_client.calls) == 1
    assert len(validator_client.calls) == 1


def test_query_proposal_live_retries_missing_nested_schema_field(tmp_path: Path) -> None:
    query_dir = tmp_path / "query_generation"
    proposal_client = QueueLLMClient(
        [
            """{"scenario": {"shared_user_scenario": "A refund request also raises verification requirements.", "jointly_satisfied_triggers": ["refund request", "unverified user"], "coupled_scope_or_path": "refunds; verification", "non_decomposability_rationale": "The refund and verification constraints determine one response.", "required_state_assumptions": ["The user has not been verified."]}, "query_text": "Can you refund this booking without verifying me first?"}""",
            """{"scenario": {"shared_user_scenario": "A refund request also raises verification requirements.", "jointly_satisfied_triggers": ["refund request", "unverified user"], "coupled_scope_or_path": "refunds; verification", "non_decomposability_rationale": "The refund and verification constraints determine one response.", "required_state_assumptions": ["The user has not been verified."], "leakage_naturalness_self_check": "Natural customer wording without policy labels."}, "query_text": "Can you refund this booking without verifying me first?"}""",
        ]
    )

    summary = run_query_proposal_stage(
        query_generation_dir=query_dir,
        accepted_compositions=[
            {
                "composition_id": "comp-1",
                "signature_proposal": "prerequisite-gating",
                "clause_ids": ["a", "b"],
                "scope_pair": ["refunds", "verification"],
            }
        ],
        facet_library={"prerequisite-gating": ("skipped-gate",)},
        execution_mode="live",
        proposal_client=proposal_client,
        proposal_model="proposal-x",
    )

    live_errors = read_jsonl(query_dir / "live_errors.jsonl")
    assert summary["candidate_query_count"] == 1
    assert len(proposal_client.calls) == 2
    assert live_errors[0]["error_type"] == "LiveSchemaError"
    assert "leakage_naturalness_self_check" in live_errors[0]["error_message"]


def test_query_proposal_live_can_generate_multiple_variants_per_facet(tmp_path: Path) -> None:
    query_dir = tmp_path / "query_generation"
    proposal_client = QueueLLMClient(
        [
            """{"scenario": {"shared_user_scenario": "A refund request also raises identity verification requirements.", "jointly_satisfied_triggers": ["refund request", "unverified user"], "coupled_scope_or_path": "refunds; verification", "non_decomposability_rationale": "The refund decision depends on the verification path.", "required_state_assumptions": ["The user has not been verified."], "leakage_naturalness_self_check": "Natural customer wording without policy labels."}, "query_text": "Can you refund this booking without verifying me first?"}""",
            """{"scenario": {"shared_user_scenario": "A refund request also raises account access and verification requirements.", "jointly_satisfied_triggers": ["refund request", "lost account access"], "coupled_scope_or_path": "refunds; verification", "non_decomposability_rationale": "The refund and access conditions must be resolved as one handling path.", "required_state_assumptions": ["The user lost access to the account."], "leakage_naturalness_self_check": "Natural customer wording without policy labels."}, "query_text": "I lost access to my account; can you still issue the refund now?"}""",
        ]
    )

    summary = run_query_proposal_stage(
        query_generation_dir=query_dir,
        accepted_compositions=[
            {
                "composition_id": "comp-1",
                "signature_proposal": "prerequisite-gating",
                "clause_ids": ["a", "b"],
                "scope_pair": ["refunds", "verification"],
            }
        ],
        facet_library={"prerequisite-gating": ("skipped-gate",)},
        execution_mode="live",
        proposal_client=proposal_client,
        proposal_model="proposal-x",
        query_variants_per_facet=2,
    )

    queries = read_jsonl(query_dir / "candidate_queries.jsonl")
    assert summary["candidate_query_count"] == 2
    assert [row["query_id"] for row in queries] == [
        "comp-1::skipped-gate::v0",
        "comp-1::skipped-gate::v1",
    ]
    assert [row["query_variant_index"] for row in queries] == [0, 1]
    assert len(proposal_client.calls) == 2
    assert "variant 1 of 2" in proposal_client.calls[0]["messages"][-1].content
    assert "variant 2 of 2" in proposal_client.calls[1]["messages"][-1].content


def test_query_proposal_live_uses_bounded_concurrency(tmp_path: Path) -> None:
    query_dir = tmp_path / "query_generation"

    class DelayedProposalClient:
        def __init__(self) -> None:
            self.active = 0
            self.max_active = 0
            self.calls: list[float] = []
            self._lock = Lock()

        def complete(self, *, model: str, messages: list[LLMMessage]) -> LLMResponse:
            with self._lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                self.calls.append(monotonic())
            try:
                sleep(0.05)
                return LLMResponse(
                    text="""{"scenario": {"shared_user_scenario": "A refund request also raises verification requirements.", "jointly_satisfied_triggers": ["refund request", "unverified user"], "coupled_scope_or_path": "refunds; verification", "non_decomposability_rationale": "The refund and verification constraints determine one response.", "required_state_assumptions": ["The user has not been verified."], "leakage_naturalness_self_check": "Natural customer wording without policy labels."}, "query_text": "Can you refund this booking without verifying me first?"}""",
                    model=model,
                )
            finally:
                with self._lock:
                    self.active -= 1

    proposal_client = DelayedProposalClient()

    summary = run_query_proposal_stage(
        query_generation_dir=query_dir,
        accepted_compositions=[
            {
                "composition_id": "comp-1",
                "signature_proposal": "prerequisite-gating",
                "clause_ids": ["a", "b"],
                "scope_pair": ["refunds", "verification"],
            }
        ],
        facet_library={
            "prerequisite-gating": (
                "skipped-gate",
                "over_gating",
                "mis_scoped_gating",
                "procedural_leakage",
            )
        },
        execution_mode="live",
        proposal_client=proposal_client,
        proposal_model="proposal-x",
        live_max_workers=4,
    )

    queries = read_jsonl(query_dir / "candidate_queries.jsonl")
    assert summary["candidate_query_count"] == 4
    assert [row["query_id"] for row in queries] == [
        "comp-1::skipped-gate",
        "comp-1::over_gating",
        "comp-1::mis_scoped_gating",
        "comp-1::procedural_leakage",
    ]
    assert proposal_client.max_active > 1


def test_query_validation_live_uses_bounded_concurrency(tmp_path: Path) -> None:
    query_dir = tmp_path / "query_generation"
    query_dir.mkdir()
    write_jsonl(
        query_dir / "candidate_queries.jsonl",
        [
            {
                "query_id": f"q{index}",
                "composition_id": "comp-1",
                "signature_proposal": "prerequisite-gating",
                "target_facet": "skipped-gate",
                "target_facets": ["skipped-gate"],
                "query_text": f"Can you refund this booking without verifying me first? {index}",
                "scenario_stub": {"clause_ids": ["a", "b"]},
            }
            for index in range(4)
        ],
    )

    class DelayedValidatorClient:
        def __init__(self) -> None:
            self.active = 0
            self.max_active = 0
            self.calls: list[float] = []
            self._lock = Lock()

        def complete(self, *, model: str, messages: list[LLMMessage]) -> LLMResponse:
            with self._lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                self.calls.append(monotonic())
            try:
                sleep(0.05)
                return LLMResponse(
                    text="""{"pass": true, "composition_validity": true, "non_separability": true, "facet_coverage": ["skipped-gate"], "scenario_level_interaction": "pass", "query_level_interaction": "pass", "independent_subrequests": false, "naturalness": "pass", "leakage": "pass", "redundancy": "novel", "validation_confidence": 0.97, "validation_rationale": "The query tests gating before fulfillment."}""",
                    model=model,
                )
            finally:
                with self._lock:
                    self.active -= 1

    validator_client = DelayedValidatorClient()

    summary = run_query_validation_stage(
        query_generation_dir=query_dir,
        execution_mode="live",
        validator_client=validator_client,
        validator_model="validator-x",
        live_max_workers=4,
    )

    accepted = read_jsonl(query_dir / "accepted_queries.jsonl")
    assert summary["accepted_count"] == 4
    assert [row["query_id"] for row in accepted] == ["q0", "q1", "q2", "q3"]
    assert validator_client.max_active > 1


def test_query_validation_live_records_schema_error_artifact_and_raises(tmp_path: Path) -> None:
    query_dir = tmp_path / "query_generation"
    query_dir.mkdir()
    write_jsonl(
        query_dir / "candidate_queries.jsonl",
        [
            {
                "query_id": "q1",
                "composition_id": "comp-1",
                "signature_proposal": "prerequisite-gating",
                "target_facet": "skipped-gate",
                "target_facets": ["skipped-gate"],
                "query_text": "Can you refund this booking without verifying me?",
                "scenario_stub": {"clause_ids": ["a", "b"]},
            }
        ],
    )
    bad_response = """{"pass": true, "composition_validity": true, "non_separability": true, "scenario_level_interaction": "pass", "query_level_interaction": "pass", "independent_subrequests": false, "naturalness": "pass", "leakage": "pass", "redundancy": "novel", "validation_confidence": 0.97, "validation_rationale": "missing facet_coverage"}"""
    validator_client = QueueLLMClient([bad_response, bad_response, bad_response])

    try:
        run_query_validation_stage(
            query_generation_dir=query_dir,
            execution_mode="live",
            validator_client=validator_client,
            validator_model="validator-x",
        )
    except LiveSchemaError as exc:
        assert "facet_coverage" in str(exc)
    else:
        raise AssertionError("LiveSchemaError was not raised")

    error_rows = read_jsonl(query_dir / "live_errors.jsonl")
    assert len(error_rows) == 3
    assert [row["attempt"] for row in error_rows] == [1, 2, 3]
    assert all(row["max_attempts"] == 3 for row in error_rows)
    assert all(row["stage_name"] == "query_validation" for row in error_rows)
    assert all(row["target_id"] == "q1" for row in error_rows)
    assert all(row["model"] == "validator-x" for row in error_rows)
    assert all(row["error_type"] == "LiveSchemaError" for row in error_rows)
    assert all(row["error_message"] == "query_validation q1 missing required field: facet_coverage" for row in error_rows)
    assert all(row["raw_response"] == bad_response for row in error_rows)


def test_coverage_downstream_and_response_judgment_live_are_split(tmp_path: Path) -> None:
    coverage_client = QueueLLMClient(
        [
            """{"signature_label": "workflow-transfer", "facet_labels": ["missed-transfer"], "coverage_rationale": "The query requires a routing path."}""",
        ]
    )
    coverage_summary = run_coverage_judge_stage(
        coverage_dir=tmp_path / "coverage",
        accepted_queries=[
            {
                "query_id": "q1",
                "composition_id": "comp-1",
                "signature_proposal": "scope-restriction",
                "target_facet": "semantic-leakage",
                "query_text": "Can you recommend a product for my diagnosis?",
                "scenario_stub": {"clause_ids": ["a", "b"]},
            }
        ],
        facet_library={
            "scope-restriction": ("semantic-leakage", "boundary-overreach"),
            "workflow-transfer": ("missed-transfer", "wrong-route"),
        },
        execution_mode="live",
        coverage_client=coverage_client,
        coverage_model="coverage-x",
    )
    assert coverage_summary["coverage_result_count"] == 1
    universes = read_jsonl(tmp_path / "coverage" / "composition_facet_universes.jsonl")
    assert universes[0]["signature"] == "workflow-transfer"
    assert universes[0]["facet_universe"] == ["missed-transfer", "wrong-route"]

    benchmark_items = [
        {
            "item_id": "q1",
            "signature": "scope-restriction",
            "facet": "semantic-leakage",
            "target_facet": "semantic-leakage",
            "query_text": "Can you recommend a product for my diagnosis?",
            "expected_handling_pattern": {"mode": "policy_composition_sensitive"},
        }
    ]
    downstream_client = QueueLLMClient(["I can give general information, but not personalized medical advice."])
    judge_client = QueueLLMClient(
        [
            """{"handling_pattern_match": true, "response_mode_correct": true, "boundary_correct": true, "required_actions_correct": true, "mandatory_content_correct": true, "overall_correct": true, "under_enforcement": false, "over_enforcement": false, "observed_handling": ["withhold_restricted_part"], "observed_facets": [], "response_outcome": "acceptable", "scoring_rationale": "The response keeps the personalized boundary.", "evidence_span": "not personalized medical advice"}""",
        ]
    )

    run_downstream_chatbot_stage(
        evaluation_dir=tmp_path / "evaluation",
        benchmark_items=benchmark_items,
        system_prompt="You are the official customer-facing assistant.",
        execution_mode="live",
        downstream_client=downstream_client,
        downstream_model="chat-x",
    )
    response_summary = run_response_judgment_stage(
        evaluation_dir=tmp_path / "evaluation",
        benchmark_items=benchmark_items,
        execution_mode="live",
        response_judge_client=judge_client,
        response_judge_model="judge-x",
    )

    responses = read_jsonl(tmp_path / "evaluation" / "chatbot_responses.jsonl")
    scores = read_jsonl(tmp_path / "evaluation" / "per_item_scores.jsonl")
    assert responses[0]["response_model"] == "chat-x"
    assert scores[0]["overall_correct"] is True
    assert scores[0]["target_facet"] == "semantic-leakage"
    assert scores[0]["observed_facets"] == []
    assert scores[0]["response_outcome"] == "acceptable"
    assert scores[0]["evidence_span"] == "not personalized medical advice"
    assert response_summary["judgment_count"] == 1
    assert len(coverage_client.calls) == 1
    assert len(downstream_client.calls) == 1
    assert len(judge_client.calls) == 1


def test_response_judgment_live_requires_evidence_span(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("COPAL_LIVE_JSON_MAX_ATTEMPTS", "1")
    benchmark_items = [
        {
            "item_id": "q1",
            "signature": "scope-restriction",
            "facet": "semantic-leakage",
            "target_facet": "semantic-leakage",
            "query_text": "Can you recommend a product for my diagnosis?",
            "expected_handling_pattern": {"mode": "policy_composition_sensitive"},
        }
    ]
    (tmp_path / "evaluation").mkdir()
    write_jsonl(
        tmp_path / "evaluation" / "chatbot_responses.jsonl",
        [
            {
                "response_id": "resp-1",
                "item_id": "q1",
                "response_model": "chat-x",
                "response_text": "I can give general information, but not personalized medical advice.",
            }
        ],
    )
    judge_client = QueueLLMClient(
        [
            """{"handling_pattern_match": true, "response_mode_correct": true, "boundary_correct": true, "required_actions_correct": true, "mandatory_content_correct": true, "overall_correct": true, "under_enforcement": false, "over_enforcement": false, "observed_handling": ["withhold_restricted_part"], "observed_facets": [], "response_outcome": "acceptable", "scoring_rationale": "The response keeps the personalized boundary."}""",
        ]
    )

    try:
        run_response_judgment_stage(
            evaluation_dir=tmp_path / "evaluation",
            benchmark_items=benchmark_items,
            execution_mode="live",
            response_judge_client=judge_client,
            response_judge_model="judge-x",
        )
    except LiveSchemaError as exc:
        assert "evidence_span" in str(exc)
    else:
        raise AssertionError("response judgment must require an evidence_span for auditability")


def test_coverage_judge_live_uses_bounded_concurrency(tmp_path: Path) -> None:
    accepted_queries = [
        {
            "query_id": f"q{index}",
            "composition_id": "comp-1",
            "signature_proposal": "scope-restriction",
            "target_facet": "semantic-leakage",
            "query_text": f"Can you recommend a product for my diagnosis? {index}",
            "scenario_stub": {"clause_ids": ["a", "b"]},
        }
        for index in range(4)
    ]

    class DelayedCoverageClient:
        def __init__(self) -> None:
            self.active = 0
            self.max_active = 0
            self.calls: list[float] = []
            self._lock = Lock()

        def complete(self, *, model: str, messages: list[LLMMessage]) -> LLMResponse:
            with self._lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                self.calls.append(monotonic())
            try:
                sleep(0.05)
                return LLMResponse(
                    text="""{"signature_label": "scope-restriction", "facet_labels": ["semantic-leakage"], "coverage_rationale": "The query asks for a prohibited semantic recommendation."}""",
                    model=model,
                )
            finally:
                with self._lock:
                    self.active -= 1

    coverage_client = DelayedCoverageClient()

    summary = run_coverage_judge_stage(
        coverage_dir=tmp_path / "coverage",
        accepted_queries=accepted_queries,
        facet_library={"scope-restriction": ("semantic-leakage", "boundary-overreach")},
        execution_mode="live",
        coverage_client=coverage_client,
        coverage_model="coverage-x",
        live_max_workers=4,
    )

    coverages = read_jsonl(tmp_path / "coverage" / "accepted_query_coverages.jsonl")
    assert summary["coverage_result_count"] == 4
    assert [row["query_id"] for row in coverages] == ["q0", "q1", "q2", "q3"]
    assert coverage_client.max_active > 1
