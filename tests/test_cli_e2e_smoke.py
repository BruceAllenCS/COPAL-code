from pathlib import Path

from copal import __version__
from copal.cli import (
    build_experiment_manifest,
    build_parser,
    build_role_config_from_args,
    limit_facets_for_live_smoke,
    limit_world_for_live_smoke,
    main,
    select_experiment_worlds,
)
from copal.data_sources import load_company_worlds, load_system_prompts
from copal.io import read_json, write_json
from copal.models import CompanyWorld, PolicyRule


def test_package_exposes_version() -> None:
    assert __version__ == "0.1.0"


def test_cli_parser_supports_run_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["run", "--company-key", "demo", "--execution-mode", "deterministic"])
    assert args.command == "run"


def test_cli_parser_requires_explicit_execution_mode() -> None:
    parser = build_parser()
    try:
        parser.parse_args(["run", "--company-key", "demo"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("execution mode must be explicit to avoid accidental deterministic runs")


def test_cli_parser_does_not_accept_llm_api_key_path() -> None:
    parser = build_parser()
    try:
        parser.parse_args(["run", "--company-key", "demo", "--api-keys-path", "local_api_keys.json"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Friday live execution should not expose --api-keys-path")


def test_cli_parser_accepts_model_name_path() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["run", "--company-key", "demo", "--execution-mode", "deterministic", "--model-name-path", "model_name.json"]
    )
    assert args.model_name_path == "model_name.json"


def test_cli_parser_accepts_explicit_live_smoke_mode() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "run",
            "--company-key",
            "demo",
            "--execution-mode",
            "live",
            "--live-smoke",
            "--smoke-rule-limit-per-side",
            "1",
            "--smoke-facet-limit-per-signature",
            "1",
            "--live-smoke-model",
            "kimi-k2.6",
        ]
    )
    assert args.live_smoke is True
    assert args.smoke_rule_limit_per_side == 1
    assert args.smoke_facet_limit_per_signature == 1
    assert args.live_smoke_model == "kimi-k2.6"


def test_cli_parser_accepts_experiment_live_smoke_mode_and_data_paths() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "experiment",
            "run",
            "--experiment-id",
            "pilot",
            "--company-limit",
            "2",
            "--execution-mode",
            "live",
            "--live-smoke",
            "--smoke-rule-limit-per-side",
            "1",
            "--smoke-facet-limit-per-signature",
            "1",
            "--live-smoke-model",
            "kimi-k2.6",
            "--policies-path",
            "custom_policies.jsonl",
            "--prompts-path",
            "custom_prompts.jsonl",
        ]
    )

    assert args.command == "experiment"
    assert args.experiment_command == "run"
    assert args.live_smoke is True
    assert args.live_smoke_model == "kimi-k2.6"
    assert args.policies_path == "custom_policies.jsonl"
    assert args.prompts_path == "custom_prompts.jsonl"


def test_cli_parser_accepts_experiment_tier_controls() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "experiment",
            "run",
            "--experiment-id",
            "yield10",
            "--company-limit",
            "10",
            "--execution-mode",
            "live",
            "--sample-strategy",
            "one-per-industry",
            "--stop-after",
            "selection",
            "--all-roles-model",
            "kimi-k2.6",
            "--live-max-workers",
            "4",
            "--composition-limit-per-signature",
            "2",
            "--composition-adjudication-limit",
            "0",
        ]
    )

    assert args.sample_strategy == "one-per-industry"
    assert args.stop_after == "selection"
    assert args.all_roles_model == "kimi-k2.6"
    assert args.live_max_workers == 4
    assert args.composition_limit_per_signature == 2
    assert args.composition_adjudication_limit == 0


def test_cli_parser_accepts_explicit_role_model_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "experiment",
            "run",
            "--experiment-id",
            "protocol-freeze",
            "--company-limit",
            "30",
            "--execution-mode",
            "live",
            "--proposal-model",
            "qwen3.5-baidu",
            "--query-proposal-model",
            "Doubao-Seed-2.0-pro",
            "--canonicalization-model",
            "qwen3.5-baidu",
            "--validator-model",
            "qwen3.5-baidu",
            "--coverage-judge-model",
            "kimi-k2.6",
            "--downstream-chatbot-model",
            "MiniMax-M2.7",
            "--response-judge-model",
            "glm-5.1",
        ]
    )

    assert args.proposal_model == "qwen3.5-baidu"
    assert args.query_proposal_model == "Doubao-Seed-2.0-pro"
    assert args.canonicalization_model == "qwen3.5-baidu"
    assert args.validator_model == "qwen3.5-baidu"
    assert args.coverage_judge_model == "kimi-k2.6"
    assert args.downstream_chatbot_model == "MiniMax-M2.7"
    assert args.response_judge_model == "glm-5.1"


def test_experiment_manifest_records_model_roster_and_role_config(
    tmp_path: Path, monkeypatch: object
) -> None:
    monkeypatch.setenv("COPAL_FRIDAY_RESPONSE_FORMAT", "json_object")
    monkeypatch.setenv("COPAL_FRIDAY_MAX_RETRIES", "4")
    monkeypatch.setenv("COPAL_FRIDAY_RETRY_BACKOFF_SECONDS", "10")
    monkeypatch.setenv("COPAL_FRIDAY_MIN_INTERVAL_SECONDS", "65")
    parser = build_parser()
    args = parser.parse_args(
        [
            "experiment",
            "run",
            "--experiment-id",
            "protocol-freeze",
            "--company-limit",
            "30",
            "--execution-mode",
            "live",
            "--sample-strategy",
            "one-per-industry",
            "--stop-after",
            "selection",
            "--proposal-model",
            "qwen3.5-baidu",
            "--query-proposal-model",
            "Doubao-Seed-2.0-pro",
            "--response-judge-model",
            "glm-5.1",
        ]
    )
    role_config = build_role_config_from_args(args)

    manifest = build_experiment_manifest(
        args=args,
        company_runs_dir=tmp_path / "runs" / "experiments" / "protocol-freeze" / "company_runs",
        composition_limit_per_signature=2,
        composition_adjudication_limit=48,
        role_config=role_config,
        model_names=("qwen3.5-baidu", "glm-5.1"),
    )

    assert manifest["model_roster"] == ["qwen3.5-baidu", "glm-5.1"]
    assert manifest["model_count"] == 2
    assert manifest["role_config"]["proposal_model"] == "qwen3.5-baidu"
    assert manifest["role_config"]["query_proposal_model"] == "Doubao-Seed-2.0-pro"
    assert manifest["role_config"]["response_judge_model"] == "glm-5.1"
    assert manifest["role_config"]["validator_model"] == "qwen3.5-baidu"
    assert manifest["live_client"]["response_format"] == {"type": "json_object"}
    assert manifest["live_client"]["max_retries"] == 4
    assert manifest["live_client"]["retry_backoff_seconds"] == 10.0
    assert manifest["live_client"]["min_interval_seconds"] == 65.0
    assert "all_roles_model" not in manifest


def test_experiment_live_smoke_requires_live_execution_mode(tmp_path) -> None:
    try:
        main(
            [
                "experiment",
                "run",
                "--experiment-id",
                "bad-smoke",
                "--company-limit",
                "1",
                "--execution-mode",
                "deterministic",
                "--live-smoke",
                "--runs-dir",
                str(tmp_path / "runs"),
                "--cache-dir",
                str(tmp_path / "cache"),
            ]
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("experiment --live-smoke must require --execution-mode live")


def test_experiment_all_roles_model_requires_live_execution_mode(tmp_path) -> None:
    try:
        main(
            [
                "experiment",
                "run",
                "--experiment-id",
                "bad-role-model",
                "--company-limit",
                "1",
                "--execution-mode",
                "deterministic",
                "--all-roles-model",
                "kimi-k2.6",
                "--runs-dir",
                str(tmp_path / "runs"),
                "--cache-dir",
                str(tmp_path / "cache"),
            ]
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("experiment --all-roles-model must require --execution-mode live")


def test_live_smoke_limits_policy_world_and_facets_without_changing_defaults() -> None:
    allowed = [
        PolicyRule(
            rule_id=f"A{index}",
            rule_text=f"allowed {index}",
            category="demo",
            severity="low",
            rationale="demo",
            verifiable=True,
            verifiability_confidence="high",
            raw={},
        )
        for index in range(2)
    ]
    prohibited = [
        PolicyRule(
            rule_id=f"P{index}",
            rule_text=f"prohibited {index}",
            category="demo",
            severity="low",
            rationale="demo",
            verifiable=True,
            verifiability_confidence="high",
            raw={},
        )
        for index in range(2)
    ]
    world = CompanyWorld(
        company_key="demo",
        industry="demo",
        company_name="Demo",
        company_index=0,
        enterprise_config={},
        allowed_behaviors=allowed,
        prohibited_behaviors=prohibited,
        quality_scores={},
        raw={},
    )

    limited_world = limit_world_for_live_smoke(world, rule_limit_per_side=1)
    limited_facets = limit_facets_for_live_smoke(
        {"scope-restriction": ("a", "b"), "prerequisite-gating": ("c", "d")},
        facet_limit_per_signature=1,
    )

    assert [rule.rule_id for rule in limited_world.allowed_behaviors] == ["A0"]
    assert [rule.rule_id for rule in limited_world.prohibited_behaviors] == ["P0"]
    assert [rule.rule_id for rule in world.allowed_behaviors] == ["A0", "A1"]
    assert limited_facets == {"scope-restriction": ("a",), "prerequisite-gating": ("c",)}


def test_select_experiment_worlds_can_take_one_company_per_industry() -> None:
    worlds = load_company_worlds(Path("data/compass_policies/compass_policies_final.jsonl"))
    selected = select_experiment_worlds(worlds, company_limit=30, sample_strategy="one-per-industry")

    assert len(selected) == 30
    assert len({world.industry for world in selected}) == 30


def test_one_per_industry_selection_uses_prompt_dataset_keys() -> None:
    worlds = load_company_worlds(Path("data/compass_policies/compass_policies_final.jsonl"))
    prompts = load_system_prompts(Path("data/compass_policies/company_system_prompts.jsonl"))

    selected = select_experiment_worlds(worlds, company_limit=30, sample_strategy="one-per-industry")
    prompt_keys = {prompt.company_key for prompt in prompts}

    assert [world.company_key for world in selected if world.company_key not in prompt_keys] == []


def test_run_command_creates_aligned_stage_directories(tmp_path) -> None:
    exit_code = main(
        [
            "run",
            "--company-key",
            "Air transportation||000||Skyline International Airways",
            "--execution-mode",
            "deterministic",
            "--runs-dir",
            str(tmp_path / "runs"),
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )
    assert exit_code == 0
    run_dir = next((tmp_path / "runs").iterdir())
    assert (run_dir / "runtime_errors.jsonl").exists()
    assert (run_dir / "validation").exists()
    assert (run_dir / "reference_subset").exists()
    assert (run_dir / "baselines").exists()
    assert (run_dir / "reports" / "summary.json").exists()
    assert (run_dir / "reports" / "run_report.md").exists()
    assert (run_dir / "reports" / "signature_facet_breakdown.md").exists()
    assert (run_dir / "reports" / "validation_calibration_report.md").exists()
    assert (run_dir / "reports" / "audit_report.md").exists()
    assert (run_dir / "reports" / "qualitative_cases.md").exists()
    assert (run_dir / "grounding" / "grounded_clause_library.jsonl").exists()
    assert (run_dir / "grounding" / "raw_clause_extractions.jsonl").exists()
    assert (run_dir / "compositions" / "accepted_compositions.jsonl").exists()
    assert (run_dir / "compositions" / "composition_adjudication_queue.jsonl").exists()
    assert (run_dir / "query_generation" / "accepted_queries.jsonl").exists()
    assert (run_dir / "query_generation" / "query_adjudication_queue.jsonl").exists()
    assert (run_dir / "coverage" / "accepted_query_coverages.jsonl").exists()
    assert (run_dir / "coverage" / "composition_facet_universes.jsonl").exists()
    assert (run_dir / "selection" / "benchmark_items_final.jsonl").exists()
    assert (run_dir / "reference_subset" / "reference_subset.jsonl").exists()
    assert (run_dir / "baselines" / "baseline_protocols.jsonl").exists()
    assert (run_dir / "baselines" / "construction_quality_metrics.jsonl").exists()
    assert (run_dir / "baselines" / "invalid_item_breakdown.jsonl").exists()
    assert (run_dir / "baselines" / "ablation_metrics.jsonl").exists()
    assert (run_dir / "baselines" / "baseline_experiment_summary.json").exists()
    assert (run_dir / "audit" / "human_audit_records.jsonl").exists()
    assert (run_dir / "evaluation" / "chatbot_responses.jsonl").exists()
    assert (run_dir / "evaluation" / "response_judge_inputs.jsonl").exists()
    assert (run_dir / "evaluation" / "evaluation_summary.json").exists()
    summary = read_json(run_dir / "reports" / "summary.json")
    assert summary["status"] in {
        "initialized",
        "grounding_completed",
        "composition_completed",
        "selection_completed",
        "evaluation_completed",
    }
    assert summary["baselines"]["method_count"] == 5


def test_run_command_reuses_completed_stage_checkpoints_for_same_run_id(tmp_path) -> None:
    args = [
        "run",
        "--company-key",
        "Air transportation||000||Skyline International Airways",
        "--execution-mode",
        "deterministic",
        "--run-id",
        "resume-demo",
        "--runs-dir",
        str(tmp_path / "runs"),
        "--cache-dir",
        str(tmp_path / "cache"),
    ]

    assert main(args) == 0
    run_dir = tmp_path / "runs" / "resume-demo"
    grounding_manifest = read_json(run_dir / "grounding" / "grounding_proposal_stage_manifest.json")
    runtime_errors_path = run_dir / "runtime_errors.jsonl"
    runtime_errors_path.write_text('{"kept": true}\n', encoding="utf-8")
    write_json(run_dir / "grounding" / "sentinel.json", {"should_survive": True})

    assert main(args) == 0

    summary = read_json(run_dir / "reports" / "summary.json")
    reused_manifest = read_json(run_dir / "grounding" / "grounding_proposal_stage_manifest.json")
    assert summary["checkpoints"]["grounding_proposal"]["checkpoint_reused"] is True
    assert reused_manifest["finished_at"] == grounding_manifest["finished_at"]
    assert runtime_errors_path.read_text(encoding="utf-8") == '{"kept": true}\n'
    assert read_json(run_dir / "grounding" / "sentinel.json") == {"should_survive": True}
