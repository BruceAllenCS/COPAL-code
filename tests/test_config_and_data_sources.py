from pathlib import Path

import pytest

from copal.config import DEFAULT_FACETS, DEFAULT_SIGNATURES, RoleConfig, RunConfig, load_model_names
from copal.data_sources import load_company_worlds, select_company_world
from copal.manifest_compat import manifests_match_for_resume
from copal.stages.run_setup import build_run_manifest


def test_load_company_worlds_reads_local_dataset() -> None:
    worlds = load_company_worlds(Path("data/compass_policies/compass_policies_final.jsonl"))
    assert len(worlds) >= 1
    assert worlds[0].company_key


def test_select_company_world_returns_matching_prompt() -> None:
    world, prompt = select_company_world(
        policies_path=Path("data/compass_policies/compass_policies_final.jsonl"),
        prompts_path=Path("data/compass_policies/company_system_prompts.jsonl"),
        company_key="Air transportation||000||Skyline International Airways",
    )
    assert world.company_key == prompt.company_key
    assert "Skyline International Airways" in prompt.system_prompt


def test_default_taxonomy_matches_current_manuscript_relation_patterns() -> None:
    assert DEFAULT_SIGNATURES == (
        "scope-restriction",
        "prerequisite-gating",
        "selective-disclosure",
        "workflow-transfer",
    )
    assert DEFAULT_FACETS == {
        "scope-restriction": ("boundary-overreach", "over-refusal", "semantic-leakage"),
        "prerequisite-gating": ("skipped-gate", "wrong-scope-gate", "pre-gate-leakage"),
        "selective-disclosure": ("protected-field-leakage", "all-withholding", "blurred-disclosure"),
        "workflow-transfer": ("missed-transfer", "wrong-route", "latent-continuation"),
    }


def test_run_manifest_records_separate_model_roles(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COPAL_FRIDAY_RESPONSE_FORMAT", "json_object")
    monkeypatch.setenv("COPAL_FRIDAY_MAX_RETRIES", "4")
    monkeypatch.setenv("COPAL_FRIDAY_RETRY_BACKOFF_SECONDS", "10")
    monkeypatch.setenv("COPAL_FRIDAY_MIN_INTERVAL_SECONDS", "65")
    config = RunConfig(
        run_id="copal_test",
        company_key="Air transportation||000||Skyline International Airways",
        execution_mode="live",
        role_config=RoleConfig(
            proposal_model="model-a",
            canonicalization_model="model-b",
            validator_model="model-c",
            coverage_judge_model="model-d",
            downstream_chatbot_model="model-e",
            response_judge_model="model-f",
        ),
    )
    manifest = build_run_manifest(config)
    assert manifest["models"]["proposal_model"] == "model-a"
    assert manifest["models"]["coverage_judge_model"] == "model-d"
    assert manifest["execution_mode"] == "live"
    assert manifest["live_client"]["provider"] == "friday"
    assert manifest["live_client"]["response_format"] == {"type": "json_object"}
    assert manifest["live_client"]["max_retries"] == 4
    assert manifest["live_client"]["retry_backoff_seconds"] == 10.0
    assert manifest["live_client"]["min_interval_seconds"] == 65.0
    assert "api_keys_path" not in manifest["inputs"]


def test_load_model_names_reads_json_array_roster(tmp_path: Path) -> None:
    path = tmp_path / "model_name.json"
    path.write_text(
        """[
  "kimi-k2.6",
  "MiniMax-M2.7",
  "qwen3.5-baidu",
  "glm-5.1",
  "deepseek-v3.2-tencent"
]
""",
        encoding="utf-8",
    )

    model_names = load_model_names(path)

    assert model_names == (
        "kimi-k2.6",
        "MiniMax-M2.7",
        "qwen3.5-baidu",
        "glm-5.1",
        "deepseek-v3.2-tencent",
    )


def test_load_model_names_rejects_empty_newline_object_or_duplicate_rosters(tmp_path: Path) -> None:
    empty_path = tmp_path / "empty_model_name.json"
    empty_path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_model_names(empty_path)

    newline_path = tmp_path / "newline_model_name.json"
    newline_path.write_text("kimi-k2.6\nqwen3.5-baidu\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON array"):
        load_model_names(newline_path)

    object_path = tmp_path / "object_model_name.json"
    object_path.write_text('{"proposal_model": "friday-proposal"}', encoding="utf-8")
    with pytest.raises(ValueError, match="JSON array"):
        load_model_names(object_path)

    duplicate_path = tmp_path / "duplicate_model_name.json"
    duplicate_path.write_text('["qwen3.5-baidu", "qwen3.5-baidu"]', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate model name"):
        load_model_names(duplicate_path)


def test_manifest_resume_compatibility_allows_legacy_selection_roster_metadata() -> None:
    existing = {
        "execution_mode": "live",
        "stop_after": "selection",
        "model_roster": ["kimi-k2.6", "glm-5.1"],
        "model_count": 2,
        "live_client": {
            "provider": "friday",
            "response_format": {"type": "json_object"},
            "max_retries": 6,
            "retry_backoff_seconds": 15.0,
        },
    }
    requested = {
        "execution_mode": "live",
        "stop_after": "selection",
        "model_roster": ["qwen3.5-baidu", "glm-5.1"],
        "model_count": 2,
        "live_client": {
            "provider": "friday",
            "response_format": {"type": "json_object"},
            "max_retries": 6,
            "retry_backoff_seconds": 15.0,
            "min_interval_seconds": 0.0,
        },
    }

    assert manifests_match_for_resume(existing, requested)


def test_manifest_resume_compatibility_allows_company_worker_rescheduling() -> None:
    existing = {
        "experiment_id": "demo",
        "execution_mode": "live",
        "stop_after": "baselines",
        "company_workers": 1,
        "live_client": {"provider": "friday", "min_interval_seconds": 0.0},
    }
    requested = {
        "experiment_id": "demo",
        "execution_mode": "live",
        "stop_after": "baselines",
        "company_workers": 4,
        "live_client": {"provider": "friday", "min_interval_seconds": 0.0},
    }

    assert manifests_match_for_resume(existing, requested)


def test_manifest_resume_compatibility_allows_stop_after_extension() -> None:
    existing_run = {
        "run_id": "demo__001",
        "execution_mode": "live",
        "stop_after": "baselines",
        "model_roster": ["glm-5.1"],
        "model_count": 1,
        "live_client": {"provider": "friday", "min_interval_seconds": 1.0},
    }
    requested_run = {
        "run_id": "demo__001",
        "execution_mode": "live",
        "model_roster": ["glm-5.1"],
        "model_count": 1,
        "live_client": {"provider": "friday", "min_interval_seconds": 1.0},
    }
    existing_experiment = {
        "experiment_id": "demo",
        "execution_mode": "live",
        "stop_after": "baselines",
        "company_workers": 4,
        "model_roster": ["glm-5.1"],
        "model_count": 1,
        "live_client": {"provider": "friday", "min_interval_seconds": 1.0},
    }
    requested_experiment = {
        "experiment_id": "demo",
        "execution_mode": "live",
        "stop_after": "evaluation",
        "company_workers": 4,
        "model_roster": ["glm-5.1"],
        "model_count": 1,
        "live_client": {"provider": "friday", "min_interval_seconds": 1.0},
    }

    assert manifests_match_for_resume(existing_run, requested_run)
    assert manifests_match_for_resume(existing_experiment, requested_experiment)


def test_manifest_resume_compatibility_rejects_stop_after_contraction() -> None:
    existing = {
        "execution_mode": "live",
        "stop_after": "audit",
        "live_client": {"provider": "friday", "min_interval_seconds": 1.0},
    }
    requested = {
        "execution_mode": "live",
        "stop_after": "baselines",
        "live_client": {"provider": "friday", "min_interval_seconds": 1.0},
    }

    assert not manifests_match_for_resume(existing, requested)


def test_manifest_resume_compatibility_preserves_evaluation_roster_and_throttle_checks() -> None:
    existing = {
        "execution_mode": "live",
        "stop_after": "evaluation",
        "model_roster": ["kimi-k2.6"],
        "model_count": 1,
        "live_client": {"provider": "friday"},
    }
    requested = {
        "execution_mode": "live",
        "stop_after": "evaluation",
        "model_roster": ["qwen3.5-baidu"],
        "model_count": 1,
        "live_client": {"provider": "friday", "min_interval_seconds": 0.0},
    }
    throttled = {
        "execution_mode": "live",
        "stop_after": "selection",
        "model_roster": ["kimi-k2.6"],
        "model_count": 1,
        "live_client": {"provider": "friday", "min_interval_seconds": 65.0},
    }

    assert not manifests_match_for_resume(existing, requested)
    assert not manifests_match_for_resume(existing | {"stop_after": "selection"}, throttled)
