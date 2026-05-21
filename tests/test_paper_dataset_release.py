from __future__ import annotations

import hashlib
from pathlib import Path

from copal.io import read_json, read_jsonl


DATASET_DIR = Path("datasets/copal-paper-v1")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_paper_dataset_manifest_counts_and_files() -> None:
    manifest = read_json(DATASET_DIR / "manifest.json")

    assert manifest["dataset_version"] == "copal-paper-v1"
    assert manifest["counts"]["industries"] == 30
    assert manifest["counts"]["companies"] == 300
    assert manifest["counts"]["policy_worlds"] == 300
    assert manifest["counts"]["system_prompts"] == 300
    assert manifest["counts"]["policy_rules"] == 8857
    assert manifest["counts"]["allowed_policy_rules"] == 4357
    assert manifest["counts"]["prohibited_policy_rules"] == 4500

    for file_record in manifest["files"]:
        path = DATASET_DIR / file_record["path"]
        assert path.exists(), path
        assert path.stat().st_size == file_record["bytes"]
        assert sha256_file(path) == file_record["sha256"]


def test_paper_dataset_rows_match_manifest() -> None:
    manifest = read_json(DATASET_DIR / "manifest.json")

    companies = read_jsonl(DATASET_DIR / "companies" / "companies.jsonl")
    policy_worlds = read_jsonl(DATASET_DIR / "policies" / "policy_worlds.jsonl")
    policy_rules = read_jsonl(DATASET_DIR / "policies" / "policy_rules.jsonl")
    system_prompts = read_jsonl(DATASET_DIR / "prompts" / "system_prompts.jsonl")

    assert len(companies) == manifest["counts"]["companies"]
    assert len(policy_worlds) == manifest["counts"]["policy_worlds"]
    assert len(policy_rules) == manifest["counts"]["policy_rules"]
    assert len(system_prompts) == manifest["counts"]["system_prompts"]
    assert {row["company_key"] for row in companies} == {row["company_key"] for row in system_prompts}
    assert {row["rule_type"] for row in policy_rules} == {"allowed", "prohibited"}


def test_paper_dataset_prompt_templates_match_source() -> None:
    template_source = (DATASET_DIR / "prompts" / "copal_prompt_templates.py").read_text(encoding="utf-8")
    current_source = Path("copal/prompts.py").read_text(encoding="utf-8")
    template_index = read_json(DATASET_DIR / "prompts" / "copal_prompt_templates.json")

    assert template_source == current_source
    assert set(template_index["system_prompts"]) == {
        "grounding",
        "validation",
        "coverage",
        "response_judge",
    }
    assert len(template_index["message_builders"]) == 8
