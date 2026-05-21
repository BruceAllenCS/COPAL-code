from __future__ import annotations

import hashlib
from pathlib import Path

from copal.io import read_json, read_jsonl


DATASET_DIR = Path("datasets/copal-paper-v1")
ARTIFACT_DIR = DATASET_DIR / "artifacts"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_paper_artifact_manifest_covers_required_families() -> None:
    manifest = read_json(ARTIFACT_DIR / "manifest.json")
    dataset_manifest = read_json(DATASET_DIR / "manifest.json")

    required = manifest["required_artifact_families"]
    assert set(required) == {
        "30_company_world_specs",
        "policy_inventories",
        "grounded_clauses",
        "composition_records",
        "generated_candidate_queries",
        "screening_mapping_logs",
        "final_selected_suites",
        "handling_contracts",
        "reconstructed_chatbot_prompts",
        "construction_and_judge_prompt_templates",
        "model_outputs",
        "automatic_judge_labels",
        "ablation_candidate_pools",
        "validation_records",
        "run_manifests",
    }
    assert manifest["source_root"] == "<COPAL_WORKSPACE_ROOT>"
    assert dataset_manifest["paper_artifacts"]["manifest_path"] == "artifacts/manifest.json"
    assert dataset_manifest["paper_artifacts"]["counts"] == manifest["counts"]


def test_paper_artifact_counts_match_files() -> None:
    manifest = read_json(ARTIFACT_DIR / "manifest.json")
    counts = manifest["counts"]

    assert counts["company_world_specs"] == 30
    assert counts["policy_inventories"] == 30
    assert counts["grounded_clauses"] == 480
    assert counts["composition_records"] == 232
    assert counts["generated_candidate_queries"] == 3827
    assert counts["screening_mapping_logs"] == 4343
    assert counts["final_selected_suite_items"] == 2340
    assert counts["handling_contracts"] == 2340
    assert counts["reconstructed_chatbot_prompts"] == 30
    assert counts["model_outputs"] == 9000
    assert counts["automatic_judge_labels"] == 9000
    assert counts["ablation_candidate_pool_items"] == 3826
    assert counts["validation_record_files"] == 18
    assert counts["run_manifest_files"] == 17

    jsonl_count_map = {
        "company_world_specs": "company_world_specs.jsonl",
        "policy_inventories": "policy_inventories.jsonl",
        "grounded_clauses": "grounded_clauses.jsonl",
        "composition_records": "composition_records.jsonl",
        "generated_candidate_queries": "generated_candidate_queries.jsonl",
        "screening_mapping_logs": "screening_mapping_logs.jsonl",
        "final_selected_suite_items": "final_selected_suites.jsonl",
        "handling_contracts": "handling_contracts.jsonl",
        "reconstructed_chatbot_prompts": "reconstructed_chatbot_prompts.jsonl",
        "model_outputs": "model_outputs.jsonl",
        "automatic_judge_labels": "automatic_judge_labels.jsonl",
        "ablation_candidate_pool_items": "ablation_candidate_pools.jsonl",
    }
    for count_key, filename in jsonl_count_map.items():
        assert len(read_jsonl(ARTIFACT_DIR / filename)) == counts[count_key]


def test_paper_artifact_file_hashes_match_manifest() -> None:
    manifest = read_json(ARTIFACT_DIR / "manifest.json")

    for file_record in manifest["files"]:
        path = ARTIFACT_DIR / file_record["path"]
        assert path.exists(), path
        assert path.stat().st_size == file_record["bytes"]
        assert sha256_file(path) == file_record["sha256"]


def test_paper_artifact_records_have_core_linkage_fields() -> None:
    selected_items = read_jsonl(ARTIFACT_DIR / "final_selected_suites.jsonl")
    contracts = read_jsonl(ARTIFACT_DIR / "handling_contracts.jsonl")
    outputs = read_jsonl(ARTIFACT_DIR / "model_outputs.jsonl")
    labels = read_jsonl(ARTIFACT_DIR / "automatic_judge_labels.jsonl")

    table3_items = [row for row in selected_items if row["suite_family"] == "table3_model_eval_composed"]
    table2_items = [row for row in selected_items if row["suite_family"] == "table2_ablation_variant"]
    assert len(table3_items) == 900
    assert len(table2_items) == 1440
    assert {row["variant_id"] for row in table2_items} == {
        "raw_policy_planning",
        "clause_only_planning",
        "without_facet_query_generation",
        "copal",
    }
    assert all(row["expected_handling"]["strict_response_contract"] for row in table3_items)
    assert all(row["generated_case_contract"]["expected_composed_handling"] for row in contracts)
    assert {row["response_id"] for row in outputs} == {row["response_id"] for row in labels}


def test_paper_artifacts_do_not_include_private_internal_terms() -> None:
    private_terms = ("meituan", "sankuai", "xiaozhi", "salesmind", "美团", "晓慧", "王晓慧")
    for path in ARTIFACT_DIR.rglob("*"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        lowered = text.lower()
        for term in private_terms:
            assert term.lower() not in lowered, path
