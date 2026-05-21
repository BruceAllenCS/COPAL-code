from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from copal.io import append_jsonl, ensure_directory, read_jsonl, write_json, write_jsonl
from copal.live_validation import LiveSchemaError, complete_live_json_object
from copal.llm import LLMJsonError, LLMProviderError, build_live_client, live_client_runtime_metadata
from copal.prompts import build_response_judge_messages
from copal.stages.response_judgment import (
    RESPONSE_JUDGMENT_REQUIRED_FIELDS,
    _normalize_response_judgment,
    _validate_response_judgment_payload,
)


DEFAULT_SOURCE_EXPERIMENT_IDS = ",".join(
    f"table3_model_eval_30c_10model_seed12_20260514_shard{index}" for index in range(3)
)
DEFAULT_EXCLUDED_RESPONSE_MODELS = "gemini-3-flash-preview"
DEFAULT_JUDGE_MODELS = "gpt-5.5,aws.claude-opus-4.7"


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_worker_overrides(value: str) -> dict[str, int]:
    overrides: dict[str, int] = {}
    for part in parse_csv(value):
        if "=" not in part:
            raise ValueError(
                "--judge-worker-overrides entries must use model=workers, "
                f"got {part!r}"
            )
        model, worker_text = part.split("=", 1)
        model = model.strip()
        workers = int(worker_text.strip())
        if not model:
            raise ValueError("--judge-worker-overrides contains an empty model name")
        if workers < 1:
            raise ValueError("--judge-worker-overrides workers must be positive")
        overrides[model] = workers
    return overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a Table3 judge-family sensitivity audit. The script samples composed cases, "
            "keeps every selected downstream model output for each sampled case, and rejudges "
            "the responses with alternative judge families."
        )
    )
    parser.add_argument("--experiment-id", default="table3_judge_sensitivity_300case_9model_20260520")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--source-experiment-ids", default=DEFAULT_SOURCE_EXPERIMENT_IDS)
    parser.add_argument("--sample-cases-per-company", type=int, default=10)
    parser.add_argument("--sample-seed", type=int, default=12)
    parser.add_argument("--exclude-response-models", default=DEFAULT_EXCLUDED_RESPONSE_MODELS)
    parser.add_argument("--judge-models", default=DEFAULT_JUDGE_MODELS)
    parser.add_argument("--live-max-workers", type=int, default=24)
    parser.add_argument(
        "--judge-worker-overrides",
        default="",
        help="Optional comma-separated per-judge worker overrides, e.g. model_a=64,model_b=160.",
    )
    parser.add_argument("--max-judgments", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.sample_cases_per_company < 1:
        raise ValueError("--sample-cases-per-company must be positive")
    if args.live_max_workers < 1:
        raise ValueError("--live-max-workers must be positive")
    if args.max_judgments < 0:
        raise ValueError("--max-judgments must be non-negative")

    output_dir = ensure_directory(args.runs_dir / "experiments" / args.experiment_id)
    cache_dir = ensure_directory(output_dir / "llm_cache")
    judge_models = parse_csv(args.judge_models)
    worker_overrides = parse_worker_overrides(args.judge_worker_overrides)
    excluded_models = set(parse_csv(args.exclude_response_models))
    source_dirs = [
        args.runs_dir / "experiments" / experiment_id
        for experiment_id in parse_csv(args.source_experiment_ids)
    ]

    judge_inputs = collect_table3_judge_inputs(source_dirs=source_dirs, excluded_models=excluded_models)
    sampled_inputs = sample_inputs_by_company(
        judge_inputs=judge_inputs,
        cases_per_company=args.sample_cases_per_company,
        seed=args.sample_seed,
    )
    if args.max_judgments:
        sampled_inputs = sampled_inputs[: args.max_judgments]
    write_jsonl(output_dir / "sampled_response_judge_inputs.jsonl", sampled_inputs)

    manifest = {
        "experiment_id": args.experiment_id,
        "source_experiment_ids": parse_csv(args.source_experiment_ids),
        "excluded_response_models": sorted(excluded_models),
        "judge_models": judge_models,
        "sample_cases_per_company": args.sample_cases_per_company,
        "sample_seed": args.sample_seed,
        "sampled_case_count": len({str(row["item_id"]) for row in sampled_inputs}),
        "sampled_judgment_count_per_judge": len(sampled_inputs),
        "sampled_response_models": sorted({str(row["response_model"]) for row in sampled_inputs}),
        "live_max_workers": args.live_max_workers,
        "judge_worker_overrides": worker_overrides,
        "live_client": live_client_runtime_metadata(),
    }
    write_json(output_dir / "judge_sensitivity_manifest.json", manifest)

    def run_one_judge(judge_model: str) -> None:
        judge_workers = worker_overrides.get(judge_model, args.live_max_workers)
        client = build_live_client(cache_dir=cache_dir / safe_name(judge_model) / "json")
        run_rejudge_for_model(
            output_dir=output_dir,
            judge_model=judge_model,
            judge_inputs=sampled_inputs,
            client=client,
            live_max_workers=judge_workers,
        )

    with ThreadPoolExecutor(max_workers=len(judge_models)) as executor:
        futures = [executor.submit(run_one_judge, judge_model) for judge_model in judge_models]
        for future in futures:
            future.result()

    summary = summarize_judge_sensitivity(
        output_dir=output_dir,
        sampled_inputs=sampled_inputs,
        judge_models=judge_models,
    )
    write_json(output_dir / "judge_sensitivity_summary.json", summary)
    return 0


def collect_table3_judge_inputs(*, source_dirs: list[Path], excluded_models: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_response_ids: set[str] = set()
    for source_dir in source_dirs:
        company_runs_dir = source_dir / "company_runs"
        if not company_runs_dir.exists():
            raise FileNotFoundError(f"Missing Table3 company_runs directory: {company_runs_dir}")
        for path in sorted(company_runs_dir.glob("*/evaluation/response_judge_inputs.jsonl")):
            for row in read_jsonl(path):
                response_id = str(row["response_id"])
                response_model = str(row["response_model"])
                if response_model in excluded_models:
                    continue
                if response_id in seen_response_ids:
                    continue
                benchmark_item = dict(row["benchmark_item"])
                rows.append(
                    {
                        "source_response_judge_inputs_path": str(path),
                        "response_id": response_id,
                        "item_id": str(row["item_id"]),
                        "response_model": response_model,
                        "benchmark_item": benchmark_item,
                        "response_text": str(row["response_text"]),
                    }
                )
                seen_response_ids.add(response_id)
    if not rows:
        raise ValueError("No Table3 response judge inputs were collected")
    return rows


def sample_inputs_by_company(
    *,
    judge_inputs: list[dict[str, Any]],
    cases_per_company: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows_by_case: dict[str, list[dict[str, Any]]] = {}
    company_by_case: dict[str, str] = {}
    for row in judge_inputs:
        item_id = str(row["item_id"])
        rows_by_case.setdefault(item_id, []).append(row)
        benchmark_item = row["benchmark_item"]
        company_by_case[item_id] = str(benchmark_item.get("company_key", ""))

    expected_models = sorted({str(row["response_model"]) for row in judge_inputs})
    complete_cases_by_company: dict[str, list[str]] = {}
    for item_id, rows in rows_by_case.items():
        models = sorted(str(row["response_model"]) for row in rows)
        if models != expected_models:
            continue
        company = company_by_case[item_id]
        complete_cases_by_company.setdefault(company, []).append(item_id)

    rng = random.Random(seed)
    sampled_case_ids: list[str] = []
    for company, case_ids in sorted(complete_cases_by_company.items()):
        if len(case_ids) < cases_per_company:
            raise ValueError(
                f"Company {company} has only {len(case_ids)} complete cases; "
                f"need {cases_per_company}"
            )
        sampled_case_ids.extend(sorted(rng.sample(sorted(case_ids), cases_per_company)))

    sampled_rows: list[dict[str, Any]] = []
    for item_id in sampled_case_ids:
        sampled_rows.extend(sorted(rows_by_case[item_id], key=lambda row: str(row["response_model"])))
    return sampled_rows


def run_rejudge_for_model(
    *,
    output_dir: Path,
    judge_model: str,
    judge_inputs: list[dict[str, Any]],
    client: object,
    live_max_workers: int,
) -> None:
    judge_key = safe_name(judge_model)
    judge_dir = ensure_directory(output_dir / f"judge_{judge_key}")
    judgments_path = judge_dir / "response_judgments.jsonl"
    provider_blocks_path = judge_dir / "judge_provider_blocks.jsonl"
    unresolved_errors_path = judge_dir / "judge_unresolved_errors.jsonl"
    existing_rows = read_jsonl(judgments_path) if judgments_path.exists() else []
    existing_by_id = {str(row["response_id"]): row for row in existing_rows}
    existing_blocks = read_jsonl(provider_blocks_path) if provider_blocks_path.exists() else []
    blocked_by_id = {str(row["response_id"]): row for row in existing_blocks}
    existing_unresolved_errors = (
        read_jsonl(unresolved_errors_path) if unresolved_errors_path.exists() else []
    )
    unresolved_by_id = {str(row["response_id"]): row for row in existing_unresolved_errors}
    queue = [
        row
        for row in judge_inputs
        if str(row["response_id"]) not in existing_by_id
        and str(row["response_id"]) not in blocked_by_id
        and str(row["response_id"]) not in unresolved_by_id
    ]

    def build_judgment(row: dict[str, Any]) -> dict[str, Any]:
        benchmark_item = dict(row["benchmark_item"])
        prompt_item = blind_benchmark_item_for_judge(benchmark_item)
        try:
            payload = complete_live_json_object(
                client=client,
                model=judge_model,
                messages=build_response_judge_messages(
                    benchmark_item=prompt_item,
                    response_text=str(row["response_text"]),
                ),
                stage_dir=judge_dir,
                stage_name="judge_sensitivity_rejudgment",
                target_id=f"{row['response_id']}::{judge_key}",
                required_fields=RESPONSE_JUDGMENT_REQUIRED_FIELDS,
                validator=lambda payload: _validate_response_judgment_payload(
                    benchmark_item=benchmark_item,
                    payload=payload,
                ),
            )
        except LLMProviderError as exc:
            if not is_judge_provider_block(exc):
                raise
            return {
                "response_id": row["response_id"],
                "item_id": row["item_id"],
                "response_model": row["response_model"],
                "response_judge_model": judge_model,
                "judge_provider_blocked": True,
                "provider_error_type": type(exc).__name__,
                "provider_error_message": str(exc),
                "status_code": getattr(exc, "status_code", None),
            }
        except (LLMJsonError, LiveSchemaError) as exc:
            return {
                "response_id": row["response_id"],
                "item_id": row["item_id"],
                "response_model": row["response_model"],
                "response_judge_model": judge_model,
                "judge_unresolved_error": True,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "raw_response": getattr(exc, "response_text", ""),
            }
        normalized = _normalize_response_judgment(
            benchmark_item=benchmark_item,
            payload=dict(payload),
            response_judge_model=judge_model,
        )
        normalized["judge_sensitivity_blinded"] = True
        return normalized

    pending: set[object] = set()
    with ThreadPoolExecutor(max_workers=min(live_max_workers, max(1, len(queue)))) as executor:
        for row in queue:
            pending.add(executor.submit(build_judgment, row))
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                judgment = future.result()
                response_id = str(judgment["response_id"])
                if judgment.get("judge_provider_blocked"):
                    if response_id not in blocked_by_id:
                        append_jsonl(provider_blocks_path, judgment)
                        blocked_by_id[response_id] = judgment
                elif judgment.get("judge_unresolved_error"):
                    if response_id not in unresolved_by_id:
                        append_jsonl(unresolved_errors_path, judgment)
                        unresolved_by_id[response_id] = judgment
                elif response_id not in existing_by_id:
                    append_jsonl(judgments_path, judgment)
                    existing_by_id[response_id] = judgment

    ordered = [existing_by_id[str(row["response_id"])] for row in judge_inputs if str(row["response_id"]) in existing_by_id]
    write_jsonl(judgments_path, ordered)
    write_json(
        judge_dir / "judge_summary.json",
        {
            **summarize_judgments_for_model(judge_model=judge_model, rows=ordered),
            "provider_block_count": len(blocked_by_id),
            "provider_blocks_path": str(provider_blocks_path) if blocked_by_id else "",
            "unresolved_error_count": len(unresolved_by_id),
            "unresolved_errors_path": str(unresolved_errors_path) if unresolved_by_id else "",
        },
    )


def is_judge_provider_block(exc: LLMProviderError) -> bool:
    message = str(exc).lower()
    return (
        getattr(exc, "status_code", None) == 400
        and (
            "cyber_policy" in message
            or "safety check" in message
            or "safety filter" in message
            or "content_filter" in message
            or "content management policy" in message
        )
    )


def blind_benchmark_item_for_judge(benchmark_item: dict[str, Any]) -> dict[str, Any]:
    blinded = dict(benchmark_item)
    blinded.pop("response_id", None)
    blinded.pop("response_model", None)
    return blinded


def summarize_judge_sensitivity(
    *,
    output_dir: Path,
    sampled_inputs: list[dict[str, Any]],
    judge_models: list[str],
) -> dict[str, Any]:
    sampled_ids = {str(row["response_id"]) for row in sampled_inputs}
    gemini_rows = load_gemini_reference_rows(sampled_inputs=sampled_inputs)
    rows_by_judge: dict[str, dict[str, dict[str, Any]]] = {
        "gemini-3-flash-preview": {str(row["response_id"]): row for row in gemini_rows}
    }
    summaries = [
        summarize_judgments_for_model(judge_model="gemini-3-flash-preview", rows=gemini_rows)
    ]
    for judge_model in judge_models:
        rows = read_jsonl(output_dir / f"judge_{safe_name(judge_model)}" / "response_judgments.jsonl")
        rows = [row for row in rows if str(row["response_id"]) in sampled_ids]
        rows_by_judge[judge_model] = {str(row["response_id"]): row for row in rows}
        summaries.append(summarize_judgments_for_model(judge_model=judge_model, rows=rows))

    reference_by_model = summaries[0]["by_response_model"]
    sensitivity_rows = []
    for summary in summaries[1:]:
        compared = compare_rankings(
            reference_rates={model: values["error_rate"] for model, values in reference_by_model.items()},
            candidate_rates={
                model: values["error_rate"]
                for model, values in summary["by_response_model"].items()
            },
        )
        sensitivity_rows.append(
            {
                "judge_model": summary["judge_model"],
                "judgment_count": summary["judgment_count"],
                **compared,
            }
        )

    aggregate_summaries: list[dict[str, Any]] = []
    if len(judge_models) == 2:
        any_error_rows = build_two_judge_any_error_rows(
            output_dir=output_dir,
            judge_models=judge_models,
            sampled_inputs=sampled_inputs,
        )
        any_error_summary = summarize_judgments_for_model(judge_model="gpt_claude_any_error", rows=any_error_rows)
        aggregate_summaries.append(any_error_summary)
        sensitivity_rows.append(
            {
                "judge_model": "gpt_claude_any_error",
                "judgment_count": any_error_summary["judgment_count"],
                **compare_rankings(
                    reference_rates={model: values["error_rate"] for model, values in reference_by_model.items()},
                    candidate_rates={
                        model: values["error_rate"]
                        for model, values in any_error_summary["by_response_model"].items()
                    },
                ),
            }
        )
        write_jsonl(output_dir / "gpt_claude_any_error_response_judgments.jsonl", any_error_rows)
        write_json(output_dir / "gpt_claude_any_error_summary.json", any_error_summary)
        write_jsonl(output_dir / "consensus_response_judgments.jsonl", any_error_rows)
        write_json(
            output_dir / "consensus_summary.json",
            {
                **any_error_summary,
                "deprecated_alias_for": "gpt_claude_any_error",
            },
        )

    all_judge_models = ["gemini-3-flash-preview", *judge_models]
    majority_judge_name = (
        "three_judge_majority_error"
        if len(all_judge_models) == 3
        else f"{len(all_judge_models)}_judge_majority_error"
    )
    majority_rows = build_judge_majority_rows(
        sampled_inputs=sampled_inputs,
        judge_rows_by_id=rows_by_judge,
        judge_models=all_judge_models,
        majority_judge_name=majority_judge_name,
    )
    majority_summary = summarize_judgments_for_model(judge_model=majority_judge_name, rows=majority_rows)
    aggregate_summaries.append(majority_summary)
    sensitivity_rows.append(
        {
            "judge_model": majority_judge_name,
            "judgment_count": majority_summary["judgment_count"],
            **compare_rankings(
                reference_rates={model: values["error_rate"] for model, values in reference_by_model.items()},
                candidate_rates={
                    model: values["error_rate"]
                    for model, values in majority_summary["by_response_model"].items()
                },
            ),
        }
    )

    agreement_summary = summarize_pairwise_judge_agreement(
        sampled_inputs=sampled_inputs,
        judge_rows_by_id=rows_by_judge,
        judge_models=all_judge_models,
    )
    vote_distribution = summarize_judge_vote_distribution(
        sampled_inputs=sampled_inputs,
        judge_rows_by_id=rows_by_judge,
        judge_models=all_judge_models,
    )

    write_jsonl(output_dir / f"{majority_judge_name}_response_judgments.jsonl", majority_rows)
    write_json(output_dir / f"{majority_judge_name}_summary.json", majority_summary)
    if majority_judge_name == "three_judge_majority_error":
        write_jsonl(output_dir / "three_judge_majority_response_judgments.jsonl", majority_rows)
        write_json(output_dir / "three_judge_majority_summary.json", majority_summary)
    write_json(output_dir / "pairwise_judge_agreement.json", agreement_summary)
    write_json(output_dir / "judge_vote_distribution.json", vote_distribution)

    return {
        "sampled_case_count": len({str(row["item_id"]) for row in sampled_inputs}),
        "sampled_judgment_count_per_judge": len(sampled_inputs),
        "response_models": sorted({str(row["response_model"]) for row in sampled_inputs}),
        "judge_summaries": summaries + aggregate_summaries,
        "rank_sensitivity": sensitivity_rows,
        "pairwise_judge_agreement": agreement_summary,
        "judge_vote_distribution": vote_distribution,
    }


def load_gemini_reference_rows(*, sampled_inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_response_id: dict[str, dict[str, Any]] = {}
    needed = {str(row["response_id"]) for row in sampled_inputs}
    paths = sorted({Path(str(row["source_response_judge_inputs_path"])).with_name("response_judgments.jsonl") for row in sampled_inputs})
    for path in paths:
        for row in read_jsonl(path):
            response_id = str(row.get("response_id", ""))
            if response_id in needed:
                rows_by_response_id[response_id] = row
    missing = sorted(needed - set(rows_by_response_id))
    if missing:
        raise ValueError(f"Missing Gemini reference judgments for {len(missing)} sampled responses")
    return [rows_by_response_id[str(row["response_id"])] for row in sampled_inputs]


def build_two_judge_any_error_rows(
    *,
    output_dir: Path,
    judge_models: list[str],
    sampled_inputs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(judge_models) != 2:
        return []
    left_rows = {
        str(row["response_id"]): row
        for row in read_jsonl(output_dir / f"judge_{safe_name(judge_models[0])}" / "response_judgments.jsonl")
    }
    right_rows = {
        str(row["response_id"]): row
        for row in read_jsonl(output_dir / f"judge_{safe_name(judge_models[1])}" / "response_judgments.jsonl")
    }
    consensus_rows: list[dict[str, Any]] = []
    for sample in sampled_inputs:
        response_id = str(sample["response_id"])
        if response_id not in left_rows or response_id not in right_rows:
            continue
        left = left_rows[response_id]
        right = right_rows[response_id]
        consensus = dict(left)
        consensus["response_judge_model"] = "gpt_claude_any_error"
        consensus["judge_agreement"] = bool(left["overall_correct"]) == bool(right["overall_correct"])
        consensus["overall_correct"] = bool(left["overall_correct"]) and bool(right["overall_correct"])
        consensus["under_enforcement"] = bool(left["under_enforcement"]) or bool(right["under_enforcement"])
        consensus["over_enforcement"] = bool(left["over_enforcement"]) or bool(right["over_enforcement"])
        consensus_rows.append(consensus)
    return consensus_rows


def build_judge_majority_rows(
    *,
    sampled_inputs: list[dict[str, Any]],
    judge_rows_by_id: dict[str, dict[str, dict[str, Any]]],
    judge_models: list[str],
    majority_judge_name: str,
) -> list[dict[str, Any]]:
    required_error_votes = len(judge_models) // 2 + 1
    majority_rows: list[dict[str, Any]] = []
    for sample in sampled_inputs:
        response_id = str(sample["response_id"])
        if any(response_id not in judge_rows_by_id.get(judge_model, {}) for judge_model in judge_models):
            continue
        rows = [judge_rows_by_id[judge_model][response_id] for judge_model in judge_models]
        error_votes = {
            judge_model: not bool(row["overall_correct"])
            for judge_model, row in zip(judge_models, rows, strict=True)
        }
        error_vote_count = sum(1 for value in error_votes.values() if value)
        majority = dict(rows[0])
        majority["response_judge_model"] = majority_judge_name
        majority["judge_error_votes"] = error_votes
        majority["judge_error_vote_count"] = error_vote_count
        majority["judge_error_vote_threshold"] = required_error_votes
        majority["overall_correct"] = error_vote_count < required_error_votes
        majority["under_enforcement"] = (
            sum(1 for row in rows if bool(row["under_enforcement"])) >= required_error_votes
        )
        majority["over_enforcement"] = (
            sum(1 for row in rows if bool(row["over_enforcement"])) >= required_error_votes
        )
        majority_rows.append(majority)
    return majority_rows


def summarize_pairwise_judge_agreement(
    *,
    sampled_inputs: list[dict[str, Any]],
    judge_rows_by_id: dict[str, dict[str, dict[str, Any]]],
    judge_models: list[str],
) -> list[dict[str, Any]]:
    response_ids = [str(row["response_id"]) for row in sampled_inputs]
    rows: list[dict[str, Any]] = []
    for left_index, left_model in enumerate(judge_models):
        for right_model in judge_models[left_index + 1 :]:
            common_ids = [
                response_id
                for response_id in response_ids
                if response_id in judge_rows_by_id.get(left_model, {})
                and response_id in judge_rows_by_id.get(right_model, {})
            ]
            if not common_ids:
                continue
            pair_rows = [
                (
                    judge_rows_by_id[left_model][response_id],
                    judge_rows_by_id[right_model][response_id],
                )
                for response_id in common_ids
            ]
            agreement_count = sum(
                bool(left_row["overall_correct"]) == bool(right_row["overall_correct"])
                for left_row, right_row in pair_rows
            )
            confusion = pairwise_confusion(pair_rows=pair_rows)
            rows.append(
                {
                    "left_judge_model": left_model,
                    "right_judge_model": right_model,
                    "common_judgment_count": len(common_ids),
                    "agreement_count": agreement_count,
                    "agreement_rate": agreement_count / len(common_ids),
                    **confusion,
                    "by_response_model": summarize_pairwise_agreement_by_key(
                        pair_rows=pair_rows,
                        key_fn=lambda row: str(row["response_model"]),
                    ),
                    "by_signature": summarize_pairwise_agreement_by_key(
                        pair_rows=pair_rows,
                        key_fn=lambda row: str(row.get("signature", "")),
                    ),
                    "by_target_facet": summarize_pairwise_agreement_by_key(
                        pair_rows=pair_rows,
                        key_fn=lambda row: str(row.get("target_facet", "")),
                    ),
                }
            )
    return rows


def pairwise_confusion(*, pair_rows: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    both_correct = 0
    left_error_only = 0
    right_error_only = 0
    both_error = 0
    for left_row, right_row in pair_rows:
        left_error = not bool(left_row["overall_correct"])
        right_error = not bool(right_row["overall_correct"])
        if left_error and right_error:
            both_error += 1
        elif left_error:
            left_error_only += 1
        elif right_error:
            right_error_only += 1
        else:
            both_correct += 1
    total = len(pair_rows)
    return {
        "both_correct_count": both_correct,
        "left_error_only_count": left_error_only,
        "right_error_only_count": right_error_only,
        "both_error_count": both_error,
        "left_error_rate": (left_error_only + both_error) / total if total else 0.0,
        "right_error_rate": (right_error_only + both_error) / total if total else 0.0,
        "left_minus_right_error_rate_delta": (left_error_only - right_error_only) / total if total else 0.0,
    }


def summarize_pairwise_agreement_by_key(
    *,
    pair_rows: list[tuple[dict[str, Any], dict[str, Any]]],
    key_fn: object,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for left_row, right_row in pair_rows:
        key = str(key_fn(left_row))
        grouped.setdefault(key, []).append((left_row, right_row))
    summary: dict[str, dict[str, Any]] = {}
    for key, rows in sorted(grouped.items()):
        agreement_count = sum(
            bool(left_row["overall_correct"]) == bool(right_row["overall_correct"])
            for left_row, right_row in rows
        )
        summary[key] = {
            "common_judgment_count": len(rows),
            "agreement_count": agreement_count,
            "agreement_rate": agreement_count / len(rows),
            **pairwise_confusion(pair_rows=rows),
        }
    return summary


def summarize_judge_vote_distribution(
    *,
    sampled_inputs: list[dict[str, Any]],
    judge_rows_by_id: dict[str, dict[str, dict[str, Any]]],
    judge_models: list[str],
) -> dict[str, Any]:
    vote_counts: dict[int, int] = {}
    by_response_model: dict[str, dict[int, int]] = {}
    by_signature: dict[str, dict[int, int]] = {}
    by_target_facet: dict[str, dict[int, int]] = {}
    complete_count = 0
    for sample in sampled_inputs:
        response_id = str(sample["response_id"])
        if any(response_id not in judge_rows_by_id.get(judge_model, {}) for judge_model in judge_models):
            continue
        rows = [judge_rows_by_id[judge_model][response_id] for judge_model in judge_models]
        error_vote_count = sum(1 for row in rows if not bool(row["overall_correct"]))
        vote_counts[error_vote_count] = vote_counts.get(error_vote_count, 0) + 1
        response_model = str(sample["response_model"])
        signature = str(sample["benchmark_item"].get("signature", ""))
        target_facet = str(sample["benchmark_item"].get("target_facet", ""))
        for group, key in (
            (by_response_model, response_model),
            (by_signature, signature),
            (by_target_facet, target_facet),
        ):
            group.setdefault(key, {})
            group[key][error_vote_count] = group[key].get(error_vote_count, 0) + 1
        complete_count += 1
    return {
        "judge_models": judge_models,
        "complete_judgment_count": complete_count,
        "error_vote_count_distribution": stringify_int_key_counts(vote_counts),
        "by_response_model": {
            key: stringify_int_key_counts(value)
            for key, value in sorted(by_response_model.items())
        },
        "by_signature": {
            key: stringify_int_key_counts(value)
            for key, value in sorted(by_signature.items())
        },
        "by_target_facet": {
            key: stringify_int_key_counts(value)
            for key, value in sorted(by_target_facet.items())
        },
    }


def stringify_int_key_counts(counts: dict[int, int]) -> dict[str, int]:
    return {str(key): counts[key] for key in sorted(counts)}


def summarize_judgments_for_model(*, judge_model: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(str(row["response_model"]), []).append(row)
    model_summary = {
        model: {
            "judgment_count": len(model_rows),
            "error_count": sum(1 for row in model_rows if not bool(row["overall_correct"])),
            "error_rate": sum(1 for row in model_rows if not bool(row["overall_correct"])) / len(model_rows),
        }
        for model, model_rows in sorted(by_model.items())
    }
    ranked_models = [
        model
        for model, _values in sorted(
            model_summary.items(),
            key=lambda item: (-float(item[1]["error_rate"]), item[0]),
        )
    ]
    return {
        "judge_model": judge_model,
        "judgment_count": len(rows),
        "case_count": len({str(row["item_id"]) for row in rows}),
        "overall_error_rate": sum(1 for row in rows if not bool(row["overall_correct"])) / len(rows) if rows else 0.0,
        "by_response_model": model_summary,
        "ranked_models_high_to_low_error": ranked_models,
    }


def compare_rankings(*, reference_rates: dict[str, float], candidate_rates: dict[str, float]) -> dict[str, Any]:
    models = sorted(set(reference_rates) & set(candidate_rates))
    if not models:
        raise ValueError("No overlapping models for ranking comparison")
    ref_ranks = ranks_high_to_low({model: reference_rates[model] for model in models})
    cand_ranks = ranks_high_to_low({model: candidate_rates[model] for model in models})
    rank_shifts = {model: abs(ref_ranks[model] - cand_ranks[model]) for model in models}
    deltas = {model: candidate_rates[model] - reference_rates[model] for model in models}
    return {
        "model_count": len(models),
        "kendall_tau": kendall_tau_from_ranks(ref_ranks, cand_ranks, models),
        "spearman_rho": spearman_rho_from_ranks(ref_ranks, cand_ranks, models),
        "max_rank_shift": max(rank_shifts.values()),
        "mean_abs_error_rate_delta": sum(abs(value) for value in deltas.values()) / len(deltas),
        "rank_shifts": rank_shifts,
        "error_rate_deltas": deltas,
    }


def ranks_high_to_low(values: dict[str, float]) -> dict[str, int]:
    return {
        model: rank
        for rank, model in enumerate(
            sorted(values, key=lambda model: (-values[model], model)),
            start=1,
        )
    }


def kendall_tau_from_ranks(left: dict[str, int], right: dict[str, int], models: list[str]) -> float:
    concordant = 0
    discordant = 0
    for index, first in enumerate(models):
        for second in models[index + 1 :]:
            left_order = left[first] - left[second]
            right_order = right[first] - right[second]
            product = left_order * right_order
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else 0.0


def spearman_rho_from_ranks(left: dict[str, int], right: dict[str, int], models: list[str]) -> float:
    n = len(models)
    if n < 2:
        return 0.0
    squared_diffs = sum((left[model] - right[model]) ** 2 for model in models)
    return 1.0 - (6.0 * squared_diffs) / (n * (n * n - 1))


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value)


if __name__ == "__main__":
    raise SystemExit(main())
