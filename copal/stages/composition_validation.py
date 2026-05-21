from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from copal.config import DEFAULT_SIGNATURES, require_execution_mode
from copal.io import ensure_directory, read_jsonl, write_json, write_jsonl
from copal.llm import LLMClient
from copal.live_validation import LiveSchemaError, complete_live_json_object, require_bool, require_str, require_str_allow_empty
from copal.prompts import build_composition_adjudication_messages
from copal.stages.validation import validate_structure_constraints

SUPPORTED_SIGNATURES = set(DEFAULT_SIGNATURES)


def _normalize_composition_adjudication(
    *,
    candidate: dict[str, object],
    payload: dict[str, object],
    validator_model: str,
) -> dict[str, object]:
    pass_value = require_bool(payload["pass"], context=f"composition_validation {candidate['composition_id']}.pass")
    raw_signature = require_str_allow_empty(
        payload["signature"],
        context=f"composition_validation {candidate['composition_id']}.signature",
    )
    signature = (
        require_supported_signature(
            raw_signature,
            context=f"composition_validation {candidate['composition_id']}.signature",
        )
        if pass_value
        else ""
    )
    raw_nonseparability_slice = require_str_allow_empty(
        payload["nonseparability_slice"],
        context=f"composition_validation {candidate['composition_id']}.nonseparability_slice",
    )
    nonseparability_slice = (
        require_str(
            raw_nonseparability_slice,
            context=f"composition_validation {candidate['composition_id']}.nonseparability_slice",
        )
        if pass_value
        else raw_nonseparability_slice
    )
    return {
        "target_type": "composition",
        "target_id": str(candidate["composition_id"]),
        "composition_id": str(candidate["composition_id"]),
        "pass": pass_value,
        "signature": signature,
        "raw_signature": raw_signature,
        "feasibility_status": require_str(
            payload["feasibility_status"],
            context=f"composition_validation {candidate['composition_id']}.feasibility_status",
        ),
        "non_separability_status": require_str(
            payload["non_separability_status"],
            context=f"composition_validation {candidate['composition_id']}.non_separability_status",
        ),
        "nonseparability_slice": nonseparability_slice,
        "adjudication_rationale": require_str(
            payload["adjudication_rationale"],
            context=f"composition_validation {candidate['composition_id']}.adjudication_rationale",
        ),
        "validator_model": validator_model,
    }


def require_supported_signature(value: str, *, context: str) -> str:
    if value not in SUPPORTED_SIGNATURES:
        allowed = ", ".join(DEFAULT_SIGNATURES)
        raise LiveSchemaError(f"{context} unsupported signature: {value}. Expected one of: {allowed}")
    return value


def _signature_budget_allows(
    counts_by_signature: dict[str, int],
    signature: str,
    composition_limit_per_signature: int | None,
) -> bool:
    if composition_limit_per_signature is None:
        return True
    return counts_by_signature[signature] < composition_limit_per_signature


def _ensure_relation_pattern(row: dict[str, object], signature: str) -> dict[str, object]:
    relation_patterns = row.get("relation_patterns")
    if isinstance(relation_patterns, list | tuple):
        patterns = [str(pattern) for pattern in relation_patterns]
    else:
        patterns = []
    if signature and signature not in patterns:
        patterns.insert(0, signature)
    return {
        **row,
        "relation_pattern": str(row.get("relation_pattern") or signature),
        "relation_patterns": patterns,
    }


def _structure_signals(row: dict[str, object]) -> dict[str, object]:
    return dict(row.get("structure_signals", {}))


def _interaction_filter(row: dict[str, object], signals: dict[str, object]) -> dict[str, object]:
    return dict(signals.get("interaction_filter", row.get("interaction_filter", {})))


def _adjudication_priority_score(row: dict[str, object]) -> int:
    signals = _structure_signals(row)
    interaction_filter = _interaction_filter(row, signals)
    interaction_status = str(interaction_filter.get("status", "")).strip()
    signature = str(row.get("signature_proposal", "")).strip()

    score = 0
    if signature in SUPPORTED_SIGNATURES:
        score += 40
    if signals.get("joint_trigger_satisfiable") is True:
        score += 35
    if interaction_status == "pass":
        score += 30
    elif interaction_status == "unresolved":
        score += 12
    elif interaction_status == "fail":
        score -= 10
    if signals.get("independently_resolvable") is False:
        score += 25
    elif signals.get("independently_resolvable") is True:
        score -= 15
    if signals.get("scope_coupled") is True:
        score += 18
    if signals.get("effect_interaction") is True:
        score += 18
    if signals.get("trigger_compatible") is True:
        score += 12
    if signals.get("same_semantic_span") is True:
        score += 15
    if signals.get("scope_overlap") is True:
        score += 10
    if signals.get("changes_scope_or_handling") is True:
        score += 8
    if signals.get("recombine_loses_expected_handling") is True:
        score += 8
    if signals.get("trigger_overlap") is True:
        score += 5
    return score


def _adjudication_bucket(row: dict[str, object]) -> str:
    signature = str(row.get("signature_proposal", "")).strip()
    if signature in SUPPORTED_SIGNATURES:
        return f"signature:{signature}"
    effect_pair = row.get("effect_pair", [])
    if isinstance(effect_pair, list) and len(effect_pair) >= 2:
        return f"effect:{effect_pair[0]}->{effect_pair[1]}"
    return "effect:unknown"


def _select_adjudication_candidate_ids(
    rows: list[dict[str, object]],
    composition_adjudication_limit: int | None,
) -> set[str]:
    if composition_adjudication_limit is None or len(rows) <= composition_adjudication_limit:
        return {str(row["composition_id"]) for row in rows}
    if composition_adjudication_limit == 0:
        return set()

    ranked = sorted(
        enumerate(rows),
        key=lambda item: (
            -_adjudication_priority_score(item[1]),
            _adjudication_bucket(item[1]),
            item[0],
        ),
    )
    selected_indexes: list[int] = []
    selected_index_set: set[int] = set()
    covered_buckets: set[str] = set()

    for index, row in ranked:
        bucket = _adjudication_bucket(row)
        if bucket in covered_buckets:
            continue
        selected_indexes.append(index)
        selected_index_set.add(index)
        covered_buckets.add(bucket)
        if len(selected_indexes) >= composition_adjudication_limit:
            return {str(rows[index]["composition_id"]) for index in selected_indexes}

    for index, _row in ranked:
        if index in selected_index_set:
            continue
        selected_indexes.append(index)
        if len(selected_indexes) >= composition_adjudication_limit:
            break

    return {str(rows[index]["composition_id"]) for index in selected_indexes}


def _validate_composition_adjudication_payload(*, candidate: dict[str, object], payload: dict[str, object]) -> None:
    _normalize_composition_adjudication(candidate=candidate, payload=payload, validator_model="schema-check")


def _adjudicate_composition_live(
    *,
    compositions_dir: Path,
    candidate: dict[str, object],
    validator_client: LLMClient,
    validator_model: str,
) -> dict[str, object]:
    payload = complete_live_json_object(
        client=validator_client,
        model=validator_model,
        messages=build_composition_adjudication_messages(candidate=candidate),
        stage_dir=compositions_dir,
        stage_name="composition_validation",
        target_id=str(candidate["composition_id"]),
        required_fields=(
            "pass",
            "signature",
            "feasibility_status",
            "non_separability_status",
            "nonseparability_slice",
            "adjudication_rationale",
        ),
        validator=lambda payload: _validate_composition_adjudication_payload(
            candidate=candidate,
            payload=payload,
        ),
    )
    return _normalize_composition_adjudication(
        candidate=candidate,
        payload=dict(payload),
        validator_model=validator_model,
    )


def run_composition_validation_stage(
    *,
    compositions_dir: Path,
    execution_mode: str,
    validator_client: LLMClient | None = None,
    validator_model: str = "",
    composition_limit_per_signature: int | None = None,
    composition_adjudication_limit: int | None = None,
    live_max_workers: int = 1,
) -> dict[str, object]:
    require_execution_mode(execution_mode)
    if composition_limit_per_signature is not None and composition_limit_per_signature < 1:
        raise ValueError("composition_limit_per_signature must be positive when set")
    if composition_adjudication_limit is not None and composition_adjudication_limit < 0:
        raise ValueError("composition_adjudication_limit must be non-negative when set")
    if live_max_workers < 1:
        raise ValueError("live_max_workers must be positive")
    ensure_directory(compositions_dir)
    candidates = read_jsonl(compositions_dir / "candidate_compositions.jsonl")
    if execution_mode == "live":
        if validator_client is None or not validator_model:
            raise ValueError("Live composition validation requires validator_client and validator_model")
        return _run_parallel_live_composition_validation_stage(
            compositions_dir=compositions_dir,
            candidates=candidates,
            validator_client=validator_client,
            validator_model=validator_model,
            composition_limit_per_signature=composition_limit_per_signature,
            composition_adjudication_limit=composition_adjudication_limit,
            live_max_workers=live_max_workers,
        )
    deterministic_rows: list[dict[str, object]] = []
    adjudication_queue: list[dict[str, object]] = []
    adjudications: list[dict[str, object]] = []
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    accepted_counts_by_signature: dict[str, int] = defaultdict(int)
    budget_excluded_count = 0
    adjudication_budget_excluded_count = 0

    def signature_budget_allows(signature: str) -> bool:
        if composition_limit_per_signature is None:
            return True
        return accepted_counts_by_signature[signature] < composition_limit_per_signature

    def accept_candidate(row: dict[str, object], *, signature: str) -> None:
        accepted.append(_ensure_relation_pattern(row, signature))
        accepted_counts_by_signature[signature] += 1

    for candidate in candidates:
        validation = validate_structure_constraints(candidate)
        merged = {
            **candidate,
            **validation,
            "stage2_required": bool(validation["requires_adjudication"]),
            "feasibility_status": validation["heuristic_feasibility_result"],
            "non_separability_status": validation["heuristic_non_separability_result"],
        }
        deterministic_rows.append(merged)
        if validation["requires_adjudication"]:
            candidate_signature = str(candidate.get("signature_proposal", "")).strip()
            if candidate_signature and not signature_budget_allows(candidate_signature):
                budget_excluded_count += 1
                rejected.append(
                    {
                        **merged,
                        "budget_excluded": True,
                        "budget_excluded_reason": "composition_limit_per_signature",
                    }
                )
                continue
            if (
                composition_adjudication_limit is not None
                and len(adjudication_queue) >= composition_adjudication_limit
            ):
                adjudication_budget_excluded_count += 1
                budget_excluded_count += 1
                rejected.append(
                    {
                        **merged,
                        "budget_excluded": True,
                        "budget_excluded_reason": "composition_adjudication_limit",
                    }
                )
                continue
            adjudication_queue.append(merged)
            if execution_mode == "live":
                if validator_client is None or not validator_model:
                    raise ValueError("Live composition validation requires validator_client and validator_model")
                adjudication = _adjudicate_composition_live(
                    compositions_dir=compositions_dir,
                    candidate=merged,
                    validator_client=validator_client,
                    validator_model=validator_model,
                )
                adjudications.append(adjudication)
                if adjudication["pass"] and adjudication["signature"]:
                    signature = str(adjudication["signature"])
                    row = {
                        **merged,
                        "signature_proposal": signature,
                        "relation_pattern": signature,
                        "relation_patterns": [signature],
                        "signature_source": "rubric_adjudicated",
                        "feasibility_status": adjudication["feasibility_status"],
                        "non_separability_status": adjudication["non_separability_status"],
                        "nonseparability_slice": adjudication["nonseparability_slice"],
                        "adjudication": adjudication,
                    }
                    if signature_budget_allows(signature):
                        accept_candidate(row, signature=signature)
                    else:
                        budget_excluded_count += 1
                        rejected.append(
                            {
                                **row,
                                "budget_excluded": True,
                                "budget_excluded_reason": "composition_limit_per_signature",
                            }
                        )
                else:
                    rejected.append({**merged, "adjudication": adjudication})
        elif (
            candidate["signature_proposal"]
            and validation["heuristic_feasibility_result"] == "pass"
            and validation["heuristic_non_separability_result"] == "pass"
        ):
            signature = str(candidate["signature_proposal"])
            if signature_budget_allows(signature):
                accept_candidate(merged, signature=signature)
            else:
                budget_excluded_count += 1
                rejected.append(
                    {
                        **merged,
                        "budget_excluded": True,
                        "budget_excluded_reason": "composition_limit_per_signature",
                    }
                )
        else:
            rejected.append(merged)

    summary = {
        "candidate_count": len(candidates),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "adjudication_queue_count": len(adjudication_queue),
        "adjudication_count": len(adjudications),
        "budget_excluded_count": budget_excluded_count,
        "adjudication_budget_excluded_count": adjudication_budget_excluded_count,
        "composition_limit_per_signature": composition_limit_per_signature,
        "composition_adjudication_limit": composition_adjudication_limit,
        "execution_mode": execution_mode,
    }

    write_jsonl(compositions_dir / "composition_deterministic_results.jsonl", deterministic_rows)
    write_jsonl(compositions_dir / "composition_adjudication_queue.jsonl", adjudication_queue)
    write_jsonl(compositions_dir / "composition_adjudications.jsonl", adjudications)
    write_jsonl(compositions_dir / "accepted_compositions.jsonl", accepted)
    write_jsonl(compositions_dir / "rejected_compositions.jsonl", rejected)
    write_json(compositions_dir / "composition_validation_summary.json", summary)
    return summary


def _run_parallel_live_composition_validation_stage(
    *,
    compositions_dir: Path,
    candidates: list[dict[str, object]],
    validator_client: LLMClient,
    validator_model: str,
    composition_limit_per_signature: int | None,
    composition_adjudication_limit: int | None,
    live_max_workers: int,
) -> dict[str, object]:
    deterministic_rows: list[dict[str, object]] = []
    adjudication_queue: list[dict[str, object]] = []
    adjudications: list[dict[str, object]] = []
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    accepted_counts_by_signature: dict[str, int] = defaultdict(int)
    budget_excluded_count = 0
    adjudication_budget_excluded_count = 0
    planned_deterministic_counts_by_signature: dict[str, int] = defaultdict(int)
    pending_adjudication_rows: list[dict[str, object]] = []
    actions: list[tuple[str, dict[str, object], str]] = []

    def signature_budget_allows(signature: str) -> bool:
        return _signature_budget_allows(
            accepted_counts_by_signature,
            signature,
            composition_limit_per_signature,
        )

    def accept_candidate(row: dict[str, object], *, signature: str) -> None:
        accepted.append(_ensure_relation_pattern(row, signature))
        accepted_counts_by_signature[signature] += 1

    for candidate in candidates:
        validation = validate_structure_constraints(candidate)
        merged = {
            **candidate,
            **validation,
            "stage2_required": bool(validation["requires_adjudication"]),
            "feasibility_status": validation["heuristic_feasibility_result"],
            "non_separability_status": validation["heuristic_non_separability_result"],
        }
        deterministic_rows.append(merged)
        if validation["requires_adjudication"]:
            candidate_signature = str(candidate.get("signature_proposal", "")).strip()
            if candidate_signature and not _signature_budget_allows(
                planned_deterministic_counts_by_signature,
                candidate_signature,
                composition_limit_per_signature,
            ):
                actions.append(("budget_excluded", merged, "composition_limit_per_signature"))
                continue
            pending_adjudication_rows.append(merged)
            actions.append(("adjudication_candidate", merged, ""))
        elif (
            candidate["signature_proposal"]
            and validation["heuristic_feasibility_result"] == "pass"
            and validation["heuristic_non_separability_result"] == "pass"
        ):
            signature = str(candidate["signature_proposal"])
            actions.append(("deterministic_accept", merged, signature))
            if _signature_budget_allows(
                planned_deterministic_counts_by_signature,
                signature,
                composition_limit_per_signature,
            ):
                planned_deterministic_counts_by_signature[signature] += 1
        else:
            actions.append(("reject", merged, ""))

    selected_adjudication_ids = _select_adjudication_candidate_ids(
        pending_adjudication_rows,
        composition_adjudication_limit,
    )
    for action, row, _value in actions:
        if action == "adjudication_candidate" and str(row["composition_id"]) in selected_adjudication_ids:
            adjudication_queue.append(row)
        elif action == "adjudication_candidate":
            adjudication_budget_excluded_count += 1

    adjudication_results: dict[str, dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=live_max_workers) as executor:
        futures = {
            executor.submit(
                _adjudicate_composition_live,
                compositions_dir=compositions_dir,
                candidate=row,
                validator_client=validator_client,
                validator_model=validator_model,
            ): str(row["composition_id"])
            for row in adjudication_queue
        }
        for future in as_completed(futures):
            composition_id = futures[future]
            adjudication_results[composition_id] = future.result()

    for action, row, value in actions:
        if action == "adjudication_candidate":
            if str(row["composition_id"]) in selected_adjudication_ids:
                action = "adjudicate"
            else:
                action = "budget_excluded"
                value = "composition_adjudication_limit"
        if action == "budget_excluded":
            budget_excluded_count += 1
            rejected.append({**row, "budget_excluded": True, "budget_excluded_reason": value})
            continue
        if action == "reject":
            rejected.append(row)
            continue
        if action == "deterministic_accept":
            if signature_budget_allows(value):
                accept_candidate(row, signature=value)
            else:
                budget_excluded_count += 1
                rejected.append(
                    {
                        **row,
                        "budget_excluded": True,
                        "budget_excluded_reason": "composition_limit_per_signature",
                    }
                )
            continue

        adjudication = adjudication_results[str(row["composition_id"])]
        adjudications.append(adjudication)
        if adjudication["pass"] and adjudication["signature"]:
            signature = str(adjudication["signature"])
            accepted_row = {
                **row,
                "signature_proposal": signature,
                "relation_pattern": signature,
                "relation_patterns": [signature],
                "signature_source": "rubric_adjudicated",
                "feasibility_status": adjudication["feasibility_status"],
                "non_separability_status": adjudication["non_separability_status"],
                "nonseparability_slice": adjudication["nonseparability_slice"],
                "adjudication": adjudication,
            }
            if signature_budget_allows(signature):
                accept_candidate(accepted_row, signature=signature)
            else:
                budget_excluded_count += 1
                rejected.append(
                    {
                        **accepted_row,
                        "budget_excluded": True,
                        "budget_excluded_reason": "composition_limit_per_signature",
                    }
                )
        else:
            rejected.append({**row, "adjudication": adjudication})

    summary = {
        "candidate_count": len(candidates),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "adjudication_queue_count": len(adjudication_queue),
        "adjudication_count": len(adjudications),
        "budget_excluded_count": budget_excluded_count,
        "adjudication_budget_excluded_count": adjudication_budget_excluded_count,
        "composition_limit_per_signature": composition_limit_per_signature,
        "composition_adjudication_limit": composition_adjudication_limit,
        "live_max_workers": live_max_workers,
        "execution_mode": "live",
    }

    write_jsonl(compositions_dir / "composition_deterministic_results.jsonl", deterministic_rows)
    write_jsonl(compositions_dir / "composition_adjudication_queue.jsonl", adjudication_queue)
    write_jsonl(compositions_dir / "composition_adjudications.jsonl", adjudications)
    write_jsonl(compositions_dir / "accepted_compositions.jsonl", accepted)
    write_jsonl(compositions_dir / "rejected_compositions.jsonl", rejected)
    write_json(compositions_dir / "composition_validation_summary.json", summary)
    return summary
