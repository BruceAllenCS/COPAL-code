from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from copal.io import read_json, write_json


TASK_LABELS = {
    "response_judge_reliability": "响应裁判可靠性",
    "clause_grounding": "子句溯源正确性",
    "composition_interaction": "组合交互有效性",
    "handling_contract": "期望/禁止处理契约可信度",
}

FIELD_LABELS = {
    "response_correct": "响应是否正确",
    "overall_valid": "整体是否有效",
}

RELATION_PATTERN_LABELS = {
    "scope-restriction": "范围限制",
    "prerequisite-gating": "前置门控",
    "workflow-transfer": "流程转交",
    "response-boundary": "响应边界",
}

FACET_LABELS = {
    "boundary-overreach": "边界越界",
    "latent-continuation": "名义转交后的继续执行",
    "pre-gate-leakage": "门控前泄露",
    "semantic-leakage": "语义泄露",
    "missed-transfer": "漏掉转交",
    "wrong-scope-gate": "错误范围门控",
    "wrong_scope_gate": "错误范围门控",
    "over-refusal": "过度拒答",
    "over_refusal": "过度拒答",
    "unsupported-commitment": "无权限承诺",
    "unsupported_commitment": "无权限承诺",
    "prohibited-disclosure": "禁止披露",
    "prohibited_disclosure": "禁止披露",
    "skipped-gate": "跳过门控",
    "skipped_gate": "跳过门控",
    "wrong-route": "错误路由",
    "wrong_route": "错误路由",
    "none": "无",
    "other": "其他",
}

EFFECT_LABELS = {
    "permit": "允许",
    "prohibit": "禁止",
    "route": "转交",
    "require-gate": "要求门控",
    "disclose": "披露",
}

SOURCE_RULE_TYPE_LABELS = {
    "allowed": "允许规则",
    "prohibited": "禁止规则",
}


def collect_translatable_texts(record: dict[str, Any]) -> dict[str, str]:
    texts: dict[str, str] = {}
    sample_input = _require_object(record.get("input"), "record.input")
    _add_text(texts, "input.query", sample_input.get("query"))
    _add_text(texts, "input.response_text", sample_input.get("response_text"))

    for index, clause in enumerate(sample_input.get("active_clauses", []) or []):
        if not isinstance(clause, dict):
            raise ValueError(f"active_clauses[{index}] must be an object")
        prefix = f"input.active_clauses.{index}"
        _add_text(texts, f"{prefix}.clause_text", clause.get("clause_text"))
        _add_text(texts, f"{prefix}.source_span", clause.get("source_span"))
        _add_serialized(texts, f"{prefix}.trigger", clause.get("trigger"))
        _add_serialized(texts, f"{prefix}.scope", clause.get("scope"))

    contract = _optional_object(sample_input.get("adjudication_contract"))
    for index, row in enumerate(contract.get("required_obligations", []) or []):
        if isinstance(row, dict):
            _add_text(texts, f"input.adjudication_contract.required_obligations.{index}.description", row.get("description"))
    for index, row in enumerate(contract.get("forbidden_outcomes", []) or []):
        if isinstance(row, dict):
            _add_text(texts, f"input.adjudication_contract.forbidden_outcomes.{index}.description", row.get("description"))

    annotations = _require_object(record.get("annotations"), "record.annotations")
    for model in sorted(annotations):
        annotation = _require_object(annotations[model], f"record.annotations.{model}")
        _add_text(texts, f"annotations.{model}.rationale", annotation.get("rationale"))
        for index, row in enumerate(annotation.get("obligation_checks", []) or []):
            if isinstance(row, dict):
                _add_text(texts, f"annotations.{model}.obligation_checks.{index}.description", row.get("description"))
        for index, row in enumerate(annotation.get("forbidden_checks", []) or []):
            if isinstance(row, dict):
                _add_text(texts, f"annotations.{model}.forbidden_checks.{index}.description", row.get("description"))
    return texts


def apply_chinese_display(record: dict[str, Any], translations: dict[str, str]) -> dict[str, Any]:
    translated = copy.deepcopy(record)
    sample_input = _require_object(record.get("input"), "record.input")
    decision = _optional_object(record.get("decision"))
    strata = _optional_object(record.get("strata"))

    display_input: dict[str, Any] = {
        "query": _translation(translations, "input.query", sample_input.get("query")),
        "response_text": _translation(translations, "input.response_text", sample_input.get("response_text")),
        "active_clauses": [],
        "adjudication_contract": {
            "required_obligations": [],
            "forbidden_outcomes": [],
        },
    }

    for index, clause in enumerate(sample_input.get("active_clauses", []) or []):
        if not isinstance(clause, dict):
            raise ValueError(f"active_clauses[{index}] must be an object")
        prefix = f"input.active_clauses.{index}"
        display_input["active_clauses"].append(
            {
                "clause_id": clause.get("clause_id"),
                "source_rule_ids": clause.get("source_rule_ids", []),
                "source_rule_type": _map_or_original(SOURCE_RULE_TYPE_LABELS, clause.get("source_rule_type")),
                "effect": _map_or_original(EFFECT_LABELS, clause.get("effect")),
                "clause_text": _translation(translations, f"{prefix}.clause_text", clause.get("clause_text")),
                "trigger": _translation(translations, f"{prefix}.trigger", _json_text(clause.get("trigger"))),
                "scope": _translation(translations, f"{prefix}.scope", _json_text(clause.get("scope"))),
                "source_span": _translation(translations, f"{prefix}.source_span", clause.get("source_span")),
                "confidence": clause.get("confidence"),
            }
        )

    contract = _optional_object(sample_input.get("adjudication_contract"))
    for index, row in enumerate(contract.get("required_obligations", []) or []):
        if isinstance(row, dict):
            display_input["adjudication_contract"]["required_obligations"].append(
                {
                    "description": _translation(
                        translations,
                        f"input.adjudication_contract.required_obligations.{index}.description",
                        row.get("description"),
                    )
                }
            )
    for index, row in enumerate(contract.get("forbidden_outcomes", []) or []):
        if isinstance(row, dict):
            display_input["adjudication_contract"]["forbidden_outcomes"].append(
                {
                    "description": _translation(
                        translations,
                        f"input.adjudication_contract.forbidden_outcomes.{index}.description",
                        row.get("description"),
                    )
                }
            )

    display_annotations: dict[str, Any] = {}
    annotations = _require_object(record.get("annotations"), "record.annotations")
    for model in sorted(annotations):
        annotation = _require_object(annotations[model], f"record.annotations.{model}")
        display_annotations[model] = {
            "rationale": _translation(translations, f"annotations.{model}.rationale", annotation.get("rationale")),
            "error_type": _map_or_original(FACET_LABELS, annotation.get("error_type")),
            "confidence": annotation.get("confidence"),
            "satisfies_required_obligations": annotation.get("satisfies_required_obligations"),
            "triggers_forbidden_outcome": annotation.get("triggers_forbidden_outcome"),
            "obligation_checks": _translated_checks(
                annotation.get("obligation_checks", []),
                translations=translations,
                prefix=f"annotations.{model}.obligation_checks",
                bool_key="satisfied",
            ),
            "forbidden_checks": _translated_checks(
                annotation.get("forbidden_checks", []),
                translations=translations,
                prefix=f"annotations.{model}.forbidden_checks",
                bool_key="triggered",
            ),
        }

    translated["display"] = {
        "language": "zh-CN",
        "task_label": _map_or_original(TASK_LABELS, record.get("task")),
        "decision_field_label": _map_or_original(FIELD_LABELS, decision.get("field")),
        "strata": {
            key: _display_strata_value(key, value)
            for key, value in strata.items()
            if key not in {"response_model", "gemini_correct"}
        },
        "input": display_input,
        "annotations": display_annotations,
    }
    return translated


def write_chinese_adjudication_payload(
    *,
    source_json: Path,
    translations_jsonl: Path,
    output_json: Path,
) -> dict[str, Any]:
    payload = read_json(source_json)
    translations_by_review = _load_translations(translations_jsonl)
    records = []
    missing: list[str] = []
    for record in payload["records"]:
        review_id = str(record["review_id"])
        translations = translations_by_review.get(review_id)
        if translations is None:
            missing.append(review_id)
            continue
        records.append(apply_chinese_display(record, translations))
    if missing:
        raise ValueError(f"missing translations for review ids: {', '.join(missing)}")
    payload["records"] = records
    payload["metadata"] = {
        **payload.get("metadata", {}),
        "language": "zh-CN",
        "translation_artifact": str(translations_jsonl),
    }
    write_json(output_json, payload)
    return payload


def _translated_checks(
    checks: Any,
    *,
    translations: dict[str, str],
    prefix: str,
    bool_key: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(checks or []):
        if isinstance(row, dict):
            rows.append(
                {
                    "description": _translation(translations, f"{prefix}.{index}.description", row.get("description")),
                    bool_key: row.get(bool_key),
                }
            )
    return rows


def _load_translations(path: Path) -> dict[str, dict[str, str]]:
    translations: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            review_id = str(row["review_id"])
            if review_id in translations:
                raise ValueError(f"duplicate translation row for {review_id}")
            value = row["translations"]
            if not isinstance(value, dict):
                raise ValueError(f"translations must be an object for {review_id}")
            translations[review_id] = {str(key): str(text) for key, text in value.items()}
    return translations


def _display_strata_value(key: str, value: Any) -> str:
    if key == "relation_pattern":
        return _map_or_original(RELATION_PATTERN_LABELS, value)
    if key == "target_facet":
        return _map_or_original(FACET_LABELS, value)
    return str(value)


def _translation(translations: dict[str, str], key: str, original: Any) -> str:
    if key in translations:
        return translations[key]
    return "" if original is None else str(original)


def _add_text(texts: dict[str, str], key: str, value: Any) -> None:
    if isinstance(value, str) and value.strip():
        texts[key] = value


def _add_serialized(texts: dict[str, str], key: str, value: Any) -> None:
    serialized = _json_text(value)
    if serialized:
        texts[key] = serialized


def _json_text(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _map_or_original(mapping: dict[str, str], value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return mapping.get(text, text)


def _require_object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return value


def _optional_object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
