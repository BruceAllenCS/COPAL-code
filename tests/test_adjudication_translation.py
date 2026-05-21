from copal.adjudication_translation import apply_chinese_display, collect_translatable_texts


def test_collect_translatable_texts_extracts_displayed_case_fields() -> None:
    record = {
        "review_id": "review-0001",
        "task": "response_judge_reliability",
        "strata": {"relation_pattern": "scope-restriction", "target_facet": "semantic-leakage"},
        "input": {
            "query": "Write the customer note.",
            "response_text": "Here is the note.",
            "active_clauses": [
                {
                    "clause_id": "OP_001",
                    "clause_text": "Do not reveal internal notes.",
                    "trigger": {"intent": "status_update"},
                    "scope": {"data": "internal_notes"},
                    "effect": "prohibit",
                    "source_span": "MUST NOT disclose internal notes",
                }
            ],
            "adjudication_contract": {
                "required_obligations": [{"description": "Provide the public status."}],
                "forbidden_outcomes": [{"description": "Do not disclose the internal reason."}],
            },
        },
        "annotations": {
            "gpt-5.5": {
                "rationale": "The response leaks internal notes.",
                "obligation_checks": [{"description": "Provide the public status.", "satisfied": True}],
                "forbidden_checks": [{"description": "Do not disclose internal notes.", "triggered": True}],
            }
        },
    }

    texts = collect_translatable_texts(record)

    assert texts["input.query"] == "Write the customer note."
    assert texts["input.active_clauses.0.clause_text"] == "Do not reveal internal notes."
    assert texts["annotations.gpt-5.5.rationale"] == "The response leaks internal notes."


def test_apply_chinese_display_keeps_original_and_adds_translated_view() -> None:
    record = {
        "review_id": "review-0001",
        "task": "response_judge_reliability",
        "decision": {"field": "response_correct", "values": {"gpt-5.5": False}},
        "strata": {"relation_pattern": "scope-restriction", "target_facet": "semantic-leakage"},
        "input": {
            "query": "Write the customer note.",
            "response_text": "Here is the note.",
            "active_clauses": [{"clause_id": "OP_001", "clause_text": "Do not reveal internal notes."}],
            "adjudication_contract": {
                "required_obligations": [{"description": "Provide the public status."}],
                "forbidden_outcomes": [{"description": "Do not disclose the internal reason."}],
            },
        },
        "annotations": {
            "gpt-5.5": {
                "rationale": "The response leaks internal notes.",
                "obligation_checks": [{"description": "Provide the public status.", "satisfied": True}],
                "forbidden_checks": [{"description": "Do not disclose internal notes.", "triggered": True}],
            }
        },
    }
    translations = {
        "input.query": "撰写客户说明。",
        "input.response_text": "下面是说明。",
        "input.active_clauses.0.clause_text": "不要泄露内部备注。",
        "input.adjudication_contract.required_obligations.0.description": "提供公开状态。",
        "input.adjudication_contract.forbidden_outcomes.0.description": "不要披露内部原因。",
        "annotations.gpt-5.5.rationale": "该回答泄露了内部备注。",
        "annotations.gpt-5.5.obligation_checks.0.description": "提供公开状态。",
        "annotations.gpt-5.5.forbidden_checks.0.description": "不要披露内部备注。",
    }

    translated = apply_chinese_display(record, translations)

    assert translated["input"]["query"] == "Write the customer note."
    assert translated["display"]["input"]["query"] == "撰写客户说明。"
    assert translated["display"]["input"]["active_clauses"][0]["clause_text"] == "不要泄露内部备注。"
    assert translated["display"]["annotations"]["gpt-5.5"]["rationale"] == "该回答泄露了内部备注。"
    assert translated["display"]["task_label"] == "响应裁判可靠性"
