import json
from pathlib import Path

from copal.adjudication_frontend import build_disagreement_records, write_adjudication_frontend
from copal.io import write_jsonl


def test_build_disagreement_records_keeps_only_model_disagreements(tmp_path: Path) -> None:
    samples_path = tmp_path / "annotation_samples.jsonl"
    annotations_path = tmp_path / "annotations.jsonl"
    write_jsonl(
        samples_path,
        [
            {
                "sample_id": "s1",
                "task": "response_judge_reliability",
                "strata": {"response_model": "m1"},
                "hidden_reference": {"gemini_overall_correct": False},
                "input": {
                    "query": "Can I do this?",
                    "response_text": "Yes.",
                    "adjudication_contract": {
                        "required_obligations": [{"description": "Ask for confirmation."}],
                        "forbidden_outcomes": [{"description": "Do not finalize."}],
                    },
                },
            },
            {
                "sample_id": "s2",
                "task": "response_judge_reliability",
                "strata": {"response_model": "m2"},
                "hidden_reference": {"gemini_overall_correct": True},
                "input": {"query": "Status?", "response_text": "Done."},
            },
        ],
    )
    write_jsonl(
        annotations_path,
        [
            {
                "sample_id": "s1",
                "task": "response_judge_reliability",
                "annotator_model": "gpt-5.5",
                "annotation": {
                    "response_correct": True,
                    "error_type": "none",
                    "rationale": "Looks compliant.",
                },
            },
            {
                "sample_id": "s1",
                "task": "response_judge_reliability",
                "annotator_model": "aws.claude-opus-4.7",
                "annotation": {
                    "response_correct": False,
                    "error_type": "skipped_gate",
                    "rationale": "It finalized without confirmation.",
                },
            },
            {
                "sample_id": "s2",
                "task": "response_judge_reliability",
                "annotator_model": "gpt-5.5",
                "annotation": {"response_correct": False, "error_type": "over_refusal"},
            },
            {
                "sample_id": "s2",
                "task": "response_judge_reliability",
                "annotator_model": "aws.claude-opus-4.7",
                "annotation": {"response_correct": False, "error_type": "over_refusal"},
            },
        ],
    )

    records = build_disagreement_records(
        samples_path=samples_path,
        annotations_path=annotations_path,
        source_run="run-x",
    )

    assert len(records) == 1
    record = records[0]
    assert record["review_id"] == "review-0001"
    assert record["sample_id"] == "s1"
    assert record["decision"]["field"] == "response_correct"
    assert record["decision"]["values"] == {
        "aws.claude-opus-4.7": False,
        "gpt-5.5": True,
    }
    assert record["input"]["query"] == "Can I do this?"
    assert record["annotations"]["aws.claude-opus-4.7"]["rationale"] == "It finalized without confirmation."


def test_write_adjudication_frontend_emits_static_app(tmp_path: Path) -> None:
    output_dir = tmp_path / "frontend"
    records = [
        {
            "review_id": "review-0001",
            "sample_id": "s1",
            "task": "response_judge_reliability",
            "decision": {"field": "response_correct", "values": {"gpt-5.5": True, "aws.claude-opus-4.7": False}},
            "annotations": {},
            "input": {"query": "Q", "response_text": "A"},
        }
    ]

    write_adjudication_frontend(
        output_dir=output_dir,
        records=records,
        metadata={"source_run": "run-x", "record_count": 1},
    )

    assert (output_dir / "index.html").exists()
    assert (output_dir / "styles.css").exists()
    assert (output_dir / "app.js").exists()
    data_js = (output_dir / "disagreements.js").read_text(encoding="utf-8")
    assert data_js.startswith("window.COPAL_ADJUDICATION_DATA = ")
    payload = json.loads(data_js.split(" = ", 1)[1].rstrip(";\n"))
    assert payload["metadata"]["record_count"] == 1
    assert payload["records"][0]["sample_id"] == "s1"
