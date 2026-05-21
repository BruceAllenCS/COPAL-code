from __future__ import annotations

from scripts.run_copal_framework import build_parser, parse_header


def test_framework_cli_exposes_construct_and_adapter_commands() -> None:
    parser = build_parser()
    construct = parser.parse_args(
        [
            "construct",
            "--workspace-key",
            "demo-support",
            "--run-id",
            "demo",
            "--policies-path",
            "examples/policy_worlds.jsonl",
            "--prompts-path",
            "examples/system_prompts.jsonl",
            "--execution-mode",
            "deterministic",
        ]
    )
    assert construct.command == "construct"
    assert construct.workspace_key == "demo-support"

    probe = parser.parse_args(
        [
            "probe-http",
            "--run-dir",
            "runs_framework/demo",
            "--endpoint",
            "http://localhost:8000/chat",
            "--header",
            "Authorization: Bearer test",
        ]
    )
    assert probe.command == "probe-http"
    assert probe.response_json_key == "response_text"


def test_framework_cli_header_parser_requires_name_value() -> None:
    assert parse_header(["Authorization: Bearer token"]) == {"Authorization": "Bearer token"}
    try:
        parse_header(["Authorization"])
    except ValueError as exc:
        assert "Name: value" in str(exc)
    else:
        raise AssertionError("malformed header must fail")
