from __future__ import annotations

import argparse
import shlex
from pathlib import Path

from copal.cli import main as copal_cli_main
from copal.framework import (
    load_benchmark_items,
    load_system_prompt,
    run_command_chatbot_probe,
    run_http_chatbot_probe,
    write_imported_responses,
)
from copal.llm import build_live_client
from copal.stages.response_judgment import run_response_judgment_stage


def parse_header(values: list[str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for value in values:
        if ":" not in value:
            raise ValueError(f"Header must use 'Name: value' format: {value}")
        key, raw_header_value = value.split(":", 1)
        key = key.strip()
        header_value = raw_header_value.strip()
        if not key or not header_value:
            raise ValueError(f"Header must include non-empty name and value: {value}")
        headers[key] = header_value
    return headers


def construct_command(args: argparse.Namespace) -> None:
    run_args = [
        "run",
        "--company-key",
        args.workspace_key,
        "--run-id",
        args.run_id,
        "--runs-dir",
        str(args.runs_dir),
        "--cache-dir",
        str(args.cache_dir),
        "--policies-path",
        str(args.policies_path),
        "--prompts-path",
        str(args.prompts_path),
        "--execution-mode",
        args.execution_mode,
        "--stop-after",
        "selection",
        "--live-max-workers",
        str(args.live_max_workers),
        "--composition-limit-per-signature",
        str(args.composition_limit_per_signature),
        "--composition-adjudication-limit",
        str(args.composition_adjudication_limit),
        "--query-variants-per-facet",
        str(args.query_variants_per_facet),
        "--selection-variants-per-facet",
        str(args.selection_variants_per_facet),
    ]
    if args.all_roles_model:
        run_args.extend(["--all-roles-model", args.all_roles_model])
    for flag_name in (
        "proposal_model",
        "query_proposal_model",
        "canonicalization_model",
        "validator_model",
        "coverage_judge_model",
        "response_judge_model",
    ):
        value = getattr(args, flag_name)
        if value:
            run_args.extend([f"--{flag_name.replace('_', '-')}", value])
    copal_cli_main(run_args)


def probe_http_command(args: argparse.Namespace) -> None:
    benchmark_items = load_benchmark_items(args.run_dir)
    system_prompt = load_system_prompt(args.run_dir)
    run_http_chatbot_probe(
        evaluation_dir=args.run_dir / "evaluation",
        benchmark_items=benchmark_items,
        system_prompt=system_prompt,
        endpoint=args.endpoint,
        response_json_key=args.response_json_key,
        bot_id=args.bot_id,
        live_max_workers=args.live_max_workers,
        timeout=args.timeout,
        headers=parse_header(args.header),
    )


def probe_command_command(args: argparse.Namespace) -> None:
    benchmark_items = load_benchmark_items(args.run_dir)
    system_prompt = load_system_prompt(args.run_dir)
    command = shlex.split(args.command)
    if not command:
        raise ValueError("--command must parse to at least one argv element")
    run_command_chatbot_probe(
        evaluation_dir=args.run_dir / "evaluation",
        benchmark_items=benchmark_items,
        system_prompt=system_prompt,
        command=command,
        output_mode=args.output_mode,
        response_json_key=args.response_json_key,
        bot_id=args.bot_id,
        live_max_workers=args.live_max_workers,
        timeout=args.timeout,
    )


def import_responses_command(args: argparse.Namespace) -> None:
    write_imported_responses(
        evaluation_dir=args.run_dir / "evaluation",
        benchmark_items=load_benchmark_items(args.run_dir),
        responses_path=args.responses_path,
        bot_id=args.bot_id,
    )


def judge_command(args: argparse.Namespace) -> None:
    benchmark_items = load_benchmark_items(args.run_dir)
    response_judge_client = None
    if args.execution_mode == "live":
        response_judge_client = build_live_client(cache_dir=args.cache_dir / "framework_judge")
    run_response_judgment_stage(
        evaluation_dir=args.run_dir / "evaluation",
        benchmark_items=benchmark_items,
        execution_mode=args.execution_mode,
        response_judge_client=response_judge_client,
        response_judge_model=args.judge_model,
        live_max_workers=args.live_max_workers,
    )


def evaluate_http_command(args: argparse.Namespace) -> None:
    probe_http_command(args)
    judge_command(args)


def add_construct_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--all-roles-model", help="Use one LLM alias for all COPAL construction roles.")
    parser.add_argument("--proposal-model")
    parser.add_argument("--query-proposal-model")
    parser.add_argument("--canonicalization-model")
    parser.add_argument("--validator-model")
    parser.add_argument("--coverage-judge-model")
    parser.add_argument("--response-judge-model")


def add_probe_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bot-id", default="target-chatbot")
    parser.add_argument("--live-max-workers", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=60)


def add_judge_args(parser: argparse.ArgumentParser, *, include_workers: bool = True) -> None:
    parser.add_argument("--execution-mode", choices=("deterministic", "live"), required=True)
    parser.add_argument("--judge-model", default="")
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    if include_workers:
        parser.add_argument("--live-max-workers", type=int, default=1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "COPAL framework entrypoint for constructing composed-policy probes and "
            "evaluating an arbitrary chatbot adapter."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    construct = subparsers.add_parser("construct", help="Construct composed-policy probes from a policy file.")
    construct.add_argument("--workspace-key", required=True)
    construct.add_argument("--run-id", required=True)
    construct.add_argument("--policies-path", type=Path, required=True)
    construct.add_argument("--prompts-path", type=Path, required=True)
    construct.add_argument("--runs-dir", type=Path, default=Path("runs_framework"))
    construct.add_argument("--cache-dir", type=Path, default=Path("cache"))
    construct.add_argument("--execution-mode", choices=("deterministic", "live"), required=True)
    construct.add_argument("--live-max-workers", type=int, default=1)
    construct.add_argument("--composition-limit-per-signature", type=int, default=0)
    construct.add_argument("--composition-adjudication-limit", type=int, default=-1)
    construct.add_argument("--query-variants-per-facet", type=int, default=1)
    construct.add_argument("--selection-variants-per-facet", type=int, default=1)
    add_construct_model_args(construct)
    construct.set_defaults(func=construct_command)

    http_probe = subparsers.add_parser("probe-http", help="Send selected probes to an HTTP chatbot adapter.")
    add_probe_args(http_probe)
    http_probe.add_argument("--endpoint", required=True)
    http_probe.add_argument("--response-json-key", default="response_text")
    http_probe.add_argument("--header", action="append", default=[])
    http_probe.set_defaults(func=probe_http_command)

    command_probe = subparsers.add_parser("probe-command", help="Send selected probes to a command-line chatbot adapter.")
    add_probe_args(command_probe)
    command_probe.add_argument("--command", required=True)
    command_probe.add_argument("--output-mode", choices=("json", "text"), default="json")
    command_probe.add_argument("--response-json-key", default="response_text")
    command_probe.set_defaults(func=probe_command_command)

    importer = subparsers.add_parser("import-responses", help="Import externally collected chatbot responses.")
    importer.add_argument("--run-dir", type=Path, required=True)
    importer.add_argument("--responses-path", type=Path, required=True)
    importer.add_argument("--bot-id", default="target-chatbot")
    importer.set_defaults(func=import_responses_command)

    judge = subparsers.add_parser("judge", help="Judge chatbot responses against COPAL handling contracts.")
    judge.add_argument("--run-dir", type=Path, required=True)
    add_judge_args(judge)
    judge.set_defaults(func=judge_command)

    evaluate_http = subparsers.add_parser(
        "evaluate-http",
        help="Convenience command: probe an HTTP chatbot and then judge its responses.",
    )
    add_probe_args(evaluate_http)
    evaluate_http.add_argument("--endpoint", required=True)
    evaluate_http.add_argument("--response-json-key", default="response_text")
    evaluate_http.add_argument("--header", action="append", default=[])
    add_judge_args(evaluate_http, include_workers=False)
    evaluate_http.set_defaults(func=evaluate_http_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
