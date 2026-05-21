from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = Path("runs_release")
DEFAULT_TABLE2_ID = "release_table2_demo"
DEFAULT_TABLE3_ID = "release_table3_demo"
DEFAULT_PAIRED_ID = "release_paired_demo"
DEFAULT_JUDGE_ID = "release_judge_sensitivity_demo"


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def run_command(command: list[str], *, cwd: Path) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def require_live_provider_configured() -> None:
    provider = os.environ.get("COPAL_LIVE_PROVIDER", "").strip().lower()
    if provider != "openrouter":
        raise SystemExit(
            "Live release runs require COPAL_LIVE_PROVIDER=openrouter. "
            "Set COPAL_OPENROUTER_API_KEY or COPAL_OPENROUTER_API_KEY_FILE as well."
        )
    if not (
        os.environ.get("COPAL_OPENROUTER_API_KEY", "").strip()
        or os.environ.get("COPAL_OPENROUTER_API_KEY_FILE", "").strip()
    ):
        raise SystemExit(
            "Live release runs require COPAL_OPENROUTER_API_KEY or COPAL_OPENROUTER_API_KEY_FILE."
        )


def smoke_command(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        "-m",
        "copal.cli",
        "experiment",
        "run",
        "--experiment-id",
        args.experiment_id,
        "--company-limit",
        str(args.company_limit),
        "--execution-mode",
        "deterministic",
        "--sample-strategy",
        args.sample_strategy,
        "--stop-after",
        args.stop_after,
        "--runs-dir",
        str(args.runs_dir),
        "--company-workers",
        str(args.company_workers),
    ]
    run_command(command, cwd=REPO_ROOT)


def table2_command(args: argparse.Namespace) -> None:
    require_live_provider_configured()
    command = [
        sys.executable,
        "scripts/run_table2_ablation_pilot.py",
        "--experiment-id",
        args.experiment_id,
        "--company-limit",
        str(args.company_limit),
        "--company-offset",
        str(args.company_offset),
        "--sample-strategy",
        args.sample_strategy,
        "--runs-dir",
        str(args.runs_dir),
        "--grounding-model",
        args.grounding_model,
        "--composition-model",
        args.composition_model,
        "--query-model",
        args.query_model,
        "--mapping-model",
        args.mapping_model,
        "--screening-model",
        args.screening_model,
        "--judge-model",
        args.judge_model,
        "--eval-models",
        args.eval_models,
        "--max-compositions-per-company",
        str(args.max_compositions_per_company),
        "--direct-candidate-count",
        str(args.direct_candidate_count),
        "--query-variants-per-composition",
        str(args.query_variants_per_composition),
        "--query-variants-per-facet",
        str(args.query_variants_per_facet),
        "--selected-per-company",
        str(args.selected_per_company),
        "--live-max-workers",
        str(args.live_max_workers),
        "--stop-after",
        args.stop_after,
    ]
    run_command(command, cwd=REPO_ROOT)


def table3_command(args: argparse.Namespace) -> None:
    require_live_provider_configured()
    command = [
        sys.executable,
        "scripts/run_table3_model_eval.py",
        "--experiment-id",
        args.experiment_id,
        "--source-experiment-ids",
        args.source_experiment_ids,
        "--runs-dir",
        str(args.runs_dir),
        "--eval-models",
        args.eval_models,
        "--judge-model",
        args.judge_model,
        "--max-items-per-company",
        str(args.max_items_per_company),
        "--live-max-workers",
        str(args.live_max_workers),
        "--max-companies",
        str(args.max_companies),
    ]
    run_command(command, cwd=REPO_ROOT)


def paired_command(args: argparse.Namespace) -> None:
    require_live_provider_configured()
    command = [
        sys.executable,
        "scripts/run_paired_single_composed_from_table3.py",
        "--experiment-id",
        args.experiment_id,
        "--source-table3-experiment-ids",
        args.source_table3_experiment_ids,
        "--runs-dir",
        str(args.runs_dir),
        "--eval-models",
        args.eval_models,
        "--judge-model",
        args.judge_model,
        "--live-max-workers",
        str(args.live_max_workers),
        "--max-companies",
        str(args.max_companies),
    ]
    run_command(command, cwd=REPO_ROOT)


def judge_sensitivity_command(args: argparse.Namespace) -> None:
    require_live_provider_configured()
    command = [
        sys.executable,
        "scripts/run_table3_judge_sensitivity.py",
        "--experiment-id",
        args.experiment_id,
        "--source-experiment-ids",
        args.source_experiment_ids,
        "--runs-dir",
        str(args.runs_dir),
        "--sample-cases-per-company",
        str(args.sample_cases_per_company),
        "--sample-seed",
        str(args.sample_seed),
        "--exclude-response-models",
        args.exclude_response_models,
        "--judge-models",
        args.judge_models,
        "--live-max-workers",
        str(args.live_max_workers),
        "--judge-worker-overrides",
        args.judge_worker_overrides,
    ]
    run_command(command, cwd=REPO_ROOT)


def paper_demo_command(args: argparse.Namespace) -> None:
    require_live_provider_configured()
    table2_args = argparse.Namespace(
        experiment_id=args.table2_id,
        company_limit=args.company_limit,
        company_offset=args.company_offset,
        sample_strategy=args.sample_strategy,
        runs_dir=args.runs_dir,
        grounding_model=args.grounding_model,
        composition_model=args.composition_model,
        query_model=args.query_model,
        mapping_model=args.mapping_model,
        screening_model=args.screening_model,
        judge_model=args.judge_model,
        eval_models=args.eval_models,
        max_compositions_per_company=args.max_compositions_per_company,
        direct_candidate_count=args.direct_candidate_count,
        query_variants_per_composition=args.query_variants_per_composition,
        query_variants_per_facet=args.query_variants_per_facet,
        selected_per_company=args.selected_per_company,
        live_max_workers=args.live_max_workers,
        stop_after="evaluation",
    )
    table2_command(table2_args)

    table3_args = argparse.Namespace(
        experiment_id=args.table3_id,
        source_experiment_ids=args.table2_id,
        runs_dir=args.runs_dir,
        eval_models=args.eval_models,
        judge_model=args.judge_model,
        max_items_per_company=args.selected_per_company,
        live_max_workers=args.live_max_workers,
        max_companies=args.company_limit,
    )
    table3_command(table3_args)

    paired_args = argparse.Namespace(
        experiment_id=args.paired_id,
        source_table3_experiment_ids=args.table3_id,
        runs_dir=args.runs_dir,
        eval_models=args.eval_models,
        judge_model=args.judge_model,
        live_max_workers=args.live_max_workers,
        max_companies=args.company_limit,
    )
    paired_command(paired_args)


def add_live_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--grounding-model", default="gemini-3-flash-preview")
    parser.add_argument("--composition-model", default="gemini-3-flash-preview")
    parser.add_argument("--query-model", default="gpt-5.5")
    parser.add_argument("--mapping-model", default="gemini-3-flash-preview")
    parser.add_argument("--screening-model", default="gemini-3-flash-preview")
    parser.add_argument("--judge-model", default="gemini-3-flash-preview")
    parser.add_argument("--eval-models", default="Doubao-Seed-2.0-pro,gemini-3.1-pro-preview")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Release-friendly COPAL automation entrypoint.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="Run a deterministic one-company pipeline smoke test.")
    smoke.add_argument("--experiment-id", default="release_smoke")
    smoke.add_argument("--company-limit", type=int, default=1)
    smoke.add_argument("--sample-strategy", choices=("first", "one-per-industry"), default="first")
    smoke.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    smoke.add_argument("--stop-after", choices=("selection", "screening", "baselines", "audit", "evaluation"), default="evaluation")
    smoke.add_argument("--company-workers", type=int, default=1)
    smoke.set_defaults(func=smoke_command)

    table2 = subparsers.add_parser("table2", help="Run the Table 2 component-ablation construction/evaluation.")
    table2.add_argument("--experiment-id", default=DEFAULT_TABLE2_ID)
    table2.add_argument("--company-limit", type=int, default=1)
    table2.add_argument("--company-offset", type=int, default=0)
    table2.add_argument("--sample-strategy", choices=("first", "one-per-industry"), default="one-per-industry")
    table2.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    add_live_model_args(table2)
    table2.add_argument("--max-compositions-per-company", type=int, default=8)
    table2.add_argument("--direct-candidate-count", type=int, default=36)
    table2.add_argument("--query-variants-per-composition", type=int, default=4)
    table2.add_argument("--query-variants-per-facet", type=int, default=2)
    table2.add_argument("--selected-per-company", type=int, default=12)
    table2.add_argument("--live-max-workers", type=int, default=6)
    table2.add_argument("--stop-after", choices=("screening", "evaluation"), default="evaluation")
    table2.set_defaults(func=table2_command)

    table3 = subparsers.add_parser("table3", help="Run Table 3 model evaluation from completed Table 2 artifacts.")
    table3.add_argument("--experiment-id", default=DEFAULT_TABLE3_ID)
    table3.add_argument("--source-experiment-ids", default=DEFAULT_TABLE2_ID)
    table3.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    table3.add_argument("--eval-models", default="Doubao-Seed-2.0-pro,gemini-3.1-pro-preview")
    table3.add_argument("--judge-model", default="gemini-3-flash-preview")
    table3.add_argument("--max-items-per-company", type=int, default=12)
    table3.add_argument("--live-max-workers", type=int, default=6)
    table3.add_argument("--max-companies", type=int, default=1)
    table3.set_defaults(func=table3_command)

    paired = subparsers.add_parser("paired", help="Run single-policy vs composed-policy paired contrast.")
    paired.add_argument("--experiment-id", default=DEFAULT_PAIRED_ID)
    paired.add_argument("--source-table3-experiment-ids", default=DEFAULT_TABLE3_ID)
    paired.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    paired.add_argument("--eval-models", default="Doubao-Seed-2.0-pro,gemini-3.1-pro-preview")
    paired.add_argument("--judge-model", default="gemini-3-flash-preview")
    paired.add_argument("--live-max-workers", type=int, default=6)
    paired.add_argument("--max-companies", type=int, default=1)
    paired.set_defaults(func=paired_command)

    sensitivity = subparsers.add_parser("judge-sensitivity", help="Run alternative-judge robustness audit.")
    sensitivity.add_argument("--experiment-id", default=DEFAULT_JUDGE_ID)
    sensitivity.add_argument("--source-experiment-ids", default=DEFAULT_TABLE3_ID)
    sensitivity.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    sensitivity.add_argument("--sample-cases-per-company", type=int, default=3)
    sensitivity.add_argument("--sample-seed", type=int, default=12)
    sensitivity.add_argument("--exclude-response-models", default="gemini-3-flash-preview")
    sensitivity.add_argument("--judge-models", default="gpt-5.5,aws.claude-opus-4.7")
    sensitivity.add_argument("--live-max-workers", type=int, default=12)
    sensitivity.add_argument("--judge-worker-overrides", default="")
    sensitivity.set_defaults(func=judge_sensitivity_command)

    demo = subparsers.add_parser("paper-demo", help="Run a small end-to-end live demo: Table2 -> Table3 -> paired contrast.")
    demo.add_argument("--table2-id", default=DEFAULT_TABLE2_ID)
    demo.add_argument("--table3-id", default=DEFAULT_TABLE3_ID)
    demo.add_argument("--paired-id", default=DEFAULT_PAIRED_ID)
    demo.add_argument("--company-limit", type=int, default=1)
    demo.add_argument("--company-offset", type=int, default=0)
    demo.add_argument("--sample-strategy", choices=("first", "one-per-industry"), default="one-per-industry")
    demo.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    add_live_model_args(demo)
    demo.add_argument("--max-compositions-per-company", type=int, default=8)
    demo.add_argument("--direct-candidate-count", type=int, default=36)
    demo.add_argument("--query-variants-per-composition", type=int, default=4)
    demo.add_argument("--query-variants-per-facet", type=int, default=2)
    demo.add_argument("--selected-per-company", type=int, default=12)
    demo.add_argument("--live-max-workers", type=int, default=6)
    demo.set_defaults(func=paper_demo_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
