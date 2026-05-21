# COPAL

COPAL is the code release for constructing and evaluating composed-policy test cases for organizational-policy chatbots. It turns company policy rules into grounded clauses, composes interacting clauses through relation patterns, generates targeted queries, runs downstream response models, and judges policy-handling outcomes.

This repository is organized as a reproducible research artifact. It includes the public COPAL pipeline, the synthetic COMPASS-style policy worlds used by the paper experiments, paper-summary result artifacts, and release-friendly scripts for smoke tests and live reproduction.

## What Is Included

- `copal/`: core construction, evaluation, judging, checkpointing, and analysis modules.
- `scripts/run_copal_release.py`: one entrypoint for deterministic smoke tests and live paper-demo runs.
- `scripts/run_table2_ablation_pilot.py`: Table 2 component-ablation runner.
- `scripts/run_table3_model_eval.py`: Table 3 downstream model evaluation runner.
- `scripts/run_paired_single_composed_from_table3.py`: single-policy vs composed-policy paired contrast runner.
- `scripts/run_table3_judge_sensitivity.py`: alternative-judge sensitivity audit runner.
- `data/compass_policies/`: synthetic company policy worlds and system prompts.
- `results/paper_summaries/`: compact JSON summaries for the paper tables and judge-sensitivity audit.
- `paper_final/manifests/`: paper-facing experiment manifests and write-ups.

The release excludes local caches, raw long-running experiment directories, local API keys, and private deployed-bot policy files.

## Installation

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

## Quick Smoke Test

The deterministic smoke test does not call external LLM APIs:

```bash
python scripts/run_copal_release.py smoke \
  --experiment-id release_smoke \
  --company-limit 1 \
  --runs-dir runs_release
```

The output is written under `runs_release/experiments/release_smoke/`.

## Live LLM Configuration

For public reproduction, use the OpenRouter route explicitly:

```bash
export COPAL_LIVE_PROVIDER=openrouter
export COPAL_OPENROUTER_API_KEY="your-key"
export COPAL_OPENROUTER_RESPONSE_FORMAT=json_object
export COPAL_OPENROUTER_MODEL_MAP="$(python - <<'PY'
import json
print(json.dumps({
  "gemini-3-flash-preview": "provider/model-id-for-json-judge",
  "gemini-3.1-pro-preview": "provider/model-id-for-gemini-pro",
  "gpt-5.5": "provider/model-id-for-gpt-5.5",
  "Doubao-Seed-2.0-pro": "provider/model-id-for-doubao",
  "aws.claude-opus-4.7": "provider/model-id-for-claude-opus"
}))
PY
)"
```

Edit the provider model IDs to match the models available to your account. COPAL model names are local aliases; `COPAL_OPENROUTER_MODEL_MAP` maps them to provider model IDs.

## Small Live Paper Demo

This runs a one-company live demo through construction, Table 3 model evaluation, and the paired single-policy contrast:

```bash
python scripts/run_copal_release.py paper-demo \
  --company-limit 1 \
  --selected-per-company 12 \
  --eval-models Doubao-Seed-2.0-pro,gemini-3.1-pro-preview \
  --runs-dir runs_release \
  --live-max-workers 6
```

For larger reproduction, increase `--company-limit` to 30 and use the model set reported in the paper. The scripts are checkpointed at the filesystem level and can be resumed with the same experiment IDs and configuration.

## Paper Results

Compact result summaries are stored in `results/paper_summaries/`:

- Table 2 component ablation: `results/paper_summaries/table2_ablation/table2_summary.json`
- Table 3 downstream model evaluation: `results/paper_summaries/table3_model_eval/table3_summary.json`
- Single-policy vs composed-policy contrast: `results/paper_summaries/paired_single_composed/paired_single_composed_summary.json`
- Judge-family sensitivity audit: `results/paper_summaries/judge_sensitivity/judge_sensitivity_summary.json`
- Judge disagreement analysis: `results/paper_summaries/judge_sensitivity/judge_disagreement_analysis.json`

The curated paper-facing reading path is `paper_final/README.md`.

## Verification

```bash
python -m pytest tests
```

Some live-integration paths require external provider credentials. The deterministic smoke test above is the fastest sanity check for a clean checkout.

## Citation

If you use this code, cite the COPAL paper. A placeholder citation file is provided in `CITATION.cff`; update it with the final title, author list, venue, and DOI/arXiv identifier when available.
