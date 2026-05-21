# Reproducibility Guide

This guide records the public reproduction path for the COPAL release.

## Modes

COPAL has two execution modes:

- `deterministic`: no external LLM calls; useful for smoke tests and CI.
- `live`: calls configured LLM providers for clause grounding, composition validation, query generation, downstream model responses, and response judging.

The paper experiments used live LLM calls. The public release defaults to explicit OpenRouter routing for live runs.

## Deterministic Smoke Test

```bash
python scripts/run_copal_release.py smoke \
  --experiment-id release_smoke \
  --company-limit 1 \
  --runs-dir runs_release
```

Expected output files include:

- `runs_release/experiments/release_smoke/experiment_manifest.json`
- `runs_release/experiments/release_smoke/company_status.jsonl`
- `runs_release/experiments/release_smoke/experiment_summary.json`

## Live Provider Setup

Set an OpenRouter key and map COPAL model aliases to provider model IDs:

```bash
export COPAL_LIVE_PROVIDER=openrouter
export COPAL_OPENROUTER_API_KEY="your-key"
export COPAL_OPENROUTER_RESPONSE_FORMAT=json_object
export COPAL_OPENROUTER_MODEL_MAP='{
  "gemini-3-flash-preview": "provider/model-id-for-json-judge",
  "gemini-3.1-pro-preview": "provider/model-id-for-gemini-pro",
  "gpt-5.5": "provider/model-id-for-gpt-5.5",
  "Doubao-Seed-2.0-pro": "provider/model-id-for-doubao",
  "aws.claude-opus-4.7": "provider/model-id-for-claude-opus"
}'
```

Use `COPAL_OPENROUTER_API_KEY_FILE` instead of `COPAL_OPENROUTER_API_KEY` if you prefer a local JSON file:

```json
{
  "OPENROUTER_API_KEY": "your-key"
}
```

Do not commit that key file.

## One-Company Live Demo

```bash
python scripts/run_copal_release.py paper-demo \
  --company-limit 1 \
  --selected-per-company 12 \
  --eval-models Doubao-Seed-2.0-pro,gemini-3.1-pro-preview \
  --runs-dir runs_release \
  --live-max-workers 6
```

This executes:

1. Table 2-style COPAL construction/ablation for one company.
2. Table 3-style downstream response evaluation on selected composed-policy cases.
3. Paired single-policy projection evaluation for the same selected composed-policy items.

## Full-Scale Paper Shape

The paper-scale setting uses 30 companies. The exact model availability and cost depend on the provider account.

```bash
python scripts/run_copal_release.py paper-demo \
  --company-limit 30 \
  --selected-per-company 30 \
  --eval-models gpt-5.5,aws.claude-sonnet-4.6,gemini-3.1-pro-preview,Doubao-Seed-2.0-pro,kimi-k2.6,MiniMax-M2.7,qwen3.5-baidu,glm-5.1,deepseek-v3.2-tencent \
  --runs-dir runs_release \
  --live-max-workers 24
```

Then run the judge-family sensitivity audit from the completed Table 3 outputs:

```bash
python scripts/run_copal_release.py judge-sensitivity \
  --source-experiment-ids release_table3_demo \
  --sample-cases-per-company 10 \
  --judge-models gpt-5.5,aws.claude-opus-4.7,gemini-3.1-pro-preview,deepseek-v3.2-tencent \
  --runs-dir runs_release \
  --live-max-workers 64
```

## Checkpointing

The runners write manifests and per-company outputs under `runs_release/experiments/<experiment-id>/`. Re-run the same command with the same experiment ID and configuration to resume completed work. If configuration changes, use a new experiment ID.

## Paper Summary Artifacts

Use the compact JSON summaries in `results/paper_summaries/` for table reconstruction. Full raw live-response artifacts are intentionally not included in the public release to avoid shipping large cache/run directories.
