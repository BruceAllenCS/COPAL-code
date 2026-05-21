# Reproducibility Guide

COPAL's public release has two reproducibility paths:

1. Framework reproducibility: verify that a fresh checkout can construct probes, call an arbitrary chatbot adapter, and judge responses.
2. Paper reproducibility: optionally reproduce the paper-specific Table 2, Table 3, paired contrast, and judge-sensitivity workflows.

The framework path is the primary open-source contribution.

## Framework Smoke Test

```bash
python scripts/run_copal_framework.py construct \
  --workspace-key demo-support \
  --run-id demo_framework \
  --policies-path examples/policy_worlds.jsonl \
  --prompts-path examples/system_prompts.jsonl \
  --runs-dir runs_framework \
  --execution-mode deterministic \
  --composition-limit-per-signature 1

python scripts/run_copal_framework.py probe-command \
  --run-dir runs_framework/demo_framework \
  --command "python examples/mock_chatbot.py" \
  --bot-id demo-mock \
  --live-max-workers 2

python scripts/run_copal_framework.py judge \
  --run-dir runs_framework/demo_framework \
  --execution-mode deterministic
```

Expected outputs:

- `runs_framework/demo_framework/selection/benchmark_items_final.jsonl`
- `runs_framework/demo_framework/evaluation/chatbot_responses.jsonl`
- `runs_framework/demo_framework/evaluation/evaluation_summary.json`

## Live Framework Run

Configure an OpenRouter-compatible provider:

```bash
export COPAL_LIVE_PROVIDER=openrouter
export COPAL_OPENROUTER_API_KEY="your-key"
export COPAL_OPENROUTER_RESPONSE_FORMAT=json_object
export COPAL_OPENROUTER_MODEL_MAP='{
  "gpt-5.5": "provider/model-id-for-gpt-5.5",
  "gemini-3-flash-preview": "provider/model-id-for-json-judge"
}'
```

Construct probes:

```bash
python scripts/run_copal_framework.py construct \
  --workspace-key your-workspace \
  --run-id your_run \
  --policies-path path/to/policy_worlds.jsonl \
  --prompts-path path/to/system_prompts.jsonl \
  --runs-dir runs_framework \
  --execution-mode live \
  --all-roles-model gpt-5.5 \
  --live-max-workers 8
```

Evaluate a target chatbot through HTTP:

```bash
python scripts/run_copal_framework.py evaluate-http \
  --run-dir runs_framework/your_run \
  --endpoint http://localhost:8000/chat \
  --response-json-key response_text \
  --bot-id my-chatbot \
  --execution-mode live \
  --judge-model gemini-3-flash-preview \
  --live-max-workers 16
```

## Paper-Specific Reproduction

The paper input dataset is committed under `datasets/copal-paper-v1/`. It contains:

- all 300 synthetic company/workspace records;
- all source policy worlds and flattened policy rules;
- all 300 deployment system prompts;
- COPAL construction, coverage, and response-judge prompt templates;
- the curated final paper artifact bundle, including the 30-company paper slice, grounded clauses, composition records, generated candidates, screening/mapping logs, selected suites, handling contracts, model outputs, automatic judge labels, ablation candidate pools, validation records, and run manifests;
- a manifest with counts, file hashes, and source provenance.

Regenerate the base synthetic dataset with:

```bash
python scripts/export_paper_dataset.py
```

Regenerate the curated paper artifact bundle from a full COPAL workspace with:

```bash
python scripts/export_paper_artifacts.py --copal-root /path/to/full/COPAL
```

Paper scripts remain available for readers who want to reproduce reported tables:

- `scripts/run_copal_release.py`
- `scripts/run_table2_ablation_pilot.py`
- `scripts/run_table3_model_eval.py`
- `scripts/run_paired_single_composed_from_table3.py`
- `scripts/run_table3_judge_sensitivity.py`

Compact paper summaries live under `results/paper_summaries/`. The full curated paper-response and judge-label artifacts are under `datasets/copal-paper-v1/artifacts/`; provider caches and private/internal real-bot deployment probes are not included.

## Checkpointing

Run directories include stage manifests. Re-running the same command with the same run ID and configuration reuses completed checkpoints. If you change construction settings, use a new run ID.
