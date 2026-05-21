# COPAL Framework Guide

This guide describes COPAL as a reusable framework for evaluating arbitrary chatbots against composed organizational policies.

## Workflow

1. Write policy rules in COPAL's policy-world JSONL schema.
2. Run `construct` to build composed-policy probes.
3. Run one chatbot adapter: `probe-http`, `probe-command`, or `import-responses`.
4. Run `judge` to score chatbot responses against the generated handling contracts.

Each stage writes checkpointed files under a run directory, so repeated commands with the same configuration resume existing work.

## Policy Input Schema

`policy_worlds.jsonl` contains one JSON object per policy workspace:

```json
{
  "company_key": "demo-support",
  "industry": "customer-support",
  "enterprise_config": {
    "company_name": "Demo Support Bot"
  },
  "policies": {
    "allowed_behaviors": [
      {
        "rule_id": "allow_refund_window",
        "rule_text": "The assistant may explain the standard refund window...",
        "category": "refund-support",
        "severity": "low",
        "rationale": "Refund instructions are intended customer support content.",
        "verifiable": true,
        "verifiability_confidence": "high"
      }
    ],
    "prohibited_behaviors": [
      {
        "rule_id": "deny_internal_token",
        "rule_text": "The assistant must not reveal internal access tokens...",
        "category": "data-protection",
        "severity": "high",
        "rationale": "Internal operational data is sensitive.",
        "verifiable": true,
        "verifiability_confidence": "high"
      }
    ]
  },
  "quality_scores": {}
}
```

`company_key` is a stable workspace identifier. It is named `company_key` for compatibility with the paper code, but it can represent any chatbot deployment, product, domain, tenant, or policy workspace.

`system_prompts.jsonl` contains one JSON object per workspace:

```json
{
  "company_key": "demo-support",
  "industry": "customer-support",
  "company_name": "Demo Support Bot",
  "company_index": 0,
  "system_prompt": "You are the Demo Support Bot..."
}
```

The `company_key` must match the selected workspace.

## Construct Probes

```bash
python scripts/run_copal_framework.py construct \
  --workspace-key demo-support \
  --run-id demo_framework \
  --policies-path examples/policy_worlds.jsonl \
  --prompts-path examples/system_prompts.jsonl \
  --runs-dir runs_framework \
  --execution-mode live \
  --all-roles-model gpt-5.5 \
  --live-max-workers 8
```

For a dependency-free sanity check, use `--execution-mode deterministic`. Deterministic mode is for smoke tests and CI; live mode is the intended construction path for meaningful probes.

Important outputs:

- `grounding/grounded_clause_library.jsonl`
- `compositions/accepted_compositions.jsonl`
- `query_generation/accepted_queries.jsonl`
- `selection/benchmark_items_final.jsonl`

## HTTP Chatbot Adapter

```bash
python scripts/run_copal_framework.py probe-http \
  --run-dir runs_framework/demo_framework \
  --endpoint http://localhost:8000/chat \
  --response-json-key response_text \
  --bot-id production-bot \
  --live-max-workers 16
```

COPAL sends one POST request per selected probe:

```json
{
  "item_id": "a43d447967ac::boundary-overreach",
  "query": "User-facing probe text",
  "system_prompt": "Deployment system prompt",
  "messages": [
    {"role": "system", "content": "Deployment system prompt"},
    {"role": "user", "content": "User-facing probe text"}
  ],
  "metadata": {
    "signature": "scope-restriction",
    "target_facet": "boundary-overreach",
    "target_facets": ["boundary-overreach"]
  }
}
```

The endpoint must return a JSON object:

```json
{"response_text": "The chatbot response"}
```

Missing keys, non-2xx statuses, non-JSON bodies, and empty response text fail the run instead of silently producing placeholder results.

## Command Chatbot Adapter

```bash
python scripts/run_copal_framework.py probe-command \
  --run-dir runs_framework/demo_framework \
  --command "python examples/mock_chatbot.py" \
  --bot-id local-wrapper \
  --output-mode json \
  --response-json-key response_text
```

The command receives the same JSON payload on stdin. With `--output-mode json`, stdout must be:

```json
{"response_text": "The chatbot response"}
```

With `--output-mode text`, stdout itself is treated as the response text and must be non-empty.

## Import Responses

If your chatbot is tested in another harness, import the collected responses:

```bash
python scripts/run_copal_framework.py import-responses \
  --run-dir runs_framework/demo_framework \
  --responses-path collected_responses.jsonl \
  --bot-id production-bot
```

The response file must contain exactly one row for every selected probe:

```json
{"item_id": "a43d447967ac::boundary-overreach", "response_text": "The chatbot response"}
```

COPAL fails on missing, duplicate, or extra item IDs.

## Judge Responses

```bash
python scripts/run_copal_framework.py judge \
  --run-dir runs_framework/demo_framework \
  --execution-mode live \
  --judge-model gemini-3-flash-preview \
  --live-max-workers 16
```

The judge reads `selection/benchmark_items_final.jsonl` and `evaluation/chatbot_responses.jsonl`. It writes:

- `evaluation/response_judge_inputs.jsonl`
- `evaluation/response_judgments.jsonl`
- `evaluation/per_item_scores.jsonl`
- `evaluation/per_signature_scores.json`
- `evaluation/per_facet_scores.json`
- `evaluation/evaluation_summary.json`

If you cannot send chatbot responses to an external judge, stop after the adapter stage and use `evaluation/chatbot_responses.jsonl` with your own review process.

## Live Provider Notes

The public release supports OpenRouter-style routing through:

- `COPAL_LIVE_PROVIDER=openrouter`
- `COPAL_OPENROUTER_API_KEY` or `COPAL_OPENROUTER_API_KEY_FILE`
- `COPAL_OPENROUTER_MODEL_MAP`

`COPAL_OPENROUTER_MODEL_MAP` maps local COPAL aliases, such as `gpt-5.5`, to provider model IDs available to your account.
