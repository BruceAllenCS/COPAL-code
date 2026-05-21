# COPAL

COPAL is a framework for testing whether a chatbot can handle composed organizational policies. Given a policy file, COPAL extracts grounded clauses, finds interacting clause compositions, generates composed-policy probes, sends those probes to a target chatbot, and judges whether the chatbot satisfies the expected handling contract.

The target chatbot can be any system you can wrap behind an HTTP endpoint, a command-line adapter, or a JSONL response file. The paper experiments are included as optional research artifacts; the main public interface is the framework workflow below.

## What COPAL Does

1. Convert policy rules into grounded clauses with trigger, scope, and effect fields.
2. Select non-separable clause compositions using relation patterns.
3. Generate probes that target policy-composition failure facets.
4. Run those probes against your chatbot adapter.
5. Judge responses against each probe's expected and forbidden handling contract.

## Installation

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

## Minimal Framework Smoke Test

This uses the included demo policy and a local command-line mock chatbot. It does not call external LLM APIs.

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

Outputs are written under `runs_framework/demo_framework/`. The key files are:

- `selection/benchmark_items_final.jsonl`: generated composed-policy probes.
- `evaluation/chatbot_requests.jsonl`: requests sent to the chatbot adapter.
- `evaluation/chatbot_responses.jsonl`: target chatbot responses.
- `evaluation/response_judgments.jsonl`: response-level correctness judgments.
- `evaluation/evaluation_summary.json`: aggregate accuracy/error summary.

## Use COPAL With Your Chatbot

Prepare two JSONL files:

- Policy worlds: follow `examples/policy_worlds.jsonl`.
- System prompts: follow `examples/system_prompts.jsonl`.

Then construct probes:

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

Probe an HTTP chatbot:

```bash
python scripts/run_copal_framework.py probe-http \
  --run-dir runs_framework/your_run \
  --endpoint http://localhost:8000/chat \
  --response-json-key response_text \
  --bot-id my-chatbot \
  --live-max-workers 16
```

The HTTP endpoint receives:

```json
{
  "item_id": "...",
  "query": "...",
  "system_prompt": "...",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "metadata": {
    "signature": "scope-restriction",
    "target_facet": "boundary-overreach",
    "target_facets": ["boundary-overreach"]
  }
}
```

It must return a JSON object containing `response_text`, or whichever key you pass with `--response-json-key`.

You can also import responses collected elsewhere:

```bash
python scripts/run_copal_framework.py import-responses \
  --run-dir runs_framework/your_run \
  --responses-path path/to/chatbot_responses.jsonl \
  --bot-id my-chatbot
```

The imported JSONL must contain one row per selected probe:

```json
{"item_id": "probe-id", "response_text": "chatbot answer"}
```

Finally judge the responses:

```bash
python scripts/run_copal_framework.py judge \
  --run-dir runs_framework/your_run \
  --execution-mode live \
  --judge-model gemini-3-flash-preview \
  --live-max-workers 16
```

## Live LLM Configuration

For public live runs, configure an OpenRouter-compatible route explicitly:

```bash
export COPAL_LIVE_PROVIDER=openrouter
export COPAL_OPENROUTER_API_KEY="your-key"
export COPAL_OPENROUTER_RESPONSE_FORMAT=json_object
export COPAL_OPENROUTER_MODEL_MAP='{
  "gpt-5.5": "provider/model-id-for-gpt-5.5",
  "gemini-3-flash-preview": "provider/model-id-for-json-judge"
}'
```

Do not commit local key files.

## Repository Layout

- `copal/`: framework library, stages, adapters, checkpointing, and judging.
- `scripts/run_copal_framework.py`: framework entrypoint for third-party chatbot testing.
- `examples/`: minimal policy world, system prompt, and command-line chatbot adapter.
- `docs/FRAMEWORK.md`: input schemas and adapter contracts.
- `datasets/copal-paper-v1/`: paper reproducibility dataset with all synthetic companies, policies, deployment system prompts, and COPAL prompt templates.
- `scripts/run_copal_release.py` and table scripts: optional paper reproduction utilities.
- `data/compass_policies/`, `results/paper_summaries/`, `paper_final/`: compact paper-facing artifacts.

## Verification

```bash
python -m pytest tests
```

GitHub Actions runs installation, the full test suite, and the framework smoke test.

## Citation

If you use this framework, cite the COPAL paper. A placeholder citation file is provided in `CITATION.cff`; update it with the final title, author list, venue, and DOI/arXiv identifier when available.
