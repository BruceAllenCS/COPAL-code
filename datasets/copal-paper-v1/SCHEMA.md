# Dataset Schema

## `companies/companies.jsonl`

One row per synthetic policy workspace.

Required fields:

- `company_key`: stable workspace identifier used by COPAL.
- `industry`: industry label.
- `company_name`: synthetic company name.
- `subtype`: company subtype.
- `company_size`: synthetic organization size.
- `geographic_focus`: geographic scope.
- `customer_segment`: B2B/B2C/customer segment description.
- `primary_offering`: business offering description.
- `chatbot_use_case`: intended chatbot deployment use case.
- `regulatory_constraints`: list of relevant regulatory or operational constraints.
- `allowed_rule_count`: number of allowed behavior rules.
- `prohibited_rule_count`: number of prohibited behavior rules.
- `total_rule_count`: total policy rules.
- `quality_scores`: source quality metadata.

## `policies/policy_worlds.jsonl`

One row per synthetic company policy world. This is the canonical paper input.

Required top-level fields:

- `industry`
- `enterprise_config`
- `policies`
- `quality_scores`

`enterprise_config` contains the company description and deployment context.

`policies.allowed_behaviors` and `policies.prohibited_behaviors` are arrays of policy rules. Each policy rule contains:

- `rule_id`
- `rule_text`
- `category`
- `severity` or legacy spelling variants retained in `raw_rule`
- `rationale`
- `verifiable`
- `verifiability_confidence`

## `policies/policy_rules.jsonl`

One flattened row per policy rule.

Required fields:

- `company_key`
- `industry`
- `company_name`
- `rule_type`: `allowed` or `prohibited`
- `rule_id`
- `rule_text`
- `category`
- `severity`
- `rationale`
- `verifiable`
- `verifiability_confidence`
- `raw_rule`: original source rule object for traceability.

## `prompts/system_prompts.jsonl`

One deployment system prompt per company.

Required fields:

- `company_key`
- `industry`
- `company_name`
- `company_index`
- `generation_mode`
- `model`
- `policy_counts`
- `system_prompt`

## `prompts/copal_prompt_templates.json`

Prompt-template index for COPAL construction and evaluation.

Fields:

- `dataset_version`
- `template_source_file`
- `system_prompts`: static COPAL role/system prompts.
- `message_builders`: prompt-builder names, stages, and purposes.

The exact dynamic prompt-builder source is stored in `prompts/copal_prompt_templates.py`.

## `artifacts/`

Curated paper experiment artifacts. These are exported from the final paper runs and are separate from the full 300-company source-policy inventory.

Top-level JSONL files:

- `company_world_specs.jsonl`: the 30 company policy worlds used in the final paper experiments, normalized with `company_key`.
- `policy_inventories.jsonl`: one inventory row per paper company, with flattened source-rule metadata.
- `grounded_clauses.jsonl`: grounded trigger/scope/effect clauses extracted from source policies.
- `composition_records.jsonl`: accepted relation-pattern composition records, including active clauses and scenario seeds.
- `generated_candidate_queries.jsonl`: raw generated candidate queries before final selection, annotated with company run and ablation variant.
- `screening_mapping_logs.jsonl`: query screening summaries, selected-query logs, post-hoc mapping labels, and variant summaries.
- `final_selected_suites.jsonl`: selected benchmark items for Table 3 and Table 2 ablation variants.
- `handling_contracts.jsonl`: extracted expected/forbidden handling contracts for selected items.
- `reconstructed_chatbot_prompts.jsonl`: one reconstructed deployment prompt per final paper company.
- `model_outputs.jsonl`: 9,000 downstream model responses for the 30-company Table 3 run.
- `automatic_judge_labels.jsonl`: automatic Gemini 3 Flash response-judge labels for those 9,000 responses.
- `ablation_candidate_pools.jsonl`: labeled candidate pools used by Table 2 ablation variants.

Subdirectories:

- `validation_records/`: curated LLM annotation, validation, and manual adjudication records. Provider caches are excluded.
- `run_manifests/`: final run summaries, repair manifests, judge-sensitivity summaries, and paper-final manifest snapshots.

`artifacts/manifest.json` records counts, file hashes, source-run paths relative to a full COPAL workspace, and the required artifact-family checklist.

## `manifest.json`

Release manifest with source provenance, counts, industry list, file paths, SHA-256 hashes, file sizes, and a pointer to `artifacts/manifest.json`.
