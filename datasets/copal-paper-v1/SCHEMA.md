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

## `manifest.json`

Release manifest with source provenance, counts, industry list, file paths, SHA-256 hashes, and file sizes.
