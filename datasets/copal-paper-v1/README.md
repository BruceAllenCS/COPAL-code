# COPAL Paper Reproducibility Dataset

This directory contains the synthetic organizational-policy chatbot dataset used for the COPAL paper experiments.

## Contents

- `companies/companies.jsonl`: one company/workspace metadata row per synthetic chatbot deployment.
- `policies/policy_worlds.jsonl`: one full policy-world object per company, including allowed and prohibited policy rules.
- `policies/policy_rules.jsonl`: one flattened policy-rule row per rule, with company metadata and rule type.
- `prompts/system_prompts.jsonl`: one deployment system prompt per company.
- `prompts/copal_prompt_templates.json`: index of COPAL construction/evaluation prompt roles and builder purposes.
- `prompts/copal_prompt_templates.py`: exact prompt-builder source used by this release.
- `metadata/dataset_summary.json`: aggregate counts and quality-summary metadata from the source policy generator.
- `manifest.json`: dataset counts, file hashes, source provenance, and industry list.
- `SCHEMA.md`: field-level schema notes.

## Counts

The dataset contains:

- 30 industries
- 300 synthetic company policy worlds
- 300 deployment system prompts
- 8,857 policy rules
- 4,357 allowed policy rules
- 4,500 prohibited policy rules

## Reproducibility

Regenerate this directory from the repository source files with:

```bash
python scripts/export_paper_dataset.py
```

The exporter reads:

- `data/compass_policies/compass_policies_final.jsonl`
- `data/compass_policies/company_system_prompts.jsonl`
- `data/compass_policies/dataset_summary.json`
- `copal/prompts.py`

The release tests verify that the committed dataset counts, file hashes, and prompt-template copy match the current repository state.

## License

Dataset files are released under the repository license unless otherwise noted.
