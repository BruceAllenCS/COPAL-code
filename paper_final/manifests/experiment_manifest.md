# Final Paper Experiment Manifest

Original artifacts remain under `runs/experiments`. This manifest only marks what should be read as final, provenance-only, or still running.

## Final Results

- `table2_ablation_30c_shard0_20260513`
  - role: Table2 ablation source shard
  - path: `runs/experiments/table2_ablation_30c_shard0_20260513`
  - summary: `runs/experiments/table2_ablation_30c_shard0_20260513/table2_summary.json`

- `table2_ablation_30c_shard1_20260513`
  - role: Table2 ablation source shard
  - path: `runs/experiments/table2_ablation_30c_shard1_20260513`
  - summary: `runs/experiments/table2_ablation_30c_shard1_20260513/table2_summary.json`

- `table2_ablation_30c_shard2_20260513`
  - role: Table2 ablation source shard
  - path: `runs/experiments/table2_ablation_30c_shard2_20260513`
  - summary: `runs/experiments/table2_ablation_30c_shard2_20260513/table2_summary.json`

- `table2_ablation_30c_hiddenfix_20260519`
  - role: Table2 ablation hidden-info repaired final artifact
  - path: `runs/experiments/table2_ablation_30c_hiddenfix_20260519`
  - summary: `runs/experiments/table2_ablation_30c_hiddenfix_20260519/table2_summary.json`
  - source/superseded artifact: `runs/experiments/table2_ablation_30c_merged_20260513/table2_summary.json`
  - judgments: 2880
  - error-rate labeling: automatic Gemini 3 Flash response judge
  - repair audit: `runs/experiments/table2_hidden_info_repair_20260519`
  - repair manifest: `runs/experiments/table2_ablation_30c_hiddenfix_20260519/hidden_info_repair_manifest.json`
  - hidden-info repairs applied: 62 selected items (16 raw-policy planning, 17 clause-only planning, 26 w/o facet query generation, 3 COPAL)
  - repair types: 13 existing-candidate replacements; 49 generated self-contained rewrites
  - post-repair hidden-info audit: 0 remaining issues across all four variants
  - construction-quality annotation: all four Table2 variants are sampled by `construction_quality_validation_20260515` (60 samples per variant) and double-annotated by GPT-5.5 and Claude Opus, with manual adjudication of GPT/Claude disagreements
  - response-judge annotation boundary: the four Table2 ablation variants' probe-error labels are not separately double-annotated or manually adjudicated; `llm_annotation_response_judge_v11_gpt55_claude240` audits the automatic response-judge pipeline on sampled COPAL response judgments, not per-variant Table2 ablation errors

- `table2_ablation_30c_merged_20260513`
  - role: Table2 ablation merged artifact, superseded by hidden-info repaired final artifact
  - path: `runs/experiments/table2_ablation_30c_merged_20260513`
  - summary: `runs/experiments/table2_ablation_30c_merged_20260513/table2_summary.json`
  - judgments: 2880
  - error-rate labeling: automatic Gemini 3 Flash response judge
  - status: provenance only; do not use for final paper Table2
  - construction-quality annotation: all four Table2 variants are sampled by `construction_quality_validation_20260515` (60 samples per variant) and double-annotated by GPT-5.5 and Claude Opus, with manual adjudication of GPT/Claude disagreements
  - response-judge annotation boundary: the four Table2 ablation variants' probe-error labels are not separately double-annotated or manually adjudicated; `llm_annotation_response_judge_v11_gpt55_claude240` audits the automatic response-judge pipeline on sampled COPAL response judgments, not per-variant Table2 ablation errors

- `construction_quality_validation_20260515`
  - role: Table2 construction-quality validation for Naturalness and Diagnosticity
  - path: `runs/experiments/construction_quality_validation_20260515`
  - summary: `runs/experiments/construction_quality_validation_20260515/construction_quality_summary.json`
  - samples: 240 (4 methods x 30 companies x 2 selected queries)
  - annotations: 480
  - annotators: `gpt-5.5`, `aws.claude-opus-4.7`
  - fully annotated: 240/240
  - agreement: Naturalness 93.3%; Diagnosticity 90.4%
  - adjudication frontend: `paper_final/annotation_adjudication/construction_quality_20260515/index.html`
  - adjudication display language: Chinese, with English backup at `paper_final/annotation_adjudication/construction_quality_20260515/disagreements.en.json`
  - disagreement records requiring manual adjudication: 39
  - manual adjudication export: `paper_final/annotation_adjudication/construction_quality_20260515/manual_adjudication_export.json`
  - post-adjudication summary: `paper_final/annotation_adjudication/construction_quality_20260515/manual_adjudication_summary.json`
  - detailed adjudication note: `paper_final/annotation_adjudication/construction_quality_20260515/manual_adjudication_detailed_results.md`
  - manual adjudication: 39/39 metric-level disagreements completed
  - post-adjudication pass rates: Naturalness 235/240 (97.9%); Diagnosticity 235/240 (97.9%); both checks 231/240 (96.3%)
  - interpretation: aggregate quality audit only; Diagnosticity is saturated and should not be used as a per-method comparison metric

- `table3_model_eval_30c_10model_seed12_20260514_shard0`
  - role: Table3 10-model evaluation shard
  - path: `runs/experiments/table3_model_eval_30c_10model_seed12_20260514_shard0`
  - summary: `runs/experiments/table3_model_eval_30c_10model_seed12_20260514_shard0/table3_summary.json`
  - completed companies: 10
  - judgments: 3000

- `table3_model_eval_30c_10model_seed12_20260514_shard1`
  - role: Table3 10-model evaluation shard
  - path: `runs/experiments/table3_model_eval_30c_10model_seed12_20260514_shard1`
  - summary: `runs/experiments/table3_model_eval_30c_10model_seed12_20260514_shard1/table3_summary.json`
  - completed companies: 10
  - judgments: 3000

- `table3_model_eval_30c_10model_seed12_20260514_shard2`
  - role: Table3 10-model evaluation shard
  - path: `runs/experiments/table3_model_eval_30c_10model_seed12_20260514_shard2`
  - summary: `runs/experiments/table3_model_eval_30c_10model_seed12_20260514_shard2/table3_summary.json`
  - completed companies: 10
  - judgments: 3000

- `table3_model_eval_30c_10model_seed12_20260514_merged`
  - role: Table3 composed-policy evaluation source artifact plus case examples and surface controls
  - path: `runs/experiments/table3_model_eval_30c_10model_seed12_20260514_merged`
  - summary: `runs/experiments/table3_model_eval_30c_10model_seed12_20260514_merged/table3_summary.json`
  - source judgments: 9000 across 10 response-model runs
  - paper-view exclusion: `gemini-3-flash-preview` is excluded from reported downstream models because it is the automatic response judge
  - paper-view judgments: 8100 across 9 downstream response models
  - paper-view overall error rate: 33.14%
  - error-rate labeling: automatic Gemini 3 Flash response judge
  - sampled audit: response-judge labels are audited by `llm_annotation_response_judge_v11_gpt55_claude240` on 240 sampled response judgments; GPT/Claude disagreements are manually adjudicated in `paper_final/annotation_adjudication/response_judge_v11`

- `table3_judge_sensitivity_300case_9model_5judge_20260521`
  - role: Table3 five-judge sensitivity audit for reviewer concern about single-judge dependence
  - path: `runs/experiments/table3_judge_sensitivity_300case_9model_5judge_20260521`
  - summary: `runs/experiments/table3_judge_sensitivity_300case_9model_5judge_20260521/judge_sensitivity_summary.json`
  - paper-ready write-up: `paper_final/manifests/judge_sensitivity_writeup.md`
  - sampled cases: 300 (10 per company)
  - sampled response judgments per judge: 2700
  - downstream response models: 9, excluding `gemini-3-flash-preview`
  - alternative judges: `gpt-5.5`, `aws.claude-opus-4.7`, `gemini-3.1-pro-preview`, `deepseek-v3.2-tencent`
  - provider safety blocks: GPT-5.5 blocked 1/2700 judge requests; metrics requiring GPT use 2699 judgments
  - unresolved judge refusals: DeepSeek refused 1/2700 judge requests; metrics requiring DeepSeek use 2699 judgments
  - rank robustness vs Gemini judge: Claude Kendall tau 0.889 / Spearman rho 0.967 / max rank shift 1; Gemini Pro 0.889 / 0.967 / 1; DeepSeek 0.722 / 0.867 / 3; GPT 0.778 / 0.900 / 2
  - five-judge majority error: 42.55% overall, Kendall tau 0.889 / Spearman rho 0.967 / max rank shift 1 vs Gemini sample ranking
  - pairwise agreement vs Gemini Flash: Claude 90.6%, Gemini Pro 84.4%, DeepSeek 73.0%, GPT 62.2%
  - interpretation: absolute error rates are judge-family sensitive because GPT-5.5 and DeepSeek are much stricter; model ranking remains stable under Claude, Gemini Pro, and the five-judge majority

- `paired_single_composed_table3_30c_2model_seed12_20260514_shard0`
  - role: Table3 paired single-vs-composed shard
  - path: `runs/experiments/paired_single_composed_table3_30c_2model_seed12_20260514_shard0`
  - summary: `runs/experiments/paired_single_composed_table3_30c_2model_seed12_20260514_shard0/paired_single_composed_summary.json`
  - completed companies: 10
  - composed judgments: 600
  - single-policy projection judgments: 1514

- `paired_single_composed_table3_30c_2model_seed12_20260514_shard1`
  - role: Table3 paired single-vs-composed shard
  - path: `runs/experiments/paired_single_composed_table3_30c_2model_seed12_20260514_shard1`
  - summary: `runs/experiments/paired_single_composed_table3_30c_2model_seed12_20260514_shard1/paired_single_composed_summary.json`
  - completed companies: 10
  - composed judgments: 600
  - single-policy projection judgments: 1524

- `paired_single_composed_table3_30c_2model_seed12_20260514_shard2`
  - role: Table3 paired single-vs-composed shard
  - path: `runs/experiments/paired_single_composed_table3_30c_2model_seed12_20260514_shard2`
  - summary: `runs/experiments/paired_single_composed_table3_30c_2model_seed12_20260514_shard2/paired_single_composed_summary.json`
  - completed companies: 10
  - composed judgments: 600
  - single-policy projection judgments: 1592

- `paired_single_composed_table3_30c_2model_seed12_20260514_merged`
  - role: Table3 paired single-vs-composed merged artifact plus confound controls
  - path: `runs/experiments/paired_single_composed_table3_30c_2model_seed12_20260514_merged`
  - summary: `runs/experiments/paired_single_composed_table3_30c_2model_seed12_20260514_merged/paired_single_composed_summary.json`
  - composed judgments: 1800
  - single-policy projection judgments: 4630

- `paired_single_composed_table3_30c_3model_seed12_20260520_merged`
  - role: Supplemental paired single-vs-composed artifact for added models (`gpt-5.5`, `aws.claude-sonnet-4.6`, `deepseek-v3.2-tencent`)
  - path: `runs/experiments/paired_single_composed_table3_30c_3model_seed12_20260520_merged`
  - summary: `runs/experiments/paired_single_composed_table3_30c_3model_seed12_20260520_merged/paired_single_composed_summary.json`
  - completed companies: 30
  - composed judgments: 2700
  - single-policy projection judgments: 6945
  - execution note: transient Friday/Bedrock Claude access errors were made retryable in `copal/llm.py` after a successful Claude smoke test showed the model was not stably unavailable.

- `paired_single_composed_table3_30c_5model_seed12_20260520_merged`
  - role: Current paper paired single-vs-composed merged artifact; combines the original 2-model result with the added 3-model result
  - path: `runs/experiments/paired_single_composed_table3_30c_5model_seed12_20260520_merged`
  - summary: `runs/experiments/paired_single_composed_table3_30c_5model_seed12_20260520_merged/paired_single_composed_summary.json`
  - composed judgments: 4500
  - single-policy projection judgments: 11575

## Supplemental / Appendix Results

- `real_chatbot_copal_inputs_20260521`
  - role: COPAL policy/prompt inputs reconstructed from real public chatbot policy and instruction material for the deployed-chatbot probe
  - path: `runs/experiments/real_chatbot_copal_inputs_20260521`
  - policies: `runs/experiments/real_chatbot_copal_inputs_20260521/policies.jsonl`
  - prompts: `runs/experiments/real_chatbot_copal_inputs_20260521/prompts.jsonl`
  - chatbot mapping: `runs/experiments/real_chatbot_copal_inputs_20260521/company_chatbot_mapping.json`
  - public deployments: City of Tampa ASK TAMI, City and County of Denver Sunny, City of Seabrook Ask Sunny
  - policy counts: Tampa 13 clauses; Denver 13 clauses; Seabrook 16 clauses

- `real_chatbot_copal_construction_30q_20260521`
  - role: Preliminary deployed-chatbot COPAL probe
  - path: `runs/experiments/real_chatbot_copal_construction_30q_20260521`
  - run README: `runs/experiments/real_chatbot_copal_construction_30q_20260521/README.md`
  - detailed paper-final note: `paper_final/manifests/deployed_chatbot_probe_20260521.md`
  - construction summary: `runs/experiments/real_chatbot_copal_construction_30q_20260521/pilot_summary.json`
  - external evaluation summary: `runs/experiments/real_chatbot_copal_construction_30q_20260521/external_evaluation_summary.json`
  - external evaluation manifest: `runs/experiments/real_chatbot_copal_construction_30q_20260521/external_evaluation_manifest.json`
  - public deployed chatbots: 3
  - candidate queries: 186
  - COPAL-selected benchmark items: 90
  - selected items per chatbot: 30
  - external chatbot responses: 90
  - COPAL response judgments: 90
  - judge model: `gemini-3-flash-preview`
  - overall result: 46/90 correct, 51.1% accuracy, 44/90 errors
  - interpretation: supplemental deployment-facing evidence that COPAL-style policy-composition probes can be run against public chatbot deployments; not a general deployed-chatbot failure-rate claim
  - safety note: generated queries are sanitized before public endpoint submission to remove address-like strings, IDs, emails, and phone numbers, while preserving the original COPAL-generated query in the artifact

## Provenance Only

- `paired_single_composed_from_table2_20260513`: Legacy 12-case paired result reused as seed prefill provenance
  - path: `runs/experiments/paired_single_composed_from_table2_20260513`
  - summary: `runs/experiments/paired_single_composed_from_table2_20260513/paired_single_composed_summary.json`

- `non_interacting_multiclause_control_30c_5per_20260514`: Non-interacting multi-clause control (clauses do not interact). Used as a confound control showing that the paired single-vs-composed gap is not explained by simply placing multiple clauses in the same prompt.
  - path: `runs/experiments/non_interacting_multiclause_control_30c_5per_20260514`
  - summary: `runs/experiments/non_interacting_multiclause_control_30c_5per_20260514/non_interacting_control_summary.json`
  - completed companies: 30
  - benchmark items: 150
  - judgments: 300 (2 response models)
  - models: `Doubao-Seed-2.0-pro`, `gemini-3.1-pro-preview`
  - judge: `gemini-3-flash-preview`
  - control result: 1.33% error rate for both response models (paired composed-policy: 43.89% Doubao / 18.33% gemini)

- `llm_human_validation_20260514_v2`: LLM annotation validation main delivery for clause_grounding, composition_interaction, and handling_contract
  - path: `runs/experiments/llm_human_validation_20260514_v2`
  - summary: `runs/experiments/llm_human_validation_20260514_v2/llm_human_validation_summary.json`
  - annotators: `gpt-5.5`, `aws.claude-opus-4.7`
  - sample count: 640 (clause_grounding 120, composition_interaction 160, handling_contract 120, response_judge_reliability 240)
  - fully annotated: 483/640
  - completed tasks: clause_grounding 120/120 (agreement 80.8%); composition_interaction 160/160 (agreement 96.3%); handling_contract 120/120 (agreement 100%)
  - response_judge_reliability in this run is partial/provenance only: 83/240 fully annotated

- `llm_annotation_response_judge_v11_gpt55_claude240`: LLM annotation result for response_judge_reliability
  - path: `runs/experiments/llm_annotation_response_judge_v11_gpt55_claude240`
  - summary: `runs/experiments/llm_annotation_response_judge_v11_gpt55_claude240/llm_human_validation_summary.json`
  - role: sampled audit of the automatic response-judge labels used for COPAL response-level error estimates
  - annotators: `gpt-5.5`, `aws.claude-opus-4.7`
  - sample count: 240
  - fully annotated: 240/240
  - agreement: 77.9%
  - manual adjudication export: `paper_final/annotation_adjudication/response_judge_v11/manual_adjudication_export.json`
  - post-adjudication summary: `paper_final/annotation_adjudication/response_judge_v11/manual_adjudication_summary.json`
  - detailed audit note: `paper_final/annotation_adjudication/response_judge_v11/reliability_detailed_results.md`
  - manual adjudication: 53/53 GPT/Claude disagreement records completed
  - final response-judge reliability: Gemini judge matches the final reference on 180/240 samples (75.0%)
  - interpretation: moderate agreement with lenient bias; 59 false-correct judge labels vs 1 false-incorrect judge label
  - paper wording: COPAL response-level error rates are automatic-judge estimates with sampled audit support; Table2 ablation probe-error labels in the hidden-info repaired final artifact are not separately double-annotated or manually adjudicated per variant

## Provenance Only

- `paired_single_composed_from_table2_20260513`: Legacy 12-case paired result reused as seed prefill provenance
  - path: `runs/experiments/paired_single_composed_from_table2_20260513`
  - summary: `runs/experiments/paired_single_composed_from_table2_20260513/paired_single_composed_summary.json`

- `llm_human_validation_20260514` (v1): Initial LLM annotation run; superseded by `v2`. Only clause_grounding 118/120 annotated; other 3 tasks at 0. Annotator positive-rate divergence was large (Opus 58.5% vs GPT 22.0% on clause_grounding) before the v2 protocol fix.
  - path: `runs/experiments/llm_human_validation_20260514`

## Supplemental Running

- `llm_human_validation_20260514_v3_judge`: Earlier repeat of `response_judge_reliability` only, 240 samples; 239/240 fully annotated. Superseded by `llm_annotation_response_judge_v11_gpt55_claude240`.
  - path: `runs/experiments/llm_human_validation_20260514_v3_judge`

- `llm_human_validation_20260514_v4_judge_pilot60` through `v8_judge_pilot60`: 60-sample pilots iterating on the response_judge_reliability annotator prompt/setup. Use these only as protocol-tuning provenance.
  - paths: `runs/experiments/llm_human_validation_20260514_v{4,5,6,7,8}_judge_pilot60`

- `llm_human_validation_20260514_v9_judge_pilot60_t0`: 60-sample pilot with temperature 0. Produced 0 annotations; only `annotation_errors/live_errors.jsonl` exists. Failed run; do not use.
  - path: `runs/experiments/llm_human_validation_20260514_v9_judge_pilot60_t0`
