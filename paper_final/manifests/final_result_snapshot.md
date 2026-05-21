# Final Result Snapshot

This is a compact snapshot of the current paper-final numbers. The JSON summaries under `runs/experiments` remain the source of truth.

## Table2 Ablation

Source: `runs/experiments/table2_ablation_30c_hiddenfix_20260519/table2_summary.json`

Supersedes: `runs/experiments/table2_ablation_30c_merged_20260513/table2_summary.json`

- Companies: 30
- Judgments: 2880
- Overall probe error rate: 13.54%
- Doubao-Seed-2.0-pro probe error rate: 18.82%
- gemini-3.1-pro-preview probe error rate: 8.26%
- Hidden-info repair: 62 selected items repaired, 0 pending
- Repair types: 13 existing-candidate replacements; 49 generated self-contained rewrites
- Post-repair hidden-info audit: 0 remaining issues across all four variants

| Variant | Coverage@12 | Probe error |
| --- | ---: | ---: |
| raw_policy_planning | 8.97 | 4.03% |
| clause_only_planning | 7.80 | 7.50% |
| without_facet_query_generation | 7.00 | 10.69% |
| copal | 11.80 | 31.94% |

Use `Coverage@12` wording in the paper: average number of relation-pattern x target-facet cells covered by the 12 selected queries per company. Treat probe error as failure yield, not as construction quality by itself.

Table2 ablation annotation boundary: the construction-quality sample for Table2 covers all four variants and is double-annotated by GPT-5.5 and Claude Opus, with manual adjudication of GPT/Claude disagreements. The Table2 `Probe error` labels themselves are produced by the automatic Gemini 3 Flash response judge on all 2,880 judgments and are not manually corrected or separately double-annotated per ablation variant. The response-judge audit below validates the automatic judging pipeline on 240 sampled COPAL response judgments; it should not be described as per-variant adjudication of Table2 ablation error rates.

Table2 hidden-info repair boundary: after detecting selected cases whose judge-visible contract depended on facts absent from `query_text`, we repaired only those selected items and reran the affected downstream responses/judgments. The old merged Table2 artifact is retained as provenance only; use the hiddenfix artifact for paper tables.

## Construction Quality Validation

Source: `runs/experiments/construction_quality_validation_20260515/construction_quality_summary.json`

Post-adjudication source: `paper_final/annotation_adjudication/construction_quality_20260515/manual_adjudication_summary.json`

- Samples: 240 (4 methods x 30 companies x 2 selected queries)
- Annotations: 480
- Annotators: `gpt-5.5` and `aws.claude-opus-4.7`
- Fully annotated: 240 / 240
- Disagreement records manually adjudicated: 39 / 39
- Adjudication frontend: `paper_final/annotation_adjudication/construction_quality_20260515/index.html` (Chinese display, English backup retained as `disagreements.en.json`)
- Detailed adjudication note: `paper_final/annotation_adjudication/construction_quality_20260515/manual_adjudication_detailed_results.md`

Pre-adjudication LLM annotation:

| Variant | Naturalness agreement | Naturalness consensus pass | Diagnosticity agreement | Diagnosticity consensus pass |
| --- | ---: | ---: | ---: | ---: |
| raw_policy_planning | 95.0% | 100.0% | 90.0% | 98.1% |
| clause_only_planning | 81.7% | 100.0% | 81.7% | 100.0% |
| without_facet_query_generation | 100.0% | 100.0% | 93.3% | 100.0% |
| copal | 96.7% | 100.0% | 96.7% | 100.0% |

Post-adjudication final reference:

| Target | Pass rate |
| --- | ---: |
| Naturalness | 235 / 240 = 97.9% |
| Diagnosticity | 235 / 240 = 97.9% |
| Both Naturalness and Diagnosticity | 231 / 240 = 96.3% |

Use these post-adjudication rates as an aggregate quality audit, not as per-method comparison evidence. Diagnosticity is saturated under this protocol because the audited items already expose active clauses and expected/forbidden handling contracts to annotators. In the main paper, Table 2 should use Coverage@12 and Probe error as the method-comparison columns, with Naturalness/Diagnosticity validation reported separately as a pass/fail quality screen.

## Table3 Composed-Policy Evaluation

Source: `runs/experiments/table3_model_eval_30c_10model_seed12_20260514_merged/table3_summary.json`

Paper view: excludes `gemini-3-flash-preview` as a downstream response model because the same model family is used as the automatic response judge. The source artifact retains that 10th response-model run for provenance only.

- Companies: 30
- Cases: 900
- Downstream models reported in paper: 9
- Judgments reported in paper: 8100
- Overall error rate: 33.14%
- Overall PHS: 66.86%

| Model | Error rate |
| --- | ---: |
| deepseek-v3.2-tencent | 49.33% |
| MiniMax-M2.7 | 46.67% |
| Doubao-Seed-2.0-pro | 43.89% |
| qwen3.5-baidu | 42.89% |
| kimi-k2.6 | 28.89% |
| aws.claude-sonnet-4.6 | 27.00% |
| glm-5.1 | 25.33% |
| gemini-3.1-pro-preview | 18.33% |
| gpt-5.5 | 15.89% |

Error-rate provenance: Table3 error labels are produced by the automatic Gemini 3 Flash response judge. They are not manually corrected on the full 8,100 reported judgments. The reported model error rates should be described as automatic judge estimates with sampled response-level audit support, not as fully human-adjudicated labels.

## Judge-Family Sensitivity Audit

Source: `runs/experiments/table3_judge_sensitivity_300case_9model_5judge_20260521/judge_sensitivity_summary.json`

Paper-ready write-up: `paper_final/manifests/judge_sensitivity_writeup.md`

This audit samples 300 composed-policy cases (10 per company) and rejudges all 9 downstream model responses per sampled case. It is a held-out robustness check for Table3 ranking under alternative judge families; it does not replace the full-table Gemini judge estimate.

- Sampled cases: 300
- Sampled response judgments per judge: 2700
- Downstream response models: 9
- Alternative judges: `gpt-5.5`, `aws.claude-opus-4.7`, `gemini-3.1-pro-preview`, `deepseek-v3.2-tencent`
- GPT provider safety blocks: 1 / 2700; excluded from metrics requiring GPT
- DeepSeek unresolved judge refusals: 1 / 2700; excluded from metrics requiring DeepSeek
- Five-judge majority error: response is counted incorrect only when at least three of the five judges mark it incorrect

| Judge view | Judgments | Overall error | Kendall tau vs Gemini | Spearman rho vs Gemini | Max rank shift |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemini 3 Flash sample | 2700 | 32.56% | n/a | n/a | n/a |
| Claude Opus 4.7 | 2700 | 33.37% | 0.889 | 0.967 | 1 |
| Gemini 3.1 Pro | 2700 | 46.19% | 0.889 | 0.967 | 1 |
| DeepSeek v3.2 | 2699 | 55.09% | 0.722 | 0.867 | 3 |
| GPT-5.5 | 2699 | 69.73% | 0.778 | 0.900 | 2 |
| Five-judge majority | 2698 | 42.55% | 0.889 | 0.967 | 1 |

Pairwise judge agreement on the sampled set: Gemini Flash vs Claude 90.6%, Gemini Flash vs Gemini Pro 84.4%, Gemini Flash vs DeepSeek 73.0%, Gemini Flash vs GPT 62.2%. Interpretation: the main disagreement source is judge strictness rather than random inconsistency. Claude is closest to Gemini Flash; Gemini Pro, DeepSeek, and GPT are progressively stricter in absolute error labels. Ranking remains stable under Claude, Gemini Pro, and the five-judge majority; DeepSeek is stricter and shifts rankings more.

| Model | Gemini Flash | Claude Opus | Gemini Pro | DeepSeek | GPT-5.5 | Five-judge majority |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| deepseek-v3.2-tencent | 50.33% | 51.33% | 64.33% | 70.67% | 81.33% | 60.67% |
| MiniMax-M2.7 | 44.00% | 40.67% | 63.33% | 75.67% | 85.62% | 59.53% |
| Doubao-Seed-2.0-pro | 41.67% | 46.00% | 55.00% | 59.00% | 82.33% | 51.67% |
| qwen3.5-baidu | 41.00% | 40.33% | 58.33% | 65.00% | 75.67% | 53.67% |
| aws.claude-sonnet-4.6 | 27.33% | 26.67% | 41.00% | 45.15% | 69.33% | 35.45% |
| kimi-k2.6 | 27.33% | 32.67% | 38.00% | 52.67% | 60.00% | 38.00% |
| glm-5.1 | 25.00% | 23.00% | 40.33% | 46.33% | 66.33% | 34.33% |
| gemini-3.1-pro-preview | 19.67% | 21.33% | 30.67% | 45.33% | 62.67% | 27.33% |
| gpt-5.5 | 16.67% | 18.33% | 24.67% | 36.00% | 44.33% | 22.33% |

Disagreement concentration: GPT and DeepSeek mainly add errors in `scope-restriction` and `workflow-transfer` cases. The lowest-agreement facets against Gemini Flash are `all-withholding`, `boundary-overreach`, `latent-continuation`, `over-refusal`, and `semantic-leakage`, suggesting stricter judges penalize latent continuation, incomplete selective fulfillment, and boundary leakage more aggressively than Gemini Flash.

## Preliminary Deployed-Chatbot Probe

Source: `runs/experiments/real_chatbot_copal_construction_30q_20260521/external_evaluation_summary.json`

Detailed note: `paper_final/manifests/deployed_chatbot_probe_20260521.md`

This supplemental probe applies the COPAL construction and response-judge pipeline to three public deployed municipal chatbot assistants. Real public chatbot policy/instruction material was converted into COPAL `policies.jsonl` and `prompts.jsonl`; COPAL generated and selected 30 cases per chatbot; the resulting queries were submitted to the live chatbot endpoints; and responses were judged with the standard COPAL response judge. Before external submission, the runner sanitized address-like strings, IDs, emails, and phone numbers while preserving the original COPAL-generated query in the artifact.

- Public deployed chatbots: 3
- COPAL-selected cases: 90 total, 30 per chatbot
- External chatbot responses: 90
- COPAL response judgments: 90
- Overall accuracy: 46 / 90 = 51.1%
- Overall error rate: 44 / 90 = 48.9%
- Judge model: `gemini-3-flash-preview`

| Chatbot | Correct / Total | Accuracy | Error count |
| --- | ---: | ---: | ---: |
| City of Tampa ASK TAMI | 17 / 30 | 56.7% | 13 |
| City and County of Denver Sunny | 15 / 30 | 50.0% | 15 |
| City of Seabrook Ask Sunny | 14 / 30 | 46.7% | 16 |
| Total | 46 / 90 | 51.1% | 44 |

Error counts by target facet: `all-withholding` 6, `blurred-disclosure` 6, `pre-gate-leakage` 6, `skipped-gate` 6, `protected-field-leakage` 5, `missed-transfer` 4, `semantic-leakage` 3, `wrong-route` 3, `wrong-scope-gate` 3, `latent-continuation` 2.

Paper wording: frame this as a preliminary deployment-facing probe showing that COPAL-style policy-composition tests can be run against real public chatbot deployments. Do not treat it as a general deployed-chatbot failure-rate estimate: it covers only 3 deployments, 90 cases, and one automatic judge, and the reconstructed policy inputs come from public policy/instruction material rather than private deployment specifications.

## Paired Single-Policy Vs Composed-Policy Contrast

Source: `runs/experiments/paired_single_composed_table3_30c_5model_seed12_20260520_merged/paired_single_composed_summary.json`

- Companies: 30
- Composed judgments: 4500
- Single-policy projection judgments: 11575

| Model | Single-policy error | Composed-policy error | Gap | Composition-induced failure |
| --- | ---: | ---: | ---: | ---: |
| deepseek-v3.2-tencent | 4.15% | 49.33% | 45.19% | 43.33% |
| Doubao-Seed-2.0-pro | 3.54% | 43.89% | 40.35% | 38.78% |
| aws.claude-sonnet-4.6 | 3.07% | 27.00% | 23.93% | 25.00% |
| gemini-3.1-pro-preview | 2.42% | 18.33% | 15.91% | 16.22% |
| gpt-5.5 | 2.94% | 15.89% | 12.95% | 13.56% |

## Surface Controls

Source: `runs/experiments/paired_single_composed_table3_30c_2model_seed12_20260514_merged/confound_controls/surface_complexity_summary.json`

- Single-policy mean word count: 15.91; error rate: 2.98%
- Composed-policy mean word count: 76.37; error rate: 31.11%
- Shortest composed quartile error rate: 29.25%
- Longest composed quartile error rate: 30.65%
- Two-active-clause composed error rate: 31.36%
- Three-active-clause composed error rate: 31.09%

These controls support the claim that the paired gap is not explained by length alone.

## LLM Annotation And Error-Rate Audit

Sources:

- `runs/experiments/llm_human_validation_20260514_v2/llm_human_validation_summary.json`
- `runs/experiments/llm_annotation_response_judge_v11_gpt55_claude240/llm_human_validation_summary.json`
- `paper_final/annotation_adjudication/response_judge_v11/manual_adjudication_summary.json`
- `paper_final/annotation_adjudication/response_judge_v11/reliability_detailed_results.md`

Annotators: `gpt-5.5` and `aws.claude-opus-4.7`.

| Validation target | Fully annotated | LLM annotator agreement | Manual adjudication | Final judge/reference agreement |
| --- | ---: | ---: | ---: | ---: |
| Clause grounding | 120 / 120 | 80.8% | n/a | n/a |
| Composition interaction | 160 / 160 | 96.3% | n/a | n/a |
| Handling contract | 120 / 120 | 100.0% | n/a | n/a |
| Response judge reliability | 240 / 240 | 77.9% | 53 / 53 disagreements | 75.0% |

For response-judge reliability, the final reference uses GPT/Claude agreement for 187 originally agreed samples and manual adjudication for the 53 GPT/Claude disagreement samples. Gemini 3 Flash matches this final reference on 180/240 samples.

Detailed response-judge audit interpretation: Gemini's 75.0% agreement is not a high-reliability claim. The confusion matrix is asymmetric: 59 false-correct labels versus 1 false-incorrect label. This means the automatic judge is lenient on the audited hard cases, so main-table failure rates should be described as conservative failure-yield estimates rather than exact ground-truth error rates.

Paper wording: describe Table3 and COPAL-response error rates as automatic-judge estimates with a sampled response-level audit. For Table2 ablation specifically, describe only the construction-quality sample as double-annotated and manually adjudicated; do not claim that the four ablation variants' probe-error labels were double-annotated or manually adjudicated.
