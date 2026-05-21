# Response-Judge Reliability Audit

This note records the detailed result for the response-judge reliability audit after manual adjudication of GPT/Claude disagreements.

## Source Artifacts

- LLM annotation run: `runs/experiments/llm_annotation_response_judge_v11_gpt55_claude240`
- Disagreement frontend: `paper_final/annotation_adjudication/response_judge_v11/index.html`
- Manual adjudication export: `paper_final/annotation_adjudication/response_judge_v11/manual_adjudication_export.json`
- Post-adjudication summary: `paper_final/annotation_adjudication/response_judge_v11/manual_adjudication_summary.json`

## Protocol

The audit evaluates the automatic response judge used for COPAL response-level error estimates. For each sampled model response, the annotators saw the user query, active policy clauses, expected/forbidden handling contract, and model response. The original Gemini judge label was not shown in the manual adjudication interface.

The final reference label is constructed as follows:

- If GPT-5.5 and Claude Opus agree, use their shared label.
- If GPT-5.5 and Claude Opus disagree, use the manual adjudication label from the frontend.

The audit covers 240 sampled COPAL response-judge validation records from the same response-level labeling pipeline used for the main composed-policy model evaluation. GPT-5.5 and Claude Opus agreed on 187 samples and disagreed on 53 samples. All 53 disagreement samples were manually adjudicated. This audit should not be described as per-variant manual adjudication of the Table 2 ablation probe-error labels.

## Main Result

| Quantity | Value |
| --- | ---: |
| Validation samples | 240 |
| GPT/Claude annotations | 480 |
| GPT/Claude agreement | 187 / 240 = 77.9% |
| Manually adjudicated disagreements | 53 / 53 |
| Final reference labels | 240 / 240 |
| Gemini judge vs final reference | 180 / 240 = 75.0% |

The 75.0% number should not be described as high reliability. It is better described as a moderate audit agreement with a clear, asymmetric error pattern.

## Confusion Matrix

Rows are Gemini automatic judge labels. Columns are final reference labels after GPT/Claude agreement plus manual adjudication.

| Gemini judge label | Final correct | Final incorrect | Row total |
| --- | ---: | ---: | ---: |
| Correct | 61 | 59 | 120 |
| Incorrect | 1 | 119 | 120 |
| Column total | 62 | 178 | 240 |

This shows that Gemini's errors are strongly asymmetric:

- False-correct cases: 59 / 240. Gemini marked the response correct, but the final reference marked it incorrect.
- False-incorrect cases: 1 / 240. Gemini marked the response incorrect, but the final reference marked it correct.

If policy-handling error is treated as the positive class, the automatic judge has:

| Metric | Value |
| --- | ---: |
| Error-detection precision | 119 / 120 = 99.2% |
| Error-detection recall | 119 / 178 = 66.9% |
| Correct-response specificity | 61 / 62 = 98.4% |
| Correct-label precision | 61 / 120 = 50.8% |

The key interpretation is that the automatic judge is highly precise when it calls a response incorrect, but it misses a substantial number of policy-handling errors by calling them correct.

## Where The Disagreement Concentrates

| Subset | Samples | Gemini match with final reference |
| --- | ---: | ---: |
| GPT/Claude agreed subset | 187 | 155 / 187 = 82.9% |
| GPT/Claude disagreement subset | 53 | 25 / 53 = 47.2% |

The lower reliability is concentrated in the hard cases where GPT-5.5 and Claude Opus also disagreed. This supports using manual adjudication for these ambiguous cases, and it cautions against presenting the automatic judge as an oracle.

## Implication For Main Experiment Results

The audit does not support a claim that the automatic judge is near-perfect. Instead, it supports a more careful claim:

- The judge has a measurable lenient bias.
- Most judge errors are false-correct labels, not false-incorrect labels.
- Therefore, COPAL composed-policy failure rates are unlikely to be inflated by judge strictness.
- If anything, the automatic judge may undercount policy-handling failures on difficult responses.

This means the reported failure rates should be framed as automatic-judge failure-yield estimates with sampled audit support, rather than exact full human-adjudicated ground-truth error rates.

## Recommended Paper Wording

Use wording like:

> We compute response-level policy-handling error rates with an automatic Gemini 3 Flash response judge, and audit this judge with independent GPT-5.5 and Claude Opus annotations on 240 sampled COPAL response judgments. The two LLM annotators agree on 187/240 samples (77.9%); the remaining 53 disagreements are manually adjudicated. Against the resulting final reference labels, the Gemini judge matches 180/240 cases (75.0%). The disagreement is highly asymmetric: 59 cases are false-correct judge labels while only 1 case is a false-incorrect label. Thus, the automatic judge is not treated as a perfect oracle; its observed bias is lenient, suggesting that reported failure rates are conservative rather than inflated.

Avoid wording like:

- "The automatic judge is highly reliable."
- "Gemini judge accuracy is sufficient to replace human review."
- "The 75.0% agreement validates all automatic labels."
- "The Table 2 ablation probe-error labels are double-annotated or manually adjudicated per variant."

## Boundary

The manual export should be used for correctness adjudication only. The exported `final_error_type` field is `none` for all 53 manually adjudicated records, so this artifact should not be used to report final error-type distributions.
