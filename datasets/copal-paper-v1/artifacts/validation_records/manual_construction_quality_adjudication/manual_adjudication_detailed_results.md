# Construction-Quality Manual Adjudication Results

This note records the post-adjudication result for the Table 2 construction-quality validation. It covers the Naturalness and Diagnosticity checks used to validate generated queries independently of downstream model failure rates.

## Source Artifacts

- LLM annotation run: `runs/experiments/construction_quality_validation_20260515`
- Disagreement frontend: `paper_final/annotation_adjudication/construction_quality_20260515/index.html`
- Manual adjudication export: `paper_final/annotation_adjudication/construction_quality_20260515/manual_adjudication_export.json`
- Post-adjudication summary: `paper_final/annotation_adjudication/construction_quality_20260515/manual_adjudication_summary.json`

## Protocol

Each sampled query is evaluated along two independent construction-quality dimensions:

- Naturalness: the query should read like a plausible user request, without explicit policy quotation, mechanical stitching, or artificial adversarial wording.
- Diagnosticity: the query should jointly activate multiple policy clauses and require a non-separable composed-policy handling decision with a clear expected/forbidden handling contract.

The final reference label for each metric decision is constructed as follows:

- If GPT-5.5 and Claude Opus agree, use their shared label.
- If GPT-5.5 and Claude Opus disagree, use the manual adjudication label from the frontend.

The validation covers 240 samples: 4 construction methods x 30 companies x 2 selected queries per method/company. Since each sample has two metrics, the final reference contains 480 metric-level decisions.

## Annotation And Adjudication Counts

| Quantity | Value |
| --- | ---: |
| Samples | 240 |
| Metric-level decisions | 480 |
| GPT/Claude-agreed metric decisions | 441 |
| Manually adjudicated metric disagreements | 39 / 39 |
| Manual adjudication pass decisions | 30 |
| Manual adjudication fail decisions | 9 |

Breakdown of the 39 manually adjudicated disagreements:

| Metric | Pass | Fail | Total |
| --- | ---: | ---: | ---: |
| Naturalness | 11 | 5 | 16 |
| Diagnosticity | 19 | 4 | 23 |
| Total | 30 | 9 | 39 |

## Final Construction-Quality Rates

| Final reference target | Pass | Fail | Pass rate |
| --- | ---: | ---: | ---: |
| Naturalness | 235 | 5 | 97.9% |
| Diagnosticity | 235 | 5 | 97.9% |
| Both Naturalness and Diagnosticity | 231 | 9 | 96.3% |

The post-adjudication result supports that the sampled queries are generally natural and diagnostic. This should be reported as an independent validation of construction quality, separate from downstream probe error.

Important interpretation: Diagnosticity is saturated under this protocol. It is a binary validity screen, not a method-comparison metric. All four construction variants in this audit are sampled after query selection and are shown to annotators with active clauses and expected/forbidden handling contracts. Under that setup, even ablated variants can pass Diagnosticity if they still produce a multi-clause, non-separable handling decision. Therefore, the Diagnosticity pass rate should not be used to argue that COPAL is better than the ablations.

## By-Method Final Pass Rates

| Method | Naturalness | Diagnosticity | Both metrics |
| --- | ---: | ---: | ---: |
| Raw policy planning | 60 / 60 = 100.0% | 58 / 60 = 96.7% | 58 / 60 = 96.7% |
| Clause-only planning | 56 / 60 = 93.3% | 59 / 60 = 98.3% | 56 / 60 = 93.3% |
| w/o facet query generation | 60 / 60 = 100.0% | 59 / 60 = 98.3% | 59 / 60 = 98.3% |
| COPAL | 59 / 60 = 98.3% | 59 / 60 = 98.3% | 58 / 60 = 96.7% |

These by-method rates should not be used in the main paper as comparative evidence. The near-identical Diagnosticity rates for clause-only planning, w/o facet query generation, and COPAL are a symptom of the metric's intended role as a pass/fail validity screen. Instead, they show that all compared construction variants pass the independent quality screen at high rates, so Table 2 can interpret downstream probe error as failure yield rather than as an artifact of obviously unnatural or invalid queries.

## Relationship To Table 2

Table 2 should separate construction coverage and probe yield:

- Coverage@12 measures how many relation-pattern x target-facet cells are covered under the same 12-query budget.
- Naturalness and Diagnosticity are independently validated with LLM annotation plus manual adjudication of disagreements.
- Probe error is downstream failure yield after the generated cases pass the construction-quality screen.

This avoids arguing that a higher downstream error rate alone proves construction quality. The construction-quality validation establishes that the generated queries are broadly plausible and diagnostic; the Table 2 probe-error column then measures how often those valid probes expose model failures.

## Recommended Paper Wording

Use wording like:

> We independently audit construction quality on 240 sampled queries across the four Table 2 construction variants. GPT-5.5 and Claude Opus annotate Naturalness and Diagnosticity for each query; the 39 metric-level disagreements are manually adjudicated. After adjudication, 235/240 queries pass Naturalness, 235/240 pass Diagnosticity, and 231/240 pass both checks. This validation is separate from downstream model error and supports interpreting Table 2 probe error as failure yield among independently validated test cases.

Avoid wording like:

- "COPAL is more natural than all baselines."
- "COPAL has higher Diagnosticity than all baselines."
- "Higher probe error itself proves construction quality."
- "Diagnosticity pass rate explains the Table 2 method difference."
- "The construction-quality annotation is human agreement."

The annotation protocol should be described as independent LLM annotation with manual adjudication of disagreements.
