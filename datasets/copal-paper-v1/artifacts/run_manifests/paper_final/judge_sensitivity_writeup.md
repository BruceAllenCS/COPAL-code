# Judge Reliability And Sensitivity Write-Up

This note is the paper-ready write-up for the judge-family sensitivity experiment. It is meant to be copied into the paper or appendix, while the raw JSON artifacts remain the source of truth.

Source artifacts:

- Main summary: `runs/experiments/table3_judge_sensitivity_300case_9model_5judge_20260521/judge_sensitivity_summary.json`
- Disagreement analysis: `runs/experiments/table3_judge_sensitivity_300case_9model_5judge_20260521/judge_disagreement_analysis.md`
- Sampled judge inputs: `runs/experiments/table3_judge_sensitivity_300case_9model_5judge_20260521/sampled_response_judge_inputs.jsonl`
- Judge outputs: `runs/experiments/table3_judge_sensitivity_300case_9model_5judge_20260521/judge_*/response_judgments.jsonl`

## Paper Placement

Recommended placement:

- Main paper: a short paragraph and Table 1 below after the Table 3 composed-policy model evaluation.
- Appendix: Table 2, Table 3, and the disagreement-source discussion.

The purpose is not to replace the full Table 3 result. The full Table 3 still uses Gemini 3 Flash for all 8,100 reported judgments so that the main evaluation is fixed and reproducible. This audit checks whether the main model ranking collapses when the same responses are rejudged by other model families.

## Main-Text Paragraph

```latex
\paragraph{Judge-family sensitivity.}
Our full Table~\ref{tab:model_eval} uses Gemini 3 Flash as a fixed automatic response judge. To test whether the conclusions depend on a single judge family, we run a held-out judge-family sensitivity audit. We sample 300 composed-policy cases, stratified as 10 cases per company, and rejudge all nine downstream model responses for each case, yielding 2,700 response-level judgments per judge. In addition to the original Gemini 3 Flash labels, we use GPT-5.5, Claude Opus 4.7, Gemini 3.1 Pro, and DeepSeek v3.2 as alternative judges, and also report a five-judge majority label.

Absolute error rates are judge-sensitive: stricter judges assign higher failure rates, ranging from 33.37\% under Claude Opus to 69.73\% under GPT-5.5. However, the relative model ranking is stable. Claude Opus and Gemini 3.1 Pro both achieve Kendall's $\tau=0.889$ and Spearman's $\rho=0.967$ against the Gemini Flash ranking, with maximum rank shift 1. The five-judge majority has the same rank robustness ($\tau=0.889$, $\rho=0.967$, max shift 1) while producing a higher absolute error estimate of 42.55\%. These results indicate that the reported error rates should be read as automatic-judge estimates, but the main comparative conclusion is robust to alternative judge families.
```

## Table 1: Main Judge-Family Sensitivity Table

Markdown version:

| Judge view | Judgments | Overall error | Kendall tau vs Gemini | Spearman rho vs Gemini | Max rank shift |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemini 3 Flash sample | 2700 | 32.56% | n/a | n/a | n/a |
| Claude Opus 4.7 | 2700 | 33.37% | 0.889 | 0.967 | 1 |
| Gemini 3.1 Pro | 2700 | 46.19% | 0.889 | 0.967 | 1 |
| DeepSeek v3.2 | 2699 | 55.09% | 0.722 | 0.867 | 3 |
| GPT-5.5 | 2699 | 69.73% | 0.778 | 0.900 | 2 |
| Five-judge majority | 2698 | 42.55% | 0.889 | 0.967 | 1 |

LaTeX version:

```latex
\begin{table}[t]
\centering
\small
\begin{tabular}{lrrrr}
\toprule
Judge view & Error & Kendall $\tau$ & Spearman $\rho$ & Max shift \\
\midrule
Gemini 3 Flash & 32.56 & -- & -- & -- \\
Claude Opus 4.7 & 33.37 & 0.889 & 0.967 & 1 \\
Gemini 3.1 Pro & 46.19 & 0.889 & 0.967 & 1 \\
DeepSeek v3.2 & 55.09 & 0.722 & 0.867 & 3 \\
GPT-5.5 & 69.73 & 0.778 & 0.900 & 2 \\
Five-judge majority & 42.55 & 0.889 & 0.967 & 1 \\
\bottomrule
\end{tabular}
\caption{Judge-family sensitivity on 300 sampled composed-policy cases. Each judge re-evaluates all nine downstream model responses for the sampled cases. Correlations and rank shifts are computed against the Gemini 3 Flash sample ranking.}
\label{tab:judge_family_sensitivity}
\end{table}
```

Notes for the table caption or footnote:

- GPT-5.5 has one provider safety block, so GPT-dependent metrics use 2,699 judgments.
- DeepSeek has one unresolved judge refusal, so DeepSeek-dependent metrics use 2,699 judgments.
- Five-judge majority uses 2,698 complete rows with all five judge labels.

## Table 2: Per-Model Error Rates Under Each Judge

This table is better suited for the appendix unless space permits.

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

Interpretation:

- The high-error group remains the same across judge families: `deepseek-v3.2-tencent`, `MiniMax-M2.7`, `Doubao-Seed-2.0-pro`, and `qwen3.5-baidu`.
- The low-error group also remains stable: `gpt-5.5`, `gemini-3.1-pro-preview`, and `glm-5.1`.
- Most rank changes are adjacent swaps. The five-judge majority differs from Gemini Flash by at most one rank position.

## Table 3: Rank Stability By Judge

Rank 1 means the highest measured error rate.

| Model | Gemini Flash | Claude Opus | Gemini Pro | DeepSeek | GPT-5.5 | Five-judge majority | Rank range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| deepseek-v3.2-tencent | 1 | 1 | 1 | 2 | 3 | 1 | 1-3 |
| MiniMax-M2.7 | 2 | 3 | 2 | 1 | 1 | 2 | 1-3 |
| Doubao-Seed-2.0-pro | 3 | 2 | 4 | 4 | 2 | 4 | 2-4 |
| qwen3.5-baidu | 4 | 4 | 3 | 3 | 4 | 3 | 3-4 |
| aws.claude-sonnet-4.6 | 5 | 6 | 5 | 8 | 5 | 6 | 5-8 |
| kimi-k2.6 | 6 | 5 | 7 | 5 | 8 | 5 | 5-8 |
| glm-5.1 | 7 | 7 | 6 | 6 | 6 | 7 | 6-7 |
| gemini-3.1-pro-preview | 8 | 8 | 8 | 7 | 7 | 8 | 7-8 |
| gpt-5.5 | 9 | 9 | 9 | 9 | 9 | 9 | 9-9 |

Paper wording:

```latex
The rank changes are concentrated in adjacent model pairs rather than reversing the overall conclusion. The worst-performing group and the best-performing group are stable across judges; the strongest response model, GPT-5.5, remains rank 9 under every judge view, and the five-judge majority changes no model by more than one position relative to Gemini Flash.
```

## Disagreement-Source Analysis

Definitions used in the analysis:

- `under_enforcement`: the response answers or executes something it should not, such as prohibited disclosure, skipped gating, boundary leakage, or latent continuation after a nominal refusal/escalation.
- `over_enforcement`: the response withholds or refuses content that should be answered or routed, such as over-refusal, all-withholding, wrong route, or missing allowed content.
- `mixed`: both under-enforcement and over-enforcement are present.
- `other`: the judge marks the response incorrect for another contract violation or incomplete obligation.

Main observations:

- Complete all-five-judge rows: 2,698 / 2,700.
- Non-unanimous rows: 1,370 / 2,698 = 50.8%.
- Among non-unanimous rows, the dominant issue is 32.5% under-enforcement, 27.8% over-enforcement, 13.9% mixed, and 25.8% other.
- Pairwise agreement with Gemini Flash is highest for Claude Opus (90.6%), then Gemini Pro (84.4%), DeepSeek (73.0%), and GPT-5.5 (62.2%).

| Judge | Error rate | Under among errors | Over among errors | Mixed among errors | Other among errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemini Flash | 32.6% | 44.9% | 47.9% | 1.0% | 6.1% |
| Claude Opus | 33.4% | 48.6% | 40.7% | 0.1% | 10.5% |
| Gemini Pro | 46.2% | 40.2% | 43.4% | 1.8% | 14.7% |
| DeepSeek | 55.1% | 18.8% | 60.4% | 0.3% | 20.5% |
| GPT-5.5 | 69.7% | 46.4% | 23.3% | 8.9% | 21.4% |

Interpretation:

- Claude is nearly aligned with Gemini Flash in both absolute rate and ranking.
- Gemini Pro is moderately stricter but preserves the ranking.
- DeepSeek is mostly stricter on over-enforcement: it penalizes responses that refuse, withhold, or route too much.
- GPT-5.5 is the strictest judge overall: it penalizes under-enforcement and contract-completeness failures that Gemini Flash often accepts.
- Therefore, disagreement is mainly a difference in strictness and boundary interpretation, not random label noise.

Highest-disagreement facets:

| Facet | n | Non-unanimous rate |
| --- | ---: | ---: |
| boundary-overreach | 360 | 62.2% |
| latent-continuation | 234 | 54.3% |
| all-withholding | 252 | 54.0% |
| over-refusal | 432 | 53.9% |
| protected-field-leakage | 189 | 52.4% |
| skipped-gate | 162 | 49.4% |
| missed-transfer | 126 | 49.2% |
| semantic-leakage | 359 | 48.2% |

Appendix wording:

```latex
Disagreements are concentrated in boundary-sensitive partial-handling cases. The highest non-unanimous facets are boundary overreach, latent continuation, all-withholding, over-refusal, and protected-field leakage. These are precisely the cases where a response may appear superficially safe while still leaking restricted content, continuing a forbidden workflow, or refusing allowed content. This explains why stricter judges increase the absolute error rate while leaving the model ranking largely intact.
```

## Recommended Claim Boundary

Use the following claim:

```latex
The main Table~\ref{tab:model_eval} error rates are automatic-judge estimates, but the relative ranking and the conclusion that composed-policy prompts expose substantial policy-handling failures are robust under alternative judge families.
```

Avoid these stronger claims:

- Do not say the 8,100 Table 3 labels are fully human-adjudicated.
- Do not say Gemini 3 Flash is a high-accuracy ground-truth judge.
- Do not treat the five-judge majority as a replacement for the full Table 3 result; it is a sampled robustness estimate.
- Do not hide the absolute-rate sensitivity. The honest framing is that absolute failure rates move, while ranking and qualitative conclusions remain stable.
