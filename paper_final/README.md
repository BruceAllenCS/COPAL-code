# COPAL Paper-Final Index

This directory contains paper-facing manifests and write-ups for the public COPAL release. The full raw run directories are not included; compact summary JSON files are provided under `results/paper_summaries/`.

## Reading Order

1. `manifests/code_manifest.md`: scripts, modules, tests, configuration, and data used by the paper pipeline.
2. `manifests/experiment_manifest.md`: final experiment families and provenance notes.
3. `manifests/final_result_snapshot.md`: compact paper-table numbers.
4. `manifests/judge_sensitivity_writeup.md`: paper-ready judge-family sensitivity text and tables.
5. `manifests/table2_ablation_diagnosis.md`: Table 2 design notes and case-level diagnosis.

## Included Summary Artifacts

- Table 2 ablation summary: `results/paper_summaries/table2_ablation/table2_summary.json`
- Table 3 model-evaluation summary: `results/paper_summaries/table3_model_eval/table3_summary.json`
- Paired single-policy vs composed-policy summary: `results/paper_summaries/paired_single_composed/paired_single_composed_summary.json`
- Judge-sensitivity summary: `results/paper_summaries/judge_sensitivity/judge_sensitivity_summary.json`
- Judge-disagreement analysis: `results/paper_summaries/judge_sensitivity/judge_disagreement_analysis.json`

## Boundary

Response-level error rates in the paper are automatic-judge estimates with sampled audit support. The public release includes the code and compact summary artifacts needed to inspect and reproduce the pipeline, but omits local provider caches, raw long-running response logs, and manual-adjudication web frontends.
