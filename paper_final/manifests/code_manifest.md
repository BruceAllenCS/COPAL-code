# Final Paper Code Manifest

This file lists the code paths that matter for the current paper experiments. Other modules may still be imported transitively, but these are the stable reading entry points.

## Entry Scripts

- `scripts/run_table2_ablation_pilot.py`
- `scripts/run_table3_model_eval.py`
- `scripts/run_table3_judge_sensitivity.py`
- `scripts/build_real_chatbot_copal_inputs.py`
- `scripts/run_real_chatbot_copal_external_eval.py`
- `scripts/run_paired_single_composed_from_table3.py`
- `scripts/run_paired_single_composed_from_table2.py`
- `scripts/summarize_paper_tables_20260513.py`
- `scripts/run_llm_human_validation.py`
- `scripts/build_adjudication_frontend.py`
- `scripts/translate_adjudication_frontend_zh.py`
- `scripts/run_construction_quality_validation.py`
- `scripts/build_construction_quality_adjudication_frontend.py`
- `scripts/translate_construction_quality_adjudication_frontend_zh.py`

## Core Modules

- `copal/table2_ablation.py`
- `copal/table3_model_eval.py`
- `copal/fast_pilot.py`
- `copal/experiment_analysis.py`
- `copal/stages/difficulty_screening.py`
- `copal/stages/response_judgment.py`
- `copal/stages/downstream_chatbot.py`
- `copal/prompts.py`
- `copal/llm.py`
- `copal/io.py`
- `copal/config.py`
- `copal/data_sources.py`
- `copal/llm_human_validation.py`
- `copal/adjudication_frontend.py`
- `copal/adjudication_translation.py`
- `copal/construction_quality_validation.py`

## Focused Tests

- `tests/test_table2_ablation_pipeline.py`
- `tests/test_table3_model_eval.py`
- `tests/test_paired_single_from_table3.py`
- `tests/test_paired_single_from_table2.py`
- `tests/test_fast_pilot_pipeline.py`
- `tests/test_difficulty_screening_stage.py`
- `tests/test_llm_live_integration.py`
- `tests/test_llm_human_validation.py`
- `tests/test_adjudication_frontend.py`
- `tests/test_adjudication_translation.py`
- `tests/test_construction_quality_validation.py`

## Configuration And Data

- `model_name.json`
- `data/compass_policies/compass_policies_final.jsonl`
- `data/compass_policies/company_system_prompts.jsonl`
