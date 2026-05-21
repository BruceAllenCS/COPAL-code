from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_SIGNATURES: tuple[str, ...] = (
    "scope-restriction",
    "prerequisite-gating",
    "selective-disclosure",
    "workflow-transfer",
)

DEFAULT_FACETS: dict[str, tuple[str, ...]] = {
    "scope-restriction": (
        "boundary-overreach",
        "over-refusal",
        "semantic-leakage",
    ),
    "prerequisite-gating": (
        "skipped-gate",
        "wrong-scope-gate",
        "pre-gate-leakage",
    ),
    "selective-disclosure": (
        "protected-field-leakage",
        "all-withholding",
        "blurred-disclosure",
    ),
    "workflow-transfer": (
        "missed-transfer",
        "wrong-route",
        "latent-continuation",
    ),
}
DEFAULT_POLICIES_PATH = Path("data/compass_policies/compass_policies_final.jsonl")
DEFAULT_PROMPTS_PATH = Path("data/compass_policies/company_system_prompts.jsonl")

EXECUTION_MODES: tuple[str, ...] = ("deterministic", "live")
STOP_AFTER_STAGES: tuple[str, ...] = ("selection", "screening", "baselines", "audit", "evaluation")


def require_execution_mode(execution_mode: str) -> None:
    if execution_mode not in EXECUTION_MODES:
        allowed = ", ".join(EXECUTION_MODES)
        raise ValueError(f"Unsupported execution_mode: {execution_mode}. Expected one of: {allowed}")


def require_stop_after(stop_after: str) -> None:
    if stop_after not in STOP_AFTER_STAGES:
        allowed = ", ".join(STOP_AFTER_STAGES)
        raise ValueError(f"Unsupported stop_after: {stop_after}. Expected one of: {allowed}")


@dataclass(slots=True)
class RoleConfig:
    proposal_model: str
    canonicalization_model: str
    validator_model: str
    coverage_judge_model: str
    downstream_chatbot_model: str
    response_judge_model: str
    query_proposal_model: str = ""


def default_role_config() -> RoleConfig:
    return RoleConfig(
        proposal_model="qwen3.5-baidu",
        canonicalization_model="qwen3.5-baidu",
        validator_model="qwen3.5-baidu",
        coverage_judge_model="qwen3.5-baidu",
        downstream_chatbot_model="qwen3.5-baidu",
        response_judge_model="qwen3.5-baidu",
    )


def load_model_names(path: Path) -> tuple[str, ...]:
    if not path.exists():
        raise FileNotFoundError(f"model name file does not exist: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"model name file is empty: {path}")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"model name file must be a JSON array of model-name strings: {path}") from exc
    if not isinstance(payload, list):
        raise ValueError(f"model name file must be a JSON array of model-name strings: {path}")

    model_names: list[str] = []
    for index, value in enumerate(payload):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"model name at index {index} must be a non-empty string")
        model_names.append(value.strip())
    if not model_names:
        raise ValueError(f"model name file has no model names: {path}")
    seen: set[str] = set()
    for model_name in model_names:
        if model_name in seen:
            raise ValueError(f"duplicate model name in model name file: {model_name}")
        seen.add(model_name)
    return tuple(model_names)


@dataclass(slots=True)
class RunConfig:
    run_id: str
    company_key: str
    execution_mode: str
    policies_path: Path = DEFAULT_POLICIES_PATH
    prompts_path: Path = DEFAULT_PROMPTS_PATH
    model_name_path: Path = Path("model_name.json")
    runs_dir: Path = Path("runs")
    cache_dir: Path = Path("cache")
    model_names: tuple[str, ...] = ()
    random_seed: int = 0
    reference_subset_size: int = 24
    audit_sample_size: int = 24
    stop_after: str = "evaluation"
    live_max_workers: int = 1
    composition_limit_per_signature: int | None = None
    composition_adjudication_limit: int | None = None
    query_variants_per_facet: int = 1
    selection_variants_per_facet: int = 1
    screening_model: str = ""
    screening_min_score: float = 2.0
    screening_hard_suite_size: int = 0
    screening_use_hard_suite: bool = False
    live_smoke: bool = False
    smoke_rule_limit_per_side: int | None = None
    smoke_facet_limit_per_signature: int | None = None
    role_config: RoleConfig = field(default_factory=default_role_config)
    signatures: tuple[str, ...] = DEFAULT_SIGNATURES
    facet_library: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {key: tuple(values) for key, values in DEFAULT_FACETS.items()}
    )
