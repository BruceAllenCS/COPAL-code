from __future__ import annotations

from collections.abc import Iterable

CANONICAL_EFFECTS: tuple[str, ...] = (
    "permit",
    "prohibit",
    "require-gate",
    "disclose",
    "withhold",
    "route",
    "other/unsupported",
)

EFFECT_ALIASES: dict[str, str] = {
    "allow": "permit",
    "allowed": "permit",
    "approve": "permit",
    "deny": "prohibit",
    "forbid": "prohibit",
    "forbidden": "prohibit",
    "require_gate": "require-gate",
    "gate": "require-gate",
    "escalate": "route",
    "escalation": "route",
    "redirect": "route",
    "reroute": "route",
    "routing": "route",
    "override": "other/unsupported",
    "exception": "other/unsupported",
    "authority-limit": "prohibit",
    "authority_limit": "prohibit",
    "authority": "prohibit",
    "unsupported": "other/unsupported",
    "other": "other/unsupported",
}

RELATION_PATTERN_PRIORITY: tuple[str, ...] = (
    "workflow-transfer",
    "prerequisite-gating",
    "selective-disclosure",
    "scope-restriction",
)


def normalize_effect_label(effect: object) -> str:
    normalized = str(effect).replace("_", "-").strip().lower()
    normalized = EFFECT_ALIASES.get(normalized, normalized)
    if normalized not in CANONICAL_EFFECTS:
        allowed = ", ".join(CANONICAL_EFFECTS)
        raise ValueError(f"Unsupported COPAL effect label: {effect}. Expected one of: {allowed}")
    return normalized


def normalize_effects(effects: Iterable[object]) -> tuple[str, ...]:
    return tuple(normalize_effect_label(effect) for effect in effects)


def relation_patterns_for_effects(effects: Iterable[object]) -> tuple[str, ...]:
    effect_set = {effect for effect in normalize_effects(effects) if effect != "other/unsupported"}
    patterns: list[str] = []
    if "route" in effect_set and effect_set - {"route"}:
        patterns.append("workflow-transfer")
    if "require-gate" in effect_set and effect_set & {"permit", "disclose", "route"}:
        patterns.append("prerequisite-gating")
    if "disclose" in effect_set and "withhold" in effect_set:
        patterns.append("selective-disclosure")
    if effect_set & {"permit", "disclose"} and effect_set & {"prohibit", "withhold"}:
        patterns.append("scope-restriction")
    return tuple(pattern for pattern in RELATION_PATTERN_PRIORITY if pattern in patterns)


def primary_relation_pattern(effects: Iterable[object]) -> str:
    patterns = relation_patterns_for_effects(effects)
    return patterns[0] if patterns else ""
