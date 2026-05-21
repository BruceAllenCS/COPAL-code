from __future__ import annotations

from copy import deepcopy
from typing import Any

STOP_AFTER_ORDER = {
    "selection": 0,
    "screening": 1,
    "baselines": 2,
    "audit": 3,
    "evaluation": 4,
}


def manifests_match_for_resume(existing: dict[str, Any], requested: dict[str, Any]) -> bool:
    normalized_existing = _normalized_for_resume(existing)
    normalized_requested = _normalized_for_resume(requested)
    if _allows_stop_after_extension(normalized_existing, normalized_requested):
        normalized_existing.pop("stop_after", None)
        normalized_requested.pop("stop_after", None)
    return normalized_existing == normalized_requested


def _normalized_for_resume(manifest: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(manifest)
    live_client = normalized.get("live_client")
    if isinstance(live_client, dict) and live_client.get("provider") == "friday":
        live_client.setdefault("min_interval_seconds", 0.0)
    if normalized.get("stop_after") == "selection":
        normalized.pop("model_roster", None)
        normalized.pop("model_count", None)
    normalized.pop("company_workers", None)
    return normalized


def _allows_stop_after_extension(existing: dict[str, Any], requested: dict[str, Any]) -> bool:
    existing_stop_after = _stop_after_value(existing)
    requested_stop_after = _stop_after_value(requested)
    if existing_stop_after not in STOP_AFTER_ORDER or requested_stop_after not in STOP_AFTER_ORDER:
        return False
    return STOP_AFTER_ORDER[requested_stop_after] >= STOP_AFTER_ORDER[existing_stop_after]


def _stop_after_value(manifest: dict[str, Any]) -> str:
    return str(manifest.get("stop_after", "evaluation"))
