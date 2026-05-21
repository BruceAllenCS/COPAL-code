from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from copal.io import ensure_directory, read_json, write_json

STAGE_MANIFEST_NAME = "stage_manifest.json"


def build_stage_fingerprint(*, input_paths: Sequence[Path], config: dict[str, Any]) -> str:
    payload = {
        "inputs": [
            {
                "path": str(path),
                "sha256": _file_sha256(path),
            }
            for path in sorted(input_paths, key=lambda item: str(item))
        ],
        "config": config,
    }
    return sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()


def run_checkpointed_stage(
    *,
    stage_name: str,
    stage_dir: Path,
    input_paths: Sequence[Path],
    config: dict[str, Any],
    output_files: Sequence[str],
    runner: Callable[[], dict[str, object]],
    manifest_file: str = STAGE_MANIFEST_NAME,
) -> dict[str, object]:
    ensure_directory(stage_dir)
    manifest_path = stage_dir / manifest_file
    fingerprint = build_stage_fingerprint(input_paths=input_paths, config=config)
    output_paths = [stage_dir / output_file for output_file in output_files]

    if _is_completed_manifest_reusable(
        manifest_path=manifest_path,
        stage_name=stage_name,
        fingerprint=fingerprint,
        output_paths=output_paths,
    ):
        manifest = read_json(manifest_path)
        return {
            "checkpoint_reused": True,
            "summary": dict(manifest["summary"]),
            "manifest_path": str(manifest_path),
        }

    started_at = _utc_now()
    write_json(
        manifest_path,
        {
            "stage_name": stage_name,
            "status": "running",
            "input_fingerprint": fingerprint,
            "config": config,
            "output_files": list(output_files),
            "started_at": started_at,
            "finished_at": None,
            "summary": {},
        },
    )
    try:
        summary = runner()
        _require_outputs(stage_name=stage_name, output_paths=output_paths)
    except Exception as exc:
        write_json(
            manifest_path,
            {
                "stage_name": stage_name,
                "status": "failed",
                "input_fingerprint": fingerprint,
                "config": config,
                "output_files": list(output_files),
                "started_at": started_at,
                "finished_at": _utc_now(),
                "summary": {},
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            },
        )
        raise

    write_json(
        manifest_path,
        {
            "stage_name": stage_name,
            "status": "completed",
            "input_fingerprint": fingerprint,
            "config": config,
            "output_files": list(output_files),
            "started_at": started_at,
            "finished_at": _utc_now(),
            "summary": dict(summary),
        },
    )
    return {
        "checkpoint_reused": False,
        "summary": dict(summary),
        "manifest_path": str(manifest_path),
    }


def _is_completed_manifest_reusable(
    *,
    manifest_path: Path,
    stage_name: str,
    fingerprint: str,
    output_paths: Sequence[Path],
) -> bool:
    if not manifest_path.exists():
        return False
    manifest = read_json(manifest_path)
    if manifest["stage_name"] != stage_name:
        return False
    if manifest["status"] != "completed":
        return False
    if manifest["input_fingerprint"] != fingerprint:
        return False
    return all(path.exists() for path in output_paths)


def _require_outputs(*, stage_name: str, output_paths: Sequence[Path]) -> None:
    missing = [str(path) for path in output_paths if not path.exists()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"{stage_name} did not write required output files: {joined}")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
