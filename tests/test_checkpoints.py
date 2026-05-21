from pathlib import Path

from copal.checkpoints import build_stage_fingerprint, run_checkpointed_stage
from copal.io import read_json, write_json


def test_checkpointed_stage_reuses_completed_outputs_when_inputs_and_config_match(tmp_path: Path) -> None:
    input_path = tmp_path / "input.json"
    write_json(input_path, {"company_key": "demo", "rules": ["r1"]})
    stage_dir = tmp_path / "stage"
    calls: list[str] = []

    def run_stage() -> dict[str, object]:
        calls.append("called")
        write_json(stage_dir / "output.json", {"ok": True})
        return {"row_count": 1}

    first = run_checkpointed_stage(
        stage_name="demo_stage",
        stage_dir=stage_dir,
        input_paths=[input_path],
        config={"model": "m1", "execution_mode": "live"},
        output_files=["output.json"],
        runner=run_stage,
    )
    second = run_checkpointed_stage(
        stage_name="demo_stage",
        stage_dir=stage_dir,
        input_paths=[input_path],
        config={"model": "m1", "execution_mode": "live"},
        output_files=["output.json"],
        runner=run_stage,
    )

    manifest = read_json(stage_dir / "stage_manifest.json")
    assert calls == ["called"]
    assert first["checkpoint_reused"] is False
    assert second["checkpoint_reused"] is True
    assert second["summary"] == {"row_count": 1}
    assert manifest["status"] == "completed"
    assert manifest["stage_name"] == "demo_stage"


def test_checkpointed_stage_reruns_when_output_file_is_missing(tmp_path: Path) -> None:
    input_path = tmp_path / "input.json"
    write_json(input_path, {"company_key": "demo"})
    stage_dir = tmp_path / "stage"
    calls = 0

    def run_stage() -> dict[str, object]:
        nonlocal calls
        calls += 1
        write_json(stage_dir / "output.json", {"call": calls})
        return {"call": calls}

    run_checkpointed_stage(
        stage_name="demo_stage",
        stage_dir=stage_dir,
        input_paths=[input_path],
        config={"model": "m1"},
        output_files=["output.json"],
        runner=run_stage,
    )
    (stage_dir / "output.json").unlink()
    second = run_checkpointed_stage(
        stage_name="demo_stage",
        stage_dir=stage_dir,
        input_paths=[input_path],
        config={"model": "m1"},
        output_files=["output.json"],
        runner=run_stage,
    )

    assert calls == 2
    assert second["checkpoint_reused"] is False
    assert second["summary"] == {"call": 2}


def test_checkpointed_stage_reruns_when_config_changes(tmp_path: Path) -> None:
    input_path = tmp_path / "input.json"
    write_json(input_path, {"company_key": "demo"})
    stage_dir = tmp_path / "stage"
    calls = 0

    def run_stage() -> dict[str, object]:
        nonlocal calls
        calls += 1
        write_json(stage_dir / "output.json", {"call": calls})
        return {"call": calls}

    run_checkpointed_stage(
        stage_name="demo_stage",
        stage_dir=stage_dir,
        input_paths=[input_path],
        config={"model": "m1"},
        output_files=["output.json"],
        runner=run_stage,
    )
    second = run_checkpointed_stage(
        stage_name="demo_stage",
        stage_dir=stage_dir,
        input_paths=[input_path],
        config={"model": "m2"},
        output_files=["output.json"],
        runner=run_stage,
    )

    assert calls == 2
    assert second["checkpoint_reused"] is False


def test_checkpointed_stage_records_failure_without_swallowing_error(tmp_path: Path) -> None:
    input_path = tmp_path / "input.json"
    write_json(input_path, {"company_key": "demo"})
    stage_dir = tmp_path / "stage"

    def run_stage() -> dict[str, object]:
        raise RuntimeError("model request failed")

    try:
        run_checkpointed_stage(
            stage_name="live_stage",
            stage_dir=stage_dir,
            input_paths=[input_path],
            config={"execution_mode": "live"},
            output_files=["output.json"],
            runner=run_stage,
        )
    except RuntimeError as exc:
        assert "model request failed" in str(exc)
    else:
        raise AssertionError("stage error must be re-raised")

    manifest = read_json(stage_dir / "stage_manifest.json")
    assert manifest["status"] == "failed"
    assert manifest["error"]["type"] == "RuntimeError"
    assert manifest["error"]["message"] == "model request failed"


def test_stage_fingerprint_changes_when_input_file_changes(tmp_path: Path) -> None:
    input_path = tmp_path / "input.json"
    write_json(input_path, {"value": 1})
    first = build_stage_fingerprint(input_paths=[input_path], config={"model": "m1"})
    write_json(input_path, {"value": 2})
    second = build_stage_fingerprint(input_paths=[input_path], config={"model": "m1"})

    assert first != second
