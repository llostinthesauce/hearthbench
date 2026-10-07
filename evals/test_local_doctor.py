"""Doctor checks for failures that otherwise appear only when a model launches."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import local_doctor as doctor


def _write_weight(path: Path) -> None:
    header = b'{"tensor":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
    prefix = len(header).to_bytes(8, "little") + header
    path.write_bytes(prefix + b"\0" * (1024 * 1024 - len(prefix)))


def _config(tmp_path: Path) -> dict:
    model_root = tmp_path / "models"
    return {
        "models": {
            "root": model_root,
            "registry": tmp_path / "models.local.json",
        },
        "endpoints": {
            "llamacpp": {"base_url": "http://127.0.0.1:8080/v1"},
            "mlx": {"base_url": "http://127.0.0.1:8085/v1"},
            "lmstudio": {"base_url": "http://127.0.0.1:1234/v1"},
        },
        "external_projects": [],
    }


def _registry(path: Path, model_path: Path, backend: str = "mlx") -> None:
    entry = ({"repo": str(model_path), "quant": "4bit"} if backend == "mlx"
             else {"path": str(model_path), "quant": "Q4_K_M"})
    family = {
        "id": "fixture",
        "name": "Fixture",
        "family": "fixture",
        "gguf": [] if backend == "mlx" else [entry],
        "mlx": [entry] if backend == "mlx" else [],
    }
    path.write_text(json.dumps({"_schema_version": 2, "model_families": [family]}))


def _codes(checks):
    return {check.code: check for check in checks}


def test_model_checks_report_outside_root_registry_entry(tmp_path):
    config = _config(tmp_path)
    config["models"]["root"].mkdir()
    outside = tmp_path / "outside" / "model"
    outside.mkdir(parents=True)
    (outside / "config.json").write_text("{}")
    (outside / "tokenizer.json").write_text("{}")
    _write_weight(outside / "model.safetensors")
    _registry(config["models"]["registry"], outside)

    checks = doctor.check_models(config)

    assert _codes(checks)["model.outside_root"].severity == "FAIL"


def test_model_checks_report_metadata_only_directory(tmp_path):
    config = _config(tmp_path)
    partial = config["models"]["root"] / "org" / "partial-model"
    partial.mkdir(parents=True)
    (partial / "config.json").write_text("{}")
    (partial / "tokenizer.json").write_text("{}")
    config["models"]["registry"].write_text(
        json.dumps({"_schema_version": 2, "model_families": [
            {"id": "other-family", "gguf": [], "mlx": []}
        ]})
    )

    checks = doctor.check_models(config)

    item = _codes(checks)["model.incomplete"]
    assert item.severity == "WARN"
    assert "partial-model" in item.message


def test_endpoint_checks_reject_non_loopback_binding(tmp_path):
    config = _config(tmp_path)
    config["endpoints"]["mlx"]["base_url"] = "http://0.0.0.0:8085/v1"

    checks = doctor.check_endpoint_safety(config)

    assert _codes(checks)["endpoint.mlx.non_loopback"].severity == "FAIL"


def test_external_project_check_reports_missing_path(tmp_path):
    config = _config(tmp_path)
    config["external_projects"] = [{
        "name": "missing", "path": tmp_path / "gone", "model": "fixture",
    }]

    checks = doctor.check_external_projects(config)

    assert _codes(checks)["project.missing"].severity == "WARN"


def test_external_project_check_reports_stale_model_reference(tmp_path):
    config = _config(tmp_path)
    project_path = tmp_path / "project"
    project_path.mkdir()
    config["external_projects"] = [{
        "name": "context", "path": project_path, "model": "stale-family",
    }]
    config["models"]["registry"].write_text(
        json.dumps({"_schema_version": 2, "model_families": [
            {"id": "other-family", "gguf": [], "mlx": []}
        ]})
    )

    checks = doctor.check_external_projects(config)

    assert _codes(checks)["project.model_missing"].severity == "WARN"


def test_runtime_check_names_each_missing_program():
    checks = doctor.check_runtime_commands(which=lambda _name: None)

    codes = _codes(checks)
    assert codes["runtime.llama-server"].severity == "WARN"
    assert codes["runtime.lms"].severity == "WARN"
    assert "runtime.omlx" not in codes


def test_default_run_excludes_retired_omlx_and_opencode_checks(tmp_path, monkeypatch):
    config = _config(tmp_path)
    config["models"]["root"].mkdir()
    config["models"]["registry"].write_text(
        json.dumps({"_schema_version": 2, "model_families": []})
    )
    monkeypatch.setattr(doctor, "check_runtime_commands", lambda: [])

    codes = _codes(doctor.run(config))

    assert not any(code.startswith("omlx.") for code in codes)
    assert not any(code.startswith("opencode.") for code in codes)
    assert "endpoint.omlx.loopback" not in codes


def test_healthy_model_registry_has_no_failure(tmp_path):
    config = _config(tmp_path)
    model = config["models"]["root"] / "org" / "complete"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    (model / "tokenizer.json").write_text("{}")
    _write_weight(model / "model.safetensors")
    _registry(config["models"]["registry"], model)

    checks = doctor.check_models(config)

    assert not [check for check in checks if check.severity == "FAIL"]
    assert _codes(checks)["models.registry"].severity == "OK"


def test_json_report_never_contains_a_secret():
    report = doctor.as_json([
        doctor.Check("auth", "OK", "configured", {"Authorization": "Bearer secret"})
    ])

    assert "secret" not in report
    assert "<redacted>" in report
