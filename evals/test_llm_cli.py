"""Tests for the stable local-ai orchestration behaviors."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import llm


def _write_weight(path: Path) -> None:
    header = b'{"tensor":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
    prefix = len(header).to_bytes(8, "little") + header
    path.write_bytes(prefix + b"\0" * (1024 * 1024 - len(prefix)))


def _config(tmp_path: Path) -> dict:
    return {
        "models": {
            "root": tmp_path / "models",
            "registry": tmp_path / "models.local.json",
        },
        "endpoints": {
            "llamacpp": {"base_url": "http://127.0.0.1:9001/v1"},
            "mlx": {"base_url": "http://127.0.0.1:9002/v1"},
            "omlx": {"base_url": "http://127.0.0.1:9003/v1"},
            "lmstudio": {"base_url": "http://127.0.0.1:9004/v1"},
        },
    }


def test_status_probes_configured_urls_instead_of_fixed_ports(tmp_path):
    config = _config(tmp_path)
    seen = []

    rows = llm.collect_status(config, probe=lambda url, _headers: seen.append(url) or (False, None))

    assert seen == [
        "http://127.0.0.1:9001/v1",
        "http://127.0.0.1:9002/v1",
        "http://127.0.0.1:9004/v1",
    ]
    assert [row[0] for row in rows] == ["llama.cpp", "direct MLX", "LM Studio"]


def test_model_sync_writes_configured_registry_from_configured_root(tmp_path):
    config = _config(tmp_path)
    root = config["models"]["root"]
    model = root / "org" / "Fixture-4bit"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    (model / "tokenizer.json").write_text("{}")
    _write_weight(model / "model.safetensors")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({
        "_schema_version": 1,
        "defaults": {},
        "model_families": [{
            "id": "fixture",
            "name": "Fixture",
            "family": "fixture",
            "mlx_patterns": ["Fixture-4bit"],
            "gguf_patterns": [],
        }],
    }))

    count = llm.sync_models(config, catalog=catalog, known_only=False)

    registry = json.loads(config["models"]["registry"].read_text())
    assert count == 1
    assert registry["model_families"][0]["mlx"][0]["repo"] == str(model)


def test_auto_backend_uses_the_configured_registry(tmp_path, monkeypatch):
    registry = tmp_path / "custom-models.json"
    seen = []

    def resolve(_selector, backend, config_path):
        seen.append((backend, config_path))
        return {"exists": backend == "mlx"}

    monkeypatch.setattr(llm.model_registry, "resolve", resolve)

    assert llm._auto_backend("fixture", registry) == "mlx"
    assert seen == [("llamacpp", registry), ("mlx", registry)]


def test_help_alias_and_pasted_nbsp(capsys):
    import pytest
    for command in (["help"], ["help\u00a0"], ["--help"]):
        with pytest.raises(SystemExit) as exc:
            llm.main(command)
        assert exc.value.code == 0
        output = capsys.readouterr().out
        assert output.startswith("usage: llm ")
        assert "local-ai" not in output


def test_help_subcommand(capsys):
    import pytest
    with pytest.raises(SystemExit) as exc:
        llm.main(["help", "serve"])
    assert exc.value.code == 0
    assert "usage: llm serve" in capsys.readouterr().out


def test_explicit_context_forwarded_to_launcher(monkeypatch):
    seen = []
    monkeypatch.setattr(llm.os, "execv", lambda _path, cmd: seen.append(cmd))
    llm.main(["serve", "fixture", "--backend", "mlx", "--ctx", "131072", "--dry-run"])
    assert seen[-1][-2:] == ["--ctx", "131072"]


def test_model_inspection_resolves_alias_before_policy(monkeypatch, capsys):
    monkeypatch.setattr(llm.model_registry, 'resolve', lambda *args: {'path':'/models/example'})
    import model_policy
    monkeypatch.setattr(model_policy, 'resolve_policy', lambda path, backend: {'model_type':'fixture','path':path,'backend':backend})
    llm.main(['models','inspect','fixture','--backend','mlx'])
    assert json.loads(capsys.readouterr().out) == {'model_type':'fixture','path':'/models/example','backend':'mlx'}
