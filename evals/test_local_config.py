"""Machine configuration contracts that prevent path, port, and secret drift."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import local_config as lc


def test_missing_config_uses_loopback_and_lmstudio_root(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    config = lc.load_config(tmp_path / "absent.toml")

    assert config["models"]["root"] == tmp_path / ".lmstudio" / "models"
    assert config["endpoints"]["llamacpp"]["base_url"] == "http://127.0.0.1:8080/v1"
    assert config["endpoints"]["mlx"]["base_url"] == "http://127.0.0.1:8085/v1"
    assert config["endpoints"]["lmstudio"]["base_url"] == "http://127.0.0.1:1234/v1"
    # Exactly the endpoints this machine serves. :8000 used to be oMLX's
    # default and is Mycelium's backend now; nothing here may probe it.
    assert set(config["endpoints"]) == {"llamacpp", "mlx", "lmstudio"}


def test_toml_overrides_ports_and_reads_external_projects(tmp_path):
    config_path = tmp_path / "local.toml"
    config_path.write_text("""
[models]
root = "/Volumes/Models/lmstudio"

[endpoints.mlx]
base_url = "http://127.0.0.1:8071/v1"

[[external_projects]]
name = "life-goals"
path = "/private/life-goals/local"
model = "qwen3_27b_dense"
kind = "direct-mlx-kv-cache"
""")

    config = lc.load_config(config_path)

    assert config["models"]["root"] == Path("/Volumes/Models/lmstudio")
    assert config["endpoints"]["mlx"]["base_url"] == "http://127.0.0.1:8071/v1"
    assert config["external_projects"] == [{
        "name": "life-goals",
        "path": Path("/private/life-goals/local"),
        "model": "qwen3_27b_dense",
        "kind": "direct-mlx-kv-cache",
    }]


def test_path_within_model_root_rejects_sibling_prefix(tmp_path):
    root = tmp_path / "models"
    inside = root / "org" / "model"
    sibling = tmp_path / "models-private" / "model"

    assert lc.is_within(inside, root) is True
    assert lc.is_within(sibling, root) is False


def test_malformed_local_config_has_actionable_error(tmp_path):
    path = tmp_path / "local.toml"
    path.write_text("[models\n")

    try:
        lc.load_config(path)
        assert False, "expected a configuration error"
    except lc.ConfigError as exc:
        assert str(path) in str(exc)


def test_shell_values_expose_paths_and_ports_but_never_auth(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    config = lc.load_config(tmp_path / "absent.toml")

    values = lc.shell_values(config)

    assert values["LOCAL_MODEL_ROOT"] == str(tmp_path / ".lmstudio" / "models")
    assert values["LOCAL_LLAMACPP_HOST"] == "127.0.0.1"
    assert values["LOCAL_LLAMACPP_PORT"] == "8080"
    assert values["LOCAL_MLX_PORT"] == "8085"
    assert not any("KEY" in name or "TOKEN" in name or "AUTH" in name for name in values)
