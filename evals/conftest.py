"""Shared fixtures for tests that run the real launcher."""
from __future__ import annotations

import json
import os

import pytest


@pytest.fixture
def launcher_env(tmp_path):
    """A throwaway model root, registry and config for `serve_local.sh`.

    One family, `fixture`, with a minimal but complete MLX directory and GGUF
    file, so launcher tests exercise real selector resolution and model-path
    validation without depending on what happens to be installed — or on the
    machine's generated `configs/models.local.json` existing at all.
    """
    model_root = tmp_path / "models"
    gguf = model_root / "org" / "fixture-gguf" / "fixture.gguf"
    gguf.parent.mkdir(parents=True)
    gguf.write_bytes(b"GGUF" + b"x" * (1024 * 1024))
    mlx = model_root / "org" / "fixture-mlx"
    mlx.mkdir(parents=True)
    (mlx / "config.json").write_text("{}")
    (mlx / "tokenizer.json").write_text("{}")
    header = b'{"tensor":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
    prefix = len(header).to_bytes(8, "little") + header
    (mlx / "model.safetensors").write_bytes(prefix + b"\0" * (1024 * 1024 - len(prefix)))

    registry = tmp_path / "models.local.json"
    registry.write_text(json.dumps({
        "_schema_version": 2,
        "model_families": [{
            "id": "fixture",
            "name": "Fixture",
            "family": "fixture",
            "ctx_cap": 4096,
            "gguf": [{"path": str(gguf), "quant": "Q4_K_M"}],
            "mlx": [{"repo": str(mlx), "quant": "4bit"}],
        }],
    }))
    config = tmp_path / "local.toml"
    config.write_text(f'[models]\nroot = "{model_root}"\nregistry = "{registry}"\n')
    return {**os.environ, "LOCAL_AI_CONFIG": str(config)}
