"""Launcher integration tests for current flags, configured ports, and secrets."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import llama_serve_menu as menu


def test_menu_rows_filters_unavailable_models(monkeypatch):
    variants = [
        {"family_id": "present", "backend": "mlx", "exists": True, "quant": "4bit", "name": "present-4bit", "path": "/models/present"},
        {"family_id": "absent", "backend": "mlx", "exists": False, "quant": "4bit", "name": "absent-4bit", "path": "/models/absent"},
    ]
    monkeypatch.setattr(menu.model_registry, "iter_models", lambda _path: variants)
    rows = menu._rows({"models": {"registry": "/unused"}})
    family_ids = [r["family_id"] for r in rows]
    assert "present" in family_ids
    assert "absent" not in family_ids


def test_menu_excludes_omlx_from_backends(monkeypatch):
    monkeypatch.setattr(menu.model_registry, "resolve", lambda selector, backend, registry: {"exists": True})
    backends = menu._supported_backends("fixture", {"models": {"registry": "/unused"}})
    assert "omlx" not in backends


def test_llamacpp_dry_run_uses_load_mode_not_deprecated_flags(tmp_path):
    model_root = tmp_path / "models"
    model = model_root / "org" / "fixture" / "fixture.gguf"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"GGUF" + b"x" * (1024 * 1024))
    registry = tmp_path / "models.local.json"
    registry.write_text(json.dumps({
        "_schema_version": 2,
        "model_families": [{
            "id": "fixture",
            "name": "Fixture",
            "family": "fixture",
            "ctx_cap": 4096,
            "gguf": [{"path": str(model), "quant": "Q4_K_M"}],
            "mlx": [],
        }],
    }))
    config = tmp_path / "local.toml"
    config.write_text(f'''\n[models]\nroot = "{model_root}"\nregistry = "{registry}"\n''')
    env = os.environ.copy()
    env["LOCAL_AI_CONFIG"] = str(config)

    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "serve_local.sh"), "fixture", "--dry-run"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--load-mode mlock" in result.stdout
    assert "--mlock" not in result.stdout
    assert "--no-mmap" not in result.stdout


def test_qwen_dry_runs_set_stable_reasoning_defaults_and_api_alias(tmp_path):
    model_root = tmp_path / "models"
    gguf = model_root / "org" / "qwen" / "qwen.gguf"
    gguf.parent.mkdir(parents=True)
    gguf.write_bytes(b"GGUF" + b"x" * (1024 * 1024))
    mlx = model_root / "org" / "qwen-mlx"
    mlx.mkdir(parents=True)
    (mlx / "config.json").write_text("{}")
    (mlx / "tokenizer.json").write_text("{}")
    header = b'{"tensor":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
    prefix = len(header).to_bytes(8, "little") + header
    (mlx / "model.safetensors").write_bytes(
        prefix + b"\0" * (1024 * 1024 - len(prefix))
    )
    registry = tmp_path / "models.local.json"
    registry.write_text(json.dumps({
        "_schema_version": 2,
        "model_families": [{
            "id": "qwen",
            "name": "Qwen",
            "family": "qwen3",
            "ctx_cap": 4096,
            "enable_thinking": False,
            "reasoning_effort": "medium",
            "reasoning_budget": 4096,
            "gguf": [{"path": str(gguf), "quant": "Q4_K_M"}],
            "mlx": [{"repo": str(mlx), "quant": "4bit"}],
        }],
    }))
    config = tmp_path / "local.toml"
    config.write_text(f'''\n[models]\nroot = "{model_root}"\nregistry = "{registry}"\n''')
    env = {**os.environ, "LOCAL_AI_CONFIG": str(config)}

    llama = subprocess.run(
        ["bash", str(ROOT / "scripts" / "serve_local.sh"), "qwen", "--backend", "llamacpp", "--dry-run"],
        cwd=ROOT, env=env, text=True, capture_output=True, check=False,
    )
    mlx_result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "serve_local.sh"), "qwen", "--backend", "mlx", "--dry-run"],
        cwd=ROOT, env=env, text=True, capture_output=True, check=False,
    )

    assert llama.returncode == 0, llama.stderr
    assert "--alias qwen.gguf" in llama.stdout
    assert "--reasoning-effort medium" in llama.stdout
    assert "--reasoning-budget 4096" in llama.stdout
    assert "--reasoning off" in llama.stdout
    assert mlx_result.returncode == 0, mlx_result.stderr
    assert "--chat-template-args" in mlx_result.stdout
    assert "reasoning_effort" in mlx_result.stdout
    assert "medium" in mlx_result.stdout
    assert "enable_thinking" in mlx_result.stdout
    assert "false" in mlx_result.stdout


def test_serve_script_does_not_shell_eval_user_model_selector():
    source = (ROOT / "scripts" / "serve_local.sh").read_text()

    assert 'eval echo "$MODEL_ARG"' not in source


def test_menu_keeps_each_quantization_selectable(monkeypatch):
    variants = [
        {"family_id": "qwen", "backend": "mlx", "exists": True,
         "quant": quant, "name": f"qwen-{quant}", "path": f"/models/qwen-{quant}"}
        for quant in ("4bit", "8bit")
    ]
    monkeypatch.setattr(menu.model_registry, "iter_models", lambda _path: variants)
    monkeypatch.setattr(menu, "_choose", lambda rows, _title, _label: rows[1])

    assert menu._choose_variant("qwen", "mlx-kv", {"models": {"registry": "/unused"}}) == "/models/qwen-8bit"


def test_explicit_model_path_cannot_bypass_canonical_storage(tmp_path):
    model = tmp_path / "outside.gguf"
    model.write_bytes(b"GGUF" + b"x" * (1024 * 1024))
    root = tmp_path / "canonical"
    root.mkdir()
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"_schema_version": 2, "model_families": []}))
    config = tmp_path / "local.toml"
    config.write_text(f'[models]\nroot = "{root}"\nregistry = "{registry}"\n')

    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "serve_local.sh"), str(model), "--dry-run"],
        env={**os.environ, "LOCAL_AI_CONFIG": str(config)}, cwd=ROOT,
        capture_output=True, text=True, timeout=15,
    )

    assert result.returncode != 0
    assert "configured LM Studio root" in result.stderr


_SLOW_TO_EXIT_LISTENER = """
import signal, socket, sys, time
sock = socket.socket()
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(("127.0.0.1", 0))
sock.listen()
print(sock.getsockname()[1], flush=True)
def _linger(*_):
    time.sleep(1.5)   # an MLX server releasing Metal buffers
    sys.exit(0)
signal.signal(signal.SIGTERM, _linger)
while True:
    time.sleep(0.1)
"""


def test_picker_waits_for_the_port_to_free_before_launching(monkeypatch):
    """The replacement server must not race the old one for its port.

    MLX servers take seconds to exit after SIGTERM. Sending the signal and
    exec'ing the launcher straight away left the new server to fail its bind
    against a process that was still shutting down.
    """
    import serving_lifecycle

    listener = subprocess.Popen(
        [sys.executable, "-c", _SLOW_TO_EXIT_LISTENER], stdout=subprocess.PIPE, text=True,
    )
    try:
        port = int(listener.stdout.readline())
        for _ in range(50):
            if serving_lifecycle.port_pids(port):
                break
            time.sleep(0.1)
        held_at_exec = []

        def fake_execv(_path, _argv):
            held_at_exec.append(serving_lifecycle.port_pids(port))
            raise SystemExit(0)

        monkeypatch.setattr(menu.os, "execv", fake_execv)
        config = {"endpoints": {
            "llamacpp": {"base_url": f"http://127.0.0.1:{port}/v1"},
            "mlx": {"base_url": f"http://127.0.0.1:{port}/v1"},
        }}
        with pytest.raises(SystemExit):
            menu._launch("fixture", "llamacpp", str(port), None, False, True, config)

        assert held_at_exec == [()]
    finally:
        listener.kill()
        listener.wait(timeout=5)


@pytest.mark.parametrize("selector,relative", [
    ("qwen3_27b_dense", "bartowski/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q5_K_M.gguf"),
    ("qwen3_8_27b_abliterated",
     "OBLITERATUS/Qwen3.8-27B-OBLITERATED/Qwen3.8-27B-OBLITERATED-Q6_K.gguf"),
])
def test_qwen38_27b_uses_native_context_by_default(tmp_path, selector, relative):
    """Historical local recommendations are advisory, not serving caps."""
    import discover_models

    model_root = tmp_path / "models"
    gguf = model_root / relative
    gguf.parent.mkdir(parents=True)
    gguf.write_bytes(b"GGUF" + b"x" * (1024 * 1024))
    registry = tmp_path / "models.local.json"
    registry.write_text(json.dumps(discover_models.discover(
        [model_root], ROOT / "configs" / "model_catalog.json", include_unmatched=True,
    )))
    config = tmp_path / "local.toml"
    config.write_text(f'[models]\nroot = "{model_root}"\nregistry = "{registry}"\n')

    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "serve_local.sh"), selector,
         "--backend", "llamacpp", "--dry-run"],
        env={**os.environ, "LOCAL_AI_CONFIG": str(config)},
        capture_output=True, text=True, timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    command = next(line for line in result.stdout.splitlines() if line.startswith("DRY RUN:"))
    assert " -c 0 " in command


def test_llamacpp_default_has_one_full_context_slot(tmp_path):
    model_root = tmp_path / 'models'
    model_root.mkdir()
    model = model_root / 'fixture.gguf'
    model.write_bytes(b'GGUF' + b'x' * (1024 * 1024))
    registry = tmp_path / 'models.json'
    registry.write_text(json.dumps({'_schema_version': 2, 'model_families': []}))
    config = tmp_path / 'local.toml'
    config.write_text(f'[models]\nroot = "{model_root}"\nregistry = "{registry}"\n')
    result = subprocess.run(['bash', str(ROOT/'scripts/serve_local.sh'), str(model), '--dry-run'],
                            env={**os.environ, 'LOCAL_AI_CONFIG': str(config)},
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    import shlex
    args = shlex.split(result.stdout.split('DRY RUN:')[1])
    assert args[args.index('-np') + 1] == '1'
    assert args[args.index('--alias') + 1] == 'fixture.gguf'
    assert args[args.index('--min-p') + 1] == '0.0'


def test_explicit_embedding_mode_uses_embedding_endpoint_and_last_pooling(tmp_path):
    model_root = tmp_path / 'models'
    model_root.mkdir()
    model = model_root / 'Qwen3-Embedding-8B-Q8_0.gguf'
    model.write_bytes(b'GGUF' + b'x' * (1024 * 1024))
    registry = tmp_path / 'models.json'
    registry.write_text(json.dumps({'_schema_version': 2, 'model_families': []}))
    config = tmp_path / 'local.toml'
    config.write_text(f'[models]\nroot = "{model_root}"\nregistry = "{registry}"\n')
    result = subprocess.run(['bash', str(ROOT/'scripts/serve_local.sh'), str(model), '--embedding', '--dry-run'],
                            env={**os.environ, 'LOCAL_AI_CONFIG': str(config)},
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    import shlex
    args = shlex.split(result.stdout.split('DRY RUN:')[1])
    assert '--embedding' in args
    assert args[args.index('--pooling') + 1] == 'last'
    assert args[args.index('-c') + 1] == '0'
    assert '--mmproj' not in args
