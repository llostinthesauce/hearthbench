"""`serve_local.sh` serves MLX only from the core runtime, and only when patched.

Two ways the launcher used to reintroduce the drift this repository exists to
remove, silently: falling back to whatever `python3` / `mlx_lm.server` was on
PATH when the core `.venv` was absent (under the Mycelium app that is
Homebrew's MLX 0.32.0, which carries none of the fixes), and starting an
`mlx_lm.server` whose local fixes a reinstall had reverted.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVE = ROOT / "scripts" / "serve_local.sh"


def _unpatched_mlx_lm(tmp_path: Path) -> str:
    """A PYTHONPATH entry whose `mlx_lm` looks like a freshly reinstalled one."""
    package = tmp_path / "site" / "mlx_lm"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "server.py").write_text("# upstream server.py: none of the local fixes\n")
    return str(package.parent)


def _serve(*args, env, script=SERVE):
    return subprocess.run(
        ["bash", str(script), *args, "--dry-run"],
        capture_output=True, text=True, timeout=120, env=env,
    )


def test_mlx_launch_refuses_a_runtime_missing_a_required_fix(launcher_env, tmp_path):
    env = {**launcher_env, "PYTHONPATH": _unpatched_mlx_lm(tmp_path)}

    result = _serve("fixture", "--backend", "mlx", env=env)

    assert result.returncode != 0
    assert "DRY RUN:" not in result.stdout
    assert "reasoning_content_alias" in result.stderr
    assert "apply_serving_patches.py" in result.stderr


def test_allow_unpatched_serves_anyway_and_says_so(launcher_env, tmp_path):
    """Measuring upstream behaviour on purpose must stay possible."""
    env = {**launcher_env, "PYTHONPATH": _unpatched_mlx_lm(tmp_path)}

    result = _serve("fixture", "--backend", "mlx", "--allow-unpatched", env=env)

    assert result.returncode == 0, result.stderr
    assert "DRY RUN:" in result.stdout
    assert "penalty_defaults" in result.stderr


def test_the_fix_check_does_not_block_llama_cpp(launcher_env, tmp_path):
    env = {**launcher_env, "PYTHONPATH": _unpatched_mlx_lm(tmp_path)}

    result = _serve("fixture", "--backend", "llamacpp", env=env)

    assert result.returncode == 0, result.stderr
    assert "DRY RUN:" in result.stdout


def _copy_launcher(tmp_path: Path) -> Path:
    """The launcher and its helpers in a checkout of their own."""
    repo = tmp_path / "repo"
    shutil.copytree(ROOT / "scripts", repo / "scripts",
                    ignore=shutil.ignore_patterns("__pycache__"))
    return repo


def test_a_missing_core_venv_is_an_error_not_a_path_fallback(launcher_env, tmp_path):
    repo = _copy_launcher(tmp_path)

    result = _serve("fixture", "--backend", "mlx", env=launcher_env,
                    script=repo / "scripts" / "serve_local.sh")

    assert result.returncode != 0
    assert "DRY RUN:" not in result.stdout
    assert str(repo / ".venv" / "bin" / "python3") in result.stderr


def test_the_mlx_server_never_resolves_from_path(launcher_env, tmp_path):
    """A core venv without `mlx_lm.server` must fail, not borrow PATH's."""
    repo = _copy_launcher(tmp_path)
    venv_bin = repo / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    wrapper = venv_bin / "python3"
    wrapper.write_text(f'#!/bin/bash\nexec "{ROOT / ".venv" / "bin" / "python3"}" "$@"\n')
    wrapper.chmod(0o755)

    result = _serve("fixture", "--backend", "mlx", env=launcher_env,
                    script=repo / "scripts" / "serve_local.sh")

    assert result.returncode != 0
    assert "DRY RUN:" not in result.stdout
    assert "mlx_lm.server" in result.stderr
    assert os.fspath(venv_bin) in result.stderr
