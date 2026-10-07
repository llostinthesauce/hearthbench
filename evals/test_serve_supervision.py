"""`--fatal-worker` must turn an invisible hang into an observable death.

`mlx_lm.server` can outlive its own generation thread: the HTTP server keeps
answering `/v1/models` while no request can ever complete. A parent that
supervises the server process — Mycelium's `BackendSupervisor` — treats a
reachable port as a live runtime and waits forever.

This coverage moved here from Mycelium when the behaviour moved here, so the
test lives beside the code it exercises rather than beside the consumer that
needs it.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUARD = ROOT / "scripts" / "mlx_server_guard.py"
SERVE = ROOT / "scripts" / "serve_local.sh"


def test_generation_worker_crash_terminates_the_process(tmp_path):
    """A synthetic `_generate` failure must exit non-zero, not hang."""
    package = tmp_path / "mlx_lm"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "server.py").write_text(
        "import threading, time\n"
        "def _generate():\n"
        '    raise RuntimeError("synthetic generation failure")\n'
        "threading.Thread(target=_generate).start()\n"
        "time.sleep(30)\n"
    )
    result = subprocess.run(
        [sys.executable, str(GUARD)],
        env=dict(os.environ, PYTHONPATH=str(tmp_path)),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1
    assert "synthetic generation failure" in result.stderr
    assert "generation worker crashed" in result.stderr


def test_a_failure_outside_generation_does_not_kill_the_server(tmp_path):
    """Only the generation thread is fatal.

    Killing the process on any thread exception would make unrelated
    background failures look like a dead runtime.
    """
    package = tmp_path / "mlx_lm"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "server.py").write_text(
        "import threading, time\n"
        "def _unrelated():\n"
        '    raise RuntimeError("synthetic unrelated failure")\n'
        "t = threading.Thread(target=_unrelated)\n"
        "t.start(); t.join()\n"
        "time.sleep(0.2)\n"
        'print("server still alive")\n'
    )
    result = subprocess.run(
        [sys.executable, str(GUARD)],
        env=dict(os.environ, PYTHONPATH=str(tmp_path)),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "server still alive" in result.stdout
    assert "generation worker crashed" not in result.stderr


def _dry_run(env, *args):
    return subprocess.run(
        ["bash", str(SERVE), *args, "--dry-run"],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=ROOT,
        env=env,
    )


def test_omitting_the_flag_leaves_the_command_unchanged(launcher_env):
    """The flag is additive: without it, nothing about serving changes."""
    plain = _dry_run(launcher_env, "fixture", "--backend", "mlx")
    assert plain.returncode == 0, plain.stderr
    command = next(
        line for line in plain.stdout.splitlines() if line.startswith("DRY RUN:")
    )
    assert "mlx_server_guard.py" not in command
    assert "mlx_lm.server" in command


def test_the_flag_routes_through_the_guard(launcher_env):
    guarded = _dry_run(launcher_env, "fixture", "--backend", "mlx", "--fatal-worker")
    assert guarded.returncode == 0, guarded.stderr
    command = next(
        line for line in guarded.stdout.splitlines() if line.startswith("DRY RUN:")
    )
    assert "mlx_server_guard.py" in command
    # Sampling and offline-serving behaviour must survive the detour.
    assert "--repetition-context-size 2048" in command
    assert "HF_HUB_OFFLINE=1" in command


def test_the_flag_is_skipped_on_mlx_vlm(launcher_env):
    """The guard runs `mlx_lm.server`; claiming to guard mlx_vlm would lie."""
    result = _dry_run(launcher_env, "fixture", "--backend", "mlx-vlm", "--fatal-worker")
    assert result.returncode == 0
    assert "--fatal-worker skipped" in result.stdout
    assert "mlx_vlm_server.py" in result.stdout
    assert "mlx_server_guard.py" not in result.stdout
