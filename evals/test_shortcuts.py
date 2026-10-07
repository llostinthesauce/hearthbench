"""Machine shortcuts resolve through the stable repository CLI."""
from __future__ import annotations

import sys
import subprocess
import shutil
import os
import pty
import select
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import shortcuts



def test_wrapper_forwards_to_the_one_cli_instead_of_duplicating_it(tmp_path):
    """`llama-serve` is `llm serve`, not a second copy of how to reach the repo.

    It used to embed the interpreter and entrypoint paths itself, so the
    machine had two places that knew where this checkout lived and a moved
    repo could leave one of them stale. Now only the `llm` wrapper knows.
    """
    cli = tmp_path / "bin" / "llm"
    content = shortcuts.wrapper_content(tmp_path / "repo", cli=cli)

    assert str(cli) in content
    assert " serve \"$@\"" in content
    # No second copy of the repo-location knowledge.
    assert ".venv/bin/python" not in content
    assert "llm.py" not in content
    assert "/old/repo" not in content


def test_double_click_launcher_survives_stale_console_script(tmp_path):
    repo = tmp_path / "moved repo"
    (repo / ".venv/bin").mkdir(parents=True)
    (repo / ".venv/bin/python").symlink_to(sys.executable)
    stale = repo / ".venv/bin/local-ai"
    stale.write_text("#!/no/longer/existing/python\n")
    stale.chmod(0o755)
    (repo / "llm.py").write_text("import sys; print('LAUNCHED',sys.argv[1:])\n")
    command = repo / "Local AI.command"
    shutil.copy2(ROOT / "Local AI.command", command)
    result = subprocess.run(["bash", str(command), "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "LAUNCHED ['serve', '--help']" in result.stdout


def test_splash_benchmark_entry_moves_with_repository():
    source = '''OPTIONS = [
    ("benchmarks", "cd /old/repo && python bench_tui.py"),
    ("llama-serve", "old"),
]'''
    updated = shortcuts.update_splash_text(source)
    assert "/old/repo" not in updated
    assert str(ROOT) in updated


def test_splash_rewrite_uses_one_stable_wrapper():
    source = '''OPTIONS = [
    ("quit to terminal", ""),
    ("llama-serve", 'echo "old"; cd /old/repo && python3 old.py'),
]'''

    updated = shortcuts.update_splash_text(source)

    assert updated.count('("llama-serve"') == 1
    assert "~/.local/bin/llama-serve" in updated
    assert "/old/repo" not in updated


def test_cli_shortcut_exposes_every_verb_not_just_serve(tmp_path):
    """`llm status`, `llm stop`, `llm doctor` need a full-CLI entry point.

    `llama-serve` hard-codes the `serve` subcommand, and pyproject's `local-ai`
    console script only exists if the package is pip-installed, which it is
    not. Without this wrapper every documented `llm <verb>` command is simply
    missing from the machine.
    """
    content = shortcuts.cli_content(tmp_path / "repo")

    assert str(tmp_path / "repo" / ".venv/bin/python") in content
    assert str(tmp_path / "repo" / "llm.py") in content
    # No pinned subcommand: the caller's first argument chooses the verb.
    assert ' "$@"' in content
    assert " serve " not in content


def test_sync_manages_the_cli_alongside_the_serve_wrapper(tmp_path, monkeypatch):
    splash = tmp_path / "splash.py"
    splash.write_text(
        '("llama-serve", "old"),\n("benchmarks", "old"),\n'
    )
    wrapper = tmp_path / "bin" / "llama-serve"
    cli = tmp_path / "bin" / "llm"
    alias = tmp_path / "bin" / "local-ai"

    planned = shortcuts.sync(wrapper=wrapper, splash=splash, cli=cli,
                             cli_alias=alias, apply=False)
    assert any("llm" in line for line in planned)
    assert not cli.exists(), "preview must not write"

    shortcuts.sync(wrapper=wrapper, splash=splash, cli=cli, cli_alias=alias,
                   apply=True)
    for path in (wrapper, cli, alias):
        assert path.is_file(), path
        assert os.access(path, os.X_OK), f"{path} is not executable"

    # Idempotent: a second sync has nothing to do.
    assert shortcuts.sync(wrapper=wrapper, splash=splash, cli=cli,
                          cli_alias=alias, apply=False) == []


def test_installed_cli_runs_every_documented_verb():
    """The real shortcut on this machine, not a synthetic one."""
    installed = Path.home() / ".local" / "bin" / "llm"
    if not installed.is_file():
        import pytest

        pytest.skip("llm shortcut not installed; run `llm shortcuts sync --apply`")
    result = subprocess.run([str(installed), "--help"], capture_output=True,
                            text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    for verb in ("serve", "stop", "use", "restart", "status", "doctor", "web"):
        assert verb in result.stdout, f"`llm {verb}` is not reachable"
