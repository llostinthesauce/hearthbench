#!/usr/bin/env python3
"""Preview or synchronize the machine's stable local inference shortcuts."""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN_DIR = Path.home() / ".local" / "bin"
DEFAULT_WRAPPER = BIN_DIR / "llama-serve"
# The full CLI, not just `serve`. Without these, every documented `llm <verb>`
# command is unavailable: `llama-serve` hard-codes the `serve` subcommand, and
# pyproject's `local-ai` console script is only present if the package is
# pip-installed, which it is not.
DEFAULT_CLI = BIN_DIR / "llm"
DEFAULT_CLI_ALIAS = BIN_DIR / "local-ai"
DEFAULT_SPLASH = Path.home() / ".splash.py"


def wrapper_content(repo_root: Path = ROOT, cli: Path = None) -> str:
    """`llama-serve` — the original entry point, now just `llm serve`.

    It forwards to the `llm` wrapper rather than re-deriving the interpreter
    and entrypoint itself. That keeps exactly one place on the machine that
    knows where this checkout lives: move the repo, run `llm shortcuts sync
    --apply`, and both commands follow. It used to be a second copy of that
    knowledge, which is the same duplication this repository exists to remove.

    Old muscle memory keeps working, including `llama-serve --host 127.0.0.1`
    from the splash menu, because the `serve` subcommand is supplied here.
    """
    target = (cli or DEFAULT_CLI).resolve()
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "# Legacy alias for `llm serve`. Managed by scripts/shortcuts.py.\n"
        f'exec "{target}" serve "$@"\n'
    )


def cli_content(repo_root: Path = ROOT) -> str:
    """The whole CLI: `llm status`, `llm stop`, `llm doctor`, and the rest."""
    executable = repo_root.resolve() / ".venv" / "bin" / "python"
    entrypoint = repo_root.resolve() / "llm.py"
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f'exec "{executable}" "{entrypoint}" "$@"\n'
    )


def update_splash_text(source: str) -> str:
    replacement = (
        '    ("llama-serve", \'echo "starting local model server..."; '
        '~/.local/bin/llama-serve --host 127.0.0.1\'),'
    )
    updated, count = re.subn(
        r'(?m)^\s*\("llama-serve",.*\),\s*$', replacement, source
    )
    if count != 1:
        raise ValueError(f"expected one llama-serve splash option, found {count}")
    benchmark_command = f"cd {shlex.quote(str(ROOT))} && .venv/bin/python bench_tui.py"
    updated, count = re.subn(
        r'(?m)^\s*\("benchmarks",.*\),\s*$',
        lambda _: f'    ("benchmarks", {json.dumps(benchmark_command)}),\n', updated
    )
    if count > 1:
        raise ValueError(f"expected at most one benchmarks splash option, found {count}")
    return updated


def _write_with_backup(path: Path, content: str, *, executable: bool = False) -> Path | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup: Path | None = None
    if path.exists():
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        backup = path.with_name(f"{path.name}.backup-{stamp}")
        shutil.copy2(path, backup)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(content)
        if executable:
            os.chmod(temporary, 0o755)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return backup


def _needs_write(path: Path, desired: str) -> bool:
    return not path.exists() or path.read_text() != desired


def sync(*, wrapper: Path = DEFAULT_WRAPPER, splash: Path = DEFAULT_SPLASH,
         cli: Path = DEFAULT_CLI, cli_alias: Path = DEFAULT_CLI_ALIAS,
         apply: bool = False) -> list[str]:
    desired_wrapper = wrapper_content(cli=cli)
    desired_cli = cli_content()
    source_splash = splash.read_text()
    desired_splash = update_splash_text(source_splash)

    executables = [("wrapper", wrapper, desired_wrapper),
                   ("cli", cli, desired_cli),
                   ("cli alias", cli_alias, desired_cli)]

    changes = []
    for label, path, desired in executables:
        if _needs_write(path, desired):
            changes.append(f"{label}: {path}")
    if source_splash != desired_splash:
        changes.append(f"splash: {splash}")
    if not apply:
        return changes

    changes = []
    for label, path, desired in executables:
        if _needs_write(path, desired):
            backup = _write_with_backup(path, desired, executable=True)
            changes.append(f"{label}: {path}" + (f" (backup {backup})" if backup else ""))
    if source_splash != desired_splash:
        backup = _write_with_backup(splash, desired_splash)
        changes.append(f"splash backup: {backup}")
    return changes


def main() -> None:
    parser = argparse.ArgumentParser(description="Synchronize splash, llm, and llama-serve shortcuts")
    parser.add_argument("--apply", action="store_true", help="Write after timestamped backups")
    args = parser.parse_args()
    changes = sync(apply=args.apply)
    if not changes:
        print("Shortcuts already aligned.")
    else:
        print("\n".join(changes))
        if not args.apply:
            print("Preview only. Re-run with --apply to write timestamped backups.")


if __name__ == "__main__":
    main()
