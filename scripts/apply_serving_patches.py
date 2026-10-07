#!/usr/bin/env python3
"""Reapply the mlx_lm serving fixes after a reinstall reverted them.

Any reinstall of mlx-lm — `pip install -U mlx-lm`, `uv sync --reinstall`, a
version bump — silently puts back the upstream `server.py`, and `llm doctor`
then names each missing fix. This restores exactly the missing ones from
`patches/mlx_lm/`.

All or nothing: missing patches are staged in order, and if any hunk no longer
applies (typically after an upstream bump) nothing is written. A half-patched
server would carry some fixes and not others, which is the drift this
repository exists to prevent. Port the failing patch by hand instead; see
`patches/mlx_lm/README.md`.

    .venv/bin/python scripts/apply_serving_patches.py            # repair the core .venv
    .venv/bin/python scripts/apply_serving_patches.py --dry-run  # report only
"""
from __future__ import annotations

import argparse
import os
import tempfile
import shutil
import subprocess
import sys
from pathlib import Path

import serving_runtime as sr


def _patch(target: Path, patch_file: Path, *, dry_run: bool) -> subprocess.CompletedProcess:
    # --no-backup-if-mismatch: a hunk that lands at an offset would otherwise
    # leave server.py.orig behind in site-packages.
    cmd = ["patch", "--forward", "--batch", "--no-backup-if-mismatch",
           str(target), "-i", str(patch_file)]
    if dry_run:
        cmd.insert(1, "--dry-run")
    return subprocess.run(cmd, capture_output=True, text=True, timeout=60)


def _core_server_file() -> Path:
    python = sr.serving_python()
    source = sr.server_source(python) if python else None
    if source is None:
        raise SystemExit(
            f"The core serving interpreter cannot import mlx_lm: {sr.VENV_PYTHON}\n"
            "Run `uv sync --extra test` first."
        )
    return source


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--server-file", type=Path,
                        help="mlx_lm/server.py to patch (default: the core .venv's)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be applied; write nothing")
    args = parser.parse_args(argv)

    if shutil.which("patch") is None:
        raise SystemExit("The `patch` command is required and was not found on PATH.")

    target = args.server_file or _core_server_file()
    missing = sorted(sr.missing_patches(target), key=lambda fix: fix.patch)
    if not missing:
        print(f"All {len(sr.PATCHES)} serving fixes present in {target}")
        return 0

    original = target.read_bytes()
    # Apply in order to a private copy: later fixes may depend on earlier ones.
    # Publish only a complete, verified source, never a half-patched runtime.
    with tempfile.TemporaryDirectory(prefix="mlx-patches-", dir=target.parent) as directory:
        staged = Path(directory) / target.name
        shutil.copy2(target, staged)
        for fix in missing:
            result = _patch(staged, sr.PATCH_DIR / fix.patch, dry_run=False)
            if result.returncode != 0:
                print(f"{fix.patch} no longer applies ({fix.key}):\n"
                      f"{result.stdout}{result.stderr}", file=sys.stderr)
                print(f"Nothing was written to {target}. Port the patch by hand; "
                      "see patches/mlx_lm/README.md.", file=sys.stderr)
                return 1
        still_missing = sr.missing_patches(staged)
        if still_missing:
            print(f"Still missing after staging: {[fix.key for fix in still_missing]}",
                  file=sys.stderr)
            return 1
        if args.dry_run:
            for fix in missing:
                print(f"would apply {fix.patch} ({fix.key})")
            return 0
        if target.read_bytes() != original:
            print("Runtime changed during patching; nothing written.", file=sys.stderr)
            return 1
        os.replace(staged, target)
        for fix in missing:
            print(f"applied {fix.patch} ({fix.key})")

    print(f"All {len(sr.PATCHES)} serving fixes present in {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
