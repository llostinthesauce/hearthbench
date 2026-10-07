"""Reverted serving fixes must be restorable with one command.

Any mlx-lm reinstall silently reverts the fixes in `patches/mlx_lm/`. These
tests run the real apply script against a copy of the real patched
`server.py` with fixes reversed out of it, so they exercise the actual patch
files, not a fixture that only resembles them.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import serving_runtime as sr

SCRIPT = ROOT / "scripts" / "apply_serving_patches.py"
PATCH_DIR = ROOT / "patches" / "mlx_lm"


def _real_patched_server() -> Path:
    report = sr.status()
    if not report.get("ok"):
        pytest.skip(f"core serving runtime is not fully patched: {report.get('reason')}")
    return Path(report["server_source"])


def _copy_with_reverted(tmp_path: Path, *patch_names: str) -> tuple[Path, bytes]:
    """A copy of the real server.py with the named patches reversed out of it."""
    target = tmp_path / "mlx_lm" / "server.py"
    target.parent.mkdir()
    shutil.copy2(_real_patched_server(), target)
    original = target.read_bytes()
    for name in patch_names:
        subprocess.run(
            ["patch", "--batch", "--no-backup-if-mismatch", "-R", str(target),
             "-i", str(PATCH_DIR / name)],
            check=True, capture_output=True, timeout=30,
        )
    return target, original


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, timeout=120,
    )


def test_reverted_fixes_are_restored_byte_for_byte(tmp_path):
    target, original = _copy_with_reverted(
        tmp_path, "01-reasoning-content-alias.patch", "03-metal-oom-shield.patch",
    )
    assert {p.key for p in sr.missing_patches(target)} == {
        "reasoning_content_alias", "metal_oom_shield",
    }

    result = _run("--server-file", str(target))

    assert result.returncode == 0, result.stdout + result.stderr
    assert target.read_bytes() == original
    assert sr.missing_patches(target) == []
    # patch(1) must not litter site-packages with .orig or .rej files.
    assert sorted(p.name for p in target.parent.iterdir()) == ["server.py"]


def test_fully_patched_file_is_left_untouched(tmp_path):
    target, original = _copy_with_reverted(tmp_path)

    result = _run("--server-file", str(target))

    assert result.returncode == 0, result.stdout + result.stderr
    assert target.read_bytes() == original


def test_dry_run_names_the_missing_fix_without_writing(tmp_path):
    target, _ = _copy_with_reverted(tmp_path, "04-explicit-zero-penalties.patch", "02-server-penalties.patch")
    before = target.read_bytes()

    result = _run("--server-file", str(target), "--dry-run")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "penalty_defaults" in result.stdout
    assert "explicit_zero_penalties" in result.stdout
    assert target.read_bytes() == before


def test_a_patch_that_no_longer_applies_changes_nothing(tmp_path):
    """After an upstream bump a hunk may stop matching. Writing the patches
    that do apply and skipping the rest would leave a half-patched server that
    the doctor can no longer explain, so nothing may be written at all."""
    target = tmp_path / "mlx_lm" / "server.py"
    target.parent.mkdir()
    target.write_text('print("not the upstream server")\n')
    before = target.read_bytes()

    result = _run("--server-file", str(target))

    assert result.returncode == 1
    assert "01-reasoning-content-alias.patch" in result.stderr
    assert target.read_bytes() == before
    assert sorted(p.name for p in target.parent.iterdir()) == ["server.py"]
