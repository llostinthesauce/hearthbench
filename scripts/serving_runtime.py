#!/usr/bin/env python3
"""The one serving runtime: which interpreter serves models, and is it patched.

This repository is the single core place local inference is served from. Every
consumer — the TUI, the benchmarks, opencode, pi, Mycelium — borrows this
interpreter rather than resolving its own. The rule is: core decides *how* to
serve, consumers decide *what* to serve.

`mlx_lm.server` carries four local fixes that are not upstream, recorded as
patches in `patches/mlx_lm/`. They live in `site-packages`, so any reinstall
of mlx-lm silently reverts them and the symptom shows up days later as a model
that loops or an app that hangs. Nothing prevents that — the owner declined a
lock — so instead we detect it by name and let `llm doctor` say which fix went
missing.

Every entry below must have a patch that reintroduces its marker; a required
fix nothing in the repo can restore is a FAIL no one can clear.

Detection is marker-based, not a file hash. A hash fails on every upstream
version bump and cannot say what is wrong; a marker survives unrelated upstream
changes and names the specific fix.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The interpreter that serves models. `serve_local.sh` uses the same path and
# has no fallback; keep them together.
VENV_PYTHON = ROOT / ".venv" / "bin" / "python3"


PATCH_DIR = ROOT / "patches" / "mlx_lm"


@dataclass(frozen=True)
class Patch:
    """One local fix, identified by a string that only its edit introduces."""

    key: str
    marker: str
    why: str
    # The file in PATCH_DIR that restores it.
    patch: str


PATCHES: tuple[Patch, ...] = (
    Patch(
        key="explicit_zero_penalties",
        patch="04-explicit-zero-penalties.patch",
        marker='self.presence_penalty = self.body.get("presence_penalty", getattr(',
        why="explicit request penalties of zero must override nonzero server defaults",
    ),
    Patch(
        key="reasoning_content_alias",
        patch="01-reasoning-content-alias.patch",
        marker='message["reasoning_content"] = reasoning',
        why=(
            "server emits thinking as `reasoning`, Qwen templates read "
            "`reasoning_content`; without the alias multi-turn tool use "
            "degenerates into verbatim repetition after ~10 turns"
        ),
    ),
    Patch(
        key="metal_oom_shield",
        patch="03-metal-oom-shield.patch",
        marker="Error during batch_generator.next()",
        why=(
            "a Metal OOM otherwise kills the generation thread and leaves "
            "every client hanging for its full timeout instead of erroring"
        ),
    ),
    Patch(
        key="penalty_defaults",
        patch="02-server-penalties.patch",
        marker='MLX_REPETITION_CONTEXT_SIZE", 2048',
        why=(
            "upstream's 20-token lookback cannot see repeated reasoning "
            "cycles ~1000-1200 tokens apart; 2048 is load-bearing"
        ),
    ),
)


def serving_python() -> Path | None:
    """The interpreter that serves models, or None if the core venv is absent."""
    return VENV_PYTHON if VENV_PYTHON.is_file() else None


def server_source(python: Path) -> Path | None:
    """Locate `mlx_lm/server.py` for the given interpreter."""
    try:
        out = subprocess.run(
            [str(python), "-c", "import mlx_lm, os; print(os.path.dirname(mlx_lm.__file__))"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    path = Path(out.stdout.strip()) / "server.py"
    return path if path.is_file() else None


def runtime_versions(python: Path) -> dict[str, str]:
    """mlx / mlx-lm versions reported by the given interpreter."""
    code = (
        "import json, sys, mlx.core, mlx_lm;"
        "print(json.dumps({'python': sys.version.split()[0],"
        "'mlx': mlx.core.__version__, 'mlx_lm': mlx_lm.__version__}))"
    )
    try:
        out = subprocess.run(
            [str(python), "-c", code], capture_output=True, text=True, timeout=30, check=True
        )
        return json.loads(out.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        return {}


def installed_server_source() -> Path | None:
    """`mlx_lm/server.py` for the running interpreter, located without importing mlx.

    `serve_local.sh` runs this under the serving interpreter before every
    mlx_lm launch, so it has to cost a Python startup, not an MLX import.
    """
    spec = importlib.util.find_spec("mlx_lm")
    locations = spec.submodule_search_locations if spec else None
    if not locations:
        return None
    path = Path(next(iter(locations))) / "server.py"
    return path if path.is_file() else None


def require_patched() -> int:
    """Exit status for the launcher: 0 when every fix is present."""
    source = installed_server_source()
    if source is None:
        print(f"mlx_lm is not importable by {sys.executable}", file=sys.stderr)
        return 1
    missing = missing_patches(source)
    for patch in missing:
        print(f"  MISSING {patch.key}: {patch.why}", file=sys.stderr)
    return 1 if missing else 0


def missing_patches(source: Path) -> list[Patch]:
    """Which local fixes are absent from this `server.py`."""
    try:
        text = source.read_text(errors="replace")
    except OSError:
        return list(PATCHES)
    return [patch for patch in PATCHES if patch.marker not in text]


def status() -> dict[str, object]:
    """Everything `llm doctor` needs to report on the serving runtime."""
    python = serving_python()
    if python is None:
        return {
            "ok": False,
            "reason": "missing_venv",
            "python": str(VENV_PYTHON),
        }
    source = server_source(python)
    if source is None:
        return {
            "ok": False,
            "reason": "mlx_lm_not_importable",
            "python": str(python),
        }
    absent = missing_patches(source)
    return {
        "ok": not absent,
        "reason": "patches_missing" if absent else "healthy",
        "python": str(python),
        "server_source": str(source),
        "versions": runtime_versions(python),
        "applied": [p.key for p in PATCHES if p not in absent],
        "missing": [{"key": p.key, "why": p.why} for p in absent],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="Emit structured JSON")
    parser.add_argument("--require-patched", action="store_true",
                        help="Check this interpreter's mlx_lm only; exit 1 if a fix is missing")
    args = parser.parse_args()
    if args.require_patched:
        raise SystemExit(require_patched())
    report = status()
    if args.json:
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report["ok"] else 1)

    print("Serving runtime")
    print("-" * 72)
    print(f"  interpreter  {report.get('python')}")
    versions = report.get("versions") or {}
    if versions:
        print(
            f"  versions     python {versions.get('python')} · "
            f"mlx {versions.get('mlx')} · mlx-lm {versions.get('mlx_lm')}"
        )
    if report["ok"]:
        print(f"  patches      all {len(PATCHES)} applied")
    else:
        print(f"  problem      {report.get('reason')}")
        for entry in report.get("missing", []):
            print(f"    MISSING {entry['key']}: {entry['why']}")
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
