"""Read-only diagnostics for local inference configuration and model storage."""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import local_config
import model_registry
import serving_runtime
from model_files import is_complete_gguf, is_complete_mlx


ACTIVE_ENDPOINTS = ("llamacpp", "mlx", "lmstudio")


@dataclass(frozen=True)
class Check:
    code: str
    severity: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def check_endpoint_safety(config: dict[str, Any]) -> list[Check]:
    checks: list[Check] = []
    loopback = {"127.0.0.1", "localhost", "::1"}
    endpoints = config.get("endpoints", {})
    for name in ACTIVE_ENDPOINTS:
        endpoint_config = endpoints.get(name)
        if not endpoint_config:
            continue
        url = str(endpoint_config.get("base_url", ""))
        host = urlparse(url).hostname
        if host in loopback:
            # This is a binding-safety check, not a liveness check. Say so:
            # a bare "[OK] llamacpp: http://..." reads as "the server is up",
            # and it prints identically when nothing is listening at all.
            # Use `llm status` for liveness.
            checks.append(Check(
                f"endpoint.{name}.loopback", "OK",
                f"{name} endpoint is local-only: {url}",
            ))
        else:
            checks.append(Check(
                f"endpoint.{name}.non_loopback",
                "FAIL",
                f"{name} is configured on non-loopback host {host!r}",
                {"base_url": url},
            ))
    return checks


def _variant_paths(registry: dict[str, Any]) -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = []
    for family in registry.get("model_families", []):
        for item in family.get("gguf", []):
            if item.get("path"):
                paths.append(("llamacpp", Path(str(item["path"])).expanduser()))
        for item in family.get("mlx", []):
            if item.get("repo"):
                paths.append(("mlx", Path(str(item["repo"])).expanduser()))
    return paths


def _has_complete_gguf(path: Path) -> bool:
    try:
        return any(is_complete_gguf(item) for item in path.rglob("*.gguf"))
    except OSError:
        return False


def _incomplete_model_dirs(root: Path) -> list[Path]:
    incomplete: list[Path] = []
    if not root.is_dir():
        return incomplete
    try:
        organizations = [item for item in root.iterdir() if item.is_dir() and not item.name.startswith(".")]
    except OSError:
        return incomplete
    for organization in organizations:
        try:
            models = [item for item in organization.iterdir() if item.is_dir() and not item.name.startswith(".")]
        except OSError:
            continue
        for model in models:
            if is_complete_mlx(model) or _has_complete_gguf(model):
                continue
            try:
                has_download_artifacts = any(item.is_file() for item in model.rglob("*"))
            except OSError:
                has_download_artifacts = False
            if has_download_artifacts:
                incomplete.append(model)
    return sorted(incomplete)


def check_models(config: dict[str, Any]) -> list[Check]:
    root = Path(config["models"]["root"])
    registry_path = Path(config["models"]["registry"])
    checks: list[Check] = []
    if not root.is_dir():
        return [Check("models.root", "FAIL", f"Model root does not exist: {root}")]
    checks.append(Check("models.root", "OK", f"Canonical model root: {root}"))

    registry = _read_json(registry_path)
    if registry is None or registry.get("_schema_version") != 2:
        checks.append(Check(
            "models.registry",
            "WARN",
            f"Missing or invalid registry: {registry_path}; run `local-ai models sync`",
        ))
    else:
        variants = _variant_paths(registry)
        checks.append(Check(
            "models.registry", "OK", f"Registry contains {len(variants)} model variant(s)"
        ))
        for backend, path in variants:
            if not local_config.is_within(path, root):
                checks.append(Check(
                    "model.outside_root",
                    "FAIL",
                    f"Registered {backend} model is outside the LM Studio root: {path}",
                ))
            complete = is_complete_mlx(path) if backend == "mlx" else is_complete_gguf(path)
            if not complete:
                checks.append(Check(
                    "model.registered_incomplete",
                    "WARN",
                    f"Registered {backend} model is incomplete or missing: {path}",
                ))

    incomplete = _incomplete_model_dirs(root)
    if incomplete:
        checks.append(Check(
            "model.incomplete",
            "WARN",
            "Incomplete model folders: " + ", ".join(str(path) for path in incomplete),
            {"paths": [str(path) for path in incomplete]},
        ))
    else:
        checks.append(Check("model.incomplete", "OK", "No incomplete model folders found"))
    return checks


def check_runtime_commands(
    which: Callable[[str], str | None] = shutil.which,
) -> list[Check]:
    checks: list[Check] = []
    for command in ("llama-server", "lms"):
        resolved = which(command)
        checks.append(Check(
            f"runtime.{command}",
            "OK" if resolved else "WARN",
            f"{command}: {resolved}" if resolved else f"{command} is not on PATH",
        ))
    return checks


def check_external_projects(config: dict[str, Any]) -> list[Check]:
    checks: list[Check] = []
    projects = config.get("external_projects", [])
    if not projects:
        return [Check("projects.none", "OK", "No external local projects registered")]
    registry = _read_json(Path(config["models"]["registry"])) or {}
    known_models: set[str] = set()
    for family in registry.get("model_families", []):
        family_id = str(family.get("id", ""))
        known_models.add(family_id)
        for backend_key, path_key in (("gguf", "path"), ("mlx", "repo")):
            for variant in family.get(backend_key, []):
                if variant.get(path_key):
                    known_models.add(Path(str(variant[path_key])).name)
        known_models.update(
            alias for alias, target in model_registry.LEGACY_ALIASES.items()
            if target == family_id
        )
    for project in projects:
        path = Path(project["path"])
        name = str(project["name"])
        if not path.is_dir():
            checks.append(Check(
                "project.missing", "WARN", f"External project is missing: {name} ({path})",
                {"model": project.get("model", ""), "kind": project.get("kind", "")},
            ))
            continue
        model = str(project.get("model", ""))
        if model and model not in known_models:
            checks.append(Check(
                "project.model_missing", "WARN",
                f"External project {name} references an unregistered model: {model}",
                {"path": str(path), "kind": project.get("kind", "")},
            ))
            continue
        checks.append(Check(
            "project.available", "OK", f"{name}: {path}",
            {"model": model, "kind": project.get("kind", "")},
        ))
    return checks


def check_serving_runtime(
    status: Callable[[], dict[str, Any]] = serving_runtime.status,
) -> list[Check]:
    """Report the one serving interpreter and whether its local fixes survive.

    The fixes live in hand-edited `site-packages`, so `pip install -U mlx-lm`
    reverts them silently. Nothing prevents that; this makes it visible here
    instead of surfacing days later as a model that loops or an app that hangs.
    """
    report = status()
    if report.get("reason") == "missing_venv":
        return [Check(
            "serving.runtime_missing", "FAIL",
            f"Serving interpreter is missing: {report.get('python')}. "
            "Follow One-time setup in README.md.",
        )]
    if report.get("reason") == "mlx_lm_not_importable":
        return [Check(
            "serving.mlx_lm_missing", "FAIL",
            f"Serving interpreter cannot import mlx_lm: {report.get('python')}",
        )]

    versions = report.get("versions") or {}
    checks = [Check(
        "serving.runtime", "OK",
        "Serving runtime: mlx {mlx} · mlx-lm {mlx_lm} (python {python})".format(
            mlx=versions.get("mlx", "?"),
            mlx_lm=versions.get("mlx_lm", "?"),
            python=versions.get("python", "?"),
        ),
        {"python": report.get("python")},
    )]

    missing = report.get("missing") or []
    if missing:
        for entry in missing:
            checks.append(Check(
                f"serving.patch_missing.{entry['key']}", "FAIL",
                f"Serving fix reverted: {entry['key']} — {entry['why']}. "
                "Reapply: .venv/bin/python scripts/apply_serving_patches.py",
            ))
    else:
        applied = report.get("applied") or []
        checks.append(Check(
            "serving.patches", "OK",
            f"All {len(applied)} local serving fixes applied",
            {"applied": applied},
        ))
    return checks


def run(config: dict[str, Any]) -> list[Check]:
    return [
        *check_endpoint_safety(config),
        *check_serving_runtime(),
        *check_runtime_commands(),
        *check_models(config),
        *check_external_projects(config),
    ]


def as_json(checks: list[Check]) -> str:
    payload = {
        "summary": {
            severity: sum(check.severity == severity for check in checks)
            for severity in ("OK", "WARN", "FAIL")
        },
        "checks": [asdict(check) for check in checks],
    }
    return json.dumps(local_config.redact(payload), indent=2, default=str)


def format_text(checks: list[Check]) -> str:
    lines = ["Local inference doctor", "=" * 72]
    for check in checks:
        lines.append(f"[{check.severity:<4}] {check.message}")
    counts = {severity: sum(c.severity == severity for c in checks) for severity in ("OK", "WARN", "FAIL")}
    lines.extend(["-" * 72, f"{counts['OK']} OK, {counts['WARN']} warning(s), {counts['FAIL']} failure(s)"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose local inference setup without changing it")
    parser.add_argument("--config", type=Path, default=local_config.DEFAULT_CONFIG)
    parser.add_argument("--json", action="store_true", help="Emit structured JSON")
    args = parser.parse_args()
    try:
        config = local_config.load_config(args.config)
    except local_config.ConfigError as exc:
        raise SystemExit(str(exc)) from exc
    checks = run(config)
    print(as_json(checks) if args.json else format_text(checks))
    raise SystemExit(1 if any(check.severity == "FAIL" for check in checks) else 0)


if __name__ == "__main__":
    main()
