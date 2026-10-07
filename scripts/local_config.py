"""Load the repository's safe defaults and ignored machine configuration."""
from __future__ import annotations

import copy
import argparse
import json
import os
import shlex
import tomllib
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "local.toml"


class ConfigError(RuntimeError):
    """Raised when the machine configuration cannot be safely interpreted."""


def _defaults() -> dict[str, Any]:
    return {
        "version": 1,
        "models": {
            "root": Path.home() / ".lmstudio" / "models",
            "registry": ROOT / "configs" / "models.local.json",
        },
        "endpoints": {
            "llamacpp": {"base_url": "http://127.0.0.1:8080/v1"},
            "mlx": {"base_url": "http://127.0.0.1:8085/v1"},
            "lmstudio": {"base_url": "http://127.0.0.1:1234/v1"},
        },
        "serving": {
            "sampling": {
                "repetition_penalty": 1.05,
                "presence_penalty": 0.2,
                "repetition_context_size": 2048,
            },
        },
        "external_projects": [],
    }


def _merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def _path(value: str | Path, *, relative_to: Path = ROOT) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return expanded if expanded.is_absolute() else relative_to / expanded


def _normalize(config: dict[str, Any]) -> dict[str, Any]:
    config["models"]["root"] = _path(config["models"]["root"])
    config["models"]["registry"] = _path(config["models"]["registry"])

    projects: list[dict[str, Any]] = []
    for raw in config.get("external_projects", []):
        project = dict(raw)
        if "path" not in project or "name" not in project:
            raise ConfigError("Every external project needs both name and path")
        project["path"] = _path(project["path"])
        projects.append(project)
    config["external_projects"] = projects

    for name, endpoint in config.get("endpoints", {}).items():
        base_url = str(endpoint.get("base_url", ""))
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or not parsed.port:
            raise ConfigError(f"Invalid base_url for endpoints.{name}: {base_url!r}")
        endpoint["base_url"] = base_url.rstrip("/")
    return config


def default_config_path() -> Path:
    return Path(os.environ.get("LOCAL_AI_CONFIG", str(DEFAULT_CONFIG))).expanduser()


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load local TOML over safe defaults; a missing file is valid."""
    path = path or default_config_path()
    config = copy.deepcopy(_defaults())
    try:
        raw = tomllib.loads(path.read_text()) if path.is_file() else {}
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"Invalid local configuration {path}: {exc}") from exc
    if raw.get("version", 1) != 1:
        raise ConfigError(f"Unsupported local configuration version in {path}")
    return _normalize(_merge(config, raw))


def is_within(path: Path, root: Path) -> bool:
    """Return whether path resolves inside root (not merely sharing a prefix)."""
    try:
        path.expanduser().resolve(strict=False).relative_to(
            root.expanduser().resolve(strict=False)
        )
        return True
    except ValueError:
        return False


def endpoint(config: dict[str, Any], name: str) -> str:
    try:
        return str(config["endpoints"][name]["base_url"])
    except KeyError as exc:
        raise ConfigError(f"No endpoint configured for {name!r}") from exc


def endpoint_host_port(config: dict[str, Any], name: str) -> tuple[str, int]:
    parsed = urlparse(endpoint(config, name))
    if not parsed.hostname or not parsed.port:
        raise ConfigError(f"Endpoint {name!r} has no host/port")
    return parsed.hostname, parsed.port


def shell_values(config: dict[str, Any]) -> dict[str, str]:
    values = {
        "LOCAL_MODEL_ROOT": str(config["models"]["root"]),
        "LOCAL_MODEL_REGISTRY": str(config["models"]["registry"]),
    }
    for service in ("llamacpp", "mlx", "lmstudio"):
        host, port = endpoint_host_port(config, service)
        prefix = f"LOCAL_{service.upper()}"
        values[f"{prefix}_HOST"] = host
        values[f"{prefix}_PORT"] = str(port)
        values[f"{prefix}_BASE_URL"] = endpoint(config, service)
    sampling = config.get("serving", {}).get("sampling", {})
    if "repetition_penalty" in sampling:
        values["LOCAL_DEFAULT_REPETITION_PENALTY"] = str(sampling["repetition_penalty"])
    if "presence_penalty" in sampling:
        values["LOCAL_DEFAULT_PRESENCE_PENALTY"] = str(sampling["presence_penalty"])
    if "repetition_context_size" in sampling:
        values["LOCAL_DEFAULT_REPETITION_CONTEXT_SIZE"] = str(sampling["repetition_context_size"])
    return values


_SECRET_KEYS = {"authorization", "api_key", "apikey", "password", "secret", "token"}


def redact(value: Any) -> Any:
    """Return a recursively redacted copy safe for diagnostics."""
    if isinstance(value, dict):
        return {
            key: "<redacted>" if key.lower() in _SECRET_KEYS and item else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Read the local inference machine configuration")
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("command", choices=["shell", "json"])
    args = parser.parse_args()
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        raise SystemExit(str(exc)) from exc
    if args.command == "shell":
        for key, value in shell_values(config).items():
            print(f"{key}={shlex.quote(value)}")
    else:
        print(json.dumps(redact(config), indent=2, default=str))


if __name__ == "__main__":
    main()
