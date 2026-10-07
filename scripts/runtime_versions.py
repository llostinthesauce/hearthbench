"""Read-only installed/latest checks for the local inference runtimes."""
from __future__ import annotations

import importlib.metadata
import json
import re
import ssl
import subprocess
import urllib.request
from dataclasses import asdict, dataclass
from typing import Callable

try:
    import certifi
except ImportError:  # system Python fallback; project environment pins certifi
    certifi = None


@dataclass(frozen=True)
class RuntimeSpec:
    name: str
    installed_version: Callable[[], str]
    latest_version: Callable[[], str]


@dataclass(frozen=True)
class RuntimeStatus:
    name: str
    installed: str | None
    latest: str | None
    state: str
    note: str = ""


def extract_version(text: str) -> str:
    labeled = re.search(r"\bversion:\s*v?b?([0-9]+(?:\.[0-9]+)*(?:a\d+|b\d+|rc\d+)?)", text, re.I)
    if labeled:
        return labeled.group(1)
    semantic = re.search(r"(?<!\d)(\d+\.\d+(?:\.\d+)?(?:a\d+|b\d+|rc\d+)?)", text, re.I)
    if semantic:
        return semantic.group(1)
    build = re.search(r"(?<![A-Za-z0-9])b?(\d{3,})(?![A-Za-z0-9])", text)
    if build:
        return build.group(1)
    raise ValueError(f"No version found in: {text.strip()[:120]}")


def extract_llamacpp_build(text: str) -> str:
    """Return llama.cpp's comparable build number, not a packaging version.

    Homebrew's current package prints ``version: 0.3.0 (build 10621, ...)``
    while older builds printed ``version: 10360 (...)``.  Upstream releases
    are build tags (for example ``b10814``), so the build is the only useful
    installed/latest comparison.
    """
    build = re.search(r"\bbuild\s+b?(\d+)\b", text, re.I)
    return build.group(1) if build else extract_version(text)


def _version_key(value: str) -> tuple[tuple[int, ...], int, int]:
    clean = value.strip().lower().lstrip("v")
    if clean.startswith("b") and clean[1:].isdigit():
        return ((int(clean[1:]),), 3, 0)
    match = re.match(r"(\d+(?:\.\d+)*)(?:(a|b|rc)(\d+))?$", clean)
    if not match:
        numbers = tuple(int(item) for item in re.findall(r"\d+", clean))
        return (numbers, 0, 0)
    numbers = tuple(int(item) for item in match.group(1).split("."))
    stage = {"a": 0, "b": 1, "rc": 2, None: 3}[match.group(2)]
    serial = int(match.group(3) or 0)
    return (numbers, stage, serial)


def is_newer(latest: str, installed: str) -> bool:
    return _version_key(latest) > _version_key(installed)


def _command_version(command: str, *args: str) -> str:
    completed = subprocess.run(
        [command, *args], text=True, capture_output=True, timeout=15, check=False
    )
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    if completed.returncode != 0:
        raise RuntimeError(output.strip() or f"{command} exited {completed.returncode}")
    return extract_version(output)


def _llamacpp_version() -> str:
    completed = subprocess.run(
        ["llama-server", "--version"], text=True, capture_output=True,
        timeout=15, check=False,
    )
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    if completed.returncode != 0:
        raise RuntimeError(output.strip() or f"llama-server exited {completed.returncode}")
    return extract_llamacpp_build(output)


def _package_version(distribution: str) -> str:
    return importlib.metadata.version(distribution)


def _fetch_json(url: str) -> object:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "local-ai-version-check/1"},
    )
    context = ssl.create_default_context(cafile=certifi.where() if certifi else None)
    with urllib.request.urlopen(request, timeout=10, context=context) as response:
        value = json.loads(response.read())
    return value


def _github_latest(repo: str) -> str:
    value = _fetch_json(f"https://api.github.com/repos/{repo}/releases/latest")
    if not isinstance(value, dict):
        raise ValueError(f"Unexpected releases response for {repo}")
    tag = str(value["tag_name"])
    return tag.lstrip("v")


def _github_latest_build(repo: str) -> str:
    value = _fetch_json(f"https://api.github.com/repos/{repo}/releases?per_page=100")
    if not isinstance(value, list):
        raise ValueError(f"Unexpected releases response for {repo}")
    builds = []
    for item in value:
        if not isinstance(item, dict):
            continue
        tag = str(item.get("tag_name") or item.get("name") or "")
        match = re.fullmatch(r"b(\d+)", tag)
        if match:
            builds.append((int(match.group(1)), tag))
    if not builds:
        raise ValueError(f"No build releases returned for {repo}")
    return max(builds)[1]


def _pypi_latest(distribution: str) -> str:
    value = _fetch_json(f"https://pypi.org/pypi/{distribution}/json")
    if not isinstance(value, dict):
        raise ValueError(f"Unexpected PyPI response for {distribution}")
    return str(value["info"]["version"])


def specs() -> list[RuntimeSpec]:
    return [
        RuntimeSpec(
            "llama.cpp",
            _llamacpp_version,
            lambda: _github_latest_build("ggml-org/llama.cpp"),
        ),
        RuntimeSpec("mlx", lambda: _package_version("mlx"), lambda: _pypi_latest("mlx")),
        RuntimeSpec("mlx-lm", lambda: _package_version("mlx-lm"), lambda: _pypi_latest("mlx-lm")),
        RuntimeSpec("mlx-vlm", lambda: _package_version("mlx-vlm"), lambda: _pypi_latest("mlx-vlm")),
    ]


def check_one(spec: RuntimeSpec) -> RuntimeStatus:
    try:
        installed = spec.installed_version()
    except Exception as exc:
        return RuntimeStatus(spec.name, None, None, "missing", str(exc))
    try:
        latest = spec.latest_version()
    except Exception as exc:
        return RuntimeStatus(spec.name, installed, None, "unknown", str(exc))
    state = "outdated" if is_newer(latest, installed) else "current"
    return RuntimeStatus(spec.name, installed, latest, state)


def check_all() -> list[RuntimeStatus]:
    return [check_one(spec) for spec in specs()]


def as_json(statuses: list[RuntimeStatus]) -> str:
    return json.dumps([asdict(status) for status in statuses], indent=2)


def format_text(statuses: list[RuntimeStatus]) -> str:
    lines = ["Runtime updates", "=" * 64]
    for status in statuses:
        installed = status.installed or "not installed"
        latest = status.latest or "unknown"
        lines.append(f"{status.name:<12} {status.state:<8} installed={installed:<12} latest={latest}")
        if status.note:
            lines.append(f"  {status.note}")
    return "\n".join(lines)
