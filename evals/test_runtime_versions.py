"""Version parsing and update reporting for heterogeneous local runtimes."""
from __future__ import annotations

import sys
from pathlib import Path
import ssl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import runtime_versions as rv


def test_extract_version_handles_llamacpp_build_and_python_packages():
    assert rv.extract_version("version: 10360 (48d22e295)") == "10360"
    assert rv.extract_version(
        "version: 10360 (48d22e295)\nbuilt with AppleClang 21.0.0.21000101"
    ) == "10360"
    assert rv.extract_version("mlx-lm 0.31.3") == "0.31.3"
    assert rv.extract_version("oMLX 0.4.2rc1") == "0.4.2rc1"


def test_extract_llamacpp_build_prefers_build_over_homebrew_package_version():
    assert rv.extract_llamacpp_build(
        "version: 0.3.0 (build 10621, commit c1d0e7a00)"
    ) == "10621"
    assert rv.extract_llamacpp_build("version: 10360 (48d22e295)") == "10360"


def test_version_comparison_understands_build_tags_and_prereleases():
    assert rv.is_newer("b10809", "10360") is True
    assert rv.is_newer("0.32.2", "0.32.0") is True
    assert rv.is_newer("0.6.4", "0.4.2rc1") is True
    assert rv.is_newer("0.31.3", "0.31.3") is False


def test_check_one_reports_unknown_latest_as_unknown():
    status = rv.check_one(
        rv.RuntimeSpec("mlx", lambda: "0.32.0", lambda: (_ for _ in ()).throw(OSError("offline")))
    )

    assert status.state == "unknown"
    assert status.installed == "0.32.0"
    assert status.latest is None
    assert "offline" in status.note


def test_check_one_reports_outdated_and_current():
    old = rv.check_one(rv.RuntimeSpec("mlx", lambda: "0.32.0", lambda: "0.32.2"))
    current = rv.check_one(rv.RuntimeSpec("mlx-lm", lambda: "0.31.3", lambda: "0.31.3"))

    assert old.state == "outdated"
    assert current.state == "current"


def test_fetch_json_supplies_an_explicit_trust_context(monkeypatch):
    seen = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"ok": true}'

    def open_url(_request, *, timeout, context):
        seen["timeout"] = timeout
        seen["context"] = context
        return Response()

    monkeypatch.setattr(rv.urllib.request, "urlopen", open_url)

    assert rv._fetch_json("https://example.test/releases") == {"ok": True}
    assert seen["timeout"] == 10
    assert isinstance(seen["context"], ssl.SSLContext)


def test_llamacpp_latest_uses_highest_build_tag(monkeypatch):
    monkeypatch.setattr(rv, "_fetch_json", lambda _url: [
        {"name": "b10807"},
        {"name": "master-old"},
        {"name": "b10809"},
        {"name": "b10808"},
    ])

    assert rv._github_latest_build("ggml-org/llama.cpp") == "b10809"
