#!/usr/bin/env python3
"""Unit tests for the pure logic in webgui/serve.py.

These cover the data-shaping the GUI depends on — the parts where a wrong
answer silently corrupts what the user can launch — not the HTTP plumbing.
Run: python3 -m pytest webgui/test_serve.py   (or: python3 webgui/test_serve.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import serve  # noqa: E402


# --- port routing -----------------------------------------------------------

def test_port_for_routes_llamacpp_to_8080():
    assert serve.port_for("llamacpp") == 8080


def test_port_for_routes_all_mlx_variants_to_8085():
    # mlx, mlx-kv and mlx-vlm are three flavors of the same server/port; a
    # regression that split them would proxy chat to a dead port.
    assert serve.port_for("mlx") == 8085
    assert serve.port_for("mlx-kv") == 8085
    assert serve.port_for("mlx-vlm") == 8085


def test_port_for_unknown_backend_falls_back_to_llamacpp():
    assert serve.port_for("nonsense") == 8080


# --- instruct family shaping ------------------------------------------------

def _fake_rows(monkeypatch, rows):
    monkeypatch.setattr(serve.model_registry, "iter_models", lambda _cfg: rows)


def test_instruct_family_offers_only_plain_backends(monkeypatch):
    # mlx-kv (mlx_vlm + KV q8) is intentionally NOT offered in the GUI picker —
    # it's a niche CLI-only variant. A plain on-disk MLX family shows just "mlx".
    _fake_rows(monkeypatch, [
        {"family_id": "fam", "aliases": ["fam"], "backend": "mlx",
         "exists": True, "use_case": "x", "quant": "4bit", "mtp_supported": False},
    ])
    fams = serve.instruct_families()
    assert len(fams) == 1
    assert fams[0]["backends"] == ["mlx"]
    assert fams[0]["type"] == "instruct"


def test_instruct_family_surfaces_mlx_vlm_pin(monkeypatch):
    # The resolved MLX server is surfaced so the UI can name the exact backend.
    _fake_rows(monkeypatch, [
        {"family_id": "fam", "aliases": ["fam"], "backend": "mlx", "exists": True,
         "use_case": "", "quant": "4bit", "mtp_supported": False, "mlx_server": "mlx_vlm"},
    ])
    assert serve.instruct_families()[0]["mlx_server"] == "mlx_vlm"


def test_instruct_family_hides_backends_not_on_disk(monkeypatch):
    # A registry row whose weights are missing must not be offered — launching
    # it would error or (for MLX repo-ids) trigger a download.
    _fake_rows(monkeypatch, [
        {"family_id": "fam", "aliases": ["fam"], "backend": "llamacpp",
         "exists": False, "use_case": "", "quant": "?", "mtp_supported": True},
    ])
    assert serve.instruct_families() == []


def test_instruct_family_flags_mtp_only_from_ondisk_llamacpp(monkeypatch):
    _fake_rows(monkeypatch, [
        {"family_id": "fam", "aliases": ["fam"], "backend": "llamacpp",
         "exists": True, "use_case": "", "quant": "Q4", "mtp_supported": True},
    ])
    assert serve.instruct_families()[0]["mtp_supported"] is True


def test_model_catalog_has_no_outside_registry_base_models(monkeypatch):
    _fake_rows(monkeypatch, [])

    assert serve.model_catalog() == {"instruct": [], "base": []}


# --- serve_model command construction ---------------------------------------

class _FakeChild:
    pid = 4242


def _capture_launch(monkeypatch):
    """Stub out resolve / kill / Popen; return a list that records the launch cmd."""
    captured = []
    monkeypatch.setattr(serve.model_registry, "resolve", lambda *a, **k: {"exists": True})
    monkeypatch.setattr(serve, "_kill_port", lambda port: None)
    monkeypatch.setattr(serve.subprocess, "Popen",
                        lambda cmd, **kw: captured.append(cmd) or _FakeChild())
    return captured


def test_serve_model_omits_no_mtp_by_default(monkeypatch):
    captured = _capture_launch(monkeypatch)
    result = serve.serve_model("qwen35", "llamacpp")
    assert "--no-mtp" not in captured[0]
    assert result["port"] == 8080 and result["no_mtp"] is False


def test_serve_model_passes_no_mtp_when_requested(monkeypatch):
    # The UI's MTP-off path must actually reach serve_local.sh, or toggling it
    # would silently do nothing.
    captured = _capture_launch(monkeypatch)
    serve.serve_model("qwen35", "llamacpp", no_mtp=True)
    assert "--no-mtp" in captured[0]


def test_serve_model_rejects_unknown_backend(monkeypatch):
    _capture_launch(monkeypatch)
    try:
        serve.serve_model("qwen35", "bogus")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_serve_model_resolves_through_configured_registry(monkeypatch):
    seen = []
    monkeypatch.setattr(
        serve.model_registry,
        "resolve",
        lambda selector, backend, registry: seen.append((selector, backend, registry))
        or {"exists": True, "path": "/models/fixture"},
    )
    monkeypatch.setattr(serve, "_kill_port", lambda _port: None)
    monkeypatch.setattr(serve.subprocess, "Popen", lambda _cmd, **_kw: _FakeChild())

    serve.serve_model("fixture", "mlx")

    assert seen == [("fixture", "mlx", serve.MODEL_REGISTRY)]


# --- proxy model-field injection --------------------------------------------
# The MLX servers disagree on the "model" field: mlx_vlm NEEDS the launched path
# (else it loads a default model and 500s offline); mlx_lm 404s on a wrong cached
# id but accepts its own path. The proxy must echo back exactly what was launched.

import json as _json


def _inject(body, port):
    # _inject_model doesn't touch self; call it unbound for a focused test.
    return serve.Handler._inject_model(None, body, port)


def test_inject_pins_launched_path_when_model_absent(monkeypatch):
    monkeypatch.setattr(serve, "_PORT_MODEL", {8085: "/models/gemma-e4b"})
    out = _inject(b'{"messages": [], "stream": true}', 8085)
    assert _json.loads(out)["model"] == "/models/gemma-e4b"


def test_inject_does_not_override_explicit_client_model(monkeypatch):
    monkeypatch.setattr(serve, "_PORT_MODEL", {8085: "/models/gemma-e4b"})
    out = _inject(b'{"model": "chosen", "messages": []}', 8085)
    assert _json.loads(out)["model"] == "chosen"


def test_inject_uses_the_running_servers_own_path_when_this_gui_did_not_launch_it(monkeypatch):
    # A server started in a prior process (or from the CLI) has no _PORT_MODEL
    # record, but its `--model` argument is exactly what it serves under — so
    # this is the running server's own answer, not a guessed id.
    monkeypatch.setattr(serve, "_PORT_MODEL", {})
    monkeypatch.setattr(
        serve.serving_lifecycle, "loaded_model", lambda port: "/models/from-argv"
    )
    out = _inject(b'{"messages": []}', 8085)
    assert _json.loads(out)["model"] == "/models/from-argv"


def test_inject_noop_when_nothing_is_running(monkeypatch):
    # Nothing to read a path from → unchanged, rather than injecting an id that
    # would 404 on mlx_lm.
    monkeypatch.setattr(serve, "_PORT_MODEL", {})
    monkeypatch.setattr(serve.serving_lifecycle, "loaded_model", lambda port: None)
    body = b'{"messages": []}'
    assert _inject(body, 8085) == body


def test_inject_passes_through_non_json(monkeypatch):
    monkeypatch.setattr(serve, "_PORT_MODEL", {8085: "/m"})
    assert _inject(b"not json", 8085) == b"not json"


def test_serve_model_records_launched_path(monkeypatch):
    monkeypatch.setattr(serve.model_registry, "resolve",
                        lambda *a, **k: {"exists": True, "path": "/models/qwen27"})
    monkeypatch.setattr(serve, "_kill_port", lambda port: None)
    monkeypatch.setattr(serve.subprocess, "Popen", lambda cmd, **kw: _FakeChild())
    monkeypatch.setattr(serve, "_PORT_MODEL", {})
    result = serve.serve_model("qwen27", "mlx")
    assert result["model_path"] == "/models/qwen27"
    assert serve._PORT_MODEL[8085] == "/models/qwen27"


# --- chat template must not be force-overridden -----------------------------

def test_gemma4_chat_template_not_forced():
    # Forcing --chat-template gemma2 on the gemma-4-26B QAT GGUF (which carries
    # its own <|turn>/<|channel> thinking template) mangles the prompt and the
    # model collapses into garbage ("9b 9b 9b…"). Empty = use the embedded one.
    assert serve.model_registry._chat_template("gemma4") == ""


# --- idle watchdog ----------------------------------------------------------

def test_should_shutdown_true_when_idle_past_timeout(monkeypatch):
    monkeypatch.setattr(serve, "_active_requests", 0)
    monkeypatch.setattr(serve, "_last_activity", serve.time.time() - 600)
    assert serve._should_shutdown(300) is True


def test_should_shutdown_false_when_recently_active(monkeypatch):
    monkeypatch.setattr(serve, "_active_requests", 0)
    monkeypatch.setattr(serve, "_last_activity", serve.time.time())
    assert serve._should_shutdown(300) is False


def test_should_shutdown_false_while_generation_in_flight(monkeypatch):
    # A long generation must NOT be killed mid-stream: in-flight requests block
    # the idle shutdown even if the last-activity stamp looks old.
    monkeypatch.setattr(serve, "_active_requests", 1)
    monkeypatch.setattr(serve, "_last_activity", serve.time.time() - 600)
    assert serve._should_shutdown(300) is False


def test_enter_leave_request_balances_active_count(monkeypatch):
    monkeypatch.setattr(serve, "_active_requests", 0)
    serve._enter_request()
    assert serve._active_requests == 1
    serve._leave_request()
    assert serve._active_requests == 0


def test_unload_all_models_kills_tracked_ports(monkeypatch):
    killed = []
    monkeypatch.setattr(serve, "_kill_port", lambda p: killed.append(p))
    monkeypatch.setattr(serve, "_PORT_MODEL", {8085: "/m"})
    serve._unload_all_models()
    assert killed == [8085]


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))


def test_status_reports_only_the_serving_ports(monkeypatch):
    """:8000 is Mycelium's backend now. Reporting it as an always-on oMLX
    server showed another app's process as one of ours."""
    monkeypatch.setattr(serve, "_probe", lambda _port: (False, None))
    assert [row["port"] for row in serve.status()] == [8080, 8085]
