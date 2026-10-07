"""TUI server readiness and model identity regressions."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bench_tui


class _Response:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def test_server_identity_accepts_full_path_for_expected_basename(monkeypatch):
    monkeypatch.setattr(
        bench_tui.requests,
        "get",
        lambda *_args, **_kwargs: _Response({
            "data": [{"id": "/models/Qwen3.8-27B-Q4_K_M.gguf"}],
        }),
    )

    assert bench_tui._server_has_expected_model("Qwen3.8-27B-Q4_K_M.gguf")


def test_tui_server_launchers_do_not_use_undrained_stderr_pipes():
    source = Path(bench_tui.__file__).read_text()

    assert "stderr=subprocess.PIPE" not in source
