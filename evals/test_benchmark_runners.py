"""Runner lifecycle regressions that can deadlock or leak sampling threads."""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import bench_llamacpp_api
import bench_mlx_api


def test_llamacpp_api_starts_one_memory_sampler_per_attempt():
    source = inspect.getsource(bench_llamacpp_api.run_benchmark)

    assert source.count("sampler = MemSampler(); sampler.start()") == 1


def test_mlx_server_does_not_use_an_undrained_stderr_pipe():
    source = inspect.getsource(bench_mlx_api._start_server)

    assert "stderr=subprocess.PIPE" not in source


def test_ordinary_mlx_benchmark_uses_canonical_server_launcher():
    source = inspect.getsource(bench_mlx_api._start_server)

    assert 'serve_script = SCRIPT_DIR / "serve_local.sh"' in source
    assert '"--backend", backend' in source


def test_mlx_vlm_run_refuses_options_it_would_drop(monkeypatch):
    """A drafted or concurrent mlx_vlm speed run used to launch a plain
    `mlx_vlm.server` found on PATH — possibly not the core runtime at all —
    and quietly ignore the requested option, so the result measured something
    other than what was asked."""
    import pytest

    def refuse_launch(cmd, **_kwargs):
        raise AssertionError(f"must not launch a server: {cmd}")

    monkeypatch.setattr(bench_mlx_api.subprocess, "Popen", refuse_launch)
    with pytest.raises(ValueError, match="mlx_vlm"):
        bench_mlx_api._start_server("/models/vlm", 18085, draft_model="/models/draft",
                                    server="mlx_vlm")
    with pytest.raises(ValueError, match="mlx_vlm"):
        bench_mlx_api._start_server("/models/vlm", 18085, decode_concurrency=4,
                                    server="mlx_vlm")
