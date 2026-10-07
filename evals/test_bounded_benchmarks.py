"""Deadline cleanup must include servers that create separate process sessions."""
import importlib
import importlib.util
from pathlib import Path
import sys
import time

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def supervisor():
    assert importlib.util.find_spec("run_practical_benchmarks"), "bounded benchmark supervisor is missing"
    return importlib.import_module("run_practical_benchmarks")


def test_deadline_stops_worker_and_separate_session_child(tmp_path):
    module = supervisor()
    code = ("import subprocess,sys,time; "
            "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True); "
            "print(p.pid,flush=True); time.sleep(60)")
    log = tmp_path / "child.log"
    started = time.monotonic()
    result = module.run_bounded([sys.executable, "-c", code], log,
                                deadline=time.monotonic() + 1, min_available_bytes=0)
    assert result["state"] == "timed_out"
    assert time.monotonic() - started < 8
    pid = int(log.read_text().strip())
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE


def test_success_is_not_mislabeled_as_timeout(tmp_path):
    result = supervisor().run_bounded([sys.executable, "-c", "print('done')"],
                                      tmp_path / "ok.log", deadline=time.monotonic() + 5,
                                      min_available_bytes=0)
    assert result == {"state": "complete", "exit_code": 0}


def test_expired_budget_does_not_start_work(tmp_path):
    log = tmp_path / "not_started.log"
    result = supervisor().run_bounded([sys.executable, "-c", "raise Exception('must not launch')"],
                                      log, deadline=time.monotonic() - 1,
                                      min_available_bytes=0)
    assert result["state"] == "timed_out"
    assert not log.exists()


def test_mlx_jobs_launch_through_the_canonical_launcher(launcher_env, tmp_path):
    """Benchmarks must measure the runtime daily serving uses (docs/SERVING.md).

    MLX jobs used to start `omlx serve`, a different server with different
    defaults that is no longer installed at all.
    """
    import local_config

    module = supervisor()
    config = local_config.load_config(Path(launcher_env["LOCAL_AI_CONFIG"]))
    job = next(j for j in module.build_jobs(config, tmp_path / "out") if j["backend"] == "mlx")
    serve_local = Path(__file__).resolve().parents[1] / "scripts" / "serve_local.sh"
    assert job["command"][:6] == ["bash", str(serve_local), job["path"], "18080",
                                  "--backend", "mlx"]
    assert job["model"] == job["path"]
