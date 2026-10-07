"""Run the complete local catalog sequentially with strict wall-clock budgets."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time

import psutil

import benchmark_profiles
import local_config
import model_registry
from model_files import validate_model_path

ROOT = Path(__file__).resolve().parents[1]
SERVE_LOCAL = ROOT / "scripts" / "serve_local.sh"


def stop_owned(processes: dict[int, psutil.Process]) -> None:
    # Process objects retain creation times, preventing signals to reused PIDs.
    alive = []
    for process in reversed(list(processes.values())):
        try:
            if process.is_running():
                process.terminate()
                alive.append(process)
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(alive, timeout=3)
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=3)


def run_bounded(command: list[str], log_path: Path, *, deadline: float,
                min_available_bytes: int = 8 * 1024**3) -> dict:
    if time.monotonic() >= deadline:
        return {"state": "timed_out", "exit_code": None}
    owned: dict[int, psutil.Process] = {}
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=log,
                                   start_new_session=True,
                                   env={**os.environ, "HF_HUB_OFFLINE": "1",
                                        "TRANSFORMERS_OFFLINE": "1", "PYTHONUNBUFFERED": "1"})
        parent = psutil.Process(process.pid)
        owned[parent.pid] = parent
        low_since = None
        try:
            while process.poll() is None:
                try:
                    for child in parent.children(recursive=True):
                        owned[child.pid] = child
                except psutil.NoSuchProcess:
                    pass
                now = time.monotonic()
                if now >= deadline:
                    return {"state": "timed_out", "exit_code": None}
                low = psutil.virtual_memory().available < min_available_bytes
                low_since = (low_since or now) if low else None
                if low_since is not None and now - low_since >= 10:
                    return {"state": "resource_stopped", "exit_code": None}
                time.sleep(min(0.2, max(0, deadline - now)))
            return {"state": "complete" if process.returncode == 0 else "failed",
                    "exit_code": process.returncode}
        finally:
            stop_owned(owned)
            process.wait(timeout=5)


def build_jobs(config: dict, output: Path) -> list[dict]:
    rows = [row for row in model_registry.iter_models(Path(config["models"]["registry"]))
            if row.get("exists")]
    rows.sort(key=lambda row: ("bonsai" not in row["family_id"], row["family_id"], row["quant"]))
    jobs = []
    for row in rows:
        validate_model_path(Path(row["path"]), Path(config["models"]["root"]), row["backend"])
        job_id = row["family_id"] + "_" + row["quant"]
        if row["backend"] == "mlx":
            # The canonical launcher, so this measures the runtime daily serving
            # uses. mlx_lm.server loads whatever a request names, so the API
            # model id has to be the loaded path itself.
            model = row["path"]
            binary = str(SERVE_LOCAL)
            command = ["bash", binary, row["path"], "18080", "--backend", "mlx",
                       "--host", "127.0.0.1"]
        else:
            model = job_id
            binary = shutil.which("llama-server")
            command = [binary, "-m", row["path"], "--alias", model, "--host", "127.0.0.1",
                       "--port", "18080", "-ngl", "all", "-c", "49152", "-np", "1",
                       "-ctk", "q8_0", "-ctv", "q8_0", "-fa", "on",
                       "--cache-ram", "1024", "--reasoning-budget", "0"]
        if not binary:
            raise ValueError(f"Missing serving runtime for {model}")
        jobs.append({"id": job_id, "model": model, "path": row["path"],
                     "backend": "mlx" if row["backend"] == "mlx" else "llamacpp",
                     "command": command, "state": "queued"})
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=local_config.DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    config = local_config.load_config(args.config)
    jobs = build_jobs(config, output)
    profile = benchmark_profiles.get_profile(benchmark_profiles.load_profiles(), "local_practical")
    state = {"profile": profile, "jobs": jobs, "max_hours": 8, "per_model_minutes": 70}
    if args.dry_run:
        print(json.dumps(state, indent=2))
        return
    # Verify the cached tokenizer before starting any model or consuming the budget.
    from bench_profile_api import load_encoding
    load_encoding()
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    deadline = time.monotonic() + 8 * 3600 - 15  # reserve cleanup time
    state.update(started_at=started, stop_by=started + 8 * 3600)

    def save():
        state["updated_at"] = time.time()
        temporary = output / "queue.json.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(output / "queue.json")

    save()
    try:
        for job in jobs:
            if time.monotonic() >= deadline:
                break
            job.update(state="running", started_at=time.time())
            save()
            print(f"START {job['model']}", flush=True)
            command = [sys.executable, str(ROOT / "scripts/bench_profile_api.py"),
                       "--profile", "local_practical", "--config", str(args.config.resolve()),
                       "--url", "http://127.0.0.1:18080/v1", "--model", job["model"],
                       "--server-command", shlex.join(job["command"]),
                       "--output-dir", str(output / job["id"]), "--timeout", "300", "--no-thinking"]
            try:
                result = run_bounded(command, output / (job["id"] + ".log"),
                                     deadline=min(deadline, time.monotonic() + 70 * 60))
                job.update(result, finished_at=time.time())
                summary = output / job["id"] / "summary.json"
                if summary.exists():
                    job["profile_conformant"] = json.loads(summary.read_text())["profile_conformant"]
                    job["state"] = "complete" if job["profile_conformant"] else "complete_nonconformant"
            except (KeyboardInterrupt, SystemExit):
                job.update(state="interrupted", finished_at=time.time())
                raise
            save()
            print(f"END {job['model']}: {job['state']}", flush=True)
            if job["state"] == "resource_stopped":
                break
    finally:
        for job in jobs:
            if job["state"] == "queued":
                job["state"] = "not_run"
        state["finished_at"] = time.time()
        save()


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    main()
