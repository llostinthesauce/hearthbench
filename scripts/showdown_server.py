#!/usr/bin/env python3
"""
Supervised local servers for the showdown matrix, plus the systems measurements.

Split out of the runner because three things here are easy to get wrong and are
worth being able to test on their own:

  * **Readiness.** A listening port is not readiness and neither is /v1/models —
    llama-server answers the latter with 200 while the model is still loading.
    Only a real completion means the same thing across llama.cpp, mlx_lm and
    mlx_vlm, so `EvalClient.probe` is what gates every arm.

  * **Teardown.** `serve_local.sh` exec's the real server, and the MLX servers
    spawn children. Killing the PID we started is not enough; the whole process
    group has to go, and the port has to be observed free before the next arm
    starts, or arm N+1 measures arm N's resident weights.

  * **Startup failure is not a quality result.** An arm that never became ready
    records `status=startup_failed` with the server log path and is never handed
    to the eval suite, so it cannot appear in the matrix as a model that scored
    zero.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import memutil  # noqa: E402
from evals.core import Case, EvalClient  # noqa: E402


@dataclass
class MemorySample:
    """Unified-memory figures around one arm.

    On Apple Silicon there is no separate VRAM figure to read, so the two useful
    numbers are the server process tree's resident set (what this model costs)
    and system-wide used memory (whether the machine is under pressure). Both are
    recorded; neither alone answers "will this configuration be usable".
    """

    baseline_used_bytes: int = 0
    peak_used_bytes: int = 0
    peak_proc_rss_bytes: int = 0
    steady_proc_rss_bytes: int = 0
    peak_system_percent: float = 0.0
    min_available_bytes: int = 0
    swap_delta_bytes: int = 0

    @property
    def attributed_peak_bytes(self) -> int:
        """System-wide used memory above the pre-launch baseline.

        This, not RSS, is the number to quote for an MLX arm. MLX allocates
        weights through Metal buffers that do not appear in the process resident
        set: a measured 15 GB 4-bit checkpoint reported 7.9 GB of RSS while
        system used memory rose by ~12.6 GB. RSS is still recorded because it is
        the right figure for llama.cpp, but it systematically under-reports MLX.

        The baseline is whatever else the machine was doing, so this is an
        attribution rather than an isolated measurement; anything else running
        during the arm inflates or deflates it.
        """
        return max(0, self.peak_used_bytes - self.baseline_used_bytes)


class MemoryWatcher:
    """Samples the server tree and the system in the background."""

    def __init__(self, pid: int, interval: float = 1.0) -> None:
        self.pid = pid
        self.interval = interval
        self.sample = MemorySample()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        try:
            self._swap_start = psutil.swap_memory().used
        except Exception:  # noqa: BLE001
            self._swap_start = 0
        vm = memutil.virtual_memory()
        self.sample.baseline_used_bytes = int(vm.used)
        self.sample.min_available_bytes = int(vm.available)

    def start(self) -> "MemoryWatcher":
        self._thread.start()
        return self

    def _tree_rss(self) -> int:
        try:
            parent = psutil.Process(self.pid)
            total = parent.memory_info().rss
            for child in parent.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except psutil.Error:
                    pass
            return total
        except psutil.Error:
            return 0

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                vm = memutil.virtual_memory()
                rss = self._tree_rss()
                self.sample.peak_used_bytes = max(self.sample.peak_used_bytes, int(vm.used))
                self.sample.peak_proc_rss_bytes = max(self.sample.peak_proc_rss_bytes, rss)
                self.sample.peak_system_percent = max(self.sample.peak_system_percent, float(vm.percent))
                self.sample.min_available_bytes = min(self.sample.min_available_bytes, int(vm.available))
                if rss:
                    self.sample.steady_proc_rss_bytes = rss
            except Exception:  # noqa: BLE001 — sampling must never kill an arm
                pass
            self._stop.wait(self.interval)

    def stop(self) -> MemorySample:
        self._stop.set()
        self._thread.join(timeout=5)
        try:
            self.sample.swap_delta_bytes = max(0, psutil.swap_memory().used - self._swap_start)
        except Exception:  # noqa: BLE001
            pass
        return self.sample


def port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.5)
        return probe.connect_ex((host, port)) != 0


def wait_for_port_free(host: str, port: int, timeout: float = 90.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if port_is_free(host, port):
            return True
        time.sleep(1.0)
    return False


@dataclass
class ServerHandle:
    process: subprocess.Popen
    command: List[str]
    log_path: Path
    host: str
    port: int
    watcher: MemoryWatcher
    launched_at: float
    ready_at: Optional[float] = None
    owned: Dict[int, psutil.Process] = field(default_factory=dict)

    @property
    def load_seconds(self) -> Optional[float]:
        return round(self.ready_at - self.launched_at, 2) if self.ready_at else None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"


def launch(command: List[str], *, host: str, port: int, log_path: Path,
           env: Optional[Dict[str, str]] = None) -> ServerHandle:
    """Start a server via serve_local.sh, draining its output to a file.

    The output goes to a file rather than a pipe on purpose: an unread pipe fills
    its buffer and wedges the server, which then looks exactly like a model that
    stopped generating.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle_env = {**os.environ, **(env or {})}
    log = log_path.open("w")
    process = subprocess.Popen(
        command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True, env=handle_env,
    )
    watcher = MemoryWatcher(process.pid).start()
    return ServerHandle(process=process, command=list(command), log_path=log_path,
                        host=host, port=port, watcher=watcher, launched_at=time.monotonic())


def await_ready(handle: ServerHandle, model_id: str, *, timeout: int = 900,
                api_key: str = "") -> Optional[str]:
    """Block until a real completion succeeds. Returns an error string on failure."""
    client = EvalClient(api_base=handle.base_url, model=model_id, api_key=api_key,
                        timeout=120, enable_thinking=False)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if handle.process.poll() is not None:
            tail = _log_tail(handle.log_path)
            return (f"server exited with code {handle.process.returncode} before becoming ready. "
                    f"Log tail: {tail}")
        problem = client.probe(wait_s=10)
        if problem is None:
            handle.ready_at = time.monotonic()
            return None
    return (f"server did not answer a completion within {timeout}s. "
            f"Log tail: {_log_tail(handle.log_path)}")


def _log_tail(path: Path, lines: int = 12) -> str:
    try:
        content = path.read_text(errors="replace").strip().splitlines()
    except OSError:
        return "(no log)"
    return " | ".join(content[-lines:])[-1200:]


def shutdown(handle: ServerHandle, *, grace: float = 10.0) -> MemorySample:
    """Terminate the whole process group and wait for the port to go quiet."""
    sample = handle.watcher.stop()
    procs: List[psutil.Process] = []
    try:
        parent = psutil.Process(handle.process.pid)
        procs = [parent] + parent.children(recursive=True)
    except psutil.Error:
        procs = []

    try:
        os.killpg(os.getpgid(handle.process.pid), 15)
    except (ProcessLookupError, PermissionError, OSError):
        for proc in procs:
            try:
                proc.terminate()
            except psutil.Error:
                pass

    _, alive = psutil.wait_procs(procs, timeout=grace)
    for proc in alive:
        try:
            proc.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(alive, timeout=5)
    try:
        handle.process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    wait_for_port_free(handle.host, handle.port, timeout=90)
    return sample


# ---------------------------------------------------------------------------
# throughput / latency
# ---------------------------------------------------------------------------

PERF_PROBES = (
    # (label, approximate prompt tokens, generated tokens)
    ("short", 256, 256),
    ("medium", 8192, 256),
    ("long", 32768, 256),
)

_FILLER = (
    "The archivist documented the quarterly variance report before the end of the fiscal "
    "quarter. A field technician revised three unresolved maintenance tickets under the "
    "revised handling procedure. The night operator catalogued the calibration log for bay "
    "four while the primary system was offline. "
)


def _prompt_of(target_tokens: int) -> str:
    # ~1.15 BPE tokens per whitespace word, the same calibration niah uses, so a
    # "32K" probe here means the same thing as a 32K eval case.
    words_needed = max(8, int(target_tokens / 1.15))
    words = _FILLER.split()
    repeated = (words * (words_needed // len(words) + 1))[:words_needed]
    return " ".join(repeated)


def measure_throughput(base_url: str, model_id: str, *, api_key: str = "",
                       probes=PERF_PROBES, timeout: int = 600) -> List[Dict[str, Any]]:
    """TTFT, prompt tok/s and generation tok/s at a few prompt sizes.

    Uses the server's own `usage.prompt_tokens` for the prompt rate rather than a
    local tokenizer count, because the question is how fast *this stack*
    processed what it actually received.
    """
    import urllib.error
    import urllib.request

    results: List[Dict[str, Any]] = []
    # tokens-per-word for THIS model's tokenizer, measured once on the cheap
    # probe. See _derive_prompt_tokens for why this exists.
    calibration: Dict[str, float] = {}
    for label, prompt_tokens, gen_tokens in probes:
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": "Continue the passage in plain prose. Do not summarize."},
                {"role": "user", "content": _prompt_of(prompt_tokens) + "\n\nContinue this passage."},
            ],
            "max_tokens": gen_tokens,
            "temperature": 0.0,
            "stream": True,
            # llama.cpp honours this and emits a final usage chunk. mlx_lm.server
            # does not, which is why there is a fallback below — without it every
            # MLX arm would report prompt_tps=None and the prefill cost of a
            # quantization would be invisible.
            "stream_options": {"include_usage": True},
            "enable_thinking": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode(), method="POST", headers=headers)

        row: Dict[str, Any] = {"probe": label, "requested_prompt_tokens": prompt_tokens}
        started = time.perf_counter()
        ttft: Optional[float] = None
        produced = 0
        served_prompt_tokens = 0
        try:
            with urllib.request.urlopen(request, timeout=timeout) as stream:
                for raw in stream:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if body == "[DONE]":
                        break
                    try:
                        chunk = json.loads(body)
                    except ValueError:
                        continue
                    usage = chunk.get("usage") or {}
                    if usage.get("prompt_tokens"):
                        served_prompt_tokens = int(usage["prompt_tokens"])
                    choices = chunk.get("choices") or [{}]
                    delta = choices[0].get("delta") or {}
                    piece = delta.get("content") or ""
                    if piece:
                        if ttft is None:
                            ttft = time.perf_counter() - started
                        produced += 1
            elapsed = time.perf_counter() - started
            source = "stream_usage"
            if not served_prompt_tokens:
                served_prompt_tokens, source = _derive_prompt_tokens(
                    base_url, model_id, payload["messages"], prompt_tokens,
                    calibration, api_key=api_key)
            row.update({
                "status": "ok" if produced else "no_tokens",
                "ttft_s": round(ttft, 3) if ttft is not None else None,
                "served_prompt_tokens": served_prompt_tokens or None,
                "prompt_token_source": source if served_prompt_tokens else None,
                "generated_chunks": produced,
                "total_s": round(elapsed, 3),
                "gen_tps": round(produced / (elapsed - (ttft or 0)), 2) if produced and elapsed > (ttft or 0) else None,
                # Prefill rate. TTFT includes queueing and sampling setup as well
                # as prefill, so this is an end-to-end prefill rate, not a pure
                # matmul figure — comparable across arms, not against a vendor number.
                "prompt_tps": (round(served_prompt_tokens / ttft, 1)
                               if served_prompt_tokens and ttft else None),
            })
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
            row.update({"status": f"error: {type(exc).__name__}: {exc}"[:200],
                        "ttft_s": None, "gen_tps": None, "prompt_tps": None})
        results.append(row)
        if row["status"] != "ok":
            # A stack that failed at 8K will fail at 32K too, and each failure
            # costs a full timeout. Stop climbing.
            break
    return results


def _usage_prompt_tokens(base_url: str, model_id: str, messages, *,
                         api_key: str = "", timeout: int = 300) -> int:
    """Read `usage.prompt_tokens` from a one-token non-streaming completion."""
    import urllib.error
    import urllib.request

    payload = {"model": model_id, "messages": messages, "max_tokens": 1,
               "temperature": 0.0, "stream": False}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(f"{base_url.rstrip('/')}/chat/completions",
                                     data=json.dumps(payload).encode(),
                                     method="POST", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as raw:
            data = json.loads(raw.read())
        return int((data.get("usage") or {}).get("prompt_tokens") or 0)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError, ValueError):
        return 0


def _derive_prompt_tokens(base_url: str, model_id: str, messages, requested: int,
                          calibration: Dict[str, float], *, api_key: str = ""):
    """Prompt-token count for a server whose stream carries no usage block.

    `mlx_lm.server` emits no usage in its streaming chunks and ignores
    `stream_options`, so the count has to come from somewhere else. The obvious
    move — repeat the request non-streaming and read its usage — costs a second
    full prefill, and at 32K that was measured at ~80 seconds per probe per arm.
    Across a 19-arm matrix that is half an hour of re-prefilling to learn a
    number that does not change.

    So the ratio of tokens to whitespace words is measured **once**, on the
    cheapest probe, against this exact model's tokenizer, and the larger probes
    are derived from their own word counts. The source is recorded on every row
    (`stream_usage` exact, `calibrated` derived) so a rate is never presented as
    measured when it was inferred.
    """
    words = sum(len(str(m.get("content") or "").split()) for m in messages)
    if "tokens_per_word" not in calibration:
        # Only pay for this on a small prompt; a big one is the thing we are
        # avoiding. `requested` is the probe's own target size.
        if requested > 2048:
            return 0, "unavailable"
        exact = _usage_prompt_tokens(base_url, model_id, messages, api_key=api_key)
        if not exact or not words:
            return 0, "unavailable"
        calibration["tokens_per_word"] = exact / words
        return exact, "measured_usage"
    return int(round(words * calibration["tokens_per_word"])), "calibrated"


def probe_max_context(base_url: str, model_id: str, *, api_key: str = "",
                      ladder=(32768, 65536, 131072, 262144), timeout: int = 900,
                      min_available_bytes: int = 8 * 1024 ** 3,
                      max_context: int = 0) -> Dict[str, Any]:
    """Largest prompt this stack answers correctly before it breaks or thrashes.

    Deliberately asks a question about the prompt rather than just accepting it:
    a server will happily ingest 128K and answer from nothing, and "it did not
    crash" is not the same as "the context is usable". The ladder stops at the
    first failure and records why it stopped, separating a refusal or error from
    memory pressure from a wrong answer.
    """
    client = EvalClient(api_base=base_url, model=model_id, api_key=api_key,
                        timeout=timeout, enable_thinking=False)
    rungs: List[Dict[str, Any]] = []
    best = 0
    for size in ladder:
        if max_context and size > max_context:
            rungs.append({"requested_tokens": size, "outcome": "above_configured_ctx_cap"})
            break
        marker = 100000 + size % 899999
        filler = _prompt_of(max(256, size - 256))
        prompt = (
            f"Remember this number: {marker}. It appears once, right here.\n\n"
            f"{filler}\n\n"
            "What was the number you were asked to remember? Reply with the digits only."
        )
        # Check BEFORE climbing, not only after. On a 64 GB machine a 29 GB
        # checkpoint plus a 262K prompt can push the system into swap, and by the
        # time the request returns the damage (and the wall-clock cost) is already
        # done. Refusing the rung leaves the previous, verified figure standing.
        available_before = int(memutil.virtual_memory().available)
        if available_before < min_available_bytes:
            rungs.append({
                "requested_tokens": size,
                "outcome": "not_attempted_memory_pressure",
                "available_before_bytes": available_before,
            })
            break
        started = time.perf_counter()
        response = client.complete(Case(case_id=f"ctx{size}", prompt=prompt,
                                        max_tokens=48, temperature=0.0))
        elapsed = round(time.perf_counter() - started, 1)
        available_after = int(memutil.virtual_memory().available)
        rung = {
            "requested_tokens": size,
            "served_prompt_tokens": response.prompt_tokens or None,
            "wall_s": elapsed,
            "available_after_bytes": available_after,
        }
        if not response.ok:
            rung["outcome"] = "serving_failure"
            rung["error"] = response.error[:200]
            rungs.append(rung)
            break
        if str(marker) in "".join(ch if ch.isdigit() else " " for ch in response.text):
            rung["outcome"] = "ok"
            best = size
        else:
            rung["outcome"] = "wrong_answer"
            rung["answer"] = response.text[:120]
        rungs.append(rung)
        if rung["outcome"] != "ok":
            break
        if available_after < min_available_bytes:
            rung["outcome"] = "ok_but_memory_pressure"
            rungs[-1] = rung
            break
    return {
        "practical_max_context": best,
        "stopped_because": rungs[-1]["outcome"] if rungs else "not_probed",
        "rungs": rungs,
    }
