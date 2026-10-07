#!/usr/bin/env python3
"""Run versioned local service workloads with streaming measurements.

Warm runs may use an existing loopback service. Cold runs require an explicit
server command; only that owned process is restarted, once per measured batch.
Token counts use o200k_base consistently across backends. Preparing its public
vocabulary is an explicit, separate operation; measurement never downloads it.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shlex
import signal
import socket
import subprocess
import threading
import time
from urllib.parse import urlparse
import uuid

import psutil
import requests
import tiktoken

import benchmark_profiles as profiles
import local_config
import runtime_versions

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_URL = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
TOKENIZER_SHA256 = profiles.REFERENCE_VOCAB_SHA256
TOKENIZER_CACHE = ROOT / ".cache" / "tiktoken"


@contextmanager
def local_request(method: str, url: str, **kwargs):
    with requests.Session() as session:
        session.trust_env = False
        with session.request(method, url, **kwargs) as response:
            yield response


def load_encoding(*, prepare: bool = False):
    cached = TOKENIZER_CACHE / hashlib.sha1(TOKENIZER_URL.encode()).hexdigest()
    if not prepare:
        if not cached.is_file() or hashlib.sha256(cached.read_bytes()).hexdigest() != TOKENIZER_SHA256:
            raise ValueError("Prepare the tokenizer once with: local-ai bench --prepare-tokenizer")
    os.environ["TIKTOKEN_CACHE_DIR"] = str(TOKENIZER_CACHE)
    return tiktoken.get_encoding("o200k_base")


def build_prompt(encoding, target: int, minimum_output: int, *, seed: int, nonce: str) -> str:
    """Build deterministic synthetic technical prose at the reference token size."""
    rng = random.Random(seed)
    prefix = f"Independent workload {nonce}.\nReference notes:\n"
    suffix = (
        "\nUsing the reference notes, write a detailed implementation guide with examples, "
        "tradeoffs, failure cases, and a testing checklist. Explain each design decision. "
        f"Write at least {minimum_output + 250} tokens. Continue until every section is complete."
    )
    available = target - len(encoding.encode(prefix)) - len(encoding.encode(suffix))
    if available < 1:
        raise ValueError("Input token target is too small for the workload instructions")
    topics = ("storage", "routing", "queues", "scheduling", "caching", "recovery", "validation")
    paragraphs = []
    # Vary facts and subjects to avoid benchmarking a single repeated sentence.
    while sum(len(item) for item in paragraphs) < available * 7:
        topic = rng.choice(topics)
        paragraphs.append(
            f"The {topic} subsystem handles {rng.randrange(20, 900)} records per interval. "
            f"Workers retry after {rng.randrange(1, 30)} seconds and retain {rng.randrange(2, 15)} snapshots. "
            "Clients verify sequence numbers before acknowledging writes. Failed requests enter "
            "a bounded recovery queue, while healthy requests continue independently. "
            "Operators inspect latency distributions, resource pressure, and replay consistency.\n"
        )
    body = encoding.decode(encoding.encode("".join(paragraphs))[:available])
    return prefix + body + suffix


def measure_request(url: str, model: str, prompt: str, max_tokens: int, encoding,
                    *, headers: dict, temperature: float, timeout: float,
                    no_thinking: bool = False) -> dict:
    started = time.perf_counter()
    first = answer = None
    content: list[str] = []
    reasoning: list[str] = []
    usage: dict = {}
    timings: dict = {}
    finish_reason = None
    status = "ok"
    last_token_at = None
    last_reasoning_at = None
    try:
        with local_request(
            "POST", f"{url.rstrip('/')}/chat/completions", headers=headers,
            json={"model": model, "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": max_tokens, "temperature": temperature, "top_p": 1,
                  "stream": True, "stream_options": {"include_usage": True},
                  **({"chat_template_kwargs": {"enable_thinking": False}} if no_thinking else {})},
            stream=True, timeout=(10, timeout), allow_redirects=False,
        ) as response:
            response.encoding = "utf-8"
            response.raise_for_status()
            if response.status_code != 200:
                raise ValueError(f"Unexpected HTTP {response.status_code}")
            for line in response.iter_lines(chunk_size=1, decode_unicode=True):
                if time.perf_counter() - started > timeout:
                    raise TimeoutError("Overall request deadline exceeded")
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("error"):
                    raise ValueError("Server reported a streaming error")
                usage.update(chunk.get("usage") or {})
                timings.update(chunk.get("timings") or {})
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    text = delta.get("content") or ""
                    thought = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    elapsed = time.perf_counter() - started
                    if text or thought:
                        first = elapsed if first is None else first
                        last_token_at = elapsed
                    if text:
                        answer = elapsed if answer is None else answer
                        content.append(text)
                    if thought:
                        reasoning.append(thought)
                        last_reasoning_at = elapsed
                    finish_reason = choice.get("finish_reason") or finish_reason
        if first is None:
            status = "no_tokens"
    except (requests.RequestException, ValueError, TimeoutError) as exc:
        status = f"error:{type(exc).__name__}"
    ended = time.perf_counter() - started
    answer_text = "".join(content)
    reasoning_text = "".join(reasoning)
    # oMLX's malformed-thinking recovery can replay all reasoning as a final
    # content delta. It is one generation, not an answer plus another generation.
    echoed_reasoning = bool(reasoning_text.strip()) and answer_text.strip().startswith(reasoning_text.strip()) and len(answer_text.strip()) - len(reasoning_text.strip()) <= 32
    if echoed_reasoning:
        answer_text = ""
        answer = None
        last_token_at = last_reasoning_at
        if status == "ok":
            status = "reasoning_echo_no_answer"
    answer_tokens = len(encoding.encode(answer_text))
    reasoning_tokens = len(encoding.encode(reasoning_text))
    visible_tokens = answer_tokens + reasoning_tokens
    duration = (last_token_at - first) if last_token_at is not None and first is not None else 0
    speed = (visible_tokens - 1) / duration if duration > 0 and visible_tokens > 1 else None
    prompt_speed = timings.get("prompt_per_second", usage.get("prompt_tokens_per_second"))
    unsupported = ["peak_memory_bytes"]
    if prompt_speed is None:
        unsupported.append("prompt_tokens_per_second")
    if speed is None:
        unsupported.append("output_tokens_per_second")
    if first is None:
        unsupported.append("time_to_first_token_s")
    if answer is None:
        unsupported.append("time_to_first_answer_s")
    return {
        "status": status, "token_count_method": "tiktoken_o200k_base",
        "reasoning_echo_detected": echoed_reasoning,
        "prompt_tokens": len(encoding.encode(prompt)), "output_tokens": answer_tokens,
        "visible_reasoning_tokens": reasoning_tokens, "server_usage": usage,
        "finish_reason": finish_reason, "time_to_first_token_s": first,
        "time_to_first_answer_s": answer, "prompt_tokens_per_second": prompt_speed,
        "prompt_speed_token_basis": "server_native" if prompt_speed is not None else None,
        "output_tokens_per_second": speed, "end_to_end_s": ended,
        "peak_memory_bytes": None, "unsupported_metrics": unsupported,
    }


class MemorySampler:
    """Sample owned server process-tree RSS; Metal allocator memory is separate."""
    def __init__(self, pid: int | None):
        self.pid = pid
        self.peak: int | None = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self.stop_event.is_set():
            try:
                parent = psutil.Process(self.pid)
                processes = [parent, *parent.children(recursive=True)]
                rss = sum(process.memory_info().rss for process in processes)
                self.peak = max(self.peak or 0, rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            self.stop_event.wait(0.05)

    def __enter__(self):
        if self.pid is not None:
            self.thread.start()
        return self

    def __exit__(self, *_exc):
        self.stop_event.set()
        if self.pid is not None:
            self.thread.join(timeout=2)


def run_batch(prompts: list[str], *, pid: int | None = None, **request_args) -> list[dict]:
    barrier = threading.Barrier(len(prompts))

    def run_one(prompt):
        barrier.wait(timeout=30)
        return measure_request(prompt=prompt, **request_args)

    started = time.perf_counter()
    with MemorySampler(pid) as sampler, ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        rows = list(pool.map(run_one, prompts))
    elapsed = time.perf_counter() - started
    batch_tps = sum(row["output_tokens"] + row["visible_reasoning_tokens"] for row in rows) / elapsed
    for row in rows:
        row.update({"batch_elapsed_s": elapsed, "batch_tokens_per_second": batch_tps,
                    "peak_memory_bytes": sampler.peak,
                    "memory_measurement": "server_process_tree_rss" if pid else "unsupported",
                    "memory_scope": "concurrent_batch"})
        if sampler.peak is not None:
            row["unsupported_metrics"].remove("peak_memory_bytes")
    return rows


@contextmanager
def owned_server(command: list[str] | None, url: str, headers: dict, log_path: Path,
                 *, model: str = "default_model", startup_timeout: float = 1800):
    """Never stop or replace a service we did not start."""
    if command is None:
        yield None
        return
    endpoint = urlparse(url)
    try:
        with socket.create_connection((endpoint.hostname, endpoint.port or 80), timeout=1):
            raise ValueError(f"A service already occupies {url}; choose a free port")
    except OSError:
        pass
    with log_path.open("a") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=log, start_new_session=True,
            cwd=ROOT,
            env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        )
        try:
            deadline = time.monotonic() + startup_timeout
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Benchmark server exited; inspect {log_path}")
                try:
                    with local_request(
                        "POST", f"{url.rstrip('/')}/chat/completions", headers=headers,
                        json={"model": model, "messages": [{"role": "user", "content": "Reply OK."}],
                              "max_tokens": 1, "stream": False},
                        timeout=(2, 30), allow_redirects=False,
                    ) as response:
                        if response.status_code == 200 and response.json().get("choices"):
                            break
                except requests.RequestException:
                    pass
                time.sleep(0.5)
            else:
                raise TimeoutError(f"Benchmark server startup timed out; inspect {log_path}")
            yield process.pid
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)


def summarize(records: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in records:
        key = (row["workload_id"], row["mode_id"], row["cache_state"])
        grouped.setdefault(key, []).append(row)
    summaries = []
    for (workload, mode, cache), rows in grouped.items():
        metrics = {}
        batches: dict[int, list[dict]] = {}
        for row in rows:
            batches.setdefault(row["sample_id"], []).append(row)
        for metric in profiles.REQUIRED_METRICS - {"cache_state", "batch_tokens_per_second"}:
            values = [row[metric] for row in rows if row["status"] == "ok" and row.get(metric) is not None]
            metrics[metric] = {
                "median": profiles._quantile(values, 0.5),
                "p95": profiles._quantile(values, 0.95), "observations": len(values),
                "independent_batches": sum(
                    any(row["status"] == "ok" and row.get(metric) is not None for row in batch)
                    for batch in batches.values()
                ),
            } if values else None
        speeds = [batch[0]["batch_tokens_per_second"] for batch in batches.values()
                  if all(row["status"] == "ok" for row in batch)]
        metrics["batch_tokens_per_second"] = {
            "median": profiles._quantile(speeds, 0.5), "p95": profiles._quantile(speeds, 0.95),
            "observations": len(speeds), "independent_batches": len(speeds),
        } if speeds else None
        summaries.append({"workload_id": workload, "mode_id": mode, "cache_state": cache,
                          "requests": len(rows), "successful": sum(row["status"] == "ok" for row in rows),
                          "metrics": metrics})
    return summaries


def runtime_provenance() -> dict:
    installed = {}
    for spec in runtime_versions.specs():
        try:
            installed[spec.name] = spec.installed_version()
        except Exception:
            installed[spec.name] = None
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, timeout=5, check=False).stdout.strip()
    return {
        "git_revision": revision or None,
        "source_sha256": hashlib.sha256(
            Path(__file__).read_bytes() + Path(profiles.__file__).read_bytes()
        ).hexdigest(),
        "runtimes": installed, "tiktoken_version": importlib.metadata.version("tiktoken"),
    }


def run_profile(profile: dict, *, url: str, model: str, encoding, output_dir: Path,
                command: list[str] | None = None, headers: dict | None = None,
                temperature: float = 0.6, timeout: float = 1800, seed: int = 0,
                no_thinking: bool = False) -> dict:
    endpoint = urlparse(url)
    if endpoint.scheme != "http" or endpoint.hostname not in {"localhost", "127.0.0.1", "::1"} or endpoint.username:
        raise ValueError("Profile benchmarks require an HTTP loopback service")
    if "cold" in profile["cache_states"] and not command:
        raise ValueError("Cold measurements require --server-command so each batch starts a fresh service")
    output_dir.mkdir(parents=True, exist_ok=False)
    headers = headers or {}
    records: list[dict] = []
    run_id = uuid.uuid4().hex
    metadata = {"profile": profile, "model": model, "url": url, "seed": seed,
                "run_id": run_id, "leaderboard_comparable": False,
                "temperature": temperature, "top_p": 1, "request_timeout_s": timeout,
                "thinking": "disabled_requested" if no_thinking else "server_default",
                "max_output_tokens_by_workload": {
                    item["id"]: item.get("max_output_tokens", item["min_output_tokens"] * 3 + 256)
                    for item in profile["workloads"]
                },
                "software": runtime_provenance(),
                "cache_definition": "cold: fresh server after a separate readiness completion, unique workload prompt; warm: identical workload prompt primed",
                "cache_limitations": "OS file cache and persistent backend caches are not purged"}
    (output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for workload in profile["workloads"]:
        for mode in profile["modes"]:
            for cache in profile["cache_states"]:
                for sample in range(profile["aggregation"]["min_samples"]):
                    batch_id = f"{workload['id']}/{mode['id']}/{cache}/{sample}"
                    prompts = [build_prompt(
                        encoding, workload["input_tokens"], workload["min_output_tokens"],
                        seed=seed + sample * mode["concurrency"] + index,
                        nonce=f"{run_id}/{batch_id}/{index}",
                    ) for index in range(mode["concurrency"])]
                    request_args = dict(url=url, model=model, encoding=encoding, headers=headers,
                                        temperature=temperature, timeout=timeout, no_thinking=no_thinking)
                    with owned_server(command, url, headers, output_dir / "server.log", model=model) as pid:
                        if cache == "warm":
                            priming = run_batch(prompts, pid=pid, max_tokens=1, **request_args)
                            if any(row["status"] not in {"ok", "reasoning_echo_no_answer"} for row in priming):
                                raise RuntimeError("Warm-up failed; no warm measurements were labeled")
                        rows = run_batch(prompts, pid=pid,
                                         max_tokens=metadata["max_output_tokens_by_workload"][workload["id"]],
                                         **request_args)
                    for index, row in enumerate(rows):
                        row.update({"schema_version": 1, "run_id": run_id, "model": model,
                                    "url": url, "profile_id": profile["id"],
                                    "tokenizer_sha256": TOKENIZER_SHA256,
                                    "temperature": temperature, "top_p": 1,
                                    "max_output_tokens": metadata["max_output_tokens_by_workload"][workload["id"]],
                                    "request_timeout_s": timeout,
                                    "thinking": metadata["thinking"],
                                    "workload_id": workload["id"], "mode_id": mode["id"],
                                    "concurrency": mode["concurrency"], "cache_state": cache,
                                    "sample_id": sample, "batch_id": batch_id, "request_index": index,
                                    "target_input_tokens": workload["input_tokens"],
                                    "target_min_output_tokens": workload["min_output_tokens"]})
                        if row["status"] == "ok" and row["output_tokens"] < workload["min_output_tokens"]:
                            row["status"] = "underfilled_output"
                    with (output_dir / "requests.jsonl").open("a") as handle:
                        for row in rows:
                            handle.write(json.dumps(row) + "\n")
                    records.extend(rows)
                    print(f"{batch_id}: {sum(row['status'] == 'ok' for row in rows)}/{len(rows)} successful", flush=True)
    errors = profiles.validate_results(profile, records)
    result = {"profile_conformant": not errors, "validation_errors": errors,
              "leaderboard_comparable": False, "groups": summarize(records)}
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="aa_local_smoke", choices=["aa_local_smoke", "aa_local_full", "local_practical"])
    parser.add_argument("--url", help="Loopback OpenAI-compatible base URL "
                        "(default: the MLX endpoint in the local config)")
    parser.add_argument("--model")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--server-command", help="Quoted foreground server command; executed directly without a shell")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-thinking", action="store_true", help="Request enable_thinking=false and record it in provenance")
    parser.add_argument("--prepare-tokenizer", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config", type=Path, default=local_config.DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    if args.prepare_tokenizer:
        load_encoding(prepare=True)
        print("o200k_base tokenizer prepared. Benchmark execution is now offline.")
        return
    profile = profiles.get_profile(profiles.load_profiles(), args.profile)
    if args.dry_run:
        print(json.dumps(profile, indent=2))
        return
    if not args.model:
        parser.error("--model is required")
    url = args.url or local_config.endpoint(local_config.load_config(args.config), "mlx")
    endpoint = urlparse(url)
    if endpoint.scheme != "http" or endpoint.hostname not in {"localhost", "127.0.0.1", "::1"} or endpoint.username:
        parser.error("--url must be an HTTP loopback service without embedded credentials")
    output_dir = args.output_dir or ROOT / "results" / f"profile_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    result = run_profile(
        profile, url=url, model=args.model, encoding=load_encoding(), output_dir=output_dir,
        command=shlex.split(args.server_command) if args.server_command else None,
        temperature=args.temperature, timeout=args.timeout, seed=args.seed,
        no_thinking=args.no_thinking,
    )
    print(f"Results: {output_dir}; profile conformant: {result['profile_conformant']}")
    if not result["profile_conformant"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
