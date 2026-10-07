"""Exercise streaming, concurrency, cache labeling, and persisted summaries."""
import json
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import threading
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bench_profile_api as runner
import benchmark_profiles as profiles


class FixtureEncoding:
    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, tokens):
        return bytes(tokens).decode("utf-8")


@pytest.fixture
def server():
    counts = {"active": 0, "peak": 0, "requests": 0, "echo": False, "payloads": []}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                counts["payloads"].append(payload)
                counts["active"] += 1
                counts["requests"] += 1
                counts["peak"] = max(counts["peak"], counts["active"])
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.end_headers()
            try:
                content = "thinking" if counts["echo"] else "café " * 20
                for delta in ({"reasoning_content": "thinking"}, {"content": content}):
                    time.sleep(0.02)
                    chunk = {"choices": [{"delta": delta}]}
                    self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
                    self.wfile.flush()
                self.wfile.write(b'data: {"usage": {"completion_tokens": 100, "prompt_tokens": 1100}}\n\n')
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            finally:
                with lock:
                    counts["active"] -= 1

        def log_message(self, *_args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}/v1", counts
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def test_streaming_separates_reasoning_from_answer_and_preserves_unicode(server):
    url, _ = server
    result = runner.measure_request(
        url, "fixture", "prompt", 100, FixtureEncoding(), headers={}, temperature=0, timeout=10,
    )

    assert result["status"] == "ok"
    assert 0 < result["time_to_first_token_s"] < result["time_to_first_answer_s"] <= result["end_to_end_s"]
    assert result["output_tokens"] == len(("café " * 20).encode())
    assert result["visible_reasoning_tokens"] == len("thinking")
    assert result["prompt_tokens_per_second"] is None
    assert result["server_usage"]["completion_tokens"] == 100


def test_batch_sends_requests_concurrently_without_inventing_memory(server):
    url, counts = server
    rows = runner.run_batch(
        ["one", "two", "three"], url=url, model="fixture", max_tokens=100,
        encoding=FixtureEncoding(), headers={}, temperature=0, timeout=10,
    )

    assert counts["peak"] >= 2
    assert len(rows) == 3
    assert all(row["batch_tokens_per_second"] > 0 for row in rows)
    assert all(row["peak_memory_bytes"] is None for row in rows)


def test_reasoning_replayed_as_content_is_not_counted_twice_or_as_an_answer(server):
    url, counts = server
    counts["echo"] = True
    result = runner.measure_request(url, "fixture", "prompt", 100, FixtureEncoding(),
                                    headers={}, temperature=0, timeout=10)

    assert result["reasoning_echo_detected"] is True
    assert result["status"] == "reasoning_echo_no_answer"
    assert result["output_tokens"] == 0
    assert result["visible_reasoning_tokens"] == len("thinking")
    assert result["time_to_first_answer_s"] is None


def test_full_profile_requires_owned_server_for_cold_batches(tmp_path):
    profile = profiles.get_profile(profiles.load_profiles(), "aa_local_full")
    with pytest.raises(ValueError, match="Cold measurements require"):
        runner.run_profile(profile, url="http://127.0.0.1:8000/v1", model="fixture",
                           encoding=FixtureEncoding(), output_dir=tmp_path / "results")
    assert not (tmp_path / "results").exists()


def test_explicit_output_cap_is_sent_and_recorded(server, tmp_path):
    url, counts = server
    profile = profiles.get_profile(profiles.load_profiles(), "aa_local_smoke")
    profile["workloads"] = [{"id": "fixture", "input_tokens": 1024,
                             "min_output_tokens": 12, "max_output_tokens": 150}]
    profile["aggregation"]["min_samples"] = 1
    output = tmp_path / "capped"
    result = runner.run_profile(profile, url=url, model="fixture", encoding=FixtureEncoding(),
                                output_dir=output, timeout=10)
    assert result["profile_conformant"], result["validation_errors"]
    assert [p["max_tokens"] for p in counts["payloads"]] == [1, 150]
    row = profiles.load_result_records(output / "requests.jsonl")[0]
    assert row["max_output_tokens"] == 150
    assert json.loads((output / "manifest.json").read_text())["max_output_tokens_by_workload"] == {"fixture": 150}


def test_owned_server_refuses_to_replace_an_existing_service(server, tmp_path):
    url, _ = server
    with pytest.raises(ValueError, match="already occupies"):
        with runner.owned_server([sys.executable, "-c", "raise Exception('must not start')"],
                                 url, {}, tmp_path / "server.log"):
            pytest.fail("must not take ownership of an existing service")


def test_owned_server_waits_for_model_completion_and_stops_only_its_process(monkeypatch, tmp_path):
    requests_seen = []
    signals = []
    statuses = iter([503, 200])

    class Process:
        pid = 987654

        def poll(self):
            return None

        def wait(self, timeout):
            return 0

    @contextmanager
    def request(method, url, **kwargs):
        requests_seen.append((method, url, kwargs["json"]))

        class Response:
            status_code = next(statuses)

            def json(self):
                return {"choices": [{"message": {"content": "OK"}}]}

        yield Response()

    def unused_port(*_args, **_kwargs):
        raise ConnectionRefusedError

    monkeypatch.setattr(runner.socket, "create_connection", unused_port)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *_args, **_kwargs: Process())
    monkeypatch.setattr(runner, "local_request", request)
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(runner.os, "killpg", lambda pid, sig: signals.append((pid, sig)))

    with runner.owned_server(["fixture"], "http://127.0.0.1:9000/v1", {},
                             tmp_path / "server.log", model="chosen-model") as pid:
        assert pid == 987654

    assert len(requests_seen) == 2
    assert all(method == "POST" and body["model"] == "chosen-model" for method, _url, body in requests_seen)
    assert signals == [(987654, runner.signal.SIGTERM)]


def test_profile_primes_warm_requests_and_writes_valid_summaries(server, tmp_path):
    url, counts = server
    profile = profiles.get_profile(profiles.load_profiles(), "aa_local_smoke")
    profile["workloads"] = [{"id": "fixture", "input_tokens": 1024, "min_output_tokens": 12}]
    profile["modes"] = [{"id": "parallel", "concurrency": 2}]
    output = tmp_path / "run"

    result = runner.run_profile(profile, url=url, model="fixture", encoding=FixtureEncoding(),
                                output_dir=output, timeout=10)

    rows = profiles.load_result_records(output / "requests.jsonl")
    assert result["profile_conformant"] is True
    assert len(rows) == 4
    assert counts["requests"] == 8  # Each measured request was primed first.
    assert result["groups"][0]["metrics"]["time_to_first_answer_s"]["p95"] > 0
    metrics = result["groups"][0]["metrics"]
    assert metrics["output_tokens_per_second"]["observations"] == 4
    assert metrics["output_tokens_per_second"]["independent_batches"] == 2
    assert metrics["batch_tokens_per_second"]["observations"] == 2
    assert json.loads((output / "manifest.json").read_text())["leaderboard_comparable"] is False


def test_tokenizer_download_requires_explicit_preparation(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "TOKENIZER_CACHE", tmp_path)
    monkeypatch.setattr(runner.tiktoken, "get_encoding", lambda *_args: pytest.fail("must stay offline"))
    with pytest.raises(ValueError, match="Prepare the tokenizer"):
        runner.load_encoding()


def test_sample_count_is_batches_not_concurrent_request_count():
    from evals.test_benchmark_profiles import _profile_result_rows

    profile = profiles.get_profile(profiles.load_profiles(), "aa_local_smoke")
    rows = _profile_result_rows(profile)
    for mode in profile["modes"]:
        mode["concurrency"] = 2
    for row in rows:
        row["concurrency"] = 2
    errors = profiles.validate_results(profile, rows)

    assert any("needs 2 successful samples; found 0" in error for error in errors)


def test_cli_dispatches_executable_profile(monkeypatch):
    import llm
    calls = []
    monkeypatch.setattr(runner, "main", lambda argv: calls.append(argv))
    llm.main(["bench", "--profile", "aa_local_smoke", "--model", "fixture", "--dry-run"])

    assert "--profile" in calls[0]
    assert "fixture" in calls[0]
    assert "--dry-run" in calls[0]
