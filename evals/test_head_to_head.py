"""Head-to-head harness: truncation must not be reported as a successful answer.

bench_head_to_head.py falls back to printing a model's chain-of-thought when it
emits no content. That fallback is useful for reading transcripts, but on its
own it turns "the model never answered" into a row that looks like a normal
result — Qwen3.6-35B spent all 6000 tokens reasoning on the LRU-cache prompt and
was tabulated as 0.6s / 65.1 t/s, indistinguishable from a real answer.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from bench_head_to_head import _pairwise_export, _stop_server, _truncated_before_answer


def test_reasoning_exhaustion_is_flagged():
    # Whole budget went to the thinking channel; no content, stopped on length.
    assert _truncated_before_answer("", "step 1... step 2...", "length")


def test_real_answer_at_length_limit_is_not_truncation():
    # A long but genuine answer that hit the cap still answered.
    assert not _truncated_before_answer("def solve():", "thinking", "length")


def test_normal_stop_is_not_truncation():
    assert not _truncated_before_answer("42", "thinking", "stop")


def test_empty_response_without_thinking_is_not_truncation():
    # Nothing generated at all is a different failure (server/error path).
    assert not _truncated_before_answer("", "", "length")


def test_thinking_only_but_stopped_cleanly_is_not_truncation():
    # Model chose to stop; not a budget problem.
    assert not _truncated_before_answer("", "thinking", "stop")


def test_pairwise_export_blinds_models_and_populates_candidate_responses():
    records = _pairwise_export(
        ["model-a", "model-b"],
        ["code_algo"],
        {
            "model-a": {"code_algo": {"text": "answer-a"}},
            "model-b": {"code_algo": {"text": "answer-b"}},
        },
        seed=3,
    )

    assert len(records) == 2
    assert records[0]["response_by_candidate"]["A"] in {"answer-a", "answer-b"}
    assert records[0]["response_by_candidate"]["A"] != records[0]["response_by_candidate"]["B"]
    assert records[0]["winner"] is None
    assert "ContextWindow" in records[0]["prompt"]


def test_stop_server_removes_its_temporary_log(tmp_path, monkeypatch):
    log = tmp_path / "server.log"
    log.write_text("output")

    class Proc:
        _server_log = str(log)

        def terminate(self):
            pass

        def wait(self, timeout):
            assert timeout == 10

    monkeypatch.setattr("bench_head_to_head.time.sleep", lambda _seconds: None)
    _stop_server(Proc())

    assert not log.exists()
