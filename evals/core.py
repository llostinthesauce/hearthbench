#!/usr/bin/env python3
"""
Shared types and the OpenAI-compatible client used by every eval.

An eval is three things:
  1. `build_cases()` — turn a config into a deterministic list of Case objects
  2. the runner       — sends each Case to a server (this module)
  3. `score()`        — turn (Case, response text) into a Score

Nothing here talks to a specific backend. Any server that speaks
`POST /v1/chat/completions` works: llama-server, mlx_lm.server, mlx_vlm.server,
oMLX, Ollama, vLLM, or a hosted API.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Case:
    """One prompt plus everything needed to score the answer."""

    case_id: str
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    system: str = ""
    # Anything score() needs: the expected answer, constraint specs, depths...
    meta: Dict[str, Any] = field(default_factory=dict)
    # OpenAI-shaped tool schemas. Present only for tool-use evals; when empty the
    # request is byte-identical to what it was before tools existed, so adding
    # this field cannot perturb any existing eval's measurement.
    tools: Tuple[Dict[str, Any], ...] = ()
    tool_choice: str = ""
    # A prebuilt conversation, used by cases that must start mid-trajectory —
    # "here is a tool result, now interpret it" cannot be expressed as one user
    # turn. When set it replaces the system+prompt pair entirely.
    messages: Tuple[Dict[str, Any], ...] = ()


@dataclass(frozen=True)
class Score:
    """Result of scoring one Case. `value` is always normalized to 0.0-1.0."""

    value: float
    passed: bool
    detail: str = ""

    @staticmethod
    def binary(passed: bool, detail: str = "") -> "Score":
        return Score(value=1.0 if passed else 0.0, passed=passed, detail=detail)


@dataclass
class Response:
    """What the server actually returned, plus timing."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    ttft_s: float = 0.0
    error: str = ""
    finish_reason: str = ""
    reasoning_chars: int = 0
    # Parsed `message.tool_calls`, normalized to {"name", "arguments", "raw"}.
    # `arguments` is the decoded object when the model emitted valid JSON and
    # None when it did not — that distinction is the whole point of a tool-call
    # eval, so it must survive into scoring rather than being coerced to {}.
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    arguments_unparseable: int = 0
    reasoning_content: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def truncated_before_answer(self) -> bool:
        """True when the model hit max_tokens without emitting any answer.

        Reasoning models spend their budget in `reasoning_content` first. Ask a
        thinking model for a one-letter answer with max_tokens=16 and it will
        return empty `content` with finish_reason="length" — it was working
        correctly and simply never got to speak.

        Scoring that as a wrong answer would report 0% for a healthy model, so
        the runner counts it as a harness error instead and the summary surfaces
        it. The fix is a larger token budget, not a different model.
        """
        return (
            self.ok
            and not self.text.strip()
            and not self.tool_calls
            and self.finish_reason == "length"
        )


@dataclass
class Eval:
    """A named, scoreable benchmark.

    tier 1 evals generate their own data and run with no network.
    tier 2 evals need a dataset download (see evals/datasets/fetch.py).
    """

    name: str
    tier: int
    description: str
    build_cases: Callable[..., List[Case]]
    score: Callable[[Case, str], Score]
    # Higher-is-better metric name reported in the summary.
    metric: str = "accuracy"
    needs_code_execution: bool = False
    # Which named set this eval belongs to. "core" is the historical tier-1
    # suite; adding a new eval to it would silently lengthen and change every
    # existing `--evals tier1` run, so behavioural additions declare their own
    # suite and are opted into explicitly.
    suite: str = "core"
    # Set when a case cannot be scored on its own — the determinism eval, for
    # example, only means anything when repeats are compared against each other.
    # When present the runner calls this instead of per-case `score`.
    score_all: Optional[Callable[[List[Case], List[str]], List[Score]]] = None
    # Tool-use and agent evals cannot be scored from the assistant's text: the
    # answer lives in `tool_calls`, or in a transcript the eval produced itself.
    # When set, the runner calls this instead of `score` and hands over the whole
    # Response. Same precedent as `score_all`.
    score_response: Optional[Callable[[Case, "Response"], Score]] = None
    # Evals that drive their own multi-turn loop rather than sending one Case.
    # The runner hands them the client and gets back (Score, Response) pairs.
    run_cases: Optional[Callable[..., List[Tuple[Score, "Response"]]]] = None


def _parse_tool_calls(message: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int]:
    """Normalize `message.tool_calls` and count arguments that would not decode.

    Every OpenAI-compatible server transports tool arguments as a JSON *string*,
    so a model that emits `{"path": "src/a.py",}` produces a well-formed HTTP
    response carrying an unparseable payload. Silently substituting {} there
    would turn the most interesting failure mode a quantized model has — valid
    shape, invalid content — into an invisible one.
    """
    raw_calls = message.get("tool_calls") or []
    calls: List[Dict[str, Any]] = []
    unparseable = 0
    for raw in raw_calls:
        function = (raw or {}).get("function") or {}
        raw_args = function.get("arguments")
        parsed: Optional[Any] = None
        if isinstance(raw_args, dict):
            parsed = raw_args
        elif isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args) if raw_args.strip() else {}
            except ValueError:
                unparseable += 1
        calls.append({
            "id": raw.get("id") or "",
            "name": function.get("name") or "",
            "arguments": parsed,
            "raw_arguments": raw_args if isinstance(raw_args, str) else json.dumps(raw_args or {}),
        })
    return calls, unparseable


class EvalClient:
    """Minimal OpenAI-compatible chat client.

    Standard library only, matching the rest of the repo's serving scripts, so
    the eval harness runs before any project virtualenv is active.
    """

    def __init__(
        self,
        api_base: str = "http://127.0.0.1:8080/v1",
        model: str = "default_model",
        api_key: str = "",
        timeout: int = 600,
        enable_thinking: Optional[bool] = False,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.enable_thinking = enable_thinking

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def complete(self, case: Case) -> Response:
        """Send one Case and return the assistant text.

        Reasoning models emit `reasoning_content` alongside `content`. Only
        `content` is scored: a model that reaches the right answer after a long
        think still answered correctly, and its scratchpad is not the answer.
        """
        if case.messages:
            messages: List[Dict[str, Any]] = [dict(m) for m in case.messages]
        else:
            messages = []
            if case.system:
                messages.append({"role": "system", "content": case.system})
            messages.append({"role": "user", "content": case.prompt})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": case.max_tokens,
            "temperature": case.temperature,
            "top_p": case.top_p,
            "stream": False,
        }
        if case.tools:
            payload["tools"] = [dict(t) for t in case.tools]
            if case.tool_choice:
                payload["tool_choice"] = case.tool_choice
        # Eval token budgets size the requested answer, not an unbounded hidden
        # scratchpad. Send both controls because mlx_lm reads the template kwargs
        # while mlx_vlm reads the top-level compatibility field; llama.cpp accepts
        # the template kwargs. None explicitly opts back into server-native policy.
        if self.enable_thinking is not None:
            payload["enable_thinking"] = self.enable_thinking
            payload["chat_template_kwargs"] = {
                "enable_thinking": self.enable_thinking,
            }
        body = json.dumps(payload).encode()
        request = urllib.request.Request(
            f"{self.api_base}/chat/completions", data=body, method="POST", headers=self._headers()
        )

        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as raw:
                data = json.loads(raw.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            return Response(text="", error=f"http_{exc.code}: {detail}")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return Response(text="", error=f"unreachable: {exc}")
        except ValueError as exc:
            return Response(text="", error=f"bad_json: {exc}")

        elapsed = time.perf_counter() - start
        choices = data.get("choices") or [{}]
        message = choices[0].get("message") or {}
        text = message.get("content") or ""
        # Reasoning models put their scratchpad here. It is never scored, but its
        # size explains a truncated answer, so it is recorded.
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        usage = data.get("usage") or {}
        calls, unparseable = _parse_tool_calls(message)
        return Response(
            text=text,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            latency_s=elapsed,
            finish_reason=str(choices[0].get("finish_reason") or ""),
            reasoning_chars=len(reasoning),
            reasoning_content=reasoning,
            tool_calls=calls,
            arguments_unparseable=unparseable,
        )

    def probe(self, wait_s: int = 0) -> Optional[str]:
        """Return an error string if the server cannot answer, else None.

        Deliberately sends a real one-token completion rather than hitting
        /v1/models. llama-server answers /v1/models with 200 *while the model is
        still loading* and only then returns 503 "Loading model" from
        /v1/chat/completions — so /v1/models reports ready before it is, and a
        benchmark that trusts it fires its first prompt into a server that has
        not finished allocating its KV cache. A completion is the only probe
        that means the same thing on llama.cpp, mlx_lm and mlx_vlm alike.

        With wait_s > 0, keeps retrying transient loading errors until the
        deadline instead of failing on the first 503.
        """
        deadline = time.monotonic() + max(0, wait_s)
        probe_case = Case(case_id="__probe__", prompt="Reply with: OK", max_tokens=4)
        last = ""
        while True:
            response = self.complete(probe_case)
            if response.ok or response.truncated_before_answer:
                return None
            last = response.error
            transient = "503" in last or "502" in last or "unreachable" in last
            if not transient or time.monotonic() >= deadline:
                return f"{self.api_base} is not serving completions ({last[:160]})"
            time.sleep(2)
