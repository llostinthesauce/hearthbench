#!/usr/bin/env python3
"""
A bounded tool-calling loop.

Every limit here exists because an unbounded local agent on a 27B model is how
you lose an afternoon: a quant that has started looping will happily emit the
same `read_file` call four hundred times, each one costing a full prefill. The
runner therefore stops on whichever of these arrives first —

    max_steps        assistant turns
    max_wall_s       wall clock for the whole trajectory
    max_output_tokens cumulative completion tokens across the trajectory
    repeat_limit     identical (tool, arguments) pairs in a row

— and records *which* limit fired, because "ran out of steps" and "looped" and
"answered" are three different outcomes that a single `success: false` would
flatten into one.

The transcript is written as JSONL so an ambiguous trajectory can be read later
without the scoring pass having to keep it in memory, and so nothing needs to
pass a model transcript through an operator's context to be useful.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..core import Case, EvalClient, Response
from .workspace import Workspace, dispatch_table, schemas_for


@dataclass
class Limits:
    max_steps: int = 14
    max_wall_s: float = 900.0
    max_output_tokens: int = 24000
    repeat_limit: int = 3
    per_request_max_tokens: int = 1024
    max_tool_calls: int = 64


@dataclass
class Trajectory:
    """Everything the scorer and a later human reader need."""

    task_id: str
    trial: int
    stop_reason: str = ""
    steps: int = 0
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    final_text: str = ""
    reasoning_content: str = ""
    finished_summary: Optional[str] = None
    findings: Dict[str, Any] = field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_s: float = 0.0
    error: str = ""
    invalid_arguments: int = 0
    unknown_tools: List[str] = field(default_factory=list)
    repeated_calls: int = 0
    transcript_path: str = ""

    @property
    def ok(self) -> bool:
        """The loop itself ran. Says nothing about whether the task succeeded.

        Keeping these apart is the point: a server that died mid-trajectory must
        never be recorded as a model that failed the task.
        """
        return not self.error


def _signature(call: Dict[str, Any]) -> str:
    return json.dumps({"n": call.get("name"), "a": call.get("arguments")}, sort_keys=True, default=str)


def run_trajectory(
    client: EvalClient,
    workspace: Workspace,
    *,
    task_id: str,
    trial: int,
    system: str,
    instruction: str,
    tools: Tuple[str, ...],
    limits: Limits,
    transcript_dir: Optional[Path] = None,
    temperature: float = 0.0,
) -> Trajectory:
    """Drive one task to a terminal state and return what happened."""
    workspace.reset()
    dispatch = dispatch_table(workspace)
    schemas = tuple(schemas_for(tools))
    allowed = set(tools)

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": instruction},
    ]
    trajectory = Trajectory(task_id=task_id, trial=trial)
    transcript: List[Dict[str, Any]] = [{"role": "system", "content": system},
                                        {"role": "user", "content": instruction}]

    started = time.monotonic()
    last_signature = ""
    consecutive = 0

    while True:
        if len(trajectory.tool_calls) >= limits.max_tool_calls:
            trajectory.stop_reason = "max_tool_calls"
            break
        if trajectory.steps >= limits.max_steps:
            trajectory.stop_reason = "max_steps"
            break
        elapsed = time.monotonic() - started
        if elapsed >= limits.max_wall_s:
            trajectory.stop_reason = "max_wall_clock"
            break
        if trajectory.completion_tokens >= limits.max_output_tokens:
            trajectory.stop_reason = "max_output_tokens"
            break

        case = Case(
            case_id=f"{task_id}#{trajectory.steps}",
            prompt="",
            max_tokens=min(limits.per_request_max_tokens,
                           limits.max_output_tokens - trajectory.completion_tokens),
            temperature=temperature,
            tools=schemas,
            messages=tuple(messages),
        )
        # Respect the remaining trajectory wall budget for real HTTP clients.
        old_timeout = getattr(client, "timeout", None)
        try:
            if old_timeout is not None:
                client.timeout = min(old_timeout, limits.max_wall_s - elapsed)
            response: Response = client.complete(case)
        finally:
            if old_timeout is not None:
                client.timeout = old_timeout
        trajectory.steps += 1
        trajectory.prompt_tokens += response.prompt_tokens
        trajectory.completion_tokens += response.completion_tokens
        trajectory.invalid_arguments += response.arguments_unparseable
        trajectory.reasoning_content += response.reasoning_content

        # A request capped by this trajectory's deadline is a task budget
        # failure, not an independent server outage to exclude from scoring.
        if time.monotonic() - started >= limits.max_wall_s:
            trajectory.stop_reason = "max_wall_clock"
            break

        if not response.ok:
            trajectory.error = response.error[:300]
            trajectory.stop_reason = "server_error"
            break

        transcript.append({
            "role": "assistant", "content": response.text,
            "reasoning_content": response.reasoning_content,
            "tool_calls": response.tool_calls,
            "finish_reason": response.finish_reason,
            "completion_tokens": response.completion_tokens,
        })

        if not response.tool_calls:
            trajectory.final_text = response.text
            trajectory.stop_reason = "answered" if response.finish_reason != "length" else "truncated"
            messages.append({"role": "assistant", "content": response.text})
            break

        # An assistant turn carrying tool calls must be replayed verbatim, or the
        # next turn's tool results have no call to attach to and strict servers
        # reject the request outright.
        messages.append({
            "role": "assistant",
            "content": response.text or None,
            "tool_calls": [{
                "id": c["id"] or f"call_{trajectory.steps}_{i}",
                "type": "function",
                "function": {"name": c["name"], "arguments": c["raw_arguments"]},
            } for i, c in enumerate(response.tool_calls)],
        })

        if response.reasoning_content:
            messages[-1]["reasoning_content"] = response.reasoning_content
        terminal = False
        for index, call in enumerate(response.tool_calls):
            if len(trajectory.tool_calls) >= limits.max_tool_calls:
                trajectory.stop_reason = "max_tool_calls"
                terminal = True
                break
            name = call["name"]
            call_id = call["id"] or f"call_{trajectory.steps}_{index}"
            arguments = call["arguments"]

            if name not in allowed:
                trajectory.unknown_tools.append(name)
                result: Dict[str, Any] = {
                    "error": f"unknown tool '{name}'",
                    "available": sorted(allowed),
                }
            elif not isinstance(arguments, dict):
                result = {"error": "arguments were not valid JSON; re-send this call with valid JSON"}
            else:
                try:
                    result = dispatch[name](**arguments)
                except TypeError as exc:
                    result = {"error": f"bad arguments for {name}: {exc}"}
                except Exception as exc:  # noqa: BLE001 — a tool fault is data, not a crash
                    result = {"error": f"{type(exc).__name__}: {exc}"}

            trajectory.tool_calls.append({
                "step": trajectory.steps, "name": name,
                "arguments": arguments, "result": result,
                "is_error": isinstance(result, dict) and "error" in result,
            })
            transcript.append({"role": "tool", "tool_call_id": call_id, "name": name,
                               "arguments": arguments, "content": result})
            messages.append({"role": "tool", "tool_call_id": call_id, "name": name,
                             "content": json.dumps(result)[:8000]})

            signature = _signature(call)
            if signature == last_signature:
                consecutive += 1
                trajectory.repeated_calls += 1
            else:
                consecutive = 0
                last_signature = signature

            if name == "finish" and name in allowed and not trajectory.tool_calls[-1]["is_error"]:
                trajectory.stop_reason = "finished"
                terminal = True
                break
            if consecutive + 1 >= limits.repeat_limit:
                trajectory.stop_reason = "repetition_loop"
                terminal = True
                break

        if terminal:
            break

    trajectory.wall_s = round(time.monotonic() - started, 2)
    trajectory.findings = dict(workspace.findings)
    trajectory.finished_summary = workspace.finished

    if transcript_dir is not None:
        transcript_dir.mkdir(parents=True, exist_ok=True)
        path = transcript_dir / f"{task_id}_trial{trial}.jsonl"
        with path.open("w") as handle:
            for entry in transcript:
                handle.write(json.dumps(entry, default=str) + "\n")
        trajectory.transcript_path = str(path)

    return trajectory
