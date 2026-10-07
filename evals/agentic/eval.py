#!/usr/bin/env python3
"""
The agent suite as a registry-shaped eval.

Unlike every other eval here, this one cannot be expressed as "send a prompt,
grade the text". Each case is a whole trajectory: the runner sends turns, executes
tools against a scratch workspace, and the verdict comes from the workspace state
and the tool-call trace. The `Eval.run_cases` hook exists for exactly this, so the
suite still appears in the same catalog, writes the same CSV columns, and joins
into the same aggregate as the single-prompt evals.

A trajectory that failed because the *server* stopped answering is reported as a
harness error, never as a task the model failed. That distinction is the reason
this file does not simply return a boolean.
"""
from __future__ import annotations

import shutil
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..core import Case, EvalClient, Response, Score
from .runner import run_trajectory
from .tasks import TASKS, AgentTask, by_id, summarize_checks
from .workspace import Workspace


def build_cases(
    seed: int = 20260918,
    repeats: int = 0,
    limit: int = 0,
    agent_tasks: Tuple[str, ...] = (),
    **_ignored: object,
) -> List[Case]:
    """One Case per (task x trial). Carries no prompt work of its own.

    `repeats` overrides each task's own trial count when given, so the same flag
    that raises reliability sampling everywhere else raises it here too.
    """
    presets = {"local_short": ("agent_local_config_repair", "agent_local_config_recovery",
                                "agent_doc_provenance")}
    ids = [task_id for selector in agent_tasks for task_id in presets.get(selector, (selector,))]
    selected = [by_id(t) for t in dict.fromkeys(ids)] if ids else list(TASKS)
    cases: List[Case] = []
    for task in selected:
        trials = max(1, int(repeats)) if repeats else task.trials
        for trial in range(trials):
            cases.append(Case(
                case_id=f"{task.task_id}#t{trial}",
                prompt=task.instruction,
                system=task.system,
                max_tokens=task.limits.per_request_max_tokens,
                temperature=0.0,
                meta={"category": "agent", "group": task.capability,
                      "task_id": task.task_id, "repeat": trial,
                      "max_steps": task.limits.max_steps},
            ))
    return cases[:limit] if limit else cases


def run_cases(
    cases: List[Case],
    client: EvalClient,
    *,
    workspace_root: Optional[Path] = None,
    transcript_dir: Optional[Path] = None,
    **_ignored: object,
) -> List[Tuple[Score, Response]]:
    """Execute each trajectory and return (Score, Response) in case order.

    The Response is synthetic: it carries the trajectory's token totals and
    latency so the CSV's timing columns mean the same thing they do elsewhere,
    and its `error` field is set only when the *server* failed.
    """
    owned_root = workspace_root is None
    root = Path(workspace_root) if workspace_root else Path(tempfile.mkdtemp(prefix="agentws_"))
    results: List[Tuple[Score, Response]] = []
    try:
        results = _run_all(cases, client, root, transcript_dir)
    finally:
        # Each trial cleans its own workspace, but the scratch root this function
        # created is its own to remove — otherwise a 19-arm matrix leaves 19
        # orphaned directories in the system temp folder.
        if owned_root:
            shutil.rmtree(root, ignore_errors=True)
    return results


def _run_all(cases: List[Case], client: EvalClient, root: Path,
             transcript_dir: Optional[Path]) -> List[Tuple[Score, Response]]:
    results: List[Tuple[Score, Response]] = []
    root.mkdir(parents=True, exist_ok=True)
    for case in cases:
        task: AgentTask = by_id(case.meta["task_id"])
        trial = int(case.meta["repeat"])
        scratch = Path(tempfile.mkdtemp(prefix=f"{task.task_id}_t{trial}_", dir=root))
        workspace = Workspace(scratch, fixture=task.fixture)
        try:
            trajectory = run_trajectory(
                client, workspace,
                task_id=task.task_id, trial=trial,
                system=task.system, instruction=task.instruction,
                tools=task.tools, limits=replace(task.limits, per_request_max_tokens=case.max_tokens),
                transcript_dir=Path(transcript_dir) if transcript_dir else None,
            )
            if not trajectory.ok:
                score = Score(value=0.0, passed=False,
                              detail=f"harness: server failed mid-trajectory ({trajectory.error[:120]})")
                response = Response(text="", error=trajectory.error,
                                    prompt_tokens=trajectory.prompt_tokens,
                                    completion_tokens=trajectory.completion_tokens,
                                    latency_s=trajectory.wall_s)
            else:
                verdict = summarize_checks(task.assess(workspace, trajectory))
                score = Score(
                    value=(verdict["outcomes_passed"] / verdict["outcomes_total"]
                           if verdict["outcomes_total"] else 0.0),
                    passed=bool(verdict["passed"]),
                    detail=_detail(verdict, trajectory),
                )
                response = Response(
                    text=trajectory.final_text or (trajectory.finished_summary or ""),
                    prompt_tokens=trajectory.prompt_tokens,
                    completion_tokens=trajectory.completion_tokens,
                    latency_s=trajectory.wall_s,
                    finish_reason=("length" if trajectory.stop_reason == "truncated"
                                   else trajectory.stop_reason),
                    reasoning_content=trajectory.reasoning_content,
                    reasoning_chars=len(trajectory.reasoning_content),
                )
                # Stash the full verdict where the CSV writer can reach it
                # without the scorer having to return two objects.
                case.meta["verdict"] = verdict
                case.meta["stop_reason"] = trajectory.stop_reason
                case.meta["steps"] = trajectory.steps
                case.meta["transcript"] = trajectory.transcript_path
        finally:
            # Always remove the scratch tree. Transcripts already hold everything
            # worth keeping, and fifteen configurations times ten trials of a
            # left-behind workspace is a slow disk leak.
            workspace.cleanup()
        results.append((score, response))
    return results


def _detail(verdict: Dict[str, Any], trajectory) -> str:
    parts = [f"{verdict['outcomes_passed']}/{verdict['outcomes_total']} outcomes",
             f"stop={trajectory.stop_reason}", f"steps={trajectory.steps}"]
    if verdict["constraints_violated"]:
        parts.append("violated=" + ",".join(verdict["constraints_violated"]))
    if verdict["outcomes_missed"]:
        parts.append("missed=" + ",".join(verdict["outcomes_missed"][:4]))
    return " · ".join(parts)


def score(case: Case, response_text: str) -> Score:
    """Never used — trajectories are scored inside `run_cases`."""
    return Score(value=0.0, passed=False, detail="agent cases are scored by run_cases")


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per task: how many trials passed outright, and why the rest did not.

    Reported as `passed/trials` rather than a mean because 1-of-3 is the single
    most decision-relevant outcome in this whole suite — a quant that completes
    an agent task intermittently is worse than useless for unattended work, and
    a 0.33 hides that.
    """
    by_task: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        if row.get("status", "ok") != "ok":
            continue
        by_task.setdefault(str(row.get("case_id") or "").split("#")[0], []).append(row)

    out: Dict[str, Any] = {}
    stop_reasons: Dict[str, int] = {}
    for task_id, trials in sorted(by_task.items()):
        passed = sum(int(t.get("passed") or 0) for t in trials)
        out[task_id] = {
            "passed": passed,
            "trials": len(trials),
            "verdict": ("reliable" if passed == len(trials)
                        else "never" if passed == 0
                        else "intermittent"),
            "mean_outcomes": round(sum(float(t.get("score") or 0.0) for t in trials) / len(trials), 3),
        }
        for trial in trials:
            for token in str(trial.get("detail") or "").split(" · "):
                if token.startswith("stop="):
                    stop_reasons[token[5:]] = stop_reasons.get(token[5:], 0) + 1
    return {
        "by_task": out,
        "stop_reasons": dict(sorted(stop_reasons.items())),
        "intermittent_tasks": [k for k, v in out.items() if v["verdict"] == "intermittent"],
    }
