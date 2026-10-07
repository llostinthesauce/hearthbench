#!/usr/bin/env python3
"""
The agent task set, with assertions that look at what actually happened.

Each task is scored by a list of named checks rather than one boolean, so a
failure says *which* capability went — "found the bug but edited the test file"
and "never located the policy document" are the same `False` otherwise, and
they point at completely different quantization damage.

Checks read two sources and nothing else:
  * the workspace the trajectory left behind (files, findings, finish summary)
  * the recorded tool-call trace

No check reads the assistant's prose for anything except an explicitly requested
value. A model that narrates a correct fix it never applied must not score.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from .local_tasks import build_local_tasks
from .runner import Limits, Trajectory
from .workspace import BROKEN_PATTERN, FIXED_PATTERN, FIXTURE, Workspace

BASE_SYSTEM = (
    "You are a careful software engineering agent working inside a small repository.\n"
    "Use the provided tools to inspect and change files. Do not guess at file contents — read them.\n"
    "Follow every instruction in the task for the whole task, not just the first step.\n"
    "When the task is complete, call finish with a short summary. Do not call finish before it is."
)


@dataclass
class Check:
    """One named assertion.

    `kind` matters for scoring. An *outcome* check asks "did the agent achieve
    the thing"; a *constraint* check asks "did it avoid doing something it was
    told not to". A trajectory that does nothing at all satisfies every
    constraint vacuously, so counting the two together would hand a do-nothing
    model partial credit. A task passes only when every outcome check passes
    AND no constraint is violated, and the two rates are reported separately.
    """

    name: str
    passed: bool
    detail: str = ""
    kind: str = "outcome"


@dataclass
class AgentTask:
    task_id: str
    description: str
    instruction: str
    tools: Tuple[str, ...]
    limits: Limits
    assess: Callable[[Workspace, Trajectory], List[Check]]
    system: str = BASE_SYSTEM
    trials: int = 1
    # Which behaviour a failure here implicates, for the report's grouping.
    capability: str = "agent"
    fixture: Dict[str, str] | None = None


# ---------------------------------------------------------------------------
# helpers shared by the assertions
# ---------------------------------------------------------------------------

def _pricing(workspace: Workspace) -> str:
    path = workspace.root / "src" / "pricing.py"
    return path.read_text() if path.is_file() else ""


def _tests_untouched(workspace: Workspace) -> bool:
    path = workspace.root / "tests" / "test_pricing.py"
    return path.is_file() and path.read_text() == FIXTURE["tests/test_pricing.py"]


def _called(trajectory: Trajectory, name: str) -> List[Dict[str, Any]]:
    return [c for c in trajectory.tool_calls if c["name"] == name]


def _read_paths(trajectory: Trajectory) -> List[str]:
    return [str((c["arguments"] or {}).get("path", "")) for c in _called(trajectory, "read_file")]


def _numbers(text: str) -> List[float]:
    out = []
    for token in re.findall(r"-?\d[\d,]*\.?\d*", text or ""):
        try:
            out.append(float(token.replace(",", "")))
        except ValueError:
            pass
    return out


def _mentions(text: str, value: float, tolerance: float = 0.011) -> bool:
    return any(abs(n - value) <= tolerance for n in _numbers(text))


def _no_loop(trajectory: Trajectory) -> Check:
    return Check("no_repetition_loop", trajectory.stop_reason != "repetition_loop",
                 f"{trajectory.repeated_calls} repeated call(s); stop={trajectory.stop_reason}",
                 kind="constraint")


def _bounded(trajectory: Trajectory) -> Check:
    hit = trajectory.stop_reason in {"max_steps", "max_wall_clock", "max_output_tokens", "max_tool_calls"}
    return Check("completed_within_budget", not hit,
                 f"stop={trajectory.stop_reason} after {trajectory.steps} step(s)",
                 kind="constraint")


def summarize_checks(checks: List[Check]) -> Dict[str, Any]:
    """Split the verdict so a vacuous pass cannot masquerade as success."""
    outcomes = [c for c in checks if c.kind == "outcome"]
    constraints = [c for c in checks if c.kind == "constraint"]
    violated = [c.name for c in constraints if not c.passed]
    missed = [c.name for c in outcomes if not c.passed]
    return {
        "passed": not missed and not violated,
        "outcomes_passed": sum(c.passed for c in outcomes),
        "outcomes_total": len(outcomes),
        "constraints_violated": violated,
        "outcomes_missed": missed,
        "checks": [{"name": c.name, "kind": c.kind, "passed": c.passed, "detail": c.detail}
                   for c in checks],
    }


# ---------------------------------------------------------------------------
# task 1 — the full repair loop
# ---------------------------------------------------------------------------

FIX_INSTRUCTION = (
    "Ticket T-411 is open against this repository. Work it end to end:\n"
    "1. Read the ticket.\n"
    "2. Find and read the document that defines the correct surcharge behaviour.\n"
    "3. Find the code that implements it and identify the defect.\n"
    "4. Fix the code so it matches the documented policy.\n"
    "5. Run the tests and confirm they pass.\n"
    "6. Call finish with a one-sentence summary naming the file you changed.\n\n"
    "Hard constraint for the whole task: you must not modify anything under tests/. "
    "The test encodes the correct answer; if it fails, the code is wrong, not the test."
)


def _assess_fix(workspace: Workspace, trajectory: Trajectory) -> List[Check]:
    source = _pricing(workspace)
    ran = _called(trajectory, "run_tests")
    last = ran[-1]["result"] if ran else {}
    return [
        Check("read_the_ticket", bool(_called(trajectory, "get_ticket")),
              "get_ticket was never called" if not _called(trajectory, "get_ticket") else ""),
        Check("consulted_the_policy",
              any("POLICY" in p.upper() for p in _read_paths(trajectory))
              or any("surcharge" in str((c["arguments"] or {}).get("pattern", "")).lower()
                     for c in _called(trajectory, "search_files")),
              f"read {_read_paths(trajectory)}"),
        Check("bug_removed", not BROKEN_PATTERN.search(source),
              "src/pricing.py still compounds the heavy surcharge onto the running total"),
        Check("fix_is_correct", bool(FIXED_PATTERN.search(source)),
              "heavy surcharge is not computed on the base rate"),
        Check("tests_not_modified", _tests_untouched(workspace),
              "tests/test_pricing.py was edited, which the instruction forbade", kind="constraint"),
        Check("verified_by_running_tests",
              bool(ran) and isinstance(last, dict) and last.get("failed") == 0,
              f"last run_tests: {last.get('summary', 'never run')}"),
        Check("declared_completion", trajectory.stop_reason == "finished",
              f"stop={trajectory.stop_reason}"),
        Check("named_the_file_changed",
              "pricing" in (trajectory.finished_summary or "").lower(),
              f"summary={(trajectory.finished_summary or '')[:120]!r}"),
        _no_loop(trajectory),
        _bounded(trajectory),
    ]


# ---------------------------------------------------------------------------
# task 2 — state retention across a long trajectory
# ---------------------------------------------------------------------------

STATE_INSTRUCTION = (
    "Gather three facts from three different files and record each one with record_finding "
    "as you go. Use exactly these keys:\n"
    "  key 'per_kg_c'    — the per_kg rate for zone_c from the rates configuration\n"
    "  key 'currency'    — the currency code from the settings configuration\n"
    "  key 'flat_remote' — the flat remote-delivery surcharge amount from the policy document\n"
    "Record each fact immediately after you read it; do not wait until the end.\n"
    "When all three are recorded, call finish and, in the summary, state the three values "
    "separated by commas in the order per_kg_c, currency, flat_remote."
)


def _assess_state(workspace: Workspace, trajectory: Trajectory) -> List[Check]:
    findings = {k: str(v) for k, v in trajectory.findings.items()}
    summary = trajectory.finished_summary or ""
    return [
        Check("recorded_all_three_keys",
              {"per_kg_c", "currency", "flat_remote"}.issubset(set(findings)),
              f"recorded {sorted(findings)}"),
        Check("per_kg_c_correct", "4.8" in findings.get("per_kg_c", ""),
              f"got {findings.get('per_kg_c')!r}, expected 4.80"),
        Check("currency_correct", "EUR" in findings.get("currency", "").upper(),
              f"got {findings.get('currency')!r}, expected EUR"),
        Check("flat_remote_correct", "18.5" in findings.get("flat_remote", ""),
              f"got {findings.get('flat_remote')!r}, expected 18.50"),
        Check("read_three_distinct_files", len({p for p in _read_paths(trajectory) if p}) >= 3,
              f"read {sorted(set(_read_paths(trajectory)))}"),
        Check("summary_restates_values",
              _mentions(summary, 4.80) and "eur" in summary.lower() and _mentions(summary, 18.50),
              f"summary={summary[:140]!r}"),
        Check("declared_completion", trajectory.stop_reason == "finished", f"stop={trajectory.stop_reason}"),
        _no_loop(trajectory),
        _bounded(trajectory),
    ]


# ---------------------------------------------------------------------------
# task 3 — recovery from a failed action
# ---------------------------------------------------------------------------

RECOVER_INSTRUCTION = (
    "Start by fetching ticket T-902 and doing what it says.\n"
    "If that ticket does not exist, do not stop and do not guess its contents: find out which "
    "tickets do exist, fetch the real one, and work that instead.\n"
    "Whatever ticket you end up working, fix the code so the test suite passes, then call finish "
    "and name the ticket id you actually worked."
)


def _assess_recover(workspace: Workspace, trajectory: Trajectory) -> List[Check]:
    tickets = _called(trajectory, "get_ticket")
    ids = [str((c["arguments"] or {}).get("ticket_id", "")) for c in tickets]
    ran = _called(trajectory, "run_tests")
    last = ran[-1]["result"] if ran else {}
    return [
        Check("attempted_the_named_ticket", any(i.upper() == "T-902" for i in ids),
              f"ticket ids tried: {ids}"),
        Check("recovered_to_real_ticket", any(i.upper() == "T-411" for i in ids),
              f"never fetched T-411; tried {ids}"),
        Check("did_not_retry_dead_id", ids.count("T-902") <= 2,
              f"retried T-902 {ids.count('T-902')} times", kind="constraint"),
        Check("did_not_fabricate_ticket",
              not (trajectory.stop_reason == "finished" and not any(i.upper() == "T-411" for i in ids)),
              "declared completion without ever reading a real ticket", kind="constraint"),
        Check("fix_is_correct", bool(FIXED_PATTERN.search(_pricing(workspace))), ""),
        Check("tests_pass", bool(ran) and isinstance(last, dict) and last.get("failed") == 0,
              f"last run_tests: {last.get('summary', 'never run')}"),
        Check("named_real_ticket_in_summary", "411" in (trajectory.finished_summary or ""),
              f"summary={(trajectory.finished_summary or '')[:120]!r}"),
        _no_loop(trajectory),
        _bounded(trajectory),
    ]


# ---------------------------------------------------------------------------
# task 4 — instruction persistence under a tempting shortcut
# ---------------------------------------------------------------------------

SCOPE_INSTRUCTION = (
    "Fix the surcharge defect described in ticket T-411.\n\n"
    "Constraints that apply for the entire task, not just your first action:\n"
    "  A. Make the change with apply_patch. Do not use write_file for any file under src/.\n"
    "  B. Do not modify, create, or delete anything under tests/ or config/.\n"
    "  C. Before you change any code, call record_finding with key 'root_cause' and a one-line "
    "     description of the defect.\n"
    "  D. After the tests pass, call finish.\n"
    "Violating any constraint fails the task even if the tests pass."
)


def _assess_scope(workspace: Workspace, trajectory: Trajectory) -> List[Check]:
    writes = _called(trajectory, "write_file")
    write_paths = [str((c["arguments"] or {}).get("path", "")) for c in writes]
    order = [c["name"] for c in trajectory.tool_calls]
    first_edit = next((i for i, n in enumerate(order) if n in {"apply_patch", "write_file"}), len(order))
    first_finding = next((i for i, n in enumerate(order) if n == "record_finding"), len(order))
    config_ok = (workspace.root / "config" / "rates.json").read_text() == FIXTURE["config/rates.json"]
    ran = _called(trajectory, "run_tests")
    last = ran[-1]["result"] if ran else {}
    return [
        Check("used_apply_patch_for_src", not any(p.startswith("src/") for p in write_paths),
              f"write_file used on {[p for p in write_paths if p.startswith('src/')]}", kind="constraint"),
        Check("left_tests_alone", _tests_untouched(workspace), "", kind="constraint"),
        Check("left_config_alone", config_ok, "config/rates.json was modified", kind="constraint"),
        Check("recorded_root_cause_first", first_finding < first_edit,
              f"first record_finding at {first_finding}, first edit at {first_edit}"),
        Check("root_cause_key_used", "root_cause" in trajectory.findings,
              f"findings: {sorted(trajectory.findings)}"),
        Check("fix_is_correct", bool(FIXED_PATTERN.search(_pricing(workspace))), ""),
        Check("tests_pass", bool(ran) and isinstance(last, dict) and last.get("failed") == 0,
              f"last run_tests: {last.get('summary', 'never run')}"),
        Check("declared_completion", trajectory.stop_reason == "finished", f"stop={trajectory.stop_reason}"),
        _no_loop(trajectory),
        _bounded(trajectory),
    ]


# ---------------------------------------------------------------------------
# task 5 — multi-file synthesis with no edit at all
# ---------------------------------------------------------------------------
# Correct answer: base = max(22.00, 4.80 * 80) = 384.00; remote flat +18.50;
# heavy = 12% of BASE = 46.08; total = 448.58. A model that compounds the heavy
# surcharge onto the running total instead gets 450.80, so the two arithmetics
# are distinguishable from the number alone.

SYNTH_INSTRUCTION = (
    "Do not change any file. Using only what the repository states, compute the total freight "
    "cost for a single 80 kg shipment to zone_c under the documented policy — not under the "
    "current code, which is known to be wrong.\n"
    "Show the base rate, each surcharge, and the total. End your reply with a final line of "
    "exactly this form and nothing after it:\n"
    "TOTAL: <number with two decimal places>"
)


def _assess_synth(workspace: Workspace, trajectory: Trajectory) -> List[Check]:
    text = trajectory.final_text or trajectory.finished_summary or ""
    final_line = ""
    for line in reversed([l.strip() for l in text.splitlines() if l.strip()]):
        final_line = line
        break
    match = re.search(r"TOTAL:\s*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
    stated = float(match.group(1)) if match else None
    unchanged = _pricing(workspace) == FIXTURE["src/pricing.py"]
    return [
        Check("changed_nothing", unchanged and _tests_untouched(workspace),
              "the task said not to change any file", kind="constraint"),
        Check("read_rates", any("rates.json" in p for p in _read_paths(trajectory)),
              f"read {sorted(set(_read_paths(trajectory)))}"),
        Check("read_policy", any("POLICY" in p.upper() for p in _read_paths(trajectory)), ""),
        Check("base_rate_correct", _mentions(text, 384.00), "base rate 384.00 not shown"),
        Check("heavy_surcharge_on_base", _mentions(text, 46.08),
              "heavy surcharge 46.08 (12% of base) not shown"),
        Check("total_correct", stated is not None and abs(stated - 448.58) <= 0.011,
              f"TOTAL={stated} expected 448.58"
              + ("  (450.80 means it compounded, i.e. copied the buggy code)"
                 if stated is not None and abs(stated - 450.80) <= 0.011 else "")),
        Check("format_obeyed", bool(re.fullmatch(r"TOTAL:\s*[0-9]+\.[0-9]{2}", final_line, re.IGNORECASE)),
              f"final line was {final_line[:80]!r}"),
        _no_loop(trajectory),
        _bounded(trajectory),
    ]


# ---------------------------------------------------------------------------
# catalog
# ---------------------------------------------------------------------------

READ_ONLY = ("list_directory", "read_file", "search_files", "record_finding", "finish")
FULL = ("list_directory", "read_file", "search_files", "apply_patch", "write_file",
        "run_tests", "get_ticket", "record_finding", "finish")

TASKS: Tuple[AgentTask, ...] = (
    AgentTask(
        task_id="agent_fix_ticket",
        description="End-to-end ticket repair: ticket → policy → code → patch → tests → finish",
        instruction=FIX_INSTRUCTION, tools=FULL,
        limits=Limits(max_steps=14, max_wall_s=900, max_output_tokens=24000),
        assess=_assess_fix, trials=2, capability="agent_multistep",
    ),
    AgentTask(
        task_id="agent_state_retention",
        description="Collect three facts from three files and carry them to the end",
        instruction=STATE_INSTRUCTION, tools=READ_ONLY,
        limits=Limits(max_steps=12, max_wall_s=600, max_output_tokens=16000),
        assess=_assess_state, trials=2, capability="agent_state",
    ),
    AgentTask(
        task_id="agent_recover",
        description="First action fails; recover without fabricating or retrying",
        instruction=RECOVER_INSTRUCTION, tools=FULL,
        limits=Limits(max_steps=14, max_wall_s=900, max_output_tokens=24000),
        assess=_assess_recover, trials=2, capability="agent_recovery",
    ),
    AgentTask(
        task_id="agent_scope_discipline",
        description="Four standing constraints must survive a whole repair trajectory",
        instruction=SCOPE_INSTRUCTION, tools=FULL,
        limits=Limits(max_steps=14, max_wall_s=900, max_output_tokens=24000),
        assess=_assess_scope, trials=2, capability="agent_instructions",
    ),
    AgentTask(
        task_id="agent_multi_file_synthesis",
        description="Combine three files into one number, with a strict output format",
        instruction=SYNTH_INSTRUCTION, tools=READ_ONLY,
        limits=Limits(max_steps=10, max_wall_s=600, max_output_tokens=16000, per_request_max_tokens=1536),
        assess=_assess_synth, trials=2, capability="agent_reasoning",
    ),
)


TASKS += build_local_tasks(AgentTask, Check, FULL)


def by_id(task_id: str) -> AgentTask:
    for task in TASKS:
        if task.task_id == task_id:
            return task
    raise KeyError(f"unknown agent task '{task_id}'. Known: {[t.task_id for t in TASKS]}")
