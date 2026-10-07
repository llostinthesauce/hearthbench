#!/usr/bin/env python3
"""
The eval catalog. Adding a benchmark means adding one entry here.

Tier 1 generates its own data and runs with no network — that is the default
suite, and it is what makes this harness useful on an air-gapped machine.
Tier 2 needs a public dataset and only runs after an explicit `--fetch`.
"""
from __future__ import annotations

from typing import Dict, List

from .core import Eval
from .datasets import gsm8k, humaneval, ifeval, mmlu_pro
from .agentic import eval as agentic
from .offline import determinism, grounding, ifeval_local, longctx, niah, reasoning_hard, toolcall

EVALS: Dict[str, Eval] = {
    # -- tier 1: offline, self-generating ----------------------------------
    "niah": Eval(
        name="niah",
        tier=1,
        description="Needle-in-a-haystack retrieval across the advertised context window",
        build_cases=niah.build_cases,
        score=niah.score,
        metric="retrieval_rate",
    ),
    "ifeval_local": Eval(
        name="ifeval_local",
        tier=1,
        description="Programmatically verifiable instruction following (IFEval method, local prompts)",
        build_cases=ifeval_local.build_cases,
        score=ifeval_local.score,
        metric="constraint_rate",
    ),
    "determinism": Eval(
        name="determinism",
        tier=1,
        description="Byte-identical output across repeats at temperature 0",
        build_cases=determinism.build_cases,
        score=determinism.score,
        score_all=determinism.score_all,
        metric="stability",
    ),
    "toolcall": Eval(
        name="toolcall",
        tier=1,
        suite="behavioral",
        description="Native tool calling: emission, selection, arguments, parallelism, recovery, restraint",
        build_cases=toolcall.build_cases,
        score=toolcall.score,
        score_response=toolcall.score_response,
        metric="tool_call_rate",
    ),
    "longctx": Eval(
        name="longctx",
        tier=1,
        suite="behavioral",
        description="Long-context synthesis with an explicit fabrication and instruction-drift taxonomy",
        build_cases=longctx.build_cases,
        score=longctx.score,
        metric="long_context_rate",
    ),
    "grounding": Eval(
        name="grounding",
        tier=1,
        suite="behavioral",
        description="Answerable vs unanswerable, contradictions, and partial evidence",
        build_cases=grounding.build_cases,
        score=grounding.score,
        metric="grounding_rate",
    ),
    "reasoning_hard": Eval(
        name="reasoning_hard",
        tier=1,
        suite="behavioral",
        description="Distractor, constraint, trap, and multi-step problems repeated for reliability",
        build_cases=reasoning_hard.build_cases,
        score=reasoning_hard.score,
        metric="reasoning_rate",
    ),
    "agentic": Eval(
        name="agentic",
        tier=1,
        suite="behavioral",
        description="Bounded multi-step agent trajectories against a scratch repository fixture",
        build_cases=agentic.build_cases,
        score=agentic.score,
        run_cases=agentic.run_cases,
        metric="task_success",
    ),
    # -- tier 2: public datasets, opt-in download --------------------------
    "gsm8k": Eval(
        name="gsm8k",
        tier=2,
        description="Grade-school math word problems, exact-match final answer",
        build_cases=gsm8k.build_cases,
        score=gsm8k.score,
    ),
    "humaneval": Eval(
        name="humaneval",
        tier=2,
        description="Python functional correctness, pass@1 by executing reference tests",
        build_cases=humaneval.build_cases,
        score=humaneval.score,
        metric="pass@1",
        needs_code_execution=True,
    ),
    "ifeval": Eval(
        name="ifeval",
        tier=2,
        description="Google IFEval prompt set scored with the supported verifiers",
        build_cases=ifeval.build_cases,
        score=ifeval.score,
        metric="constraint_rate",
    ),
    "mmlu_pro": Eval(
        name="mmlu_pro",
        tier=2,
        description="Ten-option multiple-choice knowledge across academic categories",
        build_cases=mmlu_pro.build_cases,
        score=mmlu_pro.score,
    ),
}

# Per-eval extra reporting, when the module provides it.
SUMMARIZERS = {
    "niah": niah.summarize,
    "toolcall": toolcall.summarize,
    "longctx": longctx.summarize,
    "grounding": grounding.summarize,
    "reasoning_hard": reasoning_hard.summarize,
    "agentic": agentic.summarize,
    "determinism": determinism.summarize,
    "ifeval": ifeval.summarize,
    "mmlu_pro": mmlu_pro.summarize,
}


def tier1_names() -> List[str]:
    """The historical offline suite.

    Deliberately filtered to `suite == "core"`. The behavioural evals added for
    the quantization showdown are also tier 1 — they download nothing and
    generate their own data — but folding them into this shorthand would change
    the meaning of every `--evals tier1` result already on disk and turn a
    four-minute diagnostic into an hour-long one. They are reachable as
    `behavioral`, `showdown`, or by name.
    """
    return [name for name, spec in EVALS.items() if spec.tier == 1 and spec.suite == "core"]


def behavioral_names() -> List[str]:
    return [name for name, spec in EVALS.items() if spec.suite == "behavioral"]


def tier2_names() -> List[str]:
    return [name for name, spec in EVALS.items() if spec.tier == 2]


# The behavioural set the Qwen3.8-27B showdown runs. Named here rather than in
# the runner so the suite is one catalog entry away from any other caller.
SHOWDOWN = ["toolcall", "agentic", "reasoning_hard", "grounding", "longctx",
            "ifeval_local", "niah", "determinism"]

# Enough to prove the pipeline end to end without paying for the long-context or
# agent passes. Used by the smoke mode.
SHOWDOWN_SMOKE = ["toolcall", "grounding", "reasoning_hard"]


def resolve(names: List[str]) -> List[Eval]:
    """Expand names plus the 'tier1' / 'tier2' / 'all' / 'showdown' shorthands."""
    selected: List[str] = []
    for name in names:
        if name == "all":
            selected.extend(EVALS)
        elif name == "showdown":
            selected.extend(SHOWDOWN)
        elif name == "showdown_smoke":
            selected.extend(SHOWDOWN_SMOKE)
        elif name == "tier1":
            selected.extend(tier1_names())
        elif name == "tier1_all":
            selected.extend(n for n, spec in EVALS.items() if spec.tier == 1)
        elif name == "behavioral":
            selected.extend(behavioral_names())
        elif name == "tier2":
            selected.extend(tier2_names())
        elif name in EVALS:
            selected.append(name)
        else:
            raise SystemExit(
                f"Unknown eval '{name}'. Available: {', '.join(sorted(EVALS))} "
                f"(or tier1, tier1_all, tier2, behavioral, showdown, showdown_smoke, all)"
            )
    seen, ordered = set(), []
    for name in selected:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return [EVALS[name] for name in ordered]
