#!/usr/bin/env python3
"""
Reasoning hard enough for quantization damage to become visible.

GSM8K saturates: a 27B model at 4-bit and the same model at 8-bit both score in
the nineties, and the two-point gap is inside the noise. This set is chosen for
*discrimination* instead — multi-constraint problems, planted distractors, and
premises that punish pattern-matching — and every case is repeated so that
"sometimes right" is reported as sometimes right rather than averaged into a
number that implies a stable capability.

Groups:
  distractor  the problem contains a number that is irrelevant. Using it gives a
              specific, recognisable wrong answer, so the failure is diagnosable
              rather than just wrong.
  constraint  several conditions must all hold; satisfying most gives a
              near-miss that a fuzzy grader would accept and this one does not.
  trap        the surface pattern suggests a standard problem whose answer is
              wrong here.
  multistep   five or more dependent steps, where one slip propagates.

Repeats run at the eval's own temperature, not at 0, because the question is
whether the capability is *reliable*, and a greedy decode answers a different
question. The per-case pass rate and its spread are both reported.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from ..core import Case, Score

SYSTEM = (
    "Solve the problem carefully. Some problems contain information that is not needed; "
    "ignore it. End your reply with a final line of exactly `ANSWER: <value>` and nothing after it."
)

PROBLEMS = [
    # (id, group, text, expected, wrong_if_distracted, note)
    ("rh_distract_freight", "distractor",
     "A depot ships 340 pallets on Monday and 275 on Tuesday. Each pallet weighs 42 kg. "
     "The depot employs 18 loaders and operates 6 forklifts. On Wednesday it ships 15% "
     "fewer pallets than the Monday and Tuesday total combined. How many pallets does it "
     "ship on Wednesday? Round to the nearest whole pallet.",
     523, 615, "18 and 6 and 42 are all irrelevant"),
    ("rh_distract_interest", "distractor",
     "An account holds 12,000 at the start of year one. It earns 5% simple interest per "
     "year on the original amount. A separate account, opened the same day by a different "
     "person, holds 30,000 at 7%. After 4 years, how much is in the FIRST account?",
     14400, 38400, "the second account is a distractor"),
    ("rh_distract_speed", "distractor",
     "A van leaves at 08:00 travelling 72 km/h. A second van leaves the same depot at 09:30 "
     "travelling 96 km/h on a different route to a different city 400 km away. How far has "
     "the FIRST van travelled by 13:00?",
     360, 336, "second van and the 400 km are irrelevant"),

    ("rh_constraint_seating", "constraint",
     "Four inspectors — Ada, Bo, Cy, Dee — sit in a row of four numbered seats 1 to 4, left "
     "to right. Ada is not in seat 1 or seat 4. Bo is not in seat 4. Bo sits immediately to "
     "the right of Ada. "
     "Dee sits in a lower-numbered seat than Cy. Who sits in seat 4?",
     "Cy", None, "unique: Dee=1, Ada=2, Bo=3, Cy=4 — verified by enumeration"),
    ("rh_constraint_mixture", "constraint",
     "A tank must end with exactly 200 litres of a solution that is 26% salt. Only two "
     "stocks are available: 40% salt and 15% salt. How many litres of the 40% stock are "
     "needed? Give a number rounded to two decimal places.",
     88.0, None, "0.40x + 0.15(200-x) = 52"),
    ("rh_constraint_schedule", "constraint",
     "A job has five tasks. T1 takes 3 hours. T2 takes 5 hours and cannot start until T1 "
     "finishes. T3 takes 2 hours and cannot start until T1 finishes. T4 takes 4 hours and "
     "cannot start until BOTH T2 and T3 finish. T5 takes 1 hour and cannot start until T3 "
     "finishes. Tasks with no unmet dependency may run in parallel, with unlimited workers. "
     "What is the minimum total elapsed time in hours?",
     12, 15, "critical path T1->T2->T4 = 3+5+4"),

    ("rh_trap_bat_ball", "trap",
     "A crate and its contents together cost 220. The contents cost 200 more than the empty "
     "crate. What does the empty crate cost?",
     10, 20, "the classic 'bat and ball' trap"),
    ("rh_trap_average_speed", "trap",
     "A truck drives 60 km at 30 km/h, then the same 60 km back at 60 km/h. What is its "
     "average speed for the whole round trip, in km/h? Give a number rounded to two decimals.",
     40.0, 45.0, "harmonic not arithmetic mean"),
    ("rh_trap_percent_back", "trap",
     "A price was reduced by 20%, then the reduced price was increased by 20%. The final "
     "price is 96.00. What was the original price? Give a number to two decimal places.",
     100.0, 96.0, "0.8 * 1.2 = 0.96, not 1.0"),

    ("rh_multistep_inventory", "multistep",
     "A warehouse starts the week with 1,240 units. Monday it ships 18% of stock. Tuesday it "
     "receives 350 units. Wednesday it ships a quarter of whatever is then on hand. Thursday "
     "it writes off 37 damaged units. Friday it receives twice what it wrote off on Thursday. "
     "How many units are on hand at the end of Friday? Give a whole number; if any day's "
     "figure is fractional, round down to the nearest whole unit before the next day.",
     1061, None, "1240 -18%-> 1016 (floor) +350-> 1366 -25%-> 1024 (floor) -37-> 987 +74-> 1061"),
    # -- harder: a healthy 27B answered every problem above 11/11 at greedy
    # decoding, so the set had no deterministic headroom left. These three are
    # chosen to need an exact enumeration, an exact fraction, or an exact
    # rounding rule applied at each step, where a small logit perturbation
    # changes the answer rather than the wording. All three ground truths are
    # re-derived independently in evals/test_showdown.py.
    ("rh_hard_probability", "hard",
     "A bag holds 7 red, 5 blue and 4 green marbles. Three are drawn at random without "
     "replacement. What is the probability that all three are the same colour? Give an "
     "exact fraction in lowest terms, formatted as a/b.",
     "7/80", None, "C(7,3)+C(5,3)+C(4,3)=49 over C(16,3)=560"),
    ("rh_hard_assignment", "hard",
     "Five services A, B, C, D and E are each assigned a different port from 8080, 8081, "
     "8082, 8083, 8084. A's port is higher than B's. C's port is exactly 2 more than D's. "
     "E has the lowest port of the five. B's port is not 8081. Which port is assigned to C?",
     8083, None, "unique solution A=8084 B=8082 C=8083 D=8081 E=8080, verified by enumeration"),
    ("rh_hard_compounding", "hard",
     "A tank holds 2450 litres. Step 1: 13% of the contents are drawn off. Step 2: 260 "
     "litres are added. Step 3: 15% of the contents evaporate. Step 4: 95 litres are drawn "
     "off. After each of steps 1 and 3, round the tank's contents DOWN to a whole litre "
     "before the next step begins. How many litres remain at the end?",
     1937, 1938,
     "2450 -> 2131.5 floor 2131 -> 2391 -> 2032.35 floor 2032 -> 1937; "
     "ignoring the floors gives 1937.775, i.e. 1938"),

    ("rh_multistep_dilution", "multistep",
     "A 500 ml container is full of a 12% solution. 150 ml is removed and replaced with pure "
     "solvent. Then 100 ml of the new mixture is removed and replaced with pure solvent. What "
     "is the final concentration as a percentage, rounded to three decimal places?",
     6.72, None, "12*(350/500)*(400/500) = 6.72"),
]


def build_cases(seed: int = 20260918, repeats: int = 3, limit: int = 0,
                temperature: float = 0.6, **_ignored: object) -> List[Case]:
    """One Case per (problem x repeat).

    `repeats` deliberately shares the flag `bench_quality` already passes to the
    determinism eval, so raising trial count raises it for both reliability
    measurements at once.
    """
    trials = max(1, int(repeats))
    cases: List[Case] = []
    for problem_id, group, text, expected, distracted, note in PROBLEMS:
        for trial in range(trials):
            cases.append(Case(
                case_id=f"{problem_id}#r{trial}",
                prompt=text,
                system=SYSTEM,
                # The `hard` problems need a longer derivation — an exact
                # enumeration or a four-step rounding chain — and a healthy 27B
                # was measured running out of a 1536-token budget mid-working on
                # the assignment problem, emitting no `ANSWER:` line at all. That
                # scores identically to a wrong answer while actually being a
                # budget artifact, so it would have read as reasoning damage on
                # every arm. Size the budget to the problem, not the group.
                max_tokens=3072 if group == "hard" else 1536,
                # Trial 0 is greedy so there is always one deterministic sample
                # to compare a flaky case against; the rest sample.
                temperature=0.0 if trial == 0 else temperature,
                meta={"category": "reasoning", "group": group, "problem": problem_id,
                      "repeat": trial, "expected": expected,
                      "distracted_answer": distracted, "note": note},
            ))
    return cases[:limit] if limit else cases


def _final_answer(text: str) -> str:
    match = re.findall(r"ANSWER:\s*(.+?)\s*$", text or "", re.IGNORECASE | re.MULTILINE)
    return match[-1].strip() if match else ""


def _as_float(token: str):
    cleaned = re.sub(r"[^\d.\-]", "", (token or "").replace(",", ""))
    try:
        return float(cleaned)
    except ValueError:
        return None


def score(case: Case, response: str) -> Score:
    expected = case.meta["expected"]
    stated = _final_answer(response)
    if not stated:
        # Falling back to the last number in the body would credit a model that
        # ignored the format; instead this is reported as a format failure, which
        # is what it is. A long response that stops without the line is almost
        # certainly a budget miss rather than a refusal to follow the format, and
        # is labelled so the two are distinguishable in the breakdown.
        kind = "budget" if len(response or "") > 2000 else "format"
        return Score(value=0.0, passed=False, detail=f"no `ANSWER:` line ({kind})")

    if isinstance(expected, str):
        passed = expected.lower() in stated.lower()
        return Score.binary(passed, f"answered {stated[:40]!r}, expected {expected!r}")

    value = _as_float(stated)
    if value is None:
        return Score(value=0.0, passed=False, detail=f"non-numeric answer {stated[:40]!r}")
    tolerance = 0.011 if isinstance(expected, float) else 0.5
    if abs(value - float(expected)) <= tolerance:
        return Score.binary(True, f"{value}")
    distracted = case.meta.get("distracted_answer")
    if distracted is not None and abs(value - float(distracted)) <= tolerance:
        return Score(value=0.0, passed=False,
                     detail=f"{value} — used the planted distractor ({case.meta['note']})")
    return Score.binary(False, f"{value}, expected {expected}")


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-problem reliability, not a mean.

    A problem answered correctly 1 time in 3 is reported as flaky. Averaging it
    with two saturated problems would produce a number that describes nothing.
    """
    by_problem: Dict[str, List[int]] = {}
    by_group: Dict[str, List[int]] = {}
    distracted = 0
    format_misses = 0
    budget_misses = 0
    for row in rows:
        problem = str(row.get("case_id") or "").split("#")[0]
        passed = int(row.get("passed") or 0)
        by_problem.setdefault(problem, []).append(passed)
        by_group.setdefault(str(row.get("group") or "?"), []).append(passed)
        detail = str(row.get("detail") or "")
        if "planted distractor" in detail:
            distracted += 1
        if "no `ANSWER:` line" in detail:
            format_misses += 1
            if "(budget)" in detail:
                budget_misses += 1

    flaky = {p: f"{sum(v)}/{len(v)}" for p, v in sorted(by_problem.items())
             if 0 < sum(v) < len(v)}
    always = [p for p, v in by_problem.items() if v and sum(v) == len(v)]
    never = [p for p, v in by_problem.items() if v and sum(v) == 0]
    return {
        "by_group": {g: round(sum(v) / len(v), 3) for g, v in sorted(by_group.items())},
        "reliable": len(always),
        "flaky": flaky,
        "never": sorted(never),
        "distractor_captures": distracted,
        "format_misses": format_misses,
        # A subset of format_misses that ran out of tokens mid-working. These are
        # a harness budget problem, not evidence about the model's reasoning, and
        # must not be read as capability loss.
        "budget_misses": budget_misses,
    }
