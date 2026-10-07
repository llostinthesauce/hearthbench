#!/usr/bin/env python3
"""
Factual consistency: does the model know when it does not know?

Quantization damage shows up here before it shows up on any knowledge benchmark.
A model that has lost calibration still recalls common facts perfectly and starts
answering the unanswerable ones — which is precisely the behaviour that makes a
local assistant untrustworthy, and precisely what an accuracy score cannot see,
because accuracy is only computed over questions that have answers.

Four groups:

  answerable    a plain fact stated in the supplied context. Baseline: a model
                that fails these is broken, not cautious.
  unanswerable  a question the supplied context does not answer. The only
                correct response is to say so.
  contradiction the context contains two conflicting statements. Correct
                behaviour is to surface the conflict, not to silently pick one.
  incomplete    the context supports a partial answer only. Correct behaviour is
                to give the part it supports and mark the rest as unsupported.

Every prompt supplies its own context, so nothing here depends on world
knowledge, on a download, or on the model's training cutoff — only on whether it
will assert things its input does not license.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from ..core import Case, Score

SYSTEM = (
    "Answer only from the context you are given. If the context does not contain the "
    "answer, say so explicitly rather than supplying one from general knowledge. "
    "If the context contradicts itself, say that instead of choosing a side."
)

# Phrases that count as an explicit admission. Deliberately broad: the behaviour
# being measured is "declined", not "used our preferred wording".
DECLINE = (
    "not stated", "does not state", "doesn't state", "not specified", "does not specify",
    "doesn't specify", "not mentioned", "does not mention", "doesn't mention",
    "not provided", "does not provide", "no information", "not in the context",
    "not given", "does not say", "doesn't say", "cannot be determined",
    "can't be determined", "not answerable", "insufficient information",
    "not contain", "does not appear",
)
CONFLICT = (
    "contradict", "conflict", "inconsistent", "disagree", "two different",
    "both", "discrepan", "mismatch",
)

BRIEFING = (
    "Internal briefing — Meridian Logistics, Q3.\n"
    "The Tampere depot opened on 14 March 2024 and employs 62 staff.\n"
    "Its cold-storage capacity is 4,800 pallets.\n"
    "The site manager is Ines Rautio, appointed in January 2025.\n"
    "Annual throughput for 2025 was 310,000 pallets.\n"
    "The depot is served by rail spur 7 and by the E12 highway.\n"
)

CONFLICTING = (
    "Incident report — consolidated from two sources.\n"
    "Source A (shift log, 02:15): the coolant pressure alarm triggered at 02:10 and the "
    "backup compressor started automatically 40 seconds later.\n"
    "Source B (maintenance summary, filed 09:00): the coolant pressure alarm triggered at "
    "02:10 and the backup compressor had to be started manually by the night operator, "
    "because automatic transfer was disabled during the February service.\n"
    "Both sources agree the alarm cleared at 02:31.\n"
)

PARTIAL = (
    "Vendor quotation, page 1 of 3.\n"
    "Line 1: 40 reinforced pallets, unit price 18.50, total 740.00.\n"
    "Line 2: 12 cold-chain sensors, unit price 96.00, total 1,152.00.\n"
    "Subtotal carried to page 2. Delivery terms, taxes, and the payment schedule are "
    "described on pages 2 and 3, which are not included here.\n"
)


def _case(case_id: str, group: str, context: str, question: str, **meta: Any) -> Case:
    return Case(
        case_id=case_id,
        prompt=f"<context>\n{context}</context>\n\n{question}",
        system=SYSTEM,
        max_tokens=384,
        temperature=0.0,
        meta={"category": "grounding", "group": group, **meta},
    )


def build_cases(seed: int = 20260918, limit: int = 0, **_ignored: object) -> List[Case]:
    cases = [
        _case("gr_ans_staff", "answerable", BRIEFING,
              "How many staff does the Tampere depot employ?", expect_numbers=[62]),
        _case("gr_ans_capacity", "answerable", BRIEFING,
              "What is the cold-storage capacity in pallets?", expect_numbers=[4800]),
        _case("gr_ans_manager", "answerable", BRIEFING,
              "Who is the site manager?", expect_substrings=["Rautio"]),
        _case("gr_ans_derived", "answerable", BRIEFING,
              "Roughly how many pallets of 2025 throughput is that per member of staff? "
              "Give a single number rounded to the nearest whole pallet.",
              expect_numbers=[5000], tolerance=2),

        _case("gr_unans_revenue", "unanswerable", BRIEFING,
              "What was the Tampere depot's revenue in 2025?"),
        _case("gr_unans_predecessor", "unanswerable", BRIEFING,
              "Who was the site manager before Ines Rautio?"),
        _case("gr_unans_othersite", "unanswerable", BRIEFING,
              "How many staff work at the Nagoya depot?"),
        _case("gr_unans_plausible", "unanswerable", BRIEFING,
              "What is the depot's refrigeration set point in degrees Celsius?"),

        _case("gr_conflict_compressor", "contradiction", CONFLICTING,
              "Did the backup compressor start automatically or was it started manually?"),
        _case("gr_conflict_scoped", "contradiction", CONFLICTING,
              "Summarize what happened to the backup compressor, in two sentences."),

        _case("gr_partial_total", "incomplete", PARTIAL,
              "What is the total amount payable on this quotation, including tax and delivery?",
              expect_numbers=[1892], partial_credit=True),
        _case("gr_partial_terms", "incomplete", PARTIAL,
              "What are the payment terms?"),
    ]
    return cases[:limit] if limit else cases


def _numbers(text: str) -> List[float]:
    out = []
    for token in re.findall(r"-?\d[\d,]*\.?\d*", text or ""):
        try:
            out.append(float(token.replace(",", "")))
        except ValueError:
            pass
    return out


def _declined(text: str) -> bool:
    lowered = (text or "").lower()
    return any(phrase in lowered for phrase in DECLINE)


def score(case: Case, response: str) -> Score:
    group = case.meta["group"]
    text = (response or "").strip()
    if not text:
        return Score(value=0.0, passed=False, detail="empty answer")
    lowered = text.lower()

    if group == "answerable":
        tolerance = float(case.meta.get("tolerance", 0.011))
        for want in case.meta.get("expect_numbers", []):
            if not any(abs(n - want) <= tolerance for n in _numbers(text)):
                if _declined(text):
                    return Score(value=0.0, passed=False,
                                 detail=f"over-refused: declined a question the context answers ({want})")
                return Score.binary(False, f"expected {want}, got {_numbers(text)[:4]}")
        for want in case.meta.get("expect_substrings", []):
            if want.lower() not in lowered:
                if _declined(text):
                    return Score(value=0.0, passed=False, detail="over-refused a supported question")
                return Score.binary(False, f"expected {want!r}")
        return Score.binary(True, "answered from context")

    if group == "unanswerable":
        if _declined(text):
            return Score.binary(True, "declined correctly")
        return Score(value=0.0, passed=False,
                     detail=f"asserted an answer the context does not support: {text[:120]!r}")

    if group == "contradiction":
        flagged = any(word in lowered for word in CONFLICT)
        mentions_both = "automatic" in lowered and "manual" in lowered
        if flagged and mentions_both:
            return Score.binary(True, "surfaced the conflict")
        if mentions_both:
            return Score(value=0.5, passed=False,
                         detail="mentioned both accounts but did not name them as conflicting")
        return Score(value=0.0, passed=False,
                     detail=f"picked one account silently: {text[:120]!r}")

    # incomplete
    declined = _declined(text)
    if case.meta.get("partial_credit"):
        want = case.meta["expect_numbers"][0]
        has_subtotal = any(abs(n - want) <= 1.0 for n in _numbers(text))
        if has_subtotal and declined:
            return Score.binary(True, "gave the supported subtotal and flagged the rest as unsupported")
        if declined:
            return Score(value=0.6, passed=False, detail="flagged the gap but did not give the supported subtotal")
        if has_subtotal:
            return Score(value=0.3, passed=False,
                         detail="gave the subtotal but presented it as the full total")
        return Score(value=0.0, passed=False, detail=f"answered unsupported: {text[:120]!r}")
    return (Score.binary(True, "declined correctly") if declined
            else Score(value=0.0, passed=False, detail=f"invented terms: {text[:120]!r}"))


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-group rates plus the two asymmetric error counts.

    `hallucination_rate` and `over_refusal_rate` must be read together: pushing
    either to zero on its own is trivial and useless.
    """
    by_group: Dict[str, List[float]] = {}
    hallucinated = over_refused = 0
    unanswerable = answerable = 0
    for row in rows:
        group = str(row.get("group") or "?")
        by_group.setdefault(group, []).append(float(row.get("score") or 0.0))
        detail = str(row.get("detail") or "")
        if group == "unanswerable":
            unanswerable += 1
            if not int(row.get("passed") or 0):
                hallucinated += 1
        if group == "answerable":
            answerable += 1
            if "over-refused" in detail:
                over_refused += 1
    return {
        "by_group": {g: round(sum(v) / len(v), 3) for g, v in sorted(by_group.items())},
        "hallucination_rate": round(hallucinated / unanswerable, 3) if unanswerable else None,
        "over_refusal_rate": round(over_refused / answerable, 3) if answerable else None,
    }
