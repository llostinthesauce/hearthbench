#!/usr/bin/env python3
"""
Long context that requires *using* the context, plus a failure taxonomy.

`niah` already answers "is the value still retrievable at depth D". This eval
answers the harder question the agent workspace actually depends on: can the
model find several facts scattered through a long document and combine them,
while still obeying a format instruction given at the very top of a 128K prompt?

Three case families per context length:

  combine   two planted values at different depths must be arithmetically
            combined. Retrieving one and guessing is worth nothing.
  absent    a fact that looks exactly like the planted ones is asked for and is
            NOT in the document. The only correct answer is to say so. This is
            the fabrication probe, and it is the reason a long-context score that
            only counts hits is misleading — a model that answers everything
            confidently scores 100% on retrieval and is useless.
  drift     the output format is specified in the first 200 tokens of a very
            long prompt and must still be obeyed at the end of it.

Outcomes are classified rather than averaged. "Scored 0.4" tells you nothing;
"retrieved both values but ignored the format" and "fabricated a plausible
number" are different defects with different fixes.
"""
from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Tuple

from ..core import Case, Score
from . import niah

DEFAULT_CONTEXTS = (8192, 32768, 65536, 131072)

# Reuse the niah haystack generator: same prose model, same token calibration,
# so a result here is comparable with a result there rather than being a second
# incompatible notion of "32K".
_haystack_words = niah._haystack_words
_reserve_for = niah._reserve_for

FACILITIES = [
    ("Lisbon", "harbor access"), ("Tampere", "cold storage"), ("Nagoya", "freight dock"),
    ("Valparaiso", "customs gate"), ("Reykjavik", "fuel depot"), ("Kigali", "transit hub"),
    ("Hobart", "dry dock"), ("Tromso", "relay mast"),
]
ABSENT_FACILITY = ("Antofagasta", "bonded warehouse")

FORMAT_TAG = "RESULT"


def _fact(city: str, kind: str, value: int) -> str:
    return f"Record: the {kind} at {city} was assessed at {value} units in the most recent audit."


def _insert(words: List[str], items: List[Tuple[str, str, int]], depths: List[float]) -> str:
    placements = sorted(
        ((int(depth * len(words)), item) for depth, item in zip(depths, items)),
        key=lambda pair: pair[0], reverse=True,
    )
    out = list(words)
    for index, (city, kind, value) in placements:
        out[index:index] = _fact(city, kind, value).split()
    return " ".join(out)


def build_cases(
    ctx_cap: int = 131072,
    contexts: Tuple[int, ...] = DEFAULT_CONTEXTS,
    seed: int = 20260918,
    limit: int = 0,
    **_ignored: object,
) -> List[Case]:
    cases: List[Case] = []
    usable = [c for c in contexts if c <= ctx_cap] or [min(contexts[0], ctx_cap)]

    for context_len in usable:
        body_tokens = max(256, context_len - _reserve_for(context_len))
        rng = random.Random(f"{seed}:{context_len}")
        words = _haystack_words(body_tokens, rng)

        chosen = rng.sample(FACILITIES, k=3)
        planted = [(city, kind, rng.randint(1000, 9999)) for city, kind in chosen]
        document = _insert(words, planted, [0.08, 0.52, 0.93])
        (city_a, kind_a, value_a), (city_b, kind_b, value_b), (city_c, kind_c, value_c) = planted

        # combine — needs the first and last plant, i.e. both ends of the window
        cases.append(Case(
            case_id=f"lc_combine_{context_len}",
            prompt=(
                "Read the document below, then answer the question after it.\n\n"
                f"<document>\n{document}\n</document>\n\n"
                f"Question: add the assessed value for the {kind_a} at {city_a} to the "
                f"assessed value for the {kind_c} at {city_c}. Reply with the sum as a "
                "single integer and nothing else."
            ),
            max_tokens=256, temperature=0.0,
            meta={"category": "long_context", "group": "combine", "context_len": context_len,
                  "expected": value_a + value_c, "parts": [value_a, value_c]},
        ))

        # absent — the fabrication probe
        cases.append(Case(
            case_id=f"lc_absent_{context_len}",
            prompt=(
                "Read the document below, then answer the question after it.\n\n"
                f"<document>\n{document}\n</document>\n\n"
                f"Question: what assessed value does the document give for the "
                f"{ABSENT_FACILITY[1]} at {ABSENT_FACILITY[0]}?\n"
                "If the document does not state it, reply with exactly: NOT IN DOCUMENT"
            ),
            max_tokens=256, temperature=0.0,
            meta={"category": "long_context", "group": "absent", "context_len": context_len,
                  "planted_values": [value_a, value_b, value_c]},
        ))

        # drift — the instruction is at the top, the work is at the bottom
        cases.append(Case(
            case_id=f"lc_drift_{context_len}",
            prompt=(
                f"FORMAT INSTRUCTION (applies to your entire reply): your last line must be "
                f"exactly `{FORMAT_TAG}: <integer>` with no trailing text, and you must not use "
                f"the word 'document' anywhere in your reply.\n\n"
                "Read the following, then answer the question after it.\n\n"
                f"<document>\n{document}\n</document>\n\n"
                f"Question: what is the assessed value for the {kind_b} at {city_b}? "
                "Explain in one short sentence, then give the required final line."
            ),
            max_tokens=384, temperature=0.0,
            meta={"category": "long_context", "group": "drift", "context_len": context_len,
                  "expected": value_b, "forbidden_word": "document"},
        ))

    return cases[:limit] if limit else cases


def _integers(text: str) -> List[int]:
    return [int(t.replace(",", "")) for t in re.findall(r"-?\d[\d,]*", text or "")]


def score(case: Case, response: str) -> Score:
    group = case.meta["group"]
    text = (response or "").strip()

    if not text:
        return Score(value=0.0, passed=False, detail="empty_answer")

    if group == "combine":
        expected = int(case.meta["expected"])
        found = _integers(text)
        if expected in found:
            return Score.binary(True, f"combined correctly ({expected})")
        parts = case.meta["parts"]
        hits = [p for p in parts if p in found]
        if len(hits) == len(parts):
            return Score(value=0.5, passed=False,
                         detail=f"retrieved both values {parts} but did not combine them")
        if hits:
            return Score(value=0.25, passed=False,
                         detail=f"retrieved only {hits} of {parts} — partial retrieval")
        return Score(value=0.0, passed=False, detail=f"no planted value retrieved; said {found[:4]}")

    if group == "absent":
        declined = "not in document" in text.lower()
        if declined:
            return Score.binary(True, "correctly declined")
        numbers = _integers(text)
        planted = set(case.meta["planted_values"])
        if any(n in planted for n in numbers):
            return Score(value=0.0, passed=False,
                         detail=f"answered with a value planted for a DIFFERENT facility ({numbers[:3]}) — misattribution")
        if numbers:
            return Score(value=0.0, passed=False,
                         detail=f"fabricated a value not present anywhere in the document ({numbers[:3]})")
        return Score(value=0.0, passed=False, detail="did not use the required refusal string")

    # drift
    expected = int(case.meta["expected"])
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    final = lines[-1] if lines else ""
    format_ok = bool(re.fullmatch(rf"{FORMAT_TAG}:\s*-?\d+", final))
    word_ok = case.meta["forbidden_word"] not in text.lower()
    value_ok = expected in _integers(final if format_ok else text)
    if format_ok and word_ok and value_ok:
        return Score.binary(True, "retrieved and obeyed both constraints")
    failures = []
    if not value_ok:
        failures.append("wrong/absent value")
    if not format_ok:
        failures.append(f"final line {final[:60]!r}")
    if not word_ok:
        failures.append("used the forbidden word")
    # Retrieval and compliance are scored separately; a model that found the
    # value but drifted on format has a different problem from one that lost it.
    value = 0.5 if value_ok else 0.0
    return Score(value=value, passed=False, detail="instruction drift: " + "; ".join(failures))


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per (context length x group) outcome, plus an explicit fabrication tally.

    `usable_context` is the largest length at which `combine` still passed. That
    is the number worth quoting — not the ctx_cap the config advertises.
    """
    grid: Dict[str, Dict[str, List[float]]] = {}
    fabricated = misattributed = errors = 0
    for row in rows:
        group = str(row.get("group") or "?")
        context = str(row.get("context_len") or "?")
        grid.setdefault(group, {}).setdefault(context, []).append(float(row.get("score") or 0.0))
        detail = str(row.get("detail") or "")
        if "fabricated" in detail:
            fabricated += 1
        if "misattribution" in detail:
            misattributed += 1
        if str(row.get("status") or "ok") != "ok":
            errors += 1

    combine = grid.get("combine", {})
    passed_lengths = [int(c) for c, scores in combine.items()
                      if c.isdigit() and scores and min(scores) >= 1.0]
    return {
        "by_group_and_context": {
            group: {context: round(sum(v) / len(v), 3) for context, v in sorted(inner.items(), key=lambda kv: int(kv[0]))}
            for group, inner in sorted(grid.items())
        },
        "usable_context": max(passed_lengths) if passed_lengths else 0,
        "fabricated_answers": fabricated,
        "misattributed_answers": misattributed,
        "request_failures": errors,
    }
