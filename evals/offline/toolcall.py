#!/usr/bin/env python3
"""
Native tool calling — the capability that degrades first and shows up last.

Throughput benchmarks and multiple-choice evals both miss this entirely. A quant
that has lost tool-call fidelity still answers MMLU at nearly full accuracy and
still generates at the same speed; it simply starts naming a plausible-but-wrong
tool, or emitting `{"path": "a.py",}`, or calling `search` when the answer was
already in the conversation. Every one of those is a silent agent failure.

Seven case groups, each isolating one failure mode so a regression points at a
cause rather than at "tool use got worse":

  emit        does a tool-shaped request produce a tool call at all
  select      the right tool among deliberately plausible neighbours
  arguments   required parameters, correct names, correct types, valid JSON
  parallel    two independent facts in one turn
  interpret   a tool result is already in the history; use it, do not re-call
  recover     the tool returned an error; do something other than repeat it
  restraint   the answer needs no tool; calling one is the failure

Scoring is per-group and deliberately blunt: these are pass/fail behaviours, and
a decimal average over them would invent precision the underlying signal does not
have. `summarize()` reports each group separately for exactly that reason.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from ..core import Case, Response, Score

# ---------------------------------------------------------------------------
# A small, fixed toolset. Deliberately contains near-neighbours (`read_file` vs
# `search_files` vs `list_directory`) because "picked a plausible sibling" is
# the characteristic tool-selection error of a degraded quant, and a toolset of
# obviously-distinct tools cannot detect it.
# ---------------------------------------------------------------------------

def _fn(name: str, description: str, properties: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


TOOLS: Tuple[Dict[str, Any], ...] = (
    _fn("read_file", "Return the full text of one file at an exact path.",
        {"path": {"type": "string", "description": "Repository-relative file path."}},
        ["path"]),
    _fn("search_files", "Search file contents for a regular expression and return matching lines.",
        {"pattern": {"type": "string", "description": "Regular expression."},
         "glob": {"type": "string", "description": "Optional filename filter, e.g. '*.py'."}},
        ["pattern"]),
    _fn("list_directory", "List the entries of one directory. Does not read file contents.",
        {"path": {"type": "string"}}, ["path"]),
    _fn("get_weather", "Current weather for a city.",
        {"city": {"type": "string"},
         "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
        ["city"]),
    _fn("get_stock_price", "Latest trade price for a ticker symbol.",
        {"symbol": {"type": "string", "description": "Ticker symbol, uppercase."}},
        ["symbol"]),
    _fn("convert_currency", "Convert an amount between two ISO-4217 currencies.",
        {"amount": {"type": "number"},
         "from_currency": {"type": "string"},
         "to_currency": {"type": "string"}},
        ["amount", "from_currency", "to_currency"]),
    _fn("run_tests", "Run the test suite, optionally filtered to one node id.",
        {"node_id": {"type": "string", "description": "Optional pytest node id."}},
        []),
    _fn("get_historical_price",
        "Closing price for a ticker on a specific past date. Not for the current price.",
        {"symbol": {"type": "string"}, "date": {"type": "string", "description": "YYYY-MM-DD."}},
        ["symbol", "date"]),
    _fn("get_forecast",
        "Multi-day weather forecast for a city. Not for current conditions.",
        {"city": {"type": "string"}, "days": {"type": "integer", "description": "1-10."}},
        ["city", "days"]),
    _fn("batch_convert_currency",
        "Convert several amounts from one currency into another in a single call.",
        {"amounts": {"type": "array", "items": {"type": "number"}},
         "from_currency": {"type": "string"}, "to_currency": {"type": "string"}},
        ["amounts", "from_currency", "to_currency"]),
)

SYSTEM = (
    "You are a precise assistant with access to tools. Call a tool when it is the "
    "only way to get information you do not already have. Answer directly when you "
    "already have what you need. Never invent tool results."
)


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------

def _case(case_id: str, group: str, prompt: str, expect: Dict[str, Any],
          messages: Tuple[Dict[str, Any], ...] = (), max_tokens: int = 512) -> Case:
    return Case(
        case_id=case_id,
        prompt=prompt,
        system=SYSTEM,
        max_tokens=max_tokens,
        temperature=0.0,
        tools=TOOLS,
        messages=messages,
        meta={"group": group, "category": "tool_use", **expect},
    )


def _assistant_call(call_id: str, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }],
    }


def build_cases(seed: int = 20260918, limit: int = 0, **_ignored: object) -> List[Case]:
    cases: List[Case] = []

    # -- emit: an unambiguous tool request ---------------------------------
    cases.append(_case(
        "tc_emit_weather", "emit",
        "What is the current temperature in Reykjavik? Use the tools available.",
        {"expect_tools": ["get_weather"], "expect_call": True}))
    cases.append(_case(
        "tc_emit_stock", "emit",
        "Look up the latest trade price for the ticker NVDA.",
        {"expect_tools": ["get_stock_price"], "expect_call": True}))

    # -- select: plausible neighbours --------------------------------------
    cases.append(_case(
        "tc_select_search_not_read", "select",
        "Somewhere in this repository a function named `resolve_backend` is defined, "
        "but I do not know which file it is in. Find it.",
        {"expect_tools": ["search_files"], "expect_call": True,
         "distractors": ["read_file", "list_directory"]}))
    cases.append(_case(
        "tc_select_read_not_search", "select",
        "Show me the complete contents of scripts/model_files.py.",
        {"expect_tools": ["read_file"], "expect_call": True,
         "distractors": ["search_files", "list_directory"]}))
    cases.append(_case(
        "tc_select_list_not_read", "select",
        "What files are in the configs/ directory? I only want the names.",
        {"expect_tools": ["list_directory"], "expect_call": True,
         "distractors": ["read_file", "search_files"]}))

    # -- arguments: required names, types, valid JSON ----------------------
    cases.append(_case(
        "tc_args_currency", "arguments",
        "Convert 1499.50 US dollars to Japanese yen.",
        {"expect_tools": ["convert_currency"], "expect_call": True,
         "expect_args": {"amount": 1499.5, "from_currency": "USD", "to_currency": "JPY"},
         "required_args": ["amount", "from_currency", "to_currency"],
         "numeric_args": ["amount"]}))
    cases.append(_case(
        "tc_args_enum", "arguments",
        "Give me the weather in Osaka in fahrenheit.",
        {"expect_tools": ["get_weather"], "expect_call": True,
         "expect_args": {"city": "Osaka", "units": "fahrenheit"},
         "required_args": ["city"],
         "enum_args": {"units": ["celsius", "fahrenheit"]}}))
    cases.append(_case(
        "tc_args_optional_omitted", "arguments",
        "Run the whole test suite. Do not filter it to a single test.",
        {"expect_tools": ["run_tests"], "expect_call": True,
         "forbid_args": ["node_id"]}))

    # -- parallel: two independent lookups ---------------------------------
    cases.append(_case(
        "tc_parallel_two_cities", "parallel",
        "I need the current weather in both Lisbon and Nagoya.",
        {"expect_tools": ["get_weather"], "expect_call": True, "expect_min_calls": 2,
         "expect_arg_values": {"city": ["Lisbon", "Nagoya"]}}))
    cases.append(_case(
        "tc_parallel_mixed", "parallel",
        "Tell me the stock price of AMD and the weather in Kigali.",
        {"expect_tools": ["get_stock_price", "get_weather"], "expect_call": True,
         "expect_min_calls": 2, "expect_tool_set": ["get_stock_price", "get_weather"]}))

    # -- interpret: the result is already in the history --------------------
    cases.append(_case(
        "tc_interpret_result", "interpret",
        "",
        {"expect_call": False, "expect_substrings": ["17"], "forbid_tools": ["get_weather"]},
        messages=(
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "What is the temperature in Valparaiso, and is that above 15 degrees?"},
            _assistant_call("call_1", "get_weather", {"city": "Valparaiso", "units": "celsius"}),
            {"role": "tool", "tool_call_id": "call_1", "name": "get_weather",
             "content": json.dumps({"city": "Valparaiso", "temperature_c": 17, "conditions": "overcast"})},
        )))
    cases.append(_case(
        "tc_interpret_arithmetic", "interpret",
        "",
        {"expect_call": False, "expect_substrings": ["224925", "224,925"], "substring_mode": "any",
         "forbid_tools": ["convert_currency"]},
        messages=(
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Convert 1499.50 USD to JPY, then tell me the result rounded to the nearest whole yen."},
            _assistant_call("call_1", "convert_currency",
                            {"amount": 1499.5, "from_currency": "USD", "to_currency": "JPY"}),
            {"role": "tool", "tool_call_id": "call_1", "name": "convert_currency",
             "content": json.dumps({"amount": 1499.5, "rate": 150.0, "converted": 224925.0,
                                    "from": "USD", "to": "JPY"})},
        )))

    # -- recover: the tool failed ------------------------------------------
    cases.append(_case(
        "tc_recover_bad_path", "recover",
        "",
        {"expect_call": True, "forbid_identical_retry": {"name": "read_file", "args": {"path": "src/uttils.py"}},
         "expect_tools": ["search_files", "list_directory", "read_file"]},
        messages=(
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Read the helper module src/uttils.py and tell me what it exports."},
            _assistant_call("call_1", "read_file", {"path": "src/uttils.py"}),
            {"role": "tool", "tool_call_id": "call_1", "name": "read_file",
             "content": json.dumps({"error": "ENOENT: no such file or directory: 'src/uttils.py'"})},
        )))
    cases.append(_case(
        "tc_recover_empty_result", "recover",
        "",
        {"expect_call": True, "expect_tools": ["search_files", "list_directory", "read_file"],
         "forbid_identical_retry": {"name": "search_files", "args": {"pattern": "resolve_backend"}}},
        messages=(
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Find where resolve_backend is defined."},
            _assistant_call("call_1", "search_files", {"pattern": "resolve_backend"}),
            {"role": "tool", "tool_call_id": "call_1", "name": "search_files",
             "content": json.dumps({"matches": [], "note": "0 matches"})},
        )))

    # -- restraint: no tool is needed --------------------------------------
    cases.append(_case(
        "tc_restraint_arithmetic", "restraint",
        "What is 47 multiplied by 12? Answer with the number.",
        {"expect_call": False, "expect_substrings": ["564"]}))
    cases.append(_case(
        "tc_restraint_definition", "restraint",
        "In one sentence, what does the acronym 'KV cache' refer to in transformer inference?",
        {"expect_call": False, "min_chars": 20}))
    cases.append(_case(
        "tc_restraint_given_context", "restraint",
        "Here is the file's content:\n\n```python\nDEFAULT_PORT = 8085\nDEFAULT_HOST = '127.0.0.1'\n```\n\n"
        "What is DEFAULT_PORT set to? Answer with the number only.",
        {"expect_call": False, "expect_substrings": ["8085"]}))


    # -- hard: where a degraded quant plausibly diverges from a healthy one ---
    # Each case has a near-neighbour tool that is wrong for a specific, checkable
    # reason, or an argument that must be derived rather than copied. The three
    # extra tools above also raise the difficulty of the `select` cases, which a
    # healthy 27B answered 3/3 on a seven-tool set.
    cases.append(_case(
        "tc_hard_current_not_historical", "hard",
        "What is NVDA trading at right now?",
        {"expect_tools": ["get_stock_price"], "expect_call": True,
         "forbid_tools": ["get_historical_price"]}))
    cases.append(_case(
        "tc_hard_forecast_not_current", "hard",
        "Will it rain in Lisbon at any point over the next three days?",
        {"expect_tools": ["get_forecast"], "expect_call": True,
         "expect_args": {"city": "Lisbon", "days": 3},
         "numeric_args": ["days"], "forbid_tools": ["get_weather"]}))
    cases.append(_case(
        "tc_hard_batch_not_repeated", "hard",
        "Convert 120, 340.50 and 89 euros into US dollars. Do it in one call, not three.",
        {"expect_tools": ["batch_convert_currency"], "expect_call": True,
         "expect_max_calls": 1, "forbid_tools": ["convert_currency"],
         "expect_args": {"from_currency": "EUR", "to_currency": "USD"},
         "array_args": {"amounts": [120, 340.5, 89]}}))
    cases.append(_case(
        "tc_hard_derived_argument", "hard",
        "A shipment weighs 145 pounds. Our carrier quotes in kilograms, rounded to one "
        "decimal place. Convert that weight yourself (1 lb = 0.45359237 kg) and report it. "
        "No tool here performs weight conversion.",
        {"expect_call": False, "expect_substrings": ["65.8"]}))
    cases.append(_case(
        "tc_hard_stale_result", "hard", "",
        {"expect_call": True, "expect_tools": ["get_weather"],
         "expect_arg_values": {"city": ["Tampere"]}},
        messages=(
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "What is the weather in Nagoya?"},
            _assistant_call("call_1", "get_weather", {"city": "Nagoya", "units": "celsius"}),
            {"role": "tool", "tool_call_id": "call_1", "name": "get_weather",
             "content": json.dumps({"city": "Nagoya", "temperature_c": 21, "conditions": "clear"})},
            {"role": "assistant", "content": "Nagoya is 21 degrees and clear."},
            {"role": "user", "content": "And in Tampere?"},
        )))
    cases.append(_case(
        "tc_hard_contradicted_premise", "hard", "",
        {"expect_call": False, "expect_substrings": ["not", "no", "does not exist", "missing"],
         "substring_mode": "any", "forbid_tools": ["read_file"]},
        messages=(
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Read infra/deploy_target.py and tell me which region it deploys to."},
            _assistant_call("call_1", "read_file", {"path": "infra/deploy_target.py"}),
            {"role": "tool", "tool_call_id": "call_1", "name": "read_file",
             "content": json.dumps({"error": "ENOENT: no such file or directory: 'infra/deploy_target.py'",
                                    "note": "the repository contains no deploy script"})},
            {"role": "user", "content": "So which region is it? Answer in one sentence."},
        )))

    return cases[:limit] if limit else cases


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def _names(response: Response) -> List[str]:
    return [c["name"] for c in response.tool_calls]


def score_response(case: Case, response: Response) -> Score:
    """Blunt pass/fail against the one behaviour this case isolates."""
    meta = case.meta
    calls = response.tool_calls
    names = _names(response)

    # Invalid JSON arguments fail everything else that could be said about the
    # call, so it is checked first and reported by name.
    if response.arguments_unparseable:
        return Score.binary(False, f"{response.arguments_unparseable} tool call(s) had unparseable JSON arguments")

    if meta.get("expect_call") is False:
        forbidden = set(meta.get("forbid_tools") or [])
        if calls and (not forbidden or any(n in forbidden for n in names)):
            return Score.binary(False, f"called {names} when no tool was needed")
        if calls:
            return Score.binary(False, f"called {names}; the answer was already available")
        return _score_text(case, response.text)

    if not calls:
        return Score.binary(False, f"no tool call emitted (finish_reason={response.finish_reason})")

    expected = set(meta.get("expect_tools") or [])
    if expected and not set(names) & expected:
        return Score.binary(False, f"called {names}, expected one of {sorted(expected)}")

    if meta.get("expect_tool_set"):
        want = set(meta["expect_tool_set"])
        if not want.issubset(set(names)):
            return Score.binary(False, f"called {sorted(set(names))}, needed all of {sorted(want)}")

    minimum = int(meta.get("expect_min_calls") or 0)
    if minimum and len(calls) < minimum:
        return Score.binary(False, f"{len(calls)} call(s), expected at least {minimum}")

    maximum = int(meta.get("expect_max_calls") or 0)
    if maximum and len(calls) > maximum:
        return Score.binary(False, f"{len(calls)} call(s), expected at most {maximum} "
                                   f"— did not use the batching tool as instructed")

    forbidden = set(meta.get("forbid_tools") or [])
    if forbidden & set(names):
        return Score.binary(False, f"called {sorted(forbidden & set(names))}, "
                                   f"which is the wrong tool for this request")

    retry = meta.get("forbid_identical_retry")
    if retry:
        for call in calls:
            if call["name"] == retry["name"] and call["arguments"] == retry["args"]:
                return Score.binary(False, f"repeated the failed call verbatim: {retry['name']}({retry['args']})")

    chosen = next((c for c in calls if c["name"] in expected), calls[0]) if expected else calls[0]
    arguments = chosen["arguments"]
    if arguments is None:
        return Score.binary(False, "tool arguments did not decode as JSON")

    for key in meta.get("required_args") or []:
        if key not in arguments:
            return Score.binary(False, f"missing required argument '{key}' (got {sorted(arguments)})")
    for key in meta.get("forbid_args") or []:
        if key in arguments and arguments[key] not in (None, ""):
            return Score.binary(False, f"sent optional argument '{key}' when told not to")
    for key in meta.get("numeric_args") or []:
        if key in arguments and not isinstance(arguments[key], (int, float)):
            return Score.binary(False, f"argument '{key}' is {type(arguments[key]).__name__}, expected a number")
    for key, allowed in (meta.get("enum_args") or {}).items():
        if key in arguments and arguments[key] not in allowed:
            return Score.binary(False, f"argument '{key}'={arguments[key]!r} is outside {allowed}")

    for key, want in (meta.get("expect_args") or {}).items():
        if key not in arguments:
            continue
        got = arguments[key]
        if isinstance(want, (int, float)) and isinstance(got, (int, float)):
            if abs(float(got) - float(want)) > 1e-6:
                return Score.binary(False, f"argument '{key}'={got} expected {want}")
        elif str(got).strip().lower() != str(want).strip().lower():
            return Score.binary(False, f"argument '{key}'={got!r} expected {want!r}")

    for key, wanted_list in (meta.get("array_args") or {}).items():
        got = arguments.get(key)
        if not isinstance(got, list):
            return Score.binary(False, f"argument '{key}' is {type(got).__name__}, expected an array")
        if len(got) != len(wanted_list) or any(
            not isinstance(g, (int, float)) or abs(float(g) - float(w)) > 1e-6
            for g, w in zip(got, wanted_list)
        ):
            return Score.binary(False, f"argument '{key}'={got} expected {wanted_list}")

    for key, values in (meta.get("expect_arg_values") or {}).items():
        seen = {str((c["arguments"] or {}).get(key, "")).strip().lower() for c in calls}
        for value in values:
            if value.strip().lower() not in seen:
                return Score.binary(False, f"no call carried {key}={value!r} (saw {sorted(seen)})")

    return Score.binary(True, f"{len(calls)} call(s): {names}")


def _score_text(case: Case, text: str) -> Score:
    wanted = case.meta.get("expect_substrings") or []
    if wanted:
        normalized = text.replace(",", "").lower()
        hits = [w for w in wanted if w.replace(",", "").lower() in normalized]
        mode = case.meta.get("substring_mode", "all")
        passed = bool(hits) if mode == "any" else len(hits) == len(wanted)
        return Score.binary(passed, f"answered without a tool; matched {hits or 'nothing'}")
    minimum = int(case.meta.get("min_chars") or 0)
    if minimum:
        return Score.binary(len(text.strip()) >= minimum,
                            f"answered without a tool ({len(text.strip())} chars)")
    return Score.binary(True, "answered without a tool")


def score(case: Case, response_text: str) -> Score:
    """Text-only fallback for callers that never saw the Response.

    Only the `restraint` and `interpret` groups are decidable from text alone;
    everything else is flagged rather than guessed, because scoring a tool-call
    case from its prose would quietly reward a model for narrating a call it
    never made.
    """
    if case.meta.get("expect_call") is False:
        return _score_text(case, response_text)
    return Score(value=0.0, passed=False, detail="needs_response_scoring")


def summarize(rows: List[Dict[str, object]]) -> Dict[str, object]:
    """Per-group pass rate. Groups fail for different reasons and a single
    averaged number over them would hide which capability actually moved."""
    by_group: Dict[str, List[int]] = {}
    for row in rows:
        by_group.setdefault(str(row.get("group") or "?"), []).append(int(row.get("passed") or 0))
    return {
        "by_group": {
            group: {"passed": sum(v), "of": len(v)}
            for group, v in sorted(by_group.items())
        }
    }
