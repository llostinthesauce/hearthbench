#!/usr/bin/env python3
"""Validate benchmark methodology profiles and analyze blind pairwise records."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import random
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROFILES = ROOT / "configs" / "benchmark_profiles.json"
REFERENCE_VOCAB_SHA256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"

REQUIRED_METRICS = {
    "time_to_first_token_s",
    "time_to_first_answer_s",
    "prompt_tokens_per_second",
    "output_tokens_per_second",
    "batch_tokens_per_second",
    "end_to_end_s",
    "peak_memory_bytes",
    "cache_state",
}


def load_profiles(path: Path = DEFAULT_PROFILES) -> dict:
    value = json.loads(path.read_text())
    if value.get("schema_version") != 1 or not isinstance(value.get("profiles"), dict):
        raise ValueError("benchmark profile file must use schema_version 1")
    return value


def get_profile(document: dict, name: str) -> dict:
    try:
        profile = document["profiles"][name]
    except KeyError as exc:
        names = ", ".join(sorted(document.get("profiles", {})))
        raise ValueError(f"unknown benchmark profile {name!r}; available: {names}") from exc
    errors = validate_profile(profile)
    if errors:
        raise ValueError("invalid benchmark profile: " + "; ".join(errors))
    return profile


def validate_profile(profile: dict) -> list[str]:
    errors: list[str] = []
    if profile.get("leaderboard_comparable") is not False:
        errors.append("leaderboard_comparable must be false for local methodology profiles")
    workloads = profile.get("workloads")
    if not isinstance(workloads, list) or not workloads:
        errors.append("workloads must be a non-empty list")
    else:
        for workload in workloads:
            if not isinstance(workload.get("input_tokens"), int) or workload["input_tokens"] <= 0:
                errors.append("each workload needs a positive integer input_tokens")
            if not isinstance(workload.get("min_output_tokens"), int) or workload["min_output_tokens"] <= 0:
                errors.append("each workload needs a positive integer min_output_tokens")
            cap = workload.get("max_output_tokens")
            if cap is not None and (not isinstance(cap, int) or cap < workload.get("min_output_tokens", 1)):
                errors.append("max_output_tokens must be an integer at least min_output_tokens")
    modes = profile.get("modes")
    if not isinstance(modes, list) or not modes or any(
        not isinstance(mode.get("concurrency"), int) or mode["concurrency"] <= 0
        for mode in modes or []
    ):
        errors.append("modes must contain positive integer concurrency values")
    cache_states = profile.get("cache_states")
    if not isinstance(cache_states, list) or not cache_states or not set(cache_states or []) <= {"cold", "warm"}:
        errors.append("cache_states may contain only cold and warm")
    missing_metrics = REQUIRED_METRICS - set(profile.get("metrics") or [])
    if missing_metrics:
        errors.append("missing metrics: " + ", ".join(sorted(missing_metrics)))
    aggregation = profile.get("aggregation") or {}
    if not isinstance(aggregation.get("min_samples"), int) or aggregation["min_samples"] < 1:
        errors.append("aggregation.min_samples must be a positive integer")
    if aggregation.get("primary") != "median" or "p95" not in aggregation.get("tail", []):
        errors.append("aggregation must use median with a p95 tail")
    pairwise = profile.get("pairwise") or {}
    required_pairwise = {
        "blind": True,
        "randomize_order": True,
        "swap_order": True,
        "scoring": "bradley_terry",
    }
    for key, expected in required_pairwise.items():
        if pairwise.get(key) != expected:
            errors.append(f"pairwise.{key} must be {expected!r}")
    if pairwise.get("judge") not in {"human", "local", "human_or_local"}:
        errors.append("pairwise.judge must remain human or local")
    return errors


def validate_results(profile: dict, records: list[dict]) -> list[str]:
    """Validate whether result records actually satisfy a profile contract.

    Missing backend measurements must be explicit ``null`` values and named in
    ``unsupported_metrics``. The validator checks coverage and measured output;
    it cannot prove that an operator really restarted a server for a row labeled
    ``cold``.
    """
    errors: list[str] = []
    workloads = {item["id"]: item for item in profile["workloads"]}
    modes = {item["id"]: item for item in profile["modes"]}
    cache_states = set(profile["cache_states"])
    metrics = set(profile["metrics"])
    batches: dict[tuple, set[int]] = {}
    seen: set[tuple] = set()
    provenance: set[tuple] = set()
    batch_values: dict[tuple, tuple] = {}

    for index, record in enumerate(records):
        prefix = f"row {index + 1}"
        workload_id = record.get("workload_id")
        mode_id = record.get("mode_id")
        cache_state = record.get("cache_state")
        cell = (str(workload_id), str(mode_id), str(cache_state))
        valid = True
        if (
            record.get("schema_version") != 1
            or record.get("profile_id") != profile["id"]
            or record.get("token_count_method") != "tiktoken_o200k_base"
            or record.get("tokenizer_sha256") != REFERENCE_VOCAB_SHA256
            or not all(isinstance(record.get(key), str) and record[key] for key in ("run_id", "model", "url"))
        ):
            errors.append(f"{prefix}: missing or incompatible run/model/profile/tokenizer provenance")
            valid = False
        provenance.add(tuple(record.get(key) for key in ("run_id", "model", "url", "profile_id", "temperature", "thinking")))
        endpoint = urlparse(str(record.get("url", "")))
        if endpoint.scheme != "http" or endpoint.hostname not in {"localhost", "127.0.0.1", "::1"} or endpoint.username:
            errors.append(f"{prefix}: endpoint must be HTTP loopback")
            valid = False
        if not isinstance(record.get("temperature"), (int, float)) or record.get("top_p") != 1 or record.get("thinking") not in {"server_default", "disabled_requested"}:
            errors.append(f"{prefix}: missing sampling configuration")
            valid = False
        if workload_id not in workloads or mode_id not in modes or cache_state not in cache_states:
            errors.append(f"{prefix}: unexpected profile cell {cell}")
            continue
        workload = workloads[workload_id]
        mode = modes[mode_id]
        required_values = {
            "concurrency": mode["concurrency"],
            "target_input_tokens": workload["input_tokens"],
            "target_min_output_tokens": workload["min_output_tokens"],
            "max_output_tokens": workload.get("max_output_tokens", workload["min_output_tokens"] * 3 + 256),
        }
        for key, expected in required_values.items():
            if record.get(key) != expected:
                errors.append(f"{prefix}: {key} must be {expected}")
                valid = False
        if record.get("status") != "ok":
            errors.append(f"{prefix}: status is not ok")
            valid = False
        if not isinstance(record.get("prompt_tokens"), int) or not workload["input_tokens"] * 0.9 <= record["prompt_tokens"] <= workload["input_tokens"] * 1.1:
            errors.append(f"{prefix}: input outside target tolerance")
            valid = False
        if not isinstance(record.get("output_tokens"), int) or record["output_tokens"] < workload["min_output_tokens"]:
            errors.append(f"{prefix}: underfilled output")
            valid = False
        unsupported = set(record.get("unsupported_metrics") or [])
        for metric in metrics - {"cache_state"}:
            if metric not in record:
                errors.append(f"{prefix}: missing metric {metric}")
                valid = False
            elif record[metric] is None and metric not in unsupported:
                errors.append(f"{prefix}: {metric} is null but not declared unsupported")
                valid = False
            elif record[metric] is not None and (
                isinstance(record[metric], bool) or not isinstance(record[metric], (int, float))
                or not math.isfinite(record[metric]) or record[metric] < 0
            ):
                errors.append(f"{prefix}: {metric} must be finite and nonnegative, or null")
                valid = False
        sample_id = record.get("sample_id")
        request_index = record.get("request_index", 0)
        identity = (*cell, sample_id, request_index)
        batch = (*cell, sample_id)
        value = (record.get("batch_tokens_per_second"), record.get("batch_elapsed_s"))
        if not isinstance(value[1], (int, float)) or not math.isfinite(value[1]) or value[1] <= 0:
            errors.append(f"{prefix}: batch_elapsed_s must be positive and finite")
            valid = False
        if batch in batch_values and value != batch_values[batch]:
            errors.append(f"{prefix}: inconsistent batch throughput or elapsed time")
            valid = False
        batch_values[batch] = value
        if not isinstance(sample_id, int) or not isinstance(request_index, int) or not 0 <= request_index < mode["concurrency"]:
            errors.append(f"{prefix}: invalid sample_id or request_index")
            valid = False
        if identity in seen:
            errors.append(f"{prefix}: duplicate sample/request identity")
            valid = False
        seen.add(identity)
        if valid:
            batches.setdefault((*cell, sample_id), set()).add(request_index)

    if len(provenance) > 1:
        errors.append("Result set mixes runs, models, endpoints, or profiles")
    minimum = int(profile["aggregation"]["min_samples"])
    counts: Counter[tuple] = Counter()
    for batch, indices in batches.items():
        cell = batch[:3]
        if len(indices) == modes[cell[1]]["concurrency"]:
            counts[cell] += 1
    for workload_id in workloads:
        for mode_id in modes:
            for cache_state in profile["cache_states"]:
                cell = (workload_id, mode_id, cache_state)
                if counts[cell] < minimum:
                    errors.append(
                        f"profile cell {cell} needs {minimum} successful samples; found {counts[cell]}"
                    )
    return errors


def load_result_records(path: Path) -> list[dict]:
    """Load either a JSON array or newline-delimited JSON result file."""
    text = path.read_text()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError("result file must contain a JSON array or JSONL objects")
    return value


def build_pairwise_records(
    model_ids: list[str], prompt_ids: list[str], *, seed: int = 0
) -> list[dict]:
    """Return blinded, randomized A/B templates with a reverse-order control."""
    if len(model_ids) != 2 or len(set(model_ids)) != 2:
        raise ValueError("pairwise records require exactly two distinct models")
    rng = random.Random(seed)
    records: list[dict] = []
    for prompt_id in prompt_ids:
        ordered = list(model_ids)
        rng.shuffle(ordered)
        for order_index, pair in enumerate((ordered, list(reversed(ordered)))):
            records.append({
                "prompt_id": prompt_id,
                "order_index": order_index,
                "candidates": ["A", "B"],
                "model_by_candidate": {"A": pair[0], "B": pair[1]},
                "response_by_candidate": {"A": None, "B": None},
                "winner": None,
                "judge": "human_or_local",
            })
    return records


def bradley_terry_scores(records: list[dict], *, iterations: int = 200) -> dict[str, float]:
    """Fit simple Bradley-Terry strengths from completed A/B records.

    Ties (winner ``tie``) contribute half a win to each model. Unjudged records
    are ignored. Scores are normalized to sum to one.
    """
    completed = [record for record in records if record.get("winner") in {"A", "B", "tie"}]
    models = sorted({
        model
        for record in completed
        for model in record.get("model_by_candidate", {}).values()
    })
    if not models:
        return {}
    wins = {model: 0.0 for model in models}
    contests = {(a, b): 0 for a in models for b in models if a != b}
    for record in completed:
        mapping = record.get("model_by_candidate", {})
        if set(mapping) != {"A", "B"} or record.get("winner") not in {"A", "B", "tie"}:
            continue
        a, b = mapping["A"], mapping["B"]
        contests[(a, b)] += 1
        contests[(b, a)] += 1
        if record["winner"] == "tie":
            wins[a] += 0.5
            wins[b] += 0.5
        else:
            wins[mapping[record["winner"]]] += 1.0
    strengths = {model: 1.0 for model in models}
    for _ in range(iterations):
        updated: dict[str, float] = {}
        for model in models:
            denominator = sum(
                contests[(model, other)] / max(strengths[model] + strengths[other], 1e-12)
                for other in models if other != model
            )
            updated[model] = max(wins[model] / denominator, 1e-9) if denominator else strengths[model]
        scale = math.exp(sum(math.log(value) for value in updated.values()) / len(updated))
        strengths = {model: value / scale for model, value in updated.items()}
    total = sum(strengths.values())
    return {model: strengths[model] / total for model in models}


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate a quantile from no values")
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_bradley_terry(
    records: list[dict], *, samples: int = 1000, seed: int = 0
) -> dict[str, dict[str, float]]:
    """Return seeded 95% bootstrap intervals around Bradley-Terry scores."""
    completed = [r for r in records if r.get("winner") in {"A", "B", "tie"}]
    if not completed:
        return {}
    if samples <= 0:
        raise ValueError("samples must be positive")
    point = bradley_terry_scores(completed)
    draws = {model: [] for model in point}
    rng = random.Random(seed)
    grouped: dict[str, list[dict]] = {}
    for index, record in enumerate(completed):
        prompt_id = str(record.get("prompt_id", f"__row_{index}"))
        grouped.setdefault(prompt_id, []).append(record)
    prompt_groups = list(grouped.values())
    for _ in range(samples):
        resample = [
            record
            for _ in prompt_groups
            for record in rng.choice(prompt_groups)
        ]
        scores = bradley_terry_scores(resample)
        for model in draws:
            draws[model].append(scores.get(model, 0.0))
    return {
        model: {
            "score": score,
            "low": _quantile(draws[model], 0.025),
            "high": _quantile(draws[model], 0.975),
        }
        for model, score in point.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect versioned local benchmark profiles")
    parser.add_argument("--config", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--profile", default="aa_local_full")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--validate-results", type=Path, metavar="PATH")
    args = parser.parse_args()
    document = load_profiles(args.config)
    profile = get_profile(document, args.profile)
    if args.validate_results:
        errors = validate_results(profile, load_result_records(args.validate_results))
        if errors:
            print("Result set does not satisfy the profile:", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
            raise SystemExit(1)
        print(f"Result set satisfies {args.profile}: {args.validate_results}")
        return
    if args.json:
        print(json.dumps(profile, indent=2))
        return
    print(f"{args.profile}: {profile['description']}")
    for workload in profile["workloads"]:
        print(
            f"  {workload['id']:<14} input={workload['input_tokens']:<7} "
            f"minimum_output={workload['min_output_tokens']}"
        )
    modes = ", ".join(f"{m['id']} ({m['concurrency']}x)" for m in profile["modes"])
    print(f"  modes: {modes}")
    print("  result scope: local profile; validate collected results before claiming conformance")


if __name__ == "__main__":
    main()
