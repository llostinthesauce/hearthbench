"""Versioned benchmark profiles and local blind-pairwise helpers."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import benchmark_profiles as bp


def _profiles() -> dict:
    return json.loads((ROOT / "configs" / "benchmark_profiles.json").read_text())


def test_default_profile_has_fixed_aa_shaped_workloads():
    profile = bp.get_profile(_profiles(), "aa_local_full")

    assert [(w["input_tokens"], w["min_output_tokens"]) for w in profile["workloads"]] == [
        (1_000, 1_000), (10_000, 1_500), (100_000, 2_000)
    ]
    assert {mode["concurrency"] for mode in profile["modes"]} == {1, 10}
    assert profile["cache_states"] == ["cold", "warm"]


def test_output_cap_cannot_be_smaller_than_required_answer():
    profile = bp.get_profile(_profiles(), "aa_local_smoke")
    profile["workloads"][0]["max_output_tokens"] = 1
    assert any("max_output_tokens" in error for error in bp.validate_profile(profile))


def test_profile_requires_latency_throughput_memory_and_cache_metrics():
    profile = bp.get_profile(_profiles(), "aa_local_full")
    required = {
        "time_to_first_token_s", "time_to_first_answer_s",
        "prompt_tokens_per_second", "output_tokens_per_second",
        "end_to_end_s", "peak_memory_bytes", "cache_state",
    }

    assert required <= set(profile["metrics"])
    assert profile["aggregation"]["primary"] == "median"
    assert "p95" in profile["aggregation"]["tail"]


def test_smoke_profile_is_reduced_and_not_leaderboard_comparable():
    profile = bp.get_profile(_profiles(), "aa_local_smoke")

    assert profile["reduced"] is True
    assert profile["leaderboard_comparable"] is False
    assert max(w["input_tokens"] for w in profile["workloads"]) < 100_000


def test_validator_rejects_unsafe_or_misleading_profiles():
    profile = bp.get_profile(_profiles(), "aa_local_full")
    broken = json.loads(json.dumps(profile))
    broken["leaderboard_comparable"] = True

    errors = bp.validate_profile(broken)

    assert any("leaderboard_comparable" in error for error in errors)


def test_pairwise_records_are_blind_randomized_and_order_swapped():
    records = bp.build_pairwise_records(
        ["model-a", "model-b"], ["prompt-1", "prompt-2"], seed=17
    )

    assert len(records) == 4
    assert all(set(r["candidates"]) == {"A", "B"} for r in records)
    for prompt_id in ("prompt-1", "prompt-2"):
        rows = [r for r in records if r["prompt_id"] == prompt_id]
        assert rows[0]["model_by_candidate"] == {
            "A": rows[1]["model_by_candidate"]["B"],
            "B": rows[1]["model_by_candidate"]["A"],
        }
        assert all(r["winner"] is None for r in rows)


def test_bradley_terry_hook_ranks_repeated_winner_first():
    records = [
        {"model_by_candidate": {"A": "alpha", "B": "beta"}, "winner": "A"},
        {"model_by_candidate": {"A": "beta", "B": "alpha"}, "winner": "B"},
        {"model_by_candidate": {"A": "alpha", "B": "gamma"}, "winner": "A"},
        {"model_by_candidate": {"A": "gamma", "B": "beta"}, "winner": "A"},
    ]

    scores = bp.bradley_terry_scores(records)

    assert scores["alpha"] > scores["gamma"] > scores["beta"]


def test_bootstrap_intervals_are_seeded_and_bound_point_scores():
    records = [
        {"model_by_candidate": {"A": "alpha", "B": "beta"}, "winner": "A"},
        {"model_by_candidate": {"A": "beta", "B": "alpha"}, "winner": "B"},
        {"model_by_candidate": {"A": "alpha", "B": "beta"}, "winner": "A"},
    ]

    first = bp.bootstrap_bradley_terry(records, samples=40, seed=9)
    second = bp.bootstrap_bradley_terry(records, samples=40, seed=9)

    assert first == second
    assert first["alpha"]["low"] <= first["alpha"]["high"]


def test_unjudged_pairwise_records_do_not_create_fake_even_scores():
    records = bp.build_pairwise_records(["a", "b"], ["prompt"])

    assert bp.bradley_terry_scores(records) == {}


def test_bootstrap_resamples_prompt_pairs_together(monkeypatch):
    records = []
    for prompt_id, winner in (("p1", "A"), ("p2", "B")):
        for order_index in (0, 1):
            records.append({
                "prompt_id": prompt_id,
                "order_index": order_index,
                "model_by_candidate": {"A": "a", "B": "b"},
                "winner": winner,
            })
    observed = []
    original = bp.bradley_terry_scores

    def capture(sample, **kwargs):
        observed.append([row["prompt_id"] for row in sample])
        return original(sample, **kwargs)

    monkeypatch.setattr(bp, "bradley_terry_scores", capture)
    bp.bootstrap_bradley_terry(records, samples=4, seed=3)

    for draw in observed[1:]:
        assert len(draw) == 4
        assert all(draw.count(prompt_id) % 2 == 0 for prompt_id in set(draw))


def _profile_result_rows(profile):
    rows = []
    samples = profile["aggregation"]["min_samples"]
    for workload in profile["workloads"]:
        for mode in profile["modes"]:
            for cache_state in profile["cache_states"]:
                for sample in range(samples):
                    rows.append({
                        "schema_version": 1,
                        "profile_id": profile["id"],
                        "run_id": "fixture-run",
                        "model": "fixture-model",
                        "url": "http://127.0.0.1:8000/v1",
                        "token_count_method": "tiktoken_o200k_base",
                        "tokenizer_sha256": bp.REFERENCE_VOCAB_SHA256,
                        "temperature": 0.6,
                        "top_p": 1,
                        "thinking": "server_default",
                        "request_timeout_s": 1800,
                        "max_output_tokens": workload["min_output_tokens"] * 3 + 256,
                        "workload_id": workload["id"],
                        "mode_id": mode["id"],
                        "concurrency": mode["concurrency"],
                        "cache_state": cache_state,
                        "sample_id": sample,
                        "status": "ok",
                        "target_input_tokens": workload["input_tokens"],
                        "prompt_tokens": workload["input_tokens"],
                        "target_min_output_tokens": workload["min_output_tokens"],
                        "output_tokens": workload["min_output_tokens"],
                        "time_to_first_token_s": 0.1,
                        "time_to_first_answer_s": 0.2,
                        "prompt_tokens_per_second": None,
                        "output_tokens_per_second": 20.0,
                        "batch_tokens_per_second": 20.0,
                        "batch_elapsed_s": 1.0,
                        "end_to_end_s": 1.0,
                        "peak_memory_bytes": None,
                        "unsupported_metrics": [
                            "prompt_tokens_per_second", "peak_memory_bytes",
                        ],
                    })
    return rows


def test_result_validator_requires_every_profile_cell_and_real_sample_count():
    profile = bp.get_profile(_profiles(), "aa_local_smoke")
    rows = _profile_result_rows(profile)

    assert bp.validate_results(profile, rows) == []

    errors = bp.validate_results(profile, rows[:-1])
    assert any("needs 2 successful samples" in error for error in errors)


def test_result_validator_rejects_unlabeled_missing_metrics_and_underfilled_output():
    profile = bp.get_profile(_profiles(), "aa_local_smoke")
    rows = _profile_result_rows(profile)
    rows[0]["unsupported_metrics"].remove("peak_memory_bytes")
    rows[1]["output_tokens"] = 1

    errors = bp.validate_results(profile, rows)

    assert any("peak_memory_bytes is null but not declared unsupported" in error for error in errors)
    assert any("underfilled output" in error for error in errors)


def test_validator_rejects_mixed_models_or_token_count_methods():
    profile = bp.get_profile(_profiles(), "aa_local_smoke")
    rows = _profile_result_rows(profile)
    rows[0]["model"] = "different-model"
    rows[1]["token_count_method"] = "word_fallback"

    errors = bp.validate_results(profile, rows)

    assert any("mixes runs, models" in error for error in errors)
    assert any("incompatible run/model/profile/tokenizer" in error for error in errors)
