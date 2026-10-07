"""Matrix construction, launch commands, acquisition bookkeeping, and reporting."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import showdown_acquire as acquire  # noqa: E402
import showdown_report as report  # noqa: E402
import showdown_run as run  # noqa: E402
from evals.offline import grounding, longctx, reasoning_hard, toolcall  # noqa: E402
from evals.core import Case, Response  # noqa: E402
from evals.registry import EVALS, SHOWDOWN, behavioral_names, resolve, tier1_names  # noqa: E402


# --- the committed selection ----------------------------------------------

def test_selection_is_loadable_and_every_artifact_is_pinned():
    selection = acquire.load_selection()
    assert selection["experiment"] == "qwen38_27b_showdown"
    for artifact in selection["artifacts"]:
        assert len(artifact["revision"]) == 40, artifact["id"]
        assert artifact["runtime"] in {"mlx", "llamacpp"}
        assert artifact["tier"] in {"core", "method", "derivative", "optional"}


def test_artifact_ids_are_unique():
    ids = [a["id"] for a in acquire.load_selection()["artifacts"]]
    assert len(ids) == len(set(ids))


def test_every_rejected_candidate_records_a_reason():
    for entry in acquire.load_selection()["rejected_candidates"]:
        assert entry["reason"].strip()


def test_the_mlx_ladder_is_one_provider_and_one_method():
    """The ladder only answers 'what do bits cost' if nothing else varies."""
    by_id = {a["id"]: a for a in acquire.load_selection()["artifacts"]}
    ladder = [by_id[i] for i in run.LADDER_MLX]
    assert len({a["repo"].split("/")[0] for a in ladder}) == 1
    assert all("uniform" in a["quant_method"] for a in ladder)
    bpw = [a["measured_bpw"] for a in ladder]
    assert bpw == sorted(bpw), "ladder must be ordered by increasing precision"


def test_the_gguf_ladder_is_one_provider():
    by_id = {a["id"]: a for a in acquire.load_selection()["artifacts"]}
    assert len({by_id[i]["repo"] for i in run.LADDER_GGUF}) == 1


# --- acquisition -----------------------------------------------------------

def test_repo_furniture_never_makes_an_artifact_incomplete():
    assert acquire._is_optional("README.md")
    assert acquire._is_optional(".gitattributes")
    assert not acquire._is_optional("model-00001-of-00003.safetensors")
    assert not acquire._is_optional("Qwen3.8-27B-Q4_K_M.gguf")


def test_local_dir_follows_the_lm_studio_layout(tmp_path):
    artifact = {"repo": "lmstudio-community/Qwen3.8-27B-MLX-6bit"}
    assert acquire.local_dir_for(artifact, tmp_path) == tmp_path / "lmstudio-community" / "Qwen3.8-27B-MLX-6bit"


def test_a_missing_named_file_is_an_error_not_a_silent_substitution():
    artifact = {"id": "x", "repo": "r", "revision": "0" * 40, "files": ["wanted.gguf"]}
    published = [{"path": "other.gguf", "size": 1, "sha256": "a"}]
    with pytest.raises(SystemExit, match="not in r@"):
        acquire.wanted_files(artifact, published)


def test_wanted_files_ignores_nested_and_non_runtime_formats():
    artifact = {"id": "x", "repo": "r", "revision": "0" * 40}
    published = [
        {"path": "config.json", "size": 1, "sha256": ""},
        {"path": "model.safetensors", "size": 2, "sha256": "b"},
        {"path": "pytorch_model.bin", "size": 3, "sha256": "c"},
        {"path": "onnx/model.onnx", "size": 4, "sha256": "d"},
    ]
    assert {f["path"] for f in acquire.wanted_files(artifact, published)} == {"config.json", "model.safetensors"}


def test_status_classification(tmp_path, monkeypatch):
    artifact = {"id": "a", "tier": "core", "runtime": "llamacpp", "repo": "o/r",
                "revision": "0" * 40, "quant_label": "Q4_K_M", "files": ["m.gguf"]}
    directory = tmp_path / "o" / "r"
    directory.mkdir(parents=True)
    published = [{"path": "m.gguf", "size": 10, "sha256": "x" * 64}]
    monkeypatch.setattr(acquire, "remote_files", lambda *_a, **_k: published)

    assert acquire.inspect_artifact(artifact, tmp_path, with_remote=True)["status"] == "absent"
    (directory / "m.gguf").write_bytes(b"1234")
    assert acquire.inspect_artifact(artifact, tmp_path, with_remote=True)["status"] == "partial"
    (directory / "m.gguf").write_bytes(b"0123456789")
    record = acquire.inspect_artifact(artifact, tmp_path, with_remote=True)
    assert record["status"] == "complete"
    # A complete file set is still not a loadable model.
    assert record["loadable"] is False


def test_deep_verification_detects_a_wrong_hash(tmp_path, monkeypatch):
    artifact = {"id": "a", "tier": "core", "runtime": "llamacpp", "repo": "o/r",
                "revision": "0" * 40, "quant_label": "Q", "files": ["m.gguf"]}
    directory = tmp_path / "o" / "r"
    directory.mkdir(parents=True)
    (directory / "m.gguf").write_bytes(b"0123456789")
    monkeypatch.setattr(acquire, "remote_files",
                        lambda *_a, **_k: [{"path": "m.gguf", "size": 10, "sha256": "d" * 64}])
    record = acquire.inspect_artifact(artifact, tmp_path, with_remote=True, deep=True)
    assert record["status"] == "corrupt"
    assert record["files"][0]["state"] == "hash_mismatch"


# --- arms ------------------------------------------------------------------

def test_arm_ids_are_unique_in_every_mode():
    selection = acquire.load_selection()
    for mode in ("smoke", "core", "full"):
        ids = [a.arm_id for a in run.build_arms(selection, mode)]
        assert len(ids) == len(set(ids)), mode


def test_core_mode_varies_only_the_artifact():
    arms = run.build_arms(acquire.load_selection(), "core")
    assert {a.spec_mode for a in arms} == {"none"}
    assert {a.reasoning for a in arms} == {"off"}
    for arm in arms:
        expected = "native" if arm.runtime == "mlx" else "ctk_q8_0"
        assert arm.kv_mode == expected


def test_smoke_mode_is_a_single_arm():
    assert len(run.build_arms(acquire.load_selection(), "smoke")) == 1


def test_kv_study_includes_a_same_server_control():
    """Without an mlx_vlm KV-native arm, server and KV mode change together."""
    arms = [a for a in run.build_arms(acquire.load_selection(), "full") if a.group == "kv_study"]
    mlx = [a for a in arms if a.runtime == "mlx"]
    assert any(a.kv_mode == "native" and a.server_name == "mlx_vlm" for a in mlx)
    assert {a.artifact_id for a in mlx} == {run.PIVOT_MLX}, "KV study must hold weights fixed"
    llama = [a for a in arms if a.runtime == "llamacpp"]
    assert any(a.kv_mode == "ctk_f16" for a in llama), "llama.cpp default is already q8_0"


def test_mtp_arms_pair_with_a_drafter_and_a_matching_control():
    arms = run.build_arms(acquire.load_selection(), "full")
    mtp = [a for a in arms if a.group == "mtp_study"]
    assert mtp and all(a.draft_artifact_id for a in mtp)
    for arm in mtp:
        control = [c for c in arms if c.artifact_id == arm.artifact_id
                   and c.server_name == arm.server_name and c.kv_mode == arm.kv_mode
                   and c.spec_mode == "none"]
        assert control, f"{arm.arm_id} has no non-MTP control on the same server and KV mode"


def test_reasoning_study_stays_at_or_below_medium():
    """The serving notes record a reproduced hidden-reasoning loop above this."""
    arms = run.build_arms(acquire.load_selection(), "full")
    assert {a.reasoning for a in arms if a.group == "reasoning_study"} <= {"low", "medium"}


def test_derivatives_are_their_own_group():
    arms = run.build_arms(acquire.load_selection(), "full")
    assert {a.artifact_id for a in arms if a.group == "derivative"} == set(run.DERIVATIVES)


# --- launch commands -------------------------------------------------------

def _arm(**kwargs):
    base = dict(artifact_id="a", runtime="mlx", server_name="mlx_lm", backend="mlx",
                kv_mode="native", spec_mode="none", reasoning="off")
    base.update(kwargs)
    return run.Arm(**base)


def test_default_mlx_arm_adds_no_kv_or_draft_flags():
    """Only the variables this arm actually varies may appear.

    `--ctx` and `--thinking` are present on every arm by design: both are
    experimental variables that must be stated rather than inherited from the
    registry. KV mode and drafting are not varied here and must be absent.
    """
    command = run.server_command(_arm(), "/m", "", 8085)
    assert not {"--kv-bits", "--kv-scheme", "--kv-type", "--draft-model",
                "--draft-kind"} & set(command)
    assert command[command.index("--thinking") + 1] == "off"


def test_mlx_kv_native_control_still_selects_the_kv_capable_server():
    command = run.server_command(_arm(server_name="mlx_vlm", backend="mlx-kv"), "/m", "", 8085)
    assert "--kv-bits" in command and command[command.index("--kv-bits") + 1] == "0"


def test_kv_quantized_arms_carry_bits_and_scheme():
    command = run.server_command(
        _arm(server_name="mlx_vlm", backend="mlx-kv", kv_mode="q4_uniform"), "/m", "", 8085)
    assert command[command.index("--kv-bits") + 1] == "4"
    assert command[command.index("--kv-scheme") + 1] == "uniform"


def test_llamacpp_non_mtp_arms_are_explicitly_non_mtp():
    command = run.server_command(
        _arm(runtime="llamacpp", server_name="llama.cpp", backend="llamacpp", kv_mode="ctk_q8_0"),
        "/m.gguf", "", 8080)
    assert "--no-mtp" in command


def test_llamacpp_kv_native_arm_overrides_the_q8_default():
    command = run.server_command(
        _arm(runtime="llamacpp", server_name="llama.cpp", backend="llamacpp", kv_mode="ctk_f16"),
        "/m.gguf", "", 8080)
    assert command[command.index("--kv-type") + 1] == "f16"


def test_an_mtp_arm_without_a_drafter_is_an_error_not_a_silent_baseline():
    with pytest.raises(ValueError, match="no drafter"):
        run.server_command(_arm(spec_mode="mtp", draft_artifact_id="d"), "/m", "", 8085)


def test_model_id_matches_what_each_server_publishes():
    assert run.model_id_for(_arm(), "/models/dir") == "/models/dir"
    assert run.model_id_for(
        _arm(runtime="llamacpp", server_name="llama.cpp", backend="llamacpp"),
        "/models/x/Qwen3.8-27B-Q6_K.gguf") == "Qwen3.8-27B-Q6_K.gguf"


def test_launch_environment_no_longer_carries_reasoning():
    """It moved to CLI flags; see test_reasoning_is_passed_as_a_flag_not_an_environment_variable."""
    assert run._reasoning_env(_arm()) == {}
    assert run._reasoning_env(_arm(reasoning="medium")) == {}


def test_servable_path_skips_mmproj_but_keeps_a_drafters_own_file(tmp_path):
    directory = tmp_path / "repo"
    directory.mkdir()
    for name in ("mmproj-x.gguf", "Qwen3.8-27B-Q6_K.gguf"):
        (directory / name).write_bytes(b"GGUF" + b"0" * 2_000_000)
    artifact = {"runtime": "llamacpp", "local_dir": str(directory),
                "files": [{"path": "mmproj-x.gguf"}, {"path": "Qwen3.8-27B-Q6_K.gguf"}]}
    assert run._servable_path(artifact).endswith("Qwen3.8-27B-Q6_K.gguf")

    drafter_dir = tmp_path / "drafter"
    drafter_dir.mkdir()
    (drafter_dir / "mtp-Qwen3.8-27B-Q4_0.gguf").write_bytes(b"GGUF")
    drafter = {"runtime": "llamacpp", "local_dir": str(drafter_dir), "is_drafter": True,
               "files": [{"path": "mtp-Qwen3.8-27B-Q4_0.gguf"}]}
    assert run._servable_path(drafter).endswith("mtp-Qwen3.8-27B-Q4_0.gguf")


def test_endpoint_parsing():
    endpoints = {"llamacpp": {"base_url": "http://127.0.0.1:8080/v1"},
                 "mlx": {"base_url": "http://127.0.0.1:8085/v1"}}
    assert run._endpoint_for(_arm(), endpoints) == ("127.0.0.1", 8085)
    assert run._endpoint_for(_arm(runtime="llamacpp"), endpoints) == ("127.0.0.1", 8080)


# --- reporting -------------------------------------------------------------

def _record(arm_id, *, status="complete", evals=(), group="ladder_mlx", **extra):
    record = {
        "arm_id": arm_id, "status": status,
        "arm": {"server_name": "mlx_lm", "kv_mode": "native", "spec_mode": "none",
                "reasoning": "off", "group": group, "note": ""},
        "artifact": {"id": arm_id.split(".")[0], "repo": "o/r", "revision": "0" * 40,
                     "quant_label": "Q", "disk_bytes": 16_000_000_000},
        "quality": {"evals": [dict(e) for e in evals]},
    }
    record.update(extra)
    return record


def _eval(name, rate, *, status="ok", scored=10, breakdown=None):
    row = {"eval": name, "status": status, "scored": scored, "strict_pass_rate": rate}
    if breakdown is not None:
        row["breakdown"] = breakdown
    return row


def test_too_few_scorable_cases_reads_as_na_not_as_zero():
    record = _record("a.m.native.none.think_off", evals=[_eval("toolcall", 1.0, scored=2)])
    assert report.axis_scores(record)["toolcall"] is None
    assert report._fmt(None) == "n/a"


def test_a_failed_eval_is_not_scored_as_zero():
    record = _record("a.m.native.none.think_off",
                     evals=[_eval("toolcall", 0.0, status="all_requests_failed")])
    assert report.axis_scores(record)["toolcall"] is None


def test_capability_index_needs_at_least_three_axes():
    two = _record("a.m.native.none.think_off",
                  evals=[_eval("toolcall", 1.0), _eval("grounding", 1.0)])
    assert report.capability_index(two) is None
    three = _record("a.m.native.none.think_off",
                    evals=[_eval("toolcall", 1.0), _eval("grounding", 0.5), _eval("niah", 0.6)])
    assert report.capability_index(three) == pytest.approx(0.7, abs=1e-3)


def _filler(n=2, score=0.2):
    """Plainly-inferior arms so a scenario clears MIN_ARMS_TO_COMPARE.

    Deliberately weak on every axis: they make the comparison legal without
    competing for any of the three recommendations.
    """
    return [_record(f"filler{i}.m.native.none.think_off",
                    evals=[_eval("toolcall", score), _eval("grounding", score),
                           _eval("niah", score)],
                    memory={"peak_proc_rss_gb": 60.0},
                    context={"practical_max_context": 8192},
                    throughput=[{"probe": "short", "gen_tps": 1.0}])
            for i in range(n)]


def test_a_single_scoring_arm_is_not_ranked_as_a_winner():
    """One arm is a measurement, not a comparison."""
    one = _record("only.m.native.none.think_off",
                  evals=[_eval("toolcall", 0.9), _eval("grounding", 0.9), _eval("niah", 0.9)])
    picks = report.recommend([one])
    assert "error" in picks
    assert "only 1 stock arm" in picks["error"]
    assert picks["scored_arms"] == ["only.m.native.none.think_off"]
    assert "best_quality" not in picks


def test_the_refusal_to_rank_is_visible_in_the_rendered_report(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "only.m.native.none.think_off").mkdir(parents=True)
    (run_dir / "only.m.native.none.think_off" / "result.json").write_text(json.dumps(
        _record("only.m.native.none.think_off",
                evals=[_eval("toolcall", 0.9), _eval("grounding", 0.9), _eval("niah", 0.9)])))
    (run_dir / "state.json").write_text(json.dumps({"mode": "core", "evals": [], "exhaustive": False}))
    state, records = report.load_run(run_dir)
    markdown, _summary = report.render(run_dir, state, records)
    assert "Not derivable" in markdown
    assert "Best Quality" not in markdown


def test_startup_failures_are_excluded_from_recommendations():
    records = [
        _record("good.m.native.none.think_off",
                evals=[_eval("toolcall", 0.9), _eval("grounding", 0.9), _eval("niah", 0.9)]),
        _record("dead.m.native.none.think_off", status="startup_failed"),
    ] + _filler()
    picks = report.recommend(records)
    assert picks["best_quality"]["arm_id"] == "good.m.native.none.think_off"
    assert all(v is None or v.get("arm_id") != "dead.m.native.none.think_off"
               for k, v in picks.items() if isinstance(v, dict) and "arm_id" in v)


def test_derivatives_never_win_a_recommendation():
    records = [
        _record("stock.m.native.none.think_off",
                evals=[_eval("toolcall", 0.6), _eval("grounding", 0.6), _eval("niah", 0.6)]),
        _record("deriv_x.m.native.none.think_off", group="derivative",
                evals=[_eval("toolcall", 1.0), _eval("grounding", 1.0), _eval("niah", 1.0)]),
    ] + _filler()
    picks = report.recommend(records)
    assert picks["best_quality"]["artifact"] == "stock"


def test_an_intermittent_agent_task_disqualifies_the_daily_driver():
    flaky = _record(
        "flaky.m.native.none.think_off",
        evals=[_eval("toolcall", 1.0), _eval("grounding", 1.0),
               _eval("agentic", 0.5, breakdown={"by_task": {}, "intermittent_tasks": ["agent_fix_ticket"]})],
        memory={"peak_proc_rss_gb": 20.0}, throughput=[{"probe": "short", "gen_tps": 40}])
    steady = _record(
        "steady.m.native.none.think_off",
        evals=[_eval("toolcall", 0.9), _eval("grounding", 0.9),
               _eval("agentic", 0.9, breakdown={"by_task": {}, "intermittent_tasks": []})],
        memory={"peak_proc_rss_gb": 22.0}, throughput=[{"probe": "short", "gen_tps": 30}])
    assert report._reliable(flaky) is False
    picks = report.recommend([flaky, steady] + _filler())
    assert picks["best_daily_driver"]["arm_id"] == "steady.m.native.none.think_off"


def test_workspace_pick_prefers_usable_context_over_raw_capability():
    big = _record("big.m.native.none.think_off",
                  evals=[_eval("toolcall", 0.7), _eval("grounding", 0.7), _eval("agentic", 0.7)],
                  context={"practical_max_context": 131072}, memory={"peak_proc_rss_gb": 18.0})
    sharp = _record("sharp.m.native.none.think_off",
                    evals=[_eval("toolcall", 1.0), _eval("grounding", 1.0), _eval("agentic", 1.0)],
                    context={"practical_max_context": 32768}, memory={"peak_proc_rss_gb": 34.0})
    picks = report.recommend([big, sharp] + _filler())
    assert picks["best_quality"]["arm_id"] == "sharp.m.native.none.think_off"
    assert picks["best_long_context_workspace"]["arm_id"] == "big.m.native.none.think_off"
    assert len(picks["keep_multiple"]["distinct_artifacts"]) > 1


def test_report_renders_and_marks_a_partial_matrix(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "a.m.native.none.think_off").mkdir(parents=True)
    (run_dir / "a.m.native.none.think_off" / "result.json").write_text(json.dumps(
        _record("a.m.native.none.think_off",
                evals=[_eval("toolcall", 0.9), _eval("grounding", 0.8), _eval("niah", 1.0)],
                memory={"peak_proc_rss_gb": 17.0}, context={"practical_max_context": 65536},
                throughput=[{"probe": "short", "gen_tps": 42.0, "ttft_s": 0.4, "prompt_tps": 900}])))
    for index, filler in enumerate(_filler()):
        (run_dir / f"f{index}").mkdir(parents=True)
        (run_dir / f"f{index}" / "result.json").write_text(json.dumps(filler))
    (run_dir / "state.json").write_text(json.dumps(
        {"mode": "core", "evals": ["toolcall"], "exhaustive": False}))
    state, records = report.load_run(run_dir)
    markdown, summary = report.render(run_dir, state, records)
    assert "Exhaustive: **no**" in markdown
    assert "partial matrix" in markdown
    assert "Best Daily Driver" in markdown
    target = next(a for a in summary["arms"] if a["arm_id"].startswith("a."))
    assert target["capability_index"] == pytest.approx(0.9, abs=1e-3)


# --- eval catalog ----------------------------------------------------------

def test_the_historical_tier1_suite_is_unchanged():
    assert tier1_names() == ["niah", "ifeval_local", "determinism"]


def test_behavioral_evals_are_opt_in():
    assert set(behavioral_names()) == {"toolcall", "longctx", "grounding", "reasoning_hard", "agentic"}
    assert not set(behavioral_names()) & set(tier1_names())


def test_showdown_covers_every_required_capability():
    names = {e.name for e in resolve(["showdown"])}
    assert {"toolcall", "agentic", "reasoning_hard", "grounding", "longctx",
            "ifeval_local", "niah", "determinism"} <= names


def test_every_showdown_eval_builds_offline():
    for spec in resolve(SHOWDOWN):
        cases = spec.build_cases(ctx_cap=8192, repeats=1, seed=1)
        assert cases, spec.name


# --- behavioural scoring ---------------------------------------------------

def test_unparseable_tool_arguments_fail_before_anything_else():
    case = toolcall.build_cases()[0]
    response = Response(text="", arguments_unparseable=1, tool_calls=[
        {"id": "1", "name": "get_weather", "arguments": None, "raw_arguments": "{bad"}])
    assert not toolcall.score_response(case, response).passed


def test_calling_a_tool_when_none_was_needed_fails_restraint():
    case = next(c for c in toolcall.build_cases() if c.case_id == "tc_restraint_arithmetic")
    called = Response(text="", tool_calls=[
        {"id": "1", "name": "get_weather", "arguments": {"city": "x"}, "raw_arguments": "{}"}])
    assert not toolcall.score_response(case, called).passed
    assert toolcall.score_response(case, Response(text="564")).passed


def test_repeating_a_failed_call_verbatim_fails_recovery():
    case = next(c for c in toolcall.build_cases() if c.case_id == "tc_recover_bad_path")
    repeat = Response(text="", tool_calls=[
        {"id": "1", "name": "read_file", "arguments": {"path": "src/uttils.py"},
         "raw_arguments": '{"path": "src/uttils.py"}'}])
    assert "repeated the failed call" in toolcall.score_response(case, repeat).detail
    recovered = Response(text="", tool_calls=[
        {"id": "1", "name": "search_files", "arguments": {"pattern": "uttils"},
         "raw_arguments": "{}"}])
    assert toolcall.score_response(case, recovered).passed


def test_long_context_separates_fabrication_from_misattribution():
    case = next(c for c in longctx.build_cases(ctx_cap=8192) if "absent" in c.case_id)
    planted = case.meta["planted_values"][0]
    assert "fabricated" in longctx.score(case, "The value is 4242.").detail
    assert "misattribution" in longctx.score(case, f"It is {planted}.").detail
    assert longctx.score(case, "NOT IN DOCUMENT").passed


def test_long_context_partial_credit_distinguishes_retrieval_from_combination():
    case = next(c for c in longctx.build_cases(ctx_cap=8192) if "combine" in c.case_id)
    a, b = case.meta["parts"]
    assert longctx.score(case, str(case.meta["expected"])).passed
    partial = longctx.score(case, f"{a} and {b}")
    assert not partial.passed and partial.value == 0.5


def test_grounding_penalises_both_hallucination_and_over_refusal():
    unanswerable = next(c for c in grounding.build_cases() if c.case_id == "gr_unans_revenue")
    assert grounding.score(unanswerable, "The context does not state it.").passed
    assert not grounding.score(unanswerable, "About 4.2 million.").passed

    answerable = next(c for c in grounding.build_cases() if c.case_id == "gr_ans_staff")
    over = grounding.score(answerable, "That is not stated in the context.")
    assert not over.passed and "over-refused" in over.detail


def test_reasoning_requires_the_answer_line_and_flags_distractors():
    case = next(c for c in reasoning_hard.build_cases(repeats=1)
                if c.meta["problem"] == "rh_trap_average_speed")
    assert reasoning_hard.score(case, "work\nANSWER: 40.00").passed
    captured = reasoning_hard.score(case, "work\nANSWER: 45")
    assert not captured.passed and "planted distractor" in captured.detail
    assert "no `ANSWER:` line" in reasoning_hard.score(case, "The answer is 40.").detail


def test_reasoning_summary_reports_flakiness_rather_than_a_mean():
    rows = [{"case_id": "rh_x#r0", "passed": 1, "group": "trap", "detail": ""},
            {"case_id": "rh_x#r1", "passed": 0, "group": "trap", "detail": ""},
            {"case_id": "rh_y#r0", "passed": 1, "group": "trap", "detail": ""}]
    out = reasoning_hard.summarize(rows)
    assert out["flaky"] == {"rh_x": "1/2"}
    assert out["reliable"] == 1


def test_reasoning_ground_truth_is_self_consistent():
    """Every problem with a planted distractor must have a distinct wrong answer."""
    for _id, _g, _t, expected, distracted, _n in reasoning_hard.PROBLEMS:
        if distracted is None:
            continue
        assert float(expected) != float(distracted), _id


# --- arm lifecycle ---------------------------------------------------------

class _FakeHandle:
    def __init__(self, log_path):
        self.log_path = log_path
        self.base_url = "http://127.0.0.1:9/v1"
        self.load_seconds = 1.0


def _fake_sample():
    from showdown_server import MemorySample
    return MemorySample(baseline_used_bytes=10_000_000_000, peak_used_bytes=30_000_000_000,
                        peak_proc_rss_bytes=8_000_000_000)


def _stub_launch(monkeypatch, tmp_path, shutdowns):
    monkeypatch.setattr(run.server, "port_is_free", lambda *_a, **_k: True)
    monkeypatch.setattr(run.server, "launch",
                        lambda *a, **k: _FakeHandle(k["log_path"]))

    def shutdown(handle, **_kwargs):
        shutdowns.append(handle)
        return _fake_sample()

    monkeypatch.setattr(run.server, "shutdown", shutdown)


def _artifact(tmp_path):
    directory = tmp_path / "org" / "model"
    directory.mkdir(parents=True)
    (directory / "m.gguf").write_bytes(b"GGUF" + b"0" * 2_000_000)
    return {"a": {"id": "a", "repo": "org/model", "revision": "0" * 40, "quant_label": "Q",
                  "runtime": "llamacpp", "local_dir": str(directory), "disk_bytes": 2_000_000,
                  "files": [{"path": "m.gguf"}]}}


ENDPOINTS = {"llamacpp": {"base_url": "http://127.0.0.1:8080/v1"},
             "mlx": {"base_url": "http://127.0.0.1:8085/v1"}}


def _run(monkeypatch, tmp_path, shutdowns):
    arm = run.Arm(artifact_id="a", runtime="llamacpp", server_name="llama.cpp",
                  backend="llamacpp", kv_mode="ctk_q8_0", spec_mode="none", reasoning="off")
    return run.run_arm(
        arm, artifacts=_artifact(tmp_path), run_dir=tmp_path / "run", endpoints=ENDPOINTS,
        eval_names=["toolcall"], ctx_cap=8192, repeats=1, timeout=5, ready_timeout=5,
        skip_context_probe=True, verbose=False), arm


def test_startup_failure_is_recorded_completely_and_never_scored(monkeypatch, tmp_path):
    shutdowns = []
    _stub_launch(monkeypatch, tmp_path, shutdowns)
    monkeypatch.setattr(run.server, "await_ready", lambda *a, **k: "did not answer in time")

    def explode(*_a, **_k):
        raise AssertionError("the eval suite must never run against a server that never came up")

    monkeypatch.setattr(run, "_measure_and_score", explode)
    record, _ = _run(monkeypatch, tmp_path, shutdowns)

    assert record["status"] == "startup_failed"
    assert "did not answer" in record["error"]
    # Teardown happens exactly once, and the memory block is attached BEFORE the
    # record is written, so the file and the return value agree.
    assert len(shutdowns) == 1
    assert record["memory"]["attributed_peak_gb"] == 20.0
    written = json.loads((tmp_path / "run" / record["arm_id"] / "result.json").read_text())
    assert written["status"] == "startup_failed"
    assert written["memory"]["attributed_peak_gb"] == 20.0
    assert "quality" not in written


def test_a_harness_error_still_tears_the_server_down_once(monkeypatch, tmp_path):
    shutdowns = []
    _stub_launch(monkeypatch, tmp_path, shutdowns)
    monkeypatch.setattr(run.server, "await_ready", lambda *a, **k: None)
    monkeypatch.setattr(run, "_measure_and_score",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("probe blew up")))
    record, _ = _run(monkeypatch, tmp_path, shutdowns)
    assert record["status"] == "harness_error"
    assert "probe blew up" in record["error"]
    assert len(shutdowns) == 1
    assert "memory" in json.loads((tmp_path / "run" / record["arm_id"] / "result.json").read_text())


def test_a_busy_port_is_refused_before_anything_launches(monkeypatch, tmp_path):
    shutdowns = []
    _stub_launch(monkeypatch, tmp_path, shutdowns)
    monkeypatch.setattr(run.server, "port_is_free", lambda *_a, **_k: False)
    monkeypatch.setattr(run.server, "launch",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not launch")))
    record, _ = _run(monkeypatch, tmp_path, shutdowns)
    assert record["status"] == "port_busy"
    assert shutdowns == []


def test_a_missing_artifact_is_reported_not_launched(monkeypatch, tmp_path):
    shutdowns = []
    _stub_launch(monkeypatch, tmp_path, shutdowns)
    monkeypatch.setattr(run.server, "launch",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not launch")))
    arm = run.Arm(artifact_id="a", runtime="llamacpp", server_name="llama.cpp",
                  backend="llamacpp", kv_mode="ctk_q8_0", spec_mode="none", reasoning="off")
    artifacts = {"a": {"id": "a", "repo": "o/r", "revision": "0" * 40, "quant_label": "Q",
                       "runtime": "llamacpp", "local_dir": str(tmp_path / "nope"), "files": []}}
    record = run.run_arm(arm, artifacts=artifacts, run_dir=tmp_path / "run", endpoints=ENDPOINTS,
                         eval_names=["toolcall"], ctx_cap=8192, repeats=1, timeout=5,
                         ready_timeout=5, skip_context_probe=True, verbose=False)
    assert record["status"] == "artifact_missing"


def test_attributed_peak_prefers_the_system_delta_over_rss():
    from showdown_server import MemorySample
    sample = MemorySample(baseline_used_bytes=30_000_000_000,
                          peak_used_bytes=42_600_000_000,
                          peak_proc_rss_bytes=7_900_000_000)
    assert sample.attributed_peak_bytes == 12_600_000_000
    record = {"memory": {"attributed_peak_gb": 12.6, "peak_proc_rss_gb": 7.9}}
    assert report._peak_gb(record) == 12.6
    assert report._peak_gb({"memory": {"peak_proc_rss_gb": 7.9}}) == 7.9


def test_prompt_token_calibration_avoids_a_second_large_prefill():
    """Re-prefilling a 32K prompt to read usage costs ~80s per arm."""
    import showdown_server as srv
    calibration = {}
    messages = [{"role": "user", "content": " ".join(["word"] * 1000)}]
    # No calibration yet and an expensive probe: refuses rather than re-prefilling.
    assert srv._derive_prompt_tokens("http://127.0.0.1:1", "m", messages, 32768,
                                     calibration) == (0, "unavailable")
    calibration["tokens_per_word"] = 1.2
    tokens, source = srv._derive_prompt_tokens("http://127.0.0.1:1", "m", messages, 32768,
                                               calibration)
    assert (tokens, source) == (1200, "calibrated")


# --- catalog classification ------------------------------------------------

import discover_models  # noqa: E402


def _families():
    catalog = json.loads((ROOT / "configs" / "model_catalog.json").read_text())
    return {f["id"]: f for f in catalog["model_families"]}, catalog["model_families"]


def _classify(path_text, kind="gguf"):
    """First family whose patterns match, mirroring discovery's first-match-wins."""
    _by_id, ordered = _families()
    for family in ordered:
        if discover_models._matches(Path(path_text), family.get(f"{kind}_patterns", [])):
            return family["id"]
    return None


# A synthetic model root. Deliberately not a home-shaped path: scripts/smoke_test.py
# rejects local/development paths in files Git would track, and these patterns
# only care about the relative shape below the root anyway.
ROOT_PREFIX = "/models/root/.lmstudio/models"


@pytest.mark.parametrize("path_text,expected", [
    (f"{ROOT_PREFIX}/bartowski/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q6_K.gguf", "qwen3_27b_dense"),
    (f"{ROOT_PREFIX}/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf", "qwen3_27b_dense"),
    (f"{ROOT_PREFIX}/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF/Huihui-Qwen3.8-27B-abliterated-Q6_K.gguf",
     "qwen3_8_27b_abliterated"),
    (f"{ROOT_PREFIX}/OBLITERATUS/Qwen3.8-27B-OBLITERATED/Qwen3.8-27B-OBLITERATED-Q6_K.gguf",
     "qwen3_8_27b_abliterated"),
])
def test_gguf_classification(path_text, expected):
    assert _classify(path_text) == expected


def test_a_drafter_inside_a_stock_repo_never_reads_as_a_stock_model():
    """The directory is named Qwen3.8-27B-GGUF, which an unanchored pattern matched."""
    drafter = f"{ROOT_PREFIX}/ggml-org/Qwen3.8-27B-GGUF/mtp-Qwen3.8-27B-Q4_0.gguf"
    assert _classify(drafter) != "qwen3_27b_dense"
    # Belt and braces: discovery drops it before matching ever happens.
    assert discover_models._skip_candidate(Path(drafter))


@pytest.mark.parametrize("path_text,expected", [
    (f"{ROOT_PREFIX}/lmstudio-community/Qwen3.8-27B-MLX-6bit", "qwen3_27b_dense"),
    (f"{ROOT_PREFIX}/mlx-community/Qwen3.8-27B-oQ4", "qwen3_27b_dense"),
    (f"{ROOT_PREFIX}/mlx-community/Qwen3.8-27B-MTP-4bit", "qwen3_8_27b_mtp_draft"),
    (f"{ROOT_PREFIX}/mlx-community/Qwen3.8-27B-Uncensored-OptiQ-4bit", "qwen3_8_27b_abliterated"),
])
def test_mlx_classification(path_text, expected):
    assert _classify(path_text, kind="mlx") == expected


def test_stock_and_derivative_families_never_both_claim_a_path():
    by_id, _ = _families()
    paths = [
        f"{ROOT_PREFIX}/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF/Huihui-Qwen3.8-27B-abliterated-Q6_K.gguf",
        f"{ROOT_PREFIX}/OBLITERATUS/Qwen3.8-27B-OBLITERATED/Qwen3.8-27B-OBLITERATED-Q6_K.gguf",
    ]
    for path_text in paths:
        assert not discover_models._matches(
            Path(path_text), by_id["qwen3_27b_dense"]["gguf_patterns"]), path_text


def test_the_abliterated_family_is_ordered_before_the_stock_family():
    """Discovery is first-match-wins, so order is the real guarantee."""
    _by_id, ordered = _families()
    ids = [f["id"] for f in ordered]
    assert ids.index("qwen3_8_27b_abliterated") < ids.index("qwen3_27b_dense")
    assert ids.index("qwen3_8_27b_mtp_draft") < ids.index("qwen3_27b_dense")


@pytest.mark.parametrize("name,expected", [
    ("Qwen3.8-27B-Q6_K.gguf", "Q6_K"),
    ("Qwen3.8-27B-MLX-6bit", "6bit"),
    ("Qwen3.8-27B-oQ4", "oQ4"),
    ("Qwen3.8-27B-OptiQ-4bit", "OptiQ-4bit"),
])
def test_mixed_precision_quants_are_labelled_not_unknown(name, expected):
    assert discover_models._infer_quant(Path(name)) == expected


def test_context_probe_refuses_a_rung_it_cannot_afford(monkeypatch):
    """Climbing into swap costs wall clock and leaves a worthless measurement."""
    import showdown_server as srv
    out = srv.probe_max_context("http://127.0.0.1:1", "m",
                                min_available_bytes=10 ** 15, timeout=2)
    assert out["stopped_because"] == "not_attempted_memory_pressure"
    assert out["practical_max_context"] == 0
    assert out["rungs"][0]["outcome"] == "not_attempted_memory_pressure"


def test_context_probe_respects_a_configured_ctx_cap():
    import showdown_server as srv
    out = srv.probe_max_context("http://127.0.0.1:1", "m", max_context=16384,
                                min_available_bytes=0, timeout=2)
    assert out["rungs"][0]["outcome"] == "above_configured_ctx_cap"


def test_a_calibrated_prompt_rate_is_visibly_distinct_from_a_measured_one():
    measured = {"prompt_tps": 288.3, "prompt_token_source": "stream_usage"}
    derived = {"prompt_tps": 385.7, "prompt_token_source": "calibrated"}
    assert report._prompt_tps_cell(measured) == "288.3"
    assert report._prompt_tps_cell(derived) == "385.7~"
    assert report._prompt_tps_cell({}) == "n/a"


def test_agent_scratch_root_is_removed(monkeypatch, tmp_path):
    """A 19-arm matrix must not leave 19 orphaned temp trees."""
    import evals.agentic.eval as agentic_eval
    created = {}
    real = agentic_eval.tempfile.mkdtemp

    def spy(*a, **k):
        created["path"] = real(*a, **k)
        return created["path"]

    monkeypatch.setattr(agentic_eval.tempfile, "mkdtemp", spy)
    monkeypatch.setattr(agentic_eval, "_run_all", lambda *a, **k: [])
    agentic_eval.run_cases([], object())
    assert created and not Path(created["path"]).exists()


def test_serving_context_is_explicit_on_every_arm_and_bounds_the_probe():
    """Left implicit, llama.cpp would allocate its 262144 ctx_cap up front."""
    arms = run.build_arms(acquire.load_selection(), "full")
    assert {a.ctx for a in arms} == {run.DEFAULT_SERVE_CTX}
    assert run.DEFAULT_SERVE_CTX == 131072 + run.CTX_HEADROOM
    for arm in arms:
        assert "--ctx" in run.server_command(arm, "/m", "/d", 8080)


def test_serve_ctx_leaves_room_for_the_largest_eval_case():
    from evals.offline import niah
    largest = max(niah.build_cases(ctx_cap=131072), key=lambda c: c.meta["context_len"])
    estimated = int(len(largest.prompt.split()) * niah.TOKENS_PER_WORD)
    assert estimated + largest.max_tokens < run.DEFAULT_SERVE_CTX


def test_long_context_combine_cannot_be_answered_by_keyword_match():
    """The answer must not appear anywhere in the document."""
    for case in longctx.build_cases(ctx_cap=131072):
        if "combine" not in case.case_id:
            continue
        document = case.prompt.split("<document>")[1].split("</document>")[0]
        assert str(case.meta["expected"]) not in document, case.case_id
        a, b = case.meta["parts"]
        # and the two required values must be far apart, not adjacent
        separation = abs(document.index(str(b)) - document.index(str(a))) / len(document)
        assert separation > 0.5, f"{case.case_id}: only {separation:.0%} apart"


# --- reasoning mode is actually varied -------------------------------------

def test_reasoning_is_passed_as_a_flag_not_an_environment_variable():
    """serve_local.sh re-assigns MODEL_ENABLE_THINKING from the registry after the
    caller's environment is read, so an env-var approach silently served every
    arm with the catalog default and made the reasoning study measure nothing."""
    assert run._reasoning_env(_arm(reasoning="medium")) == {}
    off = run.server_command(_arm(reasoning="off"), "/m", "", 8085)
    assert off[off.index("--thinking") + 1] == "off"
    assert "--reasoning-effort" not in off
    for effort in ("low", "medium"):
        on = run.server_command(_arm(reasoning=effort), "/m", "", 8085)
        assert on[on.index("--thinking") + 1] == "on"
        assert on[on.index("--reasoning-effort") + 1] == effort


def test_reasoning_arms_produce_distinct_server_commands():
    arms = run.build_arms(acquire.load_selection(), "full")
    pivot = [a for a in arms if a.artifact_id == run.PIVOT_MLX and a.server_name == "mlx_lm"]
    commands = {a.reasoning: " ".join(run.server_command(a, "/m", "", 8085)) for a in pivot}
    assert set(commands) == {"off", "low", "medium"}
    assert len(set(commands.values())) == 3, "reasoning arms must not collapse into one config"


def test_thinking_arms_let_the_server_govern_effort():
    """The client can only send enable_thinking; sending it would replace the
    server's chat-template args and drop reasoning_effort, collapsing low into
    medium."""
    assert run._client_thinking(_arm(reasoning="off")) is False
    assert run._client_thinking(_arm(reasoning="low")) is None
    assert run._client_thinking(_arm(reasoning="medium")) is None


def test_report_warns_that_mlx_vlm_arms_are_sampling_confounded(tmp_path):
    """mlx_vlm.server takes no sampling flags, so its arms differ from the
    mlx_lm ladder by more than the KV mode alone."""
    run_dir = tmp_path / "run"
    records = [
        _record("pivot.mlx_vlm.native.none.think_off",
                evals=[_eval("toolcall", 0.9), _eval("grounding", 0.9), _eval("niah", 0.9)],
                group="kv_study"),
    ] + _filler()
    records[0]["arm"]["server_name"] = "mlx_vlm"
    for index, record in enumerate(records):
        (run_dir / f"a{index}").mkdir(parents=True)
        (run_dir / f"a{index}" / "result.json").write_text(json.dumps(record))
    (run_dir / "state.json").write_text(json.dumps({"mode": "full", "evals": [], "exhaustive": True}))
    state, loaded = report.load_run(run_dir)
    markdown, _ = report.render(run_dir, state, loaded)
    assert "Compare within the group" in markdown
    assert "pivot.mlx_vlm.native.none.think_off" in markdown


def test_run_parameters_reach_the_report_header(tmp_path):
    run_dir = tmp_path / "run"
    for index, record in enumerate(_filler(3)):
        (run_dir / f"a{index}").mkdir(parents=True)
        (run_dir / f"a{index}" / "result.json").write_text(json.dumps(record))
    (run_dir / "state.json").write_text(json.dumps({
        "mode": "core", "evals": [], "exhaustive": True, "serve_ctx": 139264,
        "ctx_cap": 131072, "repeats": 3,
        "runtime_versions": {"llama.cpp": "10809", "mlx-lm": "0.31.3"}}))
    state, loaded = report.load_run(run_dir)
    markdown, summary = report.render(run_dir, state, loaded)
    assert "Served context: 139264" in markdown
    assert "llama.cpp 10809" in markdown
    assert summary["serve_ctx"] == 139264


def test_the_added_hard_reasoning_ground_truths_hold_independently():
    """Two of the original problems shipped with wrong answers; re-derive these."""
    from itertools import permutations
    from math import comb, floor
    from fractions import Fraction
    expected = {p[0]: p[3] for p in reasoning_hard.PROBLEMS}

    assert str(Fraction(comb(7, 3) + comb(5, 3) + comb(4, 3), comb(16, 3))) == expected["rh_hard_probability"]

    solutions = [p for p in permutations([8080, 8081, 8082, 8083, 8084])
                 if p[0] > p[1] and p[2] == p[3] + 2 and p[4] == min(p) and p[1] != 8081]
    assert len(solutions) == 1, "the assignment problem must have exactly one solution"
    assert solutions[0][2] == expected["rh_hard_assignment"]

    volume = floor(2450 * 0.87) + 260
    volume = floor(volume * 0.85) - 95
    assert volume == expected["rh_hard_compounding"]
    # The per-step flooring must actually change the answer, or the instruction
    # it tests is inert — which is exactly how the first draft of this problem
    # was wrong: 2400 x 0.875 and 2360 x 0.85 are both already whole numbers.
    unfloored = (2450 * 0.87 + 260) * 0.85 - 95
    distractor = {p[0]: p[4] for p in reasoning_hard.PROBLEMS}["rh_hard_compounding"]
    assert round(unfloored) == distractor
    assert volume != distractor


def test_the_added_hard_tool_cases_exist_and_are_scorable():
    cases = {c.case_id: c for c in toolcall.build_cases()}
    hard = [c for c in cases.values() if c.meta["group"] == "hard"]
    assert len(hard) == 6
    # the confusable neighbours must actually be offered, or the case is trivial
    names = {t["function"]["name"] for t in hard[0].tools}
    assert {"get_historical_price", "get_forecast", "batch_convert_currency"} <= names

    # picking the near-neighbour fails for a stated reason
    case = cases["tc_hard_current_not_historical"]
    wrong = Response(text="", tool_calls=[{"id": "1", "name": "get_historical_price",
                                           "arguments": {"symbol": "NVDA", "date": "2026-09-17"},
                                           "raw_arguments": "{}"}])
    assert not toolcall.score_response(case, wrong).passed
    right = Response(text="", tool_calls=[{"id": "1", "name": "get_stock_price",
                                           "arguments": {"symbol": "NVDA"}, "raw_arguments": "{}"}])
    assert toolcall.score_response(case, right).passed

    # three separate calls fail the batching case
    batch = cases["tc_hard_batch_not_repeated"]
    three = Response(text="", tool_calls=[
        {"id": str(i), "name": "convert_currency",
         "arguments": {"amount": a, "from_currency": "EUR", "to_currency": "USD"},
         "raw_arguments": "{}"} for i, a in enumerate([120, 340.5, 89])])
    assert not toolcall.score_response(batch, three).passed
    one = Response(text="", tool_calls=[{"id": "1", "name": "batch_convert_currency",
                                         "arguments": {"amounts": [120, 340.5, 89],
                                                       "from_currency": "EUR", "to_currency": "USD"},
                                         "raw_arguments": "{}"}])
    assert toolcall.score_response(batch, one).passed


def test_an_aborted_eval_contributes_nothing_rather_than_a_zero():
    """A long-context eval that ran out of memory partway must not score as
    capability loss, and its truncated case set must not score as a pass."""
    record = _record("a.m.native.none.think_off", evals=[
        _eval("toolcall", 1.0), _eval("grounding", 0.9), _eval("niah", 1.0),
        _eval("longctx", 1.0, status="aborted", scored=9)])
    scores = report.axis_scores(record)
    assert scores["longctx"] is None
    assert "longctx" not in report.axes_used(record)


def test_report_flags_arms_whose_indices_cover_different_axes(tmp_path):
    run_dir = tmp_path / "run"
    full = _record("full.m.native.none.think_off", evals=[
        _eval("toolcall", 1.0), _eval("grounding", 0.9), _eval("niah", 1.0), _eval("longctx", 0.8)])
    lost = _record("lost.m.native.none.think_off", evals=[
        _eval("toolcall", 1.0), _eval("grounding", 0.9), _eval("niah", 1.0),
        _eval("longctx", 1.0, status="aborted", scored=9)])
    for index, record in enumerate([full, lost] + _filler()):
        (run_dir / f"a{index}").mkdir(parents=True)
        (run_dir / f"a{index}" / "result.json").write_text(json.dumps(record))
    (run_dir / "state.json").write_text(json.dumps({"mode": "core", "evals": [], "exhaustive": True}))
    state, records = report.load_run(run_dir)
    markdown, summary = report.render(run_dir, state, records)
    assert "different axis sets" in markdown
    assert "lost.m.native.none.think_off" in markdown
    assert "longctx" in next(a["axes_used"] for a in summary["arms"]
                             if a["arm_id"] == "full.m.native.none.think_off")
