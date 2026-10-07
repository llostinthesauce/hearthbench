#!/usr/bin/env python3
"""
showdown_run.py — the Qwen3.8-27B configuration matrix.

One arm is one fully specified configuration: an artifact, a serving path, a
KV-cache mode, a speculative-decoding mode, and a reasoning mode. The runner
takes an arm through five separable stages and records each one independently:

    launch  →  readiness  →  systems measurement  →  behavioural suite  →  teardown

Separating them is the whole design. A server that never became ready records
`startup_failed` and is never handed to the eval suite, so it cannot show up in
the final matrix as a model that scored zero — which is exactly how a bad
comparison gets made. Likewise a suite that aborted mid-way is `partial`, not a
low score.

Every arm's raw output lands in its own directory before the next arm starts, so
an interrupted matrix loses at most the arm that was running, and `--resume`
picks up from there.

Experimental variables are kept apart on purpose:

    weight quantization   the artifact
    KV-cache quantization --kv-bits on MLX, -ctk/-ctv on llama.cpp
    serving runtime       mlx_lm vs mlx_vlm vs llama.cpp
    speculative decoding  MTP drafter on/off
    reasoning             thinking off / low / medium

KV quantization and MTP are only available on `mlx_vlm.server`, never on
`mlx_lm.server`. Any arm that varies those therefore also varies the server, so
the matrix always pairs them with an `mlx_vlm` KV-native control arm. Without
that control, a measured difference cannot be attributed to KV bits rather than
to the server change that came with them.

Modes
-----
    smoke   one artifact, one arm, a three-eval subset. Proves the pipeline.
    core    the weight-quantization ladder on its primary serving path.
    full    core + KV-cache arms + MTP arms + reasoning arms + derivatives.

Examples
--------
    python3 scripts/showdown_run.py --list
    python3 scripts/showdown_run.py --mode smoke
    python3 scripts/showdown_run.py --mode core --run-dir results/qwen38_showdown/core01
    python3 scripts/showdown_run.py --mode full --resume --run-dir results/.../full01
    python3 scripts/showdown_run.py --arm mlx_lms_6bit.mlx_lm.kv_native.nospec.think_off
    python3 scripts/showdown_run.py --mode core --evals toolcall agentic
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for entry in (str(SCRIPT_DIR), str(ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import local_config  # noqa: E402
import showdown_acquire as acquire  # noqa: E402
import showdown_server as server  # noqa: E402
from bench_quality import run_suite  # noqa: E402
from evals.registry import SHOWDOWN, SHOWDOWN_SMOKE  # noqa: E402

SERVE = str(ROOT / "scripts" / "serve_local.sh")
DEFAULT_RUN_ROOT = ROOT / "results" / "qwen38_showdown"

# Serving context for every arm, and the ceiling of the practical-context probe.
#
# Left to the registry, llama.cpp would serve at its ctx_cap of 262144. That is a
# FIXED, up-front allocation: at -ctk/-ctv q8_0 this model's 16 full-attention
# layers (4 KV heads x 256 head_dim) cost ~9.1 GB of KV at 262144 against ~4.6 GB
# at 131072 — paid on every GGUF arm whether or not a long prompt ever arrives.
# On 64 GB that is the difference between a Q6_K arm sitting at ~28 GB and ~33 GB
# before the OS and everything else.
#
# It is also a cross-runtime confound: MLX grows its cache lazily, so an MLX arm
# that never sees 262K never pays for it while the llama.cpp arm always does.
# Capping both at the same value makes "peak memory" and "usable context"
# comparable between runtimes instead of partly a serving-flag artifact.
#
# The value is the eval suite's own --ctx-cap default (131072) plus headroom for
# the answer: a 131072-token NIAH case served at exactly -c 131072 has nowhere to
# put its 512 generated tokens and would be truncated or refused, which would read
# as a long-context failure rather than a serving-flag mistake.
#
# The serving notes record llama-server wedging on a 131K prompt when served at
# -c 262144. Raise this deliberately with --serve-ctx when that is the experiment.
CTX_HEADROOM = 8192
DEFAULT_SERVE_CTX = 131072 + CTX_HEADROOM


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Arm:
    """One fully specified configuration. `arm_id` is stable across runs, which
    is what makes `--resume` and cross-run comparison work."""

    artifact_id: str
    runtime: str            # llamacpp | mlx
    server_name: str        # mlx_lm | mlx_vlm | llama.cpp
    backend: str            # the serve_local.sh --backend value
    kv_mode: str            # native | q8_turbo | q4_uniform | ctk_q8_0 | ctk_f16
    spec_mode: str          # none | mtp
    reasoning: str          # off | low | medium
    ctx: int = 0
    draft_artifact_id: str = ""
    group: str = "core"
    note: str = ""

    @property
    def arm_id(self) -> str:
        return ".".join((self.artifact_id, self.server_name, self.kv_mode,
                         self.spec_mode, f"think_{self.reasoning}"))


# The weight ladder, in the order a report should read them.
LADDER_MLX = ["mlx_lms_4bit", "mlx_lms_5bit", "mlx_lms_6bit", "mlx_lms_8bit"]
LADDER_GGUF = ["gguf_bartowski_q4_k_m", "gguf_bartowski_q5_k_m", "gguf_bartowski_q6_k"]
METHOD_CONTRAST = ["mlx_oq4", "gguf_lms_q4_k_m"]
DERIVATIVES = ["deriv_huihui_q6_k", "deriv_obliteratus_q6_k"]

# Which artifact carries the KV / MTP / reasoning sub-studies. One artifact, held
# fixed, so those variables are not tangled with weight bits. 6-bit is the
# hypothesis under test, which makes it the honest place to spend the extra arms.
PIVOT_MLX = "mlx_lms_6bit"
PIVOT_GGUF = "gguf_bartowski_q5_k_m"


def _base_arm(artifact: Dict[str, Any], **overrides: Any) -> Arm:
    runtime = artifact["runtime"]
    defaults: Dict[str, Any] = dict(
        artifact_id=artifact["id"], runtime=runtime,
        server_name="llama.cpp" if runtime == "llamacpp" else "mlx_lm",
        backend="llamacpp" if runtime == "llamacpp" else "mlx",
        kv_mode="ctk_q8_0" if runtime == "llamacpp" else "native",
        spec_mode="none", reasoning="off",
    )
    defaults.update(overrides)
    return Arm(**defaults)


def build_arms(selection: Dict[str, Any], mode: str, serve_ctx: int = DEFAULT_SERVE_CTX) -> List[Arm]:
    by_id = {a["id"]: a for a in selection["artifacts"]}
    arms: List[Arm] = []

    def add(artifact_id: str, **overrides: Any) -> None:
        artifact = by_id.get(artifact_id)
        if artifact is not None:
            overrides.setdefault("ctx", serve_ctx)
            arms.append(_base_arm(artifact, **overrides))

    if mode == "smoke":
        add(LADDER_MLX[0], group="smoke", note="pipeline validation only")
        return arms

    # --- weight-quantization ladder (both runtimes, their primary paths) ---
    for artifact_id in LADDER_MLX:
        add(artifact_id, group="ladder_mlx",
            note="mlx_lm, native KV — the daily serving path")
    for artifact_id in LADDER_GGUF:
        add(artifact_id, group="ladder_gguf",
            note="llama.cpp at its -ctk/-ctv q8_0 default")
    for artifact_id in METHOD_CONTRAST:
        add(artifact_id, group="method_contrast",
            note="same nominal precision, different conversion method")

    if mode == "core":
        return arms

    # --- KV-cache study, weights held at the pivot -------------------------
    # The mlx_vlm KV-native arm is the control: without it, a q8 arm differs from
    # the mlx_lm ladder by BOTH server and KV mode.
    add(PIVOT_MLX, server_name="mlx_vlm", backend="mlx-kv", kv_mode="native",
        group="kv_study", note="CONTROL: mlx_vlm with a native KV cache")
    add(PIVOT_MLX, server_name="mlx_vlm", backend="mlx-kv", kv_mode="q8_turbo",
        group="kv_study", note="KV 8-bit turboquant")
    add(PIVOT_MLX, server_name="mlx_vlm", backend="mlx-kv", kv_mode="q4_uniform",
        group="kv_study", note="KV 4-bit uniform")
    add(PIVOT_GGUF, kv_mode="ctk_f16", group="kv_study",
        note="CONTROL: llama.cpp KV native (the default is already q8_0)")

    # --- speculative decoding, weights and KV held ------------------------
    add(PIVOT_MLX, server_name="mlx_vlm", backend="mlx-kv", kv_mode="native",
        spec_mode="mtp", draft_artifact_id="mlx_mtp_draft_4bit", group="mtp_study",
        note="MTP drafting against the KV-native mlx_vlm control")
    add(PIVOT_GGUF, spec_mode="mtp", draft_artifact_id="gguf_mtp_draft",
        group="mtp_study", note="llama.cpp --spec-type draft-mtp")

    # --- reasoning mode, everything else held -----------------------------
    # Capped at medium: the serving notes record a reproduced hidden-reasoning loop at
    # low and medium on exact-count prompts, and Pi already caps xhigh at medium.
    # Going higher here would be re-running a known failure, not an experiment.
    for effort in ("low", "medium"):
        add(PIVOT_MLX, reasoning=effort, group="reasoning_study",
            note=f"thinking on at effort={effort}; compare against the think_off ladder arm")

    # --- derivatives, clearly separated -----------------------------------
    for artifact_id in DERIVATIVES:
        add(artifact_id, group="derivative",
            note="abliterated derivative; NOT comparable to a stock-weights result")

    return arms


# ---------------------------------------------------------------------------
# launching
# ---------------------------------------------------------------------------

KV_FLAGS = {
    "native": [],
    "q8_turbo": ["--kv-bits", "8", "--kv-scheme", "turboquant"],
    "q4_uniform": ["--kv-bits", "4", "--kv-scheme", "uniform"],
    "ctk_q8_0": [],                      # serve_local.sh's own default
    "ctk_f16": ["--kv-type", "f16"],
}


def server_command(arm: Arm, model_path: str, draft_path: str, port: int) -> List[str]:
    command = ["bash", SERVE, model_path, str(port), "--backend", arm.backend]
    command += KV_FLAGS.get(arm.kv_mode, [])
    # mlx_vlm with a native KV cache still needs --kv-bits 0 to be selected at all.
    if arm.backend == "mlx-kv" and arm.kv_mode == "native":
        command += ["--kv-bits", "0"]
    if arm.spec_mode == "mtp":
        if not draft_path:
            raise ValueError(f"{arm.arm_id}: spec_mode=mtp with no drafter on disk")
        command += ["--draft-model", draft_path]
        if arm.runtime == "mlx":
            command += ["--draft-kind", "mtp"]
    elif arm.runtime == "llamacpp":
        # Keep a non-MTP llama.cpp arm honestly non-MTP even if the registry
        # flags the family, so the MTP study is the only place drafting appears.
        command += ["--no-mtp"]
    if arm.ctx:
        command += ["--ctx", str(arm.ctx)]
    # Reasoning is an arm variable, so it is stated explicitly on every arm
    # rather than inherited — including the "off" arms, so a think_off result is
    # provably off rather than merely defaulted.
    if arm.reasoning == "off":
        command += ["--thinking", "off"]
    else:
        command += ["--thinking", "on", "--reasoning-effort", arm.reasoning]
    return command


def model_id_for(arm: Arm, model_path: str) -> str:
    """What the server expects in the request's `model` field.

    mlx_vlm.server requires the loaded absolute path; llama-server publishes the
    `--alias` that serve_local.sh sets from the registry name; mlx_lm.server
    accepts the loaded path too. Using the wrong one makes a healthy server
    answer 404, which reads as a dead model.
    """
    if arm.runtime == "llamacpp":
        return Path(model_path).name
    return model_path


# ---------------------------------------------------------------------------
# one arm
# ---------------------------------------------------------------------------

class _NotReady(Exception):
    """Raised to skip the measurement stages while still running teardown."""


def run_arm(
    arm: Arm,
    *,
    artifacts: Dict[str, Dict[str, Any]],
    run_dir: Path,
    endpoints: Dict[str, Any],
    eval_names: List[str],
    ctx_cap: int,
    repeats: int,
    timeout: int,
    ready_timeout: int,
    skip_context_probe: bool,
    verbose: bool,
) -> Dict[str, Any]:
    """Take one configuration through every stage and return its record."""
    artifact = artifacts[arm.artifact_id]
    arm_dir = run_dir / arm.arm_id
    arm_dir.mkdir(parents=True, exist_ok=True)

    record: Dict[str, Any] = {
        "arm_id": arm.arm_id,
        "arm": asdict(arm),
        "artifact": {
            "id": artifact["id"], "repo": artifact["repo"], "revision": artifact["revision"],
            "quant_label": artifact["quant_label"], "quant_method": artifact.get("quant_method", ""),
            "measured_bpw": artifact.get("measured_bpw"),
            "disk_bytes": artifact.get("disk_bytes", 0),
            "local_dir": artifact["local_dir"],
            "modification": artifact.get("modification", ""),
        },
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "eval_names": list(eval_names),
    }

    model_path = _servable_path(artifact)
    if not model_path:
        record.update(status="artifact_missing",
                      error=f"no loadable weights under {artifact['local_dir']}")
        return _finish(record, arm_dir)

    draft_path = ""
    if arm.spec_mode == "mtp":
        drafter = artifacts.get(arm.draft_artifact_id)
        draft_path = _servable_path(drafter) if drafter else ""
        if not draft_path:
            record.update(status="drafter_missing",
                          error=f"drafter {arm.draft_artifact_id} is not on disk")
            return _finish(record, arm_dir)

    host, port = _endpoint_for(arm, endpoints)
    if not server.port_is_free(host, port):
        record.update(status="port_busy",
                      error=f"{host}:{port} is already in use; stop the other server first")
        return _finish(record, arm_dir)

    command = server_command(arm, model_path, draft_path, port)
    record["server_command"] = shlex.join(command)
    record["endpoint"] = f"http://{host}:{port}/v1"
    model_id = model_id_for(arm, model_path)
    record["model_id"] = model_id

    print(f"\n=== {arm.arm_id}", flush=True)
    print(f"    {record['server_command']}", flush=True)

    handle = server.launch(command, host=host, port=port, log_path=arm_dir / "server.log",
                           env=_reasoning_env(arm))
    try:
        problem = server.await_ready(handle, model_id, timeout=ready_timeout)
        if problem:
            # The branch that keeps the matrix honest: a failed launch is recorded
            # as a failed launch, and the eval suite never runs against it.
            #
            # It deliberately does NOT return early. `finally` below still has to
            # tear the server down and attach the memory block, and an early
            # return would write result.json before that happened — leaving the
            # on-disk record missing measurements the in-memory one has.
            record.update(status="startup_failed", error=problem,
                          server_log=str(handle.log_path))
            raise _NotReady

        record["load_seconds"] = handle.load_seconds
        print(f"    ready in {handle.load_seconds}s", flush=True)
        _measure_and_score(
            record, handle, arm, artifact, model_id, arm_dir,
            eval_names=eval_names, ctx_cap=ctx_cap, repeats=repeats,
            timeout=timeout, skip_context_probe=skip_context_probe, verbose=verbose,
        )
    except _NotReady:
        pass  # status and error are already recorded
    except KeyboardInterrupt:
        record.update(status="interrupted")
        raise
    except Exception as exc:  # noqa: BLE001 — one arm must not end the matrix
        record.update(status="harness_error", error=f"{type(exc).__name__}: {exc}"[:400])
    finally:
        sample = server.shutdown(handle)
        record["memory"] = asdict(sample)
        record["memory"]["peak_proc_rss_gb"] = round(sample.peak_proc_rss_bytes / 1e9, 2)
        record["memory"]["attributed_peak_gb"] = round(sample.attributed_peak_bytes / 1e9, 2)
        record["memory"]["_rss_note"] = (
            "peak_proc_rss_gb under-reports MLX (weights sit in Metal buffers). "
            "attributed_peak_gb is system used memory above the pre-launch baseline.")
        record["memory"]["peak_system_used_gb"] = round(sample.peak_used_bytes / 1e9, 2)
        record["memory"]["min_available_gb"] = round(sample.min_available_bytes / 1e9, 2)
    return _finish(record, arm_dir)


def _measure_and_score(
    record: Dict[str, Any],
    handle: server.ServerHandle,
    arm: Arm,
    artifact: Dict[str, Any],
    model_id: str,
    arm_dir: Path,
    *,
    eval_names: List[str],
    ctx_cap: int,
    repeats: int,
    timeout: int,
    skip_context_probe: bool,
    verbose: bool,
) -> None:
    """Systems measurement then the behavioural suite, against a ready server.

    Split out of `run_arm` so the stages can be exercised independently: a test
    can assert that a failed launch never reaches this function at all.
    """
    record["throughput"] = server.measure_throughput(handle.base_url, model_id, timeout=timeout)
    for row in record["throughput"]:
        print(f"    perf/{row['probe']:<7} ttft={row.get('ttft_s')} "
              f"gen_tps={row.get('gen_tps')} prompt_tps={row.get('prompt_tps')} "
              f"[{row['status']}]", flush=True)

    csv_path = run_suite(
        model=model_id,
        eval_names=eval_names,
        output_dir=arm_dir / "quality",
        api_base=handle.base_url,
        backend=arm.server_name,
        quant=artifact["quant_label"],
        ctx_cap=ctx_cap,
        repeats=repeats,
        timeout=timeout,
        verbose=verbose,
        enable_thinking=_client_thinking(arm),
        # A thinking arm needs headroom or it spends its answer budget on the
        # scratchpad and gets recorded as truncated rather than as thinking.
        token_scale=1.0 if arm.reasoning == "off" else 3.0,
        run_kwargs={"transcript_dir": arm_dir / "transcripts"},
    )
    record["quality_csv"] = str(csv_path)
    summary_path = Path(str(csv_path)).with_suffix(".json")
    if summary_path.is_file():
        record["quality"] = json.loads(summary_path.read_text())
        record["status"] = _quality_status(record["quality"])
    else:
        record["status"] = "quality_missing"

    if skip_context_probe:
        record["context"] = {"practical_max_context": None, "stopped_because": "skipped"}
    else:
        record["context"] = server.probe_max_context(
            handle.base_url, model_id, timeout=timeout, max_context=arm.ctx or 0)
        print(f"    practical max context: {record['context']['practical_max_context']} "
              f"({record['context']['stopped_because']})", flush=True)


def _runtime_versions() -> Dict[str, str]:
    """Versions of everything that could change a number between runs."""
    import importlib.metadata as metadata
    versions: Dict[str, str] = {"python": sys.version.split()[0]}
    for distribution in ("mlx", "mlx-lm", "mlx-vlm", "transformers", "huggingface-hub"):
        try:
            versions[distribution] = metadata.version(distribution)
        except Exception:  # noqa: BLE001
            versions[distribution] = "not installed"
    try:
        sys.path.insert(0, str(SCRIPT_DIR))
        import runtime_versions
        versions["llama.cpp"] = runtime_versions._llamacpp_version()
    except Exception as exc:  # noqa: BLE001
        versions["llama.cpp"] = f"unknown ({type(exc).__name__})"
    return versions


def _quality_status(quality: Dict[str, Any]) -> str:
    """`complete` only when every selected eval produced scores."""
    statuses = [e.get("status", "") for e in quality.get("evals", [])]
    if not statuses:
        return "quality_empty"
    if all(s == "ok" for s in statuses):
        return "complete"
    if any(s == "ok" for s in statuses):
        return "partial"
    return "quality_failed"


def _servable_path(artifact: Optional[Dict[str, Any]]) -> str:
    """The path serve_local.sh should be handed for this artifact."""
    if not artifact:
        return ""
    directory = Path(artifact["local_dir"])
    if not directory.is_dir():
        return ""
    if artifact["runtime"] == "llamacpp":
        wanted = [f["path"] for f in artifact.get("files", [])
                  if f["path"].endswith(".gguf")
                  and not Path(f["path"]).name.startswith("mmproj")]
        # A drafter artifact's only file IS the mtp-*.gguf, so it must not be
        # filtered out the way a target model's companion drafter would be.
        if not artifact.get("is_drafter"):
            wanted = [w for w in wanted if not Path(w).name.startswith("mtp-")] or wanted
        for name in wanted:
            candidate = directory / name
            if candidate.is_file():
                return str(candidate)
        return ""
    return str(directory)


def _endpoint_for(arm: Arm, endpoints: Dict[str, Any]) -> Tuple[str, int]:
    key = "llamacpp" if arm.runtime == "llamacpp" else "mlx"
    url = endpoints[key]["base_url"]
    rest = url.split("://", 1)[1].split("/", 1)[0]
    host, _, port = rest.partition(":")
    return host, int(port or 80)


def _reasoning_env(arm: Arm) -> Dict[str, str]:
    """No longer carries reasoning; see `server_command`.

    Kept as the single place a launch environment is assembled. Reasoning moved
    to explicit CLI flags because serve_local.sh re-assigns MODEL_ENABLE_THINKING
    from the registry after the caller's environment is read, which silently
    defeated the environment-variable approach.
    """
    return {}


def _client_thinking(arm: Arm) -> Optional[bool]:
    """What the eval client should send per request.

    `False` on a think_off arm, as belt and braces against a server default.
    `None` on a thinking arm: the client only knows how to send `enable_thinking`,
    and sending it would replace the server's chat-template arguments — taking
    `reasoning_effort` with them and collapsing the low and medium arms into the
    same configuration. Letting the server govern keeps the two arms distinct.
    """
    return False if arm.reasoning == "off" else None


def _finish(record: Dict[str, Any], arm_dir: Path) -> Dict[str, Any]:
    """Write the arm's record.

    Teardown has already run by the time this is called, so what lands on disk is
    exactly what the caller receives.
    """
    record["finished_at"] = datetime.now(timezone.utc).isoformat()
    # Written via a temporary file: a plain write_text that is interrupted leaves
    # truncated JSON behind, and the next --resume then dies reading it rather
    # than simply re-running the arm.
    target = arm_dir / "result.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, default=str) + "\n")
    temporary.replace(target)
    print(f"    → {record['status']}"
          + (f": {record.get('error','')[:160]}" if record.get("error") else ""), flush=True)
    return record


# ---------------------------------------------------------------------------
# the matrix
# ---------------------------------------------------------------------------

def load_artifacts(manifest_path: Path, config_path: Path,
                   selection: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Prefer a written manifest; fall back to inspecting disk offline.

    Offline on purpose: a matrix that has to reach Hugging Face to start is a
    matrix that cannot run on a machine that is mid-download or off the network.
    """
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        merged = {}
        for record in manifest.get("artifacts", []):
            spec = next((a for a in selection["artifacts"] if a["id"] == record["id"]), {})
            merged[record["id"]] = {**record, "is_drafter": bool(spec.get("is_drafter"))}
        return merged
    config = local_config.load_config(config_path)
    root = Path(os.path.expanduser(config["models"]["root"]))
    return {
        a["id"]: {**acquire.inspect_artifact(a, root, with_remote=False),
                  "files": [{"path": f} for f in (a.get("files") or [])],
                  "is_drafter": bool(a.get("is_drafter"))}
        for a in selection["artifacts"]
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["smoke", "core", "full"], default="core")
    parser.add_argument("--run-dir", type=Path,
                        help="Where raw results go. Default: results/qwen38_showdown/<mode>_<stamp>")
    parser.add_argument("--resume", action="store_true",
                        help="Skip arms that already have a complete result.json in --run-dir")
    parser.add_argument("--resume-accept-partial", action="store_true",
                        help="Also skip arms whose previous run only partly scored "
                             "(default: re-run them, so a degraded arm cannot become final)")
    parser.add_argument("--serve-ctx", type=int, default=DEFAULT_SERVE_CTX,
                        help=f"Serving context for every arm and the ceiling of the "
                             f"practical-context probe (default {DEFAULT_SERVE_CTX}). "
                             f"llama.cpp allocates this up front; raise it deliberately.")
    parser.add_argument("--arm", nargs="+", help="Run only these arm id(s)")
    parser.add_argument("--artifact", nargs="+", help="Run only arms using these artifact id(s)")
    parser.add_argument("--group", nargs="+",
                        help="Run only these arm groups: ladder_mlx ladder_gguf method_contrast "
                             "kv_study mtp_study reasoning_study derivative")
    parser.add_argument("--evals", nargs="+",
                        help="Override the eval set (names, or showdown/showdown_smoke/tier1)")
    parser.add_argument("--ctx-cap", type=int, default=131072,
                        help="Upper bound for long-context case sizing")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Trials for the reliability-sensitive evals")
    parser.add_argument("--timeout", type=int, default=900, help="Per-request timeout (s)")
    parser.add_argument("--ready-timeout", type=int, default=900,
                        help="How long to wait for a server to answer its first completion")
    parser.add_argument("--skip-context-probe", action="store_true",
                        help="Skip the practical-max-context ladder (it is the slowest stage)")
    parser.add_argument("--config", type=Path, default=local_config.DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=acquire.DEFAULT_MANIFEST)
    parser.add_argument("--list", action="store_true", help="Print the arms and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print each arm's server command and exit; starts nothing")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    selection = acquire.load_selection()
    arms = build_arms(selection, args.mode, serve_ctx=args.serve_ctx)
    if args.group:
        arms = [a for a in arms if a.group in set(args.group)]
    if args.artifact:
        arms = [a for a in arms if a.artifact_id in set(args.artifact)]
    if args.arm:
        wanted = set(args.arm)
        arms = [a for a in arms if a.arm_id in wanted]
        unknown = wanted - {a.arm_id for a in arms}
        if unknown:
            raise SystemExit(f"Unknown arm id(s): {', '.join(sorted(unknown))}")
    if not arms:
        raise SystemExit("No arms selected.")

    if args.serve_ctx < args.ctx_cap + CTX_HEADROOM:
        adjusted = args.ctx_cap + CTX_HEADROOM
        print(f"NOTE: --serve-ctx {args.serve_ctx} leaves no room for the answer on a "
              f"{args.ctx_cap}-token case; raising it to {adjusted}. "
              f"Lower --ctx-cap instead if you meant to serve smaller.")
        args.serve_ctx = adjusted
        arms = build_arms(selection, args.mode, serve_ctx=args.serve_ctx)
        if args.group:
            arms = [a for a in arms if a.group in set(args.group)]
        if args.artifact:
            arms = [a for a in arms if a.artifact_id in set(args.artifact)]
        if args.arm:
            arms = [a for a in arms if a.arm_id in set(args.arm)]

    eval_names = args.evals or (SHOWDOWN_SMOKE if args.mode == "smoke" else SHOWDOWN)
    artifacts = load_artifacts(args.manifest, args.config, selection)

    if args.list:
        print(f"{'ARM':<62} {'GROUP':<16} {'ON DISK':<9} NOTE")
        for arm in arms:
            record = artifacts.get(arm.artifact_id, {})
            state = "yes" if _servable_path(record) else "MISSING"
            print(f"{arm.arm_id:<62} {arm.group:<16} {state:<9} {arm.note}")
        print(f"\n{len(arms)} arm(s) · evals: {', '.join(eval_names)}")
        return

    run_dir = args.run_dir or (DEFAULT_RUN_ROOT /
                               f"{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    config = local_config.load_config(args.config)
    endpoints = config["endpoints"]

    if args.dry_run:
        for arm in arms:
            record = artifacts.get(arm.artifact_id, {})
            path = _servable_path(record)
            draft = _servable_path(artifacts.get(arm.draft_artifact_id, {})) if arm.draft_artifact_id else ""
            host, port = _endpoint_for(arm, endpoints)
            if not path:
                print(f"{arm.arm_id}\n    SKIP: artifact not on disk")
                continue
            try:
                print(f"{arm.arm_id}\n    {shlex.join(server_command(arm, path, draft, port))}")
            except ValueError as exc:
                print(f"{arm.arm_id}\n    SKIP: {exc}")
        print(f"\nDry run — nothing started. {len(arms)} arm(s).")
        return

    state_path = run_dir / "state.json"
    state: Dict[str, Any] = {
        "experiment": selection["experiment"],
        "mode": args.mode,
        "evals": eval_names,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "arms": {a.arm_id: "queued" for a in arms},
        "base_model": selection["base_model"],
        "serving_support": selection.get("serving_support", {}),
        "serve_ctx": args.serve_ctx,
        "ctx_cap": args.ctx_cap,
        "repeats": args.repeats,
        # Without these a result cannot be compared against a later run: the same
        # artifact served by a different llama.cpp build is a different measurement.
        "runtime_versions": _runtime_versions(),
    }
    if args.resume and state_path.is_file():
        state.update(json.loads(state_path.read_text()))
        state["arms"].update({a.arm_id: state["arms"].get(a.arm_id, "queued") for a in arms})

    def save() -> None:
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        temporary = state_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, indent=2, default=str) + "\n")
        temporary.replace(state_path)

    save()
    print(f"Run dir : {run_dir}")
    print(f"Arms    : {len(arms)} · mode={args.mode}")
    print(f"Evals   : {', '.join(eval_names)}")

    completed = 0
    try:
        for index, arm in enumerate(arms, start=1):
            existing = run_dir / arm.arm_id / "result.json"
            if args.resume and existing.is_file():
                try:
                    previous = json.loads(existing.read_text())
                except (OSError, json.JSONDecodeError) as exc:
                    # An unreadable record is not a completed arm.
                    print(f"[{index}/{len(arms)}] {arm.arm_id}: unreadable previous result "
                          f"({exc}); re-running")
                    previous = {}
                # `partial` means some evals produced no scores. Skipping it would
                # silently bake a degraded arm into the final matrix, so by default
                # only a fully `complete` arm is accepted as done.
                accepted = {"complete", "partial"} if args.resume_accept_partial else {"complete"}
                if previous.get("status") in accepted:
                    print(f"[{index}/{len(arms)}] {arm.arm_id}: resume — already {previous['status']}")
                    state["arms"][arm.arm_id] = previous["status"]
                    completed += 1
                    save()
                    continue
                if previous.get("status") == "partial":
                    print(f"[{index}/{len(arms)}] {arm.arm_id}: previous run was partial; "
                          f"re-running (pass --resume-accept-partial to keep it)")
            state["arms"][arm.arm_id] = "running"
            save()
            print(f"\n[{index}/{len(arms)}]", end="")
            record = run_arm(
                arm, artifacts=artifacts, run_dir=run_dir, endpoints=endpoints,
                eval_names=eval_names, ctx_cap=args.ctx_cap, repeats=args.repeats,
                timeout=args.timeout, ready_timeout=args.ready_timeout,
                skip_context_probe=args.skip_context_probe, verbose=args.verbose,
            )
            state["arms"][arm.arm_id] = record["status"]
            completed += 1
            save()
    except KeyboardInterrupt:
        state["interrupted"] = True
        print("\nInterrupted. Completed arms are saved; re-run with --resume.")
    finally:
        state["finished_at"] = datetime.now(timezone.utc).isoformat()
        state["completed_arms"] = completed
        state["exhaustive"] = (completed == len(arms) and not state.get("interrupted"))
        save()

    print(f"\nState → {state_path}")
    if not state.get("exhaustive"):
        print("This run is NOT exhaustive. Do not report it as a full matrix.")
    print(f"Report with:  .venv/bin/python scripts/showdown_report.py {run_dir}")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    main()
