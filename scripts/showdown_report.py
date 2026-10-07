#!/usr/bin/env python3
"""
showdown_report.py — turn a matrix run into a comparable table and a recommendation.

Reads the per-arm `result.json` files a run left behind and emits Markdown plus a
machine-readable summary. It deliberately does several things that a naive
aggregator does not:

  * **Failed launches stay visible and stay unscored.** An arm that never became
    ready appears in its own section with its log path, never as a row with a
    zero in the quality columns.

  * **Capability loss is measured against the best *tested* configuration**, not
    against an absolute. The best arm in a run is the reference; everything else
    is reported as a delta from it, and the reference is named.

  * **Derivative models are in a separate table.** An abliterated fine-tune is
    not a data point about quantization, and putting it in the same table invites
    exactly that reading.

  * **Nothing is ranked on throughput.** The recommendations weigh behavioural
    capability first and use speed and memory as tie-breakers and constraints,
    because a fast configuration that loses tool calling is not a faster version
    of the same assistant.

  * **Numbers that are not trustworthy are not printed as numbers.** A capability
    measured over fewer than a threshold of scorable cases is shown as `n/a` with
    the reason, and an arm whose evals partly failed is labelled `partial`.

Usage
-----
    python3 scripts/showdown_report.py results/qwen38_showdown/core_20260918_120000
    python3 scripts/showdown_report.py <run_dir> --out REPORT.md --json summary.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]

# Behavioural axes, in the order the table reads. The key is the eval name; the
# label is the column head.
AXES: Tuple[Tuple[str, str], ...] = (
    ("toolcall", "Tool calls"),
    ("agentic", "Agent tasks"),
    ("reasoning_hard", "Reasoning"),
    ("grounding", "Grounding"),
    ("longctx", "Long ctx"),
    ("ifeval_local", "Instructions"),
    ("niah", "Retrieval"),
    ("determinism", "Stability"),
)

# Below this many scorable cases an axis is reported as n/a rather than as a
# number. Three cases cannot distinguish 0.67 from 1.00 in any useful way.
MIN_SCORABLE = 4

# Below this many scoring stock arms there is no comparison to report, only a
# measurement of whichever arm happened to run.
MIN_ARMS_TO_COMPARE = 3

RUNNABLE = {"complete", "partial"}


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_run(run_dir: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    state_path = run_dir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    records = []
    for result in sorted(run_dir.glob("*/result.json")):
        try:
            records.append(json.loads(result.read_text()))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"warning: unreadable {result}: {exc}", file=sys.stderr)
    return state, records


def axis_scores(record: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Strict pass rate per eval, or None when there is too little to say."""
    out: Dict[str, Optional[float]] = {}
    for spec in (record.get("quality") or {}).get("evals", []):
        name = spec.get("eval")
        if spec.get("status") != "ok" or int(spec.get("scored") or 0) < MIN_SCORABLE:
            out[name] = None
            continue
        out[name] = float(spec.get("strict_pass_rate") or 0.0)
    return out


def axes_used(record: Dict[str, Any]) -> List[str]:
    return sorted(k for k, v in axis_scores(record).items() if v is not None)


def capability_index(record: Dict[str, Any]) -> Optional[float]:
    """Mean of the available behavioural axes.

    An unweighted mean on purpose. Any weighting here would be this report
    asserting a preference the operator has not stated, and the three
    recommendations below re-weight explicitly and say so.
    """
    scores = [v for v in axis_scores(record).values() if v is not None]
    return round(statistics.fmean(scores), 4) if len(scores) >= 3 else None


def throughput_of(record: Dict[str, Any], probe: str = "short") -> Dict[str, Any]:
    for row in record.get("throughput") or []:
        if row.get("probe") == probe:
            return row
    return {}


def _prompt_tps_cell(perf: Dict[str, Any]) -> str:
    """Prefill rate, marked when the token count behind it was inferred.

    llama.cpp reports `usage.prompt_tokens` in its stream, so its rate is
    measured end to end. mlx_lm.server reports nothing, so the count is derived
    from a tokens-per-word ratio calibrated once on the short probe. Both are
    useful; printing them identically would let a derived number be read as a
    measured one when the two runtimes are compared side by side.
    """
    value = perf.get("prompt_tps")
    if not value:
        return "n/a"
    source = perf.get("prompt_token_source")
    return f"{value}~" if source == "calibrated" else str(value)


def _fmt(value: Optional[float], digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _gb(value: Optional[float]) -> str:
    return "n/a" if not value else f"{value:.1f}"


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

MATRIX_HEADERS = [
    "Arm", "Model / derivative", "Runtime", "Weight quant", "KV mode", "Gen mode",
    "Artifact @ revision", "Disk GB", "Peak UM GB", "Usable ctx", "Prompt t/s",
    "Gen t/s", "TTFT s",
] + [label for _, label in AXES] + ["Capability", "Δ vs best", "Notes"]


def matrix_row(record: Dict[str, Any], reference: Optional[float]) -> List[str]:
    arm = record["arm"]
    artifact = record["artifact"]
    scores = axis_scores(record)
    perf = throughput_of(record)
    memory = record.get("memory") or {}
    context = record.get("context") or {}
    index = capability_index(record)
    delta = ("n/a" if index is None or not reference
             else f"{(index - reference) * 100:+.1f} pts")
    notes = []
    if record["status"] == "partial":
        failed = [e["eval"] for e in (record.get("quality") or {}).get("evals", [])
                  if e.get("status") != "ok"]
        notes.append("partial: " + ",".join(failed[:4]))
    if context.get("stopped_because") not in (None, "ok", "skipped"):
        notes.append(f"ctx stop: {context['stopped_because']}")
    if memory.get("swap_delta_bytes", 0) > 2e9:
        notes.append(f"swapped +{memory['swap_delta_bytes']/1e9:.1f} GB")
    if arm.get("note"):
        notes.append(arm["note"])

    return [
        record["arm_id"],
        artifact["id"],
        arm["server_name"],
        artifact["quant_label"],
        arm["kv_mode"],
        arm["spec_mode"] + (f"/think_{arm['reasoning']}" if arm["reasoning"] != "off" else ""),
        f"{artifact['repo']}@{artifact['revision'][:8]}",
        _gb(artifact.get("disk_bytes", 0) / 1e9 if artifact.get("disk_bytes") else None),
        _gb(_peak_gb(record) or None),
        str(context.get("practical_max_context") or "n/a"),
        _prompt_tps_cell(perf),
        str(perf.get("gen_tps") or "n/a"),
        str(perf.get("ttft_s") or "n/a"),
    ] + [_fmt(scores.get(name)) for name, _ in AXES] + [
        _fmt(index), delta, "; ".join(notes) or "—",
    ]


def render_table(headers: List[str], rows: List[List[str]]) -> str:
    if not rows:
        return "_No rows._\n"
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# recommendations
# ---------------------------------------------------------------------------

def _usable_ctx(record: Dict[str, Any]) -> int:
    return int((record.get("context") or {}).get("practical_max_context") or 0)


def _gen_tps(record: Dict[str, Any]) -> float:
    return float(throughput_of(record).get("gen_tps") or 0.0)


def _peak_gb(record: Dict[str, Any]) -> float:
    """Peak unified memory attributable to this arm.

    Prefers the system-used delta over process RSS: MLX allocates weights through
    Metal buffers that never enter the resident set, so RSS under-reports an MLX
    arm by more than half. Falls back to RSS when the delta is unavailable.
    """
    memory = record.get("memory") or {}
    attributed = memory.get("attributed_peak_gb")
    if attributed:
        return float(attributed)
    return float(memory.get("peak_proc_rss_gb") or 0.0)


def _reliable(record: Dict[str, Any]) -> bool:
    """No intermittent agent task and no serving failure. A configuration that
    completes an agent task sometimes is unusable for unattended work regardless
    of its averages, so this gates every recommendation."""
    if record["status"] != "complete":
        return False
    for spec in (record.get("quality") or {}).get("evals", []):
        if spec.get("eval") == "agentic":
            breakdown = spec.get("breakdown") or {}
            if breakdown.get("intermittent_tasks"):
                return False
    return True


def recommend(records: List[Dict[str, Any]], *, memory_budget_gb: float = 64.0) -> Dict[str, Any]:
    """Three recommendations with explicitly different objectives.

    None of them is a ranking of one number. Each states its own rule so the
    choice can be disagreed with rather than merely trusted.
    """
    stock = [r for r in records
             if r["status"] in RUNNABLE and r["arm"].get("group") != "derivative"
             and capability_index(r) is not None]
    if not stock:
        return {"error": "no stock arm produced a scorable capability index"}
    if len(stock) < MIN_ARMS_TO_COMPARE:
        # "Best Quality: <the only arm that ran>" is not a finding, and a reader
        # skimming the headings would take it for one. Refuse rather than rank.
        return {
            "error": (f"only {len(stock)} stock arm(s) produced scores; "
                      f"{MIN_ARMS_TO_COMPARE} are needed before one configuration can be "
                      f"recommended over another"),
            "scored_arms": [r["arm_id"] for r in stock],
        }

    def best(candidates: List[Dict[str, Any]], key) -> Optional[Dict[str, Any]]:
        return max(candidates, key=key) if candidates else None

    quality = best(stock, lambda r: (capability_index(r), _usable_ctx(r), _gen_tps(r)))

    # Daily driver: must be reliable, must leave real headroom on a 64 GB
    # machine (a model that peaks at 45 GB leaves nothing for the rest of the
    # desktop), then the best capability, then speed.
    headroom_gb = memory_budget_gb * 0.55
    daily_pool = [r for r in stock if _reliable(r) and 0 < _peak_gb(r) <= headroom_gb] or \
                 [r for r in stock if _reliable(r)] or stock
    daily = best(daily_pool, lambda r: (round(capability_index(r), 2), _gen_tps(r)))

    # Long context / agent workspace: context first, then agent + tool axes,
    # then whatever capability is left. Explicitly NOT the capability ranking.
    def agent_axis(record: Dict[str, Any]) -> float:
        scores = axis_scores(record)
        parts = [scores.get("agentic"), scores.get("toolcall"), scores.get("longctx")]
        present = [p for p in parts if p is not None]
        return statistics.fmean(present) if present else 0.0

    workspace = best(stock, lambda r: (_usable_ctx(r), round(agent_axis(r), 2),
                                       -_peak_gb(r), capability_index(r)))

    reference = max(capability_index(r) for r in stock)

    def describe(record: Optional[Dict[str, Any]], rule: str) -> Optional[Dict[str, Any]]:
        if record is None:
            return None
        index = capability_index(record)
        return {
            "arm_id": record["arm_id"],
            "artifact": record["artifact"]["id"],
            "repo": f"{record['artifact']['repo']}@{record['artifact']['revision'][:8]}",
            "serve": record.get("server_command", ""),
            "capability_index": index,
            "capability_loss_vs_best_pts": (None if index is None
                                            else round((reference - index) * 100, 1)),
            "peak_unified_memory_gb": _peak_gb(record) or None,
            "usable_context": _usable_ctx(record) or None,
            "gen_tps": _gen_tps(record) or None,
            "reliable": _reliable(record),
            "rule": rule,
        }

    picks = {
        "best_quality": describe(quality, "highest capability index among stock arms"),
        "best_daily_driver": describe(
            daily,
            f"reliable (no intermittent agent task), peak memory <= {headroom_gb:.0f} GB, "
            f"then capability rounded to 2 dp, then generation speed"),
        "best_long_context_workspace": describe(
            workspace,
            "largest verified usable context, then the agent/tool/long-context axes, "
            "then lowest peak memory"),
    }
    # `describe` output, not the raw records: the raw record's "artifact" is a
    # dict and cannot be compared for distinctness.
    return {
        "reference_capability_index": reference,
        **picks,
        "keep_multiple": _keep_multiple(*picks.values()),
    }


def _keep_multiple(*chosen: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Whether the three roles actually want different artifacts on disk."""
    picked = [c for c in chosen if c]
    artifacts = [c["artifact"] for c in picked]
    distinct = sorted(set(artifacts))
    return {
        "distinct_artifacts": distinct,
        "verdict": ("one artifact serves all three roles" if len(distinct) <= 1
                    else f"keep {len(distinct)} artifacts on disk: {', '.join(distinct)}"),
    }


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def render(run_dir: Path, state: Dict[str, Any], records: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    scored = [r for r in records if r["status"] in RUNNABLE]
    failed = [r for r in records if r["status"] not in RUNNABLE]
    stock = [r for r in scored if r["arm"].get("group") != "derivative"]
    derivative = [r for r in scored if r["arm"].get("group") == "derivative"]
    indices = [capability_index(r) for r in stock]
    reference = max([i for i in indices if i is not None], default=None)

    lines: List[str] = []
    lines.append("# Qwen3.8-27B configuration showdown\n")
    lines.append(f"- Run: `{run_dir}`")
    lines.append(f"- Mode: `{state.get('mode', '?')}` · evals: {', '.join(state.get('evals', []))}")
    lines.append(f"- Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    lines.append(f"- Arms: {len(records)} recorded, {len(scored)} produced scores, "
                 f"{len(failed)} did not run to completion")
    exhaustive = bool(state.get("exhaustive"))
    lines.append(f"- Exhaustive: **{'yes' if exhaustive else 'no'}**"
                 + ("" if exhaustive else " — this is a partial matrix and must not be "
                                          "described as a complete one"))
    serve_ctx = state.get("serve_ctx")
    if serve_ctx:
        # Usable context can never exceed what the servers were told to allocate,
        # so the cap has to be visible next to the column it bounds.
        lines.append(f"- Served context: {serve_ctx} on every arm "
                     f"(eval sizing cap {state.get('ctx_cap', '?')}, "
                     f"repeats {state.get('repeats', '?')}). "
                     f"`Usable ctx` cannot exceed this.")
    versions = state.get("runtime_versions") or {}
    if versions:
        lines.append("- Runtimes: " + ", ".join(f"{k} {v}" for k, v in sorted(versions.items())))
    lines.append("")

    base = state.get("base_model") or {}
    if base.get("_arch_notes"):
        lines.append("## Architecture caveat\n")
        for note in base["_arch_notes"][:2]:
            lines.append(f"- {note}")
        lines.append("")

    lines.append("## Stock quantization matrix\n")
    lines.append(render_table(MATRIX_HEADERS, [matrix_row(r, reference) for r in stock]))
    lines.append("A `~` after `Prompt t/s` means the prompt-token count behind it was "
                 "calibrated from a tokens-per-word ratio rather than reported by the server "
                 "(mlx_lm emits no streaming usage; llama.cpp does). Compare marked against "
                 "marked.\n")
    lines.append("`Peak UM GB` is peak system unified memory above the pre-launch baseline, "
                 "which is the right figure on Apple Silicon: MLX weights live in Metal buffers "
                 "and barely appear in process RSS. It is an attribution, so anything else "
                 "running during an arm moves it.\n")
    lines.append("`Capability` is the unweighted mean of the behavioural axes that produced at "
                 f"least {MIN_SCORABLE} scorable cases; `n/a` means too little evidence, not zero. "
                 "`Δ vs best` is measured against the best arm **in this run**, which is a "
                 "relative statement about the configurations tested here.\n")

    coverage = {r["arm_id"]: axes_used(r) for r in stock if capability_index(r) is not None}
    distinct_coverage = {tuple(v) for v in coverage.values()}
    if len(distinct_coverage) > 1:
        lines.append("## Capability indices cover different axes\n")
        lines.append(
            "An eval that aborted contributes nothing rather than a zero, which is correct — "
            "but it means these arms' capability indices are means over **different axis sets** "
            "and are not directly comparable. Compare the per-axis columns, not the index.\n")
        for arm_id, axes in sorted(coverage.items()):
            missing = sorted({a for _n, _l in AXES for a in [_n]} - set(axes))
            if missing:
                lines.append(f"- `{arm_id}` is missing: {', '.join(missing)}")
        lines.append("")

    mlx_vlm_arms = [r for r in stock if r["arm"].get("server_name") == "mlx_vlm"]
    if mlx_vlm_arms:
        lines.append("## Read KV and MTP arms against their own control\n")
        lines.append(
            "`mlx_vlm.server` accepts no `--temp/--top-p/--top-k` or repetition-penalty "
            "flags, so those arms run with the server's own sampling defaults while the "
            "`mlx_lm` ladder runs with the registry's (top_k 20, repeat 1.05, presence 0.2). "
            "Temperature and top_p are sent per request and so are controlled everywhere; "
            "top_k and the penalties are not. Repetition penalty is known to matter for this "
            "family.\n")
        lines.append(
            "Consequence: a `Δ vs best` for an `mlx_vlm` arm mixes the KV or MTP effect with "
            "a server and sampling change. **Compare within the group** — each KV and MTP arm "
            "has a KV-native `mlx_vlm` control arm in the same group, and that pairwise "
            "difference is the clean measurement. Cross-group deltas for these arms are "
            "directional only.\n")
        lines.append("Affected arms: "
                     + ", ".join(f"`{r['arm_id']}`" for r in mlx_vlm_arms) + "\n")

    if derivative:
        lines.append("## Derivative models (separate study)\n")
        lines.append("These are refusal-ablated third-party fine-tunes. They are **not** data "
                     "points about quantization and are tabulated apart for that reason.\n")
        lines.append(render_table(MATRIX_HEADERS, [matrix_row(r, reference) for r in derivative]))

    if failed:
        lines.append("## Arms that did not produce scores\n")
        lines.append(render_table(
            ["Arm", "Status", "Detail", "Server log"],
            [[r["arm_id"], r["status"], (r.get("error") or "")[:180],
              r.get("server_log", str(run_dir / r["arm_id"] / "server.log"))] for r in failed]))
        lines.append("A failed launch is recorded here and **excluded from every score**. "
                     "It is not a model that answered badly.\n")

    # -- per-axis detail that the wide table cannot carry -------------------
    lines.append("## Notable behavioural detail\n")
    any_detail = False
    for record in scored:
        bullets = _detail_bullets(record)
        if bullets:
            any_detail = True
            lines.append(f"**{record['arm_id']}**")
            lines += [f"- {b}" for b in bullets]
            lines.append("")
    if not any_detail:
        lines.append("_Nothing flagged._\n")

    recommendations = recommend(scored)
    lines.append("## Recommendations for this 64 GB machine\n")
    if "error" in recommendations:
        lines.append(f"**Not derivable:** {recommendations['error']}.\n")
        if recommendations.get("scored_arms"):
            lines.append("Arms that did score: "
                         + ", ".join(f"`{a}`" for a in recommendations["scored_arms"]) + "\n")
    else:
        for key, title in (("best_quality", "Best Quality"),
                           ("best_daily_driver", "Best Daily Driver"),
                           ("best_long_context_workspace", "Best Long-Context / Agent Workspace")):
            pick = recommendations.get(key)
            lines.append(f"### {title}\n")
            if not pick:
                lines.append("_No arm qualified._\n")
                continue
            lines.append(f"**{pick['arm_id']}** — `{pick['repo']}`\n")
            lines.append(f"- capability index {_fmt(pick['capability_index'])} "
                         f"({pick['capability_loss_vs_best_pts']} pts below the best tested arm)")
            lines.append(f"- peak unified memory {_gb(pick['peak_unified_memory_gb'])} GB · "
                         f"usable context {pick['usable_context'] or 'n/a'} · "
                         f"{pick['gen_tps'] or 'n/a'} gen tok/s")
            lines.append(f"- reliable across agent trials: {'yes' if pick['reliable'] else 'NO'}")
            lines.append(f"- selection rule: {pick['rule']}")
            if pick["serve"]:
                lines.append(f"\n```bash\n{pick['serve']}\n```\n")
            lines.append("")
        lines.append(f"### Keeping more than one\n\n{recommendations['keep_multiple']['verdict']}\n")

    summary = {
        "run_dir": str(run_dir),
        "mode": state.get("mode"),
        "exhaustive": exhaustive,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reference_capability_index": reference,
        "serve_ctx": state.get("serve_ctx"),
        "ctx_cap": state.get("ctx_cap"),
        "repeats": state.get("repeats"),
        "runtime_versions": state.get("runtime_versions"),
        "arms": [
            {
                "arm_id": r["arm_id"], "status": r["status"], "group": r["arm"].get("group"),
                "artifact": r["artifact"], "arm": r["arm"],
                "capability_index": capability_index(r),
                "axes": axis_scores(r),
                "axes_used": axes_used(r),
                "throughput": r.get("throughput"),
                "memory": r.get("memory"),
                "context": r.get("context"),
                "load_seconds": r.get("load_seconds"),
                "error": r.get("error"),
            }
            for r in records
        ],
        "recommendations": recommendations,
    }
    return "\n".join(lines), summary


def _detail_bullets(record: Dict[str, Any]) -> List[str]:
    """The qualitative failures worth naming, pulled out of each summarizer."""
    bullets: List[str] = []
    for spec in (record.get("quality") or {}).get("evals", []):
        name = spec.get("eval")
        breakdown = spec.get("breakdown") or {}
        if name == "agentic":
            for task, info in (breakdown.get("by_task") or {}).items():
                if info["verdict"] == "intermittent":
                    bullets.append(f"agent task `{task}` succeeded {info['passed']}/{info['trials']} "
                                   f"— intermittent, not a capability you can rely on")
                elif info["verdict"] == "never":
                    bullets.append(f"agent task `{task}` never succeeded ({info['trials']} trials)")
            loops = (breakdown.get("stop_reasons") or {}).get("repetition_loop", 0)
            if loops:
                bullets.append(f"{loops} agent trajector(ies) ended in a repetition loop")
        if name == "longctx":
            if breakdown.get("fabricated_answers"):
                bullets.append(f"{breakdown['fabricated_answers']} fabricated long-context answer(s)")
            if breakdown.get("misattributed_answers"):
                bullets.append(f"{breakdown['misattributed_answers']} misattributed value(s) "
                               f"(returned a fact planted for a different entity)")
            if breakdown.get("usable_context"):
                bullets.append(f"long-context synthesis still correct at "
                               f"{breakdown['usable_context']} tokens")
        if name == "grounding":
            rate = breakdown.get("hallucination_rate")
            if rate:
                bullets.append(f"answered {rate:.0%} of deliberately unanswerable questions")
            over = breakdown.get("over_refusal_rate")
            if over:
                bullets.append(f"over-refused {over:.0%} of questions the context did answer")
        if name == "toolcall":
            for group, info in (breakdown.get("by_group") or {}).items():
                if info["passed"] < info["of"]:
                    bullets.append(f"tool-use `{group}`: {info['passed']}/{info['of']}")
        if name == "reasoning_hard":
            if breakdown.get("flaky"):
                bullets.append("flaky reasoning: " + ", ".join(
                    f"{k} {v}" for k, v in list(breakdown["flaky"].items())[:4]))
            if breakdown.get("distractor_captures"):
                bullets.append(f"{breakdown['distractor_captures']} answer(s) used a planted distractor")
    return bullets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, help="Markdown output (default: <run_dir>/REPORT.md)")
    parser.add_argument("--json", type=Path, help="Summary JSON (default: <run_dir>/summary.json)")
    parser.add_argument("--print", action="store_true", help="Also write the report to stdout")
    args = parser.parse_args()

    if not args.run_dir.is_dir():
        raise SystemExit(f"Not a run directory: {args.run_dir}")
    state, records = load_run(args.run_dir)
    if not records:
        raise SystemExit(f"No arm results found under {args.run_dir}")

    markdown, summary = render(args.run_dir, state, records)
    out = args.out or (args.run_dir / "REPORT.md")
    summary_path = args.json or (args.run_dir / "summary.json")
    out.write_text(markdown)
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    if args.print:
        print(markdown)
    print(f"Report  → {out}")
    print(f"Summary → {summary_path}")


if __name__ == "__main__":
    main()
