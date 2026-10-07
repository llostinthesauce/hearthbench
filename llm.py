#!/usr/bin/env python3
"""
llm — unified local inference control plane

  llm serve  [selector] [--backend] [--port] [--host] [--dry-run]
  llm stop   [--port N | --all]
  llm use    <selector> [--backend] [--port]       # stop, then serve
  llm restart [--port N]                           # same model, fresh server
  llm bench  [--results-dir] [--max-pass]
  llm smoke  [--dry-run]
  llm list   [--backend]
  llm status
  llm doctor [--json]
  llm models sync|list
  llm web    [--host] [--port] [--no-open]

Delegates to existing scripts — nothing is re-implemented here.
  serve  → scripts/serve_local.sh  (os.execv; signal goes straight to server)
  menu   → scripts/llama_serve_menu.py  (interactive model+backend picker)
  bench  → bench_tui.py            (interactive five-backend TUI)
  smoke  → scripts/smoke_test.py
  list   → scripts/model_registry.py   (imported directly)
  status → HTTP health probes on inference ports
  web    → webgui/serve.py          (browser GUI over serve/status/chat)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import model_registry  # noqa: E402  (after sys.path insert)
import discover_models  # noqa: E402
import local_config  # noqa: E402
import local_doctor  # noqa: E402
import runtime_versions  # noqa: E402
import benchmark_profiles  # noqa: E402
import shortcuts  # noqa: E402
import serving_lifecycle  # noqa: E402
import serving_runtime  # noqa: E402

SERVE_SH   = SCRIPTS / "serve_local.sh"
MENU_PY    = SCRIPTS / "llama_serve_menu.py"
BENCH_TUI  = ROOT / "bench_tui.py"
SMOKE_PY   = SCRIPTS / "smoke_test.py"
WEB_PY     = ROOT / "webgui" / "serve.py"

SERVICES = (
    ("llamacpp", "llama.cpp"),
    ("mlx", "direct MLX"),
    ("lmstudio", "LM Studio"),
)

BACKENDS = ["llamacpp", "mlx", "mlx-kv", "mlx-vlm"]


def _python() -> str:
    venv = ROOT / ".venv" / "bin" / "python3"
    return str(venv) if venv.is_file() else sys.executable


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def _probe(base_url: str, headers: dict[str, str] | None = None) -> tuple[bool, int | None]:
    base_url = base_url.rstrip("/")
    root_url = base_url[:-3] if base_url.endswith("/v1") else base_url
    for url in (f"{root_url}/health", f"{base_url}/models"):
        try:
            request = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(request, timeout=2) as r:
                return True, r.status
        except Exception:
            pass
    return False, None


def collect_status(
    config: dict,
    probe=_probe,
) -> list[tuple[str, str, bool, int | None]]:
    rows: list[tuple[str, str, bool, int | None]] = []
    for key, label in SERVICES:
        url = local_config.endpoint(config, key)
        up, code = probe(url, {})
        rows.append((label, url, up, code))
    return rows


def _load_local(args: argparse.Namespace) -> dict:
    return local_config.load_config(getattr(args, "config", local_config.DEFAULT_CONFIG))


def cmd_status(args: argparse.Namespace) -> None:
    """What is running, and which model is loaded on it.

    Up/down alone was not enough to answer the question it looks like it
    answers: two servers can both be "UP" on :8085 across a restart while
    serving different models.
    """
    print("Local inference status")
    print("─" * 78)
    print(f"  {'ENDPOINT':<13} {'STATE':<12} {'MODEL':<34} PORT")
    print("  " + "─" * 74)
    for state in serving_lifecycle.inspect():
        if state.up:
            label = "serving"
        elif state.listening:
            label = "loading"  # bound the port, not answering yet
        else:
            label = "stopped"
        print(
            f"  {state.label:<13} {label:<12} "
            f"{serving_lifecycle.model_label(state.model):<34} {state.port}"
        )
    print()
    report = serving_runtime.status()
    versions = report.get("versions") or {}
    if report.get("ok") and versions:
        print(
            f"  runtime: mlx {versions.get('mlx')} · mlx-lm {versions.get('mlx_lm')} "
            f"· all {len(report.get('applied') or [])} serving fixes applied"
        )
    else:
        missing = [entry["key"] for entry in report.get("missing") or []]
        detail = ", ".join(missing) if missing else report.get("reason", "unknown")
        print(f"  runtime: PROBLEM — {detail}.  Run `llm doctor` for detail.")


def cmd_stop(args: argparse.Namespace) -> None:
    """Stop model servers started from this machine.

    :1234 is skipped unless named explicitly: LM Studio manages that port and
    the owner did not start it from here.
    """
    if args.port:
        targets = [int(args.port)]
    elif args.all:
        targets = [state.port for state in serving_lifecycle.inspect() if state.owned]
        if not targets:
            print("Nothing to stop.")
            return
    else:
        raise SystemExit("Pass --port N or --all. `llm status` shows what is running.")

    failed = False
    for port in targets:
        ok, message = serving_lifecycle.stop(port)
        print(("  " if ok else "  FAILED: ") + message)
        failed = failed or not ok
    raise SystemExit(1 if failed else 0)


def cmd_use(args: argparse.Namespace) -> None:
    """Switch the model on a port: stop what is there, then serve the new one.

    This is the "select a model" verb. It execs `serve` at the end, so the
    server takes over this process and Ctrl-C reaches it directly.
    """
    port = int(args.port or 8085)
    ok, message = serving_lifecycle.stop(port)
    if not ok:
        raise SystemExit(message)
    print(f"  {message}")
    args.port = str(port)
    cmd_serve(args)


def cmd_restart(args: argparse.Namespace) -> None:
    """Restart a port with whatever it is already serving.

    Useful after interrupting a generation: an MLX server that served an
    aborted request has been observed running two orders of magnitude slower
    on prompt processing until restarted.
    """
    port = int(args.port or 8085)
    _, model = serving_lifecycle.probe(port)
    if not model:
        raise SystemExit(
            f"Nothing is serving on {port}, or it does not report a model. "
            f"Use `llm use <model> --port {port}` to choose one explicitly."
        )
    print(f"  restarting {port} with {serving_lifecycle.model_label(model)}")
    args.selector = model
    args.port = str(port)
    cmd_use(args)


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

def cmd_list(args: argparse.Namespace) -> None:
    backend_filter = getattr(args, "backend", "auto")
    registry = Path(_load_local(args)["models"]["registry"])
    rows = model_registry.iter_models(registry)
    if backend_filter != "auto":
        rows = [r for r in rows if r["backend"] == backend_filter]

    print(f"{'BACKEND':<9} {'FAMILY':<22} {'QUANT':<8} {'ON-DISK':<8} ALIASES / PATH")
    print("─" * 100)
    for r in rows:
        aliases = ", ".join(r["aliases"])
        exists  = "yes" if r["exists"] else "MISSING"
        print(f"{r['backend']:<9} {r['family_id']:<22} {r['quant']:<8} {exists:<8} {aliases}")
        print(f"{'':>9} {'':>22} {'':>8} {'':>8} {r['path']}")


def sync_models(config: dict, *, catalog: Path = discover_models.DEFAULT_CATALOG,
                known_only: bool = False) -> int:
    root = Path(config["models"]["root"])
    registry = Path(config["models"]["registry"])
    payload = discover_models.discover([root], catalog, include_unmatched=not known_only)
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(json.dumps(payload, indent=2) + "\n")
    return sum(
        len(family.get("gguf", [])) + len(family.get("mlx", []))
        for family in payload["model_families"]
    )


def cmd_models_inspect(args: argparse.Namespace) -> None:
    from model_policy import resolve_policy
    config = _load_local(args)
    registry = Path(config["models"]["registry"])
    backend = args.backend or _auto_backend(args.selector, registry)
    if backend is None:
        raise SystemExit(f"No model found for {args.selector!r}. Run `llm models sync`.")
    row = model_registry.resolve(args.selector, backend, registry)
    print(json.dumps(resolve_policy(row["path"], backend), indent=2))


def cmd_models_sync(args: argparse.Namespace) -> None:
    config = _load_local(args)
    count = sync_models(config, known_only=args.known_only)
    print(f"Wrote {config['models']['registry']} ({count} complete model variant(s))")


def cmd_doctor(args: argparse.Namespace) -> None:
    checks = local_doctor.run(_load_local(args))
    print(local_doctor.as_json(checks) if args.json else local_doctor.format_text(checks))
    if any(check.severity == "FAIL" for check in checks):
        raise SystemExit(1)


def cmd_update(args: argparse.Namespace) -> None:
    statuses = runtime_versions.check_all()
    print(runtime_versions.as_json(statuses) if args.json else runtime_versions.format_text(statuses))


def cmd_shortcuts_sync(args: argparse.Namespace) -> None:
    changes = shortcuts.sync(apply=args.apply)
    print("\n".join(changes) if changes else "Shortcuts already aligned.")
    if changes and not args.apply:
        print("\nPreview only. Re-run with --apply to write timestamped, recoverable backups.")


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------

def _auto_backend(selector: str, registry: Path) -> str | None:
    """Return the first on-disk backend for selector, or None."""
    for backend in ("llamacpp", "mlx"):
        try:
            row = model_registry.resolve(selector, backend, registry)
            if row.get("exists"):
                return backend
        except SystemExit:
            pass
    return None


def cmd_serve(args: argparse.Namespace) -> None:
    selector: str | None = args.selector
    backend:  str | None = args.backend
    config = _load_local(args)
    os.environ["LOCAL_AI_CONFIG"] = str(args.config)

    if getattr(args, "list", False):
        cmd = [sys.executable, str(MENU_PY), "--list"]
        os.execv(sys.executable, cmd)

    # No selector → hand off to the interactive menu (it handles model + backend pick).
    if not selector:
        cmd = [sys.executable, str(MENU_PY)]
        if backend:
            cmd += ["--backend", backend]
        if args.port:
            cmd += ["--port", args.port]
        if args.host:
            cmd += ["--host", args.host]
        if args.dry_run:
            cmd.append("--dry-run")
        if getattr(args, "ctx", None) is not None:
            cmd += ["--ctx", str(args.ctx)]
        os.execv(sys.executable, cmd)  # replaces this process

    # Selector given without backend → auto-detect.
    if not backend:
        backend = _auto_backend(selector, Path(config["models"]["registry"]))
        if backend is None:
            raise SystemExit(
                f"No on-disk model found for '{selector}'. "
                f"Run `llm list` to see what is available."
            )

    # Direct launch via serve_local.sh — os.execv so signals reach the server.
    cmd = [str(SERVE_SH), selector, "--backend", backend]
    if args.port:
        cmd.append(args.port)
    if args.host:
        cmd += ["--host", args.host]
    if args.dry_run:
        cmd.append("--dry-run")
    if getattr(args, "ctx", None) is not None:
        cmd += ["--ctx", str(args.ctx)]
    os.execv(str(SERVE_SH), cmd)


# ---------------------------------------------------------------------------
# bench
# ---------------------------------------------------------------------------

def cmd_bench(args: argparse.Namespace) -> None:
    if args.profile_info:
        document = benchmark_profiles.load_profiles()
        profile = benchmark_profiles.get_profile(document, args.profile_info)
        print(json.dumps(profile, indent=2) if args.json else _format_profile(args.profile_info, profile))
        return
    if args.profile or args.prepare_tokenizer:
        import bench_profile_api
        options = ["--config", str(args.config)]
        if args.prepare_tokenizer:
            options.append("--prepare-tokenizer")
        else:
            options += ["--profile", args.profile]
            for key in ("url", "model", "server_command", "temperature", "timeout", "seed"):
                value = getattr(args, key)
                if value is not None:
                    options += [f"--{key.replace('_', '-')}", str(value)]
            if args.results_dir:
                options += ["--output-dir", str(args.results_dir)]
            if args.dry_run:
                options.append("--dry-run")
            if args.no_thinking:
                options.append("--no-thinking")
        bench_profile_api.main(options)
        return
    cmd = [_python(), str(BENCH_TUI)]
    if args.results_dir:
        cmd += ["--results-dir", str(args.results_dir)]
    if args.max_pass:
        cmd.append("--include-model-max-pass")
    subprocess.run(cmd, cwd=ROOT, check=False)


def _format_profile(name: str, profile: dict) -> str:
    lines = [f"{name}: {profile['description']}"]
    for workload in profile["workloads"]:
        lines.append(
            f"  {workload['id']:<14} input={workload['input_tokens']:<7} "
            f"minimum_output={workload['min_output_tokens']}"
        )
    modes = ", ".join(f"{item['id']} ({item['concurrency']}x)" for item in profile["modes"])
    lines.append(f"  modes: {modes}")
    lines.append("  scope: local profile; collected results require conformance validation")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------

def cmd_showdown(args: argparse.Namespace) -> None:
    """Thin pass-through so the showdown scripts keep one CLI surface.

    Deliberately not re-declaring their flags here: a second copy of an argument
    parser is a second thing to keep in sync, and these scripts are also the
    documented way to run the experiment directly.
    """
    script = ROOT / "scripts" / args.script
    command = [sys.executable, str(script), *args.rest]
    raise SystemExit(subprocess.call(command, cwd=ROOT))


def cmd_smoke(args: argparse.Namespace) -> None:
    cmd = [_python(), str(SMOKE_PY)]
    if args.dry_run:
        cmd.append("--dry-run")
    subprocess.run(cmd, cwd=ROOT, check=False)


# ---------------------------------------------------------------------------
# web
# ---------------------------------------------------------------------------

def cmd_web(args: argparse.Namespace) -> None:
    cmd = [_python(), str(WEB_PY)]
    if args.host:
        cmd += ["--host", args.host]
    if args.port:
        cmd += ["--port", str(args.port)]
    if args.idle_timeout is not None:
        cmd += ["--idle-timeout", str(args.idle_timeout)]
    if args.no_open:
        cmd.append("--no-open")
    os.execv(cmd[0], cmd)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Pasted terminal commands sometimes carry NBSP after "help". Normalize
    # only the command word, never model paths or other user-provided values.
    if argv and argv[0].strip() == "help":
        argv = [*argv[1:], "--help"]
    ap = argparse.ArgumentParser(
        prog="llm",
        description="Local inference control plane — models, serve, status, doctor, bench",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  llm serve                          # interactive model + backend picker
  llm serve gemma26                  # auto-detect backend
  llm serve gemma26 --backend mlx-kv # mlx_vlm + KV q8 turboquant
  llm serve gemma26 --dry-run
  llm bench
  llm smoke
  llm list
  llm list --backend mlx
  llm models sync                  # explicit local registry refresh
  llm doctor                       # read-only machine audit
  llm update --check               # installed versus current releases
  llm shortcuts sync               # preview splash/wrapper alignment
  llm status                       # what is running, and which model
  llm stop --all                   # stop everything started from here
  llm use qwen27 --port 8085       # switch the loaded model
  llm restart                      # fresh server, same model
  llm web                          # browser control surface
""",
    )
    ap.add_argument("--config", type=Path, default=local_config.DEFAULT_CONFIG,
                    help="Machine configuration (default: configs/local.toml)")
    sub = ap.add_subparsers(dest="command", required=True)

    # -- serve ----------------------------------------------------------------
    sp = sub.add_parser("serve", help="Launch a model server")
    sp.add_argument("selector", nargs="?",
                    help="Model alias, family id, or path (omit for interactive picker)")
    sp.add_argument("--backend", choices=BACKENDS,
                    help="Serving backend (auto-detected when omitted)")
    sp.add_argument("--port",  help="Port override")
    sp.add_argument("--host",  help="Host override (default: 127.0.0.1)")
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--list", action="store_true", help="List serving families")
    sp.set_defaults(func=cmd_serve)

    # -- stop / use / restart -------------------------------------------------
    stp = sub.add_parser("stop", help="Stop a model server")
    stp.add_argument("--port", help="Port to stop (e.g. 8085)")
    stp.add_argument("--all", action="store_true",
                     help="Stop every server started from here (leaves LM Studio alone)")
    stp.set_defaults(func=cmd_stop)

    usp = sub.add_parser("use", help="Switch the model on a port (stop, then serve)")
    usp.add_argument("selector", help="Model alias, family id, or path")
    usp.add_argument("--backend", choices=BACKENDS,
                     help="Serving backend (auto-detected when omitted)")
    usp.add_argument("--port", help="Port to take over (default: 8085)")
    usp.add_argument("--host", help="Host override (default: 127.0.0.1)")
    usp.add_argument("--dry-run", action="store_true")
    usp.set_defaults(func=cmd_use, list=False)

    rsp = sub.add_parser("restart", help="Restart a port with the model it already serves")
    rsp.add_argument("--port", help="Port to restart (default: 8085)")
    rsp.add_argument("--backend", choices=BACKENDS, help="Override the detected backend")
    rsp.add_argument("--host", help="Host override (default: 127.0.0.1)")
    rsp.add_argument("--dry-run", action="store_true")
    rsp.set_defaults(func=cmd_restart, list=False, selector=None)

    # -- bench ----------------------------------------------------------------
    bp = sub.add_parser("bench", help="Run the interactive benchmark TUI")
    bp.add_argument("--results-dir", type=Path,
                    help="Where to write CSV results (default: results/)")
    bp.add_argument("--max-pass", action="store_true",
                    help="Pre-select the model-max context pass")
    bp.add_argument("--profile-info", choices=["aa_local_full", "aa_local_smoke", "local_practical"],
                    help="Describe a versioned methodology profile instead of launching the TUI")
    bp.add_argument("--json", action="store_true", help="Emit profile information as JSON")
    bp.add_argument("--profile", choices=["aa_local_full", "aa_local_smoke", "local_practical"],
                    help="Execute versioned streaming service workloads")
    bp.add_argument("--prepare-tokenizer", action="store_true", help="Download the public measurement tokenizer once")
    bp.add_argument("--url", help="Loopback OpenAI-compatible endpoint for profile measurement")
    bp.add_argument("--model", help="Model ID exposed by the service")
    bp.add_argument("--server-command", help="Quoted foreground command owned by the profile runner; required for cold batches")
    bp.add_argument("--temperature", type=float)
    bp.add_argument("--timeout", type=float)
    bp.add_argument("--seed", type=int)
    bp.add_argument("--no-thinking", action="store_true", help="Request thinking disabled for service performance tests")
    bp.add_argument("--dry-run", action="store_true", help="Inspect a profile without launching or measuring")
    bp.set_defaults(func=cmd_bench)

    # -- smoke ----------------------------------------------------------------
    shp = sub.add_parser("showdown", help="Qwen3.8-27B quantization showdown (see docs/QWEN38_SHOWDOWN.md)")
    showdown_sub = shp.add_subparsers(dest="showdown_command", required=True)
    for name, script, helptext in (
        ("models", "showdown_acquire.py", "Inventory, plan, download, or verify benchmark artifacts"),
        ("run", "showdown_run.py", "Execute the configuration matrix"),
        ("report", "showdown_report.py", "Aggregate a matrix run into a report"),
    ):
        entry = showdown_sub.add_parser(name, help=helptext, add_help=False)
        entry.add_argument("rest", nargs=argparse.REMAINDER,
                           help="Arguments passed straight through to the script")
        entry.set_defaults(func=cmd_showdown, script=script)

    kp = sub.add_parser("smoke", help="Run the benchmark pipeline smoke test")
    kp.add_argument("--dry-run", action="store_true",
                    help="Skip Python compilation check")
    kp.set_defaults(func=cmd_smoke)

    # -- list -----------------------------------------------------------------
    lp = sub.add_parser("list", help="List models from the registry")
    lp.add_argument("--backend", choices=["auto", "llamacpp", "mlx"], default="auto")
    lp.set_defaults(func=cmd_list)

    # -- status ---------------------------------------------------------------
    tp = sub.add_parser("status", help="Check inference port health")
    tp.set_defaults(func=cmd_status)

    # -- doctor ---------------------------------------------------------------
    dp = sub.add_parser("doctor", help="Read-only audit of runtimes, models, and external projects")
    dp.add_argument("--json", action="store_true", help="Emit structured JSON")
    dp.set_defaults(func=cmd_doctor)

    # -- update ---------------------------------------------------------------
    up = sub.add_parser("update", help="Check local inference runtime versions")
    up.add_argument("--check", action="store_true",
                    help="Read-only installed/latest check (currently the only mode)")
    up.add_argument("--json", action="store_true", help="Emit structured JSON")
    up.set_defaults(func=cmd_update)

    # -- models ---------------------------------------------------------------
    mp = sub.add_parser("models", help="Inspect or refresh the generated model registry")
    model_sub = mp.add_subparsers(dest="models_command", required=True)
    mlp = model_sub.add_parser("list", help="List registered models")
    mlp.add_argument("--backend", choices=["auto", "llamacpp", "mlx"], default="auto")
    mlp.set_defaults(func=cmd_list)
    mip = model_sub.add_parser("inspect", help="Show exact model context, sampling and source provenance as JSON")
    mip.add_argument("selector", help="Model alias, family, or exact path")
    mip.add_argument("--backend", choices=["mlx", "llamacpp"])
    mip.set_defaults(func=cmd_models_inspect)
    msp = model_sub.add_parser("sync", help="Regenerate the registry from the configured LM Studio root")
    msp.add_argument("--known-only", action="store_true", help="Skip unmatched custom models")
    msp.set_defaults(func=cmd_models_sync)

    # -- shortcuts ------------------------------------------------------------
    cp = sub.add_parser("shortcuts", help="Align splash and llama-serve shortcuts")
    shortcut_sub = cp.add_subparsers(dest="shortcuts_command", required=True)
    csp = shortcut_sub.add_parser("sync", help="Preview or apply stable launcher shortcuts")
    csp.add_argument("--apply", action="store_true", help="Write changes after timestamped backups")
    csp.set_defaults(func=cmd_shortcuts_sync)

    # -- web ------------------------------------------------------------------
    wp = sub.add_parser("web", help="Launch the optional legacy browser GUI")
    wp.add_argument("--host", help="Bind host (default: 127.0.0.1)")
    wp.add_argument("--port", type=int, help="Bind port (default: 7860)")
    wp.add_argument("--idle-timeout", type=int,
                    help="Seconds idle before auto-unload + shutdown (0 disables; default 300)")
    wp.add_argument("--no-open", action="store_true", help="Do not open a browser")
    wp.set_defaults(func=cmd_web)

    for parser in (sp, usp, rsp):
        parser.add_argument("--ctx", type=int,
                            help="Explicit context override; 0 selects the model's native window")
    args = ap.parse_args(argv)
    if getattr(args, "ctx", None) is not None and args.ctx < 0:
        ap.error("--ctx must be zero or positive")
    try:
        args.func(args)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
