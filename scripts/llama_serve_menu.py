#!/usr/bin/env python3
"""
Interactive and CLI launcher for local model serving (the "python llama TUI").

This is the repo-owned replacement for hardcoded global launcher menus. Global
entrypoints should call this file, and this file resolves models from
configs/models.local.json via scripts/model_registry.py, then hands off to
scripts/serve_local.sh.

Backends offered: llamacpp, mlx, mlx-kv (KV q8 turboquant). When a model
has mtp_supported=true in the registry, the llama.cpp option is tagged "+ MTP"
and serve_local.sh auto-enables Multi-Token Prediction speculative decoding.
"""
from __future__ import annotations

import argparse
import os
import select
import subprocess
import sys
import termios
import tty
from pathlib import Path

import local_config
import model_registry
import serving_lifecycle


ROOT = Path(__file__).resolve().parents[1]
SERVE_SCRIPT = ROOT / "scripts" / "serve_local.sh"

def _rows(config: dict) -> list[dict]:
    rows = model_registry.iter_models(Path(config["models"]["registry"]))
    families: dict[str, dict] = {}
    for row in rows:
        family_id = row["family_id"]
        if family_id not in families:
            aliases = row.get("aliases", [])
            label = aliases[0] if aliases else family_id
            families[family_id] = {
                "family_id": family_id,
                "label": label,
                "aliases": aliases,
                "backends": set(),
                "quant": row.get("quant", "?"),
                "use_case": row.get("use_case", ""),
            }
        # Only surface a backend if its model is actually on disk. A registry row
        # whose file is missing must not be offered — selecting it would either
        # error or (for MLX repo-ids) trigger a download.
        if row.get("exists"):
            families[family_id]["backends"].add(row["backend"])
    result = []
    for item in families.values():
        if not item["backends"]:
            continue
        item["backends"] = sorted(item["backends"])
        result.append(item)
    return result


def _display_label(row: dict) -> str:
    backends = ", ".join(b for b in row["backends"] if b in ("llamacpp", "mlx"))
    base = f"{row['family_id']:<20} [{backends or 'unavailable'}]"
    use_case = row.get("use_case", "")
    return f"{base}  — {use_case}" if use_case else base


def _supported_backends(selector: str, config: dict) -> list[str]:
    """Backends whose model for `selector` resolves AND exists on disk."""
    found = []
    for backend in ("llamacpp", "mlx"):
        try:
            row = model_registry.resolve(selector, backend, Path(config["models"]["registry"]))
        except SystemExit:
            continue
        if row.get("exists"):
            found.append(backend)
    if "mlx" in found:
        found.append("mlx-kv")
    return found


def _choose(items: list[dict], title: str, label_fn) -> dict:
    if not sys.stdin.isatty():
        raise SystemExit("Interactive selection requires a TTY. Pass a selector argument instead.")
    selected = 0
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    def render() -> None:
        sys.stdout.write("\x1b[2J\x1b[H")
        print(title)
        print("=" * len(title))
        for index, item in enumerate(items):
            marker = ">" if index == selected else " "
            print(f"{marker} {label_fn(item)}")
        print("\nUse up/down arrows, Enter to select, Ctrl+C to cancel.")
        sys.stdout.flush()

    try:
        tty.setcbreak(fd)
        render()
        while True:
            ready, _, _ = select.select([sys.stdin], [], [])
            if not ready:
                continue
            char = os.read(fd, 3)
            if char in (b"\x1b[A", b"\x1bOA"):
                selected = max(0, selected - 1)
                render()
            elif char in (b"\x1b[B", b"\x1bOB"):
                selected = min(len(items) - 1, selected + 1)
                render()
            elif char in (b"\n", b"\r"):
                return items[selected]
            elif char == b"\x03":
                raise KeyboardInterrupt
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        sys.stdout.write("\x1b[?25h\n")
        sys.stdout.flush()


def _free_port(port: str, dry_run: bool) -> None:
    """Stop whatever holds `port`, and wait until it has actually let go.

    Delegates to serving_lifecycle.stop, which waits for the port to free and
    escalates to SIGKILL. Signalling and exec'ing straight away let the new
    server race an MLX process still releasing Metal buffers for the port.
    """
    if dry_run:
        pids = serving_lifecycle.port_pids(int(port))
        if pids:
            print(f"DRY RUN: would stop port {port} pids: {', '.join(map(str, pids))}")
        return
    ok, message = serving_lifecycle.stop(int(port))
    if not ok:
        raise SystemExit(message)


def _launch(selector: str, backend: str, port: str | None, host: str | None,
            dry_run: bool, kill_port: bool, config: dict, ctx: int | None = None) -> None:
    cmd = [str(SERVE_SCRIPT), selector, "--backend", backend]
    if port:
        cmd.append(port)
    if host:
        cmd += ["--host", host]
    if dry_run:
        cmd.append("--dry-run")

    if ctx is not None:
        cmd += ["--ctx", str(ctx)]

    endpoint_name = "mlx" if backend in ("mlx", "mlx-kv", "mlx-vlm") else "llamacpp"
    configured_host, configured_port = local_config.endpoint_host_port(config, endpoint_name)
    effective_port = port or str(configured_port)
    if kill_port:
        _free_port(effective_port, dry_run)

    if dry_run:
        print("DRY RUN launcher:", " ".join(cmd))
        sys.stdout.flush()
        subprocess.run(cmd, cwd=ROOT, check=True)
        return
    os.execv(str(SERVE_SCRIPT), cmd)


def _print_list(config: dict) -> None:
    for row in _rows(config):
        print(_display_label(row))


def _choose_variant(family_id: str, backend: str, config: dict) -> str:
    registry_backend = "llamacpp" if backend == "llamacpp" else "mlx"
    variants = [
        row for row in model_registry.iter_models(Path(config["models"]["registry"]))
        if row["family_id"] == family_id and row["backend"] == registry_backend and row["exists"]
    ]
    if len(variants) <= 1:
        return variants[0]["path"] if variants else family_id
    selected = _choose(variants, "Model variant", lambda row: f"{row['quant']}: {row['name']}")
    return selected["path"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a local model API server")
    parser.add_argument("selector", nargs="?", help="Model alias, family id, path, or repo")
    parser.add_argument("--backend", choices=["llamacpp", "mlx", "mlx-kv", "mlx-vlm"], help="Serving backend")
    parser.add_argument("--port", help="Port override")
    parser.add_argument("--ctx", type=int, help="Explicit context override (0: native)")
    parser.add_argument("--host", help="Host override")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-kill", action="store_true", help="Do not kill existing listener on the target port")
    parser.add_argument("--list", action="store_true", help="List serving families")
    parser.add_argument("--config", type=Path, default=local_config.default_config_path())
    args = parser.parse_args()
    config = local_config.load_config(args.config)
    os.environ["LOCAL_AI_CONFIG"] = str(args.config)

    if args.list:
        _print_list(config)
        return

    selector = args.selector
    backend = args.backend

    if not selector:
        family = _choose(_rows(config), "Local model server", _display_label)
        selector = family["family_id"]
        supported = list(family["backends"])
        if "mlx" in supported:
            supported.append("mlx-kv")
    else:
        supported = _supported_backends(selector, config)

    if not backend:
        llama_host, llama_port = local_config.endpoint_host_port(config, "llamacpp")
        mlx_host, mlx_port = local_config.endpoint_host_port(config, "mlx")
        # Tag the llama.cpp option when this model carries an MTP head, so the
        # picker shows that serve_local.sh will turn on speculative decoding.
        mtp_tag = ""
        try:
            ll_row = model_registry.resolve(selector, "llamacpp", Path(config["models"]["registry"]))
            if ll_row.get("mtp_supported"):
                mtp_tag = " + MTP"
        except SystemExit:
            pass
        labels = {
            "llamacpp": f"llama.cpp API on {args.host or llama_host}:{args.port or llama_port}{mtp_tag}",
            "mlx": f"MLX API on {args.host or mlx_host}:{args.port or mlx_port} (mlx_lm, no KV quant)",
            "mlx-kv": f"MLX API on {args.host or mlx_host}:{args.port or mlx_port} (mlx_vlm + KV q8 turboquant)",
        }
        # Only offer backends this model actually has on disk — a one-backend
        # model skips the prompt entirely instead of falsely showing both.
        backend_choices = [
            {"backend": b, "label": labels[b]} for b in ("llamacpp", "mlx", "mlx-kv") if b in supported
        ]
        if not backend_choices:
            raise SystemExit(
                f"No serving backend is available on disk for '{selector}'. "
                f"Check 'python3 scripts/model_registry.py list'."
            )
        if len(backend_choices) == 1:
            backend = backend_choices[0]["backend"]
            print(f"Backend: {backend} (only on-disk option for {selector})")
        elif args.selector:
            backend = backend_choices[0]["backend"]
        else:
            selected_backend = _choose(backend_choices, "Serving backend", lambda item: item["label"])
            backend = selected_backend["backend"]

    if not args.selector:
        selector = _choose_variant(selector, backend, config)
    _launch(selector, backend, args.port, args.host, args.dry_run, not args.no_kill, config, args.ctx)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
