#!/usr/bin/env python3
"""Start, inspect, and stop local model servers — one implementation.

`webgui/serve.py` and `llama_serve_menu.py` each grew their own copy of the
lsof port-kill and the health probe. Two copies of "how do I stop a server"
is the same drift this repository is trying to remove one level up, so they
live here and both callers borrow them.

Nothing in this module launches a server: `scripts/serve_local.sh` is the only
thing that does that. This module stops them and reports what is running.
"""
from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# The endpoints this machine serves on. Kept in step with configs/local.toml.
DEFAULT_PORTS: tuple[tuple[int, str], ...] = (
    (8080, "llama.cpp"),
    (8085, "direct MLX"),
    (1234, "LM Studio"),
)


@dataclass(frozen=True)
class PortState:
    port: int
    label: str
    pids: tuple[int, ...]
    up: bool
    model: str | None

    @property
    def listening(self) -> bool:
        return bool(self.pids)

    @property
    def owned(self) -> bool:
        """True when a process we could stop is listening here.

        LM Studio listens on 1234 but is managed by its own app; reporting it
        as ours would invite `llm stop --all` to kill something the owner did
        not start from here.
        """
        return self.listening and self.port != 1234


def port_pids(port: int) -> tuple[int, ...]:
    """PIDs LISTENing on `port`, newest last. Empty when nothing is bound."""
    try:
        out = subprocess.check_output(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return ()
    pids: list[int] = []
    for line in out.split():
        try:
            pids.append(int(line))
        except ValueError:
            continue
    return tuple(pids)


def reachable(port: int, timeout: float = 2.0) -> bool:
    """Whether the server answers HTTP at all.

    A listening socket is not readiness and neither is this: `/v1/models` can
    answer while weights are still loading. Only a real completion proves
    readiness. This distinguishes "bound the port" from "answering".
    """
    url = f"http://127.0.0.1:{port}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            json.loads(response.read())
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return False
    return True


def loaded_model(port: int) -> str | None:
    """Which model the server on `port` is actually serving.

    Read from the process arguments, not from `/v1/models`. `mlx_lm.server`
    answers `/v1/models` with a scan of the whole Hugging Face cache and the
    loaded model is *not* first in that list, so `data[0]` reports an unrelated
    model — observed naming a TTS checkpoint while Qwen3.8-27B-oQ4 was loaded.
    `serve_local.sh` always passes the path (`--model` on MLX, `-m` on
    llama.cpp), which is unambiguous.
    """
    pids = port_pids(port)
    if not pids:
        return None
    try:
        out = subprocess.check_output(
            ["ps", "-o", "command=", "-p", str(pids[0])],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None
    argv = out.split()
    for flag in ("--model", "-m"):
        if flag in argv:
            index = argv.index(flag) + 1
            if index < len(argv):
                return argv[index]
    return None


def probe(port: int, timeout: float = 2.0) -> tuple[bool, str | None]:
    """(reachable, loaded model). Kept for callers that want both at once."""
    return reachable(port, timeout=timeout), loaded_model(port)


def inspect(ports: tuple[tuple[int, str], ...] = DEFAULT_PORTS) -> list[PortState]:
    states: list[PortState] = []
    for port, label in ports:
        pids = port_pids(port)
        up = reachable(port) if pids else False
        model = loaded_model(port) if pids else None
        states.append(PortState(port=port, label=label, pids=pids, up=up, model=model))
    return states


def stop(port: int, timeout: float = 12.0) -> tuple[bool, str]:
    """Stop whatever listens on `port`. SIGTERM, then SIGKILL if it lingers.

    `serve_local.sh` exec()s the real server, so the PID holding the port *is*
    the server and a plain TERM reaches it. MLX servers can take a few seconds
    to release Metal buffers, hence the wait before escalating.
    """
    pids = port_pids(port)
    if not pids:
        return True, f"nothing listening on {port}"

    for pid in pids:
        subprocess.run(["kill", str(pid)], check=False)

    deadline = timeout
    step = 0.25
    while deadline > 0:
        if not port_pids(port):
            return True, f"stopped {port} (pid {', '.join(map(str, pids))})"
        time.sleep(step)
        deadline -= step

    remaining = port_pids(port)
    for pid in remaining:
        subprocess.run(["kill", "-9", str(pid)], check=False)
    if port_pids(port):
        return False, f"port {port} still held by pid {', '.join(map(str, port_pids(port)))}"
    return True, f"force-stopped {port} (pid {', '.join(map(str, remaining))})"


def model_label(model: str | None) -> str:
    """Shorten an MLX model path to something readable in a status table."""
    if not model:
        return "—"
    if "/" in model:
        parts = Path(model).parts
        return "/".join(parts[-2:]) if len(parts) >= 2 else model
    return model
