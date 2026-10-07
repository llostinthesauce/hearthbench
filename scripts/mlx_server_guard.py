"""Run `mlx_lm.server` so a dead generation worker kills the process.

`mlx_lm.server` can survive its own generation thread dying: the HTTP server
keeps answering `/v1/models` while no request can ever complete. A supervising
parent that treats a reachable port as a healthy runtime will then wait
forever on a server that cannot generate.

Exiting on that failure turns an invisible hang into an observable death, so a
parent can restart or report. Mycelium's `BackendSupervisor` needs this; it is
useful to anything that owns a server process, so it lives in the core rather
than in one consumer.

Selected with `serve_local.sh --fatal-worker`. Without that flag nothing here
runs and the built command is unchanged.

Not included: the uncompiled-sampler replacement that Mycelium's own copy
carried for MLX 0.32.0, where the compiled sampler captured `mx.random.state`
on the import thread and every temperature > 0 request decoded identically.
Verified absent on the core runtime (mlx 0.32.2, mlx-lm 0.31.3) on 2026-09-21,
both by probing the sampler directly and by five identical temperature-0.7
requests against a live Qwen3.8-27B-oQ4 server, which returned four distinct
completions. Reinstate it only against a fresh measurement.
"""
from __future__ import annotations

import os
import runpy
import sys
import threading


def _worker_failed(args) -> None:
    """Exit the process when the generation thread raises."""
    threading.__excepthook__(args)
    traceback = args.exc_traceback
    while traceback is not None:
        frame = traceback.tb_frame
        if (
            frame.f_globals.get("__name__") in ("mlx_lm.server", "__main__")
            and frame.f_code.co_name == "_generate"
        ):
            print(
                "serve_local.sh: generation worker crashed; exiting so the "
                "supervising process can observe the failure.",
                file=sys.stderr,
                flush=True,
            )
            os._exit(1)
        traceback = traceback.tb_next


def main() -> None:
    threading.excepthook = _worker_failed
    runpy.run_module("mlx_lm.server", run_name="__main__")


if __name__ == "__main__":
    main()
