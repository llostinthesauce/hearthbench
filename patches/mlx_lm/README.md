# mlx_lm server fixes

Four edits to `mlx_lm/server.py` that are not upstream. They are made against
**mlx-lm 0.32.0** (pinned in `pyproject.toml`), apply in numeric order, and use
`a/` `b/` paths relative to `site-packages`.

Any reinstall of mlx-lm reverts them silently. `llm doctor` names each missing
fix; `scripts/serving_runtime.py` holds the marker that identifies each one.
Restore whatever is missing with:

    .venv/bin/python scripts/apply_serving_patches.py            # --dry-run to preview

## 01 — `reasoning_content` alias

The server emits assistant thinking as `reasoning`; Qwen chat templates read
past thinking from `reasoning_content`. A client that echoes the server's own
field back loses every prior turn's reasoning, and after ~10 tool-using turns
the model degenerates into verbatim repetition. Captured from a failing pi
session and replayed against the server: the unmodified request looped (`x12`
10-gram repeat, `finish=length`); the same request with the field renamed, or
the unmodified request against the patched server, finished cleanly with
`tool_calls`. Server-side, so it covers pi, opencode and Mycelium at once.

## 02 — server-side penalty defaults

Adds `--repetition-penalty`, `--presence-penalty`, `--repetition-context-size`
and `--presence-context-size` (also readable from `MLX_*` environment
variables), and falls back to them when a request omits the field. Upstream
defaults to no penalty and a 20-token window. The 2048-token lookback is
load-bearing: Qwen's reasoning loops recur ~1000–1200 tokens apart, which a
short window cannot see. `serve_local.sh` passes the catalog values.

## 03 — Metal OOM shield

Upstream calls `batch_generator.next()` unguarded. A prompt that exceeds the
~41.7 GB Metal working-set limit raises `[METAL] Command buffer execution
failed: Insufficient Memory`, which kills the generation thread silently; every
waiting client then hangs until its own timeout (900 s per request in the
showdown harness). The shield fails the pending requests at once, closes the
batch generator, and calls `mx.clear_cache()` so the next request does not
start against a wedged allocator. Measured in the September showdown: the two
long-context evals failed fast instead of burning a full timeout per request.

## 04 — explicit zero request penalties

An explicit `0` must override a nonzero launcher default. The original penalty
patch used Python `or`, treating zero as an omitted field. The correction uses
key presence for repetition, presence, frequency and their lookback values.
Patch 04 depends on patch 02; the installer applies missing patches in order to
a temporary copy and atomically publishes only after all markers validate.
Both an upgraded runtime and a clean upstream 0.32.0 source were verified.

## Removed: `_safe_logprob`

A wrapper around the per-token logprob lookup, added 2026-09-18 for a
`Slice indices must be 32-bit integers` crash that only Homebrew's MLX 0.32.0
server produced, and slated for removal after that run. Removed 2026-09-22:
MLX 0.32.2 indexes correctly, the wrapper's int32 clamp would silently read a
different token's logprob, and its `0.0` fallback records a failure as
probability 1.0.

## After an mlx-lm upgrade

Run the apply script with `--dry-run` first. A hunk that no longer applies must be
ported by hand, then the patch regenerated with `diff -u` against the new
upstream file, keeping the `a/mlx_lm/server.py` / `b/mlx_lm/server.py` labels.
An unmodified upstream file can be verified against the `sha256` for
`mlx_lm/server.py` in the package's `dist-info/RECORD`.
