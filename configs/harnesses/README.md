# Harness configs for the local MLX server

`pi.models.json`     -> ~/.pi/agent/models.json   (copy of the live file, 2026-09-23)
`hermes.config.yaml` -> ~/.hermes/config.yaml     (UNVERIFIED - hermes not installed)

The live `~/.pi/agent/models.json` is the source of truth; refresh this copy
after editing it. Both local endpoints advertise a 65,536-token context window:
the measured ceiling for Qwen3.8-27B on this Mac (`docs/SERVING.md` §4), so a
client compacts before the server runs out of Metal memory. Raise it only for a
session on a model measured to hold more.

Pi also needs `~/.pi/agent/settings.json`:

    "defaultProvider": "mlx", "defaultModel": "default_model", "defaultThinkingLevel": "off"

so a bare `pi` opens on whatever is loaded on :8085.

---

## ROOT CAUSE of the agentic repetition loop: reasoning / reasoning_content

**Fixed 2026-09-07 by `patches/mlx_lm/01-reasoning-content-alias.patch`.**

`mlx_lm.server` EMITS assistant thinking as `choices[].message.reasoning`. Pi (and any
OpenAI-compatible client that echoes the field back) sends it to the next turn as `reasoning`.
Qwen3 chat templates read historical thinking from **`message.reasoning_content`**. The names
do not match, so every prior turn's reasoning renders as an empty `<think></think>`.

After ~10 tool-using turns the model is looking at a history in which it appears to have
thought nothing, re-derives from scratch each turn, and degenerates into verbatim repetition.

### Evidence

Captured the real first-bad request from a failing pi session (43 messages, 13 assistant
turns each carrying `reasoning` and none carrying `reasoning_content`, ~57k prompt tokens,
`reasoning_effort: medium`) and replayed it directly against `mlx_lm.server`, outside pi:

| arm | result | out | finish_reason | 10-gram repeat |
|---|---|---|---|---|
| A exact request, as pi sends it        | **LOOPED** | 900 | `length`     | x12 |
| B same + `reasoning`->`reasoning_content` | clean   | 399 | `tool_calls` | x1  |
| C earlier healthy prefix (20 msgs)     | clean      | 108 | `tool_calls` | x1  |
| A' exact request, PATCHED server       | clean      | 726 | `tool_calls` | x1  |

One field renamed is the only difference between A and B. A' proves the server-side patch
fixes the unmodified client request. Fix is server-side, so it covers pi, opencode and
mycelium at once.

### Reapplying the patch

Any mlx-lm reinstall overwrites it. See `patches/mlx_lm/README.md` for how to
reapply all three serving fixes to the core `.venv`.

---

## Qwen thinking is opt-in; medium is the ceiling

The reasoning-history alias remains required for multi-turn tool use, but it does not make
`xhigh` universally safe. A later single-turn exact-word-count request looped for 28,598
reasoning characters on the patched server because `xhigh` injected repeated validation and
recounting instructions. A live follow-up also showed `medium` and `low` each consume a
512-token answer budget without reaching content on the same constrained prompt. The stable
serving default is therefore thinking off; explicit thinking is capped at `medium` in Pi.

Verified on the patched server:

| test | before patch | after patch |
|---|---|---|
| exact 57k bad request @ xhigh | (looped even at medium) | clean, 582 tok, `finish=tool_calls` |
| open-ended "explain pi" via real pi @ xhigh | 5+ turns, 7+ min, off-task, never answered | **complete, good answer, 2m56s** |

Pi's local mapping deliberately caps its `xhigh` and `max` selections to the Qwen template's
`medium` effort. This avoids reintroducing the known loop through the UI; selecting any
non-off Pi thinking level still opts in to Qwen thinking. Other providers are unaffected.

`medium` remains the template's empty branch - literally no reasoning instruction - if you ever
want the model's untuned default.

## BrokenPipeError is a symptom of client cancellation

The failing session's aborted turns record `stopReason: aborted`, `errorMessage: "Operation
aborted"`, and zero usage - i.e. the operator interrupted a runaway generation. That closes
the socket and the server logs `BrokenPipeError`. It is a consequence, not a cause.

## Ruled out by test

- **Structural corruption**: none. 27 tool calls <-> 27 results, correctly paired, no
  duplicate ids, no orphans, no duplicated assistant turns, no partial persisted output.
- **Client retries**: none. The repeated text lives inside a single `thinking` content block
  in one completion, not across turns.
- **Context-window exhaustion**: no. ~57k of a 262,144 window. The `5751/5751` in the server
  log is a prompt-cache miss suffix, not the whole prompt.

## Observed but NOT established

A server that had served an aborted/broken-pipe request measured ~2.8 prompt tok/s, while a
freshly restarted server measured 414-473 prompt tok/s on the same machine. Observed once and
confounded by ~24 GB of leftover swap, so it is not proven that the abort caused it.
Practical mitigation until tested: restart the MLX server after interrupting a generation.

## Server-side repetition mitigation

Fixed 2026-09-13 by `patches/mlx_lm/02-server-penalties.patch`.

`mlx_lm.server` previously exposed no CLI flags or server-side defaults for repetition/presence penalties, defaulting to 0.0 unless passed in the per-request HTTP body.

The patch:
1. Adds `--repetition-penalty`, `--presence-penalty`, `--repetition-context-size`, and `--presence-context-size` to `mlx_lm.server` CLI.
2. In `APIHandler.do_POST`, falls back to server CLI defaults if omitted by incoming client requests.
3. Integrated into `serve_local.sh`, `configs/local.toml` (`[serving.sampling]`), and `configs/models.local.json` (defaults: `rep_pen=1.05`, `pres_pen=0.2`, `context_size=2048`).
