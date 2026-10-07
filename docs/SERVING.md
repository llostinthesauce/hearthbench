# Serving, in plain language

This is the map of how local inference works on this machine: what lives where,
how to start and stop it, and what each knob actually does. Every claim here is
either verifiable by running the command shown or came out of a measurement in
this repository — not from vendor documentation.

If you read one thing: **this folder is the one place models get served from.
Everything else borrows.**

---

## 1. The map

```
hearthbench/                <-- the one core place
├── .venv/                  the Python that serves models (patched, see §6)
├── scripts/serve_local.sh  the only thing that starts a server
├── configs/
│   ├── model_catalog.json  committed: which models exist, and their settings
│   ├── models.local.json   generated: what is actually on this disk
│   └── local.toml          machine paths and endpoints
└── llm.py                  the commands you type
```

Three things borrow from it:

| Who | How it borrows | Cares which model is loaded? |
| --- | --- | --- |
| opencode | points at `127.0.0.1:8080` / `:8085` | No — uses whatever is loaded |
| pi | same, via `configs/harnesses/pi.models.json` | No |
| Mycelium | its `scripts/serve_mlx.sh` calls `serve_local.sh` | Yes — it picks a model |

The opencode/pi pattern is the good one: they aim at a **port**, not a model. You
change what is loaded and they follow, with no config edit.

Mycelium is different because it starts its own server on demand. Its
`serve_mlx.sh` is a shim: it passes the chosen model, `--thinking off`, and `--fatal-worker`
(with `--ctx` only when explicitly configured) to this folder's launcher and does
nothing else. Until 2026-09-21 it was a full second implementation on a
different Python with a different MLX version — see §6 for why that mattered.

**Nothing here is "global."** The only machine-level pieces are three one-line
wrappers in `~/.local/bin` (`llm`, `local-ai`, `llama-serve`) and Homebrew's
`llama-server` binary. Models all live under `~/.lmstudio/models`, and the
launcher refuses paths outside it.

### The ports

| Port | What | Started by |
| --- | --- | --- |
| 8080 | llama.cpp (GGUF files) | you, or Mycelium |
| 8085 | MLX (model directories) | you, or Mycelium |
| 1234 | LM Studio | the LM Studio app — not ours, `stop --all` leaves it alone |
| 8000 | Mycelium's own backend | the Mycelium app |

---

## 2. Starting and stopping

```bash
llm help                        # command reference (also: llm help serve)
llm status                      # what is running, and which model
llm serve                       # interactive picker
llm serve qwen27                # start the family's best model
llm use qwen27 --port 8085      # switch the loaded model (stop, then serve)
llm restart                     # fresh server, same model
llm stop --port 8085            # stop one
llm stop --all                  # stop everything we started
llm bench                       # the benchmark TUI
llm web                         # browser control surface
llm doctor                      # is anything broken
```

`llama-serve` is the same system: it now forwards to `llm serve`, so old muscle
memory keeps working and there is only one place on the machine that knows where
this checkout lives. `local-ai` is the same command as `llm`.

All three are installed by `llm shortcuts sync --apply` into `~/.local/bin`. If
you move this repository, run that once and every command follows.

Two things worth knowing:

- **A listening port is not readiness.** `/v1/models` answers while weights are
  still loading. `llm status` shows `loading` for that state and `serving` once
  the server answers. Only a real completion proves it can generate.
- **`llm status` reads the model from the server's process arguments, not from
  `/v1/models`.** MLX's `/v1/models` returns a scan of your whole Hugging Face
  cache and the loaded model is not first in the list — it once reported a TTS
  checkpoint while Qwen3.8-27B was loaded.

**Restart after interrupting a generation.** A server that served an aborted
request was measured at ~2.8 prompt tokens/sec against 414–473 on a fresh one.
Observed once and confounded by leftover swap, so it is not proven — but
`llm restart` is cheap.

---

## 3. Two backends, two shapes on disk

| | llama.cpp | MLX |
| --- | --- | --- |
| On disk | one `.gguf` file | a directory with `config.json` + `.safetensors` |
| Server | `llama-server` | `mlx_lm.server` or `mlx_vlm.server` |
| Port here | 8080 | 8085 |
| Speed (27B) | ~13 tok/s at Q5_K_M | ~17 tok/s at oQ4 |
| RAM / heat | 46–48 GB, 85–87 °C | ~61 GB, 90–95 °C |

MLX is faster and more energy-efficient on this machine; llama.cpp is cooler,
leaner on RAM, and the fallback when MLX cannot load something.

`model_type` in a model's `config.json` decides which MLX server can serve it.
`mlx_lm` handles ordinary text models. Some architectures it simply cannot load
(`gemma4_unified`, `muse_glimmer`), so the launcher reads `config.json` and
routes those to `mlx_vlm` automatically — you do not pass a flag for it.

**A quant label is not a specification.** Two publishers' `Q4_K_M` of the same
model measured 0.984 and 0.938; the second failed an agent task 3/3 and dropped
to 0.833 on instruction following. Same model, same nominal precision, same
serving path. The registry therefore distinguishes variants by **path**, never
by quant label.

---

## 4. The knobs

### Context: model capacity, allocation, and tested workloads

Default serving uses the installed model's native window. llama.cpp receives
`-c 0` and one slot; MLX grows its cache dynamically. The old 65,536 catalog
value is no longer a default serving ceiling. `llm serve MODEL --ctx N` makes
an explicit allocation choice for llama.cpp; on MLX the launcher reports that
this flag does not enforce a cache limit.

A declared 262,144-token window is architecture metadata, not a promise that
this machine can prefill that many tokens with every backend, quantization,
image count and KV format. Earlier 131K MLX trials exhausted Metal memory;
that evidence remains relevant, but it is not a universal model limit. Model
policy keeps declared context separate from recommended and verified workloads.
It does not silently apply RoPE/YaRN extensions beyond native context.

Mycelium borrows this metadata and preserves explicit user context overrides.
Existing externally generated pi/opencode settings may retain their own 64K
client budgets; this pass does not rewrite those machine configurations.

### KV cache: two different knobs with similar names

The KV cache stores the attention Keys and Values of tokens already processed
so they are not recomputed. It is usually the thing that runs you out of memory.

| Flag | Where | What |
| --- | --- | --- |
| `-ctk` / `-ctv` | llama.cpp | the **data type** of the cache (`f16`, `q8_0`, …) |
| `--kv-bits`, `--kv-quant-scheme` | launcher routes to `mlx_vlm` | **quantizing** the cache |

These are not the same knob and they are not interchangeable.

Two traps:

- **llama.cpp's default here is already quantized.** `serve_local.sh` passes
  `-ctk q8_0 -ctv q8_0`. A genuine "unquantized KV" llama.cpp baseline needs an
  explicit `--kv-type f16`.
- **The launcher still routes KV experiments to `mlx_vlm`.** MLX-LM 0.32.0
  now offers native KV quantization, but that path is not enabled here. Asking for KV quantization
  switches you to `mlx_vlm.server`, which accepts *no sampling flags whatsoever*
  — so you silently change `top_k` and the repetition penalties at the same
  time. Any KV comparison therefore needs an `mlx_vlm --kv-bits 0` control arm,
  or you cannot attribute the result to KV rather than to the server swap.

Also model-specific: Qwen3.8-27B is `model_type qwen3_5`, a hybrid attention
design with `full_attention_interval = 4`, so **only 16 of its 64 layers hold a
KV cache at all**. KV sizing intuitions from an all-full-attention model do not
transfer. Measure.

### Sampling

The defaults live in `configs/model_catalog.json` and are applied server-side:

```
temperature 0.7 · top_p 0.95 · top_k 20
repetition_penalty 1.05 · presence_penalty 0.2 · repetition_context_size 2048
```

- **temperature** — randomness. 0 is greedy/deterministic.
- **top_p 0.95** — sample from the smallest set of tokens covering 95% of
  probability.
- **top_k 20** — never consider more than the 20 likeliest tokens.
- **repetition/presence penalty** — push down tokens that already appeared.
- **repetition_context_size 2048** — *how far back* those penalties look. This
  one is load-bearing: upstream's default is 20 tokens, and a window that short
  cannot see repeated reasoning cycles that recur 1000–1200 tokens apart, which
  is exactly the shape of the Qwen thinking loop.

Which server accepts them matters: `mlx_lm.server` accepts all of the above
(only because of a local patch — see §6). `mlx_vlm.server` accepts none of them;
sampling there is per-request only. `llama-server` has its own spellings
(`--repeat-penalty`, `--repeat-last-n`).

### Thinking and reasoning effort

Three separate things that are easy to conflate:

1. **`enable_thinking`** — whether the model produces hidden reasoning. Off by
   default everywhere here. Not a level; it is a boolean, and "off" means no
   `reasoning_effort` key at all.
2. **`reasoning_effort`** — how much, when thinking is on. **Capped at
   `medium`.** `high`/`xhigh`/`max` map down, because Qwen3.8-27B looped for
   28,598 reasoning characters at `xhigh` on an exact-word-count prompt.
   On the same prompt `low` and `medium` used up a 512-token answer budget
   before reaching content — budget exhaustion, weaker evidence than the xhigh
   loop, but not proof they are safe either. Disabling thinking is what
   terminated cleanly.
3. **`reasoning_content`** — the field name for *past* turns' thinking in the
   conversation history. See §6; getting this wrong breaks multi-turn tool use.

A response that terminates is not proof of instruction compliance. Exact-count
and format misses are a different failure from a serving loop; report them
separately.

### MTP (speculative decoding)

A small "draft" head proposes several tokens and the big model verifies them in
one pass. Two on-disk shapes, both auto-detected: an embedded head
(Qwen3.6-35B-A3B-MTP) and a separate draft file (Gemma 4's `mtp-*.gguf`).
llama.cpp cannot combine it with `--mmproj` or `-np > 1`, so MTP mode forces
`-np 1` and drops mmproj. Turn it off with `--no-mtp`.

For Qwen MTP, use `mlx_vlm`: `mlx_lm` 0.32.0 has generic draft-model support
but still has no `qwen3_5_mtp` implementation.

---

## 5. Which model to use

From a 12-hour, 9-configuration matrix on this machine (September 2026):

- **Default: `mlx-community/Qwen3.8-27B-oQ4`.** Mixed per-module precision at
  4.86 bpw. Tied for the highest capability measured (0.987) while running 33%
  faster than 6-bit and using 40% less energy.
- **Fallback: `lmstudio-community/Qwen3.8-27B-MLX-4bit`** — within noise
  (0.984, 17.3 tok/s). Genuinely fine.
- **llama.cpp fallback: `bartowski/Qwen3.8-27B-Q5_K_M.gguf`.** Q5_K_M dominates
  Q6_K on capability, speed, size *and* energy, so Q6_K buys nothing.

Those three are what remain of the dense family on disk as of 2026-09-21.
Removed that day, 102 GB in total: MLX 5-bit and 6-bit, bartowski Q4_K_M and
Q6_K, the three Qwen3.8 MTP draft heads, and huihui-ai's abliterated Q6_K (one
of two near-identical uncensored builds). Every one was dominated by something
still present, or a duplicate of it. `llm serve qwen27` picks the right variant
per backend, so nothing referenced them by path and nothing broke.

Also still here, deliberately: `OBLITERATUS/Qwen3.8-27B-OBLITERATED-Q6_K`
(uncensored 27B, never benchmarked — untested, not known-bad),
`HauhauCS/Qwen3.6-35B-A3B-Uncensored` (a 35B MoE, alias `qwen35-uncensored`,
a different family), and `lmstudio-community/Qwen3.8-27B-GGUF`, which is only
the 888 MB `mmproj` vision projector — the machine's only one. Vision was never
tested in the matrix.
- **Never 6-bit or 8-bit, and never escalate to them.** Capability is flat from
  4-bit to 8-bit — seven of nine arms scored 0.976–0.987, and tool calling,
  instruction following, retrieval and determinism were **perfect on every
  arm**. 8-bit was the *worst* configuration tested: lowest capability of the
  MLX ladder (0.945), slowest (9.5 vs 17.3 tok/s), hottest (95 °C), 61.9 GB
  RAM, and it failed a multi-file agent task 3/3. At 29.5 GB of weights it is
  memory-bound, not compute-bound; its low average wattage is the GPU starving,
  not efficiency.

The practical upshot: **there is no quality tier above 4-bit-class to escalate
to.** If any code routes hard requests to a bigger quant, delete it.

`llm serve qwen27` resolves to the right one automatically. Asking for a
specific path always gets exactly that path.

---

## 6. The patches (and why serving used to be unreliable)

`mlx_lm.server` carries four fixes that are **not upstream**. They are recorded
as patches in `patches/mlx_lm/` and applied into
`.venv/lib/python3.13/site-packages/mlx_lm/server.py`:

| Fix | Without it |
| --- | --- |
| `reasoning_content` alias | multi-turn tool use degenerates into verbatim repetition after ~10 turns |
| 2048-token penalty lookback | the repetition penalty cannot see the loop it exists to stop |
| Metal OOM shield | an OOM kills the generation thread; every client hangs for its full timeout |
| Explicit zero penalties | requests cannot disable nonzero server penalty defaults |

The alias one is worth understanding because it caused a bug that looked
unfixable. The server *emits* assistant thinking as `reasoning`. Qwen's chat
template *reads* past thinking from `reasoning_content`. A client that echoes
the server's own field back therefore loses every prior turn's reasoning, so
the model sees a history in which it appears to have thought nothing,
re-derives from scratch each turn, and eventually just repeats itself. One
field renamed was the entire difference between a looping session and a clean
one.

**This is why the two stacks mattered.** Until 2026-09-21 this folder's `.venv`
had two of the four fixes and Homebrew's Python had all four. `serve_local.sh`
used the venv; Mycelium used Homebrew. Whichever one started your server decided
which bugs you got. Both now use this folder's `.venv`.

**Any reinstall of mlx-lm silently reverts all four** — `pip install -U mlx-lm`,
`uv sync --reinstall`, or a version bump. Nothing prevents that.
Instead `llm doctor` detects it by name:

```
[FAIL] Serving fix reverted: reasoning_content_alias — ... Reapply: .venv/bin/python scripts/apply_serving_patches.py
```

That command restores exactly the missing fixes from `patches/mlx_lm/`. It is
all or nothing: if any patch no longer applies (typically after an mlx-lm
version bump) it writes nothing and names the patch to port by hand.

`serve_local.sh` checks the same markers before every `mlx_lm` launch and
**refuses to start a server that lost a fix**, printing the same command —
consumers such as Mycelium never see doctor output, so the refusal is what
makes a silent revert visible. `--allow-unpatched` serves anyway, for measuring
upstream on purpose. The launcher also never falls back to a `python3` or
`mlx_lm.server` found on `PATH`; without the core `.venv` it stops with an
error instead.

One fix was deliberately *dropped*: MLX 0.32.0 compiled its sampler with the
random state captured on the import thread, and since that state is
thread-local the server's generation thread drew the same value every time —
every temperature > 0 request decoded identically. Mycelium carried a
workaround. Re-measured on 0.32.2 on 2026-09-21 and found absent: five
identical temperature-0.7 requests returned four distinct completions. After
any MLX upgrade, re-check that repeated requests differ before trusting output.

An older edit was *removed* on 2026-09-22: a `_safe_logprob()` wrapper added
during the September showdown for a `Slice indices must be 32-bit integers`
crash that only the Homebrew MLX 0.32.0 server produced. It was slated for
removal when that run finished, then got enforced by accident in the 09-21
consolidation. It was worse than unnecessary: its int32 clamp would read a
*different* token's logprob, and its `0.0` fallback records a failure as
probability 1.0. Nothing here requests logprobs, so removing it changes no
measurement.

---

## 7. When something is wrong

| Symptom | Look here |
| --- | --- |
| Is anything broken? | `llm doctor` — it names reverted fixes and bad paths |
| Launcher says "refusing to serve" | a reinstall reverted a fix; run `.venv/bin/python scripts/apply_serving_patches.py` |
| What is running? | `llm status` — state and loaded model per port |
| Model repeats itself in a long tool session | `llm doctor`; suspect the `reasoning_content` alias |
| Every answer identical at temperature > 0 | the sampler bug in §6; re-measure |
| App freezes on a big paste | context above the ceiling; §4 |
| Server up but nothing completes | dead generation thread; use `--fatal-worker` |
| Suddenly very slow prompt processing | `llm restart` |
| Which command will actually run? | add `--dry-run` to `serve_local.sh` |

`--dry-run` is the honest answer to "what is this thing going to do" — it prints
the exact command and exits without starting anything.


## 8. October 4 audit and current runtime

The serving environment is pinned to MLX 0.32.3, mlx-lm 0.32.0 and mlx-vlm
0.7.4. All three patches were reapplied and doctor checked after upgrading. A
fourth patch (`04-explicit-zero-penalties`) was added on October 5; all four are
now required.
Nine installed chat variants returned correct short completions before and after
the upgrade; this is not full context, tool-use or quality acceptance.

llama.cpp now uses one slot so `-c` is the per-conversation window. Both text
backends explicitly disable min-p (`0.0`); previously llama.cpp silently used
its native `0.05` while MLX used `0.0`. This is a sampling-policy change: do not
treat new quality measurements as identical to the September configuration.

Muse GGUF now resolves with `glimmer` and uses its card's 1.0 / 0.95 / 64
sampling and neutral penalties. Its reasoning cannot be disabled. Bonsai now
resolves with `bonsai`, with explicit non-thinking Qwen policy. Its 65,536 working
budget is `context_policy.recommended_tokens`, inherited from stock Qwen3.8-27B
(added October 5), not a newly measured maximum.

The Qwen embedding model is intentionally excluded from the chat picker. Serve
it through the same launcher (port 8080 must be free):

```bash
MODEL_ROOT=~/.lmstudio/models
bash scripts/serve_local.sh \
  "$MODEL_ROOT/Qwen/Qwen3-Embedding-8B-GGUF/Qwen3-Embedding-8B-Q8_0.gguf" \
  --embedding
```

This mode uses last-token pooling and a 32K context, and exposes
`POST /v1/embeddings`. Query strings should include a task instruction; document
strings should not. A live probe returned a normalized 4096-dimensional vector.
Use `llm stop --port 8080` when finished.

See [the agent benchmark](AGENTIC_BENCHMARK.md) for repeatable local workflow
comparisons.

### October 5 policy follow-up

Native context is now the default; `llm models inspect MODEL` exposes exact resolved metadata and source recipes. The VLM wrapper supplies omitted sampler fields, and patch 04 preserves explicit zero penalties. Mycelium now borrows this metadata rather than enforcing a universal context cap.
