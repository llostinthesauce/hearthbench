# Qwen3.8-27B configuration showdown

A reproducible experiment that answers one question:

> What behavioural capability do I lose by reducing precision, and what
> memory/context capacity do I gain in exchange?

It is deliberately *not* a throughput benchmark. Tokens per second is recorded,
but nothing is ranked on it, because a fast configuration that has lost tool
calling is not a faster version of the same assistant.

---

## What this model actually is

`Qwen/Qwen3.8-27B` is `model_type: qwen3_5`, and two properties of it shape the
whole design:

**Hybrid attention.** Of 64 text layers, `layer_types` alternates three
`linear_attention` layers then one `full_attention` (`full_attention_interval:
4`). Only 16 layers hold a conventional KV cache; the other 48 carry a recurrent
linear-attention state. So KV-cache growth with context is roughly a quarter of
what a comparable all-full-attention 27B would show, and **KV quantization buys
far less here than the usual rules of thumb suggest**. That is a hypothesis the
`kv_study` arms measure rather than assume.

**Multimodal.** `vision_config` is present and image/video token ids are
defined. `mlx_lm.server` loads it as text; the GGUF ships a separate `mmproj`.
Nothing in this experiment tests vision.

Other metadata worth knowing: `attn_output_gate: true`, `head_dim: 256`,
`hidden_size: 5120`, and a model card sampling config of temperature 1.0 /
top_p 0.95 / top_k 20. Scored evals run at temperature 0 except the reliability
repeats, and the value used is recorded per arm.

---

## Serving-path capability (verified 2026-09-18)

| | `mlx_lm.server` 0.31.3 | `mlx_vlm.server` 0.6.17 | `llama-server` b10809 |
|---|---|---|---|
| Loads `qwen3_5` | yes | yes | yes |
| KV quantization | **no** (`--kv-bits` does not exist) | `--kv-bits`, `--kv-key-bits`, `--kv-value-bits`, `uniform`/`turboquant`, `--kv-group-size` | `-ctk` / `-ctv` |
| Speculative drafting | `--draft-model` exists, but there is **no `qwen3_5_mtp` implementation** in `mlx_lm.models` | `--draft-model` + `--draft-kind mtp` | `--spec-type draft-mtp` |
| Default KV in this repo | native fp16 | native unless `--kv-bits` | **`q8_0`** via `serve_local.sh` |

Two consequences drive the matrix:

1. **KV quantization and MTP live only on `mlx_vlm.server`.** Comparing a
   quantized-KV `mlx_vlm` arm against the `mlx_lm` ladder would change the
   server and the KV mode at the same time. Every KV and MTP arm therefore has
   an `mlx_vlm` **KV-native control arm**, selected with `--kv-bits 0`.

2. **llama.cpp's default is already KV-quantized.** `serve_local.sh` passes
   `-ctk q8_0 -ctv q8_0`. The GGUF ladder runs at that default — stated, not
   hidden — and the KV study adds an explicit `--kv-type f16` baseline.

---

## The artifact selection

Committed, path-free, in [`configs/qwen38_showdown.json`](../configs/qwen38_showdown.json).
Every artifact is pinned to a 40-character revision SHA and verified against the
sha256 Hugging Face publishes as each file's LFS object id.

**The MLX ladder is one provider, one method, one date.** All four
`lmstudio-community/Qwen3.8-27B-MLX-{4,5,6,8}bit` builds are `affine`,
`group_size=64`, with no per-module overrides, converted by the same team on the
same day. Only `bits` differs, which is what makes the ladder answer the
question it is asked. Measured bits-per-weight: 4.69 / 5.67 / 6.65 / 8.61.

**The GGUF ladder is one provider.** `bartowski/Qwen3.8-27B-GGUF` is the only
reputable source carrying the complete Q4_K_M / Q5_K_M / Q6_K set for this
model; `lmstudio-community` has no Q5_K_M at all.

**Two files labelled the same thing are not the same file.** Three "Q4_K_M"
builds of this model exist at 16.81 GB (lmstudio-community), 17.44 GB
(bartowski, imatrix) and 18.97 GB (ggml-org). The already-on-disk
lmstudio-community build is kept as a zero-cost `provider_contrast` arm, so any
gap between it and bartowski's Q4_K_M is attributable to conversion recipe
rather than to quant level.

**The method contrast.** `mlx-community/Qwen3.8-27B-oQ4` is an oQ mixed-precision
conversion: base 4-bit with 161 per-module bit overrides, landing at 4.86 bpw
against the uniform 4-bit's 4.69. Near-identical footprint, different bit
allocation — the controlled test of sensitivity-aware quantization at the
memory-constrained end. The artifact is ordinary MLX safetensors; **using it does
not reintroduce oMLX as a serving path**, which stays retired per the serving notes.

**Derivatives, chosen after research on all five candidates.** Selected:
`huihui-ai/Huihui-Qwen3.8-27B-abliterated` (highest adoption, still maintained,
single-axis modification with a documented layer range) and
`OBLITERATUS/Qwen3.8-27B-OBLITERATED` (the only candidate publishing a full
machine-readable `abliteration_metadata.json`). Both ship a Q6_K that is
byte-for-byte the same size as the stock Q6_K, indicating the same quant recipe
and therefore a clean single-variable comparison. The three rejected candidates
and the reason for each are recorded in the selection file — briefly: HauhauCS
requires a patched llama.cpp, 0bserverx ships test blobs and mismatched file
names, and DavidAU's base model is itself a multi-stage merge, so no measured
difference could be attributed to any one modification.

---

## What is measured

**Systems.** Model load time to first successful completion, peak unified
memory, steady-state memory, prompt (prefill) tok/s, generation tok/s, TTFT at
256 / 8K / 32K prompts, practical maximum context, actual disk footprint, and
every server failure.

> Peak memory is reported as **system used memory above the pre-launch
> baseline**, not process RSS. MLX allocates weights through Metal buffers that
> never enter the resident set — a measured 15 GB 4-bit checkpoint reported
> 7.9 GB of RSS while system used memory rose ~12.6 GB. RSS is still recorded
> (it is the right figure for llama.cpp) but it systematically under-reports MLX.

**Behavioural**, one eval per capability, each with its own failure taxonomy:

| Eval | Capability | What its failure modes distinguish |
|---|---|---|
| `toolcall` | native tool use | emission · selection among plausible siblings · argument names/types/JSON validity · parallel calls · interpreting a returned result · recovering from a tool error · *not* calling a tool when none is needed |
| `agentic` | 5–15 step trajectories | task success · state retention · recovery from a failed first action · instruction persistence · multi-file synthesis · repetition loops |
| `reasoning_hard` | hard reasoning | planted distractors · multi-constraint · classic traps · multi-step propagation, each repeated for reliability |
| `grounding` | hallucination | answerable vs unanswerable · contradictory sources · partial evidence · **and over-refusal**, which the opposite fix causes |
| `longctx` | long context | successful retrieval · failure to *combine* two retrieved values · instruction drift · fabricated answers · misattributed answers |
| `ifeval_local`, `niah`, `determinism` | existing suite | unchanged, reused |

Coding is covered by the `agentic` tasks — debugging, interpreting a test
failure, targeted modification, multi-file reasoning, and using tools rather
than hallucinating repository state are all assertions in that fixture — plus
the existing tier-2 `humaneval` when explicitly enabled.

### Scoring honesty

- A capability with fewer than 4 scorable cases prints `n/a`, never `0.000`.
- An arm whose server never became ready is `startup_failed`, is excluded from
  every score, and appears in its own table with its log path. **A failed launch
  is never a low quality score.**
- Agent tasks report `passed/trials` and a verdict of `reliable` /
  `intermittent` / `never`. A task that succeeds 1 time in 3 is reported as
  intermittent, not as 0.33.
- Agent assertions separate **outcome** checks ("did it achieve the thing") from
  **constraint** checks ("did it avoid what it was told not to do"). A
  do-nothing trajectory satisfies every constraint vacuously, so constraints
  alone earn nothing.
- `Δ vs best` is measured against the best arm **in this run**, and the
  reference arm is named. It is a relative claim about the configurations
  tested, not an absolute quality scale.

---

## Bounded agent testing

`evals/agentic/` drives a real tool-using loop against a scratch fixture
repository, with every bound made explicit:

- fixed fixture, rebuilt byte-identically before every trial;
- an isolated scratch directory — every tool refuses a path that escapes it;
- caps on steps, wall clock, cumulative output tokens, and identical repeated
  calls, with the **stop reason recorded** so "ran out of steps", "looped" and
  "answered" stay distinguishable;
- a per-task tool allowlist; a tool outside it is refused, not executed;
- full JSONL transcripts, so an ambiguous trajectory can be read later without
  any of it passing through an operator's context;
- process cleanup after every arm, including waiting for the port to go quiet.

**`run_tests` does not execute model-written code.** It is a static checker that
inspects file contents and renders pytest-shaped output. The behaviour under test
is "can the model read a failure, locate the cause, and fix it", which that
reproduces exactly. The repository already treats executing model code as an
explicit opt-in (`--allow-code-execution`), and an unattended matrix across a
dozen quants is the wrong place to turn it on by default.

---

## Commands

### Inspect what is on disk

```bash
.venv/bin/python scripts/showdown_acquire.py inventory --tier all
```

### Download the matrix

```bash
.venv/bin/python scripts/showdown_acquire.py plan --tier core method
.venv/bin/python scripts/showdown_acquire.py download --tier core method derivative
.venv/bin/python scripts/showdown_acquire.py verify --deep
```

Resumable, skips complete files, never deletes, refuses to start if fewer than
50 GB would remain. After a download, refresh the registry:

```bash
.venv/bin/python scripts/discover_models.py --write configs/models.local.json
```

### Smoke benchmark

```bash
.venv/bin/python scripts/showdown_run.py --mode smoke --skip-context-probe --repeats 1
```

### Core showdown — the weight-quantization ladder

```bash
.venv/bin/python scripts/showdown_run.py --mode core \
  --run-dir results/qwen38_showdown/core01
```

### Full matrix — adds KV, MTP, reasoning, and derivatives

```bash
.venv/bin/python scripts/showdown_run.py --mode full \
  --run-dir results/qwen38_showdown/full01
```

### Generate the report

```bash
.venv/bin/python scripts/showdown_report.py results/qwen38_showdown/full01
```

### Useful narrowings

```bash
# see the arms without running anything
.venv/bin/python scripts/showdown_run.py --mode full --list
.venv/bin/python scripts/showdown_run.py --mode full --dry-run

# resume an interrupted matrix
.venv/bin/python scripts/showdown_run.py --mode full --resume --run-dir <dir>

# one sub-study, one artifact, or one capability
.venv/bin/python scripts/showdown_run.py --mode full --group kv_study
.venv/bin/python scripts/showdown_run.py --mode full --artifact mlx_lms_6bit
.venv/bin/python scripts/showdown_run.py --mode core --evals toolcall agentic
```

---

## Serving context

Every arm is served at a single explicit context, `--serve-ctx`, default
**139264** (the eval suite's 131072 sizing cap plus 8192 of answer headroom).
This is a deliberate departure from the registry's `ctx_cap` of 262144:

- llama.cpp allocates `-c` **up front**. With `-ctk/-ctv q8_0` and this model's
  16 full-attention layers (4 KV heads x 256 head dim), that is ~9.1 GB of KV at
  262144 against ~4.9 GB at 139264 — paid on every GGUF arm whether or not a
  long prompt ever arrives. On 64 GB that is the difference between a Q6_K arm
  sitting near 28 GB and near 33 GB before anything else on the machine.
- MLX grows its cache lazily, so an MLX arm that never sees 262K never pays for
  it. Leaving the two runtimes on different effective allocations would make
  `Peak UM GB` and `Usable ctx` partly a serving-flag artifact rather than a
  property of the configuration.
- The serving notes record llama-server wedging on a 131K prompt when served
  at `-c 262144`.

The practical-context probe is capped at the same value, so both runtimes climb
the same ladder and `Usable ctx` can never exceed what was actually allocated.
The runner refuses a `--serve-ctx` smaller than `--ctx-cap` plus headroom and
says so. Explore higher deliberately:

```bash
.venv/bin/python scripts/showdown_run.py --mode core --serve-ctx 270336 --ctx-cap 262144
```

## Runtime expectations

The core matrix is 9 arms. Each pays a model load, three throughput probes (the
32K probe alone costs ~80 s of prefill on 4-bit MLX), a context ladder, and the
eight-eval behavioural suite. Budget hours, not minutes, and run it detached:

```bash
tmux new-session -d -s qwen38 -c "$PWD" \
  '.venv/bin/python scripts/showdown_run.py --mode core --run-dir results/qwen38_showdown/core01 2>&1 | tee results/qwen38_showdown/core01.log'
```

Every arm writes `result.json` before the next one starts, so an interruption
costs at most the running arm and `--resume` picks up from there. A run that did
not finish is marked `exhaustive: false` and the report says so in its header —
per the serving notes, a partial matrix must not be described as a complete one.

---

## Known limits on discriminative power

Measured on stock MLX 4-bit, the *bottom* of the ladder, at greedy decoding:

| Axis | 4-bit score | Headroom |
|---|---|---|
| `toolcall` (original 17) | 17/17 | none |
| `agentic` (1 trial each) | 5/5 | none |
| `reasoning_hard` (greedy only) | 11/11 | none |
| `ifeval_local` | 12/12 | none |
| `grounding` | 11/12 | some |

A 27B model does not lose these capabilities at 4-bit, so most axes will read
near 1.0 for every arm and the interesting differences have to come from
somewhere else. Three things carry that load:

1. **Reliability, not accuracy.** `reasoning_hard` and `agentic` repeat at
   `--repeats 3` / 2 trials with sampling on trials after the first. The probe
   above ran one greedy trial each, which switches that dimension off entirely.
   A quant that answers correctly 2 times in 3 is reported `intermittent`, and
   that is the signal to read.
2. **Harder fixtures added after the probe** — which 4-bit also passed. Six
   `hard` tool-call cases against a 10-tool set with deliberately confusable
   neighbours (current vs historical price, forecast vs current weather, batch
   vs repeated conversion, a derived argument no tool provides, a stale prior
   result, a contradicted premise) scored **6/6 on 4-bit**, and the three added
   `hard` reasoning problems scored **2/3** — the one miss being a token-budget
   artifact (no `ANSWER:` line after 1536 tokens of working), since fixed by
   giving that group a 3072-token budget. They are better tests
   than what they joined — a 10-tool set makes the original `select` cases
   harder, and the reasoning additions need an exact fraction, an exact
   enumeration, or a per-step flooring rule — but they did **not** open up
   headroom. That is evidence about the model, not a reason to keep escalating:
   single-shot behavioural accuracy is simply not where this model degrades
   between 4 and 8 bit.
3. **Long context and memory.** `longctx` at 32K/64K/128K and the practical
   context probe are where a smaller quant has the most to gain and the most to
   lose, and nothing there is saturated.

If the core run still returns near-identical behavioural scores across 4/5/6/8
bit, the honest reading is that **this model at this size does not lose measurable
capability across that range on these tasks** — and the decision should then be
made on memory, context headroom, and speed. That is a real result, not a
harness failure. What it does not license is the reverse claim that all quants
are equivalent for tasks this suite does not cover.

## What this experiment does not establish

- Nothing about vision, despite the model being multimodal.
- Nothing about quality above the best artifact tested; `Δ vs best` is relative
  to this run.
- No claim that a derivative's behaviour generalizes beyond the one quant tested.
- Reasoning arms stop at `medium`. The serving notes record a reproduced
  hidden-reasoning loop at both `low` and `medium` on exact-word-count prompts,
  and Pi already caps `xhigh` at `medium`. Running higher would re-run a known
  failure, not add an experiment.
