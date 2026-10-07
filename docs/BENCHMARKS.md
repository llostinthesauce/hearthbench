# Benchmarks

Two independent measurements, joined at the end.

- **Speed** — four passes of increasing context and output length, recording
  throughput, time-to-first-token, and peak memory.
- **Quality** — seven evals, all scored by code. No judge model, no API key, no
  rubric that can drift between runs.

The join is the point. Either half alone will mislead you.

## What is comparable to what

### Recommended pass for a primary workstation

Run all complete catalog models, one at a time, with an eight-hour total budget:

```bash
.venv/bin/python scripts/run_practical_benchmarks.py --output-dir results/practical_RUN_ID
```

Use a new output directory for each run. Add `--dry-run` to inspect the exact
models and server settings without starting anything. Run the command in a
persistent terminal session for unattended use.

`local_practical` measures 1K/8K/32K inputs, with a 512-answer-token minimum
and a 768-server-token generation cap, two repeats, cold and warm, and one
request at a time: 12 measured requests per model. Tokenizers differ; short
answers remain flagged rather than silently counted as valid. Two repeats
provide a basic comparison, not a robust tail-latency estimate.

The catalog runner gives each model at most 70 minutes, each request a 300-second
timeout, and the whole queue at most eight hours including a cleanup allowance.
It tracks and stops only its workers and their model-server descendants, even
when servers create separate process sessions. Sustained low available system
memory stops the queue. MLX models start through `scripts/serve_local.sh` on a
private port, so they run on the same runtime and defaults as daily serving;
GGUF models use a one-request llama.cpp configuration. These are workstation-safe
test settings, not changes to daily serving configuration. GPU use can still
make other applications slower during generation. (The September 5 pass predates
this and served MLX through oMLX, since removed; its numbers are not comparable
to later runs.)

Completed batches survive timeouts in `requests.jsonl`; each model's log and
the top-level `queue.json` distinguish completed, non-conformant, timed-out,
and unrun jobs. The eight-hour cap bounds resource use, not guaranteed model
success. It does not require or perform the 100K/concurrency-10 stress test.

### Full methodology and diagnostic lanes

There are now three deliberately separate lanes:

| Lane | Purpose | Scoring/aggregation | Claim |
|---|---|---|---|
| Legacy speed passes | Fast local regression and context stress | Per-run metrics | Diagnostic only |
| Deterministic quality suite | Retrieval, instruction following, code, math | Code-scored | Comparable across this repo's controlled runs |
| `aa_local_full` profile | Standard workload shapes and streaming service metrics | Median plus p95 | Locally conformant only after result validation |

The full profile fixes short/medium/long workload classes at 1K/1K,
10K/1.5K, and 100K/2K input/minimum-output tokens. It requires sequential and
10-request concurrency modes, explicit cold/warm cache labels, at least five
samples, and these fields:

- time to first token and time to first answer;
- prompt and output tokens per second;
- end-to-end time;
- peak memory in bytes;
- cache state.

If a backend does not expose a measurement, the field remains unsupported. For
example, a client cannot infer trustworthy prompt processing speed or
process-specific memory merely from wall time. `aa_local_smoke` uses smaller
workloads and is permanently marked reduced and non-comparable.

Inspect either profile without contacting a model:

```bash
.venv/bin/local-ai bench --profile-info aa_local_full
python3 scripts/benchmark_profiles.py --profile aa_local_smoke --json
```

Execute the reduced profile against a running service:

```bash
.venv/bin/local-ai bench --prepare-tokenizer
.venv/bin/local-ai bench --profile aa_local_smoke \
  --url http://127.0.0.1:8085/v1 --model MODEL_ID --no-thinking
```

The tokenizer preparation downloads only the public `o200k_base` vocabulary and
verifies its hash. Normal measurement refuses to download a missing vocabulary.
Synthetic prompts are sized with that shared tokenizer; answer and visible
reasoning tokens are counted separately. The output minimum applies to answer
tokens. Native server usage is retained separately.

`--no-thinking` requests `enable_thinking=false` from compatible templates and
records that request in the run. Omit it to measure the server's default reasoning
mode. Sampling, output caps, timeouts, runtime versions and source identity are
saved with the run. If a server repeats unfinished reasoning as answer content,
the runner flags it and counts those tokens only once.

For cold and warm runs, supply a foreground server command on a free port:

```bash
.venv/bin/local-ai bench --profile aa_local_full \
  --url http://127.0.0.1:8085/v1 --model default_model \
  --server-command 'bash scripts/serve_local.sh qwen27 --backend mlx 8085'
```

`default_model` is the installed `mlx_lm.server` alias for its CLI-selected model.
For `mlx_vlm`, use the full model directory as the API ID.

The runner owns and restarts only this process, once per batch. It waits for a
separate one-token completion so model loading is excluded from the measurement.
A cold batch uses the fresh service and unique prompt; a warm batch first primes the exact
prompts it measures. OS file caches and persistent backend caches are not
purged. Each mode runs the configured number of batches; a 10-request batch
does not count as 10 independent repeat samples. This is a substantial workload,
especially the 100K/10-way case, so check model context and available memory.

`requests.jsonl` records each request with run/model/profile/tokenizer provenance.
`summary.json` reports median/p95 values, failures and profile conformance.
Per-request speed is separate from total batch throughput (visible tokens divided
by the batch's end-to-end wall time). Summaries distinguish individual observations
from independent repeated batches. Memory means sampled server process-tree RSS
for the whole concurrent batch; it does not claim total Metal allocation. An
external server's memory is unsupported. Prompt speed is present only when the
server exposes it, and uses the server's native tokenizer. Visible output speed
uses `o200k_base`; hidden reasoning cannot be measured by this client.

Validate a saved result set independently:

```bash
python3 scripts/benchmark_profiles.py --profile aa_local_full \
  --validate-results results/YOUR_RUN/requests.jsonl
```

The validator requires all workload/mode/cache combinations, complete repeated
batches, input lengths within 10% of target, minimum answer lengths, and explicit
unsupported metrics. Failed and underfilled requests remain visible and prevent
a conformance claim. The cache labels describe the protocol above, not verified
hardware cache misses or public leaderboard equivalence.

The exhaustive runner snapshots the committed contract beside each run, while
its existing micro/normal/high/max passes retain their legacy names. This avoids
turning an old diagnostic into a new methodology by changing only its label.

### Blind local pairwise review

`bench_head_to_head.py` exports two files when exactly two models are compared:

- `pairwise_judging_*.jsonl` contains only Candidate A/B responses, randomized
  per prompt and repeated in swapped order;
- `pairwise_key_*.json` reveals the model identities after judgments are saved.

Set each judging record's winner to `A`, `B`, or `tie`, then use the
`bradley_terry_scores()` analysis hook in `scripts/benchmark_profiles.py`.
Bootstrap sample counts are pinned in the profile for confidence-interval work.
Resampling keeps a prompt's forward/reverse judgments together and reports raw
percentile intervals. Unjudged records do not produce scores.
The evaluator must be a human or a fully local model; this workflow never sends
private responses to a cloud judge.

Methodology references: [Artificial Analysis performance benchmarking](https://artificialanalysis.ai/methodology/performance-benchmarking)
for workload and service-latency shape, and [Arena-Hard](https://blog.lmarena.ai/blog/2024/arena-hard/)
for swapped-order pairwise comparison and Bradley-Terry aggregation. This repo
adopts those useful structures without claiming identical prompts, hardware,
traffic, judges, or leaderboard comparability.

---

## Design rules

Everything here follows four rules, and they explain most of the odd decisions
elsewhere in the repo.

**1. The grader must not be able to be wrong.** Every eval is exact-match,
programmatic constraint checking, or test execution. An LLM judge introduces a
second model's failures into your measurement of the first, and its verdicts
change when the judge is updated — which makes historical runs incomparable.

**2. Offline by default.** Tier 1 generates its own data. A benchmark that pauses
to download has already invalidated its own timing, and an air-gapped machine is
a normal place to be evaluating local models.

**3. Deterministic.** Every generated eval is seeded. Two runs of `niah` on
different machines produce byte-identical prompts, so scores are comparable
across hardware.

**4. Report what was actually sent.** Prompt sizing uses an estimated tokens-per-word
ratio, but reports come from the server's own `prompt_tokens`. Estimates size the
work; measurements describe it.

---

## Speed passes

| Pass | Context | Output | Prompt |
|---|---|---|---|
| `micro` | 1K | 128 tok | Fibonacci function |
| `normal` | 16K | 1K tok | Async rate-limited API client |
| `high` | 64K budget | 2K tok | Django monolith → microservices architecture |
| `max` | fills `ctx_cap` | 4K tok | Distributed time-series DB spec, with synthetic filler to fill the window |

All prompt text lives in `scripts/prompts.py` and is imported everywhere. It is
never duplicated — `smoke_test.py` fails the build if it is, because a prompt
that drifts between two runners silently makes their numbers incomparable.

### Token counting honesty

Not every backend reports real token counts. Results carry a `token_count_method`
and the aggregator ranks it:

| Method | Rank | Meaning |
|---|---|---|
| `hf_tokenizer`, `mlx_native`, `llama_bench_native` | 5 | Real count |
| `openai_usage` | 4 | Server's `usage` block |
| `word_fallback*` | 1–2 | **Estimated from whitespace** |

Anything at rank ≤ 3 is flagged `approximate`, and the aggregator reports a
separate "trusted" winner alongside the raw fastest. A model whose throughput was
computed from word counts can look 20% faster than one measured properly — the
distinction is not pedantry.

---

## Quality evals

### Tier 1 — offline, self-generating

#### `niah` — needle-in-a-haystack

Builds a deterministic prose haystack at a target token length, inserts numeric
needles at fixed relative depths, and asks for them back. Scored on exact digit
match.

Reported per context length, which is the shape that shows *where* a window stops
working:

```json
"by_context": { "1024": 1.0, "16384": 1.0, "65536": 0.6, "131072": 0.2 }
```

That model has a real working context somewhere between 16K and 64K, whatever its
`ctx_cap` claims. The haystack is varied prose, not a repeated token: repetition
compresses in ways that make retrieval artificially easy.

#### `ifeval_local` — verifiable instruction following

Locally authored prompts, each carrying constraints a function can check: word
counts, valid JSON with required keys, no commas, exact bullet counts, forbidden
words, casing, `***`-separated paragraphs.

Two scores, matching the IFEval paper:

- **strict** — the response satisfies every constraint as written
- **loose** — the same check after stripping code fences and boilerplate openers
  ("Sure! Here is…"), which otherwise fail formatting rules for reasons unrelated
  to instruction following

Instruction-following degrades before fluency does. A 4-bit model that still
writes clean prose will start ignoring "exactly three bullet points" well before
anything else looks wrong.

#### `determinism` — reproducibility at temp 0

Same prompt, `temperature=0`, N repeats. Greedy decoding is a pure function;
identical inputs must give identical outputs. When they do not, something in the
stack is non-deterministic and every other number inherits that noise.

- **exact_match_rate** — fraction byte-identical to the first response
- **prefix_stability** — mean shared-prefix length, which localizes the drift.
  Divergence at token 3000 is a long-context or cache problem; at token 5 it is a
  sampler still sampling despite `temperature=0`.

### Tier 2 — public datasets, opt-in

Nothing downloads without `--fetch`. Files land in `evals/.cache/` (git-ignored)
and are recorded in a manifest with their SHA-256, so an upstream change to a
supposedly-immutable file gets surfaced loudly.

| Eval | Source | Scoring |
|---|---|---|
| `gsm8k` | openai/grade-school-math | Exact match on `#### <n>`, falling back to the last number |
| `humaneval` | openai/human-eval | pass@1 by executing the reference tests |
| `ifeval` | google-research IFEval | The `ifeval_local` verifiers, applied to the real prompt set |
| `mmlu_pro` | TIGER-Lab/MMLU-Pro via the HF datasets-server | Exact match on the chosen letter |

**Two caveats worth stating plainly:**

- `ifeval` scores only the instruction ids this repo implements verifiers for,
  and reports that fraction as `coverage`. Scores are comparable **across models
  you run yourself**, but not to published IFEval numbers.
- `mmlu_pro` samples a subset. Sampling is seeded, so the subset is identical
  across models — but a 150-question sample has real confidence intervals, and
  a 2-point gap between two models is noise.

#### HumanEval executes model-written code

That is inherent to the benchmark — functional correctness cannot be checked
without running the function. Mitigations: subprocess per problem, wall-clock
timeout, CPU and address-space rlimits, scratch working directory, and a guard
prelude that nulls `os.system`, `subprocess.*`, and `shutil.rmtree` in the child.

**None of that is a security boundary.** It stops a buggy completion from doing
damage. It does not stop a deliberately malicious one — the child is a normal
process running as you, with network and filesystem access. Evaluating a model
you do not trust belongs in a VM or container.

Requires `--allow-code-execution` on top of `--fetch`.

---

## Serving quirks that will bite you

Measured on an M5 Pro, llama.cpp b10090 / mlx-lm 0.31.3, and all three cost real
benchmark runs before being understood:

| Symptom | Cause | What to do |
|---|---|---|
| Server "up" but first prompt hangs forever | `/v1/models` returns 200 *while the model is still loading*; `/v1/chat/completions` returns 503 | Probe readiness with a real one-token completion — what this harness now does |
| Server stops answering mid-run, 0% CPU, never recovers | llama-server at `-c 262144` wedges permanently on a ~131K-token prompt | Cap `--ctx-cap` (16K was reliable here). The runner aborts after 3 consecutive failures rather than grinding through timeouts |
| "server never came up" on a big MLX model | Cold-loading ~28 GB of 6-bit safetensors took longer than a 600s budget | Readiness now waits up to 1800s; waiting is free when the server is healthy |

The general rule for telling "slow" from "wedged": check whether the server is
actually burning CPU. `ps aux | grep llama-server` and watch the TIME column —
an idle server with a waiting client is wedged, not thinking.

### Reusing an already-running MLX server

Both API runners can measure an existing server. Pass `--api-base` to
`bench_mlx_api.py`; it then skips startup and measures that endpoint. Without
the option, the MLX runner owns the service lifecycle on port 8085.

## Running them

```bash
# Offline suite
python3 scripts/bench_quality.py --model qwen35 --url http://127.0.0.1:8080/v1

# Everything, including downloads and code execution
python3 scripts/bench_quality.py --model qwen35 \
    --evals all --fetch --allow-code-execution

# Quick pass while iterating
python3 scripts/bench_quality.py --model qwen35 --limit 20

# See the plan without contacting anything
python3 scripts/bench_quality.py --model qwen35 --evals all --dry-run
```

Useful flags: `--ctx-cap` (bounds `niah`), `--limit` (caps cases per eval),
`--repeats` (determinism repeats), `--seed`, `--backend` / `--quant` (labels that
end up in the CSV and drive the join).

## Reading the output

```bash
python3 scripts/aggregate_results.py results/
```

Produces `summary.md` and `summary.json` with:

- **Speed × Accuracy** — the combined leaderboard, sorted by eval mean then
  throughput. Joined on `(family_id, quant)`, deliberately *not* on backend:
  accuracy is a property of the weights, so a family's scores apply to whichever
  engine served it fastest.
- **Quality Evals** — per model × eval, with case counts, errors, mean score, and
  strict pass rate
- **Serving Verdict** — fastest and *trusted-fastest* variants
- **MTP deltas** — speculative-decoding speedup where measured

### Two numbers that are easy to confuse

**Mean vs strict.** `mean_score` gives partial credit (3 of 4 needles found =
0.75). `strict_pass_rate` counts only fully-correct cases. A large gap means the
model is *nearly* right a lot — usually a formatting problem rather than a
knowledge one.

**Errors vs failures.** `errors` counts requests that never completed — a dead
socket, a timeout, an HTTP 500. Those are excluded from the score rather than
counted as wrong, because blaming the model for a transport failure understates
it. A run with a high error count should be re-run, not interpreted.

> One aggregation detail worth knowing, because it bit this repo during
> development: throughput averages skip zeros (0 t/s means "never measured"),
> while score averages must include them (0.0 means "got it wrong"). They use
> different helpers — `_mean` and `_score_mean` — and there is a regression test
> pinning the difference.


## Qwen3.8-27B configuration showdown

The quantization/KV/MTP/reasoning matrix has its own methodology document:
[QWEN38_SHOWDOWN.md](QWEN38_SHOWDOWN.md). It reuses this suite's definitions of
readiness, truncation, and scoring, and adds five behavioural evals
(`toolcall`, `agentic`, `reasoning_hard`, `grounding`, `longctx`) that are
opt-in — the `tier1` shorthand is unchanged, and the new set is reachable as
`behavioral`, `tier1_all`, or `showdown`.
