# Local LLM Benchmarking

A local-first control plane for **serving, inventorying and benchmarking LLMs on
Apple Silicon**. It finds the models already on disk, launches the right backend
(`mlx_lm`, `mlx_vlm` or `llama.cpp`) on fixed loopback ports, and runs
reproducible speed and behavioural benchmarks against whatever is being served.

Built for and measured on a single 64 GB M-series Mac. Nothing leaves the
machine: no cloud calls, telemetry or non-loopback serving.

**[Results viewer →](docs/results/index.html)** · [summary data](docs/results/summary.json)

## Highlights

- **One serving path.** `scripts/serve_local.sh` is the only thing that starts a
  model server; the TUI, benchmarks and coding agents (opencode, pi) all borrow
  it. Ports are fixed: `:8085` MLX, `:8080` llama.cpp.
- **Patched `mlx_lm` server.** Four small, documented fixes
  ([patches/mlx_lm](patches/mlx_lm/README.md)): a `reasoning_content` alias,
  repetition and presence penalties, a Metal OOM shield, and respecting explicit
  zero penalties. A doctor command detects when a reinstall has reverted them,
  and a script reapplies them.
- **Safe model inventory.** Only complete models get registered: GGUF headers are
  checked, split GGUFs must have every shard, and MLX safetensors are checked for
  intact headers and tensor ranges. Nothing downloads during serving.
- **Memory-aware benchmarking.** A supervisor stops a queue when free unified
  memory falls below a floor. Peak memory is measured as system unified memory,
  not process RSS, because MLX weights live in Metal buffers.
- **Eval suite.** Tool calling, multi-step agent tasks, hard reasoning,
  grounding, long-context synthesis, needle-in-a-haystack, IFEval-style
  instruction following, and determinism. Optional dataset evals (HumanEval,
  MMLU-Pro, GSM8K) need explicit opt-in flags.
- **Honest reporting.** Partial runs are labelled `exhaustive: false`, aborted
  evals count as missing rather than zero, and capability indices built from
  different sets of axes are flagged as not comparable.

## Headline result: Qwen3.8-27B quantization showdown

Nine quantization and runtime configurations of the same 27B model, each served
at 73,728 context and scored on eight behavioural axes, three repeats each:

| Configuration | Runtime | Disk GB | Gen tok/s | Capability |
|---|---|---:|---:|---:|
| `mlx-community` oQ4 (mixed precision) | mlx_lm | 16.7 | 16.8 | **0.987** |
| bartowski Q5_K_M | llama.cpp | 20.9 | 12.9 | **0.987** |
| lmstudio MLX 4-bit | mlx_lm | 16.1 | **17.3** | 0.984 |
| bartowski Q4_K_M | llama.cpp | 17.4 | 14.9 | 0.984 |
| lmstudio MLX 8-bit | mlx_lm | 29.5 | 9.5 | 0.945 (partial) |

Takeaways for this machine: the mixed-precision 4-bit MLX build matches the best
quality of any configuration at the smallest footprint and nearly the top speed.
Going to 8-bit costs about half the generation speed and buys no measurable
capability. 65,536 tokens is the practical context ceiling on 64 GB; 131K OOMs
during prefill. Full table, per-axis scores and the September serving pass are
in the [results viewer](docs/results/index.html). Methodology:
[docs/QWEN38_SHOWDOWN.md](docs/QWEN38_SHOWDOWN.md).

## Layout

```text
llm.py                  CLI entry point (`llm serve`, `status`, `doctor`, `models` …)
bench_tui.py            interactive benchmark/serving TUI
scripts/                serving lifecycle, model registry, benchmark runners, reports
evals/                  eval suite (offline, agentic, dataset) and the test suite
configs/                model catalog, benchmark profiles, example local config
patches/mlx_lm/         the serving fixes and their README
docs/                   serving guide, benchmark methodology, results viewer
webgui/                 optional legacy web UI
```

## Setup

Requires macOS on Apple Silicon, Python 3.13 and [`uv`](https://docs.astral.sh/uv/).
Models are expected under `~/.lmstudio/models/<org>/<model>/`, which is LM Studio's layout.

```bash
uv sync --extra test --python 3.13
.venv/bin/python scripts/apply_serving_patches.py    # mlx_lm fixes
cp configs/local.example.toml configs/local.toml     # machine-only, gitignored
.venv/bin/python llm.py models sync                  # scan disk for complete models
.venv/bin/python llm.py doctor                       # runtime, patches, storage, endpoints
```

## Usage

```bash
llm serve                # model/backend picker
llm serve qwen27         # a family's preferred variant
llm status               # what is running on each port
llm stop --all           # stop every server started from here
llm models inspect qwen27 --backend mlx   # resolved settings and context provenance
```

`Local AI.command` is a double-click launcher for the same picker.

Benchmarks:

```bash
# showdown: inventory → smoke → core matrix → report
.venv/bin/python scripts/showdown_acquire.py inventory --tier all
.venv/bin/python scripts/showdown_run.py --mode smoke --skip-context-probe --repeats 1
.venv/bin/python scripts/showdown_run.py --mode core --run-dir results/qwen38_showdown/core01
.venv/bin/python scripts/showdown_report.py results/qwen38_showdown/core01
```

Runs can be resumed (`--resume`), narrowed (`--group`, `--arm`, `--evals`) and
previewed without running anything (`--list`, `--dry-run`). Results go to
`results/`, which is gitignored.

## Tests

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/smoke_test.py
```

The smoke test is a static gate. Besides checking that scripts compile and
schemas agree, it fails if any tracked file contains a local path, a
credential-like value, or a name listed in your private `.leakpatterns` file
(see `.leakpatterns.example`).

## Documentation

| Document | Purpose |
|---|---|
| [docs/SERVING.md](docs/SERVING.md) | What is served where, lifecycle, every knob, the mlx_lm fixes |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | Methodology, metrics, eval scoring, interpretation |
| [docs/QWEN38_SHOWDOWN.md](docs/QWEN38_SHOWDOWN.md) | The Qwen3.8-27B quantization experiment |
| [docs/AGENTIC_BENCHMARK.md](docs/AGENTIC_BENCHMARK.md) | Short local agent-task preset |
| [docs/ADDING_MODELS.md](docs/ADDING_MODELS.md) | Model storage, discovery, catalog fields |
| [docs/MODEL_NOTES.md](docs/MODEL_NOTES.md) | Model landscape notes |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development conventions |

## License

MIT
