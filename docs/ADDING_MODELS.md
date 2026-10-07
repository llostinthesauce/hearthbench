# Adding a Model

Two config files, and most of the time you only touch one.

| File | Committed? | Contains |
|---|---|---|
| `configs/model_catalog.json` | yes | Public matching rules: family names, architecture routing, sampling recipes and provenance, filename patterns |
| `configs/models.local.json` | **no** (git-ignored) | Generated. Real filesystem paths for your machine |

The split exists so the repo can describe *what a Qwen3.6-35B is* without ever
committing where yours happens to live.

## The fast path

```bash
# 1. Download weights under a root you already scan
hf download poolside/Laguna-XS-2.1-GGUF \
  --local-dir ~/.lmstudio/models/poolside/Laguna-XS-2.1-GGUF

# 2. Re-scan
llm models sync

# 3. Confirm
python3 scripts/model_registry.py list
```

Unmatched models are registered as `custom`. `scripts/model_policy.py` reads
installed MLX text metadata/generation settings or GGUF architecture/context
metadata. Known architecture routing is shared; unknown architectures retain
neutral sampling and require a real acceptance test before claiming support.
Run `llm models inspect MODEL --backend mlx` (or `llamacpp`) to inspect sources.
A familiar brand name alone does not establish future-generation compatibility.

Audit an old root before migrating it:

```bash
python3 scripts/discover_models.py --roots ~/old-model-folder
```

That command only prints what it finds. Move any usable weights into
`~/.lmstudio/models/<org>/<repo>`, then run `llm models sync`. Do not write
a multi-root registry: the control plane intentionally uses one canonical root.

## Registering it properly

To get a real alias, documented sampling exceptions and a friendly alias, add a family to
`configs/model_catalog.json`:

```json
{
  "id": "laguna_xs_33b_moe",
  "name": "Poolside Laguna XS 2.1 (33B-A3B MoE)",
  "family": "laguna",
  "architecture": "moe",
  "context_policy": {"default": "native"},
  "temperature": 0.7,
  "top_p": 0.8,
  "top_k": 20,
  "gguf_patterns": ["Laguna-XS-2\\.1.*\\.gguf$"],
  "mlx_patterns": ["Laguna-XS-2\\.1.*(?:MLX|mlx|4bit|5bit|6bit|8bit)"]
}
```

### Fields

| Field | Why it matters |
|---|---|
| `id` | Stable identifier. Results group by it, so changing it splits your history |
| `family` | Drives per-family behaviour — e.g. `gemma4` gets no system prompt, because its template does not take one |
| `architecture` | `moe` or `dense`. Reported in results; MoE and dense are not comparable on tokens/sec alone |
| `context_policy` | Native serving by default; advisory recommendations and verified workload evidence remain separate |
| `sampling_recipe` / `sampling_modes` | Reference shared recipes with source links; modes can differ without duplicating each quant |
| `architecture_policy` | Shared backend routing by installed `model_type`; add only verified runtime support |
| `temperature` / `top_p` / `top_k` | Model-card values. Wrong sampling makes a good model look broken |
| `gguf_patterns` / `mlx_patterns` | Python regex, matched case-insensitively against both the full path and the bare filename |
| `preferred` | Which variant a family-level selector resolves to, per backend. See below |

### `preferred` — choosing the family's default variant

A family usually has several variants on disk (4-bit, 5-bit, a GGUF). Asking for
the family (`llm serve qwen27`) used to return whichever one discovery listed
first, so the default was decided by filesystem ordering rather than by
measurement — which is how `qwen27` kept resolving to MLX-4bit after oQ4 was
measured better at the same footprint.

Name the winner per backend:

```json
"preferred": {
  "mlx": "Qwen3.8-27B-oQ4",
  "llamacpp": "Qwen3.8-27B-Q5_K_M.gguf"
}
```

The value is the variant's **filename or directory name**, not a full path.
Rules:

- Only family-level asks are affected. An exact path, registry selector, or
  variant name is never redirected — `llm serve <path>` always serves that path.
- If the preferred variant is not on disk, resolution falls back to the first
  complete variant, so a deleted default degrades instead of breaking.
- Like `mlx_server` and `mtp_supported`, it belongs on the **catalog family**.
  `discover_models.py` carries it into each generated entry, so it survives a
  re-scan; a value hand-written into `models.local.json` does not.

Verify with `bash scripts/serve_local.sh <alias> --backend <b> --dry-run`.

Add short aliases in `LEGACY_ALIASES` in `scripts/model_registry.py`:

```python
"laguna": "laguna_xs_33b_moe",
"laguna-xs": "laguna_xs_33b_moe",
```

Then re-run `discover_models.py` and check your work:

```bash
python3 scripts/model_registry.py resolve laguna --backend llamacpp --format json
bash scripts/serve_local.sh laguna --backend llamacpp --dry-run
```

`--dry-run` prints the exact command without launching, which is the quickest way
to catch a wrong path or a bad engine choice.

## Engine routing

The interpreter that runs these servers is this repository's `.venv`, and it is
the only one on the machine that serves models — Mycelium and any other consumer
borrow it rather than resolving their own. `docs/SERVING.md` covers that
contract; adding a model does not change it.

`serve_local.sh` picks the engine from the file type and the registry:

| Weights | Engine | Port |
|---|---|---|
| `.gguf` | `llama-server` | 8080 |
| MLX directory | `mlx_lm.server` | 8085 |
| MLX directory pinned by the registry | `mlx_vlm.server` | 8085 |

Some architectures cannot load under `mlx_lm` at all — Gemma 4 E4B's elastic
weights, the `gemma4_unified` 12B, and `muse_glimmer` are the current examples.
Pin those with `mlx_server` on the **family in `configs/model_catalog.json`**:

```json
{ "id": "gemma4_e4b_dense", "architecture": "dense", "mlx_server": "mlx_vlm", ... }
```

`discover_models.py` copies that value into every generated entry, so the pin is
re-derived on each scan. Do **not** hand-write it into `models.local.json` — that
file is regenerated, and the pin silently disappears on the next `--write`,
leaving the model to fall back to `mlx_lm` and die with a weight-mismatch error.

The same applies to `mtp_supported`, `spec_type` and `spec_draft_n_max`: put
them on the catalog family, never in `models.local.json`.

### Family order matters

Discovery is **first-match-wins** — a weights file belongs to exactly one
family. Put a specific variant *before* the general family that would also match
it, or the general one claims it:

```
qwen3_35b_moe_mtp          <- must come first
qwen3_35b_moe_uncensored   <- must come first
qwen3_35b_moe              <- general
```

This matters because patterns are tested against the full path **and** the bare
filename. `Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` has no
"MTP" in its filename, so a `(?!.*MTP)` lookahead on the general family does not
exclude it. `evals/test_discovery.py` pins the required order.

Draft heads named `mtp-<base>.gguf` are skipped by discovery; a self-speculative
build whose *directory* is named `*-MTP-GGUF` is not.

`serve_local.sh` and `bench_head_to_head.py` also sniff `config.json` for
`gemma4_unified` and `muse_glimmer` and route automatically. Note that
`models.local.json` is *regenerated* by `discover_models.py`, so a hand-written
`mlx_server` pin does not survive a re-scan — the sniff, not the pin, is what
actually keeps these models servable.

### Client-side model IDs differ by server

This trips everyone up once:

```bash
# mlx_lm.server — the id is literally "default_model"
curl http://127.0.0.1:8085/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"default_model","messages":[{"role":"user","content":"hi"}]}'

# mlx_vlm.server — the id must be the full filesystem path
curl http://127.0.0.1:8085/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"/full/path/to/model-dir","messages":[{"role":"user","content":"hi"}]}'
```

The full path satisfies both, which is why the web GUI and the eval runner always
send it.

## Chat templates

The registry deliberately does **not** force `--chat-template`. GGUF files carry
their own, and overriding it corrupts the prompt in ways that look like the model
being broken rather than the harness being wrong.

Forcing `gemma2` on a Gemma 4 QAT GGUF produced `"9b 9b 9b…"` where the embedded
template produced `"Two plus two equals four."` Same class of failure as forcing
`chatml` on Qwen, which silently breaks tool calling. `_validate_chat_template()`
hard-refuses the known-bad combinations.

## Known architecture gotchas

| Model | Issue |
|---|---|
| Cohere North Mini Code 1.0 | GGUF uses `cohere2moe`; stock llama.cpp cannot load it until PR #24260 lands. Build from that branch |
| Gemma 4 (all sizes) | Google republished under the **same names** on 2026-07-15/16 with chat-template and tool-calling fixes. Weights pulled before that date are materially different — re-download |
| Gemma 4 E4B | Elastic weights; `mlx_lm` cannot load it. Needs `mlx_vlm` |
| Gemma 4 12B | `gemma4_unified`, encoder-free. Needs `mlx_vlm >= 0.6.1` |
| Muse Glimmer 30B | `muse_glimmer`, a dense **VLM** (not MoE). `mlx_lm` has no such arch at any version — needs `mlx_vlm >= 0.6.12`. Reasoning model: streams on `reasoning_content` with `content: null`, and `reasoning_strength` defaults to `high`, so budget 3-4x the tokens and well over the 600s default request timeout at ~17 t/s |

## Operating conventions

1. **One model root.** Everything under `~/.lmstudio/models` as `<org>/<repo>`.
   The HuggingFace hub cache is intentionally left empty.
2. **Serving never downloads.** MLX servers launch with `HF_HUB_OFFLINE=1`. A
   benchmark that pauses to fetch weights has already invalidated its own timing.
3. **Never commit `models.local.json`.** It is git-ignored, and `smoke_test.py`
   fails the build if a tracked file contains a local path.

Legacy registry `ctx_cap` is now a benchmark workload bound derived from declared context (32K fallback if unknown), not a serving default. Do not pass benchmark bounds as implicit daily-serving overrides. Quantization alone does not select a different recipe; record conversion-specific overrides only with evidence. Exact downloaded revision remains unknown unless recorded locally.
