#!/usr/bin/env bash
set -e

# ==========================================
# Local Model Server Launcher
#
# Registry-driven (configs/models.local.json via scripts/model_registry.py).
# Paired with scripts/llama_serve_menu.py (the interactive "python llama TUI").
#
# Backends:
#   llamacpp  llama-server, GGUF. Auto-enables MTP speculative decoding when the
#             resolved model has mtp_supported=true (see MTP block below).
#   mlx       mlx_lm.server     — MLX text; KV experiments retain the VLM route.
#   mlx-kv    mlx_vlm.server    — MLX + KV cache q8 turboquant.
#   mlx-vlm   mlx_vlm.server    — MLX vision tower.
#
# MTP (Multi-Token Prediction) has two on-disk shapes, both auto-detected:
#   self-speculative / embedded head  (Qwen3.6-35B-A3B-MTP): bare --spec-type draft-mtp
#   separate draft head               (Gemma 4 26B mtp-*.gguf): adds --spec-draft-model
# llama.cpp cannot combine MTP with --mmproj or -np > 1, so MTP mode forces
# -np 1 and suppresses mmproj.
# Gemma 4 MTP needs llama.cpp >= b9610 (Gemma4 MTP landed in PR #23398, 2026-06-07);
# older builds fail with "unknown model architecture: 'gemma4-assistant'".
# KV cache: default -ctk/-ctv q8_0 is fine WITH MTP (verified: Qwen ~0.80,
# Gemma ~0.58 draft acceptance). This is unrelated to the MLX mlx-kv backend.
# ==========================================

cd "$(dirname "$0")/.."

# The core .venv is the only interpreter that serves models (docs/SERVING.md §6).
# No PATH fallback: under the Mycelium app PATH's python3 is Homebrew's (MLX
# 0.32.0, none of the fixes), and silently serving from it is how two stacks
# carrying different halves of the same fixes came about.
PYTHON_BIN=".venv/bin/python3"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: the serving interpreter is missing: $(pwd)/$PYTHON_BIN" >&2
    echo "Create it: uv sync --extra test && .venv/bin/python scripts/apply_serving_patches.py" >&2
    exit 1
fi

# One source of truth for machine paths and default endpoints. LOCAL_AI_CONFIG
# lets tests and alternate machines select a different ignored TOML file.
LOCAL_CONFIG_ENV=$("$PYTHON_BIN" scripts/local_config.py --config "${LOCAL_AI_CONFIG:-configs/local.toml}" shell)
eval "$LOCAL_CONFIG_ENV"

PORT=""
HOST=""
BACKEND="llamacpp"
BACKEND_EXPLICIT=0
DRY_RUN=0
FORCE_NO_MTP=0
# Experiment overrides. All empty by default: with none of them passed, every
# command this script builds is byte-identical to what it built before they
# existed, so daily serving cannot drift because a benchmark needed a knob.
KV_BITS_OVERRIDE=""
KV_SCHEME_OVERRIDE=""
KV_TYPE_OVERRIDE=""
DRAFT_MODEL_OVERRIDE=""
DRAFT_KIND_OVERRIDE=""
CTX_OVERRIDE=""
THINKING_OVERRIDE=""
REASONING_EFFORT_OVERRIDE=""
FATAL_WORKER=0
ALLOW_UNPATCHED=0
EMBEDDING=0
MODEL_ARG="$1"

usage() {
    echo "Usage: $0 <model_alias_or_path> [port] [--backend <llamacpp|mlx|mlx-kv|mlx-vlm>] [--host <host>] [--dry-run] [--no-mtp] [--fatal-worker] [--allow-unpatched]"
    echo "              [--ctx <tokens>] [--kv-bits <n>] [--kv-scheme <uniform|turboquant>]"
    echo "              [--kv-type <f16|q8_0|q5_1|q4_0>] [--draft-model <path>] [--draft-kind <mtp|dflash|eagle3>]"
    echo ""
    echo "Backends:"
    echo " llamacpp — llama-server (GGUF files, OpenAI-compatible API)"
    echo " mlx     — mlx_lm.server (MLX text directories, OpenAI-compatible API, KV experiments routed to mlx_vlm)"
    echo " mlx-kv  — mlx_vlm.server + --kv-bits 8 --kv-quant-scheme turboquant (KV cache q8)"
    echo " mlx-vlm — mlx_vlm.server (MLX vision directories, OpenAI-compatible API)"
    echo ""
    echo "Auto-detection rules (when --backend not specified):"
    echo " *.gguf files → llamacpp"
    echo " MLX directories with config.json → mlx (text). Most local omni models"
    echo "   (Qwen3.5/3.6, Gemma 4) carry a vision_config but are served as text;"
    echo "   pass --backend mlx-kv for KV cache quantization, --backend mlx-vlm for vision."
    echo " Otherwise → llamacpp (default)"
    echo ""
    echo "Experiment overrides (all optional; omitting them preserves the default commands):"
    echo " --embedding     Qwen3 GGUF embedding mode: last pooling, native context, /v1/embeddings."
    echo " --ctx N         context size. llama.cpp -c; informational on MLX."
    echo " --kv-bits N     MLX KV cache bit depth. Implies mlx_vlm.server (launcher preserves mlx_vlm experiment routing)."
    echo "                 --kv-bits 0 serves under mlx_vlm with a NATIVE KV cache, which is the"
    echo "                 control arm any KV-quantization comparison needs: without it, server and"
    echo "                 KV mode change together and neither effect can be attributed."
    echo " --kv-scheme S   uniform|turboquant (mlx_vlm only). Default turboquant when --kv-bits > 0."
    echo " --kv-type T     llama.cpp -ctk/-ctv type. Default is q8_0, i.e. already quantized;"
    echo "                 pass f16 for a llama.cpp KV-native baseline."
    echo " --draft-model P speculative draft head. llama.cpp --spec-draft-model, MLX --draft-model."
    echo " --draft-kind K  mtp|dflash|eagle3 (mlx_vlm only; llama.cpp uses --spec-type)."
    echo " --fatal-worker  exit the process if mlx_lm's generation thread dies, instead of"
    echo "                 leaving a reachable server that can never complete a request."
    echo "                 mlx_lm guard only; skipped with a diagnostic on mlx_vlm."
    echo " --allow-unpatched  serve an mlx_lm runtime that is missing its local fixes"
    echo "                 (patches/mlx_lm/). Refused by default; for measuring upstream on purpose."
    echo " --thinking V    on|off. Overrides the registry's enable_thinking for this launch."
    echo " --reasoning-effort E  low|medium|... Overrides the registry's reasoning_effort."
    echo "                 Both must be flags, not environment variables: the registry"
    echo "                 resolution step assigns MODEL_ENABLE_THINKING and would clobber them."
    echo ""
    echo "Aliases come from configs/models.local.json."
    echo "Create it with:"
    echo "  python3 llm.py models sync"
    exit 1
}

if [[ -z "$MODEL_ARG" ]]; then
    usage
fi

shift
while [[ $# -gt 0 ]]; do
    case "$1" in
        --embedding)
            EMBEDDING=1
            shift
            ;;
        --backend)
            BACKEND="$2"
            BACKEND_EXPLICIT=1
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --no-mtp)
            FORCE_NO_MTP=1
            shift
            ;;
        --fatal-worker)
            FATAL_WORKER=1
            shift
            ;;
        --allow-unpatched)
            ALLOW_UNPATCHED=1
            shift
            ;;
        --ctx)
            CTX_OVERRIDE="$2"
            shift 2
            ;;
        --kv-bits)
            KV_BITS_OVERRIDE="$2"
            shift 2
            ;;
        --kv-scheme)
            KV_SCHEME_OVERRIDE="$2"
            shift 2
            ;;
        --kv-type)
            KV_TYPE_OVERRIDE="$2"
            shift 2
            ;;
        --draft-model)
            DRAFT_MODEL_OVERRIDE="$2"
            shift 2
            ;;
        --draft-kind)
            DRAFT_KIND_OVERRIDE="$2"
            shift 2
            ;;
        --thinking)
            THINKING_OVERRIDE="$2"
            shift 2
            ;;
        --reasoning-effort)
            REASONING_EFFORT_OVERRIDE="$2"
            shift 2
            ;;
        ''|*[!0-9]*)
            echo "ERROR: Unknown argument: $1"
            usage
            ;;
        *)
            PORT="$1"
            shift
            ;;
    esac
done

# Expand ~ and environment-variable path notation without executing selector
# text as shell code. Aliases remain unchanged and are resolved below.
MODEL_ARG_PATH=$("$PYTHON_BIN" -c 'import os, sys; print(os.path.expandvars(os.path.expanduser(sys.argv[1])))' "$MODEL_ARG")

# Auto-detect backend if not specified
if [[ "$BACKEND_EXPLICIT" == "0" && "$MODEL_ARG_PATH" == *.gguf ]]; then
    BACKEND="llamacpp"
elif [[ "$BACKEND_EXPLICIT" == "0" && -d "$MODEL_ARG_PATH" && -f "$MODEL_ARG_PATH/config.json" ]]; then
    BACKEND="mlx"
fi

if [[ "$EMBEDDING" == "1" && "$BACKEND" != "llamacpp" ]]; then
    echo "ERROR: --embedding requires a GGUF model with --backend llamacpp." >&2
    exit 1
fi

# --kv-bits (including 0) selects the KV-capable server. Resolving this before
# host/port selection keeps an MLX arm on the MLX endpoint either way.
if [[ -n "$KV_BITS_OVERRIDE" && "$BACKEND" == "mlx" ]]; then
    BACKEND="mlx-kv"
    BACKEND_EXPLICIT=1
fi

if [[ -z "$HOST" ]]; then
    if [[ "$BACKEND" == "mlx" || "$BACKEND" == "mlx-vlm" || "$BACKEND" == "mlx-kv" ]]; then
        HOST="$LOCAL_MLX_HOST"
    else
        HOST="$LOCAL_LLAMACPP_HOST"
    fi
fi

if [[ -z "$PORT" ]]; then
    if [[ "$BACKEND" == "mlx" || "$BACKEND" == "mlx-vlm" || "$BACKEND" == "mlx-kv" ]]; then
        PORT="$LOCAL_MLX_PORT"
    else
        PORT="$LOCAL_LLAMACPP_PORT"
    fi
fi

# The registry catalogs all MLX models under backend "mlx"; ask it that way.
RESOLVE_BACKEND="$BACKEND"
[[ "$RESOLVE_BACKEND" == "mlx-vlm" || "$RESOLVE_BACKEND" == "mlx-kv" ]] && RESOLVE_BACKEND="mlx"

if [[ -f "scripts/model_registry.py" ]]; then
    if RESOLVED_ENV=$("$PYTHON_BIN" scripts/model_registry.py --config "$LOCAL_MODEL_REGISTRY" resolve "$MODEL_ARG" --backend "$RESOLVE_BACKEND" --format shell 2>/tmp/serve_local_resolve.err); then
        eval "$RESOLVED_ENV"
        MODEL_ARG="$MODEL_PATH"
    elif [[ "$MODEL_ARG_PATH" != *.gguf && ! -e "$MODEL_ARG_PATH" ]]; then
        cat /tmp/serve_local_resolve.err
        exit 1
    fi
fi

# Registry-provided sampling defaults (per-family, e.g. Gemma 4's
# temp=1.0/top_p=0.95/top_k=64 model-card config). Only set if resolution
# populated them; per-request overrides from clients still take precedence.
# Apply the same storage policy to explicit paths as to discovered aliases.
MODEL_ARG=$("$PYTHON_BIN" -c '
import os, sys
from pathlib import Path
sys.path.insert(0, "scripts")
from model_files import validate_model_path
try:
    print(validate_model_path(Path(os.path.expandvars(sys.argv[1])), Path(sys.argv[2]), sys.argv[3]))
except ValueError as exc:
    raise SystemExit(str(exc))
' "$MODEL_ARG" "$LOCAL_MODEL_ROOT" "$RESOLVE_BACKEND")

# Resolve exact-path policy too: explicit paths and newly discovered models use
# the same installed metadata and architecture routing as family selectors.
POLICY_ENV=$("$PYTHON_BIN" - "$MODEL_ARG" "$RESOLVE_BACKEND" "$THINKING_OVERRIDE" <<'PY_POLICY'
import sys, shlex
sys.path.insert(0, "scripts")
from model_policy import resolve_policy
policy = resolve_policy(sys.argv[1], sys.argv[2])
sampling = policy["sampling"]
if sys.argv[3].lower() in {"1", "true", "yes", "on"}:
    sampling.update(policy["sampling_recipes"].get("thinking", {}).get("sampling", {}))
for key, value in sampling.items():
    print("MODEL_" + key.upper() + "=" + shlex.quote(str(value)))
print("MODEL_MLX_SERVER=" + shlex.quote(policy["mlx_server"]))
PY_POLICY
)
eval "$POLICY_ENV"

MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_ARG")}"

SAMPLING_ARGS=(--min-p "${MODEL_MIN_P:-0.0}")
if [[ -n "${MODEL_TEMPERATURE:-}" ]]; then
    SAMPLING_ARGS+=(--temp "$MODEL_TEMPERATURE" --top-p "${MODEL_TOP_P:-1.0}" --top-k "${MODEL_TOP_K:-0}")
    echo " Sampling: temp=$MODEL_TEMPERATURE top_p=${MODEL_TOP_P:-1.0} top_k=${MODEL_TOP_K:-0}"
fi

REPETITION_PENALTY="${MLX_REPETITION_PENALTY:-${MODEL_REPETITION_PENALTY:-${LOCAL_DEFAULT_REPETITION_PENALTY:-1.0}}}"
PRESENCE_PENALTY="${MLX_PRESENCE_PENALTY:-${MODEL_PRESENCE_PENALTY:-${LOCAL_DEFAULT_PRESENCE_PENALTY:-0.0}}}"
REPETITION_CONTEXT_SIZE="${MLX_REPETITION_CONTEXT_SIZE:-${MODEL_REPETITION_CONTEXT_SIZE:-${LOCAL_DEFAULT_REPETITION_CONTEXT_SIZE:-2048}}}"
ENABLE_THINKING="${MODEL_ENABLE_THINKING:-}"
REASONING_EFFORT="${MODEL_REASONING_EFFORT:-}"
REASONING_BUDGET="${MODEL_REASONING_BUDGET:-}"

# Explicit flags win over the registry. They have to be applied HERE, after the
# `eval "$RESOLVED_ENV"` above, because that eval assigns MODEL_ENABLE_THINKING
# from the catalog and so overwrites any value the caller exported. An
# experiment that varies reasoning mode by exporting the environment variable
# would therefore have served every arm with the catalog's setting and measured
# nothing — the arms would have been identical while claiming to differ.
if [[ -n "$THINKING_OVERRIDE" ]]; then
    ENABLE_THINKING="$THINKING_OVERRIDE"
fi
if [[ -n "$REASONING_EFFORT_OVERRIDE" ]]; then
    REASONING_EFFORT="$REASONING_EFFORT_OVERRIDE"
fi

if [[ -n "$REPETITION_PENALTY" && "$REPETITION_PENALTY" != "0" && "$REPETITION_PENALTY" != "0.0" ]]; then
    echo " Penalties: rep_pen=$REPETITION_PENALTY pres_pen=$PRESENCE_PENALTY context_size=$REPETITION_CONTEXT_SIZE"
fi
if [[ -n "$REASONING_EFFORT" ]]; then
    echo " Reasoning: default=${ENABLE_THINKING:-native} effort=$REASONING_EFFORT${REASONING_BUDGET:+ budget=$REASONING_BUDGET}"
fi

echo "=========================================="
echo " Local Inference Server"
echo " Backend: $BACKEND"
echo " Host: $HOST"
echo " Port: $PORT"
echo " Model: $MODEL_ARG"
[[ -n "${MODEL_USE_CASE:-}" ]] && echo " Use case: $MODEL_USE_CASE"
echo "=========================================="

case "$BACKEND" in
    llamacpp)
        MODEL_PATH=$("$PYTHON_BIN" -c 'import os, sys; print(os.path.expandvars(os.path.expanduser(sys.argv[1])))' "$MODEL_ARG")
        if [[ ! -e "$MODEL_PATH" ]]; then
            echo "ERROR: Model path does not exist: $MODEL_PATH"
            exit 1
        fi

        # llama.cpp -c 0 requests the model's native training context.
        CTX_SIZE="0"
        [[ -n "$CTX_OVERRIDE" ]] && CTX_SIZE="$CTX_OVERRIDE"

        echo " Context: $CTX_SIZE"
        echo ""

        LLAMA_SERVER_BIN="${MODEL_SERVER_BINARY:-}"
        if [[ -z "$LLAMA_SERVER_BIN" ]]; then
            LLAMA_SERVER_BIN="$(command -v llama-server || true)"
        fi

        if [[ -z "$LLAMA_SERVER_BIN" || ! -x "$LLAMA_SERVER_BIN" ]]; then
            echo "ERROR: llama-server not found. Install via: brew install llama.cpp"
            exit 1
        fi

        # Do NOT blanket-force a chat template. GGUFs that carry their own
        # embedded template (the gemma-4-26B QAT GGUF uses a <|turn>/<|channel>
        # thinking format) are corrupted by an override — forcing gemma2 made the
        # model emit garbage ("9b 9b 9b…"). Only honour an explicit registry
        # template (MODEL_CHAT_TEMPLATE) below; otherwise let llama.cpp use the
        # GGUF's embedded template.
        EXTRA_ARGS=()
        if [[ -n "${MODEL_CHAT_TEMPLATE:-}" ]]; then
            TEMPLATE_LC="$(printf '%s' "$MODEL_CHAT_TEMPLATE" | tr '[:upper:]' '[:lower:]')"
            if [[ "$TEMPLATE_LC" == "chatml" ]]; then
                echo "ERROR: Refusing to force --chat-template chatml. Qwen/Granite GGUF models carry embedded templates." >&2
                exit 1
            fi
            EXTRA_ARGS=("--chat-template" "$MODEL_CHAT_TEMPLATE")
        fi

        # --- MTP (Multi-Token Prediction) speculative decoding ---
        # Gated on registry mtp_supported=true. Handles both on-disk shapes:
        #   * separate draft head  (Gemma: mtp-*.gguf beside the GGUF) → --spec-draft-model
        #   * self-speculative      (Qwen3.6 MTP: head embedded in the GGUF) → bare --spec-type
        # MTP cannot coexist with --mmproj or -np > 1 in llama.cpp, so when active we
        # pin -np 1 here and suppress mmproj in the block below (MTP_ACTIVE flag).
        MTP_ARGS=()
        MTP_ACTIVE=0
        MODEL_MTP_LC="$(printf '%s' "${MODEL_MTP_SUPPORTED:-}" | tr '[:upper:]' '[:lower:]')"
        if [[ "$MODEL_MTP_LC" == "true" && "$FORCE_NO_MTP" == "1" ]]; then
            echo " MTP: forced OFF (--no-mtp) — serving as plain GGUF baseline."
        fi
        if [[ "$MODEL_MTP_LC" == "true" && "$FORCE_NO_MTP" != "1" ]]; then
            if ! "$LLAMA_SERVER_BIN" --help 2>&1 | grep -Eq -- '--spec-type.*draft-mtp|--spec-type.*\bmtp\b'; then
                echo "ERROR: $MODEL_NAME requires a llama-server build with --spec-type draft-mtp support." >&2
                echo "Configured binary does not expose MTP: $LLAMA_SERVER_BIN" >&2
                echo "Fix: brew upgrade llama.cpp  (Gemma 4 MTP needs >= b9610 / PR #23398)." >&2
                exit 1
            fi
            # NOTE: the --help probe passes on builds that expose the flag but cannot
            # load the gemma4-assistant draft arch (e.g. b9410), which then crash at
            # model load with "unknown model architecture". If that happens, upgrade:
            #   brew upgrade llama.cpp
            MTP_ACTIVE=1
            # Optional separate draft head: explicit registry value, else auto-detect
            # mtp-*.gguf beside the main GGUF. Absent for self-speculative models.
            DRAFT_MODEL_PATH="${MODEL_DRAFT_MODEL:-}"
            if [[ -z "$DRAFT_MODEL_PATH" ]]; then
                _MODEL_DIR="$(dirname "$MODEL_PATH")"
                shopt -s nullglob
                _DRAFT_CANDIDATES=("$_MODEL_DIR"/mtp-*.gguf "$_MODEL_DIR"/MTP/*.gguf)
                shopt -u nullglob
                [[ ${#_DRAFT_CANDIDATES[@]} -gt 0 ]] && DRAFT_MODEL_PATH="${_DRAFT_CANDIDATES[0]}"
            fi
            MTP_ARGS=(--spec-type "${MODEL_SPEC_TYPE:-draft-mtp}" --spec-draft-n-max "${MODEL_SPEC_DRAFT_N_MAX:-2}" -np 1)
            if [[ -n "$DRAFT_MODEL_PATH" ]]; then
                echo " MTP: ON — separate draft head $(basename "$DRAFT_MODEL_PATH") (n-max=${MODEL_SPEC_DRAFT_N_MAX:-2})"
                MTP_ARGS+=(--spec-draft-model "$DRAFT_MODEL_PATH")
            else
                echo " MTP: ON — self-speculative / embedded head (n-max=${MODEL_SPEC_DRAFT_N_MAX:-2})"
            fi
        fi

        if [[ -n "$DRAFT_MODEL_OVERRIDE" && "$MTP_ACTIVE" == "0" && "$FORCE_NO_MTP" != "1" ]]; then
            DRAFT_ABS=$("$PYTHON_BIN" -c 'import os, sys; print(os.path.expandvars(os.path.expanduser(sys.argv[1])))' "$DRAFT_MODEL_OVERRIDE")
            if [[ ! -f "$DRAFT_ABS" ]]; then
                echo "ERROR: --draft-model not found: $DRAFT_ABS" >&2
                exit 1
            fi
            if ! "$LLAMA_SERVER_BIN" --help 2>&1 | grep -Eq -- '--spec-type'; then
                echo "ERROR: this llama-server build has no --spec-type; cannot use --draft-model." >&2
                exit 1
            fi
            MTP_ACTIVE=1
            MTP_ARGS=(--spec-type "${MODEL_SPEC_TYPE:-draft-mtp}" --spec-draft-n-max "${MODEL_SPEC_DRAFT_N_MAX:-2}" -np 1 --spec-draft-model "$DRAFT_ABS")
            echo " MTP: ON (explicit) — draft head $(basename "$DRAFT_ABS") (n-max=${MODEL_SPEC_DRAFT_N_MAX:-2})"
        fi

        MMPROJ_ARGS=()
        if [[ "$MTP_ACTIVE" == "1" ]]; then
            # llama.cpp does not support --mmproj together with MTP.
            if [[ -n "${MODEL_MMPROJ_PATH:-}" ]]; then
                echo " Note: mmproj suppressed — llama.cpp cannot combine --mmproj with MTP."
            fi
        else
            MMPROJ_PATH="${MODEL_MMPROJ_PATH:-}"
            if [[ -z "$MMPROJ_PATH" ]]; then
                MODEL_DIR="$(dirname "$MODEL_PATH")"
                shopt -s nullglob
                MMPROJ_CANDIDATES=("$MODEL_DIR"/*mmproj*.gguf "$MODEL_DIR"/mmproj*.gguf)
                shopt -u nullglob
                if [[ ${#MMPROJ_CANDIDATES[@]} -gt 0 ]]; then
                    MMPROJ_PATH="${MMPROJ_CANDIDATES[0]}"
                fi
            fi
            if [[ -n "$MMPROJ_PATH" ]]; then
                if [[ ! -e "$MMPROJ_PATH" ]]; then
                    echo "ERROR: mmproj path does not exist: $MMPROJ_PATH"
                    exit 1
                fi
                echo " Multimodal projector: $MMPROJ_PATH"
                MMPROJ_ARGS=(--mmproj "$MMPROJ_PATH")
            fi
        fi

        LLAMA_SAMPLING_ARGS=("${SAMPLING_ARGS[@]}")
        if [[ -n "$REPETITION_PENALTY" && "$REPETITION_PENALTY" != "0" && "$REPETITION_PENALTY" != "0.0" ]]; then
            LLAMA_SAMPLING_ARGS+=(--repeat-penalty "$REPETITION_PENALTY" --presence-penalty "$PRESENCE_PENALTY" --repeat-last-n "$REPETITION_CONTEXT_SIZE")
        fi

        LLAMA_REASONING_ARGS=()
        if [[ -n "$ENABLE_THINKING" ]]; then
            ENABLE_THINKING_LC=$(printf '%s' "$ENABLE_THINKING" | tr '[:upper:]' '[:lower:]')
            if [[ "$ENABLE_THINKING_LC" =~ ^(1|true|yes|on)$ ]]; then
                LLAMA_REASONING_ARGS+=(--reasoning on)
            else
                LLAMA_REASONING_ARGS+=(--reasoning off)
            fi
        fi
        if [[ -n "$REASONING_EFFORT" ]]; then
            LLAMA_REASONING_ARGS+=(--reasoning-effort "$REASONING_EFFORT")
        fi
        if [[ -n "$REASONING_BUDGET" ]]; then
            LLAMA_REASONING_ARGS+=(--reasoning-budget "$REASONING_BUDGET" --reasoning-budget-message "Reasoning budget reached; provide the best final answer now.")
        fi

        # -ctk/-ctv default to q8_0, which is itself a KV quantization. A
        # weight-quantization comparison that leaves it there is comparing
        # weight bits with the KV cache held at q8 — fine, as long as it is
        # stated. --kv-type f16 gives the KV-native baseline.
        EMBEDDING_ARGS=()
        if [[ "$EMBEDDING" == "1" ]]; then
            EMBEDDING_ARGS=(--embedding --pooling last)
            MMPROJ_ARGS=()
            MTP_ARGS=()
            LLAMA_REASONING_ARGS=()
            LLAMA_SAMPLING_ARGS=()
        fi
        KV_TYPE="${KV_TYPE_OVERRIDE:-q8_0}"
        echo " KV cache: -ctk $KV_TYPE -ctv $KV_TYPE"
        CMD=("$LLAMA_SERVER_BIN" -m "$MODEL_PATH" --alias "$MODEL_NAME" -np 1 -c "$CTX_SIZE" --port "$PORT" --host "$HOST" -ngl all -ctk "$KV_TYPE" -ctv "$KV_TYPE" -fa on --load-mode mlock "${LLAMA_SAMPLING_ARGS[@]}" "${LLAMA_REASONING_ARGS[@]}" "${EXTRA_ARGS[@]}" "${MTP_ARGS[@]}" "${MMPROJ_ARGS[@]}" "${EMBEDDING_ARGS[@]}")
        if [[ "$DRY_RUN" == "1" ]]; then
            printf 'DRY RUN:'
            printf ' %q' "${CMD[@]}"
            printf '\n'
            exit 0
        fi

        # Replace the wrapper shell so callers can signal the actual server.
        exec "${CMD[@]}"
        ;;

    mlx|mlx-vlm|mlx-kv)
        MODEL_PATH=$("$PYTHON_BIN" -c 'import os, sys; print(os.path.expandvars(os.path.expanduser(sys.argv[1])))' "$MODEL_ARG")
        if [[ ! -d "$MODEL_PATH" ]]; then
            echo "ERROR: MLX model directory not found on disk: $MODEL_PATH" >&2
            echo "Refusing to launch: a non-local --model would make the MLX server download from Hugging Face." >&2
            exit 1
        fi

        # Pick the runner. mlx_lm serves text MLX dirs; mlx_vlm serves vision/omni
        # towers. mlx_lm CANNOT load the gemma4_unified omni arch, so route those to
        # mlx_vlm even when the caller asked for plain --backend mlx. (Plain
        # "has vision_config" is intentionally NOT enough — most omni models serve
        # fine and faster as text under mlx_lm.)
        MLX_SERVER="mlx_lm"
        [[ "$BACKEND" == "mlx-vlm" || "$BACKEND" == "mlx-kv" ]] && MLX_SERVER="mlx_vlm"
        # Registry can pin a model to a specific MLX server (models.local.json
        # "mlx_server") for archs/weights mlx_lm cannot load: gemma4_unified (12B)
        # and the gemma4 E4B elastic checkpoint. This is the reliable signal.
        if [[ "${MODEL_MLX_SERVER:-}" == "mlx_vlm" && "$MLX_SERVER" == "mlx_lm" ]]; then
            echo " Note: registry pins this model to mlx_vlm.server (mlx_lm cannot load it)."
            MLX_SERVER="mlx_vlm"
        fi
        # The fixes in patches/mlx_lm/ are what make mlx_lm.server safe for
        # multi-turn tool use and oversized prompts, and any reinstall reverts
        # them without a word. Refuse to serve their known failures rather than
        # leave it to whoever next runs `llm doctor`. mlx_vlm is unpatched upstream.
        if [[ "$MLX_SERVER" == "mlx_lm" ]]; then
            if ! PATCH_REPORT=$("$PYTHON_BIN" scripts/serving_runtime.py --require-patched 2>&1); then
                if [[ "$ALLOW_UNPATCHED" == "1" ]]; then
                    echo "WARNING: serving mlx_lm without its local fixes (--allow-unpatched):" >&2
                    echo "$PATCH_REPORT" >&2
                else
                    echo "ERROR: refusing to serve: this mlx_lm runtime is missing local fixes." >&2
                    echo "$PATCH_REPORT" >&2
                    echo "Reapply: .venv/bin/python scripts/apply_serving_patches.py  (or pass --allow-unpatched)" >&2
                    exit 1
                fi
            fi
        fi

        # Serve strictly from disk. HF_HUB_OFFLINE stops the server from silently
        # downloading a model when a client request names a repo-id that is not the
        # loaded local path (mlx_lm/mlx_vlm reload per-request "model").
        MLX_ENV=(env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MLX_REPETITION_PENALTY="$REPETITION_PENALTY" MLX_PRESENCE_PENALTY="$PRESENCE_PENALTY" MLX_REPETITION_CONTEXT_SIZE="$REPETITION_CONTEXT_SIZE" MLX_PRESENCE_CONTEXT_SIZE="$REPETITION_CONTEXT_SIZE")

        # mlx_vlm.server has no --temp/--top-p/--top-k flags (sampling is
        # per-request only); only mlx_lm.server accepts registry sampling defaults.
        MLX_SAMPLING_ARGS=()
        if [[ "$MLX_SERVER" == "mlx_lm" ]]; then
            MLX_SAMPLING_ARGS=("${SAMPLING_ARGS[@]}")
            if [[ -n "$REPETITION_PENALTY" && "$REPETITION_PENALTY" != "0" && "$REPETITION_PENALTY" != "0.0" ]]; then
                MLX_SAMPLING_ARGS+=(--repetition-penalty "$REPETITION_PENALTY" --presence-penalty "$PRESENCE_PENALTY" --repetition-context-size "$REPETITION_CONTEXT_SIZE" --presence-context-size "$REPETITION_CONTEXT_SIZE")
            fi
            if [[ -n "$REASONING_EFFORT" ]]; then
                REASONING_JSON=$("$PYTHON_BIN" -c '
import json, sys
enabled = sys.argv[1].lower() in {"1", "true", "yes", "on"}
print(json.dumps({"enable_thinking": enabled, "reasoning_effort": sys.argv[2]}))
' "$ENABLE_THINKING" "$REASONING_EFFORT")
                MLX_SAMPLING_ARGS+=(--chat-template-args "$REASONING_JSON")
            fi
        elif [[ ${#SAMPLING_ARGS[@]} -gt 0 ]]; then
            echo " Sampling: core request middleware supplies omitted model policy defaults for mlx_vlm."
        fi

        # KV cache quantization — preserve historical mlx_vlm experiment routing.
        KV_ARGS=()
        if [[ "$BACKEND" == "mlx-kv" ]]; then
            KV_BITS="${KV_BITS_OVERRIDE:-8}"
            if [[ "$KV_BITS" == "0" ]]; then
                # The control arm: mlx_vlm.server with a native KV cache. Pairing
                # a quantized-KV run against an mlx_lm run would change the server
                # and the KV mode at once.
                echo " KV cache: native (mlx_vlm.server, no quantization)"
            else
                KV_ARGS=(--kv-bits "$KV_BITS" --kv-quant-scheme "${KV_SCHEME_OVERRIDE:-turboquant}")
                echo " KV cache: ${KV_BITS}-bit ${KV_SCHEME_OVERRIDE:-turboquant} (mlx_vlm.server)"
            fi
        fi
        if [[ -n "$DRAFT_MODEL_OVERRIDE" ]]; then
            DRAFT_ABS=$("$PYTHON_BIN" -c 'import os, sys; print(os.path.expandvars(os.path.expanduser(sys.argv[1])))' "$DRAFT_MODEL_OVERRIDE")
            if [[ ! -d "$DRAFT_ABS" ]]; then
                echo "ERROR: --draft-model directory not found: $DRAFT_ABS" >&2
                exit 1
            fi
            if [[ "$MLX_SERVER" != "mlx_vlm" ]]; then
                echo "ERROR: speculative drafting needs mlx_vlm.server; mlx_lm has no qwen3_5_mtp." >&2
                echo "Re-run with --backend mlx-vlm, or with --kv-bits to select it." >&2
                exit 1
            fi
            KV_ARGS+=(--draft-model "$DRAFT_ABS")
            [[ -n "$DRAFT_KIND_OVERRIDE" ]] && KV_ARGS+=(--draft-kind "$DRAFT_KIND_OVERRIDE")
            echo " Speculative drafting: $(basename "$DRAFT_ABS")${DRAFT_KIND_OVERRIDE:+ (kind=$DRAFT_KIND_OVERRIDE)}"
        fi
        if [[ "$MLX_SERVER" == "mlx_vlm" && -n "$REASONING_BUDGET" ]]; then
            KV_ARGS+=(--thinking-budget "$REASONING_BUDGET")
        fi
        ENABLE_THINKING_LC=$(printf '%s' "$ENABLE_THINKING" | tr '[:upper:]' '[:lower:]')
        if [[ "$MLX_SERVER" == "mlx_vlm" && "$ENABLE_THINKING_LC" =~ ^(1|true|yes|on)$ ]]; then
            KV_ARGS+=(--enable-thinking)
        fi

        [[ -n "$CTX_OVERRIDE" ]] && echo " Context: $CTX_OVERRIDE (MLX servers size the cache dynamically; recorded, not enforced)"
        MLX_MAX_TOKENS="${MLX_MAX_TOKENS:-32768}"
        MLX_MAX_TOKENS_ARG=(--max-tokens "$MLX_MAX_TOKENS")
        echo " Max tokens: $MLX_MAX_TOKENS"
        echo ""
        # --fatal-worker routes through a core wrapper that exits when mlx_lm's
        # generation thread dies, instead of leaving a reachable server that can
        # never complete a request. It replaces the console script, so it is
        # checked before the normal resolution order below.
        if [[ "$FATAL_WORKER" == "1" && "$MLX_SERVER" != "mlx_lm" ]]; then
            echo " Note: --fatal-worker skipped for ${MLX_SERVER}.server; the mlx_lm thread guard does not apply."
        fi
        if [[ "$FATAL_WORKER" == "1" && "$MLX_SERVER" == "mlx_lm" ]]; then
            GUARD="$(cd "$(dirname "$0")" && pwd)/mlx_server_guard.py"
            if [[ ! -f "$GUARD" ]]; then
                echo "ERROR: --fatal-worker needs $GUARD" >&2
                exit 1
            fi
            echo " Supervision: generation-worker failures are fatal"
            CMD=("${MLX_ENV[@]}" "$PYTHON_BIN" "$GUARD" --model "$MODEL_PATH" --host "$HOST" --port "$PORT" "${MLX_MAX_TOKENS_ARG[@]}" "${MLX_SAMPLING_ARGS[@]}" "${KV_ARGS[@]}")
            if [[ "$DRY_RUN" == "1" ]]; then
                printf 'DRY RUN:'
                printf ' %q' "${CMD[@]}"
                printf '\n'
                exit 0
            fi
            exec "${CMD[@]}"
        fi

        # The server comes from the core venv or not at all: a PATH lookup here
        # is how Homebrew's unpatched MLX 0.32.0 server used to get launched.
        VENV_SERVER="$(dirname "$PYTHON_BIN")/${MLX_SERVER}.server"
        if [[ ! -x "$VENV_SERVER" ]]; then
            echo "ERROR: ${MLX_SERVER}.server is not installed in the serving venv: $(pwd)/$VENV_SERVER" >&2
            echo "Fix: uv sync --extra test" >&2
            exit 1
        fi
        SERVER_COMMAND=("$VENV_SERVER")
        if [[ "$MLX_SERVER" == "mlx_vlm" ]]; then
            SERVER_COMMAND=("$PYTHON_BIN" scripts/mlx_vlm_server.py)
        fi
        CMD=("${MLX_ENV[@]}" "${SERVER_COMMAND[@]}" --model "$MODEL_PATH" --host "$HOST" --port "$PORT" "${MLX_MAX_TOKENS_ARG[@]}" "${MLX_SAMPLING_ARGS[@]}" "${KV_ARGS[@]}")
        if [[ "$DRY_RUN" == "1" ]]; then
            printf 'DRY RUN:'
            printf ' %q' "${CMD[@]}"
            printf '\n'
            exit 0
        fi
        exec "${CMD[@]}"
        ;;

    *)
        echo "ERROR: Unknown backend '$BACKEND'. Use: llamacpp, mlx, mlx-kv, or mlx-vlm"
        exit 1
        ;;
esac
