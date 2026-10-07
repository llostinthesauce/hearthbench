#!/usr/bin/env python3
"""
Resolve local model aliases from configs/models.local.json.

This keeps serving scripts and benchmark scripts pointed at the same registry.
It intentionally uses only the Python standard library so it can run before a
project virtualenv is activated.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any

from model_policy import resolve_policy
from model_files import is_complete_model


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "models.local.json"
DEFAULT_CATALOG = ROOT / "configs" / "model_catalog.json"

LEGACY_ALIASES = {
    "qwen35": "qwen3_35b_moe",
    "qwen-moe-35b": "qwen3_35b_moe",
    "qwen35-mtp": "qwen3_35b_moe_mtp",
    "qwen35mtp": "qwen3_35b_moe_mtp",
    "qwen-moe-35b-mtp": "qwen3_35b_moe_mtp",
    "qwen27": "qwen3_27b_dense",
    "qwen-dense-27b": "qwen3_27b_dense",
    "gemma26": "gemma4_26b_moe",
    "gemma-26b": "gemma4_26b_moe",
    "gemma31": "gemma4_31b_dense",
    "gemma-31b": "gemma4_31b_dense",
    "gemmae4b": "gemma4_e4b_dense",
    "gemma-e4b": "gemma4_e4b_dense",
    "gemma4e4b": "gemma4_e4b_dense",
    "gemma4-e4b": "gemma4_e4b_dense",
    "gemma12": "gemma4_12b_dense",
    "gemma-12b": "gemma4_12b_dense",
    "gemma4-12b": "gemma4_12b_dense",
    "gemma4_12b": "gemma4_12b_dense",
    "gemma12-8bit": "gemma4_12b_8bit",
    "gemma12-8b": "gemma4_12b_8bit",
    "gemma4-12b-8bit": "gemma4_12b_8bit",
    "qwen35-uncensored": "qwen3_35b_moe_uncensored",
    "qwen35uncensored": "qwen3_35b_moe_uncensored",
    "laguna": "laguna_xs_33b_moe",
    "laguna-xs": "laguna_xs_33b_moe",
    "lagunaxs": "laguna_xs_33b_moe",
    "north": "north_mini_code_30b_moe",
    "north-mini": "north_mini_code_30b_moe",
    "northmini": "north_mini_code_30b_moe",
    "bonsai": "ternary_bonsai_27b",
    "glimmer": "muse_glimmer_30b_dense",
    "muse-glimmer": "muse_glimmer_30b_dense",
    "museglimmer": "muse_glimmer_30b_dense",
    "glimmer-30b": "muse_glimmer_30b_dense",
    "hemmingway": "hemmingway_1_27b",
    "hemmingway-1": "hemmingway_1_27b",
}


def _load_config(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        raise SystemExit(
            f"Local model registry not found: {path}\n"
            f"Create it with:\n"
            "  python3 llm.py models sync"
        )
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}")
    if data.get("_schema_version") != 2:
        raise SystemExit(f"Unsupported config schema in {path}: expected _schema_version=2")
    return data


FORBIDDEN_TEMPLATES = frozenset({"chatml"})


def _chat_template(family: str) -> str:
    # Empty = do NOT force a template; let llama.cpp use the GGUF's embedded one.
    # The gemma-4-26B QAT GGUF carries its own <|turn>/<|channel> thinking-format
    # template; forcing --chat-template gemma2 mangles the prompt and the model
    # collapses into garbage (verified: forced gemma2 → "9b 9b 9b…"; embedded →
    # "Two plus two equals four."). Same hazard as forcing chatml on Qwen.
    return ""


def _validate_chat_template(family: str, template: str) -> None:
    if template and template.lower() in FORBIDDEN_TEMPLATES:
        raise SystemExit(
            f"Refusing to serve {family} model with --chat-template {template}. "
            f"Qwen/Granite GGUF models carry embedded templates; "
            f"forcing {template} corrupts tool calling."
        )


def _expand(path_or_repo: str) -> str:
    if path_or_repo.startswith("tools/") or path_or_repo.startswith("scripts/"):
        return str(ROOT / path_or_repo)
    return os.path.expanduser(path_or_repo)


def _family_aliases(family: dict[str, Any]) -> set[str]:
    family_id = family.get("id", "")
    aliases = {family_id}
    for alias, target in LEGACY_ALIASES.items():
        if target == family_id:
            aliases.add(alias)
    return aliases


def iter_models(config: Path) -> list[dict[str, Any]]:
    data = _load_config(config)
    rows: list[dict[str, Any]] = []
    for family in data.get("model_families", []):
        family_id = family.get("id", "")
        family_name = family.get("family", "")
        aliases = sorted(_family_aliases(family))
        chat_tpl = _chat_template(family_name)
        base = {
            "family_id": family_id,
            "family": family_name,
            "use_case": family.get("use_case", ""),
            "architecture": family.get("architecture", "dense"),
            "ctx_cap": 0,
            "temperature": family.get("temperature", 0.7),
            "top_p": family.get("top_p", 0.8),
            "top_k": family.get("top_k", 20),
            "min_p": family.get("min_p", 0.0),
            "repetition_penalty": family.get("repetition_penalty", 1.05),
            "presence_penalty": family.get("presence_penalty", 0.2),
            "repetition_context_size": family.get("repetition_context_size", 2048),
            "enable_thinking": family.get("enable_thinking", ""),
            "reasoning_effort": family.get("reasoning_effort", ""),
            "reasoning_budget": family.get("reasoning_budget", ""),
            "aliases": aliases,
            "chat_template": chat_tpl,
            # {backend: variant filename} — which variant a family-level ask
            # resolves to. See `resolve`.
            "preferred": family.get("preferred", {}),
        }
        for entry in family.get("gguf", []):
            path = _expand(entry.get("path", ""))
            if not path:
                continue
            rows.append({
                **base,
                "backend": "llamacpp",
                "selector": path,
                "path": path,
                "name": Path(path).name,
                "quant": entry.get("quant", "?"),
                "exists": is_complete_model(Path(path), "llamacpp"),
                "note": entry.get("_note", ""),
                # Fall back to the committed catalog for the same reason as
                # mlx_server: models.local.json is regenerated on every scan.
                "mtp_supported": entry.get("mtp_supported", family.get("mtp_supported", False)),
                "draft_model": _expand(entry.get("draft_model", "")),
                "server_binary": _expand(entry.get("server_binary", "")),
                "mmproj_path": _expand(entry.get("mmproj_path", "")),
                "spec_type": entry.get("spec_type") or family.get("spec_type", ""),
                "spec_draft_n_max": entry.get("spec_draft_n_max") or family.get("spec_draft_n_max", ""),
            })
        for entry in family.get("mlx", []):
            repo = entry.get("repo", "")
            if not repo:
                continue
            repo_path = _expand(repo)
            rows.append({
                **base,
                "backend": "mlx",
                "selector": repo,
                "path": repo_path,
                "name": repo_path.rstrip("/").split("/")[-1],
                "quant": entry.get("quant", "?"),
                "exists": is_complete_model(Path(repo_path), "mlx"),
                "note": entry.get("_note", ""),
                "mtp_supported": entry.get("mtp_supported", False),
                "draft_model": entry.get("draft_model", ""),
                # models.local.json is regenerated by discover_models.py, so a pin
                # written there is lost on the next re-scan. Fall back to the
                # committed catalog, which is what actually persists.
                "mlx_server": entry.get("mlx_server") or family.get("mlx_server", ""),
            })
    for row in rows:
        policy = resolve_policy(row["path"], row["backend"])
        row.update(policy["sampling"])
        # Legacy benchmark field remains a positive workload budget. Serving
        # ignores it and uses native allocation; zero would break max passes.
        row["ctx_cap"] = policy["declared_context"] or 32768
        row["benchmark_context_source"] = "installed_metadata" if policy["declared_context"] else "fallback_workload_budget"
        row.update({"policy": policy, "declared_context": policy["declared_context"],
                    "context_policy": policy["context_policy"], "model_type": policy["model_type"],
                    "mlx_server": row.get("mlx_server") or policy["mlx_server"]})
    return rows


def _preferred(matches: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick a family's recorded-best variant, else the first one on disk.

    Family resolution used to be first-match-wins over discovery order, which
    meant a family's default was decided by filesystem ordering. For
    qwen3_27b_dense that silently resolved `qwen27` to MLX-4bit even though the
    September matrix measured oQ4 as the better arm at the same footprint.
    """
    for model in matches:
        wanted = (model.get("preferred") or {}).get(model["backend"])
        if wanted and Path(model["path"]).name == wanted and model["exists"]:
            return model
    # No preference recorded for this backend, or the preferred variant is not
    # on disk: prefer anything complete over a folder that cannot serve.
    for model in matches:
        if model["exists"]:
            return model
    return matches[0]


def resolve(selector: str, backend: str, config: Path) -> dict[str, Any]:
    selector_expanded = _expand(selector)
    selector_norm = selector.lower()
    target_family = LEGACY_ALIASES.get(selector_norm, selector_norm)
    candidates = [m for m in iter_models(config) if backend == "auto" or m["backend"] == backend]

    # An exact ask — a path, a registry selector, or a variant's own name — is
    # never redirected. Asking for one specific quant must get that quant.
    for model in candidates:
        if selector_expanded == model["path"] or selector == model["selector"]:
            return model
        if selector_norm == model["name"].lower():
            return model

    # A family-level ask ("qwen27") means "the right one for this family".
    family_matches = [
        model for model in candidates
        if selector_norm in {a.lower() for a in model["aliases"]}
        or target_family == model["family_id"].lower()
    ]
    if family_matches:
        return _preferred(family_matches)

    valid = ", ".join(sorted({a for m in candidates for a in m["aliases"]}))
    raise SystemExit(f"No {backend} model found for selector '{selector}'. Valid aliases: {valid}")


def _print_shell(model: dict[str, Any]) -> None:
    for key in (
        "backend",
        "path",
        "name",
        "family",
        "family_id",
        "use_case",
        "ctx_cap",
        "quant",
        "chat_template",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
        "presence_penalty",
        "repetition_context_size",
        "enable_thinking",
        "reasoning_effort",
        "reasoning_budget",
        "mlx_server",
        "mtp_supported",
        "draft_model",
        "server_binary",
        "mmproj_path",
        "spec_type",
        "spec_draft_n_max",
    ):
        value = str(model.get(key, ""))
        print(f"MODEL_{key.upper()}={shlex.quote(value)}")


def cmd_list(args: argparse.Namespace) -> None:
    rows = [m for m in iter_models(args.config) if args.backend == "auto" or m["backend"] == args.backend]
    for model in rows:
        aliases = ",".join(model["aliases"])
        exists = "ok" if model["exists"] else "missing"
        print(
            f"{model['backend']:<9} {model['family_id']:<18} "
            f"{model['quant']:<8} {exists:<7} {aliases:<32} {model['path']}"
        )
        if model.get("use_case"):
            print(f"{'':<9} → {model['use_case']}")


def cmd_resolve(args: argparse.Namespace) -> None:
    model = resolve(args.selector, args.backend, args.config)
    if args.format == "json":
        print(json.dumps(model, indent=2, sort_keys=True))
    elif args.format == "shell":
        _print_shell(model)
    else:
        raise SystemExit(f"Unknown format: {args.format}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve local model registry aliases")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list", help="List models")
    list_parser.add_argument("--backend", choices=["auto", "llamacpp", "mlx"], default="auto")
    list_parser.set_defaults(func=cmd_list)

    resolve_parser = sub.add_parser("resolve", help="Resolve a selector")
    resolve_parser.add_argument("selector")
    resolve_parser.add_argument("--backend", choices=["auto", "llamacpp", "mlx"], default="llamacpp")
    resolve_parser.add_argument("--format", choices=["json", "shell"], default="json")
    resolve_parser.set_defaults(func=cmd_resolve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
