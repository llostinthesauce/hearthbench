#!/usr/bin/env python3
"""
Discover local GGUF and MLX models.

The committed catalog contains generic matching rules. This generated local
registry contains machine-specific paths and should stay out of git. Discovery
prints JSON by default; writing requires an explicit ``--write`` destination.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

from model_policy import resolve_policy
from model_files import is_complete_gguf as _is_complete_gguf
from model_files import is_complete_mlx as _is_mlx_dir

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "configs" / "model_catalog.json"
DEFAULT_OUTPUT = ROOT / "configs" / "models.local.json"


def _default_roots() -> list[Path]:
    env_roots = os.environ.get("BENCH_MODEL_ROOTS", "")
    if env_roots:
        return [Path(os.path.expanduser(p)) for p in env_roots.split(os.pathsep) if p]
    return [
        Path.home() / ".lmstudio" / "models",
    ]


def _load_catalog(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        raise SystemExit(f"Catalog not found: {path}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}")
    if data.get("_schema_version") != 1:
        raise SystemExit(f"Unsupported catalog schema in {path}: expected _schema_version=1")
    return data


def _safe_id(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "model"


def _infer_quant(path: Path) -> str:
    text = path.name
    patterns = [
        r"(Q\d+_[A-Za-z0-9_]+)",
        r"(IQ\d+_[A-Za-z0-9_]+)",
        r"\b(BF16|F16|F32)\b",
        # Mixed-precision MLX conversions label themselves oQ4 / oQ6 / OptiQ-4bit.
        # These come BEFORE the bare [2-8]bit rule on purpose: "OptiQ-4bit" would
        # otherwise be recorded as plain "4bit", which in a report reads as the
        # uniform quantization it is specifically not.
        r"(OptiQ-?\d?(?:bit)?|oQ\d(?:\.\d)?e?)",
        r"\b(mxfp[48]|nvfp4)\b",
        r"\b([2-8]bit)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return "?"


def _skip_candidate(path: Path) -> bool:
    text = str(path).lower()
    skip_markers = (
        "mmproj",
        "embedding",
        "embed-",
        "-embed",
        "reranker",
        "rerank-",
        "-rerank",
        "gpt2",
    )
    if any(marker in text for marker in skip_markers):
        return True
    # Draft heads are named "mtp-<base>.gguf" and sit beside the model they
    # speculate for; they are not standalone models. Match the FILENAME prefix,
    # not the path — "mtp" anywhere in the path also matches legitimate
    # self-speculative builds like Qwen3.6-35B-A3B-MTP-GGUF/, which silently
    # hid a 22.9GB model from discovery and made MTP unreachable entirely.
    return path.name.lower().startswith("mtp-")


def _scan_gguf(roots: list[Path]) -> list[Path]:
    found: list[Path] = []
    split_pattern = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.gguf"):
            if not _is_complete_gguf(path) or _skip_candidate(path):
                continue
            match = split_pattern.match(path.name)
            if not match:
                found.append(path)
                continue
            prefix, index_text, total_text = match.groups()
            total = int(total_text)
            if int(index_text) != 1:
                continue
            shards = [
                path.with_name(f"{prefix}-{index:05d}-of-{total:05d}.gguf")
                for index in range(1, total + 1)
            ]
            if all(_is_complete_gguf(shard) for shard in shards):
                found.append(path)
    return sorted(set(found))


def _scan_mlx(roots: list[Path]) -> list[Path]:
    found: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for current, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in {".git", "__pycache__"}]
            path = Path(current)
            if "config.json" in files and _is_mlx_dir(path) and not _skip_candidate(path):
                found.append(path)
                dirs[:] = []
    return sorted(set(found))


def _matches(path: Path, patterns: list[str]) -> bool:
    haystack = str(path)
    name = path.name
    return any(re.search(pattern, haystack, re.IGNORECASE) or re.search(pattern, name, re.IGNORECASE) for pattern in patterns)


def _family_base(family: dict[str, Any], defaults: dict[str, Any]) -> OrderedDict[str, Any]:
    return OrderedDict(
        [
            ("id", family["id"]),
            ("name", family.get("name", family["id"])),
            ("family", family.get("family", "custom")),
            ("architecture", family.get("architecture", "dense")),
            ("ctx_cap", 0),
            ("temperature", family.get("temperature", defaults.get("temperature", 0.7))),
            ("top_p", family.get("top_p", defaults.get("top_p", 0.8))),
            ("top_k", family.get("top_k", defaults.get("top_k", 20))),
            ("min_p", family.get("min_p", defaults.get("min_p", 0.0))),
            ("repetition_penalty", family.get("repetition_penalty", defaults.get("repetition_penalty", 1.05))),
            ("presence_penalty", family.get("presence_penalty", defaults.get("presence_penalty", 0.2))),
            ("repetition_context_size", family.get("repetition_context_size", defaults.get("repetition_context_size", 2048))),
            ("enable_thinking", family.get("enable_thinking", "")),
            ("reasoning_effort", family.get("reasoning_effort", "")),
            ("reasoning_budget", family.get("reasoning_budget", "")),
            # Carried from the committed catalog so the pin is re-derived on every
            # scan. Archs mlx_lm cannot load (gemma4 E4B elastic weights,
            # gemma4_unified, muse_glimmer) are unservable without it, and a pin
            # hand-written into models.local.json is wiped by the next re-scan.
            ("mlx_server", family.get("mlx_server", "")),
            # Same reason as mlx_server: without this, MTP speculative decoding
            # can never be enabled, because serve_local.sh reads mtp_supported
            # from the generated entry and a re-scan would always drop it.
            ("mtp_supported", family.get("mtp_supported", False)),
            # Which variant a family-level selector ("qwen27") should resolve to
            # per backend. Without it, resolution is first-match-wins over
            # discovery order, so a family's default is decided by filesystem
            # ordering rather than by measurement. Carried for the same reason
            # as mlx_server: a re-scan would otherwise drop it.
            ("preferred", family.get("preferred", {})),
            ("spec_type", family.get("spec_type", "")),
            ("spec_draft_n_max", family.get("spec_draft_n_max", "")),
            ("gguf", []),
            ("mlx", []),
        ]
    )


def _probe_mlx(path: Path) -> dict[str, Any]:
    policy = resolve_policy(path)
    return {"ctx_cap": 0, **policy["sampling"], "policy": policy}


def _entry_policy(path: Path, backend: str, catalog_path: Path) -> dict[str, Any]:
    policy = resolve_policy(path, backend, catalog_path=catalog_path)
    return {"policy": policy, "declared_context": policy["declared_context"],
            "context_policy": policy["context_policy"], "model_type": policy["model_type"]}


def _add_unmatched_gguf(families: OrderedDict[str, OrderedDict[str, Any]], path: Path, defaults: dict[str, Any]) -> None:
    family_id = _safe_id(path.stem)
    base = _family_base(
        {
            "id": family_id,
            "name": path.stem,
            "family": "custom",
            "architecture": "dense",
        },
        defaults,
    )
    base["gguf"].append({"quant": _infer_quant(path), "path": str(path)})
    families[family_id] = base


def _add_unmatched_mlx(families: OrderedDict[str, OrderedDict[str, Any]], path: Path, defaults: dict[str, Any]) -> None:
    family_id = _safe_id(path.name)
    base = _family_base(
        {
            "id": family_id,
            "name": path.name,
            "family": "custom",
            "architecture": "dense",
            **_probe_mlx(path),
        },
        defaults,
    )
    base["mlx"].append({"quant": _infer_quant(path), "repo": str(path)})
    families[family_id] = base


def discover(roots: list[Path], catalog_path: Path, include_unmatched: bool) -> dict[str, Any]:
    catalog = _load_catalog(catalog_path)
    defaults = catalog.get("defaults", {})
    families: OrderedDict[str, OrderedDict[str, Any]] = OrderedDict()
    matched_gguf: set[Path] = set()
    matched_mlx: set[Path] = set()
    gguf_paths = _scan_gguf(roots)
    mlx_paths = _scan_mlx(roots)

    for family in catalog.get("model_families", []):
        row = _family_base(family, defaults)
        # First match wins: a weights file belongs to exactly one family. Without
        # this, a specific build is also claimed by its general family and shows
        # up twice — e.g. the MTP GGUF is matched by qwen3_35b_moe too, because
        # only its *directory* says MTP and _matches() also tests the bare
        # filename. Order the catalog specific-before-general.
        for path in gguf_paths:
            if path in matched_gguf:
                continue
            if _matches(path, family.get("gguf_patterns", [])):
                row["gguf"].append({"quant": _infer_quant(path), "path": str(path), **_entry_policy(path, "llamacpp", catalog_path)})
                matched_gguf.add(path)
        for path in mlx_paths:
            if path in matched_mlx:
                continue
            if _matches(path, family.get("mlx_patterns", [])):
                row["mlx"].append({"quant": _infer_quant(path), "repo": str(path), **_entry_policy(path, "mlx", catalog_path)})
                matched_mlx.add(path)
        if row["gguf"] or row["mlx"]:
            families[row["id"]] = row

    if include_unmatched:
        for path in gguf_paths:
            if path not in matched_gguf:
                _add_unmatched_gguf(families, path, defaults)
        for path in mlx_paths:
            if path not in matched_mlx:
                _add_unmatched_mlx(families, path, defaults)

    for row in families.values():
        for backend, key, path_key in (("mlx", "mlx", "repo"), ("llamacpp", "gguf", "path")):
            for entry in row[key]:
                if "policy" not in entry:
                    entry.update(_entry_policy(Path(entry[path_key]), backend, catalog_path))

    return OrderedDict(
        [
            (
                "_comment",
                "Generated by scripts/discover_models.py. Machine-specific paths; do not commit.",
            ),
            ("_schema_version", 2),
            ("model_families", list(families.values())),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover local GGUF and MLX models")
    parser.add_argument("--roots", nargs="+", type=Path, default=_default_roots(), help="Model roots to scan")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument(
        "--write",
        type=Path,
        metavar="PATH",
        help=f"Write the registry to PATH (recommended: {DEFAULT_OUTPUT})",
    )
    parser.add_argument("--known-only", action="store_true", help="Skip unmatched custom models")
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print JSON to stdout (deprecated; printing is now the default)",
    )
    args = parser.parse_args()

    roots = [Path(os.path.expanduser(str(p))).resolve() for p in args.roots]
    payload = discover(roots, args.catalog, include_unmatched=not args.known_only)
    text = json.dumps(payload, indent=2)

    if args.write is not None and not args.print:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(text + "\n")
        total = sum(len(f.get("gguf", [])) + len(f.get("mlx", [])) for f in payload["model_families"])
        print(f"Wrote {args.write} ({total} model variant(s))")
    else:
        print(text)


if __name__ == "__main__":
    main()
