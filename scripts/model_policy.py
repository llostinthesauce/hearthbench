#!/usr/bin/env python3
"""Offline model policy shared by discovery, serving and external consumers.

Installed metadata describes capacity, never a measured hardware guarantee.
Unknown architectures use generation_config and neutral sampling, not a guessed
family recipe. This module imports only the standard library.
"""

from __future__ import annotations
import argparse
import json
import os
import re
import struct
from pathlib import Path
from typing import Any

CATALOG = Path(__file__).resolve().parents[1] / "configs/model_catalog.json"
SAMPLING_KEYS = (
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
)
NEUTRAL = dict(
    temperature=1.0,
    top_p=1.0,
    top_k=0,
    min_p=0.0,
    repetition_penalty=1.0,
    presence_penalty=0.0,
    repetition_context_size=2048,
)


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def match_family(path: Path, backend: str, catalog: dict[str, Any]) -> dict[str, Any]:
    key = "gguf_patterns" if backend == "llamacpp" else "mlx_patterns"
    for family in catalog.get("model_families", []):
        if any(re.search(pattern, str(path), re.I) for pattern in family.get(key, [])):
            return family
    return {}


def _gguf(path: Path) -> dict[str, Any]:
    """Read only GGUF v2/v3 metadata; skip tokenizer arrays and all tensors.

    Bounded lengths and recursion reject malformed files rather than allocating
    according to untrusted headers. No GGUF package or model runtime is needed.
    """
    formats = {
        0: "B",
        1: "b",
        2: "H",
        3: "h",
        4: "I",
        5: "i",
        6: "f",
        7: "?",
        10: "Q",
        11: "q",
        12: "d",
    }
    with path.open("rb") as file:
        size = path.stat().st_size

        def read(n):
            value = file.read(n)
            if len(value) != n:
                raise ValueError("truncated GGUF metadata")
            return value

        def number(fmt):
            return struct.unpack("<" + fmt, read(struct.calcsize("<" + fmt)))[0]

        def string(keep):
            n = number("Q")
            if n > size - file.tell():
                raise ValueError("invalid GGUF string length")
            if keep:
                if n > 1048576:
                    raise ValueError("oversized GGUF scalar")
                return read(n).decode("utf-8")
            file.seek(n, 1)

        def value(kind, keep, depth=0):
            if kind in formats:
                if keep:
                    return number(formats[kind])
                file.seek(struct.calcsize("<" + formats[kind]), 1)
            elif kind == 8:
                return string(keep)
            elif kind == 9:
                if depth >= 4:
                    raise ValueError("nested GGUF arrays")
                element = number("I")
                count = number("Q")
                if count > size:
                    raise ValueError("invalid GGUF array length")
                if element in formats:
                    n = count * struct.calcsize("<" + formats[element])
                    if n > size - file.tell():
                        raise ValueError("truncated GGUF array")
                    file.seek(n, 1)
                else:
                    for _ in range(count):
                        value(element, False, depth + 1)
            else:
                raise ValueError("unknown GGUF metadata type")

        if read(4) != b"GGUF" or number("I") not in (2, 3):
            raise ValueError("unsupported GGUF header")
        number("Q")
        count = number("Q")
        if count > 100000:
            raise ValueError("oversized GGUF metadata")
        result = {}
        for _ in range(count):
            key = string(True)
            kind = number("I")
            keep = key == "general.architecture" or key.endswith(".context_length")
            parsed = value(kind, keep)
            if keep:
                result[key] = parsed
        return result


def resolve_policy(
    model_path: str | Path, backend: str | None = None, *, catalog_path: Path = CATALOG
) -> dict[str, Any]:
    path = Path(os.path.expandvars(os.path.expanduser(str(model_path))))
    backend = backend or ("llamacpp" if path.suffix.lower() == ".gguf" else "mlx")
    if backend in ("mlx-vlm", "mlx-kv"):
        backend = "mlx"
    if backend not in ("mlx", "llamacpp"):
        raise ValueError(f"Unsupported backend: {backend}")
    catalog = _json(catalog_path)
    family = match_family(path, backend, catalog)
    config = _json(path / "config.json") if backend == "mlx" else {}
    generation = _json(path / "generation_config.json") if backend == "mlx" else {}
    diagnostics = []
    metadata = {}
    if backend == "llamacpp":
        try:
            metadata = _gguf(path)
        except (OSError, ValueError, UnicodeError, struct.error) as exc:
            diagnostics.append(f"GGUF metadata unavailable: {exc}")
    text = config.get("text_config")
    if not isinstance(text, dict):
        text = config
    model_type = config.get("model_type") or metadata.get("general.architecture")
    declared = (
        text.get("max_position_embeddings")
        if backend == "mlx"
        else metadata.get(f"{model_type}.context_length")
    )
    if not isinstance(declared, int) or isinstance(declared, bool) or declared <= 0:
        declared = None
    if declared is None:
        diagnostics.append(
            "Text context length is not declared in readable installed metadata."
        )
    sources = {key: "neutral_fallback" for key in NEUTRAL}
    sources.update(
        {
            key: "installed_generation_config"
            for key in SAMPLING_KEYS
            if key in generation
        }
    )
    architecture = catalog.get("architecture_policy", {}).get(model_type, {})
    sources.update(
        {key: "architecture_policy" for key in architecture.get("sampling", {})}
    )
    sampling = {
        **NEUTRAL,
        **architecture.get("sampling", {}),
        **{k: generation[k] for k in SAMPLING_KEYS if k in generation},
    }
    overrides = {k: family[k] for k in SAMPLING_KEYS if k in family}
    sampling.update(overrides)
    sources.update({key: "catalog_local_tuning" for key in overrides})
    recipes = catalog.get("sampling_recipes", {})
    recipe_id = family.get("sampling_recipe")
    recipe = recipes.get(recipe_id, {})
    sampling.update(recipe.get("sampling", {}))
    sources.update(
        {
            key: recipe.get("source", "catalog_recipe")
            for key in recipe.get("sampling", {})
        }
    )
    context = {"default": "native", "recommended_tokens": None, "verified_tokens": None}
    context.update(family.get("context_policy", {}))
    modalities = ["text"]
    vision = bool(config.get("vision_config")) or config.get("model_type") in (
        "gemma4_unified",
        "muse_glimmer",
    )
    projector = (
        next(iter(path.parent.glob("*mmproj*.gguf")), None)
        if backend == "llamacpp"
        else None
    )
    if vision or projector:
        modalities.append("image")
    # A config declaration records potential inputs, not serving acceptance.
    if config.get("audio_config"):
        modalities.append("audio")
    return {
        "family_id": family.get("id")
        or re.sub(r"[^a-z0-9]+", "_", path.stem.lower()).strip("_"),
        "family": family.get("family", "custom"),
        "model_type": model_type,
        "declared_context": declared,
        "context_policy": context,
        "sampling": sampling,
        "modalities": modalities,
        "alternative_sampling_recipes": {
            ref: recipes.get(ref, {})
            for ref in family.get("alternative_sampling_recipes", [])
        },
        "sampling_recipes": {
            mode: recipes.get(ref, {})
            for mode, ref in family.get("sampling_modes", {}).items()
        },
        "mlx_server": family.get("mlx_server")
        or architecture.get("mlx_server", "mlx_lm"),
        "provenance": {
            "installed_revision": None,
            "model_card": family.get("model_card") or recipe.get("source"),
            "metadata_source": str(path / "config.json")
            if backend == "mlx"
            else str(path),
            "generation_config": str(path / "generation_config.json")
            if generation
            else None,
            "sampling_recipe": recipe_id,
            "sampling_source": recipe.get("source")
            or (
                "catalog_local_tuning"
                if overrides
                else "installed_generation_config"
                if generation
                else "neutral_fallback"
            ),
            "recipe": recipe,
            "sampling_sources": sources,
            "diagnostics": diagnostics,
            "modality_evidence": {
                "vision_config": vision,
                "adjacent_projector": str(projector) if projector else None,
                "live_verified": False,
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path")
    parser.add_argument("--backend", choices=["mlx", "llamacpp", "mlx-vlm", "mlx-kv"])
    args = parser.parse_args()
    try:
        result = resolve_policy(args.model_path, args.backend)
    except ValueError as exc:
        parser.exit(2, str(exc) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
