"""Small, dependency-free checks for loadable local model artifacts."""
from __future__ import annotations

import json
from pathlib import Path


MIN_GGUF_BYTES = 1024 * 1024
MIN_SAFETENSORS_BYTES = 1024 * 1024
MAX_SAFETENSORS_HEADER_BYTES = 100 * 1024 * 1024


def is_complete_gguf(path: Path) -> bool:
    """Return whether *path* looks like a completed GGUF download."""
    try:
        if not path.is_file() or path.stat().st_size < MIN_GGUF_BYTES:
            return False
        with path.open("rb") as handle:
            return handle.read(4) == b"GGUF"
    except OSError:
        return False


def is_complete_mlx(path: Path) -> bool:
    """Return whether an MLX checkout has config plus local weight shards."""
    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    if not any((path / name).is_file() for name in (
        "tokenizer.json", "tokenizer.model", "spiece.model", "vocab.json"
    )):
        return False
    try:
        index_path = path / "model.safetensors.index.json"
        if index_path.is_file():
            index = json.loads(index_path.read_text())
            shard_names = set(index.get("weight_map", {}).values())
            if not shard_names or not all(isinstance(name, str) for name in shard_names):
                return False
            weights = []
            for name in shard_names:
                candidate = path / name
                if candidate.parent.resolve() != path.resolve() or not candidate.resolve().is_relative_to(path.resolve()):
                    return False
                weights.append(candidate)
        else:
            weights = list(path.glob("*.safetensors"))
        return bool(weights) and all(
            weight.resolve().is_relative_to(path.resolve()) and _is_readable_safetensors(weight)
            for weight in weights
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False


def _is_readable_safetensors(path: Path) -> bool:
    try:
        size = path.stat().st_size
        if not path.is_file() or size < MIN_SAFETENSORS_BYTES:
            return False
        with path.open("rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                return False
            header_length = int.from_bytes(raw_length, "little")
            if not 2 <= header_length <= min(MAX_SAFETENSORS_HEADER_BYTES, size - 8):
                return False
            header = json.loads(handle.read(header_length))
        if not isinstance(header, dict) or not header:
            return False
        tensors = [value for name, value in header.items() if name != "__metadata__"]
        if not tensors:
            return False
        for tensor in tensors:
            if not isinstance(tensor, dict):
                return False
            offsets = tensor.get("data_offsets")
            if not isinstance(offsets, list) or len(offsets) != 2:
                return False
            start, end = offsets
            if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start <= end <= size - header_length - 8:
                return False
        return True
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return False


def is_complete_model(path: Path, backend: str) -> bool:
    if backend == "llamacpp":
        return is_complete_gguf(path)
    if backend == "mlx":
        return is_complete_mlx(path)
    return False


def validate_model_path(path: Path, root: Path, backend: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(root.expanduser().resolve()):
        raise ValueError(f"Model must live under the configured LM Studio root: {root}")
    if not is_complete_model(resolved, backend):
        raise ValueError(f"Model is incomplete or unsupported for {backend}: {resolved}")
    return resolved
