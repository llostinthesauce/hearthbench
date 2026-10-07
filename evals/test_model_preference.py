"""A family-level selector must resolve by measurement, not by scan order.

`resolve` used to be first-match-wins over discovery order, so `qwen27`
returned whichever variant the filesystem happened to list first — MLX-4bit,
even though the September matrix measured oQ4 as the better arm at the same
footprint. The committed catalog now records the choice per backend and
discovery carries it into the generated registry.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import model_registry as mr

ROOT = Path(__file__).resolve().parent.parent


def _registry(tmp_path: Path, preferred: dict | None) -> Path:
    """Two MLX variants, the preferred one listed SECOND so order cannot win."""
    family = {
        "id": "fam",
        "family": "qwen3",
        "aliases": ["fam"],
        "mlx": [
            {"repo": str(tmp_path / "First-4bit"), "quant": "4bit"},
            {"repo": str(tmp_path / "Second-oQ4"), "quant": "oQ4"},
        ],
        "gguf": [],
    }
    if preferred is not None:
        family["preferred"] = preferred
    for name in ("First-4bit", "Second-oQ4"):
        directory = tmp_path / name
        directory.mkdir(exist_ok=True)
        (directory / "config.json").write_text("{}")
        # is_complete_mlx also requires a tokenizer beside the weights.
        (directory / "tokenizer.json").write_text("{}")
        _write_weight(directory / "model.safetensors")
    path = tmp_path / "models.local.json"
    path.write_text(json.dumps({"_schema_version": 2, "model_families": [family]}))
    return path


def _write_weight(path: Path) -> None:
    header = b'{"tensor":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
    prefix = len(header).to_bytes(8, "little") + header
    path.write_bytes(prefix + b"\0" * (1024 * 1024 - len(prefix)))


def test_family_selector_honors_the_recorded_preference(tmp_path):
    config = _registry(tmp_path, {"mlx": "Second-oQ4"})
    assert mr.resolve("fam", "mlx", config)["name"] == "Second-oQ4"


def test_without_a_preference_scan_order_still_decides(tmp_path):
    config = _registry(tmp_path, None)
    assert mr.resolve("fam", "mlx", config)["name"] == "First-4bit"


def test_an_exact_path_is_never_redirected(tmp_path):
    config = _registry(tmp_path, {"mlx": "Second-oQ4"})
    exact = str(tmp_path / "First-4bit")
    assert mr.resolve(exact, "mlx", config)["path"] == exact


def test_a_variant_name_is_never_redirected(tmp_path):
    config = _registry(tmp_path, {"mlx": "Second-oQ4"})
    assert mr.resolve("First-4bit", "mlx", config)["name"] == "First-4bit"


def test_a_preference_naming_an_absent_variant_falls_back(tmp_path):
    config = _registry(tmp_path, {"mlx": "Not-On-Disk"})
    # Must still return something servable rather than an unusable pin.
    resolved = mr.resolve("fam", "mlx", config)
    assert resolved["exists"]


def test_committed_catalog_pins_the_measured_winners():
    catalog = json.loads((ROOT / "configs" / "model_catalog.json").read_text())
    dense = next(
        family for family in catalog["model_families"]
        if family["id"] == "qwen3_27b_dense"
    )
    # 8-bit was the worst arm in the matrix and 6-bit bought nothing over
    # Q5_K_M. Pinning either as a default would contradict the measurement.
    assert dense["preferred"]["mlx"] == "Qwen3.8-27B-oQ4"
    assert dense["preferred"]["llamacpp"] == "Qwen3.8-27B-Q5_K_M.gguf"


def test_discovery_carries_preference_into_the_generated_registry():
    # models.local.json is regenerated on every scan; a preference that does
    # not survive that is a preference that silently stops applying. The file
    # is generated machine state and gitignored, so a fresh checkout has none
    # until `llm models sync` runs.
    path = ROOT / "configs" / "models.local.json"
    if not path.is_file():
        import pytest

        pytest.skip("registry not generated yet; run `llm models sync`")
    generated = json.loads(path.read_text())
    dense = next(
        (family for family in generated["model_families"]
         if family["id"] == "qwen3_27b_dense"),
        None,
    )
    if dense is None:
        import pytest

        pytest.skip("qwen3_27b_dense is not on this machine")
    assert dense.get("preferred", {}).get("mlx") == "Qwen3.8-27B-oQ4"
