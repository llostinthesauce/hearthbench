"""Native context and metadata grounded policy resolution."""

import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def policy(path, backend=None):
    from model_policy import resolve_policy

    return resolve_policy(path, backend)


def test_matched_model_reads_text_context_without_capping(tmp_path):
    path = tmp_path / "Qwen3.8-27B-oQ4"
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "vision_config": {"max_position_embeddings": 999},
                "text_config": {"max_position_embeddings": 262144},
            }
        )
    )
    result = policy(path)
    assert result["declared_context"] == 262144
    assert result["context_policy"]["default"] == "native"
    assert result["sampling"]["presence_penalty"] == 1.5
    assert result["sampling"]["top_p"] == 0.8


def test_unmatched_generation_and_architecture_fallback(tmp_path):
    path = tmp_path / "new-local-name"
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "muse_glimmer",
                "text_config": {"max_position_embeddings": 8192},
            }
        )
    )
    (path / "generation_config.json").write_text(
        json.dumps({"temperature": 0.42, "top_k": 12})
    )
    result = policy(path)
    assert result["mlx_server"] == "mlx_vlm"
    assert result["sampling"]["temperature"] == 0.42
    assert result["declared_context"] == 8192
    assert result["provenance"]["installed_revision"] is None


def test_vision_context_never_becomes_text_context(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {"model_type": "unknown", "vision_config": {"max_position_embeddings": 999}}
        )
    )
    result = policy(tmp_path)
    assert result["declared_context"] is None
    assert result["sampling"]["repetition_penalty"] == 1.0
    assert result["sampling"]["presence_penalty"] == 0.0


def test_gguf_metadata_without_loading_tensors(tmp_path):
    path = tmp_path / "custom.gguf"

    def string(value):
        raw = value.encode()
        return struct.pack("<Q", len(raw)) + raw

    path.write_bytes(
        b"GGUF"
        + struct.pack("<IQQ", 3, 0, 2)
        + string("general.architecture")
        + struct.pack("<I", 8)
        + string("qwen3_5")
        + string("qwen3_5.context_length")
        + struct.pack("<II", 4, 262144)
    )
    result = policy(path, "llamacpp")
    assert result["model_type"] == "qwen3_5"
    assert result["declared_context"] == 262144
    assert result["context_policy"]["recommended_tokens"] is None


def test_unknown_qwen_conversion_keeps_reasoning_opt_in(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5", "max_position_embeddings": 262144})
    )
    result = policy(tmp_path)
    assert result["sampling"]["enable_thinking"] is False
    assert (
        result["provenance"]["sampling_sources"]["enable_thinking"]
        == "architecture_policy"
    )


def test_thinking_and_historical_recipes_are_separate(tmp_path):
    path = tmp_path / "Qwen3.8-27B-oQ4"
    path.mkdir()
    result = policy(path)
    assert result["sampling_recipes"]["thinking"]["sampling"]["temperature"] == 1.0
    assert result["sampling"]["temperature"] == 0.7
    assert (
        result["alternative_sampling_recipes"]["qwen38_local_tuning"]["sampling"][
            "repetition_penalty"
        ]
        == 1.05
    )
    assert result["provenance"]["sampling_sources"]["top_p"].startswith("https://")


def test_registry_keeps_positive_benchmark_budget_separate_from_native_serving(
    tmp_path,
):
    import model_registry

    model = tmp_path / "fixture"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"model_type": "unknown", "max_position_embeddings": 98304})
    )
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "_schema_version": 2,
                "model_families": [{"id": "fixture", "mlx": [{"repo": str(model)}]}],
            }
        )
    )
    row = model_registry.iter_models(registry)[0]
    assert row["ctx_cap"] == 98304
    assert row["context_policy"]["default"] == "native"
