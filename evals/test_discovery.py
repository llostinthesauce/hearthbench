"""Discovery rules that silently hide or duplicate models when they regress.

Both bugs here made a model unusable without any error message:

  * `_skip_candidate` matched "mtp" anywhere in the path, so the 22.9GB
    self-speculative Qwen3.6-35B-A3B-MTP-GGUF build was skipped entirely and
    MTP was unreachable — while the alias `qwen35-mtp` still existed.
  * Discovery let every matching family claim a file, so a specific build was
    also claimed by its general family and appeared twice in the registry.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import discover_models as dm

ROOT = Path("/models")


def _write_weight(path: Path) -> None:
    header = b'{"tensor":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
    prefix = len(header).to_bytes(8, "little") + header
    path.write_bytes(prefix + b"\0" * (1024 * 1024 - len(prefix)))


def test_mtp_draft_head_is_skipped():
    # "mtp-<base>.gguf" beside a model is a draft head, not a standalone model.
    assert dm._skip_candidate(ROOT / "unsloth/gemma-4-26B-A4B-it-qat-GGUF/mtp-gemma-4-26B-A4B-it.gguf")


def test_self_speculative_mtp_model_is_not_skipped():
    # Only the DIRECTORY says MTP; this is the real 22.9GB model.
    assert not dm._skip_candidate(ROOT / "unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf")


def test_projector_and_embedding_still_skipped():
    assert dm._skip_candidate(ROOT / "x/mmproj-BF16.gguf")
    assert dm._skip_candidate(ROOT / "Qwen/Qwen3-Embedding-8B-GGUF/Qwen3-Embedding-8B-Q8_0.gguf")
    assert dm._skip_candidate(ROOT / "Qwen/Qwen3-Reranker-0.6B-GGUF/qwen3-reranker-q8_0.gguf")
    assert dm._skip_candidate(ROOT / "local/gpt2/model.safetensors")


def test_ordinary_model_is_not_skipped():
    assert not dm._skip_candidate(ROOT / "mlx-community/Qwen3.6-35B-A3B-4bit/model.gguf")


def test_default_discovery_root_is_only_lmstudio(monkeypatch):
    monkeypatch.delenv("BENCH_MODEL_ROOTS", raising=False)

    assert dm._default_roots() == [Path.home() / ".lmstudio" / "models"]


def test_infer_quant_recognizes_common_mlx_six_bit_directory():
    assert dm._infer_quant(ROOT / "mlx-community/gemma-4-26B-A4B-it-qat-6bit") == "6bit"


def test_catalog_orders_specific_families_before_general():
    """A general family must not precede a variant it would also match.

    discover() is first-match-wins, so if qwen3_35b_moe came first it would
    claim the MTP GGUF (its bare filename carries no "MTP") and the MTP family
    would silently end up empty.
    """
    import json
    cat = json.loads((Path(__file__).resolve().parent.parent /
                      "configs" / "model_catalog.json").read_text())
    order = [f["id"] for f in cat["model_families"]]
    for specific, general in (("qwen3_35b_moe_mtp", "qwen3_35b_moe"),
                              ("qwen3_35b_moe_uncensored", "qwen3_35b_moe"),
                              ("gemma4_12b_8bit", "gemma4_12b_dense")):
        assert order.index(specific) < order.index(general), \
            f"{specific} must precede {general} in model_catalog.json"


def test_qwen38_quant_variants_group_under_qwen27_family(tmp_path):
    for quant in ("4bit", "8bit"):
        model = tmp_path / "lmstudio-community" / f"Qwen3.8-27B-MLX-{quant}"
        model.mkdir(parents=True)
        (model / "config.json").write_text("{}")
        (model / "tokenizer.json").write_text("{}")
        _write_weight(model / "model.safetensors")

    registry = dm.discover(
        [tmp_path],
        Path(__file__).resolve().parent.parent / "configs" / "model_catalog.json",
        include_unmatched=True,
    )

    family = next(row for row in registry["model_families"] if row["id"] == "qwen3_27b_dense")
    assert {item["quant"] for item in family["mlx"]} == {"4bit", "8bit"}
    assert family["reasoning_effort"] == "medium"
    assert family["reasoning_budget"] == 4096
    assert family["enable_thinking"] is False
    assert not [row for row in registry["model_families"] if row["id"].startswith("qwen3_8_27b")]


def test_every_alias_resolves_to_a_real_family():
    """A dangling alias fails at serve time with a confusing 'no model' error."""
    import json
    import model_registry as mr
    cat = json.loads((Path(__file__).resolve().parent.parent /
                      "configs" / "model_catalog.json").read_text())
    ids = {f["id"] for f in cat["model_families"]}
    dangling = {a: t for a, t in mr.LEGACY_ALIASES.items() if t not in ids}
    assert not dangling, f"aliases point at missing families: {dangling}"


def _write_catalog(path: Path) -> None:
    path.write_text(json.dumps({
        "_schema_version": 1,
        "defaults": {},
        "model_families": [],
    }))


def test_mlx_directory_requires_actual_weight_files(tmp_path):
    """A cache directory with metadata only must not be offered as loadable."""
    model = tmp_path / "org" / "unfinished-model"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    (model / "tokenizer.json").write_text("{}")

    assert dm._is_mlx_dir(model) is False


def test_mlx_directory_with_safetensors_is_complete(tmp_path):
    model = tmp_path / "org" / "complete-model"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    (model / "tokenizer.json").write_text("{}")
    _write_weight(model / "model.safetensors")

    assert dm._is_mlx_dir(model) is True


def test_mlx_directory_requires_every_indexed_weight_shard(tmp_path):
    model = tmp_path / "org" / "partial-sharded-model"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    (model / "tokenizer.json").write_text("{}")
    _write_weight(model / "model-00001-of-00002.safetensors")
    (model / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {
            "a": "model-00001-of-00002.safetensors",
            "b": "model-00002-of-00002.safetensors",
        }
    }))

    assert dm._is_mlx_dir(model) is False


def test_mlx_directory_requires_tokenizer_assets(tmp_path):
    model = tmp_path / "org" / "weights-only"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    _write_weight(model / "model.safetensors")

    assert dm._is_mlx_dir(model) is False


def test_mlx_rejects_a_large_but_truncated_weight_payload(tmp_path):
    model = tmp_path / "org" / "truncated"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    (model / "tokenizer.json").write_text("{}")
    header = json.dumps({"tensor": {"dtype": "F32", "shape": [1000000], "data_offsets": [0, 4000000]}}).encode()
    (model / "model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header + b"x" * (1024 * 1024))

    assert dm._is_mlx_dir(model) is False


def test_gguf_requires_magic_and_nontrivial_size(tmp_path):
    tiny = tmp_path / "tiny.gguf"
    tiny.write_bytes(b"GGUF")
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"NOPE" + b"x" * (1024 * 1024))
    valid = tmp_path / "valid.gguf"
    valid.write_bytes(b"GGUF" + b"x" * (1024 * 1024))

    assert dm._is_complete_gguf(tiny) is False
    assert dm._is_complete_gguf(bad) is False
    assert dm._is_complete_gguf(valid) is True


def test_split_gguf_registers_once_only_when_all_shards_are_present(tmp_path):
    complete = tmp_path / "complete"
    complete.mkdir()
    for index in (1, 2):
        (complete / f"model-{index:05d}-of-00002.gguf").write_bytes(
            b"GGUF" + b"x" * (1024 * 1024)
        )
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "other-00001-of-00002.gguf").write_bytes(
        b"GGUF" + b"x" * (1024 * 1024)
    )

    assert dm._scan_gguf([tmp_path]) == [complete / "model-00001-of-00002.gguf"]


def test_cli_prints_without_writing_by_default(monkeypatch, tmp_path, capsys):
    catalog = tmp_path / "catalog.json"
    _write_catalog(catalog)
    default_output = tmp_path / "must-not-be-created.json"
    monkeypatch.setattr(dm, "DEFAULT_OUTPUT", default_output)
    monkeypatch.setattr(sys, "argv", [
        "discover_models.py", "--roots", str(tmp_path), "--catalog", str(catalog),
    ])

    dm.main()

    assert not default_output.exists()
    assert json.loads(capsys.readouterr().out)["_schema_version"] == 2


def test_cli_writes_only_to_explicit_path(monkeypatch, tmp_path, capsys):
    catalog = tmp_path / "catalog.json"
    _write_catalog(catalog)
    output = tmp_path / "registry.json"
    monkeypatch.setattr(sys, "argv", [
        "discover_models.py", "--roots", str(tmp_path), "--catalog", str(catalog),
        "--write", str(output),
    ])

    dm.main()

    assert json.loads(output.read_text())["_schema_version"] == 2
    assert "Wrote" in capsys.readouterr().out


def test_muse_gguf_is_catalogued_with_its_own_sampling(tmp_path):
    model = tmp_path / 'lmstudio-community/Muse-Glimmer-30B-GGUF/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf'
    model.parent.mkdir(parents=True)
    model.write_bytes(b'GGUF' + b'\0' * (1024 * 1024))
    registry = dm.discover([tmp_path], dm.DEFAULT_CATALOG, include_unmatched=True)
    family = registry['model_families'][0]
    assert family['id'] == 'muse_glimmer_30b_dense'
    assert (family['temperature'], family['top_p'], family['top_k']) == (1.0, 0.95, 64)
    assert family['repetition_penalty'] == 1.0
    assert family['presence_penalty'] == 0.0


def test_bonsai_gets_explicit_qwen_thinking_policy(tmp_path):
    model = tmp_path / 'prism-ml/Ternary-Bonsai-27B-mlx-2bit'
    model.mkdir(parents=True)
    (model / 'config.json').write_text('{"model_type":"qwen3_5"}')
    (model / 'tokenizer.json').write_text('{}')
    _write_weight(model / 'model.safetensors')
    registry = dm.discover([tmp_path], dm.DEFAULT_CATALOG, include_unmatched=True)
    family = registry['model_families'][0]
    assert family['id'] == 'ternary_bonsai_27b'
    assert family['enable_thinking'] is False
    assert family['repetition_context_size'] == 2048
