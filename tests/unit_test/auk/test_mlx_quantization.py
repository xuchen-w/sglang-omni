# SPDX-License-Identifier: Apache-2.0
"""Selective q8 loading, floating compute dtypes, and native artifact contracts."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

mx = pytest.importorskip("mlx.core")
import mlx.nn as nn
from mlx.utils import tree_flatten
from transformers import (
    Qwen2_5OmniAudioEncoderConfig,
    Qwen2_5OmniTextConfig,
    Qwen2_5OmniThinkerConfig,
    Qwen2_5OmniVisionEncoderConfig,
)
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniThinkerForConditionalGeneration,
)

from sglang_omni.models.auk import stages as torch_stages
from sglang_omni.models.auk.dit import AuKDit as TorchDiT
from sglang_omni.models.auk.flow_matching import AuKFlowMatching as TorchFlow
from sglang_omni.models.auk.mlx import conditioning, loader
from sglang_omni.models.auk.mlx.conditioning import AuKMlxConditionEncoder
from sglang_omni.models.auk.mlx.convert import convert_checkpoint
from sglang_omni.models.auk.mlx.flow_matching import AuKSampleItem
from sglang_omni.models.auk.mlx.quantization import (
    ARTIFACT_FORMAT,
    ARTIFACT_VERSION,
    QUANTIZATION,
    layer_dtype,
)


@pytest.fixture(autouse=True)
def cpu_stream():
    with mx.stream(mx.cpu):
        yield


@pytest.fixture
def flow_checkpoint(tmp_path):
    torch.manual_seed(53)
    arch = dict(
        dim=64,
        heads=4,
        dim_head=16,
        latent_dim=64,
        text_hidden_dim=64,
        num_layers=1,
        num_single_layers=1,
    )
    model = TorchFlow(TorchDiT(**arch), num_llm_layers=2).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim == 2:
                parameter.normal_(std=0.03)
    root = tmp_path / "source"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps({"model": {"name": "AuK", "arch": arch, "vae": {"latent_dim": 64}}})
    )
    save_file(
        {name: value.detach().clone() for name, value in model.state_dict().items()},
        root / "model.safetensors",
    )
    save_file({"global_mean": torch.zeros(64)}, root / "vae.safetensors")
    return root


@pytest.fixture
def conditioner_checkpoint(tmp_path, monkeypatch):
    config = Qwen2_5OmniThinkerConfig(
        text_config=Qwen2_5OmniTextConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 1000000,
                "mrope_section": [2, 3, 3],
            },
        ),
        audio_config=Qwen2_5OmniAudioEncoderConfig(
            num_mel_bins=8,
            encoder_layers=1,
            encoder_attention_heads=4,
            encoder_ffn_dim=64,
            d_model=32,
            output_dim=64,
            n_window=4,
            max_source_positions=8,
        ),
        vision_config=Qwen2_5OmniVisionEncoderConfig(
            depth=1,
            hidden_size=16,
            intermediate_size=24,
            num_heads=2,
            out_hidden_size=64,
        ),
        audio_token_index=120,
        image_token_index=121,
        video_token_index=122,
    )
    torch.manual_seed(54)
    reference = Qwen2_5OmniThinkerForConditionalGeneration(config).eval()
    root = tmp_path / "conditioner"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps({"model_type": "qwen2_5_omni", "thinker_config": config.to_dict()})
    )
    save_file(
        {
            f"thinker.{name}": value.detach().clone()
            for name, value in reference.state_dict().items()
        },
        root / "model.safetensors",
    )
    monkeypatch.setattr(
        conditioning.Qwen2_5OmniConfig,
        "from_pretrained",
        lambda path: SimpleNamespace(thinker_config=config),
    )
    monkeypatch.setattr(
        conditioning.Qwen2_5OmniProcessor,
        "from_pretrained",
        lambda path: object(),
    )
    return root


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_q8_flow_uses_floating_compute_and_reduces_resident_weights(
    flow_checkpoint, dtype
):
    floating = loader.load_flow(str(flow_checkpoint), dtype)
    quantized = loader.load_flow(str(flow_checkpoint), dtype, "mlx_q8")
    assert isinstance(quantized.transformer.proj_out, nn.QuantizedLinear)
    assert quantized.transformer.proj_out.weight.dtype == mx.uint32
    assert quantized.transformer.dtype == dtype
    assert layer_dtype(quantized.transformer.time_embed.time_mlp[0]) == dtype
    assert quantized.transformer.rotary_embed.inv_freq.dtype == mx.float32
    assert quantized.layer_weights.dtype == quantized.layer_scale.dtype == mx.float32
    qbytes = sum(value.nbytes for _, value in tree_flatten(quantized.parameters()))
    fbytes = sum(value.nbytes for _, value in tree_flatten(floating.parameters()))
    assert qbytes < fbytes * (0.7 if dtype == mx.bfloat16 else 0.4)
    generator = np.random.default_rng(2)
    item = AuKSampleItem(
        conditioning=mx.array(generator.normal(size=(3, 64)).astype(np.float32)),
        text_mask=mx.ones((3,), dtype=mx.bool_),
        target_frames=5,
        noise=mx.array(generator.normal(size=(5, 64)).astype(np.float32)),
    )
    target = np.array(floating.sample(item, steps=2, cfg_strength=2))
    actual = np.array(quantized.sample(item, steps=2, cfg_strength=2))
    assert actual.dtype == np.float32
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, target, atol=0.03, rtol=0.02)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_q8_conditioner_keeps_audio_tower_floating(conditioner_checkpoint, dtype):
    floating = AuKMlxConditionEncoder(str(conditioner_checkpoint), dtype=dtype)
    quantized = AuKMlxConditionEncoder(
        str(conditioner_checkpoint), dtype=dtype, quantization="mlx_q8"
    )
    assert isinstance(quantized.model.model.embed_tokens, nn.QuantizedEmbedding)
    assert isinstance(
        quantized.model.model.layers[0].self_attn.q_proj, nn.QuantizedLinear
    )
    assert isinstance(quantized.model.audio_tower.layers[0].self_attn.q_proj, nn.Linear)
    assert quantized.model.audio_tower.layers[0].self_attn.q_proj.weight.dtype == dtype
    ids = np.array([[2, 120, 120, 120, 3]])
    mask = np.ones_like(ids)
    features = np.random.default_rng(3).normal(size=(1, 8, 11)).astype(np.float32)
    feature_mask = np.ones((1, 11))
    target = np.array(floating.model(ids, mask, features, feature_mask))
    actual = np.array(quantized.model(ids, mask, features, feature_mask))
    assert actual.dtype == np.float32
    relative_error = np.linalg.norm(actual - target, axis=-1) / np.maximum(
        np.linalg.norm(target, axis=-1), 1e-8
    )
    assert np.max(relative_error) < 0.025


def write_native_flow(root, model, source_config, dtype, quantization, weights=None):
    root.mkdir()
    marker = dict(
        format=ARTIFACT_FORMAT,
        version=ARTIFACT_VERSION,
        component="flow",
        layout="mlx",
        dtype=dtype,
        quantization=quantization,
        source={"repo_or_path": "unit-test"},
        vae_layout="torch",
    )
    config = json.loads(source_config.read_text())
    config["mlx_artifact"] = marker
    (root / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(
        str(root / "model.safetensors"),
        dict(tree_flatten(model.parameters())) if weights is None else weights,
    )


@pytest.mark.parametrize("quantization", [None, "mlx_q8"])
def test_native_flow_roundtrip_preserves_parameters_and_fusion(
    flow_checkpoint, tmp_path, monkeypatch, quantization
):
    original = loader.load_flow(str(flow_checkpoint), mx.bfloat16, quantization)
    native = tmp_path / "native"
    write_native_flow(
        native,
        original,
        flow_checkpoint / "config.json",
        "bfloat16",
        QUANTIZATION if quantization else None,
    )
    monkeypatch.setattr(
        loader,
        "prepare_weight",
        Mock(side_effect=AssertionError("native weights converted twice")),
    )
    restored = loader.load_flow(str(native), mx.bfloat16, quantization)
    for (name, expected), (loaded_name, value) in zip(
        tree_flatten(original.parameters()), tree_flatten(restored.parameters())
    ):
        assert name == loaded_name
        np.testing.assert_array_equal(
            np.array(value.astype(mx.float32)), np.array(expected.astype(mx.float32))
        )
    expected_fusion = loader.load_fusion(str(flow_checkpoint))
    for value, expected in zip(loader.load_fusion(str(native)), expected_fusion):
        np.testing.assert_array_equal(np.array(value), np.array(expected))


def test_native_artifact_requires_matching_dtype_and_quantization(
    flow_checkpoint, tmp_path
):
    model = loader.load_flow(str(flow_checkpoint), mx.bfloat16, "mlx_q8")
    root = tmp_path / "native"
    write_native_flow(
        root, model, flow_checkpoint / "config.json", "bfloat16", QUANTIZATION
    )
    with pytest.raises(ValueError, match="artifact dtype"):
        loader.load_flow(str(root), mx.float32, "mlx_q8")
    with pytest.raises(ValueError, match="artifact quantization"):
        loader.load_flow(str(root), mx.bfloat16)
    config = json.loads((root / "config.json").read_text())
    config["mlx_artifact"]["dtype"] = "float32"
    (root / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="tensor .* must use"):
        loader.load_flow(str(root), mx.float32, "mlx_q8")


@pytest.mark.parametrize(
    "factory", ["create_conditioning_executor", "create_auk_engine_executor"]
)
def test_q8_setting_is_forwarded_only_to_native_backend(monkeypatch, factory):
    from sglang_omni.models.auk.mlx import stages

    native = Mock(return_value="native")
    monkeypatch.setattr(stages, factory, native)
    monkeypatch.setattr("sglang.srt.hardware_backend.mlx.runtime.use_mlx", lambda: True)
    assert getattr(torch_stages, factory)("unused", quantization="mlx_q8") == "native"
    assert native.call_args.kwargs["quantization"] == "mlx_q8"
    monkeypatch.setattr(
        "sglang.srt.hardware_backend.mlx.runtime.use_mlx", lambda: False
    )
    with pytest.raises(ValueError, match="requires the native MLX backend"):
        getattr(torch_stages, factory)("unused", quantization="mlx_q8")


def test_unsupported_quantization_rejected_before_loading(monkeypatch):
    monkeypatch.setattr(
        conditioning,
        "resolve_checkpoint",
        Mock(side_effect=AssertionError("downloaded")),
    )
    with pytest.raises(ValueError, match="None or mlx_q8"):
        AuKMlxConditionEncoder("unused", dtype=mx.bfloat16, quantization="mlx_q4")
    with pytest.raises(ValueError, match="None or mlx_q8"):
        loader.load_flow("unused", mx.bfloat16, "mlx_q4")


def test_native_cache_limit_is_explicit_and_idempotent(monkeypatch):
    monkeypatch.setattr(
        loader, "resolve_concrete_device", lambda *args: torch.device("mps", 0)
    )
    monkeypatch.setattr(
        loader, "current_platform", SimpleNamespace(is_mps=lambda: True)
    )
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    setting = Mock()
    monkeypatch.setattr(mx, "set_cache_limit", setting)
    monkeypatch.setenv("SGLANG_MLX_CACHE_LIMIT_GB", "0.5")
    loader.validate_device("mps", 0)
    loader.validate_device("mps", 0)
    assert [call.args for call in setting.call_args_list] == [
        (512 * 1024**2,),
        (512 * 1024**2,),
    ]
    monkeypatch.delenv("SGLANG_MLX_CACHE_LIMIT_GB")
    loader.validate_device("mps", 0)
    assert setting.call_count == 2


def test_negative_native_cache_limit_fails_before_download(monkeypatch):
    from sglang_omni.models.auk.mlx import stages

    monkeypatch.setattr(
        loader, "resolve_concrete_device", lambda *args: torch.device("mps", 0)
    )
    monkeypatch.setattr(
        loader, "current_platform", SimpleNamespace(is_mps=lambda: True)
    )
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setenv("SGLANG_MLX_CACHE_LIMIT_GB", "-1")
    monkeypatch.setattr(
        stages, "resolve_checkpoint", Mock(side_effect=AssertionError("downloaded"))
    )
    with pytest.raises(ValueError, match="SGLANG_MLX_CACHE_LIMIT_GB must be >= 0"):
        stages.create_decode_executor(
            "unused", device="mps", gpu_id=0, max_batch_size=1, max_batch_wait_ms=0
        )


@pytest.mark.parametrize("quantization", [None, "mlx_q8"])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_converter_roundtrip_preserves_conditioning_and_generation(
    flow_checkpoint, conditioner_checkpoint, tmp_path, quantization, dtype
):
    output = convert_checkpoint(
        str(flow_checkpoint),
        str(tmp_path / "converted"),
        text_encoder_path=str(conditioner_checkpoint),
        dtype=dtype,
        quantization=quantization,
        shard_bytes=4096,
        chunk_bytes=1024,
    )
    compute = getattr(mx, dtype)
    original_conditioner = AuKMlxConditionEncoder(
        str(conditioner_checkpoint), dtype=compute, quantization=quantization
    )
    restored_conditioner = AuKMlxConditionEncoder(
        str(output / "conditioner"), dtype=compute, quantization=quantization
    )
    ids = np.array([[2, 120, 120, 120, 3]])
    mask = np.ones_like(ids)
    features = np.random.default_rng(9).normal(size=(1, 8, 11)).astype(np.float32)
    feature_mask = np.ones((1, 11))
    original_hidden = original_conditioner.model(ids, mask, features, feature_mask)
    restored_hidden = restored_conditioner.model(ids, mask, features, feature_mask)
    np.testing.assert_array_equal(np.array(restored_hidden), np.array(original_hidden))
    original_flow = loader.load_flow(str(flow_checkpoint), compute, quantization)
    restored_flow = loader.load_flow(str(output), compute, quantization)
    item = AuKSampleItem(
        conditioning=original_hidden[0, -1],
        text_mask=mx.ones((5,), dtype=mx.bool_),
        target_frames=5,
        seed=7,
    )
    np.testing.assert_array_equal(
        np.array(restored_flow.sample(item, steps=2, cfg_strength=2)),
        np.array(original_flow.sample(item, steps=2, cfg_strength=2)),
    )
    for value, expected in zip(
        loader.load_fusion(str(output)), loader.load_fusion(str(flow_checkpoint))
    ):
        np.testing.assert_array_equal(np.array(value), np.array(expected))


@pytest.mark.parametrize("override", [None, "custom-conditioner"])
def test_conditioning_factory_selects_bundled_or_explicit_encoder(
    flow_checkpoint, tmp_path, monkeypatch, override
):
    from sglang_omni.models.auk import constants
    from sglang_omni.models.auk.mlx import stages

    model = loader.load_flow(str(flow_checkpoint), mx.bfloat16, "mlx_q8")
    root = tmp_path / "native"
    write_native_flow(
        root, model, flow_checkpoint / "config.json", "bfloat16", QUANTIZATION
    )
    encoder = Mock(return_value=object())
    monkeypatch.setattr(stages, "validate_device", lambda *args: None)
    monkeypatch.setattr(stages, "AuKMlxConditionEncoder", encoder)
    monkeypatch.setattr(stages, "load_vae", lambda *args: object())
    monkeypatch.setattr(stages, "load_fusion", lambda *args: object())
    stages.create_conditioning_executor(
        str(root),
        device="mps",
        gpu_id=0,
        dtype="bfloat16",
        text_encoder_path=override or constants.DEFAULT_TEXT_ENCODER,
        max_batch_size=1,
        max_batch_wait_ms=0,
        reference_encoding="sample",
        quantization="mlx_q8",
    )
    assert encoder.call_args.args == (override or str(root / "conditioner"),)
    assert encoder.call_args.kwargs["quantization"] == "mlx_q8"


@pytest.mark.parametrize("shape", [(), (16,), (2, 2, 2)])
def test_native_artifact_rejects_nonmatrix_packed_tensor(
    flow_checkpoint, tmp_path, shape
):
    model = loader.load_flow(str(flow_checkpoint), mx.bfloat16, "mlx_q8")
    root = tmp_path / "malformed-native"
    weights = dict(tree_flatten(model.parameters()))
    weights["transformer.proj_out.weight"] = mx.zeros(shape, dtype=mx.uint32)
    write_native_flow(
        root,
        model,
        flow_checkpoint / "config.json",
        "bfloat16",
        QUANTIZATION,
        weights=weights,
    )
    with pytest.raises(ValueError, match="Packed AuK tensor must be a matrix"):
        loader.load_flow(str(root), mx.bfloat16, "mlx_q8")
