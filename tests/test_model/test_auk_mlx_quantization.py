# SPDX-License-Identifier: Apache-2.0
"""Opt-in real-weight q8 artifact regression without downloads or parallel models."""

from __future__ import annotations

import gc
import hashlib
import os
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import numpy as np
import pytest

pytestmark = pytest.mark.accelerator


@pytest.fixture
def local_checkpoints() -> tuple[Path, Path, Path]:
    names = (
        "AUK_MLX_CHECKPOINT",
        "AUK_QWEN_CHECKPOINT",
        "AUK_MLX_QUANTIZED_CHECKPOINT",
    )
    if not all(os.environ.get(name) for name in names):
        pytest.skip(
            f"Set {', '.join(names)} to local original and converted checkpoints"
        )
    paths = [Path(os.environ[name]).resolve() for name in names]
    for name, path in zip(names, paths):
        if not path.is_dir():
            pytest.fail(f"{name} must name an existing local checkpoint directory")
    if not (paths[2] / "conditioner").is_dir():
        pytest.fail("The converted checkpoint must contain its bundled conditioner")
    return paths[0], paths[1], paths[2]


@pytest.fixture
def metal_backend(
    local_checkpoints: tuple[Path, Path, Path], monkeypatch
) -> Iterator[ModuleType]:
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Real AuK q8 regression requires Apple Metal")
    elif os.environ.get("MLX_ENABLE_TF32") != "0":
        pytest.fail("Set MLX_ENABLE_TF32=0 before starting the real-checkpoint test")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", True)
    from sglang_omni.models.auk.mlx.loader import validate_device

    validate_device("mps", 0)
    with (
        mx.stream(mx.gpu),
        patch(
            "huggingface_hub.snapshot_download",
            side_effect=AssertionError("No downloads allowed"),
        ),
        patch(
            "huggingface_hub.hf_hub_download",
            side_effect=AssertionError("No downloads allowed"),
        ),
    ):
        try:
            yield mx
        finally:
            mx.synchronize()
            gc.collect()
            mx.clear_cache()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def test_converted_artifact_preserves_original_vae(
    local_checkpoints: tuple[Path, Path, Path],
) -> None:
    from sglang_omni.models.auk.weight_loader import resolve_vae_file

    original, _, artifact = local_checkpoints
    source = resolve_vae_file(str(original))
    assert source is not None, "The original checkpoint has no VAE weights"
    assert file_sha256(artifact / "vae.safetensors") == file_sha256(source)


def condition_checkpoint(
    checkpoint: Path, encoder_path: Path, mx: ModuleType
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    from sglang_omni.models.auk.mlx.conditioning import AuKMlxConditionEncoder
    from sglang_omni.models.auk.mlx.flow_matching import fuse_hidden_states
    from sglang_omni.models.auk.mlx.loader import load_fusion
    from sglang_omni.models.auk.reference_encode import build_messages

    encoder = AuKMlxConditionEncoder(
        str(encoder_path), dtype=mx.bfloat16, quantization="mlx_q8"
    )
    fusion = load_fusion(str(checkpoint))
    audio = (0.1 * np.sin(2 * np.pi * 220 * np.arange(8000) / 16000)).astype(np.float32)
    messages = [
        build_messages('Say "Welcome home." in a warm voice.', False),
        build_messages('Say "你好，欢迎回家。" with the same voice.', True),
    ]
    encodings = encoder.encode_batch(messages, [None, audio])
    return [
        (
            np.array(hidden),
            np.array(mask),
            np.array(fuse_hidden_states(hidden[None], *fusion)[0]),
        )
        for hidden, mask in encodings
    ]


def sample_checkpoint(
    checkpoint: Path,
    encodings: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    mx: ModuleType,
) -> list[np.ndarray]:
    from sglang_omni.models.auk.constants import FLASH_T_GRID
    from sglang_omni.models.auk.hf_config import make_runtime_config
    from sglang_omni.models.auk.mlx.flow_matching import AuKSampleItem
    from sglang_omni.models.auk.mlx.loader import load_flow

    config = make_runtime_config(str(checkpoint))
    flow = load_flow(str(checkpoint), mx.bfloat16, "mlx_q8")
    generator = np.random.default_rng(91)
    items = []
    for (_, mask, conditioning), frames, reference_frames in zip(
        encodings, (11, 7), (0, 3)
    ):
        reference = generator.standard_normal(
            (reference_frames, config.latent_dim)
        ).astype(np.float32)
        noise = generator.standard_normal((frames, config.latent_dim)).astype(
            np.float32
        )
        items.append(
            AuKSampleItem(
                conditioning=mx.array(conditioning),
                text_mask=mx.array(mask),
                target_frames=frames,
                ref_latent=mx.array(reference) if reference_frames else None,
                ref_length=reference_frames,
                noise=mx.array(noise),
            )
        )
    return [
        np.array(value)
        for value in flow.sample_batch(
            items,
            steps=4 if config.is_flash else 3,
            cfg_strength=0 if config.is_flash else 2,
            sway_sampling_coef=None if config.is_flash else -1,
            t_grid=FLASH_T_GRID if config.is_flash else None,
        )
    ]


def test_native_q8_artifact_matches_original_on_load_quantization(
    local_checkpoints: tuple[Path, Path, Path], metal_backend: ModuleType
) -> None:
    original, qwen, artifact = local_checkpoints
    mx = metal_backend
    expected = condition_checkpoint(original, qwen, mx)
    gc.collect()
    mx.clear_cache()
    actual = condition_checkpoint(artifact, artifact / "conditioner", mx)
    gc.collect()
    mx.clear_cache()
    assert len(actual) == len(expected) == 2
    for (hidden, mask, fused), (target_hidden, target_mask, target_fused) in zip(
        actual, expected
    ):
        np.testing.assert_array_equal(mask, target_mask)
        assert hidden.dtype == fused.dtype == np.float32
        assert np.isfinite(hidden).all() and np.isfinite(fused).all()
        np.testing.assert_allclose(hidden, target_hidden, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(fused, target_fused, rtol=1e-6, atol=1e-6)
    del actual
    target_latents = sample_checkpoint(original, expected, mx)
    gc.collect()
    mx.clear_cache()
    actual_latents = sample_checkpoint(artifact, expected, mx)
    gc.collect()
    mx.clear_cache()
    assert len(actual_latents) == len(target_latents) == 2
    for latent, target in zip(actual_latents, target_latents):
        assert latent.dtype == np.float32 and np.isfinite(latent).all()
        np.testing.assert_allclose(latent, target, rtol=1e-6, atol=1e-6)
