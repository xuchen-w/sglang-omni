"""Torch CPU versus MLX Metal parity for AuK audio encoding and synthesis."""

import os
from unittest.mock import patch

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")

from sglang_omni.models.auk.hf_config import AuKVAEConfig
from sglang_omni.models.auk.mlx.flow_matching import request_key
from sglang_omni.models.auk.mlx.vae import BigVGANFlowVAE
from sglang_omni.models.auk.mlx.vae_layers import Activation1d
from sglang_omni.models.auk.vae import Activation1d as TorchActivation1d
from sglang_omni.models.auk.vae import BigVGANFlowVAE as TorchVAE
from sglang_omni.models.auk.vae import SnakeBeta as TorchSnakeBeta

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
# note (Codex): M5 Metal defaults to TF32; MLX_ENABLE_TF32=0 verifies strict FP32 parity.
RELATIVE_TOLERANCE, ABSOLUTE_TOLERANCE = (
    (2e-5, 2e-6) if os.environ.get("MLX_ENABLE_TF32") == "0" else (2e-3, 5e-4)
)


def make_models(
    causal: bool = True, act_causal: bool = True, remove_weight_norm: bool = False
) -> tuple[TorchVAE, BigVGANFlowVAE]:
    torch.manual_seed(31)
    config = AuKVAEConfig(
        upsample_rates=[3, 2],
        upsample_kernel_sizes=[6, 4],
        upsample_initial_channel=32,
        resblock_kernel_sizes=[3, 5],
        resblock_dilation_sizes=[[1, 3], [1, 3]],
        downsample_rates=[2, 3],
        downsample_channels=[4, 8, 16],
        latent_dim=4,
        flow_hidden_channels=8,
        causal=causal,
        act_causal=act_causal,
    )
    reference = TorchVAE(config).eval()
    reference.global_mean.copy_(torch.tensor([-0.3, 0.7, 0.2, -0.8]))
    reference.global_log_std.copy_(torch.tensor([0.5, 1.5, 0.8, 2.0]))
    if remove_weight_norm:
        reference.remove_weight_norm()
    model = BigVGANFlowVAE(config)
    weights = {
        name: mx.array(value.detach().numpy())
        for name, value in reference.state_dict().items()
    }
    model.load_weights(list(model.sanitize(weights).items()), strict=True)
    mx.eval(model.parameters())
    return reference, model


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("act_causal", [True, False])
@pytest.mark.parametrize("remove_weight_norm", [True, False])
def test_decoder_matches_torch_at_boundaries(
    causal: bool, act_causal: bool, remove_weight_norm: bool
) -> None:
    reference, model = make_models(causal, act_causal, remove_weight_norm)
    latents = np.random.default_rng(5).normal(size=(2, 4, 7)).astype(np.float32)
    latents[:, :, 0] = 3
    latents[:, :, -1] = -3
    with torch.no_grad():
        expected = reference.inference_from_latents(torch.from_numpy(latents)).numpy()
    with mx.stream(mx.gpu):
        actual = model.inference_from_latents(mx.array(latents))
        mx.eval(actual)
    assert actual.shape == expected.shape
    np.testing.assert_allclose(
        np.array(actual), expected, rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_TOLERANCE
    )


@pytest.mark.parametrize("samples", [54, 55, 59])
def test_encoder_posterior_and_partial_hop_lengths_match_torch(samples: int) -> None:
    reference, model = make_models()
    waveform = torch.from_numpy(
        np.random.default_rng(4).normal(size=(2, 1, samples)).astype(np.float32)
    )
    lengths = torch.tensor([samples, samples - 8])
    with torch.no_grad():
        stats = reference.audio_encoder(waveform)
        noise = torch.randn(
            (2, 4, stats.shape[-1]), generator=torch.Generator().manual_seed(9)
        )
        expected, expected_lengths = reference.encoding_and_normalization(
            waveform, lengths, generator=torch.Generator().manual_seed(9)
        )
    with mx.stream(mx.gpu):
        actual, actual_lengths = model.encode(
            mx.array(waveform.numpy()),
            mx.array(lengths.numpy()),
            noise=mx.array(noise.numpy()),
        )
        mx.eval(actual, actual_lengths)
    np.testing.assert_allclose(
        np.array(actual),
        expected.numpy(),
        rtol=RELATIVE_TOLERANCE,
        atol=ABSOLUTE_TOLERANCE,
    )
    np.testing.assert_array_equal(np.array(actual_lengths), expected_lengths.numpy())
    assert actual.shape[1] == stats.shape[-1]


def test_normalized_decode_matches_torch() -> None:
    reference, model = make_models()
    latents = np.random.default_rng(8).normal(size=(2, 5, 4)).astype(np.float32)
    with torch.no_grad():
        expected = reference.decode(torch.from_numpy(latents)).numpy()
    with mx.stream(mx.gpu):
        actual = model.decode(mx.array(latents))
        mx.eval(actual)
    np.testing.assert_allclose(
        np.array(actual), expected, rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_TOLERANCE
    )


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("snake_logscale", [True, False])
def test_alias_free_activation_matches_torch(
    causal: bool, snake_logscale: bool
) -> None:
    reference = TorchActivation1d(
        TorchSnakeBeta(3, alpha_logscale=snake_logscale), causal=causal
    )
    model = Activation1d(3, causal, snake_logscale)
    signal = np.random.default_rng(3).normal(size=(2, 9, 3)).astype(np.float32)
    signal[:, 0, :] = 2
    signal[:, -1, :] = -2
    with torch.no_grad():
        expected = reference(torch.from_numpy(signal).transpose(1, 2))
    with mx.stream(mx.gpu):
        actual = model(mx.array(signal))
        mx.eval(actual)
    np.testing.assert_allclose(
        np.array(actual),
        expected.transpose(1, 2).numpy(),
        rtol=RELATIVE_TOLERANCE,
        atol=ABSOLUTE_TOLERANCE,
    )


def test_request_key_is_reproducible_without_global_rng_changes() -> None:
    _, model = make_models()
    sample = mx.zeros((1, 1, 59))
    key = mx.random.key(17)
    first, lengths = model.encode(sample, key=key)
    mx.random.seed(999)
    second, _ = model.encode(sample, key=key)
    other, _ = model.encode(sample, key=mx.random.key(18))
    mx.eval(first, second, other, lengths)
    np.testing.assert_array_equal(np.array(first), np.array(second))
    assert not np.array_equal(np.array(first), np.array(other))
    assert lengths.tolist() == [59 // model.hop_size]


def test_bfloat16_inputs_preserve_float32_vae_computation() -> None:
    _, model = make_models()
    waveform = mx.array(
        np.random.default_rng(2).normal(size=(1, 1, 54)).astype(np.float32)
    ).astype(mx.bfloat16)
    key = mx.random.key(11)
    actual_latents, actual_lengths = model.encode(waveform, key=key)
    expected_latents, expected_lengths = model.encode(
        waveform.astype(mx.float32), key=key
    )
    latents = actual_latents.astype(mx.bfloat16)
    actual_audio = model.decode(latents)
    expected_audio = model.decode(latents.astype(mx.float32))
    mx.eval(actual_latents, expected_latents, actual_audio, expected_audio)
    assert actual_latents.dtype == mx.float32
    assert actual_audio.dtype == mx.float32
    np.testing.assert_array_equal(np.array(actual_latents), np.array(expected_latents))
    np.testing.assert_array_equal(np.array(actual_lengths), np.array(expected_lengths))
    np.testing.assert_array_equal(np.array(actual_audio), np.array(expected_audio))


@pytest.mark.parametrize("posterior_mode", ["sample", "mean"])
def test_posterior_policy_cpu_matches_torch_with_fixed_noise(posterior_mode):
    with mx.stream(mx.cpu):
        reference, model = make_models()
        waveform = np.random.default_rng(13).normal(size=(2, 1, 59)).astype(np.float32)
        lengths = torch.tensor([59, 51])
        with torch.no_grad():
            stats = reference.audio_encoder(torch.from_numpy(waveform))
            noise = torch.randn(
                (2, 4, stats.shape[-1]), generator=torch.Generator().manual_seed(23)
            )
            expected, expected_lengths = reference.encoding_and_normalization(
                torch.from_numpy(waveform),
                lengths,
                torch.Generator().manual_seed(23),
                posterior_mode=posterior_mode,
            )
            if posterior_mode == "sample":
                original, _ = reference.encoding_and_normalization(
                    torch.from_numpy(waveform),
                    lengths,
                    torch.Generator().manual_seed(23),
                )
                torch.testing.assert_close(expected, original, rtol=0, atol=0)
        actual, actual_lengths = model.encode(
            mx.array(waveform),
            mx.array(lengths.numpy()),
            noise=mx.array(noise.numpy()) if posterior_mode == "sample" else None,
            posterior_mode=posterior_mode,
        )
        np.testing.assert_allclose(
            np.array(actual), expected.numpy(), rtol=2e-5, atol=2e-6
        )
        np.testing.assert_array_equal(
            np.array(actual_lengths), expected_lengths.numpy()
        )
        if posterior_mode == "sample":
            original, _ = model.encode(
                mx.array(waveform), noise=mx.array(noise.numpy())
            )
            np.testing.assert_array_equal(np.array(actual), np.array(original))


def test_posterior_mean_cpu_does_not_draw_noise_or_consume_generator():
    with mx.stream(mx.cpu):
        reference, model = make_models()
        waveform = np.zeros((1, 1, 59), dtype=np.float32)
        generator = torch.Generator().manual_seed(99)
        generator_state = generator.get_state().clone()
        with (
            patch("torch.randn", side_effect=AssertionError("drew posterior noise")),
            patch(
                "mlx.core.random.normal",
                side_effect=AssertionError("drew posterior noise"),
            ),
            torch.no_grad(),
        ):
            expected, _ = reference.encoding_and_normalization(
                torch.from_numpy(waveform), generator=generator, posterior_mode="mean"
            )
            actual, _ = model.encode(
                mx.array(waveform), key=mx.random.key(99), posterior_mode="mean"
            )
            mx.eval(actual)
        torch.testing.assert_close(
            generator.get_state(), generator_state, rtol=0, atol=0
        )
        np.testing.assert_allclose(
            np.array(actual), expected.numpy(), rtol=2e-5, atol=2e-6
        )


def test_posterior_policy_cpu_rejects_invalid_values_before_encoding():
    with mx.stream(mx.cpu):
        reference, model = make_models()
        with patch.object(
            reference.audio_encoder, "forward", side_effect=AssertionError("encoded")
        ):
            with pytest.raises(ValueError, match="reference_encoding"):
                reference.encoding_and_normalization(
                    torch.zeros(1, 1, 59), posterior_mode="auto"
                )
        with patch.object(
            type(model.audio_encoder), "__call__", side_effect=AssertionError("encoded")
        ):
            with pytest.raises(ValueError, match="reference_encoding"):
                model.encode(mx.zeros((1, 1, 59)), posterior_mode="auto")
            with pytest.raises(ValueError, match="only supported in sample mode"):
                model.encode(
                    mx.zeros((1, 1, 59)),
                    posterior_mode="mean",
                    noise=mx.zeros((1, 4, 9)),
                )


@pytest.mark.parametrize("posterior_mode", ["sample", "mean"])
@pytest.mark.parametrize("seed", [None, 17])
def test_posterior_policy_cpu_preserves_mlx_rng_streams(posterior_mode, seed):
    with mx.stream(mx.cpu):
        _, model = make_models()
        waveform = mx.zeros((1, 1, 59))
        target_key = request_key(seed, stream="generation")
        expected_target = mx.random.normal((7, 4), key=target_key)
        mx.random.seed(99)
        expected_global = mx.random.normal((8,))
        mx.eval(expected_target, expected_global)
        mx.random.seed(99)
        latent, _ = model.encode(
            waveform,
            key=(
                request_key(seed, stream="reference")
                if posterior_mode == "sample"
                else None
            ),
            posterior_mode=posterior_mode,
        )
        actual_target = mx.random.normal((7, 4), key=target_key)
        actual_global = mx.random.normal((8,))
        np.testing.assert_array_equal(
            np.array(actual_target), np.array(expected_target)
        )
        np.testing.assert_array_equal(
            np.array(actual_global), np.array(expected_global)
        )
        if seed is not None or posterior_mode == "mean":
            model.encode(waveform[:, :, :54], key=request_key(39, stream="reference"))
            repeated, _ = model.encode(
                waveform,
                key=(
                    request_key(seed, stream="reference")
                    if posterior_mode == "sample"
                    else None
                ),
                posterior_mode=posterior_mode,
            )
            np.testing.assert_array_equal(np.array(latent), np.array(repeated))
