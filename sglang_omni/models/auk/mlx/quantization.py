# SPDX-License-Identifier: Apache-2.0
"""Selective AuK weight quantization and native artifact metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import mlx.core as mx
import mlx.nn as nn
import torch
from pydantic import BaseModel, ConfigDict, Field

from sglang_omni.models.auk import constants as C
from sglang_omni.models.auk.hf_config import Quantization
from sglang_omni.models.auk.weight_loader import strip_prefix

ARTIFACT_FORMAT = "sglang-omni-auk-mlx"
ARTIFACT_VERSION = 1
Component = Literal["flow", "conditioner"]

CONDITIONER_DIRNAME = "conditioner"
CONDITIONER_PREFIXES = ("thinker.model.", "thinker.audio_tower.")
CONDITIONER_SKIPPED = ("audio_bos_eos_token.weight", "rotary_emb.inv_freq")
CONV_SUFFIXES = {
    "flow": ("conv_pos_embed.conv1d.0.weight", "conv_pos_embed.conv1d.2.weight"),
    "conditioner": ("audio_tower.conv1.weight", "audio_tower.conv2.weight"),
}


class QuantizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bits: Literal[8]
    group_size: Literal[64]
    mode: Literal["affine"]


QUANTIZATION = QuantizationConfig(bits=8, group_size=64, mode="affine").model_dump()


class ArtifactConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: Literal["sglang-omni-auk-mlx"]
    version: Literal[1]
    component: Component
    layout: Literal["mlx"]
    dtype: Literal["bfloat16", "float32"]
    quantization: QuantizationConfig | None
    source: dict[str, str] = Field(min_length=1)
    vae_layout: Literal["torch"] | None = None


def read_artifact_config(
    path: str | Path, *, component: Component
) -> ArtifactConfig | None:
    """Return a validated native marker, or None for an original checkpoint."""
    config_path = Path(path) / "config.json"
    marker = (
        json.loads(config_path.read_text()).get("mlx_artifact")
        if config_path.is_file()
        else None
    )
    if marker is None:
        return None
    else:
        artifact = ArtifactConfig.model_validate(marker)
        if artifact.component != component:
            raise ValueError(
                f"Expected {component} artifact, found {artifact.component}"
            )
        elif component == "flow" and artifact.vae_layout != "torch":
            raise ValueError("AuK MLX artifacts require a raw Torch VAE")
        else:
            return artifact


def source_name(original: str, *, component: Component) -> str | None:
    """Map an original checkpoint key to its native name, or None to skip it."""
    if component == "conditioner":
        if original.startswith(CONDITIONER_PREFIXES) and not original.endswith(
            CONDITIONER_SKIPPED
        ):
            return original.removeprefix("thinker.")
        else:
            return None
    else:
        name = strip_prefix(original)
        return None if name.startswith("text_encoder.") else name


def preserves_float32(name: str, *, component: Component) -> bool:
    """Hidden-state fusion and rotary frequencies stay FP32 at every precision."""
    return component == "flow" and (
        name in C.FUSION_PARAMETERS or name.endswith("rotary_embed.inv_freq")
    )


def is_quantizable(name: str, shape: tuple[int, ...], *, component: Component) -> bool:
    """Select large text and DiT matrices, retaining audio and fusion precision."""
    if not name.endswith(".weight") or len(shape) != 2:
        return False
    elif shape[-1] % QUANTIZATION["group_size"]:
        return False
    elif component == "flow":
        return name.startswith("transformer.")
    else:
        return name == "model.embed_tokens.weight" or name.startswith("model.layers.")


def quantize_model(model: nn.Module, *, component: Component) -> None:
    """Replace selected module skeletons before loading packed parameters."""
    nn.quantize(
        model,
        **QUANTIZATION,
        class_predicate=lambda name, module: isinstance(
            module, (nn.Linear, nn.Embedding)
        )
        and is_quantizable(f"{name}.weight", module.weight.shape, component=component),
    )


def layer_dtype(
    layer: nn.Linear | nn.Embedding | nn.QuantizedLinear | nn.QuantizedEmbedding,
) -> mx.Dtype:
    """Activation dtype of a layer that may hold packed weights."""
    if isinstance(layer, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
        return layer.scales.dtype
    else:
        return layer.weight.dtype


def prepare_weight(
    name: str,
    tensor: torch.Tensor,
    *,
    dtype: mx.Dtype,
    component: Component,
    quantization: Quantization | None,
) -> dict[str, mx.array]:
    """Convert one original tensor or row slice into native floating/q8 weights."""
    if not tensor.is_floating_point():
        raise ValueError(f"Expected original floating weights for {name}")
    else:
        value = mx.array(tensor.float().numpy()).astype(
            mx.float32 if preserves_float32(name, component=component) else dtype
        )
        if name.endswith(CONV_SUFFIXES[component]):
            value = value.transpose(0, 2, 1)
        if quantization is None or not is_quantizable(
            name, tensor.shape, component=component
        ):
            return {name: value}
        else:
            weight, scales, biases = mx.quantize(value, **QUANTIZATION)
            prefix = name.removesuffix(".weight")
            return {
                name: weight,
                f"{prefix}.scales": scales,
                f"{prefix}.biases": biases,
            }
