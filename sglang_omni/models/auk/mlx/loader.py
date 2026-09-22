# SPDX-License-Identifier: Apache-2.0
"""Load official AuK checkpoints into native MLX components."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import mlx.core as mx
from safetensors import safe_open
from sglang.srt.environ import envs

from sglang_omni.models.auk import constants as C
from sglang_omni.models.auk.hf_config import (
    AuKDitConfig,
    AuKVAEConfig,
    Quantization,
    make_runtime_config,
    validate_quantization,
)
from sglang_omni.models.auk.mlx.dit import AuKDit
from sglang_omni.models.auk.mlx.flow_matching import AuKFlowMatching
from sglang_omni.models.auk.mlx.quantization import (
    QUANTIZATION,
    Component,
    is_quantizable,
    prepare_weight,
    preserves_float32,
    quantize_model,
    read_artifact_config,
    source_name,
)
from sglang_omni.models.auk.mlx.vae import BigVGANFlowVAE
from sglang_omni.models.auk.weight_loader import (
    normalize_state_dict,
    resolve_vae_file,
    resolve_weight_file,
    strip_prefix,
)
from sglang_omni.platforms import current_platform
from sglang_omni.utils.device import resolve_concrete_device

MLX_DTYPES = {
    "float32": mx.float32,
    "bfloat16": mx.bfloat16,
}


def validate_device(device: str | None, gpu_id: int | None) -> None:
    resolved = resolve_concrete_device(device, gpu_id)
    if not current_platform.is_mps() or resolved.type != "mps":
        raise ValueError("AuK native MLX requires an Apple Metal device")
    elif resolved.index not in (None, 0) or not mx.metal.is_available():
        raise ValueError("AuK native MLX requires the available Apple Metal device 0")
    else:
        cache_limit_gb = envs.SGLANG_MLX_CACHE_LIMIT_GB.get()
        if cache_limit_gb is None:
            return
        elif cache_limit_gb < 0:
            raise ValueError(
                f"SGLANG_MLX_CACHE_LIMIT_GB must be >= 0, got {cache_limit_gb}"
            )
        else:
            mx.set_cache_limit(int(cache_limit_gb * (1024**3)))


def resolve_dtype(name: str) -> mx.Dtype:
    try:
        return MLX_DTYPES[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported AuK MLX dtype: {name!r}") from exc


def checkpoint_files(
    path: Path, *, keep: Callable[[str], bool] | None = None
) -> list[Path]:
    """Shards holding every parameter, or only those whose name is kept."""
    index = path / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
        return sorted(
            {
                path / filename
                for name, filename in weight_map.items()
                if keep is None or keep(name)
            }
        )
    else:
        return [path / "model.safetensors"]


def original_files(path: Path, *, component: Component) -> list[Path]:
    """Original shards holding parameters that belong to the component."""
    return checkpoint_files(
        path, keep=lambda name: source_name(name, component=component) is not None
    )


def load_source_weights(
    files: list[Path],
    *,
    component: Component,
    dtype: mx.Dtype,
    quantization: Quantization | None,
) -> dict[str, mx.array]:
    """Convert an original checkpoint one tensor at a time to bound memory."""
    weights: dict[str, mx.array] = {}
    for file in files:
        with safe_open(str(file), framework="pt", device="cpu") as checkpoint:
            for original in checkpoint.keys():
                name = source_name(original, component=component)
                if name is not None:
                    converted = prepare_weight(
                        name,
                        checkpoint.get_tensor(original),
                        dtype=dtype,
                        component=component,
                        quantization=quantization,
                    )
                    mx.eval(converted)
                    weights.update(converted)
    return weights


def load_native_weights(
    path: str | Path,
    *,
    component: Component,
    dtype: mx.Dtype,
    quantization: Quantization | None,
) -> dict[str, mx.array] | None:
    """Load pre-converted weights, or None when path holds an original checkpoint."""
    validate_quantization(quantization)
    marker = read_artifact_config(path, component=component)
    if marker is None:
        return None
    elif MLX_DTYPES[marker.dtype] != dtype:
        raise ValueError(
            f"Native AuK artifact dtype {marker.dtype} does not match requested {dtype}"
        )
    elif (marker.quantization is None) != (quantization is None):
        raise ValueError(
            "Native AuK artifact quantization does not match the explicit quantization setting"
        )
    else:
        weights = {}
        for file in checkpoint_files(Path(path)):
            shard = mx.load(str(file))
            for name, value in shard.items():
                if value.dtype == mx.uint32:
                    if value.ndim != 2:
                        raise ValueError(f"Packed AuK tensor must be a matrix: {name}")
                    packed = (
                        value.shape[0],
                        value.shape[-1] * (32 // QUANTIZATION["bits"]),
                    )
                    if quantization is None or not is_quantizable(
                        name, packed, component=component
                    ):
                        raise ValueError(
                            f"Packed tensor is outside the AuK quantization scope: {name}"
                        )
                else:
                    expected = (
                        mx.float32
                        if preserves_float32(name, component=component)
                        else dtype
                    )
                    if value.dtype != expected:
                        raise ValueError(
                            f"Native AuK tensor {name} must use {expected}, found {value.dtype}"
                        )
            mx.eval(shard)
            weights.update(shard)
        return weights


def load_component_weights(
    checkpoint: str | Path,
    *,
    component: Component,
    dtype: mx.Dtype,
    quantization: Quantization | None,
) -> dict[str, mx.array]:
    """Read pre-converted weights, converting the original shards otherwise."""
    weights = load_native_weights(
        checkpoint, component=component, dtype=dtype, quantization=quantization
    )
    if weights is not None:
        return weights
    else:
        files = (
            [resolve_weight_file(str(checkpoint))]
            if component == "flow"
            else original_files(Path(checkpoint), component=component)
        )
        return load_source_weights(
            files, component=component, dtype=dtype, quantization=quantization
        )


def load_fusion(checkpoint: str) -> tuple[mx.array, mx.array]:
    """Read the FP32 hidden-state fusion weights without building the flow model."""
    files = (
        checkpoint_files(Path(checkpoint), keep=C.FUSION_PARAMETERS.__contains__)
        if read_artifact_config(checkpoint, component="flow") is not None
        else [resolve_weight_file(checkpoint)]
    )
    loaded = {}
    for file in files:
        with safe_open(str(file), framework="pt") as weights:
            for key in weights.keys():
                name = strip_prefix(key)
                if name in C.FUSION_PARAMETERS:
                    loaded[name] = mx.array(weights.get_tensor(key).float().numpy())
    values = tuple(loaded[name] for name in C.FUSION_PARAMETERS)
    mx.eval(values)
    return values


@lru_cache(maxsize=1)
def load_vae(checkpoint: str) -> BigVGANFlowVAE:
    # note (Claude): Native artifacts must promise a raw Torch VAE to be read here.
    read_artifact_config(checkpoint, component="flow")
    config = make_runtime_config(checkpoint)
    path = resolve_vae_file(checkpoint)
    if path is None:
        raise FileNotFoundError(f"No AuK VAE weights found under {checkpoint}")
    else:
        model = BigVGANFlowVAE(AuKVAEConfig.from_dict(config.vae_init_kwargs))
        weights = normalize_state_dict(mx.load(str(path)))
        model.load_weights(list(model.sanitize(weights).items()), strict=True)
        del weights
        model.eval()
        mx.eval(model.parameters())
        return model


def load_flow(
    checkpoint: str, dtype: mx.Dtype, quantization: Quantization | None = None
) -> AuKFlowMatching:
    validate_quantization(quantization)
    if dtype not in MLX_DTYPES.values():
        raise ValueError("AuK MLX weights require bfloat16 or float32")
    config = make_runtime_config(checkpoint)
    arch = asdict(AuKDitConfig.from_dict(config.arch))
    # note (Codex): The native backbone excludes training-only hyperparameters.
    arch.pop("depth")
    arch.pop("dropout")
    arch["latent_dim"] = config.latent_dim
    weights = load_component_weights(
        checkpoint, component="flow", dtype=dtype, quantization=quantization
    )
    flow = AuKFlowMatching(AuKDit(**arch), num_llm_layers=weights["layer_weights"].size)
    if quantization is not None:
        quantize_model(flow, component="flow")
    flow.load_weights(list(weights.items()), strict=True)
    del weights
    flow.eval()
    mx.eval(flow.parameters())
    return flow
