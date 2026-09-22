# SPDX-License-Identifier: Apache-2.0
"""Convert original AuK checkpoints into bounded-memory native MLX artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import mlx.core as mx
from safetensors import safe_open

from sglang_omni.models.auk.constants import DEFAULT_TEXT_ENCODER
from sglang_omni.models.auk.hf_config import (
    CONFIG_YAML_NAMES,
    Quantization,
    load_yaml,
    validate_quantization,
)
from sglang_omni.models.auk.mlx.loader import MLX_DTYPES, original_files
from sglang_omni.models.auk.mlx.quantization import (
    ARTIFACT_FORMAT,
    ARTIFACT_VERSION,
    CONDITIONER_DIRNAME,
    QUANTIZATION,
    ArtifactConfig,
    Component,
    prepare_weight,
    read_artifact_config,
    source_name,
)
from sglang_omni.models.auk.weight_loader import resolve_vae_file, resolve_weight_file
from sglang_omni.utils.checkpoint import resolve_checkpoint

MIB = 1024**2
PROCESSOR_ASSETS = (
    "tokenizer*",
    "*token*.json",
    "vocab.*",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "chat_template*",
    "generation_config.json",
)
LICENSE_ASSETS = ("LICENSE*", "LICENCE*", "NOTICE*")


def write_component(
    files: list[Path],
    destination: Path,
    *,
    component: Component,
    dtype: mx.Dtype,
    quantization: Quantization | None,
    shard_bytes: int,
    chunk_bytes: int,
) -> None:
    """Convert one tensor at a time, retaining at most one output shard."""
    destination.mkdir(parents=True, exist_ok=True)
    weights: dict[str, mx.array] = {}
    weight_map: dict[str, str] = {}
    shards: list[Path] = []
    buffered_bytes = total_bytes = 0

    def flush() -> None:
        nonlocal buffered_bytes
        filename = destination / f"model-{len(shards) + 1:05d}.safetensors"
        mx.save_safetensors(str(filename), weights, metadata={"format": "mlx"})
        weight_map.update({name: filename.name for name in weights})
        shards.append(filename)
        weights.clear()
        buffered_bytes = 0

    for filename in files:
        with safe_open(filename, framework="pt", device="cpu") as checkpoint:
            for original_name in checkpoint.keys():
                name = source_name(original_name, component=component)
                if name is None:
                    continue
                tensor_slice = checkpoint.get_slice(original_name)
                shape = tensor_slice.get_shape()
                if len(shape) == 2 and shape[0] * shape[1] * 4 > chunk_bytes:
                    # note (Codex): Row slices preserve groups along the last axis.
                    rows = max(1, chunk_bytes // (shape[1] * 4))
                    pieces: dict[str, list[mx.array]] = {}
                    for start in range(0, shape[0], rows):
                        prepared = prepare_weight(
                            name,
                            tensor_slice[start : start + rows],
                            dtype=dtype,
                            component=component,
                            quantization=quantization,
                        )
                        mx.eval(prepared)
                        for key, value in prepared.items():
                            pieces.setdefault(key, []).append(value)
                    converted = {
                        key: mx.concatenate(values, axis=0)
                        for key, values in pieces.items()
                    }
                    mx.eval(converted)
                    del pieces, prepared
                else:
                    converted = prepare_weight(
                        name,
                        checkpoint.get_tensor(original_name),
                        dtype=dtype,
                        component=component,
                        quantization=quantization,
                    )
                    mx.eval(converted)
                for key, value in converted.items():
                    if key in weight_map or key in weights:
                        raise ValueError(f"Duplicate converted parameter {key}")
                    if weights and buffered_bytes + value.nbytes > shard_bytes:
                        flush()
                    weights[key] = value
                    buffered_bytes += value.nbytes
                    total_bytes += value.nbytes
                del converted
    if weights:
        flush()
    if not shards:
        raise ValueError(f"No {component} parameters found in the source checkpoint")
    else:
        names = {}
        for index, filename in enumerate(shards, start=1):
            name = (
                "model.safetensors"
                if len(shards) == 1
                else f"model-{index:05d}-of-{len(shards):05d}.safetensors"
            )
            filename.rename(destination / name)
            names[filename.name] = name
        manifest = {
            "metadata": {"total_size": total_bytes},
            "weight_map": {
                key: names[filename] for key, filename in sorted(weight_map.items())
            },
        }
        (destination / "model.safetensors.index.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )


def artifact_metadata(
    source: str,
    path: Path,
    *,
    component: Component,
    dtype: str,
    quantization: Quantization | None,
    fingerprinted: list[Path],
) -> dict[str, object]:
    provenance = {"repo_or_path": source}
    if path.parent.name == "snapshots":
        provenance["revision"] = path.name
    for filename in sorted(set(fingerprinted)):
        digest = hashlib.sha256()
        with filename.open("rb") as stream:
            while chunk := stream.read(MIB):
                digest.update(chunk)
        provenance[f"sha256:{filename.relative_to(path).as_posix()}"] = (
            digest.hexdigest()
        )
    marker = {
        "format": ARTIFACT_FORMAT,
        "version": ARTIFACT_VERSION,
        "component": component,
        "layout": "mlx",
        "dtype": dtype,
        "quantization": QUANTIZATION if quantization else None,
        "source": provenance,
    }
    if component == "flow":
        marker["vae_layout"] = "torch"
    ArtifactConfig.model_validate(marker)
    return marker


def convert_checkpoint(
    model_path: str,
    output_path: str,
    *,
    text_encoder_path: str,
    dtype: str,
    quantization: Quantization | None,
    shard_bytes: int,
    chunk_bytes: int,
) -> Path:
    """Publish a self-contained artifact only after every component is written."""
    validate_quantization(quantization)
    if dtype not in MLX_DTYPES:
        raise ValueError("AuK MLX artifacts require bfloat16 or float32")
    elif shard_bytes < 1 or chunk_bytes < 1:
        raise ValueError("Shard and conversion chunk sizes must be positive")
    else:
        output = Path(output_path).absolute()
        if output.exists():
            raise FileExistsError(f"Output already exists: {output}")
        source = Path(resolve_checkpoint(model_path))
        conditioner = Path(resolve_checkpoint(text_encoder_path))
        if (
            read_artifact_config(source, component="flow") is not None
            or read_artifact_config(conditioner, component="conditioner") is not None
        ):
            raise ValueError("Conversion requires original floating checkpoints")
        vae_file = resolve_vae_file(str(source))
        if vae_file is None:
            raise FileNotFoundError(f"No VAE weights found under {source}")
        raw_config = next(
            (source / name for name in CONFIG_YAML_NAMES if (source / name).is_file()),
            source / "config.json",
        )
        config = load_yaml(raw_config)
        thinker_config = json.loads((conditioner / "config.json").read_text())
        if "mlx_artifact" in config or "mlx_artifact" in thinker_config:
            raise ValueError("Conversion requires original floating checkpoints")
        index = conditioner / "model.safetensors.index.json"
        thinker_files = original_files(conditioner, component="conditioner")
        flow_file = resolve_weight_file(str(source))
        config["architectures"] = ["AuKForConditionalGeneration"]
        config["mlx_artifact"] = artifact_metadata(
            model_path,
            source,
            component="flow",
            dtype=dtype,
            quantization=quantization,
            fingerprinted=[raw_config, flow_file, vae_file],
        )
        thinker_config["mlx_artifact"] = artifact_metadata(
            text_encoder_path,
            conditioner,
            component="conditioner",
            dtype=dtype,
            quantization=quantization,
            fingerprinted=[
                conditioner / "config.json",
                *thinker_files,
                *([index] if index.is_file() else []),
            ],
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temp:
            temporary = Path(temp) / "artifact"
            for files, target, component in (
                ([flow_file], temporary, "flow"),
                (thinker_files, temporary / CONDITIONER_DIRNAME, "conditioner"),
            ):
                write_component(
                    files,
                    target,
                    component=component,
                    dtype=MLX_DTYPES[dtype],
                    quantization=quantization,
                    shard_bytes=shard_bytes,
                    chunk_bytes=chunk_bytes,
                )
            shutil.copy2(vae_file, temporary / "vae.safetensors")
            for root, target, patterns in (
                (source, temporary, LICENSE_ASSETS),
                (
                    conditioner,
                    temporary / CONDITIONER_DIRNAME,
                    PROCESSOR_ASSETS + LICENSE_ASSETS,
                ),
            ):
                assets = {asset for pattern in patterns for asset in root.glob(pattern)}
                for asset in sorted(assets):
                    if asset.is_file():
                        shutil.copy2(asset, target / asset.name)
                    elif asset.is_dir():
                        shutil.copytree(asset, target / asset.name, dirs_exist_ok=True)
            if raw_config.suffix in (".yaml", ".yml"):
                shutil.copy2(raw_config, temporary / raw_config.name)
            (temporary / "config.json").write_text(json.dumps(config, indent=2) + "\n")
            (temporary / CONDITIONER_DIRNAME / "config.json").write_text(
                json.dumps(thinker_config, indent=2) + "\n"
            )
            temporary.rename(output)
        return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--text-encoder-path", default=DEFAULT_TEXT_ENCODER)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--quantization", choices=["mlx_q8"])
    parser.add_argument("--shard-size-mb", type=int, default=256)
    parser.add_argument("--chunk-size-mb", type=int, default=32)
    args = parser.parse_args()
    output = convert_checkpoint(
        args.model_path,
        args.output_path,
        text_encoder_path=args.text_encoder_path,
        dtype=args.dtype,
        quantization=args.quantization,
        shard_bytes=args.shard_size_mb * MIB,
        chunk_bytes=args.chunk_size_mb * MIB,
    )
    print(f"Wrote AuK MLX artifact to {output}")


if __name__ == "__main__":
    main()
