# SPDX-License-Identifier: Apache-2.0
"""Native artifact round trips and selective quantization contracts."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

mx = pytest.importorskip("mlx.core")

from sglang_omni.models.auk.hf_config import load_auk_config
from sglang_omni.models.auk.mlx import convert
from sglang_omni.models.auk.mlx.convert import convert_checkpoint
from sglang_omni.models.auk.mlx.quantization import QUANTIZATION, read_artifact_config


@pytest.fixture(autouse=True)
def cpu_stream():
    with mx.stream(mx.cpu):
        yield


@pytest.fixture
def raw_checkpoints(tmp_path):
    torch.manual_seed(19)
    source = tmp_path / "source"
    conditioner = tmp_path / "thinker"
    source.mkdir()
    conditioner.mkdir()
    (source / "config.yaml").write_text(
        "model:\n  name: AuK-Flash\n  arch:\n    dim: 64\n  schedule:\n    nfe: 4\n"
    )
    (conditioner / "config.json").write_text(
        json.dumps({"model_type": "qwen2_5_omni", "thinker_config": {}})
    )
    (conditioner / "tokenizer_config.json").write_text('{"padding_side":"left"}')
    (conditioner / "chat_template.jinja").write_text("synthetic-template")
    flow = {
        "transformer.txt_proj.weight": torch.randn(8, 64) / 4,
        "transformer.txt_proj.bias": torch.randn(8),
        "transformer.audio_embed.conv_pos_embed.conv1d.0.weight": torch.randn(4, 4, 3),
        "transformer.txt_norm.weight": torch.randn(8),
        "transformer.rotary_embed.inv_freq": torch.tensor([1.0, 0.1234567]),
        "layer_weights": torch.tensor([0.1234567, 0.7654321]),
        "layer_scale": torch.tensor([1.234567]),
    }
    thinker = {
        "thinker.model.embed_tokens.weight": torch.randn(13, 64) / 4,
        "thinker.model.layers.0.mlp.up_proj.weight": torch.randn(8, 64) / 4,
        "thinker.model.norm.weight": torch.randn(64),
        "thinker.audio_tower.proj.weight": torch.randn(4, 64) / 4,
        "thinker.audio_tower.conv1.weight": torch.randn(4, 4, 3),
        "thinker.audio_tower.audio_bos_eos_token.weight": torch.randn(2, 64),
        "thinker.visual.weight": torch.randn(4, 4),
    }
    save_file(flow, str(source / "auk_flash.safetensors"))
    save_file({"global_mean": torch.zeros(64)}, str(source / "vae.safetensors"))
    save_file(thinker, str(conditioner / "model.safetensors"))
    return source, conditioner, flow, thinker


def read_component(path: Path):
    index = json.loads((path / "model.safetensors.index.json").read_text())
    weights = {}
    for filename in sorted(set(index["weight_map"].values())):
        weights.update(mx.load(str(path / filename)))
    assert set(weights) == set(index["weight_map"])
    assert index["metadata"]["total_size"] == sum(v.nbytes for v in weights.values())
    return weights


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("quantization", [None, "mlx_q8"])
def test_artifacts_preserve_layout_scope_metadata_and_processor_assets(
    tmp_path, raw_checkpoints, dtype, quantization
):
    source, thinker_path, flow, thinker = raw_checkpoints
    output = convert_checkpoint(
        str(source),
        str(tmp_path / "converted"),
        text_encoder_path=str(thinker_path),
        dtype=dtype,
        quantization=quantization,
        shard_bytes=1024,
        chunk_bytes=512,
    )
    assert load_auk_config(str(output)).is_flash
    assert (output / "vae.safetensors").read_bytes() == (
        source / "vae.safetensors"
    ).read_bytes()
    assert (output / "config.yaml").read_bytes() == (
        source / "config.yaml"
    ).read_bytes()
    assert (
        output / "conditioner/chat_template.jinja"
    ).read_text() == "synthetic-template"
    assert (output / "conditioner/tokenizer_config.json").read_bytes() == (
        thinker_path / "tokenizer_config.json"
    ).read_bytes()
    native_flow = read_component(output)
    native_thinker = read_component(output / "conditioner")
    for component, path in (("flow", output), ("conditioner", output / "conditioner")):
        marker = read_artifact_config(path, component=component)
        assert marker.dtype == dtype
        assert (marker.quantization is None) is (quantization is None)
        assert marker.source["repo_or_path"] == str(
            source if component == "flow" else thinker_path
        )
    for name in ("layer_weights", "layer_scale", "transformer.rotary_embed.inv_freq"):
        assert native_flow[name].dtype == mx.float32
        np.testing.assert_array_equal(np.array(native_flow[name]), flow[name].numpy())
    audio_name = "audio_tower.proj.weight"
    expected_audio = mx.array(thinker[f"thinker.{audio_name}"].numpy()).astype(
        getattr(mx, dtype)
    )
    assert native_thinker[audio_name].dtype == getattr(mx, dtype)
    np.testing.assert_array_equal(
        np.array(native_thinker[audio_name].astype(mx.float32)),
        np.array(expected_audio.astype(mx.float32)),
    )
    assert not any(
        "audio_tower" in key and key.endswith("scales") for key in native_thinker
    )
    assert not any("audio_bos" in key or "visual" in key for key in native_thinker)
    conv_name = "transformer.audio_embed.conv_pos_embed.conv1d.0.weight"
    expected_conv = (
        mx.array(flow[conv_name].numpy()).astype(getattr(mx, dtype)).transpose(0, 2, 1)
    )
    np.testing.assert_array_equal(
        np.array(native_flow[conv_name].astype(mx.float32)),
        np.array(expected_conv.astype(mx.float32)),
    )
    for name, weights in (
        ("transformer.txt_proj", native_flow),
        ("model.embed_tokens", native_thinker),
        ("model.layers.0.mlp.up_proj", native_thinker),
    ):
        assert (f"{name}.scales" in weights) == bool(quantization)
        assert weights[f"{name}.weight"].dtype == (
            mx.uint32 if quantization else getattr(mx, dtype)
        )


def test_row_chunking_preserves_packed_weights_and_projection(
    tmp_path, raw_checkpoints
):
    source, thinker_path, flow, _ = raw_checkpoints
    weights = []
    for chunk_bytes in (256, 1024 * 1024):
        output = convert_checkpoint(
            str(source),
            str(tmp_path / str(chunk_bytes)),
            text_encoder_path=str(thinker_path),
            dtype="float32",
            quantization="mlx_q8",
            shard_bytes=4096,
            chunk_bytes=chunk_bytes,
        )
        weights.append(read_component(output))
    for key in weights[0]:
        np.testing.assert_array_equal(
            np.array(weights[0][key]), np.array(weights[1][key])
        )
    inputs = np.random.default_rng(3).normal(size=(2, 64)).astype(np.float32)
    packed = weights[0]
    actual = (
        mx.quantized_matmul(
            mx.array(inputs),
            packed["transformer.txt_proj.weight"],
            packed["transformer.txt_proj.scales"],
            packed["transformer.txt_proj.biases"],
            transpose=True,
            **QUANTIZATION,
        )
        + packed["transformer.txt_proj.bias"]
    )
    expected = torch.nn.functional.linear(
        torch.from_numpy(inputs),
        flow["transformer.txt_proj.weight"],
        flow["transformer.txt_proj.bias"],
    ).numpy()
    np.testing.assert_allclose(np.array(actual), expected, atol=0.03, rtol=0.005)


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", 2),
        ("format", "another-layout"),
        ("layout", "torch"),
        ("dtype", "float16"),
        ("quantization", {"bits": 4, "group_size": 64, "mode": "affine"}),
        ("quantization", {"bits": 8, "group_size": 128, "mode": "affine"}),
        ("vae_layout", "mlx"),
        ("source", {}),
    ],
)
def test_incompatible_metadata_is_rejected(tmp_path, field, value):
    marker = {
        "format": "sglang-omni-auk-mlx",
        "version": 1,
        "component": "flow",
        "layout": "mlx",
        "dtype": "bfloat16",
        "quantization": deepcopy(QUANTIZATION),
        "source": {"repo_or_path": "synthetic"},
        "vae_layout": "torch",
    }
    marker[field] = value
    (tmp_path / "config.json").write_text(json.dumps({"mlx_artifact": marker}))
    with pytest.raises(ValueError):
        read_artifact_config(tmp_path, component="flow")


def test_existing_destination_and_unsupported_quantization_are_rejected(tmp_path):
    common = dict(
        text_encoder_path="unused", dtype="float32", shard_bytes=1024, chunk_bytes=512
    )
    with pytest.raises(ValueError, match="mlx_q8"):
        convert_checkpoint(
            "unused", str(tmp_path / "new"), quantization="mlx_q4", **common
        )
    with pytest.raises(FileExistsError):
        convert_checkpoint("unused", str(tmp_path), quantization="mlx_q8", **common)


def test_failed_conversion_does_not_publish_partial_artifacts(
    tmp_path, raw_checkpoints
):
    source, thinker_path, _, thinker = raw_checkpoints
    thinker["thinker.model.embed_tokens.weight"] = torch.zeros(
        13, 64, dtype=torch.int32
    )
    save_file(thinker, str(thinker_path / "model.safetensors"))
    output = tmp_path / "incomplete"
    with pytest.raises(ValueError, match="original floating weights"):
        convert_checkpoint(
            str(source),
            str(output),
            text_encoder_path=str(thinker_path),
            dtype="float32",
            quantization="mlx_q8",
            shard_bytes=1024,
            chunk_bytes=512,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".incomplete-*"))


def test_native_floating_artifacts_cannot_be_converted_twice(tmp_path, raw_checkpoints):
    source, thinker_path, _, _ = raw_checkpoints
    kwargs = dict(dtype="float32", quantization=None, shard_bytes=4096, chunk_bytes=512)
    native = convert_checkpoint(
        str(source),
        str(tmp_path / "native"),
        text_encoder_path=str(thinker_path),
        **kwargs,
    )
    with pytest.raises(ValueError, match="original floating checkpoints"):
        convert_checkpoint(
            str(native),
            str(tmp_path / "twice"),
            text_encoder_path=str(native / "conditioner"),
            **kwargs,
        )


def test_readonly_tokenizer_assets_are_copied_successfully(tmp_path, raw_checkpoints):
    source, thinker_path, _, _ = raw_checkpoints
    tokenizer = thinker_path / "tokenizer.json"
    tokenizer.write_text('{"version":"1.0"}')
    assets = [tokenizer, thinker_path / "tokenizer_config.json"]
    for asset in assets:
        asset.chmod(0o444)
    output = convert_checkpoint(
        str(source),
        str(tmp_path / "readonly-assets"),
        text_encoder_path=str(thinker_path),
        dtype="float32",
        quantization="mlx_q8",
        shard_bytes=4096,
        chunk_bytes=512,
    )
    for asset in assets:
        assert (output / "conditioner" / asset.name).read_bytes() == asset.read_bytes()
        assert asset.stat().st_mode & 0o777 == 0o444
    assert read_artifact_config(output, component="flow") is not None


def test_asset_copy_failure_does_not_publish_partial_artifacts(
    tmp_path, raw_checkpoints, monkeypatch
):
    source, thinker_path, _, _ = raw_checkpoints
    copy = convert.shutil.copy2

    def fail_tokenizer_copy(source, destination, **kwargs):
        if Path(source).name == "tokenizer_config.json":
            raise PermissionError("Tokenizer copy denied")
        else:
            return copy(source, destination, **kwargs)

    monkeypatch.setattr(convert.shutil, "copy2", fail_tokenizer_copy)
    output = tmp_path / "copy-failure"
    with pytest.raises(PermissionError, match="Tokenizer copy denied"):
        convert_checkpoint(
            str(source),
            str(output),
            text_encoder_path=str(thinker_path),
            dtype="float32",
            quantization="mlx_q8",
            shard_bytes=4096,
            chunk_bytes=512,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".copy-failure-*"))


def test_license_notices_and_readonly_overlapping_assets_are_preserved(
    tmp_path, raw_checkpoints, monkeypatch
):
    source, thinker_path, _, _ = raw_checkpoints
    names = ("LICENSE", "LICENSE.txt", "LICENCE.md", "NOTICE", "NOTICE-tokenizer.json")
    for directory in (source, thinker_path):
        for name in names:
            file = directory / name
            file.write_text(f"{directory.name} {name}\n")
            file.chmod(0o444)
    original_copy = convert.shutil.copy2
    copies = []

    def copy_once(source, target, **kwargs):
        pair = (Path(source), Path(target))
        assert pair not in copies
        copies.append(pair)
        return original_copy(source, target, **kwargs)

    monkeypatch.setattr(convert.shutil, "copy2", copy_once)
    output = convert_checkpoint(
        str(source),
        str(tmp_path / "licensed"),
        text_encoder_path=str(thinker_path),
        dtype="bfloat16",
        quantization="mlx_q8",
        shard_bytes=4096,
        chunk_bytes=512,
    )
    for original, copied in ((source, output), (thinker_path, output / "conditioner")):
        for name in names:
            assert (copied / name).read_bytes() == (original / name).read_bytes()
            assert (original / name).stat().st_mode & 0o777 == 0o444


@pytest.mark.parametrize(
    "component,filename",
    [
        ("flow", "config.yaml"),
        ("flow", "auk_flash.safetensors"),
        ("flow", "vae.safetensors"),
        ("conditioner", "config.json"),
        ("conditioner", "model.safetensors"),
    ],
)
def test_source_bytes_are_fingerprinted_and_changes_update_provenance(
    tmp_path, raw_checkpoints, component, filename
):
    source, thinker_path, _, _ = raw_checkpoints
    paths = {
        "flow": (source, ["config.yaml", "auk_flash.safetensors", "vae.safetensors"]),
        "conditioner": (thinker_path, ["config.json", "model.safetensors"]),
    }
    metadata = []
    for iteration in range(2):
        if iteration:
            file = paths[component][0] / filename
            if file.suffix == ".safetensors":
                tensors = load_file(str(file))
                name = next(iter(tensors))
                tensors[name] = tensors[name] + 0.5
                save_file(tensors, str(file))
            else:
                file.write_text(file.read_text() + "\n")
        output = convert_checkpoint(
            str(source),
            str(tmp_path / f"fingerprinted-{iteration}"),
            text_encoder_path=str(thinker_path),
            dtype="float32",
            quantization="mlx_q8",
            shard_bytes=4096,
            chunk_bytes=512,
        )
        provenance = {}
        for kind, (directory, names) in paths.items():
            artifact = output if kind == "flow" else output / "conditioner"
            recorded = read_artifact_config(artifact, component=kind).source
            provenance[kind] = recorded
            for name in names:
                assert (
                    recorded[f"sha256:{name}"]
                    == hashlib.sha256((directory / name).read_bytes()).hexdigest()
                )
        metadata.append(provenance)
    key = f"sha256:{filename}"
    assert metadata[0][component][key] != metadata[1][component][key]
    for kind in paths:
        for field, value in metadata[0][kind].items():
            if (kind, field) != (component, key):
                assert metadata[1][kind][field] == value


def test_sharded_sources_record_only_used_files_and_preserve_hf_revision(
    tmp_path, raw_checkpoints, monkeypatch
):
    source, thinker_path, _, thinker = raw_checkpoints
    revision = "a" * 40
    snapshot = tmp_path / "snapshots" / revision
    snapshot.parent.mkdir()
    thinker_path.rename(snapshot)
    (snapshot / "model.safetensors").unlink()
    model = {
        name: value
        for name, value in thinker.items()
        if name.startswith("thinker.model.")
    }
    audio = {
        name: value
        for name, value in thinker.items()
        if name.startswith("thinker.audio_tower.")
    }
    unused = {
        name: value
        for name, value in thinker.items()
        if name.startswith("thinker.visual.")
    }
    mapping = {}
    for filename, tensors in (
        ("text.safetensors", model),
        ("audio.safetensors", audio),
        ("vision.safetensors", unused),
    ):
        save_file(tensors, str(snapshot / filename))
        mapping.update({name: filename for name in tensors})
    index = snapshot / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": mapping}))
    resolver = convert.resolve_checkpoint
    monkeypatch.setattr(
        convert,
        "resolve_checkpoint",
        lambda path: str(snapshot) if path == "synthetic/qwen" else resolver(path),
    )
    output = convert_checkpoint(
        str(source),
        str(tmp_path / "sharded-source"),
        text_encoder_path="synthetic/qwen",
        dtype="float32",
        quantization="mlx_q8",
        shard_bytes=4096,
        chunk_bytes=512,
    )
    provenance = read_artifact_config(
        output / "conditioner", component="conditioner"
    ).source
    assert provenance["repo_or_path"] == "synthetic/qwen"
    assert provenance["revision"] == revision
    for name in (
        "config.json",
        "model.safetensors.index.json",
        "text.safetensors",
        "audio.safetensors",
    ):
        assert (
            provenance[f"sha256:{name}"]
            == hashlib.sha256((snapshot / name).read_bytes()).hexdigest()
        )
    assert "sha256:vision.safetensors" not in provenance


def test_native_marker_in_source_yaml_is_rejected(tmp_path, raw_checkpoints):
    source, thinker_path, _, _ = raw_checkpoints
    config = source / "config.yaml"
    config.write_text(
        config.read_text() + "mlx_artifact: {format: sglang-omni-auk-mlx}\n"
    )
    output = tmp_path / "yaml-native"
    with pytest.raises(ValueError, match="original floating checkpoints"):
        convert_checkpoint(
            str(source),
            str(output),
            text_encoder_path=str(thinker_path),
            dtype="float32",
            quantization="mlx_q8",
            shard_bytes=4096,
            chunk_bytes=512,
        )
    assert not output.exists()
