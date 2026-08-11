#!/usr/bin/env python3
"""Strict static accounting for the Bonsai Q1_0 decode stream.

This module deliberately accounts only the bytes described by tensor metadata.
Observed DDR traffic is a separate measurement and must not be inferred from
the logical Q1 payload total.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from numbers import Integral
from pathlib import Path
import sys
import os
import tempfile
from types import MappingProxyType
from typing import Any


SCHEMA_VERSION = "q1-memory-accounting/v1"
_MODEL_ID = "prism-ml/Ternary-Bonsai-27B-gguf"
_MODEL_QUANTIZATION = "Q1_0"
_MODEL_SIZE_BYTES = 3_803_452_480
_MODEL_SHA256 = "17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0"
_MODEL_FILENAME = "Bonsai-27B-Q1_0.gguf"
_ARCHITECTURE = "qwen35"
_PINNED_TENSOR_RECORDS_SHA256 = "bda0c7f1e7e41355886584eedf412d60d6cec4a001db4bc6f1ac7565b3e796db"
_PAYLOAD_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


class Q1MemoryAccountingError(ValueError):
    """A Q1 tensor or accounting manifest violates the static contract."""


@dataclass(frozen=True)
class Q1Tensor:
    name: str
    index: int
    ggml_type: str
    shape: tuple[int, ...]
    source_offset_bytes: int
    size_bytes: int
    payload_sha256: str


def classify_decode_tensor(name: str) -> str | None:
    """Return the decode family for *name*, or ``None`` for embeddings.

    The classifier is intentionally allow-list based.  A new tensor name must
    be classified explicitly before it can enter the memory contract.
    """

    if name == "token_embd.weight":
        return None
    if name == "output.weight":
        return "lm_head"
    if re.fullmatch(r"blk\.\d+\.ffn_(gate|up|down)\.weight", name):
        return "ffn"
    if re.fullmatch(r"blk\.\d+\.(attn_qkv|attn_gate|ssm_(alpha|beta|out))\.weight", name):
        return "recurrent"
    if re.fullmatch(r"blk\.\d+\.attn_(q|k|v|output)\.weight", name):
        return "full_attention"
    raise Q1MemoryAccountingError(f"unclassified Q1 tensor: {name}")


def _integer(value: object, context: str, *, non_negative: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise Q1MemoryAccountingError(f"{context}: expected integer")
    if non_negative and value < 0:
        raise Q1MemoryAccountingError(f"{context}: expected non-negative integer")
    return int(value)


def _validate_tensor(tensor: Q1Tensor) -> None:
    if not isinstance(tensor.name, str) or not tensor.name:
        raise Q1MemoryAccountingError("tensor name must be a non-empty string")
    _integer(tensor.index, f"{tensor.name}.index", non_negative=True)
    if tensor.ggml_type != "Q1_0":
        raise Q1MemoryAccountingError(f"{tensor.name}: expected ggml_type Q1_0")
    if isinstance(tensor.shape, (str, bytes)) or not isinstance(tensor.shape, Sequence):
        raise Q1MemoryAccountingError(f"{tensor.name}: shape must be a sequence")
    if len(tensor.shape) != 2:
        raise Q1MemoryAccountingError(f"{tensor.name}: expected two-dimensional shape")
    ne0 = _integer(tensor.shape[0], f"{tensor.name}.shape[0]", non_negative=True)
    ne1 = _integer(tensor.shape[1], f"{tensor.name}.shape[1]", non_negative=True)
    if ne0 == 0 or ne1 == 0:
        raise Q1MemoryAccountingError(f"{tensor.name}: shape dimensions must be positive")
    if ne0 % 128 != 0:
        raise Q1MemoryAccountingError(f"{tensor.name}: ne0 must be divisible by 128")
    if ne1 % 16 != 0:
        raise Q1MemoryAccountingError(f"{tensor.name}: ne1 must be divisible by 16")
    _integer(
        tensor.source_offset_bytes, f"{tensor.name}.source_offset_bytes", non_negative=True
    )
    size = _integer(tensor.size_bytes, f"{tensor.name}.size_bytes", non_negative=True)
    expected_size = ne0 * ne1 // 128 * 18
    if size != expected_size:
        raise Q1MemoryAccountingError(
            f"{tensor.name}: size_bytes={size} does not match expected {expected_size}"
        )
    if not isinstance(tensor.payload_sha256, str) or _PAYLOAD_SHA256.fullmatch(tensor.payload_sha256) is None:
        raise Q1MemoryAccountingError(f"{tensor.name}: payload_sha256 must be 64 hex characters")
    # Run the strict allow-list check even for excluded embeddings so unknown
    # names can never be silently omitted from accounting.
    classify_decode_tensor(tensor.name)


def _as_tensor(value: object, position: int) -> Q1Tensor:
    if isinstance(value, Q1Tensor):
        return value
    if not isinstance(value, Mapping):
        raise Q1MemoryAccountingError(f"manifest tensor {position}: expected object")
    required = (
        "name", "index", "ggml_type", "shape", "source_offset_bytes",
        "size_bytes", "payload_sha256",
    )
    missing = [field for field in required if field not in value]
    if missing:
        raise Q1MemoryAccountingError(
            f"manifest tensor {position}: missing fields {', '.join(missing)}"
        )
    shape = value["shape"]
    if isinstance(shape, list):
        shape = tuple(shape)
    elif isinstance(shape, tuple):
        shape = tuple(shape)
    return Q1Tensor(
        name=value["name"],
        index=value["index"],
        ggml_type=value["ggml_type"],
        shape=shape,
        source_offset_bytes=value["source_offset_bytes"],
        size_bytes=value["size_bytes"],
        payload_sha256=value["payload_sha256"],
    )


def _manifest_tensors(manifest: object) -> tuple[Q1Tensor, ...]:
    if isinstance(manifest, Mapping):
        if "tensors" not in manifest:
            raise Q1MemoryAccountingError("manifest must contain a tensors field")
        manifest = manifest["tensors"]
    if isinstance(manifest, (str, bytes)) or not isinstance(manifest, Iterable):
        raise Q1MemoryAccountingError("manifest must be an iterable of Q1 tensors")
    return tuple(_as_tensor(value, position) for position, value in enumerate(manifest))


def account_decode_stream(tensors: Iterable[Q1Tensor | Mapping[str, Any]]) -> dict[str, Any]:
    """Validate tensors and return deterministic logical decode accounting."""

    inventory = _manifest_tensors(tensors)
    names: set[str] = set()
    indices: set[int] = set()
    families: dict[str, dict[str, int]] = {
        "ffn": {"logical_gemv_count": 0, "q1_bytes_per_token": 0},
        "recurrent": {"logical_gemv_count": 0, "q1_bytes_per_token": 0},
        "full_attention": {"logical_gemv_count": 0, "q1_bytes_per_token": 0},
        "lm_head": {"logical_gemv_count": 0, "q1_bytes_per_token": 0},
    }
    for tensor in inventory:
        _validate_tensor(tensor)
        if tensor.name in names:
            raise Q1MemoryAccountingError(f"duplicate tensor name: {tensor.name}")
        if tensor.index in indices:
            raise Q1MemoryAccountingError(f"duplicate tensor index: {tensor.index}")
        names.add(tensor.name)
        indices.add(tensor.index)
    _validate_payload_ranges(inventory)
    excluded_embedding_count = 0
    logical_gemv_count = 0
    decode_bytes = 0
    payload_bytes = 0
    for tensor in inventory:
        payload_bytes += tensor.size_bytes
        family = classify_decode_tensor(tensor.name)
        if family is None:
            excluded_embedding_count += 1
            continue
        logical_gemv_count += 1
        decode_bytes += tensor.size_bytes
        families[family]["logical_gemv_count"] += 1
        families[family]["q1_bytes_per_token"] += tensor.size_bytes

    return {
        "q1_tensor_count": len(inventory),
        "q1_payload_bytes": payload_bytes,
        "logical_gemv_count": logical_gemv_count,
        "excluded_embedding_count": excluded_embedding_count,
        "decode_q1_bytes_per_token": decode_bytes,
        "families": families,
    }


def _validate_payload_ranges(tensors: Sequence[Q1Tensor]) -> None:
    ranges = sorted(
        (tensor.source_offset_bytes, tensor.source_offset_bytes + tensor.size_bytes, tensor.name)
        for tensor in tensors
    )
    for previous, current in zip(ranges, ranges[1:]):
        if current[0] < previous[1]:
            raise Q1MemoryAccountingError(
                f"overlapping payload ranges: {previous[2]} and {current[2]}"
            )


_EXPECTED = MappingProxyType({
    "q1_tensor_count": 498,
    "q1_payload_bytes": 3_781_877_760,
    "logical_gemv_count": 497,
    "decode_q1_bytes_per_token": 3_603_087_360,
    "families": MappingProxyType({
        "ffn": MappingProxyType({"logical_gemv_count": 192, "q1_bytes_per_token": 2_406_481_920}),
        "recurrent": MappingProxyType({"logical_gemv_count": 240, "q1_bytes_per_token": 781_885_440}),
        "full_attention": MappingProxyType({"logical_gemv_count": 64, "q1_bytes_per_token": 235_929_600}),
        "lm_head": MappingProxyType({"logical_gemv_count": 1, "q1_bytes_per_token": 178_790_400}),
    }),
})

_REQUIRED_TOP_LEVEL = frozenset({
    "architecture", "architecture_metadata", "counts",
    "declared_decode_q1_bytes_per_token", "decode_q1_bytes_per_token",
    "excluded_embedding_count", "families", "f32_tensor_count",
    "logical_gemv_count", "model", "observed_ddr_bytes_per_token",
    "q1_payload_bytes", "q1_tensor_count", "schema_version", "sidecar",
    "tensors",
})
_REQUIRED_MODEL_FIELDS = frozenset({
    "architecture", "filename", "id", "quantization", "sha256", "size_bytes",
})
_REQUIRED_TENSOR_FIELDS = frozenset({
    "ggml_type", "index", "name", "ne0", "ne1", "payload_sha256", "shape",
    "sidecar_offset_bytes", "sidecar_payload_sha256", "size_bytes",
    "source_offset_bytes", "tile_bytes", "tile_k", "tile_m",
})
_REQUIRED_SIDECAR_FIELDS = frozenset({"alignment_bytes", "payload_sha256", "total_payload_bytes"})
_EXPECTED_ARCHITECTURE_METADATA = {
    "attention.head_count": 24,
    "attention.head_count_kv": 4,
    "attention.key_length": 256,
    "attention.layer_norm_rms_epsilon": 9.999999974752427e-07,
    "attention.value_length": 256,
    "block_count": 64,
    "context_length": 262144,
    "embedding_length": 5120,
    "feed_forward_length": 17408,
    "full_attention_interval": 4,
    "rope.dimension_count": 64,
    "rope.dimension_sections": [11, 11, 10, 0],
    "rope.freq_base": 10000000.0,
    "ssm.conv_kernel": 4,
    "ssm.group_count": 16,
    "ssm.inner_size": 6144,
    "ssm.state_size": 128,
    "ssm.time_step_rank": 48,
}


def _validate_manifest_identity(manifest: object) -> Mapping[str, Any]:
    if not isinstance(manifest, Mapping):
        raise Q1MemoryAccountingError("Bonsai manifest must be an object")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise Q1MemoryAccountingError(
            f"schema_version must be {SCHEMA_VERSION}"
        )
    model = manifest.get("model")
    if not isinstance(model, Mapping):
        raise Q1MemoryAccountingError("model identity must be an object")
    identity = (
        ("id", _MODEL_ID),
        ("quantization", _MODEL_QUANTIZATION),
        ("size_bytes", _MODEL_SIZE_BYTES),
        ("sha256", _MODEL_SHA256),
    )
    for field, expected in identity:
        if type(model.get(field)) is not type(expected) or model.get(field) != expected:
            raise Q1MemoryAccountingError(
                f"model.{field} must equal the pinned Bonsai identity"
            )
    return model


def validate_bonsai_manifest(manifest: object) -> dict[str, Any]:
    """Validate a manifest against the pinned Bonsai Q1 accounting totals."""

    _validate_manifest_identity(manifest)
    _validate_complete_manifest_contract(manifest)
    tensors = _manifest_tensors(manifest)
    result = account_decode_stream(tensors)
    for field, expected in _EXPECTED.items():
        if result[field] != expected:
            raise Q1MemoryAccountingError(
                f"Bonsai manifest mismatch for {field}: expected={expected!r} got={result[field]!r}"
            )
    _validate_manifest_summary(manifest, result)
    return result


def _assert_exact_json_value(got: object, expected: object, context: str) -> None:
    if type(got) is not type(expected):
        raise Q1MemoryAccountingError(
            f"{context}: expected JSON type {type(expected).__name__}"
        )
    if isinstance(expected, Mapping):
        if set(got) != set(expected):  # type: ignore[arg-type]
            raise Q1MemoryAccountingError(f"{context}: object fields mismatch")
        for key, expected_value in expected.items():
            _assert_exact_json_value(got[key], expected_value, f"{context}.{key}")  # type: ignore[index]
    elif isinstance(expected, list):
        if len(got) != len(expected):  # type: ignore[arg-type]
            raise Q1MemoryAccountingError(f"{context}: list length mismatch")
        for index, expected_value in enumerate(expected):
            _assert_exact_json_value(got[index], expected_value, f"{context}[{index}]")  # type: ignore[index]
    elif got != expected:
        raise Q1MemoryAccountingError(f"{context}: value mismatch")


def _require_builtin_int(value: object, context: str) -> int:
    if type(value) is not int:
        raise Q1MemoryAccountingError(f"{context}: expected JSON integer")
    return value


def _validate_complete_manifest_contract(manifest: Mapping[str, Any]) -> None:
    if set(manifest) != _REQUIRED_TOP_LEVEL:
        missing = sorted(_REQUIRED_TOP_LEVEL - set(manifest))
        extra = sorted(set(manifest) - _REQUIRED_TOP_LEVEL)
        raise Q1MemoryAccountingError(
            f"manifest top-level fields mismatch: missing={missing!r} extra={extra!r}"
        )
    model = manifest["model"]
    if not isinstance(model, Mapping) or set(model) != _REQUIRED_MODEL_FIELDS:
        raise Q1MemoryAccountingError("manifest model fields mismatch")
    _assert_exact_json_value(
        model,
        {
            "architecture": _ARCHITECTURE,
            "filename": _MODEL_FILENAME,
            "id": _MODEL_ID,
            "quantization": _MODEL_QUANTIZATION,
            "sha256": _MODEL_SHA256,
            "size_bytes": _MODEL_SIZE_BYTES,
        },
        "model",
    )
    if model["architecture"] != _ARCHITECTURE:
        raise Q1MemoryAccountingError("model.architecture must equal pinned architecture")
    if model["filename"] != _MODEL_FILENAME:
        raise Q1MemoryAccountingError("model.filename must equal pinned model filename")
    if manifest["architecture"] != _ARCHITECTURE:
        raise Q1MemoryAccountingError("architecture must equal pinned architecture")
    _assert_exact_json_value(
        manifest["architecture_metadata"], _EXPECTED_ARCHITECTURE_METADATA, "architecture_metadata"
    )
    counts = manifest["counts"]
    _assert_exact_json_value(counts, {"F32": 353, "Q1_0": 498}, "counts")
    sidecar = manifest["sidecar"]
    if not isinstance(sidecar, Mapping) or set(sidecar) != _REQUIRED_SIDECAR_FIELDS:
        raise Q1MemoryAccountingError("manifest sidecar fields mismatch")
    _require_builtin_int(sidecar["alignment_bytes"], "sidecar.alignment_bytes")
    _require_builtin_int(sidecar["total_payload_bytes"], "sidecar.total_payload_bytes")
    if sidecar["alignment_bytes"] != 64 or type(sidecar["payload_sha256"]) is not type(None):
        raise Q1MemoryAccountingError("sidecar must use 64-byte alignment and null hash")
    if manifest["observed_ddr_bytes_per_token"] is not None:
        raise Q1MemoryAccountingError("observed_ddr_bytes_per_token must be null")
    tensors = manifest["tensors"]
    if not isinstance(tensors, list) or not tensors or not all(isinstance(tensor, Mapping) for tensor in tensors):
        raise Q1MemoryAccountingError("manifest tensors must be a non-empty object list")
    previous_index = -1
    expected_sidecar_offset = 0
    for position, tensor in enumerate(tensors):
        if set(tensor) != _REQUIRED_TENSOR_FIELDS:
            raise Q1MemoryAccountingError(f"tensor {position}: fields mismatch")
        normalized = _as_tensor(tensor, position)
        _validate_tensor(normalized)
        if not isinstance(tensor["shape"], list):
            raise Q1MemoryAccountingError(f"tensor {position}: shape must be a JSON list")
        _require_builtin_int(tensor["index"], f"tensor {position}.index")
        _require_builtin_int(tensor["ne0"], f"tensor {position}.ne0")
        _require_builtin_int(tensor["ne1"], f"tensor {position}.ne1")
        _require_builtin_int(tensor["size_bytes"], f"tensor {position}.size_bytes")
        _require_builtin_int(tensor["source_offset_bytes"], f"tensor {position}.source_offset_bytes")
        for field in ("tile_m", "tile_k", "tile_bytes"):
            _require_builtin_int(tensor[field], f"tensor {position}.{field}")
        _require_builtin_int(tensor["sidecar_offset_bytes"], f"tensor {position}.sidecar_offset_bytes")
        if type(tensor["name"]) is not str or type(tensor["ggml_type"]) is not str:
            raise Q1MemoryAccountingError(f"tensor {position}: names/types must be JSON strings")
        if type(tensor["payload_sha256"]) is not str:
            raise Q1MemoryAccountingError(f"tensor {position}: payload hash must be a JSON string")
        if type(tensor["sidecar_payload_sha256"]) is not type(None):
            raise Q1MemoryAccountingError(f"tensor {position}: sidecar payload hash must be null")
        if tensor["index"] <= previous_index:
            raise Q1MemoryAccountingError("manifest tensors must be sorted by increasing index")
        previous_index = tensor["index"]
        if tensor["shape"] != [tensor["ne0"], tensor["ne1"]]:
            raise Q1MemoryAccountingError(f"tensor {position}: shape must equal [ne0, ne1]")
        if (tensor["tile_m"], tensor["tile_k"], tensor["tile_bytes"]) != (16, 128, 288):
            raise Q1MemoryAccountingError(f"tensor {position}: unsupported tile constants")
        sidecar_offset = tensor["sidecar_offset_bytes"]
        if sidecar_offset < 0:
            raise Q1MemoryAccountingError(f"tensor {position}: invalid sidecar offset")
        if sidecar_offset % 64 != 0 or sidecar_offset != _align_up(expected_sidecar_offset, 64):
            raise Q1MemoryAccountingError(f"tensor {position}: non-deterministic sidecar offset")
        if tensor["sidecar_payload_sha256"] is not None:
            raise Q1MemoryAccountingError(f"tensor {position}: sidecar payload hash must be null")
        expected_sidecar_offset = sidecar_offset + tensor["size_bytes"]
    if sidecar["total_payload_bytes"] != expected_sidecar_offset:
        raise Q1MemoryAccountingError("sidecar total_payload_bytes mismatch")
    canonical_records = json.dumps(
        tensors, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if hashlib.sha256(canonical_records).hexdigest() != _PINNED_TENSOR_RECORDS_SHA256:
        raise Q1MemoryAccountingError("tensor records do not match pinned canonical SHA-256")


def _validate_manifest_summary(manifest: Mapping[str, Any], result: Mapping[str, Any]) -> None:
    for field in (
        "q1_tensor_count", "q1_payload_bytes", "logical_gemv_count",
        "excluded_embedding_count", "decode_q1_bytes_per_token",
    ):
        _require_builtin_int(manifest[field], f"manifest.{field}")
        if manifest[field] != result[field]:
            raise Q1MemoryAccountingError(
                f"manifest {field} does not match derived accounting"
            )
    _assert_exact_json_value(manifest["families"], result["families"], "manifest.families")
    _require_builtin_int(
        manifest["declared_decode_q1_bytes_per_token"],
        "manifest.declared_decode_q1_bytes_per_token",
    )
    if manifest["declared_decode_q1_bytes_per_token"] != result["decode_q1_bytes_per_token"]:
        raise Q1MemoryAccountingError("declared decode bytes do not match derived accounting")
    _require_builtin_int(manifest["f32_tensor_count"], "manifest.f32_tensor_count")
    if manifest["f32_tensor_count"] != 353:
        raise Q1MemoryAccountingError("f32_tensor_count does not match pinned model")


def _reader_payload_bytes(tensor: object, n_bytes: int) -> bytes:
    """Return exactly the declared payload bytes from a GGUF reader tensor."""

    try:
        view = memoryview(getattr(tensor, "data"))
        try:
            view = view.cast("B")
        except TypeError:
            # A non-contiguous byte-like object still has a well-defined byte
            # representation through ``tobytes``.
            pass
        payload = view.tobytes()
    except (AttributeError, TypeError):
        try:
            payload = bytes(getattr(tensor, "data"))
        except (AttributeError, TypeError, ValueError) as exc:
            raise Q1MemoryAccountingError("reader tensor data must be byte-like") from exc
    if len(payload) < n_bytes:
        raise Q1MemoryAccountingError(
            f"reader tensor {getattr(tensor, 'name', '<unknown>')}: "
            f"payload is truncated ({len(payload)} < {n_bytes})"
        )
    return payload[:n_bytes]


def inventory_from_reader(reader: object) -> tuple[Q1Tensor, ...]:
    """Build Q1 inventory entries from a lazy ``GGUFReader`` instance.

    ``ReaderTensor.data_offset`` is the payload's file offset.  The associated
    ``ReaderField.offset`` points into the tensor-info header and is never a
    valid payload location.
    """

    reader_tensors = getattr(reader, "tensors", None)
    if reader_tensors is None:
        raise Q1MemoryAccountingError("reader must expose a tensors collection")
    inventory: list[Q1Tensor] = []
    for index, tensor in enumerate(reader_tensors):
        tensor_type = getattr(getattr(tensor, "tensor_type", None), "name", None)
        if tensor_type != "Q1_0":
            continue
        try:
            n_bytes = _integer(getattr(tensor, "n_bytes"), f"reader tensor {index}.n_bytes", non_negative=True)
            source_offset = _integer(
                getattr(tensor, "data_offset"),
                f"reader tensor {index}.data_offset",
                non_negative=True,
            )
            raw_name = getattr(tensor, "name")
            if not isinstance(raw_name, str) or not raw_name:
                raise Q1MemoryAccountingError(f"reader tensor {index}.name must be a non-empty string")
            raw_shape = getattr(tensor, "shape")
            if isinstance(raw_shape, (str, bytes)) or not isinstance(raw_shape, Iterable):
                raise Q1MemoryAccountingError(f"reader tensor {index}.shape must be a sequence")
            shape = tuple(
                _integer(dimension, f"reader tensor {index}.shape[{dimension_index}]", non_negative=True)
                for dimension_index, dimension in enumerate(raw_shape)
            )
            name = raw_name
        except Q1MemoryAccountingError:
            raise
        except (AttributeError, TypeError, ValueError) as exc:
            raise Q1MemoryAccountingError(f"reader tensor {index}: incomplete metadata") from exc
        payload = _reader_payload_bytes(tensor, n_bytes)
        inventory.append(
            Q1Tensor(
                name=name,
                index=index,
                ggml_type=tensor_type,
                shape=shape,
                source_offset_bytes=source_offset,
                size_bytes=n_bytes,
                payload_sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
    return tuple(sorted(inventory, key=lambda tensor: tensor.index))


def load_gguf_reader(model: Path, gguf_python_root: Path) -> object:
    """Load ``GGUFReader`` lazily so accounting unit tests need no gguf-py."""

    sys.path.insert(0, str(gguf_python_root))
    from gguf import GGUFReader

    return GGUFReader(model, "r")


def _stream_sha256(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def _json_value(value: object) -> object:
    """Convert numpy scalar/array metadata to standard JSON values."""

    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_value(item())
        except (TypeError, ValueError):
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _json_value(tolist())
        except (TypeError, ValueError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _reader_field_value(reader: object, key: str) -> object:
    fields = getattr(reader, "fields", {})
    field = fields.get(key) if hasattr(fields, "get") else None
    if field is None:
        return None
    contents = getattr(field, "contents", None)
    if not callable(contents):
        return None
    return _json_value(contents())


def _architecture_metadata(reader: object) -> tuple[str, dict[str, object]]:
    architecture_value = _reader_field_value(reader, "general.architecture")
    architecture = str(architecture_value)
    prefix = f"{architecture}."
    fields = getattr(reader, "fields", {})
    metadata: dict[str, object] = {}
    for key in fields:
        if key.startswith(prefix):
            metadata[key[len(prefix):]] = _reader_field_value(reader, key)
    return architecture, dict(sorted(metadata.items()))


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _manifest_from_reader(model: Path, reader: object, model_size: int, model_sha256: str) -> dict[str, object]:
    inventory = inventory_from_reader(reader)
    accounting = account_decode_stream(inventory)
    architecture, architecture_metadata = _architecture_metadata(reader)
    reader_tensors = getattr(reader, "tensors")
    type_counts: dict[str, int] = {}
    for tensor in reader_tensors:
        type_name = str(getattr(getattr(tensor, "tensor_type", None), "name", "<unknown>"))
        type_counts[type_name] = type_counts.get(type_name, 0) + 1

    sidecar_offset = 0
    tensor_records: list[dict[str, object]] = []
    for tensor in inventory:
        sidecar_offset = _align_up(sidecar_offset, 64)
        ne0, ne1 = tensor.shape
        tensor_records.append(
            {
                "ggml_type": tensor.ggml_type,
                "index": tensor.index,
                "name": tensor.name,
                "ne0": ne0,
                "ne1": ne1,
                "payload_sha256": tensor.payload_sha256,
                "shape": list(tensor.shape),
                "sidecar_offset_bytes": sidecar_offset,
                "sidecar_payload_sha256": None,
                "size_bytes": tensor.size_bytes,
                "source_offset_bytes": tensor.source_offset_bytes,
                "tile_bytes": 288,
                "tile_k": 128,
                "tile_m": 16,
            }
        )
        sidecar_offset += tensor.size_bytes

    return {
        "architecture": architecture,
        "architecture_metadata": architecture_metadata,
        "counts": {
            "F32": type_counts.get("F32", 0),
            "Q1_0": type_counts.get("Q1_0", 0),
        },
        "declared_decode_q1_bytes_per_token": accounting["decode_q1_bytes_per_token"],
        "excluded_embedding_count": accounting["excluded_embedding_count"],
        "families": accounting["families"],
        "f32_tensor_count": type_counts.get("F32", 0),
        "model": {
            "architecture": architecture,
            "filename": model.name,
            "id": _MODEL_ID,
            "quantization": _MODEL_QUANTIZATION,
            "sha256": model_sha256,
            "size_bytes": model_size,
        },
        "observed_ddr_bytes_per_token": None,
        "q1_payload_bytes": accounting["q1_payload_bytes"],
        "q1_tensor_count": accounting["q1_tensor_count"],
        "schema_version": SCHEMA_VERSION,
        "sidecar": {
            "alignment_bytes": 64,
            "payload_sha256": None,
            "total_payload_bytes": sidecar_offset,
        },
        "tensors": tensor_records,
        "logical_gemv_count": accounting["logical_gemv_count"],
        "decode_q1_bytes_per_token": accounting["decode_q1_bytes_per_token"],
    }


def _write_manifest(path: Path, manifest: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as destination:
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
        # A hard link is atomic and never replaces a path that won a race.
        # Both paths are in the same directory/filesystem by construction.
        os.link(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--model", type=Path)
    mode.add_argument("--check", type=Path)
    parser.add_argument("--gguf-python-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.check is not None:
        manifest = json.loads(args.check.read_text(encoding="utf-8"))
        validate_bonsai_manifest(manifest)
        print(f"PASS {SCHEMA_VERSION}")
        return 0
    if args.gguf_python_root is None or args.output is None:
        parser.error("--model requires --gguf-python-root and --output")
    model_size, model_sha256 = _stream_sha256(args.model)
    if model_size != _MODEL_SIZE_BYTES or model_sha256 != _MODEL_SHA256:
        raise Q1MemoryAccountingError(
            "model identity mismatch: expected pinned Bonsai size/SHA-256"
        )
    reader = load_gguf_reader(args.model, args.gguf_python_root)
    manifest = _manifest_from_reader(args.model, reader, model_size, model_sha256)
    validate_bonsai_manifest(manifest)
    _write_manifest(args.output, manifest)
    print(f"model_sha256={model_sha256}")
    print(f"q1_tensor_count={manifest['q1_tensor_count']}")
    print(f"logical_gemv_count={manifest['logical_gemv_count']}")
    print(f"decode_q1_bytes_per_token={manifest['decode_q1_bytes_per_token']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(_main())
    except Q1MemoryAccountingError as exc:
        raise SystemExit(f"ERROR: {exc}")
