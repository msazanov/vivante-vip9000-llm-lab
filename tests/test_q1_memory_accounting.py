import unittest
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from tooling.q1_memory_accounting import (
    Q1MemoryAccountingError,
    Q1Tensor,
    _validate_manifest_identity,
    account_decode_stream,
    classify_decode_tensor,
    inventory_from_reader,
    _align_up,
    _write_manifest,
    validate_bonsai_manifest,
)


MODEL_SHA256 = "17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0"
MODEL_SIZE_BYTES = 3_803_452_480


def bonsai_manifest(tensors: tuple[Q1Tensor, ...] | list[Q1Tensor]) -> dict[str, object]:
    return {
        "schema_version": "q1-memory-accounting/v1",
        "model": {
            "id": "prism-ml/Ternary-Bonsai-27B-gguf",
            "sha256": MODEL_SHA256,
            "size_bytes": MODEL_SIZE_BYTES,
            "quantization": "Q1_0",
        },
        "tensors": tensors,
    }


def q1(name: str, index: int, ne0: int, ne1: int) -> Q1Tensor:
    return Q1Tensor(
        name=name,
        index=index,
        ggml_type="Q1_0",
        shape=(ne0, ne1),
        source_offset_bytes=(0, 1_000_000_000, 2_000_000_000, 3_000_000_000, 4_000_000_000)[index],
        size_bytes=ne0 * ne1 // 128 * 18,
        payload_sha256=f"{index + 1:064x}",
    )


class MemoryAccountingTests(unittest.TestCase):
    def test_reader_uses_payload_offset(self) -> None:
        class FakeField:
            offset = 999_999

        class FakeReaderTensor:
            def __init__(self, name: str, offset: int, payload: bytes) -> None:
                self.name = name
                self.tensor_type = type("TensorType", (), {"name": "Q1_0"})()
                self.shape = (128, 16)
                self.n_bytes = len(payload)
                self.data_offset = offset
                self.data = bytearray(payload + b"trailing bytes not in payload")
                self.field = FakeField()

        payloads = (
            FakeReaderTensor("blk.0.ffn_gate.weight", 4096, bytes(range(36))),
            FakeReaderTensor("output.weight", 8192, bytes(reversed(range(36)))),
        )

        inventory = inventory_from_reader(type("Reader", (), {"tensors": payloads})())

        self.assertEqual([tensor.index for tensor in inventory], [0, 1])
        self.assertEqual([tensor.source_offset_bytes for tensor in inventory], [4096, 8192])
        self.assertEqual(
            [tensor.payload_sha256 for tensor in inventory],
            [
                "5d7e2d9b1dcbc85e7c890036a2cf2f9fe7b66554f2df08cec6aa9c0a25c99c21",
                "ca792194a1c1862fcacc2becdacd04bbabc4574495f6806549ce61491ce496b1",
            ],
        )

    def test_reader_rejects_coerced_dimensions_and_names(self) -> None:
        class TensorType:
            name = "Q1_0"

        def reader_tensor(*, name: object, shape: object) -> object:
            return SimpleNamespace(
                name=name,
                tensor_type=TensorType(),
                shape=shape,
                n_bytes=36,
                data_offset=4096,
                data=bytes(36),
            )

        for tensor in (
            reader_tensor(name="output.weight", shape=(128.9, 16)),
            reader_tensor(name="output.weight", shape=(True, 16)),
            reader_tensor(name=123, shape=(128, 16)),
        ):
            with self.subTest(tensor=tensor):
                with self.assertRaises(Q1MemoryAccountingError):
                    inventory_from_reader(SimpleNamespace(tensors=(tensor,)))

    def test_manifest_validator_rejects_missing_mutated_and_reordered_contract(self) -> None:
        manifest = json.loads(
            (Path(__file__).parents[1] / "benchmarks/workloads/bonsai-27b-q1.json").read_text()
        )
        required_top_level = (
            "architecture", "architecture_metadata", "counts",
            "declared_decode_q1_bytes_per_token", "decode_q1_bytes_per_token",
            "excluded_embedding_count", "families", "f32_tensor_count",
            "logical_gemv_count", "model", "observed_ddr_bytes_per_token",
            "q1_payload_bytes", "q1_tensor_count", "schema_version", "sidecar",
            "tensors",
        )
        for field in required_top_level:
            candidate = json.loads(json.dumps(manifest))
            del candidate[field]
            with self.subTest(missing=field):
                with self.assertRaises(Q1MemoryAccountingError):
                    validate_bonsai_manifest(candidate)

        for field, value in (
            ("architecture", "wrong"),
            ("logical_gemv_count", 496),
            ("declared_decode_q1_bytes_per_token", 0),
            ("f32_tensor_count", 0),
        ):
            candidate = json.loads(json.dumps(manifest))
            candidate[field] = value
            with self.subTest(mutated=field):
                with self.assertRaises(Q1MemoryAccountingError):
                    validate_bonsai_manifest(candidate)

        for path, value in (
            (("model", "filename"), "other.gguf"),
            (("model", "architecture"), "wrong"),
            (("architecture_metadata", "block_count"), 63),
            (("counts", "F32"), 0),
            (("families", "ffn"), {"logical_gemv_count": 0, "q1_bytes_per_token": 0}),
            (("sidecar", "alignment_bytes"), 32),
            (("sidecar", "total_payload_bytes"), 0),
        ):
            candidate = json.loads(json.dumps(manifest))
            candidate[path[0]][path[1]] = value
            with self.subTest(mutated_path=".".join(path)):
                with self.assertRaises(Q1MemoryAccountingError):
                    validate_bonsai_manifest(candidate)

        for field in ("name", "index", "ggml_type", "shape", "ne0", "ne1", "size_bytes", "source_offset_bytes", "payload_sha256", "tile_m", "tile_k", "tile_bytes", "sidecar_offset_bytes", "sidecar_payload_sha256"):
            candidate = json.loads(json.dumps(manifest))
            del candidate["tensors"][0][field]
            with self.subTest(missing_tensor_field=field):
                with self.assertRaises(Q1MemoryAccountingError):
                    validate_bonsai_manifest(candidate)

        candidate = json.loads(json.dumps(manifest))
        candidate["tensors"][0]["shape"] = [16, 128]
        with self.assertRaises(Q1MemoryAccountingError):
            validate_bonsai_manifest(candidate)
        candidate = json.loads(json.dumps(manifest))
        candidate["tensors"][0]["tile_bytes"] = 1
        with self.assertRaises(Q1MemoryAccountingError):
            validate_bonsai_manifest(candidate)
        candidate = json.loads(json.dumps(manifest))
        candidate["tensors"][0]["sidecar_offset_bytes"] = 1
        with self.assertRaises(Q1MemoryAccountingError):
            validate_bonsai_manifest(candidate)
        candidate = json.loads(json.dumps(manifest))
        candidate["tensors"][0]["sidecar_payload_sha256"] = "0" * 64
        with self.assertRaises(Q1MemoryAccountingError):
            validate_bonsai_manifest(candidate)
        candidate = json.loads(json.dumps(manifest))
        candidate["tensors"] = list(reversed(candidate["tensors"]))
        with self.assertRaises(Q1MemoryAccountingError):
            validate_bonsai_manifest(candidate)
        candidate = json.loads(json.dumps(manifest))
        candidate["tensors"] = tuple(candidate["tensors"])
        with self.assertRaises(Q1MemoryAccountingError):
            validate_bonsai_manifest(candidate)
        for field, value in (
            ("source_offset_bytes", 1),
            ("payload_sha256", "0" * 64),
        ):
            candidate = json.loads(json.dumps(manifest))
            candidate["tensors"][0][field] = value
            with self.subTest(mutated_tensor_field=field):
                with self.assertRaises(Q1MemoryAccountingError):
                    validate_bonsai_manifest(candidate)
        for path, value in (
            (("q1_tensor_count",), 498.0),
            (("f32_tensor_count",), 353.0),
            (("logical_gemv_count",), 497.0),
            (("counts", "F32"), 353.0),
            (("model", "size_bytes"), 3803452480.0),
            (("architecture_metadata", "block_count"), 64.0),
            (("sidecar", "alignment_bytes"), 64.0),
            (("sidecar", "total_payload_bytes"), 3781877760.0),
            (("tensors", 0, "tile_bytes"), 288.0),
        ):
            candidate = json.loads(json.dumps(manifest))
            target = candidate
            for component in path[:-1]:
                target = target[component]
            target[path[-1]] = value
            with self.subTest(non_exact_type=".".join(str(part) for part in path)):
                with self.assertRaises(Q1MemoryAccountingError):
                    validate_bonsai_manifest(candidate)
        self.assertEqual(_align_up(289, 64), 320)

    def test_public_manifest_validator_has_no_legacy_fixture_bypass(self) -> None:
        tensors = [
            q1("output.weight", 0, 128, 16),
        ]
        with self.assertRaisesRegex(Q1MemoryAccountingError, "top-level"):
            validate_bonsai_manifest(bonsai_manifest(tensors))

    def test_manifest_writer_refuses_existing_output_without_overwrite(self) -> None:
        manifest = json.loads(
            (Path(__file__).parents[1] / "benchmarks/workloads/bonsai-27b-q1.json").read_text()
        )
        with TemporaryDirectory() as directory:
            output = Path(directory) / "manifest.json"
            output.write_bytes(b"sentinel\n")
            with self.assertRaises(FileExistsError):
                _write_manifest(output, manifest)
            self.assertEqual(output.read_bytes(), b"sentinel\n")

    def test_manifest_writer_write_failure_leaves_no_final_or_temp_file(self) -> None:
        manifest = json.loads(
            (Path(__file__).parents[1] / "benchmarks/workloads/bonsai-27b-q1.json").read_text()
        )
        with TemporaryDirectory() as directory:
            output = Path(directory) / "manifest.json"
            with patch("tooling.q1_memory_accounting.os.fsync", side_effect=OSError("injected fsync failure")):
                with self.assertRaises(OSError):
                    _write_manifest(output, manifest)
            self.assertFalse(output.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_classifies_decode_families_and_excludes_embedding(self) -> None:
        tensors = (
            q1("blk.0.ffn_gate.weight", 0, 5120, 17408),
            q1("blk.0.attn_qkv.weight", 1, 5120, 10240),
            q1("blk.3.attn_q.weight", 2, 5120, 12288),
            q1("output.weight", 3, 5120, 248320),
            q1("token_embd.weight", 4, 5120, 248320),
        )
        result = account_decode_stream(tensors)
        self.assertEqual(result["logical_gemv_count"], 4)
        self.assertEqual(result["excluded_embedding_count"], 1)
        self.assertEqual(
            sum(family["q1_bytes_per_token"] for family in result["families"].values()),
            result["decode_q1_bytes_per_token"],
        )

    def test_classification_is_strict_and_excludes_only_embedding(self) -> None:
        self.assertIsNone(classify_decode_tensor("token_embd.weight"))
        self.assertEqual(classify_decode_tensor("output.weight"), "lm_head")
        self.assertEqual(classify_decode_tensor("blk.12.ffn_down.weight"), "ffn")
        self.assertEqual(classify_decode_tensor("blk.12.ssm_beta.weight"), "recurrent")
        self.assertEqual(classify_decode_tensor("blk.12.attn_output.weight"), "full_attention")
        with self.assertRaises(Q1MemoryAccountingError):
            classify_decode_tensor("blk.12.ffn_gate.bias")

    def test_rejects_invalid_tensor_metadata_and_overlapping_payloads(self) -> None:
        valid = q1("output.weight", 0, 128, 16)
        invalid = (
            replace(valid, ggml_type="Q8_0"),
            replace(valid, shape=(127, 16)),
            replace(valid, shape=(128, 15)),
            replace(valid, size_bytes=1),
            replace(valid, payload_sha256="not-a-hash"),
            replace(valid, source_offset_bytes=-1),
        )
        for tensor in invalid:
            with self.subTest(tensor=tensor):
                with self.assertRaises(Q1MemoryAccountingError):
                    account_decode_stream((tensor,))
        with self.assertRaises(Q1MemoryAccountingError):
            account_decode_stream((valid, replace(valid, name="token_embd.weight", index=0)))
        with self.assertRaises(Q1MemoryAccountingError):
            validate_bonsai_manifest((valid, replace(valid, name="token_embd.weight", index=1)))

    def test_validates_the_pinned_bonsai_totals(self) -> None:
        tensors: list[Q1Tensor] = []
        index = 0
        offset = 0
        family_specs = (
            ("ffn", 192, ("gate", "up", "down"), 8_355_649),
            ("recurrent", 240, ("alpha", "beta", "out", "alpha"), 2_714_641),
            ("full_attention", 64, ("q", "k", "v", "output"), 819_137),
        )
        # Use unique synthetic block numbers while preserving each classifier family.
        for family, count, labels, final_factor in family_specs:
            for ordinal in range(count):
                prefix = "ffn_" if family == "ffn" else "ssm_" if family == "recurrent" else "attn_"
                name = f"blk.{ordinal}.{prefix}{labels[ordinal % len(labels)]}.weight"
                factor = 1 if ordinal < count - 1 else final_factor
                size = factor * 288
                tensors.append(Q1Tensor(name, index, "Q1_0", (128, 16 * factor), offset, size, f"{index + 1:064x}"))
                index += 1
                offset += size
        lm_head_size = 178_790_400
        tensors.append(Q1Tensor("output.weight", index, "Q1_0", (128, 16 * (lm_head_size // 288)), offset, lm_head_size, f"{index + 1:064x}"))
        index += 1
        offset += lm_head_size
        embedding_size = 178_790_400
        tensors.append(Q1Tensor("token_embd.weight", index, "Q1_0", (128, 16 * (embedding_size // 288)), offset, embedding_size, f"{index + 1:064x}"))

        result = account_decode_stream(tensors)
        self.assertEqual(
            result,
            {
                "q1_tensor_count": 498,
                "q1_payload_bytes": 3_781_877_760,
                "logical_gemv_count": 497,
                "excluded_embedding_count": 1,
                "decode_q1_bytes_per_token": 3_603_087_360,
                "families": {
                    "ffn": {"logical_gemv_count": 192, "q1_bytes_per_token": 2_406_481_920},
                    "recurrent": {"logical_gemv_count": 240, "q1_bytes_per_token": 781_885_440},
                    "full_attention": {"logical_gemv_count": 64, "q1_bytes_per_token": 235_929_600},
                    "lm_head": {"logical_gemv_count": 1, "q1_bytes_per_token": 178_790_400},
                },
            },
        )

        with self.assertRaises(Q1MemoryAccountingError):
            validate_bonsai_manifest(bonsai_manifest(tensors[:-1]))

    def test_manifest_requires_exact_schema_and_pinned_model_identity(self) -> None:
        tensors = (q1("output.weight", 0, 128, 16),)
        valid = bonsai_manifest(list(tensors))
        candidates = []
        wrong_schema = dict(valid)
        wrong_schema["schema_version"] = "q1-memory-accounting/v0"
        candidates.append(("schema_version", wrong_schema))
        wrong_size = dict(valid)
        wrong_size["model"] = {**valid["model"], "size_bytes": MODEL_SIZE_BYTES + 1}
        candidates.append(("model.size_bytes", wrong_size))
        wrong_hash = dict(valid)
        wrong_hash["model"] = {**valid["model"], "sha256": "0" * 64}
        candidates.append(("model.sha256", wrong_hash))
        for field, candidate in candidates:
            with self.subTest(field=field):
                with self.assertRaisesRegex(Q1MemoryAccountingError, field.split(".")[0]):
                    _validate_manifest_identity(candidate)

    def test_example_schema_keeps_declared_and_observed_memory_separate(self) -> None:
        example = json.loads((Path(__file__).parents[1] / "benchmarks/schema/q1-memory-accounting.example.json").read_text())
        self.assertEqual(example["schema_version"], "q1-memory-accounting/v1")
        self.assertEqual(example["model"]["id"], "prism-ml/Ternary-Bonsai-27B-gguf")
        self.assertEqual(example["model"]["quantization"], "Q1_0")
        self.assertEqual(example["model"]["size_bytes"], MODEL_SIZE_BYTES)
        self.assertEqual(example["model"]["sha256"], MODEL_SHA256)
        self.assertEqual(example["q1_tensor_count"], 498)
        self.assertEqual(example["q1_payload_bytes"], 3_781_877_760)
        self.assertEqual(example["logical_gemv_count"], 497)
        self.assertEqual(example["excluded_embedding_count"], 1)
        self.assertEqual(example["decode_q1_bytes_per_token"], 3_603_087_360)
        self.assertEqual(example["declared_decode_q1_bytes_per_token"], 3_603_087_360)
        self.assertIsInstance(example["declared_decode_q1_bytes_per_token"], int)
        self.assertIsNone(example["observed_ddr_bytes_per_token"])
        self.assertEqual(
            example["families"],
            {
                "ffn": {"logical_gemv_count": 192, "q1_bytes_per_token": 2_406_481_920},
                "recurrent": {"logical_gemv_count": 240, "q1_bytes_per_token": 781_885_440},
                "full_attention": {"logical_gemv_count": 64, "q1_bytes_per_token": 235_929_600},
                "lm_head": {"logical_gemv_count": 1, "q1_bytes_per_token": 178_790_400},
            },
        )
