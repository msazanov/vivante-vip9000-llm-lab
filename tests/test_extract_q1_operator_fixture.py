from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import struct
import sys
import tempfile
import unittest
from unittest import mock

import tooling.extract_q1_operator_fixture as extractor
from tooling.extract_q1_operator_fixture import FixtureError, extract_tensor


ROOT = Path(__file__).resolve().parents[1]


class ExtractQ1OperatorFixtureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.work = Path(self.temp_dir.name)
        self.model = self.work / "model.gguf"
        self.prefix = b"header-not-payload"
        self.payload = bytes(range(256)) + b"q1-payload" + b"-" * 22
        self.suffix = b"trailing-data"
        self.model.write_bytes(self.prefix + self.payload + self.suffix)
        self.tensor = {
            "name": "blk.0.ffn_gate.weight",
            "index": 7,
            "ggml_type": "Q1_0",
            "shape": [128, 16],
            "source_offset_bytes": len(self.prefix),
            "size_bytes": len(self.payload),
            "payload_sha256": hashlib.sha256(self.payload).hexdigest(),
        }

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _model_sha(self) -> str:
        return hashlib.sha256(self.model.read_bytes()).hexdigest()

    def test_extracts_exact_pread_slice_and_manifest_hashes(self) -> None:
        output = self.work / "fixture"
        result = extract_tensor(
            self.model,
            self.tensor,
            output,
            expected_model_sha256=self._model_sha(),
            seed=0x733,
        )

        self.assertEqual((output / "weights.q1_0.bin").read_bytes(), self.payload)
        activation = (output / "activation.f32.bin").read_bytes()
        self.assertEqual(len(activation), 128 * 4)
        self.assertEqual(result["schema"], "q1-cpu-operator-fixture/v1")
        manifest = json.loads((output / "fixture.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["model"]["sha256"], self._model_sha())
        self.assertEqual(manifest["tensor"]["sha256"], hashlib.sha256(self.payload).hexdigest())
        self.assertEqual(manifest["tensor"]["size_bytes"], len(self.payload))
        self.assertEqual(manifest["activation"]["size_bytes"], len(activation))
        self.assertEqual(
            manifest["activation"]["sha256"], hashlib.sha256(activation).hexdigest()
        )

    def test_activation_is_code_owned_deterministic_and_bounded(self) -> None:
        first = self.work / "first"
        second = self.work / "second"
        extract_tensor(self.model, self.tensor, first, expected_model_sha256=self._model_sha(), seed=91)
        extract_tensor(self.model, self.tensor, second, expected_model_sha256=self._model_sha(), seed=91)
        self.assertEqual(
            (first / "activation.f32.bin").read_bytes(),
            (second / "activation.f32.bin").read_bytes(),
        )
        self.assertNotEqual(
            (first / "activation.f32.bin").read_bytes(),
            (self.work / "model.gguf").read_bytes(),
        )
        values = struct.unpack("<128f", (first / "activation.f32.bin").read_bytes())
        self.assertTrue(all(math.isfinite(value) and -1.0 <= value <= 1.0 for value in values))
        self.assertEqual(
            values[:3],
            (-0.9890313744544983, -0.13624773919582367, 0.8659814596176147),
        )
        third = self.work / "third"
        extract_tensor(self.model, self.tensor, third, expected_model_sha256=self._model_sha(), seed=92)
        self.assertNotEqual(
            (first / "activation.f32.bin").read_bytes(),
            (third / "activation.f32.bin").read_bytes(),
        )

    def test_rejects_model_sha_mismatch(self) -> None:
        with self.assertRaises(FixtureError):
            extract_tensor(self.model, self.tensor, self.work / "fixture", expected_model_sha256="0" * 64)

    def test_rejects_same_size_model_mutation_between_hash_and_payload_read(self) -> None:
        output = self.work / "mutated"
        original_pread = extractor.os.pread
        mutated = False

        def mutate_before_pread(fd: int, size: int, offset: int) -> bytes:
            nonlocal mutated
            if not mutated:
                mutated = True
                current = self.model.read_bytes()
                self.model.write_bytes(current[:-1] + bytes([current[-1] ^ 1]))
            return original_pread(fd, size, offset)

        with mock.patch.object(extractor.os, "pread", side_effect=mutate_before_pread):
            with self.assertRaises(FixtureError):
                extract_tensor(self.model, self.tensor, output, expected_model_sha256=self._model_sha())
        self.assertFalse(output.exists())

    def test_rejects_wrong_shape_or_type(self) -> None:
        for field, value in (("shape", [129, 16]), ("ggml_type", "F32")):
            tensor = dict(self.tensor)
            tensor[field] = value
            with self.subTest(field=field), self.assertRaises(FixtureError):
                extract_tensor(self.model, tensor, self.work / field, expected_model_sha256=self._model_sha())

    def test_rejects_truncated_exact_pread(self) -> None:
        tensor = dict(self.tensor)
        tensor["source_offset_bytes"] = len(self.model.read_bytes()) - 10
        with self.assertRaises(FixtureError):
            extract_tensor(self.model, tensor, self.work / "truncated", expected_model_sha256=self._model_sha())

    def test_rejects_existing_output_file_without_overwrite(self) -> None:
        output = self.work / "fixture"
        output.mkdir()
        existing = output / "weights.q1_0.bin"
        existing.write_bytes(b"keep-me")
        with self.assertRaises(FixtureError):
            extract_tensor(self.model, self.tensor, output, expected_model_sha256=self._model_sha())
        self.assertEqual(existing.read_bytes(), b"keep-me")

    def test_rejects_existing_output_directory(self) -> None:
        output = self.work / "fixture"
        output.mkdir()
        with self.assertRaises(FixtureError):
            extract_tensor(self.model, self.tensor, output, expected_model_sha256=self._model_sha())

    def test_rejects_output_path_inside_repository_after_resolution(self) -> None:
        link = self.work / "repo-link"
        link.symlink_to(ROOT, target_is_directory=True)
        with self.assertRaises(FixtureError):
            extract_tensor(
                self.model,
                self.tensor,
                link / ".task3-fixture",
                expected_model_sha256=self._model_sha(),
            )

    def test_rejects_direct_repository_output_and_cleans_created_directory(self) -> None:
        output = ROOT / ".task3-direct-policy-regression"
        self.assertFalse(output.exists())
        with self.assertRaises(FixtureError):
            extract_tensor(self.model, self.tensor, output, expected_model_sha256=self._model_sha())
        self.assertFalse(output.exists())

    def test_rejects_rename_into_repository_before_first_publication(self) -> None:
        base = Path(tempfile.mkdtemp(prefix="task3-rename-", dir="/tmp"))
        output = base / "fixture"
        target = ROOT / ".task3-rename-boundary-regression"
        self.assertFalse(target.exists())
        real_publish = extractor._exclusive_write_at
        renamed = False

        def rename_before_first_write(directory_fd: int, name: str, payload: bytes) -> None:
            nonlocal renamed
            if not renamed:
                renamed = True
                output.rename(target)
            real_publish(directory_fd, name, payload)

        try:
            with mock.patch.object(extractor, "_exclusive_write_at", side_effect=rename_before_first_write):
                with self.assertRaises(FixtureError):
                    extract_tensor(self.model, self.tensor, output, expected_model_sha256=self._model_sha())
            self.assertTrue(renamed)
            self.assertFalse((target / "weights.q1_0.bin").exists())
            self.assertFalse((target / "activation.f32.bin").exists())
            self.assertFalse((target / "fixture.json").exists())
        finally:
            if target.exists():
                for child in target.iterdir():
                    child.unlink()
                target.rmdir()
            shutil.rmtree(base)

    def test_cleanup_leaves_unrelated_replacement_after_original_rename(self) -> None:
        base = self.work / "cleanup-base"
        base.mkdir()
        output = base / "fixture"
        moved = base / "fixture-original"
        replacement_created = False
        def replace_before_failure(directory_fd: int, name: str, payload: bytes) -> None:
            nonlocal replacement_created
            output.rename(moved)
            output.mkdir()
            replacement_created = True
            raise FixtureError("injected publication failure")

        with mock.patch.object(extractor, "_exclusive_write_at", side_effect=replace_before_failure):
            with self.assertRaises(FixtureError):
                extract_tensor(self.model, self.tensor, output, expected_model_sha256=self._model_sha())
        self.assertTrue(replacement_created)
        self.assertTrue(output.is_dir())
        self.assertTrue(moved.is_dir())
        self.assertEqual(list(output.iterdir()), [])

    def test_dirfd_output_survives_parent_symlink_swap(self) -> None:
        base = self.work / "base"
        base.mkdir()
        output = base / "safe" / "fixture"
        escape = self.work / "escape"
        escape.mkdir()
        real_open = extractor.os.open
        swapped = False

        def swap_before_output_file(path: object, flags: int, *args: object, **kwargs: object) -> int:
            nonlocal swapped
            path_text = os.fsdecode(path)
            if not swapped and (path_text.endswith("/fixture/weights.q1_0.bin") or path_text == "weights.q1_0.bin"):
                swapped = True
                safe = base / "safe"
                moved = base / "safe-real"
                safe.rename(moved)
                safe.symlink_to(escape, target_is_directory=True)
            return real_open(path, flags, *args, **kwargs)

        with mock.patch.object(extractor.os, "open", side_effect=swap_before_output_file):
            extract_tensor(self.model, self.tensor, output, expected_model_sha256=self._model_sha())
        self.assertTrue(swapped)
        self.assertFalse((escape / "fixture" / "weights.q1_0.bin").exists())
        self.assertEqual((base / "safe-real" / "fixture" / "weights.q1_0.bin").read_bytes(), self.payload)

    def test_script_entrypoint_loads_project_accounting_from_tooling_directory(self) -> None:
        workload = self.work / "empty-workload.json"
        workload.write_text("{}", encoding="utf-8")
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tooling" / "extract_q1_operator_fixture.py"),
                "--model",
                str(self.model),
                "--workload",
                str(workload),
                "--tensor",
                "missing",
                "--output-dir",
                str(self.work / "fixture"),
            ],
            cwd=self.work,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertTrue(completed.stderr.startswith("ERROR: "), completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)
        self.assertNotIn("No module named 'tooling'", completed.stderr)

    def test_script_entrypoint_rejects_invalid_utf8_without_traceback(self) -> None:
        workload = self.work / "invalid-utf8.json"
        workload.write_bytes(b"\xff")
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tooling" / "extract_q1_operator_fixture.py"),
                "--model",
                str(self.model),
                "--workload",
                str(workload),
                "--tensor",
                "missing",
                "--output-dir",
                str(self.work / "fixture"),
            ],
            cwd=self.work,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertTrue(completed.stderr.startswith("ERROR: "), completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)


if __name__ == "__main__":
    unittest.main()
