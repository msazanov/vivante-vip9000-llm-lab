import json
import hashlib
import os
import struct
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from tooling.summarize_target_run import _consume_phases
from tooling.summarize_q1_cpu_operator import _parse_bundle


ROOT = Path(__file__).resolve().parents[1]


class Q1CpuOperatorRunnerTests(unittest.TestCase):
    def test_literal_q1_repack_graph_and_wire_contract(self):
        runner_name = os.environ.get("Q1_CPU_OPERATOR_RUNNER")
        if not runner_name:
            self.skipTest("Q1_CPU_OPERATOR_RUNNER is unset; target CPU_REPACK runtime test skipped")
        runner = Path(runner_name)
        if not runner.is_file():
            self.fail(f"Q1_CPU_OPERATOR_RUNNER does not name an executable: {runner}")

        with tempfile.TemporaryDirectory(prefix="q1-runner-test-") as directory:
            root = Path(directory)
            fixture = root / "fixture"
            output = root / "output"
            fixture.mkdir()

            # Literal canonical Q1_0: one 128-value block per row, 16 rows.
            # Every activation is one, so each row's expected dot is the signed
            # population count times d=0.5.  This expected list is deliberately
            # independent of Prism and is hand-derived from the payload below.
            q1 = bytearray()
            for row in range(16):
                q1 += struct.pack("<H", 0x3800)  # FP16 0.5
                q1 += bytes([0xFF] * (row + 1))
                q1 += bytes([0x00] * (16 - row - 1))
            self.assertEqual(len(q1), 16 * 18)
            (fixture / "weights.q1_0.bin").write_bytes(q1)
            activation = struct.pack("<128f", *([1.0] * 128))
            (fixture / "activation.f32.bin").write_bytes(activation)
            (fixture / "fixture.json").write_text(json.dumps({
                "schema": "q1-cpu-operator-fixture/v1",
                "model": {
                    "filename": "Bonsai-27B-Q1_0.gguf",
                    "sha256": "17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0",
                    "size_bytes": 3803452480,
                },
                "tensor": {
                    "file": "weights.q1_0.bin",
                    "ggml_type": "Q1_0",
                    "name": "literal.q1",
                    "sha256": hashlib.sha256(q1).hexdigest(),
                    "shape": [128, 16],
                    "size_bytes": len(q1),
                },
                "activation": {
                    "dtype": "F32",
                    "file": "activation.f32.bin",
                    "sha256": hashlib.sha256(activation).hexdigest(),
                    "shape": [128],
                    "size_bytes": len(activation),
                },
            }, sort_keys=True), encoding="utf-8")

            phase_file = root / "phase.jsonl"
            phase_file.write_text(json.dumps({
                "event": "profiler_start", "monotonic_ns": time.monotonic_ns(), "step": 0,
            }) + "\n", encoding="utf-8")
            env = dict(os.environ, VIP9000_PHASE_FILE=str(phase_file))
            completed = subprocess.run(
                [str(runner), "--self-test", "--fixture-dir", str(fixture),
                 "--run-id", "q1-literal-001", "--iterations", "2", "--output-dir", str(output)],
                text=True,
                capture_output=True,
                env=env,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

            records = [json.loads(line) for line in (output / "runner.jsonl").read_text().splitlines()]
            self.assertEqual(
                [row["record"] for row in records],
                ["identity", "memory", "sample", "sample", "sample", "result"],
            )
            self.assertEqual(records[0]["weight_buffer_type"], "CPU_REPACK")
            self.assertEqual(records[0]["run_id"], "q1-literal-001")
            self.assertEqual(records[0]["backend"], "ggml-cpu")
            self.assertEqual(records[0]["kernel"], "q1_0_4x4_q8_0")
            self.assertTrue(records[0]["strict_mode"])
            self.assertEqual(records[0]["model"]["id"], "prism-ml/Ternary-Bonsai-27B-gguf")
            self.assertEqual(records[0]["model"]["size_bytes"], 3803452480)
            self.assertEqual(records[0]["tensor"]["name"], "literal.q1")
            self.assertEqual(records[0]["tensor"]["sha256"], hashlib.sha256(q1).hexdigest())
            self.assertEqual(records[1]["expanded_weight_ddr_bytes"], 0)
            self.assertEqual(records[1]["cpu_fallback_count"], 0)
            self.assertEqual(records[-1], {
                "schema_version": "q1-cpu-operator-run/v1",
                "record": "result",
                "output_f32_bytes": 16 * 4,
            })

            values = struct.unpack("<16f", (output / "output.f32.bin").read_bytes())
            expected = [
                -56.0, -48.0, -40.0, -32.0, -24.0, -16.0, -8.0, 0.0,
                8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0, 64.0,
            ]
            for actual, hand_derived in zip(values, expected):
                self.assertAlmostEqual(actual, hand_derived, delta=1e-2)
            self.assertEqual((output / "activation.q8_0.bin").stat().st_size, 4 * 34)

            with phase_file.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({
                    "event": "profiler_complete", "monotonic_ns": time.monotonic_ns(), "step": 0,
                }) + "\n")
            all_phase_rows = [json.loads(line) for line in phase_file.read_text().splitlines()]
            self.assertEqual([all_phase_rows[0]["event"], all_phase_rows[-1]["event"]],
                             ["profiler_start", "profiler_complete"])
            phase_rows = all_phase_rows[1:-1]
            self.assertEqual([set(row) for row in phase_rows],
                             [{"event", "monotonic_ns", "step"}] * 10)
            self.assertEqual([row["step"] for row in phase_rows], list(range(1, 11)))
            self.assertTrue(all(row["monotonic_ns"] >= 0 for row in phase_rows))
            self.assertEqual(_consume_phases(phase_file)[0], 12)
            phase_events = [row["event"] for row in phase_rows]
            self.assertEqual(phase_events, [
                "setup_begin", "repack_begin", "repack_end", "warmup_begin",
                "warmup_end", "compute_begin", "compute_end", "output_read_begin",
                "output_read_end", "teardown",
            ])

            prior = {path.name: path.read_bytes() for path in output.iterdir()}
            rerun = subprocess.run(
                [str(runner), "--self-test", "--fixture-dir", str(fixture),
                 "--run-id", "q1-literal-001", "--iterations", "2", "--output-dir", str(output)],
                text=True, capture_output=True, env=env,
            )
            self.assertNotEqual(rerun.returncode, 0)
            self.assertEqual({path.name: path.read_bytes() for path in output.iterdir()}, prior)
            self.assertEqual(list(output.glob(".*.tmp.*")), [])

            bad_fixture = root / "bad-fixture"
            bad_fixture.mkdir()
            (bad_fixture / "weights.q1_0.bin").write_bytes(q1)
            failed_output = root / "failed-output"
            failed = subprocess.run(
                [str(runner), "--self-test", "--fixture-dir", str(bad_fixture),
                 "--run-id", "q1-invalid-001", "--iterations", "2", "--output-dir", str(failed_output)],
                text=True, capture_output=True, env=env,
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertFalse(failed_output.exists())

    def test_normal_production_fixture_is_strict_and_task5_consumable(self):
        runner_name = os.environ.get("Q1_CPU_OPERATOR_RUNNER")
        fixture_name = os.environ.get("Q1_CPU_OPERATOR_PRODUCTION_FIXTURE")
        if not runner_name and not fixture_name:
            self.skipTest("Q1_CPU_OPERATOR_RUNNER and Q1_CPU_OPERATOR_PRODUCTION_FIXTURE are required")
        if not runner_name or not fixture_name:
            self.fail("Q1_CPU_OPERATOR_RUNNER and Q1_CPU_OPERATOR_PRODUCTION_FIXTURE must be set together")
        runner = Path(runner_name)
        fixture = Path(fixture_name)
        manifest = json.loads((fixture / "fixture.json").read_text())
        shape = manifest["tensor"]["shape"]
        self.assertEqual(shape, [5120, 17408])
        with tempfile.TemporaryDirectory(prefix="q1-production-run-") as directory:
            output = Path(directory)
            phase_file = output / "phases.jsonl"
            env = dict(os.environ, VIP9000_PHASE_FILE=str(phase_file))
            completed = subprocess.run([
                str(runner), "--run-id", "bonsai-q1-gate-a55-001",
                "--weights", str(fixture / "weights.q1_0.bin"),
                "--activation", str(fixture / "activation.f32.bin"),
                "--ne0", "5120", "--ne1", "17408", "--threads", "6",
                "--warmup", "1", "--iterations", "50",
                "--output-jsonl", str(output / "runner.jsonl"),
                "--output-f32", str(output / "output.f32.bin"),
                "--output-q8", str(output / "activation.q8_0.bin"),
            ], text=True, capture_output=True, env=env)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            records = [json.loads(line) for line in (output / "runner.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 54)
            _parse_bundle(records)
            self.assertEqual(set(records[0]), {
                "schema_version", "record", "run_id", "model", "tensor", "layout", "workload",
                "weight_buffer_type", "backend", "kernel", "strict_mode", "threads", "logical_gemv_count",
            })
            self.assertEqual(set(records[1]), {
                "schema_version", "record", "declared_canonical_q1_bytes", "declared_cpu_repack_bytes",
                "expanded_weight_ddr_bytes", "activation_f32_bytes", "activation_q8_bytes", "output_f32_bytes",
                "observed_ddr_read_bytes", "observed_ddr_write_bytes", "physical_resident_weight_copies",
                "cpu_fallback_count",
            })
            self.assertEqual(records[0]["run_id"], "bonsai-q1-gate-a55-001")
            self.assertEqual(records[0]["model"], manifest_model := {
                "id": "prism-ml/Ternary-Bonsai-27B-gguf",
                "sha256": manifest["model"]["sha256"],
                "size_bytes": manifest["model"]["size_bytes"],
                "quantization": "Q1_0",
            })
            self.assertEqual(records[0]["tensor"]["name"], "blk.0.ffn_gate.weight")
            self.assertEqual(records[0]["tensor"]["shape"], [5120, 17408])
            self.assertEqual(_consume_phases(phase_file)[0], 10)
            prior = {path.name: path.read_bytes() for path in output.iterdir() if path.name != "phases.jsonl"}
            rerun = subprocess.run([
                str(runner), "--run-id", "bonsai-q1-gate-a55-001",
                "--weights", str(fixture / "weights.q1_0.bin"),
                "--activation", str(fixture / "activation.f32.bin"),
                "--ne0", "5120", "--ne1", "17408", "--threads", "6",
                "--warmup", "1", "--iterations", "50",
                "--output-jsonl", str(output / "runner.jsonl"),
                "--output-f32", str(output / "output.f32.bin"),
                "--output-q8", str(output / "activation.q8_0.bin"),
            ], text=True, capture_output=True, env=env)
            self.assertNotEqual(rerun.returncode, 0)
            self.assertEqual({path.name: path.read_bytes() for path in output.iterdir() if path.name != "phases.jsonl"}, prior)
            self.assertEqual(list(output.glob(".*.tmp.*")), [])


if __name__ == "__main__":
    unittest.main()
