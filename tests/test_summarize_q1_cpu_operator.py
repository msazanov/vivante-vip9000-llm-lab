import json
import math
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import tooling.summarize_q1_cpu_operator as summarizer


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tooling" / "summarize_q1_cpu_operator.py"


def _f32(values):
    return b"".join(struct.pack("<f", value) for value in values)


def _run_bundle(root: Path, *, samples=50, mutate=None, output_values=None,
                tensor_name="blk.0.ffn_gate.weight", sidecar_mutate=None,
                observed_ddr=None):
    gate = tensor_name.endswith("ffn_gate.weight")
    k, m = ((5120, 17408) if gate else (17408, 5120))
    tensor_hash = ("0f42ca3b81099f540ed67941809ee7fdc0a672563bf852b87db75034135fc6fe"
                   if gate else "ab3ef165a5940b14ff1279843af15cbde0fa6cffc2e8eb063be70f7541f0fd04")
    reference = [float(index + 1) for index in range(m)]
    output = reference if output_values is None else output_values
    (root / "reference.f32.bin").write_bytes(_f32(reference))
    (root / "output.f32.bin").write_bytes(_f32(output))
    rows = [
        {
            "schema_version": "q1-cpu-operator-run/v1",
            "record": "identity",
            "run_id": "bonsai-q1-gate-a55-001",
            "model": {"id": "prism-ml/Ternary-Bonsai-27B-gguf", "sha256": "17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0", "size_bytes": 3803452480,
                      "quantization": "Q1_0"},
            "tensor": {"name": tensor_name, "sha256": tensor_hash,
                       "shape": [k, m], "packed_bytes": 12533760},
            "layout": {"logical_type": "GGML_TYPE_Q1_0", "source_layout": "GGUF_Q1_0/v1",
                       "executor_layout": "CPU_REPACK_Q1_0_4x4"},
            "workload": {"activation_f32_elements": k, "output_f32_elements": m},
            "weight_buffer_type": "CPU_REPACK", "backend": "ggml-cpu",
            "kernel": "q1_0_4x4_q8_0", "strict_mode": True, "threads": 2,
            "logical_gemv_count": 1,
        },
        {
            "schema_version": "q1-cpu-operator-run/v1", "record": "memory",
            "declared_canonical_q1_bytes": 12533760, "declared_cpu_repack_bytes": 12533760,
            "expanded_weight_ddr_bytes": 0, "activation_f32_bytes": k * 4,
            "activation_q8_bytes": (k // 32) * 34, "output_f32_bytes": m * 4,
            "observed_ddr_read_bytes": None if observed_ddr is None else observed_ddr,
            "observed_ddr_write_bytes": None if observed_ddr is None else observed_ddr // 2,
            "physical_resident_weight_copies": 1, "cpu_fallback_count": 0,
        },
    ]
    rows.append({"schema_version": "q1-cpu-operator-run/v1", "record": "sample",
                 "kind": "warmup", "iteration": 0, "host_us": 101.0, "compute_us": 100.0})
    for index in range(samples):
        rows.append({"schema_version": "q1-cpu-operator-run/v1", "record": "sample",
                     "kind": "measured", "iteration": index,
                     "host_us": 100.0 + index * 0.1, "compute_us": 90.0 + index * 0.1})
    rows.append({"schema_version": "q1-cpu-operator-run/v1", "record": "result",
                 "output_f32_bytes": len(_f32(output))})
    if mutate:
        mutate(rows)
    runner = root / "runner.jsonl"
    runner.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    telemetry = {
        "schema_version": "q1-cpu-operator-telemetry/v1", "swap_delta_kib": 0,
        "thermal_guard_passed": True, "throttle_detected": False,
        "oom_detected": False, "fault_detected": False,
    }
    guard_rows = [
        {"event": "start", "monotonic_ns": 1, "command": ["runner"], "limit_mc": 85000, "interval_ms": 10},
        {"event": "inventory", "monotonic_ns": 2,
         "thermal_zones": [{"path": "/sys/class/thermal/thermal_zone0/temp", "type": "cpu", "identity": "1:1"}],
         "cpu_policies": [{"path": "/sys/devices/system/cpu/cpufreq/policy0", "cur_khz": 1000,
                           "max_khz": 2000, "governor": "performance", "affected_cpus": "0-5", "identity": "2:2"}],
         "cooling_devices": [{"path": "/sys/class/thermal/cooling_device0", "type": "pwm-fan",
                              "cur_state": 0, "max_state": 4, "identity": "3:3"}]},
        {"event": "child_started", "monotonic_ns": 3, "pid": 1234},
        {"kind": "sample", "sample_seq": 0, "monotonic_ns": 4, "scheduled_deadline_ns": 4,
         "lateness_ns": 0, "process": {"pid": 1234, "available": False},
         "thermal_zones": [{"path": "/sys/class/thermal/thermal_zone0/temp", "type": "cpu", "millidegrees_c": 40000}],
         "cpu_policies": [{"path": "/sys/devices/system/cpu/cpufreq/policy0", "cur_khz": 1000,
                           "max_khz": 2000, "governor": "performance", "affected_cpus": "0-5", "identity": "2:2"}],
         "cooling_devices": [{"path": "/sys/class/thermal/cooling_device0", "type": "pwm-fan",
                              "cur_state": 0, "max_state": 4, "identity": "3:3"}]},
        {"event": "child_exit", "monotonic_ns": 5, "returncode": 0},
        {"event": "exit", "monotonic_ns": 6, "status": 0},
    ]
    if sidecar_mutate:
        sidecar_mutate(telemetry, guard_rows)
    (root / "telemetry.json").write_text(json.dumps(telemetry), encoding="utf-8")
    (root / "thermal-guard.jsonl").write_text("\n".join(json.dumps(row) for row in guard_rows) + "\n", encoding="utf-8")
    return runner, root / "reference.f32.bin", root / "output.f32.bin", root / "telemetry.json", root / "thermal-guard.jsonl"


class SummarizeQ1CpuOperatorTests(unittest.TestCase):
    def run_summary(self, mutate=None, sidecar_mutate=None, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            runner, reference, output, telemetry, guard = _run_bundle(
                Path(directory), mutate=mutate, sidecar_mutate=sidecar_mutate, **kwargs)
            target = Path(directory) / "summary.json"
            completed = subprocess.run(
                [sys.executable, str(SCRIPT), str(runner), "--golden-f32", str(reference),
                 "--output-f32", str(output), "--telemetry", str(telemetry),
                 "--thermal-guard", str(guard), "--output", str(target)],
                text=True, capture_output=True,
            )
            return completed, (json.loads(target.read_text()) if target.exists() else None)

    def test_happy_path_has_exact_stats_hashes_and_qualification(self):
        completed, result = self.run_summary()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(result["schema_version"], "q1-cpu-operator-baseline/v1")
        self.assertEqual(set(result), {"schema_version", "run_id", "model", "tensor", "layout",
                                       "workload", "executor", "memory", "execution", "timing",
                                       "correctness", "telemetry", "qualification"})
        self.assertEqual(result["timing"]["compute_us"]["min"], 90.0)
        self.assertEqual(result["timing"]["compute_us"]["median"], 92.45)
        self.assertEqual(result["timing"]["compute_us"]["max"], 94.9)
        self.assertAlmostEqual(result["timing"]["compute_us"]["p90"], 94.41, places=12)
        values = [90.0 + index * 0.1 for index in range(50)]
        mean = sum(values) / len(values)
        cv = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)) / mean
        self.assertAlmostEqual(result["timing"]["compute_us"]["population_cv"], cv, places=12)
        self.assertEqual(result["memory"]["observed_ddr_read_bytes"], None)
        self.assertEqual(result["memory"]["observed_ddr_write_bytes"], None)
        self.assertTrue(result["correctness"]["passed"])
        self.assertTrue(result["qualification"]["passed"])

    def test_each_invalid_gate_fails_closed(self):
        cases = [
            ("49 samples", lambda rows: rows.pop(4)),
            ("duplicate indices", lambda rows: rows[5].update(iteration=0)),
            ("nonfinite timing", lambda rows: rows[4].update(compute_us=float("nan"))),
            ("cv above two percent", lambda rows: [row.update(compute_us=1.0 if row["iteration"] % 2 else 2.0)
                                                    for row in rows if row["record"] == "sample" and row["kind"] == "measured"]),
            ("wrong tensor hash", lambda rows: rows[0]["tensor"].update(sha256="c" * 64)),
            ("missing CPU_REPACK", lambda rows: rows[0].update(weight_buffer_type="CPU")),
            ("expanded bytes", lambda rows: rows[1].update(expanded_weight_ddr_bytes=1)),
            ("output mismatch", lambda rows: rows[-1].update(output_f32_bytes=12)),
            ("swap delta", lambda rows: None),
            ("thermal failure", lambda rows: None),
        ]
        for name, mutate in cases:
            with self.subTest(name=name):
                sidecar = None
                if name == "swap delta":
                    sidecar = lambda telemetry, _guard: telemetry.update(swap_delta_kib=1)
                elif name == "thermal failure":
                    sidecar = lambda _telemetry, guard: guard.insert(0, {"event": "abort"})
                completed, result = self.run_summary(mutate, sidecar_mutate=sidecar)
                self.assertNotEqual(completed.returncode, 0)
                self.assertIsNone(result)

    def test_check_is_exclusive_and_accepts_deterministic_example(self):
        example = ROOT / "benchmarks/schema/q1-cpu-operator.example.json"
        completed = subprocess.run([sys.executable, str(SCRIPT), "--check", str(example)],
                                   text=True, capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("PASS q1-cpu-operator-baseline/v1", completed.stdout)
        bad = subprocess.run([sys.executable, str(SCRIPT), "--check", str(example),
                              "--output", str(example.with_suffix(".out.json"))],
                             text=True, capture_output=True)
        self.assertNotEqual(bad.returncode, 0)

    def test_generated_summary_round_trips_for_gate_and_down(self):
        for tensor_name in ("blk.0.ffn_gate.weight", "blk.0.ffn_down.weight"):
            with self.subTest(tensor_name=tensor_name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                runner, reference, output, telemetry, guard = _run_bundle(root, tensor_name=tensor_name)
                target = root / "summary.json"
                command = [sys.executable, str(SCRIPT), str(runner), "--golden-f32", str(reference),
                           "--output-f32", str(output), "--telemetry", str(telemetry),
                           "--thermal-guard", str(guard), "--output", str(target)]
                generated = subprocess.run(command, text=True, capture_output=True)
                self.assertEqual(generated.returncode, 0, generated.stderr)
                checked = subprocess.run([sys.executable, str(SCRIPT), "--check", str(target)],
                                         text=True, capture_output=True)
                self.assertEqual(checked.returncode, 0, checked.stderr)

    def test_canonical_runner_fixture_has_exact_wire_records(self):
        fixture = ROOT / "benchmarks/schema/q1-cpu-operator-run.example.jsonl"
        rows = [json.loads(line) for line in fixture.read_text().splitlines()]
        self.assertEqual(len(rows), 54)
        self.assertEqual([row["record"] for row in rows[:3]], ["identity", "memory", "sample"])
        self.assertEqual(rows[-1], {"schema_version": "q1-cpu-operator-run/v1",
                                    "record": "result", "output_f32_bytes": 69632})
        self.assertEqual(set(rows[0]), {"schema_version", "record", "run_id", "model", "tensor",
                                        "layout", "workload", "weight_buffer_type", "backend",
                                        "kernel", "strict_mode", "threads", "logical_gemv_count"})
        self.assertEqual(set(rows[1]), {"schema_version", "record", "declared_canonical_q1_bytes",
                                        "declared_cpu_repack_bytes", "expanded_weight_ddr_bytes",
                                        "activation_f32_bytes", "activation_q8_bytes", "output_f32_bytes",
                                        "observed_ddr_read_bytes", "observed_ddr_write_bytes",
                                        "physical_resident_weight_copies", "cpu_fallback_count"})

    def test_measured_ddr_counters_round_trip_as_nonnegative_integers(self):
        completed, result = self.run_summary(observed_ddr=12345)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(result["memory"]["observed_ddr_read_bytes"], 12345)
        self.assertEqual(result["memory"]["observed_ddr_write_bytes"], 6172)

    def test_check_rejects_forged_correctness_decisions(self):
        value = json.loads((ROOT / "benchmarks/schema/q1-cpu-operator.example.json").read_text())
        mutations = [
            lambda row: row["correctness"].update(cosine=0.0),
            lambda row: row["correctness"].update(max_abs_error=999.0),
            lambda row: row["correctness"].update(cosine_passed=False),
            lambda row: row["correctness"].update(max_abs_error_passed=False),
            lambda row: row["correctness"].update(finite_passed=False),
            lambda row: row["correctness"].update(reference_abs_max=1.0),
            lambda row: (row["tensor"].update(packed_bytes=1),
                         row["memory"].update(declared_canonical_q1_bytes=1,
                                              declared_cpu_repack_bytes=1)),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                candidate = json.loads(json.dumps(value))
                mutate(candidate)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "mutated.json"
                    path.write_text(json.dumps(candidate))
                    completed = subprocess.run([sys.executable, str(SCRIPT), "--check", str(path)],
                                               text=True, capture_output=True)
                    self.assertNotEqual(completed.returncode, 0)

    def test_atomic_writer_cleans_partial_temp_on_injected_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "summary.json"
            original_dump = summarizer.json.dump
            def fail_after_partial(value, handle, **kwargs):
                handle.write('{"partial":')
                raise OSError("injected write failure")
            summarizer.json.dump = fail_after_partial
            try:
                with self.assertRaises(OSError):
                    summarizer._write_exclusive_json(target, {"ok": True})
            finally:
                summarizer.json.dump = original_dump
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(directory).glob(".*.tmp")), [])

    def test_check_rejects_mutation_in_every_summary_section(self):
        value = json.loads((ROOT / "benchmarks/schema/q1-cpu-operator.example.json").read_text())
        sections = {"model", "tensor", "layout", "workload", "executor", "memory", "execution",
                    "timing", "correctness", "telemetry", "qualification"}
        for section in sections:
            with self.subTest(section=section), tempfile.TemporaryDirectory() as directory:
                candidate = json.loads(json.dumps(value))
                candidate[section] = None
                path = Path(directory) / "mutated.json"
                path.write_text(json.dumps(candidate))
                completed = subprocess.run([sys.executable, str(SCRIPT), "--check", str(path)],
                                           text=True, capture_output=True)
                self.assertNotEqual(completed.returncode, 0)

    def test_measured_ddr_counters_reject_zero_negative_and_boolean(self):
        for value in (0, -1, True):
            with self.subTest(value=value):
                completed, result = self.run_summary(
                    mutate=lambda rows, value=value: rows[1].update(observed_ddr_read_bytes=value,
                                                                     observed_ddr_write_bytes=1))
                self.assertNotEqual(completed.returncode, 0)
                self.assertIsNone(result)

    def test_thermal_lifecycle_is_exact_and_strictly_typed(self):
        mutations = [
            lambda _telemetry, guard: guard[3].update(sample_seq=0.0),
            lambda _telemetry, guard: guard[3].update(sample_seq=False),
            lambda _telemetry, guard: guard[4].update(returncode=False),
            lambda _telemetry, guard: guard[5].update(status=0.0),
            lambda _telemetry, guard: guard[0].update(unexpected=True),
            lambda _telemetry, guard: guard[3].update(extra=True),
            lambda _telemetry, guard: guard.insert(0, dict(guard[0])),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                completed, result = self.run_summary(sidecar_mutate=mutate)
                self.assertNotEqual(completed.returncode, 0)
                self.assertIsNone(result)

    def test_nested_thermal_schema_and_cross_correlation_are_strict(self):
        mutations = [
            lambda _telemetry, guard: guard[1].update(thermal_zones=[]),
            lambda _telemetry, guard: guard[3].update(thermal_zones=[]),
            lambda _telemetry, guard: guard[3]["thermal_zones"].append(dict(guard[3]["thermal_zones"][0])),
            lambda _telemetry, guard: (guard[3]["thermal_zones"].append(dict(guard[3]["thermal_zones"][0])),
                                       guard[3]["thermal_zones"].reverse()),
            lambda _telemetry, guard: guard[3]["thermal_zones"][0].update(type="other"),
            lambda _telemetry, guard: guard[3]["cpu_policies"][0].update(identity="wrong"),
            lambda _telemetry, guard: guard[3]["cooling_devices"][0].update(max_state=5),
            lambda _telemetry, guard: guard[3]["process"].update(pid=9999),
            lambda _telemetry, guard: guard[1]["thermal_zones"][0].update(unknown=[]),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                completed, result = self.run_summary(sidecar_mutate=mutate)
                self.assertNotEqual(completed.returncode, 0)
                self.assertIsNone(result)

    def test_thermal_timestamps_deadline_lateness_and_temperature_are_strict(self):
        mutations = [
            lambda _telemetry, guard: guard[1].update(monotonic_ns=0),
            lambda _telemetry, guard: guard[3].update(lateness_ns=1),
            lambda _telemetry, guard: guard[3].update(scheduled_deadline_ns=0),
            lambda _telemetry, guard: guard[3]["thermal_zones"][0].update(millidegrees_c=85000),
            lambda _telemetry, guard: guard[0].update(command=[]),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                completed, result = self.run_summary(sidecar_mutate=mutate)
                self.assertNotEqual(completed.returncode, 0)
                self.assertIsNone(result)

    def test_emitter_integer_ranges_and_empty_later_command_args_are_accepted(self):
        def mutate(_telemetry, guard):
            guard[0]["command"] = ["runner", ""]
            guard[3]["thermal_zones"][0]["millidegrees_c"] = -5000
            guard[3]["cpu_policies"][0].update(cur_khz=0, max_khz=-1)
            guard[1]["cpu_policies"][0].update(cur_khz=0, max_khz=-1)
            guard[3]["cooling_devices"][0].update(cur_state=-2, max_state=-1)
            guard[1]["cooling_devices"][0].update(cur_state=-2, max_state=-1)
        completed, result = self.run_summary(sidecar_mutate=mutate)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(result["qualification"]["passed"])

    def test_path_identity_and_command_authentication_reject_forgery(self):
        mutations = [
            lambda _telemetry, guard: guard[1]["thermal_zones"][0].update(path="relative/temp"),
            lambda _telemetry, guard: guard[1]["cpu_policies"][0].update(path="/sys/../bad"),
            lambda _telemetry, guard: guard[1]["cooling_devices"][0].update(identity="x-y"),
            lambda _telemetry, guard: guard[1]["cpu_policies"][0].update(
                path=guard[1]["thermal_zones"][0]["path"]),
            lambda _telemetry, guard: (guard[1]["thermal_zones"][0].update(identity="4:4"),
                                       guard[1]["cpu_policies"][0].update(identity="4:4")),
            lambda _telemetry, guard: guard[1]["thermal_zones"][0].update(identity="04:4"),
            lambda _telemetry, guard: guard[0].update(command=[""]),
            lambda _telemetry, guard: guard[0].update(command=["runner", 1]),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                completed, result = self.run_summary(sidecar_mutate=mutate)
                self.assertNotEqual(completed.returncode, 0)
                self.assertIsNone(result)

    def test_invalid_utf8_check_is_concise_rc2(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_bytes(b"\xff\xfe\xfd")
            completed = subprocess.run([sys.executable, str(SCRIPT), "--check", str(path)],
                                       text=True, capture_output=True)
            self.assertEqual(completed.returncode, 2)
            self.assertNotIn("Traceback", completed.stderr)

    def test_malformed_runner_and_summary_shapes_fail_without_traceback(self):
        for malformed in ({}, []):
            with self.subTest(malformed=malformed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                runner, reference, output, telemetry, guard = _run_bundle(root)
                rows = [json.loads(line) for line in runner.read_text().splitlines()]
                rows[3]["record"] = malformed
                runner.write_text("".join(json.dumps(row) + "\n" for row in rows))
                completed = subprocess.run(
                    [sys.executable, str(SCRIPT), str(runner), "--golden-f32", str(reference),
                     "--output-f32", str(output), "--telemetry", str(telemetry),
                     "--thermal-guard", str(guard), "--output", str(root / "summary.json")],
                    text=True, capture_output=True)
                self.assertEqual(completed.returncode, 2)
                self.assertNotIn("Traceback", completed.stderr)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = json.loads((ROOT / "benchmarks/schema/q1-cpu-operator.example.json").read_text())
            value["correctness"]["thresholds"] = [{}]
            path = root / "summary.json"
            path.write_text(json.dumps(value))
            completed = subprocess.run([sys.executable, str(SCRIPT), "--check", str(path)],
                                       text=True, capture_output=True)
            self.assertEqual(completed.returncode, 2)
            self.assertNotIn("Traceback", completed.stderr)

    def test_summary_telemetry_uses_strict_integer_and_boolean_types(self):
        value = json.loads((ROOT / "benchmarks/schema/q1-cpu-operator.example.json").read_text())
        mutations = [
            lambda row: row["telemetry"].update(swap_delta_kib=0.0),
            lambda row: row["telemetry"].update(swap_delta_kib=False),
            lambda row: row["telemetry"].update(swap_delta_kib=[]),
            lambda row: row["telemetry"].update(swap_delta_kib={}),
            lambda row: row["telemetry"].update(thermal_guard_passed=1),
            lambda row: row["telemetry"].update(throttle_detected="false"),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate), tempfile.TemporaryDirectory() as directory:
                candidate = json.loads(json.dumps(value))
                mutate(candidate)
                path = Path(directory) / "mutated.json"
                path.write_text(json.dumps(candidate))
                completed = subprocess.run([sys.executable, str(SCRIPT), "--check", str(path)],
                                           text=True, capture_output=True)
                self.assertEqual(completed.returncode, 2)
                self.assertNotIn("Traceback", completed.stderr)

    def test_duplicate_json_members_are_rejected_in_runner_telemetry_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner, reference, output, telemetry, guard = _run_bundle(root)
            lines = runner.read_text().splitlines()
            lines[3] = lines[3].replace('"iteration": 0', '"iteration": 0, "iteration": 0')
            runner.write_text("\n".join(lines) + "\n")
            command = [sys.executable, str(SCRIPT), str(runner), "--golden-f32", str(reference),
                       "--output-f32", str(output), "--telemetry", str(telemetry),
                       "--thermal-guard", str(guard), "--output", str(root / "summary.json")]
            completed = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(completed.returncode, 2)
            self.assertNotIn("Traceback", completed.stderr)
            second = root / "second"
            second.mkdir()
            runner, reference, output, telemetry, guard = _run_bundle(second)
            telemetry.write_text('{"schema_version":"q1-cpu-operator-telemetry/v1",'
                                 '"swap_delta_kib":0,"swap_delta_kib":0,"thermal_guard_passed":true,'
                                 '"throttle_detected":false,"oom_detected":false,"fault_detected":false}')
            command = [sys.executable, str(SCRIPT), str(runner), "--golden-f32", str(reference),
                       "--output-f32", str(output), "--telemetry", str(telemetry),
                       "--thermal-guard", str(guard), "--output", str(root / "second" / "summary.json")]
            completed = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(completed.returncode, 2)
            self.assertNotIn("Traceback", completed.stderr)
            summary = root / "summary.json"
            summary.write_text('{"schema_version":"q1-cpu-operator-baseline/v1",'
                               '"schema_version":"q1-cpu-operator-baseline/v1"}')
            completed = subprocess.run([sys.executable, str(SCRIPT), "--check", str(summary)],
                                       text=True, capture_output=True)
            self.assertEqual(completed.returncode, 2)
            self.assertNotIn("Traceback", completed.stderr)


if __name__ == "__main__":
    unittest.main()
