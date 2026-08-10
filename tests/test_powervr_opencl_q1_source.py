import json
import os
import re
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "E007-powervr-opencl-q1"
RUNNER = EXPERIMENT / "powervr_opencl_q1_runner.c"
KERNEL = EXPERIMENT / "q1_packed_gemv.cl"
README = EXPERIMENT / "README.md"


class PowerVROpenCLQ1SourceTests(unittest.TestCase):
    def _build_runner(self, directory: Path) -> Path:
        compiler = shutil.which("cc")
        if compiler is None:
            self.skipTest("a C compiler is not installed")
        binary = directory / "powervr-opencl-q1-runner"
        completed = subprocess.run(
            [
                compiler,
                "-std=c11",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Wpedantic",
                "-Werror",
                str(RUNNER),
                "-ldl",
                "-lm",
                "-o",
                str(binary),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return binary

    def test_compiled_runner_reports_the_pinned_runtime_and_layout_contract(self):
        self.assertTrue(RUNNER.is_file(), f"missing runner: {RUNNER}")
        self.assertTrue(KERNEL.is_file(), f"missing kernel: {KERNEL}")
        with tempfile.TemporaryDirectory(prefix="powervr-opencl-contract-") as directory:
            binary = self._build_runner(Path(directory))
            completed = subprocess.run(
                [str(binary), "--print-contract"],
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        contract = json.loads(completed.stdout)
        self.assertEqual(contract, {
            "schema": "powervr-opencl-q1-contract/v1",
            "runtime": "/usr/lib/libPVROCL.so.1",
            "generic_icd_allowed": False,
            "kernel_language": "OpenCL C 1.2",
            "weight_layout": "Q1_0:fp16-scale+16-sign-bytes-per-128",
            "activation_dtype": "F32",
            "output_dtype": "F32",
            "explicit_host_expanded_weight_allocation_bytes": 0,
            "runtime_internal_expansion_observed": "unknown",
            "allowed_local_sizes": [32, 64, 128],
            "qualified_steady_minimum_warmup": 1,
            "qualified_steady_minimum_iterations": 50,
            "timing_fields": [
                "weights_h2d_event_ms",
                "weights_h2d_host_ms",
                "activation_h2d_event_ms",
                "activation_h2d_host_ms",
                "d2h_event_ms",
                "d2h_host_ms",
                "host_total_ms",
            ],
            "host_total_scope": "weights_h2d_start_through_final_d2h_finish",
            "thermal_wrapper": "external",
        })

    def test_kernel_compiles_as_strict_opencl_c_12(self):
        clang = shutil.which("clang")
        if clang is None:
            self.skipTest("clang is not installed")
        completed = subprocess.run(
            [
                clang,
                "-x",
                "cl",
                "-cl-std=CL1.2",
                "-Werror",
                "-fsyntax-only",
                str(KERNEL),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_kernel_uses_powervr_compiler_safe_builtin_fp16_load(self):
        source = KERNEL.read_text(encoding="utf-8")
        self.assertIn("#pragma OPENCL EXTENSION cl_khr_fp16 : enable", source)
        self.assertIn("vload_half", source)
        self.assertNotIn("q1_half_to_float", source)

    def test_opencl_profiling_constants_measure_start_to_end_execution(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertRegex(source, r"CL_PROFILING_COMMAND_START\s*=\s*0x1282")
        self.assertRegex(source, r"CL_PROFILING_COMMAND_END\s*=\s*0x1283")
        self.assertNotRegex(source, r"CL_PROFILING_COMMAND_START\s*=\s*0x1280")
        self.assertNotRegex(source, r"CL_PROFILING_COMMAND_END\s*=\s*0x1281")

    def _write_literal_fixture(self, directory: Path) -> tuple[Path, Path]:
        weights = bytearray()
        signs = [0x00, 0xFF, 0xAA] + [0x00] * 125
        for sign in signs:
            weights += struct.pack("<H", 0x3800)  # FP16 0.5
            weights += bytes([sign] * 16)
        weights_path = directory / "weights.q1_0.bin"
        activation_path = directory / "activation.f32.bin"
        weights_path.write_bytes(weights)
        activation_path.write_bytes(struct.pack("<128f", *([1.0] * 128)))
        return weights_path, activation_path

    def test_cpu_reference_mode_emits_hand_derived_golden_and_memory_accounting(self):
        with tempfile.TemporaryDirectory(prefix="powervr-opencl-golden-") as directory:
            root = Path(directory)
            binary = self._build_runner(root)
            weights, activation = self._write_literal_fixture(root)
            jsonl = root / "run.jsonl"
            output = root / "output.f32.bin"
            completed = subprocess.run(
                [
                    str(binary),
                    "--cpu-reference-only",
                    "--weights", str(weights),
                    "--activation", str(activation),
                    "--m", "128",
                    "--k", "128",
                    "--warmup", "1",
                    "--iterations", "50",
                    "--local-size", "64",
                    "--run-id", "literal-q1-001",
                    "--output-jsonl", str(jsonl),
                    "--output-f32", str(output),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            rows = [json.loads(line) for line in jsonl.read_text().splitlines()]
            self.assertEqual([row["record"] for row in rows], ["identity", "memory", "result"])
            self.assertEqual(rows[0]["run_id"], "literal-q1-001")
            self.assertEqual(rows[0]["backend"], "scalar-c11-reference")
            self.assertEqual(rows[0]["shape"], {"m": 128, "k": 128})
            self.assertEqual(rows[1], {
                "schema": "powervr-opencl-q1-run/v1",
                "record": "memory",
                "packed_weight_bytes": 2304,
                "explicit_host_expanded_weight_allocation_bytes": 0,
                "runtime_internal_expansion_observed": "unknown",
                "activation_f32_bytes": 512,
                "output_f32_bytes": 512,
                "resident_weight_upload_count": 0,
            })
            self.assertEqual(rows[2]["golden"], "scalar-c11-q1_0-fp32")
            self.assertTrue(rows[2]["golden_pass"])
            self.assertTrue(rows[2]["cpu_reference_only"])
            values = struct.unpack("<128f", output.read_bytes())
            self.assertEqual(values[:3], (-64.0, 64.0, 0.0))
            self.assertEqual(values[3:], (-64.0,) * 125)

    def test_cpu_reference_handles_two_packed_blocks_per_row(self):
        with tempfile.TemporaryDirectory(prefix="powervr-opencl-two-block-") as directory:
            root = Path(directory)
            binary = self._build_runner(root)
            row_blocks = [
                ((0x3800, 0x00), (0x3C00, 0xFF)),
                ((0x3800, 0xFF), (0x3C00, 0x00)),
                ((0x3800, 0xAA), (0x3C00, 0xAA)),
            ] + [((0x3800, 0xFF), (0x3400, 0xFF))] * 125
            packed = bytearray()
            for blocks in row_blocks:
                for scale_bits, signs in blocks:
                    packed += struct.pack("<H", scale_bits)
                    packed += bytes([signs] * 16)
            weights = root / "weights-k256.q1_0.bin"
            activation = root / "activation-k256.f32.bin"
            weights.write_bytes(packed)
            activation.write_bytes(struct.pack("<256f", *([1.0] * 128 + [2.0] * 128)))
            jsonl = root / "run.jsonl"
            output = root / "output.f32.bin"
            completed = subprocess.run(
                [
                    str(binary), "--cpu-reference-only",
                    "--weights", str(weights), "--activation", str(activation),
                    "--m", "128", "--k", "256", "--local-size", "64",
                    "--warmup", "1", "--iterations", "50",
                    "--output-jsonl", str(jsonl), "--output-f32", str(output),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            rows = [json.loads(line) for line in jsonl.read_text().splitlines()]
            self.assertEqual(rows[1]["packed_weight_bytes"], 4608)
            self.assertEqual(rows[1]["activation_f32_bytes"], 1024)
            values = struct.unpack("<128f", output.read_bytes())
            self.assertEqual(values[:3], (192.0, -192.0, 0.0))
            self.assertEqual(values[3:], (128.0,) * 125)

    def test_cpu_reference_rejects_non_finite_accumulator_without_outputs(self):
        with tempfile.TemporaryDirectory(prefix="powervr-opencl-overflow-") as directory:
            root = Path(directory)
            binary = self._build_runner(root)
            weights, activation = self._write_literal_fixture(root)
            activation.write_bytes(struct.pack("<128f", *([3.4e38] * 128)))
            jsonl = root / "run.jsonl"
            output = root / "output.f32.bin"
            completed = subprocess.run(
                [
                    str(binary), "--cpu-reference-only",
                    "--weights", str(weights), "--activation", str(activation),
                    "--m", "128", "--k", "128", "--local-size", "64",
                    "--warmup", "1", "--iterations", "50",
                    "--output-jsonl", str(jsonl), "--output-f32", str(output),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("non-finite scalar accumulator/output", completed.stderr)
            self.assertFalse(jsonl.exists())
            self.assertFalse(output.exists())

    def test_runner_rejects_unqualified_steady_settings_without_outputs(self):
        with tempfile.TemporaryDirectory(prefix="powervr-opencl-unqualified-") as directory:
            root = Path(directory)
            binary = self._build_runner(root)
            weights, activation = self._write_literal_fixture(root)
            cases = (
                ("warmup", "0", "50", "warmup must be at least 1"),
                ("iterations", "1", "49", "iterations must be at least 50"),
            )
            for name, warmup, iterations, message in cases:
                with self.subTest(name=name):
                    jsonl = root / f"{name}.jsonl"
                    output = root / f"{name}.f32.bin"
                    completed = subprocess.run(
                        [
                            str(binary), "--cpu-reference-only",
                            "--weights", str(weights), "--activation", str(activation),
                            "--m", "128", "--k", "128", "--local-size", "64",
                            "--warmup", warmup, "--iterations", iterations,
                            "--output-jsonl", str(jsonl), "--output-f32", str(output),
                        ],
                        text=True,
                        capture_output=True,
                        check=False,
                    )
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertIn(message, completed.stderr)
                    self.assertFalse(jsonl.exists())
                    self.assertFalse(output.exists())

    def test_publication_is_atomic_no_replace_under_a_competing_writer(self):
        with tempfile.TemporaryDirectory(prefix="powervr-opencl-publish-race-") as directory:
            root = Path(directory)
            compiler = shutil.which("cc")
            if compiler is None:
                self.skipTest("a C compiler is not installed")
            binary = self._build_runner(root)
            injector_source = root / "publish-race-injector.c"
            injector = root / "publish-race-injector.so"
            injector_source.write_text(
                r'''
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>

static int injected = 0;

static void inject_competing_file(const char *target) {
    if (injected != 0) return;
    injected = 1;
    int fd = open(target, O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd >= 0) {
        (void) write(fd, "racer", 5);
        (void) close(fd);
    }
}

static int call_next(const char *name, const char *old_path, const char *new_path) {
    void *symbol = dlsym(RTLD_NEXT, name);
    int (*function)(const char *, const char *) = 0;
    memcpy(&function, &symbol, sizeof(function));
    return function(old_path, new_path);
}

int rename(const char *old_path, const char *new_path) {
    inject_competing_file(new_path);
    return call_next("rename", old_path, new_path);
}

int link(const char *old_path, const char *new_path) {
    inject_competing_file(new_path);
    return call_next("link", old_path, new_path);
}
''',
                encoding="utf-8",
            )
            built = subprocess.run(
                [compiler, "-std=c11", "-shared", "-fPIC", str(injector_source),
                 "-ldl", "-o", str(injector)],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(built.returncode, 0, built.stderr)
            weights, activation = self._write_literal_fixture(root)
            jsonl = root / "run.jsonl"
            output = root / "output.f32.bin"
            completed = subprocess.run(
                [
                    str(binary), "--cpu-reference-only",
                    "--weights", str(weights), "--activation", str(activation),
                    "--m", "128", "--k", "128", "--local-size", "64",
                    "--warmup", "1", "--iterations", "50",
                    "--output-jsonl", str(jsonl), "--output-f32", str(output),
                ],
                text=True,
                capture_output=True,
                check=False,
                env={**os.environ, "LD_PRELOAD": str(injector)},
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("refusing to replace competing output", completed.stderr)
            self.assertEqual(output.read_bytes(), b"racer")
            self.assertFalse(jsonl.exists())
            self.assertEqual(list(root.glob("*.tmp.*")), [])

    def test_invalid_shape_fails_before_creating_outputs(self):
        with tempfile.TemporaryDirectory(prefix="powervr-opencl-invalid-shape-") as directory:
            root = Path(directory)
            binary = self._build_runner(root)
            weights, activation = self._write_literal_fixture(root)
            jsonl = root / "run.jsonl"
            output = root / "output.f32.bin"
            completed = subprocess.run(
                [
                    str(binary), "--cpu-reference-only",
                    "--weights", str(weights), "--activation", str(activation),
                    "--m", "129", "--k", "128", "--local-size", "64",
                    "--output-jsonl", str(jsonl), "--output-f32", str(output),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("multiples of 128", completed.stderr)
            self.assertFalse(jsonl.exists())
            self.assertFalse(output.exists())

    def test_unapproved_local_size_fails_before_creating_outputs(self):
        with tempfile.TemporaryDirectory(prefix="powervr-opencl-invalid-local-") as directory:
            root = Path(directory)
            binary = self._build_runner(root)
            weights, activation = self._write_literal_fixture(root)
            jsonl = root / "run.jsonl"
            output = root / "output.f32.bin"
            completed = subprocess.run(
                [
                    str(binary), "--cpu-reference-only",
                    "--weights", str(weights), "--activation", str(activation),
                    "--m", "128", "--k", "128", "--local-size", "16",
                    "--output-jsonl", str(jsonl), "--output-f32", str(output),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("local size must be 32, 64, or 128", completed.stderr)
            self.assertFalse(jsonl.exists())
            self.assertFalse(output.exists())

    def test_missing_direct_powervr_runtime_fails_without_publishing_results(self):
        if Path("/usr/lib/libPVROCL.so.1").exists():
            self.skipTest("direct PowerVR runtime is present; target execution is outside this source test")
        with tempfile.TemporaryDirectory(prefix="powervr-opencl-no-runtime-") as directory:
            root = Path(directory)
            binary = self._build_runner(root)
            weights, activation = self._write_literal_fixture(root)
            jsonl = root / "run.jsonl"
            output = root / "output.f32.bin"
            completed = subprocess.run(
                [
                    str(binary), "--weights", str(weights),
                    "--activation", str(activation), "--kernel", str(KERNEL),
                    "--m", "128", "--k", "128", "--warmup", "1",
                    "--iterations", "50", "--local-size", "64",
                    "--output-jsonl", str(jsonl), "--output-f32", str(output),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("/usr/lib/libPVROCL.so.1", completed.stderr)
            self.assertFalse(jsonl.exists())
            self.assertFalse(output.exists())

    def test_gpu_source_has_one_checked_status_path_for_every_opencl_call(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertNotIn("libOpenCL.so", source)
        self.assertIn("#define CL_CALL", source)
        checked_calls = re.findall(
            r"CL_CALL(?:_OR_RETURN)?\((.*?)\);",
            source,
            re.DOTALL,
        )
        for api in (
            "clGetPlatformIDs", "clGetDeviceIDs", "clCreateContext",
            "clCreateCommandQueue", "clCreateProgramWithSource", "clBuildProgram",
            "clCreateKernel", "clCreateBuffer", "clSetKernelArg",
            "clEnqueueWriteBuffer", "clEnqueueNDRangeKernel", "clEnqueueReadBuffer",
            "clFinish", "clGetEventProfilingInfo",
        ):
            self.assertTrue(
                any(re.search(rf"\b{re.escape(api)}\b", call) for call in checked_calls),
                f"{api} must be routed through the fail-closed CL_CALL status check",
            )
        self.assertIn('"-cl-std=CL1.2"', source)
        self.assertIn("explicit_host_expanded_weight_allocation_bytes\\\":0", source)
        self.assertIn("runtime_internal_expansion_observed\\\":\\\"unknown", source)
        for timing_field in (
            "weights_h2d_event_ms", "weights_h2d_host_ms",
            "activation_h2d_event_ms", "activation_h2d_host_ms",
            "d2h_event_ms", "d2h_host_ms", "host_total_ms",
        ):
            self.assertIn(timing_field, source)

    def test_gpu_records_host_wall_and_event_time_for_each_transfer(self):
        source = RUNNER.read_text(encoding="utf-8")
        for phase in ("weights_h2d", "activation_h2d", "d2h"):
            self.assertRegex(source, rf"double {phase}_event_ms = 0\.0;")
            self.assertRegex(source, rf"double {phase}_host_ms = 0\.0;")
            self.assertIn(f"{phase}_host_start_ns = monotonic_ns()", source)
            self.assertIn(f"{phase}_host_end_ns = monotonic_ns()", source)
        self.assertIn(
            "host_total_ms = (double) (host_total_end_ns - host_total_start_ns)",
            source,
        )
        self.assertIn("weights_h2d_host_ms + activation_h2d_host_ms", source)

    def test_readme_states_qualification_timing_and_memory_limits(self):
        readme = README.read_text(encoding="utf-8")
        self.assertIn("warmup >= 1", readme)
        self.assertIn("iterations >= 50", readme)
        self.assertNotIn("--iterations 5 ", readme)
        self.assertIn("explicit_host_expanded_weight_allocation_bytes=0", readme)
        self.assertIn('runtime_internal_expansion_observed="unknown"', readme)
        self.assertIn("host_total_ms", readme)
        self.assertIn("weights H2D", readme)
        self.assertIn("final D2H", readme)
        self.assertIn("5120×5120` — synthetic/non-model", readme)
        self.assertIn("`17408×5120`", readme)
        self.assertIn("`5120×17408`", readme)


if __name__ == "__main__":
    unittest.main()
