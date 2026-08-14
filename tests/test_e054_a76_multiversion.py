import json
import hashlib
import math
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "E054-a76-q1-multiversion"
PATCH = EXPERIMENT / "patches" / "0001-a76-q1-multiversion.patch"


def extract_added_file(patch_text: str, path: str) -> str:
    marker = f"diff --git a/{path} b/{path}\n"
    start = patch_text.index(marker) + len(marker)
    section = patch_text[start:]
    next_diff = section.find("\ndiff --git ")
    if next_diff >= 0:
        section = section[:next_diff]
    lines = section.splitlines()
    body_started = False
    output = []
    for line in lines:
        if line.startswith("@@ "):
            body_started = True
            continue
        if not body_started:
            continue
        if line.startswith("+") and not line.startswith("+++"):
            output.append(line[1:])
        elif line.startswith(" "):
            output.append(line[1:])
    return "\n".join(output) + "\n"


class E054PatchContractTest(unittest.TestCase):
    def test_sensitive_binary_captures_are_externalized(self):
        manifest_path = EXPERIMENT / "data" / "external-sensitive-artifacts.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["schema_version"], "e054-external-sensitive-artifacts/v1"
        )
        entries = manifest["external_sensitive_artifacts"]
        self.assertEqual(len(entries), 46)
        paths = [entry["path"] for entry in entries]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertEqual(
            sum(entry["origin"] == "real_model_derived" for entry in entries), 40
        )
        self.assertEqual(
            sum(entry["origin"] == "synthetic_fixture" for entry in entries), 6
        )
        for entry in entries:
            self.assertTrue(entry["path"].endswith(".bin"))
            self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")
            self.assertGreater(entry["size_bytes"], 0)
            self.assertFalse(entry["published"])
            self.assertEqual(entry["retention"], "local_only")

            local_path = EXPERIMENT / entry["path"]
            if local_path.is_file():
                self.assertEqual(local_path.stat().st_size, entry["size_bytes"])
                self.assertEqual(
                    hashlib.sha256(local_path.read_bytes()).hexdigest(),
                    entry["sha256"],
                )

        tracked = subprocess.run(
            ["git", "ls-files", "-z", "--", str(EXPERIMENT.relative_to(ROOT))],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout.split(b"\0")
        self.assertFalse([path for path in tracked if path.endswith(b".bin")])

        raw_manifest = (
            EXPERIMENT / "data" / "raw-manifest.sha256"
        ).read_text(encoding="utf-8")
        self.assertNotIn(".bin", raw_manifest)

    def test_microgate_runner_has_fixed_pair_cardinality_and_safety(self):
        script = EXPERIMENT / "run_microgate.sh"
        subprocess.run(["bash", "-n", str(script)], check=True)
        text = script.read_text(encoding="utf-8")
        self.assertIn("for pair in 1 2 3 4 5", text)
        self.assertIn("--iterations 50", text)
        self.assertIn("--warmup 1", text)
        self.assertIn("--threads 2", text)
        self.assertIn("taskset -c 6-7", text)
        self.assertIn("--limit-mc 85000", text)
        self.assertIn('GGML_Q1_A76_DISPATCH="$variant"', text)
        self.assertIn('--output-dir "$run_dir/profile/$run_id"', text)
        self.assertIn('--phase-file phases.jsonl', text)

    def test_patch_is_bound_to_pinned_baseline(self):
        manifest = json.loads((EXPERIMENT / "baseline.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["llama_commit"], "38c66ad0241da4f9fcce541cda8edc219086cec5")
        self.assertEqual(
            manifest["files"]["ggml/src/ggml-cpu/arch/arm/repack.cpp"],
            "6a96da05d38f693bcf259ef063c0e4adf762c006a92252fd83133f7cf626b76d",
        )
        self.assertEqual(
            manifest["files"]["ggml/src/ggml-cpu/CMakeLists.txt"],
            "7e17ca2625178776864bc8ecef6f85a346c6536c8556d4a8910b9a133b1690aa",
        )

    def test_patch_has_separate_a76_tu_and_tuning_only_there(self):
        text = PATCH.read_text(encoding="utf-8")
        self.assertIn("ggml-cpu/arch/arm/repack-a76.cpp", text)
        self.assertIn('COMPILE_OPTIONS "-mtune=cortex-a76;-fno-lto"', text)
        self.assertIn("ggml_gemv_q1_0_4x4_q8_0_stock", text)
        self.assertIn("ggml_gemv_q1_0_4x4_q8_0_a76", text)
        self.assertIn('std::getenv("GGML_Q1_A76_DISPATCH")', text)
        self.assertIn("sched_getcpu()", text)
        a76_source = extract_added_file(text, "ggml/src/ggml-cpu/arch/arm/repack-a76.cpp")
        self.assertIn('#include "simd-mappings.h"', a76_source)
        self.assertNotIn("q1_0_4x8", a76_source)

    def test_dispatch_contract_compiles_and_selects_only_cpu6_7(self):
        patch_text = PATCH.read_text(encoding="utf-8")
        header = extract_added_file(
            patch_text, "ggml/src/ggml-cpu/arch/arm/repack-a76.h"
        )
        harness = r'''
#include "repack-a76.h"

static void stock(int, float *, size_t, const void *, const void *, int, int) {}
static void a76(int, float *, size_t, const void *, const void *, int, int) {}

int main() {
    using mode = ggml_q1_a76_dispatch_mode;
    if (ggml_q1_a76_parse_dispatch_mode(nullptr) != mode::stock) return 1;
    if (ggml_q1_a76_parse_dispatch_mode("null") != mode::null_dispatch) return 2;
    if (ggml_q1_a76_parse_dispatch_mode("candidate") != mode::candidate) return 3;
    if (ggml_q1_a76_parse_dispatch_mode("garbage") != mode::stock) return 4;
    for (int cpu = -1; cpu < 12; ++cpu) {
        if (ggml_q1_a76_resolve_kernel(mode::stock, cpu, stock, a76) != stock) return 5;
        if (ggml_q1_a76_resolve_kernel(mode::null_dispatch, cpu, stock, a76) != stock) return 6;
        const auto expected = cpu == 6 || cpu == 7 ? a76 : stock;
        if (ggml_q1_a76_resolve_kernel(mode::candidate, cpu, stock, a76) != expected) return 7;
    }
    return 0;
}
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "repack-a76.h").write_text(header, encoding="utf-8")
            (root / "ggml.h").write_text("#pragma once\n#define GGML_RESTRICT __restrict\n", encoding="utf-8")
            (root / "main.cpp").write_text(harness, encoding="utf-8")
            subprocess.run(
                ["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", str(root),
                 str(root / "main.cpp"), "-o", str(root / "dispatch-test")],
                check=True,
            )
            subprocess.run([str(root / "dispatch-test")], check=True)

    def test_a76_math_keeps_stock_accumulation_sequence(self):
        text = extract_added_file(
            PATCH.read_text(encoding="utf-8"),
            "ggml/src/ggml-cpu/arch/arm/repack-a76.cpp",
        )
        ordered = [
            "ret = vdotq_laneq_s32(ret, signs0, q_tiles, 0);",
            "ret = vdotq_laneq_s32(ret, signs1, q_tiles, 1);",
            "ret = vdotq_laneq_s32(ret, signs2, q_tiles, 2);",
            "ret = vdotq_laneq_s32(ret, signs3, q_tiles, 3);",
            "accb = vfmaq_n_f32(accb, vcvtq_f32_s32(ret), ad);",
            "acc = vfmaq_f32(acc, accb, b_d);",
        ]
        positions = [text.index(item) for item in ordered]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(text.count("vdotq_laneq_s32"), 4)
        self.assertEqual(text.count("ncols_interleaved = 4"), 1)


class E054AnalysisTest(unittest.TestCase):
    @staticmethod
    def document(candidate_us: float = 80.0, pair_count: int = 5):
        pairs = []
        for index in range(pair_count):
            pairs.append(
                {
                    "pair_id": f"p{index + 1:02d}",
                    "stock": {
                        "run_id": f"stock-{index}",
                        "samples_us": [100.0] * 50,
                        "output_f32_sha256": "a" * 64,
                        "activation_q8_sha256": "b" * 64,
                    },
                    "candidate": {
                        "run_id": f"candidate-{index}",
                        "samples_us": [candidate_us] * 50,
                        "output_f32_sha256": "a" * 64,
                        "activation_q8_sha256": "b" * 64,
                    },
                }
            )
        return {
            "schema_version": "e054-a76-microgate/v1",
            "gate": "a76_candidate",
            "minimum_pair_gain_percent": 12.0,
            "pairs": pairs,
        }

    def test_five_exact_pairs_above_twelve_percent_pass(self):
        from tooling.e054_analyze import analyze_document

        result = analyze_document(self.document())
        self.assertEqual(result["decision"], "PASS_FULL_GATE")
        self.assertAlmostEqual(result["median_pair_gain_percent"], 20.0)
        self.assertTrue(result["golden_exact"])

    def test_gain_below_twelve_percent_rejects(self):
        from tooling.e054_analyze import analyze_document

        result = analyze_document(self.document(candidate_us=88.01))
        self.assertEqual(result["decision"], "REJECT_MICROGATE")
        self.assertLess(result["median_pair_gain_percent"], 12.0)

    def test_less_than_five_pairs_is_invalid(self):
        from tooling.e054_analyze import AnalysisError, analyze_document

        with self.assertRaises(AnalysisError):
            analyze_document(self.document(pair_count=4))

    def test_nonfinite_timing_is_invalid(self):
        from tooling.e054_analyze import AnalysisError, analyze_document

        document = self.document()
        document["pairs"][0]["candidate"]["samples_us"][0] = math.nan
        with self.assertRaises(AnalysisError):
            analyze_document(document)

    def test_mismatched_f32_or_q8_golden_is_invalid(self):
        from tooling.e054_analyze import AnalysisError, analyze_document

        for key in ("output_f32_sha256", "activation_q8_sha256"):
            with self.subTest(key=key):
                document = self.document()
                document["pairs"][2]["candidate"][key] = "c" * 64
                with self.assertRaises(AnalysisError):
                    analyze_document(document)


class E054LiteralFixtureTest(unittest.TestCase):
    def test_literal_fixture_has_hand_derived_contract(self):
        from tooling.e054_make_literal_fixture import write_fixture

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fixture"
            manifest = write_fixture(root)
            self.assertEqual(manifest["tensor"]["shape"], [128, 16])
            self.assertEqual((root / "weights.q1_0.bin").stat().st_size, 16 * 18)
            self.assertEqual((root / "activation.f32.bin").stat().st_size, 128 * 4)
            self.assertEqual(
                manifest["expected_output_f32"],
                [-56.0, -48.0, -40.0, -32.0, -24.0, -16.0, -8.0, 0.0,
                 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0, 64.0],
            )
            with self.assertRaises(FileExistsError):
                write_fixture(root)


class E054PlotContractTest(unittest.TestCase):
    def test_plot_contract_requires_five_real_pairs(self):
        from tooling.e054_plot import PlotError, validate_result

        result = json.loads((EXPERIMENT / "data" / "candidate-result.json").read_text())
        rows = validate_result(result)
        self.assertEqual(len(rows), 5)
        broken = dict(result)
        broken["pairs"] = result["pairs"][:1]
        with self.assertRaises(PlotError):
            validate_result(broken)


if __name__ == "__main__":
    unittest.main()
