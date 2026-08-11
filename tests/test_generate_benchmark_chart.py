import importlib.util
import json
import math
import sys
from dataclasses import replace
import subprocess
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "generate_benchmark_chart", ROOT / "tooling/generate_benchmark_chart.py"
)
chart = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = chart
SPEC.loader.exec_module(chart)


class BenchmarkChartTests(unittest.TestCase):
    def test_real_evidence_selects_comparable_cpu_and_resident_npu(self):
        data = chart.collect_chart_data(
            ROOT / "benchmarks/results/model-runs.jsonl",
            ROOT / "benchmarks/results",
        )
        self.assertEqual(
            sorted({row.run_id for row in data.cpu}),
            [
                "bonsai27b-q1-cpu-a55-pp512-tg128-001",
                "bonsai27b-q1-cpu-a76-pp512-tg128-003",
            ],
        )
        self.assertEqual(len(data.cpu), 4)
        self.assertEqual(len(data.npu), 2)
        self.assertEqual({row.median for row in data.npu}, {2.803, 2.846})
        # E009 fused-FC and E011-E014 Q8-reuse summaries intentionally use
        # experiment-specific schemas and must stay outside the generic chart.
        self.assertEqual(data.out_of_scope, 13)
        cpu_values = {
            (row.run_id, row.metric): (row.p10, row.median, row.p90)
            for row in data.cpu
        }
        self.assertEqual(
            cpu_values[("bonsai27b-q1-cpu-a55-pp512-tg128-001", "prompt_tps")],
            (1.443938, 1.44464, 1.444666),
        )
        self.assertEqual(
            cpu_values[("bonsai27b-q1-cpu-a55-pp512-tg128-001", "decode_tps")],
            (0.7257002, 0.726175, 0.7263472),
        )
        self.assertEqual(
            cpu_values[("bonsai27b-q1-cpu-a76-pp512-tg128-003", "prompt_tps")],
            (1.582594, 1.58287, 1.584104),
        )
        self.assertEqual(
            cpu_values[("bonsai27b-q1-cpu-a76-pp512-tg128-003", "decode_tps")],
            (0.6494914, 0.649836, 0.652178),
        )


def _cpu_row(run_id="cpu-001", *, status="unqualified", samples=None):
    samples = samples or [1.0, 2.0, 3.0, 4.0, 5.0]
    stats = {
        "p10": 1.4,
        "median": 3.0,
        "p90": 4.6,
        "samples": samples,
    }
    return {
        "schema_version": 1,
        "run_id": run_id,
        "status": status,
        "evidence": {"thermal_peak_millidegrees_c": 69089},
        "model": {
            "id": "model", "sha256": "model-sha", "quantization": "q",
            "size_bytes": 1,
        },
        "configuration": {
            "backend": "cpu", "partition": "A55", "context": 512,
            "batch": 512, "ubatch": 512, "threads": 4,
        },
        "software": {
            "repository_commit": "repo", "runtime": "runtime",
            "runtime_commit": "runtime-commit", "compiler": "compiler",
            "sdk": "sdk", "driver": "driver", "kernel": "kernel",
            "command": "command",
        },
        "workload": {
            "tokenizer": "tok", "prompt_suite": "suite", "prompt_tokens": 512,
            "generated_tokens": 128, "seed": 0, "deterministic": True,
            "warmup_iterations": 1,
        },
        "performance": {
            "repetitions": len(samples),
            "statistics": {"prompt_tps": stats, "decode_tps": stats.copy()},
        },
    }


def _npu_row(run_id="npu-001", *, status="performance-observed-unqualified"):
    assets = {field: f"{field}-hash" for field in chart.NPU_ASSET_FIELDS}
    target = {
        field: (1 if field in ("device_count", "logical_core_count") else f"{field}-value")
        for field in chart.NPU_TARGET_FIELDS
    }
    config = {
        "repository_commit": "repo", "profiler_guard_affinity": "CPU 6",
        "host_workload_affinity": "CPU 7", "device_core": 0,
        "npu_hz": 1008000000, "loops": 3, "warmup_loops": 1,
        "measured_loops": 2, "profiler_interval_ms": 10,
        "preload": False, "npd": False, "bypass_output": True,
        "command": "command",
    }
    stat = {"p10": 1000, "median": 1100, "p90": 1200, "samples": 2}
    return {
        "schema_version": "vip9000-capability-run/v1", "run_id": run_id,
        "status": status, "asset_sha256": assets, "target": target,
        "configuration": config, "host_run_us": stat.copy(),
        "device_inference_us": stat.copy(),
        "profiling": {"npu_peak_millidegrees_c": 35650},
    }


def _synthetic_chart_data(*, unique_runs=1, npu_cohorts=1):
    cpu = [
        chart.Series("cpu", "cohort-cpu", "cpu-001", "A55 & A76 <probe>", "prompt_tps", "qualified", "commit-cpu", 69.089, 1.0, 1.2, 1.4),
        chart.Series("cpu", "cohort-cpu", "cpu-001", "A55 & A76 <probe>", "decode_tps", "qualified", "commit-cpu", 69.089, 0.5, 0.649836, 0.8),
    ]
    npu = [
        chart.Series("npu", "cohort-npu", "npu-001", "VIP9000 resident", "host_run_ms", "unqualified", "commit-npu", 35.650, 2.5, 2.846, 3.1),
        chart.Series("npu", "cohort-npu", "npu-001", "VIP9000 resident", "device_inference_ms", "unqualified", "commit-npu", 35.650, 2.2, 2.803, 3.0),
    ]
    for index in range(2, unique_runs + 1):
        cpu.extend(
            chart.Series("cpu", f"cohort-cpu-{index}", f"cpu-{index:03d}", "A55 & A76 <probe>", metric, "unqualified", "commit-cpu", 69.089, 0.5, 0.65, 0.8)
            for metric in ("prompt_tps", "decode_tps")
        )
    for index in range(2, npu_cohorts + 1):
        npu.extend(
            chart.Series("npu", f"cohort-npu-{index}", f"npu-{index:03d}", "VIP9000 alternate", metric, "unqualified", "commit-npu", 35.650, 2.0, 2.4, 2.8)
            for metric in ("host_run_ms", "device_inference_ms")
        )
    return chart.ChartData(
        cpu=tuple(cpu),
        npu=tuple(npu),
        omissions=(("failed_status", 1), ("missing_metrics", 2), ("non_resident", 3), ("singleton_cpu_cohort", 4)),
        out_of_scope=0,
    )


class BenchmarkChartValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.ledger = self.root / "ledger.jsonl"
        self.results = self.root / "results"
        self.results.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def write_ledger(self, *rows):
        self.ledger.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    def write_npu(self, row, directory=None):
        directory = directory or row["run_id"]
        target = self.results / directory
        target.mkdir(parents=True)
        (target / "summary.json").write_text(json.dumps(row), encoding="utf-8")

    def test_duplicate_cpu_run_id_is_fatal(self):
        row = _cpu_row()
        self.write_ledger(row, dict(row))
        with self.assertRaisesRegex(chart.EvidenceError, "duplicate CPU run_id"):
            chart.collect_chart_data(self.ledger, self.results)

    def test_cpu_samples_must_match_repetitions(self):
        row = _cpu_row()
        row["performance"]["repetitions"] = 4
        self.write_ledger(row)
        with self.assertRaisesRegex(chart.EvidenceError, "samples.*repetitions"):
            chart.collect_chart_data(self.ledger, self.results)

    def test_zero_repetitions_with_empty_statistics_is_missing_metrics(self):
        row = _cpu_row()
        row["performance"] = {"repetitions": 0, "statistics": {}}
        self.write_ledger(row)
        data = chart.collect_chart_data(self.ledger, self.results)
        self.assertEqual(dict(data.omissions)["missing_metrics"], 1)

    def test_cpu_schema_scalar_fields_are_strictly_typed(self):
        cases = (
            (("model", "quantization"), {}),
            (("model", "size_bytes"), {}),
            (("configuration", "backend"), {}),
            (("configuration", "partition"), {}),
            (("software", "runtime"), {}),
            (("software", "compiler"), {}),
            (("software", "driver"), {}),
            (("software", "kernel"), {}),
            (("software", "sdk"), {}),
            (("software", "command"), {}),
            (("workload", "tokenizer"), {}),
            (("workload", "prompt_suite"), {}),
        )
        for (section, field), value in cases:
            with self.subTest(section=section, field=field):
                row = _cpu_row()
                row[section][field] = value
                self.write_ledger(row)
                with self.assertRaises(chart.EvidenceError):
                    chart.collect_chart_data(self.ledger, self.results)
                self.ledger.unlink()

    def test_cpu_schema_version_is_a_v1_integer(self):
        row = _cpu_row()
        row["schema_version"] = "1"
        self.write_ledger(row)
        with self.assertRaises(chart.EvidenceError):
            chart.collect_chart_data(self.ledger, self.results)

    def test_npu_resident_string_fields_are_strictly_typed(self):
        cases = (
            ("repository_commit", "configuration"),
            ("profiler_guard_affinity", "configuration"),
            ("host_workload_affinity", "configuration"),
            ("command", "configuration"),
        )
        for field, section in cases:
            with self.subTest(section=section, field=field):
                self.write_ledger()
                row = _npu_row(f"npu-{field}")
                row[section][field] = {}
                self.write_npu(row)
                with self.assertRaises(chart.EvidenceError):
                    chart.collect_chart_data(self.ledger, self.results)
                for child in self.results.iterdir():
                    for item in child.iterdir():
                        item.unlink()
                    child.rmdir()

    def test_duplicate_npu_run_id_is_reported_directly(self):
        self.write_ledger()
        self.write_npu(_npu_row("npu-duplicate"), directory="npu-duplicate")
        self.write_npu(_npu_row("npu-duplicate"), directory="npu-duplicate-copy")
        with self.assertRaisesRegex(chart.EvidenceError, "duplicate NPU run_id"):
            chart.collect_chart_data(self.ledger, self.results)

    def test_failed_records_are_omitted_before_metric_validation(self):
        cpu = _cpu_row(status="failed-before-launch")
        cpu["performance"] = {"repetitions": "not-an-integer"}
        self.write_ledger(cpu)
        npu = _npu_row(status="failed-before-launch")
        npu["configuration"] = {"unexpected": "value"}
        npu.pop("asset_sha256")
        self.write_npu(npu)
        data = chart.collect_chart_data(self.ledger, self.results)
        self.assertEqual(dict(data.omissions)["failed_status"], 2)

    def test_cohorts_are_sorted_by_full_signature(self):
        rows = []
        for seed, prefix in ((0, "a"), (1, "b")):
            for suffix in ("1", "2"):
                row = _cpu_row(f"{prefix}{suffix}")
                row["workload"]["seed"] = seed
                rows.append(row)
        self.write_ledger(*rows)
        data = chart.collect_chart_data(self.ledger, self.results)
        self.assertEqual(
            [row.run_id for row in data.cpu],
            ["a1", "a1", "a2", "a2", "b1", "b1", "b2", "b2"],
        )

    def test_npu_run_id_must_match_directory(self):
        self.write_ledger()
        self.write_npu(_npu_row(), directory="different-directory")
        with self.assertRaisesRegex(chart.EvidenceError, "run_id.*directory"):
            chart.collect_chart_data(self.ledger, self.results)

    def test_npu_missing_measured_loops_is_fatal_for_resident_blocks(self):
        self.write_ledger()
        row = _npu_row()
        del row["configuration"]["measured_loops"]
        self.write_npu(row)
        with self.assertRaisesRegex(chart.EvidenceError, "measured_loops"):
            chart.collect_chart_data(self.ledger, self.results)

    def test_nan_negative_and_misordered_statistics_are_fatal(self):
        for field, value, pattern in (
            ("p10", math.nan, "finite"),
            ("median", -1, "non-negative"),
            ("p90", 0.5, "p10 <= median <= p90"),
        ):
            with self.subTest(field=field):
                row = _cpu_row()
                row["performance"]["statistics"]["prompt_tps"][field] = value
                self.write_ledger(row)
                with self.assertRaisesRegex(chart.EvidenceError, pattern):
                    chart.collect_chart_data(self.ledger, self.results)
                self.ledger.unlink()

    def test_cpu_quantiles_must_match_linear_interpolation(self):
        row = _cpu_row()
        row["performance"]["statistics"]["prompt_tps"]["p10"] = 1.0
        self.write_ledger(row)
        with self.assertRaisesRegex(chart.EvidenceError, "inconsistent"):
            chart.collect_chart_data(self.ledger, self.results)

    def test_partial_resident_statistics_is_fatal(self):
        self.write_ledger()
        row = _npu_row()
        del row["device_inference_us"]
        self.write_npu(row)
        with self.assertRaisesRegex(chart.EvidenceError, "device_inference_us"):
            chart.collect_chart_data(self.ledger, self.results)

    def test_unknown_resident_execution_field_is_fatal(self):
        self.write_ledger()
        row = _npu_row()
        row["configuration"]["new_execution_mode"] = "unexpected"
        self.write_npu(row)
        with self.assertRaisesRegex(chart.EvidenceError, "unknown field"):
            chart.collect_chart_data(self.ledger, self.results)

    def test_nonresident_legacy_variants_are_omitted_without_opening_raw_result(self):
        self.write_ledger()
        for run_id in ("single", "output"):
            row = {
                "schema_version": "vip9000-capability-run/v1", "run_id": run_id,
                "status": "output-captured-unqualified", "raw_result": "../../sentinel.json",
            }
            self.write_npu(row)
        data = chart.collect_chart_data(self.ledger, self.results)
        self.assertEqual(data.npu, ())
        self.assertEqual(dict(data.omissions)["non_resident"], 2)

    def test_temperature_requires_finite_nonnegative_cpu_millidegrees(self):
        for value in (None, True, 69089.5, -1, math.nan, math.inf):
            with self.subTest(value=value):
                row = _cpu_row()
                row["evidence"]["thermal_peak_millidegrees_c"] = value
                self.write_ledger(row, _cpu_row("cpu-002"))
                with self.assertRaises(chart.EvidenceError):
                    chart.collect_chart_data(self.ledger, self.results)
                self.ledger.unlink()

    def test_temperature_requires_finite_nonnegative_npu_millidegrees(self):
        for index, value in enumerate((None, False, 35650.5, -1, math.nan, math.inf)):
            with self.subTest(value=value):
                self.write_ledger()
                row = _npu_row(f"npu-temp-{index}")
                row["profiling"]["npu_peak_millidegrees_c"] = value
                self.write_npu(row)
                with self.assertRaises(chart.EvidenceError):
                    chart.collect_chart_data(self.ledger, self.results)
                for child in self.results.iterdir():
                    for item in child.iterdir():
                        item.unlink()
                    child.rmdir()

    def test_nice_axis_handles_subnormal_positive_values(self):
        for maximum in (5e-324, 1e-320, 1e-309, 1e-100):
            with self.subTest(maximum=maximum):
                ceiling, ticks = chart.nice_axis(maximum)
                self.assertGreater(ceiling, 0.0)
                self.assertGreaterEqual(ceiling, maximum)
                self.assertEqual(ticks[-1], ceiling)

    def test_nice_axis_rejects_nonpositive_or_nonfinite_values(self):
        for maximum in (0.0, -1.0, math.nan, math.inf):
            with self.subTest(maximum=maximum):
                with self.assertRaises(chart.EvidenceError):
                    chart.nice_axis(maximum)

    def test_zero_derived_medians_are_rejected(self):
        data = _synthetic_chart_data()
        zero_decode = replace(data.cpu[1], median=0.0)
        with self.assertRaises(chart.EvidenceError):
            chart.render_svg(replace(data, cpu=(data.cpu[0], zero_decode)))
        zero_npu = replace(data.npu[0], median=0.0)
        with self.assertRaises(chart.EvidenceError):
            chart.render_svg(replace(data, npu=(zero_npu, data.npu[1])))


class BenchmarkChartRenderTests(unittest.TestCase):
    def test_real_temperature_metadata_is_preserved(self):
        data = chart.collect_chart_data(
            ROOT / "benchmarks/results/model-runs.jsonl",
            ROOT / "benchmarks/results",
        )
        self.assertEqual(
            {row.temperature_c for row in data.cpu},
            {69.089, 63.612},
        )
        self.assertEqual({row.temperature_c for row in data.npu}, {35.650})

    def test_svg_has_visible_structural_contract(self):
        data = chart.ChartData(
            cpu=(
                chart.Series("cpu", "cohort-cpu", "cpu-001", "A55 & A76 <probe>", "prompt_tps", "qualified", "commit-cpu", 69.089, 1.0, 1.2, 1.4),
                chart.Series("cpu", "cohort-cpu", "cpu-001", "A55 & A76 <probe>", "decode_tps", "qualified", "commit-cpu", 69.089, 0.5, 0.649836, 0.8),
            ),
            npu=(
                chart.Series("npu", "cohort-npu", "npu-001", "VIP9000 resident", "host_run_ms", "unqualified", "commit-npu", 35.650, 2.5, 2.846, 3.1),
                chart.Series("npu", "cohort-npu", "npu-001", "VIP9000 resident", "device_inference_ms", "unqualified", "commit-npu", 35.650, 2.2, 2.803, 3.0),
            ),
            omissions=(("failed_status", 1), ("missing_metrics", 2), ("non_resident", 3), ("singleton_cpu_cohort", 4)),
            out_of_scope=0,
        )
        payload = chart.render_svg(data)
        root = ET.fromstring(payload)
        text = "".join(root.itertext())
        self.assertIn("CPU: пропускная способность LLM — больше лучше", text)
        self.assertIn("VIP9000: задержка резидентного запуска — меньше лучше", text)
        self.assertIn("токенов/с", text)
        self.assertIn("мс", text)
        self.assertEqual(len(root.findall(".//*[@data-role='median-bar']")), 4)
        self.assertEqual(len(root.findall(".//*[@data-role='whisker']")), 4)
        self.assertEqual(len(root.findall(".//*[@data-role='whisker-cap']")), 8)
        self.assertEqual(len(root.findall(".//*[@data-role='median-label']")), 4)
        self.assertIn("0.650 ток/с · 1.539 с/ток", text)
        self.assertIn("2.846 мс · 351.37 инф/с", text)
        self.assertIn("пик CPU 69.1 °C", text)
        self.assertIn("пик NPU 35.6 °C", text)

    def test_svg_is_deterministic_and_has_dynamic_height(self):
        data = _synthetic_chart_data(unique_runs=3)
        first = chart.render_svg(data)
        second = chart.render_svg(data)
        self.assertEqual(first, second)
        root = ET.fromstring(first)
        self.assertEqual(root.attrib["width"], "1200")
        self.assertEqual(root.attrib["height"], "972")
        self.assertTrue(first.endswith(b"\n"))

    def test_npu_cohort_headers_and_separator_are_visible(self):
        data = _synthetic_chart_data(npu_cohorts=2)
        root = ET.fromstring(chart.render_svg(data))
        npu = root.find(".//*[@data-panel='npu']")
        self.assertIsNotNone(npu)
        self.assertEqual(len(npu.findall(".//*[@data-role='cohort-header']")), 2)
        self.assertEqual(len(npu.findall(".//*[@data-role='cohort-separator']")), 1)

    def test_svg_literal_p10_median_p90_geometry(self):
        root = ET.fromstring(chart.render_svg(_synthetic_chart_data()))
        groups = {
            group.attrib["data-metric"]: group
            for group in root.findall(".//*[@data-metric]")
            if "data-status" in group.attrib
        }
        expected = {
            "prompt_tps": (756.666667, 822.0, 887.333333, 392.0),
            "decode_tps": (593.333333, 642.279467, 691.333333, 212.279467),
            "host_run_ms": (736.25, 778.635, 809.75, 348.635),
            "device_inference_ms": (699.5, 773.3675, 797.5, 343.3675),
        }
        for metric, (p10_x, median_x, p90_x, width) in expected.items():
            with self.subTest(metric=metric):
                group = groups[metric]
                bar = group.find("./*[@data-role='median-bar']")
                whisker = group.find("./*[@data-role='whisker']")
                caps = group.findall("./*[@data-role='whisker-cap']")
                self.assertAlmostEqual(float(bar.attrib["x"]), 430.0, places=5)
                self.assertAlmostEqual(float(bar.attrib["width"]), width, places=3)
                self.assertAlmostEqual(float(whisker.attrib["x1"]), p10_x, places=3)
                self.assertAlmostEqual(float(whisker.attrib["x2"]), p90_x, places=3)
                self.assertAlmostEqual(float(caps[0].attrib["x1"]), p10_x, places=3)
                self.assertAlmostEqual(float(caps[1].attrib["x1"]), p90_x, places=3)
                self.assertAlmostEqual(float(bar.attrib["x"]) + float(bar.attrib["width"]), median_x, places=3)

    def test_svg_title_and_desc_are_nonempty_russian_accessibility_text(self):
        root = ET.fromstring(chart.render_svg(_synthetic_chart_data()))
        namespace = "{http://www.w3.org/2000/svg}"
        title = root.find(f"{namespace}title")
        desc = root.find(f"{namespace}desc")
        self.assertIsNotNone(title)
        self.assertIsNotNone(desc)
        self.assertRegex(title.text or "", r"[А-Яа-яЁё]")
        self.assertRegex(desc.text or "", r"[А-Яа-яЁё]")

    def test_svg_footer_has_exact_omissions_and_relative_sources(self):
        root = ET.fromstring(chart.render_svg(_synthetic_chart_data()))
        footer = root.find(".//*[@data-role='footer']")
        self.assertIsNotNone(footer)
        text = "".join(footer.itertext())
        self.assertIn("failed_status=1", text)
        self.assertIn("missing_metrics=2", text)
        self.assertIn("non_resident=3", text)
        self.assertIn("singleton_cpu_cohort=4", text)
        self.assertIn("out_of_scope=0", text)
        self.assertIn("benchmarks/results/model-runs.jsonl", text)
        self.assertIn("benchmarks/results/*/summary.json", text)

    def test_svg_npu_caveat_and_dom_order_are_stable(self):
        data = _synthetic_chart_data(unique_runs=3, npu_cohorts=2)
        root = ET.fromstring(chart.render_svg(data))
        npu = root.find(".//*[@data-panel='npu']")
        self.assertIsNotNone(npu)
        self.assertIn("Накладные расходы настройки и процесса, а также корректность не представлены этими столбцами.", "".join(npu.itertext()))
        groups = [
            (item.attrib["data-panel"], item.attrib["data-cohort-id"], item.attrib["data-run-id"], item.attrib["data-metric"])
            for item in root.findall(".//*[@data-metric]")
            if "data-status" in item.attrib
        ]
        self.assertEqual(
            groups,
            [
                ("cpu", "cohort-cpu", "cpu-001", "prompt_tps"),
                ("cpu", "cohort-cpu", "cpu-001", "decode_tps"),
                ("cpu", "cohort-cpu-2", "cpu-002", "prompt_tps"),
                ("cpu", "cohort-cpu-2", "cpu-002", "decode_tps"),
                ("cpu", "cohort-cpu-3", "cpu-003", "prompt_tps"),
                ("cpu", "cohort-cpu-3", "cpu-003", "decode_tps"),
                ("npu", "cohort-npu", "npu-001", "host_run_ms"),
                ("npu", "cohort-npu", "npu-001", "device_inference_ms"),
                ("npu", "cohort-npu-2", "npu-002", "host_run_ms"),
                ("npu", "cohort-npu-2", "npu-002", "device_inference_ms"),
            ],
        )

    def test_large_cpu_layout_keeps_rows_at_least_72px_apart(self):
        data = _synthetic_chart_data(unique_runs=30)
        root = ET.fromstring(chart.render_svg(data))
        self.assertEqual(root.attrib["height"], "2916")
        cpu = root.find(".//*[@data-panel='cpu']")
        run_y = sorted(float(item.attrib["y"]) for item in cpu.findall(".//*[@data-role='run-id']"))
        self.assertEqual(len(run_y), 30)
        self.assertGreaterEqual(min(b - a for a, b in zip(run_y, run_y[1:])), 72.0)
        bars_by_run = {}
        for group in cpu.findall(".//*[@data-metric]"):
            if "data-status" not in group.attrib:
                continue
            bar = group.find("./*[@data-role='median-bar']")
            bars_by_run.setdefault(group.attrib["data-run-id"], []).append(float(bar.attrib["y"]) + 6.0)
        centers = [sorted(bars_by_run[f"cpu-{index:03d}"]) for index in range(1, 31)]
        self.assertTrue(all(len(values) == 2 for values in centers))
        self.assertGreaterEqual(min(next_values[0] - values[-1] for values, next_values in zip(centers, centers[1:])), 54.0)

    def test_renderer_preserves_first_appearance_cohort_order(self):
        data = _synthetic_chart_data(unique_runs=2)
        cpu = tuple(replace(item, cohort_id="z-first" if item.run_id == "cpu-001" else "a-second") for item in data.cpu)
        root = ET.fromstring(chart.render_svg(replace(data, cpu=cpu)))
        cpu_groups = [
            item.attrib["data-cohort-id"]
            for item in root.findall(".//*[@data-metric]")
            if item.attrib.get("data-panel") == "cpu" and "data-status" in item.attrib
        ]
        self.assertEqual(cpu_groups[:2], ["z-first", "z-first"])
        self.assertEqual(cpu_groups[2:], ["a-second", "a-second"])

    def test_svg_copy_is_russian_and_status_star_is_visible(self):
        data = _synthetic_chart_data()
        rejected = replace(data.cpu[0], status="rejected")
        root = ET.fromstring(chart.render_svg(replace(data, cpu=(rejected, data.cpu[1]))))
        text = "".join(root.itertext())
        for forbidden in ("throughput", "prompt", "decode", "Setup", "process overhead", "solid"):
            self.assertNotIn(forbidden.casefold(), text.casefold())
        for expected in ("промпт", "декод", "хост", "устройство", "сплошн", "пунктир", "Накладные расходы"):
            self.assertIn(expected.casefold(), text.casefold())
        rejected_labels = [
            group.find("./*[@data-role='median-label']").text
            for group in root.findall(".//*[@data-metric]")
            if group.get("data-status") == "rejected"
        ]
        self.assertTrue(rejected_labels)
        self.assertTrue(all("*" in label for label in rejected_labels))

    def test_svg_contains_full_run_id_text_nodes(self):
        root = ET.fromstring(chart.render_svg(_synthetic_chart_data(npu_cohorts=2)))
        ids = {item.text for item in root.findall(".//*[@data-role='run-id']")}
        self.assertIn("cpu-001", ids)
        self.assertIn("npu-001", ids)
        self.assertIn("npu-002", ids)
        for item in root.findall(".//*[@data-role='run-id']"):
            self.assertEqual(item.text, item.attrib["data-run-id"])

    def test_zero_cpu_prompt_median_is_rendered(self):
        data = _synthetic_chart_data()
        zero_prompt = replace(data.cpu[0], p10=0.0, median=0.0, p90=0.0)
        payload = chart.render_svg(replace(data, cpu=(zero_prompt, data.cpu[1])))
        self.assertIn("0.000 ток/с", "".join(ET.fromstring(payload).itertext()))

    def test_svg_geometry_status_provenance_footer_and_order_contract(self):
        data = _synthetic_chart_data(unique_runs=3)
        payload = chart.render_svg(data)
        root = ET.fromstring(payload)
        bars = root.findall(".//*[@data-role='median-bar']")
        self.assertEqual(len(bars), 8)
        self.assertTrue(any("stroke-dasharray" in item.attrib for item in bars))
        qualified_bars = [
            group.find("./*[@data-role='median-bar']")
            for group in root.findall(".//*[@data-metric]")
            if group.attrib["data-status"] == "qualified"
        ]
        self.assertFalse(any("stroke-dasharray" in item.attrib for item in qualified_bars))
        for group in root.findall(".//*[@data-metric]"):
            self.assertEqual(
                set(("data-panel", "data-cohort-id", "data-run-id", "data-metric", "data-status")),
                set(group.attrib) & {"data-panel", "data-cohort-id", "data-run-id", "data-metric", "data-status"},
            )
            whisker = group.find("./*[@data-role='whisker']")
            p10 = group.find("./*[@data-role='whisker-cap']")
            p90 = group.findall("./*[@data-role='whisker-cap']")[1]
            median = group.find("./*[@data-role='median-bar']")
            self.assertLessEqual(float(p10.attrib["x1"]), float(median.attrib["x"]) + float(median.attrib["width"]))
            self.assertEqual(float(whisker.attrib["x1"]), float(p10.attrib["x1"]))
            self.assertEqual(float(whisker.attrib["x2"]), float(p90.attrib["x1"]))
        colors = {
            group.attrib["data-metric"]: group.find("./*[@data-role='median-bar']").attrib["fill"]
            for group in root.findall(".//*[@data-metric]")
        }
        self.assertEqual(len(set(colors.values())), 4)
        cpu_prompt = root.find(".//*[@data-metric='prompt_tps']/*[@data-role='median-bar']")
        self.assertAlmostEqual(float(cpu_prompt.attrib["x"]), 430.0)
        self.assertAlmostEqual(float(cpu_prompt.attrib["width"]), 392.0, places=3)
        self.assertNotIn(str(ROOT), payload.decode("utf-8"))
        text = "".join(root.itertext())
        self.assertIn("data-repository-commit", payload.decode("utf-8"))
        self.assertIn("benchmarks/results/model-runs.jsonl", text)
        self.assertIn("benchmarks/results/*/summary.json", text)
        self.assertIn("cohort-separator", payload.decode("utf-8"))
        self.assertIn("промпт", text)
        self.assertIn("декод", text)
        self.assertIn("хост", text)
        self.assertIn("устройство", text)
        for label in root.findall(".//*[@data-role='median-label']"):
            self.assertLessEqual(float(label.attrib["x"]) + float(label.attrib["textLength"]), 1200.0)
        for run_id in root.findall(".//*[@data-role='run-id']"):
            self.assertLessEqual(float(run_id.attrib["x"]) + float(run_id.attrib["textLength"]), 430.0)


class BenchmarkChartCliTests(unittest.TestCase):
    SCRIPT = ROOT / "tooling/generate_benchmark_chart.py"
    LEDGER = ROOT / "benchmarks/results/model-runs.jsonl"
    RESULTS = ROOT / "benchmarks/results"

    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, str(self.SCRIPT), *map(str, args)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_default_write_then_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "charts" / "overview.svg"
            common = ("--ledger", self.LEDGER, "--results-dir", self.RESULTS, "--output", output)
            first = self.run_cli(*common)
            second = self.run_cli(*common)
            self.assertEqual(first.returncode, 0)
            self.assertEqual(second.returncode, 2)
            self.assertIn("ошибка:", second.stderr)
            self.assertTrue(output.is_file())

    def test_check_current_stale_missing_and_malformed(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "overview.svg"
            common = ("--ledger", self.LEDGER, "--results-dir", self.RESULTS, "--output", output)
            self.assertEqual(self.run_cli(*common, "--force").returncode, 0)
            self.assertEqual(self.run_cli(*common, "--check").returncode, 0)
            output.write_bytes(b"stale\n")
            self.assertEqual(self.run_cli(*common, "--check").returncode, 1)
            output.unlink()
            self.assertEqual(self.run_cli(*common, "--check").returncode, 1)
            malformed = Path(directory) / "bad.jsonl"
            malformed.write_text("not json\n", encoding="utf-8")
            missing_parent = Path(directory) / "missing" / "overview.svg"
            result = self.run_cli("--ledger", malformed, "--results-dir", self.RESULTS, "--output", missing_parent, "--check")
            self.assertEqual(result.returncode, 2)
            self.assertFalse(missing_parent.parent.exists())

    def test_force_replaces_and_cleans_sibling_temp(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "overview.svg"
            common = ("--ledger", self.LEDGER, "--results-dir", self.RESULTS, "--output", output)
            self.assertEqual(self.run_cli(*common, "--force").returncode, 0)
            old_inode = output.stat().st_ino
            output.write_bytes(b"old")
            self.assertEqual(self.run_cli(*common, "--force").returncode, 0)
            self.assertNotEqual(output.stat().st_ino, old_inode)
            self.assertEqual(list(output.parent.glob(f".{output.name}.*.tmp")), [])

    def test_force_and_check_are_mutually_exclusive(self):
        result = self.run_cli("--force", "--check")
        self.assertEqual(result.returncode, 2)

    def test_cli_allows_zero_cpu_prompt_median(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = root / "model-runs.jsonl"
            results = root / "results"
            shutil.copytree(self.RESULTS, results)
            rows = [json.loads(line) for line in self.LEDGER.read_text(encoding="utf-8").splitlines()]
            for row in rows:
                performance = row.get("performance")
                if not isinstance(performance, dict) or performance.get("repetitions", 0) <= 0:
                    continue
                prompt = performance["statistics"]["prompt_tps"]
                prompt.update({"p10": 0.0, "median": 0.0, "p90": 0.0, "samples": [0.0] * performance["repetitions"]})
                performance["prompt_tps"] = 0.0
            ledger.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            output = root / "charts" / "overview.svg"
            result = self.run_cli("--ledger", ledger, "--results-dir", results, "--output", output, "--force")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("0.000 ток/с", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
