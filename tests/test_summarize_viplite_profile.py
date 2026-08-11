import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tooling" / "summarize_viplite_profile.py"


VALID_LOG = """\
VIPLite driver software version 2.0.3.2-AW-2024-08-30
phase,name=init,time_us=1000.000
phase,name=create_network,time_us=2000.000
network,inputs=1,outputs=1
tensor,kind=input,index=0,format=2,quant=2,dims=2,shape=4x2,scale=0.5,zero_point=3
buffer,kind=input,index=0,bytes=8
tensor,kind=output,index=0,format=2,quant=2,dims=1,shape=2,scale=0.25,zero_point=128
buffer,kind=output,index=0,bytes=2
phase,name=create_buffers,time_us=300.000
phase,name=prepare,time_us=400.000
iteration,index=0,kind=first,h2d_us=8.000,run_us=40.000,d2h_us=2.000,device_profile_status=0,device_us=30,cycles=30000,repeat_equal=1
iteration,index=1,kind=steady,h2d_us=4.000,run_us=36.000,d2h_us=1.000,device_profile_status=0,device_us=30,cycles=30001,repeat_equal=1
iteration,index=2,kind=steady,h2d_us=6.000,run_us=42.000,d2h_us=3.000,device_profile_status=0,device_us=32,cycles=32000,repeat_equal=0
iteration,index=3,kind=steady,h2d_us=8.000,run_us=48.000,d2h_us=5.000,device_profile_status=0,device_us=34,cycles=34000,repeat_equal=1
"""


class SummarizeVIPLiteProfileTests(unittest.TestCase):
    def invoke(self, text: str, *args: str) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stdout.log"
            path.write_text(text, encoding="utf-8")
            return subprocess.run(
                [sys.executable, str(SCRIPT), str(path), *args],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

    def test_summarizes_first_and_steady_phases_without_calling_repeat_a_golden(self) -> None:
        completed = self.invoke(VALID_LOG, "--run-id", "synthetic-001")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)

        self.assertEqual(result["schema_version"], "vip9000-viplite-phase-profile/v1")
        self.assertEqual(result["status"], "performance-observed-unqualified")
        self.assertEqual(result["run_id"], "synthetic-001")
        self.assertEqual(result["driver_software"], "2.0.3.2-AW-2024-08-30")
        self.assertEqual(result["iterations"], {"first": 1, "steady": 3, "total": 4})
        self.assertEqual(result["bytes"], {"h2d": 8, "d2h": 2})
        self.assertEqual(result["setup_us"]["prepare"], 400.0)
        self.assertEqual(result["stats_us"]["first"]["end_to_end"]["median"], 50.0)
        self.assertEqual(result["stats_us"]["steady"]["h2d"], {
            "min": 4.0, "median": 6.0, "p95": 7.8, "max": 8.0,
        })
        self.assertEqual(result["stats_us"]["steady"]["run_minus_device"]["median"], 10.0)
        self.assertEqual(result["stats_us"]["steady"]["end_to_end_minus_device"]["median"], 19.0)
        self.assertAlmostEqual(result["bandwidth_gbps"]["steady"]["h2d_at_median"], 0.0013333333333333333)
        self.assertEqual(result["correctness"], {
            "golden_checked": False,
            "repeat_equal_failures": 1,
            "repeat_equal_is_not_golden": True,
        })
        self.assertEqual(result["device_profile_failures"], 0)
        self.assertEqual(result["outputs"][0]["zero_point"], 128)

    def test_rejects_missing_or_non_contiguous_iterations_and_profile_errors(self) -> None:
        missing = VALID_LOG.replace("iteration,index=2,", "iteration,index=4,")
        completed = self.invoke(missing)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("contiguous", completed.stderr.lower())

        profile_error = VALID_LOG.replace("device_profile_status=0", "device_profile_status=-1", 1)
        completed = self.invoke(profile_error)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("profil", completed.stderr.lower())

    def test_rejects_malformed_numeric_fields_and_inconsistent_kind(self) -> None:
        malformed = VALID_LOG.replace("h2d_us=8.000", "h2d_us=nan", 1)
        completed = self.invoke(malformed)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("number", completed.stderr.lower())

        wrong_kind = VALID_LOG.replace("iteration,index=1,kind=steady", "iteration,index=1,kind=first")
        completed = self.invoke(wrong_kind)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("kind", completed.stderr.lower())

    def test_accepts_scientific_notation_emitted_by_percent_g(self) -> None:
        scientific = VALID_LOG.replace("scale=0.5", "scale=1e-05")
        completed = self.invoke(scientific)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["inputs"][0]["scale"], 1e-05)

    def test_output_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "stdout.log"
            output = Path(temporary) / "summary.json"
            log.write_text(VALID_LOG, encoding="utf-8")
            command = [sys.executable, str(SCRIPT), str(log), "--output", str(output)]
            first = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
            second = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("overwrite", second.stderr.lower())

    def test_merges_ordered_thermal_and_npu_clock_evidence(self) -> None:
        samples = [
            {"kind": "sample", "sample_seq": 0, "monotonic_ns": 10,
             "thermal_zones": [{"type": "npu_thermal_zone", "millidegrees_c": 35000}],
             "cooling_devices": [{"type": "pwm-fan", "cur_state": 4},
                                  {"type": "devfreq-3600000.npu", "cur_state": 0}]},
            {"kind": "sample", "sample_seq": 1, "monotonic_ns": 20,
             "thermal_zones": [{"type": "npu_thermal_zone", "millidegrees_c": 42000}],
             "cooling_devices": [{"type": "pwm-fan", "cur_state": 4},
                                  {"type": "devfreq-3600000.npu", "cur_state": 0}]},
            {"kind": "sample", "sample_seq": 2, "monotonic_ns": 30,
             "thermal_zones": [{"type": "npu_thermal_zone", "millidegrees_c": 41000}],
             "cooling_devices": [{"type": "pwm-fan", "cur_state": 4},
                                  {"type": "devfreq-3600000.npu", "cur_state": 0}]},
        ]
        telemetry = [
            {"monotonic_ns": 5, "system": {"npu_frequencies": {"3600000.npu_hz": 492000000}}},
            {"monotonic_ns": 10, "system": {"npu_frequencies": {"3600000.npu_hz": 1008000000}}},
            {"monotonic_ns": 20, "system": {"npu_frequencies": {"3600000.npu_hz": 1008000000}}},
            {"monotonic_ns": 35, "system": {"npu_frequencies": {"3600000.npu_hz": 492000000}}},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log = root / "stdout.log"
            guard = root / "thermal-guard.jsonl"
            telemetry_path = root / "telemetry.jsonl"
            log.write_text(VALID_LOG, encoding="utf-8")
            guard.write_text("".join(json.dumps(row) + "\n" for row in samples), encoding="utf-8")
            telemetry_path.write_text(
                "".join(json.dumps(row) + "\n" for row in telemetry), encoding="utf-8"
            )
            completed = subprocess.run(
                [sys.executable, str(SCRIPT), str(log), "--thermal-guard", str(guard),
                 "--telemetry", str(telemetry_path)],
                cwd=ROOT, text=True, capture_output=True, check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual(result["thermal"], {
            "clock_variation_evidence": False,
            "npu_clock_max_hz": 1008000000,
            "npu_clock_min_hz": 1008000000,
            "npu_cooling_max_state": 0,
            "npu_end_c": 41.0,
            "npu_peak_c": 42.0,
            "npu_start_c": 35.0,
            "pwm_fan_min_state": 4,
            "sample_count": 3,
            "throttling_evidence": False,
        })


if __name__ == "__main__":
    unittest.main()
