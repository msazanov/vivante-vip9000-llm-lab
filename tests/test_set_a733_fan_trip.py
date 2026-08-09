from pathlib import Path
import os
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "tooling" / "set_a733_fan_trip.sh"
UNIT = ROOT / "tooling" / "systemd" / "a733-fan-trip-30c.service"


class FanSysfsFixture:
    def __init__(
        self,
        root: Path,
        *,
        include_cpub: bool = True,
        fan_type: str = "pwm-fan",
        duplicate_cpub: bool = False,
        multiple_fan_trips: bool = False,
        trip0_value: str = "55000",
    ):
        self.root = root
        thermal = root / "class" / "thermal"
        thermal.mkdir(parents=True)
        zones = (("thermal_zone7", "cpub_thermal_zone"), ("thermal_zone12", "cpul_thermal_zone"))
        if duplicate_cpub:
            zones += (("thermal_zone99", "cpub_thermal_zone"),)
        for zone_name, zone_type in zones:
            if zone_type == "cpub_thermal_zone" and not include_cpub:
                continue
            zone = thermal / zone_name
            zone.mkdir()
            (zone / "type").write_text(zone_type + "\n")
            (zone / "trip_point_0_type").write_text("passive\n")
            (zone / "trip_point_0_temp").write_text(trip0_value + "\n")
            (zone / "trip_point_1_type").write_text("passive\n")
            (zone / "trip_point_1_temp").write_text("90000\n")
            (zone / "trip_point_2_type").write_text("critical\n")
            (zone / "trip_point_2_temp").write_text("110000\n")
            cooling = thermal / f"cooling_device_{zone_name}"
            cooling.mkdir()
            (cooling / "type").write_text(fan_type + "\n")
            (zone / "cdev0").symlink_to(Path("..") / cooling.name)
            (zone / "cdev0_trip_point").write_text("0\n")

            cpu_cooling = thermal / f"cpu_cooling_{zone_name}"
            cpu_cooling.mkdir()
            (cpu_cooling / "type").write_text("cpufreq\n")
            (zone / "cdev1").symlink_to(Path("..") / cpu_cooling.name)
            (zone / "cdev1_trip_point").write_text("1\n")

            if multiple_fan_trips:
                second_fan = thermal / f"second_fan_{zone_name}"
                second_fan.mkdir()
                (second_fan / "type").write_text("pwm-fan\n")
                (zone / "cdev2").symlink_to(Path("..") / second_fan.name)
                (zone / "cdev2_trip_point").write_text("1\n")


class SetA733FanTripTests(unittest.TestCase):
    def invoke(
        self,
        root: Path,
        *,
        args: list[str] | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["A733_FAN_TRIP_TEST_SYSFS_ROOT"] = str(root)
        if extra_env:
            environment.update(extra_env)
        return subprocess.run(
            ["bash", str(HELPER), *(args or [])],
            cwd=ROOT,
            text=True,
            capture_output=True,
            env=environment,
            check=False,
        )

    def assert_trip_values(self, root: Path, zone: str, *, trip0: str = "55000") -> None:
        zone_root = root / "class" / "thermal" / zone
        self.assertEqual((zone_root / "trip_point_0_temp").read_text().strip(), trip0)
        self.assertEqual((zone_root / "trip_point_1_temp").read_text().strip(), "90000")
        self.assertEqual((zone_root / "trip_point_2_temp").read_text().strip(), "110000")

    def test_sets_only_the_pwm_fan_passive_trip_in_both_cpu_zones(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sys"
            FanSysfsFixture(root)

            completed = self.invoke(root)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            verified = self.invoke(root, args=["--verify"])
            self.assertEqual(verified.returncode, 0, verified.stderr)
            for zone in ("thermal_zone7", "thermal_zone12"):
                self.assert_trip_values(root, zone, trip0="30000")

    def test_bounded_wait_ready_mode_checks_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sys"
            FanSysfsFixture(root)

            completed = self.invoke(root, args=["--wait-ready"])

            self.assertEqual(completed.returncode, 0, completed.stderr)
            for zone in ("thermal_zone7", "thermal_zone12"):
                self.assert_trip_values(root, zone)

    def test_refuses_to_write_when_a_cpu_zone_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sys"
            FanSysfsFixture(root, include_cpub=False)

            completed = self.invoke(root)

            self.assertNotEqual(completed.returncode, 0)
            self.assertNotIn("30000", completed.stdout)
            self.assert_trip_values(root, "thermal_zone12")

    def test_refuses_to_write_when_cpu_trip_is_not_bound_to_pwm_fan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sys"
            FanSysfsFixture(root, fan_type="cpu-cooling")

            completed = self.invoke(root)

            self.assertNotEqual(completed.returncode, 0)
            for zone in ("thermal_zone7", "thermal_zone12"):
                self.assert_trip_values(root, zone)

    def test_rolls_back_every_original_value_after_write_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sys"
            FanSysfsFixture(root)

            completed = self.invoke(root, extra_env={"A733_FAN_TRIP_TEST_FAIL_WRITE_INDEX": "1"})

            self.assertNotEqual(completed.returncode, 0)
            for zone in ("thermal_zone7", "thermal_zone12"):
                self.assert_trip_values(root, zone)

    def test_rolls_back_both_original_values_after_readback_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sys"
            FanSysfsFixture(root)

            completed = self.invoke(root, extra_env={"A733_FAN_TRIP_TEST_READBACK_ERROR_INDEX": "0"})

            self.assertNotEqual(completed.returncode, 0)
            for zone in ("thermal_zone7", "thermal_zone12"):
                self.assert_trip_values(root, zone)

    def test_rolls_back_both_original_values_after_readback_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sys"
            FanSysfsFixture(root)

            completed = self.invoke(root, extra_env={"A733_FAN_TRIP_TEST_FAIL_READBACK_INDEX": "0"})

            self.assertNotEqual(completed.returncode, 0)
            for zone in ("thermal_zone7", "thermal_zone12"):
                self.assert_trip_values(root, zone)

    def test_warns_when_fault_injected_rollback_is_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sys"
            FanSysfsFixture(root)

            completed = self.invoke(
                root,
                extra_env={
                    "A733_FAN_TRIP_TEST_FAIL_READBACK_INDEX": "0",
                    "A733_FAN_TRIP_TEST_FAIL_ROLLBACK_INDEX": "0",
                },
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("WARNING: rollback was incomplete", completed.stderr)
            self.assert_trip_values(root, "thermal_zone7", trip0="30000")
            self.assert_trip_values(root, "thermal_zone12")

    def test_refuses_non_numeric_original_trip_before_any_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sys"
            FanSysfsFixture(root, trip0_value="not-a-number")
            (root / "class" / "thermal" / "thermal_zone12" / "trip_point_0_temp").write_text("55000\n")

            completed = self.invoke(root)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("not numeric", completed.stderr)
            self.assert_trip_values(root, "thermal_zone7", trip0="not-a-number")
            self.assert_trip_values(root, "thermal_zone12")

    def test_refuses_duplicate_cpu_zone_before_any_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sys"
            FanSysfsFixture(root, duplicate_cpub=True)

            completed = self.invoke(root)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("multiple cpub_thermal_zone", completed.stderr)
            self.assert_trip_values(root, "thermal_zone7")
            self.assert_trip_values(root, "thermal_zone12")

    def test_refuses_multiple_pwm_fan_trips_before_any_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sys"
            FanSysfsFixture(root, multiple_fan_trips=True)

            completed = self.invoke(root)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("exactly one passive pwm-fan trip", completed.stderr)
            for zone in ("thermal_zone7", "thermal_zone12"):
                self.assert_trip_values(root, zone)

    def test_production_hook_guard_and_unit_environment_contract(self) -> None:
        helper = HELPER.read_text()
        unit = UNIT.read_text()
        hooks = (
            "A733_FAN_TRIP_TEST_SYSFS_ROOT",
            "A733_FAN_TRIP_TEST_FAIL_WRITE_INDEX",
            "A733_FAN_TRIP_TEST_READBACK_ERROR_INDEX",
            "A733_FAN_TRIP_TEST_FAIL_READBACK_INDEX",
            "A733_FAN_TRIP_TEST_FAIL_ROLLBACK_INDEX",
        )

        self.assertIn("/usr/local/sbin/*)", helper)
        self.assertIn("A733_*_TEST_*", helper)
        self.assertIn("for attempt in 1 2 3 4 5", helper)
        self.assertIn("sleep 1", helper)
        self.assertIn("UnsetEnvironment=" + " ".join(hooks), unit)

    def test_systemd_unit_is_root_oneshot_with_exact_helper_and_ordering(self) -> None:
        unit = UNIT.read_text()

        self.assertIn("Type=oneshot", unit)
        self.assertIn("User=root", unit)
        self.assertIn("Group=root", unit)
        self.assertIn("ExecStart=/usr/local/sbin/set_a733_fan_trip.sh", unit)
        self.assertIn("After=systemd-modules-load.service", unit)
        self.assertIn("Before=multi-user.target", unit)
        self.assertIn("WantedBy=multi-user.target", unit)
        self.assertIn("RemainAfterExit=yes", unit)
        self.assertIn("NoNewPrivileges=yes", unit)
        self.assertIn("ProtectSystem=strict", unit)
        self.assertIn("ReadWritePaths=/sys/class/thermal", unit)
        self.assertIn("ProtectHome=yes", unit)
        self.assertIn("PrivateTmp=yes", unit)
        self.assertIn("ExecStartPre=/usr/local/sbin/set_a733_fan_trip.sh --wait-ready", unit)
        self.assertIn("TimeoutStartSec=8s", unit)
