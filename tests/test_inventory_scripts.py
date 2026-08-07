from pathlib import Path
import os
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
TARGET_INVENTORY = ROOT / "tooling" / "target_inventory.sh"
ACUITY_INVENTORY = ROOT / "tooling" / "inspect_acuitylite.sh"


class InventoryScriptTests(unittest.TestCase):
    def test_target_inventory_writes_sanitized_files_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "target"
            first = subprocess.run(
                ["bash", str(TARGET_INVENTORY), "--output-dir", str(output_dir)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertIn("[uname]", (output_dir / "system.txt").read_text())
            self.assertIn("[npu-devfreq]", (output_dir / "system.txt").read_text())
            self.assertTrue((output_dir / "runtime-files.sha256").is_file())
            before = {
                "system.txt": (output_dir / "system.txt").read_bytes(),
                "runtime-files.sha256": (output_dir / "runtime-files.sha256").read_bytes(),
            }

            second = subprocess.run(
                ["bash", str(TARGET_INVENTORY), "--output-dir", str(output_dir)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("refusing to overwrite", second.stderr)
            self.assertEqual((output_dir / "system.txt").read_bytes(), before["system.txt"])
            self.assertEqual(
                (output_dir / "runtime-files.sha256").read_bytes(),
                before["runtime-files.sha256"],
            )

    def test_target_inventory_mandatory_failure_leaves_no_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "target"
            command_stub_dir = root / "bin"
            command_stub_dir.mkdir()
            uname_stub = command_stub_dir / "uname"
            uname_stub.write_text("#!/bin/sh\nprintf 'uname failed\\n' >&2\nexit 23\n")
            uname_stub.chmod(0o755)
            environment = {**os.environ, "PATH": f"{command_stub_dir}:{os.environ['PATH']}"}

            completed = subprocess.run(
                ["bash", str(TARGET_INVENTORY), "--output-dir", str(output_dir)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                env=environment,
                check=False,
            )

            self.assertEqual(completed.returncode, 23, completed.stderr)
            self.assertFalse(output_dir.exists())
            self.assertEqual(list(root.glob("target.tmp.*")), [])

    def test_acuity_inspector_invokes_an_isolated_read_only_container(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docker_stub = root / "docker-stub"
            output = root / "acuity.txt"
            docker_stub.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n")
            docker_stub.chmod(0o755)

            completed = subprocess.run(
                [
                    "bash",
                    str(ACUITY_INVENTORY),
                    "--docker-bin",
                    str(docker_stub),
                    "--image",
                    "acuity-test:1",
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            arguments = output.read_text().splitlines()
            self.assertEqual(arguments[0], "run")
            self.assertIn("--rm", arguments)
            self.assertIn("--network", arguments)
            self.assertIn("none", arguments)
            self.assertIn("--read-only", arguments)
            self.assertIn("--tmpfs", arguments)
            self.assertIn("--cap-drop", arguments)
            self.assertIn("ALL", arguments)
            self.assertIn("--security-opt", arguments)
            self.assertIn("no-new-privileges", arguments)
            self.assertIn("--pids-limit", arguments)
            self.assertIn("128", arguments)
            self.assertIn("--memory", arguments)
            self.assertIn("1g", arguments)
            self.assertIn("acuity-test:1", arguments)

    def test_acuity_rejects_malicious_image_before_invoking_docker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docker_marker = root / "docker-invoked"
            docker_stub = root / "docker-stub"
            output = root / "acuity.txt"
            docker_stub.write_text(
                f"#!/bin/sh\nprintf invoked > {docker_marker}\nexit 0\n"
            )
            docker_stub.chmod(0o755)

            completed = subprocess.run(
                [
                    "bash",
                    str(ACUITY_INVENTORY),
                    "--docker-bin",
                    str(docker_stub),
                    "--image",
                    "--privileged",
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(output.exists())
            self.assertFalse(docker_marker.exists())

    def test_acuity_rejects_empty_docker_binary_before_invoking_docker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "acuity.txt"
            completed = subprocess.run(
                [
                    "bash",
                    str(ACUITY_INVENTORY),
                    "--docker-bin",
                    "",
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(output.exists())

    def test_acuity_failure_propagates_and_leaves_requested_output_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docker_stub = root / "docker-stub"
            output = root / "acuity.txt"
            docker_stub.write_text("#!/bin/sh\nprintf partial\nexit 37\n")
            docker_stub.chmod(0o755)

            completed = subprocess.run(
                [
                    "bash",
                    str(ACUITY_INVENTORY),
                    "--docker-bin",
                    str(docker_stub),
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 37, completed.stderr)
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob("acuity.txt.tmp.*")), [])

    def test_acuity_requests_license_hash_and_redacted_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docker_stub = root / "docker-stub"
            output = root / "acuity.txt"
            docker_stub.write_text("#!/bin/sh\nprintf 'CONFIDENTIAL-LICENSE-TEXT\\n'\nexit 0\n")
            docker_stub.chmod(0o755)

            completed = subprocess.run(
                [
                    "bash",
                    str(ACUITY_INVENTORY),
                    "--docker-bin",
                    str(docker_stub),
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("CONFIDENTIAL-LICENSE-TEXT", output.read_text())
            source = ACUITY_INVENTORY.read_text()
            self.assertNotIn('cat "$sdk_root/vsi_sdk/licence.txt"', source)
            self.assertIn('sha256sum "$license_file"', source)
            self.assertIn("<redacted>", source)


if __name__ == "__main__":
    unittest.main()
