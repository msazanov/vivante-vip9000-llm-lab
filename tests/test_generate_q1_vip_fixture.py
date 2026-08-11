import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tooling" / "generate_q1_vip_fixture.py"
FILES = ("q1_canonical.bin", "q1_vip.bin", "q8.bin", "expected_f32.bin", "manifest.json")


class GenerateQ1VipFixtureTests(unittest.TestCase):
    def run_generator(self, output_dir: Path, *extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--output-dir", str(output_dir), *extra],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )

    def test_generates_deterministic_canonical_fixture_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as parent:
            first = Path(parent) / "first"
            second = Path(parent) / "second"
            first_run = self.run_generator(first)
            second_run = self.run_generator(second)
            self.assertEqual(first_run.returncode, 0, first_run.stderr)
            self.assertEqual(second_run.returncode, 0, second_run.stderr)
            self.assertEqual(sorted(path.name for path in first.iterdir()), sorted(FILES))
            self.assertEqual(sorted(path.name for path in second.iterdir()), sorted(FILES))
            self.assertEqual(
                {name: (first / name).read_bytes() for name in FILES},
                {name: (second / name).read_bytes() for name in FILES},
            )

            expected_sizes = {
                "q1_canonical.bin": 288,
                "q1_vip.bin": 288,
                "q8.bin": 136,
                "expected_f32.bin": 64,
                "manifest.json": 0,
            }
            for name, size in expected_sizes.items():
                if name != "manifest.json":
                    self.assertEqual((first / name).stat().st_size, size)

            expected_hashes = {
                "q1_canonical.bin": "29b15bffce3cc00e2cbcec6048e13fe63a6b783715670a0e69a6d817ac3becde",
                "q1_vip.bin": "0f820597d5981d013bf24b9fba8308d580456b0838c3b628846f498e1c045d00",
                "q8.bin": "f277e2ead476201039aab761b20111443cbde1e1be78a5740a01060f52394a0b",
                "expected_f32.bin": "9c907b6b3558da8b756e142f95630060ecc9a1923714dd17206366271cd5b491",
            }
            for name, digest in expected_hashes.items():
                self.assertEqual(hashlib.sha256((first / name).read_bytes()).hexdigest(), digest)

            manifest_bytes = (first / "manifest.json").read_bytes()
            self.assertTrue(manifest_bytes.endswith(b"\n"))
            self.assertNotIn(b"\r", manifest_bytes)
            self.assertEqual(manifest_bytes[-2:-1], b"}")
            self.assertEqual(manifest_bytes, manifest_bytes.decode("utf-8").encode("utf-8"))
            manifest = json.loads(manifest_bytes.decode("utf-8"))
            self.assertEqual(manifest["schema"], "q1-vip-c0-fixture/v1")
            self.assertEqual(manifest["shape"], {"K": 128, "M": 16})
            self.assertEqual(manifest["layout"], "Q1_VIP_16x128/v1")
            self.assertEqual(manifest["types"], {"q1": "Q1_0", "q8": "Q8_0", "expected": "FP32"})
            self.assertEqual(manifest["expected_output_count"], 16)
            self.assertRegex(manifest["generator_repository_commit"], r"^[0-9a-f]{40}$")
            self.assertEqual(
                manifest["files"],
                {
                    name: {"sha256": digest, "size_bytes": (first / name).stat().st_size}
                    for name, digest in expected_hashes.items()
                },
            )
            self.assertEqual(list(manifest), sorted(manifest))
            encoded = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            self.assertEqual(manifest_bytes.decode("utf-8"), encoded)

    def test_missing_output_dir_argument_fails(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT)], cwd=ROOT, text=True, capture_output=True
        )
        self.assertNotEqual(completed.returncode, 0)

    def test_existing_output_directory_fails_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as parent:
            output = Path(parent) / "existing"
            output.mkdir()
            sentinel = output / "sentinel"
            sentinel.write_bytes(b"keep")
            completed = self.run_generator(output)
            self.assertNotEqual(completed.returncode, 0)
            self.assertEqual(sentinel.read_bytes(), b"keep")
            self.assertEqual(sorted(path.name for path in output.iterdir()), ["sentinel"])

    def test_symlink_output_path_fails_without_writing_target(self) -> None:
        with tempfile.TemporaryDirectory() as parent:
            root = Path(parent)
            target = root / "target"
            target.mkdir()
            sentinel = target / "sentinel"
            sentinel.write_bytes(b"keep")
            link = root / "link"
            link.symlink_to(target, target_is_directory=True)
            completed = self.run_generator(link)
            self.assertNotEqual(completed.returncode, 0)
            self.assertEqual(sentinel.read_bytes(), b"keep")
            self.assertEqual(sorted(path.name for path in target.iterdir()), ["sentinel"])

    def test_partially_existing_target_fails_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as parent:
            output = Path(parent) / "partial"
            output.mkdir()
            existing = output / "q1_canonical.bin"
            existing.write_bytes(b"keep")
            completed = self.run_generator(output)
            self.assertNotEqual(completed.returncode, 0)
            self.assertEqual(existing.read_bytes(), b"keep")
            self.assertEqual(sorted(path.name for path in output.iterdir()), ["q1_canonical.bin"])


if __name__ == "__main__":
    unittest.main()
