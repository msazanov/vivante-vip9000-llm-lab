#!/usr/bin/env python3
"""Regression checks for the PowerVR 16 KiB shared-memory patch."""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import tempfile
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
PATCH = REPO_ROOT / "experiments/E006-powervr-vulkan-q1/powervr-q1-shmem.patch"
PINNED_SOURCE = pathlib.Path("/home/random/src/llama-prismml")
VULKAN_SOURCE = "ggml/src/ggml-vulkan/ggml-vulkan.cpp"


class PowervrVulkanShmemPatchTest(unittest.TestCase):
    def test_patch_applies_to_pinned_prismml_and_keeps_q1_path(self) -> None:
        """A future patch must apply cleanly and replace the unconditional abort."""
        self.assertTrue(PATCH.is_file(), f"missing patch: {PATCH}")
        self.assertTrue(
            (PINNED_SOURCE / VULKAN_SOURCE).is_file(),
            f"missing pinned checkout: {PINNED_SOURCE / VULKAN_SOURCE}",
        )

        with tempfile.TemporaryDirectory(prefix="powervr-vulkan-shmem-") as tmp:
            checkout = pathlib.Path(tmp) / "llama-prismml"
            shutil.copytree(PINNED_SOURCE, checkout, symlinks=True)

            check = subprocess.run(
                ["git", "apply", "--check", str(PATCH)],
                cwd=checkout,
                text=True,
                capture_output=True,
            )
            self.assertEqual(check.returncode, 0, check.stderr or check.stdout)

            apply = subprocess.run(
                ["git", "apply", str(PATCH)],
                cwd=checkout,
                text=True,
                capture_output=True,
            )
            self.assertEqual(apply.returncode, 0, apply.stderr or apply.stdout)

            patched = (checkout / VULKAN_SOURCE).read_text(encoding="utf-8")
            self.assertIn("device->mul_mat_s[i] = false;", patched)
            self.assertIn("device->mul_mat_m[i] = false;", patched)
            self.assertIn("device->mul_mat_l[i] = false;", patched)
            self.assertIn("GGML_TYPE_Q1_0", patched)
            self.assertNotIn(
                "throw std::runtime_error(\"Shared memory size too small for matrix multiplication.\");",
                patched,
            )

            diff_check = subprocess.run(
                ["git", "diff", "--check"],
                cwd=checkout,
                text=True,
                capture_output=True,
            )
            self.assertEqual(diff_check.returncode, 0, diff_check.stderr or diff_check.stdout)


if __name__ == "__main__":
    unittest.main()
