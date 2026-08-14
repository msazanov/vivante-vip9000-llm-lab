from __future__ import annotations

import fcntl
import os
import pathlib
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
HEADER = ROOT / "patches" / "llama.cpp" / "38c66ad" / "e049d-steady-pmu.h"
PATCH = ROOT / "patches" / "llama.cpp" / "38c66ad" / "e049d-steady-pmu.patch"


HARNESS = r"""
#include "e049d-steady-pmu.h"

#include <cstdint>
#include <cstdlib>
#include <cstdio>

int main() {
    const char * ack_fd = std::getenv("E049D_TEST_ACK_FD");
    const char * marker_fd = std::getenv("E049D_TEST_MARKER_FD");
    if (ack_fd != nullptr && marker_fd != nullptr) {
        if (dup2(std::atoi(ack_fd), 8) < 0 || dup2(std::atoi(marker_fd), 9) < 0) {
            return 4;
        }
    }
    const uint32_t token_counts[] = { 4, 1, 1, 1, 1, 1 };
    e049d_steady_pmu gate;
    for (uint32_t n_tokens : token_counts) {
        bool measured = false;
        if (!gate.before_graph(n_tokens, measured)) {
            return 2;
        }
        std::printf("%d ", measured ? 1 : 0);
        if (!gate.after_graph(measured)) {
            return 3;
        }
    }
    std::printf("\n");
    return 0;
}
"""


class E049dSteadyPmuTest(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(HEADER.is_file(), f"missing production header: {HEADER}")
        self.tmp = tempfile.TemporaryDirectory()
        self.work = pathlib.Path(self.tmp.name)
        source = self.work / "harness.cpp"
        source.write_text(HARNESS, encoding="utf-8")
        self.binary = self.work / "harness"
        subprocess.run(
            [
                "c++",
                "-std=c++11",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-I",
                str(HEADER.parent),
                str(source),
                "-o",
                str(self.binary),
            ],
            check=True,
            text=True,
            capture_output=True,
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_env_absent_runs_without_marker_fds_or_measured_window(self) -> None:
        env = os.environ.copy()
        env.pop("LLAMA_E049D_STEADY_PMU", None)
        proc = subprocess.run(
            [str(self.binary)], env=env, text=True, capture_output=True, timeout=5
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout, "0 0 0 0 0 0 \n")

    def test_patch_places_gate_immediately_around_graph_compute(self) -> None:
        self.assertTrue(PATCH.is_file(), f"missing integration patch: {PATCH}")
        patch = PATCH.read_text(encoding="utf-8")
        before = patch.index("e049d_pmu.before_graph")
        compute = patch.index("const auto status = graph_compute")
        after = patch.index("e049d_pmu.after_graph")
        self.assertLess(before, compute)
        self.assertLess(compute, after)
        self.assertIn('#include "e049d-steady-pmu.h"', patch)
        self.assertIn("e049d_steady_pmu e049d_pmu;", patch)

    def test_enabled_gate_skips_first_single_decode_and_marks_next_three(self) -> None:
        ack_read, ack_write = os.pipe()
        marker_read, marker_write = os.pipe()
        ack_read_high = fcntl.fcntl(ack_read, fcntl.F_DUPFD, 100)
        marker_write_high = fcntl.fcntl(marker_write, fcntl.F_DUPFD, 100)
        os.close(ack_read)
        os.close(marker_write)

        env = os.environ.copy()
        env["LLAMA_E049D_STEADY_PMU"] = "1"
        env["E049D_TEST_ACK_FD"] = str(ack_read_high)
        env["E049D_TEST_MARKER_FD"] = str(marker_write_high)
        try:
            proc = subprocess.Popen(
                [str(self.binary)],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                pass_fds=(ack_read_high, marker_write_high),
            )
            os.close(ack_read_high)
            os.close(marker_write_high)
            self.assertEqual(os.read(marker_read, 1), b"S")
            self.assertEqual(os.write(ack_write, b"A"), 1)
            self.assertEqual(os.read(marker_read, 1), b"E")
            stdout, stderr = proc.communicate(timeout=5)
            self.assertEqual(proc.returncode, 0, stderr)
            self.assertEqual(stdout, "0 0 1 1 1 0 \n")
            self.assertEqual(os.read(marker_read, 1), b"")
        finally:
            for fd in (ack_write, marker_read, ack_read_high, marker_write_high):
                try:
                    os.close(fd)
                except OSError:
                    pass


if __name__ == "__main__":
    unittest.main()
