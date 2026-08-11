import hashlib
from pathlib import Path
import subprocess
import struct
import sys
import tempfile
import unittest

from tooling.q1_vip_layout import pack_tensor
from tooling.q1_vip_golden import (
    dot_canonical_block,
    dot_canonical_matrix,
    dot_packed_tile,
)


def q8_block(scale_bits: bytes, values: list[int]) -> bytes:
    return scale_bits + bytes(v & 255 for v in values)


ROOT = Path(__file__).resolve().parents[1]


class Q1VipGoldenTests(unittest.TestCase):
    def test_dot_canonical_matrix(self):
        # Keep this expected-value path deliberately explicit: the matrix
        # implementation must agree with two independently accumulated rows.
        q1_rows = [
            b"".join([
                bytes.fromhex("003c") + b"\xff" * 16,
                bytes.fromhex("0040") + b"\x55" * 16,
            ]),
            b"".join([
                bytes.fromhex("b83a") + b"\x00" * 16,
                bytes.fromhex("0038") + b"\xaa" * 16,
            ]),
        ]
        q1 = b"".join(q1_rows)
        q8 = b"".join([
            q8_block(bytes.fromhex("003c"), [1, -2, 3, -4] * 8),
            q8_block(bytes.fromhex("0040"), [-5, 6, -7, 8] * 8),
            q8_block(bytes.fromhex("0038"), [9, -10, 11, -12] * 8),
            q8_block(bytes.fromhex("00bc"), [13, -14, 15, -16] * 8),
            q8_block(bytes.fromhex("0040"), [-16, 15, -14, 13] * 8),
            q8_block(bytes.fromhex("0038"), [-12, 11, -10, 9] * 8),
            q8_block(bytes.fromhex("00bc"), [8, -7, 6, -5] * 8),
            q8_block(bytes.fromhex("003c"), [4, -3, 2, -1] * 8),
        ])

        expected = []
        for row in q1_rows:
            row_total = 0.0
            for block in range(2):
                q1_offset = block * 18
                q8_offset = block * 136
                row_total += dot_canonical_block(
                    row[q1_offset:q1_offset + 18],
                    q8[q8_offset:q8_offset + 136],
                )
            expected.append(row_total)

        self.assertEqual(dot_canonical_matrix(q1, q8, 256, 2), tuple(expected))

        invalid_cases = (
            (q1[:-1], q8, 256, 2),
            (q1, q8[:-1], 256, 2),
            (q1, q8 + b"\0", 256, 2),
            (q1, q8, 255, 2),
            (q1, q8, 256, 0),
            (q1, q8, 256, -1),
            (q1 + b"\0", q8, 256, 2),
        )
        for candidate_q1, candidate_q8, ne0, ne1 in invalid_cases:
            with self.subTest(ne0=ne0, ne1=ne1, q1_len=len(candidate_q1), q8_len=len(candidate_q8)):
                with self.assertRaises(ValueError):
                    dot_canonical_matrix(candidate_q1, candidate_q8, ne0, ne1)

        bad_q1 = bytearray(q1)
        bad_q1[0:2] = bytes.fromhex("007c")
        with self.assertRaises(ValueError):
            dot_canonical_matrix(bytes(bad_q1), q8, 256, 2)
        bad_q8 = bytearray(q8)
        bad_q8[0:2] = bytes.fromhex("00fe")
        with self.assertRaises(ValueError):
            dot_canonical_matrix(q1, bytes(bad_q8), 256, 2)

    def test_matrix_cli_writes_exclusive_little_endian_f32_and_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "weights.q1_0.bin"
            q8 = root / "activation.q8_0.bin"
            output = root / "expected.f32.bin"
            valid_weights = (
                bytes.fromhex("003c") + b"\xff" * 16
                + bytes.fromhex("0040") + b"\x00" * 16
            )
            valid_q8 = b"".join(q8_block(bytes.fromhex("003c"), [1] * 32) for _ in range(4))
            weights.write_bytes(valid_weights)
            q8.write_bytes(valid_q8)
            expected = struct.pack("<2f", 128.0, -256.0)

            command = [
                sys.executable,
                str(ROOT / "tooling" / "q1_vip_golden.py"),
                "--weights", str(weights),
                "--q8", str(q8),
                "--ne0", "128",
                "--ne1", "2",
                "--output-f32", str(output),
            ]
            completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(output.read_bytes(), expected)
            self.assertEqual(
                completed.stdout.strip(),
                f"output_sha256={hashlib.sha256(expected).hexdigest()}",
            )

            output.write_bytes(b"keep-me")
            rejected = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertEqual(output.read_bytes(), b"keep-me")

            invalid_inputs = (
                ("short-weights", weights, valid_weights[:-1]),
                ("extra-weights", weights, valid_weights + b"\0"),
                ("short-q8", q8, valid_q8[:-1]),
                ("extra-q8", q8, valid_q8 + b"\0"),
            )
            for name, input_path, payload in invalid_inputs:
                candidate = root / f"{name}.f32.bin"
                weights.write_bytes(valid_weights)
                q8.write_bytes(valid_q8)
                input_path.write_bytes(payload)
                invalid = subprocess.run(
                    [*command[:-1], str(candidate)],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(invalid.returncode, 0)
                self.assertFalse(candidate.exists())

    def test_four_q8_scales_are_applied_independently(self):
        q1 = bytes.fromhex("003c") + b"\xff" * 16
        q8 = b"".join([
            q8_block(bytes.fromhex("003c"), [1] * 32),
            q8_block(bytes.fromhex("0040"), [1] * 32),
            q8_block(bytes.fromhex("0038"), [1] * 32),
            q8_block(bytes.fromhex("00bc"), [1] * 32),
        ])
        self.assertEqual(dot_canonical_block(q1, q8), 80.0)

    def test_all_plus_q8_raw_128_is_signed_minus_128(self):
        q1 = bytes.fromhex("003c") + b"\xff" * 16
        q8 = b"".join(q8_block(bytes.fromhex("003c"), [128] * 32) for _ in range(4))
        self.assertEqual(dot_canonical_block(q1, q8), -16384.0)

    def test_all_minus_q8_raw_128_is_signed_minus_128(self):
        q1 = bytes.fromhex("003c") + b"\x00" * 16
        q8 = b"".join(q8_block(bytes.fromhex("003c"), [128] * 32) for _ in range(4))
        self.assertEqual(dot_canonical_block(q1, q8), 16384.0)

    def test_alternating_signs_with_zero_q8(self):
        q1 = bytes.fromhex("003c") + b"\x55" * 16
        q8 = b"".join(q8_block(bytes.fromhex("003c"), [0] * 32) for _ in range(4))
        self.assertEqual(dot_canonical_block(q1, q8), 0.0)

    def test_alternating_signs_use_lsb_first_order(self):
        q1 = bytes.fromhex("003c") + b"\x55" * 16
        q8 = b"".join(q8_block(bytes.fromhex("003c"), [1, -1] * 16) for _ in range(4))
        self.assertEqual(dot_canonical_block(q1, q8), 128.0)

    def test_q1_scale_two_doubles_the_independent_scale_golden(self):
        q1 = bytes.fromhex("0040") + b"\xff" * 16
        q8 = b"".join([
            q8_block(bytes.fromhex("003c"), [1] * 32),
            q8_block(bytes.fromhex("0040"), [1] * 32),
            q8_block(bytes.fromhex("0038"), [1] * 32),
            q8_block(bytes.fromhex("00bc"), [1] * 32),
        ])
        self.assertEqual(dot_canonical_block(q1, q8), 160.0)

    def test_q8_endpoint_values_have_signed_byte_semantics(self):
        q1 = bytes.fromhex("003c") + b"\xff" * 16
        q8 = b"".join([
            q8_block(bytes.fromhex("003c"), [-128] * 32),
            q8_block(bytes.fromhex("003c"), [127] * 32),
            q8_block(bytes.fromhex("003c"), [0] * 32),
            q8_block(bytes.fromhex("003c"), [127, -128] * 16),
        ])
        self.assertEqual(dot_canonical_block(q1, q8), -48.0)

    def test_packed_tile_matches_each_canonical_row(self):
        patterns = [
            b"\xff" * 16,
            b"\x00" * 16,
            b"\x55" * 16,
            b"\xaa" * 16,
            bytes([0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80]) * 2,
        ]
        rows = [bytes.fromhex("003c") + patterns[row % len(patterns)] for row in range(16)]
        canonical = b"".join(rows)
        q8 = b"".join([
            q8_block(bytes.fromhex("003c"), [1, -2, 3, -4] * 8),
            q8_block(bytes.fromhex("0040"), [-5, 6, -7, 8] * 8),
            q8_block(bytes.fromhex("0038"), [9, -10, 11, -12] * 8),
            q8_block(bytes.fromhex("00bc"), [13, -14, 15, -16] * 8),
        ])
        tile = pack_tensor(canonical, 128, 16)

        self.assertEqual(
            dot_packed_tile(tile, q8),
            tuple(dot_canonical_block(row, q8) for row in rows),
        )

    def test_scalar_rejects_non_exact_lengths(self):
        with self.assertRaises(ValueError):
            dot_canonical_block(b"\0" * 17, b"\0" * 136)
        with self.assertRaises(ValueError):
            dot_canonical_block(b"\0" * 18, b"\0" * 135)

    def test_packed_rejects_non_exact_lengths(self):
        with self.assertRaises(ValueError):
            dot_packed_tile(b"\0" * 287, b"\0" * 136)
        with self.assertRaises(ValueError):
            dot_packed_tile(b"\0" * 288, b"\0" * 135)

    def test_nan_and_inf_fp16_scales_are_rejected(self):
        finite_q1 = bytes.fromhex("003c") + b"\xff" * 16
        finite_q8 = b"".join(q8_block(bytes.fromhex("003c"), [1] * 32) for _ in range(4))
        for invalid_scale in (
            bytes.fromhex("007c"), bytes.fromhex("00fc"),
            bytes.fromhex("007e"), bytes.fromhex("00fe"),
        ):
            with self.subTest(invalid_scale=invalid_scale):
                with self.assertRaises(ValueError):
                    dot_canonical_block(invalid_scale + finite_q1[2:], finite_q8)
                bad_q8 = invalid_scale + finite_q8[2:]
                with self.assertRaises(ValueError):
                    dot_canonical_block(finite_q1, bad_q8)

        tile = pack_tensor(b"".join([finite_q1] * 16), 128, 16)
        bad_tile = bytearray(tile)
        bad_tile[272:274] = bytes.fromhex("007c")
        with self.assertRaises(ValueError):
            dot_packed_tile(bytes(bad_tile), finite_q8)


if __name__ == "__main__":
    unittest.main()
