import unittest


from tooling.e023_q1_carrier_adapter import (
    E022_Q1_BLOCK_BYTES,
    GGML_Q1_BLOCK_BYTES,
    e022_tile_supported,
    repack_ggml_q1_rows_to_e022,
)


class E023Q1CarrierAdapterTest(unittest.TestCase):
    def test_one_standard_block_moves_scale_after_sign_bytes(self):
        standard = bytes.fromhex("003c 0123456789abcdef 1020304050607080")
        converted = repack_ggml_q1_rows_to_e022(standard, rows=1, k=128)
        self.assertEqual(len(converted), E022_Q1_BLOCK_BYTES)
        self.assertEqual(converted[:16], standard[2:18])
        self.assertEqual(converted[16:18], standard[:2])

    def test_multiple_rows_preserve_each_block_without_expansion(self):
        rows = 4
        k = 256
        row_bytes = (k // 8) + 2 * (k // 128)
        standard = b"".join(
            bytes((row * 17 + i) & 0xFF for i in range(row_bytes))
            for row in range(rows)
        )
        original = bytes(standard)
        converted = repack_ggml_q1_rows_to_e022(standard, rows=rows, k=k)
        self.assertEqual(len(converted), len(standard))
        self.assertEqual(standard, original)
        for row in range(rows):
            start = row * row_bytes
            for block in range(k // 128):
                src = standard[start + block * GGML_Q1_BLOCK_BYTES:start + (block + 1) * GGML_Q1_BLOCK_BYTES]
                dst = converted[start + block * E022_Q1_BLOCK_BYTES:start + (block + 1) * E022_Q1_BLOCK_BYTES]
                self.assertEqual(dst, src[2:] + src[:2])

    def test_invalid_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            repack_ggml_q1_rows_to_e022(bytes(18), rows=1, k=64)
        with self.assertRaises(ValueError):
            repack_ggml_q1_rows_to_e022(bytes(17), rows=1, k=128)

    def test_tile_contract_is_narrow_and_explicit(self):
        self.assertTrue(e022_tile_supported(rows=1024, k=5120))
        self.assertTrue(e022_tile_supported(rows=4, k=5120))
        self.assertFalse(e022_tile_supported(rows=1028, k=5120))
        self.assertFalse(e022_tile_supported(rows=1024, k=17408))

    def test_reordered_block_keeps_q1_q8_dot_exact(self):
        scale = (0x3C00).to_bytes(2, "little")  # FP16 1.0
        signs = bytes((0x00, 0xFF, 0x55, 0xAA) * 4)
        standard = scale + signs
        converted = repack_ggml_q1_rows_to_e022(standard, rows=1, k=128)
        q8 = tuple((-128 + (i * 37) % 256) for i in range(128))
        expected = sum(
            (q8[i] if (signs[i // 8] >> (i % 8)) & 1 else -q8[i])
            for i in range(128)
        )
        selected = sum(
            q8[i] if (converted[i // 8] >> (i % 8)) & 1 else 0
            for i in range(128)
        )
        total = sum(q8)
        self.assertEqual(2 * selected - total, expected)


if __name__ == "__main__":
    unittest.main()
