import unittest

from tooling.q1_vip_layout import (
    Q1VipLayoutError,
    canonical_nbytes,
    pack_tensor,
    packed_nbytes,
    unpack_tensor,
)


class Q1VipLayoutTests(unittest.TestCase):
    def test_production_shapes_preserve_q1_byte_count(self):
        self.assertEqual(canonical_nbytes(5120, 17408), 12_533_760)
        self.assertEqual(packed_nbytes(5120, 17408), 12_533_760)
        self.assertEqual(canonical_nbytes(17408, 5120), 12_533_760)
        self.assertEqual(packed_nbytes(17408, 5120), 12_533_760)

    def test_pack_one_tile_writes_all_signs_before_scales(self):
        blocks = []
        signs = []
        scales = []
        for row in range(16):
            scale = bytes((row, 255 - row))
            sign = bytes((row * 16 + i) & 255 for i in range(16))
            blocks.append(scale + sign)
            signs.append(sign)
            scales.append(scale)
        self.assertEqual(pack_tensor(b"".join(blocks), 128, 16),
                         b"".join(signs + scales))

    def test_pack_two_tiles_orders_each_k_block_inside_m_tile(self):
        blocks = []
        expected_tiles = []
        for m_tile in range(2):
            for row in range(16):
                absolute_row = m_tile * 16 + row
                for k_block in range(2):
                    scale = bytes((absolute_row, k_block))
                    sign = bytes((absolute_row * 2 + k_block,)) * 16
                    blocks.append(scale + sign)
            tile_blocks = []
            for k_block in range(2):
                tile_signs = []
                tile_scales = []
                for row in range(16):
                    absolute_row = m_tile * 16 + row
                    tile_signs.append(bytes((absolute_row * 2 + k_block,)) * 16)
                    tile_scales.append(bytes((absolute_row, k_block)))
                tile_blocks.append(b"".join(tile_signs + tile_scales))
            expected_tiles.append(b"".join(tile_blocks))
        self.assertEqual(pack_tensor(b"".join(blocks), 256, 32),
                         b"".join(expected_tiles))

    def test_unpack_round_trip_preserves_canonical_bytes(self):
        source = bytes((i * 37) & 255 for i in range(32 * 2 * 18))
        self.assertEqual(unpack_tensor(pack_tensor(source, 256, 32), 256, 32), source)

    def test_invalid_dimensions_fail_closed(self):
        for ne0, ne1 in ((0, 16), (128, 0), (127, 16), (128, 15),
                         (True, 16), (128, False)):
            with self.subTest(ne0=ne0, ne1=ne1):
                with self.assertRaises(Q1VipLayoutError):
                    canonical_nbytes(ne0, ne1)

    def test_input_lengths_must_match_exact_shape(self):
        for source in (b"\0" * 287, b"\0" * 289):
            with self.subTest(length=len(source)):
                with self.assertRaises(Q1VipLayoutError):
                    pack_tensor(source, 128, 16)
                with self.assertRaises(Q1VipLayoutError):
                    unpack_tensor(source, 128, 16)

    def test_dimension_product_beyond_uint64_fails_closed(self):
        with self.assertRaises(Q1VipLayoutError):
            canonical_nbytes(1 << 60, 16)
