"""Byte-exact Q1_0 layout conversion for the VIP 16x128 tile."""

QK = 128
M_TILE = 16
Q1_BLOCK_BYTES = 18
TILE_BYTES = 288
MAX_U64 = (1 << 64) - 1


class Q1VipLayoutError(ValueError):
    """Raised when a Q1 VIP layout or input buffer is invalid."""


def _shape(ne0: int, ne1: int) -> tuple[int, int]:
    if type(ne0) is not int or type(ne1) is not int:
        raise Q1VipLayoutError("ne0 и ne1 должны быть целыми")
    if ne0 <= 0 or ne1 <= 0 or ne0 % QK or ne1 % M_TILE:
        raise Q1VipLayoutError("форма должна иметь ne0 % 128 == 0 и ne1 % 16 == 0")
    if ne0 > MAX_U64 or ne1 > MAX_U64:
        raise Q1VipLayoutError("форма выходит за uint64")
    return ne0 // QK, ne1 // M_TILE


def _nbytes(ne0: int, ne1: int, k_blocks: int, m_tiles: int) -> int:
    elements = ne0 * ne1
    if elements > MAX_U64:
        raise Q1VipLayoutError("размер формы выходит за uint64")
    blocks = k_blocks * m_tiles
    if blocks > MAX_U64 or blocks > MAX_U64 // TILE_BYTES:
        raise Q1VipLayoutError("размер буфера выходит за uint64")
    return blocks * TILE_BYTES


def canonical_nbytes(ne0: int, ne1: int) -> int:
    """Return canonical row-major Q1_0 storage size in bytes."""
    k_blocks, m_tiles = _shape(ne0, ne1)
    return _nbytes(ne0, ne1, k_blocks, m_tiles)


def packed_nbytes(ne0: int, ne1: int) -> int:
    """Return packed Q1_VIP_16x128 storage size in bytes."""
    k_blocks, m_tiles = _shape(ne0, ne1)
    return _nbytes(ne0, ne1, k_blocks, m_tiles)


def pack_tensor(source: bytes, ne0: int, ne1: int) -> bytes:
    """Pack canonical row-major Q1_0 blocks into VIP tiles."""
    k_blocks, m_tiles = _shape(ne0, ne1)
    expected = _nbytes(ne0, ne1, k_blocks, m_tiles)
    if not isinstance(source, bytes):
        raise Q1VipLayoutError("данные должны быть bytes")
    if len(source) != expected:
        raise Q1VipLayoutError("размер canonical буфера не совпадает с формой")
    packed = bytearray(expected)
    for m_tile in range(m_tiles):
        row_start = m_tile * M_TILE
        for k_block in range(k_blocks):
            destination = (m_tile * k_blocks + k_block) * TILE_BYTES
            signs_start = destination
            for row in range(M_TILE):
                source_offset = ((row_start + row) * k_blocks + k_block) * Q1_BLOCK_BYTES
                sign_destination = destination + row * (Q1_BLOCK_BYTES - 2)
                packed[sign_destination:sign_destination + Q1_BLOCK_BYTES - 2] = source[
                    source_offset + 2:source_offset + Q1_BLOCK_BYTES
                ]
            scales_start = signs_start + (Q1_BLOCK_BYTES - 2) * M_TILE
            for row in range(M_TILE):
                source_offset = ((row_start + row) * k_blocks + k_block) * Q1_BLOCK_BYTES
                scale_destination = scales_start + row * 2
                packed[scale_destination:scale_destination + 2] = source[
                    source_offset:source_offset + 2
                ]
    return bytes(packed)


def unpack_tensor(packed: bytes, ne0: int, ne1: int) -> bytes:
    """Unpack VIP tiles into canonical row-major Q1_0 blocks."""
    k_blocks, m_tiles = _shape(ne0, ne1)
    expected = _nbytes(ne0, ne1, k_blocks, m_tiles)
    if not isinstance(packed, bytes):
        raise Q1VipLayoutError("данные должны быть bytes")
    if len(packed) != expected:
        raise Q1VipLayoutError("размер packed буфера не совпадает с формой")
    canonical = bytearray(expected)
    for m_tile in range(m_tiles):
        row_start = m_tile * M_TILE
        for k_block in range(k_blocks):
            tile_start = (m_tile * k_blocks + k_block) * TILE_BYTES
            scales_start = tile_start + (Q1_BLOCK_BYTES - 2) * M_TILE
            for row in range(M_TILE):
                destination = ((row_start + row) * k_blocks + k_block) * Q1_BLOCK_BYTES
                sign_start = tile_start + row * (Q1_BLOCK_BYTES - 2)
                canonical[destination + 2:destination + Q1_BLOCK_BYTES] = packed[
                    sign_start:sign_start + Q1_BLOCK_BYTES - 2
                ]
                canonical[destination:destination + 2] = packed[
                    scales_start + row * 2:scales_start + row * 2 + 2
                ]
    return bytes(canonical)
