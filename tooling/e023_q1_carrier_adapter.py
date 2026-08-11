"""Format-safe adapter between GGML Q1_0 and the E022 VIP carrier.

The standard GGML Q1_0 block is ``[fp16 scale][16 packed sign bytes]``.
The E022 diagnostic NBG intentionally uses ``[16 sign bytes][fp16 scale]``.
Both representations are 18 bytes per 128 weights, so this adapter only
reorders bytes; it never creates an expanded +/-1 or INT8 weight tensor.
"""

from __future__ import annotations


GGML_Q1_BLOCK_BYTES = 18
E022_Q1_BLOCK_BYTES = 18
Q1_BLOCK_WEIGHTS = 128


def e022_tile_supported(*, rows: int, k: int) -> bool:
    """Return whether the first E022 target gate can represent a tile.

    The current NBG reads K=5120 and has one work-item produce four output
    rows.  It has been target-verified for at most 1024 rows.  This predicate
    is deliberately conservative: returning false is required to preserve a
    CPU fallback for all other GGML graph shapes.
    """

    return (
        rows > 0
        and rows <= 1024
        and rows % 4 == 0
        and k == 5120
    )


def _validate_shape(data: bytes | bytearray | memoryview, *, rows: int, k: int) -> int:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if k <= 0 or k % Q1_BLOCK_WEIGHTS:
        raise ValueError("k must be a positive multiple of 128")
    row_bytes = (k // 8) + 2 * (k // Q1_BLOCK_WEIGHTS)
    expected = rows * row_bytes
    if len(data) != expected:
        raise ValueError(f"expected {expected} GGML Q1 bytes, got {len(data)}")
    return row_bytes


def repack_ggml_q1_rows_to_e022(
    data: bytes | bytearray | memoryview,
    *,
    rows: int,
    k: int,
) -> bytes:
    """Reorder standard GGML Q1_0 rows into the E022 byte carrier.

    ``data`` must contain exactly ``rows`` contiguous row-major GGML Q1_0
    rows.  The returned buffer has exactly the same byte count and each 18-byte
    block is transformed as ``[d][qs] -> [qs][d]``.  The input is not mutated.
    """

    source = bytes(data)
    row_bytes = _validate_shape(source, rows=rows, k=k)
    blocks_per_row = k // Q1_BLOCK_WEIGHTS
    result = bytearray(len(source))

    for row in range(rows):
        row_start = row * row_bytes
        for block in range(blocks_per_row):
            src_start = row_start + block * GGML_Q1_BLOCK_BYTES
            dst_start = row_start + block * E022_Q1_BLOCK_BYTES
            block_bytes = source[src_start:src_start + GGML_Q1_BLOCK_BYTES]
            result[dst_start:dst_start + E022_Q1_BLOCK_BYTES] = (
                block_bytes[2:] + block_bytes[:2]
            )

    return bytes(result)
