"""Scalar golden for E030's paired 4-row Q1/Q8 activation-reuse path.

The target kernel keeps the existing ``block_q1_0x4`` representation.  Four
Q1 rows are packed into 64 sign bytes per 128-weight block and the adjacent
Q8 activation block is consumed twice, once for each neighbouring four-row
group.  This module deliberately models only the integer dot-product part;
FP16 scale multiplication is unchanged from the existing ggml path.
"""

from __future__ import annotations


QK1 = 128
QK8 = 32
ROWS_PER_GROUP = 4


def _validate(signs: tuple[tuple[int, ...], ...], q8: tuple[tuple[int, ...], ...]) -> None:
    if not signs or len(signs) % ROWS_PER_GROUP:
        raise ValueError("sign rows must be a non-empty multiple of four")
    k = len(signs[0])
    if k == 0 or k % QK1:
        raise ValueError("sign rows must have a positive K multiple of 128")
    if any(len(row) != k for row in signs):
        raise ValueError("all sign rows must have equal length")
    if len(q8) != k // QK8 or any(len(block) != QK8 for block in q8):
        raise ValueError("Q8 blocks must cover K as 32-value blocks")
    if any(value not in (-1, 1) for row in signs for value in row):
        raise ValueError("sign rows must contain only -1 or +1")
    if any(value < -128 or value > 127 for block in q8 for value in block):
        raise ValueError("Q8 values must fit signed int8")


def _step(seed: int) -> int:
    """Small deterministic generator; no process-global random state."""

    return (1664525 * seed + 1013904223) & 0xFFFFFFFF


def make_sign_rows(*, rows: int, k: int, pattern: str) -> tuple[tuple[int, ...], ...]:
    if rows <= 0 or rows % ROWS_PER_GROUP or k <= 0 or k % QK1:
        raise ValueError("rows must be a positive multiple of four and K a multiple of 128")

    result: list[tuple[int, ...]] = []
    seed = 0xE030
    for row in range(rows):
        values: list[int] = []
        for index in range(k):
            if pattern == "zeros":
                bit = 0
            elif pattern == "ones":
                bit = 1
            elif pattern == "alternating":
                bit = (row + index) & 1
            elif pattern == "random":
                seed = _step(seed)
                bit = (seed >> 31) & 1
            elif pattern == "extremes":
                bit = ((row * 17 + index * 13) ^ (index >> 3)) & 1
            else:
                raise ValueError(f"unknown sign pattern: {pattern}")
            values.append(1 if bit else -1)
        result.append(tuple(values))
    return tuple(result)


def make_q8_blocks(*, k: int, pattern: str) -> tuple[tuple[int, ...], ...]:
    if k <= 0 or k % QK8:
        raise ValueError("K must be a positive multiple of 32")

    result: list[tuple[int, ...]] = []
    seed = 0xA733
    extreme_values = (-128, -127, -1, 0, 1, 126, 127)
    for block_index in range(k // QK8):
        values: list[int] = []
        for lane in range(QK8):
            index = block_index * QK8 + lane
            if pattern == "zeros":
                value = 0
            elif pattern == "ones":
                value = 1
            elif pattern == "alternating":
                value = -127 if index & 1 else 127
            elif pattern == "random":
                seed = _step(seed)
                value = ((seed >> 24) & 0xFF) - 128
            elif pattern == "extremes":
                value = extreme_values[index % len(extreme_values)]
            else:
                raise ValueError(f"unknown Q8 pattern: {pattern}")
            values.append(value)
        result.append(tuple(values))
    return tuple(result)


def scalar_outputs(
    signs: tuple[tuple[int, ...], ...],
    q8: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    """Reference row-major signed dot products without quantization scales."""

    _validate(signs, q8)
    flat_q8 = tuple(value for block in q8 for value in block)
    return tuple(sum(sign * value for sign, value in zip(row, flat_q8)) for row in signs)


def _pack_group(rows: tuple[tuple[int, ...], ...], *, block: int) -> tuple[int, ...]:
    """Produce the exact 64-byte ``block_q1_0x4.qs`` layout for one K block."""

    out: list[int] = []
    base = block * QK1
    for k8 in range(QK1 // QK8):
        for tile in range(QK8 // 4):
            packed_lo = 0
            packed_hi = 0
            weight_base = base + k8 * QK8 + tile * 4
            for pos in range(4):
                weight_index = weight_base + pos
                for row_offset in (0, 1):
                    packed_lo |= ((rows[row_offset][weight_index] == 1) << (row_offset * 4 + pos))
                for row_offset in (2, 3):
                    packed_hi |= ((rows[row_offset][weight_index] == 1) << ((row_offset - 2) * 4 + pos))
            out.extend((packed_lo, packed_hi))
    return tuple(out)


def _unpack_pair(bits0: int, bits1: int) -> tuple[int, ...]:
    """Mirror ``table_q1_signs``: low byte is rows 0/1, high byte rows 2/3."""

    result: list[int] = []
    for bits in (bits0, bits1):
        for lane in range(8):
            result.append(1 if (bits >> lane) & 1 else -1)
    return tuple(result)


def _paired_group_outputs(
    rows: tuple[tuple[int, ...], ...],
    q8: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    k = len(rows[0])
    accum = [0, 0, 0, 0]
    for block in range(k // QK1):
        packed = _pack_group(rows, block=block)
        base = block * QK1
        for q8_index in range(QK1 // QK8):
            qblock = q8[(base // QK8) + q8_index]
            packed_base = q8_index * 16
            for tile in range(QK8 // 4):
                signs = _unpack_pair(packed[packed_base + tile * 2], packed[packed_base + tile * 2 + 1])
                values = qblock[tile * 4:tile * 4 + 4]
                for row in range(4):
                    row_signs = signs[row * 4:row * 4 + 4]
                    accum[row] += sum(sign * value for sign, value in zip(row_signs, values))
    return tuple(accum)


def paired_outputs(
    signs: tuple[tuple[int, ...], ...],
    q8: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    """Golden for processing adjacent 4-row groups while reusing each Q8 block."""

    _validate(signs, q8)
    output: list[int] = []
    for start in range(0, len(signs), ROWS_PER_GROUP):
        output.extend(_paired_group_outputs(signs[start:start + ROWS_PER_GROUP], q8))
    return tuple(output)
