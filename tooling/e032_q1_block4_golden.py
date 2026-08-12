"""Independent scalar reference for the E032 Q1/Q8 block4 helper.

E032 keeps the packed Q1 representation and moves the four Q8 sub-blocks of
one 128-value Q1 row-group into one helper call.  This module deliberately
models the arithmetic scalarly so the host gate does not share NEON code with
the target implementation.
"""

from __future__ import annotations

from tooling.e030_q1_paired_golden import (
    QK1,
    QK8,
    ROWS_PER_GROUP,
    _pack_group,
    _unpack_pair,
    _validate,
)


def block4_outputs(
    signs: tuple[tuple[int, ...], ...],
    q8: tuple[tuple[int, ...], ...],
    scales: tuple[float, ...],
) -> tuple[float, ...]:
    """Reference four-row output for E032's one-call block4 pipeline."""

    _validate(signs, q8)
    if len(signs) != ROWS_PER_GROUP:
        raise ValueError("E032 expects one four-row Q1 group")
    if len(scales) != ROWS_PER_GROUP:
        raise ValueError("one scale is required for every output row")

    accum = [0, 0, 0, 0]
    for block in range(len(signs[0]) // QK1):
        packed = _pack_group(signs, block=block)
        base = block * QK1
        for q8_index in range(QK1 // QK8):
            qblock = q8[base // QK8 + q8_index]
            packed_base = q8_index * 16
            for tile in range(QK8 // 4):
                signs16 = _unpack_pair(
                    packed[packed_base + tile * 2],
                    packed[packed_base + tile * 2 + 1],
                )
                values = qblock[tile * 4:tile * 4 + 4]
                for row in range(ROWS_PER_GROUP):
                    row_slice = slice(row * 4, row * 4 + 4)
                    accum[row] += sum(
                        sign * value
                        for sign, value in zip(signs16[row_slice], values)
                    )
    return tuple(value * scales[row] for row, value in enumerate(accum))

