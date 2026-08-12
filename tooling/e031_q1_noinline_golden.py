"""Reference for E031's low-live-state paired Q1/Q8 pipeline.

The target helper processes one 128-value block for two adjacent four-row
groups, accumulates each group over its four Q8 sub-blocks, and applies the
eight independent Q1 row scales in the caller.  This Python model keeps those
boundaries explicit so a future C/NEON implementation cannot accidentally
apply one group's scale to the other.
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


def noinline_pair_outputs(
    signs: tuple[tuple[int, ...], ...],
    q8: tuple[tuple[int, ...], ...],
    scales: tuple[float, ...],
) -> tuple[float, ...]:
    """Compute E031's pair-per-Q1-block output with independent row scales."""

    _validate(signs, q8)
    if len(signs) % (2 * ROWS_PER_GROUP) != 0:
        raise ValueError("E031 requires pairs of four-row groups")
    if len(scales) != len(signs):
        raise ValueError("one scale is required for every output row")

    output: list[float] = []
    for pair_start in range(0, len(signs), 2 * ROWS_PER_GROUP):
        group0 = signs[pair_start:pair_start + ROWS_PER_GROUP]
        group1 = signs[pair_start + ROWS_PER_GROUP:pair_start + 2 * ROWS_PER_GROUP]
        accum0 = [0, 0, 0, 0]
        accum1 = [0, 0, 0, 0]
        for block in range(len(signs[0]) // QK1):
            packed0 = _pack_group(group0, block=block)
            packed1 = _pack_group(group1, block=block)
            base = block * QK1
            for q8_index in range(QK1 // QK8):
                qblock = q8[base // QK8 + q8_index]
                packed_base = q8_index * 16
                for tile in range(QK8 // 4):
                    values = qblock[tile * 4:tile * 4 + 4]
                    signs0 = _unpack_pair(
                        packed0[packed_base + tile * 2],
                        packed0[packed_base + tile * 2 + 1],
                    )
                    signs1 = _unpack_pair(
                        packed1[packed_base + tile * 2],
                        packed1[packed_base + tile * 2 + 1],
                    )
                    for row in range(ROWS_PER_GROUP):
                        row_slice = slice(row * 4, row * 4 + 4)
                        accum0[row] += sum(
                            sign * value
                            for sign, value in zip(signs0[row_slice], values)
                        )
                        accum1[row] += sum(
                            sign * value
                            for sign, value in zip(signs1[row_slice], values)
                        )
        for row, value in enumerate(accum0):
            output.append(value * scales[pair_start + row])
        for row, value in enumerate(accum1):
            output.append(value * scales[pair_start + ROWS_PER_GROUP + row])
    return tuple(output)
