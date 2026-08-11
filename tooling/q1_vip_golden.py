"""Independent CPU golden for canonical and packed Q1_0 data."""

import argparse
import hashlib
import math
import os
from pathlib import Path
import struct
import sys


Q1_BLOCK_BYTES = 18
Q8_SUBBLOCK_BYTES = 34
Q8_SUBBLOCK_COUNT = 4
Q8_VALUES_PER_SUBBLOCK = 32
Q8_VECTOR_BYTES = Q8_SUBBLOCK_BYTES * Q8_SUBBLOCK_COUNT
TILE_BYTES = 288
Q1_VALUES = 128
Q1_SIGN_BYTES = 16


class Q1VipGoldenError(ValueError):
    """Raised when a Q1/Q8 golden input is invalid."""


def _require_bytes(value: object, expected: int, name: str) -> bytes:
    if not isinstance(value, bytes):
        raise Q1VipGoldenError(f"{name} must be bytes")
    if len(value) != expected:
        raise Q1VipGoldenError(f"{name} must contain exactly {expected} bytes")
    return value


def _fp16(raw: bytes, offset: int) -> float:
    value = struct.unpack("<e", raw[offset:offset + 2])[0]
    if not math.isfinite(value):
        raise Q1VipGoldenError("FP16 scales must be finite")
    return value


def _signed_byte(raw: int) -> int:
    return raw if raw < 128 else raw - 256


def _validate_q8_scales(q8: bytes) -> None:
    for subblock in range(Q8_SUBBLOCK_COUNT):
        _fp16(q8, subblock * Q8_SUBBLOCK_BYTES)


def _dot_block(q1: bytes, q8: bytes) -> float:
    q1_scale = _fp16(q1, 0)
    result = 0.0
    for subblock in range(Q8_SUBBLOCK_COUNT):
        q8_start = subblock * Q8_SUBBLOCK_BYTES
        q8_scale = _fp16(q8, q8_start)
        integer_dot = 0
        for local in range(Q8_VALUES_PER_SUBBLOCK):
            index = subblock * Q8_VALUES_PER_SUBBLOCK + local
            q1_sign = 1 if q1[2 + index // 8] & (1 << (index % 8)) else -1
            integer_dot += q1_sign * _signed_byte(q8[q8_start + 2 + local])
        result += float(q1_scale) * float(q8_scale) * integer_dot
    return result


def dot_canonical_block(q1: bytes, q8: bytes) -> float:
    """Return the scalar Q1_0 × Q8_0 K=128 dot product."""
    q1 = _require_bytes(q1, Q1_BLOCK_BYTES, "q1")
    q8 = _require_bytes(q8, Q8_VECTOR_BYTES, "q8")
    _validate_q8_scales(q8)
    return _dot_block(q1, q8)


def _matrix_dimensions(ne0: int, ne1: int) -> tuple[int, int]:
    if type(ne0) is not int or type(ne1) is not int:
        raise Q1VipGoldenError("matrix dimensions must be integers")
    if ne0 <= 0 or ne0 % Q1_VALUES:
        raise Q1VipGoldenError("ne0 must be a positive multiple of 128")
    if ne1 <= 0:
        raise Q1VipGoldenError("ne1 must be positive")
    return ne0 // Q1_VALUES, ne1


def _require_matrix_lengths(q1: bytes, q8: bytes, ne0: int, ne1: int) -> int:
    blocks_per_row, _ = _matrix_dimensions(ne0, ne1)
    _require_bytes(q1, ne1 * blocks_per_row * Q1_BLOCK_BYTES, "q1")
    _require_bytes(q8, blocks_per_row * Q8_VECTOR_BYTES, "q8")
    return blocks_per_row


def dot_canonical_matrix(q1: bytes, q8: bytes, ne0: int, ne1: int) -> tuple[float, ...]:
    """Return row-wise Q1_0 × Q8_0 dots for a canonical ``ne1 × ne0`` matrix.

    ``q1`` is row-major canonical Q1_0 storage.  The Q8 input contains one
    Q8 vector (one 128-value vector) for each K block, shared by every row.
    Rows are accumulated block by block so the reference does not construct a
    second matrix-sized representation.
    """
    blocks_per_row = _require_matrix_lengths(q1, q8, ne0, ne1)
    _validate_q8_scales(q8)
    output = []
    for row in range(ne1):
        total = 0.0
        for block in range(blocks_per_row):
            q1_offset = (row * blocks_per_row + block) * Q1_BLOCK_BYTES
            q8_offset = block * Q8_VECTOR_BYTES
            total += dot_canonical_block(
                q1[q1_offset:q1_offset + Q1_BLOCK_BYTES],
                q8[q8_offset:q8_offset + Q8_VECTOR_BYTES],
            )
        output.append(total)
    return tuple(output)


def dot_packed_tile(tile: bytes, q8: bytes) -> tuple[float, ...]:
    """Return 16 scalar dots from one packed 16-row Q1 VIP tile."""
    tile = _require_bytes(tile, TILE_BYTES, "tile")
    q8 = _require_bytes(q8, Q8_VECTOR_BYTES, "q8")
    _validate_q8_scales(q8)
    scales_start = Q1_SIGN_BYTES * Q1_VALUES // 8
    values = []
    for row in range(16):
        signs_start = row * Q1_SIGN_BYTES
        q1 = tile[scales_start + row * 2:scales_start + row * 2 + 2]
        q1 += tile[signs_start:signs_start + Q1_SIGN_BYTES]
        values.append(_dot_block(q1, q8))
    return tuple(values)


def _exclusive_write(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    complete = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        complete = True
    finally:
        if descriptor != -1:
            os.close(descriptor)
        if not complete:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def _require_new_output(path: Path) -> None:
    # This preflight avoids reading/processing a production matrix when the
    # requested destination is already occupied.  O_EXCL remains the final
    # race-safe check in _exclusive_write().
    if os.path.lexists(path):
        raise Q1VipGoldenError(f"refusing to overwrite existing output: {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate an independent canonical Q1 F32 golden")
    parser.add_argument("--weights", type=Path, required=True, help="canonical Q1_0 matrix bytes")
    parser.add_argument("--q8", type=Path, required=True, help="canonical Q8_0 activation bytes")
    parser.add_argument("--ne0", type=int, required=True, help="matrix K dimension")
    parser.add_argument("--ne1", type=int, required=True, help="matrix M dimension")
    parser.add_argument("--output-f32", type=Path, required=True, help="new little-endian F32 output")
    args = parser.parse_args(argv)
    try:
        _require_new_output(args.output_f32)
        values = dot_canonical_matrix(
            args.weights.read_bytes(), args.q8.read_bytes(), args.ne0, args.ne1
        )
        output = struct.pack("<" + "f" * len(values), *values)
        _exclusive_write(args.output_f32, output)
    except (OSError, Q1VipGoldenError, OverflowError, struct.error) as error:
        print(f"golden error: {error}", file=sys.stderr)
        return 2
    print(f"output_sha256={hashlib.sha256(output).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
