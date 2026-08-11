#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Minimal packed-Q1 carrier probe: one Q1_0 block produces one output value.
// The 18-byte weight block is FP16 scale followed by 128 LSB-first sign bits.
__kernel void q1_vip_1x128(
    __read_only image2d_t packed_weights,
    __read_only image2d_t activation,
    __write_only image2d_t output) {
    if (get_global_id(0) != 0u) {
        return;
    }

    const uint scale_lo = read_imageui(packed_weights, (int2)(0, 0)).x;
    const uint scale_hi = read_imageui(packed_weights, (int2)(1, 0)).x;
    const ushort scale_bits = (ushort)(scale_lo | (scale_hi << 8));
    const float scale = convert_float(as_half(scale_bits));
    float accumulator = 0.0f;
    for (size_t lane = 0; lane < 128u; ++lane) {
        const uint signs = read_imageui(
            packed_weights, (int2)(2u + lane / 8u, 0)).x;
        const float value = read_imagef(activation, (int2)(lane, 0)).x;
        accumulator += ((signs >> (lane & 7u)) & 1u) != 0u ? value : -value;
    }
    write_imagef(output, (int2)(0, 0), (float4)(scale * accumulator, 0, 0, 0));
}
