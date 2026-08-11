#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// One work-item consumes one packed 18-byte Q1_0 row and the shared activation.
__kernel void q1_vip_mx128(
    __read_only image2d_t packed_weights,
    __read_only image2d_t activation,
    __write_only image2d_t output) {
    const size_t row = get_global_id(0);
    const uint scale_lo = read_imageui(packed_weights, (int2)(0, row)).x;
    const uint scale_hi = read_imageui(packed_weights, (int2)(1, row)).x;
    const ushort scale_bits = (ushort)(scale_lo | (scale_hi << 8));
    const float scale = convert_float(as_half(scale_bits));

    float accumulator = 0.0f;
    for (size_t lane = 0; lane < 128u; ++lane) {
        const uint signs = read_imageui(
            packed_weights, (int2)(2u + lane / 8u, row)).x;
        const float value = read_imagef(activation, (int2)(lane, 0)).x;
        accumulator += ((signs >> (lane & 7u)) & 1u) != 0u ? value : -value;
    }
    write_imagef(output, (int2)(row, 0),
                 (float4)(scale * accumulator, 0, 0, 0));
}
