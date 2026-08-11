#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Exact Bonsai hidden-size probe: 40 packed Q1_0 blocks per output row.
__kernel void q1_vip_mx5120(
    __read_only image2d_t packed_weights,
    __read_only image2d_t activation,
    __write_only image2d_t output) {
    const size_t row = get_global_id(0);
    float accumulator = 0.0f;

    for (size_t block = 0; block < 40u; ++block) {
        const size_t byte_base = block * 18u;
        const uint scale_lo = read_imageui(
            packed_weights, (int2)(byte_base, row)).x;
        const uint scale_hi = read_imageui(
            packed_weights, (int2)(byte_base + 1u, row)).x;
        const ushort scale_bits = (ushort)(scale_lo | (scale_hi << 8));
        const float scale = convert_float(as_half(scale_bits));
        float block_dot = 0.0f;

        for (size_t lane = 0; lane < 128u; ++lane) {
            const uint signs = read_imageui(
                packed_weights, (int2)(byte_base + 2u + lane / 8u, row)).x;
            const float value = read_imagef(
                activation, (int2)(block * 128u + lane, 0)).x;
            block_dot += ((signs >> (lane & 7u)) & 1u) != 0u ? value : -value;
        }
        accumulator += scale * block_dot;
    }
    write_imagef(output, (int2)(row, 0),
                 (float4)(accumulator, 0, 0, 0));
}
