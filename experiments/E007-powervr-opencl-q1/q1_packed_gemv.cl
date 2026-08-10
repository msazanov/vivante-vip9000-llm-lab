#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Portable OpenCL C 1.2 baseline. One work-item computes one output row.

__kernel void q1_packed_gemv(
    __global const uchar *packed_weights,
    __global const float *activation,
    const ulong rows,
    const ulong columns,
    __global float *output) {
    const size_t row = get_global_id(0);
    if ((ulong) row >= rows) {
        return;
    }

    const size_t blocks_per_row = (size_t) (columns / 128ul);
    float accumulator = 0.0f;
    for (size_t block = 0; block < blocks_per_row; ++block) {
        const size_t base = (row * blocks_per_row + block) * 18u;
        const float scale =
            vload_half(0, (__global const half *) (packed_weights + base));
        float block_dot = 0.0f;
        for (size_t lane = 0; lane < 128u; ++lane) {
            const uchar signs = packed_weights[base + 2u + lane / 8u];
            const float value = activation[block * 128u + lane];
            block_dot += ((signs >> (lane & 7u)) & 1u) != 0u ? value : -value;
        }
        accumulator += scale * block_dot;
    }
    output[row] = accumulator;
}
