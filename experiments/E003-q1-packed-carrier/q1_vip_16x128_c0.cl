// Portable OpenCL C 1.2 C0 probe for Q1_VIP_16x128/v1.
// The first four arguments are read-only byte/scale planes; the fifth is output.
__kernel void q1_vip_16x128_c0(
    __global const uchar *packed_signs,
    __global const float *q1_scales,
    __global const uchar *q8_values,
    __global const float *q8_scales,
    __global float *output) {
    const size_t row = get_global_id(0);
    if (row >= 16) return;

    float acc = 0.0f;
    for (uint block = 0; block < 4; ++block) {
        int signed_dot = 0;
        for (uint j = 0; j < 32; ++j) {
            const uint k = block * 32 + j;
            const uchar sign_byte = packed_signs[row * 16 + k / 8];
            const int sign = ((sign_byte >> (k & 7)) & 1) ? 1 : -1;
            const uint raw = q8_values[k];
            const int value = raw <= 127 ? (int)raw : (int)raw - 256;
            signed_dot += sign * value;
        }
        acc += q8_scales[block] * (float)signed_dot;
    }
    output[row] = q1_scales[row] * acc;
}
