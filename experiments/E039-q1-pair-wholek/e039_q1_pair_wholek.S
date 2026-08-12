// E039: whole-K native packed-Q1 pair, AArch64 DOTPROD.
//
// One call processes nb native Q1 block pairs.  Q1 packed bytes are expanded
// only in SIMD registers through table_q1_signs immediately before SDOT.
// No expanded sign buffer, call, stack frame, or callee-saved register is
// used.

        .text
        .align  2
        .global e039_q1_pair_wholek
        .type   e039_q1_pair_wholek, %function
e039_q1_pair_wholek:

// AAPCS64 argument/register map at entry:
//   x0  dst: writable float[8], rows b0[0..3] then b1[0..3], contiguous
//   x1  b0: nb native block_q1_0x4 blocks, 72 bytes per block
//   x2  b1: nb native block_q1_0x4 blocks, 72 bytes per block
//   x3  q8: 4*nb native block_q8_0 blocks, 34 bytes per block
//   x4  table_q1_signs: exact uint64_t[256], 8 bytes per entry
//   w5  nb: uint32_t number of Q1 blocks; zero is accepted and stores zero
//       output.  Values with bit 31 set are rejected by the entry guard.
//
// Native block_q1_0x4: d[4] at +0 (8 bytes), qs[64] at +8 (72 bytes total).
// Native block_q8_0: d at +0 (2 bytes), qs[32] at +2 (34 bytes total).
//
// Persistent whole-K state:
//   v0/v1  global FP32 output accumulators across l
//   x1/x2  current b0/b1 block pointers, advanced by 72 after each l
//   x3     current q8 block pointer, advanced by 4*34 after each l
//   x4     table pointer (constant)
//   w5     remaining l count
//
// Per-l state:
//   v2/v3  b0/b1 FP32 accb vectors reset at each l
//   v20/v21 integer dot accumulators reset at each q8 block
//   v22/v23 converted FP32 dots; v24/v25 Q1 scale vectors
//   v16 common 16-byte Q8 load; v17/v19 register-only expanded signs
//   v18 one Q8 four-byte tile replicated to all four lanes
//   v31 current Q8 FP16 scale converted to FP32 in s31
//   x16/x17 scratch (x17 is restored before every second Q8 load)
//
// Clobbers: caller-saved x0..x5,x16,x17 and v0..v3,v16..v25,v31.
// x6..x15, x18..x30, v8..v15, sp, and source memory are untouched.

// Expand packed q1 bytes [off0] and [off1] through the caller-supplied table.
// This matches ggml_q1_0_unpack_pair: two uint64 table entries become one
// 16-byte sign vector.  The result never reaches memory.
        .macro  E039_B0_PAIR off0, off1
        ldrb    w16, [x1, #\off0]
        add     x17, x4, x16, lsl #3
        ldr     x16, [x17]
        fmov    d17, x16
        ldrb    w16, [x1, #\off1]
        add     x17, x4, x16, lsl #3
        ldr     x16, [x17]
        ins     v17.d[1], x16
        .endm

        .macro  E039_B1_PAIR off0, off1
        ldrb    w16, [x2, #\off0]
        add     x17, x4, x16, lsl #3
        ldr     x16, [x17]
        fmov    d19, x16
        ldrb    w16, [x2, #\off1]
        add     x17, x4, x16, lsl #3
        ldr     x16, [x17]
        ins     v19.d[1], x16
        .endm

// One common Q8 tile feeds both four-row groups.  Each SDOT updates four
// output lanes, one lane per row (the table expansion is row-pair ordered).
        .macro  E039_DOT tile, off0, off1
        dup     v18.4s, v16.s[\tile]
        E039_B0_PAIR \off0, \off1
        sdot    v20.4s, v17.16b, v18.16b
        E039_B1_PAIR \off0, \off1
        sdot    v21.4s, v19.16b, v18.16b
        .endm

        .cfi_startproc
        movi    v0.4s, #0
        movi    v1.4s, #0
        // The public count is unsigned.  Reject a signed-negative/high-bit
        // value before touching any input so a bad caller cannot wrap the
        // decrement-and-branch loop into an unbounded walk.
        tbnz    w5, #31, .L_e039_store
        cbz     w5, .L_e039_store

.L_e039_l:
        // qk=0, Q1 qs offsets 8..23; q8 block 0.
        ldrh    w16, [x3]
        fmov    s31, w16
        fcvt    s31, h31
        add     x17, x3, #2
        movi    v2.4s, #0
        movi    v3.4s, #0
        movi    v20.4s, #0
        movi    v21.4s, #0
        ld1     {v16.16b}, [x17], #16
        E039_DOT 0, 8, 9
        E039_DOT 1, 10, 11
        E039_DOT 2, 12, 13
        E039_DOT 3, 14, 15
        add     x17, x3, #18
        ld1     {v16.16b}, [x17]
        E039_DOT 0, 16, 17
        E039_DOT 1, 18, 19
        E039_DOT 2, 20, 21
        E039_DOT 3, 22, 23
        scvtf   v22.4s, v20.4s
        scvtf   v23.4s, v21.4s
        fmla    v2.4s, v22.4s, v31.s[0]
        fmla    v3.4s, v23.4s, v31.s[0]
        add     x3, x3, #34

        // qk=1, Q1 qs offsets 24..39; q8 block 1.
        ldrh    w16, [x3]
        fmov    s31, w16
        fcvt    s31, h31
        add     x17, x3, #2
        movi    v20.4s, #0
        movi    v21.4s, #0
        ld1     {v16.16b}, [x17], #16
        E039_DOT 0, 24, 25
        E039_DOT 1, 26, 27
        E039_DOT 2, 28, 29
        E039_DOT 3, 30, 31
        add     x17, x3, #18
        ld1     {v16.16b}, [x17]
        E039_DOT 0, 32, 33
        E039_DOT 1, 34, 35
        E039_DOT 2, 36, 37
        E039_DOT 3, 38, 39
        scvtf   v22.4s, v20.4s
        scvtf   v23.4s, v21.4s
        fmla    v2.4s, v22.4s, v31.s[0]
        fmla    v3.4s, v23.4s, v31.s[0]
        add     x3, x3, #34

        // qk=2, Q1 qs offsets 40..55; q8 block 2.
        ldrh    w16, [x3]
        fmov    s31, w16
        fcvt    s31, h31
        add     x17, x3, #2
        movi    v20.4s, #0
        movi    v21.4s, #0
        ld1     {v16.16b}, [x17], #16
        E039_DOT 0, 40, 41
        E039_DOT 1, 42, 43
        E039_DOT 2, 44, 45
        E039_DOT 3, 46, 47
        add     x17, x3, #18
        ld1     {v16.16b}, [x17]
        E039_DOT 0, 48, 49
        E039_DOT 1, 50, 51
        E039_DOT 2, 52, 53
        E039_DOT 3, 54, 55
        scvtf   v22.4s, v20.4s
        scvtf   v23.4s, v21.4s
        fmla    v2.4s, v22.4s, v31.s[0]
        fmla    v3.4s, v23.4s, v31.s[0]
        add     x3, x3, #34

        // qk=3, Q1 qs offsets 56..71; q8 block 3.
        ldrh    w16, [x3]
        fmov    s31, w16
        fcvt    s31, h31
        add     x17, x3, #2
        movi    v20.4s, #0
        movi    v21.4s, #0
        ld1     {v16.16b}, [x17], #16
        E039_DOT 0, 56, 57
        E039_DOT 1, 58, 59
        E039_DOT 2, 60, 61
        E039_DOT 3, 62, 63
        add     x17, x3, #18
        ld1     {v16.16b}, [x17]
        E039_DOT 0, 64, 65
        E039_DOT 1, 66, 67
        E039_DOT 2, 68, 69
        E039_DOT 3, 70, 71
        scvtf   v22.4s, v20.4s
        scvtf   v23.4s, v21.4s
        fmla    v2.4s, v22.4s, v31.s[0]
        fmla    v3.4s, v23.4s, v31.s[0]
        // Advance to the next l's first Q8 block after qk=3.
        add     x3, x3, #34

        // Apply Q1 row scales only after all four q8 blocks for this l.
        ld1     {v24.4h}, [x1]
        fcvtl   v24.4s, v24.4h
        ld1     {v25.4h}, [x2]
        fcvtl   v25.4s, v25.4h
        fmla    v0.4s, v2.4s, v24.4s
        fmla    v1.4s, v3.4s, v25.4s

        add     x1, x1, #72
        add     x2, x2, #72
        sub     w5, w5, #1
        cbnz    w5, .L_e039_l

.L_e039_store:
        st1     {v0.4s}, [x0]
        add     x17, x0, #16
        st1     {v1.4s}, [x17]
        ret
        .cfi_endproc
        .size e039_q1_pair_wholek, .-e039_q1_pair_wholek
