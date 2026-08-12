// E040: one native block_q1_0x4 output group, whole-K.
//
// Packed Q1 bytes remain in the Prism layout.  For each two-byte tile, four
// nibbles are formed in registers and expanded through a 64-byte TBL table:
// T0..T3 each map a four-bit nibble to four exact {0xff,0x01} signs.  No
// expanded signs or temporary buffers are written to memory.

        .arch armv8.2-a+dotprod+fp16
        .text
        .align  2
        .global e040_q1_group_wholek
        .type   e040_q1_group_wholek, %function
e040_q1_group_wholek:

// AAPCS64 argument/register map:
//   x0  dst: float[4]
//   x1  b: nb native block_q1_0x4 blocks, 72 bytes each
//   x2  q8: 4*nb native block_q8_0 blocks, 34 bytes each
//   x3  nibble_lut: 64 bytes, T0,T1,T2,T3 concatenated
//   w4  nb: uint32_t number of Q1 blocks
//
// Persistent state: v0 global FP32 accumulator, x1/x2 current block pointers,
// w4 remaining block count, v24..v27 LUT, v28 base indices, v29 nibble lane
// selectors.  All used registers are caller-saved; no stack or BL is used.

// The first four bytes of v23 are [a&15,a>>4,b&15,b>>4].  zip1's remaining
// lanes are irrelevant because v29 selects only lanes 0..3.  There is no
// AdvSIMD LD1 .2b arrangement, so the two packed bytes are loaded as a zero-
// extended halfword and moved into d30; this is bounded even at qs[62..63].
        .macro  E040_UNPACK off
        ldrh    w16, [x1, #\off]
        fmov    d30, x16
        ushr    v22.8b, v30.8b, #4
        and     v30.16b, v30.16b, v7.16b
        zip1    v23.8b, v30.8b, v22.8b
        tbl     v30.16b, {v23.16b}, v29.16b
        add     v30.16b, v30.16b, v28.16b
        tbl     v17.16b, {v24.16b-v27.16b}, v30.16b
        .endm

        .macro  E040_DOT tile, off
        dup     v18.4s, v16.s[\tile]
        E040_UNPACK \off
        sdot    v20.4s, v17.16b, v18.16b
        .endm

        movi    v0.4s, #0
        tbnz    w4, #31, .L_e040_store
        cbz     w4, .L_e040_store

        // Load the immutable 64-byte table and the two index vectors once.
        ld1     {v24.16b-v27.16b}, [x3]
        movi    v7.16b, #0xf
        adr     x6, .L_e040_base
        ld1     {v28.16b}, [x6]
        adr     x6, .L_e040_selectors
        ld1     {v29.16b}, [x6]

.L_e040_l:
        movi    v2.4s, #0

        // qk=0, Q1 qs offsets 8..23; q8 block 0.
        ldrh    w16, [x2]
        fmov    s31, w16
        fcvt    s31, h31
        add     x17, x2, #2
        movi    v20.4s, #0
        ld1     {v16.16b}, [x17], #16
        E040_DOT 0, 8
        E040_DOT 1, 10
        E040_DOT 2, 12
        E040_DOT 3, 14
        add     x17, x2, #18
        ld1     {v16.16b}, [x17]
        E040_DOT 0, 16
        E040_DOT 1, 18
        E040_DOT 2, 20
        E040_DOT 3, 22
        scvtf   v22.4s, v20.4s
        fmla    v2.4s, v22.4s, v31.s[0]
        add     x2, x2, #34

        // qk=1, Q1 qs offsets 24..39; q8 block 1.
        ldrh    w16, [x2]
        fmov    s31, w16
        fcvt    s31, h31
        add     x17, x2, #2
        movi    v20.4s, #0
        ld1     {v16.16b}, [x17], #16
        E040_DOT 0, 24
        E040_DOT 1, 26
        E040_DOT 2, 28
        E040_DOT 3, 30
        add     x17, x2, #18
        ld1     {v16.16b}, [x17]
        E040_DOT 0, 32
        E040_DOT 1, 34
        E040_DOT 2, 36
        E040_DOT 3, 38
        scvtf   v22.4s, v20.4s
        fmla    v2.4s, v22.4s, v31.s[0]
        add     x2, x2, #34

        // qk=2, Q1 qs offsets 40..55; q8 block 2.
        ldrh    w16, [x2]
        fmov    s31, w16
        fcvt    s31, h31
        add     x17, x2, #2
        movi    v20.4s, #0
        ld1     {v16.16b}, [x17], #16
        E040_DOT 0, 40
        E040_DOT 1, 42
        E040_DOT 2, 44
        E040_DOT 3, 46
        add     x17, x2, #18
        ld1     {v16.16b}, [x17]
        E040_DOT 0, 48
        E040_DOT 1, 50
        E040_DOT 2, 52
        E040_DOT 3, 54
        scvtf   v22.4s, v20.4s
        fmla    v2.4s, v22.4s, v31.s[0]
        add     x2, x2, #34

        // qk=3, Q1 qs offsets 56..71; q8 block 3.
        ldrh    w16, [x2]
        fmov    s31, w16
        fcvt    s31, h31
        add     x17, x2, #2
        movi    v20.4s, #0
        ld1     {v16.16b}, [x17], #16
        E040_DOT 0, 56
        E040_DOT 1, 58
        E040_DOT 2, 60
        E040_DOT 3, 62
        add     x17, x2, #18
        ld1     {v16.16b}, [x17]
        E040_DOT 0, 64
        E040_DOT 1, 66
        E040_DOT 2, 68
        E040_DOT 3, 70
        scvtf   v22.4s, v20.4s
        fmla    v2.4s, v22.4s, v31.s[0]
        add     x2, x2, #34

        // Apply Q1 row scales only after all four q8 blocks for this l.
        ld1     {v6.4h}, [x1]
        fcvtl   v6.4s, v6.4h
        fmla    v0.4s, v2.4s, v6.4s
        add     x1, x1, #72
        sub     w4, w4, #1
        cbnz    w4, .L_e040_l

.L_e040_store:
        st1     {v0.4s}, [x0]
        ret

        .align  4
.L_e040_base:
        // Per nibble: [T0+n,T1+n,T2+n,T3+n].  Repeated for n0..n3.
        .byte   0,16,32,48, 0,16,32,48, 0,16,32,48, 0,16,32,48
.L_e040_selectors:
        .byte   0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3

        .size e040_q1_group_wholek, .-e040_q1_group_wholek
