#pragma once
// INKNVFP4K Q8A16: Q8_0 decode matvec with full-precision activations (no q8_1 activation quantization).
// Weights are dequantized in registers (int8 -> f32 by the 2^23 magic: byte_perm + one FADD, exact), multiplied by the
// f32 activation and accumulated in f32. It reads the same weight bytes as MMVQ (34-byte block_q8_0, no repack); the
// activation is read as f32 straight from src1 (L1/L2 resident, K*4 bytes per column).
// Thread layout: a warp owns RW rows (one activation load feeds all of them); lane l takes q8_0 block l/4 + 8i and
// the 8 weights (l%4)*8 .. +7 of it. A CTA holds NW warps (NW*RW rows). Grid: member blocks back to back
// (the INKGLUE group layout: block b runs member k's row block b - blk_end[k-1]).

#include "common.cuh"
#include "vecdotq.cuh"

#define Q8A16_MAX_GROUP 8
#define Q8A16_MAX_COLS  8
#ifndef Q8A16_UNROLL
#define Q8A16_UNROLL 2
#endif
static constexpr int q8a16_unroll = Q8A16_UNROLL;

struct q8a16_args {
    const char  * vx[Q8A16_MAX_GROUP];   // weights, row r at vx + r*srow
    const float * x[Q8A16_MAX_GROUP];    // activation column 0; column c at x + c*sx (floats)
    float       * dst[Q8A16_MAX_GROUP];  // output column 0; column c at dst + c*sd (floats)
    int64_t       srow[Q8A16_MAX_GROUP]; // bytes per weight row
    uint32_t      blk_end[Q8A16_MAX_GROUP];
    int           nrows[Q8A16_MAX_GROUP];
    int           n;
    int64_t       sx[Q8A16_MAX_GROUP];  // member column strides in floats (src1, dst)
    int64_t       sd[Q8A16_MAX_GROUP];
    const int32_t * ids;                 // MUL_MAT_ID: member k uses expert ids[k] of vx[0] (stride sexp bytes)
    int64_t       sexp;
};

// 4 int8 in q -> 4 exact floats: byte k of q ^ 0x80 is w + 128; 0x4B0000xx is the float 2^23 + xx
static __device__ __forceinline__ void q8a16_cvt4(const int q, float * f) {
    const uint32_t u = (uint32_t) q ^ 0x80808080u;
    f[0] = __uint_as_float(__byte_perm(u, 0x4B000000u, 0x7540)) - 8388736.0f;
    f[1] = __uint_as_float(__byte_perm(u, 0x4B000000u, 0x7541)) - 8388736.0f;
    f[2] = __uint_as_float(__byte_perm(u, 0x4B000000u, 0x7542)) - 8388736.0f;
    f[3] = __uint_as_float(__byte_perm(u, 0x4B000000u, 0x7543)) - 8388736.0f;
}

template <int T, int RW, int NW>
__launch_bounds__(32*NW)
static __global__ void k_q8a16(const q8a16_args a, const int nb /* q8_0 blocks per row */) {
    static_assert(T >= 1 && T <= Q8A16_MAX_COLS && RW*T <= 32, "columns");
    const uint32_t b = blockIdx.x;
    int      k  = 0;
    uint32_t b0 = 0;
    while (k < a.n - 1 && b >= a.blk_end[k]) {
        b0 = a.blk_end[k];
        ++k;
    }
    const int warp  = threadIdx.x / 32;
    const int lane  = threadIdx.x % 32;
    const int nrows = a.nrows[k];
    const int row0  = ((int) (b - b0)*NW + warp)*RW;     // this warp's RW rows share every activation load
    if (row0 >= nrows) {
        return;
    }
    const char * vxk = a.ids ? a.vx[0] + (int64_t) a.ids[k]*a.sexp : a.vx[k];
    const block_q8_0 * w[RW];
#pragma unroll
    for (int r = 0; r < RW; ++r) {
        const int row = row0 + r < nrows ? row0 + r : nrows - 1; // clamped duplicate, never written
        w[r] = (const block_q8_0 *) (vxk + (int64_t) row*a.srow[k]);
    }
    const float * x = a.x[k];

    float acc[RW][T];
#pragma unroll
    for (int r = 0; r < RW; ++r) {
#pragma unroll
        for (int t = 0; t < T; ++t) {
            acc[r][t] = 0.0f;
        }
    }
    const int iq = lane & 3;
#pragma unroll q8a16_unroll
    for (int ib = lane >> 2; ib < nb; ib += 8) {
        float4 x0[T], x1[T];
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float4 * xp = (const float4 *) (x + t*a.sx[k] + (int64_t) ib*QK8_0 + iq*8);
            x0[t] = xp[0];
            x1[t] = xp[1];
        }
#pragma unroll
        for (int r = 0; r < RW; ++r) {
            const block_q8_0 * bw = w[r] + ib;
            const int   q0 = get_int_b2(bw->qs, 2*iq + 0);
            const int   q1 = get_int_b2(bw->qs, 2*iq + 1);
            const float d  = __half2float(bw->d);
            float f[8];
            q8a16_cvt4(q0, f + 0);
            q8a16_cvt4(q1, f + 4);
#pragma unroll
            for (int t = 0; t < T; ++t) {
                float p = f[0]*x0[t].x;
                p = fmaf(f[1], x0[t].y, p);
                p = fmaf(f[2], x0[t].z, p);
                p = fmaf(f[3], x0[t].w, p);
                p = fmaf(f[4], x1[t].x, p);
                p = fmaf(f[5], x1[t].y, p);
                p = fmaf(f[6], x1[t].z, p);
                p = fmaf(f[7], x1[t].w, p);
                acc[r][t] = fmaf(d, p, acc[r][t]);
            }
        }
    }
    // lane j < RW*T writes row row0 + j/T, column j%T
    float out = 0.0f;
#pragma unroll
    for (int r = 0; r < RW; ++r) {
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float s = warp_reduce_sum<32>(acc[r][t]);
            out = lane == r*T + t ? s : out;
        }
    }
    if (lane < RW*T) {
        const int r = lane / T;
        const int t = lane % T;
        if (row0 + r < nrows) {
            a.dst[k][row0 + r + t*a.sd[k]] = out;
        }
    }
}
