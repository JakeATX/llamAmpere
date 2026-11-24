#pragma once
// INKNVFP4K Q8A16 MoE at verify width T = 2..8: Q8_0 MUL_MAT_ID on f32 activations, one weight read per distinct expert.
// Grid (row blocks, n_used*T slots). The CTA of slot s = (k, t) looks up expert e = ids[k, t]; if an earlier slot routes
// to e the CTA exits, otherwise it computes every slot routed to e (at most one per token) from the same weight loads.
// Per (row, slot) the lane split, the k-block order, the f32 math and the warp reduction are k_q8a16's (q8a16-kern.cuh),
// so every output is bitwise what the T = 1 Q8A16 MUL_MAT_ID computes for that token.

#include "q8a16-kern.cuh"

#define Q8A16M_MAX_SLOTS 64

struct q8a16m_args {
    const char    * vx;        // expert e row r at vx + e*sexp + r*srow
    int64_t         sexp;
    int64_t         srow;
    const float   * x;         // slot (k, t) activation at x + k*sxk + t*sxt (floats)
    int64_t         sxk;
    int64_t         sxt;
    float         * dst;       // slot (k, t) output at dst + k*sdk + t*sdt (floats)
    int64_t         sdk;
    int64_t         sdt;
    const int32_t * ids;       // ids[k + t*sid]
    int64_t         sid;
    int             n_used;
    int             nrows;
};

template <int M, int RW>
static __device__ __forceinline__ void q8a16m_rows(const q8a16m_args & a, const block_q8_0 * const * w, const int nb, const int row0,
        const int * col_k, const int * col_t, const int lane) {
    const float * xs[M];
#pragma unroll
    for (int c = 0; c < M; ++c) {
        xs[c] = a.x + col_k[c]*a.sxk + col_t[c]*a.sxt;
    }
    float acc[RW][M];
#pragma unroll
    for (int r = 0; r < RW; ++r) {
#pragma unroll
        for (int c = 0; c < M; ++c) {
            acc[r][c] = 0.0f;
        }
    }
    const int iq = lane & 3;
#pragma unroll q8a16_unroll
    for (int ib = lane >> 2; ib < nb; ib += 8) {
        float4 x0[M], x1[M];
#pragma unroll
        for (int c = 0; c < M; ++c) {
            const float4 * xp = (const float4 *) (xs[c] + (int64_t) ib*QK8_0 + iq*8);
            x0[c] = xp[0];
            x1[c] = xp[1];
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
            for (int c = 0; c < M; ++c) {
                float p = f[0]*x0[c].x;
                p = fmaf(f[1], x0[c].y, p);
                p = fmaf(f[2], x0[c].z, p);
                p = fmaf(f[3], x0[c].w, p);
                p = fmaf(f[4], x1[c].x, p);
                p = fmaf(f[5], x1[c].y, p);
                p = fmaf(f[6], x1[c].z, p);
                p = fmaf(f[7], x1[c].w, p);
                acc[r][c] = fmaf(d, p, acc[r][c]);
            }
        }
    }
    float out = 0.0f;
#pragma unroll
    for (int r = 0; r < RW; ++r) {
#pragma unroll
        for (int c = 0; c < M; ++c) {
            const float s = warp_reduce_sum<32>(acc[r][c]);
            out = lane == r*M + c ? s : out;
        }
    }
    if (lane < RW*M) {
        const int r = lane / M;
        const int c = lane % M;
        if (row0 + r < a.nrows) {
            a.dst[col_k[c]*a.sdk + col_t[c]*a.sdt + row0 + r] = out;
        }
    }
}

template <int TMAX, int RW, int NW>
__launch_bounds__(32*NW)
static __global__ void k_q8a16_moe(const q8a16m_args a, const int nb /* q8_0 blocks per row */) {
    static_assert(TMAX >= 1 && TMAX <= 8 && RW*TMAX <= 32, "columns");
    const int warp  = threadIdx.x / 32;
    const int lane  = threadIdx.x % 32;
    const int row0  = ((int) blockIdx.x*NW + warp)*RW;
    if (row0 >= a.nrows) {
        return;
    }
    const int n_used = a.n_used;
    const int nslot  = (int) gridDim.y;                // n_used*T <= Q8A16M_MAX_SLOTS
    const int s      = blockIdx.y;
    const int e      = a.ids[(s % n_used) + (int64_t) (s / n_used)*a.sid];

    // warp-wide scan of the (<= 64) slots: exit if an earlier slot owns e, else collect the slots routed to e
    unsigned long long mine = 0;
#pragma unroll
    for (int j0 = 0; j0 < Q8A16M_MAX_SLOTS; j0 += 32) {
        const int  j   = j0 + lane;
        const bool hit = j < nslot && a.ids[(j % n_used) + (int64_t) (j / n_used)*a.sid] == e;
        const unsigned m = __ballot_sync(0xFFFFFFFFu, hit);
        mine |= (unsigned long long) m << j0;
    }
    if (mine & ((1ull << s) - 1ull)) {
        return; // an earlier slot's CTA computes this expert
    }
    const char * vxe = a.vx + (int64_t) e*a.sexp;
    const block_q8_0 * w[RW];
#pragma unroll
    for (int r = 0; r < RW; ++r) {
        const int row = row0 + r < a.nrows ? row0 + r : a.nrows - 1; // clamped duplicate, never written
        w[r] = (const block_q8_0 *) (vxe + (int64_t) row*a.srow);
    }
    // normally <= 1 slot per token (top-k ids are distinct), so one pass; duplicate ids just take more passes
    for (unsigned long long left = mine; left; ) {
        int col_k[TMAX], col_t[TMAX];
        int m = 0;
#pragma unroll
        for (int c = 0; c < TMAX; ++c) {
            col_k[c] = 0;
            col_t[c] = 0;
        }
        for (; left && m < TMAX; left &= left - 1ull) {
            const int jj = __ffsll((long long) left) - 1;
#pragma unroll
            for (int c = 0; c < TMAX; ++c) {
                if (c == m) {
                    col_k[c] = jj % n_used;
                    col_t[c] = jj / n_used;
                }
            }
            ++m;
        }
        switch (m) {
            case 1: q8a16m_rows<1, RW>(a, w, nb, row0, col_k, col_t, lane); break;
            case 2: if constexpr (TMAX >= 2) { q8a16m_rows<2, RW>(a, w, nb, row0, col_k, col_t, lane); } break;
            case 3: if constexpr (TMAX >= 3) { q8a16m_rows<3, RW>(a, w, nb, row0, col_k, col_t, lane); } break;
            case 4: if constexpr (TMAX >= 4) { q8a16m_rows<4, RW>(a, w, nb, row0, col_k, col_t, lane); } break;
            case 5: if constexpr (TMAX >= 5) { q8a16m_rows<5, RW>(a, w, nb, row0, col_k, col_t, lane); } break;
            case 6: if constexpr (TMAX >= 6) { q8a16m_rows<6, RW>(a, w, nb, row0, col_k, col_t, lane); } break;
            case 7: if constexpr (TMAX >= 7) { q8a16m_rows<7, RW>(a, w, nb, row0, col_k, col_t, lane); } break;
            case 8: if constexpr (TMAX >= 8) { q8a16m_rows<8, RW>(a, w, nb, row0, col_k, col_t, lane); } break;
            default: break;
        }
    }
}
