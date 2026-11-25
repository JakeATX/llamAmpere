// NEONKERN NK_DENSE_TC tail kernel: 0-SMEM DP4A head kernel for K % 256 == 128 (nb % 8 == 4).
// Separate translation unit keeps k_nk_head_grid_dp4a (K % 256 == 0) codegen unchanged.

#include "nk-dense-tc-tail.cuh"
#include "ggml-impl.h"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

static __device__ __forceinline__ uint64_t nk_tail_l2_evict_first() {
#if defined(AMPERE_MMA_AVAILABLE) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    uint64_t p = 0;
    asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(p));
    return p;
#else
    return 0;
#endif
}

template <bool L2_EF>
static __device__ __forceinline__ uint2 nk_tail_ldg_na_u64(const uint2 * p, uint64_t l2_pol) {
#if defined(AMPERE_MMA_AVAILABLE) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    uint2 v;
    if constexpr (L2_EF) {
        asm("ld.global.L1::no_allocate.L2::cache_hint.v2.u32 {%0, %1}, [%2], %3;\n"
            : "=r"(v.x), "=r"(v.y) : "l"(p), "l"(l2_pol));
    } else {
        asm("ld.global.L1::no_allocate.v2.u32 {%0, %1}, [%2];\n"
            : "=r"(v.x), "=r"(v.y) : "l"(p));
    }
    return v;
#else
    GGML_UNUSED_VARS(l2_pol);
    return *p;
#endif
}

// 0-SMEM streaming DP4A head kernel for nb >= 4 with nb % 8 == 4 (K % 256 == 128).
// Main loop processes nb_main = nb - 4 blocks in 272 B (8-block) chunks.
// Tail processes the last 4 blocks (136 B = 17 uint2): lanes 0..16 load weights, lanes 0..15 accumulate.
template <int T, int RW, int NW, int MINB, bool L2_EF>
static __global__ void __launch_bounds__(32 * NW, MINB) k_nk_head_grid_dp4a_tail4(
        const char  * __restrict__ vx,
        const int2  * __restrict__ qx_q,
        const float * __restrict__ qx_d,
        float       * __restrict__ dst,
        const int64_t srow,
        const int64_t sd,
        const int     nr,
        const int     nb) {
#if defined(AMPERE_MMA_AVAILABLE) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    static_assert(T >= 1 && T <= 8 && RW * T <= 32, "head_grid_dp4a_tail4 shape");
    const int warp = (int) threadIdx.x >> 5;
    const int lane = (int) threadIdx.x & 31;
    const int row0 = ((int) blockIdx.x * NW + warp) * RW;
    if (row0 >= nr) {
        return;
    }

    const int pair_in_iter = lane >> 3;
    const int g            = lane & 7;
    const int is_odd       = g >> 2;
    const int ib_lane      = lane >> 2;
    const int iq           = lane & 3;
    const int pair_base_w  = pair_in_iter * 17 + (is_odd ? (9 + 2 * iq) : (2 * iq));
    const int s_lo         = (lane < 30) ? (pair_base_w >> 1) : 0;
    const int s_hi         = (s_lo + 1) & 31;
    const int parity       = pair_base_w & 1;
    const int d_src_lane   = (lane & ~7) + (is_odd ? 3 : 0);

    const char * wp0 = vx + (int64_t) row0 * srow;
    int64_t row_off[RW];
#pragma unroll
    for (int r = 0; r < RW; ++r) {
        row_off[r] = (row0 + r < nr) ? (int64_t) r * srow : 0;
    }

    const int nb_main = nb - 4;
    const int2  * pq = qx_q + ((nb_main > 0 || lane < 16) ? ((int64_t) (ib_lane * T) * 4 + iq) : 0);
    const float * pd = qx_d + ((nb_main > 0 || lane < 16) ? ((int64_t) (ib_lane * T)) : 0);

    const uint64_t l2_pol = L2_EF ? nk_tail_l2_evict_first() : 0ull;

    float acc[RW][T];
#pragma unroll
    for (int r = 0; r < RW; ++r) {
#pragma unroll
        for (int t = 0; t < T; ++t) {
            acc[r][t] = 0.0f;
        }
    }

    for (int ib0 = 0; ib0 < nb_main; ib0 += 8) {
        int2  xq[T];
        float xd[T];
#pragma unroll
        for (int t = 0; t < T; ++t) {
            xq[t] = __ldg(pq + t * 4);
            xd[t] = __ldg(pd + t);
        }
        pq += 8 * T * 4;
        pd += 8 * T;

        uint2 v0[RW], v1[RW];
#pragma unroll
        for (int r = 0; r < RW; ++r) {
            const uint2 * rp = (const uint2 *) (wp0 + row_off[r]);
            v0[r] = nk_tail_ldg_na_u64<L2_EF>(rp + lane, l2_pol);
            v1[r] = (lane >= 30) ? nk_tail_ldg_na_u64<L2_EF>(rp + 2 + lane, l2_pol) : make_uint2(0u, 0u);
        }
        wp0 += 272;

#pragma unroll
        for (int r = 0; r < RW; ++r) {
            const uint32_t a       = __shfl_sync(0xffffffffu, v0[r].x, s_lo);
            const uint32_t b       = __shfl_sync(0xffffffffu, v0[r].y, s_lo);
            const uint32_t c       = __shfl_sync(0xffffffffu, v0[r].x, s_hi);
            const uint32_t d       = __shfl_sync(0xffffffffu, v0[r].y, s_hi);
            const uint32_t u0      = (lane >= 30) ? v1[r].x : (parity ? b : a);
            const uint32_t u1      = (lane >= 30) ? v1[r].y : (parity ? c : b);
            const uint32_t next_w  = parity ? d : c;
            const uint32_t q0      = is_odd ? u0 : __byte_perm(u0, u1, 0x5432);
            const uint32_t q1      = is_odd ? u1 : __byte_perm(u1, next_w, 0x5432);
            const uint32_t d_carry = (g == 3) ? next_w : u0;
            const uint32_t d_word  = __shfl_sync(0xffffffffu, d_carry, d_src_lane);
            const float dw = __half2float(__ushort_as_half(
                (uint16_t) (is_odd ? (d_word >> 16) : (d_word & 0xFFFFu))));
#pragma unroll
            for (int t = 0; t < T; ++t) {
                int dot = __dp4a((int) q0, xq[t].x, 0);
                dot     = __dp4a((int) q1, xq[t].y, dot);
                acc[r][t] = fmaf(__int2float_rn(dot), dw * xd[t], acc[r][t]);
            }
        }
    }

    // 4-block tail (136 B = 17 uint2): lanes 16..31 get zero activation/scale and do not accumulate.
    {
        int2  xq[T];
        float xd[T];
#pragma unroll
        for (int t = 0; t < T; ++t) {
            xq[t] = (lane < 16) ? __ldg(pq + t * 4) : make_int2(0, 0);
            xd[t] = (lane < 16) ? __ldg(pd + t) : 0.0f;
        }

        uint2 v0[RW];
#pragma unroll
        for (int r = 0; r < RW; ++r) {
            const uint2 * rp = (const uint2 *) (wp0 + row_off[r]);
            v0[r] = (lane < 17) ? nk_tail_ldg_na_u64<L2_EF>(rp + lane, l2_pol) : make_uint2(0u, 0u);
        }

#pragma unroll
        for (int r = 0; r < RW; ++r) {
            const uint32_t a       = __shfl_sync(0xffffffffu, v0[r].x, s_lo);
            const uint32_t b       = __shfl_sync(0xffffffffu, v0[r].y, s_lo);
            const uint32_t c       = __shfl_sync(0xffffffffu, v0[r].x, s_hi);
            const uint32_t d       = __shfl_sync(0xffffffffu, v0[r].y, s_hi);
            const uint32_t u0      = parity ? b : a;
            const uint32_t u1      = parity ? c : b;
            const uint32_t next_w  = parity ? d : c;
            const uint32_t q0      = is_odd ? u0 : __byte_perm(u0, u1, 0x5432);
            const uint32_t q1      = is_odd ? u1 : __byte_perm(u1, next_w, 0x5432);
            const uint32_t d_carry = (g == 3) ? next_w : u0;
            const uint32_t d_word  = __shfl_sync(0xffffffffu, d_carry, d_src_lane);
            if (lane < 16) {
                const float dw = __half2float(__ushort_as_half(
                    (uint16_t) (is_odd ? (d_word >> 16) : (d_word & 0xFFFFu))));
#pragma unroll
                for (int t = 0; t < T; ++t) {
                    int dot = __dp4a((int) q0, xq[t].x, 0);
                    dot     = __dp4a((int) q1, xq[t].y, dot);
                    acc[r][t] = fmaf(__int2float_rn(dot), dw * xd[t], acc[r][t]);
                }
            }
        }
    }

    float out = 0.0f;
#pragma unroll
    for (int r = 0; r < RW; ++r) {
#pragma unroll
        for (int t = 0; t < T; ++t) {
            float s = acc[r][t];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                s += __shfl_xor_sync(0xffffffffu, s, off);
            }
            if (lane == r * T + t) {
                out = s;
            }
        }
    }
    if (lane < RW * T) {
        const int r = lane / T;
        const int t = lane - r * T;
        if (row0 + r < nr) {
            dst[(int64_t) t * sd + row0 + r] = out;
        }
    }
#else
    GGML_UNUSED_VARS(vx, qx_q, qx_d, dst, srow, sd, nr, nb);
    NO_DEVICE_CODE;
#endif
}

template <int T, int RW, int NW, int MINB, bool L2_EF>
static void nk_head_grid_tail4_launch_t(
        const char * vx, const int2 * qx_q, const float * qx_d, float * dst,
        const int64_t srow, const int64_t sd, const int nr, const int nb, cudaStream_t stream) {
    const uint32_t grid = (uint32_t) (((int64_t) nr + RW * NW - 1) / (RW * NW));
    k_nk_head_grid_dp4a_tail4<T, RW, NW, MINB, L2_EF><<<grid, 32 * NW, 0, stream>>>(
        vx, qx_q, qx_d, dst, srow, sd, nr, nb);
}

template <bool L2_EF>
static void nk_head_grid_tail4_dispatch(
        const char * vx, const int2 * qx_q, const float * qx_d, float * dst,
        const int64_t srow, const int64_t sd, const int nr, const int nb, const int T, cudaStream_t stream) {
    switch (T) {
        case 1: nk_head_grid_tail4_launch_t<1, 4, 4, 8, L2_EF>(vx, qx_q, qx_d, dst, srow, sd, nr, nb, stream); break;
        case 2: nk_head_grid_tail4_launch_t<2, 4, 4, 8, L2_EF>(vx, qx_q, qx_d, dst, srow, sd, nr, nb, stream); break;
        case 3: nk_head_grid_tail4_launch_t<3, 4, 4, 8, L2_EF>(vx, qx_q, qx_d, dst, srow, sd, nr, nb, stream); break;
        case 4: nk_head_grid_tail4_launch_t<4, 4, 4, 8, L2_EF>(vx, qx_q, qx_d, dst, srow, sd, nr, nb, stream); break;
        case 5: nk_head_grid_tail4_launch_t<5, 4, 4, 8, L2_EF>(vx, qx_q, qx_d, dst, srow, sd, nr, nb, stream); break;
        case 6: nk_head_grid_tail4_launch_t<6, 3, 4, 8, L2_EF>(vx, qx_q, qx_d, dst, srow, sd, nr, nb, stream); break;
        case 7: nk_head_grid_tail4_launch_t<7, 2, 4, 8, L2_EF>(vx, qx_q, qx_d, dst, srow, sd, nr, nb, stream); break;
        case 8: nk_head_grid_tail4_launch_t<8, 2, 4, 8, L2_EF>(vx, qx_q, qx_d, dst, srow, sd, nr, nb, stream); break;
        default: GGML_ABORT("nk_head_grid_dp4a_tail4: invalid T");
    }
}

void ggml_cuda_nk_head_grid_tail4_launch(
        const char  * vx,
        const int2  * qx_q,
        const float * qx_d,
        float       * dst,
        int64_t       srow,
        int64_t       sd,
        int           nr,
        int           nb,
        int           T,
        bool          l2_ef,
        cudaStream_t  stream) {
    if (l2_ef) {
        nk_head_grid_tail4_dispatch<true>(vx, qx_q, qx_d, dst, srow, sd, nr, nb, T, stream);
    } else {
        nk_head_grid_tail4_dispatch<false>(vx, qx_q, qx_d, dst, srow, sd, nr, nb, T, stream);
    }
}

#else

void ggml_cuda_nk_head_grid_tail4_launch(
        const char *, const int2 *, const float *, float *,
        int64_t, int64_t, int, int, int, bool, cudaStream_t) {
    GGML_ABORT("nk_head_grid_dp4a_tail4: unsupported backend");
}

#endif
