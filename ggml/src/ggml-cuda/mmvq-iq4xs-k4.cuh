#pragma once

#include "common.cuh"
#include "vecdotq.cuh"

#include <cstdlib>
#include <cstring>

static constexpr int IQ4XS_TC_ROWS = 16;

enum iq4xs_k4_variant { IQ4XS_K4_OFF, IQ4XS_K4, IQ4XS_K4A };

static iq4xs_k4_variant ggml_cuda_iq4xs_k4_variant() {
    static const iq4xs_k4_variant value = [] {
        const char * env = getenv("GGML_CUDA_IQ4XS_K4");
        if (env && strcmp(env, "k4") == 0) {
            return IQ4XS_K4;
        }
        if (env && strcmp(env, "k4a") == 0) {
            return IQ4XS_K4A;
        }
        return IQ4XS_K4_OFF;
    }();
    return value;
}

static int ggml_cuda_iq4xs_k4_min_n() {
    static const int value = [] {
        const int fallback = ggml_cuda_iq4xs_k4_variant() == IQ4XS_K4A ? 5 : 6;
        const char * env = getenv("GGML_CUDA_IQ4XS_K4_MIN_N");
        if (!env) {
            return fallback;
        }
        char * end = nullptr;
        const long n = strtol(env, &end, 10);
        return end != env && *end == '\0' && n >= 1 && n <= 8 ? (int) n : fallback;
    }();
    return value;
}

// k4a stores each lane's activation words together for int4 loads.
template <bool vec>
static __global__ void quantize_iq4xs_tc_q8k(
        const float * __restrict__ x, int8_t * __restrict__ yq, float * __restrict__ yd,
        const int ncols_x, const size_t stride_col_x) {
    ggml_cuda_pdl_lc();
    const int col = blockIdx.y;
    const int kb  = blockIdx.x;
    const int nsb = ncols_x / QK_K;
    const int tid = threadIdx.x;

    __shared__ float smax[8];

    ggml_cuda_pdl_sync();
    const float xi = x[(size_t) col * stride_col_x + (size_t) kb * QK_K + tid];
    float amax = warp_reduce_max(fabsf(xi));
    if (tid % 32 == 0) {
        smax[tid / 32] = amax;
    }
    __syncthreads();
    amax = smax[0];
#pragma unroll
    for (int i = 1; i < 8; ++i) {
        amax = fmaxf(amax, smax[i]);
    }
    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : (int8_t) roundf(xi / d);

    const int ib = tid / 32;
    const int k  = tid % 32;
    const int pos = ib * 32 + 2*(k % 16) + k/16;
    const int posv = vec ? 4 * (((k % 16) / 4) * 16 + ib * 2 + ((k % 16) / 2) % 2) + pos % 4 : pos;
    yq[(size_t) col * nsb * QK_K + (size_t) kb * QK_K + posv] = q;
    if (tid == 0) {
        yd[col * nsb + kb] = d;
    }
}

// Each warp owns 16 rows and every fourth K block; integer sums reset after 256 elements.
// The bound 256 * 127 * 127 * 32 = 132128768 fits int32.
template <bool vec>
__launch_bounds__(128, 1)
static __global__ void mul_mat_iq4_xs_q8k_tc(
        const void * __restrict__ vx, const int8_t * __restrict__ yq, const float * __restrict__ yd, float * __restrict__ dst,
        const int ncols_x, const int ncols_dst, const size_t stride_row_x, const size_t stride_col_dst) {
    ggml_cuda_pdl_lc();

    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int g    = lane / 4;
    const int t    = lane % 4;

    const int row0 = blockIdx.x * IQ4XS_TC_ROWS;
    const int nsb  = ncols_x / QK_K;

    constexpr int nwarps = 4;

    __shared__ float red[nwarps][IQ4XS_TC_ROWS][8];

    ggml_cuda_pdl_sync();

    const block_iq4_xs * xg  = (const block_iq4_xs *) vx + (size_t) (row0 + g)     * stride_row_x;
    const block_iq4_xs * xg8 = (const block_iq4_xs *) vx + (size_t) (row0 + g + 8) * stride_row_x;

    const bool     has_col = g < ncols_dst;
    const int8_t * yg      = yq + (size_t) (has_col ? g : 0) * nsb * QK_K;
    const bool     has_c0  = 2*t     < ncols_dst;
    const bool     has_c1  = 2*t + 1 < ncols_dst;
    const float *  yd0     = yd + (has_c0 ? 2*t     : 0) * nsb;
    const float *  yd1     = yd + (has_c1 ? 2*t + 1 : 0) * nsb;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int kb = warp; kb < nsb; kb += nwarps) {
        const block_iq4_xs * xbg  = xg  + kb;
        const block_iq4_xs * xbg8 = xg8 + kb;

        const uint2 sg  = *(const uint2 *) xbg;
        const uint2 sg8 = *(const uint2 *) xbg8;

        const int  * ybg  = (const int  *) (yg + (size_t) kb * QK_K);
        const int4 * ybg4 = (const int4 *) (yg + (size_t) kb * QK_K) + 4*t;

        int acc_i[4] = {0, 0, 0, 0};

#pragma unroll
        for (int sb = 0; sb < 8; ++sb) {
            const int q4g  = get_int_b4(xbg->qs,  4*sb + t);
            const int q4g8 = get_int_b4(xbg8->qs, 4*sb + t);

            int b0 = 0, b1 = 0;
            if constexpr (vec) {
                if (has_col) {
                    const int4 v = ybg4[sb/2];
                    b0 = (sb % 2 == 0) ? v.x : v.z;
                    b1 = (sb % 2 == 0) ? v.y : v.w;
                }
            } else if (has_col) {
                b0 = ybg[sb*8 + 2*t];
                b1 = ybg[sb*8 + 2*t + 1];
            }

            const int2 vg  = get_int_from_table_16_interleaved(q4g,  kvalues_iq4nl);
            const int2 vg8 = get_int_from_table_16_interleaved(q4g8, kvalues_iq4nl);

            // A holds rows g/g+8, B holds column g; C holds rows g/g+8 at columns 2*t/2*t+1.
            int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE && !defined(GGML_USE_HIP)
            asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                : "r"(vg.x), "r"(vg8.x), "r"(vg.y), "r"(vg8.y), "r"(b0), "r"(b1));
#else
            GGML_UNUSED_VARS(vg, vg8, b0, b1);
            NO_DEVICE_CODE;
#endif

            const int lsg  = (int) ((sg.y  >> (4*sb)) & 0x0F) | (int) (((sg.x  >> (16 + 2*sb)) & 0x03) << 4);
            const int lsg8 = (int) ((sg8.y >> (4*sb)) & 0x0F) | (int) (((sg8.x >> (16 + 2*sb)) & 0x03) << 4);

            acc_i[0] += c0 * (lsg  - 32);
            acc_i[1] += c1 * (lsg  - 32);
            acc_i[2] += c2 * (lsg8 - 32);
            acc_i[3] += c3 * (lsg8 - 32);
        }

        const float dg  = __half2float(__ushort_as_half((unsigned short) (sg.x  & 0xFFFF)));
        const float dg8 = __half2float(__ushort_as_half((unsigned short) (sg8.x & 0xFFFF)));
        const float dc0 = has_c0 ? yd0[kb] : 0.0f;
        const float dc1 = has_c1 ? yd1[kb] : 0.0f;
        acc[0] += (float) acc_i[0] * (dg  * dc0);
        acc[1] += (float) acc_i[1] * (dg  * dc1);
        acc[2] += (float) acc_i[2] * (dg8 * dc0);
        acc[3] += (float) acc_i[3] * (dg8 * dc1);
    }

    red[warp][g][2*t]         = acc[0];
    red[warp][g][2*t + 1]     = acc[1];
    red[warp][g + 8][2*t]     = acc[2];
    red[warp][g + 8][2*t + 1] = acc[3];
    __syncthreads();

    for (int i = threadIdx.x; i < IQ4XS_TC_ROWS * 8; i += nwarps * 32) {
        const int r = i / 8;
        const int c = i % 8;
        if (c < ncols_dst) {
            float s = 0.0f;
#pragma unroll
            for (int w = 0; w < nwarps; ++w) {
                s += red[w][r][c];
            }
            dst[(size_t) c * stride_col_dst + row0 + r] = s;
        }
    }
}

template <bool vec>
static void ggml_cuda_mul_mat_iq4xs_k4(
        ggml_backend_cuda_context & ctx, const void * vx, const float * x, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const size_t stride_row_x, const size_t stride_col_x, const size_t stride_col_dst) {
    const int nsb = ncols_x / QK_K;
    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<int8_t> yq(ctx.pool(), (size_t) ncols_dst * ncols_x + (size_t) ncols_dst * nsb * sizeof(float));
    int8_t * yq_d = yq.get();
    float * yd_d = (float *) (yq_d + (size_t) ncols_dst * ncols_x);

    const ggml_cuda_kernel_launch_params quant(dim3(nsb, ncols_dst), dim3(QK_K), 0, stream);
    ggml_cuda_kernel_launch(quantize_iq4xs_tc_q8k<vec>, quant, x, yq_d, yd_d, ncols_x, stride_col_x);

    const ggml_cuda_kernel_launch_params mat(dim3(nrows_x / IQ4XS_TC_ROWS), dim3(128), 0, stream);
    ggml_cuda_kernel_launch(mul_mat_iq4_xs_q8k_tc<vec>, mat, vx, (const int8_t *) yq_d, (const float *) yd_d, dst,
                           ncols_x, ncols_dst, stride_row_x, stride_col_dst);

    static const bool logged = [=] {
        GGML_LOG_DEBUG("ggml_cuda_iq4xs_k4: route=%s n=%d min_n=%d K=%d M=%d\n", vec ? "k4a" : "k4",
                       ncols_dst, ggml_cuda_iq4xs_k4_min_n(), ncols_x, nrows_x);
        return true;
    }();
    GGML_UNUSED(logged);
}
