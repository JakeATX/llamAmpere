#pragma once

// [PARW] IQ4_XS x f32 for 9..32 activation columns: the multi-slot speculative verify widths (--parallel N x (depth+1)).
// Above MMVQ_MAX_BATCH_SIZE these shapes took MMQ, whose tiles cost ~1.5x (w9-16) to ~1.8x (w17-24) of a w1 weight pass.
// One block owns 16*MR weight rows; its 4 warps split K. Each warp feeds every int8 m16n8k32 MMA's A fragment (two
// weight rows) to NT column tiles of 8, so the weights are read once for all columns. Activations are quantized to int8
// with one scale per 32 elements (the q8_1 granularity MMQ uses), in the IQ4 nibble-interleaved order so a lane's two
// B registers line up with get_int_from_table_16_interleaved.

#include "common.cuh"
#include "vecdotq.cuh"

#include <cstdlib>

static constexpr int IQ4XS_WIDE_MIN_N = MMVQ_MAX_BATCH_SIZE + 1;
static constexpr int IQ4XS_WIDE_MAX_N = 32;
static constexpr int IQ4XS_WIDE_MAX_N_EXT = 48;

static bool ggml_cuda_iq4xs_wide_48_enabled() {
    static const bool value = [] {
        const char * env = getenv("GGML_CUDA_IQ4XS_WIDE_48");
        return env != nullptr && env[0] == '1' && env[1] == '\0';
    }();
    return value;
}

static bool ggml_cuda_iq4xs_wide_stream_enabled() {
    static const bool value = [] {
        const char * env = getenv("GGML_CUDA_IQ4XS_WIDE_STREAM");
        return env != nullptr && env[0] == '1' && env[1] == '\0';
    }();
    return value;
}

// 0 = off, else the row tiles per warp (1, 2 or 4); GGML_CUDA_IQ4XS_WIDE=0 restores MMQ for these widths
static int ggml_cuda_iq4xs_wide_mr() {
    static const int value = [] {
        const char * env = getenv("GGML_CUDA_IQ4XS_WIDE");
        if (env == nullptr) {
            return 2;
        }
        const int v = atoi(env);
        return v == 0 ? 0 : v == 1 ? 1 : v == 4 ? 4 : 2;
    }();
    return value;
}

static __global__ void quantize_iq4xs_wide_q8(
        const float * __restrict__ x, int8_t * __restrict__ yq, float * __restrict__ yd,
        const int ncols_x, const size_t stride_col_x) {
    ggml_cuda_pdl_lc();
    const int col = blockIdx.y;
    const int kb  = blockIdx.x;
    const int nsb = ncols_x / QK_K;
    const int tid = threadIdx.x;

    ggml_cuda_pdl_sync();
    const float xi   = x[(size_t) col * stride_col_x + (size_t) kb * QK_K + tid];
    const float amax = warp_reduce_max(fabsf(xi));
    const float d    = amax / 127.0f;
    const int8_t q   = amax == 0.0f ? 0 : (int8_t) roundf(xi / d);

    // IQ4 interleaved byte p within the sub-block; lane t of the MMA reads its 16 B-fragment ints for the whole
    // 256-block contiguously: int (t, sb, h) at ((col*nsb + kb)*4 + t)*16 + 2*sb + h
    const int ib = tid / 32;
    const int k  = tid % 32;
    const int p  = 2*(k % 16) + k/16;
    const int w  = p / 4;
    yq[((((size_t) col * nsb + kb) * 4 + w/2) * 16 + 2*ib + w%2) * 4 + p%4] = q;
    if (k == 0) {
        yd[(size_t) col * nsb * (QK_K/32) + (size_t) kb * (QK_K/32) + ib] = d;
    }
}

template <int NT, int MR, int NW, int OCC, bool STREAM = false>
__launch_bounds__(32*NW, OCC)
static __global__ void mul_mat_iq4_xs_q8_wide(
        const void * __restrict__ vx, const int8_t * __restrict__ yq, const float * __restrict__ yd, float * __restrict__ dst,
        const int ncols_x, const int ncols_dst, const size_t stride_row_x, const size_t stride_col_dst) {
    ggml_cuda_pdl_lc();

    constexpr int nwarps = NW;
    constexpr int ROWS   = 16 * MR;
    constexpr int COLS   = 8 * NT;

    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int g    = lane / 4;
    const int t    = lane % 4;

    const int row0 = blockIdx.x * ROWS;
    const int nsb  = ncols_x / QK_K;

    // Four padding words give each lane a distinct bank when columns differ by two.
    __shared__ float red[nwarps][COLS][ROWS + (STREAM ? 4 : 0)];

    ggml_cuda_pdl_sync();

    const block_iq4_xs * xr[MR][2];
#pragma unroll
    for (int mr = 0; mr < MR; ++mr) {
        xr[mr][0] = (const block_iq4_xs *) vx + (size_t) (row0 + 16*mr + g)     * stride_row_x;
        xr[mr][1] = (const block_iq4_xs *) vx + (size_t) (row0 + 16*mr + g + 8) * stride_row_x;
    }

    // B: column 8*nt + g; scales: columns 8*nt + 2*t and 8*nt + 2*t + 1 (absent columns read column 0, weighted 0)
    const int8_t * yb[NT];
    const float  * ys0[NT];
    const float  * ys1[NT];
    bool hb[NT], h0[NT], h1[NT];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        const int cb = 8*nt + g;
        const int c0 = 8*nt + 2*t;
        hb[nt]  = cb     < ncols_dst;
        h0[nt]  = c0     < ncols_dst;
        h1[nt]  = c0 + 1 < ncols_dst;
        yb[nt]  = yq + (size_t) (hb[nt] ? cb     : 0) * nsb * QK_K;
        ys0[nt] = yd + (size_t) (h0[nt] ? c0     : 0) * nsb * (QK_K/32);
        ys1[nt] = yd + (size_t) (h1[nt] ? c0 + 1 : 0) * nsb * (QK_K/32);
    }

    float acc[MR][NT][4];
#pragma unroll
    for (int mr = 0; mr < MR; ++mr) {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            acc[mr][nt][0] = acc[mr][nt][1] = acc[mr][nt][2] = acc[mr][nt][3] = 0.0f;
        }
    }

    for (int kb = warp; kb < nsb; kb += nwarps) {
        uint2 sg[MR][2];
        float dg[MR][2];
#pragma unroll
        for (int mr = 0; mr < MR; ++mr) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                sg[mr][h] = *(const uint2 *) (xr[mr][h] + kb);
                dg[mr][h] = __half2float(__ushort_as_half((unsigned short) (sg[mr][h].x & 0xFFFF)));
            }
        }

        // this lane's B fragments (16 ints) and the two columns' 8 sub-block scales, all vector loads
        int4   bq[NT][4];
        float4 s0[NT][2], s1[NT][2];
        if constexpr (!STREAM) {
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const int4 * yb4 = (const int4 *) (yb[nt] + ((size_t) kb * 4 + t) * 64);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    bq[nt][j] = hb[nt] ? yb4[j] : make_int4(0, 0, 0, 0);
                }
                const float4 * q0 = (const float4 *) (ys0[nt] + kb*8);
                const float4 * q1 = (const float4 *) (ys1[nt] + kb*8);
                s0[nt][0] = h0[nt] ? q0[0] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                s0[nt][1] = h0[nt] ? q0[1] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                s1[nt][0] = h1[nt] ? q1[0] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                s1[nt][1] = h1[nt] ? q1[1] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }

#pragma unroll (STREAM ? 1 : 8)
        for (int sb = 0; sb < 8; ++sb) {
            int   b0[NT], b1[NT];
            float d0[NT], d1[NT];
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                if constexpr (STREAM) {
                    const int2 * q = (const int2 *) (yb[nt] + ((size_t) kb * 4 + t) * 64);
                    const int2 v = hb[nt] ? q[sb] : make_int2(0, 0);
                    b0[nt] = v.x;
                    b1[nt] = v.y;
                    d0[nt] = h0[nt] ? ys0[nt][(size_t) kb*8 + sb] : 0.0f;
                    d1[nt] = h1[nt] ? ys1[nt][(size_t) kb*8 + sb] : 0.0f;
                } else {
                    const int4 v = bq[nt][sb/2];
                    b0[nt] = sb % 2 == 0 ? v.x : v.z;
                    b1[nt] = sb % 2 == 0 ? v.y : v.w;
                    const float4 e0 = s0[nt][sb/4];
                    const float4 e1 = s1[nt][sb/4];
                    d0[nt] = sb % 4 == 0 ? e0.x : sb % 4 == 1 ? e0.y : sb % 4 == 2 ? e0.z : e0.w;
                    d1[nt] = sb % 4 == 0 ? e1.x : sb % 4 == 1 ? e1.y : sb % 4 == 2 ? e1.z : e1.w;
                }
            }

#pragma unroll
            for (int mr = 0; mr < MR; ++mr) {
                const int2 va = get_int_from_table_16_interleaved(get_int_b4(xr[mr][0][kb].qs, 4*sb + t), kvalues_iq4nl);
                const int2 vb = get_int_from_table_16_interleaved(get_int_b4(xr[mr][1][kb].qs, 4*sb + t), kvalues_iq4nl);

                const int la = (int) ((sg[mr][0].y >> (4*sb)) & 0x0F) | (int) (((sg[mr][0].x >> (16 + 2*sb)) & 0x03) << 4);
                const int lb = (int) ((sg[mr][1].y >> (4*sb)) & 0x0F) | (int) (((sg[mr][1].x >> (16 + 2*sb)) & 0x03) << 4);
                const float wa = dg[mr][0] * (float) (la - 32);
                const float wb = dg[mr][1] * (float) (lb - 32);

#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE && !defined(GGML_USE_HIP)
                    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                        : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                        : "r"(va.x), "r"(vb.x), "r"(va.y), "r"(vb.y), "r"(b0[nt]), "r"(b1[nt]));
#else
                    GGML_UNUSED_VARS(va, vb, b0, b1);
                    NO_DEVICE_CODE;
#endif
                    acc[mr][nt][0] += (float) c0 * (wa * d0[nt]);
                    acc[mr][nt][1] += (float) c1 * (wa * d1[nt]);
                    acc[mr][nt][2] += (float) c2 * (wb * d0[nt]);
                    acc[mr][nt][3] += (float) c3 * (wb * d1[nt]);
                }
            }
        }
    }

#pragma unroll
    for (int mr = 0; mr < MR; ++mr) {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            red[warp][8*nt + 2*t    ][16*mr + g    ] = acc[mr][nt][0];
            red[warp][8*nt + 2*t + 1][16*mr + g    ] = acc[mr][nt][1];
            red[warp][8*nt + 2*t    ][16*mr + g + 8] = acc[mr][nt][2];
            red[warp][8*nt + 2*t + 1][16*mr + g + 8] = acc[mr][nt][3];
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < COLS * ROWS; i += nwarps * 32) {
        const int c = i / ROWS;
        const int r = i % ROWS;
        if (c < ncols_dst) {
            float s = 0.0f;
#pragma unroll
            for (int w = 0; w < nwarps; ++w) {
                s += red[w][c][r];
            }
            dst[(size_t) c * stride_col_dst + row0 + r] = s;
        }
    }
}

template <int MR, int OCC, bool STREAM = false>
static void ggml_cuda_mul_mat_iq4xs_wide_launch(
        const int nt, const dim3 grid, cudaStream_t stream, const void * vx, const int8_t * yq, const float * yd, float * dst,
        const int ncols_x, const int ncols_dst, const size_t stride_row_x, const size_t stride_col_dst) {
    constexpr int NW = 4;
    const ggml_cuda_kernel_launch_params mat(grid, dim3(32*NW), 0, stream);
    switch (nt) {
        case 2: ggml_cuda_kernel_launch(mul_mat_iq4_xs_q8_wide<2, MR, NW, OCC, STREAM>, mat, vx, yq, yd, dst, ncols_x, ncols_dst, stride_row_x, stride_col_dst); break;
        case 3: ggml_cuda_kernel_launch(mul_mat_iq4_xs_q8_wide<3, MR, NW, OCC, STREAM>, mat, vx, yq, yd, dst, ncols_x, ncols_dst, stride_row_x, stride_col_dst); break;
        case 4: ggml_cuda_kernel_launch(mul_mat_iq4_xs_q8_wide<4, MR, NW, OCC, STREAM>, mat, vx, yq, yd, dst, ncols_x, ncols_dst, stride_row_x, stride_col_dst); break;
        case 5:
            if constexpr (STREAM) {
                ggml_cuda_kernel_launch(mul_mat_iq4_xs_q8_wide<5, MR, NW, OCC, STREAM>, mat, vx, yq, yd, dst, ncols_x, ncols_dst, stride_row_x, stride_col_dst);
                break;
            }
            GGML_ABORT("iq4xs wide: extended columns need streaming loads");
        case 6:
            if constexpr (STREAM && MR <= 2) {
                ggml_cuda_kernel_launch(mul_mat_iq4_xs_q8_wide<6, MR, NW, OCC, STREAM>, mat, vx, yq, yd, dst, ncols_x, ncols_dst, stride_row_x, stride_col_dst);
                break;
            }
            GGML_ABORT("iq4xs wide: 48 columns need streaming loads and MR <= 2");
        default: GGML_ABORT("iq4xs wide: bad column tile count %d", nt);
    }
}

// x: f32 activations, ncols_dst columns of ncols_x; dst: ncols_dst columns of nrows_x. Caller checks the shape.
static void ggml_cuda_mul_mat_iq4xs_wide(
        ggml_backend_cuda_context & ctx, const void * vx, const float * x, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const size_t stride_row_x, const size_t stride_col_x, const size_t stride_col_dst) {
    const int nsb = ncols_x / QK_K;
    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<int8_t> buf(ctx.pool(), (size_t) ncols_dst * ncols_x + (size_t) ncols_dst * nsb * (QK_K/32) * sizeof(float));
    int8_t * yq_d = buf.get();
    float  * yd_d = (float *) (yq_d + (size_t) ncols_dst * ncols_x);

    const ggml_cuda_kernel_launch_params quant(dim3(nsb, ncols_dst), dim3(QK_K), 0, stream);
    ggml_cuda_kernel_launch(quantize_iq4xs_wide_q8, quant, x, yq_d, yd_d, ncols_x, stride_col_x);

    const int nt = (ncols_dst + 7) / 8;
    int mr = ggml_cuda_iq4xs_wide_mr();
    while (mr > 1 && nrows_x % (16*mr) != 0) {
        mr /= 2;
    }
    // MR=4 at NT=6 exceeds the 48 KiB static shared memory limit.
    if (nt == 6 && mr == 4) {
        mr = 2;
    }
    const dim3 grid(nrows_x / (16*mr));
    if (ncols_dst > IQ4XS_WIDE_MAX_N || ggml_cuda_iq4xs_wide_stream_enabled()) {
        switch (mr) {
            case 4: ggml_cuda_mul_mat_iq4xs_wide_launch<4, 2, true>(nt, grid, stream, vx, yq_d, yd_d, dst, ncols_x, ncols_dst, stride_row_x, stride_col_dst); break;
            case 2: ggml_cuda_mul_mat_iq4xs_wide_launch<2, 2, true>(nt, grid, stream, vx, yq_d, yd_d, dst, ncols_x, ncols_dst, stride_row_x, stride_col_dst); break;
            default: ggml_cuda_mul_mat_iq4xs_wide_launch<1, 2, true>(nt, grid, stream, vx, yq_d, yd_d, dst, ncols_x, ncols_dst, stride_row_x, stride_col_dst); break;
        }
        return;
    }
    switch (mr) {
        case 4: ggml_cuda_mul_mat_iq4xs_wide_launch<4, 2>(nt, grid, stream, vx, yq_d, yd_d, dst, ncols_x, ncols_dst, stride_row_x, stride_col_dst); break;
        case 2: ggml_cuda_mul_mat_iq4xs_wide_launch<2, 2>(nt, grid, stream, vx, yq_d, yd_d, dst, ncols_x, ncols_dst, stride_row_x, stride_col_dst); break;
        default: ggml_cuda_mul_mat_iq4xs_wide_launch<1, 2>(nt, grid, stream, vx, yq_d, yd_d, dst, ncols_x, ncols_dst, stride_row_x, stride_col_dst); break;
    }
}
