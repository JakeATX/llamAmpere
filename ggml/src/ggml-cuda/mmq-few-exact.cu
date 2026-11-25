#include "mmq-few-exact.cuh"
#include "cp-async.cuh"
#include "mma.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "vecdotq.cuh"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstring>
#include <type_traits>

// NVIDIA Ampere+ only: uses int8 MMA fragment layout and cp.async.
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

using namespace ggml_cuda_mma;

// Row-parallel bit-exact MMQ for Q4_K and Q6_K (2..8 columns):
// Each warp owns 16 contiguous rows strictly inside one 128-row MMQ tile (no __syncthreads(),
// no cross-warp reduction, no stream-k fixup launch). Activations are quantized with the exact
// quantize_mmq_q8_1 unquantized-sum formula, super-block dots match mmq-load-tiles.cuh +
// mmq-vec-dot.cuh, and split tiles replay mul_mat_q + mul_mat_q_stream_k_fixup segments in order.

static constexpr int MMQ_EXACT_NROWS   = 16;
static constexpr int MMQ_EXACT_NSTAGES = 2;

template <ggml_type type>
static constexpr __host__ __device__ int mmq_exact_unit_size() {
    return QK_K / ggml_cuda_type_traits<type>::qk * ggml_cuda_type_traits<type>::bs;
}

template <ggml_type type>
static constexpr __host__ __device__ bool mmq_exact_unaligned() {
    return (mmq_exact_unit_size<type>() % 16) != 0;
}

template <ggml_type type>
static constexpr __host__ __device__ int mmq_exact_nchunks() {
    constexpr int nbytes = mmq_exact_unit_size<type>();
    return mmq_exact_unaligned<type>() ? (14 + nbytes + 15) / 16 : nbytes / 16;
}

template <ggml_type type>
static constexpr __host__ __device__ int mmq_exact_sram_stride() {
    constexpr int nchunks = mmq_exact_nchunks<type>();
    return (mmq_exact_unaligned<type>() && (nchunks % 2 == 0)) ? 16 * (nchunks + 1) : 16 * nchunks;
}

template <ggml_type type, int nwarps>
static constexpr __host__ __device__ int mmq_exact_nbytes_shared() {
    return nwarps * MMQ_EXACT_NSTAGES * MMQ_EXACT_NROWS * mmq_exact_sram_stride<type>();
}

static constexpr __host__ __device__ mmq_q8_1_ds_layout mmq_exact_ds_layout(const ggml_type type) {
    return type == GGML_TYPE_Q4_K ? MMQ_Q8_1_DS_LAYOUT_DS4 : MMQ_Q8_1_DS_LAYOUT_D4;
}

// Quantize 2D contiguous src1 (<= 8 columns) into the single-column-group MMQ_FEW MMA layout
// using the exact quantize_mmq_q8_1 formula (unquantized F32 sum for DS4).
template <mmq_q8_1_ds_layout ds_layout>
static __global__ void quantize_mmq_few_exact_q8_1(
        const float * x_ptr, void * vy_ptr, const int64_t ne00, const int64_t s01, const int ne1) {
    ggml_cuda_pdl_lc();
    const float * GGML_CUDA_RESTRICT x  = x_ptr;
    int4        * GGML_CUDA_RESTRICT vy = (int4 *) vy_ptr;
    const int64_t i0 = ((int64_t) blockDim.x * blockIdx.y + threadIdx.x) * 4;

    if (i0 >= ne00) {
        return;
    }
    ggml_cuda_pdl_sync();

    const int col = blockIdx.x;
    const float4 xi = col < ne1
        ? *(const float4 *) (x + int64_t(col) * s01 + i0)
        : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));

#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }

    [[maybe_unused]] float sum = 0.0f;
    if constexpr (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = xi.x + xi.y + xi.z + xi.w;
#pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
        }
    }

    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x * d_inv);
    q.y = roundf(xi.y * d_inv);
    q.z = roundf(xi.z * d_inv);
    q.w = roundf(xi.w * d_inv);
    const float d = 1.0f / d_inv;

    int q_packed = 0;
    memcpy(&q_packed, &q, sizeof(q_packed));

    const int s = (i0 % QK_K) / QK8_1;
    const int k = i0 % QK8_1;
    const int g = col;
    int4 * yg = vy + (i0 / QK_K) * MMQ_FEW_Q8_1_INT4;
    ((int *) (yg + (s / 2) * WARP_SIZE + 4 * g + (k % 16) / 4))[2 * (s % 2) + k / 16] = q_packed;
    if (k == 0) {
        int4 * ds = yg + 4 * WARP_SIZE + 4 * (g / 2) + s / 2;
        if constexpr (ds_layout == MMQ_Q8_1_DS_LAYOUT_D4) {
            ((float *) ds)[2 * (s % 2) + g % 2] = d;
        } else {
            ((half2 *) ds)[2 * (s % 2) + g % 2] = make_half2(d, sum);
        }
    }
}

#if defined(CP_ASYNC_AVAILABLE)

struct mmq_exact_y_frag {
    int4 qs[4];
    int4 ds[4];
};

static __device__ __forceinline__ void mmq_exact_load_y(mmq_exact_y_frag & yf, const int4 * y) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        yf.qs[j] = y[j * WARP_SIZE + threadIdx.x];
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        yf.ds[j] = y[4 * WARP_SIZE + 4 * (threadIdx.x % 4) + j];
    }
}

static __device__ __forceinline__ int2 mmq_exact_get_b(const mmq_exact_y_frag & yf, const int s) {
    const int4 q = yf.qs[s / 2];
    return (s % 2 == 0) ? make_int2(q.x, q.y) : make_int2(q.z, q.w);
}

static __device__ __forceinline__ tile<16, 8, int> mmq_exact_mma(const int (&lo)[2], const int (&hi)[2], const int2 b) {
    tile<16, 8, int> A;
    A.x[0] = lo[0];
    A.x[1] = lo[1];
    A.x[2] = hi[0];
    A.x[3] = hi[1];
    tile<8, 8, int> B;
    B.x[0] = b.x;
    B.x[1] = b.y;
    tile<16, 8, int> C;
    mma(C, A, B);
    return C;
}

static __device__ __forceinline__ tile<16, 8, int> mmq_exact_mma_k16(const int (&a)[2], const int b) {
    tile<16, 4, int> A;
    A.x[0] = a[0];
    A.x[1] = a[1];
    tile<8, 4, int> B;
    B.x[0] = b;
    tile<16, 8, int> C;
    mma(C, A, B);
    return C;
}

// Matches ggml_cuda_mmq_load_tiles_q4_K + ggml_cuda_mmq_vec_dot_q4_K_q8_1_mma bit-for-bit.
static __device__ __forceinline__ void mmq_exact_vec_dot_q4_K(
        const char * const (&x)[2], const mmq_exact_y_frag & yf, float (&sum)[4]) {
    const int t = threadIdx.x % 4;

    int   qs[2][4][2];
    int   sc[2][2];
    int   mn[2][2];
    half2 dm_w[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const block_q4_K * bx = (const block_q4_K *) x[r];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            qs[r][j][0] = get_int_b4(bx->qs + 32 * j,      t);
            qs[r][j][1] = get_int_b4(bx->qs + 32 * j + 16, t);
        }
        const int4 hdr = *(const int4 *) bx; // dm and 12 scale bytes
        const int scales[3] = { hdr.y, hdr.z, hdr.w };
        sc[r][0] = unpack_scales_q45_K(scales, 0);
        sc[r][1] = unpack_scales_q45_K(scales, 1);
        mn[r][0] = unpack_scales_q45_K(scales, 2);
        mn[r][1] = unpack_scales_q45_K(scales, 3);
        dm_w[r]  = bx->dm * make_half2(1.0f, -1.0f);
    }

#pragma unroll
    for (int s = 0; s < 8; ++s) {
        const int j     = s / 2;
        const int shift = 4 * (s % 2);
        int lo[2];
        int hi[2];
        float2 dmA[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            lo[r] = (qs[r][j][0] >> shift) & 0x0F0F0F0F;
            hi[r] = (qs[r][j][1] >> shift) & 0x0F0F0F0F;
            const uint8_t sc8 = (sc[r][s / 4] >> (8 * (s % 4))) & 0xFF;
            const uint8_t m8  = (mn[r][s / 4] >> (8 * (s % 4))) & 0xFF;
            dmA[r] = __half22float2(dm_w[r] * make_half2(sc8, m8));
        }

        const tile<16, 8, int> C = mmq_exact_mma(lo, hi, mmq_exact_get_b(yf, s));

        const half2 * y_ds = (const half2 *) &yf.ds[s / 2];
        const float2 dsB[2] = {
            __half22float2(y_ds[2 * (s % 2) + 0]),
            __half22float2(y_ds[2 * (s % 2) + 1]),
        };

#pragma unroll
        for (int l = 0; l < 4; ++l) {
            sum[l] += dmA[l / 2].x * dsB[l % 2].x * C.x[l];
            sum[l] += dmA[l / 2].y * dsB[l % 2].y;
        }
    }
}

// Matches ggml_cuda_mmq_load_tiles_q6_K + ggml_cuda_mmq_vec_dot_q6_K_q8_1_mma bit-for-bit
// (two 128-value halves per 256-value super-block, each scaled by dA once).
static __device__ __forceinline__ void mmq_exact_vec_dot_q6_K(
        const char * const (&x)[2], const mmq_exact_y_frag & yf, float (&sum)[4]) {
    const int t = threadIdx.x % 4;

    float dA[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        dA[r] = __half2float(((const block_q6_K *) x[r])->d);
    }

#pragma unroll
    for (int n = 0; n < 2; ++n) {
        int ql[2][2][2];
        int qh[2][2];
        int sc[2][2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const block_q6_K * bx = (const block_q6_K *) x[r];
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                ql[r][k][0] = get_int_b2(bx->ql + 64 * n + 32 * k,      t);
                ql[r][k][1] = get_int_b2(bx->ql + 64 * n + 32 * k + 16, t);
            }
            qh[r][0] = get_int_b2(bx->qh + 32 * n,      t);
            qh[r][1] = get_int_b2(bx->qh + 32 * n + 16, t);
            sc[r][0] = get_int_b2(bx->scales + 8 * n, 0);
            sc[r][1] = get_int_b2(bx->scales + 8 * n, 1);
        }

        float tmp[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int s = 4 * n + j;
            int lo[2];
            int hi[2];
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                lo[r] = __vsub4(((ql[r][j % 2][0] >> (4 * (j / 2))) & 0x0F0F0F0F) | (((qh[r][0] >> (2 * j)) & 0x03030303) << 4), 0x20202020);
                hi[r] = __vsub4(((ql[r][j % 2][1] >> (4 * (j / 2))) & 0x0F0F0F0F) | (((qh[r][1] >> (2 * j)) & 0x03030303) << 4), 0x20202020);
            }
            const int sc_s[2]  = { sc[0][j / 2] >> (16 * (j % 2)), sc[1][j / 2] >> (16 * (j % 2)) };
            const int sc_lo[2] = { (int8_t) sc_s[0],        (int8_t) sc_s[1] };
            const int sc_hi[2] = { (int8_t) (sc_s[0] >> 8), (int8_t) (sc_s[1] >> 8) };

            const int2 b = mmq_exact_get_b(yf, s);
            const tile<16, 8, int> C_lo = mmq_exact_mma_k16(lo, b.x);
            const tile<16, 8, int> C_hi = mmq_exact_mma_k16(hi, b.y);

            const float * y_d = (const float *) &yf.ds[s / 2];
            const float dB[2] = { y_d[2 * (s % 2) + 0], y_d[2 * (s % 2) + 1] };

#pragma unroll
            for (int l = 0; l < 4; ++l) {
                tmp[l] += (C_lo.x[l] * sc_lo[l / 2] + C_hi.x[l] * sc_hi[l / 2]) * dB[l % 2];
            }
        }
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            sum[l] += tmp[l] * dA[l / 2];
        }
    }
}

#endif // defined(CP_ASYNC_AVAILABLE)

template <ggml_type type, int nwarps, bool is_stream_k_split>
__launch_bounds__(nwarps * ggml_cuda_get_physical_warp_size(), 2)
static __global__ void mul_mat_q_few_exact(
        const char * GGML_CUDA_RESTRICT x, const int4 * GGML_CUDA_RESTRICT y, float * GGML_CUDA_RESTRICT dst,
        const int nunits, const int nrows_x, const int ncols_dst, const int64_t stride_row_x,
        const int stride_col_dst, const int nti, const int nblocks_stream_k, const bool x_is_constant) {
#if defined(CP_ASYNC_AVAILABLE)
    constexpr int  warp_size    = ggml_cuda_get_physical_warp_size();
    static_assert(warp_size == WARP_SIZE, "unsupported warp size");
    static_assert(type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q6_K, "unsupported type");
    static_assert(128 % (nwarps * MMQ_EXACT_NROWS) == 0, "CTA rows must divide MMQ I=128 tile");

    constexpr int  unit_size    = mmq_exact_unit_size<type>();
    constexpr bool unaligned    = mmq_exact_unaligned<type>();
    constexpr int  nchunks      = mmq_exact_nchunks<type>();
    constexpr int  sram_stride  = mmq_exact_sram_stride<type>();
    constexpr int  nbytes_stage = MMQ_EXACT_NROWS * sram_stride;
    constexpr int  ncopies      = MMQ_EXACT_NROWS * nchunks;

    const int row0 = (blockIdx.x * nwarps + threadIdx.y) * MMQ_EXACT_NROWS;
    if (row0 >= nrows_x) {
        return;
    }

    extern __shared__ char data_mmq_exact[];
    char * ring = data_mmq_exact + threadIdx.y * MMQ_EXACT_NSTAGES * nbytes_stage;

    auto load_stage = [&](const int stage, const int iu) {
        const unsigned int dst_stage = ggml_cuda_cvta_generic_to_shared(ring + stage * nbytes_stage);
#pragma unroll
        for (int c0 = 0; c0 < ncopies; c0 += warp_size) {
            const int c  = c0 + threadIdx.x;
            const int i  = c / nchunks;
            const int ic = c % nchunks;
            if (ncopies % warp_size == 0 || c < ncopies) {
                const int row = min(row0 + i, nrows_x - 1);
                const char * src = x + int64_t(row) * stride_row_x + int64_t(iu) * unit_size;
                if constexpr (unaligned) {
                    const int off = (uintptr_t) src % 16;
                    if (16 * ic < off + unit_size) {
                        cp_async_cg_16<0>(dst_stage + i * sram_stride + 16 * ic, src - off + 16 * ic);
                    }
                } else {
                    cp_async_cg_16<0>(dst_stage + i * sram_stride + 16 * ic, src + 16 * ic);
                }
            }
        }
    };

    int off_row[2] = { 0, 0 };
    if constexpr (unaligned) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int row = min(row0 + int(threadIdx.x) / 4 + 8 * r, nrows_x - 1);
            off_row[r] = (int) ((uintptr_t) (x + int64_t(row) * stride_row_x) & 15);
        }
    }

    if (!x_is_constant) {
        ggml_cuda_pdl_sync();
    }

    // Run one contiguous forward K segment [iu_begin, iu_end) with 2-stage cp.async overlap.
    bool pdl_synced = !x_is_constant;
    auto run_segment = [&](const int iu_begin, const int iu_end, float (&seg)[4]) {
        const int seg_len = iu_end - iu_begin;
        load_stage(0, iu_begin);
        cp_async_commit_group();

        if (!pdl_synced) {
            ggml_cuda_pdl_sync();
            pdl_synced = true;
        }

        mmq_exact_y_frag yf;
        mmq_exact_load_y(yf, y + int64_t(iu_begin) * MMQ_FEW_Q8_1_INT4);

        for (int idx = 0; idx < seg_len; ++idx) {
            const int iu = iu_begin + idx;
            if (idx + 1 < seg_len) {
                load_stage((idx + 1) & 1, iu + 1);
            }
            cp_async_commit_group();
            cp_async_wait_group<1>();
            ggml_cuda_syncwarp();

            const char * xs = ring + (idx & 1) * nbytes_stage;
            const int off_stage = unaligned ? (((iu & 15) * (unit_size & 15)) & 15) : 0;
            const char * const xr[2] = {
                xs + (threadIdx.x / 4 + 0) * sram_stride + ((off_row[0] + off_stage) & 15),
                xs + (threadIdx.x / 4 + 8) * sram_stride + ((off_row[1] + off_stage) & 15),
            };
            if constexpr (type == GGML_TYPE_Q4_K) {
                mmq_exact_vec_dot_q4_K(xr, yf, seg);
            } else {
                mmq_exact_vec_dot_q6_K(xr, yf, seg);
            }

            if (idx + 1 < seg_len) {
                mmq_exact_load_y(yf, y + int64_t(iu + 1) * MMQ_FEW_Q8_1_INT4);
            }
            ggml_cuda_syncwarp();
        }
        cp_async_wait_all();
    };

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    if constexpr (!is_stream_k_split) {
        run_segment(0, nunits, acc);
    } else {
        const int it = row0 / 128;
        const int T0 = it * nunits;
        const int T1 = T0 + nunits;
        const int total_work = nti * nunits;
        auto kbc = [&](const int b) -> int {
            return (int) ((int64_t(b) * total_work) / nblocks_stream_k);
        };

        int b_last = (int) ((int64_t(T1) * nblocks_stream_k - 1) / total_work);
        while (b_last + 1 < nblocks_stream_k && kbc(b_last + 1) < T1) {
            ++b_last;
        }
        while (b_last > 0 && kbc(b_last) >= T1) {
            --b_last;
        }

        const int k_last_start = kbc(b_last);
        if (k_last_start <= T0) {
            run_segment(0, nunits, acc);
        } else {
            run_segment(k_last_start - T0, nunits, acc);
            float acc_fix[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
            int b = b_last - 1;
            while (b >= 0) {
                const int k0 = kbc(b);
                const int k1 = kbc(b + 1);
                if (k1 > k0) {
                    const int seg_lo = max(T0, k0) - T0;
                    const int seg_hi = min(T1, k1) - T0;
                    float seg[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
                    run_segment(seg_lo, seg_hi, seg);
#pragma unroll
                    for (int l = 0; l < 4; ++l) {
                        acc_fix[l] += seg[l];
                    }
                    if (k0 <= T0) {
                        break;
                    }
                }
                --b;
            }
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                acc[l] += acc_fix[l];
            }
        }
    }

    ggml_cuda_pdl_lc();

#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const int i = row0 + tile<16, 8, int>::get_i(l);
        const int j = tile<16, 8, int>::get_j(l);
        if (i < nrows_x && j < ncols_dst) {
            dst[int64_t(j) * stride_col_dst + i] = acc[l];
        }
    }
#else
    GGML_UNUSED_VARS(x, y, dst, nunits, nrows_x, ncols_dst, stride_row_x,
        stride_col_dst, nti, nblocks_stream_k, x_is_constant);
    NO_DEVICE_CODE;
#endif // defined(CP_ASYNC_AVAILABLE)
}

static __global__ void mmq_few_exact_check_kernel(const float * actual, const float * reference, const int64_t n) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n && __float_as_uint(actual[i]) != __float_as_uint(reference[i])) {
#if defined(__CUDA_ARCH__)
        // Active in Release builds too; zero host sync so CUDA graph capture works.
        asm volatile("trap;");
#endif
    }
}

static bool mmq_exact_is_constant(const ggml_tensor * src0) {
    const ggml_tensor * x_root = src0->view_src ? src0->view_src : src0;
    return x_root->buffer && ggml_backend_buffer_get_usage(x_root->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        !(x_root->flags & GGML_TENSOR_FLAG_PARAM);
}

template <ggml_type type, int nwarps, bool is_stream_k_split>
static void launch_mul_mat_q_few_exact_impl(
        const ggml_tensor * src0, ggml_tensor * dst, const int4 * src1_q,
        const int nti, const int nblocks_stream_k, cudaStream_t stream) {
    const int nunits        = (int) (src0->ne[0] / QK_K);
    const int nrows_x       = (int) src0->ne[1];
    const int ncols_dst     = (int) dst->ne[1];
    const int warp_size     = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int nbytes_shared = mmq_exact_nbytes_shared<type, nwarps>();
    CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q_few_exact<type, nwarps, is_stream_k_split>), nbytes_shared);

    const bool x_is_constant = mmq_exact_is_constant(src0);
    const int  rows_per_cta  = nwarps * MMQ_EXACT_NROWS;
    const dim3 block_nums((nrows_x + rows_per_cta - 1) / rows_per_cta, 1, 1);
    const dim3 block_dims(warp_size, nwarps, 1);
    const ggml_cuda_kernel_launch_params params(block_nums, block_dims, nbytes_shared, stream);
    ggml_cuda_kernel_launch(mul_mat_q_few_exact<type, nwarps, is_stream_k_split>, params,
        (const char *) src0->data, src1_q, (float *) dst->data,
        nunits, nrows_x, ncols_dst, (int64_t) src0->nb[1], (int) (dst->nb[1] / sizeof(float)),
        nti, nblocks_stream_k, x_is_constant);
}

template <ggml_type type>
static void launch_mul_mat_q_few_exact(
        const ggml_tensor * src0, ggml_tensor * dst, const int4 * src1_q,
        const int nti, const int nblocks_stream_k, const bool is_stream_k_split, cudaStream_t stream) {
    // Use 1 warp (16 rows/CTA) on narrow-M weights (< 2048 rows) to fill more SMs, 2 warps (32 rows/CTA) otherwise.
    if (src0->ne[1] < 2048) {
        if (is_stream_k_split) {
            launch_mul_mat_q_few_exact_impl<type, 1, true>(src0, dst, src1_q, nti, nblocks_stream_k, stream);
        } else {
            launch_mul_mat_q_few_exact_impl<type, 1, false>(src0, dst, src1_q, nti, nblocks_stream_k, stream);
        }
    } else {
        if (is_stream_k_split) {
            launch_mul_mat_q_few_exact_impl<type, 2, true>(src0, dst, src1_q, nti, nblocks_stream_k, stream);
        } else {
            launch_mul_mat_q_few_exact_impl<type, 2, false>(src0, dst, src1_q, nti, nblocks_stream_k, stream);
        }
    }
}

size_t ggml_cuda_mmq_few_exact_pool_bytes(const ggml_tensor * src0, const ggml_tensor * src1) {
    GGML_UNUSED(src1);
    const int64_t nunits = src0->ne[0] / QK_K;
    return size_t(nunits * MMQ_FEW_Q8_1_INT4) * sizeof(int4);
}

bool ggml_cuda_should_use_mmq_few_exact(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, const int cc) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    GGML_UNUSED_VARS(src0, src1, dst, cc);
    return false;
#endif
    if (!ampere_mma_available(cc) || cc >= GGML_CUDA_CC_RUBIN) {
        return false;
    }
    if (src0 == nullptr || src1 == nullptr) {
        return false;
    }
    if (src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_Q6_K) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32) {
        return false;
    }
    if (dst != nullptr && (dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst))) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
        return false;
    }

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne11 = src1->ne[1];
    if (ne00 != src1->ne[0] || ne00 < 8 * QK_K || (ne00 % QK_K) != 0 || ne00 > INT_MAX) {
        return false;
    }
    if (ne01 <= 0 || ne01 > INT_MAX || ne11 < 2 || ne11 > MMQ_FEW_NCOLS) {
        return false;
    }

    const int64_t nti    = (ne01 + 127) / 128;
    const int64_t nunits = ne00 / QK_K;
    if (nti * nunits >= (1LL << 30)) {
        return false;
    }

    if (!ggml_cuda_is_aligned(src1, 16)) {
        return false;
    }
    if (src0->type == GGML_TYPE_Q4_K) {
        if (!ggml_cuda_is_aligned(src0, 16)) {
            return false;
        }
    } else {
        // Q6_K rows are 2-byte aligned; cp.async loads from preceding 16-byte boundary to trailing 16-byte boundary.
        if (!ggml_cuda_is_aligned(src0, 2) || ((uintptr_t) src0->data % 16) != 0) {
            return false;
        }
        const size_t nbytes = ggml_nbytes(src0);
        if ((nbytes % 16) != 0) {
            if (src0->view_src != nullptr || src0->buffer == nullptr ||
                    ggml_backend_buffer_get_alloc_size(src0->buffer, src0) < GGML_PAD(nbytes, 16)) {
                return false;
            }
        }
    }

    // Verify regular MMQ uses I=128, J=8, K_vram=256 on this GPU so stream-k replay matches exactly.
    const int    id       = ggml_cuda_get_device();
    const size_t smpbo    = ggml_cuda_info().devices[id].smpbo;
    const bool   fallback = (ne01 % 128) != 0;
    const ggml_cuda_mmq_config cfg = ggml_cuda_mmq_get_config(src0->type, 8, fallback, cc, GGML_PREC_Q8);
    if (cfg.type != src0->type || cfg.I != 128 || cfg.J != 8 || cfg.K_vram != QK_K ||
            mmq_get_nbytes_shared(cfg, cc) > smpbo) {
        return false;
    }
    return true;
}

void ggml_cuda_mul_mat_q_few_exact(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const bool self_check) {
    GGML_ASSERT(src0->type == GGML_TYPE_Q4_K || src0->type == GGML_TYPE_Q6_K);
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->ne[0] % QK_K == 0 && src1->ne[1] <= MMQ_FEW_NCOLS);
    GGML_ASSERT(src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1);

    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<float> ref_buf(ctx.pool());
    if (self_check) {
        ggml_tensor ref = *dst;
        ref.data = ref_buf.alloc(ggml_nelements(dst));
        ggml_cuda_mul_mat_q(ctx, src0, src1, nullptr, &ref);
    }

    const int64_t ne00   = src0->ne[0];
    const int64_t ne01   = src0->ne[1];
    const int64_t ne11   = src1->ne[1];
    const int     nunits = (int) (ne00 / QK_K);
    const int     nti    = (int) ((ne01 + 127) / 128);

    const int    id       = ggml_cuda_get_device();
    const int    cc       = ggml_cuda_info().devices[id].cc;
    const int    nsm      = ggml_cuda_info().devices[id].nsm;
    const bool   fallback = (ne01 % 128) != 0;
    const ggml_cuda_mmq_config cfg = ggml_cuda_mmq_get_config(src0->type, 8, fallback, cc, GGML_PREC_Q8);

    // Exact launch_mul_mat_q stream-k block count (mmq.cuh:1603-1629; [PARW] requires ncols_max > 8 so inactive here).
    const int  tiles_nwaves = (nti + nsm - 1) / nsm;
    const int  tiles_eff    = 100 * nti / (nsm * tiles_nwaves);
    const int  nwork        = nti * nunits;
    const bool grid_v063    = !ggml_cuda_mmq_sk_upstream();
    const int  nblocks_sk   = (GGML_CUDA_CC_IS_NVIDIA(cc) && tiles_eff >= 90)
        ? nti
        : (grid_v063 ? nsm : std::min(nsm, nwork));
    const bool is_sk_split  = cfg.stream_k && (nti % nblocks_sk != 0);

    ggml_cuda_pool_alloc<int4> src1_q(ctx.pool(), int64_t(nunits) * MMQ_FEW_Q8_1_INT4);
    const int64_t q_blocks_y = (ne00 + 4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    const ggml_cuda_kernel_launch_params q_params(
        dim3(MMQ_FEW_NCOLS, (unsigned int) q_blocks_y, 1),
        dim3(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1),
        0, stream);
    const int64_t s01 = src1->nb[1] / sizeof(float);
    if (mmq_exact_ds_layout(src0->type) == MMQ_Q8_1_DS_LAYOUT_DS4) {
        ggml_cuda_kernel_launch(quantize_mmq_few_exact_q8_1<MMQ_Q8_1_DS_LAYOUT_DS4>, q_params,
            (const float *) src1->data, src1_q.get(), ne00, s01, (int) ne11);
    } else {
        ggml_cuda_kernel_launch(quantize_mmq_few_exact_q8_1<MMQ_Q8_1_DS_LAYOUT_D4>, q_params,
            (const float *) src1->data, src1_q.get(), ne00, s01, (int) ne11);
    }

    if (src0->type == GGML_TYPE_Q4_K) {
        launch_mul_mat_q_few_exact<GGML_TYPE_Q4_K>(src0, dst, src1_q.get(), nti, nblocks_sk, is_sk_split, stream);
    } else {
        launch_mul_mat_q_few_exact<GGML_TYPE_Q6_K>(src0, dst, src1_q.get(), nti, nblocks_sk, is_sk_split, stream);
    }

    if (self_check) {
        const int64_t n = ggml_nelements(dst);
        mmq_few_exact_check_kernel<<<(unsigned int) ((n + 255) / 256), 256, 0, stream>>>(
            (const float *) dst->data, ref_buf.get(), n);
        CUDA_CHECK(cudaGetLastError());
    }
}

#else // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

bool ggml_cuda_should_use_mmq_few_exact(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc) {
    GGML_UNUSED_VARS(src0, src1, dst, cc);
    return false;
}

size_t ggml_cuda_mmq_few_exact_pool_bytes(const ggml_tensor * src0, const ggml_tensor * src1) {
    GGML_UNUSED_VARS(src0, src1);
    return 0;
}

void ggml_cuda_mul_mat_q_few_exact(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, bool self_check) {
    GGML_UNUSED_VARS(ctx, src0, src1, dst, self_check);
    GGML_ABORT("bit-exact few-row MMQ is not supported on HIP or MUSA");
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
