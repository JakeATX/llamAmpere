// NEONKERN NK_DENSE_TC: Ampere int8 tensor-core kernel (mma.m16n8k32.s32.s8)
// for dense Q8_0 decode/verify projections (T=1..8), single and grouped.
// Weights stay in stock 34-byte block_q8_0 layout; 4 warps cooperate on a
// 3-stage cp.async.cg.L2::128B shared ring (272 B = 8 blocks per chunk) with
// overlapped in-register per-32 f32 -> int8 + fp32-scale activation quantization.
// Opt-in via GGML_CUDA_NK_DENSE_TC=1 (default OFF).

#include "nk-dense-tc.cuh"
#include "ggml-impl.h"

#include <atomic>
#include <climits>
#include <cstdlib>
#include <cstring>

#define NK_DTC_MAX_GROUP   8
#define NK_DTC_STAGES      3
#define NK_DTC_ROW_WORDS   68
#define NK_DTC_STAGE_WORDS (16 * NK_DTC_ROW_WORDS)
#define NK_DTC_STAGE_BYTES (NK_DTC_STAGE_WORDS * 4)
#define NK_DTC_RED_FLOATS  (3 * 32 * 4)

static std::atomic<int64_t> g_nk_dense_tc_fires{0};

bool ggml_cuda_nk_dense_tc_enabled() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_NK_DENSE_TC");
        return e != nullptr && strcmp(e, "1") == 0;
    }();
    return on;
}

int64_t ggml_cuda_nk_dense_tc_count() {
    return g_nk_dense_tc_fires.load(std::memory_order_relaxed);
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

static int nk_dense_tc_min_t() {
    static const int v = [] {
        const char * e = getenv("GGML_CUDA_NK_DENSE_TC_MINT");
        const int x = e ? atoi(e) : 1;
        return x < 1 ? 1 : x > 8 ? 8 : x;
    }();
    return v;
}

static int nk_dense_tc_max_rows() {
    static const int v = [] {
        const char * e = getenv("GGML_CUDA_NK_DENSE_TC_MAX_ROWS");
        const int x = e ? atoi(e) : 0;
        return x > 0 ? x : 0;
    }();
    return v;
}

static bool nk_dense_tc_trace() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_NK_TRACE");
        return e != nullptr && strcmp(e, "1") == 0;
    }();
    return on;
}

struct nk_dense_tc_args {
    const char  * vx[NK_DTC_MAX_GROUP];
    const float * x[NK_DTC_MAX_GROUP];
    float       * dst[NK_DTC_MAX_GROUP];
    int64_t       srow[NK_DTC_MAX_GROUP];
    int64_t       sx[NK_DTC_MAX_GROUP];
    int64_t       sd[NK_DTC_MAX_GROUP];
    uint32_t      blk_end[NK_DTC_MAX_GROUP];
    int           nrows[NK_DTC_MAX_GROUP];
    int           n;
};

static __device__ __forceinline__ void nk_dtc_cp16(uint32_t smem_addr, const void * gmem) {
#if defined(AMPERE_MMA_AVAILABLE) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gmem));
#else
    GGML_UNUSED_VARS(smem_addr, gmem);
    NO_DEVICE_CODE;
#endif
}

static __device__ __forceinline__ void nk_dtc_mma_i8(
        int (&d)[4],
        uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
        uint32_t b0, uint32_t b1) {
#if defined(AMPERE_MMA_AVAILABLE) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    GGML_UNUSED_VARS(d, a0, a1, a2, a3, b0, b1);
    NO_DEVICE_CODE;
#endif
}

// Round 4 floats scaled into [-127, 127] to signed int8 via 1.5*2^23 magic bias and pack.
static __device__ __forceinline__ uint32_t nk_dtc_pack4_i8(const float4 v, const float inv_dx) {
    const uint32_t i0  = __float_as_uint(fmaf(v.x, inv_dx, 12582912.0f));
    const uint32_t i1  = __float_as_uint(fmaf(v.y, inv_dx, 12582912.0f));
    const uint32_t i2  = __float_as_uint(fmaf(v.z, inv_dx, 12582912.0f));
    const uint32_t i3  = __float_as_uint(fmaf(v.w, inv_dx, 12582912.0f));
    const uint32_t p01 = __byte_perm(i0, i1, 0x0040);
    const uint32_t p23 = __byte_perm(i2, i3, 0x0040);
    return __byte_perm(p01, p23, 0x5410);
}

static __device__ __forceinline__ void nk_dtc_quant_x_blk(
        const float4 v0, const float4 v1, const int tig,
        uint32_t & b0, uint32_t & b1, float & dx0, float & dx1) {
    float m = fmaxf(fmaxf(fabsf(v0.x), fabsf(v0.y)), fmaxf(fabsf(v0.z), fabsf(v0.w)));
    m = fmaxf(m, fmaxf(fmaxf(fabsf(v1.x), fabsf(v1.y)), fmaxf(fabsf(v1.z), fabsf(v1.w))));
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
    const float dx     = m * (1.0f / 127.0f);
    const float inv_dx = (m > 1e-30f) ? __fdividef(127.0f, m) : 0.0f;
    b0  = nk_dtc_pack4_i8(v0, inv_dx);
    b1  = nk_dtc_pack4_i8(v1, inv_dx);
    dx0 = __shfl_sync(0xffffffffu, dx, 8 * tig);
    dx1 = __shfl_sync(0xffffffffu, dx, 8 * tig + 4);
}

static __device__ __forceinline__ void nk_dtc_quant_x_pair(
        const float * xp, const bool colok, const int tig,
        uint32_t & b0_0, uint32_t & b1_0, float & dx0_0, float & dx1_0,
        uint32_t & b0_1, uint32_t & b1_1, float & dx0_1, float & dx1_1) {
    const float4 z   = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const float4 v00 = colok ? __ldg((const float4 *) (xp +  0)) : z;
    const float4 v01 = colok ? __ldg((const float4 *) (xp + 16)) : z;
    const float4 v10 = colok ? __ldg((const float4 *) (xp + 32)) : z;
    const float4 v11 = colok ? __ldg((const float4 *) (xp + 48)) : z;
    nk_dtc_quant_x_blk(v00, v01, tig, b0_0, b1_0, dx0_0, dx1_0);
    nk_dtc_quant_x_blk(v10, v11, tig, b0_1, b1_1, dx0_1, dx1_1);
}

static __global__ void __launch_bounds__(128, 6) k_nk_dense_tc(
        const nk_dense_tc_args a, const int nch, const int T) {
#if defined(AMPERE_MMA_AVAILABLE) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    __shared__ __align__(16) uint32_t sw[NK_DTC_STAGES * NK_DTC_STAGE_WORDS + NK_DTC_RED_FLOATS];

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int w    = tid >> 5;
    const int gid  = lane >> 2;
    const int tig  = lane & 3;

    int k = 0;
    while (k < a.n - 1 && blockIdx.x >= a.blk_end[k]) {
        ++k;
    }
    const int tile  = (int) (blockIdx.x - (k == 0 ? 0u : a.blk_end[k - 1]));
    const int nr    = a.nrows[k];
    const int64_t srow = a.srow[k];
    const char * vx = a.vx[k] + (int64_t) tile * 16 * srow;
    const int rmax  = nr - 1 - tile * 16;

    // Precompute the 3 coalesced 16-B copy descriptors per thread (272 = 128 + 128 + 16).
    // Pass 2 covers indices 256..271 = row 15, col 1..16 (15 * 17 = 255).
    const int r0 = tid / 17;
    const int c0 = tid - r0 * 17;
    const int i1 = tid + 128;
    const int r1 = i1 / 17;
    const int c1 = i1 - r1 * 17;

    const char * g0 = vx + (int64_t) min(r0, rmax) * srow + c0 * 16;
    const char * g1 = vx + (int64_t) min(r1, rmax) * srow + c1 * 16;
    const char * g2 = (tid < 16) ? (vx + (int64_t) min(15, rmax) * srow + (tid + 1) * 16) : nullptr;

    const uint32_t sa0_base = (uint32_t) __cvta_generic_to_shared(&sw[r0 * NK_DTC_ROW_WORDS + c0 * 4]);
    const uint32_t sa1_base = (uint32_t) __cvta_generic_to_shared(&sw[r1 * NK_DTC_ROW_WORDS + c1 * 4]);
    const uint32_t sa2_base = (tid < 16)
        ? (uint32_t) __cvta_generic_to_shared(&sw[15 * NK_DTC_ROW_WORDS + (tid + 1) * 4])
        : 0u;

    auto issue_chunk = [&](const uint32_t stage_bytes) {
        nk_dtc_cp16(sa0_base + stage_bytes, g0); g0 += 272;
        nk_dtc_cp16(sa1_base + stage_bytes, g1); g1 += 272;
        if (tid < 16) {
            nk_dtc_cp16(sa2_base + stage_bytes, g2); g2 += 272;
        }
        asm volatile("cp.async.commit_group;\n");
    };

    // Prologue: prefetch up to 2 chunks into slots 0 and 1.
    issue_chunk(0u);
    if (nch > 1) {
        issue_chunk(NK_DTC_STAGE_BYTES);
    }

    const bool colok = gid < T;
    const float * x_ptr = a.x[k] + (int64_t) (colok ? gid : 0) * a.sx[k] + (2 * w) * QK8_0 + 4 * tig;

    // Quantize activation block-pair for ch = 0 while prologue cp.async is in flight.
    uint32_t xb0_0, xb1_0, xb0_1, xb1_1;
    float    dx0_0, dx1_0, dx0_1, dx1_1;
    nk_dtc_quant_x_pair(x_ptr, colok, tig, xb0_0, xb1_0, dx0_0, dx1_0, xb0_1, xb1_1, dx0_1, dx1_1);
    x_ptr += 8 * QK8_0;

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    uint32_t cur_w = 0u;
    uint32_t nxt_b = 2u * NK_DTC_STAGE_BYTES;

    for (int ch = 0; ch < nch; ++ch) {
        if (ch + 1 < nch) {
            asm volatile("cp.async.wait_group 1;\n");
        } else {
            asm volatile("cp.async.wait_group 0;\n");
        }
        __syncthreads();

        // All warps finished reading slot (ch - 1) % 3 == slot (ch + 2) % 3 before __syncthreads.
        if (ch + 2 < nch) {
            issue_chunk(nxt_b);
        }

        const uint32_t * W0 = sw + cur_w + gid * NK_DTC_ROW_WORDS + 17 * w;
        const uint32_t * W1 = W0 + 8 * NK_DTC_ROW_WORDS;

        // Even block (2-byte offset inside 17-word block pair) + scales from u*_a (tig=0) and u*_d (tig=3).
        const uint32_t u0_a = W0[tig],     u0_b = W0[tig + 1];
        const uint32_t u0_c = W0[tig + 4], u0_d = W0[tig + 5];
        const uint32_t u1_a = W1[tig],     u1_b = W1[tig + 1];
        const uint32_t u1_c = W1[tig + 4], u1_d = W1[tig + 5];

        const uint32_t a0_e0 = __byte_perm(u0_a, u0_b, 0x5432);
        const uint32_t a2_e0 = __byte_perm(u0_c, u0_d, 0x5432);
        const uint32_t a1_e0 = __byte_perm(u1_a, u1_b, 0x5432);
        const uint32_t a3_e0 = __byte_perm(u1_c, u1_d, 0x5432);

        // Odd block (4-byte aligned in words 9..16).
        const uint32_t a0_e1 = W0[9 + tig],  a2_e1 = W0[13 + tig];
        const uint32_t a1_e1 = W1[9 + tig],  a3_e1 = W1[13 + tig];

        const uint32_t h0_e0 = __shfl_sync(0xffffffffu, u0_a, lane & ~3);
        const uint32_t h0_e1 = __shfl_sync(0xffffffffu, u0_d, (lane & ~3) + 3);
        const uint32_t h1_e0 = __shfl_sync(0xffffffffu, u1_a, lane & ~3);
        const uint32_t h1_e1 = __shfl_sync(0xffffffffu, u1_d, (lane & ~3) + 3);

        const float dw0_0 = __half2float(__ushort_as_half((uint16_t) (h0_e0 & 0xFFFFu)));
        const float dw1_0 = __half2float(__ushort_as_half((uint16_t) (h1_e0 & 0xFFFFu)));
        const float dw0_1 = __half2float(__ushort_as_half((uint16_t) (h0_e1 >> 16)));
        const float dw1_1 = __half2float(__ushort_as_half((uint16_t) (h1_e1 >> 16)));

        int d0[4] = { 0, 0, 0, 0 };
        nk_dtc_mma_i8(d0, a0_e0, a1_e0, a2_e0, a3_e0, xb0_0, xb1_0);

        int d1[4] = { 0, 0, 0, 0 };
        nk_dtc_mma_i8(d1, a0_e1, a1_e1, a2_e1, a3_e1, xb0_1, xb1_1);

        acc[0] = fmaf(__int2float_rn(d0[0]), dw0_0 * dx0_0, acc[0]);
        acc[1] = fmaf(__int2float_rn(d0[1]), dw0_0 * dx1_0, acc[1]);
        acc[2] = fmaf(__int2float_rn(d0[2]), dw1_0 * dx0_0, acc[2]);
        acc[3] = fmaf(__int2float_rn(d0[3]), dw1_0 * dx1_0, acc[3]);

        acc[0] = fmaf(__int2float_rn(d1[0]), dw0_1 * dx0_1, acc[0]);
        acc[1] = fmaf(__int2float_rn(d1[1]), dw0_1 * dx1_1, acc[1]);
        acc[2] = fmaf(__int2float_rn(d1[2]), dw1_1 * dx0_1, acc[2]);
        acc[3] = fmaf(__int2float_rn(d1[3]), dw1_1 * dx1_1, acc[3]);

        if (ch + 1 < nch) {
            nk_dtc_quant_x_pair(x_ptr, colok, tig, xb0_0, xb1_0, dx0_0, dx1_0, xb0_1, xb1_1, dx0_1, dx1_1);
            x_ptr += 8 * QK8_0;
        }

        cur_w = (cur_w == 2u * NK_DTC_STAGE_WORDS) ? 0u : (cur_w + NK_DTC_STAGE_WORDS);
        nxt_b = (nxt_b == 2u * NK_DTC_STAGE_BYTES) ? 0u : (nxt_b + NK_DTC_STAGE_BYTES);
    }

    // Deterministic 4-warp epilogue reduction into separate SMEM buffer (1 barrier total).
    float4 * red = (float4 *) &sw[NK_DTC_STAGES * NK_DTC_STAGE_WORDS];
    if (w > 0) {
        red[(w - 1) * 32 + lane] = make_float4(acc[0], acc[1], acc[2], acc[3]);
    }
    __syncthreads();
    if (w > 0) {
        return;
    }
#pragma unroll
    for (int v = 0; v < 3; ++v) {
        const float4 p = red[v * 32 + lane];
        acc[0] += p.x;
        acc[1] += p.y;
        acc[2] += p.z;
        acc[3] += p.w;
    }

    float * dst = a.dst[k];
    const int64_t sd = a.sd[k];
    const int r0_out = tile * 16 + gid;
    const int r1_out = r0_out + 8;
    const int c0_out = 2 * tig;
    const int c1_out = c0_out + 1;
    if (r0_out < nr) {
        if (c0_out < T) { dst[(int64_t) c0_out * sd + r0_out] = acc[0]; }
        if (c1_out < T) { dst[(int64_t) c1_out * sd + r0_out] = acc[1]; }
    }
    if (r1_out < nr) {
        if (c0_out < T) { dst[(int64_t) c0_out * sd + r1_out] = acc[2]; }
        if (c1_out < T) { dst[(int64_t) c1_out * sd + r1_out] = acc[3]; }
    }
#else
    GGML_UNUSED_VARS(a, nch, T);
    NO_DEVICE_CODE;
#endif
}

static bool nk_dense_tc_device_ok(const ggml_backend_cuda_context & ctx) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    return GGML_CUDA_CC_IS_NVIDIA(cc) && ampere_mma_available(cc) &&
        ggml_cuda_info().devices[ctx.device].warp_size == 32;
}

static bool nk_dense_tc_member_ok(
        const ggml_tensor * w, const ggml_tensor * x, const ggml_tensor * d,
        const int64_t ne10, const int64_t T, const int max_rows) {
    if (!w || !x || !d || w->type != GGML_TYPE_Q8_0 || !w->buffer ||
            ggml_backend_buffer_get_usage(w->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        return false;
    }
    if (w->nb[0] != ggml_type_size(GGML_TYPE_Q8_0) || w->ne[0] != ne10 ||
            w->ne[1] <= 0 || w->ne[1] > INT_MAX || (max_rows > 0 && w->ne[1] > max_rows) ||
            w->ne[2] != 1 || w->ne[3] != 1 ||
            w->nb[1] != (size_t) (ne10 / QK8_0) * sizeof(block_q8_0) ||
            (uintptr_t) w->data % 16 != 0) {
        return false;
    }
    if (x->type != GGML_TYPE_F32 || x->ne[0] != ne10 || x->ne[1] != T ||
            x->ne[2] != 1 || x->ne[3] != 1 ||
            x->nb[0] != sizeof(float) || x->nb[1] % 16 != 0 || (uintptr_t) x->data % 16 != 0) {
        return false;
    }
    if (d->type != GGML_TYPE_F32 || d->ne[0] != w->ne[1] || d->ne[1] != T ||
            d->ne[2] != 1 || d->ne[3] != 1 ||
            d->nb[0] != sizeof(float) || d->nb[1] % sizeof(float) != 0) {
        return false;
    }
    return true;
}

static void nk_dense_tc_maybe_set_carveout(const int dev, cudaStream_t stream) {
    static std::atomic<uint32_t> done_mask{0};
    if (dev < 0 || dev >= 32) {
        return;
    }
    const uint32_t bit = 1u << (unsigned) dev;
    if ((done_mask.load(std::memory_order_acquire) & bit) != 0u) {
        return;
    }
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &cap) == cudaSuccess && cap != cudaStreamCaptureStatusNone) {
        return;
    }
    if (cudaFuncSetAttribute(k_nk_dense_tc, cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxShared) == cudaSuccess) {
        done_mask.fetch_or(bit, std::memory_order_release);
    }
}

static bool nk_dense_tc_launch(
        ggml_backend_cuda_context & ctx, nk_dense_tc_args & a, const int nch, const int T) {
    uint64_t blk64 = 0;
    for (int k = 0; k < a.n; ++k) {
        blk64 += (uint64_t) ((a.nrows[k] + 15) / 16);
        if (blk64 > (uint64_t) INT_MAX) {
            return false;
        }
        a.blk_end[k] = (uint32_t) blk64;
    }
    if (blk64 == 0) {
        return false;
    }

    static std::atomic<bool> logged{false};
    if (!logged.exchange(true, std::memory_order_relaxed)) {
        GGML_LOG_WARN("[NK_DENSE_TC] active (int8 mma.m16n8k32, 3-stage cp.async.cg.L2::128B, min_t=%d, max_rows=%d)\n",
            nk_dense_tc_min_t(), nk_dense_tc_max_rows());
    }

    cudaStream_t stream = ctx.stream();
    nk_dense_tc_maybe_set_carveout(ctx.device, stream);
    k_nk_dense_tc<<<(uint32_t) blk64, 128, 0, stream>>>(a, nch, T);
    CUDA_CHECK(cudaGetLastError());

    const int64_t fire = g_nk_dense_tc_fires.fetch_add(1, std::memory_order_relaxed) + 1;
    if (nk_dense_tc_trace() && fire <= 8) {
        int total_rows = 0;
        for (int k = 0; k < a.n; ++k) {
            total_rows += a.nrows[k];
        }
        GGML_LOG_WARN("[NK_DENSE_TC] trace #%lld: n=%d rows=%d K=%d T=%d grid=%u\n",
            (long long) fire, a.n, total_rows, nch * 8 * QK8_0, T, (unsigned) blk64);
    }
    return true;
}

bool ggml_cuda_nk_dense_tc_mul_mat(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor       * dst,
        bool                force_enable) {
    if ((!force_enable && !ggml_cuda_nk_dense_tc_enabled()) || !nk_dense_tc_device_ok(ctx)) {
        return false;
    }
    if (!src0 || !src1 || !dst) {
        return false;
    }
    const int64_t ne10 = src0->ne[0];
    const int64_t T    = src1->ne[1];
    const int min_t    = force_enable ? 1 : nk_dense_tc_min_t();
    const int max_rows = force_enable ? 0 : nk_dense_tc_max_rows();
    if (T < min_t || T > 8 || ne10 < 8 * QK8_0 || ne10 % (8 * QK8_0) != 0) {
        return false;
    }
    if (!nk_dense_tc_member_ok(src0, src1, dst, ne10, T, max_rows)) {
        return false;
    }
    nk_dense_tc_args a{};
    a.n        = 1;
    a.vx[0]    = (const char *) src0->data;
    a.x[0]     = (const float *) src1->data;
    a.dst[0]   = (float *) dst->data;
    a.srow[0]  = (int64_t) src0->nb[1];
    a.sx[0]    = (int64_t) (src1->nb[1] / sizeof(float));
    a.sd[0]    = (int64_t) (dst->nb[1] / sizeof(float));
    a.nrows[0] = (int) src0->ne[1];
    return nk_dense_tc_launch(ctx, a, (int) (ne10 / (8 * QK8_0)), (int) T);
}

bool ggml_cuda_nk_dense_tc_group(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * const       * dsts,
        int                         n,
        bool                        check_only,
        bool                        force_enable) {
    if ((!force_enable && !ggml_cuda_nk_dense_tc_enabled()) || n < 1 || n > NK_DTC_MAX_GROUP ||
            !dsts || !dsts[0] || !dsts[0]->src[0] || !dsts[0]->src[1] ||
            !nk_dense_tc_device_ok(ctx)) {
        return false;
    }
    const int64_t ne10 = dsts[0]->src[0]->ne[0];
    const int64_t T    = dsts[0]->src[1]->ne[1];
    const int min_t    = force_enable ? 1 : nk_dense_tc_min_t();
    const int max_rows = force_enable ? 0 : nk_dense_tc_max_rows();
    if (T < min_t || T > 8 || ne10 < 8 * QK8_0 || ne10 % (8 * QK8_0) != 0) {
        return false;
    }
    for (int k = 0; k < n; ++k) {
        if (!dsts[k] || !nk_dense_tc_member_ok(dsts[k]->src[0], dsts[k]->src[1], dsts[k], ne10, T, max_rows)) {
            return false;
        }
    }
    if (check_only) {
        return true;
    }
    nk_dense_tc_args a{};
    a.n = n;
    for (int k = 0; k < n; ++k) {
        a.vx[k]    = (const char *) dsts[k]->src[0]->data;
        a.x[k]     = (const float *) dsts[k]->src[1]->data;
        a.dst[k]   = (float *) dsts[k]->data;
        a.srow[k]  = (int64_t) dsts[k]->src[0]->nb[1];
        a.sx[k]    = (int64_t) (dsts[k]->src[1]->nb[1] / sizeof(float));
        a.sd[k]    = (int64_t) (dsts[k]->nb[1] / sizeof(float));
        a.nrows[k] = (int) dsts[k]->src[0]->ne[1];
    }
    return nk_dense_tc_launch(ctx, a, (int) (ne10 / (8 * QK8_0)), (int) T);
}

#else

bool ggml_cuda_nk_dense_tc_mul_mat(
        ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *, ggml_tensor *, bool) {
    return false;
}

bool ggml_cuda_nk_dense_tc_group(
        ggml_backend_cuda_context &, ggml_tensor * const *, int, bool, bool) {
    return false;
}

#endif
