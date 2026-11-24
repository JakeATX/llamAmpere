// AIT5_A16: SM 8.0/8.6 FP32-activation decode kernels for GGML_TYPE_AIT5_U (dense MUL_MAT & MoE MUL_MAT_ID).
// Standalone CUDA translation unit for fast incremental builds and clean SASS auditing.
//
// SM 8.6 (GA102) microarchitecture design:
//   - Each AIT5_U 16x128 unit is 1,312 B = 32 B fp16 scales + 1,280 B (16 tiles x 80 B): 100% 16B-aligned (82 x 16B).
//   - 2-stage per-warp cp.async.cg.16 ring streams the entire 1,312 B unit contiguously (0 separate scale cp.async4).
//   - Lane layout: r = lane & 15 (row 0..15), h = lane >> 4 (cols 0..3 or 4..7 of each 16x8 tile):
//       * All 16 lanes of each half-warp broadcast the SAME float4 activation from L1 (__ldg 128-bit broadcast).
//       * 32 lanes read at most 20 SMEM words of a tile -> ZERO SMEM bank conflicts.
//       * Fused 1MAD -> IEEE-754 FP32 decode (IMAD + IDP4A + PRMT + FADD) executes on INT32 while FFMA dot products
//         dual-issue on the 128-wide FP32 pipe (~152 INT32 SM-clocks/unit < 195.4 SM-clock 936 GB/s HBM floor).
//   - Zero static SMEM for reduction (aliases dead cp.async ring after cp_async_wait<0> + __syncthreads).
//   - Deterministic fixed-order FP32 accumulation (per-chunk csum scaled by d_row, single shfl_xor 16 + warp 0..W-1 sum).

#include "ait5a16.cuh"
#include "ai4-kern.cuh"
#include "ggml-backend-impl.h"

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdlib>

namespace {
std::atomic<uint64_t> g_ait5a16_fires[2] = { {0}, {0} };
}

extern "C" GGML_BACKEND_API uint64_t ggml_cuda_ait5a16_fire_count(int which) {
    return g_ait5a16_fires[which & 1].load(std::memory_order_relaxed);
}

extern "C" GGML_BACKEND_API void ggml_cuda_ait5a16_reset_fire_count(void) {
    g_ait5a16_fires[0].store(0, std::memory_order_relaxed);
    g_ait5a16_fires[1].store(0, std::memory_order_relaxed);
}

extern "C" GGML_BACKEND_API void * ggml_cuda_ait5_backend_stream(ggml_backend_t backend) {
    if (!backend || !backend->context) {
        return nullptr;
    }
    auto * ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    return (void *) ctx->stream();
}

bool ggml_cuda_ait5a16_enabled() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_AIT5_A16");
        return e != nullptr && atoi(e) != 0; // default OFF (opt-in)
    }();
    return on;
}

bool ggml_cuda_ait5a16_id_enabled() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_AIT5_A16_ID");
        return e != nullptr && atoi(e) != 0; // default OFF (opt-in)
    }();
    return on;
}

int ggml_cuda_ait5a16_max_t() {
    static const int m = [] {
        const char * e = getenv("GGML_CUDA_AIT5_A16_MAXT");
        const int v = e ? atoi(e) : 8;
        return v < 0 ? 0 : (v > 8 ? 8 : v);
    }();
    return m;
}

#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)

bool ggml_cuda_ait5a16_mul_mat(ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *, ggml_tensor *) {
    return false;
}

bool ggml_cuda_ait5a16_mul_mat_id(ggml_backend_cuda_context &, const ggml_tensor *, float *) {
    return false;
}

#else

namespace ait5a16 {

constexpr int UB  = 1312;     // AIT5_U unit bytes: 32 B d[16] + 16 * 80 B tiles
constexpr int U16 = UB / 16;  // 82 x 16B vectors
constexpr uint32_t MULT = ai4k::lay_mul<ai4k::LAY_AIT5_U>(); // 0xAA73876Bu

static_assert(ai4k::u_bytes<ai4k::LAY_AIT5_U>() == UB, "AIT5_U unit size");
static_assert(UB % 16 == 0 && ai4k::U_SC == 32, "AIT5_U 16B alignment");

// -------------------------------------------------------------------------------------------------
// 1. Exact FP32 Activation Rotation Kernels (H256 for K % 256 == 0, H128 for K % 256 != 0)
//    Exact same sign-flip (rq4::d_sign_bits == ai4_sign_bits) + FWHT + 1/sqrt(n) as ai4_quantize_x,
//    storing full FP32 rotated activations (ZERO int8 quantization loss!).
// -------------------------------------------------------------------------------------------------

static __global__ void __launch_bounds__(128) k_ait5a16_rot256(
        const float * __restrict__ x, float * __restrict__ x_rot, int K, int N) {
    const int lane = threadIdx.x & 31;
    const int hb   = (int) blockIdx.x * 4 + (threadIdx.x >> 5);
    const int tok  = (int) blockIdx.y;
    if (hb * 256 >= K || tok >= N) {
        return;
    }
    const int64_t off = (int64_t) tok * K + (int64_t) hb * 256 + lane * 8;
    const float4 u0 = *(const float4 *) (x + off);
    const float4 u1 = *(const float4 *) (x + off + 4);
    float v[8] = { u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w };

    const uint32_t sw = rq4::d_sign_bits[lane >> 2] >> ((lane & 3) * 8);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        if ((sw >> e) & 1u) {
            v[e] = -v[e];
        }
    }
#pragma unroll
    for (int h = 1; h < 8; h <<= 1) {
#pragma unroll
        for (int i = 0; i < 8; i += 2 * h) {
#pragma unroll
            for (int j = i; j < i + h; ++j) {
                const float a = v[j], b = v[j + h];
                v[j] = a + b;
                v[j + h] = a - b;
            }
        }
    }
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const float o = __shfl_xor_sync(0xffffffffu, v[e], m);
            v[e] = (lane & m) ? (o - v[e]) : (v[e] + o);
        }
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        v[e] *= (1.0f / 16.0f);
    }
    *(float4 *) (x_rot + off)     = make_float4(v[0], v[1], v[2], v[3]);
    *(float4 *) (x_rot + off + 4) = make_float4(v[4], v[5], v[6], v[7]);
}

static __global__ void __launch_bounds__(128) k_ait5a16_rot128(
        const float * __restrict__ x, float * __restrict__ x_rot, int K, int N) {
    const int lane = threadIdx.x & 31;
    const int l    = lane & 15;
    const int cb   = ((int) blockIdx.x * 4 + (threadIdx.x >> 5)) * 2 + (lane >> 4);
    const int tok  = (int) blockIdx.y;
    const bool live = (cb * 128 < K) && (tok < N); // whole warp runs shuffles; dead half-warps skip I/O

    float v[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    const int64_t off = (int64_t) tok * K + (int64_t) cb * 128 + l * 8;
    if (live) {
        const float4 u0 = *(const float4 *) (x + off);
        const float4 u1 = *(const float4 *) (x + off + 4);
        v[0] = u0.x; v[1] = u0.y; v[2] = u0.z; v[3] = u0.w;
        v[4] = u1.x; v[5] = u1.y; v[6] = u1.z; v[7] = u1.w;
    }
    const uint32_t sw = rq4::d_sign_bits[l >> 2] >> ((l & 3) * 8);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        if ((sw >> e) & 1u) {
            v[e] = -v[e];
        }
    }
#pragma unroll
    for (int h = 1; h < 8; h <<= 1) {
#pragma unroll
        for (int i = 0; i < 8; i += 2 * h) {
#pragma unroll
            for (int j = i; j < i + h; ++j) {
                const float a = v[j], b = v[j + h];
                v[j] = a + b;
                v[j + h] = a - b;
            }
        }
    }
#pragma unroll
    for (int m = 1; m < 16; m <<= 1) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const float o = __shfl_xor_sync(0xffffffffu, v[e], m);
            v[e] = (l & m) ? (o - v[e]) : (v[e] + o);
        }
    }
    if (live) {
        constexpr float norm128 = 0.08838834764831845f; // 1/sqrt(128)
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            v[e] *= norm128;
        }
        *(float4 *) (x_rot + off)     = make_float4(v[0], v[1], v[2], v[3]);
        *(float4 *) (x_rot + off + 4) = make_float4(v[4], v[5], v[6], v[7]);
    }
}

static void launch_rotate_fp32(const float * x, float * x_rot, int K, int64_t N_rows, cudaStream_t stream) {
    const bool h256 = (K % 256 == 0);
    for (int64_t r0 = 0; r0 < N_rows; r0 += 65535) {
        const int nr = (int) std::min<int64_t>(65535, N_rows - r0);
        if (h256) {
            const dim3 grid((K / 256 + 3) / 4, nr);
            k_ait5a16_rot256<<<grid, 128, 0, stream>>>(x + r0 * K, x_rot + r0 * K, K, nr);
        } else {
            const dim3 grid((K / 128 + 7) / 8, nr);
            k_ait5a16_rot128<<<grid, 128, 0, stream>>>(x + r0 * K, x_rot + r0 * K, K, nr);
        }
    }
}

// -------------------------------------------------------------------------------------------------
// 2. Fused 1MAD Trellis State -> Exact IEEE-754 FP32 Weight Level ([-128.0f, +127.0f])
//    Proof for all s in [0, 65535]:
//      bytesum = sum of 4 unsigned bytes of (s * MULT) in [0, 1020]
//      r = __dp4a(s * MULT, 0x40404040u, 128u) = 64 * (bytesum + 2) in [128, 65408]
//      byte 1 of r = u = (bytesum + 2) >> 2 in [0, 255]
//      __byte_perm(r, 0x4B000000u, 0x7541) = 0x4B000000u | u = IEEE-754 float(8388608 + u)
//      __fsub_rn(float(8388608 + u), 8388736.0f) = exact float(u - 128) = float(aik_level_c(s, MULT))!
//    Uses __fsub_rn to guarantee zero contraction under -use_fast_math (same 1-cycle FADD instruction).
// -------------------------------------------------------------------------------------------------

static __device__ __forceinline__ float lv1_f32(const uint32_t s, const uint32_t c128, const uint32_t c4b) {
    const uint32_t r = __dp4a(s * MULT, 0x40404040u, c128);
    // Keeping c4b (0x4B000000u) in a loop-invariant register lets SM 8.6 PRMT encode 0x7541 as its immediate
    // selector (`PRMT Rd, Ra, 0x7541, Rc`), eliminating all UMOV/MOV UR5 rematerializations!
    return __fsub_rn(__uint_as_float(__byte_perm(r, c4b, 0x7541u)), 8388736.0f);
}

static __device__ __forceinline__ void load_unit_async(uint8_t * dst_stage, const uint8_t * src_unit, int lane) {
#pragma unroll
    for (int r = 0; r < 3; ++r) {
        const int p = lane + 32 * r;
        if (r < 2 || p < U16) {
            rq4::cp_async16(dst_stage + p * 16, src_unit + p * 16, true);
        }
    }
}

template <int WARPS>
constexpr __host__ __device__ int dense_smem_bytes() {
    constexpr int ring_b = WARPS * 2 * UB;
    constexpr int red_b  = (WARPS - 1) * 8 * 16 * (int) sizeof(float);
    return ring_b > red_b ? ring_b : red_b;
}

// -------------------------------------------------------------------------------------------------
// 3. Dense AIT5_A16 Matvec Kernel (T = 1..8, WARPS = 3 or 4)
//    Reads each AIT5_U weight slab ONCE for all T columns (zero verify-width weight re-reads!).
// -------------------------------------------------------------------------------------------------

template <int T, int WARPS, int MINB>
static __global__ void __launch_bounds__(WARPS * 32, MINB) k_ait5a16_dense(
        const uint8_t * __restrict__ W,
        const float   * __restrict__ Xrot,
        float         * __restrict__ Y,
        const int64_t ldy, const int K) {
    static_assert(T >= 1 && T <= 8, "AIT5_A16 batch width");
    static_assert(WARPS == 3 || WARPS == 4, "AIT5_A16 warps");

    extern __shared__ __align__(16) uint8_t smem[];
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int r    = lane & 15;
    const int h    = lane >> 4;
    const int slab = (int) blockIdx.x;

    const int nk  = K >> 7;
    const int nmy = (nk - warp + WARPS - 1) / WARPS;

    const uint8_t * wp_cur = W + ((int64_t) slab * nk + warp) * UB;
    const float   * xr_cur = Xrot + (int64_t) warp * 128 + h * 4;
    uint8_t       * ring   = smem + warp * 2 * UB;

    const ai4k::k5win win = ai4k::k5_win(r, h);
    uint32_t p0_s0_b = (uint32_t) __cvta_generic_to_shared(ring + 32 + win.w0 * 4);
    uint32_t p1_s0_b = (uint32_t) __cvta_generic_to_shared(ring + 32 + win.w1 * 4);
    uint32_t win_sh = (uint32_t) win.sh;
    // K % 128 == 0 is enforced by host dispatch: (K & 0x7Fu) == 0 holds these constants in 2 registers outside the loop
    const uint32_t k_zero = (uint32_t) K & 0x7Fu;
    uint32_t c128   = 128u | k_zero;
    uint32_t c4b    = 0x4B000000u | k_zero;
    asm volatile("" : "+r"(p0_s0_b), "+r"(p1_s0_b), "+r"(win_sh), "+r"(c128), "+r"(c4b));

    if (nmy > 0) {
        load_unit_async(ring, wp_cur, lane);
    }
    rq4::cp_async_commit();

    float acc[T];
#pragma unroll
    for (int t = 0; t < T; ++t) {
        acc[t] = 0.0f;
    }

    auto process_chunk = [&](int i, int slot) {
        rq4::cp_async_wait<0>();
        __syncwarp();
        if (i + 1 < nmy) {
            load_unit_async(ring + (slot ^ 1) * UB, wp_cur + (int64_t) WARPS * UB, lane);
        }
        rq4::cp_async_commit();

        const uint8_t * st      = ring + slot * UB;
        uint32_t        p0_addr = p0_s0_b + (uint32_t) (slot * UB);
        uint32_t        p1_addr = p1_s0_b + (uint32_t) (slot * UB);
        const float   * x_chunk = xr_cur;
        const float     dr      = __half2float(((const half *) st)[r]);

        wp_cur += (int64_t) WARPS * UB;
        xr_cur += (int64_t) WARPS * 128;

        // Pre-multiplying dr * f0..f3 is 0-ULP exact in FP32 (11-bit fp16 * 7-bit int <= 18-bit < 24-bit mantissa),
        // eliminates csum[T] (saves T regs -> 0 spills for all T=1..8), and matches aik_dequantize_slab_rotated_c.
        constexpr int CT_UNROLL = (T <= 4) ? 2 : 1;
#pragma unroll CT_UNROLL
        for (int ct = 0; ct < 16; ++ct) {
            uint32_t w0_val, w1_val;
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(w0_val) : "r"(p0_addr));
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(w1_val) : "r"(p1_addr));
            p0_addr += 80u;
            p1_addr += 80u;
            const uint32_t f = __funnelshift_l(w1_val, w0_val, (int) win_sh);
            const float w0 = lv1_f32((f >> 15) & 0xFFFFu, c128, c4b) * dr;
            const float w1 = lv1_f32((f >> 10) & 0xFFFFu, c128, c4b) * dr;
            const float w2 = lv1_f32((f >>  5) & 0xFFFFu, c128, c4b) * dr;
            const float w3 = lv1_f32( f        & 0xFFFFu, c128, c4b) * dr;
#pragma unroll
            for (int t = 0; t < T; ++t) {
                const float4 xv = __ldg((const float4 *) (x_chunk + (int64_t) t * K + ct * 8));
                acc[t] = fmaf(w0, xv.x, acc[t]);
                acc[t] = fmaf(w1, xv.y, acc[t]);
                acc[t] = fmaf(w2, xv.z, acc[t]);
                acc[t] = fmaf(w3, xv.w, acc[t]);
            }
        }
    };

#pragma unroll 1
    for (int i0 = 0; i0 < nmy; i0 += 2) {
        if (i0 < nmy)     process_chunk(i0,     0);
        if (i0 + 1 < nmy) process_chunk(i0 + 1, 1);
    }
    rq4::cp_async_wait<0>();

    // Combine h = 0 (lanes 0..15) and h = 1 (lanes 16..31)
#pragma unroll
    for (int t = 0; t < T; ++t) {
        acc[t] += __shfl_xor_sync(0xffffffffu, acc[t], 16);
    }

    __syncthreads();
    float * red = (float *) smem;
    if (warp > 0 && lane < 16) {
#pragma unroll
        for (int t = 0; t < T; ++t) {
            red[((warp - 1) * T + t) * 16 + lane] = acc[t];
        }
    }
    __syncthreads();
    if (warp == 0 && lane < 16) {
        const int row = slab * 16 + lane;
#pragma unroll
        for (int t = 0; t < T; ++t) {
            float s = acc[t];
#pragma unroll
            for (int w = 0; w < WARPS - 1; ++w) {
                s += red[(w * T + t) * 16 + lane];
            }
            Y[(int64_t) t * ldy + row] = s;
        }
    }
}

// -------------------------------------------------------------------------------------------------
// 4. MoE MUL_MAT_ID AIT5_A16 Kernel (T = 1..8, n_used * T <= 64, ZERO ids_helper launch!)
//    Inline CTA-uniform 64-route ballot dedup (handles duplicates within a token & strided ids).
//    Compile-time MAX_C specializations (1, 8) keep all accumulators in registers (0 spills!).
// -------------------------------------------------------------------------------------------------

template <int MAX_C, int WARPS>
static __device__ __forceinline__ void ait5a16_id_run(
        uint8_t * smem,
        const uint8_t * __restrict__ W_exp,
        const float   * __restrict__ Xrot,
        float         * __restrict__ Y,
        const int64_t ldy, const int K, const int slab,
        const int n_cols, const int * sxrow, const int * sycol) {
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int r    = lane & 15;
    const int h    = lane >> 4;

    const int nk  = K >> 7;
    const int nmy = (nk - warp + WARPS - 1) / WARPS;

    const uint8_t * wp_cur = W_exp + ((int64_t) slab * nk + warp) * UB;
    const float   * xr_off = Xrot + (int64_t) warp * 128 + h * 4;
    uint8_t       * ring   = smem + warp * 2 * UB;

    const ai4k::k5win win = ai4k::k5_win(r, h);
    uint32_t p0_s0_b = (uint32_t) __cvta_generic_to_shared(ring + 32 + win.w0 * 4);
    uint32_t p1_s0_b = (uint32_t) __cvta_generic_to_shared(ring + 32 + win.w1 * 4);
    uint32_t win_sh = (uint32_t) win.sh;
    const uint32_t k_zero = (uint32_t) K & 0x7Fu;
    uint32_t c128   = 128u | k_zero;
    uint32_t c4b    = 0x4B000000u | k_zero;
    asm volatile("" : "+r"(p0_s0_b), "+r"(p1_s0_b), "+r"(win_sh), "+r"(c128), "+r"(c4b));

    if (nmy > 0) {
        load_unit_async(ring, wp_cur, lane);
    }
    rq4::cp_async_commit();

    float acc[MAX_C];
#pragma unroll
    for (int c = 0; c < MAX_C; ++c) {
        acc[c] = 0.0f;
    }

    auto process_chunk = [&](int i, int slot) {
        rq4::cp_async_wait<0>();
        __syncwarp();
        if (i + 1 < nmy) {
            load_unit_async(ring + (slot ^ 1) * UB, wp_cur + (int64_t) WARPS * UB, lane);
        }
        rq4::cp_async_commit();

        const uint8_t * st      = ring + slot * UB;
        uint32_t        p0_addr = p0_s0_b + (uint32_t) (slot * UB);
        uint32_t        p1_addr = p1_s0_b + (uint32_t) (slot * UB);
        const float   * x_base  = xr_off;
        const float     dr      = __half2float(((const half *) st)[r]);

        wp_cur += (int64_t) WARPS * UB;
        xr_off += (int64_t) WARPS * 128;

        constexpr int CT_UNROLL = (MAX_C == 1) ? 2 : 1;
#pragma unroll CT_UNROLL
        for (int ct = 0; ct < 16; ++ct) {
            uint32_t w0_val, w1_val;
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(w0_val) : "r"(p0_addr));
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(w1_val) : "r"(p1_addr));
            p0_addr += 80u;
            p1_addr += 80u;
            const uint32_t f = __funnelshift_l(w1_val, w0_val, (int) win_sh);
            const float w0 = lv1_f32((f >> 15) & 0xFFFFu, c128, c4b) * dr;
            const float w1 = lv1_f32((f >> 10) & 0xFFFFu, c128, c4b) * dr;
            const float w2 = lv1_f32((f >>  5) & 0xFFFFu, c128, c4b) * dr;
            const float w3 = lv1_f32( f        & 0xFFFFu, c128, c4b) * dr;
#pragma unroll
            for (int c = 0; c < MAX_C; ++c) {
                if (MAX_C == 1 || c < n_cols) {
                    const float4 xv = __ldg((const float4 *) (x_base + (int64_t) sxrow[c] * K + ct * 8));
                    acc[c] = fmaf(w0, xv.x, acc[c]);
                    acc[c] = fmaf(w1, xv.y, acc[c]);
                    acc[c] = fmaf(w2, xv.z, acc[c]);
                    acc[c] = fmaf(w3, xv.w, acc[c]);
                }
            }
        }
    };

#pragma unroll 1
    for (int i0 = 0; i0 < nmy; i0 += 2) {
        if (i0 < nmy)     process_chunk(i0,     0);
        if (i0 + 1 < nmy) process_chunk(i0 + 1, 1);
    }
    rq4::cp_async_wait<0>();

#pragma unroll
    for (int c = 0; c < MAX_C; ++c) {
        if (MAX_C == 1 || c < n_cols) {
            acc[c] += __shfl_xor_sync(0xffffffffu, acc[c], 16);
        }
    }

    __syncthreads();
    float * red = (float *) smem;
    if (warp > 0 && lane < 16) {
#pragma unroll
        for (int c = 0; c < MAX_C; ++c) {
            if (MAX_C == 1 || c < n_cols) {
                red[((warp - 1) * MAX_C + c) * 16 + lane] = acc[c];
            }
        }
    }
    __syncthreads();
    if (warp == 0 && lane < 16) {
        const int row = slab * 16 + lane;
#pragma unroll
        for (int c = 0; c < MAX_C; ++c) {
            if (MAX_C == 1 || c < n_cols) {
                float s = acc[c];
#pragma unroll
                for (int w = 0; w < WARPS - 1; ++w) {
                    s += red[(w * MAX_C + c) * 16 + lane];
                }
                Y[(int64_t) sycol[c] * ldy + row] = s;
            }
        }
    }
}

template <int WARPS, int MINB>
static __global__ void __launch_bounds__(WARPS * 32, MINB) k_ait5a16_id(
        const uint8_t * __restrict__ W,
        const int64_t nb02,
        const char    * __restrict__ ids,
        const int64_t ids_nb0, const int64_t ids_nb1,
        const int n_used, const int ne11, const int n_expert,
        const bool direct_zero,
        const float   * __restrict__ Xrot,
        float         * __restrict__ Y,
        const int64_t ldy, const int K) {
    static_assert(WARPS == 3 || WARPS == 4, "AIT5_A16_ID warps");

    extern __shared__ __align__(16) uint8_t smem[];
    __shared__ int sxrow[8];
    __shared__ int sycol[8];
    __shared__ int sn_cols;

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int slab = (int) blockIdx.x;
    const int u    = (int) blockIdx.y;
    const int tok  = (int) blockIdx.z;
    const int own  = tok * n_used + u;

    const bool contig_ids = (ids_nb0 == (int64_t) sizeof(int32_t)) &&
                            (ids_nb1 == (int64_t) n_used * (int64_t) sizeof(int32_t));
    const int e = contig_ids
        ? ((const int32_t *) ids)[own]
        : *(const int32_t *) (ids + u * ids_nb0 + tok * ids_nb1);

    if (e < 0 || e >= n_expert) {
        if (direct_zero && tid < 16) {
            Y[(int64_t) own * ldy + slab * 16 + tid] = 0.0f;
        }
        return;
    }

    // CTA-uniform 64-route ballot dedup executed by warp 0, broadcast via __shared__
    if (warp == 0) {
        const int routes = (int) gridDim.z * n_used; // <= 64
        int id0 = -1, id1 = -1;
        if (contig_ids) {
            const int32_t * idp = (const int32_t *) ids;
            if (lane < routes) id0 = idp[lane];
            if (routes > 32 && (32 + lane) < routes) id1 = idp[32 + lane];
        } else {
            if (lane < routes) {
                const int t0 = lane / n_used, u0 = lane - t0 * n_used;
                id0 = *(const int32_t *) (ids + u0 * ids_nb0 + t0 * ids_nb1);
            }
            if (routes > 32 && (32 + lane) < routes) {
                const int r1 = 32 + lane;
                const int t1 = r1 / n_used, u1 = r1 - t1 * n_used;
                id1 = *(const int32_t *) (ids + u1 * ids_nb0 + t1 * ids_nb1);
            }
        }
        const uint32_t m0 = __ballot_sync(0xffffffffu, (lane < routes) && (id0 == e));
        const uint32_t m1 = (routes > 32) ? __ballot_sync(0xffffffffu, ((32 + lane) < routes) && (id1 == e)) : 0u;
        const uint32_t prior0 = (own < 32) ? (m0 & ((1u << own) - 1u)) : m0;
        const uint32_t prior1 = (own < 32) ? 0u : (m1 & ((1u << (own - 32)) - 1u));
        const int before = __popc(prior0) + __popc(prior1);
        if ((before & 7) != 0) {
            if (lane == 0) sn_cols = 0;
        } else {
            const uint32_t f0 = m0 & ~prior0;
            const uint32_t f1 = m1 & ~prior1;
            const int n0 = __popc(f0);
            const int nc = min(8, n0 + __popc(f1));
            if (lane == 0) sn_cols = nc;

            const bool pow2_used = (n_used & (n_used - 1)) == 0;
            const int  used_sh   = __ffs(n_used) - 1;
            auto route_to_xrow = [&](int r_idx) -> int {
                if (ne11 == n_used) return r_idx;
                if (ne11 == 1) return pow2_used ? (r_idx >> used_sh) : (r_idx / n_used);
                return (r_idx / n_used) * ne11 + (r_idx % n_used) % ne11;
            };

            const uint32_t lower = (1u << lane) - 1u;
            if (f0 & (1u << lane)) {
                const int idx = __popc(f0 & lower);
                if (idx < 8) {
                    sycol[idx] = lane;
                    sxrow[idx] = route_to_xrow(lane);
                }
            }
            if (f1 & (1u << lane)) {
                const int idx = n0 + __popc(f1 & lower);
                if (idx < 8) {
                    sycol[idx] = 32 + lane;
                    sxrow[idx] = route_to_xrow(32 + lane);
                }
            }
        }
    }
    __syncthreads();

    const int n_cols = sn_cols;
    if (n_cols == 0) {
        return;
    }

    const uint8_t * W_exp = W + (int64_t) e * nb02;
    if (n_cols == 1) {
        ait5a16_id_run<1, WARPS>(smem, W_exp, Xrot, Y, ldy, K, slab, 1, sxrow, sycol);
    } else {
        ait5a16_id_run<8, WARPS>(smem, W_exp, Xrot, Y, ldy, K, slab, n_cols, sxrow, sycol);
    }
}

template <int T>
static void launch_dense_t(const uint8_t * W, const float * x_rot, float * Y, int M, int K, cudaStream_t stream) {
    const dim3 grid(M / 16, 1, 1);
    const int nk = K / 128;
    if ((nk % 4 != 0) && (nk % 3 == 0)) {
        constexpr int smem = dense_smem_bytes<3>();
        k_ait5a16_dense<T, 3, 8><<<grid, 3 * 32, smem, stream>>>(W, x_rot, Y, M, K);
    } else {
        constexpr int smem = dense_smem_bytes<4>();
        k_ait5a16_dense<T, 4, 6><<<grid, 4 * 32, smem, stream>>>(W, x_rot, Y, M, K);
    }
}

static bool device_ok(const ggml_backend_cuda_context & ctx) {
    const auto & dev = ggml_cuda_info().devices[ctx.device];
    return GGML_CUDA_CC_IS_NVIDIA(dev.cc) && dev.cc >= GGML_CUDA_CC_AMPERE && dev.warp_size == 32;
}

} // namespace ait5a16

bool ggml_cuda_ait5a16_mul_mat(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {
    if (!ggml_cuda_ait5a16_enabled() || !ait5a16::device_ok(ctx)) {
        return false;
    }
    if (src0 == nullptr || src1 == nullptr || dst == nullptr) {
        return false;
    }
    if (src0->type != GGML_TYPE_AIT5_U || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    // Decline non-contiguous, broadcast, or batched (ne02 > 1) tensors cleanly back to default ai4
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 ||
            dst->ne[2] != 1 || dst->ne[3] != 1) {
        return false;
    }
    const int64_t K = src0->ne[0];
    const int64_t M = src0->ne[1];
    const int64_t T = src1->ne[1];
    if (K <= 0 || K % 128 != 0 || K > INT_MAX || M <= 0 || M % 16 != 0 || M > INT_MAX) {
        return false;
    }
    if (T < 1 || T > ggml_cuda_ait5a16_max_t()) {
        return false;
    }
    if (src1->ne[0] != K || dst->ne[0] != M || dst->ne[1] != T) {
        return false;
    }
    if (src0->nb[1] != ggml_row_size(GGML_TYPE_AIT5_U, K)) {
        return false;
    }
    // 16B alignment required for cp.async.cg.16 on weights and float4 loads on src1
    if ((((uintptr_t) src0->data) & 15u) != 0u || (((uintptr_t) src1->data) & 15u) != 0u) {
        return false;
    }

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<float> x_rot(ctx.pool(), (size_t) T * (size_t) K);
    ait5a16::launch_rotate_fp32((const float *) src1->data, x_rot.get(), (int) K, T, stream);

    const uint8_t * W = (const uint8_t *) src0->data;
    float * Y = (float *) dst->data;
    switch ((int) T) {
        case 1: ait5a16::launch_dense_t<1>(W, x_rot.get(), Y, (int) M, (int) K, stream); break;
        case 2: ait5a16::launch_dense_t<2>(W, x_rot.get(), Y, (int) M, (int) K, stream); break;
        case 3: ait5a16::launch_dense_t<3>(W, x_rot.get(), Y, (int) M, (int) K, stream); break;
        case 4: ait5a16::launch_dense_t<4>(W, x_rot.get(), Y, (int) M, (int) K, stream); break;
        case 5: ait5a16::launch_dense_t<5>(W, x_rot.get(), Y, (int) M, (int) K, stream); break;
        case 6: ait5a16::launch_dense_t<6>(W, x_rot.get(), Y, (int) M, (int) K, stream); break;
        case 7: ait5a16::launch_dense_t<7>(W, x_rot.get(), Y, (int) M, (int) K, stream); break;
        case 8: ait5a16::launch_dense_t<8>(W, x_rot.get(), Y, (int) M, (int) K, stream); break;
        default: return false;
    }
    g_ait5a16_fires[0].fetch_add(1, std::memory_order_relaxed);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool ggml_cuda_ait5a16_mul_mat_id(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * dst,
        float * y_into) {
    if (!ggml_cuda_ait5a16_id_enabled() || !ait5a16::device_ok(ctx) || dst == nullptr) {
        return false;
    }
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];
    if (src0 == nullptr || src1 == nullptr || ids == nullptr) {
        return false;
    }
    if (src0->type != GGML_TYPE_AIT5_U || src1->type != GGML_TYPE_F32 ||
            dst->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) {
        return false;
    }
    if (ids->nb[0] != sizeof(int32_t) || ids->ne[2] != 1 || ids->ne[3] != 1 ||
            !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }
    const int64_t K        = src0->ne[0];
    const int64_t M        = src0->ne[1];
    const int64_t n_expert = src0->ne[2];
    const int64_t ne11     = src1->ne[1];
    const int64_t n_used   = ids->ne[0];
    const int64_t n_tok    = ids->ne[1];
    if (K <= 0 || K % 128 != 0 || K > INT_MAX || M <= 0 || M % 16 != 0 || M > INT_MAX) {
        return false;
    }
    if (n_expert <= 0 || n_expert > INT_MAX || n_used <= 0 || n_tok < 1 || n_tok > ggml_cuda_ait5a16_max_t()) {
        return false;
    }
    if (n_used * n_tok > 64) {
        return false;
    }
    if (src0->ne[3] != 1 || src0->nb[0] != ggml_type_size(GGML_TYPE_AIT5_U) ||
            src0->nb[1] != ggml_row_size(GGML_TYPE_AIT5_U, K) ||
            src0->nb[2] < src0->nb[1] * M || (src0->nb[2] & 15u) != 0u) {
        return false;
    }
    if (src1->ne[0] != K || (ne11 != 1 && ne11 != n_used) || src1->ne[2] != n_tok || src1->ne[3] != 1) {
        return false;
    }
    if (dst->ne[0] != M || dst->ne[1] != n_used || dst->ne[2] != n_tok || dst->ne[3] != 1) {
        return false;
    }
    if ((((uintptr_t) src0->data) & 15u) != 0u || (((uintptr_t) src1->data) & 15u) != 0u) {
        return false;
    }

    cudaStream_t stream = ctx.stream();
    const int64_t n_rows = ne11 * n_tok;
    ggml_cuda_pool_alloc<float> x_rot(ctx.pool(), (size_t) n_rows * (size_t) K);
    ait5a16::launch_rotate_fp32((const float *) src1->data, x_rot.get(), (int) K, n_rows, stream);

    float * const y_out = y_into ? y_into : (float *) dst->data;
    const bool direct_zero = (y_into == nullptr);
    const dim3 grid((unsigned int) (M / 16), (unsigned int) n_used, (unsigned int) n_tok);
    const int nk = (int) (K / 128);
    if ((nk % 4 != 0) && (nk % 3 == 0)) {
        constexpr int smem = ait5a16::dense_smem_bytes<3>();
        ait5a16::k_ait5a16_id<3, 8><<<grid, 3 * 32, smem, stream>>>(
            (const uint8_t *) src0->data, (int64_t) src0->nb[2],
            (const char *) ids->data, (int64_t) ids->nb[0], (int64_t) ids->nb[1],
            (int) n_used, (int) ne11, (int) n_expert, direct_zero,
            x_rot.get(), y_out, M, (int) K);
    } else {
        constexpr int smem = ait5a16::dense_smem_bytes<4>();
        ait5a16::k_ait5a16_id<4, 6><<<grid, 4 * 32, smem, stream>>>(
            (const uint8_t *) src0->data, (int64_t) src0->nb[2],
            (const char *) ids->data, (int64_t) ids->nb[0], (int64_t) ids->nb[1],
            (int) n_used, (int) ne11, (int) n_expert, direct_zero,
            x_rot.get(), y_out, M, (int) K);
    }
    g_ait5a16_fires[1].fetch_add(1, std::memory_order_relaxed);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
