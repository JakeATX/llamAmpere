// NEONKERN: sm_86 cliff-free int8 tensor-core NVFP4 MoE decode/verify kernels (T = 1..8).
//
// Design:
// - Weights stay in native ggml block_nvfp4 layout (36 bytes per 64 weights: 4 UE4M3 scales + 32 code bytes).
// - Each 16-row slab is a contiguous, 128-byte-aligned DRAM stream (gate/up: 20,736 B = 162 lines; down: 10,368 B = 81 lines),
//   double-buffered via 16-byte cp.async.cg with optional L2::evict_first cache policy so expert weights do not evict hot
//   activations or KV metadata from L2.
// - Each 16-element NVFP4 sub-block maps directly to one mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 with weights as A
//   (16 rows x 16 cols) and up to 8 member tokens/slots as B (16 cols x 8 cols), so T = 1..8 execute the exact same MMA work.
// - Activations (P.xq, P.act) use a private split-SOA layout inside the existing act scratch (zero extra VRAM) pre-permuted
//   to match the raw NVFP4 nibble layout so decoding 4 weights takes 3 PRMT + 2 ALU ops with zero de-interleave, and loading
//   all 4 sub-blocks of B takes one LDG.128 per 64 weights from L1 cache.
// - Every 16-element sub-block integer dot product and UE4M3/q8_1 scale is exact (identical terms to vec_dot_nvfp4_q8_1);
//   only the outer FP32 K-summation order across warps differs.

#include "nk-nvfp4-tc.cuh"

#include <atomic>
#include <cstdlib>

bool ggml_cuda_nk_nvfp4_tc_enabled() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_NK_NVFP4_TC");
        return e != nullptr && atoi(e) != 0;
    }();
    return on;
}

static bool nk_nvfp4_tc_l2_ef() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_NK_NVFP4_TC_L2");
        return e == nullptr || atoi(e) != 0;
    }();
    return on;
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

namespace nk_nvfp4_tc {

// Match k_ink_situ / nvfp4moe_situ per-element expressions exactly without pulling in nvfp4moe-kern.cuh kernels.
static __device__ __forceinline__ float nk_situ(const float gv, const float uv, const ink_situ_params & p) {
    const float a1 = p.ga_s * gv + p.ga_b;
    const float a2 = tanhf(a1);
    const float a3 = p.gb_s * a2 + p.gb_b;
    const float sg = 1.0f / (1.0f + expf(-gv));
    const float a4 = p.sig_first ? sg * a3 : a3 * sg;
    const float u1 = p.ua_s * uv + p.ua_b;
    const float u2 = tanhf(u1);
    const float u3 = p.ub_s * u2 + p.ub_b;
    return p.act_first ? a4 * u3 : u3 * a4;
}

constexpr int NK_K       = 2304;
constexpr int NK_F       = 1152;
constexpr int NK_NUSED   = 8;
constexpr int NK_NB32_K  = NK_K / 32;  // 72
constexpr int NK_NB32_F  = NK_F / 32;  // 36
constexpr int NK_NB64_K  = NK_K / 64;  // 36
constexpr int NK_NB64_F  = NK_F / 64;  // 18
constexpr int NK_ROW_GU  = NK_NB64_K * 36;  // 1296 bytes
constexpr int NK_ROW_DN  = NK_NB64_F * 36;  // 648 bytes
constexpr int NK_SLAB_GU = 16 * NK_ROW_GU;  // 20736 bytes (162 x 128B lines)
constexpr int NK_SLAB_DN = 16 * NK_ROW_DN;  // 10368 bytes (81 x 128B lines)

constexpr int NK_NW_GU   = 12; // 36 blocks / 12 warps = 3 blocks/warp; 2 CTAs/SM = 24 warps/SM
constexpr int NK_NW_DN   = 6;  // 18 blocks / 6 warps  = 3 blocks/warp; 4 CTAs/SM = 24 warps/SM

// Dynamic shared memory sizes (multiples of 128 bytes):
// gu: 2 * 20736 (ring) + 12 * 128 * 4 (red) + 256 * 4 (act_s) + 32 (s_slots) = 48672 -> pad 48768 (2 CTAs/SM = 97536 <= 101376)
// dn: 2 * 10368 (ring) +  6 * 128 * 4 (red) + 32 (s_slots)                   = 23840 -> pad 23936 (4 CTAs/SM = 95744 <= 101376)
constexpr size_t NK_SMEM_GU = ((2 * NK_SLAB_GU + NK_NW_GU * 128 * sizeof(float) + 256 * sizeof(float) + 32 + 127) & ~size_t(127));
constexpr size_t NK_SMEM_DN = ((2 * NK_SLAB_DN + NK_NW_DN * 128 * sizeof(float) + 32 + 127) & ~size_t(127));

static_assert(NK_SMEM_GU * 2 <= 101376, "NK_NVFP4_TC gu must fit 2 CTAs/SM on sm_86");
static_assert(NK_SMEM_DN * 4 <= 101376, "NK_NVFP4_TC dn must fit 4 CTAs/SM on sm_86");

// Exact signed-int-to-float via 1.5 * 2^23 binade bias (exact for |x| < 2^22; sub-block max |sumi| = 16*12*127 = 24384).
constexpr int   NK_I2F_BIAS_I = 0x4B400000;
constexpr float NK_I2F_BIAS_F = 12582912.0f;

// kvalues_mxfp4 int8 LUT in 4 register constants:
// indices 0..7:  {0, 1, 2, 3, 4, 6, 8, 12}
// indices 8..15: {0, -1, -2, -3, -4, -6, -8, -12}
constexpr uint32_t NK_LUT_T0 = 0x03020100u;
constexpr uint32_t NK_LUT_T1 = 0x0C080604u;
constexpr uint32_t NK_LUT_T2 = 0xFDFEFF00u;
constexpr uint32_t NK_LUT_T3 = 0xF4F8FAFCu;

static __device__ __forceinline__ uint64_t nk_l2_evict_first() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    uint64_t p;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
    return p;
#else
    return 0ull;
#endif
}

template <bool L2_EF>
static __device__ __forceinline__ void nk_cp16(const uint32_t dst_s, const void * src, const uint64_t policy) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if constexpr (L2_EF) {
        asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;"
            :: "r"(dst_s), "l"(src), "l"(policy) : "memory");
    } else {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
            :: "r"(dst_s), "l"(src) : "memory");
    }
#else
    GGML_UNUSED_VARS(dst_s, src, policy);
    __trap();
#endif
}

static __device__ __forceinline__ void nk_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}

template <int N>
static __device__ __forceinline__ void nk_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory");
#endif
}

static __device__ __forceinline__ void nk_mma_k16_bias(
        int & c0, int & c1, int & c2, int & c3,
        const uint32_t a0, const uint32_t a1, const uint32_t b0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const int bias = NK_I2F_BIAS_I;
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %7, %7, %7};"
        : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)
        : "r"(a0), "r"(a1), "r"(b0), "r"(bias));
#else
    GGML_UNUSED_VARS(a0, a1, b0);
    c0 = c1 = c2 = c3 = 0;
    __trap();
#endif
}

// Branchless UE4M3 -> fp32 (includes 0.5 factor; bit-identical to ggml_cuda_ue4m3_to_fp32 for all 256 bytes on sm_86).
static __device__ __forceinline__ float nk_ue4m3(const uint32_t u) {
    const uint32_t x  = u & 0x7Fu;
    const bool     sn = x < 8u;
    const float    v  = __fsub_rn(__uint_as_float((sn ? 0x3C000000u : 0x3B800000u) + (x << 20)), sn ? 0.0078125f : 0.0f);
    return u == 0x7Fu ? 0.0f : v;
}

// Decode 4 packed E2M1 nibbles in the low 16 bits of q16 straight into 4 int8 bytes in one uint32_t.
// Byte order matches {elem 2t+0, elem 2t+8, elem 2t+1, elem 2t+9}; no de-interleave needed.
static __device__ __forceinline__ uint32_t nk_dec4(const uint32_t q16) {
    const uint32_t idx = q16 & 0x7777u;
    const uint32_t lo  = __byte_perm(NK_LUT_T0, NK_LUT_T1, idx);
    const uint32_t hi  = __byte_perm(NK_LUT_T2, NK_LUT_T3, idx);
    const uint32_t sel = 0x3210u | ((q16 & 0x8888u) >> 1);
    return __byte_perm(lo, hi, sel);
}

// Quantize one 32-element block (lane 0..31 holds xi) into split-SOA qs + d layout:
// - qs stores 64-element blocks laid out as [t = 0..3][s = 0..3][4 nibs] so lane (g, t) reads all 4 sub-blocks in one uint4.
// - d stores float scale rounded through fp16 to match block_q8_1's __low2float(ds) bit-for-bit.
static __device__ __forceinline__ void nk_quant_store_32(
        const float xi, int8_t * qs_row, float * d_row, const int b32, const int lane) {
    float amax = fabsf(xi);
    amax = warp_reduce_max<32>(amax);
    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : (int8_t) roundf(xi / d);

    // Pack the 8 4-byte words (t = 0..3, sub = 0..1) via warp shuffles (all 32 lanes participate).
    const int l7   = lane & 7;
    const int t    = l7 >> 1;
    const int sub  = l7 & 1;
    const int base = (sub << 4) | (t << 1);

    const uint32_t b0 = (uint8_t) __shfl_sync(0xffffffffu, (int) q, base + 0);
    const uint32_t b1 = (uint8_t) __shfl_sync(0xffffffffu, (int) q, base + 8);
    const uint32_t b2 = (uint8_t) __shfl_sync(0xffffffffu, (int) q, base + 1);
    const uint32_t b3 = (uint8_t) __shfl_sync(0xffffffffu, (int) q, base + 9);

    if (lane < 8) {
        const int b64 = b32 >> 1;
        const int s   = ((b32 & 1) << 1) | sub;
        uint32_t * dst32 = (uint32_t *) (qs_row + b64 * 64 + t * 16);
        dst32[s] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
    }
    if (lane == 0) {
        d_row[b32] = __half2float(__float2half_rn(d));
    }
}

// Quantize P.x [T][2304] into P.xq split-SOA (xq_qs [T][2304] int8, xq_d [T][72] float).
static __global__ void __launch_bounds__(256) k_nk_nvfp4_xq(const nvfp4moe_params P) {
    const int b = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (b >= P.T * NK_NB32_K) {
        return;
    }
    const int lane = threadIdx.x & 31;
    const int tok  = b / NK_NB32_K;
    const int b32  = b - tok * NK_NB32_K;

    int8_t * xq_qs = (int8_t *) P.xq;
    float  * xq_d  = (float *) (xq_qs + (int64_t) P.T * NK_K);

    const float xi = P.x[(int64_t) tok * P.x_s2 + b32 * 32 + lane];
    nk_quant_store_32(xi, xq_qs + (int64_t) tok * NK_K, xq_d + (int64_t) tok * NK_NB32_K, b32, lane);
}

// Warp-level routing ballot over up to 64 slots (T <= 8, n_used == 8).
// Returns N (1..8) slots handled by this CTA (0 if not the chunk leader or invalid expert id).
// Handles duplicate ids with > 8 routes via 8-slot chunking ((before & 7) == 0).
static __device__ __forceinline__ int nk_route_ballot(
        const nvfp4moe_params & P, const int lane, int & e_out, uint32_t & f0_out, uint32_t & f1_out, int & n0_out) {
    const int ns  = P.T * NK_NUSED;
    const int own = (int) blockIdx.y;
    const int e   = P.ids[(own >> 3) * P.ids_s1 + (own & 7)];
    e_out = e;
    if (e < 0) {
        return 0;
    }
    const int id0 = (lane < ns)      ? P.ids[(lane >> 3) * P.ids_s1 + (lane & 7)]             : -1;
    const int id1 = (lane + 32 < ns) ? P.ids[((lane + 32) >> 3) * P.ids_s1 + ((lane + 32) & 7)] : -1;

    const uint32_t m0 = __ballot_sync(0xffffffffu, (lane < ns) && (id0 == e));
    const uint32_t m1 = (ns > 32) ? __ballot_sync(0xffffffffu, (lane + 32 < ns) && (id1 == e)) : 0u;

    const uint32_t prior0 = (own < 32) ? (m0 & ((1u << own) - 1u)) : m0;
    const uint32_t prior1 = (own < 32) ? 0u : (m1 & ((1u << (own - 32)) - 1u));
    const int before = __popc(prior0) + __popc(prior1);
    if ((before & 7) != 0) {
        return 0;
    }
    const uint32_t f0 = m0 & ~prior0;
    const uint32_t f1 = m1 & ~prior1;
    const int n0 = __popc(f0);
    f0_out = f0;
    f1_out = f1;
    n0_out = n0;
    return min(8, n0 + __popc(f1));
}

// Scatter up to 8 active slot indices into s_slots[0..N-1] using warp 0.
static __device__ __forceinline__ void nk_scatter_slots(
        int * s_slots, const uint32_t f0, const uint32_t f1, const int n0, const int lane) {
    const uint32_t lower = (1u << lane) - 1u;
    if (f0 & (1u << lane)) {
        const int idx = __popc(f0 & lower);
        if (idx < 8) {
            s_slots[idx] = lane;
        }
    }
    if (f1 & (1u << lane)) {
        const int idx = n0 + __popc(f1 & lower);
        if (idx < 8) {
            s_slots[idx] = 32 + lane;
        }
    }
}

// Compute 16 rows x up to 8 columns for one staged 16-row slab using N_BLK = 3 block_nvfp4 blocks per warp.
template <int ROW_BYTES, int N_BLK>
static __device__ __forceinline__ void nk_slab_dot(
        const char * __restrict__ slab_s,
        const int8_t * __restrict__ xq_qs_g,
        const float  * __restrict__ xq_d_c0,
        const float  * __restrict__ xq_d_c1,
        const int warp, const int lane, float (&acc)[4]) {
    const int g   = lane >> 2;
    const int t   = lane & 3;
    const int kb0 = warp * N_BLK;

    const char * row0 = slab_s + g * ROW_BYTES;
    const char * row8 = slab_s + (g + 8) * ROW_BYTES;

    acc[0] = 0.0f;
    acc[1] = 0.0f;
    acc[2] = 0.0f;
    acc[3] = 0.0f;

#pragma unroll 1
    for (int bi = 0; bi < N_BLK; ++bi) {
        const int kb = kb0 + bi;

        const uint4  bv  = xq_qs_g ? __ldg((const uint4 *)  (xq_qs_g + kb * 64 + t * 16)) : make_uint4(0u, 0u, 0u, 0u);
        const float2 dx0 = xq_d_c0 ? __ldg((const float2 *) (xq_d_c0 + 2 * kb))           : make_float2(0.0f, 0.0f);
        const float2 dx1 = xq_d_c1 ? __ldg((const float2 *) (xq_d_c1 + 2 * kb))           : make_float2(0.0f, 0.0f);

        const char * b0 = row0 + kb * 36;
        const char * b8 = row8 + kb * 36;
        const uint32_t d0_raw = *(const uint32_t *) b0;
        const uint32_t d8_raw = *(const uint32_t *) b8;

#pragma unroll
        for (int s = 0; s < 4; ++s) {
            const uint32_t b_word = (s == 0) ? bv.x : (s == 1) ? bv.y : (s == 2) ? bv.z : bv.w;
            const float    d_x0   = (s < 2)  ? dx0.x : dx0.y;
            const float    d_x1   = (s < 2)  ? dx1.x : dx1.y;

            const uint32_t q0 = *(const uint16_t *) (b0 + 4 + 8 * s + 2 * t);
            const uint32_t q8 = *(const uint16_t *) (b8 + 4 + 8 * s + 2 * t);
            const uint32_t a0 = nk_dec4(q0);
            const uint32_t a1 = nk_dec4(q8);

            int c0, c1, c2, c3;
            nk_mma_k16_bias(c0, c1, c2, c3, a0, a1, b_word);

            const float dw0 = nk_ue4m3((d0_raw >> (8 * s)) & 0xFFu);
            const float dw8 = nk_ue4m3((d8_raw >> (8 * s)) & 0xFFu);

            const float f0 = __fsub_rn(__int_as_float(c0), NK_I2F_BIAS_F);
            const float f1 = __fsub_rn(__int_as_float(c1), NK_I2F_BIAS_F);
            const float f2 = __fsub_rn(__int_as_float(c2), NK_I2F_BIAS_F);
            const float f3 = __fsub_rn(__int_as_float(c3), NK_I2F_BIAS_F);

            acc[0] = fmaf(dw0 * d_x0, f0, acc[0]);
            acc[1] = fmaf(dw0 * d_x1, f1, acc[1]);
            acc[2] = fmaf(dw8 * d_x0, f2, acc[2]);
            acc[3] = fmaf(dw8 * d_x1, f3, acc[3]);
        }
    }
}

// gate_up + scale + SITU -> split-SOA P.act.
// Grid: (F/32 = 36, T*8), 384 threads (12 warps), 2 CTAs/SM = 24 warps/SM on sm_86.
template <bool L2_EF>
static __global__ void __launch_bounds__(NK_NW_GU * 32, 2) k_nk_nvfp4_gu(const nvfp4moe_params P) {
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    int e, n0;
    uint32_t f0_m, f1_m;
    const int N = nk_route_ballot(P, lane, e, f0_m, f1_m, n0);
    if (N == 0) {
        return; // CTA-uniform, before any barrier
    }

    extern __shared__ __align__(16) char smem[];
    char  * ring    = smem;                                       // [2][20736]
    float * red     = (float *) (ring + 2 * NK_SLAB_GU);          // [12][8][16]
    float * act_s   = red + NK_NW_GU * 128;                       // [8][32]
    int   * s_slots = (int *) (act_s + 256);                      // [8]

    const uint32_t ring_s = (uint32_t) __cvta_generic_to_shared(ring);
    const uint64_t policy = L2_EF ? nk_l2_evict_first() : 0ull;

    const int feat0 = (int) blockIdx.x * 32;
    const char * wg = P.w_gu + (int64_t) e * P.gu_nb2 + (int64_t) feat0 * NK_ROW_GU;
    const char * wu = wg + (int64_t) NK_F * NK_ROW_GU;

    auto slab_src = [&](const int s) -> const char * {
        const char * base = (s & 1) ? wu : wg;
        return base + (s >= 2 ? NK_SLAB_GU : 0);
    };
    auto issue_slab = [&](const int s, const int buf) {
        const char * src = slab_src(s);
        const uint32_t dst = ring_s + (uint32_t) (buf * NK_SLAB_GU);
#pragma unroll
        for (int c = tid; c < NK_SLAB_GU / 16; c += NK_NW_GU * 32) {
            nk_cp16<L2_EF>(dst + (uint32_t) (c * 16), src + c * 16, policy);
        }
        nk_commit();
    };

    // Prologue: start DRAM stream of slab 0 immediately, scatter slot table under its latency.
    issue_slab(0, 0);
    if (warp == 0) {
        nk_scatter_slots(s_slots, f0_m, f1_m, n0, lane);
    }
    const float scale = P.s_gu ? P.s_gu[e] : 1.0f;
    nk_wait<0>();
    __syncthreads();

    const int g = lane >> 2;
    const int t = lane & 3;
    const int tok_g  = (g < N)         ? (s_slots[g]         >> 3) : -1;
    const int tok_c0 = (2 * t < N)     ? (s_slots[2 * t]     >> 3) : -1;
    const int tok_c1 = (2 * t + 1 < N) ? (s_slots[2 * t + 1] >> 3) : -1;

    const int8_t * xq_qs = (const int8_t *) P.xq;
    const float  * xq_d  = (const float *) (xq_qs + (int64_t) P.T * NK_K);

    const int8_t * xq_qs_g = tok_g  >= 0 ? xq_qs + (int64_t) tok_g  * NK_K     : nullptr;
    const float  * xq_d_c0 = tok_c0 >= 0 ? xq_d  + (int64_t) tok_c0 * NK_NB32_K : nullptr;
    const float  * xq_d_c1 = tok_c1 >= 0 ? xq_d  + (int64_t) tok_c1 * NK_NB32_K : nullptr;

#pragma unroll 1
    for (int s = 0; s < 4; ++s) {
        if (s + 1 < 4) {
            issue_slab(s + 1, (s + 1) & 1);
        }

        float acc[4];
        nk_slab_dot<NK_ROW_GU, 3>(ring + (s & 1) * NK_SLAB_GU, xq_qs_g, xq_d_c0, xq_d_c1, warp, lane, acc);

        float * rw = red + warp * 128;
        if (2 * t < N) {
            rw[(2 * t) * 16 + g]     = acc[0];
            rw[(2 * t) * 16 + g + 8] = acc[2];
        }
        if (2 * t + 1 < N) {
            rw[(2 * t + 1) * 16 + g]     = acc[1];
            rw[(2 * t + 1) * 16 + g + 8] = acc[3];
        }
        __syncthreads();

        if (tid < N * 16) {
            const int c = tid >> 4;
            const int r = tid & 15;
            float sum = 0.0f;
#pragma unroll
            for (int w = 0; w < NK_NW_GU; ++w) {
                sum += red[w * 128 + c * 16 + r];
            }
            if (P.s_gu) {
                sum *= scale;
            }
            const int foff = (s >= 2 ? 16 : 0) + r;
            if ((s & 1) == 0) {
                act_s[c * 32 + foff] = sum;
            } else {
                const float gv = act_s[c * 32 + foff];
                act_s[c * 32 + foff] = nk_situ(gv, sum, P.situ);
            }
        }
        if (s + 1 < 4) {
            nk_wait<0>();
            __syncthreads();
        }
    }

    __syncthreads();
    if (warp < N) {
        const int slot = s_slots[warp];
        int8_t * act_qs = (int8_t *) P.act;
        float  * act_d  = (float *) (act_qs + (int64_t) P.T * NK_NUSED * NK_F);
        nk_quant_store_32(
            act_s[warp * 32 + lane],
            act_qs + (int64_t) slot * NK_F,
            act_d  + (int64_t) slot * NK_NB32_F,
            (int) blockIdx.x, lane);
    }
}

// down + scale -> P.e_out.
// Grid: (K/32 = 72, T*8), 192 threads (6 warps), 4 CTAs/SM = 24 warps/SM on sm_86.
template <bool L2_EF>
static __global__ void __launch_bounds__(NK_NW_DN * 32, 4) k_nk_nvfp4_dn(const nvfp4moe_params P) {
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    int e, n0;
    uint32_t f0_m, f1_m;
    const int N = nk_route_ballot(P, lane, e, f0_m, f1_m, n0);
    if (N == 0) {
        return; // CTA-uniform, before any barrier
    }

    extern __shared__ __align__(16) char smem[];
    char  * ring    = smem;                                       // [2][10368]
    float * red     = (float *) (ring + 2 * NK_SLAB_DN);          // [6][8][16]
    int   * s_slots = (int *) (red + NK_NW_DN * 128);             // [8]

    const uint32_t ring_s = (uint32_t) __cvta_generic_to_shared(ring);
    const uint64_t policy = L2_EF ? nk_l2_evict_first() : 0ull;

    const int row_base = (int) blockIdx.x * 32;
    const char * wd = P.w_dn + (int64_t) e * P.dn_nb2 + (int64_t) row_base * NK_ROW_DN;

    auto issue_slab = [&](const int s, const int buf) {
        const char * src = wd + s * NK_SLAB_DN;
        const uint32_t dst = ring_s + (uint32_t) (buf * NK_SLAB_DN);
#pragma unroll
        for (int c = tid; c < NK_SLAB_DN / 16; c += NK_NW_DN * 32) {
            nk_cp16<L2_EF>(dst + (uint32_t) (c * 16), src + c * 16, policy);
        }
        nk_commit();
    };

    // Prologue: start DRAM stream of slab 0 immediately, scatter slot table under its latency.
    issue_slab(0, 0);
    if (warp == 0) {
        nk_scatter_slots(s_slots, f0_m, f1_m, n0, lane);
    }
    const float scale = P.s_dn ? P.s_dn[e] : 1.0f;
    nk_wait<0>();
    __syncthreads();

    const int g = lane >> 2;
    const int t = lane & 3;
    const int slot_g  = (g < N)         ? s_slots[g]         : -1;
    const int slot_c0 = (2 * t < N)     ? s_slots[2 * t]     : -1;
    const int slot_c1 = (2 * t + 1 < N) ? s_slots[2 * t + 1] : -1;

    const int8_t * act_qs = (const int8_t *) P.act;
    const float  * act_d  = (const float *) (act_qs + (int64_t) P.T * NK_NUSED * NK_F);

    const int8_t * act_qs_g = slot_g  >= 0 ? act_qs + (int64_t) slot_g  * NK_F      : nullptr;
    const float  * act_d_c0 = slot_c0 >= 0 ? act_d  + (int64_t) slot_c0 * NK_NB32_F : nullptr;
    const float  * act_d_c1 = slot_c1 >= 0 ? act_d  + (int64_t) slot_c1 * NK_NB32_F : nullptr;

#pragma unroll 1
    for (int s = 0; s < 2; ++s) {
        if (s == 0) {
            issue_slab(1, 1);
        }

        float acc[4];
        nk_slab_dot<NK_ROW_DN, 3>(ring + s * NK_SLAB_DN, act_qs_g, act_d_c0, act_d_c1, warp, lane, acc);

        float * rw = red + warp * 128;
        if (2 * t < N) {
            rw[(2 * t) * 16 + g]     = acc[0];
            rw[(2 * t) * 16 + g + 8] = acc[2];
        }
        if (2 * t + 1 < N) {
            rw[(2 * t + 1) * 16 + g]     = acc[1];
            rw[(2 * t + 1) * 16 + g + 8] = acc[3];
        }
        __syncthreads();

        if (tid < N * 16) {
            const int c = tid >> 4;
            const int r = tid & 15;
            float sum = 0.0f;
#pragma unroll
            for (int w = 0; w < NK_NW_DN; ++w) {
                sum += red[w * 128 + c * 16 + r];
            }
            if (P.s_dn) {
                sum *= scale;
            }
            P.e_out[(int64_t) s_slots[c] * NK_K + row_base + s * 16 + r] = sum;
        }
        if (s == 0) {
            nk_wait<0>();
            __syncthreads();
        }
    }
}

template <bool L2_EF>
static void launch_pair(const nvfp4moe_params & P, cudaStream_t stream) {
    static bool init[GGML_CUDA_MAX_DEVICES] = {};
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    if (!init[dev]) {
        CUDA_CHECK(cudaFuncSetAttribute(k_nk_nvfp4_gu<L2_EF>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) NK_SMEM_GU));
        CUDA_CHECK(cudaFuncSetAttribute(k_nk_nvfp4_gu<L2_EF>, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
        CUDA_CHECK(cudaFuncSetAttribute(k_nk_nvfp4_dn<L2_EF>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) NK_SMEM_DN));
        CUDA_CHECK(cudaFuncSetAttribute(k_nk_nvfp4_dn<L2_EF>, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
        init[dev] = true;
    }
    k_nk_nvfp4_xq<<<(P.T * NK_NB32_K + 7) / 8, 256, 0, stream>>>(P);
    CUDA_CHECK(cudaGetLastError());

    k_nk_nvfp4_gu<L2_EF><<<dim3(NK_F / 32, P.T * NK_NUSED), NK_NW_GU * 32, NK_SMEM_GU, stream>>>(P);
    CUDA_CHECK(cudaGetLastError());

    k_nk_nvfp4_dn<L2_EF><<<dim3(NK_K / 32, P.T * NK_NUSED), NK_NW_DN * 32, NK_SMEM_DN, stream>>>(P);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace nk_nvfp4_tc

bool ggml_cuda_nk_nvfp4_tc_accepts(const nvfp4moe_params & P) {
    using namespace nk_nvfp4_tc;
    return ggml_cuda_nk_nvfp4_tc_enabled() &&
        P.T >= 1 && P.T <= 8 && !P.write_sum && P.xq != nullptr && !P.f32act && P.n_used == NK_NUSED &&
        P.K == NK_K && P.F == NK_F && P.gu_nb1 == NK_ROW_GU && P.dn_nb1 == NK_ROW_DN &&
        P.gu_nb2 % 16 == 0 && P.dn_nb2 % 16 == 0 &&
        (uintptr_t) P.w_gu % 16 == 0 && (uintptr_t) P.w_dn % 16 == 0 &&
        (uintptr_t) P.act % 16 == 0 && (uintptr_t) P.xq % 16 == 0;
}

bool ggml_cuda_nk_nvfp4_tc_launch(const nvfp4moe_params & P, cudaStream_t stream) {
    if (!ggml_cuda_nk_nvfp4_tc_accepts(P)) {
        return false;
    }
    const bool l2_ef = nk_nvfp4_tc_l2_ef();
    static std::atomic_flag logged = ATOMIC_FLAG_INIT;
    if (!logged.test_and_set()) {
        GGML_LOG_INFO("[NK_NVFP4_TC] active (m16n8k16 s8 TC, T=1..8, l2_evict_first=%d, smem gu=%zu dn=%zu)\n",
            l2_ef ? 1 : 0, nk_nvfp4_tc::NK_SMEM_GU, nk_nvfp4_tc::NK_SMEM_DN);
    }
    if (l2_ef) {
        nk_nvfp4_tc::launch_pair<true>(P, stream);
    } else {
        nk_nvfp4_tc::launch_pair<false>(P, stream);
    }
    return true;
}

#else

bool ggml_cuda_nk_nvfp4_tc_accepts(const nvfp4moe_params &) {
    return false;
}

bool ggml_cuda_nk_nvfp4_tc_launch(const nvfp4moe_params &, cudaStream_t) {
    return false;
}

#endif
