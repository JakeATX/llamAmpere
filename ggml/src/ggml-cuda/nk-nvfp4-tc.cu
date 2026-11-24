// NEONKERN: sm_86 cliff-free int8 tensor-core NVFP4 MoE decode/verify kernels (T = 1..8), v2.
//
// Design (v2 per-warp multi-stage cp.async ring):
// - Weights stay in native ggml block_nvfp4 layout (36 bytes per 64 weights: 4 UE4M3 scales + 32 code bytes).
// - Each warp owns 16 complete output rows across the full K dimension and streams K-chunks through its own private
//   cp.async.cg ring buffer with optional L2::evict_first cache policy, accumulating in 4 FP32 registers.
//   This eliminates split-K shared-memory reductions, loop-internal __syncthreads(), and wait_group 0 stalls:
//   - gu: 2 warps/CTA (32 features, grid (36, T*8)), 9 chunks x 4 blocks (144 B/row), D=4 stages/warp (wait_group 2,
//         3 stages = 6,912 B/warp always in flight), 18 continuous gate->up steps, 19,456 B SMEM (5 CTAs/SM = 69.1 KB/SM
//         in flight), 1 single __syncthreads() in the entire kernel before P.act quantization.
//   - dn: 2 warps/CTA (64 rows = 2 slabs of 16 rows/warp, grid (36, T*8)), 3 chunks x 6 blocks (216 B/row) via
//         16B-aligned 224 B over-fetch into 240 B bank-conflict-free SMEM stride (0 bytes OOB), D=3 stages/warp
//         (wait_group 1, 2 stages = 7,168 B payload/warp always in flight), 6 continuous steps across both slabs,
//         23,040 B SMEM (4 CTAs/SM = 57.3 KB/SM in flight), 0 __syncthreads() in the entire kernel.
// - Each 16-element NVFP4 sub-block maps directly to one mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 with weights as A
//   (16 rows x 16 cols) and up to 8 member tokens/slots as B (16 cols x 8 cols), so T = 1..8 execute identical MMA work.
// - Activations (P.xq, P.act) use a private split-SOA layout inside the existing act scratch (zero extra VRAM) pre-permuted
//   to match the raw NVFP4 nibble layout so decoding 4 weights takes 3 PRMT + 2 ALU ops with zero de-interleave, and loading
//   all 4 sub-blocks of B takes one LDG.128 per 64 weights from L1 cache.
// - Every 16-element sub-block integer dot product and UE4M3/q8_1 scale is exact (identical terms to vec_dot_nvfp4_q8_1).

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

static bool nk_nvfp4_tc_trace() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_NK_TRACE");
        return e != nullptr && atoi(e) != 0;
    }();
    return on;
}

static std::atomic<int64_t> g_nk_nvfp4_tc_launches{0};

int64_t ggml_cuda_nk_nvfp4_tc_count() {
    return g_nk_nvfp4_tc_launches.load(std::memory_order_relaxed);
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
constexpr int NK_SLAB_DN = 16 * NK_ROW_DN;  // 10368 bytes

// gu per-warp ring: 9 chunks x 4 blocks = 144 B/row (9 x 16B, bank-conflict-free), D = 4 stages/warp, 2 warps/CTA.
constexpr int NK_NW_GU     = 2;
constexpr int NK_GU_CHUNKS = 9;
constexpr int NK_GU_CBLK   = 4;
constexpr int NK_GU_CBYTES = NK_GU_CBLK * 36;         // 144 B
constexpr int NK_GU_STAGE  = 16 * NK_GU_CBYTES;       // 2304 B
constexpr int NK_GU_D      = 4;
constexpr int NK_GU_WRING  = NK_GU_D * NK_GU_STAGE;   // 9216 B

// dn per-warp ring: 3 chunks x 6 blocks = 216 B/row via 16B-aligned 224 B over-fetch into 240 B SMEM row stride,
// D = 3 stages/warp, 2 warps/CTA (2 slabs of 16 rows/warp = 64 rows/CTA).
constexpr int NK_NW_DN     = 2;
constexpr int NK_DN_CHUNKS = 3;
constexpr int NK_DN_CBLK   = 6;
constexpr int NK_DN_CBYTES = NK_DN_CBLK * 36;         // 216 B
constexpr int NK_DN_SROW   = 240;                     // >= 224 B, 60 words == 28 mod 32 -> 0 bank conflicts for even/odd skew
constexpr int NK_DN_STAGE  = 16 * NK_DN_SROW;         // 3840 B
constexpr int NK_DN_D      = 3;
constexpr int NK_DN_WRING  = NK_DN_D * NK_DN_STAGE;   // 11520 B

// Dynamic shared memory sizes (multiples of 128 bytes):
// gu: 2 * 9216 (per-warp ring) + 256 * 4 (act_s) = 19456 B -> 5 CTAs/SM = 97280 <= 101376
// dn: 2 * 11520 (per-warp ring)                  = 23040 B -> 4 CTAs/SM = 92160 <= 101376
constexpr size_t NK_SMEM_GU = (size_t) (NK_NW_GU * NK_GU_WRING + 256 * (int) sizeof(float));
constexpr size_t NK_SMEM_DN = (size_t) (NK_NW_DN * NK_DN_WRING);

static_assert(NK_SMEM_GU % 128 == 0 && NK_SMEM_GU * 5 <= 101376, "NK_NVFP4_TC gu must fit 5 CTAs/SM on sm_86");
static_assert(NK_SMEM_DN % 128 == 0 && NK_SMEM_DN * 4 <= 101376, "NK_NVFP4_TC dn must fit 4 CTAs/SM on sm_86");

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
        const nvfp4moe_params & P, const int lane, int & e_out, uint32_t & f0_out, uint32_t & f1_out) {
    const int ns  = P.T * NK_NUSED;
    const int own = (int) blockIdx.y;
    const int e   = P.ids[(own >> 3) * P.ids_s1 + (own & 7)];
    e_out = e;
    if (e < 0) {
        return 0;
    }
    const int id0 = (lane < ns)      ? P.ids[(lane >> 3) * P.ids_s1 + (lane & 7)]               : -1;
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
    f0_out = f0;
    f1_out = f1;
    return min(8, __popc(f0) + __popc(f1));
}

// Warp-uniform unpack of up to 8 active slots from (f0, f1) into lane k (0..7) using 1-cycle __ffsll;
// replaces slow sm_86 __fns software emulation and needs zero SMEM or CTA barriers.
static __device__ __forceinline__ int nk_unpack_slots_lane(const uint32_t f0, const uint32_t f1, const int lane) {
    uint64_t m = ((uint64_t) f1 << 32) | (uint64_t) f0;
    int my_slot = -1;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int bit = __ffsll((long long) m) - 1;
        if (lane == k) {
            my_slot = bit;
        }
        m &= m - 1ull;
    }
    return my_slot;
}

// Accumulate 16 rows x up to 8 columns for one staged K-chunk (N_BLK block_nvfp4 blocks) into acc[0..3].
// xq_qs_gt is pre-offset by + t * 16 in the prologue.
template <int SMEM_ROW_BYTES, int N_BLK>
static __device__ __forceinline__ void nk_chunk_dot(
        const char * __restrict__ stage_s,
        const int skew,
        const int kb_base,
        const int8_t * __restrict__ xq_qs_gt,
        const float  * __restrict__ xq_d_c0,
        const float  * __restrict__ xq_d_c1,
        const bool has_g, const bool has_c0, const bool has_c1,
        const int lane,
        float (&acc)[4]) {
    const int g = lane >> 2;
    const int t = lane & 3;

    const char * row0 = stage_s + g * SMEM_ROW_BYTES + skew;
    const char * row8 = stage_s + (g + 8) * SMEM_ROW_BYTES + skew;

#pragma unroll 1
    for (int bi = 0; bi < N_BLK; ++bi) {
        const int kb = kb_base + bi;

        const uint4  bv  = has_g  ? __ldg((const uint4 *)  (xq_qs_gt + kb * 64)) : make_uint4(0u, 0u, 0u, 0u);
        const float2 dx0 = has_c0 ? __ldg((const float2 *) (xq_d_c0 + 2 * kb))   : make_float2(0.0f, 0.0f);
        const float2 dx1 = has_c1 ? __ldg((const float2 *) (xq_d_c1 + 2 * kb))   : make_float2(0.0f, 0.0f);

        const char * b0 = row0 + bi * 36;
        const char * b8 = row8 + bi * 36;
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
// Grid: (F/32 = 36, T*8), 64 threads (2 warps), 5 CTAs/SM = 10 warps/SM on sm_86.
template <bool L2_EF>
static __global__ void __launch_bounds__(NK_NW_GU * 32, 5) k_nk_nvfp4_gu(const nvfp4moe_params P) {
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    int e;
    uint32_t f0_m, f1_m;
    const int N = nk_route_ballot(P, lane, e, f0_m, f1_m);
    if (N == 0) {
        return; // CTA-uniform, before any barrier
    }

    extern __shared__ __align__(16) char smem[];
    char  * warp_ring = smem + warp * NK_GU_WRING;
    float * act_s     = (float *) (smem + NK_NW_GU * NK_GU_WRING); // [8][32]

    const uint32_t warp_ring_s = (uint32_t) __cvta_generic_to_shared(warp_ring);
    const uint64_t policy      = L2_EF ? nk_l2_evict_first() : 0ull;

    const int feat_warp = (int) blockIdx.x * 32 + warp * 16;
    const char * wg = P.w_gu + (int64_t) e * P.gu_nb2 + (int64_t) feat_warp * NK_ROW_GU;
    const char * wu = wg + (int64_t) NK_F * NK_ROW_GU;

    // Precompute per-lane source byte offsets for the 144 x 16B copy of one 16-row x 144 B stage.
    int src_off[5];
#pragma unroll
    for (int it = 0; it < 5; ++it) {
        const int c = it * 32 + lane;
        if (c < 144) {
            const int r   = c / 9;
            const int k16 = c - r * 9;
            src_off[it] = r * NK_ROW_GU + k16 * 16;
        } else {
            src_off[it] = 0;
        }
    }

    auto issue_stage = [&](const char * base, const int buf) {
        const uint32_t dst_stage = warp_ring_s + (uint32_t) (buf * NK_GU_STAGE);
#pragma unroll
        for (int it = 0; it < 4; ++it) {
            nk_cp16<L2_EF>(dst_stage + (uint32_t) ((it * 32 + lane) * 16), base + src_off[it], policy);
        }
        if (lane < 16) {
            nk_cp16<L2_EF>(dst_stage + (uint32_t) ((128 + lane) * 16), base + src_off[4], policy);
        }
        nk_commit();
    };

    // Prologue: immediately prime 3 stages in flight (6,912 B/warp) and resolve routing pointers under DRAM latency.
    issue_stage(wg + 0 * NK_GU_CBYTES, 0);
    issue_stage(wg + 1 * NK_GU_CBYTES, 1);
    issue_stage(wg + 2 * NK_GU_CBYTES, 2);

    const int g = lane >> 2;
    const int t = lane & 3;
    const int my_slot = nk_unpack_slots_lane(f0_m, f1_m, lane);
    const int slot_g  = __shfl_sync(0xffffffffu, my_slot, g);
    const int slot_c0 = __shfl_sync(0xffffffffu, my_slot, 2 * t);
    const int slot_c1 = __shfl_sync(0xffffffffu, my_slot, 2 * t + 1);
    const bool has_g  = slot_g  >= 0;
    const bool has_c0 = slot_c0 >= 0;
    const bool has_c1 = slot_c1 >= 0;

    const int8_t * xq_qs = (const int8_t *) P.xq;
    const float  * xq_d  = (const float *) (xq_qs + (int64_t) P.T * NK_K);

    const int8_t * xq_qs_gt = has_g  ? xq_qs + (int64_t) (slot_g  >> 3) * NK_K + t * 16 : nullptr;
    const float  * xq_d_c0  = has_c0 ? xq_d  + (int64_t) (slot_c0 >> 3) * NK_NB32_K     : nullptr;
    const float  * xq_d_c1  = has_c1 ? xq_d  + (int64_t) (slot_c1 >> 3) * NK_NB32_K     : nullptr;

    const float scale = P.s_gu ? P.s_gu[e] : 1.0f;

    float acc_g[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float acc_u[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    // Gate pass (s = 0..8); steps 6..8 seamlessly prefetch up chunks 0..2 so gate->up has zero pipeline bubble.
#pragma unroll 1
    for (int s = 0; s < NK_GU_CHUNKS; ++s) {
        nk_wait<2>();
        const int ns = s + 3;
        const char * next_base = (ns < NK_GU_CHUNKS)
            ? (wg + ns * NK_GU_CBYTES)
            : (wu + (ns - NK_GU_CHUNKS) * NK_GU_CBYTES);
        issue_stage(next_base, ns & 3);
        nk_chunk_dot<NK_GU_CBYTES, NK_GU_CBLK>(
            warp_ring + (s & 3) * NK_GU_STAGE, 0, s * NK_GU_CBLK,
            xq_qs_gt, xq_d_c0, xq_d_c1, has_g, has_c0, has_c1, lane, acc_g);
    }

    // Up pass (s = 9..17); explicit wait drain on the final 2 steps with zero empty commit groups.
#pragma unroll 1
    for (int k = 0; k < NK_GU_CHUNKS; ++k) {
        const int s = NK_GU_CHUNKS + k;
        if (k < NK_GU_CHUNKS - 2) {
            nk_wait<2>();
        } else if (k == NK_GU_CHUNKS - 2) {
            nk_wait<1>();
        } else {
            nk_wait<0>();
        }
        if (k + 3 < NK_GU_CHUNKS) {
            issue_stage(wu + (k + 3) * NK_GU_CBYTES, (s + 3) & 3);
        }
        nk_chunk_dot<NK_GU_CBYTES, NK_GU_CBLK>(
            warp_ring + (s & 3) * NK_GU_STAGE, 0, k * NK_GU_CBLK,
            xq_qs_gt, xq_d_c0, xq_d_c1, has_g, has_c0, has_c1, lane, acc_u);
    }

    // Epilogue: apply scale + SITU directly in registers, stage [N][32] into SMEM, single CTA barrier, quantize P.act.
    const int foff = warp * 16;
    if (has_c0) {
        float g0 = acc_g[0], u0 = acc_u[0];
        float g2 = acc_g[2], u2 = acc_u[2];
        if (P.s_gu) {
            g0 *= scale;
            u0 *= scale;
            g2 *= scale;
            u2 *= scale;
        }
        act_s[(2 * t) * 32 + foff + g]     = nk_situ(g0, u0, P.situ);
        act_s[(2 * t) * 32 + foff + g + 8] = nk_situ(g2, u2, P.situ);
    }
    if (has_c1) {
        float g1 = acc_g[1], u1 = acc_u[1];
        float g3 = acc_g[3], u3 = acc_u[3];
        if (P.s_gu) {
            g1 *= scale;
            u1 *= scale;
            g3 *= scale;
            u3 *= scale;
        }
        act_s[(2 * t + 1) * 32 + foff + g]     = nk_situ(g1, u1, P.situ);
        act_s[(2 * t + 1) * 32 + foff + g + 8] = nk_situ(g3, u3, P.situ);
    }
    __syncthreads();

    int8_t * act_qs = (int8_t *) P.act;
    float  * act_d  = (float *) (act_qs + (int64_t) P.T * NK_NUSED * NK_F);
    for (int c = warp; c < N; c += NK_NW_GU) {
        const int slot = __shfl_sync(0xffffffffu, my_slot, c);
        nk_quant_store_32(
            act_s[c * 32 + lane],
            act_qs + (int64_t) slot * NK_F,
            act_d  + (int64_t) slot * NK_NB32_F,
            (int) blockIdx.x, lane);
    }
}

// down + scale -> P.e_out.
// Grid: (K/64 = 36, T*8), 64 threads (2 warps), 4 CTAs/SM = 8 warps/SM on sm_86.
// Zero __syncthreads(): each warp independently streams 2 slabs x 3 K-chunks through its private D=3 ring.
template <bool L2_EF>
static __global__ void __launch_bounds__(NK_NW_DN * 32, 4) k_nk_nvfp4_dn(const nvfp4moe_params P) {
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    int e;
    uint32_t f0_m, f1_m;
    const int N = nk_route_ballot(P, lane, e, f0_m, f1_m);
    if (N == 0) {
        return; // CTA-uniform
    }

    extern __shared__ __align__(16) char smem[];
    char * warp_ring = smem + warp * NK_DN_WRING;

    const uint32_t warp_ring_s = (uint32_t) __cvta_generic_to_shared(warp_ring);
    const uint64_t policy      = L2_EF ? nk_l2_evict_first() : 0ull;

    const int row_warp = (int) blockIdx.x * 64 + warp * 32;
    const char * wd = P.w_dn + (int64_t) e * P.dn_nb2 + (int64_t) row_warp * NK_ROW_DN;

    // Precompute per-lane offsets for the 224 x 16B = 7 unpredicated cp.async.cg transfers per 16-row stage.
    // Row r, chunk k starts at r*648 + k*216; subtracting skew = ((r+k)&1)*8 aligns down to 16B with 0 B OOB.
    int      src_even[7];
    uint32_t dst_off[7];
    uint32_t r_odd_mask = 0u;
#pragma unroll
    for (int it = 0; it < 7; ++it) {
        const int c   = it * 32 + lane;
        const int r   = c / 14;
        const int j16 = c - r * 14;
        src_even[it]  = r * NK_ROW_DN - (r & 1) * 8 + j16 * 16;
        dst_off[it]   = (uint32_t) (r * NK_DN_SROW + j16 * 16);
        r_odd_mask   |= ((uint32_t) (r & 1)) << it;
    }

    auto issue_stage = [&](const int sl, const int k) {
        const char * slab_base = wd + sl * NK_SLAB_DN;
        const uint32_t dst_stage = warp_ring_s + (uint32_t) (k * NK_DN_STAGE);
        const int k_add_even = k * NK_DN_CBYTES - (k & 1) * 8;
        const int k_add_odd  = k * NK_DN_CBYTES + (k & 1) * 8;
#pragma unroll
        for (int it = 0; it < 7; ++it) {
            const int k_add = (r_odd_mask & (1u << it)) ? k_add_odd : k_add_even;
            nk_cp16<L2_EF>(dst_stage + dst_off[it], slab_base + src_even[it] + k_add, policy);
        }
        nk_commit();
    };

    // Prologue: immediately prime 2 stages in flight (7,168 B payload / 7,680 B SMEM per warp).
    issue_stage(0, 0);
    issue_stage(0, 1);

    const int g = lane >> 2;
    const int t = lane & 3;
    const int my_slot = nk_unpack_slots_lane(f0_m, f1_m, lane);
    const int slot_g  = __shfl_sync(0xffffffffu, my_slot, g);
    const int slot_c0 = __shfl_sync(0xffffffffu, my_slot, 2 * t);
    const int slot_c1 = __shfl_sync(0xffffffffu, my_slot, 2 * t + 1);
    const bool has_g  = slot_g  >= 0;
    const bool has_c0 = slot_c0 >= 0;
    const bool has_c1 = slot_c1 >= 0;

    const int8_t * act_qs = (const int8_t *) P.act;
    const float  * act_d  = (const float *) (act_qs + (int64_t) P.T * NK_NUSED * NK_F);

    const int8_t * act_qs_gt = has_g  ? act_qs + (int64_t) slot_g  * NK_F + t * 16 : nullptr;
    const float  * act_d_c0  = has_c0 ? act_d  + (int64_t) slot_c0 * NK_NB32_F     : nullptr;
    const float  * act_d_c1  = has_c1 ? act_d  + (int64_t) slot_c1 * NK_NB32_F     : nullptr;

    const float scale = P.s_dn ? P.s_dn[e] : 1.0f;

#pragma unroll 1
    for (int sl = 0; sl < 2; ++sl) {
        float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
#pragma unroll 1
        for (int k = 0; k < NK_DN_CHUNKS; ++k) {
            const int s = sl * NK_DN_CHUNKS + k;
            if (s < 5) {
                nk_wait<1>();
            } else {
                nk_wait<0>();
            }
            if (s + 2 < 6) {
                const int ns = s + 2;
                issue_stage(ns / NK_DN_CHUNKS, ns % NK_DN_CHUNKS);
            }
            const int skew = ((g + k) & 1) * 8;
            nk_chunk_dot<NK_DN_SROW, NK_DN_CBLK>(
                warp_ring + k * NK_DN_STAGE, skew, k * NK_DN_CBLK,
                act_qs_gt, act_d_c0, act_d_c1, has_g, has_c0, has_c1, lane, acc);
        }

        const int row_slab = row_warp + sl * 16;
        if (has_c0) {
            float v0 = acc[0];
            float v2 = acc[2];
            if (P.s_dn) {
                v0 *= scale;
                v2 *= scale;
            }
            P.e_out[(int64_t) slot_c0 * NK_K + row_slab + g]     = v0;
            P.e_out[(int64_t) slot_c0 * NK_K + row_slab + g + 8] = v2;
        }
        if (has_c1) {
            float v1 = acc[1];
            float v3 = acc[3];
            if (P.s_dn) {
                v1 *= scale;
                v3 *= scale;
            }
            P.e_out[(int64_t) slot_c1 * NK_K + row_slab + g]     = v1;
            P.e_out[(int64_t) slot_c1 * NK_K + row_slab + g + 8] = v3;
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

    k_nk_nvfp4_dn<L2_EF><<<dim3(NK_K / 64, P.T * NK_NUSED), NK_NW_DN * 32, NK_SMEM_DN, stream>>>(P);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace nk_nvfp4_tc

bool ggml_cuda_nk_nvfp4_tc_accepts(const nvfp4moe_params & P) {
    using namespace nk_nvfp4_tc;
    if (!ggml_cuda_nk_nvfp4_tc_enabled()) {
        return false;
    }
    const int dev = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[dev].cc;
    return GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_AMPERE && cc < GGML_CUDA_CC_HOPPER &&
        (P.T >= 2 || (P.T == 1 && !P.write_sum)) && P.T <= 8 && P.xq != nullptr && !P.f32act && P.n_used == NK_NUSED &&
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
        GGML_LOG_INFO("[NK_NVFP4_TC] active (m16n8k16 v2 per-warp D=4/3, T=1..8, l2_evict_first=%d, smem gu=%zu dn=%zu)\n",
            l2_ef ? 1 : 0, nk_nvfp4_tc::NK_SMEM_GU, nk_nvfp4_tc::NK_SMEM_DN);
    }
    const int64_t n = g_nk_nvfp4_tc_launches.fetch_add(1, std::memory_order_relaxed) + 1;
    if (nk_nvfp4_tc_trace()) {
        GGML_LOG_INFO("[NK_NVFP4_TC] launch #%lld T=%d write_sum=%d\n", (long long) n, P.T, P.write_sum);
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
