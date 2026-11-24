// NEONKERN: sm_86 cliff-free int8 tensor-core Q8_0 MoE decode/verify kernels (T = 1..8).
//
// Design:
// - Weights stay in native ggml block_q8_0 layout (34 bytes per 32 weights: 1 fp16 scale + 32 signed int8 weights).
// - Each 32-element block maps 1-to-1 to one full-rate mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 with weights as A
//   (16 rows x 32 cols) and up to 8 grouped member slots as B (32 cols x 8 cols), so T = 1..8 run cliff-free and every
//   unique expert is read once from DRAM per visit.
// - gate_up streams 16-row slabs in 3 K-chunks of 24 blocks (816 B/row = 13,056 B/stage) through a 3-stage cp.async.cg
//   ring with L2::evict_first and wait_group 1 (2 groups always in flight), 100% bank-conflict-free SMEM (204 words/row),
//   2 CTAs/SM = 24 warps/SM, fusing optional scale + SITU + split-SOA act quantization.
// - down streams 4 contiguous 16-row slabs (64 rows/CTA, 19,584 B/slab = 153 x 128B lines) through a 2-stage cp.async.cg
//   ring with L2::evict_first, 2 CTAs/SM = 24 warps/SM, fusing optional scale -> P.e_out.
// - Activations (P.xq, P.act) use a private split-SOA layout inside the existing act scratch (zero extra VRAM) packed for
//   single LDG.64 (uint2) loads per 32-element block from L1 cache.

#include "nk-q8-moe-tc.cuh"

#include <atomic>
#include <cstdlib>

bool ggml_cuda_nk_q8_0_tc_enabled() {
    static const bool on = [] {
        // default ON (HOME 100K ABBA -2.77% ms/round 6/6; E21 KLD |delta| <= 1e-6); =0 disables
        const char * e = getenv("GGML_CUDA_NK_Q8_0_TC");
        return !(e != nullptr && *e && atoi(e) == 0);
    }();
    return on;
}

static bool nk_q8_0_tc_l2_ef() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_NK_Q8_0_TC_L2");
        return e == nullptr || atoi(e) != 0;
    }();
    return on;
}

static bool nk_q8_0_tc_trace() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_NK_TRACE");
        return e != nullptr && atoi(e) != 0;
    }();
    return on;
}

static std::atomic<int64_t> g_nk_q8_0_tc_launches{0};

int64_t ggml_cuda_nk_q8_0_tc_count() {
    return g_nk_q8_0_tc_launches.load(std::memory_order_relaxed);
}

bool ggml_cuda_nk_q8_0_tc_supported(int cc, int64_t K, int64_t F, int64_t n_used) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(cc); GGML_UNUSED(K); GGML_UNUSED(F); GGML_UNUSED(n_used);
    return false;
#else
    return GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_AMPERE && cc < GGML_CUDA_CC_HOPPER &&
           K == 2304 && F == 1152 && n_used == 8;
#endif
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

namespace nk_q8_0_tc {

// Match k_ink_situ / nvfp4moe_situ per-element expressions without pulling in nvfp4moe-kern.cuh kernels.
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

constexpr int NK_K          = 2304;
constexpr int NK_F          = 1152;
constexpr int NK_NUSED      = 8;
constexpr int NK_NB32_K     = NK_K / 32;            // 72
constexpr int NK_NB32_F     = NK_F / 32;            // 36
constexpr int NK_BLK_BYTES  = (int) sizeof(block_q8_0); // 34
constexpr int NK_ROW_GU     = NK_NB32_K * NK_BLK_BYTES; // 2448 bytes (153 x 16B)
constexpr int NK_ROW_DN     = NK_NB32_F * NK_BLK_BYTES; // 1224 bytes (153 x 8B)

// gate_up 3-stage K-chunking: 72 blocks = 3 chunks x 24 blocks (816 B/row = 51 x 16B; 204 words/row = 12 mod 32 -> 0 bank conflicts).
constexpr int NK_KC_BLKS    = 24;
constexpr int NK_KC_CHUNKS  = NK_NB32_K / NK_KC_BLKS;   // 3
constexpr int NK_KC_ROW     = NK_KC_BLKS * NK_BLK_BYTES; // 816 bytes
constexpr int NK_KC_ROW16   = NK_KC_ROW / 16;            // 51
constexpr int NK_STAGE_GU   = 16 * NK_KC_ROW;            // 13056 bytes (102 x 128B)
constexpr int NK_STAGES_GU  = 3;

// down 2-stage full-row slabs: 16 rows x 1224 B = 19584 B (153 x 128B lines).
constexpr int NK_SLAB_DN    = 16 * NK_ROW_DN;            // 19584 bytes

constexpr int NK_NW_GU      = 12; // 24 blocks / 12 warps = 2 blocks (1 even+odd pair) per warp per stage; 2 CTAs/SM = 24 warps/SM
constexpr int NK_NW_DN      = 12; // 36 blocks / 12 warps = 3 blocks/warp; 2 CTAs/SM = 24 warps/SM

// Dynamic shared memory sizes (multiples of 128 bytes):
// gu: 3 * 13056 (ring) + 12 * 128 * 4 (red) + 256 * 4 (act_s) + 32 (s_slots) = 46368 -> pad 46464 (2 CTAs/SM = 92928 <= 101376)
// dn: 2 * 19584 (ring) + 12 * 128 * 4 (red) + 32 (s_slots)                   = 45344 -> pad 45440 (2 CTAs/SM = 90880 <= 101376)
constexpr size_t NK_SMEM_GU = ((NK_STAGES_GU * NK_STAGE_GU + NK_NW_GU * 128 * sizeof(float) + 256 * sizeof(float) + 32 + 127) & ~size_t(127));
constexpr size_t NK_SMEM_DN = ((2 * NK_SLAB_DN + NK_NW_DN * 128 * sizeof(float) + 32 + 127) & ~size_t(127));

static_assert(NK_BLK_BYTES == 34, "unexpected block_q8_0 size");
static_assert(NK_SMEM_GU * 2 <= 101376, "NK_Q8_0_TC gu must fit 2 CTAs/SM on sm_86");
static_assert(NK_SMEM_DN * 2 <= 101376, "NK_Q8_0_TC dn must fit 2 CTAs/SM on sm_86");

// Exact signed-int-to-float via 1.5 * 2^23 binade bias (exact for |x| < 2^22; block_q8_0 max |sumi| = 32*128*127 = 520192).
constexpr int   NK_I2F_BIAS_I = 0x4B400000;
constexpr float NK_I2F_BIAS_F = 12582912.0f;

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

static __device__ __forceinline__ void nk_mma_k32_bias(
        int & c0, int & c1, int & c2, int & c3,
        const uint32_t a0, const uint32_t a1, const uint32_t a2, const uint32_t a3,
        const uint32_t b0, const uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const int bias = NK_I2F_BIAS_I;
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %10, %10, %10};"
        : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(bias));
#else
    GGML_UNUSED_VARS(a0, a1, a2, a3, b0, b1);
    c0 = c1 = c2 = c3 = 0;
    __trap();
#endif
}

// Quantize one 32-element block (lane 0..31 holds xi) into split-SOA qs + d layout:
// - qs stores each 32-element block as [t = 0..3][half = 0..1][4B] so lane (g, t) reads (b0, b1) in one uint2 (LDG.64).
// - d stores float scale rounded through fp16 to match block_q8_1's __low2float(ds) bit-for-bit.
static __device__ __forceinline__ void nk_q8_quant_store_32(
        const float xi, int8_t * qs_row, float * d_row, const int b32, const int lane) {
    float amax = fabsf(xi);
    amax = warp_reduce_max<32>(amax);
    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : (int8_t) roundf(xi / d);

    const int l7   = lane & 7;
    const int t    = l7 >> 1;
    const int half = l7 & 1;
    const int base = (half << 4) | (t << 2);

    const uint32_t b0 = (uint8_t) __shfl_sync(0xffffffffu, (int) q, base + 0);
    const uint32_t b1 = (uint8_t) __shfl_sync(0xffffffffu, (int) q, base + 1);
    const uint32_t b2 = (uint8_t) __shfl_sync(0xffffffffu, (int) q, base + 2);
    const uint32_t b3 = (uint8_t) __shfl_sync(0xffffffffu, (int) q, base + 3);

    if (lane < 8) {
        uint32_t * dst32 = (uint32_t *) (qs_row + b32 * 32);
        dst32[l7] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
    }
    if (lane == 0) {
        d_row[b32] = __half2float(__float2half_rn(d));
    }
}

// Quantize P.x [T][2304] into P.xq split-SOA (xq_qs [T][2304] int8, xq_d [T][72] float).
static __global__ void __launch_bounds__(256) k_nk_q8_0_xq(const nvfp4moe_params P) {
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
    nk_q8_quant_store_32(xi, xq_qs + (int64_t) tok * NK_K, xq_d + (int64_t) tok * NK_NB32_K, b32, lane);
}

// Warp-level routing ballot over up to 64 slots (T <= 8, n_used == 8).
// Returns N (1..8) slots handled by this CTA (0 if not the chunk leader or invalid expert id).
static __device__ __forceinline__ int nk_route_ballot(
        const nvfp4moe_params & P, const int lane, int & e_out, uint32_t & f0_out, uint32_t & f1_out, int & n0_out) {
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

// Accumulate one EVEN block_q8_0 (byte offset % 4 == 0; qs starts at +2, reconstructed via aligned LDS.32 + PRMT).
static __device__ __forceinline__ void nk_q8_blk_even(
        const char * __restrict__ b0,
        const char * __restrict__ b8,
        const uint2 bv,
        const float dx0, const float dx1,
        const int t, float (&acc)[4]) {
    const float dw0 = __half2float(*(const half *) b0);
    const float dw8 = __half2float(*(const half *) b8);

    const uint32_t u00 = *(const uint32_t *) (b0 + 4 * t);
    const uint32_t u01 = *(const uint32_t *) (b0 + 4 + 4 * t);
    const uint32_t v00 = *(const uint32_t *) (b0 + 16 + 4 * t);
    const uint32_t v01 = *(const uint32_t *) (b0 + 20 + 4 * t);
    const uint32_t a0  = __byte_perm(u00, u01, 0x5432);
    const uint32_t a2  = __byte_perm(v00, v01, 0x5432);

    const uint32_t u80 = *(const uint32_t *) (b8 + 4 * t);
    const uint32_t u81 = *(const uint32_t *) (b8 + 4 + 4 * t);
    const uint32_t v80 = *(const uint32_t *) (b8 + 16 + 4 * t);
    const uint32_t v81 = *(const uint32_t *) (b8 + 20 + 4 * t);
    const uint32_t a1  = __byte_perm(u80, u81, 0x5432);
    const uint32_t a3  = __byte_perm(v80, v81, 0x5432);

    int c0, c1, c2, c3;
    nk_mma_k32_bias(c0, c1, c2, c3, a0, a1, a2, a3, bv.x, bv.y);

    const float f0 = __fsub_rn(__int_as_float(c0), NK_I2F_BIAS_F);
    const float f1 = __fsub_rn(__int_as_float(c1), NK_I2F_BIAS_F);
    const float f2 = __fsub_rn(__int_as_float(c2), NK_I2F_BIAS_F);
    const float f3 = __fsub_rn(__int_as_float(c3), NK_I2F_BIAS_F);

    acc[0] = fmaf(dw0 * dx0, f0, acc[0]);
    acc[1] = fmaf(dw0 * dx1, f1, acc[1]);
    acc[2] = fmaf(dw8 * dx0, f2, acc[2]);
    acc[3] = fmaf(dw8 * dx1, f3, acc[3]);
}

// Accumulate one ODD block_q8_0 (byte offset % 4 == 2; qs starts at +2 -> 4-byte aligned direct LDS.32).
static __device__ __forceinline__ void nk_q8_blk_odd(
        const char * __restrict__ b0,
        const char * __restrict__ b8,
        const uint2 bv,
        const float dx0, const float dx1,
        const int t, float (&acc)[4]) {
    const float dw0 = __half2float(*(const half *) b0);
    const float dw8 = __half2float(*(const half *) b8);

    const uint32_t a0 = *(const uint32_t *) (b0 + 2 + 4 * t);
    const uint32_t a2 = *(const uint32_t *) (b0 + 18 + 4 * t);
    const uint32_t a1 = *(const uint32_t *) (b8 + 2 + 4 * t);
    const uint32_t a3 = *(const uint32_t *) (b8 + 18 + 4 * t);

    int c0, c1, c2, c3;
    nk_mma_k32_bias(c0, c1, c2, c3, a0, a1, a2, a3, bv.x, bv.y);

    const float f0 = __fsub_rn(__int_as_float(c0), NK_I2F_BIAS_F);
    const float f1 = __fsub_rn(__int_as_float(c1), NK_I2F_BIAS_F);
    const float f2 = __fsub_rn(__int_as_float(c2), NK_I2F_BIAS_F);
    const float f3 = __fsub_rn(__int_as_float(c3), NK_I2F_BIAS_F);

    acc[0] = fmaf(dw0 * dx0, f0, acc[0]);
    acc[1] = fmaf(dw0 * dx1, f1, acc[1]);
    acc[2] = fmaf(dw8 * dx0, f2, acc[2]);
    acc[3] = fmaf(dw8 * dx1, f3, acc[3]);
}

// gate_up + optional scale + SITU -> split-SOA P.act.
// Grid: (F/32 = 36, T*8), 384 threads (12 warps), 2 CTAs/SM = 24 warps/SM on sm_86.
// 3-stage cp.async.cg ring across all 12 stages (4 slabs x 3 K-chunks of 24 blocks) with wait_group 1.
template <bool L2_EF>
static __global__ void __launch_bounds__(NK_NW_GU * 32, 2) k_nk_q8_0_gu(const nvfp4moe_params P) {
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
    char  * ring    = smem;                                       // [3][13056]
    float * red     = (float *) (ring + NK_STAGES_GU * NK_STAGE_GU); // [12][8][16]
    float * act_s   = red + NK_NW_GU * 128;                       // [8][32]
    int   * s_slots = (int *) (act_s + 256);                      // [8]

    const uint32_t ring_s = (uint32_t) __cvta_generic_to_shared(ring);
    const uint64_t policy = L2_EF ? nk_l2_evict_first() : 0ull;

    const int feat0 = (int) blockIdx.x * 32;
    const char * wg = P.w_gu + (int64_t) e * P.gu_nb2 + (int64_t) feat0 * NK_ROW_GU;
    const char * wu = wg + (int64_t) NK_F * NK_ROW_GU;

    auto issue_stage = [&](const int st, const int buf) {
        const int slab = st / NK_KC_CHUNKS;
        const int kc   = st - slab * NK_KC_CHUNKS;
        const char * base = ((slab & 1) ? wu : wg) + (slab >= 2 ? 16 * NK_ROW_GU : 0) + kc * NK_KC_ROW;
        const uint32_t dst = ring_s + (uint32_t) (buf * NK_STAGE_GU);
#pragma unroll
        for (int c = tid; c < NK_STAGE_GU / 16; c += NK_NW_GU * 32) {
            const int r     = c / NK_KC_ROW16;
            const int col16 = c - r * NK_KC_ROW16;
            nk_cp16<L2_EF>(dst + (uint32_t) (c * 16), base + r * NK_ROW_GU + col16 * 16, policy);
        }
        nk_commit();
    };

    // Prologue: launch stages 0 and 1 immediately (2 cp.async groups in flight), scatter slots under latency.
    issue_stage(0, 0);
    issue_stage(1, 1);
    if (warp == 0) {
        nk_scatter_slots(s_slots, f0_m, f1_m, n0, lane);
    }
    const float scale = P.s_gu ? P.s_gu[e] : 1.0f;
    nk_wait<1>();
    __syncthreads();

    const int g = lane >> 2;
    const int t = lane & 3;
    const int tok_g  = (g < N)         ? (s_slots[g]         >> 3) : -1;
    const int tok_c0 = (2 * t < N)     ? (s_slots[2 * t]     >> 3) : -1;
    const int tok_c1 = (2 * t + 1 < N) ? (s_slots[2 * t + 1] >> 3) : -1;

    const int8_t * xq_qs = (const int8_t *) P.xq;
    const float  * xq_d  = (const float *) (xq_qs + (int64_t) P.T * NK_K);

    const int8_t * xq_qs_g = tok_g  >= 0 ? xq_qs + (int64_t) tok_g  * NK_K      : nullptr;
    const float  * xq_d_c0 = tok_c0 >= 0 ? xq_d  + (int64_t) tok_c0 * NK_NB32_K : nullptr;
    const float  * xq_d_c1 = tok_c1 >= 0 ? xq_d  + (int64_t) tok_c1 * NK_NB32_K : nullptr;

    const int off_e = (2 * warp) * NK_BLK_BYTES;
    const int off_o = off_e + NK_BLK_BYTES;

#pragma unroll 1
    for (int slab = 0; slab < 4; ++slab) {
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

#pragma unroll 1
        for (int kc = 0; kc < NK_KC_CHUNKS; ++kc) {
            const int st = slab * NK_KC_CHUNKS + kc;
            if (st + 2 < 4 * NK_KC_CHUNKS) {
                issue_stage(st + 2, (st + 2) % NK_STAGES_GU);
            }

            const char * stage_s = ring + (st % NK_STAGES_GU) * NK_STAGE_GU;
            const char * row0    = stage_s + g * NK_KC_ROW;
            const char * row8    = stage_s + (g + 8) * NK_KC_ROW;

            const int kb_even = kc * NK_KC_BLKS + 2 * warp;
            const float2 dx0  = xq_d_c0 ? __ldg((const float2 *) (xq_d_c0 + kb_even)) : make_float2(0.0f, 0.0f);
            const float2 dx1  = xq_d_c1 ? __ldg((const float2 *) (xq_d_c1 + kb_even)) : make_float2(0.0f, 0.0f);

            const uint2 bv_e = xq_qs_g ? __ldg((const uint2 *) (xq_qs_g + kb_even * 32 + t * 8))       : make_uint2(0u, 0u);
            const uint2 bv_o = xq_qs_g ? __ldg((const uint2 *) (xq_qs_g + (kb_even + 1) * 32 + t * 8)) : make_uint2(0u, 0u);

            nk_q8_blk_even(row0 + off_e, row8 + off_e, bv_e, dx0.x, dx1.x, t, acc);
            nk_q8_blk_odd (row0 + off_o, row8 + off_o, bv_o, dx0.y, dx1.y, t, acc);

            if (kc < NK_KC_CHUNKS - 1) {
                if (st + 2 < 4 * NK_KC_CHUNKS) {
                    nk_wait<1>();
                } else {
                    nk_wait<0>();
                }
                __syncthreads();
            }
        }

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
            const int foff = (slab >= 2 ? 16 : 0) + r;
            if ((slab & 1) == 0) {
                act_s[c * 32 + foff] = sum;
            } else {
                const float gv = act_s[c * 32 + foff];
                act_s[c * 32 + foff] = nk_situ(gv, sum, P.situ);
            }
        }
        if (slab < 3) {
            nk_wait<1>();
            __syncthreads();
        }
    }

    __syncthreads();
    if (warp < N) {
        const int slot = s_slots[warp];
        int8_t * act_qs = (int8_t *) P.act;
        float  * act_d  = (float *) (act_qs + (int64_t) P.T * NK_NUSED * NK_F);
        nk_q8_quant_store_32(
            act_s[warp * 32 + lane],
            act_qs + (int64_t) slot * NK_F,
            act_d  + (int64_t) slot * NK_NB32_F,
            (int) blockIdx.x, lane);
    }
}

// down + optional scale -> P.e_out.
// Grid: (K/64 = 36, T*8), 384 threads (12 warps), 2 CTAs/SM = 24 warps/SM on sm_86.
// 4 contiguous 16-row slabs per CTA (19,584 B/slab = 153 x 128B lines), 2-stage cp.async.cg ring.
template <bool L2_EF>
static __global__ void __launch_bounds__(NK_NW_DN * 32, 2) k_nk_q8_0_dn(const nvfp4moe_params P) {
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
    char  * ring    = smem;                                       // [2][19584]
    float * red     = (float *) (ring + 2 * NK_SLAB_DN);          // [12][8][16]
    int   * s_slots = (int *) (red + NK_NW_DN * 128);             // [8]

    const uint32_t ring_s = (uint32_t) __cvta_generic_to_shared(ring);
    const uint64_t policy = L2_EF ? nk_l2_evict_first() : 0ull;

    const int row_base = (int) blockIdx.x * 64;
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

    // Prologue: stream slab 0 immediately, scatter slot table under its latency.
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

    const int kb0 = 3 * warp;

#pragma unroll 1
    for (int s = 0; s < 4; ++s) {
        if (s + 1 < 4) {
            issue_slab(s + 1, (s + 1) & 1);
        }

        const char * slab_s = ring + (s & 1) * NK_SLAB_DN;
        const char * row0   = slab_s + g * NK_ROW_DN;
        const char * row8   = slab_s + (g + 8) * NK_ROW_DN;

        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 1
        for (int bi = 0; bi < 3; ++bi) {
            const int   kb  = kb0 + bi;
            const int   off = kb * NK_BLK_BYTES;
            const uint2 bv  = act_qs_g ? __ldg((const uint2 *) (act_qs_g + kb * 32 + t * 8)) : make_uint2(0u, 0u);
            const float dx0 = act_d_c0 ? __ldg(act_d_c0 + kb) : 0.0f;
            const float dx1 = act_d_c1 ? __ldg(act_d_c1 + kb) : 0.0f;
            if ((kb & 1) == 0) {
                nk_q8_blk_even(row0 + off, row8 + off, bv, dx0, dx1, t, acc);
            } else {
                nk_q8_blk_odd (row0 + off, row8 + off, bv, dx0, dx1, t, acc);
            }
        }

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
        if (s + 1 < 4) {
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
        CUDA_CHECK(cudaFuncSetAttribute(k_nk_q8_0_gu<L2_EF>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) NK_SMEM_GU));
        CUDA_CHECK(cudaFuncSetAttribute(k_nk_q8_0_gu<L2_EF>, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
        CUDA_CHECK(cudaFuncSetAttribute(k_nk_q8_0_dn<L2_EF>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) NK_SMEM_DN));
        CUDA_CHECK(cudaFuncSetAttribute(k_nk_q8_0_dn<L2_EF>, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
        init[dev] = true;
    }
    k_nk_q8_0_xq<<<(P.T * NK_NB32_K + 7) / 8, 256, 0, stream>>>(P);
    CUDA_CHECK(cudaGetLastError());

    k_nk_q8_0_gu<L2_EF><<<dim3(NK_F / 32, P.T * NK_NUSED), NK_NW_GU * 32, NK_SMEM_GU, stream>>>(P);
    CUDA_CHECK(cudaGetLastError());

    k_nk_q8_0_dn<L2_EF><<<dim3(NK_K / 64, P.T * NK_NUSED), NK_NW_DN * 32, NK_SMEM_DN, stream>>>(P);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace nk_q8_0_tc

bool ggml_cuda_nk_q8_0_tc_accepts(const nvfp4moe_params & P) {
    using namespace nk_q8_0_tc;
    if (!ggml_cuda_nk_q8_0_tc_enabled()) {
        return false;
    }
    const int dev = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[dev].cc;
    return ggml_cuda_nk_q8_0_tc_supported(cc, P.K, P.F, P.n_used) &&
        (P.T >= 2 || (P.T == 1 && !P.write_sum)) && P.T <= 8 && P.xq != nullptr && !P.f32act &&
        P.gu_nb1 == NK_ROW_GU && P.dn_nb1 == NK_ROW_DN &&
        P.gu_nb2 % 16 == 0 && P.dn_nb2 % 16 == 0 &&
        (uintptr_t) P.w_gu % 16 == 0 && (uintptr_t) P.w_dn % 16 == 0 &&
        (uintptr_t) P.act % 16 == 0 && (uintptr_t) P.xq % 16 == 0;
}

bool ggml_cuda_nk_q8_0_tc_launch(const nvfp4moe_params & P, cudaStream_t stream) {
    if (!ggml_cuda_nk_q8_0_tc_accepts(P)) {
        return false;
    }
    const bool l2_ef = nk_q8_0_tc_l2_ef();
    static std::atomic_flag logged = ATOMIC_FLAG_INIT;
    if (!logged.test_and_set()) {
        GGML_LOG_WARN("[NK_Q8_0_TC] active (m16n8k32 s8 TC, T=1..8, l2_evict_first=%d, smem gu=%zu dn=%zu)\n",
            l2_ef ? 1 : 0, nk_q8_0_tc::NK_SMEM_GU, nk_q8_0_tc::NK_SMEM_DN);
    }
    const int64_t n = g_nk_q8_0_tc_launches.fetch_add(1, std::memory_order_relaxed) + 1;
    if (nk_q8_0_tc_trace()) {
        GGML_LOG_INFO("[NK_Q8_0_TC] launch #%lld T=%d write_sum=%d\n", (long long) n, P.T, P.write_sum);
    }
    if (l2_ef) {
        nk_q8_0_tc::launch_pair<true>(P, stream);
    } else {
        nk_q8_0_tc::launch_pair<false>(P, stream);
    }
    return true;
}

#else

bool ggml_cuda_nk_q8_0_tc_accepts(const nvfp4moe_params &) {
    return false;
}

bool ggml_cuda_nk_q8_0_tc_launch(const nvfp4moe_params &, cudaStream_t) {
    return false;
}

#endif
