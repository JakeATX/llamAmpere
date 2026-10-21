// [I8QK] int8-QK prefill flash attention (SageAttention-style). See fattn-i8qk.cuh for the route description.

#include "fattn-i8qk.cuh"
#include "convert.cuh"
#include "turbo-quant.cuh"

#include <cmath>
#include <cstdlib>
#include <cstring>

#define I8QK_BM 128 // query rows per CTA: 8 warps x 16 rows (one query head)
#define I8QK_BN  64 // keys per KV tile
#define I8QK_NW   8 // warps per CTA

static constexpr int i8qk_nthreads = I8QK_NW*WARP_SIZE;

bool ggml_cuda_fattn_i8qk_enabled() {
    static const bool v = [] {
        const char * e = getenv("GGML_CUDA_FA_I8QK");
        return e == nullptr || (e[0] == '1' && e[1] == '\0');
    }();
    return v;
}

// [SPQ] a sparse-attention (QSA) prompt at long context: op param 4 carries the per-query selected-cell count, and once
// the f16 MMA route would gather only the selected cells (fattn.cu shall_use_sparse, the (256, 256, 8, 8) case the SJ-KVaRN
// prefill launches for GQA > 4) the int8-QK kernel, which skips only fully masked 128x64 tiles, does several times more
// work. GGML_CUDA_FA_SPARSE_FIRST=1 lets the gather take those calls (shorter contexts and dense models keep int8 QK).
bool ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(const int cc, const ggml_tensor * dst, const int ncols1, const int ncols2);

static bool ggml_cuda_fattn_i8qk_sparse_first() {
    static const bool v = [] {
        // default on (owner ruling 2026-10-21); GGML_CUDA_FA_SPARSE_FIRST=0 turns it off
        const char * e = getenv("GGML_CUDA_FA_SPARSE_FIRST");
        return !(e != nullptr && e[0] == '0' && e[1] == '\0');
    }();
    return v;
}

static bool ggml_cuda_fattn_i8qk_yields_to_sparse(int cc, const ggml_tensor * dst) {
    if (!ggml_cuda_fattn_i8qk_sparse_first() || ggml_get_op_params_i32(dst, 4) <= 0) {
        return false;
    }
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    if (Q->ne[0] != 256 || V->ne[0] != 256 || K->ne[2] < 1 || Q->ne[2] % K->ne[2] != 0 || Q->ne[2]/K->ne[2] <= 4) {
        return false;
    }
    return ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(cc, dst, 8, 8);
}

static int ggml_cuda_fattn_i8qk_min_q() {
    static const int v = [] {
        const char * e = getenv("GGML_CUDA_FA_I8QK_MIN_Q");
        const int n = e ? atoi(e) : 64;
        return n >= 9 ? n : 64; // decode / MTP verify widths 1-8 never take the route
    }();
    return v;
}

// [I8QK v2] route variants (each read once; PV16 and KCB default on, =0 opts out; other variants stay opt-in):
//   GGML_CUDA_FA_I8QK_PV16=1  (#151) P.V on f16 MMA with an f16 accumulator (the f16 route's own V.KQ numerics on Ampere,
//                             twice the f32-accumulate MMA rate on GA10x), O rescaled in half2;
//   GGML_CUDA_FA_I8QK_KCB=1   (#152) codebook-direct int8 K for tq5_0 / tq6_0 K caches: K is decoded from the codes
//                             inside the mean pass and the quant pass (no f16 K staging, no f16 round trip), smoothed by
//                             the per-channel mean, one f32 scale per (token, 128-value block); with KBLK=0 no mean pass
//                             (pure codebook map round(c*127/c_b), c_b = largest centroid used in the block);
//   GGML_CUDA_FA_I8QK_QH=1    (#154 arm) one Q scale per (query row, 128-dim half) instead of per row;
//   GGML_CUDA_FA_I8QK_Q2L=1   (#154 arm) two-limb Q: q = s*(q1 + q2/254), two int8 MMAs per K fragment;
//   GGML_CUDA_FA_I8QK_KBLK=n  (#155 numerics) K smoothing offset: unset = one mean over all n_kv tokens (v1); 0 = no
//                             smoothing; n (multiple of 64) = a mean per n-token block, frozen per block, with the exact
//                             q.m_b correction added per (query, block) in f32 (softmax is shift-invariant only within
//                             a block). This is the accuracy arm of a persistent int8 K; the K prep still runs per op.
// KCB and QH make QK^T accumulate per 128-dim half (int32 per half, scaled once per half). Q2L and KBLK=n need the
// PV16 kernel registers and switch PV16 on.
struct i8qk_modes {
    bool pv16;
    bool kcb;
    bool qh;
    bool q2l;
    int  kblk; // -1 = global mean (v1), 0 = none, n = block-frozen mean per n tokens
};

static bool i8qk_env_on(const char * name, const bool default_on = false) {
    const char * e = getenv(name);
    return e == nullptr ? default_on : e[0] == '1' && e[1] == '\0';
}

static const i8qk_modes & ggml_cuda_fattn_i8qk_modes() {
    static const i8qk_modes m = [] {
        i8qk_modes r = {
            i8qk_env_on("GGML_CUDA_FA_I8QK_PV16", true),
            i8qk_env_on("GGML_CUDA_FA_I8QK_KCB", true),
            i8qk_env_on("GGML_CUDA_FA_I8QK_QH"),
            i8qk_env_on("GGML_CUDA_FA_I8QK_Q2L"),
            -1,
        };
        if (const char * e = getenv("GGML_CUDA_FA_I8QK_KBLK")) {
            const int n = atoi(e);
            if (n == 0 && e[0] == '0' && e[1] == '\0') {
                r.kblk = 0;
            } else if (n >= I8QK_BN && n % I8QK_BN == 0) {
                r.kblk = n;
            } else {
                GGML_LOG_WARN("%s: GGML_CUDA_FA_I8QK_KBLK=%s ignored (0 or a multiple of %d)\n", __func__, e, I8QK_BN);
            }
        }
        if (r.q2l || r.kblk > 0) {
            r.pv16 = true;
        }
        if (r.pv16 || r.kcb || r.qh || r.q2l || r.kblk >= 0) {
            GGML_LOG_INFO("%s: int8-QK v2 modes: pv16=%d kcb=%d qh=%d q2l=%d kblk=%d\n", __func__,
                          r.pv16, r.kcb, r.qh, r.q2l, r.kblk);
        }
        return r;
    }();
    return m;
}

// Quantized caches only: the int8 K and the f16 K staging / f16 V live in the f16 K/V reservation the allocator makes for
// the f16 route of a quantized pair. An f16 K or V has no such reservation (the f16 route reads it in place), so the int8
// route would need new VRAM for it: f16 caches keep the f16 route.
static bool i8qk_kv_type_ok(const ggml_tensor * t, const bool f16_ok = false) {
    switch (t->type) {
        case GGML_TYPE_F16:
            // [I8SJ] only for callers that pass their own int8 K workspace (the SJ-KVaRN prefill expand scratch):
            // read in place, 16-byte rows for the cp.async V loads
            return f16_ok && t->nb[0] == sizeof(half) && t->nb[1] % 16 == 0 && t->nb[2] % 16 == 0 &&
                   (uintptr_t) t->data % 16 == 0;
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_TURBO2_0:
        case GGML_TYPE_TURBO3_0:
        case GGML_TYPE_TURBO4_0:
        case GGML_TYPE_TQ5_0:
        case GGML_TYPE_TQ6_0: {
            // strided f16 converter (ggml_get_to_fp16_nc_cuda) on a head slice, block-contiguous rows
            const size_t ts = ggml_type_size(t->type);
            return ggml_get_to_fp16_nc_cuda(t->type) != nullptr && t->ne[0] % ggml_blck_size(t->type) == 0 &&
                   t->nb[0] == ts && t->nb[1] % ts == 0 && t->nb[2] % ts == 0 && t->nb[3] % ts == 0;
        }
        default:
            return false;
    }
}

static bool i8qk_applies_impl(int cc, const ggml_tensor * dst, const bool f16_kv) {
    if (!ggml_cuda_fattn_i8qk_enabled() || ggml_cuda_fattn_i8qk_yields_to_sparse(cc, dst)) {
        return false;
    }
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(cc); GGML_UNUSED(dst);
    return false;
#else
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_AMPERE) {
        return false;
    }
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    if (Q == nullptr || K == nullptr || V == nullptr || dst->src[4] != nullptr) {
        return false; // sinks are not implemented
    }
    for (int i = 5; i < GGML_MAX_SRC; ++i) {
        if (dst->src[i] != nullptr) {
            return false;
        }
    }
    const int64_t D = Q->ne[0];
    if ((D != 128 && D != 256) || K->ne[0] != D || V->ne[0] != D) {
        return false;
    }
    if (Q->type != GGML_TYPE_F32 || Q->nb[0] != sizeof(float) || Q->nb[1] % 16 != 0 || Q->nb[2] % 16 != 0 ||
            (uintptr_t) Q->data % 16 != 0) {
        return false;
    }
    // grid limits: query tiles and query heads go to gridDim.y of the tile-flag / attention launches
    if (Q->ne[1] < ggml_cuda_fattn_i8qk_min_q() || Q->ne[1] > int64_t(65535)*I8QK_BM || Q->ne[2] > 65535 ||
            Q->ne[3] != 1 || K->ne[3] != 1 || V->ne[3] != 1) {
        return false;
    }
    if (K->ne[2] < 1 || K->ne[2] != V->ne[2] || Q->ne[2] % K->ne[2] != 0) {
        return false;
    }
    if (K->ne[1] < I8QK_BN || K->ne[1] % I8QK_BN != 0 || K->ne[1] != V->ne[1] || K->ne[1] > (int64_t(1) << 26)) {
        return false;
    }
    if (K == V || V->view_src == K || K->view_src == V ||
            (V->view_src != nullptr && V->view_src == K->view_src && V->view_offs == K->view_offs)) {
        return false;
    }
    if (f16_kv ? (K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16 || !i8qk_kv_type_ok(K, true) || !i8qk_kv_type_ok(V, true))
               : (!i8qk_kv_type_ok(K) || !i8qk_kv_type_ok(V))) {
        return false;
    }
    if (mask != nullptr && (mask->type != GGML_TYPE_F16 || mask->nb[0] != sizeof(half) || mask->ne[0] < K->ne[1] || mask->ne[1] < Q->ne[1] ||
            mask->ne[2] != 1 || mask->ne[3] != 1 || mask->nb[1] % 4 != 0 || (uintptr_t) mask->data % 4 != 0)) {
        return false;
    }
    float max_bias = 0.0f;
    float softcap  = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || softcap != 0.0f) {
        return false;
    }
    if (dst->type != GGML_TYPE_F32 || dst->ne[0] != D || dst->ne[1] != Q->ne[2] || dst->ne[2] != Q->ne[1] ||
            dst->ne[3] != 1 || !ggml_is_contiguous(dst)) {
        return false;
    }
    return true;
#endif
}

bool ggml_cuda_fattn_i8qk_applies(int cc, const ggml_tensor * dst) {
    return i8qk_applies_impl(cc, dst, false);
}

bool ggml_cuda_fattn_i8qk_applies_f16kv(int cc, const ggml_tensor * dst) {
    return i8qk_applies_impl(cc, dst, true);
}

bool ggml_cuda_fattn_i8qk_sj_kvarn_wanted(int cc, const ggml_tensor * dst) {
    if (!ggml_cuda_fattn_i8qk_enabled() || !GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_AMPERE ||
            ggml_cuda_fattn_i8qk_yields_to_sparse(cc, dst)) {
        return false;
    }
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(dst);
    return false;
#else
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * mask = dst->src[3];
    float max_bias = 0.0f;
    float softcap  = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) dst->op_params + 2, sizeof(float));
    return Q->type == GGML_TYPE_F32 && Q->ne[0] == 256 && Q->ne[1] >= ggml_cuda_fattn_i8qk_min_q() &&
           dst->src[4] == nullptr && max_bias == 0.0f && softcap == 0.0f &&
           (mask == nullptr || (mask->type == GGML_TYPE_F16 && mask->ne[2] == 1 && mask->ne[3] == 1));
#endif
}

// ---------------------------------------------------------------------------------------------------------------------
// device helpers

static __device__ __forceinline__ uint32_t i8qk_smem_u32(const void * p) {
    return (uint32_t) __cvta_generic_to_shared(p);
}

static __device__ __forceinline__ void i8qk_cp_async_16(void * smem, const void * gmem) {
#if defined(AMPERE_MMA_AVAILABLE) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(i8qk_smem_u32(smem)), "l"(gmem));
#else
    GGML_UNUSED(smem); GGML_UNUSED(gmem);
#endif
}

static __device__ __forceinline__ void i8qk_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int n>
static __device__ __forceinline__ void i8qk_cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(n));
}

static __device__ __forceinline__ void i8qk_ldmatrix_x4(int (&r)[4], const void * smem) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(i8qk_smem_u32(smem)));
}

static __device__ __forceinline__ void i8qk_ldmatrix_x4_trans(int (&r)[4], const void * smem) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(i8qk_smem_u32(smem)));
}

// D (16x8 s32) += A (16x32 s8, row) * B (32x8 s8, col)
static __device__ __forceinline__ void i8qk_mma_s8(int (&d)[4], const int (&a)[4], const int b0, const int b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                 "{%0, %1, %2, %3};\n"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// D (16x8 f32) += A (16x16 f16, row) * B (16x8 f16, col)
static __device__ __forceinline__ void i8qk_mma_f16(float (&d)[4], const uint32_t (&a)[4], const int b0, const int b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                 "{%0, %1, %2, %3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// D (16x8 f16) += A (16x16 f16, row) * B (16x8 f16, col), f16 accumulator ([I8QK v2] PV16)
static __device__ __forceinline__ void i8qk_mma_f16_h(uint32_t (&d)[2], const uint32_t (&a)[4], const int b0, const int b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};\n"
                 : "+r"(d[0]), "+r"(d[1])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

static __device__ __forceinline__ uint32_t i8qk_pack_h2(const float x, const float y) {
    const half2 h = __floats2half2_rn(x, y);
    return *(const uint32_t *) &h;
}

static __device__ __forceinline__ int i8qk_q8(const float x, const float inv) {
    const int q = __float2int_rn(x*inv);
    return max(-127, min(127, q));
}

static __device__ __forceinline__ uint32_t i8qk_pack_s8(const int a, const int b, const int c, const int d) {
    return (uint32_t(a) & 0xFF) | ((uint32_t(b) & 0xFF) << 8) | ((uint32_t(c) & 0xFF) << 16) | ((uint32_t(d) & 0xFF) << 24);
}

// ---------------------------------------------------------------------------------------------------------------------
// K smoothing + quantization

// Partial per-channel sums of f16 K over a token chunk. grid (nchunks, heads), block D/2.
template <int D>
static __global__ void i8qk_k_mean_partial(const half * __restrict__ K, const int64_t s_tok, const int64_t s_head,
                                           const int n_kv, const int chunk, float * __restrict__ partial) {
    const int c2   = threadIdx.x; // channel pair
    const int head = blockIdx.y;
    const int t0   = blockIdx.x*chunk;
    const int t1   = min(n_kv, t0 + chunk);
    const half2 * Kh = (const half2 *) (K + head*s_head);
    float2 sum = make_float2(0.0f, 0.0f);
    for (int t = t0; t < t1; ++t) {
        const float2 v = __half22float2(Kh[(t*s_tok)/2 + c2]);
        sum.x += v.x;
        sum.y += v.y;
    }
    float * p = partial + ((size_t) head*gridDim.x + blockIdx.x)*D;
    p[2*c2 + 0] = sum.x;
    p[2*c2 + 1] = sum.y;
}

// Ordered (deterministic) reduction of the chunk sums. grid heads, block D.
template <int D>
static __global__ void i8qk_k_mean_final(const float * __restrict__ partial, const int nchunks, const int n_kv,
                                         float * __restrict__ mean) {
    const int c    = threadIdx.x;
    const int head = blockIdx.x;
    float sum = 0.0f;
    for (int i = 0; i < nchunks; ++i) {
        sum += partial[((size_t) head*nchunks + i)*D + c];
    }
    const float m = sum / n_kv;
    mean[head*D + c] = isfinite(m) ? m : 0.0f; // a non-finite padding row must not poison every key of the head
}

// [I8QK v2] KBLK: per-channel mean of each blk-token block (the last block may be partial). grid (nblk, heads), block
// D/2; sequential per thread, so deterministic.
template <int D>
static __global__ void i8qk_k_mean_block(const half * __restrict__ K, const int64_t s_tok, const int64_t s_head,
                                         const int n_kv, const int blk, float * __restrict__ mean) {
    const int c2   = threadIdx.x; // channel pair
    const int head = blockIdx.y;
    const int t0   = blockIdx.x*blk;
    const int t1   = min(n_kv, t0 + blk);
    const half2 * Kh = (const half2 *) (K + head*s_head);
    float2 sum = make_float2(0.0f, 0.0f);
    for (int t = t0; t < t1; ++t) {
        const float2 v = __half22float2(Kh[(t*s_tok)/2 + c2]);
        sum.x += v.x;
        sum.y += v.y;
    }
    const float mx = sum.x / (t1 - t0);
    const float my = sum.y / (t1 - t0);
    float * m = mean + ((size_t) head*gridDim.x + blockIdx.x)*D;
    m[2*c2 + 0] = isfinite(mx) ? mx : 0.0f;
    m[2*c2 + 1] = isfinite(my) ? my : 0.0f;
}

// One warp per (token, head): K - mean -> int8 with one f32 scale per token. grid (n_kv/8, heads), block 256.
// NS = scales stored per token (1, or D/128 copies of the token scale for the per-half kernels).
// blk > 0: mean holds nblk per-block means per head ([head][block][D]), token tok uses block tok/blk; blk 0: one mean
// per head.
template <int D, int NS>
static __global__ void i8qk_k_quant(const half * __restrict__ K, const int64_t s_tok, const int64_t s_head,
                                    const float * __restrict__ mean, const int n_kv, const int blk, const int nblk,
                                    int8_t * __restrict__ Kq, float * __restrict__ Ks) {
    constexpr int E = D/WARP_SIZE; // 8 (D 256) or 4 (D 128) channels per lane
    const int lane = threadIdx.x % WARP_SIZE;
    const int tok  = blockIdx.x*(blockDim.x/WARP_SIZE) + threadIdx.x/WARP_SIZE;
    const int head = blockIdx.y;
    if (tok >= n_kv) {
        return;
    }
    const half * src = K + head*s_head + tok*s_tok + lane*E;
    const float * mu = mean + ((size_t) head*nblk + (blk > 0 ? tok/blk : 0))*D + lane*E;

    float x[E];
    if constexpr (E == 8) {
        const uint4 raw = *(const uint4 *) src;
        const half2 * h = (const half2 *) &raw;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f = __half22float2(h[i]);
            x[2*i + 0] = f.x;
            x[2*i + 1] = f.y;
        }
    } else {
        const uint2 raw = *(const uint2 *) src;
        const half2 * h = (const half2 *) &raw;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const float2 f = __half22float2(h[i]);
            x[2*i + 0] = f.x;
            x[2*i + 1] = f.y;
        }
    }
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < E; ++i) {
        x[i] -= mu[i];
        amax = fmaxf(amax, fabsf(x[i]));
    }
#pragma unroll
    for (int o = WARP_SIZE/2; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, o));
    }
    const bool  fin = isfinite(amax);
    const float inv = fin && amax > 0.0f ? 127.0f/amax : 0.0f; // a non-finite (padding) row becomes 0 with scale 0
    uint32_t packed[E/4];
#pragma unroll
    for (int i = 0; i < E/4; ++i) {
        packed[i] = i8qk_pack_s8(i8qk_q8(x[4*i + 0], inv), i8qk_q8(x[4*i + 1], inv),
                                 i8qk_q8(x[4*i + 2], inv), i8qk_q8(x[4*i + 3], inv));
    }
    int8_t * dstq = Kq + ((size_t) head*n_kv + tok)*D + lane*E;
    if constexpr (E == 8) {
        *(uint2 *) dstq = make_uint2(packed[0], packed[1]);
    } else {
        *(uint32_t *) dstq = packed[0];
    }
    if (lane < NS) {
        Ks[((size_t) head*n_kv + tok)*NS + lane] = fin ? amax/127.0f : 0.0f;
    }
}

// [I8QK v2] KCB (#152): partial per-channel sums of a tq5_0 / tq6_0 K straight from the codes (no f16 staging).
// grid (nchunks, heads), block D/2 (one channel pair per thread); feeds i8qk_k_mean_final.
template <int D, ggml_type type>
static __global__ void i8qk_k_cb_mean_partial(const char * __restrict__ K, const int64_t nb1, const int64_t nb2,
                                              const int n_kv, const int chunk, float * __restrict__ partial) {
    const int c  = 2*threadIdx.x; // first channel of the pair
    const int b  = c/128;
    const int jj = c%128;
    const int head = blockIdx.y;
    const int t0   = blockIdx.x*chunk;
    const int t1   = min(n_kv, t0 + chunk);
    float2 sum = make_float2(0.0f, 0.0f);
    for (int t = t0; t < t1; ++t) {
        const char * row = K + head*nb2 + t*nb1;
        float2 v;
        if constexpr (type == GGML_TYPE_TQ5_0) {
            const block_tq5_0 * x = (const block_tq5_0 *) row + b;
            const float norm = __half2float(x->norm);
            v = make_float2(tq5_dequant_element(x, jj, norm), tq5_dequant_element(x, jj + 1, norm));
        } else {
            const block_tq6_0 * x = (const block_tq6_0 *) row + b;
            const float norm = __half2float(x->norm);
            v = make_float2(tq6_dequant_element(x, jj, norm), tq6_dequant_element(x, jj + 1, norm));
        }
        sum.x += v.x;
        sum.y += v.y;
    }
    float * p = partial + ((size_t) head*gridDim.x + blockIdx.x)*D;
    p[c + 0] = sum.x;
    p[c + 1] = sum.y;
}

// [I8QK v2] KCB (#152): tq5_0 / tq6_0 K codes -> int8 in one pass: x = centroid*norm - mean (mean = nullptr: no
// smoothing, i.e. the pure codebook map round(c*127/c_b) with c_b the largest centroid used in the block), one f32
// scale amax/127 per (token, 128-value block). One warp per (token, head), E = D/32 consecutive values per lane, 128/E
// lanes per block. grid (n_kv/8, heads), block 256. A non-finite row (padding) gives zero codes and scale 0.
template <int D, ggml_type type>
static __global__ void i8qk_k_codebook(const char * __restrict__ K, const int64_t nb1, const int64_t nb2, const int n_kv,
                                       const float * __restrict__ mean, int8_t * __restrict__ Kq, float * __restrict__ Ks) {
    static_assert(type == GGML_TYPE_TQ5_0 || type == GGML_TYPE_TQ6_0, "i8qk codebook: tq5_0 / tq6_0 only");
    constexpr int E  = D/WARP_SIZE;
    constexpr int NB = D/128;
    constexpr int LB = 128/E;                             // lanes per 128-value block (16 or 32)
    constexpr int NC = type == GGML_TYPE_TQ5_0 ? 16 : 64; // tq5_0: magnitude table; tq6_0: signed table over all codes
    __shared__ float lut[NC];
    if (threadIdx.x < NC) {
        lut[threadIdx.x] = type == GGML_TYPE_TQ5_0 ? TQ5_CENTROIDS[16 + threadIdx.x] : TQ6_CENTROIDS[threadIdx.x];
    }
    __syncthreads();

    const int lane = threadIdx.x % WARP_SIZE;
    const int tok  = blockIdx.x*(blockDim.x/WARP_SIZE) + threadIdx.x/WARP_SIZE;
    const int head = blockIdx.y;
    if (tok >= n_kv) {
        return;
    }
    const int b = (lane*E)/128; // block within the row
    const int j = (lane*E)%128; // first value within the block (multiple of E)
    const char * row = K + head*nb2 + tok*nb1;

    float x[E];
    float norm;
    if constexpr (type == GGML_TYPE_TQ5_0) {
        const block_tq5_0 * xb = (const block_tq5_0 *) row + b;
        norm = __half2float(xb->norm);
        const uint32_t sg = (uint32_t) xb->qh[j/8] >> (j%8);
#pragma unroll
        for (int i = 0; i < E; ++i) {
            const float m = lut[(xb->qs[(j + i)/2] >> (((j + i)%2)*4)) & 0xF];
            x[i] = ((sg >> i) & 1) ? -m : m;
        }
    } else {
        const block_tq6_0 * xb = (const block_tq6_0 *) row + b;
        norm = __half2float(xb->norm);
#pragma unroll
        for (int i = 0; i < E; ++i) {
            const int lo = (xb->qs[(j + i)/2] >> (((j + i)%2)*4)) & 0xF;
            const int hi = (xb->qh[(j + i)/4] >> (((j + i)%4)*2)) & 0x3;
            x[i] = lut[lo | (hi << 4)];
        }
    }
    const float * mu = mean ? mean + (size_t) head*D + lane*E : nullptr;
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < E; ++i) {
        x[i] = x[i]*norm - (mu ? mu[i] : 0.0f);
        amax = fmaxf(amax, fabsf(x[i]));
    }
#pragma unroll
    for (int o = LB/2; o > 0; o >>= 1) { // lanes of one block are an aligned group of LB
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, o));
    }
    const bool  fin = isfinite(norm) && isfinite(amax); // fmaxf drops NaN: test the block norm itself
    const float inv = fin && amax > 0.0f ? 127.0f/amax : 0.0f;
    uint32_t packed[E/4];
#pragma unroll
    for (int i = 0; i < E/4; ++i) {
        packed[i] = i8qk_pack_s8(i8qk_q8(x[4*i + 0], inv), i8qk_q8(x[4*i + 1], inv),
                                 i8qk_q8(x[4*i + 2], inv), i8qk_q8(x[4*i + 3], inv));
    }
    int8_t * dstq = Kq + ((size_t) head*n_kv + tok)*D + lane*E;
    if constexpr (E == 8) {
        *(uint2 *) dstq = make_uint2(packed[0], packed[1]);
    } else {
        *(uint32_t *) dstq = packed[0];
    }
    if (j == 0) {
        Ks[((size_t) head*n_kv + tok)*NB + b] = fin ? amax/127.0f : 0.0f;
    }
}

// Per (query tile, KV tile) mask class: 0 = every visible element -inf (skip), 1 = all 0 (no mask), 2 = mixed.
// grid (n_ktiles, n_qtiles), block 256.
static __global__ void i8qk_tile_flags(const half * __restrict__ mask, const int64_t s1, const int n_q,
                                       uint8_t * __restrict__ flags) {
    const int kt = blockIdx.x;
    const int qt = blockIdx.y;
    int all_ninf = 1;
    int all_zero = 1;
    for (int i = threadIdx.x; i < I8QK_BM*I8QK_BN/2; i += blockDim.x) {
        const int row = i / (I8QK_BN/2);
        const int c2  = i % (I8QK_BN/2);
        const int q   = qt*I8QK_BM + row;
        if (q < n_q) {
            const float2 m = __half22float2(((const half2 *) (mask + q*s1 + kt*I8QK_BN))[c2]);
            all_ninf &= (isinf(m.x) && m.x < 0.0f) && (isinf(m.y) && m.y < 0.0f);
            all_zero &= m.x == 0.0f && m.y == 0.0f;
        }
    }
    all_ninf = __syncthreads_and(all_ninf);
    all_zero = __syncthreads_and(all_zero);
    if (threadIdx.x == 0) {
        flags[(size_t) qt*gridDim.x + kt] = all_ninf ? 0 : (all_zero ? 1 : 2);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// attention kernel

// NS = Q and K scales per row / token: 1 (v1: per row / per token) or D/128 (per 128-dim half: KCB / QH)
template <int D, int NS>
struct i8qk_smem {
    static constexpr int q_stride = D + 16;   // bytes per int8 Q row (+16: ldmatrix rows hit distinct banks)
    static constexpr int k_stride = D + 16;   // bytes per int8 K row
    static constexpr int v_stride = 2*D + 16; // bytes per f16 V row
    static constexpr int q_off   = 0;
    static constexpr int k_off   = q_off  + I8QK_BM*q_stride;
    static constexpr int v_off   = k_off  + I8QK_BN*k_stride;
    static constexpr int ks_off  = v_off  + I8QK_BN*v_stride;
    static constexpr int qs_off  = ks_off + I8QK_BN*NS*int(sizeof(float));
    static constexpr int bytes   = qs_off + I8QK_BM*NS*int(sizeof(float));
};

// PV16: f16 P.V accumulator (#151). NS: see i8qk_smem. Q2L: two-limb Q (#154 arm); limb 1 lives in registers, limb 2
// in the Q shared-memory tile. KBF: block-frozen K offsets (#155): K tile j was smoothed by the mean m_b of block
// b = j/blk_tiles (kmean [kv head][block][D]), so q.m_b (f32, from the unquantized Q row) is added back to every score of
// the block. The v1 kernel is <D, false, 1, false, false>.
template <int D, bool PV16, int NS, bool Q2L, bool KBF>
__launch_bounds__(I8QK_NW*WARP_SIZE, 1)
static __global__ void i8qk_flash_attn(
        const char * __restrict__ Q, const int64_t q_nb1, const int64_t q_nb2,
        const int8_t * __restrict__ Kq, const float * __restrict__ Ksc,
        const char * __restrict__ V, const int64_t v_nb1, const int64_t v_nb2,
        const half * __restrict__ mask, const int64_t mask_s1,
        const uint8_t * __restrict__ flags, const int n_ktiles,
        float * __restrict__ dst, const int64_t dst_s_head, const int64_t dst_s_q,
        const int n_q, const int n_kv, const int gqa, const float scale_log2,
        const float * __restrict__ kmean, const int blk_tiles, const int nblk) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    static_assert(NS == 1 || NS == D/128, "i8qk: NS must be 1 or D/128");
    constexpr int KS_PER = (D/32)/NS; // k32 steps per scale group
    using L = i8qk_smem<D, NS>;
    extern __shared__ __align__(16) char smem[];
    int8_t * Qs  = (int8_t *) (smem + L::q_off);
    int8_t * Ks  = (int8_t *) (smem + L::k_off);
    char   * Vs  =            (smem + L::v_off);
    float  * Kss = (float  *) (smem + L::ks_off);
    float  * Qss = (float  *) (smem + L::qs_off);

    const int tid  = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;
    const int g    = lane / 4;
    const int t    = lane % 4;
    const int qt   = blockIdx.x;
    const int h    = blockIdx.y;      // query head within the launched group
    const int kvh  = h / gqa;
    const int q0   = qt*I8QK_BM;

    const uint8_t * fl = flags + (size_t) qt*n_ktiles;

    const int8_t * Kq_h  = Kq  + (size_t) kvh*n_kv*D;
    const float  * Ksc_h = Ksc + (size_t) kvh*n_kv*NS;
    const char   * V_h   = V   + kvh*v_nb2;

    auto load_K = [&](const int j) {
        constexpr int chunks_row = D/16;
        for (int i = tid; i < I8QK_BN*chunks_row; i += i8qk_nthreads) {
            const int row = i / chunks_row;
            const int c   = i % chunks_row;
            i8qk_cp_async_16(Ks + row*L::k_stride + c*16, Kq_h + ((size_t) j*I8QK_BN + row)*D + c*16);
        }
        if (tid < I8QK_BN*NS/4) {
            i8qk_cp_async_16(Kss + tid*4, Ksc_h + (size_t) j*I8QK_BN*NS + tid*4);
        }
    };
    auto load_V = [&](const int j) {
        constexpr int chunks_row = 2*D/16;
        for (int i = tid; i < I8QK_BN*chunks_row; i += i8qk_nthreads) {
            const int row = i / chunks_row;
            const int c   = i % chunks_row;
            i8qk_cp_async_16(Vs + row*L::v_stride + c*16, V_h + ((int64_t) j*I8QK_BN + row)*v_nb1 + c*16);
        }
    };
    auto next_tile = [&](int j) {
        for (++j; j < n_ktiles; ++j) {
            if (fl[j] != 0) {
                return j;
            }
        }
        return -1;
    };

    int j = next_tile(-1);
    if (j >= 0) {
        load_K(j);
    }
    i8qk_cp_async_commit();

    const int8_t * Qw = Qs + (warp*16)*L::q_stride;
    const int lrow = lane % 8;
    const int lmat = lane / 8;

    // Q tile -> int8 rows in shared memory, NS f32 scales per row (softmax scale and log2(e) folded in). Lane value
    // group i (4 values at i*128 + lane*4) lies in 128-dim half i. Two-limb Q: pass 0 writes limb 1 and moves it to
    // registers, pass 1 writes limb 2 (residual in units of s/254) over it; each warp only touches its own 16 rows.
    int QA1[Q2L ? D/32 : 1][4];
    GGML_UNUSED(QA1);
#pragma unroll
    for (int pass = 0; pass < (Q2L ? 2 : 1); ++pass) {
        for (int r = 0; r < 16; ++r) {
            const int row = warp*16 + r;
            const int q   = q0 + row;
            float4 x[D/128];
            float amax[D/128];
#pragma unroll
            for (int i = 0; i < D/128; ++i) {
                amax[i] = 0.0f;
            }
            if (q < n_q) {
                const float * Qr = (const float *) (Q + q*q_nb1 + h*q_nb2);
#pragma unroll
                for (int i = 0; i < D/128; ++i) {
                    x[i] = *(const float4 *) (Qr + i*128 + lane*4);
                    amax[NS == 1 ? 0 : i] = fmaxf(amax[NS == 1 ? 0 : i],
                        fmaxf(fmaxf(fabsf(x[i].x), fabsf(x[i].y)), fmaxf(fabsf(x[i].z), fabsf(x[i].w))));
                }
            } else {
#pragma unroll
                for (int i = 0; i < D/128; ++i) {
                    x[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                }
            }
#pragma unroll
            for (int i = 0; i < NS; ++i) {
#pragma unroll
                for (int o = WARP_SIZE/2; o > 0; o >>= 1) {
                    amax[i] = fmaxf(amax[i], __shfl_xor_sync(0xFFFFFFFF, amax[i], o));
                }
            }
#pragma unroll
            for (int i = 0; i < D/128; ++i) {
                const float am  = amax[NS == 1 ? 0 : i];
                const float inv = am > 0.0f ? 127.0f/am : 0.0f;
                uint32_t pk;
                if (!Q2L || pass == 0) {
                    pk = i8qk_pack_s8(i8qk_q8(x[i].x, inv), i8qk_q8(x[i].y, inv), i8qk_q8(x[i].z, inv), i8qk_q8(x[i].w, inv));
                } else {
                    // residual x - s*q1 (s = am/127), in units of s/254: |r| <= 127 up to rounding
                    const float s   = am/127.0f;
                    const float inv2 = am > 0.0f ? 254.0f/s : 0.0f;
                    pk = i8qk_pack_s8(i8qk_q8(x[i].x - s*i8qk_q8(x[i].x, inv), inv2),
                                      i8qk_q8(x[i].y - s*i8qk_q8(x[i].y, inv), inv2),
                                      i8qk_q8(x[i].z - s*i8qk_q8(x[i].z, inv), inv2),
                                      i8qk_q8(x[i].w - s*i8qk_q8(x[i].w, inv), inv2));
                }
                *(uint32_t *) (Qs + row*L::q_stride + i*128 + lane*4) = pk;
            }
#pragma unroll
            for (int i = 0; i < NS; ++i) {
                if (lane == i) {
                    Qss[row*NS + i] = (Q2L ? amax[i]/(127.0f*254.0f) : amax[i]/127.0f)*scale_log2;
                }
            }
        }
        if constexpr (Q2L) {
            if (pass == 0) {
                __syncwarp();
#pragma unroll
                for (int ks = 0; ks < D/32; ++ks) {
                    i8qk_ldmatrix_x4(QA1[ks], Qw + (lrow + (lmat & 1)*8)*L::q_stride + ks*32 + (lmat >> 1)*16);
                }
                __syncwarp();
            }
        }
    }
    __syncthreads();
    float sq0[NS], sq1[NS];
#pragma unroll
    for (int i = 0; i < NS; ++i) {
        sq0[i] = Qss[(warp*16 + g)*NS + i];
        sq1[i] = Qss[(warp*16 + g + 8)*NS + i];
    }

    float    O  [PV16 ? 1 : D/8][4];
    uint32_t O16[PV16 ? D/8 : 1][2];
    GGML_UNUSED(O); GGML_UNUSED(O16);
    if constexpr (PV16) {
#pragma unroll
        for (int i = 0; i < D/8; ++i) {
            O16[i][0] = O16[i][1] = 0u;
        }
    } else {
#pragma unroll
        for (int i = 0; i < D/8; ++i) {
            O[i][0] = O[i][1] = O[i][2] = O[i][3] = 0.0f;
        }
    }
    float m0 = -INFINITY, m1 = -INFINITY; // running row max (log2 domain), rows g and g + 8
    float l0 = 0.0f,      l1 = 0.0f;      // per-thread partial row sums

    int   kb_cur = -1;                    // KBF: block of the cached corrections
    float kc0 = 0.0f, kc1 = 0.0f;         // KBF: q.m_b x scale_log2 for rows g and g + 8
    GGML_UNUSED(kb_cur); GGML_UNUSED(kc0); GGML_UNUSED(kc1);

    while (j >= 0) {
        const int jn = next_tile(j);
        const uint8_t flag = fl[j];

        i8qk_cp_async_wait<0>();
        __syncthreads(); // K_j landed; every warp is done with V_{j-1}
        load_V(j);
        i8qk_cp_async_commit();

        // S = Q K^T (int8 MMA, int32 accumulation per scale group), dequantized by the row x token scales
        float Sf[I8QK_BN/8][4];
#pragma unroll
        for (int sg = 0; sg < NS; ++sg) {
            int S[I8QK_BN/8][4];
#pragma unroll
            for (int i = 0; i < I8QK_BN/8; ++i) {
                S[i][0] = S[i][1] = S[i][2] = S[i][3] = 0;
            }
#pragma unroll
            for (int limb = 0; limb < (Q2L ? 2 : 1); ++limb) {
                if (Q2L && limb == 1) {
                    // q1*254 + q2: |S| <= 127*127*D*255 < 2^31 at D 256
#pragma unroll
                    for (int i = 0; i < I8QK_BN/8; ++i) {
                        S[i][0] *= 254; S[i][1] *= 254; S[i][2] *= 254; S[i][3] *= 254;
                    }
                }
#pragma unroll
                for (int ks = sg*KS_PER; ks < (sg + 1)*KS_PER; ++ks) {
                    int A[4];
                    if constexpr (Q2L) {
                        if (limb == 0) {
                            A[0] = QA1[ks][0]; A[1] = QA1[ks][1]; A[2] = QA1[ks][2]; A[3] = QA1[ks][3];
                        } else {
                            i8qk_ldmatrix_x4(A, Qw + (lrow + (lmat & 1)*8)*L::q_stride + ks*32 + (lmat >> 1)*16);
                        }
                    } else {
                        i8qk_ldmatrix_x4(A, Qw + (lrow + (lmat & 1)*8)*L::q_stride + ks*32 + (lmat >> 1)*16);
                    }
#pragma unroll
                    for (int np = 0; np < I8QK_BN/16; ++np) {
                        int B[4];
                        i8qk_ldmatrix_x4(B, Ks + (np*16 + lrow + (lmat >> 1)*8)*L::k_stride + ks*32 + (lmat & 1)*16);
                        i8qk_mma_s8(S[2*np + 0], A, B[0], B[1]);
                        i8qk_mma_s8(S[2*np + 1], A, B[2], B[3]);
                    }
                }
            }
#pragma unroll
            for (int nt = 0; nt < I8QK_BN/8; ++nt) {
                const int col = nt*8 + 2*t;
                const float sk0 = Kss[(col + 0)*NS + sg];
                const float sk1 = Kss[(col + 1)*NS + sg];
                if (sg == 0) {
                    Sf[nt][0] = float(S[nt][0])*sq0[sg]*sk0;
                    Sf[nt][1] = float(S[nt][1])*sq0[sg]*sk1;
                    Sf[nt][2] = float(S[nt][2])*sq1[sg]*sk0;
                    Sf[nt][3] = float(S[nt][3])*sq1[sg]*sk1;
                } else {
                    Sf[nt][0] += float(S[nt][0])*sq0[sg]*sk0;
                    Sf[nt][1] += float(S[nt][1])*sq0[sg]*sk1;
                    Sf[nt][2] += float(S[nt][2])*sq1[sg]*sk0;
                    Sf[nt][3] += float(S[nt][3])*sq1[sg]*sk1;
                }
            }
        }

        if constexpr (KBF) {
            const int kb = j / blk_tiles;
            if (kb != kb_cur) {
                kb_cur = kb;
                const float * mb = kmean + ((size_t) kvh*nblk + kb)*D;
                float4 mv[D/128];
#pragma unroll
                for (int i = 0; i < D/128; ++i) {
                    mv[i] = *(const float4 *) (mb + i*128 + lane*4);
                }
                for (int r = 0; r < 16; ++r) {
                    const int q = q0 + warp*16 + r;
                    float c = 0.0f;
                    if (q < n_q) {
                        const float * Qr = (const float *) (Q + q*q_nb1 + h*q_nb2);
#pragma unroll
                        for (int i = 0; i < D/128; ++i) {
                            const float4 x = *(const float4 *) (Qr + i*128 + lane*4);
                            c += x.x*mv[i].x + x.y*mv[i].y + x.z*mv[i].z + x.w*mv[i].w;
                        }
                    }
#pragma unroll
                    for (int o = WARP_SIZE/2; o > 0; o >>= 1) {
                        c += __shfl_xor_sync(0xFFFFFFFF, c, o);
                    }
                    if (r == g) {
                        kc0 = c*scale_log2;
                    }
                    if (r == g + 8) {
                        kc1 = c*scale_log2;
                    }
                }
            }
#pragma unroll
            for (int nt = 0; nt < I8QK_BN/8; ++nt) {
                Sf[nt][0] += kc0;
                Sf[nt][1] += kc0;
                Sf[nt][2] += kc1;
                Sf[nt][3] += kc1;
            }
        }

        // mask, online softmax (log2 domain)
        float mx0 = -INFINITY, mx1 = -INFINITY;
        const int qa = q0 + warp*16 + g;
        const int qb = qa + 8;
#pragma unroll
        for (int nt = 0; nt < I8QK_BN/8; ++nt) {
            const int col = nt*8 + 2*t;
            if (flag == 2) {
                if (qa < n_q) {
                    const float2 mk = __half22float2(*(const half2 *) (mask + qa*mask_s1 + (int64_t) j*I8QK_BN + col));
                    Sf[nt][0] += mk.x*1.4426950408889634f;
                    Sf[nt][1] += mk.y*1.4426950408889634f;
                }
                if (qb < n_q) {
                    const float2 mk = __half22float2(*(const half2 *) (mask + qb*mask_s1 + (int64_t) j*I8QK_BN + col));
                    Sf[nt][2] += mk.x*1.4426950408889634f;
                    Sf[nt][3] += mk.y*1.4426950408889634f;
                }
            }
            mx0 = fmaxf(mx0, fmaxf(Sf[nt][0], Sf[nt][1]));
            mx1 = fmaxf(mx1, fmaxf(Sf[nt][2], Sf[nt][3]));
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xFFFFFFFF, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xFFFFFFFF, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xFFFFFFFF, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xFFFFFFFF, mx1, 2));

        const float mn0 = fmaxf(m0, mx0);
        const float mn1 = fmaxf(m1, mx1);
        const float mu0 = mn0 == -INFINITY ? 0.0f : mn0; // fully masked so far: keep every term at exp2(-inf) = 0
        const float mu1 = mn1 == -INFINITY ? 0.0f : mn1;
        const float a0  = exp2f(m0 - mu0);
        const float a1  = exp2f(m1 - mu1);
        m0 = mn0;
        m1 = mn1;
        l0 *= a0;
        l1 *= a1;
        if constexpr (PV16) {
            const half2 a0h = __float2half2_rn(a0);
            const half2 a1h = __float2half2_rn(a1);
#pragma unroll
            for (int i = 0; i < D/8; ++i) {
                *(half2 *) &O16[i][0] = __hmul2(*(const half2 *) &O16[i][0], a0h);
                *(half2 *) &O16[i][1] = __hmul2(*(const half2 *) &O16[i][1], a1h);
            }
        } else {
#pragma unroll
            for (int i = 0; i < D/8; ++i) {
                O[i][0] *= a0;
                O[i][1] *= a0;
                O[i][2] *= a1;
                O[i][3] *= a1;
            }
        }

        uint32_t P[I8QK_BN/16][4];
#pragma unroll
        for (int nt = 0; nt < I8QK_BN/8; ++nt) {
            const float p0 = exp2f(Sf[nt][0] - mu0);
            const float p1 = exp2f(Sf[nt][1] - mu0);
            const float p2 = exp2f(Sf[nt][2] - mu1);
            const float p3 = exp2f(Sf[nt][3] - mu1);
            l0 += p0 + p1;
            l1 += p2 + p3;
            P[nt/2][(nt % 2)*2 + 0] = i8qk_pack_h2(p0, p1);
            P[nt/2][(nt % 2)*2 + 1] = i8qk_pack_h2(p2, p3);
        }

        __syncthreads(); // every warp is done with K_j
        if (jn >= 0) {
            load_K(jn);
        }
        i8qk_cp_async_commit();
        i8qk_cp_async_wait<1>();
        __syncthreads(); // V_j landed

        // O += P V (f16 MMA; f32 accumulation, or f16 accumulation with PV16)
#pragma unroll
        for (int kc = 0; kc < I8QK_BN/16; ++kc) {
#pragma unroll
            for (int dp = 0; dp < D/16; ++dp) {
                int B[4];
                i8qk_ldmatrix_x4_trans(B, Vs + (kc*16 + lrow + (lmat & 1)*8)*L::v_stride + (dp*16 + (lmat >> 1)*8)*2);
                if constexpr (PV16) {
                    i8qk_mma_f16_h(O16[2*dp + 0], P[kc], B[0], B[1]);
                    i8qk_mma_f16_h(O16[2*dp + 1], P[kc], B[2], B[3]);
                } else {
                    i8qk_mma_f16(O[2*dp + 0], P[kc], B[0], B[1]);
                    i8qk_mma_f16(O[2*dp + 1], P[kc], B[2], B[3]);
                }
            }
        }
        j = jn;
    }
    i8qk_cp_async_wait<0>();

    l0 += __shfl_xor_sync(0xFFFFFFFF, l0, 1);
    l0 += __shfl_xor_sync(0xFFFFFFFF, l0, 2);
    l1 += __shfl_xor_sync(0xFFFFFFFF, l1, 1);
    l1 += __shfl_xor_sync(0xFFFFFFFF, l1, 2);
    const float il0 = l0 > 0.0f ? 1.0f/l0 : 0.0f;
    const float il1 = l1 > 0.0f ? 1.0f/l1 : 0.0f;

    const int qa = q0 + warp*16 + g;
    const int qb = qa + 8;
    if (qa < n_q) {
        float * o = dst + qa*dst_s_q + h*dst_s_head;
#pragma unroll
        for (int i = 0; i < D/8; ++i) {
            if constexpr (PV16) {
                const float2 v = __half22float2(*(const half2 *) &O16[i][0]);
                *(float2 *) (o + i*8 + 2*t) = make_float2(v.x*il0, v.y*il0);
            } else {
                *(float2 *) (o + i*8 + 2*t) = make_float2(O[i][0]*il0, O[i][1]*il0);
            }
        }
    }
    if (qb < n_q) {
        float * o = dst + qb*dst_s_q + h*dst_s_head;
#pragma unroll
        for (int i = 0; i < D/8; ++i) {
            if constexpr (PV16) {
                const float2 v = __half22float2(*(const half2 *) &O16[i][1]);
                *(float2 *) (o + i*8 + 2*t) = make_float2(v.x*il1, v.y*il1);
            } else {
                *(float2 *) (o + i*8 + 2*t) = make_float2(O[i][2]*il1, O[i][3]*il1);
            }
        }
    }
#else
    GGML_UNUSED_VARS(Q, q_nb1, q_nb2, Kq, Ksc, V, v_nb1, v_nb2, mask, mask_s1, flags, n_ktiles, dst, dst_s_head, dst_s_q,
                     n_q, n_kv, gqa, scale_log2, kmean, blk_tiles, nblk);
    NO_DEVICE_CODE;
#endif
}

// ---------------------------------------------------------------------------------------------------------------------
// host

template <int D, bool PV16, int NS, bool Q2L, bool KBF>
static void i8qk_launch_attn(const ggml_tensor * dst, const char * Qp, const int8_t * Kq, const float * ksc,
                             const char * Vp, const int64_t v_nb1, const int64_t v_nb2, const uint8_t * flags,
                             const int n_qtiles, const int n_ktiles, float * dstp, const int heads, const int gqa,
                             const int n_q, const int n_kv, const float scale_log2, const float * kmean,
                             const int blk_tiles, const int nblk, cudaStream_t stream) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * mask = dst->src[3];
    const int    id     = ggml_cuda_get_device();
    const size_t nbytes = i8qk_smem<D, NS>::bytes;
    static bool smem_raised[GGML_CUDA_MAX_DEVICES] = {false};
    if (!smem_raised[id]) {
        CUDA_CHECK(cudaFuncSetAttribute(i8qk_flash_attn<D, PV16, NS, Q2L, KBF>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) nbytes));
        smem_raised[id] = true;
    }
    i8qk_flash_attn<D, PV16, NS, Q2L, KBF><<<dim3(n_qtiles, heads*gqa), i8qk_nthreads, nbytes, stream>>>(
        Qp, Q->nb[1], Q->nb[2], Kq, ksc, Vp, v_nb1, v_nb2,
        mask ? (const half *) mask->data : nullptr, mask ? (int64_t) (mask->nb[1]/sizeof(half)) : 0,
        flags, n_ktiles, dstp, (int64_t) (dst->nb[1]/sizeof(float)), (int64_t) (dst->nb[2]/sizeof(float)),
        n_q, n_kv, gqa, scale_log2, kmean, blk_tiles, nblk);
    CUDA_CHECK(cudaGetLastError());
}

template <int D>
static void ggml_cuda_flash_attn_ext_i8qk_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const int first,
                                               const int heads, void * ws_k, half * ws_v) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool & pool = ctx.pool();

    const i8qk_modes & md = ggml_cuda_fattn_i8qk_modes();
    const bool kcb = md.kcb && (K->type == GGML_TYPE_TQ5_0 || K->type == GGML_TYPE_TQ6_0);
    // per-half scale groups (NS = D/128) when either side needs them; at D 128 that is one group, i.e. the v1 layout
    const int  ns  = (kcb || md.qh) ? D/128 : 1;
    // KCB: the global mean (default) or none (KBLK=0); block-frozen means are not wired into the codebook pass
    const int  kblk = kcb && md.kblk > 0 ? -1 : md.kblk;
    const bool kbf  = kblk > 0;

    const int n_q  = (int) Q->ne[1];
    const int n_kv = (int) K->ne[1];
    const int gqa  = (int) (Q->ne[2] / K->ne[2]);

    const bool k_conv = K->type != GGML_TYPE_F16;
    const bool v_conv = V->type != GGML_TYPE_F16;
    const int nchunks  = std::min(64, std::max(1, n_kv/256));
    const int chunk    = (n_kv + nchunks - 1)/nchunks;
    const int n_qtiles = (n_q + I8QK_BM - 1)/I8QK_BM;
    const int n_ktiles = n_kv/I8QK_BN;
    const int nblk     = kbf ? (n_kv + kblk - 1)/kblk : 1;

    // the big buffers live in the caller's f16 K/V reservation (no new VRAM); only scales, means and flags are pooled
    GGML_ASSERT(ws_k != nullptr && ws_v != nullptr);
    half   * f16_buf = ws_v;
    int8_t * Kq      = (int8_t *) ws_k;
    // pool allocations in construction order (the VMM pool frees LIFO, i.e. in reverse destruction order)
    ggml_cuda_pool_alloc<float>   ksc(pool, (size_t) n_kv*heads*ns);
    ggml_cuda_pool_alloc<float>   kmean_part(pool);
    ggml_cuda_pool_alloc<float>   kmean(pool);
    if (kblk < 0) {
        kmean_part.alloc((size_t) nchunks*heads*D);
    }
    if (kblk < 0 || !kcb) {
        kmean.alloc((size_t) heads*nblk*D);
    }
    ggml_cuda_pool_alloc<uint8_t> flags(pool, (size_t) n_qtiles*n_ktiles);

    if (kcb) {
        // codebook-direct int8 K straight from the cache codes (#152): mean pass + one quant pass, no f16 staging
        const char * Kp = (const char *) K->data + (size_t) first*K->nb[2];
        const dim3 grid(n_kv/8, heads);
        const float * mu = nullptr;
        if (kblk < 0) {
            if (K->type == GGML_TYPE_TQ5_0) {
                i8qk_k_cb_mean_partial<D, GGML_TYPE_TQ5_0><<<dim3(nchunks, heads), D/2, 0, stream>>>(Kp, K->nb[1], K->nb[2], n_kv, chunk, kmean_part.ptr);
            } else {
                i8qk_k_cb_mean_partial<D, GGML_TYPE_TQ6_0><<<dim3(nchunks, heads), D/2, 0, stream>>>(Kp, K->nb[1], K->nb[2], n_kv, chunk, kmean_part.ptr);
            }
            i8qk_k_mean_final<D><<<heads, D, 0, stream>>>(kmean_part.ptr, nchunks, n_kv, kmean.ptr);
            mu = kmean.ptr;
        }
        if (K->type == GGML_TYPE_TQ5_0) {
            i8qk_k_codebook<D, GGML_TYPE_TQ5_0><<<grid, 256, 0, stream>>>(Kp, K->nb[1], K->nb[2], n_kv, mu, Kq, ksc.ptr);
        } else {
            i8qk_k_codebook<D, GGML_TYPE_TQ6_0><<<grid, 256, 0, stream>>>(Kp, K->nb[1], K->nb[2], n_kv, mu, Kq, ksc.ptr);
        }
        CUDA_CHECK(cudaGetLastError());
    } else {
        // 1. f16 K: in place (f16 cache) or converted into the V workspace (overwritten by V below, stream ordered)
        const half * Kh;
        int64_t k_s_tok, k_s_head;
        if (k_conv) {
            const size_t ts = ggml_type_size(K->type);
            const to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(K->type);
            GGML_ASSERT(to_fp16 != nullptr);
            to_fp16((const char *) K->data + (size_t) first*K->nb[2], f16_buf, D, n_kv, heads, 1,
                    int64_t(K->nb[1]/ts), int64_t(K->nb[2]/ts), int64_t(K->nb[3]/ts), stream);
            CUDA_CHECK(cudaGetLastError());
            Kh = f16_buf;
            k_s_tok  = D;
            k_s_head = (int64_t) n_kv*D;
        } else {
            Kh = (const half *) ((const char *) K->data + (size_t) first*K->nb[2]);
            k_s_tok  = K->nb[1]/sizeof(half);
            k_s_head = K->nb[2]/sizeof(half);
        }

        // 2. per-channel mean (v1: two deterministic passes over all n_kv; KBLK: per block, or none),
        // 3. smoothed int8 K + per-token scales
        if (kblk < 0) {
            i8qk_k_mean_partial<D><<<dim3(nchunks, heads), D/2, 0, stream>>>(Kh, k_s_tok, k_s_head, n_kv, chunk, kmean_part.ptr);
            i8qk_k_mean_final<D><<<heads, D, 0, stream>>>(kmean_part.ptr, nchunks, n_kv, kmean.ptr);
        } else if (kblk == 0) {
            CUDA_CHECK(cudaMemsetAsync(kmean.ptr, 0, (size_t) heads*D*sizeof(float), stream));
        } else {
            i8qk_k_mean_block<D><<<dim3(nblk, heads), D/2, 0, stream>>>(Kh, k_s_tok, k_s_head, n_kv, kblk, kmean.ptr);
        }
        const int qblk = kbf ? kblk : 0;
        if (ns == 1) {
            i8qk_k_quant<D, 1><<<dim3(n_kv/8, heads), 256, 0, stream>>>(Kh, k_s_tok, k_s_head, kmean.ptr, n_kv, qblk, nblk, Kq, ksc.ptr);
        } else {
            i8qk_k_quant<D, D/128><<<dim3(n_kv/8, heads), 256, 0, stream>>>(Kh, k_s_tok, k_s_head, kmean.ptr, n_kv, qblk, nblk, Kq, ksc.ptr);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    // 4. f16 V
    const char * Vp;
    int64_t v_nb1, v_nb2;
    if (v_conv) {
        const size_t ts = ggml_type_size(V->type);
        const to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(V->type);
        GGML_ASSERT(to_fp16 != nullptr);
        to_fp16((const char *) V->data + (size_t) first*V->nb[2], f16_buf, D, n_kv, heads, 1,
                int64_t(V->nb[1]/ts), int64_t(V->nb[2]/ts), int64_t(V->nb[3]/ts), stream);
        CUDA_CHECK(cudaGetLastError());
        Vp    = (const char *) f16_buf;
        v_nb1 = (int64_t) D*sizeof(half);
        v_nb2 = (int64_t) n_kv*D*sizeof(half);
    } else {
        Vp    = (const char *) V->data + (size_t) first*V->nb[2];
        v_nb1 = V->nb[1];
        v_nb2 = V->nb[2];
    }

    // 5. mask classes per (query tile, KV tile)
    if (mask != nullptr) {
        i8qk_tile_flags<<<dim3(n_ktiles, n_qtiles), 256, 0, stream>>>((const half *) mask->data, mask->nb[1]/sizeof(half),
                                                                     n_q, flags.ptr);
        CUDA_CHECK(cudaGetLastError());
    } else {
        CUDA_CHECK(cudaMemsetAsync(flags.ptr, 1, (size_t) n_qtiles*n_ktiles, stream));
    }

    // 6. attention
    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    const float scale_log2 = scale*1.4426950408889634f;

    const char * Qp = (const char *) Q->data + (size_t) first*gqa*Q->nb[2];
    float * dstp = (float *) dst->data + (size_t) first*gqa*D;
#define I8QK_LAUNCH(PV16, NS, Q2L, KBF) i8qk_launch_attn<D, PV16, NS, Q2L, KBF>(dst, Qp, Kq, ksc.ptr, Vp, v_nb1, v_nb2, \
        flags.ptr, n_qtiles, n_ktiles, dstp, heads, gqa, n_q, n_kv, scale_log2, kmean.ptr, kblk/I8QK_BN, nblk, stream)
#define I8QK_LAUNCH_PV16(NS) \
        if (md.q2l) { if (kbf) { I8QK_LAUNCH(true, NS, true,  true); } else { I8QK_LAUNCH(true, NS, true,  false); } } \
        else        { if (kbf) { I8QK_LAUNCH(true, NS, false, true); } else { I8QK_LAUNCH(true, NS, false, false); } }
    // Q2L and KBLK=n run on the PV16 kernel only (the modes switch PV16 on)
    constexpr int NH = D/128;
    if (ns == 1) {
        if (md.pv16) { I8QK_LAUNCH_PV16(1) } else { I8QK_LAUNCH(false, 1, false, false); }
    } else {
        if (md.pv16) { I8QK_LAUNCH_PV16(NH) } else { I8QK_LAUNCH(false, NH, false, false); }
    }
#undef I8QK_LAUNCH_PV16
#undef I8QK_LAUNCH
}

void ggml_cuda_flash_attn_ext_i8qk(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const int first, const int heads,
                                   void * ws_k, half * ws_v) {
    switch (dst->src[0]->ne[0]) {
        case 128: ggml_cuda_flash_attn_ext_i8qk_impl<128>(ctx, dst, first, heads, ws_k, ws_v); break;
        case 256: ggml_cuda_flash_attn_ext_i8qk_impl<256>(ctx, dst, first, heads, ws_k, ws_v); break;
        default: GGML_ABORT("i8qk: unsupported head size");
    }
}
