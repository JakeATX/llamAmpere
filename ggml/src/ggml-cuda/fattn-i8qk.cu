// [I8QK] int8-QK prefill flash attention (SageAttention-style). See fattn-i8qk.cuh for the route description.

#include "fattn-i8qk.cuh"
#include "convert.cuh"

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
        // default ON since the I8QK gate (2026-10-07: prefill +5.21% G, decode +4.77% G at 100K); =0 disables
        return e == nullptr || !(e[0] == '0' && e[1] == '\0');
    }();
    return v;
}

static int ggml_cuda_fattn_i8qk_min_q() {
    static const int v = [] {
        const char * e = getenv("GGML_CUDA_FA_I8QK_MIN_Q");
        const int n = e ? atoi(e) : 64;
        return n >= 9 ? n : 64; // decode / MTP verify widths 1-8 never take the route
    }();
    return v;
}

// Quantized caches only: the int8 K and the f16 K staging / f16 V live in the f16 K/V reservation the allocator makes for
// the f16 route of a quantized pair. An f16 K or V has no such reservation (the f16 route reads it in place), so the int8
// route would need new VRAM for it: f16 caches keep the f16 route.
static bool i8qk_kv_type_ok(const ggml_tensor * t) {
    switch (t->type) {
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

bool ggml_cuda_fattn_i8qk_applies(int cc, const ggml_tensor * dst) {
    if (!ggml_cuda_fattn_i8qk_enabled()) {
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
    if (!i8qk_kv_type_ok(K) || !i8qk_kv_type_ok(V)) {
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

// One warp per (token, head): K - mean -> int8 with one f32 scale per token. grid (n_kv/8, heads), block 256.
template <int D>
static __global__ void i8qk_k_quant(const half * __restrict__ K, const int64_t s_tok, const int64_t s_head,
                                    const float * __restrict__ mean, const int n_kv,
                                    int8_t * __restrict__ Kq, float * __restrict__ Ks) {
    constexpr int E = D/WARP_SIZE; // 8 (D 256) or 4 (D 128) channels per lane
    const int lane = threadIdx.x % WARP_SIZE;
    const int tok  = blockIdx.x*(blockDim.x/WARP_SIZE) + threadIdx.x/WARP_SIZE;
    const int head = blockIdx.y;
    if (tok >= n_kv) {
        return;
    }
    const half * src = K + head*s_head + tok*s_tok + lane*E;
    const float * mu = mean + head*D + lane*E;

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
    if (lane == 0) {
        Ks[(size_t) head*n_kv + tok] = fin ? amax/127.0f : 0.0f;
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

template <int D>
struct i8qk_smem {
    static constexpr int q_stride = D + 16;   // bytes per int8 Q row (+16: ldmatrix rows hit distinct banks)
    static constexpr int k_stride = D + 16;   // bytes per int8 K row
    static constexpr int v_stride = 2*D + 16; // bytes per f16 V row
    static constexpr int q_off   = 0;
    static constexpr int k_off   = q_off  + I8QK_BM*q_stride;
    static constexpr int v_off   = k_off  + I8QK_BN*k_stride;
    static constexpr int ks_off  = v_off  + I8QK_BN*v_stride;
    static constexpr int qs_off  = ks_off + I8QK_BN*int(sizeof(float));
    static constexpr int bytes   = qs_off + I8QK_BM*int(sizeof(float));
};

template <int D>
__launch_bounds__(I8QK_NW*WARP_SIZE, 1)
static __global__ void i8qk_flash_attn(
        const char * __restrict__ Q, const int64_t q_nb1, const int64_t q_nb2,
        const int8_t * __restrict__ Kq, const float * __restrict__ Ksc,
        const char * __restrict__ V, const int64_t v_nb1, const int64_t v_nb2,
        const half * __restrict__ mask, const int64_t mask_s1,
        const uint8_t * __restrict__ flags, const int n_ktiles,
        float * __restrict__ dst, const int64_t dst_s_head, const int64_t dst_s_q,
        const int n_q, const int n_kv, const int gqa, const float scale_log2) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    using L = i8qk_smem<D>;
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
    const float  * Ksc_h = Ksc + (size_t) kvh*n_kv;
    const char   * V_h   = V   + kvh*v_nb2;

    auto load_K = [&](const int j) {
        constexpr int chunks_row = D/16;
        for (int i = tid; i < I8QK_BN*chunks_row; i += i8qk_nthreads) {
            const int row = i / chunks_row;
            const int c   = i % chunks_row;
            i8qk_cp_async_16(Ks + row*L::k_stride + c*16, Kq_h + ((size_t) j*I8QK_BN + row)*D + c*16);
        }
        if (tid < I8QK_BN/4) {
            i8qk_cp_async_16(Kss + tid*4, Ksc_h + (size_t) j*I8QK_BN + tid*4);
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

    // Q tile -> int8 rows in shared memory, one f32 scale per row (softmax scale and log2(e) folded in)
    for (int r = 0; r < 16; ++r) {
        const int row = warp*16 + r;
        const int q   = q0 + row;
        float4 x[D/128];
        float amax = 0.0f;
        if (q < n_q) {
            const float * Qr = (const float *) (Q + q*q_nb1 + h*q_nb2);
#pragma unroll
            for (int i = 0; i < D/128; ++i) {
                x[i] = *(const float4 *) (Qr + i*128 + lane*4);
                amax = fmaxf(amax, fmaxf(fmaxf(fabsf(x[i].x), fabsf(x[i].y)), fmaxf(fabsf(x[i].z), fabsf(x[i].w))));
            }
        } else {
#pragma unroll
            for (int i = 0; i < D/128; ++i) {
                x[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
#pragma unroll
        for (int o = WARP_SIZE/2; o > 0; o >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, o));
        }
        const float inv = amax > 0.0f ? 127.0f/amax : 0.0f;
#pragma unroll
        for (int i = 0; i < D/128; ++i) {
            *(uint32_t *) (Qs + row*L::q_stride + i*128 + lane*4) =
                i8qk_pack_s8(i8qk_q8(x[i].x, inv), i8qk_q8(x[i].y, inv), i8qk_q8(x[i].z, inv), i8qk_q8(x[i].w, inv));
        }
        if (lane == 0) {
            Qss[row] = amax/127.0f*scale_log2;
        }
    }
    __syncthreads();
    const float sq0 = Qss[warp*16 + g];
    const float sq1 = Qss[warp*16 + g + 8];

    float O[D/8][4];
#pragma unroll
    for (int i = 0; i < D/8; ++i) {
        O[i][0] = O[i][1] = O[i][2] = O[i][3] = 0.0f;
    }
    float m0 = -INFINITY, m1 = -INFINITY; // running row max (log2 domain), rows g and g + 8
    float l0 = 0.0f,      l1 = 0.0f;      // per-thread partial row sums

    const int8_t * Qw = Qs + (warp*16)*L::q_stride;
    const int lrow = lane % 8;
    const int lmat = lane / 8;

    while (j >= 0) {
        const int jn = next_tile(j);
        const uint8_t flag = fl[j];

        i8qk_cp_async_wait<0>();
        __syncthreads(); // K_j landed; every warp is done with V_{j-1}
        load_V(j);
        i8qk_cp_async_commit();

        // S = Q K^T (int8 MMA, int32 accumulation)
        int S[I8QK_BN/8][4];
#pragma unroll
        for (int i = 0; i < I8QK_BN/8; ++i) {
            S[i][0] = S[i][1] = S[i][2] = S[i][3] = 0;
        }
#pragma unroll
        for (int ks = 0; ks < D/32; ++ks) {
            int A[4];
            i8qk_ldmatrix_x4(A, Qw + (lrow + (lmat & 1)*8)*L::q_stride + ks*32 + (lmat >> 1)*16);
#pragma unroll
            for (int np = 0; np < I8QK_BN/16; ++np) {
                int B[4];
                i8qk_ldmatrix_x4(B, Ks + (np*16 + lrow + (lmat >> 1)*8)*L::k_stride + ks*32 + (lmat & 1)*16);
                i8qk_mma_s8(S[2*np + 0], A, B[0], B[1]);
                i8qk_mma_s8(S[2*np + 1], A, B[2], B[3]);
            }
        }

        // dequantize, mask, online softmax (log2 domain)
        float Sf[I8QK_BN/8][4];
        float mx0 = -INFINITY, mx1 = -INFINITY;
        const int qa = q0 + warp*16 + g;
        const int qb = qa + 8;
#pragma unroll
        for (int nt = 0; nt < I8QK_BN/8; ++nt) {
            const int col = nt*8 + 2*t;
            const float sk0 = Kss[col + 0];
            const float sk1 = Kss[col + 1];
            Sf[nt][0] = float(S[nt][0])*sq0*sk0;
            Sf[nt][1] = float(S[nt][1])*sq0*sk1;
            Sf[nt][2] = float(S[nt][2])*sq1*sk0;
            Sf[nt][3] = float(S[nt][3])*sq1*sk1;
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
#pragma unroll
        for (int i = 0; i < D/8; ++i) {
            O[i][0] *= a0;
            O[i][1] *= a0;
            O[i][2] *= a1;
            O[i][3] *= a1;
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

        // O += P V (f16 MMA, f32 accumulation)
#pragma unroll
        for (int kc = 0; kc < I8QK_BN/16; ++kc) {
#pragma unroll
            for (int dp = 0; dp < D/16; ++dp) {
                int B[4];
                i8qk_ldmatrix_x4_trans(B, Vs + (kc*16 + lrow + (lmat & 1)*8)*L::v_stride + (dp*16 + (lmat >> 1)*8)*2);
                i8qk_mma_f16(O[2*dp + 0], P[kc], B[0], B[1]);
                i8qk_mma_f16(O[2*dp + 1], P[kc], B[2], B[3]);
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
            *(float2 *) (o + i*8 + 2*t) = make_float2(O[i][0]*il0, O[i][1]*il0);
        }
    }
    if (qb < n_q) {
        float * o = dst + qb*dst_s_q + h*dst_s_head;
#pragma unroll
        for (int i = 0; i < D/8; ++i) {
            *(float2 *) (o + i*8 + 2*t) = make_float2(O[i][2]*il1, O[i][3]*il1);
        }
    }
#else
    GGML_UNUSED_VARS(Q, q_nb1, q_nb2, Kq, Ksc, V, v_nb1, v_nb2, mask, mask_s1, flags, n_ktiles, dst, dst_s_head, dst_s_q,
                     n_q, n_kv, gqa, scale_log2);
    NO_DEVICE_CODE;
#endif
}

// ---------------------------------------------------------------------------------------------------------------------
// host

template <int D>
static void ggml_cuda_flash_attn_ext_i8qk_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const int first,
                                               const int heads, void * ws_k, half * ws_v) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool & pool = ctx.pool();

    const int n_q  = (int) Q->ne[1];
    const int n_kv = (int) K->ne[1];
    const int gqa  = (int) (Q->ne[2] / K->ne[2]);
    const size_t n_elem = (size_t) n_kv*D*heads;

    const bool k_conv = K->type != GGML_TYPE_F16;
    const bool v_conv = V->type != GGML_TYPE_F16;
    const int nchunks  = std::min(64, std::max(1, n_kv/256));
    const int chunk    = (n_kv + nchunks - 1)/nchunks;
    const int n_qtiles = (n_q + I8QK_BM - 1)/I8QK_BM;
    const int n_ktiles = n_kv/I8QK_BN;

    // the big buffers live in the caller's f16 K/V reservation (no new VRAM); only scales, means and flags are pooled
    GGML_ASSERT(ws_k != nullptr && ws_v != nullptr);
    GGML_UNUSED(n_elem);
    half   * f16_buf = ws_v;
    int8_t * Kq      = (int8_t *) ws_k;
    // pool allocations in construction order (the VMM pool frees LIFO, i.e. in reverse destruction order)
    ggml_cuda_pool_alloc<float>   ksc(pool, (size_t) n_kv*heads);
    ggml_cuda_pool_alloc<float>   kmean_part(pool, (size_t) nchunks*heads*D);
    ggml_cuda_pool_alloc<float>   kmean(pool, (size_t) heads*D);
    ggml_cuda_pool_alloc<uint8_t> flags(pool, (size_t) n_qtiles*n_ktiles);

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

    // 2. per-channel mean (two deterministic passes), 3. smoothed int8 K + per-token scales
    i8qk_k_mean_partial<D><<<dim3(nchunks, heads), D/2, 0, stream>>>(Kh, k_s_tok, k_s_head, n_kv, chunk, kmean_part.ptr);
    i8qk_k_mean_final<D><<<heads, D, 0, stream>>>(kmean_part.ptr, nchunks, n_kv, kmean.ptr);
    i8qk_k_quant<D><<<dim3(n_kv/8, heads), 256, 0, stream>>>(Kh, k_s_tok, k_s_head, kmean.ptr, n_kv, Kq, ksc.ptr);
    CUDA_CHECK(cudaGetLastError());

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

    const int    id     = ggml_cuda_get_device();
    const size_t nbytes = i8qk_smem<D>::bytes;
    static bool smem_raised[GGML_CUDA_MAX_DEVICES] = {false};
    if (!smem_raised[id]) {
        CUDA_CHECK(cudaFuncSetAttribute(i8qk_flash_attn<D>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) nbytes));
        smem_raised[id] = true;
    }
    const char * Qp = (const char *) Q->data + (size_t) first*gqa*Q->nb[2];
    float * dstp = (float *) dst->data + (size_t) first*gqa*D;
    i8qk_flash_attn<D><<<dim3(n_qtiles, heads*gqa), i8qk_nthreads, nbytes, stream>>>(
        Qp, Q->nb[1], Q->nb[2], Kq, ksc.ptr, Vp, v_nb1, v_nb2,
        mask ? (const half *) mask->data : nullptr, mask ? (int64_t) (mask->nb[1]/sizeof(half)) : 0,
        flags.ptr, n_ktiles, dstp, (int64_t) (dst->nb[1]/sizeof(float)), (int64_t) (dst->nb[2]/sizeof(float)),
        n_q, n_kv, gqa, scale_log2);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_flash_attn_ext_i8qk(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const int first, const int heads,
                                   void * ws_k, half * ws_v) {
    switch (dst->src[0]->ne[0]) {
        case 128: ggml_cuda_flash_attn_ext_i8qk_impl<128>(ctx, dst, first, heads, ws_k, ws_v); break;
        case 256: ggml_cuda_flash_attn_ext_i8qk_impl<256>(ctx, dst, first, heads, ws_k, ws_v); break;
        default: GGML_ABORT("i8qk: unsupported head size");
    }
}
