// INKPREFILL inkfa: prefill attention for INKREL nodes, see inkfa.cuh.
//
// Block (NW warps, 16*NW query rows of one q head h; GQA: K/V head h / g):
//   Q    : registers, f16 A fragments of m16n8k16 (pre-scaled by the softmax scale), rows gr / gr + 8 of each warp
//   K, V : [2][64 keys][128 + 8] f16 in shared memory, filled by cp.async one tile ahead
//   S    = Q K^T  (B fragments by ldmatrix), logits in log2 units: (S + f16(mask + bias)) * log2(e)
//   O   += P V    (P = the S accumulators as f16 A fragments, V fragments by ldmatrix.trans)
// Only the key tiles some row of the block sees are visited (inkfa_tile_flags + inkfa_tile_lists list them from the base mask); a warp whose
// 16 rows see no key of a visited tile skips its MMAs.
// Fragment layout (m16n8k16, lane = 4 gr + tq): A a0 (gr, 2tq..), a1 (gr+8, 2tq..), a2 (gr, 2tq+8..), a3 (gr+8, 2tq+8..);
// B b0 (k 2tq.., n gr), b1 (k 2tq+8.., n gr); C c0,c1 (gr, 2tq..), c2,c3 (gr+8, 2tq..).

#include "inkfa.cuh"
#include "inkrel-bias.cuh"

#include <cstdlib>
#include <cstring>

namespace {

constexpr int   FA_D  = 128;
constexpr int   FA_BN = 64;         // keys per tile
constexpr int   FA_LD = FA_D + 8;   // shared row stride in halves (272 bytes: ldmatrix rows on distinct banks)
constexpr float FA_LOG2E = 1.4426950408889634f;
constexpr size_t FA_SMEM = 4*FA_BN*FA_LD*sizeof(half); // K and V, two buffers each

struct inkfa_params {
    const char * Q; int64_t nbq1, nbq2;     // f32 [D, T, H]
    const char * K; int64_t nbk1, nbk2;     // f16 [D, n_kv, Hkv]
    const char * V; int64_t nbv1, nbv2;
    const half * M; int64_t m_t;            // f16 base mask, halves per query row
    int m_al4;                              // half2 mask loads
    ink_rel_bias b;
    const float * sinks;                    // [H] or nullptr
    const float * gate; int64_t gate_t;     // [D*H, T] or nullptr (floats per token)
    char * O; int64_t nbo1, nbo2;           // f32 dst [D, H, T]
    const int32_t * tiles;                  // [n_rb][NT] live key tiles of each row block
    const int32_t * n_tiles;                // [n_rb]
    int NT, T, H, g, n_kv;
    float scale;
};

static __device__ __forceinline__ uint32_t fa_h2u(const float x, const float y) {
    const half2 h = __floats2half2_rn(x, y);
    return *reinterpret_cast<const uint32_t *>(&h);
}

static __device__ __forceinline__ void fa_mma(float (&c)[4], const uint32_t (&a)[4], const uint32_t b0, const uint32_t b1) {
#ifdef AMPERE_MMA_AVAILABLE
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    GGML_UNUSED_VARS(c, a, b0, b1);
    NO_DEVICE_CODE;
#endif // AMPERE_MMA_AVAILABLE
}

static __device__ __forceinline__ void fa_ldsm4(uint32_t & r0, uint32_t & r1, uint32_t & r2, uint32_t & r3, const half * p) {
#ifdef AMPERE_MMA_AVAILABLE
    const unsigned a = (unsigned) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
#else
    GGML_UNUSED_VARS(r0, r1, r2, r3, p);
    NO_DEVICE_CODE;
#endif // AMPERE_MMA_AVAILABLE
}

static __device__ __forceinline__ void fa_ldsm4_t(uint32_t & r0, uint32_t & r1, uint32_t & r2, uint32_t & r3, const half * p) {
#ifdef AMPERE_MMA_AVAILABLE
    const unsigned a = (unsigned) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
#else
    GGML_UNUSED_VARS(r0, r1, r2, r3, p);
    NO_DEVICE_CODE;
#endif // AMPERE_MMA_AVAILABLE
}

// 16 bytes global -> shared, zero-filled when !ok (src must still be a valid address)
static __device__ __forceinline__ void fa_cp16(half * dst, const void * src, const bool ok) {
#ifdef AMPERE_MMA_AVAILABLE
    const unsigned a = (unsigned) __cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(a), "l"(src), "r"(ok ? 16 : 0));
#else
    GGML_UNUSED_VARS(dst, src, ok);
    NO_DEVICE_CODE;
#endif // AMPERE_MMA_AVAILABLE
}

static __device__ __forceinline__ void fa_commit() {
#ifdef AMPERE_MMA_AVAILABLE
    asm volatile("cp.async.commit_group;");
#endif // AMPERE_MMA_AVAILABLE
}

template <int N>
static __device__ __forceinline__ void fa_wait() {
#ifdef AMPERE_MMA_AVAILABLE
    asm volatile("cp.async.wait_group %0;" :: "n"(N));
#endif // AMPERE_MMA_AVAILABLE
}

// K and V rows [k0, k0 + 64) of KV head kvh into Ks / Vs (rows past n_kv zero-filled)
template <int NT>
static __device__ __forceinline__ void fa_load_tile(half * Ks, half * Vs, const char * Kh, const char * Vh,
        const int64_t nbk1, const int64_t nbv1, const int k0, const int n_kv, const int tid) {
#pragma unroll
    for (int it = 0; it < FA_BN*(FA_D/8)/NT; ++it) {
        const int c   = it*NT + tid;
        const int row = c / (FA_D/8);
        const int col = (c % (FA_D/8))*8;
        const int key = k0 + row;
        const bool ok = key < n_kv;
        const int kr  = ok ? key : 0;
        fa_cp16(Ks + row*FA_LD + col, Kh + (int64_t) kr*nbk1 + col*sizeof(half), ok);
        fa_cp16(Vs + row*FA_LD + col, Vh + (int64_t) kr*nbv1 + col*sizeof(half), ok);
    }
}

// live[rb][tile] = some row of the BM-row block rb sees some key of the tile (a non -inf base mask cell)
template <int BM>
static __global__ void inkfa_tile_flags(const half * __restrict__ M, const int64_t m_t, const int T, const int n_kv, const int NT,
        uint8_t * __restrict__ live) {
    const int tile = blockIdx.x;
    const int rb   = blockIdx.y;
    bool any = false;
    for (int e = threadIdx.x; e < BM*FA_BN; e += blockDim.x) {
        const int row = rb*BM + e / FA_BN;
        const int key = tile*FA_BN + e % FA_BN;
        if (row < T && key < n_kv && __half2float(M[(int64_t) row*m_t + key]) != -INFINITY) {
            any = true;
        }
    }
    any = __syncthreads_or(any);
    if (threadIdx.x == 0) {
        live[(int64_t) rb*NT + tile] = any;
    }
}

// tiles[rb][0, n_tiles[rb]) = the live tiles of row block rb, in increasing order; one warp per row block
static __global__ void inkfa_tile_lists(const uint8_t * __restrict__ live, const int NT, int32_t * __restrict__ tiles,
        int32_t * __restrict__ n_tiles) {
    const int rb   = blockIdx.x;
    const int lane = threadIdx.x;
    int cnt = 0;
    for (int t0 = 0; t0 < NT; t0 += WARP_SIZE) {
        const int  t = t0 + lane;
        const bool l = t < NT && live[(int64_t) rb*NT + t];
        const unsigned bal = __ballot_sync(0xFFFFFFFF, l);
        if (l) {
            tiles[(int64_t) rb*NT + cnt + __popc(bal & ((1u << lane) - 1))] = t;
        }
        cnt += __popc(bal);
    }
    if (lane == 0) {
        n_tiles[rb] = cnt;
    }
}

template <int NW>
static __global__ void __launch_bounds__(NW*WARP_SIZE, 1) inkfa_prefill(const inkfa_params p) {
#ifdef AMPERE_MMA_AVAILABLE
    constexpr int NT = NW*WARP_SIZE;
    constexpr int BM = 16*NW;
    extern __shared__ __align__(16) half fa_sm[];
    half * Ks = fa_sm;                    // [2][BN][LD]
    half * Vs = fa_sm + 2*FA_BN*FA_LD;    // [2][BN][LD]

    const int tid  = threadIdx.x;
    const int warp = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int gr   = lane / 4;
    const int tq   = lane % 4;
    const int h    = blockIdx.x;
    const int rb   = blockIdx.y;
    const int kvh  = h / p.g;

    const int  ia  = rb*BM + warp*16 + gr;
    const int  ib  = ia + 8;
    const bool va_ = ia < p.T;
    const bool vb_ = ib < p.T;
    const int  ta  = va_ ? ia : p.T - 1; // clamped rows: loads stay in bounds, results are never written
    const int  tb  = vb_ ? ib : p.T - 1;

    // Q fragments, pre-scaled, f16
    uint32_t qf[FA_D/16][4];
    {
        const float * qa = (const float *) (p.Q + (int64_t) ta*p.nbq1 + (int64_t) h*p.nbq2);
        const float * qb = (const float *) (p.Q + (int64_t) tb*p.nbq1 + (int64_t) h*p.nbq2);
#pragma unroll
        for (int kk = 0; kk < FA_D/16; ++kk) {
            const int d = kk*16 + 2*tq;
            const float2 a0 = va_ ? *reinterpret_cast<const float2 *>(qa + d)     : make_float2(0.0f, 0.0f);
            const float2 b0 = vb_ ? *reinterpret_cast<const float2 *>(qb + d)     : make_float2(0.0f, 0.0f);
            const float2 a1 = va_ ? *reinterpret_cast<const float2 *>(qa + d + 8) : make_float2(0.0f, 0.0f);
            const float2 b1 = vb_ ? *reinterpret_cast<const float2 *>(qb + d + 8) : make_float2(0.0f, 0.0f);
            qf[kk][0] = fa_h2u(a0.x*p.scale, a0.y*p.scale);
            qf[kk][1] = fa_h2u(b0.x*p.scale, b0.y*p.scale);
            qf[kk][2] = fa_h2u(a1.x*p.scale, a1.y*p.scale);
            qf[kk][3] = fa_h2u(b1.x*p.scale, b1.y*p.scale);
        }
    }

    float o[FA_D/8][4];
#pragma unroll
    for (int j = 0; j < FA_D/8; ++j) {
        o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.0f;
    }
    float ma = -INFINITY, mb = -INFINITY, la = 0.0f, lb = 0.0f;

    const half * mra = p.M + (int64_t) ta*p.m_t;
    const half * mrb = p.M + (int64_t) tb*p.m_t;
    const char * Kh  = p.K + (int64_t) kvh*p.nbk2;
    const char * Vh  = p.V + (int64_t) kvh*p.nbv2;
    const int32_t * tl = p.tiles + (int64_t) rb*p.NT;
    const int cnt = p.n_tiles[rb];

    if (cnt > 0) {
        fa_load_tile<NT>(Ks, Vs, Kh, Vh, p.nbk1, p.nbv1, tl[0]*FA_BN, p.n_kv, tid);
    }
    fa_commit();

    for (int it = 0; it < cnt; ++it) {
        const int buf = it & 1;
        const int k0  = tl[it]*FA_BN;
        if (it + 1 < cnt) {
            fa_load_tile<NT>(Ks + (buf ^ 1)*FA_BN*FA_LD, Vs + (buf ^ 1)*FA_BN*FA_LD, Kh, Vh, p.nbk1, p.nbv1,
                tl[it + 1]*FA_BN, p.n_kv, tid);
        }
        fa_commit();

        // f16(mask + bias) of this lane's 2 rows x 16 keys (keys k0 + 8 j + 2 tq + e); -inf cells need no bias
        float ba[FA_BN/8][2], bb[FA_BN/8][2];
        bool any = false;
#pragma unroll
        for (int j = 0; j < FA_BN/8; ++j) {
            const int c = k0 + 8*j + 2*tq;
            float2 xa = make_float2(-INFINITY, -INFINITY);
            float2 xb = xa;
            if (p.m_al4 && c + 1 < p.n_kv) {
                xa = __half22float2(*reinterpret_cast<const half2 *>(mra + c));
                xb = __half22float2(*reinterpret_cast<const half2 *>(mrb + c));
            } else {
                if (c < p.n_kv)     { xa.x = __half2float(mra[c]);     xb.x = __half2float(mrb[c]); }
                if (c + 1 < p.n_kv) { xa.y = __half2float(mra[c + 1]); xb.y = __half2float(mrb[c + 1]); }
            }
            ba[j][0] = va_ && xa.x != -INFINITY ? ink_rel_bias_masked(xa.x, ink_rel_bias_load(p.b, h, c,     ta)) : -INFINITY;
            ba[j][1] = va_ && xa.y != -INFINITY ? ink_rel_bias_masked(xa.y, ink_rel_bias_load(p.b, h, c + 1, ta)) : -INFINITY;
            bb[j][0] = vb_ && xb.x != -INFINITY ? ink_rel_bias_masked(xb.x, ink_rel_bias_load(p.b, h, c,     tb)) : -INFINITY;
            bb[j][1] = vb_ && xb.y != -INFINITY ? ink_rel_bias_masked(xb.y, ink_rel_bias_load(p.b, h, c + 1, tb)) : -INFINITY;
            any |= ba[j][0] != -INFINITY || ba[j][1] != -INFINITY || bb[j][0] != -INFINITY || bb[j][1] != -INFINITY;
        }
        any = __any_sync(0xFFFFFFFF, any);

        fa_wait<1>(); // this tile's K/V (the next tile's copies may still be in flight)
        __syncthreads();

        if (any) {
            const half * K = Ks + buf*FA_BN*FA_LD;
            const half * V = Vs + buf*FA_BN*FA_LD;

            float s[FA_BN/8][4];
#pragma unroll
            for (int j = 0; j < FA_BN/8; ++j) {
                s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.0f;
                const half * kr = K + (8*j + (lane & 7))*FA_LD + (lane >> 3)*8; // two k-steps per ldmatrix
#pragma unroll
                for (int kk = 0; kk < FA_D/16; kk += 2) {
                    uint32_t b0, b1, b2, b3;
                    fa_ldsm4(b0, b1, b2, b3, kr + kk*16);
                    fa_mma(s[j], qf[kk],     b0, b1);
                    fa_mma(s[j], qf[kk + 1], b2, b3);
                }
            }

            float mxa = -INFINITY, mxb = -INFINITY;
#pragma unroll
            for (int j = 0; j < FA_BN/8; ++j) {
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    s[j][e]     = (s[j][e]     + ba[j][e])*FA_LOG2E;
                    s[j][2 + e] = (s[j][2 + e] + bb[j][e])*FA_LOG2E;
                    mxa = fmaxf(mxa, s[j][e]);
                    mxb = fmaxf(mxb, s[j][2 + e]);
                }
            }
            mxa = fmaxf(mxa, __shfl_xor_sync(0xFFFFFFFF, mxa, 1));
            mxa = fmaxf(mxa, __shfl_xor_sync(0xFFFFFFFF, mxa, 2));
            mxb = fmaxf(mxb, __shfl_xor_sync(0xFFFFFFFF, mxb, 1));
            mxb = fmaxf(mxb, __shfl_xor_sync(0xFFFFFFFF, mxb, 2));
            const float mna = fmaxf(ma, mxa);
            const float mnb = fmaxf(mb, mxb);
            const float ca  = mna == -INFINITY ? 1.0f : exp2f(ma - mna);
            const float cb  = mnb == -INFINITY ? 1.0f : exp2f(mb - mnb);
            const float ua  = mna == -INFINITY ? 0.0f : mna;
            const float ub  = mnb == -INFINITY ? 0.0f : mnb;
            ma = mna;
            mb = mnb;
            la *= ca;
            lb *= cb;
#pragma unroll
            for (int j = 0; j < FA_D/8; ++j) {
                o[j][0] *= ca; o[j][1] *= ca;
                o[j][2] *= cb; o[j][3] *= cb;
            }

            uint32_t pa[FA_BN/16][4];
#pragma unroll
            for (int kk = 0; kk < FA_BN/16; ++kk) {
                const float p00 = exp2f(s[2*kk][0] - ua),     p01 = exp2f(s[2*kk][1] - ua);
                const float p02 = exp2f(s[2*kk][2] - ub),     p03 = exp2f(s[2*kk][3] - ub);
                const float p10 = exp2f(s[2*kk + 1][0] - ua), p11 = exp2f(s[2*kk + 1][1] - ua);
                const float p12 = exp2f(s[2*kk + 1][2] - ub), p13 = exp2f(s[2*kk + 1][3] - ub);
                la += p00 + p01 + p10 + p11;
                lb += p02 + p03 + p12 + p13;
                pa[kk][0] = fa_h2u(p00, p01);
                pa[kk][1] = fa_h2u(p02, p03);
                pa[kk][2] = fa_h2u(p10, p11);
                pa[kk][3] = fa_h2u(p12, p13);
            }

#pragma unroll
            for (int kk = 0; kk < FA_BN/16; ++kk) {
                const half * vr = V + (16*kk + (lane & 7) + ((lane >> 3) & 1)*8)*FA_LD + (lane >> 4)*8;
#pragma unroll
                for (int j = 0; j < FA_D/8; j += 2) {
                    uint32_t b0, b1, b2, b3;
                    fa_ldsm4_t(b0, b1, b2, b3, vr + 8*j);
                    fa_mma(o[j],     pa[kk], b0, b1);
                    fa_mma(o[j + 1], pa[kk], b2, b3);
                }
            }
        }
        __syncthreads(); // buffer buf is refilled by the next iteration's prefetch
    }

    la += __shfl_xor_sync(0xFFFFFFFF, la, 1);
    la += __shfl_xor_sync(0xFFFFFFFF, la, 2);
    lb += __shfl_xor_sync(0xFFFFFFFF, lb, 1);
    lb += __shfl_xor_sync(0xFFFFFFFF, lb, 2);

    // sinks: one more logit per row with value sinks[h] and no V (log2 units here)
    float fa_ = 0.0f, fb_ = 0.0f;
    {
        float Ma = ma, Mb = mb, La = la, Lb = lb, sa = 1.0f, sb = 1.0f;
        if (p.sinks) {
            const float s2 = p.sinks[h]*FA_LOG2E;
            const float M2a = fmaxf(Ma, s2), M2b = fmaxf(Mb, s2);
            sa = Ma == -INFINITY ? 0.0f : exp2f(Ma - M2a);
            sb = Mb == -INFINITY ? 0.0f : exp2f(Mb - M2b);
            La = La*sa + exp2f(s2 - M2a);
            Lb = Lb*sb + exp2f(s2 - M2b);
        }
        fa_ = La > 0.0f ? sa / La : 0.0f;
        fb_ = Lb > 0.0f ? sb / Lb : 0.0f;
    }

#pragma unroll
    for (int i = 0; i < 2; ++i) {
        if (!(i == 0 ? va_ : vb_)) {
            continue;
        }
        const int   t = i == 0 ? ia : ib;
        const float f = i == 0 ? fa_ : fb_;
        float * out = (float *) (p.O + (int64_t) t*p.nbo2 + (int64_t) h*p.nbo1);
        const float * gt = p.gate ? p.gate + (int64_t) t*p.gate_t + (int64_t) h*FA_D : nullptr;
#pragma unroll
        for (int j = 0; j < FA_D/8; ++j) {
            const int d = 8*j + 2*tq;
            float2 r = make_float2(o[j][2*i]*f, o[j][2*i + 1]*f);
            if (gt) {
                r.x *= 1.0f / (1.0f + expf(-gt[d]));
                r.y *= 1.0f / (1.0f + expf(-gt[d + 1]));
            }
            *reinterpret_cast<float2 *>(out + d) = r;
        }
    }
#else
    GGML_UNUSED(p);
    NO_DEVICE_CODE;
#endif // AMPERE_MMA_AVAILABLE
}

static bool fa_aligned(const void * ptr, const int64_t a, const int64_t b, const int64_t n) {
    return (reinterpret_cast<uintptr_t>(ptr) % n) == 0 && a % n == 0 && b % n == 0;
}

// as fa_aligned, for a tensor that may not be allocated yet (graph planning): then its offset in the view source
// (buffers themselves are at least 32-byte aligned)
static bool fa_aligned_t(const ggml_tensor * t, const int64_t n) {
    const uintptr_t off = t->data ? reinterpret_cast<uintptr_t>(t->data) : (t->view_src ? (uintptr_t) t->view_offs : 0);
    return off % n == 0 && t->nb[1] % n == 0 && t->nb[2] % n == 0;
}

// shape and type checks (no data pointers: also used while planning the graph)
static bool inkfa_check(const int cc, const ggml_tensor * dst) {
    if (!ggml_cuda_inkfa_enabled() || !ggml_flash_attn_ext_is_inkrel(dst)) {
        return false;
    }
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || !ampere_mma_available(cc)) {
        return false;
    }
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * snk  = dst->src[4];
    const ggml_tensor * tab  = ggml_flash_attn_ext_inkrel_table(dst);
    const ggml_tensor * rp   = ggml_flash_attn_ext_inkrel_pos(dst);
    const ggml_tensor * gate = ggml_flash_attn_ext_inkrel_gate(dst);
    if (dst->type != GGML_TYPE_F32 || Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16 ||
            !mask || mask->type != GGML_TYPE_F16 || (snk && (snk->type != GGML_TYPE_F32 || !ggml_is_contiguous(snk)))) {
        return false;
    }
    if (dst->src[6] != nullptr || dst->src[9] != nullptr) {
        return false;
    }
    for (int i = 4; i < GGML_MAX_OP_PARAMS / (int) sizeof(int32_t); ++i) {
        if (ggml_get_op_params_i32(dst, i) != 0 && i != GGML_INKREL_PARAM) {
            return false;
        }
    }
    float max_bias, softcap;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || softcap != 0.0f) {
        return false;
    }
    const int64_t T = Q->ne[1], H = Q->ne[2], Hkv = K->ne[2], n_kv = K->ne[1];
    if (Q->ne[0] != FA_D || K->ne[0] != FA_D || V->ne[0] != FA_D || V->ne[1] != n_kv || V->ne[2] != Hkv || n_kv < 1 ||
            Hkv < 1 || H % Hkv != 0 || Q->ne[3] != 1 || K->ne[3] != 1 || V->ne[3] != 1 || T > INT32_MAX/2 || n_kv > INT32_MAX/2) {
        return false;
    }
    if ((H/Hkv)*T <= 16) {
        return false; // decode widths: INKATTN
    }
    if (mask->ne[0] < n_kv || mask->ne[1] < T || mask->ne[2] != 1 || mask->ne[3] != 1 || mask->nb[0] != sizeof(half) ||
            mask->nb[1] % sizeof(half) != 0) {
        return false;
    }
    if (Q->nb[0] != sizeof(float) || !fa_aligned_t(Q, 8) || K->nb[0] != sizeof(half) || !fa_aligned_t(K, 16) ||
            V->nb[0] != sizeof(half) || !fa_aligned_t(V, 16) ||
            dst->nb[0] != sizeof(float) || dst->ne[0] != FA_D || dst->ne[1] != H || dst->ne[2] != T || !fa_aligned_t(dst, 8)) {
        return false;
    }
    if (tab->type != GGML_TYPE_F32 || !ggml_is_contiguous(tab) || tab->ne[0] != H || tab->ne[2] != T || tab->ne[3] != 1 ||
            rp->type != GGML_TYPE_I32 || rp->nb[0] != sizeof(int32_t) || rp->nb[1] % sizeof(int32_t) != 0 ||
            rp->ne[0] != n_kv || rp->ne[1] != T || rp->ne[2] != 1 || rp->ne[3] != 1 || H > 256) {
        return false; // (the limits of ink_rel_bias_make, so that try() takes every node this accepts)
    }
    if (gate && (gate->type != GGML_TYPE_F32 || !ggml_is_contiguous(gate) || gate->ne[0] != FA_D*H || gate->ne[1] != T)) {
        return false;
    }
    return true;
}

template <int NW>
static void inkfa_launch(ggml_backend_cuda_context & ctx, inkfa_params & p, const ggml_tensor * mask) {
    constexpr int BM = 16*NW;
    cudaStream_t stream = ctx.stream();
    const int n_rb = (p.T + BM - 1) / BM;
    ggml_cuda_pool_alloc<int32_t> tiles(ctx.pool(), (size_t) n_rb*p.NT + n_rb);
    ggml_cuda_pool_alloc<uint8_t> live(ctx.pool(), (size_t) n_rb*p.NT);
    p.tiles   = tiles.get();
    p.n_tiles = tiles.get() + (size_t) n_rb*p.NT;
    inkfa_tile_flags<BM><<<dim3(p.NT, n_rb), 256, 0, stream>>>((const half *) mask->data, p.m_t, p.T, p.n_kv, p.NT, live.get());
    CUDA_CHECK(cudaGetLastError());
    inkfa_tile_lists<<<n_rb, WARP_SIZE, 0, stream>>>(live.get(), p.NT, (int32_t *) p.tiles, (int32_t *) p.n_tiles);
    CUDA_CHECK(cudaGetLastError());

    static bool smem_set[GGML_CUDA_MAX_DEVICES] = {false};
    if (!smem_set[ctx.device]) {
        CUDA_CHECK(cudaFuncSetAttribute(inkfa_prefill<NW>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) FA_SMEM));
        smem_set[ctx.device] = true;
    }
    inkfa_prefill<NW><<<dim3(p.H, n_rb), NW*WARP_SIZE, FA_SMEM, stream>>>(p);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

bool ggml_cuda_inkfa_enabled() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_INKFA");
        return !(e && *e && atoi(e) == 0);
    }();
    return on;
}

bool ggml_cuda_inkfa_supported(int device, const ggml_tensor * dst) {
    return inkfa_check(ggml_cuda_info().devices[device].cc, dst);
}

bool ggml_cuda_inkfa_try(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (!inkfa_check(ggml_cuda_info().devices[ctx.device].cc, dst)) {
        return false;
    }
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * snk  = dst->src[4];
    const ggml_tensor * gate = ggml_flash_attn_ext_inkrel_gate(dst);

    inkfa_params p;
    if (!ink_rel_bias_from_inkrel(dst, p.b)) {
        return false;
    }
    p.Q = (const char *) Q->data; p.nbq1 = Q->nb[1]; p.nbq2 = Q->nb[2];
    p.K = (const char *) K->data; p.nbk1 = K->nb[1]; p.nbk2 = K->nb[2];
    p.V = (const char *) V->data; p.nbv1 = V->nb[1]; p.nbv2 = V->nb[2];
    p.M = (const half *) mask->data; p.m_t = mask->nb[1]/sizeof(half);
    p.m_al4 = fa_aligned(mask->data, mask->nb[1], 0, 4);
    p.sinks  = snk ? (const float *) snk->data : nullptr;
    p.gate   = gate ? (const float *) gate->data : nullptr;
    p.gate_t = gate ? gate->nb[1]/sizeof(float) : 0;
    p.O = (char *) dst->data; p.nbo1 = dst->nb[1]; p.nbo2 = dst->nb[2];
    p.T = (int) Q->ne[1]; p.H = (int) Q->ne[2]; p.g = (int) (Q->ne[2]/K->ne[2]); p.n_kv = (int) K->ne[1];
    p.NT = (p.n_kv + FA_BN - 1) / FA_BN;
    memcpy(&p.scale, (const float *) dst->op_params + 0, sizeof(float));

    // 8 warps (128 rows) share each K/V tile (one 69.6 KB block per SM on 100 KB/SM cards either way); 4 warps only when
    // the 8-warp blocks would not cover the SMs. GGML_CUDA_INKFA_NW=4/8 forces one.
    static const int nw_env = [] { const char * e = getenv("GGML_CUDA_INKFA_NW"); return e ? atoi(e) : 0; }();
    const int nsm = ggml_cuda_info().devices[ctx.device].nsm;
    const bool wide = nw_env ? nw_env == 8 : (int64_t) ((p.T + 127)/128)*p.H >= nsm;
    if (wide) {
        inkfa_launch<8>(ctx, p, mask);
    } else {
        inkfa_launch<4>(ctx, p, mask);
    }
    return true;
}
