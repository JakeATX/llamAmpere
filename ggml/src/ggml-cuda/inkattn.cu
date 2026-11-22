// INKATTN (fork): GQA-grouped tensor-core decode attention, see inkattn.cuh.
//
// Layout of one warp step (16 keys of one KV head, all R = g*T <= 16 query rows of that head):
//   S^T[row, key] = Q[row, :] . K[key, :]   mma m16n8k16, A = Q (rows), B = K^T (k = dim, n = key), 2 n-blocks of 8 keys
//   O[row, dv]   += P[row, key] V[key, dv]  mma m16n8k16, A = P (the S^T accumulators reused as f16, no shuffles),
//                                            B = V (k = key, n = dv), 16 n-blocks over D = 128
// The dot product order over dims and the n -> dv assignment are free, so every thread loads 16-byte runs:
//   K: lane holds key n = lane/4 of each n-block and dims [32 s + 8 c, +8) (c = lane % 4) for s = 0..3; Q uses the same
//      dim -> k-slot map (k-step 2s: slots {2c,2c+1} = dims +0,1, {2c+8,2c+9} = +2,3; k-step 2s+1: +4,5 / +6,7)
//   V: lane holds keys {2c, 2c+1, 2c+8, 2c+9} of the tile and dv [16 (lane/4), +16); n-block j of the PV mma is
//      dv = 16 n + j, so the O accumulator of lane holds dv 32 c + j and 32 c + 16 + j
// Precision: f16 Q (pre-scaled), f16 K/V, f32 scores and accumulators, f16 P, like the MMA FA kernel.

#include "inkattn.cuh"

#include <cstdlib>
#include <cstdint>

namespace {

constexpr int INKATTN_D     = 128;
constexpr int INKATTN_WARPS = 4; // warps per block, each an independent run

struct inkattn_params {
    const char * Q; int64_t nbq1, nbq2, nbq3;  // f32 [D, T, H, S]
    const char * K; int64_t nbk1, nbk2, nbk3;  // f16 [D, n_kv, Hkv, S]
    const char * V; int64_t nbv1, nbv2, nbv3;
    const char * M; int64_t nbm1, nbm2, nbm3;  // f16 [>= n_kv, >= T, ne32, ne33]
    int ne32, ne33;
    int mask_al4;        // mask rows 4-byte aligned: half2 loads on full tiles
    const float   * tab; int64_t tab_h1, tab_t; // INKREL bias table f32 [H, rows, T]: floats per rel row, per token
    const int32_t * rp;  int64_t rp_t;          // INKREL rel_pos i32 [n_kv, T]: ints per token
    int n_kv, T, g, H, Hkv;
    int W;               // warps (runs) per (sequence, KV head)
    float scale;
    float  * Op;         // [S][Hkv][W][16][D]
    float2 * Mp;         // [S][Hkv][W][16]   (max, sum)
};

static __device__ __forceinline__ void inkattn_mma(float (&c)[4], const uint32_t a0, const uint32_t a1, const uint32_t a2,
        const uint32_t a3, const uint32_t b0, const uint32_t b1) {
#ifdef AMPERE_MMA_AVAILABLE
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    GGML_UNUSED_VARS(c, a0, a1, a2, a3, b0, b1);
    NO_DEVICE_CODE;
#endif // AMPERE_MMA_AVAILABLE
}

static __device__ __forceinline__ uint32_t inkattn_h2u(const float x, const float y) {
    const half2 h = __floats2half2_rn(x, y);
    return *reinterpret_cast<const uint32_t *>(&h);
}

static __device__ __forceinline__ uint4 inkattn_ld16(const char * p) {
    return __ldg(reinterpret_cast<const uint4 *>(p));
}

template <bool rows16, bool rel>
static __global__ void __launch_bounds__(INKATTN_WARPS*WARP_SIZE)
inkattn_decode(const inkattn_params p) {
#ifdef AMPERE_MMA_AVAILABLE
    constexpr int D = INKATTN_D;
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int wi   = blockIdx.x*INKATTN_WARPS + warp; // run index within (seq, kv head)
    if (wi >= p.W) {
        return;
    }
    const int kvh = blockIdx.y;
    const int seq = blockIdx.z;
    const int c   = lane % 4;
    const int ln  = lane / 4;
    const int R   = p.g*p.T;

    // query rows of this lane: r0 = ln, r1 = ln + 8; row r = token (r / g), q head kvh*g + r % g
    const int r0 = ln;
    const int r1 = ln + 8;

    // Q fragments, pre-scaled, f16: qa[ks][0..3] = A regs of k-step ks
    uint32_t qa[8][4];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int r = i == 0 ? r0 : r1;
        const bool live = r < R && (i == 0 || rows16);
        const float * q = (const float *) (p.Q + seq*p.nbq3 + (int64_t) (r / p.g)*p.nbq1 + (int64_t) (kvh*p.g + r % p.g)*p.nbq2);
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            float4 x0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4 x1 = x0;
            if (live) {
                x0 = *reinterpret_cast<const float4 *>(q + 32*s + 8*c);
                x1 = *reinterpret_cast<const float4 *>(q + 32*s + 8*c + 4);
            }
            // k-step 2s: slots {2c,2c+1} = dims +0,1 (reg 0/1 = row r0/r1), {2c+8,2c+9} = +2,3 (reg 2/3)
            qa[2*s + 0][0 + i] = inkattn_h2u(x0.x*p.scale, x0.y*p.scale);
            qa[2*s + 0][2 + i] = inkattn_h2u(x0.z*p.scale, x0.w*p.scale);
            qa[2*s + 1][0 + i] = inkattn_h2u(x1.x*p.scale, x1.y*p.scale);
            qa[2*s + 1][2 + i] = inkattn_h2u(x1.z*p.scale, x1.w*p.scale);
        }
    }

    // mask rows (clamped to a live row for the padding rows, whose results are never written)
    // INKREL: the bias row of (q head, token) in the table and the token's rel_pos row
    const half * mrow[2];
    const float   * trow[2];
    const int32_t * rrow[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        int r = i == 0 ? r0 : r1;
        r = r < R ? r : R - 1;
        const int h = kvh*p.g + r % p.g;
        mrow[i] = (const half *) (p.M + (int64_t) (seq % p.ne33)*p.nbm3 + (int64_t) (h % p.ne32)*p.nbm2 + (int64_t) (r / p.g)*p.nbm1);
        if constexpr (rel) {
            trow[i] = p.tab + h + (int64_t) (r / p.g)*p.tab_t;
            rrow[i] = p.rp + (int64_t) (r / p.g)*p.rp_t;
        } else {
            trow[i] = nullptr;
            rrow[i] = nullptr;
        }
    }

    const char * Kh = p.K + seq*p.nbk3 + kvh*p.nbk2;
    const char * Vh = p.V + seq*p.nbv3 + kvh*p.nbv2;

    const int NT = (p.n_kv + 15) / 16;
    const int t0 = (int) ((int64_t) wi*NT / p.W);
    const int t1 = (int) ((int64_t) (wi + 1)*NT / p.W);

    float O[16][4];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        O[j][0] = O[j][1] = O[j][2] = O[j][3] = 0.0f;
    }
    float mx[2] = {-INFINITY, -INFINITY};
    float sm[2] = {0.0f, 0.0f};

    for (int t = t0; t < t1; ++t) {
        const int k0 = 16*t;
        const bool full = k0 + 16 <= p.n_kv;

        // K: keys k0 + 8 nb + ln, dims [32 s + 8 c, +8)
        uint4 kr[2][4];
#pragma unroll
        for (int nb = 0; nb < 2; ++nb) {
            int key = k0 + 8*nb + ln;
            key = key < p.n_kv ? key : p.n_kv - 1;
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                kr[nb][s] = inkattn_ld16(Kh + (int64_t) key*p.nbk1 + (32*s + 8*c)*sizeof(half));
            }
        }
        // V: keys k0 + {2c, 2c+1, 2c+8, 2c+9}, dv [16 ln, +16)
        uint4 vr[4][2];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int key = k0 + 2*c + (i & 1) + 8*(i >> 1);
            key = key < p.n_kv ? key : p.n_kv - 1;
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                vr[i][u] = inkattn_ld16(Vh + (int64_t) key*p.nbv1 + (16*ln + 8*u)*sizeof(half));
            }
        }
        // mask: rows r0/r1, keys k0 + 8 nb + 2c + {0, 1}
        float2 mk[2][2];
#pragma unroll
        for (int nb = 0; nb < 2; ++nb) {
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                if (i == 1 && !rows16) {
                    mk[nb][i] = make_float2(0.0f, 0.0f);
                    continue;
                }
                const int key = k0 + 8*nb + 2*c;
                if (full && p.mask_al4) {
                    mk[nb][i] = __half22float2(*reinterpret_cast<const half2 *>(mrow[i] + key));
                } else {
                    mk[nb][i].x = key + 0 < p.n_kv ? __half2float(mrow[i][key + 0]) : -INFINITY;
                    mk[nb][i].y = key + 1 < p.n_kv ? __half2float(mrow[i][key + 1]) : -INFINITY;
                }
                if constexpr (rel) {
                    // f16(mask + bias): the value the dense path's per-head f16 mask holds
                    if (key + 0 < p.n_kv) {
                        mk[nb][i].x = __half2float(__float2half_rn(mk[nb][i].x + trow[i][(int64_t) rrow[i][key + 0]*p.tab_h1]));
                    }
                    if (key + 1 < p.n_kv) {
                        mk[nb][i].y = __half2float(__float2half_rn(mk[nb][i].y + trow[i][(int64_t) rrow[i][key + 1]*p.tab_h1]));
                    }
                }
            }
        }

        // S^T = Q K^T
        float S[2][4];
#pragma unroll
        for (int nb = 0; nb < 2; ++nb) {
            S[nb][0] = S[nb][1] = S[nb][2] = S[nb][3] = 0.0f;
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                const uint4 kk = kr[nb][s];
                inkattn_mma(S[nb], qa[2*s + 0][0], qa[2*s + 0][1], qa[2*s + 0][2], qa[2*s + 0][3], kk.x, kk.y);
                inkattn_mma(S[nb], qa[2*s + 1][0], qa[2*s + 1][1], qa[2*s + 1][2], qa[2*s + 1][3], kk.z, kk.w);
            }
            S[nb][0] += mk[nb][0].x; S[nb][1] += mk[nb][0].y;
            S[nb][2] += mk[nb][1].x; S[nb][3] += mk[nb][1].y;
        }

        // online softmax, rows r0 (regs 0,1) and r1 (regs 2,3); the 4 lanes of a row differ in c (xor 1, 2)
        float alpha[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            float m = fmaxf(fmaxf(S[0][2*i], S[0][2*i + 1]), fmaxf(S[1][2*i], S[1][2*i + 1]));
            m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFF, m, 1));
            m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFF, m, 2));
            const float mn = fmaxf(mx[i], m);
            const float mu = mn == -INFINITY ? 0.0f : mn;
            alpha[i] = __expf(mx[i] - mu);
            mx[i] = mn;
#pragma unroll
            for (int nb = 0; nb < 2; ++nb) {
                S[nb][2*i + 0] = __expf(S[nb][2*i + 0] - mu);
                S[nb][2*i + 1] = __expf(S[nb][2*i + 1] - mu);
            }
            sm[i] = sm[i]*alpha[i] + (S[0][2*i] + S[0][2*i + 1] + S[1][2*i] + S[1][2*i + 1]);
        }
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            O[j][0] *= alpha[0]; O[j][1] *= alpha[0];
            O[j][2] *= alpha[1]; O[j][3] *= alpha[1];
        }

        // P as the A operand: slots {2c,2c+1} = n-block 0, {2c+8,2c+9} = n-block 1
        const uint32_t pa0 = inkattn_h2u(S[0][0], S[0][1]);
        const uint32_t pa1 = inkattn_h2u(S[0][2], S[0][3]);
        const uint32_t pa2 = inkattn_h2u(S[1][0], S[1][1]);
        const uint32_t pa3 = inkattn_h2u(S[1][2], S[1][3]);

        // O += P V: n-block j = dv 16 n + j; b0 = keys (2c, 2c+1), b1 = keys (2c+8, 2c+9) at dv 16 ln + j
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            const uint32_t * va = reinterpret_cast<const uint32_t *>(&vr[0][u]);
            const uint32_t * vb = reinterpret_cast<const uint32_t *>(&vr[1][u]);
            const uint32_t * vc = reinterpret_cast<const uint32_t *>(&vr[2][u]);
            const uint32_t * vd = reinterpret_cast<const uint32_t *>(&vr[3][u]);
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                const int j = 8*u + 2*w;
                inkattn_mma(O[j + 0], pa0, pa1, pa2, pa3, __byte_perm(va[w], vb[w], 0x5410), __byte_perm(vc[w], vd[w], 0x5410));
                inkattn_mma(O[j + 1], pa0, pa1, pa2, pa3, __byte_perm(va[w], vb[w], 0x7632), __byte_perm(vc[w], vd[w], 0x7632));
            }
        }
    }

#pragma unroll
    for (int i = 0; i < 2; ++i) {
        sm[i] += __shfl_xor_sync(0xFFFFFFFF, sm[i], 1);
        sm[i] += __shfl_xor_sync(0xFFFFFFFF, sm[i], 2);
    }

    const int64_t base = ((int64_t) (seq*p.Hkv + kvh)*p.W + wi)*16;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int r = i == 0 ? r0 : r1;
        if (r >= R || (i == 1 && !rows16)) {
            continue;
        }
        float * o = p.Op + (base + r)*D;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            o[32*c + j]      = O[j][2*i + 0];
            o[32*c + 16 + j] = O[j][2*i + 1];
        }
        if (c == 0) {
            p.Mp[base + r] = make_float2(mx[i], sm[i]);
        }
    }
#else
    GGML_UNUSED(p);
    NO_DEVICE_CODE;
#endif // AMPERE_MMA_AVAILABLE
}

// one block per (q head, token, sequence), D threads
static __global__ void inkattn_combine(const float * __restrict__ Op, const float2 * __restrict__ Mp, const float * __restrict__ sinks,
        char * __restrict__ dst, const int64_t nbd1, const int64_t nbd2, const int64_t nbd3, float * __restrict__ lse,
        const float * __restrict__ gate, const int64_t gate_t, const int W, const int g, const int Hkv) {
    constexpr int D = INKATTN_D;
    const int h   = blockIdx.x;
    const int j   = blockIdx.y;
    const int seq = blockIdx.z;
    const int H   = gridDim.x;
    const int T   = gridDim.y;
    const int tid = threadIdx.x;
    const int kvh = h / g;
    const int r   = j*g + h % g;

    const int64_t base = (int64_t) (seq*Hkv + kvh)*W*16 + r;

    float M = -INFINITY;
    for (int w = 0; w < W; ++w) {
        M = fmaxf(M, Mp[base + (int64_t) w*16].x);
    }
    float L = 0.0f;
    float acc = 0.0f;
    if (M != -INFINITY) {
        for (int w = 0; w < W; ++w) {
            const float2 ms = Mp[base + (int64_t) w*16];
            if (ms.x == -INFINITY) {
                continue;
            }
            const float f = __expf(ms.x - M);
            L   += f*ms.y;
            acc += f*Op[(base + (int64_t) w*16)*D + tid];
        }
    }
    if (sinks) {
        const float s  = sinks[h];
        const float M2 = fmaxf(M, s);
        const float f  = M == -INFINITY ? 0.0f : __expf(M - M2);
        L   = L*f + __expf(s - M2);
        acc = acc*f;
        M   = M2;
    }
    float * d = (float *) (dst + seq*nbd3 + j*nbd2 + h*nbd1);
    float res = L > 0.0f ? acc / L : 0.0f;
    if (gate) { // INKREL output gate: out * sigmoid(gate), as GGML_OP_MUL(out, GGML_UNARY_OP_SIGMOID(gate))
        res = res * (1.0f / (1.0f + expf(-gate[(int64_t) j*gate_t + h*D + tid])));
    }
    d[tid] = res;
    if (lse && tid == 0) {
        lse[((int64_t) seq*T + j)*H + h] = L > 0.0f ? M + logf(L) : -INFINITY;
    }
}

static int inkattn_mode() {
    static const int mode = [] {
        const char * e = getenv("GGML_CUDA_INKATTN");
        return e && *e ? atoi(e) : 1;
    }();
    return mode;
}

static int inkattn_env_int(const char * name, const int def) {
    const char * e = getenv(name);
    return e && *e ? atoi(e) : def;
}

static bool inkattn_aligned(const void * ptr, const int64_t a, const int64_t b, const int64_t c, const int64_t n) {
    return (reinterpret_cast<uintptr_t>(ptr) % n) == 0 && a % n == 0 && b % n == 0 && c % n == 0;
}

} // namespace

// can INKATTN take dst on a device of compute capability cc (mode: GGML_CUDA_INKATTN scope; INKREL nodes ignore it)
static bool inkattn_check(const int cc, const ggml_tensor * dst, const int mode) {
    const bool inkrel = ggml_flash_attn_ext_is_inkrel(dst);
    if (mode == 0 && !inkrel) {
        return false;
    }
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * snk  = dst->src[4];

    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || !ampere_mma_available(cc)) {
        return false;
    }
    if (dst->type != GGML_TYPE_F32 || Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16 ||
            !mask || mask->type != GGML_TYPE_F16 || (snk && snk->type != GGML_TYPE_F32)) {
        return false;
    }
    if (Q->ne[0] != INKATTN_D || K->ne[0] != INKATTN_D || V->ne[0] != INKATTN_D || K->ne[1] != V->ne[1] || K->ne[2] != V->ne[2] ||
            K->ne[1] < 1 || Q->ne[2] % K->ne[2] != 0 || K->ne[3] != Q->ne[3] || V->ne[3] != Q->ne[3]) {
        return false;
    }
    // fork extras this kernel does not read: SJ-KVaRN, QSA / LSA block lists, kv_rows, LSA virtual keys; any op param
    // past the precision (n_kv_max hint, sparse modes, ...)
    for (int i = 5; i <= 8; ++i) {
        if (dst->src[i] != nullptr && !(inkrel && i != 6)) {
            return false;
        }
    }
    ggml_tensor * lse = ggml_flash_attn_ext_get_lse(dst);
    if (dst->src[9] != nullptr && lse == nullptr) {
        return false;
    }
    for (int i = 4; i < GGML_MAX_OP_PARAMS / (int) sizeof(int32_t); ++i) {
        if (ggml_get_op_params_i32(dst, i) != 0 && !(inkrel && i == GGML_INKREL_PARAM)) {
            return false;
        }
    }
    float max_bias, softcap;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || softcap != 0.0f) {
        return false;
    }

    const int T   = (int) Q->ne[1];
    const int H   = (int) Q->ne[2];
    const int Hkv = (int) K->ne[2];
    const int S   = (int) Q->ne[3];
    const int g   = H / Hkv;
    const int n_kv = (int) K->ne[1];
    if (g*T > 16 || mask->ne[0] < n_kv || mask->ne[1] < T || H % mask->ne[2] != 0 || S % mask->ne[3] != 0) {
        return false;
    }
    // scope: 1 = per-head masks only (the Inkling rel bias: dense decode, SWA, INKREL near, INKLSA compact);
    // 2 = also LSE nodes with a broadcast mask (INKREL far); 3 = also plain broadcast masks (any model)
    if (!(inkrel || mask->ne[2] > 1 || (lse != nullptr && mode >= 2) || mode >= 3)) {
        return false;
    }
    if (!inkattn_aligned(Q->data, Q->nb[1], Q->nb[2], Q->nb[3], 16) || Q->nb[0] != sizeof(float) ||
            !inkattn_aligned(K->data, K->nb[1], K->nb[2], K->nb[3], 16) || K->nb[0] != sizeof(half) ||
            !inkattn_aligned(V->data, V->nb[1], V->nb[2], V->nb[3], 16) || V->nb[0] != sizeof(half) ||
            mask->nb[0] != sizeof(half)) {
        return false;
    }
    if (lse && (!ggml_is_contiguous(lse) || lse->ne[0] != H || lse->ne[1] != T || lse->ne[2] != S)) {
        return false;
    }
    if (inkrel) {
        const ggml_tensor * tab  = ggml_flash_attn_ext_inkrel_table(dst);
        const ggml_tensor * rp   = ggml_flash_attn_ext_inkrel_pos(dst);
        const ggml_tensor * gate = ggml_flash_attn_ext_inkrel_gate(dst);
        if (lse || S != 1 || mask->ne[2] != 1 || mask->ne[3] != 1 ||
                tab->type != GGML_TYPE_F32 || !ggml_is_contiguous(tab) || tab->ne[0] != H || tab->ne[2] != T ||
                rp->type != GGML_TYPE_I32 || rp->nb[0] != sizeof(int32_t) || rp->ne[0] != n_kv || rp->ne[1] != T ||
                (gate && (gate->type != GGML_TYPE_F32 || !ggml_is_contiguous(gate) || gate->ne[0] != (int64_t) INKATTN_D*H ||
                          gate->ne[1] != T))) {
            return false;
        }
    }
    return true;
}

bool ggml_cuda_inkattn_supported(int device, const ggml_tensor * dst) {
    return inkattn_check(ggml_cuda_info().devices[device].cc, dst, inkattn_mode());
}

bool ggml_cuda_inkattn_try(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (!inkattn_check(ggml_cuda_info().devices[ctx.device].cc, dst, inkattn_mode())) {
        return false;
    }
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * snk  = dst->src[4];
    ggml_tensor * lse = ggml_flash_attn_ext_get_lse(dst);
    const bool inkrel = ggml_flash_attn_ext_is_inkrel(dst);
    float scale;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    const int T    = (int) Q->ne[1];
    const int H    = (int) Q->ne[2];
    const int Hkv  = (int) K->ne[2];
    const int S    = (int) Q->ne[3];
    const int g    = H / Hkv;
    const int n_kv = (int) K->ne[1];


    const bool rows16 = g*T > 8;
    const int  nsm    = ggml_cuda_info().devices[ctx.device].nsm;

    // runs per (seq, KV head): fill the card (target warps per SM) but keep >= tpw tiles per run
    static const int tgt_wps = inkattn_env_int("GGML_CUDA_INKATTN_WPS", 16);
    static const int min_tpw = inkattn_env_int("GGML_CUDA_INKATTN_TPW", 4);
    const int NT = (n_kv + 15) / 16;
    int W = (nsm*tgt_wps + S*Hkv - 1) / (S*Hkv);
    W = std::min(W, std::max(1, NT / std::max(1, min_tpw)));
    W = std::max(W, 1);

    ggml_cuda_pool_alloc<float>  Op(ctx.pool(), (size_t) S*Hkv*W*16*INKATTN_D);
    ggml_cuda_pool_alloc<float2> Mp(ctx.pool(), (size_t) S*Hkv*W*16);

    inkattn_params p;
    p.Q = (const char *) Q->data; p.nbq1 = Q->nb[1]; p.nbq2 = Q->nb[2]; p.nbq3 = Q->nb[3];
    p.K = (const char *) K->data; p.nbk1 = K->nb[1]; p.nbk2 = K->nb[2]; p.nbk3 = K->nb[3];
    p.V = (const char *) V->data; p.nbv1 = V->nb[1]; p.nbv2 = V->nb[2]; p.nbv3 = V->nb[3];
    p.M = (const char *) mask->data; p.nbm1 = mask->nb[1]; p.nbm2 = mask->nb[2]; p.nbm3 = mask->nb[3];
    p.ne32 = (int) mask->ne[2]; p.ne33 = (int) mask->ne[3];
    p.mask_al4 = inkattn_aligned(mask->data, mask->nb[1], mask->nb[2], mask->nb[3], 4);
    p.n_kv = n_kv; p.T = T; p.g = g; p.H = H; p.Hkv = Hkv; p.W = W; p.scale = scale;
    p.Op = Op.get(); p.Mp = Mp.get();
    p.tab = nullptr; p.tab_h1 = 0; p.tab_t = 0; p.rp = nullptr; p.rp_t = 0;
    const ggml_tensor * gate = nullptr;
    if (inkrel) {
        const ggml_tensor * tab = ggml_flash_attn_ext_inkrel_table(dst);
        const ggml_tensor * rp  = ggml_flash_attn_ext_inkrel_pos(dst);
        gate = ggml_flash_attn_ext_inkrel_gate(dst);
        p.tab = (const float *) tab->data; p.tab_h1 = tab->nb[1]/sizeof(float); p.tab_t = tab->nb[2]/sizeof(float);
        p.rp  = (const int32_t *) rp->data; p.rp_t = rp->nb[1]/sizeof(int32_t);
    }

    cudaStream_t stream = ctx.stream();
    const dim3 grid((W + INKATTN_WARPS - 1)/INKATTN_WARPS, Hkv, S);
    if (inkrel) {
        if (rows16) {
            inkattn_decode<true, true><<<grid, INKATTN_WARPS*WARP_SIZE, 0, stream>>>(p);
        } else {
            inkattn_decode<false, true><<<grid, INKATTN_WARPS*WARP_SIZE, 0, stream>>>(p);
        }
    } else if (rows16) {
        inkattn_decode<true, false><<<grid, INKATTN_WARPS*WARP_SIZE, 0, stream>>>(p);
    } else {
        inkattn_decode<false, false><<<grid, INKATTN_WARPS*WARP_SIZE, 0, stream>>>(p);
    }
    CUDA_CHECK(cudaGetLastError());

    inkattn_combine<<<dim3(H, T, S), INKATTN_D, 0, stream>>>(Op.get(), Mp.get(), snk ? (const float *) snk->data : nullptr,
        (char *) dst->data, dst->nb[1], dst->nb[2], dst->nb[3], lse ? (float *) lse->data : nullptr,
        gate ? (const float *) gate->data : nullptr, gate ? (int64_t) (gate->nb[1]/sizeof(float)) : 0, W, g, Hkv);
    CUDA_CHECK(cudaGetLastError());
    return true;
}
