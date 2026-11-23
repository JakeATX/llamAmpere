#pragma once
// INKNVFP4K: NVFP4 MoE decode kernels for sm_86 (no FP4 hardware: register LUT via byte_perm to int8, dp4a against q8_1).
// Two launches per MoE visit instead of quantize + mmvq(gate_up) + SITU + quantize + mmvq(down) + weighted sum:
//   k_nvfp4moe_gu : x -> q8_1 (in the CTA), gate_up matvec, per-expert scale, SITU, SITU -> q8_1 (written for the down)
//   k_nvfp4moe_dn1: down matvec of the 8 routed experts of one token, per-expert scale, routing weight and expert sum
//   k_nvfp4moe_dnm: down matvec for 2..16 tokens (each routed expert read once per CTA), per-slot output
// Weights stay in the ggml NVFP4 layout (36-byte blocks: 4 UE4M3 scales + 32 code bytes; no repack, see lanes/INKNVFP4K.md).
// Lanes own half-blocks (32 weights = one q8_1 block): lane j of a row group of L lanes takes half-blocks j, j + L, ...
// At one token the partial sums are added in the order of the generic MMVQ kernel (mul_mat_vec_q, small_k, 4 warps,
// vec_dot_nvfp4_q8_1) and of quantize_q8_1 / k_ink_situ / k_ink_moe_wsum, so the result is bit-identical to that path.

#include "nvfp4moe.cuh"

// 16-entry int8 table lookup of 8 nibbles (get_int_from_table_16 with the table in registers): .x = low nibbles, .y = high
static __device__ __forceinline__ int2 nvfp4moe_lut(const uint32_t q4, const uint32_t t0, const uint32_t t1,
                                                    const uint32_t t2, const uint32_t t3) {
    const uint32_t sel = 0x32103210 | ((q4 & 0x88888888) >> 1);
    uint32_t tmp[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint32_t sh = 16*i;
        const uint32_t lo = __byte_perm(t0, t1, q4 >> sh);
        const uint32_t hi = __byte_perm(t2, t3, q4 >> sh);
        tmp[i] = __byte_perm(lo, hi, sel >> sh);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
}

// One half-block of weights (codes c[0..3], the two UE4M3 scales in sc) against one q8_1 block (qs ints q, d8):
// vec_dot_nvfp4_q8_1 for the two sub-blocks of the half, same operations in the same order.
static __device__ __forceinline__ float nvfp4moe_hb_dot(const uint4 c, const uint32_t sc, const int * q, const float d8,
        const float * ue4m3, const uint32_t t0, const uint32_t t1, const uint32_t t2, const uint32_t t3) {
    float sum = 0.0f;
    {
        const int2 v0 = nvfp4moe_lut(c.x, t0, t1, t2, t3);
        const int2 v1 = nvfp4moe_lut(c.y, t0, t1, t2, t3);
        int sumi = ggml_cuda_dp4a(v0.x, q[0], 0);
        sumi = ggml_cuda_dp4a(v0.y, q[2], sumi);
        sumi = ggml_cuda_dp4a(v1.x, q[1], sumi);
        sumi = ggml_cuda_dp4a(v1.y, q[3], sumi);
        const float d = ue4m3[sc & 0xFF] * d8;
        sum += d * float(sumi);
    }
    {
        const int2 v0 = nvfp4moe_lut(c.z, t0, t1, t2, t3);
        const int2 v1 = nvfp4moe_lut(c.w, t0, t1, t2, t3);
        int sumi = ggml_cuda_dp4a(v0.x, q[4], 0);
        sumi = ggml_cuda_dp4a(v0.y, q[6], sumi);
        sumi = ggml_cuda_dp4a(v1.x, q[5], sumi);
        sumi = ggml_cuda_dp4a(v1.y, q[7], sumi);
        const float d = ue4m3[(sc >> 8) & 0xFF] * d8;
        sum += d * float(sumi);
    }
    return sum;
}

// Decoded half-block (8 int8x4 words + the two sub-block scales), for reuse across tokens
struct nvfp4moe_hbd {
    int v[8];
    float da, db;
};
static __device__ __forceinline__ nvfp4moe_hbd nvfp4moe_decode(const uint4 c, const uint32_t sc, const float * ue4m3,
        const uint32_t t0, const uint32_t t1, const uint32_t t2, const uint32_t t3) {
    nvfp4moe_hbd r;
    const int2 a0 = nvfp4moe_lut(c.x, t0, t1, t2, t3), a1 = nvfp4moe_lut(c.y, t0, t1, t2, t3);
    const int2 b0 = nvfp4moe_lut(c.z, t0, t1, t2, t3), b1 = nvfp4moe_lut(c.w, t0, t1, t2, t3);
    r.v[0] = a0.x; r.v[1] = a0.y; r.v[2] = a1.x; r.v[3] = a1.y;
    r.v[4] = b0.x; r.v[5] = b0.y; r.v[6] = b1.x; r.v[7] = b1.y;
    r.da = ue4m3[sc & 0xFF];
    r.db = ue4m3[(sc >> 8) & 0xFF];
    return r;
}
// same terms as nvfp4moe_hb_dot
static __device__ __forceinline__ float nvfp4moe_hbd_dot(const nvfp4moe_hbd & w, const block_q8_1 * yb) {
    const int * q = (const int *) yb->qs;
    const float d8 = __low2float(yb->ds);
    float sum = 0.0f;
    int sumi = ggml_cuda_dp4a(w.v[0], q[0], 0);
    sumi = ggml_cuda_dp4a(w.v[1], q[2], sumi);
    sumi = ggml_cuda_dp4a(w.v[2], q[1], sumi);
    sumi = ggml_cuda_dp4a(w.v[3], q[3], sumi);
    sum += (w.da * d8) * float(sumi);
    sumi = ggml_cuda_dp4a(w.v[4], q[4], 0);
    sumi = ggml_cuda_dp4a(w.v[5], q[6], sumi);
    sumi = ggml_cuda_dp4a(w.v[6], q[5], sumi);
    sumi = ggml_cuda_dp4a(w.v[7], q[7], sumi);
    sum += (w.db * d8) * float(sumi);
    return sum;
}

// Half-block hb of a row (row base 4-byte aligned): 16 code bytes at 36*(hb/2) + 4 + 16*(hb%2), scales at 36*(hb/2) + 2*(hb%2)
static __device__ __forceinline__ void nvfp4moe_load_hb(const char * __restrict__ row, const int hb, uint4 & c, uint32_t & sc) {
    const char * blk = row + 36*(hb >> 1);
    const uint32_t * cp = (const uint32_t *) (blk + 4 + 16*(hb & 1));
    c.x = __ldg(cp + 0);
    c.y = __ldg(cp + 1);
    c.z = __ldg(cp + 2);
    c.w = __ldg(cp + 3);
    sc = __ldg((const uint16_t *) (blk + 2*(hb & 1)));
}

static __device__ __forceinline__ void nvfp4moe_fill_ue4m3(float * lut) {
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        lut[i] = ggml_cuda_ue4m3_to_fp32((uint8_t) i);
    }
}

// quantize_q8_1_elem for one block of 32 values held one per lane (warp-wide; same expressions, bit-identical)
static __device__ __forceinline__ void nvfp4moe_quant_block(const float xi, block_q8_1 * y, const int lane) {
    float amax = fabsf(xi);
    amax = warp_reduce_max<QK8_1>(amax);
    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
    y->qs[lane] = q;
    const float sum = warp_reduce_sum<QK8_1>((float) q);
    if (lane == 0) {
        y->ds = make_half2(d, d * sum);
    }
}

// k_ink_situ's per-element expressions
static __device__ __forceinline__ float nvfp4moe_situ(const float gv, const float uv, const ink_situ_params & p) {
    const float a1 = p.ga_s*gv + p.ga_b;
    const float a2 = tanhf(a1);
    const float a3 = p.gb_s*a2 + p.gb_b;
    const float sg = 1.0f / (1.0f + expf(-gv));
    const float a4 = p.sig_first ? sg*a3 : a3*sg;
    const float u1 = p.ua_s*uv + p.ua_b;
    const float u2 = tanhf(u1);
    const float u3 = p.ub_s*u2 + p.ub_b;
    return p.act_first ? a4*u3 : u3*a4;
}

// The routed experts of the batch: slot s = t*n_used + k. A CTA of slot s works for expert e = ids[s] only if s is the
// first slot with e; it then serves every token that routes to e. Returns the token count (0: not the first slot).
// ids of one token must be distinct (top-k routing): n never exceeds T <= max_n, a repeat traps instead of overrunning
static __device__ __forceinline__ int nvfp4moe_gather(const nvfp4moe_params & P, int * s_ids, int * tok_t, int * tok_k,
        int & e_out, const int max_n) {
    const int n_slots = P.T*P.n_used;
    for (int s = threadIdx.x; s < n_slots; s += blockDim.x) {
        const int t = s / P.n_used;
        s_ids[s] = P.ids[t*P.ids_s1 + (s - t*P.n_used)];
    }
    __syncthreads();
    const int s0 = blockIdx.y;
    const int e  = s_ids[s0];
    e_out = e;
    for (int s = 0; s < s0; ++s) {
        if (s_ids[s] == e) {
            return 0;
        }
    }
    int n = 0;
    for (int s = s0; s < n_slots; ++s) {
        if (s_ids[s] == e) {
            if (n >= max_n) {
                __trap();
            }
            if (threadIdx.x == 0) {
                tok_t[n] = s / P.n_used;
                tok_k[n] = s % P.n_used;
            }
            n++;
        }
    }
    __syncthreads();  // n is uniform over the CTA (all threads scan the same s_ids), so the barrier is safe
    return n;
}

// ---------------------------------------------------------------------------------------------------------------------
// gate_up + scale + SITU -> q8_1. grid (F/32, T*n_used), 256 threads: 32 row groups of 8 lanes, group g = feature f0 + g
// (gate row f, up row F + f), lane j owns half-blocks j + 8m. K = 2304 here: 72 half-blocks = 9 per lane.
// x of T tokens -> q8_1 once per visit (T > 1, GGML_CUDA_NVFP4MOE_PREQ): warp w of CTA b quantizes block 8b + w with the
// expressions of the in-CTA quantization below (bit-identical), instead of every gate_up CTA quantizing its tokens again
static __global__ void __launch_bounds__(256) k_nvfp4moe_xq(const nvfp4moe_params P) {
    const int nb_k = P.K / QK8_1;
    const int b    = blockIdx.x*8 + (threadIdx.x >> 5);
    if (b >= P.T*nb_k) {
        return;  // warp-uniform
    }
    const int t = b / nb_k;
    nvfp4moe_quant_block(P.x[t*P.x_s2 + (b - t*nb_k)*QK8_1 + (threadIdx.x & 31)], P.xq + b, threadIdx.x & 31);
}

// GO (GGML_CUDA_NVFP4MOE_GORD, T > 1): every token's sums in the generic order (mul_mat_vec_q_moe: lane l holds
// 0 + p[l] + p[l+32] + p[l+64], xor 16..1 butterfly), so each token is bit-identical to the generic path; else sequential
template <int MAXM, int NHB, bool PREQ = false, bool GO = false>  // NHB: half-blocks per lane (K/32/8); PREQ: x q8_1 from P.xq (k_nvfp4moe_xq)
static __global__ void __launch_bounds__(256) k_nvfp4moe_gu(const nvfp4moe_params P) {
    constexpr int L = 8;
    extern __shared__ __align__(16) char smem_raw[];
    __shared__ float ue4m3[256];
    __shared__ int   s_ids[NVFP4MOE_MAX_TOK*8];  // n_used == 8
    __shared__ int   tok_t[MAXM], tok_k[MAXM];
    __shared__ float act_s[MAXM][32];
    block_q8_1 * xq = (block_q8_1 *) smem_raw;  // [MAXM][K/32]

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = tid / L, j = tid % L;
    const int nb_k = P.K / QK8_1;
    const int f = blockIdx.x*32 + g;

    int e, n_tok;
    if constexpr (MAXM == 1) {
        e = P.ids[(blockIdx.y / P.n_used)*P.ids_s1 + blockIdx.y % P.n_used];
        n_tok = 1;
        if (tid == 0) { tok_t[0] = blockIdx.y / P.n_used; tok_k[0] = blockIdx.y % P.n_used; }
    } else {
        n_tok = nvfp4moe_gather(P, s_ids, tok_t, tok_k, e, MAXM);
        if (n_tok == 0) {
            return;
        }
    }

    // weight loads first: the x quantization below runs under their latency
    const char * rg = P.w_gu + e*P.gu_nb2 + (int64_t) f*P.gu_nb1;
    const char * ru = rg + (int64_t) P.F*P.gu_nb1;
    uint4 cg[NHB], cu[NHB];
    uint32_t sg[NHB], su[NHB];
#pragma unroll
    for (int m = 0; m < NHB; ++m) {
        nvfp4moe_load_hb(rg, j + L*m, cg[m], sg[m]);
        nvfp4moe_load_hb(ru, j + L*m, cu[m], su[m]);
    }
    const float scale = P.s_gu ? P.s_gu[e] : 1.0f;

    nvfp4moe_fill_ue4m3(ue4m3);
    if constexpr (MAXM == 1) {
        __syncthreads();
    }
    if constexpr (PREQ) {
        static_assert(MAXM > 1, "PREQ: T > 1 only");
        for (int m = 0; m < n_tok; ++m) {
            const uint4 * src = (const uint4 *) (P.xq + tok_t[m]*nb_k);
            uint4 * dst = (uint4 *) (xq + m*nb_k);
            for (int q = tid; q < nb_k*(int) sizeof(block_q8_1)/16; q += 256) {
                dst[q] = src[q];
            }
        }
    } else {
        for (int m = 0; m < n_tok; ++m) {
            const float * xt = P.x + tok_t[m]*P.x_s2;
            for (int b = warp; b < nb_k; b += 8) {
                nvfp4moe_quant_block(xt[b*QK8_1 + lane], xq + m*nb_k + b, lane);
            }
        }
    }
    __syncthreads();

    const uint32_t t0 = 0x03020100, t1 = 0x0C080604, t2 = 0xFDFEFF00, t3 = 0xF4F8FAFC; // kvalues_mxfp4

    if constexpr (MAXM == 1) {
        constexpr int m = 0;
        const block_q8_1 * xb = xq;
        float pg[NHB], pu[NHB];
#pragma unroll
        for (int i = 0; i < NHB; ++i) {
            const block_q8_1 * yb = xb + j + L*i;
            const int * q = (const int *) yb->qs;
            int qv[8];
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                qv[l] = q[l];
            }
            const float d8 = __low2float(yb->ds);
            pg[i] = nvfp4moe_hb_dot(cg[i], sg[i], qv, d8, ue4m3, t0, t1, t2, t3);
            pu[i] = nvfp4moe_hb_dot(cu[i], su[i], qv, d8, ue4m3, t0, t1, t2, t3);
        }
        float rg_, ru_;
        if constexpr (NHB == 9) {
            // generic MMVQ order: thread h holds half-block h, s_l = p[l] + p[l+32] + p[l+64], then the xor 16..1 butterfly
            const float A = (pg[0] + pg[4]) + pg[8], B = pg[1] + pg[5], C = pg[2] + pg[6], D = pg[3] + pg[7];
            rg_ = (A + C) + (B + D);
            const float Au = (pu[0] + pu[4]) + pu[8], Bu = pu[1] + pu[5], Cu = pu[2] + pu[6], Du = pu[3] + pu[7];
            ru_ = (Au + Cu) + (Bu + Du);
        } else {
            rg_ = 0.0f; ru_ = 0.0f;
#pragma unroll
            for (int i = 0; i < NHB; ++i) { rg_ += pg[i]; ru_ += pu[i]; }
        }
#pragma unroll
        for (int o = 4; o > 0; o >>= 1) {
            rg_ += __shfl_xor_sync(0xffffffff, rg_, o, 32);
            ru_ += __shfl_xor_sync(0xffffffff, ru_, o, 32);
        }
        if (j == 0) {
            float gv = rg_, uv = ru_;
            if (P.s_gu) {
                gv *= scale;
                uv *= scale;
            }
            act_s[m][g] = nvfp4moe_situ(gv, uv, P.situ);
        }
    } else if constexpr (GO) {
        static_assert(NHB == 9, "GO: K = 2304");
        // 2..MAXM tokens, generic order per token: generic lane j + 8a (a < 4) holds half-blocks j + 8a, + 32, (a == 0) + 64
        // = m, m + 4, m + 8 here; acc a gets 0 + p[a] + p[a+4] (+ p[8]), then the xor 16 / 8 butterfly is (a0 + a2) + (a1 + a3)
        float ag[MAXM][4], au[MAXM][4];
#pragma unroll
        for (int i = 0; i < NHB; ++i) {
            const nvfp4moe_hbd wg = nvfp4moe_decode(cg[i], sg[i], ue4m3, t0, t1, t2, t3);
            const nvfp4moe_hbd wu = nvfp4moe_decode(cu[i], su[i], ue4m3, t0, t1, t2, t3);
#pragma unroll
            for (int m = 0; m < MAXM; ++m) {
                if (m < n_tok) {
                    const block_q8_1 * yb = xq + m*nb_k + j + L*i;
                    const float pg = nvfp4moe_hbd_dot(wg, yb), pu = nvfp4moe_hbd_dot(wu, yb);
                    if (i < 4) { ag[m][i] = 0.0f + pg; au[m][i] = 0.0f + pu; }
                    else       { ag[m][i & 3] += pg;   au[m][i & 3] += pu; }
                }
            }
        }
#pragma unroll
        for (int m = 0; m < MAXM; ++m) {
            if (m < n_tok) {
                float rg_ = (ag[m][0] + ag[m][2]) + (ag[m][1] + ag[m][3]);
                float ru_ = (au[m][0] + au[m][2]) + (au[m][1] + au[m][3]);
#pragma unroll
                for (int o = 4; o > 0; o >>= 1) {
                    rg_ += __shfl_xor_sync(0xffffffff, rg_, o, 32);
                    ru_ += __shfl_xor_sync(0xffffffff, ru_, o, 32);
                }
                if (j == 0) {
                    act_s[m][g] = nvfp4moe_situ(P.s_gu ? rg_*scale : rg_, P.s_gu ? ru_*scale : ru_, P.situ);
                }
            }
        }
    } else {
        // 2..MAXM tokens: decode each half-block once, dot it with every token (sequential sums, not the 1-token order)
        float ag[MAXM], au[MAXM];
#pragma unroll
        for (int m = 0; m < MAXM; ++m) { ag[m] = 0.0f; au[m] = 0.0f; }
#pragma unroll
        for (int i = 0; i < NHB; ++i) {
            const nvfp4moe_hbd wg = nvfp4moe_decode(cg[i], sg[i], ue4m3, t0, t1, t2, t3);
            const nvfp4moe_hbd wu = nvfp4moe_decode(cu[i], su[i], ue4m3, t0, t1, t2, t3);
#pragma unroll
            for (int m = 0; m < MAXM; ++m) {
                if (m < n_tok) {
                    const block_q8_1 * yb = xq + m*nb_k + j + L*i;
                    ag[m] += nvfp4moe_hbd_dot(wg, yb);
                    au[m] += nvfp4moe_hbd_dot(wu, yb);
                }
            }
        }
#pragma unroll
        for (int m = 0; m < MAXM; ++m) {
            if (m < n_tok) {
                float rg_ = ag[m], ru_ = au[m];
#pragma unroll
                for (int o = 4; o > 0; o >>= 1) {
                    rg_ += __shfl_xor_sync(0xffffffff, rg_, o, 32);
                    ru_ += __shfl_xor_sync(0xffffffff, ru_, o, 32);
                }
                if (j == 0) {
                    act_s[m][g] = nvfp4moe_situ(P.s_gu ? rg_*scale : rg_, P.s_gu ? ru_*scale : ru_, P.situ);
                }
            }
        }
    }
    __syncthreads();
    const int nb_f = P.F / QK8_1;
    for (int m = warp; m < n_tok; m += 8) {
        nvfp4moe_quant_block(act_s[m][lane], P.act + ((int64_t) tok_t[m]*P.n_used + tok_k[m])*nb_f + blockIdx.x, lane);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// gate_up + scale + SITU -> q8_1 for one token, the generic MMVQ thread layout: grid (F/(8*NF), n_used), 256 threads,
// warp w takes features f0 + w + 8i (i < NF; gate row f, up row F + f; the next rows load while one is reduced), lane l owns half-blocks l, l + 32 and (l < 8) l + 64, as thread
// l, 32 + l, 64 + l of the 4-warp small_k MMVQ CTA: s_l = (p_l + p_{l+32}) + p_{l+64}, then the xor 16..1 butterfly.
template <int NF>  // NF features per warp (256 threads: 8*NF features per CTA, F/(8*NF) CTAs per slot)
static __global__ void __launch_bounds__(256) k_nvfp4moe_gu1w(const nvfp4moe_params P) {
    __shared__ float ue4m3[256];
    __shared__ __align__(16) block_q8_1 xq[2304/QK8_1];
    __shared__ float act_s[8*NF];

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int nb_k = P.K / QK8_1;  // 72
    const int t = blockIdx.y / P.n_used, k = blockIdx.y % P.n_used;
    const int e = P.ids[t*P.ids_s1 + k];
    const char * base = P.w_gu + e*P.gu_nb2;
    const int64_t up_off = (int64_t) P.F*P.gu_nb1;
    const bool three = lane < nb_k - 64;

    uint4 cg[2][3], cu[2][3];
    uint32_t sg[2][3], su[2][3];
    auto load = [&](const int b, const int i) {
        const char * rg = base + (int64_t) (blockIdx.x*8*NF + warp + 8*i)*P.gu_nb1;
        const char * ru = rg + up_off;
#pragma unroll
        for (int m = 0; m < 2; ++m) {
            nvfp4moe_load_hb(rg, lane + 32*m, cg[b][m], sg[b][m]);
            nvfp4moe_load_hb(ru, lane + 32*m, cu[b][m], su[b][m]);
        }
        if (three) {
            nvfp4moe_load_hb(rg, lane + 64, cg[b][2], sg[b][2]);
            nvfp4moe_load_hb(ru, lane + 64, cu[b][2], su[b][2]);
        }
    };
    load(0, 0);
    if (NF > 1) {
        load(1, 1);
    }
    const float scale = P.s_gu ? P.s_gu[e] : 1.0f;

    ue4m3[tid] = ggml_cuda_ue4m3_to_fp32((uint8_t) tid);
    const float * xt = P.x + t*P.x_s2;
    for (int b = warp; b < nb_k; b += 8) {
        nvfp4moe_quant_block(xt[b*QK8_1 + lane], xq + b, lane);
    }
    __syncthreads();

    const uint32_t t0 = 0x03020100, t1 = 0x0C080604, t2 = 0xFDFEFF00, t3 = 0xF4F8FAFC; // kvalues_mxfp4
#pragma unroll
    for (int i = 0; i < NF; ++i) {
        const int b = i & 1;
        float sgv = 0.0f, suv = 0.0f;
#pragma unroll
        for (int m = 0; m < 3; ++m) {
            if (m < 2 || three) {
                const block_q8_1 * yb = xq + lane + 32*m;
                const int * q = (const int *) yb->qs;
                int qv[8];
#pragma unroll
                for (int l = 0; l < 8; ++l) {
                    qv[l] = q[l];
                }
                const float d8 = __low2float(yb->ds);
                const float pg = nvfp4moe_hb_dot(cg[b][m], sg[b][m], qv, d8, ue4m3, t0, t1, t2, t3);
                const float pu = nvfp4moe_hb_dot(cu[b][m], su[b][m], qv, d8, ue4m3, t0, t1, t2, t3);
                if (m == 0) { sgv = pg; suv = pu; } else { sgv += pg; suv += pu; }
            }
        }
        if (i + 2 < NF) {
            load(b, i + 2);
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            sgv += __shfl_xor_sync(0xffffffff, sgv, o, 32);
            suv += __shfl_xor_sync(0xffffffff, suv, o, 32);
        }
        if (lane == 0) {
            float gv = sgv, uv = suv;
            if (P.s_gu) {
                gv *= scale;
                uv *= scale;
            }
            act_s[warp + 8*i] = nvfp4moe_situ(gv, uv, P.situ);
        }
    }
    __syncthreads();
    // feature blockIdx.x*8*NF + w + 8*i sits at act_s[w + 8*i]: q8_1 block c covers act_s[32c .. 32c + 31]
    if (warp < NF/4) {
        nvfp4moe_quant_block(act_s[32*warp + lane], P.act + ((int64_t) t*P.n_used + k)*(P.F/QK8_1) + blockIdx.x*(NF/4) + warp, lane);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// F32-activation arm (GGML_CUDA_NVFP4MOE_ACT=f32): the weights are decoded exactly (int8 x2 values, UE4M3 scales) and
// dotted with the f32 activations, no q8_1 quantization of x or of the SITU output. One token.

// 4 int8 lanes -> 4 exact floats: (b ^ 0x80) as the low mantissa byte of 2^23, minus 2^23 + 128
static __device__ __forceinline__ float nvfp4moe_i8f(const int v, const int i) {
    return __int_as_float(__byte_perm(v ^ 0x80808080, 0x4B000000, 0x7650 | i)) - 8388736.0f;
}
static __device__ __forceinline__ float nvfp4moe_dot4f(const int v, const float4 x) {
    float r = nvfp4moe_i8f(v, 0)*x.x;
    r = fmaf(nvfp4moe_i8f(v, 1), x.y, r);
    r = fmaf(nvfp4moe_i8f(v, 2), x.z, r);
    r = fmaf(nvfp4moe_i8f(v, 3), x.w, r);
    return r;
}
// one half-block (32 weights) against 32 f32 activations x4[0..7] (float4)
static __device__ __forceinline__ float nvfp4moe_hb_dotf(const uint4 c, const uint32_t sc, const float4 * __restrict__ x4,
        const float * ue4m3, const uint32_t t0, const uint32_t t1, const uint32_t t2, const uint32_t t3) {
    const int2 a0 = nvfp4moe_lut(c.x, t0, t1, t2, t3), a1 = nvfp4moe_lut(c.y, t0, t1, t2, t3);
    const int2 b0 = nvfp4moe_lut(c.z, t0, t1, t2, t3), b1 = nvfp4moe_lut(c.w, t0, t1, t2, t3);
    // sub-block: word 0 low nibbles = elements 0..3, word 1 low = 4..7, word 0 high = 8..11, word 1 high = 12..15
    const float sa = (nvfp4moe_dot4f(a0.x, __ldg(x4 + 0)) + nvfp4moe_dot4f(a1.x, __ldg(x4 + 1))) +
                     (nvfp4moe_dot4f(a0.y, __ldg(x4 + 2)) + nvfp4moe_dot4f(a1.y, __ldg(x4 + 3)));
    const float sb = (nvfp4moe_dot4f(b0.x, __ldg(x4 + 4)) + nvfp4moe_dot4f(b1.x, __ldg(x4 + 5))) +
                     (nvfp4moe_dot4f(b0.y, __ldg(x4 + 6)) + nvfp4moe_dot4f(b1.y, __ldg(x4 + 7)));
    return fmaf(ue4m3[(sc >> 8) & 0xFF], sb, ue4m3[sc & 0xFF]*sa);
}

// gate_up + scale + SITU, f32 activations: same lane layout as k_nvfp4moe_gu1w, writes the f32 SITU output P.actf[k][f]
template <int NF>
static __global__ void __launch_bounds__(256) k_nvfp4moe_gu1f(const nvfp4moe_params P) {
    __shared__ float ue4m3[256];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int nb_k = P.K / QK8_1;  // 72
    const int t = blockIdx.y / P.n_used, k = blockIdx.y % P.n_used;
    const int e = P.ids[t*P.ids_s1 + k];
    const char * base = P.w_gu + e*P.gu_nb2;
    const int64_t up_off = (int64_t) P.F*P.gu_nb1;
    const bool three = lane < nb_k - 64;

    uint4 cg[2][3], cu[2][3];
    uint32_t sg[2][3], su[2][3];
    auto load = [&](const int b, const int i) {
        const char * rg = base + (int64_t) (blockIdx.x*8*NF + warp + 8*i)*P.gu_nb1;
        const char * ru = rg + up_off;
#pragma unroll
        for (int m = 0; m < 2; ++m) {
            nvfp4moe_load_hb(rg, lane + 32*m, cg[b][m], sg[b][m]);
            nvfp4moe_load_hb(ru, lane + 32*m, cu[b][m], su[b][m]);
        }
        if (three) {
            nvfp4moe_load_hb(rg, lane + 64, cg[b][2], sg[b][2]);
            nvfp4moe_load_hb(ru, lane + 64, cu[b][2], su[b][2]);
        }
    };
    load(0, 0);
    if (NF > 1) {
        load(1, 1);
    }
    const float scale = P.s_gu ? P.s_gu[e] : 1.0f;
    ue4m3[tid] = ggml_cuda_ue4m3_to_fp32((uint8_t) tid);
    __syncthreads();

    const float4 * x4 = (const float4 *) (P.x + t*P.x_s2);
    const uint32_t t0 = 0x03020100, t1 = 0x0C080604, t2 = 0xFDFEFF00, t3 = 0xF4F8FAFC; // kvalues_mxfp4
#pragma unroll
    for (int i = 0; i < NF; ++i) {
        const int b = i & 1;
        float sgv = 0.0f, suv = 0.0f;
#pragma unroll
        for (int m = 0; m < 3; ++m) {
            if (m < 2 || three) {
                const float4 * xb = x4 + 8*(lane + 32*m);
                sgv += nvfp4moe_hb_dotf(cg[b][m], sg[b][m], xb, ue4m3, t0, t1, t2, t3);
                suv += nvfp4moe_hb_dotf(cu[b][m], su[b][m], xb, ue4m3, t0, t1, t2, t3);
            }
        }
        if (i + 2 < NF) {
            load(b, i + 2);
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            sgv += __shfl_xor_sync(0xffffffff, sgv, o, 32);
            suv += __shfl_xor_sync(0xffffffff, suv, o, 32);
        }
        if (lane == 0) {
            const int f = blockIdx.x*8*NF + warp + 8*i;
            P.actf[((int64_t) t*P.n_used + k)*P.F + f] = nvfp4moe_situ(sgv*scale, suv*scale, P.situ);
        }
    }
}

// down + scale (+ routing weight and expert sum when write_sum), f32 activations, one token, n_used == 8:
// k_nvfp4moe_dn1's layout (warp = output row, lane = 4k + j, lane j owns half-blocks j + 4m of slot k)
static __global__ void __launch_bounds__(256) k_nvfp4moe_dn1f(const nvfp4moe_params P) {
    constexpr int NHB = 9;
    __shared__ float ue4m3[256];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int k = lane >> 2, j = lane & 3;
    const int r = blockIdx.x*8 + warp;
    const int e = P.ids[k];
    const char * row = P.w_dn + e*P.dn_nb2 + (int64_t) r*P.dn_nb1;
    uint4 c[NHB];
    uint32_t sc[NHB];
#pragma unroll
    for (int m = 0; m < NHB; ++m) {
        nvfp4moe_load_hb(row, j + 4*m, c[m], sc[m]);
    }
    const float scale = P.s_dn ? P.s_dn[e] : 1.0f;
    const float w = P.write_sum ? P.rw[k*P.rw_s1] : 0.0f;
    ue4m3[tid] = ggml_cuda_ue4m3_to_fp32((uint8_t) tid);
    __syncthreads();

    const float4 * a4 = (const float4 *) (P.actf + (int64_t) k*P.F);
    const uint32_t t0 = 0x03020100, t1 = 0x0C080604, t2 = 0xFDFEFF00, t3 = 0xF4F8FAFC;
    float v = 0.0f;
#pragma unroll
    for (int m = 0; m < NHB; ++m) {
        v += nvfp4moe_hb_dotf(c[m], sc[m], a4 + 8*(j + 4*m), ue4m3, t0, t1, t2, t3);
    }
    v += __shfl_xor_sync(0xffffffff, v, 2, 32);
    v += __shfl_xor_sync(0xffffffff, v, 1, 32);
    v *= scale;
    if (!P.write_sum) {
        if (j == 0) {
            P.e_out[(int64_t) k*P.K + r] = v;
        }
        return;
    }
    float x = j == 0 ? v*w : 0.0f;
    x += __shfl_xor_sync(0xffffffff, x, 16, 32);
    x += __shfl_xor_sync(0xffffffff, x,  8, 32);
    x += __shfl_xor_sync(0xffffffff, x,  4, 32);
    if (lane == 0) {
        P.out[r] = x;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// F16-activation arm (GGML_CUDA_NVFP4MOE_ACT=f16): codes decoded exactly to half2 (byte -> (value of element b, value of
// element b + 8), the 2x E2M1 integers), activations as f16 pairs (x[b], x[b+8]) pre-scaled by 1/16 (8-term half sums: no overflow below
// |x| ~ 10.9K), HFMA2 over one 16-weight sub-block, then f32: sum += UE4M3 scale * sub-block sum. No q8_1 quantization.
#define NVFP4MOE_H_PRE 0.0625f

static __device__ __forceinline__ void nvfp4moe_fill_lut2(__half2 * lut2) {
    const int b = threadIdx.x;  // 256 threads
    lut2[b] = __halves2half2(__int2half_rn(kvalues_mxfp4[b & 15]), __int2half_rn(kvalues_mxfp4[b >> 4]));
}
// one half-block (2 sub-blocks) against its 16 activation pairs xp[0..15] (sub-block s: pairs 8s .. 8s + 7)
static __device__ __forceinline__ float nvfp4moe_hb_doth(const uint4 c, const uint32_t sc, const __half2 * xp,
        const __half2 * lut2, const float * ue4m3) {
    const uint32_t w[4] = { c.x, c.y, c.z, c.w };
    float sum = 0.0f;
#pragma unroll
    for (int s = 0; s < 2; ++s) {
        __half2 acc = __float2half2_rn(0.0f);
#pragma unroll
        for (int b = 0; b < 8; ++b) {
            acc = __hfma2(lut2[(w[2*s + b/4] >> (8*(b%4))) & 0xFF], xp[8*s + b], acc);
        }
        const float2 f = __half22float2(acc);
        sum = fmaf(ue4m3[(sc >> (8*s)) & 0xFF], f.x + f.y, sum);
    }
    return sum;
}

// register decode variant: the int8 table gives (value + 128) bytes, paired with 0x64 into f16 1024 + 128 + value, minus 1152
static __device__ __forceinline__ float nvfp4moe_hb_dothr(const uint4 c, const uint32_t sc, const __half2 * xp,
        const float * ue4m3) {
    const uint32_t t0 = 0x83828180, t1 = 0x8C888684, t2 = 0x7D7E7F80, t3 = 0x74787A7C; // kvalues_mxfp4 + 128
    const uint32_t C = 0x64646464;
    const __half2 off = __float2half2_rn(1152.0f);
    const uint32_t w[4] = { c.x, c.y, c.z, c.w };
    const uint4 * xq = (const uint4 *) xp;
    float sum = 0.0f;
#pragma unroll
    for (int s = 0; s < 2; ++s) {
        __half2 acc = __float2half2_rn(0.0f);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int2 v = nvfp4moe_lut(w[2*s + h], t0, t1, t2, t3);
            const uint4 xv = xq[2*s + h];  // pairs 8s + 4h .. 8s + 4h + 3
            const uint32_t p01 = __byte_perm(v.x, v.y, 0x5140), p23 = __byte_perm(v.x, v.y, 0x7362);
            const uint32_t hb[4] = { __byte_perm(p01, C, 0x4140), __byte_perm(p01, C, 0x4342),
                                     __byte_perm(p23, C, 0x4140), __byte_perm(p23, C, 0x4342) };
            const uint32_t xs[4] = { xv.x, xv.y, xv.z, xv.w };
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const __half2 wv = __hsub2(*(const __half2 *) &hb[b], off);
                acc = __hfma2(wv, *(const __half2 *) &xs[b], acc);
            }
        }
        const float2 f = __half22float2(acc);
        sum = fmaf(ue4m3[(sc >> (8*s)) & 0xFF], f.x + f.y, sum);
    }
    return sum;
}
#ifdef NVFP4MOE_H_LUT
#define NVFP4MOE_HDOT(c, sc, xp) nvfp4moe_hb_doth(c, sc, xp, lut2, ue4m3)
#else
#define NVFP4MOE_HDOT(c, sc, xp) nvfp4moe_hb_dothr(c, sc, xp, ue4m3)
#endif

// gate_up + scale + SITU, f16 activations, one token: k_nvfp4moe_gu1w's lane layout. Writes the SITU output of the
// CTA's 8*NF features as pre-scaled f16 pairs into P.acth[k][F/2] (pair index 8*(f/16) + f%8 holds (f, f + 8)).
template <int NF>
static __global__ void __launch_bounds__(256) k_nvfp4moe_gu1h(const nvfp4moe_params P) {
    __shared__ float ue4m3[256];
#ifdef NVFP4MOE_H_LUT
    __shared__ __half2 lut2[256];
#endif
    __shared__ __align__(16) __half2 xp[2304/2];
    __shared__ float act_s[8*NF];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int nb_k = P.K / QK8_1;  // 72
    const int t = blockIdx.y / P.n_used, k = blockIdx.y % P.n_used;
    const int e = P.ids[t*P.ids_s1 + k];
    const char * base = P.w_gu + e*P.gu_nb2;
    const int64_t up_off = (int64_t) P.F*P.gu_nb1;
    const bool three = lane < nb_k - 64;

    uint4 cg[2][3], cu[2][3];
    uint32_t sg[2][3], su[2][3];
    auto load = [&](const int b, const int i) {
        const char * rg = base + (int64_t) (blockIdx.x*8*NF + warp + 8*i)*P.gu_nb1;
        const char * ru = rg + up_off;
#pragma unroll
        for (int m = 0; m < 2; ++m) {
            nvfp4moe_load_hb(rg, lane + 32*m, cg[b][m], sg[b][m]);
            nvfp4moe_load_hb(ru, lane + 32*m, cu[b][m], su[b][m]);
        }
        if (three) {
            nvfp4moe_load_hb(rg, lane + 64, cg[b][2], sg[b][2]);
            nvfp4moe_load_hb(ru, lane + 64, cu[b][2], su[b][2]);
        }
    };
    load(0, 0);
    if (NF > 1) {
        load(1, 1);
    }
    const float scale = (P.s_gu ? P.s_gu[e] : 1.0f) * (1.0f/NVFP4MOE_H_PRE);
    ue4m3[tid] = ggml_cuda_ue4m3_to_fp32((uint8_t) tid);
#ifdef NVFP4MOE_H_LUT
    nvfp4moe_fill_lut2(lut2);
#endif
    const float * xt = P.x + t*P.x_s2;
    for (int i = tid; i < P.K/2; i += 256) {  // pair i: sub-block i/8, byte i%8
        const int e0 = 16*(i >> 3) + (i & 7);
        xp[i] = __floats2half2_rn(xt[e0]*NVFP4MOE_H_PRE, xt[e0 + 8]*NVFP4MOE_H_PRE);
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < NF; ++i) {
        const int b = i & 1;
        float sgv = 0.0f, suv = 0.0f;
#pragma unroll
        for (int m = 0; m < 3; ++m) {
            if (m < 2 || three) {
                const __half2 * xb = xp + 16*(lane + 32*m);
                sgv += NVFP4MOE_HDOT(cg[b][m], sg[b][m], xb);
                suv += NVFP4MOE_HDOT(cu[b][m], su[b][m], xb);
            }
        }
        if (i + 2 < NF) {
            load(b, i + 2);
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            sgv += __shfl_xor_sync(0xffffffff, sgv, o, 32);
            suv += __shfl_xor_sync(0xffffffff, suv, o, 32);
        }
        if (lane == 0) {
            act_s[warp + 8*i] = nvfp4moe_situ(sgv*scale, suv*scale, P.situ);
        }
    }
    __syncthreads();
    // features f0 + w + 8i sit at act_s[w + 8i] (local feature index); pairs (lf, lf + 8) for lf % 16 < 8
    __half2 * dst = P.acth + ((int64_t) t*P.n_used + k)*(P.F/2) + blockIdx.x*4*NF;
    for (int i = tid; i < 4*NF; i += 256) {
        const int lf = 16*(i >> 3) + (i & 7);
        dst[i] = __floats2half2_rn(act_s[lf]*NVFP4MOE_H_PRE, act_s[lf + 8]*NVFP4MOE_H_PRE);
    }
}

// down + scale (+ routing weight and expert sum), f16 activation pairs from k_nvfp4moe_gu1h, one token, n_used == 8
static __global__ void __launch_bounds__(256) k_nvfp4moe_dn1h(const nvfp4moe_params P) {
    constexpr int NHB = 9;
    __shared__ float ue4m3[256];
#ifdef NVFP4MOE_H_LUT
    __shared__ __half2 lut2[256];
#endif
    constexpr int AP_S = 1152/2 + 4;  // padded slot stride (half2): spreads the 8 slots over banks
    __shared__ __align__(16) __half2 ap[8*AP_S];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int k = lane >> 2, j = lane & 3;
    const int r = blockIdx.x*8 + warp;
    const int e = P.ids[k];
    const char * row = P.w_dn + e*P.dn_nb2 + (int64_t) r*P.dn_nb1;
    uint4 c[NHB];
    uint32_t sc[NHB];
#pragma unroll
    for (int m = 0; m < NHB; ++m) {
        nvfp4moe_load_hb(row, j + 4*m, c[m], sc[m]);
    }
    const float scale = (P.s_dn ? P.s_dn[e] : 1.0f) * (1.0f/NVFP4MOE_H_PRE);
    const float w = P.write_sum ? P.rw[k*P.rw_s1] : 0.0f;
    ue4m3[tid] = ggml_cuda_ue4m3_to_fp32((uint8_t) tid);
#ifdef NVFP4MOE_H_LUT
    nvfp4moe_fill_lut2(lut2);
#endif
    {
        const uint4 * src = (const uint4 *) P.acth;  // 8 slots x 144 uint4
        uint4 * dst = (uint4 *) ap;
        for (int i = tid; i < 8*144; i += 256) {
            dst[(i/144)*(AP_S/4) + i%144] = src[i];
        }
    }
    __syncthreads();
    float v = 0.0f;
#pragma unroll
    for (int m = 0; m < NHB; ++m) {
        v += NVFP4MOE_HDOT(c[m], sc[m], ap + k*AP_S + 16*(j + 4*m));
    }
    v += __shfl_xor_sync(0xffffffff, v, 2, 32);
    v += __shfl_xor_sync(0xffffffff, v, 1, 32);
    v *= scale;
    if (!P.write_sum) {
        if (j == 0) {
            P.e_out[(int64_t) k*P.K + r] = v;
        }
        return;
    }
    float x = j == 0 ? v*w : 0.0f;
    x += __shfl_xor_sync(0xffffffff, x, 16, 32);
    x += __shfl_xor_sync(0xffffffff, x,  8, 32);
    x += __shfl_xor_sync(0xffffffff, x,  4, 32);
    if (lane == 0) {
        P.out[r] = x;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// down + scale + routing weight + expert sum, one token, n_used == 8. grid K/8, 256 threads: warp = output row r,
// lane = 4k + j (slot k, lane j owns half-blocks j + 4m). F = 1152: 36 half-blocks = 9 per lane.
static __global__ void __launch_bounds__(256) k_nvfp4moe_dn1(const nvfp4moe_params P) {
    constexpr int NHB = 9;
    __shared__ float ue4m3[256];
    __shared__ __align__(16) block_q8_1 aq[8*36];

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int k = lane >> 2, j = lane & 3;
    const int r = blockIdx.x*8 + warp;
    const int e = P.ids[k];

    const char * row = P.w_dn + e*P.dn_nb2 + (int64_t) r*P.dn_nb1;
    uint4 c[NHB];
    uint32_t sc[NHB];
#pragma unroll
    for (int m = 0; m < NHB; ++m) {
        nvfp4moe_load_hb(row, j + 4*m, c[m], sc[m]);
    }
    const float scale = P.s_dn ? P.s_dn[e] : 1.0f;
    const float w = P.write_sum ? P.rw[k*P.rw_s1] : 0.0f;

    nvfp4moe_fill_ue4m3(ue4m3);
    {
        const uint4 * src = (const uint4 *) P.act;
        uint4 * dst = (uint4 *) aq;
        for (int i = tid; i < (int) (sizeof(aq)/16); i += 256) {
            dst[i] = src[i];
        }
    }
    __syncthreads();

    const uint32_t t0 = 0x03020100, t1 = 0x0C080604, t2 = 0xFDFEFF00, t3 = 0xF4F8FAFC;
    const block_q8_1 * ab = aq + k*36;
    float p[NHB];
#pragma unroll
    for (int i = 0; i < NHB; ++i) {
        const block_q8_1 * yb = ab + j + 4*i;
        const int * q = (const int *) yb->qs;
        int qv[8];
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            qv[l] = q[l];
        }
        p[i] = nvfp4moe_hb_dot(c[i], sc[i], qv, __low2float(yb->ds), ue4m3, t0, t1, t2, t3);
    }
    // generic order (36 half-blocks): s_l = p[l] + p[l+32] for l < 4, else p[l]; lane j holds s_{j+4a}
    const float s0 = p[0] + p[8];
    const float t16_0 = s0   + p[4], t16_1 = p[1] + p[5], t16_2 = p[2] + p[6], t16_3 = p[3] + p[7];
    const float u0 = t16_0 + t16_2, u1 = t16_1 + t16_3;
    float v = u0 + u1;
    v += __shfl_xor_sync(0xffffffff, v, 2, 32);
    v += __shfl_xor_sync(0xffffffff, v, 1, 32);
    if (P.s_dn) {
        v *= scale;
    }
    if (!P.write_sum) {  // the scaled per-slot output (e_out[k][r]); the graph adds the weighted sum
        if (j == 0) {
            P.e_out[(int64_t) k*P.K + r] = v;
        }
        return;
    }
    // k_ink_moe_wsum: lane k holds 0 + e_k*w_k, lanes 8..31 hold 0, warp_reduce_sum
    float x = 0.0f + __fmul_rn(v, w);
    x += 0.0f;  // xor 16: lanes 16..23 are 0
    x += 0.0f;  // xor 8
    x += __shfl_xor_sync(0xffffffff, x, 16, 32);  // k ^ 4
    x += __shfl_xor_sync(0xffffffff, x,  8, 32);  // k ^ 2
    x += __shfl_xor_sync(0xffffffff, x,  4, 32);  // k ^ 1
    if (lane == 0) {
        P.out[r] = x;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// down for 2..MAXM tokens: grid (K/64, T*n_used), 256 threads: lane = 4*row + j, warp w rows 8w..8w+7 of the chunk.
// Each CTA reads the rows of its expert once and dots every token routed to it; writes e_out[t][k][r] (scaled).
// GO: generic order per token (lane l of mul_mat_vec_q_moe: 0 + p[l] + p[l+32], l < 4), as k_nvfp4moe_dn1 rebuilds it
template <int MAXM, bool GO = false>
static __global__ void __launch_bounds__(256) k_nvfp4moe_dnm(const nvfp4moe_params P) {
    constexpr int NHB = 9;
    extern __shared__ __align__(16) char smem_raw[];
    __shared__ float ue4m3[256];
    __shared__ int   s_ids[NVFP4MOE_MAX_TOK*8];  // n_used == 8
    __shared__ int   tok_t[MAXM], tok_k[MAXM];
    block_q8_1 * aq = (block_q8_1 *) smem_raw;  // [MAXM][36]

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int j = lane & 3;
    const int r = blockIdx.x*64 + warp*8 + (lane >> 2);

    int e;
    const int n_tok = nvfp4moe_gather(P, s_ids, tok_t, tok_k, e, MAXM);
    if (n_tok == 0) {
        return;
    }
    const char * row = P.w_dn + e*P.dn_nb2 + (int64_t) r*P.dn_nb1;
    uint4 c[NHB];
    uint32_t sc[NHB];
#pragma unroll
    for (int m = 0; m < NHB; ++m) {
        nvfp4moe_load_hb(row, j + 4*m, c[m], sc[m]);
    }
    const float scale = P.s_dn ? P.s_dn[e] : 1.0f;
    nvfp4moe_fill_ue4m3(ue4m3);
    const int nb_f = P.F / QK8_1;
    for (int m = 0; m < n_tok; ++m) {
        const uint4 * src = (const uint4 *) (P.act + ((int64_t) tok_t[m]*P.n_used + tok_k[m])*nb_f);
        uint4 * dst = (uint4 *) (aq + m*nb_f);
        for (int i = tid; i < nb_f*(int) sizeof(block_q8_1)/16; i += 256) {
            dst[i] = src[i];
        }
    }
    __syncthreads();

    const uint32_t t0 = 0x03020100, t1 = 0x0C080604, t2 = 0xFDFEFF00, t3 = 0xF4F8FAFC;
    float acc[MAXM];
    if constexpr (GO) {
        // lane j holds generic lanes j + 4a (a < 8) = half-blocks j + 4i, i < 8, and j + 32 (i == 8, added to a = 0 first):
        // acc4[a] = 0 + p[a] (+ p[8]); xor 16 / 8 / 4 = ((q0 + q4) + (q2 + q6)) + ((q1 + q5) + (q3 + q7))
        constexpr int ord[NHB] = {0, 8, 1, 2, 3, 4, 5, 6, 7};
        float q[MAXM][8];
#pragma unroll
        for (int ii = 0; ii < NHB; ++ii) {
            const int i = ord[ii];
            const nvfp4moe_hbd wd = nvfp4moe_decode(c[i], sc[i], ue4m3, t0, t1, t2, t3);
#pragma unroll
            for (int m = 0; m < MAXM; ++m) {
                if (m < n_tok) {
                    const float p = nvfp4moe_hbd_dot(wd, aq + m*nb_f + j + 4*i);
                    if (i == 8) { q[m][0] += p; } else { q[m][i] = 0.0f + p; }
                }
            }
        }
#pragma unroll
        for (int m = 0; m < MAXM; ++m) {
            acc[m] = m >= n_tok ? 0.0f : ((q[m][0] + q[m][4]) + (q[m][2] + q[m][6])) + ((q[m][1] + q[m][5]) + (q[m][3] + q[m][7]));
        }
    } else {
#pragma unroll
    for (int m = 0; m < MAXM; ++m) { acc[m] = 0.0f; }
#pragma unroll
    for (int i = 0; i < NHB; ++i) {
        const nvfp4moe_hbd wd = nvfp4moe_decode(c[i], sc[i], ue4m3, t0, t1, t2, t3);
#pragma unroll
        for (int m = 0; m < MAXM; ++m) {
            if (m < n_tok) {
                acc[m] += nvfp4moe_hbd_dot(wd, aq + m*nb_f + j + 4*i);
            }
        }
    }
    }
#pragma unroll
    for (int m = 0; m < MAXM; ++m) {
        if (m < n_tok) {
            float a = acc[m];
            a += __shfl_xor_sync(0xffffffff, a, 2, 32);
            a += __shfl_xor_sync(0xffffffff, a, 1, 32);
            if (j == 0) {
                if (P.s_dn) {
                    a *= scale;
                }
                P.e_out[((int64_t) tok_t[m]*P.n_used + tok_k[m])*P.K + r] = a;
            }
        }
    }
}
