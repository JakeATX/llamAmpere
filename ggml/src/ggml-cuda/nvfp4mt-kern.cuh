#pragma once
// INKNVFP4K NVFP4MT: NVFP4 MoE decode for 1..8 tokens (decode, DSpark verify, 2-user decode) on sm_86, K = 2304, F = 1152, 8 used.
// One CTA per (routed expert, tile): the first slot of an expert serves every token routed to it, so each expert's weights
// are read once per visit. The weights stream through a per-warp cp.async ring in shared memory (16-byte copies of whole
// rows, no register staging), so 2 CTAs per SM keep loads in flight while the other computes. The body is compiled per
// member count NT (1..T): an expert with one token does one token of math.
// Bit-identical to the generic path (mul_mat_vec_q_moe per token: lane l holds 0 + p[l] + p[l+32] (+ p[l+64]), then the
// xor 16..1 butterfly, then * x_scale; quantize_q8_1; k_ink_situ): each lane computes the half-blocks of fixed generic
// lanes in the generic order, folds the xor steps that stay in the lane in registers, and runs the rest as a transposed
// butterfly over the tokens (each xor step still adds own + partner for every value, so every value keeps its bits).
#include "nvfp4moe-kern.cuh"

#define NVFP4MT_ROW_GU 1296  // K/64 blocks of 36 bytes
#define NVFP4MT_ROW_DN 648   // F/64 blocks of 36 bytes
#define NVFP4MT_STEP   2592  // one ring slot: gate + up row (gu) or 4 down rows (dn)

// cp.async on sm_80+; older archs (never launched there, see ggml_cuda_nvfp4moe_supported) compile to plain copies
static __device__ __forceinline__ void nvfp4mt_cp16(void * s, const void * g) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const unsigned sa = (unsigned) __cvta_generic_to_shared(s);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(g) : "memory");
#else
    *(int4 *) s = *(const int4 *) g;
#endif
}
static __device__ __forceinline__ void nvfp4mt_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}
template <int N>
static __device__ __forceinline__ void nvfp4mt_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
#endif
}

// warp-level, no barrier (every warp computes the same): e = expert of slot blockIdx.y; 0 if an earlier slot routes to e,
// else the mask of slots >= blockIdx.y that route to e (slot s = 8*t + k; T*8 <= 64)
static __device__ __forceinline__ uint64_t nvfp4mt_members(const nvfp4moe_params & P, int & e) {
    const int lane = threadIdx.x & 31, ns = P.T*8, s0 = blockIdx.y;
    const int a = lane, b = lane + 32;
    const int ea = a < ns ? P.ids[(a >> 3)*P.ids_s1 + (a & 7)] : -1;
    const int eb = b < ns ? P.ids[(b >> 3)*P.ids_s1 + (b & 7)] : -1;
    e = P.ids[(s0 >> 3)*P.ids_s1 + (s0 & 7)];
    const uint64_t m = ((uint64_t) __ballot_sync(0xffffffff, eb == e) << 32) | __ballot_sync(0xffffffff, ea == e);
    if (m & ((1ull << s0) - 1)) {
        return 0;
    }
    return m;
}

// slot of the m-th member
static __device__ __forceinline__ int nvfp4mt_slot(uint64_t mem, const int m) {
    for (int i = 0; i < m; ++i) {
        mem &= mem - 1;
    }
    return __ffsll((long long) mem) - 1;
}

// number of values left after the transposed butterfly over offsets O, O/2, .., 1
static constexpr __host__ __device__ int nvfp4mt_nf(const int n, const int o) {
    return o == 0 ? n : nvfp4mt_nf((n + 1)/2, o/2);
}

// Transposed xor butterfly: at offset O, value pairs (2i, 2i+1) split: lanes with bit O clear keep value 2i and add the
// partner's 2i, the others keep 2i+1 and add the partner's 2i+1 (own + partner, as warp_reduce_sum); an odd last value
// does the plain step. Later offsets pair lanes that agree on bit O, so they hold the same value. id: token of each value.
template <int N, int O>
struct nvfp4mt_red {
    static __device__ __forceinline__ void run(float * w, int * id, const int lane) {
        const bool bit = lane & O;
#pragma unroll
        for (int i = 0; i < N/2; ++i) {
            const float send = bit ? w[2*i] : w[2*i + 1];
            const float keep = bit ? w[2*i + 1] : w[2*i];
            const int   kid  = bit ? id[2*i + 1] : id[2*i];
            const float recv = __shfl_xor_sync(0xffffffff, send, O, 32);
            w[i]  = keep + recv;
            id[i] = kid;
        }
        if (N & 1) {
            const float x = w[N - 1];
            w[N/2]  = x + __shfl_xor_sync(0xffffffff, x, O, 32);
            id[N/2] = id[N - 1];
        }
        nvfp4mt_red<(N + 1)/2, O/2>::run(w, id, lane);
    }
};
template <int N>
struct nvfp4mt_red<N, 0> {
    static __device__ __forceinline__ void run(float *, int *, const int) {}
};

// decoded half-block h of a row in shared memory (36-byte blocks: scales at 36*(h/2) + 2*(h%2), codes at + 4 + 16*(h%2))
static __device__ __forceinline__ nvfp4moe_hbd nvfp4mt_hb(const char * row, const int h, const float * ue) {
    const char * blk = row + 36*(h >> 1);
    const uint32_t * cp = (const uint32_t *) (blk + 4 + 16*(h & 1));
    const uint4 c = make_uint4(cp[0], cp[1], cp[2], cp[3]);
    const uint32_t sc = *(const uint16_t *) (blk + 2*(h & 1));
    return nvfp4moe_decode(c, sc, ue, 0x03020100, 0x0C080604, 0xFDFEFF00, 0xF4F8FAFC);  // kvalues_mxfp4
}

// nvfp4moe_hbd_dot with the q8_1 block split in shared memory: qs[0..3] = lo, qs[4..7] = hi, d8
static __device__ __forceinline__ float nvfp4mt_dot(const nvfp4moe_hbd & w, const int4 lo, const int4 hi, const float d8) {
    float sum = 0.0f;
    int sumi = ggml_cuda_dp4a(w.v[0], lo.x, 0);
    sumi = ggml_cuda_dp4a(w.v[1], lo.z, sumi);
    sumi = ggml_cuda_dp4a(w.v[2], lo.y, sumi);
    sumi = ggml_cuda_dp4a(w.v[3], lo.w, sumi);
    sum += (w.da * d8) * float(sumi);
    sumi = ggml_cuda_dp4a(w.v[4], hi.x, 0);
    sumi = ggml_cuda_dp4a(w.v[5], hi.z, sumi);
    sumi = ggml_cuda_dp4a(w.v[6], hi.y, sumi);
    sumi = ggml_cuda_dp4a(w.v[7], hi.w, sumi);
    sum += (w.db * d8) * float(sumi);
    return sum;
}

// q8_1 blocks -> split shared layout (lo / hi 16-byte halves of qs, d); n_rows rows of nb blocks, row r from src_row(r)
template <typename F>
static __device__ __forceinline__ void nvfp4mt_load_q8(int4 * lo, int4 * hi, float * d, const int n_rows, const int nb,
        const F & src_row) {
    for (int i = threadIdx.x; i < n_rows*nb; i += blockDim.x) {
        const int r = i / nb, b = i - r*nb;
        const block_q8_1 * src = src_row(r) + b;
        const int * q = (const int *) src->qs;
        lo[i] = make_int4(q[0], q[1], q[2], q[3]);
        hi[i] = make_int4(q[4], q[5], q[6], q[7]);
        d[i]  = __low2float(src->ds);
    }
}

// gate_up for the NT tokens of one expert: warp w takes features fb .. fb + FPW - 1, one per ring step; lanes 0..15 the
// gate row, 16..31 the up row; lane j of a half = generic lanes j and j + 16 (half-blocks j + 16q). Writes SITU to act_s.
template <int T, int NW, int FPW, int D, int NT>
static __device__ __forceinline__ void nvfp4mt_gu_body(const nvfp4moe_params & P, char * myring, const int4 * xlo,
        const int4 * xhi, const float * xd, float * act_s, const float * ue, const float scale, const int fl0,
        const char * wg, const int fb) {
    constexpr int ROW = NVFP4MT_ROW_GU, NBK = 72, FC = NW*FPW;
    const int lane = threadIdx.x & 31, h = lane >> 4, j = lane & 15;
    for (int i = 0; i < FPW; ++i) {
        nvfp4mt_wait<D - 2>();
        __syncwarp();
        {   // refill the slot of step i - 1 with step i + D - 1
            const int n = i + D - 1;
            if (n < FPW) {
                const char * rg = wg + (int64_t) (fb + n)*P.gu_nb1;
                const char * ru = rg + (int64_t) P.F*P.gu_nb1;
                char * dst = myring + (n % D)*NVFP4MT_STEP;
                for (int c = lane; c < 2*ROW/16; c += 32) {
                    const int u = c >= ROW/16, cc = c - u*(ROW/16);
                    nvfp4mt_cp16(dst + u*ROW + cc*16, (u ? ru : rg) + cc*16);
                }
            }
            nvfp4mt_commit();
        }
        const char * row = myring + (i % D)*NVFP4MT_STEP + h*ROW;
        float acc[NT][2];
#pragma unroll
        for (int q = 0; q < 5; ++q) {
            const int hb = j + 16*q;
            if (q < 4 || j < 8) {
                const nvfp4moe_hbd w = nvfp4mt_hb(row, hb, ue);
#pragma unroll
                for (int m = 0; m < NT; ++m) {
                    const float p = nvfp4mt_dot(w, xlo[m*NBK + hb], xhi[m*NBK + hb], xd[m*NBK + hb]);
                    if (q < 2) {
                        acc[m][q] = 0.0f + p;
                    } else {
                        acc[m][q & 1] += p;
                    }
                }
            }
        }
        float v[NT];
        int id[NT];
#pragma unroll
        for (int m = 0; m < NT; ++m) {
            v[m]  = acc[m][0] + acc[m][1];  // xor 16 (generic lanes j and j + 16)
            id[m] = m;
        }
        nvfp4mt_red<NT, 8>::run(v, id, lane);
        constexpr int NF = nvfp4mt_nf(NT, 8);
#pragma unroll
        for (int q = 0; q < NF; ++q) {
            const float mine  = P.s_gu ? v[q]*scale : v[q];
            const float other = __shfl_xor_sync(0xffffffff, mine, 16, 32);  // same token in the other half
            if (h == 0) {
                act_s[id[q]*FC + fl0 + i] = nvfp4moe_situ(mine, other, P.situ);
            }
        }
    }
}

// grid (F/(NW*FPW), T*8), NW*32 threads. Dynamic shared: xlo, xhi [T][72] int4, ring [NW][D][2592], xd [T][72],
// act_s [T][NW*FPW], ue4m3 [256]. Reads P.xq (x of the T tokens as q8_1, k_nvfp4moe_xq), writes P.act.
template <int T, int NW, int FPW, int D>
static __global__ void __launch_bounds__(NW*32) k_nvfp4mt_gu(const nvfp4moe_params P) {
    constexpr int NBK = 72, FC = NW*FPW;
    static_assert(FC % 32 == 0 && D >= 2, "NVFP4MT gu tile");
    extern __shared__ __align__(16) char smem_mt[];
    int4  * xlo   = (int4 *) smem_mt;
    int4  * xhi   = xlo + T*NBK;
    char  * ring  = (char *) (xhi + T*NBK);
    float * xd    = (float *) (ring + NW*D*NVFP4MT_STEP);
    float * act_s = xd + T*NBK;
    float * ue    = act_s + T*FC;

    int e;
    const uint64_t mem = nvfp4mt_members(P, e);
    if (!mem) {
        return;  // CTA-uniform, before any barrier
    }
    const int n = __popcll((long long) mem);
    if (n > T) {
        __trap();  // a token routed twice to one expert
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int fl0 = warp*FPW, fb = blockIdx.x*FC + fl0;
    const char * wg = P.w_gu + e*P.gu_nb2;
    char * myring = ring + warp*D*NVFP4MT_STEP;
#pragma unroll
    for (int s = 0; s < D - 1; ++s) {
        if (s < FPW) {
            const char * rg = wg + (int64_t) (fb + s)*P.gu_nb1;
            const char * ru = rg + (int64_t) P.F*P.gu_nb1;
            char * dst = myring + s*NVFP4MT_STEP;
            for (int c = lane; c < 2*NVFP4MT_ROW_GU/16; c += 32) {
                const int u = c >= NVFP4MT_ROW_GU/16, cc = c - u*(NVFP4MT_ROW_GU/16);
                nvfp4mt_cp16(dst + u*NVFP4MT_ROW_GU + cc*16, (u ? ru : rg) + cc*16);
            }
        }
        nvfp4mt_commit();
    }
    const float scale = P.s_gu ? P.s_gu[e] : 1.0f;
    nvfp4moe_fill_ue4m3(ue);
    nvfp4mt_load_q8(xlo, xhi, xd, n, NBK, [&](const int r) { return P.xq + (nvfp4mt_slot(mem, r) >> 3)*NBK; });
    __syncthreads();
    switch (n) {
        case 1: nvfp4mt_gu_body<T, NW, FPW, D, 1>(P, myring, xlo, xhi, xd, act_s, ue, scale, fl0, wg, fb); break;
        case 2: if constexpr (T >= 2) nvfp4mt_gu_body<T, NW, FPW, D, 2>(P, myring, xlo, xhi, xd, act_s, ue, scale, fl0, wg, fb); break;
        case 3: if constexpr (T >= 3) nvfp4mt_gu_body<T, NW, FPW, D, 3>(P, myring, xlo, xhi, xd, act_s, ue, scale, fl0, wg, fb); break;
        case 4: if constexpr (T >= 4) nvfp4mt_gu_body<T, NW, FPW, D, 4>(P, myring, xlo, xhi, xd, act_s, ue, scale, fl0, wg, fb); break;
        case 5: if constexpr (T >= 5) nvfp4mt_gu_body<T, NW, FPW, D, 5>(P, myring, xlo, xhi, xd, act_s, ue, scale, fl0, wg, fb); break;
        case 6: if constexpr (T >= 6) nvfp4mt_gu_body<T, NW, FPW, D, 6>(P, myring, xlo, xhi, xd, act_s, ue, scale, fl0, wg, fb); break;
        case 7: if constexpr (T >= 7) nvfp4mt_gu_body<T, NW, FPW, D, 7>(P, myring, xlo, xhi, xd, act_s, ue, scale, fl0, wg, fb); break;
        default: if constexpr (T >= 8) nvfp4mt_gu_body<T, NW, FPW, D, 8>(P, myring, xlo, xhi, xd, act_s, ue, scale, fl0, wg, fb); break;
    }
    nvfp4mt_wait<0>();
    __syncthreads();
    // SITU of feature blockIdx.x*FC + f sits at act_s[m][f]: q8_1 block c of the tile covers f = 32c .. 32c + 31
    const int nb_f = P.F / QK8_1;
    for (int b = warp; b < n*(FC/32); b += NW) {
        const int m = b / (FC/32), c = b - m*(FC/32);
        const int s = nvfp4mt_slot(mem, m);
        nvfp4moe_quant_block(act_s[m*FC + 32*c + lane], P.act + (int64_t) ((s >> 3)*P.n_used + (s & 7))*nb_f + blockIdx.x*(FC/32) + c, lane);
    }
}

// down for the NT tokens of one expert: warp w takes rows r0 + 4i .. r0 + 4i + 3 per ring step (4 contiguous rows),
// 8 lanes per row, lane j = generic lanes j + 8a (a < 4): half-blocks j + 8a, and j + 32 for j < 4. Writes e_out.
template <int T, int NW, int SPW, int D, int NT>
static __device__ __forceinline__ void nvfp4mt_dn_body(const nvfp4moe_params & P, char * myring, const int4 * alo,
        const int4 * ahi, const float * ad, const float * ue, const float scale, const char * wd, const int r0,
        const uint64_t mem) {
    constexpr int ROW = NVFP4MT_ROW_DN, NBF = 36;
    const int lane = threadIdx.x & 31, rq = lane >> 3, j = lane & 7;
    int sl[NT];
#pragma unroll
    for (int m = 0; m < NT; ++m) {
        sl[m] = nvfp4mt_slot(mem, m);
    }
    for (int i = 0; i < SPW; ++i) {
        nvfp4mt_wait<D - 2>();
        __syncwarp();
        {
            const int n = i + D - 1;
            if (n < SPW) {
                const char * src = wd + (int64_t) (r0 + 4*n)*P.dn_nb1;
                char * dst = myring + (n % D)*NVFP4MT_STEP;
                for (int c = lane; c < NVFP4MT_STEP/16; c += 32) {
                    nvfp4mt_cp16(dst + c*16, src + c*16);
                }
            }
            nvfp4mt_commit();
        }
        const char * row = myring + (i % D)*NVFP4MT_STEP + rq*ROW;
        float acc[NT][4];
#pragma unroll
        for (int q = 0; q < 5; ++q) {
            const int hb = q < 4 ? j + 8*q : j + 32;
            if (q < 4 || j < 4) {
                const nvfp4moe_hbd w = nvfp4mt_hb(row, hb, ue);
#pragma unroll
                for (int m = 0; m < NT; ++m) {
                    const float p = nvfp4mt_dot(w, alo[m*NBF + hb], ahi[m*NBF + hb], ad[m*NBF + hb]);
                    if (q < 4) {
                        acc[m][q] = 0.0f + p;
                    } else {
                        acc[m][0] += p;
                    }
                }
            }
        }
        float v[NT];
        int id[NT];
#pragma unroll
        for (int m = 0; m < NT; ++m) {
            v[m]  = (acc[m][0] + acc[m][2]) + (acc[m][1] + acc[m][3]);  // xor 16, xor 8
            id[m] = m;
        }
        nvfp4mt_red<NT, 4>::run(v, id, lane);
        constexpr int NF = nvfp4mt_nf(NT, 4);
        const int r = r0 + 4*i + rq;
#pragma unroll
        for (int q = 0; q < NF; ++q) {
            int s = sl[0];
#pragma unroll
            for (int m = 1; m < NT; ++m) {
                s = id[q] == m ? sl[m] : s;
            }
            P.e_out[(int64_t) ((s >> 3)*P.n_used + (s & 7))*P.K + r] = P.s_dn ? v[q]*scale : v[q];
        }
    }
}

// grid (K/(NW*SPW*4), T*8), NW*32 threads. Dynamic shared: alo, ahi [T][36] int4, ring [NW][D][2592], ad [T][36],
// ue4m3 [256]. Reads P.act (SITU output of the member slots), writes P.e_out. Rows contiguous (dn_nb1 == 648).
template <int T, int NW, int SPW, int D>
static __global__ void __launch_bounds__(NW*32) k_nvfp4mt_dn(const nvfp4moe_params P) {
    constexpr int NBF = 36;
    static_assert(D >= 2, "NVFP4MT dn ring");
    extern __shared__ __align__(16) char smem_mt[];
    int4  * alo  = (int4 *) smem_mt;
    int4  * ahi  = alo + T*NBF;
    char  * ring = (char *) (ahi + T*NBF);
    float * ad   = (float *) (ring + NW*D*NVFP4MT_STEP);
    float * ue   = ad + T*NBF;

    int e;
    const uint64_t mem = nvfp4mt_members(P, e);
    if (!mem) {
        return;
    }
    const int n = __popcll((long long) mem);
    if (n > T) {
        __trap();
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int r0 = (blockIdx.x*NW + warp)*SPW*4;
    const char * wd = P.w_dn + e*P.dn_nb2;
    char * myring = ring + warp*D*NVFP4MT_STEP;
#pragma unroll
    for (int s = 0; s < D - 1; ++s) {
        if (s < SPW) {
            const char * src = wd + (int64_t) (r0 + 4*s)*P.dn_nb1;
            char * dst = myring + s*NVFP4MT_STEP;
            for (int c = lane; c < NVFP4MT_STEP/16; c += 32) {
                nvfp4mt_cp16(dst + c*16, src + c*16);
            }
        }
        nvfp4mt_commit();
    }
    const float scale = P.s_dn ? P.s_dn[e] : 1.0f;
    nvfp4moe_fill_ue4m3(ue);
    nvfp4mt_load_q8(alo, ahi, ad, n, NBF, [&](const int r) {
        const int s = nvfp4mt_slot(mem, r);
        return P.act + (int64_t) ((s >> 3)*P.n_used + (s & 7))*NBF;
    });
    __syncthreads();
    switch (n) {
        case 1: nvfp4mt_dn_body<T, NW, SPW, D, 1>(P, myring, alo, ahi, ad, ue, scale, wd, r0, mem); break;
        case 2: if constexpr (T >= 2) nvfp4mt_dn_body<T, NW, SPW, D, 2>(P, myring, alo, ahi, ad, ue, scale, wd, r0, mem); break;
        case 3: if constexpr (T >= 3) nvfp4mt_dn_body<T, NW, SPW, D, 3>(P, myring, alo, ahi, ad, ue, scale, wd, r0, mem); break;
        case 4: if constexpr (T >= 4) nvfp4mt_dn_body<T, NW, SPW, D, 4>(P, myring, alo, ahi, ad, ue, scale, wd, r0, mem); break;
        case 5: if constexpr (T >= 5) nvfp4mt_dn_body<T, NW, SPW, D, 5>(P, myring, alo, ahi, ad, ue, scale, wd, r0, mem); break;
        case 6: if constexpr (T >= 6) nvfp4mt_dn_body<T, NW, SPW, D, 6>(P, myring, alo, ahi, ad, ue, scale, wd, r0, mem); break;
        case 7: if constexpr (T >= 7) nvfp4mt_dn_body<T, NW, SPW, D, 7>(P, myring, alo, ahi, ad, ue, scale, wd, r0, mem); break;
        default: if constexpr (T >= 8) nvfp4mt_dn_body<T, NW, SPW, D, 8>(P, myring, alo, ahi, ad, ue, scale, wd, r0, mem); break;
    }
    nvfp4mt_wait<0>();
}

template <int T, int NW, int FPW, int D>
static constexpr size_t nvfp4mt_gu_smem() {
    return (size_t) T*72*32 + (size_t) NW*D*NVFP4MT_STEP + (size_t) T*72*4 + (size_t) T*NW*FPW*4 + 256*4;
}
template <int T, int NW, int SPW, int D>
static constexpr size_t nvfp4mt_dn_smem() {
    return (size_t) T*36*32 + (size_t) NW*D*NVFP4MT_STEP + (size_t) T*36*4 + 256*4;
}
