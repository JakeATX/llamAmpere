#include "lsa-select.cuh"
#include "qsa-resident-clock.cuh"
#include "../ggml-sjkvarn.h"
#include <atomic>

// Keys: the score as an order-preserving uint32 (larger score -> larger key), eligible blocks >= 1, ineligible 0.
// A 4-pass 8-bit radix select finds the k-th largest key thr (k = min(kb, eligible) >= 1, so thr >= 1); every key
// above thr is selected, and of the keys equal to thr the need_eq lowest block indices.

#define LSA_SEL_THREADS 256
#define LSA_SEL_MAX_NBS 12288 // 48 KiB of keys in shared memory (n_kv up to ~1.5M cells at B = 128)

// cells mode: one warp per (block, row): out[t][b] = logsumexp over the block's cells of tau*s + mask (-inf if all
// masked). Replaces the full-row softmax + block sums + log of the graph path (a 200K-wide softmax row is one CTA).
template <typename mask_t>
static __global__ void k_lsa_block_lse(
        const float * __restrict__ S, const char * __restrict__ mask, const size_t mask_nb1, float * __restrict__ out,
        const int n_kv, const int nbs, const int B, const float tau) {
    const int lane = threadIdx.x & 31;
    const int b    = blockIdx.x*(blockDim.x/32) + (threadIdx.x >> 5);
    const int t    = blockIdx.y;
    if (b >= nbs) {
        return;
    }
    const int c0 = b*B;
    const int c1 = min(c0 + B, n_kv);
    const float  * s = S + (size_t) t*n_kv;
    const mask_t * m = mask ? (const mask_t *) (mask + (size_t) t*mask_nb1) : nullptr;
    float mx = -INFINITY;
    for (int c = c0 + lane; c < c1; c += 32) {
        const float x = tau*s[c] + (m ? (float) m[c] : 0.0f);
        mx = fmaxf(mx, x);
    }
    mx = warp_reduce_max(mx);
    float sum = 0.0f;
    if (mx != -INFINITY) {
        for (int c = c0 + lane; c < c1; c += 32) {
            const float x = tau*s[c] + (m ? (float) m[c] : 0.0f);
            sum += expf(x - mx);
        }
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0) {
        out[(size_t) t*nbs + b] = mx == -INFINITY ? -INFINITY : mx + logf(sum);
    }
}

// shared = 2: lse[r] = logsumexp over the nbs block scores of row r (-inf if all -inf); one CTA per row
static __global__ void k_lsa_row_lse(const float * __restrict__ S, float * __restrict__ lse, const int nbs) {
    __shared__ float red[LSA_SEL_THREADS/32];
    __shared__ float s_mx;
    const int r = blockIdx.x, tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const float * s = S + (size_t) r*nbs;
    float mx = -INFINITY;
    for (int b = tid; b < nbs; b += LSA_SEL_THREADS) {
        mx = fmaxf(mx, s[b]);
    }
    mx = warp_reduce_max(mx);
    if (lane == 0) {
        red[wid] = mx;
    }
    __syncthreads();
    if (tid == 0) {
        float m = red[0];
        for (int w = 1; w < LSA_SEL_THREADS/32; ++w) {
            m = fmaxf(m, red[w]);
        }
        s_mx = m;
    }
    __syncthreads();
    mx = s_mx;
    float sum = 0.0f;
    if (mx != -INFINITY) {
        for (int b = tid; b < nbs; b += LSA_SEL_THREADS) {
            sum += expf(s[b] - mx);
        }
    }
    sum = warp_reduce_sum(sum);
    __syncthreads();
    if (lane == 0) {
        red[wid] = sum;
    }
    __syncthreads();
    if (tid == 0) {
        float t = 0.0f;
        for (int w = 0; w < LSA_SEL_THREADS/32; ++w) {
            t += red[w];
        }
        lse[r] = mx == -INFINITY ? -INFINITY : mx + logf(t);
    }
}

// summary mode (LLAMA_LSA_SUM_OP): block score from the two F16 summary planes of block b and query row q (d channels)
// LDG = false: q (and the planes) are in shared memory (tiled prepass); same expressions in the same order -> same bits
template <bool LDG = true>
static __device__ __forceinline__ float lsa_sum_score(
        const half * __restrict__ hx, const half * __restrict__ hn, const float * __restrict__ q, const int d,
        const float tau, const int smode) {
    float a = 0.0f, c = 0.0f;
    for (int i = 0; i < d; i += 2) {
        const float2 x = __half22float2(*(const half2 *) (hx + i));
        const float2 n = __half22float2(*(const half2 *) (hn + i));
        const float q0 = LDG ? __ldg(q + i) : q[i], q1 = LDG ? __ldg(q + i + 1) : q[i + 1];
        if (smode == 2) {
            a += q0*x.x;
            a += q1*x.y;
            c += q0*q0*n.x;
            c += q1*q1*n.y;
        } else {
            a += q0 > 0.0f ? q0*x.x : q0*n.x;
            a += q1 > 0.0f ? q1*x.y : q1*n.y;
        }
    }
    return smode == 2 ? tau*a + 0.5f*tau*tau*c : a;
}

// summary scores to scratch [T][nbs] (shared = 2 and the multi-CTA path, which need every row's scores first)
static __global__ void k_lsa_block_sumscore(
        const half * __restrict__ smx, const half * __restrict__ smn, const int s_ld, const float * __restrict__ Q,
        const int d, const float tau, const int smode, float * __restrict__ out, const int nbs) {
    const int b = blockIdx.x*blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (b >= nbs) {
        return;
    }
    out[(size_t) t*nbs + b] = lsa_sum_score(smx + (size_t) b*s_ld, smn + (size_t) b*s_ld, Q + (size_t) t*d, d, tau, smode);
}

// tiled prepass (opt-in GGML_LSA_SUM_TILE=1): one CTA per LSA_SUMT_UNITS units scores them for all T rows. The two
// summary planes of the tile are copied to shared memory with coalesced half2 loads (the one-thread-per-score prepass
// has each lane walk its own 2 x d halfs, d*2 bytes apart from its neighbour, and reads every plane T times); rows are
// padded by 2 halfs so lane u's half2 reads fall in distinct banks (stride d/2 + 1 words). Each thread then runs the
// same sequential lsa_sum_score per row -> identical scores. Dynamic smem: T*d floats + 2 x units x (d + 2) halfs.
#define LSA_SUMT_UNITS 64
static __global__ void __launch_bounds__(LSA_SUMT_UNITS) k_lsa_block_sumscore_tile(
        const half * __restrict__ smx, const half * __restrict__ smn, const int s_ld, const float * __restrict__ Q,
        const int d, const int T, const float tau, const int smode, float * __restrict__ out, const int nbs) {
    extern __shared__ float s_sumt[];
    float * sq = s_sumt;                                    // [T][d]
    half  * sx = (half *) (sq + (size_t) T*d);              // [units][d + 2]
    half  * sn = sx + (size_t) LSA_SUMT_UNITS*(d + 2);      // [units][d + 2]
    const int ldh = d + 2, d2 = d/2;
    const int b0  = blockIdx.x*LSA_SUMT_UNITS;
    const int nu  = min(LSA_SUMT_UNITS, nbs - b0);
    for (int i = threadIdx.x; i < nu*d2; i += LSA_SUMT_UNITS) {
        const int u = i/d2, c = i - u*d2;
        ((half2 *) (sx + (size_t) u*ldh))[c] = ((const half2 *) (smx + (size_t) (b0 + u)*s_ld))[c];
        ((half2 *) (sn + (size_t) u*ldh))[c] = ((const half2 *) (smn + (size_t) (b0 + u)*s_ld))[c];
    }
    for (int i = threadIdx.x; i < T*d; i += LSA_SUMT_UNITS) {
        sq[i] = Q[i];
    }
    __syncthreads();
    const int u = threadIdx.x;
    if (u >= nu) {
        return;
    }
    for (int t = 0; t < T; ++t) {
        out[(size_t) t*nbs + b0 + u] = lsa_sum_score<false>(sx + (size_t) u*ldh, sn + (size_t) u*ldh, sq + (size_t) t*d, d, tau, smode);
    }
}

// LLAMA_LSA_RESIDENT_STICKY (opt-in): S[t][b] += bonus x (share of the unit's two end cells whose 128-cell record is
// in a card slot); unit b covers cells [b*cpu, (b+1)*cpu). -inf scores (masked units) stay -inf. The table is the slot
// state after the previous resolve (the mark/assign launches run later, at this layer's attention).
static __global__ void k_lsa_resident_bonus(float * __restrict__ S, const int nbs, const int T, const int cpu,
        const int32_t * __restrict__ desc, const int32_t * __restrict__ table, const int records, const float bonus) {
    const int b = blockIdx.x*blockDim.x + threadIdx.x;
    if (b >= nbs) {
        return;
    }
    const int s0 = desc[GGML_SJKVARN_DESC_S], s1 = desc[GGML_SJKVARN_DESC_B], G = desc[GGML_SJKVARN_DESC_G];
    if (G != GGML_LSA_PAGE_CELLS) {
        return;
    }
    const int64_t ends[2] = { (int64_t) b*cpu, (int64_t) b*cpu + cpu - 1 };
    float share = 0.0f;
    for (int i = 0; i < 2; ++i) {
        const int64_t c = ends[i];
        if (c < s0 || c >= s0 + (int64_t) ((s1 - s0)/G)*G) {
            continue; // sinks, window or the unsealed tail: on the card anyway, no bonus
        }
        const int g = int((c - s0)/G);
        const int r = desc[GGML_SJKVARN_DESC_TABLE] ? desc[GGML_SJKVARN_DESC_N_ENTRIES + g] : g;
        share += r >= 0 && r < records && table[r] >= 0 ? 0.5f : 0.0f;
    }
    if (share == 0.0f) {
        return;
    }
    for (int t = 0; t < T; ++t) {
        float & x = S[(size_t) t*nbs + b];
        if (x != -INFINITY) {
            x += bonus*share;
        }
    }
}

// LLAMA_LSA_RESIDENT_STICKY: the bonus inputs of a select node (src 7 = page body, src 8 = SJ-KVaRN desc, param 10 =
// bonus, param 11 = cells per scored element), or false if the node has none or the body has no record-page map
struct lsa_sticky_args {
    const int32_t * desc;
    const int32_t * table;
    uint32_t records;
    float bonus;
    int cells;
};

static bool lsa_sticky_get(const ggml_tensor * dst, lsa_sticky_args & a) {
    if (!dst->src[7] || !dst->src[8]) {
        return false;
    }
    a.bonus = ggml_get_op_params_f32(dst, 10);
    if (ggml_get_op_params_i32(dst, 12) == 1 && dst->src[7]->type == GGML_TYPE_I32) { // test mode: the table itself
        a.table   = (const int32_t *) dst->src[7]->data;
        a.records = (uint32_t) dst->src[7]->ne[0];
    } else {
        a.table = ggml_cuda_lsa_page_table(dst->src[7], &a.records);
    }
    if (a.bonus == 0.0f || a.table == nullptr) {
        return false;
    }
    a.desc  = (const int32_t *) dst->src[8]->data;
    a.cells = std::max(1, ggml_get_op_params_i32(dst, 11))*ggml_get_op_params_i32(dst, 0);
    return true;
}

static void lsa_sticky_apply(const lsa_sticky_args & a, float * S, int nbs, int T, cudaStream_t stream) {
    k_lsa_resident_bonus<<<(nbs + 255)/256, 256, 0, stream>>>(S, nbs, T, a.cells, a.desc, a.table, (int) a.records, a.bonus);
    CUDA_CHECK(cudaGetLastError());
}

// NT threads per row CTA: 256 (default) or 1024 (opt-in GGML_LSA_SEL_WIDE=1, nbs >= 2048): same keys, same radix digits,
// same tie order (chunks of NT keys walked in index order) -> identical output; a quarter of the serial loop steps.
template <int NT>
static __global__ void __launch_bounds__(NT) k_lsa_block_select(
        const float * __restrict__ S, const float * __restrict__ cpos, const float * __restrict__ qpos,
        float * __restrict__ dst, const int nbs, const int T, const int B, const int window, const int sinks,
        const int kb, const int shared, const float * __restrict__ gate, const int gate_ld, const int gate_div,
        uint32_t * __restrict__ gkeys, const float * __restrict__ rlse,
        const half * __restrict__ smx, const half * __restrict__ smn, const int s_ld, const float * __restrict__ Q,
        const int d, const float tau, const int smode, const bool fast) {
    // fast (opt-in GGML_LSA_SEL_FAST=1, same output): warp-aggregated histogram adds, stop the radix as soon as the
    // boundary bin is taken whole, and skip the ordered tie walk when every key equal to the threshold is kept.
    // smx != nullptr: summary mode inline (never with shared = 2, which reads S after a scratch pass)
    // keys in shared memory, or in global scratch (gkeys, nbs per row) for unit counts above LSA_SEL_MAX_NBS
    extern __shared__ uint32_t skeys[];
    uint32_t * keys = gkeys ? gkeys + (size_t) blockIdx.x*nbs : skeys;
    __shared__ uint32_t hist[256];
    __shared__ uint32_t warp_cnt[NT/32];
    __shared__ uint32_t s_n_el, s_digit, s_rem, s_run, s_whole;

    const int t    = blockIdx.x;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int wid  = tid >> 5;
    const float qp = qpos[t];

    if (tid == 0) {
        s_n_el = 0;
        s_run  = 0;
    }
    __syncthreads();

    uint32_t n_el = 0;
    for (int b = tid; b < nbs; b += NT) {
        float v;
        if (shared == 2 && T > 1) {
            v = -INFINITY;
            for (int r = 0; r < T; ++r) { // max over rows of the row-normalized score, as the CPU op
                const float l = rlse[r];
                if (l != -INFINITY) {
                    v = fmaxf(v, S[(size_t) r*nbs + b] - l);
                }
            }
        } else if (shared && T > 1) {
            v = 0.0f;
            for (int r = 0; r < T; ++r) { // same order as the CPU op
                v += smx ? lsa_sum_score(smx + (size_t) b*s_ld, smn + (size_t) b*s_ld, Q + (size_t) r*d, d, tau, smode)
                         : S[(size_t) r*nbs + b];
            }
        } else if (smx) {
            v = lsa_sum_score(smx + (size_t) b*s_ld, smn + (size_t) b*s_ld, Q + (size_t) t*d, d, tau, smode);
        } else {
            v = S[(size_t) t*nbs + b];
        }
        const float p = cpos[(size_t) b*B];
        const bool inwin = (p - qp) + ((float) window - 0.5f) > 0.0f;
        const bool insnk = -p + ((float) (sinks - B) + 0.5f) > 0.0f;
        const bool gated = gate && gate[(size_t) t*gate_ld + b/gate_div] <= 0.5f;
        uint32_t key = 0;
        if (!inwin && !insnk && !gated) {
            v = v == 0.0f ? 0.0f : v; // -0 and +0 tie, as the float compare does on the CPU
            uint32_t u = __float_as_uint(v);
            u   = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
            key = max(u, 1u);
            ++n_el;
        }
        keys[b] = key;
    }
    if (n_el) {
        atomicAdd(&s_n_el, n_el);
    }
    __syncthreads();

    float * out = dst + (size_t) t*nbs;
    const uint32_t k = min((uint32_t) kb, s_n_el);
    if (k == 0) {
        for (int b = tid; b < nbs; b += NT) {
            out[b] = 0.0f;
        }
        return;
    }
    // fork: a full eligible set needs no score order or tie break.
    if (k == s_n_el) {
        for (int b = tid; b < nbs; b += NT) {
            out[b] = keys[b] != 0 ? 1.0f : 0.0f;
        }
        return;
    }

    uint32_t prefix = 0, pmask = 0, rem = k;
    bool whole = false; // fast: the boundary bin is taken whole, so every key >= prefix is kept
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = tid; i < 256; i += NT) {
            hist[i] = 0;
        }
        if (tid == 0) {
            s_whole = 0;
        }
        __syncthreads();
#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
        if (fast) {
            // nearby scores share a few top-byte bins: one add per distinct bin per warp instead of up to 32
            for (int base = 0; base < nbs; base += NT) {
                const int b = base + tid;
                const uint32_t key = b < nbs ? keys[b] : 0;
                const bool on = b < nbs && (key & pmask) == prefix;
                const uint32_t bin = on ? (key >> shift) & 255u : 256u;
                const uint32_t grp = __match_any_sync(0xffffffffu, bin);
                if (on && lane == __ffs(grp) - 1) {
                    atomicAdd(&hist[bin], (uint32_t) __popc(grp));
                }
            }
        } else
#endif // __match_any_sync needs Volta+; older archs always take the plain adds
        {
            for (int b = tid; b < nbs; b += NT) {
                const uint32_t key = keys[b];
                if ((key & pmask) == prefix) {
                    atomicAdd(&hist[(key >> shift) & 255u], 1u);
                }
            }
        }
        __syncthreads();
        if (wid == 0) {
            // lane L holds bins 255-8L .. 248-8L (descending); find the bin where the running count from the top
            // reaches rem
            uint32_t c = 0;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                c += hist[255 - 8*lane - j];
            }
            uint32_t incl = c;
#pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                const uint32_t y = __shfl_up_sync(0xffffffffu, incl, off);
                if (lane >= off) {
                    incl += y;
                }
            }
            const uint32_t excl = incl - c;
            if (excl < rem && rem <= incl) {
                uint32_t cum = excl;
                for (int j = 0; j < 8; ++j) {
                    const int d = 255 - 8*lane - j;
                    const uint32_t h = hist[d];
                    if (cum + h >= rem) {
                        s_digit = d;
                        s_rem   = rem - cum;
                        s_whole = cum + h == rem;
                        break;
                    }
                    cum += h;
                }
            }
        }
        __syncthreads();
        prefix |= s_digit << shift;
        pmask  |= 255u << shift;
        rem     = s_rem;
        whole   = fast && s_whole;
        __syncthreads(); // hist is cleared again next pass
        if (whole) {
            break; // block-uniform (shared flag)
        }
    }
    if (whole) {
        // keys above the boundary bin and the whole bin are kept: key >= prefix (lower bytes zero); prefix >= 1
        // because the k-th largest key is eligible (k <= number of eligible keys)
        for (int b = tid; b < nbs; b += NT) {
            out[b] = keys[b] >= prefix ? 1.0f : 0.0f;
        }
        return;
    }
    const uint32_t thr     = prefix;
    const uint32_t need_eq = rem;

    for (int base = 0; base < nbs; base += NT) {
        const int b = base + tid;
        const uint32_t key = b < nbs ? keys[b] : 0;
        const bool eq = key == thr;
        const uint32_t bal = __ballot_sync(0xffffffffu, eq);
        if (lane == 0) {
            warp_cnt[wid] = __popc(bal);
        }
        __syncthreads();
        uint32_t before = s_run, total = 0;
#pragma unroll
        for (int w = 0; w < NT/32; ++w) {
            before += w < wid ? warp_cnt[w] : 0;
            total  += warp_cnt[w];
        }
        before += __popc(bal & ((1u << lane) - 1u));
        if (b < nbs) {
            out[b] = (key > thr || (eq && before < need_eq)) ? 1.0f : 0.0f;
        }
        __syncthreads();
        if (tid == 0) {
            s_run += total;
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Multi-CTA select (opt-in GGML_LSA_SEL_MULTI=1, only for the global-key path: nbs > LSA_SEL_MAX_NBS, i.e. the per-cell
// / small-unit pick at 100K+). k_lsa_block_select runs one CTA per row, which at 100K units makes 4 radix passes over
// 100K global keys inside a single CTA. Here every pass is spread over (nbs/2048) x T CTAs; the per-pass bin search is
// done by the last CTA of each row to finish (atomic counter), so there is no host round trip and no extra launch for it.
// Same keys, same k-th largest key (radix on 12 + 12 + 8 bits instead of 4 x 8 finds the same key and the same rank
// inside it), same tie rule (equal keys: lowest index first) -> output identical to k_lsa_block_select.
// Launches: memset of the scratch + P1 keys/hist(31..20) + P2 hist(19..8) + P3 hist(7..0) + P4 mark.

#define LSA_SEL_MC_CHUNK 2048
#define LSA_SEL_MC_BINS  4096

struct lsa_sel_mc_state {
    uint32_t n_el;    // eligible keys of the row
    uint32_t prefix;  // key bits fixed so far
    uint32_t rem;     // rank still to find inside the prefix
    uint32_t thr;     // k-th largest key
    uint32_t n_eq;    // keys == thr
    uint32_t empty;   // k == 0: nothing selected
    uint32_t cnt;     // CTAs of the row done with the current pass
    uint32_t pad;
};

// last CTA of row t for this pass? (all threads get the answer; resets the counter for the next pass)
static __device__ __forceinline__ bool lsa_sel_mc_last(lsa_sel_mc_state * st) {
    __shared__ uint32_t s_last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        s_last = atomicAdd(&st->cnt, 1u) == gridDim.x - 1;
        if (s_last) {
            st->cnt = 0;
        }
    }
    __syncthreads();
    if (s_last) {
        __threadfence();
    }
    return s_last;
}

// bins h[0..nbins) of row t (global, filled by atomics); find, counting from the top bin down, the bin where the
// running count reaches rem; return it in *s_digit, the rank inside it in *s_rem and its count in *s_cnt. Zeroes h.
// nbins = 256 * per, per in {1, 16}; one CTA of LSA_SEL_THREADS threads.
static __device__ void lsa_sel_mc_scan(uint32_t * h, const int per, const uint32_t rem,
        uint32_t * s_digit, uint32_t * s_rem, uint32_t * s_cnt) {
    __shared__ uint32_t wtot[LSA_SEL_THREADS/32];
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const int top = 256*per - 1 - per*tid; // this thread's bins: top, top-1, ..., top-per+1
    uint32_t c = 0;
    for (int j = 0; j < per; ++j) {
        c += __ldcg(h + top - j);
    }
    uint32_t incl = c;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        const uint32_t y = __shfl_up_sync(0xffffffffu, incl, off);
        if (lane >= off) {
            incl += y;
        }
    }
    if (lane == 31) {
        wtot[wid] = incl;
    }
    __syncthreads();
    for (int w = 0; w < wid; ++w) {
        incl += wtot[w];
    }
    const uint32_t excl = incl - c;
    if (excl < rem && rem <= incl) {
        uint32_t cum = excl;
        for (int j = 0; j < per; ++j) {
            const uint32_t hj = __ldcg(h + top - j);
            if (cum + hj >= rem) {
                *s_digit = top - j;
                *s_rem   = rem - cum;
                *s_cnt   = hj;
                break;
            }
            cum += hj;
        }
    }
    __syncthreads();
    for (int j = 0; j < per; ++j) {
        h[top - j] = 0;
    }
}

// P1: keys (exactly as k_lsa_block_select; cells1 = per-cell scores with unit size 1, the logsumexp of one cell computed
// inline as k_lsa_block_lse does) + eligible count + histogram of key bits 31..20. Last CTA: k, first digit.
template <typename mask_t>
static __global__ void k_lsa_sel_mc_keys(
        const float * __restrict__ S, const char * __restrict__ mask, const size_t mask_nb1, const int n_kv, const float tau,
        const bool cells1, const float * __restrict__ cpos, const float * __restrict__ qpos, const int nbs, const int T,
        const int B, const int window, const int sinks, const int kb, const int shared, const float * __restrict__ gate,
        const int gate_ld, const int gate_div, const float * __restrict__ rlse, uint32_t * __restrict__ keys,
        uint32_t * __restrict__ ghist, lsa_sel_mc_state * __restrict__ st) {
    __shared__ uint32_t sh[LSA_SEL_MC_BINS];
    __shared__ uint32_t s_n_el, s_digit, s_rem, s_cnt;
    const int t = blockIdx.y, tid = threadIdx.x;
    for (int i = tid; i < LSA_SEL_MC_BINS; i += LSA_SEL_THREADS) {
        sh[i] = 0;
    }
    if (tid == 0) {
        s_n_el = 0;
    }
    __syncthreads();
    const float qp = qpos[t];
    auto val = [&](const int r, const int b) -> float {
        if (!cells1) {
            return S[(size_t) r*nbs + b];
        }
        const float x = tau*S[(size_t) r*n_kv + b] + (mask ? (float) ((const mask_t *) (mask + (size_t) r*mask_nb1))[b] : 0.0f);
        return x == -INFINITY ? -INFINITY : x + logf(expf(x - x));
    };
    uint32_t n_el = 0;
    const int b0 = blockIdx.x*LSA_SEL_MC_CHUNK, b1 = min(b0 + LSA_SEL_MC_CHUNK, nbs);
    for (int b = b0 + tid; b < b1; b += LSA_SEL_THREADS) {
        float v;
        if (shared == 2 && T > 1) {
            v = -INFINITY;
            for (int r = 0; r < T; ++r) {
                const float l = rlse[r];
                if (l != -INFINITY) {
                    v = fmaxf(v, val(r, b) - l);
                }
            }
        } else if (shared && T > 1) {
            v = 0.0f;
            for (int r = 0; r < T; ++r) {
                v += val(r, b);
            }
        } else {
            v = val(t, b);
        }
        const float p = cpos[(size_t) b*B];
        const bool inwin = (p - qp) + ((float) window - 0.5f) > 0.0f;
        const bool insnk = -p + ((float) (sinks - B) + 0.5f) > 0.0f;
        const bool gated = gate && gate[(size_t) t*gate_ld + b/gate_div] <= 0.5f;
        uint32_t key = 0;
        if (!inwin && !insnk && !gated) {
            v = v == 0.0f ? 0.0f : v;
            uint32_t u = __float_as_uint(v);
            u   = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
            key = max(u, 1u);
            ++n_el;
            atomicAdd(&sh[key >> 20], 1u); // key 0 (not eligible) never counts: k <= n_el keeps thr >= 1
        }
        keys[(size_t) t*nbs + b] = key;
    }
    n_el = warp_reduce_sum((int) n_el);
    if ((tid & 31) == 0 && n_el) {
        atomicAdd(&s_n_el, n_el);
    }
    __syncthreads();
    uint32_t * gh = ghist + (size_t) t*LSA_SEL_MC_BINS;
    for (int i = tid; i < LSA_SEL_MC_BINS; i += LSA_SEL_THREADS) {
        if (sh[i]) {
            atomicAdd(gh + i, sh[i]);
        }
    }
    if (tid == 0 && s_n_el) {
        atomicAdd(&st[t].n_el, s_n_el);
    }
    if (!lsa_sel_mc_last(&st[t])) {
        return;
    }
    const uint32_t k = min((uint32_t) kb, __ldcg(&st[t].n_el));
    if (k == 0) {
        if (tid == 0) {
            st[t].empty = 1;
        }
        for (int i = tid; i < LSA_SEL_MC_BINS; i += LSA_SEL_THREADS) {
            gh[i] = 0;
        }
        return;
    }
    lsa_sel_mc_scan(gh, LSA_SEL_MC_BINS/256, k, &s_digit, &s_rem, &s_cnt);
    if (tid == 0) {
        st[t].prefix = s_digit;
        st[t].rem    = s_rem;
    }
}

// P2 / P3: histogram of the next key bits (shift 8: bits 19..8, 4096 bins; shift 0: bits 7..0, 256 bins) over the keys
// whose higher bits equal the prefix. Last CTA: next digit (P3: thr, need_eq = rem, n_eq = count of keys == thr).
static __global__ void k_lsa_sel_mc_hist(const uint32_t * __restrict__ keys, uint32_t * __restrict__ ghist,
        lsa_sel_mc_state * __restrict__ st, const int nbs, const int shift) {
    __shared__ uint32_t sh[LSA_SEL_MC_BINS];
    __shared__ uint32_t s_digit, s_rem, s_cnt;
    const int t = blockIdx.y, tid = threadIdx.x;
    if (__ldcg(&st[t].empty)) {
        return; // uniform over the row's CTAs: none of them touches the counter
    }
    const int      nbins  = shift == 8 ? LSA_SEL_MC_BINS : 256;
    const int      hi     = shift == 8 ? 20 : 8;
    const uint32_t prefix = __ldcg(&st[t].prefix);
    for (int i = tid; i < nbins; i += LSA_SEL_THREADS) {
        sh[i] = 0;
    }
    __syncthreads();
    const int b0 = blockIdx.x*LSA_SEL_MC_CHUNK, b1 = min(b0 + LSA_SEL_MC_CHUNK, nbs);
    for (int b = b0 + tid; b < b1; b += LSA_SEL_THREADS) {
        const uint32_t key = keys[(size_t) t*nbs + b];
        if (key && (key >> hi) == prefix) {
            atomicAdd(&sh[(key >> shift) & (uint32_t) (nbins - 1)], 1u);
        }
    }
    __syncthreads();
    uint32_t * gh = ghist + (size_t) t*LSA_SEL_MC_BINS;
    for (int i = tid; i < nbins; i += LSA_SEL_THREADS) {
        if (sh[i]) {
            atomicAdd(gh + i, sh[i]);
        }
    }
    if (!lsa_sel_mc_last(&st[t])) {
        return;
    }
    lsa_sel_mc_scan(gh, nbins/256, __ldcg(&st[t].rem), &s_digit, &s_rem, &s_cnt);
    if (tid == 0) {
        const uint32_t p = (prefix << (hi - shift)) | s_digit;
        st[t].prefix = p;
        st[t].rem    = s_rem;
        if (shift == 0) {
            st[t].thr  = p;
            st[t].n_eq = s_cnt;
        }
    }
}

// P4: out = key > thr, or key == thr and among the first need_eq equal keys by index. When every equal key is taken
// (need_eq == n_eq, the usual case) each CTA marks its own; otherwise the last CTA of the row marks the equal keys in
// index order (as the single-CTA kernel's ballot walk does).
static __global__ void k_lsa_sel_mc_mark(const uint32_t * __restrict__ keys, lsa_sel_mc_state * __restrict__ st,
        float * __restrict__ dst, const int nbs) {
    __shared__ uint32_t warp_cnt[LSA_SEL_THREADS/32];
    __shared__ uint32_t s_run;
    const int t = blockIdx.y, tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const bool     empty  = __ldcg(&st[t].empty) != 0;
    const uint32_t thr    = empty ? 0xffffffffu : __ldcg(&st[t].thr);
    const uint32_t needeq = empty ? 0 : __ldcg(&st[t].rem);
    const bool     all_eq = !empty && needeq == __ldcg(&st[t].n_eq);
    const uint32_t * kr = keys + (size_t) t*nbs;
    float * out = dst + (size_t) t*nbs;
    const int b0 = blockIdx.x*LSA_SEL_MC_CHUNK, b1 = min(b0 + LSA_SEL_MC_CHUNK, nbs);
    for (int b = b0 + tid; b < b1; b += LSA_SEL_THREADS) {
        const uint32_t key = kr[b];
        out[b] = !empty && (key > thr || (key == thr && all_eq)) ? 1.0f : 0.0f;
    }
    if (empty || all_eq) {
        return;
    }
    if (!lsa_sel_mc_last(&st[t])) {
        return;
    }
    if (tid == 0) {
        s_run = 0;
    }
    __syncthreads();
    for (int base = 0; base < nbs; base += LSA_SEL_THREADS) {
        const int b = base + tid;
        const bool eq = b < nbs && __ldcg(kr + b) == thr;
        const uint32_t bal = __ballot_sync(0xffffffffu, eq);
        if (lane == 0) {
            warp_cnt[wid] = __popc(bal);
        }
        __syncthreads();
        uint32_t before = s_run, total = 0;
#pragma unroll
        for (int w = 0; w < LSA_SEL_THREADS/32; ++w) {
            before += w < wid ? warp_cnt[w] : 0;
            total  += warp_cnt[w];
        }
        before += __popc(bal & ((1u << lane) - 1u));
        if (eq) {
            out[b] = before < needeq ? 1.0f : 0.0f;
        }
        __syncthreads();
        if (tid == 0) {
            s_run += total;
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Slice-local quota pick (op param 13 = units per slice > 0, opt-in LLAMA_LSA_SLICE): slice j of a row keeps
// q_j = floor(k*cnt_j/tot) of its own best units (+1 for the k - sum q slices with the largest remainders, ties: lower
// slice first), same keys and tie rule (equal keys: lowest unit first) as k_lsa_block_select inside each slice. Integer
// quotas, so the CPU op gives the same mask. Grid (slices, rows); keys in global scratch.

// eligibility of unit b for row t (outside the dense window and the sinks, and kept by the stage-1 gate)
static __device__ __forceinline__ bool lsa_slice_elig(
        const float * __restrict__ cpos, const float qp, const int B, const int window, const int sinks,
        const float * __restrict__ gate, const int gate_ld, const int gate_div, const int t, const int b) {
    const float p = cpos[(size_t) b*B];
    const bool inwin = (p - qp) + ((float) window - 0.5f) > 0.0f;
    const bool insnk = -p + ((float) (sinks - B) + 0.5f) > 0.0f;
    const bool gated = gate && gate[(size_t) t*gate_ld + b/gate_div] <= 0.5f;
    return !inwin && !insnk && !gated;
}

// select key of unit b for row t (exactly as k_lsa_block_select, scores from S); 0 = not eligible
static __device__ __forceinline__ uint32_t lsa_slice_key(
        const float * __restrict__ S, const float * __restrict__ cpos, const float qp, const int nbs, const int T,
        const int B, const int window, const int sinks, const int shared, const float * __restrict__ gate,
        const int gate_ld, const int gate_div, const float * __restrict__ rlse, const int t, const int b) {
    if (!lsa_slice_elig(cpos, qp, B, window, sinks, gate, gate_ld, gate_div, t, b)) {
        return 0;
    }
    float v;
    if (shared == 2 && T > 1) {
        v = -INFINITY;
        for (int r = 0; r < T; ++r) {
            const float l = rlse[r];
            if (l != -INFINITY) {
                v = fmaxf(v, S[(size_t) r*nbs + b] - l);
            }
        }
    } else if (shared && T > 1) {
        v = 0.0f;
        for (int r = 0; r < T; ++r) {
            v += S[(size_t) r*nbs + b];
        }
    } else {
        v = S[(size_t) t*nbs + b];
    }
    v = v == 0.0f ? 0.0f : v;
    uint32_t u = __float_as_uint(v);
    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return max(u, 1u);
}

// keys of one slice + its eligible count
static __global__ void k_lsa_slice_keys(
        const float * __restrict__ S, const float * __restrict__ cpos, const float * __restrict__ qpos,
        const int nbs, const int T, const int B, const int window, const int sinks, const int shared,
        const float * __restrict__ gate, const int gate_ld, const int gate_div, const float * __restrict__ rlse,
        const int upc, uint32_t * __restrict__ keys, uint32_t * __restrict__ cnt) {
    __shared__ uint32_t s_n;
    const int j = blockIdx.x, t = blockIdx.y, ns = gridDim.x;
    const int b0 = j*upc, b1 = min(nbs, b0 + upc);
    const float qp = qpos[t];
    if (threadIdx.x == 0) {
        s_n = 0;
    }
    __syncthreads();
    uint32_t n_el = 0;
    for (int b = b0 + threadIdx.x; b < b1; b += blockDim.x) {
        const uint32_t key = lsa_slice_key(S, cpos, qp, nbs, T, B, window, sinks, shared, gate, gate_ld, gate_div, rlse, t, b);
        n_el += key ? 1u : 0u;
        keys[(size_t) t*nbs + b] = key;
    }
    if (n_el) {
        atomicAdd(&s_n, n_el);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        cnt[(size_t) t*ns + j] = s_n;
    }
}

// quota of slice j from the ns eligible counts ck, then the radix select of its n keys kk (global or shared memory)
template <int NT>
static __device__ __forceinline__ void lsa_slice_select_body(
        const uint32_t * kk, const uint32_t * ck, float * __restrict__ out, const int n, const int ns, const int j,
        const int kb) {
    __shared__ uint32_t hist[256];
    __shared__ uint32_t warp_cnt[NT/32];
    __shared__ unsigned long long s_tot, s_qs;
    __shared__ uint32_t s_rank, s_digit, s_rem, s_run;
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;

    if (tid == 0) {
        s_tot = 0; s_qs = 0; s_rank = 0; s_run = 0;
    }
    __syncthreads();
    unsigned long long tp = 0;
    for (int i = tid; i < ns; i += NT) {
        tp += ck[i];
    }
    atomicAdd(&s_tot, tp);
    __syncthreads();
    const unsigned long long tot = s_tot;
    const unsigned long long k   = min((unsigned long long) kb, tot);
    const unsigned long long cj  = ck[j];
    const unsigned long long rj  = k ? k*cj % tot : 0;
    unsigned long long qp = 0;
    uint32_t rk = 0;
    for (int i = tid; i < ns && k; i += NT) {
        const unsigned long long ci = ck[i], ri = k*ci % tot;
        qp += k*ci/tot;
        rk += (ri > rj || (ri == rj && i < j)) ? 1u : 0u;
    }
    atomicAdd(&s_qs, qp);
    atomicAdd(&s_rank, rk);
    __syncthreads();
    unsigned long long kj = k ? k*cj/tot + ((unsigned long long) s_rank < k - s_qs ? 1ull : 0ull) : 0ull;
    kj = min(kj, cj);
    if (kj == 0) {
        for (int b = tid; b < n; b += NT) {
            out[b] = 0.0f;
        }
        return;
    }

    uint32_t prefix = 0, pmask = 0, rem = (uint32_t) kj;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = tid; i < 256; i += NT) {
            hist[i] = 0;
        }
        __syncthreads();
        for (int b = tid; b < n; b += NT) {
            const uint32_t key = kk[b];
            if ((key & pmask) == prefix) {
                atomicAdd(&hist[(key >> shift) & 255u], 1u);
            }
        }
        __syncthreads();
        if (wid == 0) {
            uint32_t c = 0;
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                c += hist[255 - 8*lane - q];
            }
            uint32_t incl = c;
#pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                const uint32_t y = __shfl_up_sync(0xffffffffu, incl, off);
                if (lane >= off) {
                    incl += y;
                }
            }
            const uint32_t excl = incl - c;
            if (excl < rem && rem <= incl) {
                uint32_t cum = excl;
                for (int q = 0; q < 8; ++q) {
                    const int d = 255 - 8*lane - q;
                    const uint32_t h = hist[d];
                    if (cum + h >= rem) {
                        s_digit = d;
                        s_rem   = rem - cum;
                        break;
                    }
                    cum += h;
                }
            }
        }
        __syncthreads();
        prefix |= s_digit << shift;
        pmask  |= 255u << shift;
        rem     = s_rem;
        __syncthreads();
    }
    const uint32_t thr = prefix, need_eq = rem;
    for (int base = 0; base < n; base += NT) {
        const int b = base + tid;
        const uint32_t key = b < n ? kk[b] : 0;
        const bool eq = key == thr;
        const uint32_t bal = __ballot_sync(0xffffffffu, eq);
        if (lane == 0) {
            warp_cnt[wid] = __popc(bal);
        }
        __syncthreads();
        uint32_t before = s_run, total = 0;
#pragma unroll
        for (int w = 0; w < NT/32; ++w) {
            before += w < wid ? warp_cnt[w] : 0;
            total  += warp_cnt[w];
        }
        before += __popc(bal & ((1u << lane) - 1u));
        if (b < n) {
            out[b] = (key > thr || (eq && before < need_eq)) ? 1.0f : 0.0f;
        }
        __syncthreads();
        if (tid == 0) {
            s_run += total;
        }
        __syncthreads();
    }
}

template <int NT>
static __global__ void __launch_bounds__(NT) k_lsa_slice_select(
        const uint32_t * __restrict__ keys, const uint32_t * __restrict__ cnt, float * __restrict__ dst,
        const int nbs, const int kb, const int upc) {
    const int j = blockIdx.x, t = blockIdx.y, ns = gridDim.x;
    const int b0 = j*upc, n = min(nbs, b0 + upc) - b0;
    lsa_slice_select_body<NT>(keys + (size_t) t*nbs + b0, cnt + (size_t) t*ns, dst + (size_t) t*nbs + b0, n, ns, j, kb);
}

// GGML_LSA_SLICE_FUSED=1 (opt-in): the same slice pick in ONE launch. Each CTA (slice j, row t) counts the eligible
// units of every slice of its row (positions and gate only, no scores), writes its own slice's keys to shared memory
// and selects from there: no global key scratch, no count round trip, no second launch. Same keys, quotas and tie
// rule as k_lsa_slice_keys + k_lsa_slice_select -> the same mask.
template <int NT>
static __global__ void __launch_bounds__(NT) k_lsa_slice_fused(
        const float * __restrict__ S, const float * __restrict__ cpos, const float * __restrict__ qpos,
        const int nbs, const int T, const int B, const int window, const int sinks, const int shared,
        const float * __restrict__ gate, const int gate_ld, const int gate_div, const float * __restrict__ rlse,
        const int upc, const int kb, float * __restrict__ dst) {
    extern __shared__ uint32_t s_dyn[]; // [ns] eligible counts, then [upc] keys of this slice
    const int j = blockIdx.x, t = blockIdx.y, ns = gridDim.x;
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    uint32_t * s_cnt = s_dyn;
    uint32_t * s_key = s_dyn + ns;
    const float qp = qpos[t];
    for (int i = wid; i < ns; i += NT/32) {
        const int c0 = i*upc, c1 = min(nbs, c0 + upc);
        uint32_t c = 0;
        for (int b = c0 + lane; b < c1; b += 32) {
            c += lsa_slice_elig(cpos, qp, B, window, sinks, gate, gate_ld, gate_div, t, b) ? 1u : 0u;
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            c += __shfl_xor_sync(0xffffffffu, c, off);
        }
        if (lane == 0) {
            s_cnt[i] = c;
        }
    }
    const int b0 = j*upc, n = min(nbs, b0 + upc) - b0;
    for (int b = threadIdx.x; b < n; b += NT) {
        s_key[b] = lsa_slice_key(S, cpos, qp, nbs, T, B, window, sinks, shared, gate, gate_ld, gate_div, rlse, t, b0 + b);
    }
    __syncthreads();
    lsa_slice_select_body<NT>(s_key, s_cnt, dst + (size_t) t*nbs + b0, n, ns, j, kb);
}

static bool lsa_slice_fused_enabled() {
    static const bool on = [] {
        const char * e = getenv("GGML_LSA_SLICE_FUSED");
        return e != nullptr && atoi(e) != 0;
    }();
    return on;
}

static bool lsa_sel_multi_enabled() {
    static const bool on = [] {
        const char * e = getenv("GGML_LSA_SEL_MULTI");
        return e != nullptr && atoi(e) != 0;
    }();
    return on;
}

// block summaries are scored inside the select launch only up to this many units per row (GGML_LSA_SUM_INLINE_MAX,
// default 256). Above it the single CTA per row is latency bound on the summary reads (so1b: 16-cell units at 100K,
// 6,272 units x 64 dims, 291 us at T=1 and 1,177 us for the 5-row shared sum), so the scores come from the multi-CTA
// prepass instead. Same per-score arithmetic and the same row-sum order in both routes -> identical keep masks.
static int lsa_sum_inline_max() {
    static const int v = [] {
        const char * e = getenv("GGML_LSA_SUM_INLINE_MAX");
        return e != nullptr ? atoi(e) : 256;
    }();
    return v;
}

static bool lsa_sel_fast_enabled() {
    static const bool on = [] {
        const char * e = getenv("GGML_LSA_SEL_FAST");
        return e != nullptr && atoi(e) != 0;
    }();
    return on;
}

static bool lsa_sum_tile_enabled() {
    static const bool on = [] {
        const char * e = getenv("GGML_LSA_SUM_TILE");
        return e != nullptr && atoi(e) != 0;
    }();
    return on;
}

// GGML_LSA_SUM_TILE_MIN (default 2048): the tiled prepass only runs at >= this many units per row. tl1 (11-05):
// 0.85x at 6,272 units but 1.15x slower at 784 (one 64-unit CTA per tile leaves most SMs idle); both kernels give
// the same scores, so the cut-over changes speed only
static int lsa_sum_tile_min() {
    static const int v = [] {
        const char * e = getenv("GGML_LSA_SUM_TILE_MIN");
        return e != nullptr ? atoi(e) : 2048;
    }();
    return v;
}

static bool lsa_sel_wide_enabled() {
    static const bool on = [] {
        const char * e = getenv("GGML_LSA_SEL_WIDE");
        return e != nullptr && atoi(e) != 0;
    }();
    return on;
}

// summary mode: half2 loads need an even channel count and 4-byte aligned rows
static bool lsa_sum_supported(const ggml_tensor * op) {
    const ggml_tensor * x = op->src[5];
    const ggml_tensor * n = op->src[6];
    return x && n && x->type == GGML_TYPE_F16 && n->type == GGML_TYPE_F16 && x->nb[0] == 2 && n->nb[0] == 2 &&
        x->nb[1] == n->nb[1] && x->nb[1] % 4 == 0 && x->ne[1] >= op->ne[0] && n->ne[1] >= op->ne[0] &&
        op->src[0]->ne[0] % 2 == 0 && op->src[0]->ne[0] == x->ne[0] && ((uintptr_t) x->data) % 4 == 0 &&
        ((uintptr_t) n->data) % 4 == 0 && op->src[3] == nullptr && op->src[4] == nullptr &&
        x->nb[1]/2 <= INT32_MAX;
}

bool ggml_cuda_lsa_block_select_supported(const ggml_tensor * op) {
    const ggml_tensor * a = op->src[0];
    const ggml_tensor * m = op->src[3];
    return a->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 && op->src[2]->type == GGML_TYPE_F32 &&
        op->type == GGML_TYPE_F32 && ggml_is_contiguous(a) && op->ne[0] >= 1 && op->ne[0] <= INT32_MAX/4 &&
        a->ne[1] <= 65535 && a->ne[0] <= INT32_MAX/2 && (m == nullptr || m->type == GGML_TYPE_F16 || m->type == GGML_TYPE_F32) &&
        (op->src[4] == nullptr || (op->src[4]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[4]))) &&
        (ggml_get_op_params_i32(op, 5) != 2 || lsa_sum_supported(op));
}

void ggml_cuda_lsa_block_select(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * a = dst->src[0];
    const ggml_tensor * m = dst->src[3];
    const int nbs   = (int) dst->ne[0];
    const int T     = (int) a->ne[1];
    const int B     = ggml_get_op_params_i32(dst, 0);
    const bool cells = ggml_get_op_params_i32(dst, 5) == 1;
    const bool sumq  = ggml_get_op_params_i32(dst, 5) == 2; // block summaries (src 5/6), q rows in src 0
    const int slice_u = ggml_get_op_params_i32(dst, 13);    // > 0: slice-local quota pick (k_lsa_slice_*)
    GGML_ASSERT(ggml_cuda_lsa_block_select_supported(dst));
    cudaStream_t stream = ctx.stream();
    const half * smx  = sumq ? (const half *) dst->src[5]->data : nullptr;
    const half * smn  = sumq ? (const half *) dst->src[6]->data : nullptr;
    const int    s_ld = sumq ? (int) (dst->src[5]->nb[1]/2) : 0;
    const int    sd   = sumq ? (int) a->ne[0] : 0;
    const float  stau = sumq ? ggml_get_op_params_f32(dst, 6) : 1.0f;
    const int    smod = sumq ? ggml_get_op_params_i32(dst, 8) : 0;
    auto sum_prepass = [&](float * out) {
        if (lsa_sum_tile_enabled() && sd % 2 == 0 && nbs >= lsa_sum_tile_min()) {
            const size_t smem_t = (size_t) T*sd*sizeof(float) + 2*(size_t) LSA_SUMT_UNITS*(sd + 2)*sizeof(half);
            if (smem_t <= 48*1024) {
                k_lsa_block_sumscore_tile<<<(nbs + LSA_SUMT_UNITS - 1)/LSA_SUMT_UNITS, LSA_SUMT_UNITS, smem_t, stream>>>(
                    smx, smn, s_ld, (const float *) a->data, sd, (int) T, stau, smod, out, nbs);
                CUDA_CHECK(cudaGetLastError());
                return;
            }
        }
        // 64-thread CTAs: at 6,272 units a 256-thread grid fills only 25 SMs (one thread per score, same math)
        const dim3 grid((nbs + 63)/64, T);
        k_lsa_block_sumscore<<<grid, 64, 0, stream>>>(smx, smn, s_ld, (const float *) a->data, sd, stau, smod, out, nbs);
        CUDA_CHECK(cudaGetLastError());
    };

    if (nbs > LSA_SEL_MAX_NBS && lsa_sel_multi_enabled() && slice_u <= 0) {
        const int shared_mc = ggml_get_op_params_i32(dst, 4);
        const int n_kv_mc   = (int) a->ne[0];
        const ggml_tensor * g_mc = dst->src[4];
        // per-cell scores with unit size 1: the one-cell logsumexp is computed inside P1 (not for shared = 2, which
        // needs the row logsumexp over all units first)
        const bool cells1 = cells && B == 1 && !(shared_mc == 2 && T > 1);
        const float * Smc = (const float *) a->data;
        ggml_cuda_pool_alloc<float> lse_mc(ctx.pool());
        if (sumq) {
            lse_mc.alloc((size_t) nbs*T);
            sum_prepass(lse_mc.get());
            Smc = lse_mc.get();
        } else if (cells && !cells1) {
            const float tau = ggml_get_op_params_f32(dst, 6);
            lse_mc.alloc((size_t) nbs*T);
            const dim3 grid((nbs + 7)/8, T);
            if (m && m->type == GGML_TYPE_F16) {
                k_lsa_block_lse<half><<<grid, 256, 0, stream>>>(Smc, (const char *) m->data, m->nb[1], lse_mc.get(), n_kv_mc, nbs, B, tau);
            } else {
                k_lsa_block_lse<float><<<grid, 256, 0, stream>>>(Smc, m ? (const char *) m->data : nullptr, m ? m->nb[1] : 0,
                    lse_mc.get(), n_kv_mc, nbs, B, tau);
            }
            CUDA_CHECK(cudaGetLastError());
            Smc = lse_mc.get();
        }
        lsa_sticky_args sticky;
        if (lsa_sticky_get(dst, sticky)) {
            if (!cells1) {
                if (Smc == (const float *) a->data) { // plain block scores: copy (a feeds other nodes)
                    lse_mc.alloc((size_t) nbs*T);
                    CUDA_CHECK(cudaMemcpyAsync(lse_mc.get(), Smc, (size_t) nbs*T*sizeof(float), cudaMemcpyDeviceToDevice, stream));
                    Smc = lse_mc.get();
                }
                lsa_sticky_apply(sticky, lse_mc.get(), nbs, T, stream);
            } else {
                static bool warned = false;
                if (!warned) {
                    warned = true;
                    GGML_LOG_WARN("LSA RESIDENT_STICKY: one-cell picks above %d units score inside the select; no bonus\n",
                        LSA_SEL_MAX_NBS);
                }
            }
        }
        ggml_cuda_pool_alloc<float> rlse_mc(ctx.pool());
        if (shared_mc == 2 && T > 1) {
            rlse_mc.alloc(T);
            k_lsa_row_lse<<<T, LSA_SEL_THREADS, 0, stream>>>(Smc, rlse_mc.get(), nbs);
            CUDA_CHECK(cudaGetLastError());
        }
        // scratch: [T] state + [T][4096] histogram (zeroed) + [T][nbs] keys
        const size_t n_st = (size_t) T*sizeof(lsa_sel_mc_state)/sizeof(uint32_t);
        const size_t n_z  = n_st + (size_t) T*LSA_SEL_MC_BINS;
        ggml_cuda_pool_alloc<uint32_t> scr(ctx.pool(), n_z + (size_t) T*nbs);
        CUDA_CHECK(cudaMemsetAsync(scr.get(), 0, n_z*sizeof(uint32_t), stream));
        lsa_sel_mc_state * st = (lsa_sel_mc_state *) scr.get();
        uint32_t * gh   = scr.get() + n_st;
        uint32_t * keys = scr.get() + n_z;
        const dim3 grid((nbs + LSA_SEL_MC_CHUNK - 1)/LSA_SEL_MC_CHUNK, T);
        const float tau1 = cells1 ? ggml_get_op_params_f32(dst, 6) : 1.0f;
        const float * gp = g_mc ? (const float *) g_mc->data : nullptr;
        const int gld = g_mc ? (int) g_mc->ne[0] : 0, gdiv = g_mc ? ggml_get_op_params_i32(dst, 7) : 1;
        const float * rl = shared_mc == 2 && T > 1 ? rlse_mc.get() : nullptr;
        const int win = ggml_get_op_params_i32(dst, 1), snk = ggml_get_op_params_i32(dst, 2), kb = ggml_get_op_params_i32(dst, 3);
        if (cells1 && m && m->type == GGML_TYPE_F16) {
            k_lsa_sel_mc_keys<half><<<grid, LSA_SEL_THREADS, 0, stream>>>(Smc, (const char *) m->data, m->nb[1], n_kv_mc, tau1,
                true, (const float *) dst->src[1]->data, (const float *) dst->src[2]->data, nbs, T, B, win, snk, kb, shared_mc,
                gp, gld, gdiv, rl, keys, gh, st);
        } else {
            k_lsa_sel_mc_keys<float><<<grid, LSA_SEL_THREADS, 0, stream>>>(Smc, cells1 && m ? (const char *) m->data : nullptr,
                cells1 && m ? m->nb[1] : 0, n_kv_mc, tau1, cells1, (const float *) dst->src[1]->data,
                (const float *) dst->src[2]->data, nbs, T, B, win, snk, kb, shared_mc, gp, gld, gdiv, rl, keys, gh, st);
        }
        CUDA_CHECK(cudaGetLastError());
        k_lsa_sel_mc_hist<<<grid, LSA_SEL_THREADS, 0, stream>>>(keys, gh, st, nbs, 8);
        CUDA_CHECK(cudaGetLastError());
        k_lsa_sel_mc_hist<<<grid, LSA_SEL_THREADS, 0, stream>>>(keys, gh, st, nbs, 0);
        CUDA_CHECK(cudaGetLastError());
        k_lsa_sel_mc_mark<<<grid, LSA_SEL_THREADS, 0, stream>>>(keys, st, (float *) dst->data, nbs);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const float * S = (const float *) a->data;
    ggml_cuda_pool_alloc<float> lse(ctx.pool());
    if (cells) {
        const int   n_kv = (int) a->ne[0];
        const float tau  = ggml_get_op_params_f32(dst, 6);
        lse.alloc((size_t) nbs*T);
        const dim3 grid((nbs + 7)/8, T);
        if (m && m->type == GGML_TYPE_F16) {
            k_lsa_block_lse<half><<<grid, 256, 0, stream>>>(S, (const char *) m->data, m->nb[1], lse.get(), n_kv, nbs, B, tau);
        } else {
            k_lsa_block_lse<float><<<grid, 256, 0, stream>>>(S, m ? (const char *) m->data : nullptr, m ? m->nb[1] : 0,
                lse.get(), n_kv, nbs, B, tau);
        }
        CUDA_CHECK(cudaGetLastError());
        S = lse.get();
    }
    const int shared = ggml_get_op_params_i32(dst, 4);
    const bool sum_inline = sumq && !(shared == 2 && T > 1) && nbs <= lsa_sum_inline_max() && slice_u <= 0; // scores inside the select launch
    if (sumq && !sum_inline) {
        lse.alloc((size_t) nbs*T);
        sum_prepass(lse.get());
        S = lse.get();
    }
    lsa_sticky_args sticky;
    if (lsa_sticky_get(dst, sticky)) {
        if (S == (const float *) a->data) { // plain scores or inline summaries: copy (a is a graph input of other nodes)
            if (sum_inline) {
                lse.alloc((size_t) nbs*T);
                sum_prepass(lse.get());
            } else {
                lse.alloc((size_t) nbs*T);
                CUDA_CHECK(cudaMemcpyAsync(lse.get(), S, (size_t) nbs*T*sizeof(float), cudaMemcpyDeviceToDevice, stream));
            }
            S = lse.get();
        }
        lsa_sticky_apply(sticky, lse.get(), nbs, T, stream);
    }
    const bool sel_inline = sum_inline && S == (const float *) a->data;
    const ggml_tensor * g = dst->src[4];
    ggml_cuda_pool_alloc<uint32_t> gkeys(ctx.pool());
    if (nbs > LSA_SEL_MAX_NBS || (slice_u > 0 && !(lsa_slice_fused_enabled() &&
            ((size_t) (nbs + slice_u - 1)/slice_u + (size_t) slice_u)*sizeof(uint32_t) <= 40*1024))) {
        gkeys.alloc((size_t) nbs*T);
    }
    ggml_cuda_pool_alloc<float> rlse(ctx.pool());
    if (shared == 2 && T > 1) {
        rlse.alloc(T);
        k_lsa_row_lse<<<T, LSA_SEL_THREADS, 0, stream>>>(S, rlse.get(), nbs);
        CUDA_CHECK(cudaGetLastError());
    }
    // wide: not with inline summaries (summary scoring per thread would spill at 1,024 threads)
    const bool wide = lsa_sel_wide_enabled() && nbs >= 2048 && !sel_inline;
    const size_t smem = nbs > LSA_SEL_MAX_NBS ? 0 : (size_t) nbs*sizeof(uint32_t);
    const float * cp = (const float *) dst->src[1]->data;
    const float * qp = (const float *) dst->src[2]->data;
    float * out = (float *) dst->data;
    const int win = ggml_get_op_params_i32(dst, 1), snk = ggml_get_op_params_i32(dst, 2), kb = ggml_get_op_params_i32(dst, 3);
    const float * gp = g ? (const float *) g->data : nullptr;
    const int gld = g ? (int) g->ne[0] : 0, gdiv = g ? ggml_get_op_params_i32(dst, 7) : 1;
    uint32_t * gk = nbs > LSA_SEL_MAX_NBS ? gkeys.get() : nullptr;
    const float * rl = shared == 2 && T > 1 ? rlse.get() : nullptr;
    if (slice_u > 0) {
        const int ns = (nbs + slice_u - 1)/slice_u;
        GGML_ASSERT(ns <= 65535);
        static std::atomic<bool> logged_slice{false};
        if (!logged_slice.exchange(true)) {
            GGML_LOG_WARN("LSA select kernel: slice-local quota pick, %d units per slice (%d slices, units %d, rows %d, shared %d)\n",
                slice_u, ns, nbs, T, shared);
        }
        const dim3 grid(ns, T);
        const size_t fsm = ((size_t) ns + (size_t) slice_u)*sizeof(uint32_t);
        if (lsa_slice_fused_enabled() && fsm <= 40*1024) { // one launch, keys in shared memory (else the 2-kernel route)
            static std::atomic<bool> logged_fused{false};
            if (!logged_fused.exchange(true)) {
                GGML_LOG_WARN("LSA select kernel: fused slice pick, one launch (%zu B shared, %d slices)\n", fsm, ns);
            }
            k_lsa_slice_fused<LSA_SEL_THREADS><<<grid, LSA_SEL_THREADS, fsm, stream>>>(S, cp, qp, nbs, T, B, win, snk,
                shared, gp, gld, gdiv, rl, slice_u, kb, out);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        ggml_cuda_pool_alloc<uint32_t> scnt(ctx.pool(), (size_t) ns*T);
        k_lsa_slice_keys<<<grid, LSA_SEL_THREADS, 0, stream>>>(S, cp, qp, nbs, T, B, win, snk, shared, gp, gld, gdiv, rl,
            slice_u, gkeys.get(), scnt.get());
        CUDA_CHECK(cudaGetLastError());
        k_lsa_slice_select<LSA_SEL_THREADS><<<grid, LSA_SEL_THREADS, 0, stream>>>(gkeys.get(), scnt.get(), out, nbs, kb, slice_u);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    {   // one line per distinct select variant actually launched (job guards check the opt-in kernels really ran)
        static std::atomic<uint32_t> seen{0};
        const uint32_t bit = 1u << ((wide ? 1 : 0) | (lsa_sel_fast_enabled() ? 2 : 0) | (sel_inline ? 4 : 0));
        if (!(seen.fetch_or(bit) & bit)) {
            GGML_LOG_WARN("LSA select kernel: threads %d fast %d inline-summary %d (units %d, rows %d, shared %d)\n",
                wide ? 1024 : LSA_SEL_THREADS, lsa_sel_fast_enabled() ? 1 : 0, sel_inline ? 1 : 0, nbs, T, shared);
        }
    }
    if (wide) {
        k_lsa_block_select<1024><<<T, 1024, smem, stream>>>(S, cp, qp, out, nbs, T, B, win, snk, kb, shared, gp, gld, gdiv,
            gk, rl, nullptr, nullptr, s_ld, nullptr, sd, stau, smod, lsa_sel_fast_enabled());
    } else {
        k_lsa_block_select<LSA_SEL_THREADS><<<T, LSA_SEL_THREADS, smem, stream>>>(S, cp, qp, out, nbs, T, B, win, snk, kb,
            shared, gp, gld, gdiv, gk, rl, sel_inline ? smx : nullptr, sel_inline ? smn : nullptr, s_ld,
            sel_inline ? (const float *) a->data : nullptr, sd, stau, smod, lsa_sel_fast_enabled());
    }
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------------------------------------------------
// GGML_OP_LSA_UNIT_LSE (opt-in LLAMA_LSA_UNIT_OP): one thread per cell scores its index key against up to 8 query rows
// at a time (keys read once for all rows, no [n_kv, T] score tensor), then the unit logsumexp over F consecutive cells
// (warp shuffles for F <= 32, shared memory above). Units never straddle a CTA: F divides 128.

#define LSA_ULSE_CELLS 128
#define LSA_ULSE_ROWS  8

template <typename mask_t, bool vec>
static __global__ void k_lsa_unit_lse(
        const char * __restrict__ K, const size_t knb1, const bool k_f16, const float * __restrict__ Q,
        const char * __restrict__ M, const size_t mnb1, float * __restrict__ out,
        const int d, const int n_kv, const int T, const int F, const float tau) {
    extern __shared__ float sq[]; // [LSA_ULSE_ROWS][d]
    __shared__ float sx[LSA_ULSE_CELLS];
    const int tid = threadIdx.x;
    const int c   = blockIdx.x*LSA_ULSE_CELLS + tid;
    const int nu  = n_kv/F;
    const bool live = c < n_kv;
    for (int t0 = 0; t0 < T; t0 += LSA_ULSE_ROWS) {
        const int nt = min(LSA_ULSE_ROWS, T - t0);
        for (int i = tid; i < LSA_ULSE_ROWS*d; i += LSA_ULSE_CELLS) {
            const int j = i / d;
            sq[i] = j < nt ? Q[(size_t) (t0 + j)*d + (i - j*d)] : 0.0f;
        }
        __syncthreads();
        float acc[LSA_ULSE_ROWS];
#pragma unroll
        for (int j = 0; j < LSA_ULSE_ROWS; ++j) {
            acc[j] = 0.0f;
        }
        if (live) {
            const char * row = K + (size_t) c*knb1;
            if (vec) { // F16 keys, d % 8 == 0, 16-byte aligned rows: 8 halves per load
                for (int i = 0; i < d; i += 8) {
                    const uint4 w = *(const uint4 *) (row + (size_t) i*sizeof(half));
                    const half2 * h2 = (const half2 *) &w;
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const float2 kf = __half22float2(h2[e]);
#pragma unroll
                        for (int j = 0; j < LSA_ULSE_ROWS; ++j) {
                            acc[j] += kf.x*sq[j*d + i + 2*e];
                            acc[j] += kf.y*sq[j*d + i + 2*e + 1];
                        }
                    }
                }
            } else {
                for (int i = 0; i < d; ++i) {
                    const float kv = k_f16 ? __half2float(((const half *) row)[i]) : ((const float *) row)[i];
#pragma unroll
                    for (int j = 0; j < LSA_ULSE_ROWS; ++j) {
                        acc[j] += kv*sq[j*d + i];
                    }
                }
            }
        }
#pragma unroll
        for (int j = 0; j < LSA_ULSE_ROWS; ++j) {
            if (j >= nt) {
                break;
            }
            const int t = t0 + j;
            float x = -INFINITY;
            if (live) {
                x = tau*acc[j] + (M ? (float) ((const mask_t *) (M + (size_t) t*mnb1))[c] : 0.0f);
            }
            if (F <= 32) {
                float mx = x;
                for (int off = F >> 1; off > 0; off >>= 1) {
                    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
                }
                float sum = mx == -INFINITY ? 0.0f : expf(x - mx);
                for (int off = F >> 1; off > 0; off >>= 1) {
                    sum += __shfl_xor_sync(0xffffffffu, sum, off);
                }
                if (live && (tid & (F - 1)) == 0) {
                    out[(size_t) t*nu + c/F] = mx == -INFINITY ? -INFINITY : mx + logf(sum);
                }
            } else {
                sx[tid] = x;
                __syncthreads();
                const int u0 = blockIdx.x*(LSA_ULSE_CELLS/F) + tid;
                if (tid < LSA_ULSE_CELLS/F && u0 < nu) {
                    const float * xs = sx + tid*F;
                    float mx = -INFINITY;
                    for (int e = 0; e < F; ++e) {
                        mx = fmaxf(mx, xs[e]);
                    }
                    float sum = 0.0f;
                    if (mx != -INFINITY) {
                        for (int e = 0; e < F; ++e) {
                            sum += expf(xs[e] - mx);
                        }
                    }
                    out[(size_t) t*nu + u0] = mx == -INFINITY ? -INFINITY : mx + logf(sum);
                }
                __syncthreads();
            }
        }
        __syncthreads(); // sq is reloaded for the next rows
    }
}

// GGML_LSA_ULSE_COOP=1 (opt-in, same math up to f32 summation order): L = d/8 lanes share one cell's key (16-byte
// loads, a warp reads 32/L whole rows per pass, coalesced), each lane keeps its 8-dim slice of all <= 8 query rows in
// registers (no shared-memory q reads). Partial dots are reduced with xor shuffles down to 8 lanes, then a 3-stage
// reduce-scatter leaves row (lane & 7)'s full dot in that lane (7 shuffles for 8 rows instead of 24). The scaled
// scores go to shared memory and the unit logsumexp is the same code as k_lsa_unit_lse (one thread per cell).
template <typename mask_t, int L>
static __global__ void k_lsa_unit_lse_coop(
        const char * __restrict__ K, const size_t knb1, const float * __restrict__ Q,
        const char * __restrict__ M, const size_t mnb1, float * __restrict__ out,
        const int n_kv, const int T, const int F, const float tau) {
    constexpr int d   = 8*L;
    constexpr int CPP = 32/L;                    // cells per warp pass
    __shared__ float sd[LSA_ULSE_ROWS][LSA_ULSE_CELLS];
    __shared__ float sx[LSA_ULSE_CELLS];
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int sub  = lane & (L - 1);             // dim slice [8*sub, 8*sub + 8)
    const int c0   = blockIdx.x*LSA_ULSE_CELLS;
    const int c    = c0 + tid;
    const int nu   = n_kv/F;
    const bool live = c < n_kv;
    for (int t0 = 0; t0 < T; t0 += LSA_ULSE_ROWS) {
        const int nt = min(LSA_ULSE_ROWS, T - t0);
        float q[LSA_ULSE_ROWS][8];
#pragma unroll
        for (int j = 0; j < LSA_ULSE_ROWS; ++j) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                q[j][e] = j < nt ? Q[(size_t) (t0 + j)*d + 8*sub + e] : 0.0f;
            }
        }
        for (int p = 0; p < 32/CPP; ++p) {
            const int cl = warp*32 + p*CPP + lane/L; // cell within the CTA
            const int cg = c0 + cl;
            float acc[LSA_ULSE_ROWS];
#pragma unroll
            for (int j = 0; j < LSA_ULSE_ROWS; ++j) {
                acc[j] = 0.0f;
            }
            if (cg < n_kv) {
                const uint4 w = *(const uint4 *) (K + (size_t) cg*knb1 + (size_t) sub*16);
                const half2 * h2 = (const half2 *) &w;
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const float2 kf = __half22float2(h2[e]);
#pragma unroll
                    for (int j = 0; j < LSA_ULSE_ROWS; ++j) {
                        acc[j] += kf.x*q[j][2*e];
                        acc[j] += kf.y*q[j][2*e + 1];
                    }
                }
            }
#pragma unroll
            for (int off = L >> 1; off >= 8; off >>= 1) {
#pragma unroll
                for (int j = 0; j < LSA_ULSE_ROWS; ++j) {
                    acc[j] += __shfl_xor_sync(0xffffffffu, acc[j], off);
                }
            }
            // reduce-scatter over the low 3 lane bits: lane keeps the half of the rows selected by its bit
            {
                const bool b = lane & 4;
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const float snd = b ? acc[k] : acc[k + 4];
                    const float kp  = b ? acc[k + 4] : acc[k];
                    acc[k] = kp + __shfl_xor_sync(0xffffffffu, snd, 4);
                }
            }
            {
                const bool b = lane & 2;
#pragma unroll
                for (int k = 0; k < 2; ++k) {
                    const float snd = b ? acc[k] : acc[k + 2];
                    const float kp  = b ? acc[k + 2] : acc[k];
                    acc[k] = kp + __shfl_xor_sync(0xffffffffu, snd, 2);
                }
            }
            {
                const bool b = lane & 1;
                const float snd = b ? acc[0] : acc[1];
                const float kp  = b ? acc[1] : acc[0];
                acc[0] = kp + __shfl_xor_sync(0xffffffffu, snd, 1);
            }
            const int r = lane & 7;
            if (sub < 8 && r < nt) {
                sd[r][cl] = acc[0];
            }
        }
        __syncthreads();
        for (int j = 0; j < nt; ++j) {
            const int t = t0 + j;
            float x = -INFINITY;
            if (live) {
                x = tau*sd[j][tid] + (M ? (float) ((const mask_t *) (M + (size_t) t*mnb1))[c] : 0.0f);
            }
            if (F <= 32) {
                float mx = x;
                for (int off = F >> 1; off > 0; off >>= 1) {
                    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
                }
                float sum = mx == -INFINITY ? 0.0f : expf(x - mx);
                for (int off = F >> 1; off > 0; off >>= 1) {
                    sum += __shfl_xor_sync(0xffffffffu, sum, off);
                }
                if (live && (tid & (F - 1)) == 0) {
                    out[(size_t) t*nu + c/F] = mx == -INFINITY ? -INFINITY : mx + logf(sum);
                }
            } else {
                sx[tid] = x;
                __syncthreads();
                const int u0 = blockIdx.x*(LSA_ULSE_CELLS/F) + tid;
                if (tid < LSA_ULSE_CELLS/F && u0 < nu) {
                    const float * xs = sx + tid*F;
                    float mx = -INFINITY;
                    for (int e = 0; e < F; ++e) {
                        mx = fmaxf(mx, xs[e]);
                    }
                    float sum = 0.0f;
                    if (mx != -INFINITY) {
                        for (int e = 0; e < F; ++e) {
                            sum += expf(xs[e] - mx);
                        }
                    }
                    out[(size_t) t*nu + u0] = mx == -INFINITY ? -INFINITY : mx + logf(sum);
                }
                __syncthreads();
            }
        }
        __syncthreads(); // sd is rewritten for the next rows
    }
}

static bool lsa_ulse_coop_enabled() {
    static const bool on = [] {
        const char * e = getenv("GGML_LSA_ULSE_COOP");
        return e != nullptr && atoi(e) != 0;
    }();
    return on;
}

bool ggml_cuda_lsa_unit_lse_supported(const ggml_tensor * op) {
    const ggml_tensor * k = op->src[0];
    const ggml_tensor * q = op->src[1];
    const ggml_tensor * m = op->src[2];
    const int F = ggml_get_op_params_i32(op, 0);
    return (k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_F32) && k->nb[0] == ggml_type_size(k->type) &&
        q->type == GGML_TYPE_F32 && ggml_is_contiguous(q) && op->type == GGML_TYPE_F32 && ggml_is_contiguous(op) &&
        (m == nullptr || ((m->type == GGML_TYPE_F16 || m->type == GGML_TYPE_F32) && m->nb[0] == ggml_type_size(m->type))) &&
        k->ne[0] >= 1 && k->ne[0] <= 1024 && k->ne[1] <= INT32_MAX - LSA_ULSE_CELLS && q->ne[1] <= 65535 &&
        F >= 1 && F <= LSA_ULSE_CELLS && (F & (F - 1)) == 0 && k->ne[1] % F == 0;
}

void ggml_cuda_lsa_unit_lse(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_lsa_unit_lse_supported(dst));
    const ggml_tensor * k = dst->src[0];
    const ggml_tensor * q = dst->src[1];
    const ggml_tensor * m = dst->src[2];
    const int   d    = (int) k->ne[0];
    const int   n_kv = (int) k->ne[1];
    const int   T    = (int) q->ne[1];
    const int   F    = ggml_get_op_params_i32(dst, 0);
    const float tau  = ggml_get_op_params_f32(dst, 1);
    const bool  f16  = k->type == GGML_TYPE_F16;
    const bool  vec  = f16 && d % 8 == 0 && k->nb[1] % 16 == 0 && ((uintptr_t) k->data) % 16 == 0;
    const dim3  grid((n_kv + LSA_ULSE_CELLS - 1)/LSA_ULSE_CELLS);
    const size_t smem = (size_t) LSA_ULSE_ROWS*d*sizeof(float);
    cudaStream_t stream = ctx.stream();
    const char * md = m ? (const char *) m->data : nullptr;
    const size_t mnb = m ? m->nb[1] : 0;
    {   // one line per unit-lse kernel actually launched (lane-shared or per-cell)
        static std::atomic<uint32_t> seen{0};
        const bool coop = vec && (d == 64 || d == 128 || d == 256) && lsa_ulse_coop_enabled();
        const uint32_t bit = coop ? 2u : 1u;
        if (!(seen.fetch_or(bit) & bit)) {
            GGML_LOG_WARN("LSA unit-lse kernel: %s (d %d, units of %d, rows %d)\n", coop ? "lane-shared" : "per-cell", (int) d, F, T);
        }
    }
    if (vec && (d == 64 || d == 128 || d == 256) && lsa_ulse_coop_enabled()) {
        const bool mh = m && m->type == GGML_TYPE_F16;
#define LSA_ULSE_COOP_LAUNCH(MT, LL) k_lsa_unit_lse_coop<MT, LL><<<grid, LSA_ULSE_CELLS, 0, stream>>>( \
            (const char *) k->data, k->nb[1], (const float *) q->data, md, mnb, (float *) dst->data, n_kv, T, F, tau)
        switch (d) {
            case 64:  if (mh) { LSA_ULSE_COOP_LAUNCH(half, 8);  } else { LSA_ULSE_COOP_LAUNCH(float, 8);  } break;
            case 128: if (mh) { LSA_ULSE_COOP_LAUNCH(half, 16); } else { LSA_ULSE_COOP_LAUNCH(float, 16); } break;
            default:  if (mh) { LSA_ULSE_COOP_LAUNCH(half, 32); } else { LSA_ULSE_COOP_LAUNCH(float, 32); } break;
        }
#undef LSA_ULSE_COOP_LAUNCH
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (m && m->type == GGML_TYPE_F16) {
        if (vec) {
            k_lsa_unit_lse<half, true><<<grid, LSA_ULSE_CELLS, smem, stream>>>((const char *) k->data, k->nb[1], f16,
                (const float *) q->data, md, mnb, (float *) dst->data, d, n_kv, T, F, tau);
        } else {
            k_lsa_unit_lse<half, false><<<grid, LSA_ULSE_CELLS, smem, stream>>>((const char *) k->data, k->nb[1], f16,
                (const float *) q->data, md, mnb, (float *) dst->data, d, n_kv, T, F, tau);
        }
    } else {
        if (vec) {
            k_lsa_unit_lse<float, true><<<grid, LSA_ULSE_CELLS, smem, stream>>>((const char *) k->data, k->nb[1], f16,
                (const float *) q->data, md, mnb, (float *) dst->data, d, n_kv, T, F, tau);
        } else {
            k_lsa_unit_lse<float, false><<<grid, LSA_ULSE_CELLS, smem, stream>>>((const char *) k->data, k->nb[1], f16,
                (const float *) q->data, md, mnb, (float *) dst->data, d, n_kv, T, F, tau);
        }
    }
    CUDA_CHECK(cudaGetLastError());
}
