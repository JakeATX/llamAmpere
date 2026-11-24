// [HEAD2] exact two-stage output head (LLAMA_HEAD2=1|2, default off).
//
// Stage 1 (screen) reads only the scales and the high 6 bits of every Q8_0 head row and gives each logit an interval
// [lo, hi] that provably holds the value the default route computes (bound: head2-ref.h). A per-column threshold tau
// follows from the lower bounds:
//   top-K mode (top_k > 0): tau <= the K-th largest lower bound, so every row with hi < tau is below the K-th largest
//     logit and cannot be in the top K; those rows are written as -inf.
//   mass mode (top_k = 0): tau = max lo - delta (after the softcap), so the rows below it hold less than V e^-delta of
//     the softmax mass; they are written as their upper bound.
// Stage 2 (rescore) computes every row with hi >= tau exactly, in the lane order of the default Q8_0 decode route
// (Q8A16, k_q8a16 in q8a16-kern.cuh): same weights, same products, same fma chain and the same warp butterfly, so the
// kept logits are bit-identical to the route without HEAD2. When every row survives, the head is the full head.
// Widths T > 8 (flag-off takes MMQ there) run in chunks of 8 columns with the same kernels.
//
// mode 2 keeps the 2 low bits (8 of the 34 bytes per block) in pinned host memory; the rescore reads them over PCIe.
// GGML_CUDA_HEAD2_SHADOW=1 keeps a device copy of the original head and checks every call against Q8A16 (test only).
// GGML_CUDA_HEAD2_STATS=1 counts candidates per column (printed at exit).

#include "head2.cuh"
#include "head2-ref.h"
#include "q8a16.cuh"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

#define H2_NBIN 1024

struct h2_counters {
    unsigned long long calls;
    unsigned long long cols;
    unsigned long long cand;     // candidate rows summed over calls (rows rescored)
    unsigned long long compared; // shadow: kept logits compared against the default route
    unsigned long long mism;     // shadow: kept logits that differ in any bit
    unsigned long long viol;     // shadow: dropped rows whose true logit reaches tau
    unsigned long long nrows;    // rows per column summed over columns
};

struct h2_entry {
    const ggml_tensor * t;
    int      device;
    int      mode;
    int      nb;
    int64_t  ne0;
    int64_t  nrows;
    const uint8_t * hi;
    const uint8_t * lo;
    int      top_k;
    float    delta;
    float    softcap;
    float    cslack;   // float error coefficient (x D~)
    float    cl1;      // L1 low-bit coefficient (x D~)
    float  * rn;       // [nrows] device: || d (l - 1.5) ||_2 per row, rounded up
    ggml_backend_buffer_t own_buf;
    void   * lo_host;
    void   * shadow;
};

static std::mutex               g_h2_mutex;
static std::vector<h2_entry *>  g_h2;
bool                            g_ggml_cuda_head2_on = false;
static h2_counters            * g_h2_cnt    = nullptr; // mapped pinned host memory
static bool                     g_h2_shadow = false;
static bool                     g_h2_stats  = false;

static const h2_entry * h2_find(const ggml_tensor * w) {
    if (w == nullptr) {
        return nullptr;
    }
    const ggml_tensor * root = w->view_src ? w->view_src : w;
    for (const h2_entry * e : g_h2) {
        if (e->t == root) {
            return e;
        }
    }
    return nullptr;
}

bool ggml_cuda_head2_touches(const ggml_tensor * node) {
    if (!g_ggml_cuda_head2_on || node == nullptr) {
        return false;
    }
    if (h2_find(node) != nullptr) {
        return true;
    }
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (node->src[i] && h2_find(node->src[i]) != nullptr) {
            return true;
        }
    }
    return false;
}

static float h2_env_f(const char * name, float def) {
    const char * e = getenv(name);
    return e && *e ? (float) atof(e) : def;
}

static void h2_print_stats() {
    const h2_counters * c = g_h2_cnt;
    if (c == nullptr || c->cols == 0) {
        return;
    }
    fprintf(stderr, "head2: %llu calls, %llu columns, %.1f rows rescored per column of %.0f (%.3f%%)",
            c->calls, c->cols, (double) c->cand/c->cols, (double) c->nrows/c->cols, 100.0*c->cand/(double) c->nrows);
    if (g_h2_shadow) {
        fprintf(stderr, "; shadow: %llu kept logits compared, %llu differ, %llu dropped rows reach tau -> %s",
                c->compared, c->mism, c->viol, c->mism == 0 && c->viol == 0 && c->compared > 0 ? "EXACT" : "FAIL");
    }
    fprintf(stderr, "\n");
}

// ---------------------------------------------------------------------------------------------------------------
// kernels

static __device__ __forceinline__ void h2_add(unsigned long long * p, unsigned long long v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600 && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    atomicAdd_system(p, v);
#else
    atomicAdd(p, v);
#endif
}

// per column: sums of x and |x| over each 8-weight group (one GPU lane's share of a block); clears the histogram
// and the column's || x ||_2 (double sum, rounded up). blockDim.x = 256.
static __global__ void k_h2_prep(const float * __restrict__ x, const int64_t sx, const int g8,
        float * __restrict__ s8, float * __restrict__ a8, float * __restrict__ xn, int * __restrict__ hist, unsigned * __restrict__ cnt) {
    const int t = blockIdx.x;
    const float * xt = x + t*sx;
    double q = 0.0;
    for (int g = threadIdx.x; g < g8; g += blockDim.x) {
        float s = 0.0f;
        float a = 0.0f;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const float v = xt[8*g + k];
            s += v;
            a += fabsf(v);
            q += (double) v*v;
        }
        s8[t*g8 + g] = s;
        a8[t*g8 + g] = a;
    }
    __shared__ double sq[256];
    sq[threadIdx.x] = q;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) {
        if ((int) threadIdx.x < o) {
            sq[threadIdx.x] += sq[threadIdx.x + o];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const double n = sqrt(sq[0])*(1.0 + 1e-12);
        xn[t] = isfinite(n) ? __double2float_ru(n) : INFINITY;
    }
    for (int b = threadIdx.x; b < H2_NBIN; b += blockDim.x) {
        hist[t*H2_NBIN + b] = 0;
    }
    if (t == 0 && threadIdx.x == 0) {
        *cnt = 0;
    }
}

template <int T, int RW, int NW>
__launch_bounds__(32*NW)
static __global__ void k_h2_screen(const uint8_t * __restrict__ hi, const int64_t hs, const int nb, const int nr,
        const float * __restrict__ x, const int64_t sx, const float * __restrict__ s8, const float * __restrict__ a8,
        const int g8, const float cslack, const float cl1, const float * __restrict__ rn, const float * __restrict__ xn,
        float * __restrict__ lo_o, float * __restrict__ hi_o, float * __restrict__ bmax) {
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row0 = ((int) blockIdx.x*NW + warp)*RW;
    const int iq   = lane & 3;

    float c[RW][T];
    float a[RW][T];
#pragma unroll
    for (int r = 0; r < RW; ++r) {
#pragma unroll
        for (int t = 0; t < T; ++t) {
            c[r][t] = 0.0f;
            a[r][t] = 0.0f;
        }
    }

    if (row0 < nr) {
        const uint8_t * w[RW];
#pragma unroll
        for (int r = 0; r < RW; ++r) {
            const int row = row0 + r < nr ? row0 + r : nr - 1;
            w[r] = hi + (int64_t) row*hs;
        }
        for (int ib = lane >> 2; ib < nb; ib += 8) {
            float xv[T][8];
            float sv[T];
            float av[T];
#pragma unroll
            for (int t = 0; t < T; ++t) {
                const float * xp = x + t*sx + (int64_t) ib*QK8_0 + iq*8;
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    xv[t][k] = xp[k];
                }
                sv[t] = s8[t*g8 + ib*4 + iq];
                av[t] = a8[t*g8 + ib*4 + iq];
            }
#pragma unroll
            for (int r = 0; r < RW; ++r) {
                const float    d  = __half2float(*(const half *) (w[r] + 2*ib));
                const uint32_t nw = *(const uint32_t *) (w[r] + 2*nb + 16*ib + 4*iq);
                const uint32_t hw = *(const uint16_t *) (w[r] + 18*nb + 8*ib + 2*iq);
                float h[8];
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    const int u = (int) ((((hw >> (2*k)) & 3u) << 4) | ((nw >> (4*k)) & 15u));
                    h[k] = (float) (u - ((u & 32) << 1));
                }
#pragma unroll
                for (int t = 0; t < T; ++t) {
                    float p = h[0]*xv[t][0];
#pragma unroll
                    for (int k = 1; k < 8; ++k) {
                        p = fmaf(h[k], xv[t][k], p);
                    }
                    c[r][t] = fmaf(d, fmaf(4.0f, p, 1.5f*sv[t]), c[r][t]);
                    a[r][t] = fmaf(fabsf(d), av[t], a[r][t]);
                }
            }
        }
    }

    __shared__ float smax[NW][T];
    float mymax[T];
#pragma unroll
    for (int t = 0; t < T; ++t) {
        mymax[t] = -INFINITY;
    }
#pragma unroll
    for (int r = 0; r < RW; ++r) {
        const float rnr = rn[row0 + r < nr ? row0 + r : nr - 1];
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float cs  = warp_reduce_sum<32>(c[r][t]);
            const float as  = warp_reduce_sum<32>(a[r][t]);
            // outward rounding; the 1e-30 covers the flushed subnormals of the fast-math build. fminf: an inf/NaN
            // L2 side falls back to the L1 side (a NaN L1 side gives a non-finite interval below)
            const float lowb = fminf(__fmul_ru(as, cl1), __fmul_ru(rnr, xn[t]));
            const float rho  = __fadd_ru(__fadd_ru(__fmul_ru(as, cslack), lowb), 1e-30f);
            float l = __fsub_rd(cs, rho);
            float h = __fadd_ru(cs, rho);
            if (!(isfinite(l) && isfinite(h))) {
                l = -INFINITY;
                h =  INFINITY;
            }
            if (row0 + r < nr) {
                if (lane == 0) {
                    lo_o[(int64_t) t*nr + row0 + r] = l;
                    hi_o[(int64_t) t*nr + row0 + r] = h;
                }
                mymax[t] = fmaxf(mymax[t], l);
            }
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int t = 0; t < T; ++t) {
            smax[warp][t] = mymax[t];
        }
    }
    __syncthreads();
    if (threadIdx.x < T) {
        float m = -INFINITY;
#pragma unroll
        for (int i = 0; i < NW; ++i) {
            m = fmaxf(m, smax[i][threadIdx.x]);
        }
        bmax[(int64_t) threadIdx.x*gridDim.x + blockIdx.x] = m;
    }
}

// per column: max lower bound over the screen blocks
static __global__ void k_h2_max(const float * __restrict__ bmax, const int nblk, float * __restrict__ M) {
    const int t = blockIdx.x;
    float m = -INFINITY;
    for (int i = threadIdx.x; i < nblk; i += blockDim.x) {
        m = fmaxf(m, bmax[(int64_t) t*nblk + i]);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, off));
    }
    __shared__ float sm[32];
    if (threadIdx.x % 32 == 0) {
        sm[threadIdx.x / 32] = m;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int i = 1; i < (int) blockDim.x / 32; ++i) {
            m = fmaxf(m, sm[i]);
        }
        M[t] = m;
    }
}

// per column: histogram of the lower bounds in [M - W, M], bin 0 at the top
static __global__ void k_h2_hist(const float * __restrict__ lo, const int nr, const float * __restrict__ Mv, const float W,
        int * __restrict__ hist) {
    const int t = blockIdx.y;
    __shared__ int sh[H2_NBIN];
    for (int b = threadIdx.x; b < H2_NBIN; b += blockDim.x) {
        sh[b] = 0;
    }
    __syncthreads();
    const float M = Mv[t];
    if (isfinite(M)) {
        const float scale = (float) H2_NBIN / W;
        for (int r = blockIdx.x*blockDim.x + threadIdx.x; r < nr; r += gridDim.x*blockDim.x) {
            const float v = lo[(int64_t) t*nr + r];
            if (v >= M - W) {
                int b = (int) ((M - v)*scale);
                b = b < 0 ? 0 : b >= H2_NBIN ? H2_NBIN - 1 : b;
                atomicAdd(&sh[b], 1);
            }
        }
    }
    __syncthreads();
    for (int b = threadIdx.x; b < H2_NBIN; b += blockDim.x) {
        if (sh[b]) {
            atomicAdd(&hist[t*H2_NBIN + b], sh[b]);
        }
    }
}

// per column: the threshold (same rules as head2::tau_topk / head2::tau_mass)
static __global__ void k_h2_tau(const int * __restrict__ hist, const float * __restrict__ Mv, const int T, const int nr,
        const int K, const float W, const float delta, const float softcap, float * __restrict__ tau) {
    const int t = threadIdx.x;
    if (t >= T) {
        return;
    }
    const float M = Mv[t];
    float out = -INFINITY;
    if (isfinite(M)) {
        if (K > 0) {
            if (K < nr) {
                int cum = 0;
                for (int b = 0; b < H2_NBIN; ++b) {
                    cum += hist[t*H2_NBIN + b];
                    if (cum >= K) {
                        out = __fsub_rd(M, (float) (b + 2)*(W/(float) H2_NBIN));
                        break;
                    }
                }
            }
        } else if (softcap <= 0.0f) {
            out = __fsub_rd(M, delta + 1e-3f*(1.0f + fabsf(M)));
        } else {
            const double y = (double) softcap*tanh((double) M/softcap) - delta;
            if (y > -0.999999*softcap) {
                const double v = (double) softcap*atanh(y/softcap);
                out = __double2float_rd(v - 1e-3*(1.0 + fabs(v)));
            }
        }
    }
    tau[t] = out;
}

// candidates: any column with hi >= tau; non-candidates get -inf (top-K mode) or their upper bound (mass mode)
template <int T>
static __global__ void k_h2_compact(const float * __restrict__ hib, const float * __restrict__ tau, const int nr, const int kmode,
        float * __restrict__ dst, const int64_t sd, int * __restrict__ list, unsigned * __restrict__ cnt, uint8_t * __restrict__ flag) {
    const int r    = blockIdx.x*blockDim.x + threadIdx.x;
    const int lane = threadIdx.x % 32;
    bool cand = false;
    if (r < nr) {
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float h = hib[(int64_t) t*nr + r];
            cand = cand || !(h < tau[t]);
            dst[t*sd + r] = kmode ? -INFINITY : h;
        }
        if (flag) {
            flag[r] = cand;
        }
    }
    const unsigned mask = __ballot_sync(0xffffffff, cand);
    if (mask) {
        const int leader = __ffs(mask) - 1;
        unsigned base = 0;
        if (lane == leader) {
            base = atomicAdd(cnt, (unsigned) __popc(mask));
        }
        base = __shfl_sync(0xffffffff, base, leader);
        if (cand) {
            list[base + __popc(mask & ((1u << lane) - 1u))] = r;
        }
    }
}

// exact rows, in the lane order of k_q8a16 (q8a16-kern.cuh): lane l takes blocks l/4 + 8i and weights 8*(l%4)..+7,
// p = f0*x0 then 7 fmas, acc = fma(d, p, acc), then warp_reduce_sum. Keep the two in step.
template <int T, int NW>
__launch_bounds__(32*NW)
static __global__ void k_h2_rescore(const uint8_t * __restrict__ hi, const int64_t hs, const uint8_t * __restrict__ lo, const int64_t ls,
        const int nb, const float * __restrict__ x, const int64_t sx, const int * __restrict__ list, const unsigned * __restrict__ cnt,
        float * __restrict__ dst, const int64_t sd) {
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int iq   = lane & 3;
    const int n    = (int) *cnt;
    for (int i = blockIdx.x*NW + warp; i < n; i += gridDim.x*NW) {
        const int r = list[i];
        const uint8_t * wh = hi + (int64_t) r*hs;
        const uint8_t * wl = lo + (int64_t) r*ls;
        float acc[T];
#pragma unroll
        for (int t = 0; t < T; ++t) {
            acc[t] = 0.0f;
        }
        for (int ib = lane >> 2; ib < nb; ib += 8) {
            float xv[T][8];
#pragma unroll
            for (int t = 0; t < T; ++t) {
                const float * xp = x + t*sx + (int64_t) ib*QK8_0 + iq*8;
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    xv[t][k] = xp[k];
                }
            }
            const float    d  = __half2float(*(const half *) (wh + 2*ib));
            const uint32_t nw = *(const uint32_t *) (wh + 2*nb + 16*ib + 4*iq);
            const uint32_t hw = *(const uint16_t *) (wh + 18*nb + 8*ib + 2*iq);
            const uint32_t lw = *(const uint16_t *) (wl + 8*ib + 2*iq);
            float f[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const uint32_t u = (((hw >> (2*k)) & 3u) << 6) | (((nw >> (4*k)) & 15u) << 2) | ((lw >> (2*k)) & 3u);
                f[k] = (float) (int) (int8_t) (uint8_t) u;
            }
#pragma unroll
            for (int t = 0; t < T; ++t) {
                float p = f[0]*xv[t][0];
                p = fmaf(f[1], xv[t][1], p);
                p = fmaf(f[2], xv[t][2], p);
                p = fmaf(f[3], xv[t][3], p);
                p = fmaf(f[4], xv[t][4], p);
                p = fmaf(f[5], xv[t][5], p);
                p = fmaf(f[6], xv[t][6], p);
                p = fmaf(f[7], xv[t][7], p);
                acc[t] = fmaf(d, p, acc[t]);
            }
        }
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float s = warp_reduce_sum<32>(acc[t]);
            if (lane == 0) {
                dst[t*sd + r] = s;
            }
        }
    }
}

static __global__ void k_h2_count(const unsigned * __restrict__ cnt, const int T, const int nr, h2_counters * c) {
    h2_add(&c->calls, 1ull);
    h2_add(&c->cols, (unsigned long long) T);
    h2_add(&c->cand, (unsigned long long) *cnt * T);
    h2_add(&c->nrows, (unsigned long long) nr * T);
}

// shadow: kept rows must match the default route bit for bit, dropped rows must sit below tau
static __global__ void k_h2_check(const float * __restrict__ dst, const int64_t sd, const float * __restrict__ ref, const float * __restrict__ tau,
        const uint8_t * __restrict__ flag, const int nr, const int T, h2_counters * c) {
    __shared__ unsigned s_cmp, s_mis, s_vio;
    if (threadIdx.x == 0) {
        s_cmp = 0; s_mis = 0; s_vio = 0;
    }
    __syncthreads();
    const int r = blockIdx.x*blockDim.x + threadIdx.x;
    if (r < nr) {
        for (int t = 0; t < T; ++t) {
            const float o = dst[t*sd + r];
            const float f = ref[(int64_t) t*nr + r];
            if (flag[r]) {
                atomicAdd(&s_cmp, 1u);
                if (__float_as_uint(o) != __float_as_uint(f)) {
                    atomicAdd(&s_mis, 1u);
                }
            } else if (!(f < tau[t])) {
                atomicAdd(&s_vio, 1u);
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        if (s_cmp) { h2_add(&c->compared, (unsigned long long) s_cmp); }
        if (s_mis) { h2_add(&c->mism,     (unsigned long long) s_mis); }
        if (s_vio) { h2_add(&c->viol,     (unsigned long long) s_vio); }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// host

template <int T>
static void h2_run(ggml_backend_cuda_context & ctx, const h2_entry & e, const ggml_tensor * src0, const uint8_t * hi, const uint8_t * lo,
        const int nr, const float * x, const int64_t sx, float * dst, const int64_t sd, const int64_t r0) {
    cudaStream_t stream = ctx.stream();
    const int     nb  = e.nb;
    const int     g8  = (int) (e.ne0 / 8);
    const int64_t hs  = head2::hi_row_bytes(nb);
    const int64_t ls  = head2::lo_row_bytes(nb);
    constexpr int RW  = T <= 2 ? 2 : 1;
    constexpr int NW  = 4;
    const int     nblk = (nr + RW*NW - 1) / (RW*NW);
    static const float W = h2_env_f("GGML_CUDA_HEAD2_WINDOW", 32.0f);

    ggml_cuda_pool_alloc<float>    s8  (ctx.pool(), (size_t) T*g8);
    ggml_cuda_pool_alloc<float>    a8  (ctx.pool(), (size_t) T*g8);
    ggml_cuda_pool_alloc<float>    lob (ctx.pool(), (size_t) T*nr);
    ggml_cuda_pool_alloc<float>    hib (ctx.pool(), (size_t) T*nr);
    ggml_cuda_pool_alloc<float>    bmax(ctx.pool(), (size_t) T*nblk);
    ggml_cuda_pool_alloc<float>    Mv  (ctx.pool(), T);
    ggml_cuda_pool_alloc<float>    xn  (ctx.pool(), T);
    ggml_cuda_pool_alloc<float>    tau (ctx.pool(), T);
    ggml_cuda_pool_alloc<int>      hist(ctx.pool(), (size_t) T*H2_NBIN);
    ggml_cuda_pool_alloc<int>      list(ctx.pool(), (size_t) nr);
    ggml_cuda_pool_alloc<unsigned> cnt (ctx.pool(), 1);
    ggml_cuda_pool_alloc<uint8_t>  flag;
    if (g_h2_shadow) {
        flag.alloc(ctx.pool(), (size_t) nr);
    }

    k_h2_prep<<<T, 256, 0, stream>>>(x, sx, g8, s8.get(), a8.get(), xn.get(), hist.get(), cnt.get());
    k_h2_screen<T, RW, NW><<<nblk, 32*NW, 0, stream>>>(hi, hs, nb, nr, x, sx, s8.get(), a8.get(), g8, e.cslack, e.cl1, e.rn + r0, xn.get(),
            lob.get(), hib.get(), bmax.get());
    k_h2_max<<<T, 1024, 0, stream>>>(bmax.get(), nblk, Mv.get());
    if (e.top_k > 0) {
        const int nbh = std::min(4*ggml_cuda_info().devices[ctx.device].nsm, (nr + 255) / 256);
        k_h2_hist<<<dim3(nbh, T), 256, 0, stream>>>(lob.get(), nr, Mv.get(), W, hist.get());
    }
    k_h2_tau<<<1, 32, 0, stream>>>(hist.get(), Mv.get(), T, nr, e.top_k, W, e.delta, e.softcap, tau.get());
    k_h2_compact<T><<<(nr + 255) / 256, 256, 0, stream>>>(hib.get(), tau.get(), nr, e.top_k > 0, dst, sd, list.get(), cnt.get(),
            g_h2_shadow ? flag.get() : nullptr);
    constexpr int NWR = 4;
    const int grid = 8*ggml_cuda_info().devices[ctx.device].nsm;
    k_h2_rescore<T, NWR><<<grid, 32*NWR, 0, stream>>>(hi, hs, lo, ls, nb, x, sx, list.get(), cnt.get(), dst, sd);
    CUDA_CHECK(cudaGetLastError());

    if (g_h2_cnt) {
        k_h2_count<<<1, 1, 0, stream>>>(cnt.get(), T, nr, g_h2_cnt);
    }
    if (g_h2_shadow && e.shadow) {
        // the default route on the original bytes: Q8A16, the route the head takes without HEAD2 at T <= 8
        ggml_cuda_pool_alloc<float> ref(ctx.pool(), (size_t) T*nr);
        ggml_tensor w = *src0;
        w.view_src  = nullptr;
        w.view_offs = 0;
        w.data      = (char *) e.shadow + r0*(int64_t) nb*sizeof(block_q8_0);
        w.ne[1]     = nr;
        w.nb[1]     = (size_t) nb*sizeof(block_q8_0);
        w.nb[2]     = w.nb[1]*nr;
        w.nb[3]     = w.nb[2];
        ggml_tensor xs = {};
        xs.type  = GGML_TYPE_F32;
        xs.ne[0] = e.ne0; xs.ne[1] = T; xs.ne[2] = 1; xs.ne[3] = 1;
        xs.nb[0] = sizeof(float); xs.nb[1] = sx*sizeof(float); xs.nb[2] = xs.nb[1]*T; xs.nb[3] = xs.nb[2];
        xs.data  = const_cast<float *>(x);
        ggml_tensor ds = {};
        ds.type  = GGML_TYPE_F32;
        ds.ne[0] = nr; ds.ne[1] = T; ds.ne[2] = 1; ds.ne[3] = 1;
        ds.nb[0] = sizeof(float); ds.nb[1] = (size_t) nr*sizeof(float); ds.nb[2] = ds.nb[1]*T; ds.nb[3] = ds.nb[2];
        ds.data  = ref.get();
        if (ggml_cuda_q8a16_mul_mat(ctx, &w, &xs, &ds)) {
            k_h2_check<<<(nr + 255) / 256, 256, 0, stream>>>(dst, sd, ref.get(), tau.get(), flag.get(), nr, T, g_h2_cnt);
        } else {
            static bool warned = false;
            if (!warned) {
                warned = true;
                GGML_LOG_WARN("head2: shadow check skipped: the Q8A16 route refused this call\n");
            }
        }
        cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(stream, &st));
        if (st == cudaStreamCaptureStatusNone) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            static unsigned long long last_bad = 0;
            const unsigned long long bad = g_h2_cnt->mism + g_h2_cnt->viol;
            if (bad != last_bad || g_h2_cnt->calls % 256 == 0) {
                last_bad = bad;
                h2_print_stats();
            }
        }
    }
}

bool ggml_cuda_head2_forward(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const h2_entry * e = dst->op == GGML_OP_MUL_MAT ? h2_find(dst->src[0]) : nullptr;
    if (e == nullptr) {
        if (ggml_cuda_head2_touches(dst)) {
            GGML_ABORT("head2: %s (%s) reads the relaid-out output head; only MUL_MAT is supported (unset LLAMA_HEAD2)",
                    dst->name, ggml_op_name(dst->op));
        }
        return false;
    }
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const int64_t row_bytes = (int64_t) e->nb*sizeof(block_q8_0);
    const int64_t offs      = src0->view_src ? (int64_t) src0->view_offs : 0;
    if (ctx.device != e->device || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src0->ne[0] != e->ne0 ||
            src0->nb[1] != (size_t) row_bytes || offs % row_bytes != 0 || offs/row_bytes + src0->ne[1] > e->nrows ||
            src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[0] != e->ne0 || src1->ne[2] != 1 || src1->ne[3] != 1 ||
            src1->nb[0] != sizeof(float) || src1->nb[1] % sizeof(float) != 0 || dst->nb[0] != sizeof(float) ||
            dst->nb[1] % sizeof(float) != 0 || dst->ne[0] != src0->ne[1] || dst->ne[1] != src1->ne[1] ||
            dst->ne[2] != 1 || dst->ne[3] != 1 || src0->ne[1] > INT32_MAX) {
        GGML_ABORT("head2: %s: unsupported head product (device %d/%d, src1 %s [%" PRId64 ", %" PRId64 "])", dst->name,
                ctx.device, e->device, ggml_type_name(src1->type), src1->ne[0], src1->ne[1]);
    }
    const int64_t r0 = offs/row_bytes;
    const int     nr = (int) src0->ne[1];
    const uint8_t * hi = e->hi + r0*head2::hi_row_bytes(e->nb);
    const uint8_t * lo = e->lo + r0*head2::lo_row_bytes(e->nb);
    const int64_t sx = src1->nb[1]/sizeof(float);
    const int64_t sd = dst->nb[1]/sizeof(float);
    const float * x  = (const float *) src1->data;
    float       * d  = (float *) dst->data;
    for (int64_t c0 = 0; c0 < src1->ne[1]; c0 += 8) {
        const int T = (int) std::min<int64_t>(8, src1->ne[1] - c0);
        const float * xc = x + c0*sx;
        float       * dc = d + c0*sd;
        switch (T) {
            case 1: h2_run<1>(ctx, *e, src0, hi, lo, nr, xc, sx, dc, sd, r0); break;
            case 2: h2_run<2>(ctx, *e, src0, hi, lo, nr, xc, sx, dc, sd, r0); break;
            case 3: h2_run<3>(ctx, *e, src0, hi, lo, nr, xc, sx, dc, sd, r0); break;
            case 4: h2_run<4>(ctx, *e, src0, hi, lo, nr, xc, sx, dc, sd, r0); break;
            case 5: h2_run<5>(ctx, *e, src0, hi, lo, nr, xc, sx, dc, sd, r0); break;
            case 6: h2_run<6>(ctx, *e, src0, hi, lo, nr, xc, sx, dc, sd, r0); break;
            case 7: h2_run<7>(ctx, *e, src0, hi, lo, nr, xc, sx, dc, sd, r0); break;
            case 8: h2_run<8>(ctx, *e, src0, hi, lo, nr, xc, sx, dc, sd, r0); break;
            default: GGML_ABORT("head2: width");
        }
    }
    return true;
}

bool ggml_cuda_head2_prepare(ggml_tensor * t, int mode, int device, int top_k, float delta, float softcap) {
    std::lock_guard<std::mutex> lock(g_h2_mutex);
    if (t == nullptr || t->buffer == nullptr || t->data == nullptr || (mode != 1 && mode != 2)) {
        return false;
    }
    const int64_t ne0 = t->ne[0];
    const int     nb  = (int) (ne0 / QK8_0);
    if (t->type != GGML_TYPE_Q8_0 || t->view_src || ne0 % QK8_0 != 0 || nb % 2 != 0 || t->ne[2] != 1 || t->ne[3] != 1 ||
            t->nb[1] != (size_t) nb*sizeof(block_q8_0) || t->ne[1] > INT32_MAX) {
        GGML_LOG_WARN("head2: %s: %s [%" PRId64 ", %" PRId64 "] is not a plain Q8_0 head with an even block count; head2 off\n",
                __func__, ggml_type_name(t->type), t->ne[0], t->ne[1]);
        return false;
    }
    for (const h2_entry * e : g_h2) {
        if (e->t == t) {
            return false;
        }
    }
    ggml_cuda_set_device(device);

    const int64_t nrows   = t->ne[1];
    const int64_t hs      = head2::hi_row_bytes(nb);
    const int64_t ls      = head2::lo_row_bytes(nb);
    const size_t  nbytes  = (size_t) nrows*nb*sizeof(block_q8_0);
    const bool    on_host = ggml_backend_buffer_is_host(t->buffer);

    std::vector<uint8_t> orig;
    const uint8_t * src;
    if (on_host) {
        src = (const uint8_t *) t->data;
    } else {
        orig.resize(nbytes);
        CUDA_CHECK(cudaMemcpy(orig.data(), t->data, nbytes, cudaMemcpyDeviceToHost));
        src = orig.data();
    }

    auto * e = new h2_entry{};
    e->t       = t;
    e->device  = device;
    e->mode    = mode;
    e->nb      = nb;
    e->ne0     = ne0;
    e->nrows   = nrows;
    e->top_k   = top_k;
    e->delta   = delta;
    e->softcap = softcap;
    e->cslack  = head2::coef_slack(nb);
    e->cl1     = head2::coef_l1(nb);

    g_h2_shadow = getenv("GGML_CUDA_HEAD2_SHADOW") && atoi(getenv("GGML_CUDA_HEAD2_SHADOW")) != 0;
    g_h2_stats  = g_h2_shadow || (getenv("GGML_CUDA_HEAD2_STATS") && atoi(getenv("GGML_CUDA_HEAD2_STATS")) != 0);
    if (g_h2_shadow) {
        CUDA_CHECK(cudaMalloc(&e->shadow, nbytes));
        CUDA_CHECK(cudaMemcpy(e->shadow, src, nbytes, cudaMemcpyHostToDevice));
    }

    std::vector<uint8_t> hi((size_t) nrows*hs);
    std::vector<uint8_t> lo((size_t) nrows*ls);
    std::vector<float>   rn((size_t) nrows);
    for (int64_t r = 0; r < nrows; ++r) {
        head2::split_row_q8_0(src + r*nb*sizeof(block_q8_0), nb, hi.data() + r*hs, lo.data() + r*ls);
        rn[r] = head2::row_lo_norm_q8_0(src + r*nb*sizeof(block_q8_0), nb);
    }
    CUDA_CHECK(cudaMalloc(&e->rn, rn.size()*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(e->rn, rn.data(), rn.size()*sizeof(float), cudaMemcpyHostToDevice));

    size_t freed = 0;
    if (mode == 2) {
        CUDA_CHECK(cudaHostAlloc(&e->lo_host, lo.size(), cudaHostAllocMapped | cudaHostAllocPortable));
        memcpy(e->lo_host, lo.data(), lo.size());
        void * lo_dev = nullptr;
        CUDA_CHECK(cudaHostGetDevicePointer(&lo_dev, e->lo_host, 0));
        e->lo = (const uint8_t *) lo_dev;
    }
    if (on_host) {
        // the head was loaded into host memory (-ot output.weight=CPU): its hi planes move into a device buffer
        const size_t dev_bytes = mode == 2 ? hi.size() : nbytes;
        e->own_buf = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_buffer_type(device), dev_bytes);
        if (e->own_buf == nullptr) {
            GGML_LOG_ERROR("head2: %s: cannot allocate %zu device bytes for the head planes\n", __func__, dev_bytes);
            if (e->lo_host) { cudaFreeHost(e->lo_host); }
            if (e->shadow)  { cudaFree(e->shadow); }
            delete e;
            return false;
        }
        uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(e->own_buf);
        CUDA_CHECK(cudaMemcpy(base, hi.data(), hi.size(), cudaMemcpyHostToDevice));
        if (mode == 1) {
            CUDA_CHECK(cudaMemcpy(base + hi.size(), lo.data(), lo.size(), cudaMemcpyHostToDevice));
            e->lo = base + hi.size();
        }
        e->hi     = base;
        t->buffer = e->own_buf;
        t->data   = base;
        freed     = mode == 2 ? nbytes - hi.size() : 0;
    } else {
        uint8_t * base = (uint8_t *) t->data;
        CUDA_CHECK(cudaMemcpy(base, hi.data(), hi.size(), cudaMemcpyHostToDevice));
        if (mode == 1) {
            CUDA_CHECK(cudaMemcpy(base + hi.size(), lo.data(), lo.size(), cudaMemcpyHostToDevice));
            e->lo = base + hi.size();
        } else {
            GGML_LOG_WARN("head2: mode 2 on a device-resident head keeps its VRAM; load it with -ot output.weight=CPU to free %zu MiB\n",
                    (nbytes - hi.size()) >> 20);
        }
        e->hi = base;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    if (g_h2_stats && g_h2_cnt == nullptr) {
        CUDA_CHECK(cudaHostAlloc((void **) &g_h2_cnt, sizeof(h2_counters), cudaHostAllocMapped | cudaHostAllocPortable));
        memset(g_h2_cnt, 0, sizeof(h2_counters));
        h2_counters * dev = nullptr;
        CUDA_CHECK(cudaHostGetDevicePointer((void **) &dev, g_h2_cnt, 0));
        GGML_ASSERT(dev == g_h2_cnt && "head2: stats need unified addressing");
        atexit(h2_print_stats);
    }

    g_h2.push_back(e);
    g_ggml_cuda_head2_on = true;
    GGML_LOG_INFO("head2: %s [%" PRId64 ", %" PRId64 "] Q8_0 on CUDA%d, mode %d (%s), %s, screen reads %.1f%% of the head%s%s\n",
            t->name, ne0, nrows, device, mode, mode == 1 ? "planes in VRAM" : "low planes in pinned host memory",
            top_k > 0 ? (std::string("top-") + std::to_string(top_k) + " exact").c_str() :
                        (std::string("mass mode, delta ") + std::to_string(delta)).c_str(),
            100.0*hs/(hs + ls), freed ? (", frees " + std::to_string(freed >> 20) + " MiB VRAM").c_str() : "",
            g_h2_shadow ? ", SHADOW CHECK ON" : "");
    return true;
}
