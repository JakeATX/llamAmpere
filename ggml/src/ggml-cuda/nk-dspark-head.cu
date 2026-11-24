// NEONKERN: sm_86 bandwidth-saturated DSpark folded Markov head + exact 8-bit radix top-K / Gumbel-max.
//
// Motivation (from dec.sqlite on RTX 3090 300 W, Inkling 2 Mini 100K DSpark graph):
// - Base dspark-head runs k_dspark_fold_head_tile_topk (179.5 us x 4 = 718 us/r) + k_dspark_head_merge_sample
//   (50.4 us x 4 = 201.5 us/r) = 919.5 us/r (29.7% of DSpark), achieving only 453 GB/s on 104.0 MB/call.
// - Stage 1 loses bandwidth to a 1,024-element 55-barrier bitonic sort per CTA, 1-row serial LDG->shfl->tanhf loops,
//   stride-8 scalar fold_bias/base loads, no L2::evict_first on 102.4 MB fold_w, and a 196/82 = 2.39-wave tail.
// - Stage 2 spends 50.4 us/call running a 4,096-element 78-barrier bitonic sort on a single CTA.
//
// Design (100% bit-identical numerics to base dspark-head.cu):
// - Stage 1 (k_nk_dspark_head_scan): 4x SM exact-wave grid (328 CTAs on 82-SM 3090, 336 on 84-SM 3090 Ti),
//   256 threads/CTA, 4-row unrolled streaming LDG.128 with L1::no_allocate + L2::evict_first on fold_w,
//   coalesced 4-lane parallel fold_bias/base loads and tanhf softcap, exact same FMA + butterfly order as base,
//   followed by an exact 8-bit MSB-first radix top-K selector (0 full sort, typically 1-2 passes in ~0.3 us).
// - Stage 2 (k_nk_dspark_head_merge_sample): single 512-thread CTA running the same 8-bit radix top-K selector
//   directly over L2-hot Stage-1 candidates (~2.6 KB SMEM, zero 4096-element bitonic sort) plus warp-0 64-element
//   final sort and exact Gumbel-max argmax.
// - Switch: GGML_CUDA_NK_DSPARK_HEAD=1 (default OFF; unset = exact base).

#include "nk-dspark-head.cuh"

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdint>
#include <cstdlib>

bool ggml_cuda_nk_dspark_head_enabled() {
    static const bool on = [] {
        const char * e = std::getenv("GGML_CUDA_NK_DSPARK_HEAD");
        return e != nullptr && std::atoi(e) != 0;
    }();
    return on;
}

// Returns: -1 = auto (L2::evict_first on nb==1, off on nb>1 for cross-seq L2 reuse), 0 = force off, 1 = force on.
static int nk_dspark_head_l2_mode() {
    static const int mode = [] {
        const char * e = std::getenv("GGML_CUDA_NK_DSPARK_HEAD_L2");
        if (e == nullptr) {
            return -1;
        }
        return std::atoi(e) != 0 ? 1 : 0;
    }();
    return mode;
}

static bool nk_dspark_head_trace() {
    static const bool on = [] {
        const char * e = std::getenv("GGML_CUDA_NK_TRACE");
        return e != nullptr && std::atoi(e) != 0;
    }();
    return on;
}

static std::atomic<int64_t> g_nk_dspark_head_launches{0};

int64_t ggml_cuda_nk_dspark_head_count() {
    return g_nk_dspark_head_launches.load(std::memory_order_relaxed);
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

namespace nk_dspark_head {

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
static __device__ __forceinline__ uint4 nk_ldg128_stream(const void * ptr, const uint64_t policy) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if constexpr (L2_EF) {
        uint4 r;
        asm volatile(
            "ld.global.L1::no_allocate.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;"
            : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
            : "l"(ptr), "l"(policy));
        return r;
    } else {
        return __ldg(reinterpret_cast<const uint4 *>(ptr));
    }
#else
    GGML_UNUSED_VARS(policy);
    return *reinterpret_cast<const uint4 *>(ptr);
#endif
}

// Exact match to dspark-head.cu key encoding and decoding.
static __device__ __forceinline__ uint64_t nk_topk_key(const float v, const int col, const int ncols) {
    uint32_t b = __float_as_uint(v);
    b = (b == 0x80000000u) ? 0u : b;
    b = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
    return ((uint64_t) b << 32) | (uint32_t) (ncols - 1 - col);
}

static __device__ __forceinline__ float nk_key_to_float(const uint64_t key) {
    uint32_t b = (uint32_t) (key >> 32);
    b = (b & 0x80000000u) ? (b & 0x7fffffffu) : ~b;
    return __uint_as_float(b);
}

// Exact match to dspark-head.cu 8-element F16xF32 dot accumulation order.
static __device__ __forceinline__ float nk_dot8(const uint4 w4, const float4 x0, const float4 x1) {
    const half2 * h2 = reinterpret_cast<const half2 *>(&w4);
    const float2  f0 = __half22float2(h2[0]);
    const float2  f1 = __half22float2(h2[1]);
    const float2  f2 = __half22float2(h2[2]);
    const float2  f3 = __half22float2(h2[3]);
    return f0.x * x0.x + f0.y * x0.y
         + f1.x * x0.z + f1.y * x0.w
         + f2.x * x1.x + f2.y * x1.y
         + f3.x * x1.z + f3.y * x1.w;
}

// Pure-register 64-bit warp butterfly shuffle.
static __device__ __forceinline__ uint64_t nk_shfl_xor_u64(const uint64_t v, const int mask) {
    const uint32_t lo = __shfl_xor_sync(0xffffffffu, (uint32_t) v, mask);
    const uint32_t hi = __shfl_xor_sync(0xffffffffu, (uint32_t) (v >> 32), mask);
    return ((uint64_t) hi << 32) | (uint64_t) lo;
}

// Pure-register bitonic sort of 32 uint64_t across lanes 0..31 (desc=true -> descending, false -> ascending).
static __device__ __forceinline__ uint64_t nk_warp_bitonic_sort32(uint64_t v, const int lane, const bool desc) {
    for (int size = 2; size <= 32; size <<= 1) {
        const bool dir = ((lane & size) == 0) ? desc : !desc;
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            const uint64_t o = nk_shfl_xor_u64(v, stride);
            const bool is_lo = (lane & stride) == 0;
            const bool keep_max = (is_lo == dir);
            v = keep_max ? (v > o ? v : o) : (v < o ? v : o);
        }
    }
    return v;
}

// Pure-register bitonic merge of a 32-element bitonic sequence into descending order across lanes 0..31.
static __device__ __forceinline__ uint64_t nk_warp_bitonic_merge32_desc(uint64_t v, const int lane) {
    for (int stride = 16; stride > 0; stride >>= 1) {
        const uint64_t o = nk_shfl_xor_u64(v, stride);
        const bool is_lo = (lane & stride) == 0;
        v = is_lo ? (v > o ? v : o) : (v < o ? v : o);
    }
    return v;
}

// Warp-0 descending bitonic sort of 64 uint64_t keys using pure-register warp shuffles (0 intermediate SMEM
// traffic, 0 __syncthreads, immune to non-volatile SMEM load hoisting across __syncwarp).
static __device__ __forceinline__ void nk_sort64_desc_warp0(uint64_t * __restrict__ s, const int tid) {
    if (tid < 32) {
        uint64_t a = s[tid];
        uint64_t b = s[tid + 32];
        a = nk_warp_bitonic_sort32(a, tid, false);
        b = nk_warp_bitonic_sort32(b, tid, true);
        if (a < b) {
            const uint64_t tmp = a;
            a = b;
            b = tmp;
        }
        a = nk_warp_bitonic_merge32_desc(a, tid);
        b = nk_warp_bitonic_merge32_desc(b, tid);
        s[tid]      = a;
        s[tid + 32] = b;
    }
}

struct nk_radix_smem {
    uint32_t hist[256];
    uint64_t small[64];
    int      warp_tot[8];
    int      warp_off[8];
    int      bucket;
    int      gt;
    int      cnt_eq;
    int      emit_cnt;
    int      small_cnt;
};

// Exact MSB-first 8-bit radix top-K selector (1 <= k <= 64) over n_in uint64_t keys.
// Uses prefix_mask/prefix_val filtering so no active compaction buffer is needed (works for arbitrary n_in).
// When SORT_OUT is true, s_out[0..k-1] is sorted strictly descending; otherwise it holds the exact top-K set.
template <int BLOCK, bool SORT_OUT>
static __device__ __forceinline__ void nk_radix_topk(
        const uint64_t * __restrict__ keys,
        const int n_in,
        const int k,
        uint64_t * __restrict__ s_out,
        nk_radix_smem & sm,
        const int tid) {
    if (n_in <= k) {
        if (tid < 64) {
            s_out[tid] = (tid < n_in) ? keys[tid] : 0ull;
        }
        __syncthreads();
        if constexpr (SORT_OUT) {
            nk_sort64_desc_warp0(s_out, tid);
            __syncthreads();
        }
        return;
    }

    if (n_in <= 64) {
        if (tid < 64) {
            sm.small[tid] = (tid < n_in) ? keys[tid] : 0ull;
        }
        __syncthreads();
        nk_sort64_desc_warp0(sm.small, tid);
        __syncthreads();
        if (tid < k) {
            s_out[tid] = sm.small[tid];
        }
        __syncthreads();
        return;
    }

    int found = 0;
    int need  = k;
    uint64_t prefix_mask = 0ull;
    uint64_t prefix_val  = 0ull;

    for (int shift = 56; shift >= 0 && need > 0; shift -= 8) {
        if (tid < 256) {
            sm.hist[tid] = 0u;
        }
        __syncthreads();

        for (int i = tid; i < n_in; i += BLOCK) {
            const uint64_t ky = keys[i];
            if ((ky & prefix_mask) == prefix_val) {
                atomicAdd(&sm.hist[(ky >> shift) & 0xFFu], 1u);
            }
        }
        __syncthreads();

        // Descending prefix sum over bins 255..0 across threads 0..255 (all __syncthreads remain CTA-uniform).
        int my_val  = 0;
        int my_wsum = 0;
        if (tid < 256) {
            my_val  = (int) sm.hist[255 - tid];
            my_wsum = my_val;
            const int l = tid & 31;
#pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                const int up = __shfl_up_sync(0xffffffffu, my_wsum, off);
                if (l >= off) {
                    my_wsum += up;
                }
            }
            if (l == 31) {
                sm.warp_tot[tid >> 5] = my_wsum;
            }
        }
        __syncthreads();

        if (tid < 32) {
            const int wt = (tid < 8) ? sm.warp_tot[tid] : 0;
            int inc = wt;
#pragma unroll
            for (int off = 1; off < 8; off <<= 1) {
                const int up = __shfl_up_sync(0xffffffffu, inc, off);
                if (tid >= off) {
                    inc += up;
                }
            }
            if (tid < 8) {
                sm.warp_off[tid] = inc - wt;
            }
        }
        __syncthreads();

        if (tid < 256) {
            const int incl = my_wsum + sm.warp_off[tid >> 5];
            const int excl = incl - my_val;
            if (excl < need && incl >= need) {
                sm.bucket = 255 - tid;
                sm.gt     = excl;
                sm.cnt_eq = my_val;
            }
        }
        if (tid == 0) {
            sm.emit_cnt  = found;
            sm.small_cnt = 0;
        }
        __syncthreads();

        const uint32_t B      = (uint32_t) sm.bucket;
        const int      gt     = sm.gt;
        const int      cnt_eq = sm.cnt_eq;
        const int      rem    = need - gt;

        const bool exact_eq = (cnt_eq == rem);
        const bool small_eq = (!exact_eq && cnt_eq <= 64);
        if (small_eq) {
            if (tid < 64) {
                sm.small[tid] = 0ull;
            }
            __syncthreads();
        }

        if (gt > 0 || exact_eq || small_eq) {
            for (int i = tid; i < n_in; i += BLOCK) {
                const uint64_t ky = keys[i];
                if ((ky & prefix_mask) == prefix_val) {
                    const uint32_t bval = (uint32_t) ((ky >> shift) & 0xFFu);
                    if (bval > B || (exact_eq && bval == B)) {
                        const int pos = atomicAdd(&sm.emit_cnt, 1);
                        s_out[pos] = ky;
                    } else if (small_eq && bval == B) {
                        const int pos = atomicAdd(&sm.small_cnt, 1);
                        sm.small[pos] = ky;
                    }
                }
            }
            __syncthreads();
        }

        if (exact_eq) {
            found = k;
            need  = 0;
            break;
        }
        found += gt;
        need   = rem;

        if (small_eq) {
            nk_sort64_desc_warp0(sm.small, tid);
            __syncthreads();
            if (tid < rem) {
                s_out[found + tid] = sm.small[tid];
            }
            __syncthreads();
            found = k;
            need  = 0;
            break;
        }

        prefix_mask |= (0xFFull << shift);
        prefix_val  |= ((uint64_t) B << shift);
    }

    if (need > 0) {
        if (tid < need) {
            s_out[found + tid] = prefix_val;
        }
        __syncthreads();
    }

    if constexpr (SORT_OUT) {
        if (tid >= k && tid < 64) {
            s_out[tid] = 0ull;
        }
        __syncthreads();
        nk_sort64_desc_warp0(s_out, tid);
        __syncthreads();
    }
}

// Exact 4-aligned partition of [0, n_eval) across ntiles CTAs (disjoint, complete, 16B-aligned bias/base starts).
static __device__ __forceinline__ int nk_tile_cut(const int tile, const int ntiles, const int n_eval) {
    if (tile <= 0) {
        return 0;
    }
    if (tile >= ntiles) {
        return n_eval;
    }
    return ((int) (((int64_t) tile * n_eval) / ntiles)) & ~3;
}

// Stage 1: 4-row unrolled F16 folded-head GEMV + bias + base + softcap + exact radix top-K (or greedy max).
// Grid: (nb, ntiles) so multi-seq CTAs for the same tile co-execute and reuse fold_w in L2; Block: 256 (8 warps).
template <bool HAS_IDS, bool GREEDY, bool L2_EF>
static __global__ void __launch_bounds__(256, 4) k_nk_dspark_head_scan(
        const half  * __restrict__ fold_w,
        const float * __restrict__ latent,
        const float * __restrict__ fold_bias,
        const float * __restrict__ base,
        const int   * __restrict__ ids,
        uint64_t    * __restrict__ cand_out,
        const int    n_eval,
        const int    ntiles,
        const int    k,
        const float  cap,
        const float  inv_cap,
        const size_t lat_stride,
        const size_t base_stride) {
    __shared__ float         s_x[256];
    __shared__ uint64_t      s_out[64];
    __shared__ nk_radix_smem sm_radix;
    extern __shared__ uint64_t s_keys[]; // dynamic: max_tile_rows uint64_t (0 B when GREEDY)

    const int b       = blockIdx.x;
    const int tile    = blockIdx.y;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    s_x[tid] = latent[(size_t) b * lat_stride + tid];
    __syncthreads();

    const float4 x0 = *reinterpret_cast<const float4 *>(s_x + lane * 8);
    const float4 x1 = *reinterpret_cast<const float4 *>(s_x + lane * 8 + 4);

    const float * base_b = base + (size_t) b * base_stride;
    const uint64_t policy = L2_EF ? nk_l2_evict_first() : 0ull;

    const int row_beg = nk_tile_cut(tile,     ntiles, n_eval);
    const int row_end = nk_tile_cut(tile + 1, ntiles, n_eval);
    const int n_rows  = row_end - row_beg;

    uint64_t best_greedy = 0ull;

    // Main loop: each warp processes 4 consecutive rows per iteration (32 consecutive rows/CTA-step).
    // All 4 LDG.128 weight loads + coalesced 4-lane bias/base loads issue before reduction/softcap.
#pragma unroll 1
    for (int r = row_beg + warp_id * 4; r < row_end; r += 32) {
        if (r + 3 < row_end) {
            const int wi0 = HAS_IDS ? __ldg(ids + r + 0) : (r + 0);
            const int wi1 = HAS_IDS ? __ldg(ids + r + 1) : (r + 1);
            const int wi2 = HAS_IDS ? __ldg(ids + r + 2) : (r + 2);
            const int wi3 = HAS_IDS ? __ldg(ids + r + 3) : (r + 3);

            const uint4 w0 = nk_ldg128_stream<L2_EF>(fold_w + (size_t) wi0 * 256 + lane * 8, policy);
            const uint4 w1 = nk_ldg128_stream<L2_EF>(fold_w + (size_t) wi1 * 256 + lane * 8, policy);
            const uint4 w2 = nk_ldg128_stream<L2_EF>(fold_w + (size_t) wi2 * 256 + lane * 8, policy);
            const uint4 w3 = nk_ldg128_stream<L2_EF>(fold_w + (size_t) wi3 * 256 + lane * 8, policy);

            const float fb = (lane < 4) ? __ldg(fold_bias + r + lane) : 0.0f;
            const float bb = (lane < 4) ? __ldg(base_b    + r + lane) : 0.0f;

            float acc0 = nk_dot8(w0, x0, x1);
            float acc1 = nk_dot8(w1, x0, x1);
            float acc2 = nk_dot8(w2, x0, x1);
            float acc3 = nk_dot8(w3, x0, x1);

#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                acc0 += __shfl_xor_sync(0xffffffffu, acc0, off);
                acc1 += __shfl_xor_sync(0xffffffffu, acc1, off);
                acc2 += __shfl_xor_sync(0xffffffffu, acc2, off);
                acc3 += __shfl_xor_sync(0xffffffffu, acc3, off);
            }

            if (lane < 4) {
                const float acc_j = (lane == 0) ? acc0 : (lane == 1) ? acc1 : (lane == 2) ? acc2 : acc3;
                const int   idx_j = r + lane;
                const float v     = cap * tanhf((acc_j + fb + bb) * inv_cap);
                const uint64_t ky = nk_topk_key(v, idx_j, n_eval);
                if constexpr (GREEDY) {
                    if (ky > best_greedy) {
                        best_greedy = ky;
                    }
                } else {
                    s_keys[idx_j - row_beg] = ky;
                }
            }
        } else {
            // Tail (1..3 rows at tile boundary, executes at most once per CTA).
            for (int j = 0; j < 4; ++j) {
                const int idx_j = r + j;
                if (idx_j >= row_end) {
                    break;
                }
                const int wi = HAS_IDS ? __ldg(ids + idx_j) : idx_j;
                const uint4 w4 = nk_ldg128_stream<L2_EF>(fold_w + (size_t) wi * 256 + lane * 8, policy);
                float acc = nk_dot8(w4, x0, x1);
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    acc += __shfl_xor_sync(0xffffffffu, acc, off);
                }
                if (lane == 0) {
                    const float v = cap * tanhf((acc + fold_bias[idx_j] + base_b[idx_j]) * inv_cap);
                    const uint64_t ky = nk_topk_key(v, idx_j, n_eval);
                    if constexpr (GREEDY) {
                        if (ky > best_greedy) {
                            best_greedy = ky;
                        }
                    } else {
                        s_keys[idx_j - row_beg] = ky;
                    }
                }
            }
        }
    }

    if constexpr (GREEDY) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const uint64_t other = __shfl_xor_sync(0xffffffffu, best_greedy, off);
            if (other > best_greedy) {
                best_greedy = other;
            }
        }
        if (lane == 0) {
            s_out[warp_id] = best_greedy;
        }
        __syncthreads();
        if (tid == 0) {
            uint64_t gbest = s_out[0];
#pragma unroll
            for (int w = 1; w < 8; ++w) {
                if (s_out[w] > gbest) {
                    gbest = s_out[w];
                }
            }
            cand_out[(size_t) b * ntiles + tile] = gbest;
        }
        return;
    }

    __syncthreads();
    nk_radix_topk<256, false>(s_keys, n_rows, k, s_out, sm_radix, tid);

    uint64_t * out = cand_out + ((size_t) b * ntiles + tile) * k;
    if (tid < k) {
        out[tid] = s_out[tid];
    }
}

// Stage 2A: Greedy merge (1 CTA per sequence, reduces ntiles max keys -> prev_out[b]).
template <bool HAS_IDS>
static __global__ void __launch_bounds__(256) k_nk_dspark_head_merge_argmax(
        const uint64_t * __restrict__ cand_in,
        const int      * __restrict__ ids,
        int            * __restrict__ prev_out,
        const int ntiles,
        const int n_eval) {
    __shared__ uint64_t s_warp[8];
    const int b       = blockIdx.x;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    const uint64_t * row = cand_in + (size_t) b * ntiles;
    uint64_t best = 0ull;
    for (int i = tid; i < ntiles; i += 256) {
        const uint64_t v = row[i];
        if (v > best) {
            best = v;
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const uint64_t other = __shfl_xor_sync(0xffffffffu, best, off);
        if (other > best) {
            best = other;
        }
    }
    if (lane == 0) {
        s_warp[warp_id] = best;
    }
    __syncthreads();
    if (tid == 0) {
        uint64_t gbest = s_warp[0];
#pragma unroll
        for (int w = 1; w < 8; ++w) {
            if (s_warp[w] > gbest) {
                gbest = s_warp[w];
            }
        }
        const int local_idx = n_eval - 1 - (int) (uint32_t) gbest;
        prev_out[b] = HAS_IDS ? ids[local_idx] : local_idx;
    }
}

// Stage 2B: Sampled exact 8-bit radix merge + optional shortlist remap + Gumbel-max selection.
// Uses only ~2.6 KB SMEM (vs 33.3 KB for 4096-element bitonic sort) and works for arbitrary ncand.
template <bool HAS_IDS>
static __global__ void __launch_bounds__(512) k_nk_dspark_head_merge_sample(
        const uint64_t * __restrict__ cand_in,
        const int      * __restrict__ ids,
        const float    * __restrict__ noise,
        int            * __restrict__ cand_out,
        float          * __restrict__ vals_out,
        int            * __restrict__ prev_out,
        const int    ncand,
        const int    n_eval,
        const int    k,
        const float  inv_temp,
        const size_t noise_stride,
        const size_t cand_stride,
        const size_t vals_stride) {
    __shared__ uint64_t      s_out[64];
    __shared__ nk_radix_smem sm_radix;
    __shared__ float         s_score[64];
    __shared__ int           s_tok[64];

    const int b   = blockIdx.x;
    const int tid = threadIdx.x;

    const uint64_t * c = cand_in + (size_t) b * ncand;
    nk_radix_topk<512, true>(c, ncand, k, s_out, sm_radix, tid);

    if (tid < k) {
        const uint64_t key       = s_out[tid];
        const int      local_idx = n_eval - 1 - (int) (uint32_t) key;
        const int      tok_id    = HAS_IDS ? ids[local_idx] : local_idx;
        const float    val       = nk_key_to_float(key);
        const float    g         = noise[(size_t) b * noise_stride + tid];

        cand_out[(size_t) b * cand_stride + tid] = tok_id;
        vals_out[(size_t) b * vals_stride + tid] = val;
        s_tok[tid]   = tok_id;
        s_score[tid] = val * inv_temp + g;
    }
    __syncthreads();

    // Exact argmax over k <= 64 (lowest index wins ties, matching base dspark-head.cu).
    if (tid == 0) {
        int   best_j = 0;
        float best_s = s_score[0];
        for (int j = 1; j < k; ++j) {
            const float sc = s_score[j];
            if (sc > best_s) {
                best_s = sc;
                best_j = j;
            }
        }
        prev_out[b] = s_tok[best_j];
    }
}

} // namespace nk_dspark_head

bool ggml_cuda_nk_dspark_head_launch(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * fold_w,
        const ggml_tensor * latent,
        const ggml_tensor * fold_bias,
        const ggml_tensor * base,
        const ggml_tensor * ids,
        const ggml_tensor * noise,
        ggml_tensor       * cand_out,
        ggml_tensor       * vals_out,
        ggml_tensor       * prev_out,
        int64_t             n_eval,
        int                 sk,
        float               softcap,
        float               inv_temp,
        int                 ntiles_override) {
    using namespace nk_dspark_head;

    const int dev = ctx.device;
    const int cc  = ggml_cuda_info().devices[dev].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_AMPERE) {
        return false;
    }
    if (!fold_w || !latent || !fold_bias || !base || !prev_out) {
        return false;
    }
    if (fold_w->type != GGML_TYPE_F16 || fold_w->ne[0] != 256 || !ggml_is_contiguous(fold_w)) {
        return false;
    }
    if (((uintptr_t) fold_w->data & 15u) != 0) {
        return false;
    }
    if (latent->type != GGML_TYPE_F32 || latent->ne[0] != 256 ||
            fold_bias->type != GGML_TYPE_F32 || base->type != GGML_TYPE_F32 ||
            prev_out->type != GGML_TYPE_I32) {
        return false;
    }
    const int nb = (int) latent->ne[1];
    if (nb <= 0 || nb > 64 || n_eval <= 0 || n_eval > INT_MAX || softcap <= 0.0f) {
        return false;
    }
    const bool sampled = noise != nullptr;
    if (sampled) {
        if (sk < 1 || sk > 64 || !cand_out || !vals_out ||
                cand_out->type != GGML_TYPE_I32 || vals_out->type != GGML_TYPE_F32 ||
                noise->type != GGML_TYPE_F32 || n_eval < sk ||
                cand_out->ne[0] != sk || cand_out->ne[1] != nb || cand_out->ne[2] != 1 ||
                vals_out->ne[0] != sk || vals_out->ne[1] != nb || vals_out->ne[2] != 1 ||
                cand_out->nb[0] != sizeof(int32_t) || (cand_out->nb[1] % sizeof(int32_t)) != 0 ||
                (nb > 1 && cand_out->nb[1] < (size_t) sk * sizeof(int32_t)) ||
                vals_out->nb[0] != sizeof(float) || (vals_out->nb[1] % sizeof(float)) != 0 ||
                (nb > 1 && vals_out->nb[1] < (size_t) sk * sizeof(float))) {
            return false;
        }
    } else {
        sk = 1;
    }

    const int sm_count = std::max(1, ggml_cuda_info().devices[dev].nsm);
    const int min_tiles = (int) ((n_eval + 1023) / 1024);
    int target_tiles = ntiles_override > 0 ? ntiles_override : (sm_count * 4 + nb - 1) / nb;
    if (ntiles_override <= 0 && n_eval < (int64_t) target_tiles * 64) {
        target_tiles = (int) ((n_eval + 63) / 64);
    }
    const int ntiles = std::max(1, std::max(min_tiles, target_tiles));
    const int ncand  = ntiles * sk;

    // Max tile row count under 4-aligned cuts is <= ceil(n_eval / ntiles) + 4.
    const int max_tile_rows = (int) ((n_eval + ntiles - 1) / ntiles) + 4;
    const size_t scan_smem  = sampled ? (size_t) max_tile_rows * sizeof(uint64_t) : 0u;

    const float inv_cap      = 1.0f / softcap;
    const size_t lat_stride  = latent->nb[1] / sizeof(float);
    const size_t base_stride = base->nb[1]   / sizeof(float);
    const bool has_ids       = ids != nullptr;
    const int * ids_d        = has_ids ? (const int *) ids->data : nullptr;
    const int l2_mode        = nk_dspark_head_l2_mode();
    const bool l2_ef         = (l2_mode < 0) ? (nb == 1) : (l2_mode != 0);

    static std::atomic_flag logged = ATOMIC_FLAG_INIT;
    if (!logged.test_and_set()) {
        GGML_LOG_WARN("[NK_DSPARK_HEAD] active (4-row unroll + 8-bit radix top-K + reg sort64, sm=%d, l2_mode=%d)\n",
            sm_count, l2_mode);
    }
    const int64_t n = g_nk_dspark_head_launches.fetch_add(1, std::memory_order_relaxed) + 1;
    if (nk_dspark_head_trace() && n <= 64) {
        GGML_LOG_INFO("[NK_DSPARK_HEAD] launch #%lld n_eval=%lld nb=%d sk=%d sampled=%d ids=%d ntiles=%d l2_ef=%d smem=%zu\n",
            (long long) n, (long long) n_eval, nb, sk, sampled ? 1 : 0, has_ids ? 1 : 0, ntiles, l2_ef ? 1 : 0, scan_smem);
    }

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<uint64_t> cand_buf(ctx.pool(), (size_t) nb * ncand);

    const dim3 grid1(nb, ntiles, 1);
    auto launch_scan = [&](auto ids_tag, auto greedy_tag, auto l2_tag) {
        constexpr bool I = decltype(ids_tag)::value;
        constexpr bool G = decltype(greedy_tag)::value;
        constexpr bool L = decltype(l2_tag)::value;
        k_nk_dspark_head_scan<I, G, L><<<grid1, 256, scan_smem, stream>>>(
            (const half *)  fold_w->data,
            (const float *) latent->data,
            (const float *) fold_bias->data,
            (const float *) base->data,
            ids_d,
            cand_buf.get(),
            (int) n_eval, ntiles, sk, softcap, inv_cap, lat_stride, base_stride);
    };

    auto dispatch_scan = [&](auto ids_tag, auto greedy_tag) {
        if (l2_ef) {
            launch_scan(ids_tag, greedy_tag, std::true_type{});
        } else {
            launch_scan(ids_tag, greedy_tag, std::false_type{});
        }
    };

    if (!sampled) {
        if (has_ids) {
            dispatch_scan(std::true_type{}, std::true_type{});
            k_nk_dspark_head_merge_argmax<true><<<nb, 256, 0, stream>>>(
                cand_buf.get(), ids_d, (int *) prev_out->data, ntiles, (int) n_eval);
        } else {
            dispatch_scan(std::false_type{}, std::true_type{});
            k_nk_dspark_head_merge_argmax<false><<<nb, 256, 0, stream>>>(
                cand_buf.get(), nullptr, (int *) prev_out->data, ntiles, (int) n_eval);
        }
    } else {
        const size_t noise_stride = noise->nb[1]    / sizeof(float);
        const size_t cand_stride  = cand_out->nb[1] / sizeof(int32_t);
        const size_t vals_stride  = vals_out->nb[1] / sizeof(float);

        if (has_ids) {
            dispatch_scan(std::true_type{}, std::false_type{});
            k_nk_dspark_head_merge_sample<true><<<nb, 512, 0, stream>>>(
                cand_buf.get(), ids_d,
                (const float *) noise->data,
                (int *)         cand_out->data,
                (float *)       vals_out->data,
                (int *)         prev_out->data,
                ncand, (int) n_eval, sk, inv_temp,
                noise_stride, cand_stride, vals_stride);
        } else {
            dispatch_scan(std::false_type{}, std::false_type{});
            k_nk_dspark_head_merge_sample<false><<<nb, 512, 0, stream>>>(
                cand_buf.get(), nullptr,
                (const float *) noise->data,
                (int *)         cand_out->data,
                (float *)       vals_out->data,
                (int *)         prev_out->data,
                ncand, (int) n_eval, sk, inv_temp,
                noise_stride, cand_stride, vals_stride);
        }
    }

    CUDA_CHECK(cudaGetLastError());
    return true;
}

#else

bool ggml_cuda_nk_dspark_head_launch(
        ggml_backend_cuda_context &,
        const ggml_tensor *, const ggml_tensor *, const ggml_tensor *, const ggml_tensor *,
        const ggml_tensor *, const ggml_tensor *, ggml_tensor *, ggml_tensor *, ggml_tensor *,
        int64_t, int, float, float, int) {
    return false;
}

#endif
