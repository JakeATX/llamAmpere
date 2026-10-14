#include "argsort.cuh"
#include "top-k.cuh"

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
// DeviceTopK has a race condition before CCCL 3.4.3.
// https://github.com/NVIDIA/cccl/pull/10627
#    if (CCCL_MAJOR_VERSION > 3 || \
         (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION > 4) || \
         (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION == 4 && CCCL_PATCH_VERSION >= 3))
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL >= 3.4.3
#endif      // GGML_CUDA_USE_CUB

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#endif                            // CUB_TOP_K_AVAILABLE


// Two-stage top-k for wide rows: a global top-k element has at most k-1 larger
// elements, so at most k-1 inside its own tile and tiling cannot drop a winner.
#define TOPK_CAND  1024   // argsort_f32_i32_cuda_bitonic's row limit

// measured on H200 and A10G
#define TOPK_BLOCK     256
#define TOPK_TILE_WIDE 8192
#define TOPK_TILE      4096

template <int TILE, int BLOCK>
static __global__ void topk_tile(const float * src, float * cand_val, int * cand_idx,
                                 const int ncols, const int ntiles, const int k) {
    __shared__ uint64_t smem[BLOCK];

    const int     row     = blockIdx.x / ntiles;
    const int     tile    = blockIdx.x % ntiles;
    const float * row_ptr = src + (size_t) row * ncols;

    uint64_t keys[TILE / BLOCK];
#pragma unroll
    for (int i = 0; i < TILE / BLOCK; ++i) {
        const int col = tile * TILE + threadIdx.x + i * BLOCK;
        uint32_t  b   = col < ncols ? __float_as_uint(row_ptr[col]) : 0;
        b = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
        keys[i] = col < ncols ? (((uint64_t) b << 32) | (uint32_t) (ncols - 1 - col)) : 0;
    }

    const size_t out = ((size_t) row * ntiles + tile) * k;
    for (int j = 0; j < k; ++j) {
        uint64_t local = 0;
#pragma unroll
        for (int i = 0; i < TILE / BLOCK; ++i) {
            local = max(local, keys[i]);
        }
        smem[threadIdx.x] = local;
        __syncthreads();
        for (int s = BLOCK / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                smem[threadIdx.x] = max(smem[threadIdx.x], smem[threadIdx.x + s]);
            }
            __syncthreads();
        }
        const uint64_t best = smem[0];
        if (threadIdx.x == 0) {
            const int col = ncols - 1 - (int) (best & 0xFFFFFFFFu);
            cand_val[out + j] = best ? row_ptr[col] : -INFINITY;
            cand_idx[out + j] = best ? col : 0;
        }
#pragma unroll
        for (int i = 0; i < TILE / BLOCK; ++i) {
            if (keys[i] == best) {
                keys[i] = 0;
            }
        }
        __syncthreads();
    }
}

// The argsort ranks candidates; turn its positions back into columns.
static __global__ void topk_unmap(const int * cand_idx, const int * order, int * dst,
                                  const int ncand, const int k) {
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        dst[(size_t) blockIdx.x * k + i] = cand_idx[(size_t) blockIdx.x * ncand + order[(size_t) blockIdx.x * ncand + i]];
    }
}

static bool ggml_cuda_top_k_tiled(ggml_cuda_pool & pool, const float * src, int * dst,
                                  const int ncols, const int nrows, const int k,
                                  cudaStream_t stream) {
    // Narrow rows are already handled whole by the bitonic sort below.
    const int tile   = ncols >= 65536 ? TOPK_TILE_WIDE : TOPK_TILE;
    const int ntiles = (ncols + tile - 1) / tile;
    const int ncand  = ntiles * k;
    if (ncols <= TOPK_CAND || ncand > TOPK_CAND) {
        return false;
    }

    ggml_cuda_pool_alloc<float> cand_val(pool, (size_t) nrows * ncand);
    ggml_cuda_pool_alloc<int>   cand_idx(pool, (size_t) nrows * ncand);
    ggml_cuda_pool_alloc<int>   order   (pool, (size_t) nrows * ncand);

    if (tile == TOPK_TILE_WIDE) {
        topk_tile<TOPK_TILE_WIDE, TOPK_BLOCK><<<nrows * ntiles, TOPK_BLOCK, 0, stream>>>(
                src, cand_val.get(), cand_idx.get(), ncols, ntiles, k);
    } else {
        topk_tile<TOPK_TILE, TOPK_BLOCK><<<nrows * ntiles, TOPK_BLOCK, 0, stream>>>(
                src, cand_val.get(), cand_idx.get(), ncols, ntiles, k);
    }
    argsort_f32_i32_cuda_bitonic(cand_val.get(), order.get(), ncand, nrows,
            GGML_SORT_ORDER_DESC, stream);
    topk_unmap<<<nrows, TOPK_BLOCK, 0, stream>>>(cand_idx.get(), order.get(), dst, ncand, k);
    return true;
}

#if !defined(GGML_CUDA_USE_CUB) && defined(GGML_USE_HIP)

static __device__ __forceinline__ uint32_t top_k_float_to_ordered(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t mask = (uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U;
    return bits ^ mask;
}

struct top_k_radix_state {
    uint32_t prefix;
    uint32_t prefix_mask;
    int rank;
    int greater_count;
    int equal_count;
};

static __global__ void top_k_radix_init(top_k_radix_state * states, int nrows, int k) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row] = {0, 0, k, 0, 0};
    }
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_histogram(
        const float * __restrict__ src,
        const top_k_radix_state * __restrict__ states,
        int * __restrict__ block_histograms,
        int ncols,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    __shared__ int histogram[NBINS];

    histogram[tid] = 0;
    __syncthreads();

    const top_k_radix_state state = states[row];
    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if ((key & state.prefix_mask) == state.prefix) {
            atomicAdd(&histogram[(key >> shift) & (NBINS - 1)], 1);
        }
    }
    __syncthreads();

    const size_t histogram_offset =
        ((size_t) row * blocks_per_row + row_block) * NBINS;
    block_histograms[histogram_offset + tid] = histogram[tid];
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_select(
        const int * __restrict__ block_histograms,
        top_k_radix_state * __restrict__ states,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int histogram[NBINS];

    int count = 0;
    for (int row_block = 0; row_block < blocks_per_row; ++row_block) {
        const size_t offset = ((size_t) row * blocks_per_row + row_block) * NBINS;
        count += block_histograms[offset + tid];
    }
    histogram[tid] = count;
    __syncthreads();

    if (tid == 0) {
        top_k_radix_state state = states[row];
        int bin = NBINS - 1;
        while (bin > 0 && histogram[bin] < state.rank) {
            state.rank -= histogram[bin--];
        }
        state.prefix |= (uint32_t) bin << shift;
        state.prefix_mask |= (uint32_t) (NBINS - 1) << shift;
        states[row] = state;
    }
}

static __global__ void top_k_radix_reset_counters(top_k_radix_state * states, int nrows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row].greater_count = 0;
        states[row].equal_count = 0;
    }
}

template<int BLOCK_SIZE>
static __global__ void top_k_radix_gather(
        const float * __restrict__ src,
        int * __restrict__ dst,
        top_k_radix_state * __restrict__ states,
        int ncols,
        int k,
        int blocks_per_row) {
    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    int * row_dst = dst + (size_t) row * k;
    top_k_radix_state * state = &states[row];

    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if (key > state->prefix) {
            const int pos = atomicAdd(&state->greater_count, 1);
            row_dst[pos] = col;
        } else if (key == state->prefix) {
            const int pos = atomicAdd(&state->equal_count, 1);
            if (pos < state->rank) {
                row_dst[k - state->rank + pos] = col;
            }
        }
    }
}

static void top_k_radix_cuda(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int nrows, int k, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int RADIX_BITS = 8;
    constexpr int NBINS = 1 << RADIX_BITS;
    const int blocks_per_row = std::min((ncols + 1023) / 1024, 64);

    ggml_cuda_pool_alloc<top_k_radix_state> states_alloc(pool, nrows);
    ggml_cuda_pool_alloc<int> histograms_alloc(pool, (size_t) nrows * blocks_per_row * NBINS);
    top_k_radix_state * states = states_alloc.get();
    int * histograms = histograms_alloc.get();

    top_k_radix_init<<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows, k);

    const dim3 row_grid(blocks_per_row * nrows);
    for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
        top_k_radix_histogram<BLOCK_SIZE, RADIX_BITS>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                src, states, histograms, ncols, blocks_per_row, shift);
        top_k_radix_select<BLOCK_SIZE, RADIX_BITS>
            <<<nrows, BLOCK_SIZE, 0, stream>>>(histograms, states, blocks_per_row, shift);
    }

    top_k_radix_reset_counters
        <<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows);
    top_k_radix_gather<BLOCK_SIZE>
        <<<row_grid, BLOCK_SIZE, 0, stream>>>(
            src, dst, states, ncols, k, blocks_per_row);
}

#endif // !defined(GGML_CUDA_USE_CUB) && defined(GGML_USE_HIP)

// fork: opt-in exact radix select for wide rows (GGML_CUDA_TOPK_SELECT=1, 2 = also compare with the sort).
// One block per row finds the k-th largest key in three digit passes (11, 11, 10 bits), then writes every
// index above it plus the lowest-index ties. That is the set the stable descending radix sort below keeps
// (equal keys stay in index order, -0.0 sorts as +0.0), without sorting the whole row. The output is the
// set in index order, not sorted by value: GGML_OP_TOP_K promises an unordered result.
#define TOPK_SEL_BLOCK 256
#define TOPK_SEL_BINS  2048

static __device__ __forceinline__ uint32_t topk_sel_key(const float v) {
    uint32_t b = __float_as_uint(v);
    b = b == 0x80000000u ? 0u : b;
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// exclusive prefix sum over the block in thread order; *total gets the block sum
template <int BLOCK>
static __device__ __forceinline__ int topk_sel_scan(const int v, int * s_warp, int * total) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    int x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffff, x, o);
        x += lane >= o ? y : 0;
    }
    if (lane == 31) {
        s_warp[warp] = x;
    }
    __syncthreads();
    if (warp == 0) {
        int w = lane < BLOCK/32 ? s_warp[lane] : 0;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffff, w, o);
            w += lane >= o ? y : 0;
        }
        if (lane < BLOCK/32) {
            s_warp[lane] = w;
        }
    }
    __syncthreads();
    *total = s_warp[BLOCK/32 - 1];
    const int excl = x - v + (warp > 0 ? s_warp[warp - 1] : 0);
    __syncthreads();
    return excl;
}

template <int BLOCK>
static __global__ void __launch_bounds__(BLOCK) topk_select_rows(const float * __restrict__ src, int * __restrict__ dst,
                                                                 const int ncols, const int k) {
    constexpr int BPT = TOPK_SEL_BINS / BLOCK;
    __shared__ int hist[TOPK_SEL_BINS];
    __shared__ int s_warp[BLOCK/32];
    __shared__ int s_digit;
    __shared__ int s_rank;

    const int     tid = threadIdx.x;
    const float * row = src + (size_t) blockIdx.x * ncols;
    int *         out = dst + (size_t) blockIdx.x * k;

    uint32_t prefix = 0;
    uint32_t pmask  = 0;
    int      rank   = k; // the wanted key is the rank-th largest of the keys that match the prefix

#pragma unroll 1
    for (int pass = 0; pass < 3; ++pass) {
        const int      shift = pass == 0 ? 21 : (pass == 1 ? 10 : 0);
        const uint32_t dmask = pass == 2 ? 0x3FFu : 0x7FFu;
        for (int i = tid; i < TOPK_SEL_BINS; i += BLOCK) {
            hist[i] = 0;
        }
        __syncthreads();
        for (int c = tid; c < ncols; c += BLOCK) {
            const uint32_t key = topk_sel_key(row[c]);
            if ((key & pmask) == prefix) {
                atomicAdd(&hist[(key >> shift) & dmask], 1);
            }
        }
        __syncthreads();

        // thread t owns bins from the top down: TOPK_SEL_BINS-1-BPT*t ... ; the scan counts the keys above them
        int own = 0;
#pragma unroll
        for (int j = 0; j < BPT; ++j) {
            own += hist[TOPK_SEL_BINS - 1 - BPT*tid - j];
        }
        int total;
        int above = topk_sel_scan<BLOCK>(own, s_warp, &total);
#pragma unroll
        for (int j = 0; j < BPT; ++j) {
            const int b = TOPK_SEL_BINS - 1 - BPT*tid - j;
            const int h = hist[b];
            if (above < rank && above + h >= rank) {
                s_digit = b;
                s_rank  = rank - above;
            }
            above += h;
        }
        __syncthreads();
        prefix |= (uint32_t) s_digit << shift;
        pmask  |= dmask << shift;
        rank    = s_rank;
        __syncthreads();
    }

    // prefix is now the k-th largest key; take all keys above it and its first `rank` copies in index order
    const int n_gt_want = k - rank;
    int n_gt = 0;
    int n_eq = 0;
#pragma unroll 1
    for (int base = 0; base < ncols && (n_gt < n_gt_want || n_eq < rank); base += BLOCK) {
        const int      c   = base + tid;
        const uint32_t key = c < ncols ? topk_sel_key(row[c]) : 0u;
        const int      gt  = c < ncols && key > prefix;
        const int      eq  = c < ncols && key == prefix;
        int total;
        const int excl = topk_sel_scan<BLOCK>(gt | (eq << 16), s_warp, &total);
        if (gt) {
            out[n_gt + (excl & 0xFFFF)] = c;
        }
        if (eq) {
            const int p = n_eq + (excl >> 16);
            if (p < rank) {
                out[n_gt_want + p] = c;
            }
        }
        n_gt += total & 0xFFFF;
        n_eq += total >> 16;
    }
}

static __device__ __forceinline__ unsigned long long topk_sel_mix(unsigned long long x) {
    x += 0x9e3779b97f4a7c15ull;
    x  = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x  = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

// rows whose selected live sets differ (order-free hash of the picked indices whose score is above -inf; which -inf
// entries fill a short row does not matter to the QSA selection, which sends them to dump rows)
static __global__ void topk_select_compare(const float * src, const int * a, const int * b, const int ncols, const int k,
                                           unsigned long long * n_diff) {
    __shared__ unsigned long long ha;
    __shared__ unsigned long long hb;
    if (threadIdx.x == 0) {
        ha = 0;
        hb = 0;
    }
    __syncthreads();
    const float * row = src + (size_t) blockIdx.x * ncols;
    unsigned long long sa = 0;
    unsigned long long sb = 0;
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        const int ia = a[(size_t) blockIdx.x * k + i];
        const int ib = b[(size_t) blockIdx.x * k + i];
        const bool la = ia < 0 || ia >= ncols || row[ia] > -INFINITY; // out of range counts as live, so it shows up
        const bool lb = ib < 0 || ib >= ncols || row[ib] > -INFINITY;
        sa += la ? topk_sel_mix((unsigned long long) (unsigned int) ia) : 0ull;
        sb += lb ? topk_sel_mix((unsigned long long) (unsigned int) ib) : 0ull;
    }
    atomicAdd(&ha, sa);
    atomicAdd(&hb, sb);
    __syncthreads();
    if (threadIdx.x == 0 && ha != hb) {
        atomicAdd(n_diff, 1ull);
    }
}

static void ggml_cuda_top_k_sort_fallback(ggml_cuda_pool & pool, const float * src0_d, int * dst_d, const int64_t ncols,
                                          const int64_t nrows, const int64_t k, const size_t nb01, cudaStream_t stream) {
#ifdef CUB_TOP_K_AVAILABLE
    GGML_UNUSED(nb01);
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    // Fall back to argsort + copy
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;
    const bool   use_bitonic    = shared_mem <= max_shared_mem && ncols <= 1024;
    const int    chunk_nrows    = argsort_f32_i32_cuda_cub_chunk_nrows(nb01, nrows);

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * chunk_nrows);
    int *                     tmp_dst = temp_dst_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        int iter_nrows = std::min((int64_t) chunk_nrows, nrows - i);

        if (use_bitonic) {
            argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        } else {
            argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        }
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), iter_nrows,
                                     cudaMemcpyDeviceToDevice, stream));

        src0_d += ncols * iter_nrows;
        dst_d  += k     * iter_nrows;
    }
#else                             // GGML_CUDA_USE_CUB
    GGML_UNUSED(nb01);
#if defined(GGML_USE_HIP)
    if (ncols > 1024) {
        top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
    } else {
#endif // defined(GGML_USE_HIP)
        ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
        int *                     tmp_dst = temp_dst_alloc.get();
        argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                     cudaMemcpyDeviceToDevice, stream));
#if defined(GGML_USE_HIP)
    }
#endif // defined(GGML_USE_HIP)
#endif
}

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();

    // fork: opt-in radix select for wide rows with a large k, where the tiling runs k block-wide reductions per tile
    // and the fallback sorts whole rows. 2 = also run the default path and count rows whose live sets differ.
    static const int select_mode = [] {
        const char * value = getenv("GGML_CUDA_TOPK_SELECT");
        return value != nullptr ? atoi(value) : 0;
    }();
    if (select_mode > 0 && ncols > 1024 && k >= 64 && k <= ncols && nrows <= INT_MAX) {
        topk_select_rows<TOPK_SEL_BLOCK><<<nrows, TOPK_SEL_BLOCK, 0, stream>>>(src0_d, dst_d, ncols, k);
        CUDA_CHECK(cudaGetLastError());
        if (select_mode == 2) {
            static unsigned long long n_calls = 0, n_diff_total = 0, n_rows_total = 0;
            ggml_cuda_pool_alloc<int> ref(pool, nrows * k);
            const bool tiled = ggml_cuda_top_k_tiled(pool, src0_d, ref.get(), ncols, nrows, k, stream);
            if (!tiled) {
                ggml_cuda_top_k_sort_fallback(pool, src0_d, ref.get(), ncols, nrows, k, src0->nb[1], stream);
            }
            ggml_cuda_pool_alloc<unsigned long long> n_diff(pool, 1);
            CUDA_CHECK(cudaMemsetAsync(n_diff.get(), 0, sizeof(unsigned long long), stream));
            topk_select_compare<<<nrows, 128, 0, stream>>>(src0_d, ref.get(), dst_d, ncols, k, n_diff.get());
            unsigned long long h_diff = 0;
            CUDA_CHECK(cudaMemcpyAsync(&h_diff, n_diff.get(), sizeof(h_diff), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            n_calls++; n_diff_total += h_diff; n_rows_total += nrows;
            if (h_diff != 0 || (n_calls & (n_calls - 1)) == 0) {
                ggml_log_internal(h_diff != 0 ? GGML_LOG_LEVEL_WARN : GGML_LOG_LEVEL_INFO,
                    "%s: TOPK_SELECT check call %llu (ncols %lld nrows %lld k %lld, vs %s): %llu differing rows, total %llu of %llu\n",
                    __func__, n_calls, (long long) ncols, (long long) nrows, (long long) k, tiled ? "tiled" : "sort",
                    h_diff, n_diff_total, n_rows_total);
            }
        }
        return;
    }

    if (ggml_cuda_top_k_tiled(pool, src0_d, dst_d, ncols, nrows, k, stream)) {
        return;
    }

    ggml_cuda_top_k_sort_fallback(pool, src0_d, dst_d, ncols, nrows, k, src0->nb[1], stream);
}
