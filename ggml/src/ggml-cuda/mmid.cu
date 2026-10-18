#include "common.cuh"
#include "mmid.cuh"

// To reduce shared memory use, store "it" and "iex_used" with 22/10 bits each.
struct mm_ids_helper_store {
    uint32_t data;

    __device__ mm_ids_helper_store(const uint32_t it, const uint32_t iex_used) {
        data = (it & 0x003FFFFF) | (iex_used << 22);
    }

    __device__ uint32_t it() const {
        return data & 0x003FFFFF;
    }

    __device__ uint32_t iex_used() const {
        return data >> 22;
    }
};
static_assert(sizeof(mm_ids_helper_store) == 4, "unexpected size for mm_ids_helper_store");

// the generic path passes 0, which needs no padding since it never groups lanes by token
template <int n> struct mm_ids_pow2 { static constexpr int value = 2*mm_ids_pow2<(n + 1)/2>::value; };
template <>      struct mm_ids_pow2<1> { static constexpr int value = 1; };
template <>      struct mm_ids_pow2<0> { static constexpr int value = 1; };

// Helper function for mul_mat_id, converts ids to a more convenient format.
// ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted by expert.
// ids_dst describes the same mapping but for the dst tensor.
// The upper and lower bounds for the ith expert in the compact src1 tensor are stored in expert_bounds[i:i+1].
template <int n_expert_used_template>
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
    const int expert = blockIdx.x;

    extern __shared__ char data_mm_ids_helper[];
    mm_ids_helper_store * store = (mm_ids_helper_store *) data_mm_ids_helper;

    int nex_prev   = 0; // Number of columns for experts with a lower index.
    int it_compact = 0; // Running index for the compact slice of this expert.

    if constexpr (n_expert_used_template == 0) {
        // Generic implementation:
        for (int it = 0; it < n_tokens; ++it) {
            int iex_used = -1; // The index at which the expert is used, if any.
            for (int iex = threadIdx.x; iex < n_expert_used; iex += warp_size) {
                const int expert_used = ids[it*si1 + iex];
                nex_prev += expert_used < expert;
                if (expert_used == expert) {
                    iex_used = iex;
                }
            }

            if (iex_used != -1) {
                store[it_compact] = mm_ids_helper_store(it, iex_used);
            }

            if (warp_reduce_any<warp_size>(iex_used != -1)) {
                it_compact++;
            }
        }
    } else {
        // Implementation optimized for specific numbers of experts used:
        // a warp holds a whole number of token slots, so the slot count is padded to a power of 2
        constexpr int neu_padded = mm_ids_pow2<n_expert_used_template>::value;
        static_assert(neu_padded <= warp_size && warp_size % neu_padded == 0, "bad n_expert_used");
        for (int it0 = 0; it0 < n_tokens; it0 += warp_size/neu_padded) {
            const int it = it0 + threadIdx.x / neu_padded;

            const int iex = threadIdx.x % neu_padded; // The index at which the expert is used, if any.
            const int expert_used = (neu_padded == n_expert_used || iex < n_expert_used) && it < n_tokens ?
                ids[it*si1 + iex] : INT_MAX;
            const int iex_used = expert_used == expert ? iex : -1;
            nex_prev += expert_used < expert;

            // Whether the threads at this token position have used the expert:
            const int it_compact_add_self = warp_reduce_any<neu_padded>(iex_used != -1);

            // Do a scan over threads at lower token positions in warp to get the correct index for writing data:
            int it_compact_add_lower = 0;
#pragma unroll
            for (int offset = neu_padded; offset < warp_size; offset += neu_padded) {
                const int tmp = __shfl_up_sync(0xFFFFFFFF, it_compact_add_self, offset, warp_size);
                if (threadIdx.x >= static_cast<unsigned int>(offset)) {
                    it_compact_add_lower += tmp;
                }
            }

            if (iex_used != -1) {
                store[it_compact + it_compact_add_lower] = mm_ids_helper_store(it, iex_used);
            }

            // The thread with the highest index in the warp always has the sum over the whole warp, use it to increment all threads:
            it_compact += __shfl_sync(0xFFFFFFFF, it_compact_add_lower + it_compact_add_self, warp_size - 1, warp_size);
        }
    }
    nex_prev = warp_reduce_sum<warp_size>(nex_prev);
    ggml_cuda_syncwarp();

    for (int itc = threadIdx.x; itc < it_compact; itc += warp_size) {
        const mm_ids_helper_store store_it = store[itc];
        const int it       = store_it.it();
        const int iex_used = store_it.iex_used();
        ids_dst[nex_prev + itc] = it*n_expert_used + iex_used;
        // ids_src1 holds the forward map, or the inverse map (token slot -> compact row) for quant dedup
        if (write_inverse) {
            ids_src1[it*n_expert_used + iex_used] = nex_prev + itc;
        } else {
            ids_src1[nex_prev + itc] = it*sis1 + iex_used % nchannels_y;
        }
    }

    if (threadIdx.x != 0) {
        return;
    }

    expert_bounds[expert] = nex_prev;

    if (expert < static_cast<int>(gridDim.x) - 1) {
        return;
    }

    expert_bounds[gridDim.x] = nex_prev + it_compact;
}

template <int n_expert_used_template>
static void launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    GGML_ASSERT(n_tokens          < (1 << 22) && "too few bits in mm_ids_helper_store");
    GGML_ASSERT(n_expert_used_var < (1 << 10) && "too few bits in mm_ids_helper_store");

    const int id = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[id].warp_size;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
    CUDA_SET_SHARED_MEMORY_LIMIT(mm_ids_helper<n_expert_used_template>, smpbo);

    const dim3 num_blocks(n_experts, 1, 1);
    const dim3 block_size(warp_size, 1, 1);
    const size_t nbytes_shared = n_tokens*sizeof(mm_ids_helper_store);
    GGML_ASSERT(nbytes_shared <= smpbo);
    mm_ids_helper<n_expert_used_template><<<num_blocks, block_size, nbytes_shared, stream>>>
        (ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used_var, nchannels_y, si1, sis1, write_inverse);
}

// [FNSTRATA] Opt-in parallel helper (GGML_CUDA_MMID_PAR=1): the same three outputs as mm_ids_helper, built by a
// counting sort instead of one warp per expert scanning every token. Tokens are cut into chunks of MMID_PAR_WARPS*32;
// 1) every chunk counts its entries per expert, 2) one block turns the counts into expert-major, chunk-minor offsets
// (= expert_bounds and each chunk's base inside its expert), 3) every chunk writes its entries at base + the number of
// earlier tokens of the same chunk that use the same expert. Each expert keeps its tokens in ascending order, exactly
// as in mm_ids_helper, so the outputs are identical for valid ids (ids in [0, n_experts), no expert twice per token).
// GGML_CUDA_MMID_PAR=2 also runs mm_ids_helper and counts differing entries (synchronizes, for testing only).
#define MMID_PAR_WARPS 8

template <int neu>
static __global__ void __launch_bounds__(32*MMID_PAR_WARPS) mm_ids_par_count(
        const int32_t * __restrict__ ids, int32_t * __restrict__ cnt, const int n_tokens, const int n_experts, const int si1) {
    extern __shared__ int mm_ids_par_hist[]; // [n_experts]
    for (int e = threadIdx.y*32 + threadIdx.x; e < n_experts; e += 32*MMID_PAR_WARPS) {
        mm_ids_par_hist[e] = 0;
    }
    __syncthreads();
    const int it = blockIdx.x*(32*MMID_PAR_WARPS) + threadIdx.y*32 + threadIdx.x;
    if (it < n_tokens) {
#pragma unroll
        for (int i = 0; i < neu; ++i) {
            const int e = ids[(int64_t) it*si1 + i];
            if (e >= 0 && e < n_experts) {
                atomicAdd(&mm_ids_par_hist[e], 1);
            }
        }
    }
    __syncthreads();
    for (int e = threadIdx.y*32 + threadIdx.x; e < n_experts; e += 32*MMID_PAR_WARPS) {
        cnt[(int64_t) blockIdx.x*n_experts + e] = mm_ids_par_hist[e];
    }
}

// one block, thread e owns expert e: running sum over the chunks, then a block-wide exclusive scan over the experts
static __global__ void __launch_bounds__(1024) mm_ids_par_scan(
        int32_t * __restrict__ cnt, int32_t * __restrict__ expert_bounds, const int n_experts, const int nchunks) {
    __shared__ int s_warp[32];
    const int e = threadIdx.x;
    int run = 0;
    if (e < n_experts) {
        for (int c = 0; c < nchunks; ++c) {
            const int v = cnt[(int64_t) c*n_experts + e];
            cnt[(int64_t) c*n_experts + e] = run;
            run += v;
        }
    }
    // exclusive scan of run over the block
    const int lane = e % 32, w = e / 32;
    int incl = run;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int t = __shfl_up_sync(0xFFFFFFFF, incl, o);
        if (lane >= o) {
            incl += t;
        }
    }
    if (lane == 31) {
        s_warp[w] = incl;
    }
    __syncthreads();
    if (w == 0) {
        const int nw = (blockDim.x + 31)/32;
        int v = lane < nw ? s_warp[lane] : 0;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int t = __shfl_up_sync(0xFFFFFFFF, v, o);
            if (lane >= o) {
                v += t;
            }
        }
        if (lane < nw) {
            s_warp[lane] = v; // inclusive over warps
        }
    }
    __syncthreads();
    const int off = (w > 0 ? s_warp[w - 1] : 0) + incl - run;
    if (e < n_experts) {
        expert_bounds[e] = off;
        if (e == n_experts - 1) {
            expert_bounds[n_experts] = off + run;
        }
        for (int c = 0; c < nchunks; ++c) {
            cnt[(int64_t) c*n_experts + e] += off;
        }
    }
}

template <int neu>
static __global__ void __launch_bounds__(32*MMID_PAR_WARPS) mm_ids_par_write(
        const int32_t * __restrict__ ids, const int32_t * __restrict__ base, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst,
        const int n_tokens, const int n_experts, const int nchannels_y, const int si1, const int sis1, const bool write_inverse) {
    extern __shared__ int mm_ids_par_wcnt[]; // [MMID_PAR_WARPS][n_experts] per-warp counts, then exclusive over warps
    __shared__ int s_ids[MMID_PAR_WARPS][32][neu];
    const int w = threadIdx.y, lane = threadIdx.x;
    for (int j = w*32 + lane; j < MMID_PAR_WARPS*n_experts; j += 32*MMID_PAR_WARPS) {
        mm_ids_par_wcnt[j] = 0;
    }
    const int it = blockIdx.x*(32*MMID_PAR_WARPS) + w*32 + lane;
    int my[neu];
#pragma unroll
    for (int i = 0; i < neu; ++i) {
        my[i] = it < n_tokens ? ids[(int64_t) it*si1 + i] : -1;
        s_ids[w][lane][i] = my[i];
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < neu; ++i) {
        if (my[i] >= 0 && my[i] < n_experts) {
            atomicAdd(&mm_ids_par_wcnt[w*n_experts + my[i]], 1);
        }
    }
    __syncthreads();
    // exclusive over the warps of this chunk, per expert
    for (int e = w*32 + lane; e < n_experts; e += 32*MMID_PAR_WARPS) {
        int run = 0;
#pragma unroll
        for (int ww = 0; ww < MMID_PAR_WARPS; ++ww) {
            const int v = mm_ids_par_wcnt[ww*n_experts + e];
            mm_ids_par_wcnt[ww*n_experts + e] = run;
            run += v;
        }
    }
    __syncthreads();
    if (it >= n_tokens) {
        return;
    }
    // rank inside the warp: earlier lanes (tokens) that use the same expert, at most once each
    int rank[neu];
#pragma unroll
    for (int i = 0; i < neu; ++i) {
        rank[i] = 0;
    }
    for (int l = 0; l < lane; ++l) {
#pragma unroll
        for (int j = 0; j < neu; ++j) {
            const int x = s_ids[w][l][j];
#pragma unroll
            for (int i = 0; i < neu; ++i) {
                rank[i] += x == my[i];
            }
        }
    }
#pragma unroll
    for (int i = 0; i < neu; ++i) {
        const int e = my[i];
        if (e < 0 || e >= n_experts) {
            continue;
        }
        const int pos = base[(int64_t) blockIdx.x*n_experts + e] + mm_ids_par_wcnt[w*n_experts + e] + rank[i];
        ids_dst[pos] = it*neu + i;
        if (write_inverse) {
            ids_src1[it*neu + i] = pos;
        } else {
            ids_src1[pos] = it*sis1 + i % nchannels_y;
        }
    }
}

static __global__ void mm_ids_par_diff(const int32_t * __restrict__ a, const int32_t * __restrict__ b, const int n, unsigned long long * __restrict__ ndiff) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n && a[i] != b[i]) {
        atomicAdd(ndiff, 1ull);
    }
}

template <int neu>
static bool launch_mm_ids_par(
        ggml_cuda_pool & pool, const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        const int n_experts, const int n_tokens, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    const int id = ggml_cuda_get_device();
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
    const size_t nbytes_w = (size_t) MMID_PAR_WARPS*n_experts*sizeof(int) + sizeof(int)*MMID_PAR_WARPS*32*neu;
    if (n_experts > 1024 || nbytes_w > smpbo) {
        return false;
    }
    // the opt-in limit covers static + dynamic shared memory: the dynamic maximum is what the static s_ids leaves
    CUDA_SET_SHARED_MEMORY_LIMIT(mm_ids_par_write<neu>, smpbo - sizeof(int)*MMID_PAR_WARPS*32*neu);
    const int nchunks = (n_tokens + 32*MMID_PAR_WARPS - 1)/(32*MMID_PAR_WARPS);
    ggml_cuda_pool_alloc<int32_t> cnt(pool, (size_t) nchunks*n_experts);
    const dim3 block(32, MMID_PAR_WARPS, 1);
    mm_ids_par_count<neu><<<nchunks, block, n_experts*sizeof(int), stream>>>(ids, cnt.get(), n_tokens, n_experts, si1);
    mm_ids_par_scan<<<1, ((n_experts + 31)/32)*32, 0, stream>>>(cnt.get(), expert_bounds, n_experts, nchunks);
    mm_ids_par_write<neu><<<nchunks, block, MMID_PAR_WARPS*n_experts*sizeof(int), stream>>>(
        ids, cnt.get(), ids_src1, ids_dst, n_tokens, n_experts, nchannels_y, si1, sis1, write_inverse);
    return true;
}

static void ggml_cuda_launch_mm_ids_helper_ref(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream);

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream,
        ggml_cuda_pool * pool) {
    static const int par_mode = [] { const char * e = getenv("GGML_CUDA_MMID_PAR"); return e ? atoi(e) : 0; }();
    const size_t smpbo = ggml_cuda_info().devices[ggml_cuda_get_device()].smpbo;
    const bool reference_fits = (size_t) n_tokens * sizeof(mm_ids_helper_store) <= smpbo;
    if (pool != nullptr && (n_expert_used == 1 || n_expert_used == 10) &&
            ((par_mode > 0 && n_tokens >= 64*MMID_PAR_WARPS) || !reference_fits)) {
        // A routed-row bucket uses one expert per gathered row. The existing
        // stable counting sort also handles that shape without O(tokens) shared
        // storage, so large buckets work even when the reference cannot launch.
        const auto launch = [&]() {
            return n_expert_used == 1 ?
                launch_mm_ids_par<1>(*pool, ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, nchannels_y, si1, sis1, write_inverse, stream) :
                launch_mm_ids_par<10>(*pool, ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, nchannels_y, si1, sis1, write_inverse, stream);
        };
        bool ok = false;
        if (par_mode == 2 && reference_fits) {
            const int64_t ne = (int64_t) n_tokens*n_expert_used;
            ggml_cuda_pool_alloc<int32_t> r_src1(*pool, ne), r_dst(*pool, ne), r_bounds(*pool, n_experts + 1);
            ok = launch();
            if (ok) {
                ggml_cuda_launch_mm_ids_helper_ref(ids, r_src1.get(), r_dst.get(), r_bounds.get(), n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
                ggml_cuda_pool_alloc<unsigned long long> nd(*pool, 3);
                CUDA_CHECK(cudaMemsetAsync(nd.get(), 0, 3*sizeof(unsigned long long), stream));
                mm_ids_par_diff<<<(ne + 255)/256, 256, 0, stream>>>(ids_src1, r_src1.get(), (int) ne, nd.get() + 0);
                mm_ids_par_diff<<<(ne + 255)/256, 256, 0, stream>>>(ids_dst,  r_dst.get(),  (int) ne, nd.get() + 1);
                mm_ids_par_diff<<<(n_experts + 1 + 255)/256, 256, 0, stream>>>(expert_bounds, r_bounds.get(), n_experts + 1, nd.get() + 2);
                unsigned long long h[3];
                CUDA_CHECK(cudaMemcpyAsync(h, nd.get(), sizeof(h), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                static unsigned long long ncalls = 0, nbad = 0;
                ++ncalls;
                nbad += (h[0] | h[1] | h[2]) != 0;
                if ((h[0] | h[1] | h[2]) != 0 || (ncalls & (ncalls - 1)) == 0) {
                    (h[0] | h[1] | h[2]) ? GGML_LOG_WARN("%s: MMID_PAR check call %llu (n_tokens %d n_experts %d inverse %d): differing src1 %llu dst %llu bounds %llu, bad calls %llu\n",
                            __func__, ncalls, n_tokens, n_experts, (int) write_inverse, h[0], h[1], h[2], nbad)
                        : GGML_LOG_INFO("%s: MMID_PAR check call %llu (n_tokens %d n_experts %d inverse %d): differing src1 %llu dst %llu bounds %llu, bad calls %llu\n",
                            __func__, ncalls, n_tokens, n_experts, (int) write_inverse, h[0], h[1], h[2], nbad);
                }
            }
        } else {
            ok = launch();
        }
        if (ok) {
            return;
        }
    }
    ggml_cuda_launch_mm_ids_helper_ref(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
}

static void ggml_cuda_launch_mm_ids_helper_ref(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    switch (n_expert_used) {
        case  1:
            launch_mm_ids_helper< 1>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  2:
            launch_mm_ids_helper< 2>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  4:
            launch_mm_ids_helper< 4>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  6:
            launch_mm_ids_helper< 6>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  8:
            launch_mm_ids_helper< 8>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 10:
            launch_mm_ids_helper<10>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 16:
            launch_mm_ids_helper<16>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 32:
            launch_mm_ids_helper<32>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        default:
            launch_mm_ids_helper< 0>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
    }
}
