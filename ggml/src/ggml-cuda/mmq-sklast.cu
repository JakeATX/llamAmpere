#include "mmq-sklast.cuh"
#include "mmq.cuh"

#include <cstdlib>

static bool sklast_enabled() {
    static const bool v = [] { const char * e = getenv("GGML_CUDA_MMQ_SK_LASTCTA"); return e && atoi(e) != 0; }();
    return v;
}

// Start of the K range of CTA b, exactly as in mul_mat_q / mul_mat_q_stream_k_fixup.
static __device__ __forceinline__ int sklast_kbc(const int b, const int64_t total, const uint3 blocks_per_ne00, const int blocks_per_iter) {
    int kbc = int64_t(b)*total / gridDim.x;
    kbc -= fastmodulo(kbc, blocks_per_ne00) % blocks_per_iter;
    return kbc;
}

// Called by every CTA that wrote a part of split tile T (the owner to dst, the others to tmp_fixup). The last one to
// arrive adds the partial sums of the earlier CTAs to dst in the fixup kernel's order: sum = 0; sum += tmp[O-1];
// sum += tmp[O-2] ...; dst += sum (O = the CTA that wrote the end of the tile; empty CTAs skipped).
template <int J, int I, int nthreads, bool fallback>
static __device__ __forceinline__ void sklast_arrive(
        const int T, float * __restrict__ dst, const float * __restrict__ tmp_fixup, unsigned int * __restrict__ counters,
        const int64_t total, const uint3 blocks_per_ne00, const int blocks_per_iter, const uint3 ntj,
        const int nrows_x, const int ncols_dst, const int stride_col_dst) {
    __shared__ int s_last;
    const int tid = threadIdx.y*blockDim.x + threadIdx.x;
    const int B   = blocks_per_ne00.z;
    const int tB  = T*B;
    const int tE  = tB + B;

    // CTA range [lo, hi] that wrote tile T; hi is the owner.
    int lo = blockIdx.x;
    while (sklast_kbc(lo, total, blocks_per_ne00, blocks_per_iter) > tB) {
        --lo;
    }
    int hi = blockIdx.x;
    while (sklast_kbc(hi + 1, total, blocks_per_ne00, blocks_per_iter) < tE) {
        ++hi;
    }

    // the earlier non-empty CTAs, newest first: the order of the fixup kernel's sum
    constexpr int max_prev = 32;
    __shared__ int s_prev[max_prev];
    __shared__ int s_nprev;
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        unsigned int n = 1; // the owner hi
        int np = 0;
        int kbc_stop = sklast_kbc(hi, total, blocks_per_ne00, blocks_per_iter);
        for (int b = hi - 1; b >= lo; --b) {
            const int kbc = sklast_kbc(b, total, blocks_per_ne00, blocks_per_iter);
            if (kbc != kbc_stop) {
                if (np < max_prev) {
                    s_prev[np] = b;
                }
                ++np;
                ++n;
            }
            kbc_stop = kbc;
        }
        s_nprev = np;
        const unsigned int old = atomicAdd(counters + T, 1u);
        s_last = old + 1 == n;
    }
    __syncthreads();
    if (!s_last) {
        return;
    }
    __threadfence();
    const int np = s_nprev;

    const uint2 tmp2 = fast_div_modulo(T, ntj);
    const int jt = tmp2.y;
    const int it = tmp2.x;
    const int i_max = nrows_x   - it*I - 1;
    const int j_sup = ncols_dst - jt*J;
    float * dst_tile = dst + jt*J*stride_col_dst + it*I;

    // many loads in flight per thread: the combine is on the tail of this CTA
    static_assert((J*I) % nthreads == 0, "tile not a multiple of the block");
    constexpr int per_thread = J*I/nthreads;
    constexpr int chunk = per_thread % 16 == 0 ? 16 : per_thread % 8 == 0 ? 8 : 1;
    for (int c = 0; c < per_thread; c += chunk) {
        float sum[chunk];
        bool  ok[chunk];
#pragma unroll
        for (int u = 0; u < chunk; ++u) {
            const int l = tid + (c + u)*nthreads;
            const int j = l / I;
            const int i = l % I;
            ok[u]  = j < j_sup && !(fallback && i > i_max);
            sum[u] = 0.0f;
        }
        if (np <= max_prev) {
            for (int p = 0; p < np; ++p) {
                const float * tp = tmp_fixup + int64_t(s_prev[p])*(J*I) + tid + c*nthreads;
#pragma unroll
                for (int u = 0; u < chunk; ++u) {
                    sum[u] += __ldcg(tp + u*nthreads);
                }
            }
        } else { // more earlier CTAs than the list holds: walk the ranges again
            int kbc_stop = sklast_kbc(hi, total, blocks_per_ne00, blocks_per_iter);
            for (int b = hi - 1; b >= lo; --b) {
                const int kbc = sklast_kbc(b, total, blocks_per_ne00, blocks_per_iter);
                if (kbc != kbc_stop) {
                    const float * tp = tmp_fixup + int64_t(b)*(J*I) + tid + c*nthreads;
#pragma unroll
                    for (int u = 0; u < chunk; ++u) {
                        sum[u] += __ldcg(tp + u*nthreads);
                    }
                }
                kbc_stop = kbc;
            }
        }
#pragma unroll
        for (int u = 0; u < chunk; ++u) {
            if (ok[u]) {
                const int l = tid + (c + u)*nthreads;
                float * p = dst_tile + (l / I)*stride_col_dst + l % I;
                *p = __ldcg(p) + sum[u];
            }
        }
    }
}

// Dense (one channel, one sample, no ids, no fusion) stream-K body of mul_mat_q plus the in-kernel fixup.
template <ggml_type type, int J, bool fallback>
__launch_bounds__(ggml_cuda_mmq_get_nthreads(type, J, fallback, GGML_PREC_Q8), ggml_cuda_mmq_get_occupancy(type, J, fallback, GGML_PREC_Q8))
static __global__ void mul_mat_q_sklast(
        const char * __restrict__ x, const int * __restrict__ y, float * __restrict__ dst, float * __restrict__ tmp_fixup,
        unsigned int * __restrict__ counters, const uint3 blocks_per_ne00, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int ncols_y, const int stride_col_dst, const uint3 ntj) {
    constexpr ggml_prec prec = GGML_PREC_Q8;
    if (ggml_cuda_mmq_get_config(type, J, fallback, prec).type == GGML_TYPE_COUNT) {
        NO_DEVICE_CODE;
        return;
    }
    constexpr int warp_size       = ggml_cuda_get_physical_warp_size();
    constexpr int nthreads        = ggml_cuda_mmq_get_nthreads(type, J, fallback, prec);
    constexpr int nwarps          = nthreads / warp_size;
    constexpr int qk              = ggml_cuda_type_traits<type>::qk;
    constexpr int I               = ggml_cuda_mmq_get_I(type, J, fallback, prec);
    constexpr int ITER_K          = ggml_cuda_mmq_get_K_vram(type, J, fallback, prec);
    constexpr int blocks_per_iter = ITER_K / qk;
    constexpr int sz              = sizeof(block_q8_1_mmq) / sizeof(int);

    extern __shared__ int data_mul_mat_q[]; // ids first, as in mul_mat_q
#pragma unroll
    for (int j0 = 0; j0 < J; j0 += nwarps*warp_size) {
        const int j = j0 + threadIdx.y*warp_size + threadIdx.x;
        if (j0 + nwarps*warp_size > J && j >= J) {
            break;
        }
        data_mul_mat_q[j] = j;
    }
    __syncthreads();

    const int nti = (nrows_x + I - 1) / I;
    const int B   = blocks_per_ne00.z;
    const int64_t total = int64_t(ntj.z)*nti*B;

    int kbc      = sklast_kbc(blockIdx.x,     total, blocks_per_ne00, blocks_per_iter);
    int kbc_stop = sklast_kbc(blockIdx.x + 1, total, blocks_per_ne00, blocks_per_iter);

    int kb0_start = fastmodulo(kbc, blocks_per_ne00);
    int kb0_stop  = min(B, kb0_start + kbc_stop - kbc);
    while (kbc < kbc_stop && kb0_stop == B) {
        const int T  = fastdiv(kbc, blocks_per_ne00);
        const uint2 tmp2 = fast_div_modulo(T, ntj);
        const int jt = tmp2.y;
        const int it = tmp2.x;
        const int col_low  = jt*J;
        const int col_high = min(col_low + J, ncols_dst);

        mul_mat_q_process_tile<type, J, fallback, false, prec, false>
            (x, nullptr, it*I*stride_row_x, y + col_low*sz, data_mul_mat_q, dst + jt*J*stride_col_dst + it*I, tmp_fixup,
             nullptr, GGML_GLU_OP_SWIGLU, 0.0f, nullptr, nullptr, stride_row_x, ncols_y, stride_col_dst,
             nrows_x - it*I - 1, col_high - col_low, kb0_start, kb0_stop);
        if (kb0_start != 0) { // owner of a tile whose start another CTA wrote
            sklast_arrive<J, I, nthreads, fallback>(T, dst, tmp_fixup, counters, total, blocks_per_ne00, blocks_per_iter,
                    ntj, nrows_x, ncols_dst, stride_col_dst);
        }

        kbc += B;
        kbc -= fastmodulo(kbc, blocks_per_ne00);
        kb0_start = 0;
        kb0_stop  = min(B, kbc_stop - kbc);
    }
    if (kbc >= kbc_stop) {
        return;
    }

    const int T  = fastdiv(kbc, blocks_per_ne00);
    const uint2 tmp2 = fast_div_modulo(T, ntj);
    const int jt = tmp2.y;
    const int it = tmp2.x;
    const int col_low  = jt*J;
    const int col_high = min(col_low + J, ncols_dst);

    mul_mat_q_process_tile<type, J, fallback, true, prec, false>
        (x, nullptr, it*I*stride_row_x, y + col_low*sz, data_mul_mat_q, dst + jt*J*stride_col_dst + it*I, tmp_fixup,
         nullptr, GGML_GLU_OP_SWIGLU, 0.0f, nullptr, nullptr, stride_row_x, ncols_y, stride_col_dst,
         nrows_x - it*I - 1, col_high - col_low, kb0_start, kb0_stop);
    sklast_arrive<J, I, nthreads, fallback>(T, dst, tmp_fixup, counters, total, blocks_per_ne00, blocks_per_iter,
            ntj, nrows_x, ncols_dst, stride_col_dst);
}

// Mirrors launch_mul_mat_q's stream-K grid choice; returns false where it would not use a split stream-K grid.
template <ggml_type type, int J, bool fallback>
static bool sklast_launch(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    const int id  = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[id].cc;
    const int nsm = ggml_cuda_info().devices[id].nsm;
    const int warp_size = ggml_cuda_info().devices[id].warp_size;

    const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, fallback, cc, GGML_PREC_Q8);
    if (config.type == GGML_TYPE_COUNT || !config.stream_k) {
        return false;
    }
    const int nti        = (args.nrows_x + config.I - 1) / config.I;
    const int ntiles_dst = args.ntj * nti;
    const int tiles_nwaves = (ntiles_dst + nsm - 1) / nsm;
    const int tiles_efficiency_percent = 100 * ntiles_dst / (nsm*tiles_nwaves);
    const int nwork = ntiles_dst * ((args.ncols_x + config.K_vram - 1) / config.K_vram);
    int nblocks = (GGML_CUDA_CC_IS_NVIDIA(cc) && tiles_efficiency_percent >= 90) ? ntiles_dst :
        (!ggml_cuda_mmq_sk_upstream() ? nsm : std::min(nsm, nwork));
    const int sk_max_cols = ggml_cuda_mmq_sk_48_enabled() && GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_AMPERE ? 48 : 32;
    if (const int min_iters = ggml_cuda_mmq_sk_min_iters(); min_iters > 0 && nblocks != ntiles_dst && ntiles_dst < nsm &&
            args.ncols_max > 8 && args.ncols_max <= sk_max_cols) {
        const int64_t iters_total = (int64_t) ntiles_dst * (args.ncols_x / config.K_vram);
        nblocks = (int) std::max<int64_t>(ntiles_dst, std::min<int64_t>(nblocks, iters_total / min_iters));
    }
    if (ntiles_dst % nblocks == 0) {
        return false; // no split tile: stock MMQ has no fixup either
    }
    const uint3 blocks_per_ne00_fd = init_fastdiv_values(args.ncols_x / ggml_cuda_type_traits<type>::qk);
    GGML_ASSERT(int64_t(ntiles_dst) * blocks_per_ne00_fd.z < (1 << 30));

    const int nwarps = config.nthreads / warp_size;
    const int nbytes_shared = mmq_get_nbytes_shared(config, cc);
    CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q_sklast<type, J, fallback>), nbytes_shared);

    ggml_cuda_pool & pool = ctx.pool(id);
    ggml_cuda_pool_alloc<float> tmp_fixup(pool, size_t(nblocks) * config.J*config.I);
    ggml_cuda_pool_alloc<unsigned int> counters(pool, ntiles_dst);
    CUDA_CHECK(cudaMemsetAsync(counters.ptr, 0, size_t(ntiles_dst)*sizeof(unsigned int), stream));

    mul_mat_q_sklast<type, J, fallback><<<dim3(nblocks, 1, 1), dim3(warp_size, nwarps, 1), nbytes_shared, stream>>>
        (args.x, args.y, args.dst, tmp_fixup.ptr, counters.ptr, blocks_per_ne00_fd, args.nrows_x, args.ncols_dst,
         args.stride_row_x, args.ncols_y, args.nrows_dst, init_fastdiv_values(args.ntj));
    CUDA_CHECK(cudaGetLastError());
    return true;
}

template <bool fallback>
static bool sklast_switch_J(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    switch (args.J_best) {
        case  64: return sklast_launch<GGML_TYPE_Q8_0,  64, fallback>(ctx, args, stream);
        case 112: return sklast_launch<GGML_TYPE_Q8_0, 112, fallback>(ctx, args, stream);
        case 128: return sklast_launch<GGML_TYPE_Q8_0, 128, fallback>(ctx, args, stream);
        default:  return false;
    }
}

bool ggml_cuda_mmq_sklast(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    if (!sklast_enabled() || args.type_x != GGML_TYPE_Q8_0 || args.ids_dst || args.tile_spec || args.y_scale ||
            args.fusion.gate || args.fusion.x_scale || args.fusion.x_bias || args.fusion.shared_up ||
            args.nchannels_y != 1 || args.nsamples_y != 1 || args.nchannels_x != 1 || args.nsamples_x != 1) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return false;
    }
    return ggml_cuda_mmq_needs_fallback(args.nrows_x) ? sklast_switch_J<true>(ctx, args, stream)
                                                      : sklast_switch_J<false>(ctx, args, stream);
}
