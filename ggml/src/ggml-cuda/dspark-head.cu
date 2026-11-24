#include "dspark-head.cuh"
#include "nk-dspark-head.cuh"

#include <cmath>
#include <cstdint>
#include <cstdlib>

bool ggml_cuda_dspark_head_enabled() {
    static const bool on = [] {
        const char * e = std::getenv("GGML_CUDA_DSPARK_HEAD");
        return !e || std::atoi(e) != 0;
    }();
    return on;
}

bool ggml_cuda_dspark_gru_enabled() {
    static const bool on = [] {
        const char * e = std::getenv("GGML_CUDA_DSPARK_GRU");
        return !e || std::atoi(e) != 0;
    }();
    return on;
}

// Ordered 64-bit descending sort key matching top-k.cu (topk_split_key):
// upper 32 bits = order-preserving float encoding (-0.0f folded to +0.0f),
// lower 32 bits = (ncols - 1 - col) so ties break to the lowest column index.
static __device__ __forceinline__ uint64_t dspark_topk_key(const float v, const int col, const int ncols) {
    uint32_t b = __float_as_uint(v);
    b = (b == 0x80000000u) ? 0u : b;
    b = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
    return ((uint64_t) b << 32) | (uint32_t) (ncols - 1 - col);
}

static __device__ __forceinline__ float dspark_key_to_float(const uint64_t key) {
    uint32_t b = (uint32_t) (key >> 32);
    b = (b & 0x80000000u) ? (b & 0x7fffffffu) : ~b;
    return __uint_as_float(b);
}

// Descending bitonic sort of N uint64_t keys in shared memory by BLOCK threads.
template <int N, int BLOCK>
static __device__ __forceinline__ void dspark_bitonic_sort_desc(uint64_t * __restrict__ s) {
    for (int size = 2; size <= N; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            __syncthreads();
#pragma unroll
            for (int t = threadIdx.x; t < N / 2; t += BLOCK) {
                const int i = 2 * t - (t & (stride - 1));
                const int j = i + stride;
                const uint64_t a = s[i];
                const uint64_t b = s[j];
                if ((a < b) == ((i & size) == 0)) {
                    s[i] = b;
                    s[j] = a;
                }
            }
        }
    }
    __syncthreads();
}

// ============================================================================
// 1. Capture-safe F16 1-row-expert GEMV (for shortlist MUL_MAT_ID fallback)
//    src0: [K, 1, N_full] F16 (K multiple of 8, 16-byte aligned rows)
//    src1: [K, 1, nb]     F32
//    ids:  [n_sel, nb]    I32 (may be broadcast with nb1 == 0)
//    dst:  [1, n_sel, nb] F32
// ============================================================================
static __global__ void __launch_bounds__(256) k_dspark_mul_mat_id_f16_rows(
        const half  * __restrict__ w_base,
        const float * __restrict__ x_base,
        const int   * __restrict__ ids_base,
        float       * __restrict__ dst_base,
        const int    K,
        const int    n_sel,
        const int    nb,
        const size_t x_stride_elems,
        const size_t ids_stride_elems,
        const size_t dst_stride_elems) {
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;
    const int sel_idx = (blockIdx.x * 8) + warp_id;
    const int b       = blockIdx.y;
    if (sel_idx >= n_sel || b >= nb) {
        return;
    }

    const int row_id = ids_base[(size_t) b * ids_stride_elems + sel_idx];
    const half  * w_row = w_base + (size_t) row_id * K;
    const float * x_row = x_base + (size_t) b * x_stride_elems;

    float acc = 0.0f;
    for (int k = lane * 8; k < K; k += 256) {
        const uint4  w4 = *reinterpret_cast<const uint4 *>(w_row + k);
        const half2 * h2 = reinterpret_cast<const half2 *>(&w4);
        const float4 x0 = *reinterpret_cast<const float4 *>(x_row + k);
        const float4 x1 = *reinterpret_cast<const float4 *>(x_row + k + 4);
        const float2 f0 = __half22float2(h2[0]);
        const float2 f1 = __half22float2(h2[1]);
        const float2 f2 = __half22float2(h2[2]);
        const float2 f3 = __half22float2(h2[3]);
        acc += f0.x * x0.x + f0.y * x0.y
             + f1.x * x0.z + f1.y * x0.w
             + f2.x * x1.x + f2.y * x1.y
             + f3.x * x1.z + f3.y * x1.w;
    }

    acc += __shfl_xor_sync(0xffffffffu, acc, 16);
    acc += __shfl_xor_sync(0xffffffffu, acc, 8);
    acc += __shfl_xor_sync(0xffffffffu, acc, 4);
    acc += __shfl_xor_sync(0xffffffffu, acc, 2);
    acc += __shfl_xor_sync(0xffffffffu, acc, 1);

    if (lane == 0) {
        dst_base[(size_t) b * dst_stride_elems + sel_idx] = acc;
    }
}

bool ggml_cuda_dspark_f16_rows_supported(const ggml_tensor * dst) {
    if (!dst || dst->op != GGML_OP_MUL_MAT_ID) {
        return false;
    }
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];
    if (!src0 || !src1 || !ids) {
        return false;
    }
    if (src0->type != GGML_TYPE_F16 || src1->type != GGML_TYPE_F32 ||
            ids->type != GGML_TYPE_I32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (src0->ne[1] != 1 || src0->ne[3] != 1 || src1->ne[1] != 1 || src1->ne[3] != 1 ||
            dst->ne[0] != 1 || dst->ne[3] != 1 || ids->ne[2] != 1 || ids->ne[3] != 1) {
        return false;
    }
    if ((src0->ne[0] % 8) != 0 || src0->ne[0] <= 0 || src0->ne[0] > INT_MAX ||
            dst->ne[1] <= 0 || dst->ne[1] > INT_MAX || dst->ne[2] <= 0 || dst->ne[2] > 64) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }
    // ids may be contiguous or broadcast along dim 1 (nb[1] == 0)
    if (ids->nb[0] != sizeof(int32_t) || (ids->nb[1] != 0 && ids->nb[1] % sizeof(int32_t) != 0)) {
        return false;
    }
    return true;
}

void ggml_cuda_dspark_mul_mat_id_f16_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    const int K     = (int) src0->ne[0];
    const int n_sel = (int) dst->ne[1];
    const int nb    = (int) dst->ne[2];
    const size_t x_stride   = src1->nb[2] / sizeof(float);
    const size_t ids_stride = ids->nb[1]  / sizeof(int32_t);
    const size_t dst_stride = dst->nb[2]  / sizeof(float);

    const dim3 grid((n_sel + 7) / 8, nb, 1);
    k_dspark_mul_mat_id_f16_rows<<<grid, 256, 0, ctx.stream()>>>(
        (const half *)  src0->data,
        (const float *) src1->data,
        (const int *)   ids->data,
        (float *)       dst->data,
        K, n_sel, nb, x_stride, ids_stride, dst_stride);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// 2. Fused Markov GRU state update:
//    gate      = sigmoid(joint_raw[0..R-1]     + joint_b[0..R-1])
//    candidate = tanh   (joint_raw[R..2R-1]    + joint_b[R..2R-1])
//    state_out = candidate + gate * (state_in - candidate)
//    latent    = tanh   (joint_raw[2R..3R-1]   + joint_b[2R..3R-1])
// ============================================================================
static __global__ void __launch_bounds__(256) k_dspark_gru_update(
        const float * __restrict__ joint_raw,
        const float * __restrict__ joint_b,
        const float * __restrict__ state_in,
        float       * __restrict__ state_out,
        float       * __restrict__ latent_out,
        const int    rank,
        const size_t joint_stride,
        const size_t sin_stride,
        const size_t sout_stride,
        const size_t lat_stride) {
    const int b = blockIdx.x;
    const float * jr = joint_raw  + (size_t) b * joint_stride;
    const float * si = state_in   + (size_t) b * sin_stride;
    float       * so = state_out  + (size_t) b * sout_stride;
    float       * lo = latent_out + (size_t) b * lat_stride;

    for (int i = threadIdx.x; i < rank; i += blockDim.x) {
        const float g_in = jr[i]            + joint_b[i];
        const float c_in = jr[i + rank]     + joint_b[i + rank];
        const float l_in = jr[i + 2 * rank] + joint_b[i + 2 * rank];

        const float gate = 1.0f / (1.0f + expf(-g_in));
        const float cand = tanhf(c_in);
        const float st   = si[i];
        so[i] = cand + gate * (st - cand);
        lo[i] = tanhf(l_in);
    }
}

void ggml_cuda_dspark_gru_update(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * joint_raw,
        const ggml_tensor * joint_b,
        const ggml_tensor * state_in,
        ggml_tensor       * state_out,
        ggml_tensor       * latent_out) {
    const int rank = (int) state_out->ne[0];
    const int nb   = (int) state_out->ne[1];
    const size_t j_stride    = joint_raw->nb[1]  / sizeof(float);
    const size_t sin_stride  = state_in->nb[1]   / sizeof(float);
    const size_t sout_stride = state_out->nb[1]  / sizeof(float);
    const size_t lat_stride  = latent_out->nb[1] / sizeof(float);

    k_dspark_gru_update<<<nb, 256, 0, ctx.stream()>>>(
        (const float *) joint_raw->data,
        (const float *) joint_b->data,
        (const float *) state_in->data,
        (float *)       state_out->data,
        (float *)       latent_out->data,
        rank, j_stride, sin_stride, sout_stride, lat_stride);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// 3. Fused Folded Markov Head (K=256 F16) + Bias + Base + Softcap + Tile Top-K / Argmax
//    Each CTA (256 threads = 8 warps) caches latent[256] in 1 KB shared memory,
//    then processes TILE rows (1 row per warp via ONE coalesced LDG.E.128 per lane),
//    fuses fold_bias + base + softcap(cap) in registers, and emits per-CTA top-K keys
//    (or 1 max key for greedy) into scratch `cand`.
// ============================================================================
template <int TILE, bool HAS_IDS>
static __global__ void __launch_bounds__(256) k_dspark_fold_head_tile_topk(
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
    __shared__ float    s_x[256];
    __shared__ uint64_t s_keys[TILE];
    __shared__ uint64_t s_warp_best[8];

    const int tile    = blockIdx.x;
    const int b       = blockIdx.y;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    // Cache latent[256] once per CTA (256 threads -> 1 float each)
    s_x[tid] = latent[(size_t) b * lat_stride + tid];
    __syncthreads();

    // Load this lane's 8 floats of latent into registers once across all TILE/8 row iterations!
    const float4 x0 = *reinterpret_cast<const float4 *>(s_x + lane * 8);
    const float4 x1 = *reinterpret_cast<const float4 *>(s_x + lane * 8 + 4);

    const float * base_b = base + (size_t) b * base_stride;
    const int row_beg = tile * TILE;

#pragma unroll 1
    for (int r = warp_id; r < TILE; r += 8) {
        const int local_idx = row_beg + r;
        if (local_idx < n_eval) {
            const int w_idx = HAS_IDS ? ids[local_idx] : local_idx;
            const half * w_row = fold_w + (size_t) w_idx * 256;
            // 32 lanes * 8 halves = 256 halves = 512 bytes: single coalesced LDG.E.128 per lane
            const uint4   w4 = *reinterpret_cast<const uint4 *>(w_row + lane * 8);
            const half2 * h2 = reinterpret_cast<const half2 *>(&w4);
            const float2  f0 = __half22float2(h2[0]);
            const float2  f1 = __half22float2(h2[1]);
            const float2  f2 = __half22float2(h2[2]);
            const float2  f3 = __half22float2(h2[3]);

            float acc = f0.x * x0.x + f0.y * x0.y
                      + f1.x * x0.z + f1.y * x0.w
                      + f2.x * x1.x + f2.y * x1.y
                      + f3.x * x1.z + f3.y * x1.w;

            acc += __shfl_xor_sync(0xffffffffu, acc, 16);
            acc += __shfl_xor_sync(0xffffffffu, acc, 8);
            acc += __shfl_xor_sync(0xffffffffu, acc, 4);
            acc += __shfl_xor_sync(0xffffffffu, acc, 2);
            acc += __shfl_xor_sync(0xffffffffu, acc, 1);

            if (lane == 0) {
                const float v = cap * tanhf((acc + fold_bias[local_idx] + base_b[local_idx]) * inv_cap);
                s_keys[r] = dspark_topk_key(v, local_idx, n_eval);
            }
        } else if (lane == 0) {
            s_keys[r] = 0ull;
        }
    }

    if (k == 1) {
        // Greedy fast path: block-wide max reduction of TILE keys into separate s_warp_best[8] (no SMEM race!)
        __syncthreads();
        uint64_t best = 0ull;
#pragma unroll
        for (int i = tid; i < TILE; i += 256) {
            const uint64_t v = s_keys[i];
            if (v > best) {
                best = v;
            }
        }
        // Warp reduction
        for (int off = 16; off > 0; off >>= 1) {
            const uint64_t other = __shfl_xor_sync(0xffffffffu, best, off);
            if (other > best) {
                best = other;
            }
        }
        if (lane == 0) {
            s_warp_best[warp_id] = best;
        }
        __syncthreads();
        if (tid == 0) {
            uint64_t gbest = s_warp_best[0];
#pragma unroll
            for (int w = 1; w < 8; ++w) {
                if (s_warp_best[w] > gbest) {
                    gbest = s_warp_best[w];
                }
            }
            cand_out[(size_t) b * ntiles + tile] = gbest;
        }
        return;
    }

    // Sampled path (k > 1): sort TILE keys descending in shared memory and emit top k
    dspark_bitonic_sort_desc<TILE, 256>(s_keys);
    uint64_t * out = cand_out + ((size_t) b * ntiles + tile) * k;
    for (int i = tid; i < k; i += 256) {
        out[i] = s_keys[i];
    }
}

// Stage 2A: Greedy merge (1 CTA per sequence, reduces `ntiles` max keys -> prev_out[b])
template <bool HAS_IDS>
static __global__ void __launch_bounds__(256) k_dspark_head_merge_argmax(
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

// Stage 2B: Sampled merge + optional shortlist remap + Gumbel-max selection
// Writes:
//   cand_out[b, 0..k-1] = real token ids (descending logit order)
//   vals_out[b, 0..k-1] = softcapped logits (decoded directly from key, zero global re-read!)
//   prev_out[b]         = cand_out[b, argmax_j(vals[j] * inv_temp + noise[b, j])]
template <int N_MERGE, bool HAS_IDS>
static __global__ void __launch_bounds__(512) k_dspark_head_merge_sample(
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
    __shared__ uint64_t s[N_MERGE];
    __shared__ float    s_score[64];
    __shared__ int      s_tok[64];

    const int b   = blockIdx.x;
    const int tid = threadIdx.x;

    const uint64_t * c = cand_in + (size_t) b * ncand;
#pragma unroll
    for (int i = tid; i < N_MERGE; i += 512) {
        s[i] = i < ncand ? c[i] : 0ull;
    }
    dspark_bitonic_sort_desc<N_MERGE, 512>(s);

    if (tid < k) {
        const uint64_t key       = s[tid];
        const int      local_idx = n_eval - 1 - (int) (uint32_t) key;
        const int      tok_id    = HAS_IDS ? ids[local_idx] : local_idx;
        const float    val       = dspark_key_to_float(key);
        const float    g         = noise[(size_t) b * noise_stride + tid];

        cand_out[(size_t) b * cand_stride + tid] = tok_id;
        vals_out[(size_t) b * vals_stride + tid] = val;
        s_tok[tid]   = tok_id;
        s_score[tid] = val * inv_temp + g;
    }
    __syncthreads();

    // Exact argmax over k <= 64 (lowest index wins ties, matching ggml_argmax)
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

bool ggml_cuda_dspark_head_launch_base(
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
        float               inv_temp) {
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
    }
    if (!sampled) {
        sk = 1;
    }

    // Choose adaptive tile size so ntiles * sk <= 4096 (single-CTA merge) AND high SM occupancy
    int tile_sz = 1024;
    if (n_eval <= 16384 && ((n_eval + 255) / 256) * sk <= 2048) {
        tile_sz = 256;
    } else if (n_eval <= 65536 && ((n_eval + 511) / 512) * sk <= 2048) {
        tile_sz = 512;
    } else if (((n_eval + 1023) / 1024) * sk <= 4096) {
        tile_sz = 1024;
    } else if (((n_eval + 4095) / 4096) * sk <= 4096) {
        tile_sz = 4096;
    } else {
        return false;
    }

    const int ntiles = (int) ((n_eval + tile_sz - 1) / tile_sz);
    const int ncand  = ntiles * sk;
    const float inv_cap = 1.0f / softcap;
    const size_t lat_stride  = latent->nb[1] / sizeof(float);
    const size_t base_stride = base->nb[1]   / sizeof(float);
    const bool has_ids = ids != nullptr;
    const int * ids_d  = has_ids ? (const int *) ids->data : nullptr;

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<uint64_t> cand_buf(ctx.pool(), (size_t) nb * ncand);

    const dim3 grid1(ntiles, nb, 1);
    auto launch_stage1 = [&](auto tile_tag, auto ids_tag) {
        constexpr int  T = decltype(tile_tag)::value;
        constexpr bool I = decltype(ids_tag)::value;
        k_dspark_fold_head_tile_topk<T, I><<<grid1, 256, 0, stream>>>(
            (const half *)  fold_w->data,
            (const float *) latent->data,
            (const float *) fold_bias->data,
            (const float *) base->data,
            ids_d,
            cand_buf.get(),
            (int) n_eval, ntiles, sk, softcap, inv_cap, lat_stride, base_stride);
    };

    auto dispatch_stage1 = [&](auto ids_tag) {
        switch (tile_sz) {
            case 256:  launch_stage1(std::integral_constant<int, 256>{},  ids_tag); break;
            case 512:  launch_stage1(std::integral_constant<int, 512>{},  ids_tag); break;
            case 1024: launch_stage1(std::integral_constant<int, 1024>{}, ids_tag); break;
            case 4096: launch_stage1(std::integral_constant<int, 4096>{}, ids_tag); break;
            default:   GGML_ABORT("invalid dspark tile_sz");
        }
    };

    if (has_ids) {
        dispatch_stage1(std::true_type{});
    } else {
        dispatch_stage1(std::false_type{});
    }

    if (!sampled) {
        if (has_ids) {
            k_dspark_head_merge_argmax<true><<<nb, 256, 0, stream>>>(
                cand_buf.get(), ids_d, (int *) prev_out->data, ntiles, (int) n_eval);
        } else {
            k_dspark_head_merge_argmax<false><<<nb, 256, 0, stream>>>(
                cand_buf.get(), nullptr, (int *) prev_out->data, ntiles, (int) n_eval);
        }
    } else {
        const size_t noise_stride = noise->nb[1]    / sizeof(float);
        const size_t cand_stride  = cand_out->nb[1] / sizeof(int32_t);
        const size_t vals_stride  = vals_out->nb[1] / sizeof(float);

        auto launch_stage2 = [&](auto nmerge_tag, auto ids_tag) {
            constexpr int  NM = decltype(nmerge_tag)::value;
            constexpr bool I  = decltype(ids_tag)::value;
            k_dspark_head_merge_sample<NM, I><<<nb, 512, 0, stream>>>(
                cand_buf.get(), ids_d,
                (const float *) noise->data,
                (int *)         cand_out->data,
                (float *)       vals_out->data,
                (int *)         prev_out->data,
                ncand, (int) n_eval, sk, inv_temp,
                noise_stride, cand_stride, vals_stride);
        };

        auto dispatch_stage2 = [&](auto ids_tag) {
            if (ncand <= 512) {
                launch_stage2(std::integral_constant<int, 512>{},  ids_tag);
            } else if (ncand <= 1024) {
                launch_stage2(std::integral_constant<int, 1024>{}, ids_tag);
            } else if (ncand <= 2048) {
                launch_stage2(std::integral_constant<int, 2048>{}, ids_tag);
            } else {
                launch_stage2(std::integral_constant<int, 4096>{}, ids_tag);
            }
        };

        if (has_ids) {
            dispatch_stage2(std::true_type{});
        } else {
            dispatch_stage2(std::false_type{});
        }
    }

    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool ggml_cuda_dspark_head_launch(
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
        float               inv_temp) {
    if (ggml_cuda_nk_dspark_head_enabled() &&
            ggml_cuda_nk_dspark_head_launch(ctx, fold_w, latent, fold_bias, base, ids, noise,
                cand_out, vals_out, prev_out, n_eval, sk, softcap, inv_temp)) {
        return true;
    }
    return ggml_cuda_dspark_head_launch_base(ctx, fold_w, latent, fold_bias, base, ids, noise,
        cand_out, vals_out, prev_out, n_eval, sk, softcap, inv_temp);
}
