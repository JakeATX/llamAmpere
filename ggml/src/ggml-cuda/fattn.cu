#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-mma-turbo.cuh"
#include "fattn-tile.cuh"
#include "fattn-vec.cuh"
#include "fattn.cuh"
#include "ledger.cuh"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>

// Flash-attention path census, opt-in via GGML_FATTN_PATH_STATS=1.
// Counts every ggml_cuda_flash_attn_ext dispatch keyed by (path, K type, V type,
// n_q = Q->ne[1], gqa_ratio, ncols2 when the MMA path packs GQA). Dumped to stderr at
// exit. Purpose (W5): prove or kill the premise that an f16 draft K/V cache routes the
// drafter's width-1 attention onto the GQA-packed MMA path instead of VEC.
static bool ggml_cuda_fattn_path_stats_enabled() {
    static const bool enabled = [] { const char * e = getenv("GGML_FATTN_PATH_STATS"); return e && e[0] == '1'; }();
    return enabled;
}

static void ggml_cuda_fattn_path_note(const char * path, const ggml_tensor * dst, int ncols2) {
    {
        // fallback ledger (GGML_LEDGER=1): kernel family per K/V pair and query width
        const ggml_tensor * Q = dst->src[0];
        const ggml_tensor * K = dst->src[1];
        const ggml_tensor * V = dst->src[2];
        const uint64_t id = ggml_cuda_ledger_mix(ggml_cuda_ledger_mix(ggml_cuda_ledger_mix(ggml_cuda_ledger_mix(
            (uint64_t) (uintptr_t) path, (uint64_t) K->type), (uint64_t) V->type), (uint64_t) Q->ne[0]),
            (uint64_t) ggml_cuda_ledger_width_bucket(Q->ne[1]) * 64 + (uint64_t) (ncols2 + 1));
        ggml_cuda_ledger_count("cuda.fattn", id, [&](char * buf, size_t size) {
            char w[16];
            ggml_cuda_ledger_width_str(Q->ne[1], w, sizeof(w));
            snprintf(buf, size, "path=%s K=%s V=%s D=%d n_q=%s ncols2=%d", path, ggml_type_name(K->type), ggml_type_name(V->type),
                (int) Q->ne[0], w, ncols2);
        });
    }
    if (!ggml_cuda_fattn_path_stats_enabled()) {
        return;
    }
    static std::mutex mtx;
    static std::map<std::string, uint64_t> counts;
    static const bool registered = [] {
        atexit([] {
            std::lock_guard<std::mutex> lock(mtx);
            fprintf(stderr, "fattn_path_stats: begin (%zu keys)\n", counts.size());
            for (const auto & kv : counts) {
                fprintf(stderr, "fattn_path_stats: %s count=%llu\n", kv.first.c_str(), (unsigned long long) kv.second);
            }
            fprintf(stderr, "fattn_path_stats: end\n");
        });
        return true;
    }();
    GGML_UNUSED(registered);
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    char key[256];
    snprintf(key, sizeof(key), "path=%s K=%s V=%s D=%d n_q=%d gqa=%d ncols2=%d kv_len_bucket=%dk",
        path, ggml_type_name(K->type), ggml_type_name(V->type), (int) Q->ne[0], (int) Q->ne[1],
        (int) (Q->ne[2] / K->ne[2]), ncols2, (int) (K->ne[1] / 1024));
    std::lock_guard<std::mutex> lock(mtx);
    counts[key]++;
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
__launch_bounds__(256, 1)
static __global__ void flash_attn_mask_to_sparse_indices(
        const half * mask_ptr, int32_t * indices_ptr, const int ne30, const int n_kv_max,
        const int64_t s31, const int64_t s33) {
    ggml_cuda_pdl_sync();

    constexpr int values_per_lane = 8;
    const int tid      = threadIdx.x;
    const int warp     = tid / WARP_SIZE;
    const int lane     = tid % WARP_SIZE;
    const int sequence = blockIdx.y;
    const int query    = blockIdx.x;

    const half * mask = mask_ptr + sequence*s33 + query*s31;
    int32_t * indices = indices_ptr + (int64_t(sequence)*gridDim.x + query)*n_kv_max;

    __shared__ int warp_offsets[256/WARP_SIZE];
    __shared__ int row_count;
    __shared__ int chunk_count;

    if (tid == 0) {
        row_count = 0;
    }
    __syncthreads();

    for (int i0 = 0; i0 < ne30; i0 += blockDim.x*values_per_lane) {
        uint32_t selected_warp[values_per_lane];
        int warp_count = 0;
#pragma unroll
        for (int item = 0; item < values_per_lane; ++item) {
            const int i = i0 + (warp*values_per_lane + item)*WARP_SIZE + lane;
            const bool selected = i < ne30 && isfinite(__half2float(mask[i]));
            selected_warp[item] = __ballot_sync(0xFFFFFFFF, selected);
            warp_count += __popc(selected_warp[item]);
        }

        if (lane == 0) {
            warp_offsets[warp] = warp_count;
        }
        __syncthreads();

        if (tid == 0) {
            int offset = 0;
#pragma unroll
            for (int iw = 0; iw < 256/WARP_SIZE; ++iw) {
                const int count = warp_offsets[iw];
                warp_offsets[iw] = offset;
                offset += count;
            }
            chunk_count = offset;
        }
        __syncthreads();

        const uint32_t lane_mask = lane == 0 ? 0 : (1u << lane) - 1;
        int warp_item_offset = 0;
#pragma unroll
        for (int item = 0; item < values_per_lane; ++item) {
            const int i = i0 + (warp*values_per_lane + item)*WARP_SIZE + lane;
            const int dst = row_count + warp_offsets[warp] + warp_item_offset + __popc(selected_warp[item] & lane_mask);
            if ((selected_warp[item] & (uint32_t(1) << lane)) && dst < n_kv_max) {
                indices[dst] = i;
            }
            warp_item_offset += __popc(selected_warp[item]);
        }
        __syncthreads();

        if (tid == 0) {
            row_count += chunk_count;
        }
        __syncthreads();
    }

    const int count = row_count;
    for (int i = count + tid; i < n_kv_max; i += blockDim.x) {
        indices[i] = -1;
    }
    __syncthreads();

    // the dependent grid reads indices, signal once the row is complete
    ggml_cuda_pdl_lc();
}
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

void ggml_cuda_flash_attn_ext_compact_mask(
        const ggml_tensor * mask, int32_t * indices, int32_t n_kv_max, cudaStream_t stream) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(mask, indices, n_kv_max, stream);
    GGML_ABORT("sparse flash attention is only supported on NVIDIA CUDA");
#else
    const int64_t s31 = mask->nb[1] / sizeof(half);
    const int64_t s33 = mask->nb[3] / sizeof(half);
    const dim3 blocks_num(mask->ne[1], mask->ne[3], 1);
    const dim3 block_dim(256, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params(blocks_num, block_dim, 0, stream);
    ggml_cuda_kernel_launch(flash_attn_mask_to_sparse_indices, launch_params,
        (const half *) mask->data, indices, int(mask->ne[0]), n_kv_max, s31, s33);
    CUDA_CHECK(cudaGetLastError());
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}

bool ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(ctx, dst);
    return false;
#else
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * mask = dst->src[3];
    const int cc = ggml_cuda_info().devices[ctx.device].cc;

    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    const int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);
    return GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc) &&
        mask != nullptr && n_kv_max > 0 && max_bias == 0.0f && logit_softcap == 0.0f &&
        mask->ne[0] == K->ne[1] && mask->ne[1] >= Q->ne[1] && mask->ne[2] == 1 &&
        K->ne[1] >= std::max<int64_t>(4096, 2LL*n_kv_max);
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}

template <int DKQ, int DV, int ncols2>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * Q = dst->src[0];

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if constexpr (ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(DKQ, DV, 1, ncols2)) {
        if (ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(ctx, dst)) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 1, ncols2>(ctx, dst);
            return;
        }
    }
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

    if constexpr (ncols2 <= 8) {
        if (turing_mma_available(cc) && Q->ne[1] <= 8/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 8/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if constexpr (ncols2 <= 16) {
        if (Q->ne[1] <= 16/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 16/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if (Q->ne[1] <= 32/ncols2 || (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING) ||
            (GGML_CUDA_CC_IS_AMD(cc) && DKQ > 256)) {
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 32/ncols2, ncols2>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 64/ncols2, ncols2>(ctx, dst);
}

template <int DKQ, int DV>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // Edge cases like no mask, ALiBi, unpadded K/V, or misaligned addresses for large data transfers
    //     are put into the template specialization without GQA optimizations. Quantized tensors
    //     (incl. turbo2/3/4) are skipped here: their loaders dequantize into SMEM via the
    //     swizzled/padded tile helpers rather than reading nb[] directly, so the 16-byte-stride
    //     alignment this loop checks for doesn't apply to them.
    bool use_gqa_opt = mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                use_gqa_opt = false;
                break;
            }
        }
    }

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    // On Volta the GQA optimizations aren't as impactful vs. minimizing wasted compute:
    if (cc == GGML_CUDA_CC_VOLTA) {
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            ggml_cuda_fattn_path_note("mma_f16", dst, 8);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            ggml_cuda_fattn_path_note("mma_f16", dst, 4);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
            return;
        }

        if constexpr (DKQ <= 256) {
            if (use_gqa_opt && gqa_ratio % 2 == 0) {
                ggml_cuda_fattn_path_note("mma_f16", dst, 2);
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
                return;
            }

            ggml_cuda_fattn_path_note("mma_f16", dst, 1);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
            return;
        } else {
            GGML_ABORT("fatal error");
        }
    }

    // On RDNA it is preferable to minimize wasted compute vs. duplicate I/O for the mask.
    if (amd_wmma_available(cc)) {
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 2 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
            return;
        }
    }

    if (use_gqa_opt && gqa_ratio > 4) {
        ggml_cuda_fattn_path_note("mma_f16", dst, 8);
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 2) {
        ggml_cuda_fattn_path_note("mma_f16", dst, 4);
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 1) {
        ggml_cuda_fattn_path_note("mma_f16", dst, 2);
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
        return;
    }

    if constexpr (DKQ <= 256) {
        ggml_cuda_fattn_path_note("mma_f16", dst, 1);
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
    } else {
        GGML_ABORT("fatal error");
    }
}

// ---------------------------------------------------------------------------
// turbo4 fused MMA decode dispatch (mirrors the f16 switch helpers, type-parametric).
// Only reached from the gate for turbo4 K==V, D in {128,256}, Q->ne[1] <= 4, turing MMA.
//
// The reachable (ncols1, ncols2) set for Q->ne[1] in {1..4} with GQA-packing is exactly
// {(1,8),(2,8),(4,8),(2,4),(4,4),(4,2),(8,1)} — the 7 compiled instances per D. Each ncols2
// has an explicit dispatcher so ONLY those pairs are instantiated (an unguarded ncols1=8/ncols2
// fallthrough would also instantiate uncompiled cases like (8,4) -> link error).

template <int DKQ, int DV, ggml_type type_K, ggml_type type_V>
static void ggml_cuda_flash_attn_ext_mma_turbo_dispatch_ncols1_8(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0]; // ncols2 == 8: (1,8),(2,8),(4,8)
    // GGML_Q8_TURBO3_MMA_NCOLS1_MIN pads single queries into the (2,8) instance. Default 2 since P5b:
    // the (1,8) instance runs 84 blocks at 4% occupancy (883 us at 100K under ncu) while the padded
    // (2,8) route runs 252 blocks (390 vs 700 us/launch in test-backend-ops perf). Set =1 to disable.
    static const int ncols1_min = [] { const char * e = getenv("GGML_Q8_TURBO3_MMA_NCOLS1_MIN"); const int v = e ? atoi(e) : 2; return (v == 1 || v == 2 || v == 4) ? v : 2; }();
    if (Q->ne[1] <= 1 && ncols1_min == 1) { ggml_cuda_flash_attn_ext_mma_turbo_case<DKQ, DV, 1, 8, type_K, type_V>(ctx, dst); return; }
    if (Q->ne[1] <= 2 && ncols1_min <= 2) { ggml_cuda_flash_attn_ext_mma_turbo_case<DKQ, DV, 2, 8, type_K, type_V>(ctx, dst); return; }
    if constexpr (DKQ == 256 && DV == 256 && ((type_K == GGML_TYPE_Q8_0 && (type_V == GGML_TYPE_TURBO3_0 || type_V == GGML_TYPE_Q8_0)) ||
                                             (type_K == GGML_TYPE_TQ6_0 && type_V == GGML_TYPE_TURBO3_0) ||
                                             (type_K == GGML_TYPE_TQ5_0 && type_V == GGML_TYPE_TURBO3_0) ||
                                             ((type_K == GGML_TYPE_TQ5_0 || type_K == GGML_TYPE_TQ6_0 || type_K == GGML_TYPE_Q8_0) && type_V == GGML_TYPE_TURBO4_0) ||
                                             (type_K == GGML_TYPE_TQ6_0 && type_V == GGML_TYPE_TQ5_0))) {
        if (Q->ne[1] > 4) { ggml_cuda_flash_attn_ext_mma_turbo_case<DKQ, DV, 8, 8, type_K, type_V>(ctx, dst); return; }
    }
    ggml_cuda_flash_attn_ext_mma_turbo_case<DKQ, DV, 4, 8, type_K, type_V>(ctx, dst); // Q->ne[1] in {3,4}
}
template <int DKQ, int DV, ggml_type type_K, ggml_type type_V>
static void ggml_cuda_flash_attn_ext_mma_turbo_dispatch_ncols1_4(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0]; // ncols2 == 4: (2,4),(4,4)
    if (Q->ne[1] <= 2) { ggml_cuda_flash_attn_ext_mma_turbo_case<DKQ, DV, 2, 4, type_K, type_V>(ctx, dst); return; }
    ggml_cuda_flash_attn_ext_mma_turbo_case<DKQ, DV, 4, 4, type_K, type_V>(ctx, dst); // Q->ne[1] in {3,4}
}

template <int DKQ, int DV, ggml_type type_K, ggml_type type_V>
static void ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // Mirror the f16 use_gqa_opt computation. Quantized tensors are skipped in the nb%16 loop.
    bool use_gqa_opt = mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                use_gqa_opt = false;
                break;
            }
        }
    }

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    if (use_gqa_opt && gqa_ratio > 4) {                                  // ncols2 = 8
        ggml_cuda_flash_attn_ext_mma_turbo_dispatch_ncols1_8<DKQ, DV, type_K, type_V>(ctx, dst);
        return;
    }
    if (use_gqa_opt && gqa_ratio > 2) {                                  // ncols2 = 4
        ggml_cuda_flash_attn_ext_mma_turbo_dispatch_ncols1_4<DKQ, DV, type_K, type_V>(ctx, dst);
        return;
    }
    if (use_gqa_opt && gqa_ratio > 1) {                                  // ncols2 = 2 -> (4,2)
        ggml_cuda_flash_attn_ext_mma_turbo_case<DKQ, DV, 4, 2, type_K, type_V>(ctx, dst);
        return;
    }
    ggml_cuda_flash_attn_ext_mma_turbo_case<DKQ, DV, 8, 1, type_K, type_V>(ctx, dst); // ncols2 = 1 -> (8,1)
}

// Env latch for the fused turbo MMA decode path. DEFAULT ON.
//
// The MMA path is correctness-validated (coherent output, KLD == VEC baseline 0.008396)
// and faster than VEC at every depth (beats rival "buun"), BUT it is NOT bit/token-identical
// to the VEC reference: MMA and VEC accumulate the P·V (VKQ) reduction in f16 with different
// reduction trees (tensor-core fragment order vs per-thread VEC order), so a near-tie greedy
// token can flip (~1 in ~25 tokens on a hard tie). This is the same irreducible f16-order
// difference that exists between the base f16-MMA and f16-VEC kernels — not a regression — but
// it fails strict token-identity. GGML_TURBO_MMA_FUSED=0 is the VEC kill-switch for anyone who
// needs that identity guarantee back.
static bool ggml_cuda_turbo_mma_fused() {
    static const bool v = []{
        const char * s = getenv("GGML_TURBO_MMA_FUSED");
        return !(s && s[0] == '0');  // default ON (faster GQA-packed MMA, quality-neutral); GGML_TURBO_MMA_FUSED=0 = VEC kill-switch
    }();
    return v;
}

// Fused Q8_0-K / TURBO3-V MMA path. Default ON since P5b (2026-09-09): D7/D7b census and the
// P5a temp-1.0 ABBA showed it beats the vector and generic-MMA routes at every verify width
// 1..5 for D=256. Set GGML_Q8_TURBO3_MMA_FUSED=0 to fall back to the pre-P5b routing.
static bool ggml_cuda_q8_turbo3_mma_fused() {
    static const bool value = [] {
        const char * env = getenv("GGML_Q8_TURBO3_MMA_FUSED");
        return env == nullptr || env[0] != '0';
    }();
    return value;
}

// smallest query width routed to the q8_0/turbo3 MMA path. Default 1 since P5b: width-1
// fused MMA is 797 us vs 1418 us for the vector kernel at 100K (D7b), width-2 434 us vs the
// generic MMA + f16 temporaries. GGML_Q8_TURBO3_MMA_MIN_Q=3 restores the pre-P5b routing.
static int ggml_cuda_q8_turbo3_mma_min_q() {
    static const int value = [] {
        const char * env = getenv("GGML_Q8_TURBO3_MMA_MIN_Q");
        const int v = env ? atoi(env) : 1;
        return (v >= 1 && v <= 5) ? v : 1;
    }();
    return value;
}

// largest query width routed to the fused q8_0-K MMA paths. Default 8: the (8,8) instance is a full eight-row
// tile, so widths 6..8 (MTP depth 5..7, n-gram drafts up to 7, DFlash block_size 8 verify) use it instead of
// falling to the generic mma_f16 dequant path (kv=100352 width 8: 1434 -> 429 us per layer, widths 1..5 unchanged,
// test-backend-ops 1158/1158). GGML_Q8_TURBO3_MMA_MAX_Q=5 restores the previous routing (valid 5..8).
static int ggml_cuda_q8_turbo3_mma_max_q() {
    static const int value = [] {
        const char * env = getenv("GGML_Q8_TURBO3_MMA_MAX_Q");
        const int v = env ? atoi(env) : 8;
        return (v >= 5 && v <= 8) ? v : 8;
    }();
    return value;
}

static void ggml_cuda_flash_attn_ext_mma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    switch (Q->ne[0]) {
        case 64:
            GGML_ASSERT(V->ne[0] == 64);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 64,  64>(ctx, dst);
            break;
        case 80:
            GGML_ASSERT(V->ne[0] == 80);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 80,  80>(ctx, dst);
            break;
        case 96:
            GGML_ASSERT(V->ne[0] == 96);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 96,  96>(ctx, dst);
            break;
        case 112:
            GGML_ASSERT(V->ne[0] == 112);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<112, 112>(ctx, dst);
            break;
        case 128:
            GGML_ASSERT(V->ne[0] == 128);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<128, 128>(ctx, dst);
            break;
        case 192: {
            // MiMo-V2.5 / V2.5-Pro / V2-Flash: gqa_ratio is 8 (SWA) or 16 (full attn)
            GGML_ASSERT(V->ne[0] == 128);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));
            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);
            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128, 16>(ctx, dst);
            } else {
                GGML_ASSERT(gqa_ratio % 8 == 0);
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128,  8>(ctx, dst);
            }
        } break;
        case 256:
            GGML_ASSERT(V->ne[0] == 256);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<256, 256>(ctx, dst);
            break;
        case 320:
            // For Mistral Small 4, go straight to the ncols1 switch (ncols2=32-only build).
            GGML_ASSERT(V->ne[0] == 256);
            {
                float max_bias = 0.0f;
                memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

                const bool use_gqa_opt = mask && max_bias == 0.0f;
                GGML_ASSERT(use_gqa_opt);
                GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
                const int gqa_ratio = Q->ne[2] / K->ne[2];
                GGML_ASSERT(gqa_ratio % 32 == 0);

                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<320, 256, 32>(ctx, dst);
            }
            break;
        case 512:
            GGML_ASSERT(V->ne[0] == 512);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<512, 512>(ctx, dst);
            break;
        case 576: {
            // For Deepseek, go straight to the ncols1 switch to avoid compiling unnecessary kernels.
            GGML_ASSERT(V->ne[0] == 512);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);

            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio == 20) { // GLM 4.7 Flash
                if (cc >= GGML_CUDA_CC_DGX_SPARK) {
                    if (Q->ne[1] <= 8) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_BLACKWELL) {
                    if (Q->ne[1] <= 4 && K->ne[1] >= 65536) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 4) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_TURING) {
                    if (Q->ne[1] <= 4) {
                        if (K->ne[1] <= 16384) {
                            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                            break;
                        }
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 32>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                // Volta:
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
            } else if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512,  4>(ctx, dst);
            }
        } break;
        case 640: {
            // Padded turbo KV cache for GLM-4.7 Flash (K head_dim=576 zero-padded to 640).
            // D=640 shared memory (Q storage = ncols*(DKQ/2+4)*4) exceeds hardware limit at ncols1>=4.
            // Cap at ncols1=2 (ncols=32): Q=32*324*4=41KB + KV≈37KB = ~78KB total.
            GGML_ASSERT(V->ne[0] == 512);
            if (Q->ne[1] <= 1) {
                ggml_cuda_flash_attn_ext_mma_f16_case<640, 512, 1, 16>(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_mma_f16_case<640, 512, 2, 16>(ctx, dst);
            }
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

#define FATTN_VEC_CASE(D, type_K_case, type_V_case)                                                                                \
    if constexpr (GGML_CUDA_FA_##type_K_case##_##type_V_case) {                                                                    \
        const bool type_K_okay = type_K == GGML_TYPE_##type_K_case || (type_K == GGML_TYPE_F32 && GGML_TYPE_##type_K_case == GGML_TYPE_F16); \
        const bool type_V_okay = type_V == GGML_TYPE_##type_V_case || (type_V == GGML_TYPE_F32 && GGML_TYPE_##type_V_case == GGML_TYPE_F16); \
        if (head_size == (D) && type_K_okay && type_V_okay) {                                                                      \
            return ggml_cuda_flash_attn_ext_vec_case<D, GGML_TYPE_##type_K_case, GGML_TYPE_##type_V_case>;                         \
        }                                                                                                                          \
    }                                                                                                                              \

#define FATTN_VEC_CASES_ALL_D(type_K_case, type_V_case) \
    FATTN_VEC_CASE( 64, type_K_case, type_V_case)       \
    FATTN_VEC_CASE(128, type_K_case, type_V_case)       \
    FATTN_VEC_CASE(256, type_K_case, type_V_case)       \

typedef void (* fattn_vec_case_t)(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Vector kernel for the given head size and K/V types, nullptr if its template instance was not compiled:
static fattn_vec_case_t ggml_cuda_get_fattn_vec_case(const int64_t head_size, const ggml_type type_K, const ggml_type type_V) {
    FATTN_VEC_CASES_ALL_D(F16,  F16)
    FATTN_VEC_CASES_ALL_D(Q4_0, F16)
    FATTN_VEC_CASES_ALL_D(Q4_1, F16)
    FATTN_VEC_CASES_ALL_D(Q5_0, F16)
    FATTN_VEC_CASES_ALL_D(Q5_1, F16)
    FATTN_VEC_CASES_ALL_D(Q8_0, F16)
    FATTN_VEC_CASES_ALL_D(BF16, F16)

    FATTN_VEC_CASES_ALL_D(F16,  Q4_0)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q4_0)
    FATTN_VEC_CASES_ALL_D(BF16, Q4_0)

    FATTN_VEC_CASES_ALL_D(F16,  Q4_1)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q4_1)
    FATTN_VEC_CASES_ALL_D(BF16, Q4_1)

    FATTN_VEC_CASES_ALL_D(F16,  Q5_0)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q5_0)
    FATTN_VEC_CASES_ALL_D(BF16, Q5_0)

    FATTN_VEC_CASES_ALL_D(F16,  Q5_1)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q5_1)
    FATTN_VEC_CASES_ALL_D(BF16, Q5_1)

    FATTN_VEC_CASES_ALL_D(F16,  Q8_0)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q8_0)
    FATTN_VEC_CASES_ALL_D(BF16, Q8_0)

    FATTN_VEC_CASES_ALL_D(F16,  BF16)
    FATTN_VEC_CASES_ALL_D(Q4_0, BF16)
    FATTN_VEC_CASES_ALL_D(Q4_1, BF16)
    FATTN_VEC_CASES_ALL_D(Q5_0, BF16)
    FATTN_VEC_CASES_ALL_D(Q5_1, BF16)
    FATTN_VEC_CASES_ALL_D(Q8_0, BF16)
    FATTN_VEC_CASES_ALL_D(BF16, BF16)

    // TurboQuant KV cache types (fork-only, always compiled - see ggml_cuda_fattn_vec_instances)
    FATTN_VEC_CASES_ALL_D(TURBO3_0, TURBO3_0)
    FATTN_VEC_CASES_ALL_D(TURBO3_0, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q8_0,     TURBO3_0)
    FATTN_VEC_CASES_ALL_D(F16,      TURBO3_0)
    FATTN_VEC_CASES_ALL_D(TURBO3_0, F16)

    FATTN_VEC_CASES_ALL_D(TURBO2_0, TURBO2_0)
    FATTN_VEC_CASES_ALL_D(TURBO2_0, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q8_0,     TURBO2_0)
    FATTN_VEC_CASES_ALL_D(F16,      TURBO2_0)
    FATTN_VEC_CASES_ALL_D(TURBO2_0, F16)
    FATTN_VEC_CASES_ALL_D(TURBO3_0, TURBO2_0)
    FATTN_VEC_CASES_ALL_D(TURBO2_0, TURBO3_0)

    FATTN_VEC_CASES_ALL_D(TURBO4_0, TURBO4_0)
    FATTN_VEC_CASES_ALL_D(TURBO4_0, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q8_0,     TURBO4_0)
    FATTN_VEC_CASES_ALL_D(F16,      TURBO4_0)
    FATTN_VEC_CASES_ALL_D(TURBO4_0, F16)
    FATTN_VEC_CASES_ALL_D(TURBO4_0, TURBO3_0)
    FATTN_VEC_CASES_ALL_D(TURBO3_0, TURBO4_0)
    FATTN_VEC_CASES_ALL_D(TURBO4_0, TURBO2_0)
    FATTN_VEC_CASES_ALL_D(TURBO2_0, TURBO4_0)

    // tq6 blocks hold 128 values: head dim 64 has no instance (see the cmake FA_TQ6_COMBINATIONS list)
    FATTN_VEC_CASE(128, TQ6_0,    TQ6_0)
    FATTN_VEC_CASE(256, TQ6_0,    TQ6_0)
    FATTN_VEC_CASE(128, TQ6_0,    TURBO3_0)
    FATTN_VEC_CASE(256, TQ6_0,    TURBO3_0)
    FATTN_VEC_CASE(128, TQ6_0,    Q8_0)
    FATTN_VEC_CASE(256, TQ6_0,    Q8_0)
    FATTN_VEC_CASE(128, Q8_0,     TQ6_0)
    FATTN_VEC_CASE(256, Q8_0,     TQ6_0)
    FATTN_VEC_CASE(128, TQ6_0,    F16)
    FATTN_VEC_CASE(256, TQ6_0,    F16)
    FATTN_VEC_CASE(128, F16,      TQ6_0)
    FATTN_VEC_CASE(256, F16,      TQ6_0)

    // tq5 shares the tq6 geometry (128 values per block)
    FATTN_VEC_CASE(128, TQ5_0,    TQ5_0)
    FATTN_VEC_CASE(256, TQ5_0,    TQ5_0)
    FATTN_VEC_CASE(128, TQ5_0,    TURBO3_0)
    FATTN_VEC_CASE(256, TQ5_0,    TURBO3_0)
    FATTN_VEC_CASE(128, TQ5_0,    Q8_0)
    FATTN_VEC_CASE(256, TQ5_0,    Q8_0)
    FATTN_VEC_CASE(128, Q8_0,     TQ5_0)
    FATTN_VEC_CASE(256, Q8_0,     TQ5_0)
    FATTN_VEC_CASE(128, TQ5_0,    F16)
    FATTN_VEC_CASE(256, TQ5_0,    F16)
    FATTN_VEC_CASE(128, F16,      TQ5_0)
    FATTN_VEC_CASE(256, F16,      TQ5_0)

    return nullptr;
}

static void ggml_cuda_flash_attn_ext_vec(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    fattn_vec_case_t vec_case = ggml_cuda_get_fattn_vec_case(Q->ne[0], K->type, V->type);
    if (vec_case == nullptr) {
        static bool warned = false;
        if (!warned) {
            GGML_LOG_WARN("%s: no FlashAttention vector kernel compiled for K/V types %s-%s, converting K and V to f16 instead (slow). "
                "Add \"%s-%s\" to GGML_CUDA_FA_QUANTS to compile it.\n",
                __func__, ggml_type_name(K->type), ggml_type_name(V->type), ggml_type_name(K->type), ggml_type_name(V->type));
            warned = true;
        }
        vec_case = ggml_cuda_get_fattn_vec_case(Q->ne[0], GGML_TYPE_F16, GGML_TYPE_F16);
    }
    GGML_ASSERT(vec_case != nullptr);
    vec_case(ctx, dst);
}

// Best FlashAttention kernel for a specific GPU:
enum best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE    =   0,
    BEST_FATTN_KERNEL_TILE    = 200,
    BEST_FATTN_KERNEL_VEC     = 100,
    BEST_FATTN_KERNEL_MMA_F16 = 400,
};

// K/V types for which there is a vector kernel template instance, other kernels convert these to f16:
static bool ggml_cuda_fattn_kv_type_supported(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
            return true;
        case GGML_TYPE_TURBO2_0:
        case GGML_TYPE_TURBO3_0:
        case GGML_TYPE_TURBO4_0:
            // turbo KV types; head-dim geometry is validated separately in
            // ggml_cuda_get_best_fattn_kernel (multiples of 64 only)
            return true;
        case GGML_TYPE_TQ6_0:
        case GGML_TYPE_TQ5_0:
            // tq6/tq5 KV types; head dim must be a multiple of 128 (block size), checked below
            return true;
        default:
            return false;
    }
}

static best_fattn_kernel ggml_cuda_get_best_fattn_kernel(const int device, const ggml_tensor * dst) {
#ifndef FLASH_ATTN_AVAILABLE
    GGML_UNUSED(device); GGML_UNUSED(dst);
    return BEST_FATTN_KERNEL_NONE;
#endif// FLASH_ATTN_AVAILABLE

    const ggml_tensor * KQV   = dst;
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];

    const int gqa_ratio = Q->ne[2] / K->ne[2];
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // The effective batch size for the kernel can be increased by gqa_ratio.
    // The kernel versions without this optimization are also used for ALiBi, if there is no mask, or if the KV cache is not padded,
    bool gqa_opt_applies = gqa_ratio >= 2 && mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                gqa_opt_applies = false;
                break;
            }
        }
    }

    const int cc = ggml_cuda_info().devices[device].cc;

    switch (K->ne[0]) {
        case  40:
        case  64:
        case  72:
        case  80:
        case  96:
        case 128:
        case 112:
        case 256:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 192:
            if (V->ne[0] != 128 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 8 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 320:
            if (V->ne[0] != 256 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 32 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 512:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 576:
        case 640:
            if (V->ne[0] != 512) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

    if (!ggml_cuda_fattn_kv_type_supported(K->type) || !ggml_cuda_fattn_kv_type_supported(V->type)) {
        return BEST_FATTN_KERNEL_NONE;
    }

    // turbo VEC/MMA kernels are instantiated for head dims that are multiples of 64
    {
        auto is_turbo = [](ggml_type t) {
            return t == GGML_TYPE_TURBO2_0 || t == GGML_TYPE_TURBO3_0 || t == GGML_TYPE_TURBO4_0;
        };
        if ((is_turbo(K->type) && K->ne[0] % 64 != 0) ||
            (is_turbo(V->type) && V->ne[0] % 64 != 0)) {
            return BEST_FATTN_KERNEL_NONE;
        }
        // tq6 packs 128 values per block and is only instantiated for head dims 128 and 256
        if (((K->type == GGML_TYPE_TQ6_0 || K->type == GGML_TYPE_TQ5_0) && K->ne[0] % 128 != 0) ||
            ((V->type == GGML_TYPE_TQ6_0 || V->type == GGML_TYPE_TQ5_0) && V->ne[0] % 128 != 0)) {
            return BEST_FATTN_KERNEL_NONE;
        }
    }

    if (mask && mask->ne[2] != 1) {
        return BEST_FATTN_KERNEL_NONE;
    }

    // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes:
    // 192 satisfies % 64 == 0 but has no vec instance (DKQ != DV); force it onto the MMA path.
    const bool can_use_vector_kernel = Q->ne[0] <= 256 && Q->ne[0] % 64 == 0 && Q->ne[0] != 192 && K->ne[1] % FATTN_KQ_STRIDE == 0;

#ifdef GGML_USE_HIP
    // HIP/ROCm: the TILE/MMA/WMMA FA paths allocate large f16 temp buffers for
    // quantized KV types (K_f16, V_f16 in launch_fattn). For SMALL batches (decode)
    // the VEC kernel is preferred: it does inline dequant with zero temp buffer
    // overhead, it natively supports the TurboQuant types, and it produces a
    // HIP-graph-safe op stream (no per-call cudaMalloc/cudaFree during capture).
    // For LARGE batches (prefill) the VEC kernel is far slower (sequential query
    // processing), so we deliberately fall through to the TILE/MMA path which is
    // ~3.4x faster; prefill runs eagerly (not captured) so its f16 temp buffer is
    // allocated/freed raw in launch_fattn without violating graph-capture rules.
    // Limitation: head_dim > 256 cannot use VEC (falls through to TILE).
    if ((ggml_is_quantized(K->type) || ggml_is_quantized(V->type)) && can_use_vector_kernel && Q->ne[1] <= 8) {
        return BEST_FATTN_KERNEL_VEC;
    }
#endif // GGML_USE_HIP

    // If Turing tensor cores are available, use them:
    if (turing_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel) {
            if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE && Q->ne[1] == 1 && Q->ne[3] == 1 && !(gqa_ratio > 4 && K->ne[1] >= 8192)) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            } else {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 2) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                } else {
                    if (Q->ne[1] == 1) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                }
            }
            if (!gqa_opt_applies && Q->ne[1] == 1) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    const int ncols2_max = Q->ne[0] == 320 ? 32 : ((Q->ne[0] == 576 || Q->ne[0] == 640 || Q->ne[0] == 192) ? 16 : 8);
    int gqa_ratio_eff = 1;
    while (gqa_ratio % (2*gqa_ratio_eff) == 0 && gqa_ratio_eff < ncols2_max) {
        gqa_ratio_eff *= 2;
    }

    if (volta_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel && Q->ne[1] * gqa_ratio_eff <= 2) {
            return BEST_FATTN_KERNEL_VEC;
        }
        if (Q->ne[1] * gqa_ratio_eff <= 16) {
            return BEST_FATTN_KERNEL_TILE; // On Volta tensor cores are only faster for sufficiently large matrices.
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // TQ: RDNA4 fast path for TurboQuant cache types — prefer VEC for quantized K/V at small q-cols
    if (amd_wmma_available(cc) && GGML_CUDA_CC_IS_RDNA4(cc) && gqa_opt_applies && Q->ne[0] <= 128 && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel) {
            if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
                if (Q->ne[1] == 1) {
                    if (!gqa_opt_applies) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                }
            } else {
                if (Q->ne[1] <= 2) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
        }
        int gqa_ratio_eff_rdna4 = 1;
        const int ncols2_max_rdna4 = (Q->ne[0] == 576 || Q->ne[0] == 640) ? 16 : 8;
        while (gqa_ratio % (2*gqa_ratio_eff_rdna4) == 0 && gqa_ratio_eff_rdna4 < ncols2_max_rdna4) {
            gqa_ratio_eff_rdna4 *= 2;
        }
        if (Q->ne[1] * gqa_ratio_eff_rdna4 <= 8) {
            return BEST_FATTN_KERNEL_TILE;
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // AMD MFMA needs a certain minimum batch size to outscale the tile kernel for large head sizes.
    if ((amd_mfma_available(cc) && Q->ne[0] <= 256) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if ((Q->ne[0] <= 64 && Q->ne[1] * gqa_ratio_eff > 8)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 128 && Q->ne[1] * gqa_ratio_eff > 16)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 256 && Q->ne[1] * gqa_ratio_eff > 64)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
    }

    // AMD WMMA is faster than the tile kernel if the wide tiles with high arithmetic intensity can be utilized.
    if ((amd_wmma_available(cc) && gqa_opt_applies && Q->ne[0] <= 256) && Q->ne[0] != 40 && Q->ne[0] != 72 &&
            Q->ne[1] * gqa_ratio_eff > (Q->ne[0] <= 128 ? 8 : 16)) {
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // If there are no tensor cores available, use the generic tile kernel:
    if (can_use_vector_kernel) {
        if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
            if (Q->ne[1] == 1) {
                if (!gqa_opt_applies) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
        } else {
            if (Q->ne[1] <= 2) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
    }
    return BEST_FATTN_KERNEL_TILE;
}

// The fused packed-KV routes (q8_0 K over turbo3 / q8_0 V, and the turbo / tq gate) run before the generic selector
// and read K/V in place: launch_fattn gets need_f16_K = need_f16_V = false. This one function decides the route for
// execution (ctx != nullptr: launch) and for ggml_cuda_flash_attn_ext_get_alloc_size (ctx == nullptr: only report),
// so the scratch reserved behind dst follows the kernel that runs (#64). Returns true when a fused route applies.
static bool ggml_cuda_flash_attn_ext_fused(ggml_backend_cuda_context * ctx, ggml_tensor * dst, const int device) {
#define FATTN_FUSED_NOTE(...) do { if (ctx != nullptr) { ggml_cuda_fattn_path_note(__VA_ARGS__); } } while (0)
#define FATTN_FUSED_LAUNCH(DKQ_, DV_, TK_, TV_) \
    do { if (ctx != nullptr) { ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols2<DKQ_, DV_, TK_, TV_>(*ctx, dst); } return true; } while (0)

    // Qwen3.8 verification fast path: Q8 K and Turbo3 V are decoded directly
    // into the Stream-K MMA tile. This removes full-cache FP16 conversion and
    // shares each compressed tile across the packed MTP query rows.
    {
        const ggml_tensor * Q = dst->src[0];
        const ggml_tensor * K = dst->src[1];
        const ggml_tensor * V = dst->src[2];
        const int cc = ggml_cuda_info().devices[device].cc;
        if (ggml_cuda_q8_turbo3_mma_fused() && K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_TURBO3_0 &&
                Q->ne[0] == 256 && V->ne[0] == 256 && Q->ne[1] >= ggml_cuda_q8_turbo3_mma_min_q() && Q->ne[1] <= ggml_cuda_q8_turbo3_mma_max_q() && turing_mma_available(cc)) {
            FATTN_FUSED_NOTE("q8_turbo3_fused", dst, -1);
            FATTN_FUSED_LAUNCH(256, 256, GGML_TYPE_Q8_0, GGML_TYPE_TURBO3_0);
        }
        // Same fused path for a q8_0 K / q8_0 V cache (e.g. the MTP draft cache): identical staging and
        // K decode, V decoded with the q8_0 tile loader instead of turbo3. Same routing knobs.
        if (ggml_cuda_q8_turbo3_mma_fused() && K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q8_0 &&
                Q->ne[0] == 256 && V->ne[0] == 256 && Q->ne[1] >= ggml_cuda_q8_turbo3_mma_min_q() && Q->ne[1] <= ggml_cuda_q8_turbo3_mma_max_q() && turing_mma_available(cc)) {
            FATTN_FUSED_NOTE("q8_q8_fused", dst, -1);
            FATTN_FUSED_LAUNCH(256, 256, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0);
        }
    }

    // Fused turbo MMA decode gate (DEFAULT ON, see ggml_cuda_turbo_mma_fused; GGML_TURBO_MMA_FUSED=0 disables).
    // Routes turbo2/3/4 K==V, D in {128,256}, decode (Q->ne[1] <= 4) onto the GQA-packed
    // MMA path (KV read once per head-group instead of per query head). Q is ALREADY
    // graph-rotated (src/llama-graph.cpp) and the FA output is inverse-rotated there, so this
    // path does NO inline FWHT and NO src swap. GGML_TURBO_MMA_FUSED=0 falls straight through
    // to the original VEC dispatch (kill-switch).
    {
        const ggml_tensor * Q = dst->src[0];
        const ggml_tensor * K = dst->src[1];
        const ggml_tensor * V = dst->src[2];
        const int cc = ggml_cuda_info().devices[device].cc;
        const bool turbo_matched = (K->type == V->type &&
            (K->type == GGML_TYPE_TURBO4_0 || K->type == GGML_TYPE_TURBO3_0 || K->type == GGML_TYPE_TURBO2_0 ||
             K->type == GGML_TYPE_TQ6_0 || K->type == GGML_TYPE_TQ5_0)) ||
            // asymmetric tq6/tq5 K over a turbo3 V: the pair they are meant for, K and V decoded by
            // their own tile loaders into the same GQA-packed tile.
            ((K->type == GGML_TYPE_TQ6_0 || K->type == GGML_TYPE_TQ5_0) && V->type == GGML_TYPE_TURBO3_0) ||
            // turbo4 V under a q8_0/tq6_0/tq5_0 K (D=256 only): unstaged turbo4 V tile loader, K loader as above
            ((K->type == GGML_TYPE_TQ6_0 || K->type == GGML_TYPE_TQ5_0 || K->type == GGML_TYPE_Q8_0) && V->type == GGML_TYPE_TURBO4_0 && Q->ne[0] == 256) ||
            // tq6_0 K over a tq5_0 V (D=256 only): both tiles staged, same loaders as the matched tq6/tq5 pairs
            (K->type == GGML_TYPE_TQ6_0 && V->type == GGML_TYPE_TQ5_0 && Q->ne[0] == 256);
        // the tq6_0/tq5_0 K / turbo3_0 V pairs at D=256 have an (8,8) instance, so MTP verify widths 5..8 stay fused
        const int turbo_max_q = (((K->type == GGML_TYPE_TQ6_0 || K->type == GGML_TYPE_TQ5_0 || K->type == GGML_TYPE_Q8_0) &&
                                  (V->type == GGML_TYPE_TURBO3_0 || V->type == GGML_TYPE_TURBO4_0) && Q->ne[0] == 256) ||
                                 (K->type == GGML_TYPE_TQ6_0 && V->type == GGML_TYPE_TQ5_0 && Q->ne[0] == 256)) ? 8 : 4;
        if (ggml_cuda_turbo_mma_fused() && turbo_matched
                && Q->ne[1] <= turbo_max_q && V->ne[0] == Q->ne[0] && turing_mma_available(cc)) {
            FATTN_FUSED_NOTE("turbo_fused_gate", dst, -1);
            if (K->type == GGML_TYPE_TQ6_0 && V->type == GGML_TYPE_TURBO3_0) {
                if (Q->ne[0] == 128) { FATTN_FUSED_LAUNCH(128, 128, GGML_TYPE_TQ6_0, GGML_TYPE_TURBO3_0); }
                if (Q->ne[0] == 256) { FATTN_FUSED_LAUNCH(256, 256, GGML_TYPE_TQ6_0, GGML_TYPE_TURBO3_0); }
            }
            if (V->type == GGML_TYPE_TURBO4_0 && Q->ne[0] == 256) {
                if (K->type == GGML_TYPE_TQ5_0) { FATTN_FUSED_LAUNCH(256, 256, GGML_TYPE_TQ5_0, GGML_TYPE_TURBO4_0); }
                if (K->type == GGML_TYPE_TQ6_0) { FATTN_FUSED_LAUNCH(256, 256, GGML_TYPE_TQ6_0, GGML_TYPE_TURBO4_0); }
                if (K->type == GGML_TYPE_Q8_0)  { FATTN_FUSED_LAUNCH(256, 256, GGML_TYPE_Q8_0,  GGML_TYPE_TURBO4_0); }
            }
            if (K->type == GGML_TYPE_TQ6_0 && V->type == GGML_TYPE_TQ5_0 && Q->ne[0] == 256) {
                FATTN_FUSED_LAUNCH(256, 256, GGML_TYPE_TQ6_0, GGML_TYPE_TQ5_0);
            }
            if (K->type == GGML_TYPE_TQ5_0 && V->type == GGML_TYPE_TURBO3_0) {
                if (Q->ne[0] == 128) { FATTN_FUSED_LAUNCH(128, 128, GGML_TYPE_TQ5_0, GGML_TYPE_TURBO3_0); }
                if (Q->ne[0] == 256) { FATTN_FUSED_LAUNCH(256, 256, GGML_TYPE_TQ5_0, GGML_TYPE_TURBO3_0); }
            }
            if (Q->ne[0] == 128) {
                switch (K->type) {
                    case GGML_TYPE_TURBO4_0: FATTN_FUSED_LAUNCH(128, 128, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0);
                    case GGML_TYPE_TURBO3_0: FATTN_FUSED_LAUNCH(128, 128, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0);
                    case GGML_TYPE_TURBO2_0: FATTN_FUSED_LAUNCH(128, 128, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0);
                    case GGML_TYPE_TQ6_0:    FATTN_FUSED_LAUNCH(128, 128, GGML_TYPE_TQ6_0,    GGML_TYPE_TQ6_0);
                    case GGML_TYPE_TQ5_0:    FATTN_FUSED_LAUNCH(128, 128, GGML_TYPE_TQ5_0,    GGML_TYPE_TQ5_0);
                    default: break;
                }
            }
            if (Q->ne[0] == 256) {
                switch (K->type) {
                    case GGML_TYPE_TURBO4_0: FATTN_FUSED_LAUNCH(256, 256, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0);
                    case GGML_TYPE_TURBO3_0: FATTN_FUSED_LAUNCH(256, 256, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0);
                    case GGML_TYPE_TQ6_0:    FATTN_FUSED_LAUNCH(256, 256, GGML_TYPE_TQ6_0,    GGML_TYPE_TQ6_0);
                    case GGML_TYPE_TQ5_0:    FATTN_FUSED_LAUNCH(256, 256, GGML_TYPE_TQ5_0,    GGML_TYPE_TQ5_0);
                    // turbo2 + head_dim 256: intentionally NO fused case (routes to VEC via
                    // default below). At 2-bit KV the fused path's GQA-pack saving is tiny while the
                    // dequant/no-pipeline overhead is unchanged, so it is neutral on high-BW GPUs and
                    // regresses ~1-2.5% on bandwidth-limited ones (tester @everson: Gemma-12B / RTX
                    // 5060 Ti). VEC == baseline there. turbo2 + hd128 keeps fused (a +6.6..+69% depth
                    // win on dense models); turbo3/turbo4 stay fused at both head dims.
                    default: break;
                }
            }
        }
    }


    return false;
#undef FATTN_FUSED_LAUNCH
#undef FATTN_FUSED_NOTE
}

static const char * ggml_cuda_fattn_kernel_name(const best_fattn_kernel kernel) {
    switch (kernel) {
        case BEST_FATTN_KERNEL_TILE:    return "tile";
        case BEST_FATTN_KERNEL_VEC:     return "vec";
        case BEST_FATTN_KERNEL_MMA_F16: return "mma_f16";
        case BEST_FATTN_KERNEL_NONE:    break;
    }
    return "none";
}

// GGML_CUDA_FATTN_ALLOC_ROUTE=0 restores the generic-selector sizing: f16 K+V copies for TILE/MMA (and VEC without a
// type instance) even when a fused route runs and never touches them. Default 1 (#64).
static bool ggml_cuda_fattn_alloc_route() {
    static const bool v = [] { const char * e = getenv("GGML_CUDA_FATTN_ALLOC_ROUTE"); return !(e && e[0] == '0'); }();
    return v;
}

// GGML_CUDA_FATTN_ALLOC_LOG=1: one line per new (K type, V type, n_q, log2 n_kv, route) with the scratch reserved
// behind dst and what the generic selector would reserve, to size the gap at 32K / 100K (#64).
static bool ggml_cuda_fattn_alloc_log() {
    static const bool v = [] { const char * e = getenv("GGML_CUDA_FATTN_ALLOC_LOG"); return e && e[0] == '1'; }();
    return v;
}

// ---------------------------------------------------------------------------------------------------------------------
// [#22/#29] Bounded f16 prefill. GGML_CUDA_PREFILL_KV_MIB=<MiB> (1..16384); unset, empty or 0 = off, and off runs none
// of the code below (the plan returns before reading the tensors), so sizing and launch are exactly the #64 tree's.
//
// The generic MMA route converts the whole quantized K and V cache to f16 before the kernel (launch_fattn,
// fattn-common.cuh), and ggml_cuda_flash_attn_ext_get_alloc_size reserves both copies behind dst in the compute buffer:
// 2 * n_kv * n_head_kv * D * 2 bytes, 1 GiB for 4 KV heads at D 256 and n_kv 262,144, reserved at load for the
// worst-case prefill graph. The bounded plan converts one group of whole KV heads at a time into a workspace of
// heads * (f16 K + f16 V + that group's f32 output) and runs the ordinary f16 MMA kernel per group, then scatters the
// group's output rows into dst. The attention math per head is the f16 MMA kernel's; only the grid (fewer heads per
// launch) differs, which can change the stream-k partition and so the f32 reduction order at some shapes.
//
// #29: the workspace never drops below one complete GQA head group (one KV head's f16 K+V plus the outputs of its gqa
// query heads). When the budget is below that floor at the current n_kv, the floor is used (WARN once, with sizes);
// the full-copy path stays only for shapes the plan excludes. Token-axis stripes are a separate item.
//
// One function, ggml_cuda_fattn_bounded_prefill_plan, decides eligibility and sizes for the allocator
// (ggml_cuda_flash_attn_ext_get_alloc_size), the executor (ggml_cuda_flash_attn_ext) and the CUDA graph compatibility
// check (ggml_cuda_flash_attn_ext_bounded_prefill_applies, called from ggml-cuda.cu), so they cannot disagree.
struct ggml_cuda_fattn_bounded_plan {
    int    heads      = 0;     // KV heads per group; 0 = the plan does not apply
    int    n_head_kv  = 0;
    int    gqa        = 0;     // query heads per KV head
    size_t kv_bytes   = 0;     // f16 bytes of one group's K (and, separately, V): heads * n_kv * D * 2
    size_t out_bytes  = 0;     // f32 bytes of one group's output: heads * gqa * n_q * D * 4
    size_t offset     = 0;     // workspace offset behind dst->data: ggml_nbytes(dst) padded to 128
    size_t workspace  = 0;     // 2 * kv_bytes + out_bytes
    size_t reserve    = 0;     // bytes reserved behind dst: max(workspace, budget), see below
    size_t budget     = 0;     // GGML_CUDA_PREFILL_KV_MIB in bytes
    size_t floor_bytes = 0;     // one KV head's f16 K+V plus its GQA group's output at this n_kv
    size_t full       = 0;     // the full f16 K+V copies the unbounded route reserves
    bool   floor_used = false; // budget < floor: one head per group
};

static size_t ggml_cuda_fattn_prefill_budget() {
    static const size_t budget = [] {
        const char * s = getenv("GGML_CUDA_PREFILL_KV_MIB");
        if (s == nullptr || *s == '\0') {
            return size_t(0);
        }
        size_t n = 0;
        for (const char * c = s; *c; ++c) {
            if (*c < '0' || *c > '9' || n > 16384) {
                GGML_LOG_WARN("fattn bounded prefill: GGML_CUDA_PREFILL_KV_MIB=\"%s\" is not an integer in 0..16384; "
                              "bounded prefill stays off\n", s);
                return size_t(0);
            }
            n = n*10 + size_t(*c - '0');
        }
        if (n > 16384) {
            GGML_LOG_WARN("fattn bounded prefill: GGML_CUDA_PREFILL_KV_MIB=%zu is above 16384; bounded prefill stays off\n", n);
            return size_t(0);
        }
        return n << 20;
    }();
    return budget;
}

static ggml_cuda_fattn_bounded_plan ggml_cuda_fattn_bounded_prefill_plan(const int device, const ggml_tensor * dst) {
    ggml_cuda_fattn_bounded_plan p;

    const size_t budget = ggml_cuda_fattn_prefill_budget();
    if (budget == 0) {
        return p;
    }
    GGML_ASSERT(dst->op == GGML_OP_FLASH_ATTN_EXT);

    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    if (Q == nullptr || K == nullptr || V == nullptr) {
        return p;
    }
    // any further source (hybrid / KVarN descriptors and the like) is not an ordinary quantized cache: excluded
    for (int i = 5; i < GGML_MAX_SRC; ++i) {
        if (dst->src[i] != nullptr) {
            return p;
        }
    }

    // Ampere only (the measured target; Ada and newer and every AMD device keep the #64 route)
    const int cc = ggml_cuda_info().devices[device].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_AMPERE || cc >= GGML_CUDA_CC_ADA_LOVELACE) {
        return p;
    }

    // D 256 only, batch 1, widths above 8: decode and MTP / n-gram verify widths 1-8 never take the plan (they keep
    // their routes and their CUDA graphs); every wider batch that reaches the f16 MMA kernel does, so that no runtime
    // shape smaller than the reserved worst case falls back to the full copy (see `reserve`)
    if (Q->type != GGML_TYPE_F32 || Q->nb[0] != sizeof(float) || Q->ne[0] != 256 || K->ne[0] != 256 || V->ne[0] != 256) {
        return p;
    }
    if (Q->ne[1] <= 8 || Q->ne[1] > 65536 || Q->ne[2] < 2 || Q->ne[2] > 256 || Q->ne[3] != 1) {
        return p;
    }
    if (K->ne[1] < 1 || K->ne[1] > (int64_t(1) << 24) || K->ne[1] != V->ne[1]) {
        return p;
    }
    // at least two KV heads (one head cannot be split), matching K/V head counts, integral GQA
    if (K->ne[2] < 2 || K->ne[2] != V->ne[2] || K->ne[2] > Q->ne[2] || Q->ne[2] % K->ne[2] != 0 ||
            Q->ne[2] / K->ne[2] > 32 || K->ne[3] != 1 || V->ne[3] != 1) {
        return p;
    }
    // contiguous f32 output [D, n_head, n_q, 1]: the per-group scatter below writes whole D-rows into it
    if (dst->type != GGML_TYPE_F32 || dst->ne[0] != 256 || dst->ne[1] != Q->ne[2] || dst->ne[2] != Q->ne[1] ||
            dst->ne[3] != 1 || !ggml_is_contiguous(dst)) {
        return p;
    }
    // shared K/V views (MLA) are excluded: the full path converts that tensor once for both
    if (K == V || V->view_src == K || K->view_src == V ||
            (V->view_src != nullptr && V->view_src == K->view_src && V->view_offs == K->view_offs)) {
        return p;
    }
    // ordinary quantized cache types with a strided f16 converter (ggml_get_to_fp16_nc_cuda), block-contiguous rows
    for (const ggml_tensor * t : {K, V}) {
        switch (t->type) {
            case GGML_TYPE_Q8_0:
            case GGML_TYPE_TURBO2_0:
            case GGML_TYPE_TURBO3_0:
            case GGML_TYPE_TURBO4_0:
            case GGML_TYPE_TQ5_0:
            case GGML_TYPE_TQ6_0:
                break;
            default:
                return p;
        }
        const size_t ts = ggml_type_size(t->type);
        if (t->ne[0] % ggml_blck_size(t->type) != 0 || t->nb[0] != ts || t->nb[1] % ts != 0 || t->nb[2] % ts != 0 ||
                t->nb[3] % ts != 0) {
            return p;
        }
    }
    // ALiBi slopes use the global query head index; renumbered heads would get the wrong slope
    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    if (max_bias != 0.0f) {
        return p;
    }
    // one mask shared by every head (per-head masks excluded)
    if (mask != nullptr && (mask->type != GGML_TYPE_F16 || mask->ne[0] < K->ne[1] || mask->ne[1] < Q->ne[1] ||
            mask->ne[2] != 1 || mask->ne[3] != 1)) {
        return p;
    }
    // sinks: one f32 per query head, contiguous, so a group's sinks are a pointer offset
    if (sinks != nullptr && (sinks->type != GGML_TYPE_F32 || !ggml_is_contiguous(sinks) || sinks->ne[0] != Q->ne[2] ||
            ggml_nelements(sinks) != Q->ne[2])) {
        return p;
    }

    // the route this op takes without the plan: not a fused packed-KV route, and the generic selector's f16 MMA kernel
    // (the only consumer the per-group launch below calls)
    if (ggml_cuda_flash_attn_ext_fused(nullptr, const_cast<ggml_tensor *>(dst), device)) {
        return p;
    }
    if (ggml_cuda_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_MMA_F16) {
        return p;
    }

    const int64_t n_head_kv    = K->ne[2];
    const int64_t gqa          = Q->ne[2] / n_head_kv;
    const size_t  kv_per_head  = size_t(K->ne[1]) * 256 * sizeof(half);
    const size_t  out_per_head = size_t(Q->ne[1]) * size_t(gqa) * 256 * sizeof(float);

    p.budget = budget;
    p.full   = 2 * kv_per_head * size_t(n_head_kv);
    p.floor_bytes  = 2 * kv_per_head + out_per_head;
    // the full copies fit the budget, or one group is no smaller than the full copies: unchanged route
    if (p.full <= budget || p.floor_bytes >= p.full) {
        return p;
    }

    size_t heads = budget / p.floor_bytes;
    if (heads == 0) {
        heads        = 1; // #29: never below one complete GQA head group, never back to the full copy
        p.floor_used = true;
    }
    heads = std::min(heads, size_t(n_head_kv));

    p.heads     = int(heads);
    p.n_head_kv = int(n_head_kv);
    p.gqa       = int(gqa);
    p.kv_bytes  = kv_per_head * heads;
    p.out_bytes = out_per_head * heads;
    p.offset    = GGML_PAD(ggml_nbytes(dst), 128);
    p.workspace = 2 * p.kv_bytes + p.out_bytes;
    // The compute buffer is sized once, from the worst-case graph (n_ubatch queries, full context), and every runtime
    // shape must fit what that shape reserved. Smaller runtime shapes can take more heads per group (a smaller floor) or
    // the full copies (when they fit the budget): both are <= budget. Reserving max(workspace, budget) at every shape
    // therefore keeps every runtime reservation <= the worst-case one; the extra over `workspace` is < one floor.
    p.reserve   = std::max(p.workspace, budget);

    if (p.floor_used) {
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true)) {
            GGML_LOG_WARN("fattn bounded prefill: GGML_CUDA_PREFILL_KV_MIB=%zu is below the one-group floor at n_kv=%lld "
                          "(f16 K+V of 1 KV head %.1f MiB + output of its %lld query heads at n_q=%lld %.1f MiB = %.1f MiB); "
                          "using the floor: 1 of %lld KV heads per group, workspace %.1f MiB instead of the %.1f MiB "
                          "full f16 copies (logged once)\n",
                          budget >> 20, (long long) K->ne[1], 2 * kv_per_head / 1048576.0, (long long) gqa,
                          (long long) Q->ne[1], out_per_head / 1048576.0, p.floor_bytes / 1048576.0, (long long) n_head_kv,
                          p.workspace / 1048576.0, p.full / 1048576.0);
        }
    }
    return p;
}

bool ggml_cuda_flash_attn_ext_bounded_prefill_applies(int device, const ggml_tensor * dst) {
    return ggml_cuda_fattn_bounded_prefill_plan(device, dst).heads > 0;
}

// Scatter one group's contiguous output [D, heads*gqa, n_q] into dst [D, n_head, n_q] at query head `first`.
static __global__ void ggml_cuda_fattn_bounded_prefill_scatter(
        const float * src, float * dst, const int64_t n, const int group_heads,
        const int first, const int all_heads) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int64_t row = i / 256;            // (token, head within the group)
    const int64_t tok = row / group_heads;
    const int64_t h   = row % group_heads;
    dst[(tok*all_heads + first + h)*256 + i % 256] = src[i];
}

static void ggml_cuda_flash_attn_ext_bounded_prefill(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_fattn_bounded_plan & p) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    cudaStream_t stream = ctx.stream();

    // No allocation inside CUDA graph capture. ggml_cuda_graph_check_compability (ggml-cuda.cu) marks every cgraph that
    // holds a node this plan applies to as incompatible, so such graphs always run eagerly and this assert holds by
    // construction; it is here so that a future change to the capture rules fails loudly instead of recording a pool
    // allocation into a graph.
    {
        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
        GGML_ASSERT(capture_status == cudaStreamCaptureStatusNone && "bounded prefill must not run inside CUDA graph capture");
    }

    // Workspace: the region ggml_cuda_flash_attn_ext_get_alloc_size reserved behind dst from the same plan, or the pool
    // when dst was not allocated through the buffer type (a view, or no buffer).
    ggml_cuda_pool_alloc<char> ws_pool(ctx.pool());
    char * ws = nullptr;
    const bool reserved = dst->buffer != nullptr && dst->view_src == nullptr && (uintptr_t) dst->data % 128 == 0 &&
        p.offset + p.workspace <= ggml_backend_buffer_get_alloc_size(dst->buffer, dst);
    if (reserved) {
        ws = (char *) dst->data + p.offset;
    } else {
        ws = ws_pool.alloc(p.workspace);
    }
    half  * K_ws = (half  *)  ws;
    half  * V_ws = (half  *) (ws + p.kv_bytes);
    float * O_ws = (float *) (ws + 2*p.kv_bytes);

    {
        // fallback ledger (GGML_LEDGER=1): the same f16_convert family as launch_fattn, with the bounded scratch
        const uint64_t id = ggml_cuda_ledger_mix(ggml_cuda_ledger_mix(ggml_cuda_ledger_mix(ggml_cuda_ledger_mix(ggml_cuda_ledger_mix(
            ggml_cuda_ledger_mix(0x22b0dedull, (uint64_t) K->type), (uint64_t) V->type), (uint64_t) ggml_cuda_ledger_width_bucket(Q->ne[1])),
            (uint64_t) reserved + 2), (uint64_t) p.heads), (uint64_t) p.n_head_kv);
        ggml_cuda_ledger_count("cuda.fattn", id, [&](char * buf, size_t size) {
            char w[16];
            ggml_cuda_ledger_width_str(Q->ne[1], w, sizeof(w));
            snprintf(buf, size, "f16_convert K=%s V=%s n_q=%s scratch=bounded-%s heads=%d/%d", ggml_type_name(K->type),
                ggml_type_name(V->type), w, reserved ? "reserved" : "pool", p.heads, p.n_head_kv);
        });
    }

    // Groups in ascending KV-head order on one stream: deterministic, and the workspace is reused group after group
    // (stream order serialises each group's conversion after the previous group's kernel and scatter).
    for (int first = 0; first < p.n_head_kv; first += p.heads) {
        const int heads = std::min(p.heads, p.n_head_kv - first);

        ggml_tensor q = *Q;
        ggml_tensor k = *K;
        ggml_tensor v = *V;
        ggml_tensor out = *dst;
        ggml_tensor sinks;

        q.ne[2] = int64_t(heads) * p.gqa;
        q.data  = (char *) Q->data + size_t(first) * p.gqa * Q->nb[2];

        // Converter: ggml_get_to_fp16_nc_cuda (convert.cu:934) -> dequantize_block_cuda (convert.cu:417) -> kernel
        // dequantize_block (convert.cu:10). A head slice of the KV cache view is not contiguous (the cache interleaves
        // heads per token: nb[1] = n_head_kv rows, nb[2] = one row), so this is always the strided converter with
        // block-unit strides (nb / type size), never the contiguous converter on a head slice. The output is canonical
        // contiguous f16 [D, n_kv, heads].
        ggml_tensor * kv_t[2]  = { &k, &v };
        half        * kv_ws[2] = { K_ws, V_ws };
        for (int j = 0; j < 2; ++j) {
            ggml_tensor * t = kv_t[j];
            const size_t ts = ggml_type_size(t->type);
            const to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(t->type);
            GGML_ASSERT(to_fp16 != nullptr);
            to_fp16((const char *) t->data + size_t(first) * t->nb[2], kv_ws[j],
                t->ne[0], t->ne[1], heads, 1, int64_t(t->nb[1] / ts), int64_t(t->nb[2] / ts), int64_t(t->nb[3] / ts), stream);
            CUDA_CHECK(cudaGetLastError());

            t->type  = GGML_TYPE_F16;
            t->ne[2] = heads;
            t->nb[0] = sizeof(half);
            for (int d = 1; d < GGML_MAX_DIMS; ++d) {
                t->nb[d] = t->nb[d - 1] * t->ne[d - 1];
            }
            t->data      = kv_ws[j];
            t->buffer    = nullptr;
            t->view_src  = nullptr;
            t->view_offs = 0;
        }

        out.src[0] = &q;
        out.src[1] = &k;
        out.src[2] = &v;
        out.ne[1]  = q.ne[2];
        out.nb[0]  = sizeof(float);
        for (int d = 1; d < GGML_MAX_DIMS; ++d) {
            out.nb[d] = out.nb[d - 1] * out.ne[d - 1];
        }
        out.data      = O_ws;
        out.buffer    = nullptr; // launch_fattn: no reserved-region lookup (K/V are f16 already, nothing converts)
        out.view_src  = nullptr;
        out.view_offs = 0;
        if (dst->src[4] != nullptr) {
            sinks       = *dst->src[4];
            sinks.ne[0] = q.ne[2];
            for (int d = 1; d < GGML_MAX_DIMS; ++d) {
                sinks.nb[d] = sinks.nb[d - 1] * sinks.ne[d - 1];
            }
            sinks.data  = (char *) dst->src[4]->data + size_t(first) * p.gqa * sizeof(float);
            out.src[4]  = &sinks;
        }

        // Consumer: ggml_cuda_flash_attn_ext_mma_f16 (this file) -> ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<256, 256>
        // -> ggml_cuda_flash_attn_ext_mma_f16_case (fattn-mma-f16.cuh) -> launch_fattn (fattn-common.cuh), the same kernel
        // the unbounded route runs, with the same gqa ratio (so the same ncols2), on `heads` KV heads.
        ggml_cuda_flash_attn_ext_mma_f16(ctx, &out);

        const int64_t n = ggml_nelements(&out);
        const ggml_cuda_kernel_launch_params launch(dim3((unsigned int) ((n + 255) / 256)), dim3(256), 0, stream);
        ggml_cuda_kernel_launch(ggml_cuda_fattn_bounded_prefill_scatter, launch, (const float *) O_ws, (float *) dst->data,
            n, heads * p.gqa, first * p.gqa, (int) Q->ne[2]);
        CUDA_CHECK(cudaGetLastError());
    }
}

static size_t ggml_cuda_fattn_generic_alloc_size(const int device, const ggml_tensor * dst, best_fattn_kernel * kernel_out) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const best_fattn_kernel kernel = ggml_cuda_get_best_fattn_kernel(device, dst);
    if (kernel_out) {
        *kernel_out = kernel;
    }

    bool need_f16_K = false;
    bool need_f16_V = false;

    switch (kernel) {
        case BEST_FATTN_KERNEL_TILE:
        case BEST_FATTN_KERNEL_MMA_F16:
            need_f16_K = true;
            need_f16_V = true;
            break;
        case BEST_FATTN_KERNEL_VEC: {
            const bool f16_fallback = ggml_cuda_get_fattn_vec_case(Q->ne[0], K->type, V->type) == nullptr;
            need_f16_K = K->type == GGML_TYPE_F32 || f16_fallback;
            need_f16_V = V->type == GGML_TYPE_F32 || f16_fallback;
        } break;
        case BEST_FATTN_KERNEL_NONE:
            break;
    }

    const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra =
        ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, need_f16_K, need_f16_V);

    return f16_extra.end - (uintptr_t) dst->data;
}

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst) {
    GGML_ASSERT(dst->op == GGML_OP_FLASH_ATTN_EXT);

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    GGML_ASSERT(K != nullptr);
    GGML_ASSERT(V != nullptr);

    // dry run: nothing launches and dst is not modified
    const bool fused = ggml_cuda_flash_attn_ext_fused(nullptr, const_cast<ggml_tensor *>(dst), device);

    best_fattn_kernel kernel = BEST_FATTN_KERNEL_NONE;
    const size_t size_generic = ggml_cuda_fattn_generic_alloc_size(device, dst, &kernel);
    const size_t size_fused   = ggml_nbytes(dst); // no f16 K/V copies
    // [#22/#29] the same plan the executor runs (ggml_cuda_fattn_bounded_prefill_plan); budget 0 = never applies
    const ggml_cuda_fattn_bounded_plan bounded = fused ? ggml_cuda_fattn_bounded_plan() :
        ggml_cuda_fattn_bounded_prefill_plan(device, dst);
    const size_t size_bounded = bounded.offset + bounded.reserve;
    const size_t size         = fused && ggml_cuda_fattn_alloc_route() ? size_fused :
                                bounded.heads > 0 ? size_bounded : size_generic;

    if (ggml_cuda_fattn_alloc_log()) {
        static std::mutex mtx;
        static std::map<uint64_t, bool> seen;
        int log2_kv = 0;
        while ((int64_t(1) << (log2_kv + 1)) <= K->ne[1]) {
            log2_kv++;
        }
        const uint64_t key = ggml_cuda_ledger_mix(ggml_cuda_ledger_mix(ggml_cuda_ledger_mix(ggml_cuda_ledger_mix(
            (uint64_t) K->type, (uint64_t) V->type), (uint64_t) Q->ne[1]), (uint64_t) log2_kv), (uint64_t) fused * 2 + (uint64_t) kernel);
        const uint64_t key_bp = bounded.heads > 0 ? ggml_cuda_ledger_mix(key, (uint64_t) bounded.heads) : key;
        std::lock_guard<std::mutex> lock(mtx);
        if (bounded.heads > 0 && seen.emplace(key_bp, true).second) {
            GGML_LOG_WARN("fattn alloc: dev %d K=%s V=%s n_q=%lld n_kv=%lld route=bounded_mma_f16 reserve %.1f MiB behind dst "
                          "(%d of %d KV heads per group: f16 K+V %.1f MiB + group output %.1f MiB; budget %zu MiB, "
                          "one-group floor %.1f MiB%s; generic selector: %s, %.1f MiB; saved %.1f MiB)\n",
                device, ggml_type_name(K->type), ggml_type_name(V->type), (long long) Q->ne[1], (long long) K->ne[1],
                (size - ggml_nbytes(dst)) / 1048576.0, bounded.heads, bounded.n_head_kv, 2 * bounded.kv_bytes / 1048576.0,
                bounded.out_bytes / 1048576.0, bounded.budget >> 20, bounded.floor_bytes / 1048576.0,
                bounded.floor_used ? " used" : "", ggml_cuda_fattn_kernel_name(kernel),
                (size_generic - ggml_nbytes(dst)) / 1048576.0, ((double) size_generic - (double) size) / 1048576.0);
        } else if (bounded.heads == 0 && seen.emplace(key, true).second) {
            // WARN, not INFO: ggml INFO maps to trace verbosity (4), above the server's default of 3, so INFO never prints
            GGML_LOG_WARN("fattn alloc: dev %d K=%s V=%s n_q=%lld n_kv=%lld route=%s reserve %.1f MiB behind dst "
                          "(generic selector: %s, %.1f MiB; saved %.1f MiB)\n",
                device, ggml_type_name(K->type), ggml_type_name(V->type), (long long) Q->ne[1], (long long) K->ne[1],
                fused ? "fused" : ggml_cuda_fattn_kernel_name(kernel),
                (size - ggml_nbytes(dst)) / 1048576.0, ggml_cuda_fattn_kernel_name(kernel),
                (size_generic - ggml_nbytes(dst)) / 1048576.0, (size_generic - size) / 1048576.0);
        }
    }

    return size;
}

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);

    if (ggml_cuda_flash_attn_ext_fused(&ctx, dst, ggml_cuda_get_device())) {
        return;
    }

    // [#22/#29] bounded f16 prefill: the plan the allocator sized this op's reservation with
    {
        const ggml_cuda_fattn_bounded_plan bounded = ggml_cuda_fattn_bounded_prefill_plan(ggml_cuda_get_device(), dst);
        if (bounded.heads > 0) {
            ggml_cuda_fattn_path_note("mma_f16_bounded", dst, 0);
            ggml_cuda_flash_attn_ext_bounded_prefill(ctx, dst, bounded);
            return;
        }
    }

    switch (ggml_cuda_get_best_fattn_kernel(ggml_cuda_get_device(), dst)) {
        case BEST_FATTN_KERNEL_NONE:
            GGML_ABORT("fatal error");
        case BEST_FATTN_KERNEL_TILE:
            ggml_cuda_fattn_path_note("tile", dst, 0);
            ggml_cuda_flash_attn_ext_tile(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_VEC:
            ggml_cuda_fattn_path_note("vec", dst, 0);
            ggml_cuda_flash_attn_ext_vec(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_MMA_F16:
            ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
            break;
    }
}

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst) {
    return ggml_cuda_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_NONE;
}
