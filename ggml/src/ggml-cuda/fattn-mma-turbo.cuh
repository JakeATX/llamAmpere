// Fused turbo4 (4-bit PolarQuant) MMA flash-attention DECODE launcher.
//
// This is the host-side case launcher for the GQA-packed MMA path with turbo4 KV.
// It reuses the f16 MMA device kernel (flash_attn_ext_f16 in fattn-mma-f16.cuh) but
// instantiates it with type_K/type_V = TURBO4_0 so the in-kernel load tiles dequantize
// raw turbo4 blocks straight into SRAM. Q is ALREADY rotated at the graph level
// (src/llama-graph.cpp) and the FA output is inverse-rotated there too — this path does
// NO inline FWHT and NO src swap (that would double-rotate Q).
//
// Differences vs ggml_cuda_flash_attn_ext_mma_f16_case:
//   * nstages is forced to 0 inside the kernel for turbo (synchronous dequant load), so
//     here we size shared memory for the 1-stage path.
//   * launch_fattn is called with need_f16_K = need_f16_V = false, so launch_fattn does
//     NOT pre-convert K/V to f16; the kernel receives the raw quantized bytes and the
//     true byte pitch nb11/nb21.

#pragma once

#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"

template <int DKQ, int DV, int ncols1, int ncols2, ggml_type type_K, ggml_type type_V, bool compact_q5g6, bool early_v>
static void ggml_cuda_flash_attn_ext_mma_turbo_case_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    using query_layout = ggml_fattn_query_layout<ncols1, ncols2, compact_q5g6>;

    const ggml_tensor * KQV = dst;
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;

    constexpr int ncols = ncols1 * ncols2;
    constexpr bool preserve_cand = ggml_cuda_fattn_mma_cand_pair(type_K, type_V);
    if constexpr (preserve_cand && ggml_cuda_fattn_mma_cand_shape(DKQ, DV, ncols)) {
        if (ampere_mma_available(cc)) {
            ggml_cuda_fattn_mma_note_cand(DKQ, ncols);
        }
    }

    const int  nthreads       = ggml_cuda_fattn_mma_get_nthreads<preserve_cand>      (DKQ, DV, ncols, cc);
    const int  nbatch_fa      = ggml_cuda_fattn_mma_get_nbatch_fa<preserve_cand>     (DKQ, DV, ncols, type_K, cc); // [#39] per K type
    const int  nbatch_K2      = ggml_cuda_fattn_mma_get_nbatch_K2<preserve_cand>     (DKQ, DV, ncols, cc);
    const int  nbatch_V2      = ggml_cuda_fattn_mma_get_nbatch_V2<preserve_cand>     (DKQ, DV, ncols, cc);
    const int  nbatch_combine = ggml_cuda_fattn_mma_get_nbatch_combine<preserve_cand>(DKQ, DV, ncols, cc);
    const bool Q_in_reg       = ggml_cuda_fattn_mma_get_Q_in_reg<preserve_cand>      (DKQ, DV, ncols, cc);

    // turbo path is always single-stage synchronous (nstages forced to 0 in the kernel).
    const int cols_per_warp = std::min(ncols, get_cols_per_warp(cc));
    const int warp_size_host = ggml_cuda_info().devices[ctx.device].warp_size;
    const int nwarps         = nthreads / warp_size_host;

    // turbo4 never aliases V onto K.
    constexpr bool V_is_K_view = false;

    // must match the swizzled tile stride flash_attn_ext_turbo{2,3,4}_load_tile write through
    // (fattn-mma-f16.cuh's turbo_store_h2 / swizzle_bytes), same helpers as the f16 host launcher.
    const bool swizzled     = ggml_cuda_fattn_mma_get_swizzled<preserve_cand>(DKQ, DV, ncols1, ncols2, cc);
    const int stride_tile_K = ggml_cuda_fattn_mma_get_stride_tile(nbatch_K2, swizzled);
    const int stride_tile_V = ggml_cuda_fattn_mma_get_stride_tile(nbatch_V2, swizzled);
    const size_t nbytes_shared_KV_1stage = nbatch_fa            * std::max(stride_tile_K, stride_tile_V) * sizeof(half2);
    const size_t nbytes_shared_Q         = ncols                * (DKQ/2 + 4)                             * sizeof(half2);
    const size_t nbytes_shared_mask      = query_layout::mask_rows * (nbatch_fa/2 + 4)                       * sizeof(half2);
    const size_t nbytes_shared_combine   = nwarps*cols_per_warp * (nbatch_combine + 4)                    * sizeof(half2);

    const size_t nbytes_shared_KV = nbytes_shared_KV_1stage;

    size_t nbytes_shared_total = std::max(nbytes_shared_combine, Q_in_reg ?
        std::max(nbytes_shared_Q,  nbytes_shared_KV + nbytes_shared_mask) :
                 nbytes_shared_Q + nbytes_shared_KV + nbytes_shared_mask);
    if constexpr (ggml_cuda_fattn_turbo_stage<DKQ, DV, ncols2, type_K, type_V>()) {
        // raw compressed-tile staging buffers after the Q/KV/mask region (offset mirrored in the kernel)
        const size_t stage_off = ggml_cuda_fattn_align16((int) (Q_in_reg ?
            std::max(nbytes_shared_Q, nbytes_shared_KV + nbytes_shared_mask) :
            nbytes_shared_Q + nbytes_shared_KV + nbytes_shared_mask));
        // q8_0 K rows (272 B) and q8_0 V rows are copied with 16-byte cp.async; tq6_0/tq5_0 K rows (196/164 B) and
        // turbo3 V rows (100 B) with 4-byte cp.async, since per-head rows are only 4-byte aligned
        constexpr int k_align = (type_K == GGML_TYPE_TQ6_0 || type_K == GGML_TYPE_TQ5_0) ? 4 : 16;
        constexpr int v_align = type_V == GGML_TYPE_Q8_0  ? 16 : 4;
        constexpr bool stage_v = ggml_cuda_fattn_turbo_stage_v<DKQ, DV, ncols2, type_K, type_V>(); // tq6/tq5 V: K-only staging
        GGML_ASSERT(dst->src[1]->nb[1] % k_align == 0 && dst->src[1]->nb[2] % k_align == 0 && ((uintptr_t) dst->src[1]->data) % k_align == 0);
        GGML_ASSERT(!stage_v || (dst->src[2]->nb[1] % v_align == 0 && dst->src[2]->nb[2] % v_align == 0 && ((uintptr_t) dst->src[2]->data) % v_align == 0));
        const size_t extra_v = early_v ? nbatch_fa * ggml_cuda_fattn_turbo_stage_v_pitch<DV, type_V>() : 0;
        nbytes_shared_total = std::max(nbytes_shared_total, stage_off + (size_t) ggml_cuda_fattn_turbo_stage_bytes<DKQ, DV, ncols2, type_K, type_V>(nbatch_fa) + extra_v);
    }

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));
    if constexpr (early_v) {
        ggml_ledger_addf("cuda.fattn.early_v", 1, "cols=%dx%d,compact=%d,softcap=%d", ncols1, ncols2, compact_q5g6, logit_softcap != 0.0f);
    }

#if defined(GGML_USE_HIP)
    using fattn_kernel_ptr_t = const void*;
#else
    using fattn_kernel_ptr_t = fattn_kernel_t;
#endif // defined(GGML_USE_HIP)
    fattn_kernel_t fattn_kernel;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, /* use_sparse */ false, type_K, type_V, /* output_partial */ false, preserve_cand, compact_q5g6, early_v>;

#if !defined(GGML_USE_MUSA)
        static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
        if (!shared_memory_limit_raised[id]) {
            CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
            shared_memory_limit_raised[id] = true;
        }
#endif // !defined(GGML_USE_MUSA)
    } else {
        constexpr bool use_logit_softcap = true;
        fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, /* use_sparse */ false, type_K, type_V, /* output_partial */ false, preserve_cand, compact_q5g6, early_v>;

#if !defined(GGML_USE_MUSA)
        static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
        if (!shared_memory_limit_raised[id]) {
            CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
            shared_memory_limit_raised[id] = true;
        }
#endif // !defined(GGML_USE_MUSA)
    }

    // need_f16_K = need_f16_V = false: launch_fattn does NOT convert turbo bytes to f16;
    // the kernel receives raw quantized KV + the true byte pitch. stream_k = true.
    launch_fattn<DV, ncols1, ncols2, compact_q5g6>
        (ctx, dst, fattn_kernel, nwarps, nbytes_shared_total, nbatch_fa,
         /*need_f16_K=*/false, /*need_f16_V=*/false, /*stream_k=*/true, /*use_sparse=*/false, warp_size_host);
}


template <int DKQ, int DV, int ncols1, int ncols2, ggml_type type_K, ggml_type type_V, bool compact_q5g6 = false>
void ggml_cuda_flash_attn_ext_mma_turbo_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if constexpr (DKQ == 256 && DV == 256 && ncols2 == 8 && type_K == GGML_TYPE_TQ5_0 && type_V == GGML_TYPE_TURBO4_0 &&
                  ggml_cuda_fattn_turbo_stage_v<DKQ, DV, ncols2, type_K, type_V>()) {
        static const bool early_v = [] {
            const char * e = getenv("GGML_VL_AT");
            return e != nullptr && strcmp(e, "1") == 0;
        }();
        const ggml_tensor * Q = dst->src[0];
        const ggml_tensor * K = dst->src[1];
        if (early_v && ggml_cuda_info().devices[ctx.device].cc == 860 && Q->ne[1] >= 1 && Q->ne[1] <= 8 &&
                Q->ne[2] == 6*K->ne[2] && K->ne[1] % FATTN_KQ_STRIDE == 0) {
            ggml_cuda_flash_attn_ext_mma_turbo_case_impl<DKQ, DV, ncols1, ncols2, type_K, type_V, compact_q5g6, true>(ctx, dst);
            return;
        }
    }
#endif
    // Other widths, pairs, devices, and GQA layouts keep the existing pipeline.
    ggml_cuda_flash_attn_ext_mma_turbo_case_impl<DKQ, DV, ncols1, ncols2, type_K, type_V, compact_q5g6, false>(ctx, dst);
}


#define DECL_FATTN_MMA_TURBO_CASE(DKQ, DV, ncols1, ncols2, tK, tV)                  \
    template void ggml_cuda_flash_attn_ext_mma_turbo_case                           \
    <DKQ, DV, ncols1, ncols2, tK, tV>(ggml_backend_cuda_context & ctx, ggml_tensor * dst)

// The reachable (ncols1, ncols2) set for Q->ne[1] in {1..4} with turing_mma_available
// is exactly: (1,8),(2,8),(4,8),(2,4),(4,4),(4,2),(8,1). Declare those externs only.
#define DECL_FATTN_MMA_TURBO_ALL(DKQ, DV, tK, tV)        \
    extern DECL_FATTN_MMA_TURBO_CASE(DKQ, DV, 1, 8, tK, tV); \
    extern DECL_FATTN_MMA_TURBO_CASE(DKQ, DV, 2, 8, tK, tV); \
    extern DECL_FATTN_MMA_TURBO_CASE(DKQ, DV, 4, 8, tK, tV); \
    extern DECL_FATTN_MMA_TURBO_CASE(DKQ, DV, 2, 4, tK, tV); \
    extern DECL_FATTN_MMA_TURBO_CASE(DKQ, DV, 4, 4, tK, tV); \
    extern DECL_FATTN_MMA_TURBO_CASE(DKQ, DV, 4, 2, tK, tV); \
    extern DECL_FATTN_MMA_TURBO_CASE(DKQ, DV, 8, 1, tK, tV); \

DECL_FATTN_MMA_TURBO_ALL(128, 128, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0);
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0);
DECL_FATTN_MMA_TURBO_ALL(128, 128, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0);
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0);
DECL_FATTN_MMA_TURBO_ALL(128, 128, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0);
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0);
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_Q8_0, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_MMA_TURBO_CASE(256, 256, 8, 8, GGML_TYPE_Q8_0, GGML_TYPE_TURBO3_0);
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_MMA_TURBO_CASE(256, 256, 8, 8, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0);
DECL_FATTN_MMA_TURBO_ALL(128, 128, GGML_TYPE_TQ6_0, GGML_TYPE_TQ6_0);
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TQ6_0, GGML_TYPE_TQ6_0);
DECL_FATTN_MMA_TURBO_ALL(128, 128, GGML_TYPE_TQ6_0, GGML_TYPE_TURBO3_0);
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TQ6_0, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_MMA_TURBO_CASE(256, 256, 8, 8, GGML_TYPE_TQ6_0, GGML_TYPE_TURBO3_0);
DECL_FATTN_MMA_TURBO_ALL(128, 128, GGML_TYPE_TQ5_0, GGML_TYPE_TQ5_0);
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TQ5_0, GGML_TYPE_TQ5_0);
DECL_FATTN_MMA_TURBO_ALL(128, 128, GGML_TYPE_TQ5_0, GGML_TYPE_TURBO3_0);
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TQ5_0, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_MMA_TURBO_CASE(256, 256, 8, 8, GGML_TYPE_TQ5_0, GGML_TYPE_TURBO3_0);
// turbo4 V under a q8_0 / tq6_0 / tq5_0 K (D=256 only; unstaged V tile, same loader as turbo4/turbo4)
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TQ5_0, GGML_TYPE_TURBO4_0);
extern DECL_FATTN_MMA_TURBO_CASE(256, 256, 8, 8, GGML_TYPE_TQ5_0, GGML_TYPE_TURBO4_0);
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TQ6_0, GGML_TYPE_TURBO4_0);
extern DECL_FATTN_MMA_TURBO_CASE(256, 256, 8, 8, GGML_TYPE_TQ6_0, GGML_TYPE_TURBO4_0);
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_Q8_0, GGML_TYPE_TURBO4_0);
extern DECL_FATTN_MMA_TURBO_CASE(256, 256, 8, 8, GGML_TYPE_Q8_0, GGML_TYPE_TURBO4_0);
// tq5_0 V under a tq6_0 K (D=256 only): the 6-bit-K / 5-bit-V pair, both tiles staged through cp.async
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TQ6_0, GGML_TYPE_TQ5_0);
extern DECL_FATTN_MMA_TURBO_CASE(256, 256, 8, 8, GGML_TYPE_TQ6_0, GGML_TYPE_TQ5_0);
