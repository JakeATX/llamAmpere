// KVarN region-aware MMA flash-attention launcher (any number of query rows: decode widths and prefill ubatches).
//
// Host-side case launcher for the GQA-packed MMA path over a KVarN cache: F16, Q8_0 or TQ6_0 rows for the sink and the
// ring (dst->src[1]/src[2]) plus sealed 4-bit records for the body (dst->src[5], descriptor dst->src[6]). It
// instantiates flash_attn_ext_f16 (fattn-mma-f16.cuh) with type_K = type_V = GGML_TYPE_I8, which selects the
// is_kvarn branches: the in-kernel tile loaders decode body records straight into shared memory and copy ring
// rows through their storage loader into F16 tiles for MMA. Single-stage synchronous loading, no cp.async staging, no f16 conversion in launch_fattn.

#pragma once

#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "kvarn-seal.cuh"

template <int DKQ, int DV, int ncols1, int ncols2>
void ggml_cuda_flash_attn_ext_mma_kvarn_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;

    constexpr ggml_type type_K = GGML_TYPE_I8;
    constexpr ggml_type type_V = GGML_TYPE_I8;
    constexpr int ncols = ncols1 * ncols2;

    GGML_ASSERT(dst->src[5] != nullptr && dst->src[6] != nullptr);
    GGML_ASSERT(ggml_get_op_params_i32(KQV, 5) == ((4 << 8) | 4) && "KVarN CUDA path: 4-bit K and V only");
    GGML_ASSERT(DKQ == 256 && DV == 256);
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_F16 || dst->src[1]->type == GGML_TYPE_Q8_0 || dst->src[1]->type == GGML_TYPE_TQ6_0);
    GGML_ASSERT(dst->src[2]->type == GGML_TYPE_F16 || dst->src[2]->type == GGML_TYPE_Q8_0 || dst->src[2]->type == GGML_TYPE_TQ6_0);

    const int  nthreads       = ggml_cuda_fattn_mma_get_nthreads      (DKQ, DV, ncols, cc);
    const int  nbatch_fa      = ggml_cuda_fattn_mma_get_nbatch_fa     (DKQ, DV, ncols, type_K, cc); // [#39] same per-K-type value as the kernel
    const int  nbatch_K2      = ggml_cuda_fattn_mma_get_nbatch_K2     (DKQ, DV, ncols, cc);
    const int  nbatch_V2      = ggml_cuda_fattn_mma_get_nbatch_V2     (DKQ, DV, ncols, cc);
    const int  nbatch_combine = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ, DV, ncols, cc);
    const bool Q_in_reg       = ggml_cuda_fattn_mma_get_Q_in_reg      (DKQ, DV, ncols, cc);

    const int cols_per_warp  = std::min(ncols, get_cols_per_warp(cc));
    const int warp_size_host = ggml_cuda_info().devices[ctx.device].warp_size;
    const int nwarps         = nthreads / warp_size_host;

    constexpr bool V_is_K_view = false;

    // must match the tile stride the device side uses (ggml_cuda_fattn_mma_get_swizzled / swizzle_bytes in
    // fattn-mma-f16.cuh, which flash_attn_ext_kvarn_load_tile writes through): same helpers as the turbo launcher.
    const bool swizzled     = ggml_cuda_fattn_mma_get_swizzled(DKQ, DV, ncols1, ncols2, cc);
    const int stride_tile_K = ggml_cuda_fattn_mma_get_stride_tile(nbatch_K2, swizzled);
    const int stride_tile_V = ggml_cuda_fattn_mma_get_stride_tile(nbatch_V2, swizzled);
    const size_t nbytes_shared_KV_1stage = nbatch_fa            * std::max(stride_tile_K, stride_tile_V) * sizeof(half2);
    const size_t nbytes_shared_Q         = ncols                * (DKQ/2 + 4)                             * sizeof(half2);
    const size_t nbytes_shared_mask      = ncols1               * (nbatch_fa/2 + 4)                       * sizeof(half2);
    const size_t nbytes_shared_combine   = nwarps*cols_per_warp * (nbatch_combine + 4)                    * sizeof(half2);

    const size_t nbytes_shared_KV = nbytes_shared_KV_1stage;

    const size_t nbytes_shared_total = std::max(nbytes_shared_combine, Q_in_reg ?
        std::max(nbytes_shared_Q,  nbytes_shared_KV + nbytes_shared_mask) :
                 nbytes_shared_Q + nbytes_shared_KV + nbytes_shared_mask);

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

#if defined(GGML_USE_HIP)
    using fattn_kernel_ptr_t = const void*;
#else
    using fattn_kernel_ptr_t = fattn_kernel_t;
#endif // defined(GGML_USE_HIP)
    fattn_kernel_t fattn_kernel;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, /* use_sparse */ false, type_K, type_V>;

#if !defined(GGML_USE_MUSA)
        static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
        if (!shared_memory_limit_raised[id]) {
            CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
            shared_memory_limit_raised[id] = true;
        }
#endif // !defined(GGML_USE_MUSA)
    } else {
        constexpr bool use_logit_softcap = true;
        fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, /* use_sparse */ false, type_K, type_V>;

#if !defined(GGML_USE_MUSA)
        static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
        if (!shared_memory_limit_raised[id]) {
            CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
            shared_memory_limit_raised[id] = true;
        }
#endif // !defined(GGML_USE_MUSA)
    }

    // K/V stay in their ring storage format; only shared tiles are decoded. stream_k = true.
    launch_fattn<DV, ncols1, ncols2>
        (ctx, dst, fattn_kernel, nwarps, nbytes_shared_total, nbatch_fa,
         /*need_f16_K=*/false, /*need_f16_V=*/false, /*stream_k=*/true, /*use_sparse=*/false, warp_size_host);
}

#define DECL_FATTN_MMA_KVARN_CASE(DKQ, DV, ncols1, ncols2)                          \
    template void ggml_cuda_flash_attn_ext_mma_kvarn_case                           \
    <DKQ, DV, ncols1, ncols2>(ggml_backend_cuda_context & ctx, ggml_tensor * dst)

// Reachable (ncols1, ncols2) for Q->ne[1] in {1..8}: gqa > 4 -> (1,8),(2,8),(4,8),(8,8); gqa in {3,4} -> (2,4),(4,4);
// gqa == 2 -> (4,2); gqa == 1 -> (8,1).
extern DECL_FATTN_MMA_KVARN_CASE(256, 256, 1, 8);
extern DECL_FATTN_MMA_KVARN_CASE(256, 256, 2, 8);
extern DECL_FATTN_MMA_KVARN_CASE(256, 256, 4, 8);
extern DECL_FATTN_MMA_KVARN_CASE(256, 256, 8, 8);
extern DECL_FATTN_MMA_KVARN_CASE(256, 256, 2, 4);
extern DECL_FATTN_MMA_KVARN_CASE(256, 256, 4, 4);
extern DECL_FATTN_MMA_KVARN_CASE(256, 256, 4, 2);
extern DECL_FATTN_MMA_KVARN_CASE(256, 256, 8, 1);

// fattn-kvarn-direct.cuh (template-instances/fattn-mma-kvarn-direct-instance.cu): register-streaming decode kernel
// for n_q <= 8 at GQA <= 8; GGML_KVARN_NO_DIRECT=1 falls back to the shared-memory tile kernel below.
bool ggml_cuda_flash_attn_ext_kvarn_direct_supported(const ggml_tensor * dst);
void ggml_cuda_flash_attn_ext_kvarn_direct(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_flash_attn_ext_kvarn_prefill(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

template <int DKQ, int DV>
static void ggml_cuda_flash_attn_ext_mma_kvarn_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    const int n_kv_pad = ggml_get_op_params_i32(KQV, 6);
    const bool use_gqa_opt = mask && max_bias == 0.0f && n_kv_pad % FATTN_KQ_STRIDE == 0 &&
        Q->nb[1] % 16 == 0 && (K->type != GGML_TYPE_F16 || K->nb[1] % 16 == 0) && (V->type != GGML_TYPE_F16 || V->nb[1] % 16 == 0) && mask->nb[1] % 16 == 0;
    GGML_ASSERT(use_gqa_opt && "KVarN CUDA path needs a mask, no ALiBi and n_kv_pad % 256 == 0");

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    if (ggml_get_op_params_i32(dst, 7) == GGML_TYPE_I16) {
        ggml_cuda_kvarn_trellis_cb_init();
    }
    if (ggml_cuda_flash_attn_ext_kvarn_prefill(ctx, dst)) {
        return;
    }
    if (ggml_cuda_flash_attn_ext_kvarn_direct_supported(dst)) {
        ggml_cuda_flash_attn_ext_kvarn_direct(ctx, dst);
        return;
    }
    const int gqa_ratio = Q->ne[2] / K->ne[2];
    const int n_q = Q->ne[1];

    // Larger n_q (prefill ubatches) runs on the widest instantiated ncols1 of the GQA class: the kernel loops over
    // ncols1-row query blocks (jt), exactly as the f16 MMA path does above 8 rows.
    if (gqa_ratio > 4) {
        if (n_q <= 1) { ggml_cuda_flash_attn_ext_mma_kvarn_case<DKQ, DV, 1, 8>(ctx, dst); return; }
        if (n_q <= 2) { ggml_cuda_flash_attn_ext_mma_kvarn_case<DKQ, DV, 2, 8>(ctx, dst); return; }
        if (n_q <= 4) { ggml_cuda_flash_attn_ext_mma_kvarn_case<DKQ, DV, 4, 8>(ctx, dst); return; }
        ggml_cuda_flash_attn_ext_mma_kvarn_case<DKQ, DV, 8, 8>(ctx, dst);
        return;
    }
    if (gqa_ratio > 2) {
        if (n_q <= 2) { ggml_cuda_flash_attn_ext_mma_kvarn_case<DKQ, DV, 2, 4>(ctx, dst); return; }
        ggml_cuda_flash_attn_ext_mma_kvarn_case<DKQ, DV, 4, 4>(ctx, dst);
        return;
    }
    if (gqa_ratio > 1) {
        ggml_cuda_flash_attn_ext_mma_kvarn_case<DKQ, DV, 4, 2>(ctx, dst);
        return;
    }
    ggml_cuda_flash_attn_ext_mma_kvarn_case<DKQ, DV, 8, 1>(ctx, dst);
}

// gate mirrored by ggml_cuda_flash_attn_ext_supported: widths the switch above can serve
static inline bool ggml_cuda_flash_attn_ext_kvarn_supported(const int cc, const ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    if (dst->src[5] == nullptr || dst->src[6] == nullptr) {
        return false;
    }
    if (!turing_mma_available(cc) || mask == nullptr || Q->ne[3] != 1) {
        return false;
    }
    if (Q->ne[0] != 256 || V->ne[0] != 256 || (K->type != GGML_TYPE_F16 && K->type != GGML_TYPE_Q8_0 && K->type != GGML_TYPE_TQ6_0) || (V->type != GGML_TYPE_F16 && V->type != GGML_TYPE_Q8_0 && V->type != GGML_TYPE_TQ6_0)) {
        return false;
    }
    const int bits = ggml_get_op_params_i32(dst,5);
    if (bits != ((4 << 8) | 4)) {
        if (bits != ((4 << 8) | 3) && bits != ((3 << 8) | 3) && bits != ((4 << 8) | 2) && bits != ((2 << 8) | 4) && bits != ((3 << 8) | 2) && bits != ((2 << 8) | 2)) {
            return false;
        }
        const int gqa = Q->ne[2]/K->ne[2];
        float softcap;
        memcpy(&softcap, (const float *) dst->op_params + 2, sizeof(float));
        const int body_type = ggml_get_op_params_i32(dst,7);
        const bool trellis3 = body_type == GGML_TYPE_I16 && (bits == ((3 << 8) | 3) || bits == ((3 << 8) | 2) || bits == ((2 << 8) | 2)); // 3-bit and 2-bit trellis stream payloads
        if (gqa <= 4 || gqa > 8 || dst->src[4] != nullptr || softcap != 0 || !cp_async_available(cc) || (body_type != 0 && !trellis3)) {
            return false;
        }
    }
    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    if (max_bias != 0.0f || ggml_get_op_params_i32(dst, 6) % FATTN_KQ_STRIDE != 0) {
        return false;
    }
    if (Q->ne[2] % K->ne[2] != 0) {
        return false;
    }
    if (Q->nb[1] % 16 != 0 || mask->nb[1] % 16 != 0) {
        return false;
    }
    for (const ggml_tensor * kv : {K, V}) {
        const size_t alignment = kv->type == GGML_TYPE_F16 ? 16 : sizeof(half2);
        if (kv->nb[1] % alignment != 0 || kv->nb[2] % alignment != 0 || kv->view_offs % alignment != 0) {
            return false;
        }
    }
    return true; // any n_q: the switch above tiles query rows in ncols1 blocks
}
