#include "fattn-kvarn-lowbits-stream.cuh"
#include <atomic>
#include <cstdlib>

template <int DKQ, int DV, int ncols1, int ncols2, int bits_k, int bits_v, bool rot = false>
void ggml_cuda_lowbits_prefill_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;

    constexpr ggml_type type_K = GGML_TYPE_I8;
    constexpr ggml_type type_V = GGML_TYPE_I8;
    constexpr int ncols = ncols1 * ncols2;

    GGML_ASSERT(dst->src[5] != nullptr && dst->src[6] != nullptr);
    GGML_ASSERT(ggml_get_op_params_i32(KQV, 5) == (((bits_k & 7) << 8) | bits_v)); // bits_k | 8 / | 16: word-load / byte-load trellis build
    GGML_ASSERT(DKQ == 256 && DV == 256);
    GGML_ASSERT(rot == ggml_cuda_fattn_kvarn_rot(dst)); // [#139] the rot build rotates Q and the output in-kernel
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_F16 || dst->src[1]->type == GGML_TYPE_Q8_0 || dst->src[1]->type == GGML_TYPE_TQ6_0);
    GGML_ASSERT(dst->src[2]->type == GGML_TYPE_F16 || dst->src[2]->type == GGML_TYPE_Q8_0 || dst->src[2]->type == GGML_TYPE_TQ6_0);

    const int  nthreads       = ggml_cuda_fattn_mma_get_nthreads      (DKQ, DV, ncols, cc);
    const int  nbatch_fa      = ggml_cuda_fattn_mma_get_nbatch_fa     (DKQ, DV, ncols, cc);
    const int  nbatch_K2      = ggml_cuda_fattn_mma_get_nbatch_K2     (DKQ, DV, ncols, cc);
    const int  nbatch_V2      = ggml_cuda_fattn_mma_get_nbatch_V2     (DKQ, DV, ncols, cc);
    const int  nbatch_combine = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ, DV, ncols, cc);
    const bool Q_in_reg       = ggml_cuda_fattn_mma_get_Q_in_reg      (DKQ, DV, ncols, cc);

    const int cols_per_warp  = std::min(ncols, get_cols_per_warp(cc));
    const int warp_size_host = ggml_cuda_info().devices[ctx.device].warp_size;
    const int nwarps         = nthreads / warp_size_host;

    constexpr bool V_is_K_view = false;

    const int stride_tile_K = ggml_cuda_fattn_smem_swizzle::tile_stride(nbatch_K2, cc);
    const int stride_tile_V = ggml_cuda_fattn_smem_swizzle::tile_stride(nbatch_V2, cc);
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
        fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, /* use_sparse */ false, type_K, type_V, bits_k, bits_v, rot>;

#if !defined(GGML_USE_MUSA)
        static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
        if (!shared_memory_limit_raised[id]) {
            CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
            shared_memory_limit_raised[id] = true;
        }
#endif // !defined(GGML_USE_MUSA)
    } else if constexpr (rot) {
        GGML_ABORT("KVarN fused rotation with logit softcap (the caller takes the separate passes)");
    } else {
        constexpr bool use_logit_softcap = true;
        fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, /* use_sparse */ false, type_K, type_V, bits_k, bits_v>;

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


// GGML_KVARN_NO_DIRECT=1 (same switch as the 4/4 path): every width through the tile loader, so a scalar arm can be
// compared with a trellis arm on one decoder
static bool kvarn_lowbits_no_direct() {
    static const bool v = [] { const char * e = getenv("GGML_KVARN_NO_DIRECT"); return e != nullptr && e[0] == '1'; }();
    return v;
}

// GGML_KVARN_TRELLIS_WORDS (default on; =0 selects the byte-load kernel): trellis tile loads read the payload as aligned
// 32-bit words instead of two byte loads per code (fattn_kvarn_trellis_lb_word_w, bit-identical values). Kernel build
// bits_k | 8 (byte-load: bits_k | 16); both stage the codebooks in shared memory (#126).
// Measured 2026-10-01: +19.3% (3/3) / +17.8% (3/2) decode at 100K, identical text.
static bool kvarn_trellis_words() {
    static const bool v = [] { const char * e = getenv("GGML_KVARN_TRELLIS_WORDS"); return e == nullptr || e[0] != '0'; }();
    return v;
}
// route proof for end-to-end gates: one line on the first trellis attention call that takes the word-load build
static void kvarn_trellis_words_note() {
    static std::atomic_flag done = ATOMIC_FLAG_INIT;
    if (!done.test_and_set()) {
        GGML_LOG_INFO("%s: KVarN trellis attention uses the word-load decode (GGML_KVARN_TRELLIS_WORDS=0 disables)\n", __func__);
    }
}

template<int bits_k, int bits_v, bool rot = false>
static void ggml_cuda_lowbits_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if constexpr (bits_k == 4 && bits_v == 4) {
        static_assert(!rot, "the 4/4 trellis4 tile build takes the separate rotation passes");
        // 4/4 reaches this path only as the token-axis trellis4 body (explicit nonzero GGML_KVARN_TRELLIS_TOKENS, fattn.cu)
        GGML_ASSERT(ggml_get_op_params_i32(dst,7) == GGML_TYPE_I16 && ggml_kvarn::trellis3::tokens4());
        ggml_cuda_kvarn_trellis_cb_init();
        ggml_cuda_lowbits_prefill_case<256,256,8,8,4,4>(ctx,dst);
    } else if (ggml_get_op_params_i32(dst,7) == GGML_TYPE_I16) {
        // trellis-coded body: the stream kernel has no codebook path, the tile loader decodes every width from the
        // codebooks staged in shared memory (#126). Only 3/3, 3/2 and 2/2 seal a 2/3-bit trellis body (llama-context.cpp).
        if constexpr (bits_k <= 3 && bits_v <= 3) {
            ggml_cuda_kvarn_trellis_cb_init();
            if (kvarn_trellis_words()) {
                kvarn_trellis_words_note();
                ggml_cuda_lowbits_prefill_case<256,256,8,8,bits_k | 8,bits_v,rot>(ctx,dst);
            } else if constexpr (!rot) {
                ggml_cuda_lowbits_prefill_case<256,256,8,8,bits_k | 16,bits_v>(ctx,dst);
            } else {
                GGML_ABORT("KVarN fused rotation on the byte-load trellis build (the caller takes the separate passes)");
            }
        } else {
            GGML_ABORT("KVarN trellis body on a %d/%d pair (trellis needs 3/3, 3/2, 2/2 or the 4/4 trellis4 body)", bits_k, bits_v);
        }
    } else if (dst->src[0]->ne[1] <= 8 && !kvarn_lowbits_no_direct()) {
        ggml_cuda_kvarn_lowbits_width<bits_k,bits_v,rot>(ctx,dst);
    } else {
        ggml_cuda_lowbits_prefill_case<256,256,8,8,bits_k,bits_v,rot>(ctx,dst);
    }
}

void ggml_cuda_flash_attn_kvarn_lowbits(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// [#139] fused Q / output rotation: the tile build (Ampere table: one 128-half2 combine batch at 8x8 columns), the
// word-load trellis build and the width kernel rotate in-kernel; the 4/4 trellis4 body, the byte-load trellis build,
// logit softcap and other MMA tables take the separate passes around this same dispatch.
template<int bits_k, int bits_v>
static void ggml_cuda_lowbits_case_rot(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (!ggml_cuda_fattn_kvarn_rot(dst)) {
        ggml_cuda_lowbits_case<bits_k,bits_v>(ctx,dst);
        return;
    }
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    float logit_softcap;
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    const bool trellis = ggml_get_op_params_i32(dst,7) == GGML_TYPE_I16;
    bool in_kernel = !(bits_k == 4 && bits_v == 4) && logit_softcap == 0.0f &&
        ggml_cuda_fattn_mma_get_nbatch_combine(256, 256, 64, cc) == 128 && ggml_cuda_fattn_kvarn_rot_q_aligned(dst->src[0]) &&
        (!trellis || kvarn_trellis_words());
    if constexpr (bits_k == 4 && bits_v == 4) {
        in_kernel = false;
    }
    if (!in_kernel) {
        ggml_cuda_flash_attn_ext_kvarn_rot_unfused(ctx, dst, ggml_cuda_flash_attn_kvarn_lowbits);
        return;
    }
    if constexpr (!(bits_k == 4 && bits_v == 4)) {
        ggml_cuda_lowbits_case<bits_k,bits_v,true>(ctx,dst);
    }
}

void ggml_cuda_flash_attn_kvarn_lowbits(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    switch (ggml_get_op_params_i32(dst,5)) {
        case (4 << 8) | 3: ggml_cuda_lowbits_case_rot<4,3>(ctx,dst); break;
        case (3 << 8) | 3: ggml_cuda_lowbits_case_rot<3,3>(ctx,dst); break;
        case (4 << 8) | 2: ggml_cuda_lowbits_case_rot<4,2>(ctx,dst); break;
        case (2 << 8) | 4: ggml_cuda_lowbits_case_rot<2,4>(ctx,dst); break;
        case (3 << 8) | 2: ggml_cuda_lowbits_case_rot<3,2>(ctx,dst); break;
        case (2 << 8) | 2: ggml_cuda_lowbits_case_rot<2,2>(ctx,dst); break;
        case (4 << 8) | 4: ggml_cuda_lowbits_case_rot<4,4>(ctx,dst); break; // token-axis trellis4 only (explicit nonzero GGML_KVARN_TRELLIS_TOKENS)
        default: GGML_ABORT("Unsupported KVarN lower-bit pair");
    }
}
