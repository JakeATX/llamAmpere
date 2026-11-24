// Spill-free compact SJ-KVaRN 4/4 verify attention for widths 7..8 at GQA 6 (W=6, sm_80+).
// Instantiates flash_attn_ext_sj_kvarn_stream with qp in shared memory (eliminating the 64 B/thread
// local spill at W=6, 2 CTA/SM), raw-K code bias folding, hoisted mask loads, L2::128B cp.async,
// and the pipelined tq6_0/q8_0 exact ring in a separate TU so default stream builds stay unchanged.

#define FATTN_SJKVARN_STREAM_KERNEL_ONLY
#include "fattn-sjkvarn-w8.cuh"
#include "fattn-sjkvarn-stream.cuh"
#include "qsa-resident-clock.cuh"

#include <atomic>
#include <climits>
#include <cstdlib>
#include <cstring>

namespace fattn_sj_kvarn_w8 {

using namespace fattn_sj_kvarn_stream;

constexpr int W8_WARPS = 6; // ceil(6 * n_q / 8) for n_q in {7, 8}

static constexpr size_t w8_shared_bytes(const int nstage) {
    return split_bytes(nstage, false) + (size_t) W8_WARPS * QP_WARP_BYTES;
}

// GGML_SJKVARN_W8: unset/0 = off (ggml default: every run keeps the cand verify kernels), 2 = qp-smem + decode_pairs + raw_pipe,
// 1 = qp-smem + raw-K fold + raw_pipe. common/arg.cpp sets 2 for draft-dflash (DSpine) runs only, next to the W78 rule.
static int w8_parse_env(const char * name, const long lo, const long hi, const bool allow_zero, const int default_val, const char * desc) {
    const char * e = getenv(name);
    if (e == nullptr || *e == '\0') {
        return default_val;
    }
    char * end = nullptr;
    const long v = strtol(e, &end, 10);
    if (end == e || *end != '\0' || ((v < lo || v > hi) && !(allow_zero && v == 0))) {
        GGML_LOG_WARN("SJKVARN W8: invalid %s=\"%s\" (expected %s); falling back to 0\n", name, e, desc);
        return 0;
    }
    return (int) v;
}

static int w8_mode()       { static const int v = w8_parse_env("GGML_SJKVARN_W8",        0, 2,       false, 0, "0, 1 or 2");    return v; }
static int w8_nstage_env() { static const int v = w8_parse_env("GGML_SJKVARN_W8_NSTAGE", 2, 4,       true,  0, "0, 2, 3 or 4"); return v; }
static int w8_splits_env() { static const int v = w8_parse_env("GGML_SJKVARN_W8_SPLITS", 0, INT_MAX, true,  0, ">= 0");         return v; }
// Test-only switch: force the rot occupancy-mismatch fallback (unfused W8) so tests can verify it directly.
static bool w8_force_unfused_env() { static const bool v = w8_parse_env("GGML_SJKVARN_W8_FORCE_UNFUSED", 0, 1, false, 0, "0 or 1") != 0; return v; }

template <int NSTAGE, bool rot, bool raw_k>
static fattn_kernel_t w8_kernel() {
    fattn_kernel_t kernel = flash_attn_ext_sj_kvarn_stream<
        W8_WARPS, NSTAGE, /*turbo4_body=*/false, rot, /*lsa=*/false, sj_kvarn_tile::compact,
        /*qp_smem=*/true, raw_k, /*force_raw_pipe=*/true>;
    const int id = ggml_cuda_get_device();
    static bool initialized[GGML_CUDA_MAX_DEVICES] = {false};
    if (!initialized[id]) {
        const size_t limit = ggml_cuda_info().devices[id].smpbo;
        CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void *>(kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, (int) limit));
        initialized[id] = true;
    }
    return kernel;
}

template <int NSTAGE, bool rot, bool raw_k>
static int w8_ctas_per_sm() {
    const int id = ggml_cuda_get_device();
    static std::atomic<int> cached[GGML_CUDA_MAX_DEVICES];
    if (id >= 0 && id < GGML_CUDA_MAX_DEVICES) {
        const int v = cached[id].load(std::memory_order_relaxed);
        if (v != 0) {
            return v - 1;
        }
    }
    const size_t shared = w8_shared_bytes(NSTAGE);
    int blocks = 0;
    if (shared <= ggml_cuda_info().devices[id].smpbo) {
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, w8_kernel<NSTAGE, rot, raw_k>(), WARP_SIZE * W8_WARPS, shared));
    }
    if (id >= 0 && id < GGML_CUDA_MAX_DEVICES) {
        cached[id].store(blocks + 1, std::memory_order_relaxed);
    }
    return blocks;
}

template <int NSTAGE, bool rot, bool raw_k>
static void w8_launch_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    fattn_kernel_t kernel = w8_kernel<NSTAGE, rot, raw_k>();
    const size_t shared   = w8_shared_bytes(NSTAGE);
    const int ctas        = w8_ctas_per_sm<NSTAGE, rot, raw_k>();
    GGML_ASSERT(ctas > 0);
    const int pb = w8_splits_env();

    static std::atomic_flag logged = ATOMIC_FLAG_INIT;
    if (!logged.test_and_set()) {
        cudaFuncAttributes attr;
        CUDA_CHECK(cudaFuncGetAttributes(&attr, reinterpret_cast<const void *>(kernel)));
        GGML_LOG_INFO("SJKVARN W8: W=%d (n_q=%d, n_kv=%d) at %d CTA/SM (regs %d, local %zu B, smem %zu B, nstage %d, raw_k %d, rot %d, pb_ov %d)\n",
            W8_WARPS, (int) dst->src[0]->ne[1], ggml_get_op_params_i32(dst, 6), ctas, attr.numRegs, attr.localSizeBytes, shared, NSTAGE, (int) raw_k, (int) rot, pb);
    }

    launch_fattn<D, NCOLS2, NCOLS2>(ctx, dst, kernel, W8_WARPS, shared, UNIT,
        /*need_f16_K=*/false, /*need_f16_V=*/false, /*stream_k=*/false, /*use_sparse=*/false, WARP_SIZE,
        nullptr, nullptr, pb, /*async_kv_preload=*/false, /*sj_fill_kv_splits=*/1);
}

template <bool raw_k>
static int w8_pick_nstage() {
    const int nst = w8_nstage_env();
    if (nst == 4 && w8_ctas_per_sm<4, false, raw_k>() > 0) {
        return 4;
    }
    if (nst == 2) {
        return 2;
    }
    if (nst == 3 && w8_ctas_per_sm<3, false, raw_k>() > 0) {
        return 3;
    }
    // Default: NSTAGE=3 (2 body strips + 2 tq6_0 or 1 q8_0 raw slot in flight) whenever it keeps max CTA/SM occupancy.
    // Query the plain (rot=false) build so fused and unfused rotation always pick the same stage count.
    const int c3 = w8_ctas_per_sm<3, false, raw_k>();
    const int c2 = w8_ctas_per_sm<2, false, raw_k>();
    return (c3 > 0 && c3 >= c2) ? 3 : 2;
}

template <int NSTAGE, bool raw_k>
static void w8_launch_rot_checked(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const bool rot) {
    if (!rot) {
        w8_launch_impl<NSTAGE, false, raw_k>(ctx, dst);
        return;
    }
    // Fused rot=true is bit-exact against unfused rot=false + WHT only when launch_fattn sees the same CTA/SM
    // occupancy (hence identical parallel_blocks / KV splits). Fall back to unfused W8 if they ever differ.
    const int  ctas_rot   = w8_ctas_per_sm<NSTAGE, true,  raw_k>();
    const int  ctas_plain = w8_ctas_per_sm<NSTAGE, false, raw_k>();
    const bool forced     = w8_force_unfused_env();
    if (forced || ctas_rot != ctas_plain) {
        static std::atomic_flag logged_fb = ATOMIC_FLAG_INIT;
        if (!logged_fb.test_and_set()) {
            GGML_LOG_INFO("SJKVARN W8: rot unfused fallback taken (nstage %d, raw_k %d, ctas_rot %d, ctas_plain %d, forced %d)\n",
                NSTAGE, (int) raw_k, ctas_rot, ctas_plain, (int) forced);
        }
        ggml_cuda_flash_attn_ext_sj_kvarn_rot_unfused(ctx, dst, ::ggml_cuda_flash_attn_ext_sj_kvarn_w8);
        return;
    }
    ggml_cuda_pool_alloc<float> q_rot(ctx.pool(), ggml_nelements(dst->src[0]));
    ggml_tensor q2;
    ggml_cuda_sj_kvarn_rot256_q_pass(ctx, dst->src[0], q_rot.ptr, &q2);
    ggml_tensor d2 = *dst;
    d2.src[0] = &q2;
    w8_launch_impl<NSTAGE, true, raw_k>(ctx, &d2);
}

template <bool raw_k>
static void w8_select_nstage(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const bool rot) {
    switch (w8_pick_nstage<raw_k>()) {
        case 4:  w8_launch_rot_checked<4, raw_k>(ctx, dst, rot); break;
        case 3:  w8_launch_rot_checked<3, raw_k>(ctx, dst, rot); break;
        default: w8_launch_rot_checked<2, raw_k>(ctx, dst, rot); break;
    }
}

} // namespace fattn_sj_kvarn_w8

bool ggml_cuda_flash_attn_ext_sj_kvarn_w8_supported(const ggml_tensor * dst) {
    using namespace fattn_sj_kvarn_w8;
    if (w8_mode() == 0 || dst->src[6] == nullptr) {
        return false;
    }
    if (ggml_get_op_params_i32(dst, 5) != ((4 << 8) | 4) || ggml_get_op_params_i32(dst, 7) != GGML_TYPE_F32) {
        return false;
    }
    if (ggml_flash_attn_ext_get_lsa_block(dst) > 0 || ggml_get_op_params_i32(dst, 4) > 0 ||
            dst->src[4] != nullptr || dst->src[7] != nullptr || dst->src[8] != nullptr || dst->src[9] != nullptr ||
            ggml_cuda_fattn_sj_live_records != nullptr) {
        return false;
    }
    if (dst->src[5] && ((dst->src[5]->buffer &&
            ggml_cuda_qsa_resident_is_buft(ggml_backend_buffer_get_type(dst->src[5]->buffer))) ||
            ggml_cuda_qsa_strip_body(dst->src[5]))) {
        return false;
    }
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    if (Q == nullptr || K == nullptr || V == nullptr || mask == nullptr) {
        return false;
    }
    const int64_t n_q = Q->ne[1];
    if (n_q < 7 || n_q > 8 || dst->ne[2] != n_q) {
        return false;
    }
    if (Q->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) ||
            Q->ne[0] != D || K->ne[0] != D || V->ne[0] != D || Q->nb[0] != sizeof(float) ||
            dst->ne[0] != D || dst->ne[1] != Q->ne[2] ||
            Q->ne[3] != 1 || K->ne[3] != 1 || V->ne[3] != 1 || dst->ne[3] != 1) {
        return false;
    }
    if (mask->type != GGML_TYPE_F16 || mask->ne[1] < n_q || mask->ne[2] != 1 || mask->ne[3] != 1) {
        return false;
    }
    const auto ring_type_ok = [](const ggml_type t) {
        return t == GGML_TYPE_F16 || t == GGML_TYPE_Q8_0 || t == GGML_TYPE_TQ6_0;
    };
    if (!ring_type_ok(K->type) || !ring_type_ok(V->type)) {
        return false;
    }
    const int32_t n_kv_pad = ggml_get_op_params_i32(dst, 6);
    if (n_kv_pad <= 0 || n_kv_pad % FATTN_KQ_STRIDE != 0 || mask->ne[0] < n_kv_pad) {
        return false;
    }
    if (((uintptr_t) mask->data) % 16 != 0 || mask->nb[1] % 16 != 0 ||
            ((uintptr_t) Q->data) % 8 != 0 || Q->nb[1] % 16 != 0 || Q->nb[2] % 8 != 0) {
        return false;
    }
    if (K->ne[2] <= 0 || Q->ne[2] != 6 * K->ne[2]) {
        return false;
    }
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }
    const int dev = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[dev].cc;
    return cp_async_available(cc) && turing_mma_available(cc) &&
           w8_shared_bytes(2) <= ggml_cuda_info().devices[dev].smpbo;
}

void ggml_cuda_flash_attn_ext_sj_kvarn_w8(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    using namespace fattn_sj_kvarn_w8;
    const bool rawk = w8_mode() == 1;
    const bool rot  = ggml_cuda_fattn_sj_kvarn_rot(dst);
    if (rawk) {
        w8_select_nstage<true>(ctx, dst, rot);
    } else {
        w8_select_nstage<false>(ctx, dst, rot);
    }
}
