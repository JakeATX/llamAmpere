#include "fattn-mma-turbo-d128-tq5t4.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-mma-turbo.cuh"

#include <atomic>
#include <cinttypes>
#include <cstdlib>
#include <cstring>

namespace {

static bool d128_tq5t4_enabled() {
    static const bool enabled = [] {
        const char * fused = std::getenv("GGML_TURBO_MMA_FUSED");
        if (fused != nullptr && fused[0] == '0') {
            return false;
        }
        const char * e = std::getenv("GGML_CUDA_FATTN_D128_TQ5T4");
        if (e == nullptr || *e == '\0') {
            return false;
        }
        char * end = nullptr;
        const long v = std::strtol(e, &end, 10);
        if (end != e && *end == '\0' && (v == 0 || v == 1)) {
            return v == 1;
        }
        GGML_LOG_WARN("%s: invalid GGML_CUDA_FATTN_D128_TQ5T4='%s' (expected 0 or 1); disabling\n", __func__, e);
        return false;
    }();
    return enabled;
}

static bool d128_tq5t4_use_gqa(const ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    std::memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));

    bool use_gqa_opt = mask != nullptr && mask->ne[2] == 1 && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
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
    return use_gqa_opt;
}

template <int ncols1, int ncols2>
static void d128_tq5t4_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst, int bit_idx) {
    static std::atomic<uint32_t> logged{0};
    const uint32_t bit = 1u << (bit_idx & 31);
    if (!(logged.fetch_or(bit, std::memory_order_relaxed) & bit)) {
        const ggml_tensor * Q = dst->src[0];
        const ggml_tensor * K = dst->src[1];
        GGML_LOG_INFO("FATTN D128 TQ5/TURBO4: fused MMA on (ncols1=%d ncols2=%d n_q=%" PRId64 " gqa=%" PRId64 " n_kv=%" PRId64 ")\n",
                ncols1, ncols2, Q->ne[1], Q->ne[2] / K->ne[2], K->ne[1]);
    }
    ggml_cuda_flash_attn_ext_mma_turbo_case<128, 128, ncols1, ncols2, GGML_TYPE_TQ5_0, GGML_TYPE_TURBO4_0>(ctx, dst);
}

} // namespace

bool ggml_cuda_flash_attn_ext_d128_tq5t4_try(ggml_backend_cuda_context * ctx, ggml_tensor * dst, int device) {
    if (!d128_tq5t4_enabled()) {
        return false;
    }
    if (dst->src[6] != nullptr || dst->src[8] != nullptr ||
            ggml_flash_attn_ext_get_kv_rows(dst) != nullptr ||
            ggml_flash_attn_ext_get_lse(dst) != nullptr ||
            ggml_get_op_params_i32(dst, 4) > 0) {
        return false;
    }

    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    if (Q->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            Q->ne[0] != 128 || V->ne[0] != 128 || Q->ne[1] <= 0 || Q->ne[2] <= 0 ||
            K->ne[1] <= 0 || K->ne[2] <= 0 || V->ne[1] != K->ne[1] || V->ne[2] != K->ne[2] ||
            K->type != GGML_TYPE_TQ5_0 || V->type != GGML_TYPE_TURBO4_0) {
        return false;
    }
    if (Q->nb[1] % sizeof(float2) != 0 || Q->nb[2] % sizeof(float2) != 0) {
        return false;
    }
    if (mask != nullptr) {
        if ((mask->type != GGML_TYPE_F16 && mask->type != GGML_TYPE_I32) || mask->ne[1] < Q->ne[1]) {
            return false;
        }
        if (mask->type == GGML_TYPE_I32 && (mask->nb[1] & 3) != 0) {
            return false;
        }
    }

    const int cc = ggml_cuda_info().devices[device].cc;
    if (!turing_mma_available(cc) || Q->ne[2] % K->ne[2] != 0) {
        return false;
    }

    const int64_t n_q       = Q->ne[1];
    const int64_t gqa_ratio = Q->ne[2] / K->ne[2];
    const bool    use_gqa   = d128_tq5t4_use_gqa(dst);
    // Turing builds reject ncols1 * ncols2 > 32 in flash_attn_ext_f16; non-early_v sinks read sinks_f[0..ncols2-1].
    const bool    ampere_ok = ampere_mma_available(cc);
    const bool    has_sinks = dst->src[4] != nullptr;

    int ncols1  = 0;
    int bit_idx = -1;
    if (use_gqa && gqa_ratio > 4 && (!has_sinks || gqa_ratio % 8 == 0)) {
        if (n_q <= 2)                    { ncols1 = 2; bit_idx = 0; }
        else if (n_q <= 4)               { ncols1 = 4; bit_idx = 1; }
        else if (n_q <= 8 && ampere_ok)  { ncols1 = 8; bit_idx = 2; }
        else                             { return false; }
    } else if (use_gqa && gqa_ratio > 2 && (!has_sinks || gqa_ratio % 4 == 0)) {
        if (n_q <= 2)                    { ncols1 =  2; bit_idx = 3; }
        else if (n_q <= 4)               { ncols1 =  4; bit_idx = 4; }
        else if (n_q <= 8)               { ncols1 =  8; bit_idx = 5; }
        else if (n_q <= 16 && ampere_ok) { ncols1 = 16; bit_idx = 6; }
        else                             { return false; }
    } else if (use_gqa && gqa_ratio > 1 && (!has_sinks || gqa_ratio % 2 == 0)) {
        if (n_q <= 4)                    { ncols1 = 4; bit_idx = 7; }
        else if (has_sinks && n_q <= 16) { ncols1 = 8; bit_idx = 8; }
        else                             { return false; }
    } else if (n_q <= 16) {
        ncols1  = 8;
        bit_idx = 8;
    } else {
        return false;
    }

    // Multi-sequence mask scan in launch_fattn reads whole ncols1-row tiles from mask.
    const bool scan_mask = mask != nullptr && mask->ne[2] == 1 && K->ne[1] % FATTN_KQ_STRIDE == 0 && Q->ne[3] > 1;
    if (scan_mask && mask->ne[1] < ((n_q + ncols1 - 1) / ncols1) * ncols1) {
        return false;
    }

    if (ctx != nullptr) {
        switch (bit_idx) {
            case 0: d128_tq5t4_launch< 2, 8>(*ctx, dst, 0); break;
            case 1: d128_tq5t4_launch< 4, 8>(*ctx, dst, 1); break;
            case 2: d128_tq5t4_launch< 8, 8>(*ctx, dst, 2); break;
            case 3: d128_tq5t4_launch< 2, 4>(*ctx, dst, 3); break;
            case 4: d128_tq5t4_launch< 4, 4>(*ctx, dst, 4); break;
            case 5: d128_tq5t4_launch< 8, 4>(*ctx, dst, 5); break;
            case 6: d128_tq5t4_launch<16, 4>(*ctx, dst, 6); break;
            case 7: d128_tq5t4_launch< 4, 2>(*ctx, dst, 7); break;
            case 8: d128_tq5t4_launch< 8, 1>(*ctx, dst, 8); break;
            default: GGML_ABORT("unreachable");
        }
    }
    return true;
}
