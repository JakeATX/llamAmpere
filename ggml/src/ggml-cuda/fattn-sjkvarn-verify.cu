#include "fattn-sjkvarn-verify.cuh"
#include "fattn-sjkvarn-stream-pv16.cuh"
#include "fattn-sjkvarn-direct-verify.cuh"

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace {

static bool sj_verify_env_bool(const char * name, const bool default_val = false) {
    const char * e = getenv(name);
    if (e == nullptr || *e == '\0') {
        return default_val;
    }
    return atoi(e) != 0;
}

static bool sj_verify_common_ok(const ggml_tensor * dst) {
    if (dst->src[5] == nullptr || dst->src[6] == nullptr) {
        return false;
    }
    if (ggml_get_op_params_i32(dst, 5) != ((4 << 8) | 4)) {
        return false;
    }
    // Scalar 4/4 sealed bodies only (excludes TURBO4_0 and I16 trellis bodies).
    if (ggml_get_op_params_i32(dst, 7) != GGML_TYPE_F32) {
        return false;
    }
    // Leave LSA sparse walks / virtual keys / resident card strip bodies on baseline stream path.
    if (ggml_flash_attn_ext_get_lsa_block(dst) > 0 || dst->src[7] != nullptr || dst->src[8] != nullptr) {
        return false;
    }
    if (dst->src[5]->buffer != nullptr &&
        ggml_cuda_qsa_resident_is_buft(ggml_backend_buffer_get_type(dst->src[5]->buffer))) {
        return false;
    }
    if (sj_verify_env_bool("GGML_SJKVARN_LIVE_RECORDS") ||
        sj_verify_env_bool("GGML_SJKVARN_LIVE_STRIPS") ||
        sj_verify_env_bool("GGML_SJKVARN_TILE_SKIP")) {
        return false;
    }

    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    if (Q->ne[0] != 256 || K->ne[0] != 256 || V->ne[0] != 256 || Q->ne[3] != 1) {
        return false;
    }
    if (Q->type != GGML_TYPE_F32 || Q->nb[0] != sizeof(float) || Q->nb[1] % 16 != 0) {
        return false;
    }
    if (K->type != GGML_TYPE_F16 && K->type != GGML_TYPE_Q8_0 && K->type != GGML_TYPE_TQ6_0) {
        return false;
    }
    if (V->type != GGML_TYPE_F16 && V->type != GGML_TYPE_Q8_0 && V->type != GGML_TYPE_TQ6_0) {
        return false;
    }
    if (mask == nullptr || mask->ne[3] != 1) {
        return false;
    }
    if (mask->type == GGML_TYPE_F16) {
        if (((uintptr_t) mask->data) % 16 != 0 || mask->nb[1] % 16 != 0) {
            return false;
        }
        if (mask->ne[2] != 1 && mask->ne[2] != Q->ne[2]) {
            return false;
        }
    } else if (mask->type == GGML_TYPE_I32) {
        if (mask->ne[2] != 1 || (mask->nb[1] & 3) != 0) {
            return false;
        }
    } else {
        return false;
    }
    if (Q->ne[2] % K->ne[2] != 0) {
        return false;
    }
    const int gqa = (int) (Q->ne[2] / K->ne[2]);
    if (gqa < 1 || gqa > fattn_sj_kvarn_stream::NCOLS2) {
        return false;
    }
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }
    if (ggml_get_op_params_i32(dst, 6) % FATTN_KQ_STRIDE != 0) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!ampere_mma_available(cc) || !cp_async_available(cc)) {
        return false;
    }
    return true;
}

template <bool rot>
static void run_pv16_rot(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    using namespace fattn_sj_kvarn_stream_pv16;
    static const bool kw8        = getenv("GGML_SJKVARN_DIRECT_KW8") != nullptr;
    static const bool no_compact = sj_verify_env_bool("GGML_SJKVARN_VERIFY_NO_COMPACT");
    const int n_q = (int) dst->src[0]->ne[1];
    const int gqa = (int) (dst->src[0]->ne[2] / dst->src[1]->ne[2]);
    GGML_ASSERT(n_q >= 1 && n_q <= 8);

    if (gqa == 6 && n_q >= 4 && !no_compact) {
        const int W = (6 * n_q + 7) / 8;
        switch (W) {
            case 3:  pv16_launch_select<3, rot, sj_kvarn_tile::compact>(ctx, dst, 3, 1); break;
            case 4:  pv16_launch_select<4, rot, sj_kvarn_tile::compact>(ctx, dst, 4, 1); break;
            case 5:  pv16_launch_select<5, rot, sj_kvarn_tile::compact>(ctx, dst, 5, 1); break;
            default: pv16_launch_select<6, rot, sj_kvarn_tile::compact>(ctx, dst, 6, 1); break;
        }
        return;
    }

    const int KW = n_q <= 4 ? (kw8 ? 8 / n_q : 4 / n_q) : 1;
    if (n_q * KW <= 4) {
        pv16_launch_select<4, rot, sj_kvarn_tile::rows>(ctx, dst, n_q, KW);
    } else if (n_q == 5) {
        pv16_launch_select<5, rot, sj_kvarn_tile::rows>(ctx, dst, 5, 1);
    } else if (n_q == 6) {
        pv16_launch_select<6, rot, sj_kvarn_tile::rows>(ctx, dst, 6, 1);
    } else {
        pv16_launch_select<8, rot, sj_kvarn_tile::rows>(ctx, dst, n_q, KW);
    }
}

template <bool rot>
static void run_direct_verify_rot(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    using namespace fattn_sj_kvarn_direct_verify;
    static const bool no_compact = sj_verify_env_bool("GGML_SJKVARN_VERIFY_NO_COMPACT");
    const int n_q = (int) dst->src[0]->ne[1];
    const int gqa = (int) (dst->src[0]->ne[2] / dst->src[1]->ne[2]);
    GGML_ASSERT(n_q >= 1 && n_q <= 64);

    if (gqa == 6 && n_q >= 4 && !no_compact) {
        const int W = std::min(6, (6 * n_q + 7) / 8);
        switch (W) {
            case 3:  direct_verify_launch_select<3, 8, rot, sj_kvarn_tile::compact>(ctx, dst, 3, 1); break;
            case 4:  direct_verify_launch_select<4, 8, rot, sj_kvarn_tile::compact>(ctx, dst, 4, 1); break;
            case 5:  direct_verify_launch_select<5, 8, rot, sj_kvarn_tile::compact>(ctx, dst, 5, 1); break;
            default: direct_verify_launch_select<6, 8, rot, sj_kvarn_tile::compact>(ctx, dst, 6, 1); break;
        }
        return;
    }

    if (n_q > 8) {
        direct_verify_launch_select<4, 4, rot, sj_kvarn_tile::rows>(ctx, dst, 4, 1);
        return;
    }

    const int KW = n_q <= 2 ? 4 / n_q : 1;
    if (n_q * KW <= 4) {
        direct_verify_launch_select<4, 8, rot, sj_kvarn_tile::rows>(ctx, dst, n_q, KW);
    } else if (n_q == 5) {
        direct_verify_launch_select<5, 8, rot, sj_kvarn_tile::rows>(ctx, dst, 5, 1);
    } else if (n_q == 6) {
        direct_verify_launch_select<6, 8, rot, sj_kvarn_tile::rows>(ctx, dst, 6, 1);
    } else {
        direct_verify_launch_select<8, 8, rot, sj_kvarn_tile::rows>(ctx, dst, n_q, 1);
    }
}

} // namespace

bool ggml_cuda_flash_attn_ext_sj_kvarn_stream_pv16_applies(const ggml_tensor * dst) {
    static const bool enabled = sj_verify_env_bool("GGML_SJKVARN_STREAM_PV16", false);
    if (!enabled) {
        return false;
    }
    const ggml_tensor * Q = dst->src[0];
    if (Q->ne[1] < 1 || Q->ne[1] > 8) {
        return false;
    }
    return sj_verify_common_ok(dst);
}

void ggml_cuda_flash_attn_ext_sj_kvarn_stream_pv16_run(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (ggml_cuda_fattn_sj_kvarn_rot(dst)) {
        ggml_cuda_pool_alloc<float> q_rot(ctx.pool(), ggml_nelements(dst->src[0]));
        ggml_tensor q2;
        ggml_cuda_sj_kvarn_rot256_q_pass(ctx, dst->src[0], q_rot.ptr, &q2);
        ggml_tensor d2 = *dst;
        d2.src[0] = &q2;
        run_pv16_rot<true>(ctx, &d2);
        return;
    }
    run_pv16_rot<false>(ctx, dst);
}

bool ggml_cuda_flash_attn_ext_sj_kvarn_direct_verify_applies(const ggml_tensor * dst) {
    static const int mode = [] {
        const char * e = getenv("GGML_SJKVARN_DIRECT_VERIFY");
        return (e != nullptr && *e != '\0') ? atoi(e) : 0; // default OFF (0)
    }();
    if (mode == 0) {
        return false;
    }
    static const int min_t_env = [] {
        const char * e = getenv("GGML_SJKVARN_DIRECT_VERIFY_MIN_T");
        return (e != nullptr && *e != '\0') ? std::max(1, atoi(e)) : 4;
    }();
    static const int max_t_env = [] {
        const char * e = getenv("GGML_SJKVARN_DIRECT_VERIFY_MAX_T");
        return (e != nullptr && *e != '\0') ? std::min(64, atoi(e)) : 64;
    }();
    const int min_t = mode >= 2 ? 1 : min_t_env;
    const ggml_tensor * Q = dst->src[0];
    if (Q->ne[1] < min_t || Q->ne[1] > max_t_env) {
        return false;
    }
    return sj_verify_common_ok(dst);
}

void ggml_cuda_flash_attn_ext_sj_kvarn_direct_verify_run(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (ggml_cuda_fattn_sj_kvarn_rot(dst)) {
        ggml_cuda_pool_alloc<float> q_rot(ctx.pool(), ggml_nelements(dst->src[0]));
        ggml_tensor q2;
        ggml_cuda_sj_kvarn_rot256_q_pass(ctx, dst->src[0], q_rot.ptr, &q2);
        ggml_tensor d2 = *dst;
        d2.src[0] = &q2;
        run_direct_verify_rot<true>(ctx, &d2);
        return;
    }
    run_direct_verify_rot<false>(ctx, dst);
}
