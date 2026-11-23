// This file has been hand-created ([DSPINE4 VERIFY16] SJ-KVaRN 4/4 compact verify tile, widths 9..16). Do NOT run
// generate_cu_files.py over it. Its own TU, so the 8/10/12-warp compact instances do not slow the direct TU's build.

#include "../fattn-sjkvarn-direct.cuh"
#include "../qsa-resident.cuh"

// GGML_SJKVARN_COMPACT16=1 and a node the compact tile serves at 9..16 rows. The caller (fattn.cu) has already checked
// the 4/4 body (not low-bit) and that ggml_cuda_fattn_sj_kvarn_rowsplit takes the node (no LSA inputs, one plain F16 or
// packed mask row per query, the decode kernel contract), so only the compact tile's own limits are left here.
bool ggml_cuda_flash_attn_ext_sj_kvarn_compact16_ok(const ggml_tensor * dst) {
    if (!ggml_cuda_fattn_sj_kvarn_compact16_env()) {
        return false;
    }
    const ggml_tensor * Q = dst->src[0];
    const int64_t n_q = Q->ne[1];
    if (n_q <= 8 || n_q > 16 || Q->ne[3] != 1 || dst->src[4] != nullptr) {
        return false;
    }
    // the stream kernel's opt-in walks (live records / strips / tile skip) and the debug STABLE launch: no compact build
    static const bool walks = [] {
        for (const char * n : { "GGML_SJKVARN_TILE_SKIP", "GGML_SJKVARN_LIVE_STRIPS", "GGML_SJKVARN_LIVE_RECORDS", "GGML_SJKVARN_DIRECT_STABLE" }) {
            const char * e = getenv(n);
            if (e != nullptr && atoi(e) != 0) {
                return true;
            }
        }
        return false;
    }();
    if (walks || ggml_get_op_params_i32(dst, 7) == GGML_TYPE_I16) {
        return false;
    }
    // LSA resident body (strip slots): the stream kernel resolves slots only on the rows tile
    if (dst->src[5] && dst->src[5]->buffer && ggml_cuda_qsa_resident_is_buft(ggml_backend_buffer_get_type(dst->src[5]->buffer))) {
        return false;
    }
    float logit_softcap = 0.0f;
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (logit_softcap != 0.0f || !cp_async_available(cc) || !turing_mma_available(cc)) {
        return false;
    }
    // GQA 6, no per-head mask, GGML_SJKVARN_COMPACT_ALL not 0 (no live-record list is set outside the stream kernel)
    return ggml_cuda_fattn_sj_kvarn_compact_mode_for(dst, false) != 0;
}

void ggml_cuda_flash_attn_ext_sj_kvarn_compact16_run(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const bool turbo4_body,
        const bool rot, const int nstage_env) {
    GGML_ASSERT(!(turbo4_body && rot));
    if (turbo4_body) {
        ggml_cuda_fattn_sj_kvarn_stream_compact16<true, false>(ctx, dst, nstage_env);
    } else if (rot) {
        ggml_cuda_fattn_sj_kvarn_stream_compact16<false, true>(ctx, dst, nstage_env);
    } else {
        ggml_cuda_fattn_sj_kvarn_stream_compact16<false, false>(ctx, dst, nstage_env);
    }
}
