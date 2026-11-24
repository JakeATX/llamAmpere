// This file has been hand-created ([DSPINE4 KQX] SJ-KVaRN 4/4 compact verify tile with cheaper body-strip MMA, widths 7/8).
// Do NOT run generate_cu_files.py over it. Its own TU, so the three variant instances do not slow the other FA TUs.

#include "../fattn-sjkvarn-direct.cuh"

// GGML_SJKVARN_KQX=<1|2|3> (default unset = off). See sj_kvarn_tile::compact_kq2 in fattn-sjkvarn-stream.cuh.
static int ggml_cuda_fattn_sj_kvarn_kqx() {
    static const int v = [] {
        const char * e = getenv("GGML_SJKVARN_KQX");
        const int x = e != nullptr ? atoi(e) : 0;
        const int k = x >= 1 && x <= 3 ? x : 0;
        if (e != nullptr) {
            GGML_LOG_INFO("SJ-KVaRN GGML_SJKVARN_KQX=%s: %s\n", e,
                k == 1 ? "KQ as two fp32 MMA chains (widths 7/8, mode 2)" :
                k == 2 ? "KQ two chains + fp16-accumulate V (widths 7/8, mode 2)" :
                k == 3 ? "fp16-accumulate KQ and V (widths 7/8, mode 2)" : "off (needs 1..3)");
        }
        return k;
    }();
    return v;
}

// The stream kernel asks on a mode-2 compact node (its contract: GQA 6, no LSA list, plain mask). Only the unrotated
// SJ-KVaRN 4/4 body at the 2-strip ring has a KQX instance (the DSpine 8-row verify build: rot 0, nstage 2); anything
// else returns false and keeps the compact tile.
bool ggml_cuda_flash_attn_ext_sj_kvarn_kqx_run(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const bool turbo4_body,
        const bool rot, const int nst) {
    const int k = ggml_cuda_fattn_sj_kvarn_kqx();
    if (k == 0) {
        return false;
    }
    const int n_q = (int) dst->src[0]->ne[1];
    // the stream select maps ring depths 1..3 (and 0 = occupancy pick, resolved by the caller) to 2 strips
    const char * why = turbo4_body ? "turbo4 body" : rot ? "rotated build" : n_q < 7 || n_q > 8 ? "width not 7/8"
                     : dst->src[0]->ne[2] != 6*dst->src[1]->ne[2] ? "GQA not 6" : nst >= 4 ? "ring depth not 2" : nullptr;
    if (why != nullptr) {
        static std::atomic_flag warned = ATOMIC_FLAG_INIT;
        if (!warned.test_and_set()) {
            GGML_LOG_WARN("%s: GGML_SJKVARN_KQX=%d set but not taken on a node (%s): compact tile\n", __func__, k, why);
        }
        return false;
    }
    static std::atomic<uint32_t> logged{0};
    const uint32_t bit = 1u << n_q;
    if (!(logged.fetch_or(bit) & bit)) {
        GGML_LOG_INFO("%s: SJ-KVaRN KQX=%d compact tile on at width %d (6 warps, mode 2, nstage 2)\n", __func__, k, n_q);
    }
    switch (k) {
        case 1:  ggml_cuda_flash_attn_ext_sj_kvarn_stream_launch<6, 2, false, false, sj_kvarn_tile::compact_kq2>(ctx, dst, 6, 1, 0); break;
        case 2:  ggml_cuda_flash_attn_ext_sj_kvarn_stream_launch<6, 2, false, false, sj_kvarn_tile::compact_hv >(ctx, dst, 6, 1, 0); break;
        default: ggml_cuda_flash_attn_ext_sj_kvarn_stream_launch<6, 2, false, false, sj_kvarn_tile::compact_h16>(ctx, dst, 6, 1, 0); break;
    }
    return true;
}
