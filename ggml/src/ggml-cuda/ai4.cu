// AI4 (Ampere INT4, formerly D2) CUDA backend entry: reference and fast-path matmul dispatch.
#include "ai4-mm.cuh"
#include "ai4.cuh"
#include "ai4-kern.cuh"
#include "ai4-ampere.cuh"
#include "rpb-knobs.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

// Kernel selection (read once per process), for A/B runs:
//   GGML_CUDA_AI4_REF=1        reference path: reconstruct to f32 + cuBLAS SGEMM (the D-003 KL path)
//   GGML_CUDA_AI4_REF_SLAB_MIB=n   cap in MiB of that f32 buffer (default 256). Rows are reconstructed and multiplied
//                             in slabs of a multiple of 16. 0 or negative: one slab, the whole matrix (the old buffer).
//   GGML_CUDA_AI4_REF_SLAB_ROWS=n  test only. Slab height in rows; a positive value overrides the MiB cap. Rounded
//                             down to a multiple of 16 (the storage unit). Not a runtime tuning knob.
//   GGML_CUDA_AI4_BN128_MIN=n  N at which the GEMM switches from 64- to 128-wide token tiles (default 512)
//   GGML_CUDA_AI4_MMV_STAGES=2|3|4  cp.async ring depth of the N <= 8 kernel (default 2: AI4@s2 is the configuration
//                             that passed the D-008 ship gate and was fastest of 2/3/4 at verify widths 5-8 in D-007)
//   GGML_CUDA_AI4_MMV_XSMEM=1  N <= 8 kernel stages the activations in shared memory (first version, 3 stages) instead
//                             of reading them into registers
//   GGML_CUDA_AI4_MMV_WARPS=2|4|8   warps per block of the stages-2 register kernel (default 4)
//   GGML_CUDA_AI4_MMV_MINB=2|4|6|8  launch_bounds min blocks of that kernel (default 6).
//                             8 warps with min blocks 8 does not fit sm_86 and uses 4 and 6.
//                             Stages 3/4 and XSMEM ignore both and keep 4 warps.
//   GGML_CUDA_AI4_AMPERE_MMV=1   measured width 9-32 family on sm_80/sm_86 (default off)
//   GGML_CUDA_AI4_AMPERE_MMV_MIN_N/MAX_N   inclusive per-card tuning range within 1-64 (defaults 9/32)
//   GGML_CUDA_AI4_AMPERE_GEMM    retired; keeps baseline prefill after measured regressions
//   GGML_CUDA_AI4_PREFILL_V2=1  experimental decode-once dense prefill, N >= 512 (default off)
//   GGML_CUDA_AI23_MMVR=1    rate-2/3 decode with fixed stage offsets (default off)
//   GGML_CUDA_AI23_TUNED=1   select measured rate-2/3 decode and tile defaults (default off)
//   GGML_CUDA_AI23_MMVR_STAGES=4|8  deeper dense fixed-K four-warp weight prefetch (default 2)
//   GGML_CUDA_AI23_MMVR_UNROLL=2|4  unroll dense decode without a deeper shared buffer (default 1)
//   GGML_CUDA_AI23_SMALL_SPLIT=2|4  split small-width fixed-K dense products (default off)
//   GGML_CUDA_AI23_FUSED_X=1  quantize activations inside direct one-token expert blocks (default off)
//   GGML_CUDA_AI23_DIRECT=1..8  skip expert grouping through this token width in MMVR (default off)
//   GGML_CUDA_AI23_DENSE_TILE=1..29  dense shared-activation tile controls for 9..64 tokens (default off)
//                             22/23 swap quantized activation halves to avoid LDS64 bank conflicts
//   GGML_CUDA_AI23_RESOURCE=1   print CUDA kernel resource and occupancy queries once (diagnostics only)
//   GGML_CUDA_AI23_SPLIT_K=2|4|8|16  split dense K chunks into partial tiles for 9..64 tokens (default off)
//   GGML_CUDA_AI23_NARROW_WARPS=2|4|8  override narrow dense K parallelism (default automatic)
//   GGML_CUDA_AI23_NARROW=1|2|3   one-slab dense tiles for 9..64 tokens at rates 2 and 3 (default off)
//                             2 fixes K/loop slots; 3 also uses whole-row rate-2 decode
// Existing switches also read pre-rename GGML_CUDA_D2_* spellings when the AI4 name is unset. The new Ampere switches have no legacy aliases.
//   GGML_CUDA_AI6_REF=1        AI6_S takes the rate-k reference path (reconstruct + SGEMM) instead of its rate-6 IMMA
//                             kernels (an A/B check of those kernels, like GGML_CUDA_AI5_REF for rate 5)
static int ai4_env_int(const char * name, const char * legacy_name, int def) {
    const char * e = getenv(name);
    if (!e && legacy_name) {
        e = getenv(legacy_name);
    }
    return e ? atoi(e) : def;
}
static bool ai4_ai23_tuned() {
    static const bool v = ai4_env_int("GGML_CUDA_AI23_TUNED", nullptr, 0) != 0;
    return v;
}
static bool ai4_ref() {
    static const bool v = ai4_env_int("GGML_CUDA_AI4_REF", "GGML_CUDA_D2_REF", 0) != 0;
    return v;
}
static bool ai4_ampere_mmv(int n) {
    static const bool v = ai4_env_int("GGML_CUDA_AI4_AMPERE_MMV", nullptr, 0) != 0;
    static const rpb_ai4_ampere_range range = [] {
        const auto r = rpb_ai4_ampere_resolve(getenv("GGML_CUDA_AI4_AMPERE_MMV_MIN_N"), getenv("GGML_CUDA_AI4_AMPERE_MMV_MAX_N"));
        if (r.rejected) {
            GGML_LOG_WARN("invalid GGML_CUDA_AI4_AMPERE_MMV_MIN_N/MAX_N; using 9/32 (expected 1 <= min <= max <= 64)\n");
        }
        return r;
    }();
    return v && range.contains(n);
}
static bool ai4_prefill_v2() {
    static const bool v = ai4_env_int("GGML_CUDA_AI4_PREFILL_V2", nullptr, 0) != 0;
    return v;
}
static int ai4_ampere_sms(const ggml_backend_cuda_context & ctx) {
    const auto & dev = ggml_cuda_info().devices[ctx.device];
    return dev.cc == 800 || dev.cc == 860 ? dev.nsm : 0;
}
static int ai4_ampere_bn(int N) {
    return N <= 8 ? 8 : N <= 16 ? 16 : N <= 32 ? 32 : 64;
}

template <int BN, int LAY, int XSH, bool ID, int WARPS, int KT>
static void ai23_launch_narrow_kt(const uint8_t * W, const int8_t * q, const float * s, float * Y,
                                 int M, int N, int K, cudaStream_t stream, const ai4k::idmap & idm, int max_tiles, bool perm) {
    if constexpr (ai4k::lay_rate<LAY>() == 2) {
        if (perm) {
            ai4k::launch_ampere_mmv<BN, LAY, XSH, ID, WARPS, KT, true>(W, q, s, Y, M, M, N, K, stream, idm, max_tiles);
            return;
        }
    }
    ai4k::launch_ampere_mmv<BN, LAY, XSH, ID, WARPS, KT>(W, q, s, Y, M, M, N, K, stream, idm, max_tiles);
}

template <int BN, int LAY, int XSH, bool ID, int WARPS>
static void ai4_launch_ampere_w(const uint8_t * W, const int8_t * q, const float * s, float * Y,
                               int M, int N, int K, cudaStream_t stream, const ai4k::idmap & idm, int max_tiles) {
    if constexpr (ai4k::lay_rate<LAY>() == 2 || ai4k::lay_rate<LAY>() == 3) {
        static const int narrow = ai4_env_int("GGML_CUDA_AI23_NARROW", nullptr, 0);
        if (narrow >= 2 && K == 5120) {
            ai23_launch_narrow_kt<BN, LAY, XSH, ID, WARPS, 5120>(W, q, s, Y, M, N, K, stream, idm, max_tiles, narrow >= 3);
            return;
        }
        if (narrow >= 2 && K == 17408) {
            ai23_launch_narrow_kt<BN, LAY, XSH, ID, WARPS, 17408>(W, q, s, Y, M, N, K, stream, idm, max_tiles, narrow >= 3);
            return;
        }
    }
    ai4k::launch_ampere_mmv<BN, LAY, XSH, ID, WARPS>(W, q, s, Y, M, M, N, K, stream, idm, max_tiles);
}

template <int BN, int LAY, int XSH, bool ID>
static void ai4_launch_ampere_mmv(const uint8_t * W, const int8_t * q, const float * s, float * Y,
                                  int M, int N, int K, int sms, cudaStream_t stream,
                                  const ai4k::idmap & idm, int max_tiles) {
    if constexpr (ai4k::lay_rate<LAY>() == 2 || ai4k::lay_rate<LAY>() == 3) {
        static const int warps = ai4_env_int("GGML_CUDA_AI23_NARROW_WARPS", nullptr, 0);
        if (warps == 8) {
            ai4_launch_ampere_w<BN, LAY, XSH, ID, 8>(W, q, s, Y, M, N, K, stream, idm, max_tiles);
            return;
        }
        if (warps == 4) {
            ai4_launch_ampere_w<BN, LAY, XSH, ID, 4>(W, q, s, Y, M, N, K, stream, idm, max_tiles);
            return;
        }
        if (warps == 2) {
            ai4_launch_ampere_w<BN, LAY, XSH, ID, 2>(W, q, s, Y, M, N, K, stream, idm, max_tiles);
            return;
        }
    }
    if (sms >= 80 && BN == 8 && K >= 1024) {
        ai4_launch_ampere_w<BN, LAY, XSH, ID, 8>(W, q, s, Y, M, N, K, stream, idm, max_tiles);
    } else if (sms > 28 && BN <= 32) {
        ai4_launch_ampere_w<BN, LAY, XSH, ID, 4>(W, q, s, Y, M, N, K, stream, idm, max_tiles);
    } else {
        ai4_launch_ampere_w<BN, LAY, XSH, ID, 2>(W, q, s, Y, M, N, K, stream, idm, max_tiles);
    }
}

template <int LAY, int XSH, bool ID>
static void ai4_ampere_width(const uint8_t * W, const int8_t * q, const float * s, float * Y,
                             int M, int N, int K, int bn, int sms, cudaStream_t stream,
                             const ai4k::idmap & idm = ai4k::idmap{}, int max_tiles = 1) {
    switch (bn) {
        case 8:  ai4_launch_ampere_mmv<8,  LAY, XSH, ID>(W, q, s, Y, M, N, K, sms, stream, idm, max_tiles); break;
        case 16: ai4_launch_ampere_mmv<16, LAY, XSH, ID>(W, q, s, Y, M, N, K, sms, stream, idm, max_tiles); break;
        case 32: ai4_launch_ampere_mmv<32, LAY, XSH, ID>(W, q, s, Y, M, N, K, sms, stream, idm, max_tiles); break;
        default: ai4_launch_ampere_mmv<64, LAY, XSH, ID>(W, q, s, Y, M, N, K, sms, stream, idm, max_tiles); break;
    }
}

static int ai4_bn128_min() {
    static const int v = ai4_env_int("GGML_CUDA_AI4_BN128_MIN", "GGML_CUDA_D2_BN128_MIN", 512);
    return v;
}
static int ai4_mmv_stages() {
    static const int v = ai4_env_int("GGML_CUDA_AI4_MMV_STAGES", "GGML_CUDA_D2_MMV_STAGES", 2);
    return v;
}
static bool ai4_mmv_xsmem() {
    static const bool v = ai4_env_int("GGML_CUDA_AI4_MMV_XSMEM", "GGML_CUDA_D2_MMV_XSMEM", 0) != 0;
    return v;
}

static_assert(ai4k::mmv_stage_bytes<true, ai4k::LAY_SB>() == rpb_ai4_stage_bytes(true, false), "ai4 stage bytes");
static_assert(ai4k::mmv_stage_bytes<true, ai4k::LAY_U5>() == rpb_ai4_stage_bytes(true, true), "ai4 k5 stage bytes");
static_assert(ai4k::mmv_stage_bytes<false, ai4k::LAY_S5>() == rpb_ai4_stage_bytes(false, true), "ai4 xsmem k5 stage bytes");
static_assert(ai4k::mmv_smem_bytes<2, true, ai4k::LAY_SB, 4>() == rpb_ai4_dynamic_smem(4, 2, true, false), "ai4 default dynamic smem");

// Read once. A value outside the set warns once and uses the default. Unset is 4 and 6.
static rpb_ai4_mmv ai4_mmv_knobs() {
    static const rpb_ai4_mmv v = [] {
        const char * we = getenv("GGML_CUDA_AI4_MMV_WARPS");
        const char * me = getenv("GGML_CUDA_AI4_MMV_MINB");
        char wb[64];
        char mb[64];
        wb[0] = '\0';
        mb[0] = '\0';
        const char * wt = nullptr;
        const char * mt = nullptr;
        if (we != nullptr) {
            std::strncpy(wb, we, sizeof(wb) - 1);
            wb[sizeof(wb) - 1] = '\0';
            wt = wb;
        }
        if (me != nullptr) {
            std::strncpy(mb, me, sizeof(mb) - 1);
            mb[sizeof(mb) - 1] = '\0';
            mt = mb;
        }
        rpb_ai4_mmv r = rpb_ai4_resolve(wt, mt);
        if (r.rejected_warps) {
            GGML_LOG_WARN("GGML_CUDA_AI4_MMV_WARPS='%s' is not one of 2, 4, 8; using %d\n",
                          wb, rpb_ai4_warps_default);
        }
        if (r.rejected_minb) {
            GGML_LOG_WARN("GGML_CUDA_AI4_MMV_MINB='%s' is not one of 2, 4, 6, 8; using %d\n",
                          mb, rpb_ai4_minb_default);
        }
        // k5 is the larger stage. A pair that fails there fails every layout.
        if (!r.rejected_warps && !r.rejected_minb && !rpb_ai4_mmv_ok(r.warps, r.minb, 2, true, true)) {
            GGML_LOG_WARN("GGML_CUDA_AI4_MMV_WARPS=%d MINB=%d exceeds sm_86 launch_bounds (48 warps, 1536 threads per SM); using %d and %d\n",
                          r.warps, r.minb, rpb_ai4_warps_default, rpb_ai4_minb_default);
            r.warps = rpb_ai4_warps_default;
            r.minb  = rpb_ai4_minb_default;
        }
        return r;
    }();
    return v;
}

// Stages 3/4 and the shared-memory activation kernel are not in the sweep. One warning.
static void ai4_mmv_warn_ignored() {
    const rpb_ai4_mmv k = ai4_mmv_knobs();
    if (k.warps == rpb_ai4_warps_default && k.minb == rpb_ai4_minb_default) {
        return;
    }
    static bool warned = false;
    if (!warned) {
        warned = true;
        GGML_LOG_WARN("GGML_CUDA_AI4_MMV_WARPS/MINB apply only to the stages-2 register kernel; this launch keeps warps %d min blocks %d\n",
                      rpb_ai4_warps_default, rpb_ai4_minb_default);
    }
}

// Pre-instantiated (warps, min blocks) pairs of launch_mmv<2, MINB, XREG=true, ..., WARPS>.
// The 4/6 pair is not in this list: callers use the historical launch_mmv<2, 6, ...> for it.
template <int LAY, int XSH, bool ID>
static void ai4_launch_mmv_tuned(const uint8_t * W, const int8_t * q, const float * s, float * Y,
                                int M, int N, int K, cudaStream_t stream, int warps, int minb,
                                const ai4k::idmap & idm = ai4k::idmap{}, int max_tiles = 1) {
#define AI4_MMV_GO(WV, MB) \
    do { ai4k::launch_mmv<2, MB, true, LAY, XSH, ID, WV>(W, q, s, Y, M, M, N, K, stream, idm, max_tiles); return; } while (0)
    if (warps == 2 && minb == 2) AI4_MMV_GO(2, 2);
    if (warps == 2 && minb == 4) AI4_MMV_GO(2, 4);
    if (warps == 2 && minb == 6) AI4_MMV_GO(2, 6);
    if (warps == 2 && minb == 8) AI4_MMV_GO(2, 8);
    if (warps == 4 && minb == 2) AI4_MMV_GO(4, 2);
    if (warps == 4 && minb == 4) AI4_MMV_GO(4, 4);
    if (warps == 4 && minb == 8) AI4_MMV_GO(4, 8);
    if (warps == 8 && minb == 2) AI4_MMV_GO(8, 2);
    if (warps == 8 && minb == 4) AI4_MMV_GO(8, 4);
    if (warps == 8 && minb == 6) AI4_MMV_GO(8, 6);
#undef AI4_MMV_GO
    ai4k::launch_mmv<2, 6, true, LAY, XSH, ID>(W, q, s, Y, M, M, N, K, stream, idm, max_tiles);
}

// Activations: rotate + int8-quantize X [N rows of K] once. XSH = 1: H256 and one scale per 256 (track B's kernel, used by
// AI4 and by AI4_U with K % 256 == 0); XSH = 0: H128 and one scale per 128 (AI4_U with K % 256 != 0).
// gridDim.y carries the rows (<= 65535), so longer inputs run in pieces.
template <bool SWAP8 = false>
static void ai4_quantize_x(const float * X, int64_t N, int K, int xsh, int8_t * q, float * s, cudaStream_t stream) {
    const int64_t ns = K >> (7 + xsh);
    for (int64_t r0 = 0; r0 < N; r0 += 65535) {
        const int nr = (int) std::min<int64_t>(65535, N - r0);
        if (xsh) {
            rq4::launch_quantize_act<SWAP8>(X + r0*K, K, q + r0*K, s + r0*ns, K, nr, stream);
        } else {
            ai4k::launch_quantize_act128<SWAP8>(X + r0*K, K, q + r0*K, s + r0*ns, K, nr, stream);
        }
    }
}

// GGML_CUDA_AI23_MMVR=1: rates 2 and 3 run the mmvr decode kernel (loop overhead hoisted, rate-2 whole-row windows) in
// place of mmv for N <= 8 and mmv-ID. GGML_CUDA_AI23_MMVR_CFG=WxB picks warps x min blocks (4x6 default, 2x12, 4x8, 8x3, 8x4, 16x2 dense unit long-K),
// GGML_CUDA_AI23_MMVR_KT=0 keeps K a runtime value (default: K in {640, 2048, 2560, 5120, 17408} is a template constant).
static bool ai4_mmvr_on() {
    static const bool v = ai4_env_int("GGML_CUDA_AI23_MMVR", nullptr, ai4_ai23_tuned() ? 1 : 0) != 0;
    return v;
}
static int ai4_mmvr_cfg() {
    static const int v = [] {
        const char * e = getenv("GGML_CUDA_AI23_MMVR_CFG");
        if (!e) return ai4_ai23_tuned() ? 0 : 46;
        if (!strcmp(e, "auto")) return 0;
        if (!strcmp(e, "4x6")) return 46;
        if (!strcmp(e, "2x12")) return 212;
        if (!strcmp(e, "4x8")) return 48;
        if (!strcmp(e, "8x3")) return 83;
        if (!strcmp(e, "8x4")) return 84;
        if (!strcmp(e, "16x2")) return 162;
        GGML_LOG_WARN("GGML_CUDA_AI23_MMVR_CFG='%s' is not one of auto, 4x6, 2x12, 4x8, 8x3, 8x4, 16x2; using 4x6\n", e);
        return 46;
    }();
    return v;
}
static bool ai4_mmvr_kt() {
    static const bool v = ai4_env_int("GGML_CUDA_AI23_MMVR_KT", nullptr, 1) != 0;
    return v;
}

template <int LAY, int XSH, int KT>
constexpr bool ai4_mmvr_kt_ok() {
    if constexpr (KT == 0) { return true; }
    else if constexpr (ai4k::lay_span<LAY>()) { return XSH == 1 && KT % 1024 == 0; }
    else { return XSH == 1 ? KT % 256 == 0 : KT % 256 != 0; }
}

template <int LAY, int XSH, bool ID, int KT, int STAGES = 2, int WARPS = 4, int MINB = 6, bool DIRECT = false, bool FUSED_X = false>
static void ai23_launch_mmvr(const uint8_t * W, const int8_t * q, const float * s, float * Y, int64_t ldy,
                             int M, int N, int K, cudaStream_t stream, const ai4k::idmap & idm, int max_tiles) {
    if constexpr (!ID && WARPS == 4 && (KT == 5120 || KT == 17408)) {
        static const int stages = ai4_env_int("GGML_CUDA_AI23_MMVR_STAGES", nullptr, 2);
        if (stages == 4) {
            ai4k::launch_mmvr<LAY, XSH, ID, KT, 4, WARPS, MINB, DIRECT, FUSED_X>(W, q, s, Y, ldy, M, N, K, stream, idm, max_tiles);
            return;
        }
        if (stages == 8) {
            ai4k::launch_mmvr<LAY, XSH, ID, KT, 8, WARPS, MINB, DIRECT, FUSED_X>(W, q, s, Y, ldy, M, N, K, stream, idm, max_tiles);
            return;
        }
    }
    if constexpr (!ID && ((WARPS == 4 && (KT == 5120 || KT == 17408)) ||
                          ((WARPS == 8 || WARPS == 16) && KT == 17408))) {
        static const int unroll = ai4_env_int("GGML_CUDA_AI23_MMVR_UNROLL", nullptr, 1);
        if (unroll == 2) {
            ai4k::launch_mmvr<LAY, XSH, ID, KT, STAGES, WARPS, MINB, DIRECT, FUSED_X, 1, 2>(W, q, s, Y, ldy, M, N, K, stream, idm, max_tiles);
            return;
        }
        if (unroll == 4) {
            ai4k::launch_mmvr<LAY, XSH, ID, KT, STAGES, WARPS, MINB, DIRECT, FUSED_X, 1, 4>(W, q, s, Y, ldy, M, N, K, stream, idm, max_tiles);
            return;
        }
    }
    ai4k::launch_mmvr<LAY, XSH, ID, KT, STAGES, WARPS, MINB, DIRECT, FUSED_X>(W, q, s, Y, ldy, M, N, K, stream, idm, max_tiles);
}

template <int LAY, int XSH, bool ID, int KT>
static bool ai4_mmvr_go(const uint8_t * W, const int8_t * q, const float * s, float * Y, int M, int N, int K,
                        cudaStream_t stream, const ai4k::idmap & idm, int max_tiles) {
    if constexpr (!ai4_mmvr_kt_ok<LAY, XSH, KT>()) {
        return false;
    } else {
        if (KT != 0 && K != KT) {
            return false;
        }
        const int cfg = ai4_mmvr_cfg();
        if constexpr (!ID && !ai4k::lay_span<LAY>() && KT == 17408) {
            if (cfg == 162) {
                ai23_launch_mmvr<LAY, XSH, ID, KT, 2, 16, 2>(W, q, s, Y, M, M, N, K, stream, idm, max_tiles);
                return true;
            }
        }
        const bool two_warps = cfg == 212 || (cfg == 0 && ID && N > 1 && K >= 1024);
        if constexpr (ID) {
            if (idm.direct_ids != nullptr) {
                if (idm.direct_x != nullptr) {
                    ai23_launch_mmvr<LAY, XSH, ID, KT, 2, 4, 6, true, true>(W, q, s, Y, M, M, N, K, stream, idm, max_tiles);
                    return true;
                }
                if (two_warps) {
                    ai23_launch_mmvr<LAY, XSH, ID, KT, ai4k::lay_span<LAY>() ? 4 : 2, 2, 12, true>(W, q, s, Y, M, M, N, K, stream, idm, max_tiles);
                } else {
                    ai23_launch_mmvr<LAY, XSH, ID, KT, 2, 4, 6, true>(W, q, s, Y, M, M, N, K, stream, idm, max_tiles);
                }
                return true;
            }
        }
        if (two_warps) {
            ai23_launch_mmvr<LAY, XSH, ID, KT, ai4k::lay_span<LAY>() ? 4 : 2, 2, 12>(W, q, s, Y, M, M, N, K, stream, idm, max_tiles);
            return true;
        }
        if constexpr (KT != 0) {
            if constexpr (!ID && (KT == 5120 || KT == 17408)) {
                if (cfg == 48) {
                    ai23_launch_mmvr<LAY, XSH, ID, KT, 2, 4, 8>(W, q, s, Y, M, M, N, K, stream, idm, max_tiles);
                    return true;
                }
                if (cfg == 83) {
                    ai23_launch_mmvr<LAY, XSH, ID, KT, 2, 8, 3>(W, q, s, Y, M, M, N, K, stream, idm, max_tiles);
                    return true;
                }
                if (cfg == 84) {
                    ai23_launch_mmvr<LAY, XSH, ID, KT, 2, 8, 4>(W, q, s, Y, M, M, N, K, stream, idm, max_tiles);
                    return true;
                }
            }
            ai23_launch_mmvr<LAY, XSH, ID, KT>(W, q, s, Y, M, M, N, K, stream, idm, max_tiles);
        } else {
            switch (ai4_mmvr_cfg()) {
                case 48: ai23_launch_mmvr<LAY, XSH, ID, KT, 2, 4, 8>(W, q, s, Y, M, M, N, K, stream, idm, max_tiles); break;
                case 83: ai23_launch_mmvr<LAY, XSH, ID, KT, 2, 8, 3>(W, q, s, Y, M, M, N, K, stream, idm, max_tiles); break;
                case 84: ai23_launch_mmvr<LAY, XSH, ID, KT, 2, 8, 4>(W, q, s, Y, M, M, N, K, stream, idm, max_tiles); break;
                default: ai23_launch_mmvr<LAY, XSH, ID, KT>(W, q, s, Y, M, M, N, K, stream, idm, max_tiles); break;
            }
        }
        return true;
    }
}

// true when mmvr ran (rate 2 / 3 layouts with GGML_CUDA_AI23_MMVR=1)
template <int LAY, int XSH, bool ID>
static bool ai4_mmvr_try(const uint8_t * W, const int8_t * q, const float * s, float * Y, int M, int N, int K,
                         cudaStream_t stream, const ai4k::idmap & idm = ai4k::idmap{}, int max_tiles = 1) {
    constexpr int rate = ai4k::lay_rate<LAY>();
    if constexpr (rate != 2 && rate != 3) {
        return false;
    } else {
        if (!ai4_mmvr_on()) {
            return false;
        }
        const int cfg = ai4_mmvr_cfg();
        const bool dense_forced = !ID && (K == 5120 || K == 17408) && (cfg == 48 || cfg == 83 || cfg == 84 || cfg == 162);
        if (ai4_mmvr_kt() && (cfg == 0 || cfg == 46 || cfg == 212 || dense_forced)) {
            if (ai4_mmvr_go<LAY, XSH, ID,   640>(W, q, s, Y, M, N, K, stream, idm, max_tiles)) return true;
            if (ai4_mmvr_go<LAY, XSH, ID,  2048>(W, q, s, Y, M, N, K, stream, idm, max_tiles)) return true;
            if (ai4_mmvr_go<LAY, XSH, ID,  2560>(W, q, s, Y, M, N, K, stream, idm, max_tiles)) return true;
            if (ai4_mmvr_go<LAY, XSH, ID,  5120>(W, q, s, Y, M, N, K, stream, idm, max_tiles)) return true;
            if (ai4_mmvr_go<LAY, XSH, ID, 17408>(W, q, s, Y, M, N, K, stream, idm, max_tiles)) return true;
        }
        return ai4_mmvr_go<LAY, XSH, ID, 0>(W, q, s, Y, M, N, K, stream, idm, max_tiles);
    }
}

// Dense controls keep the existing shared activation tile while varying live columns and cooperative N warps.
template <int BN, int LAY, int XSH>
static void ai23_dense_tile(const uint8_t * W, const int8_t * q, const float * s, float * Y, int M, int N, int K,
                            cudaStream_t stream, int cfg) {
    switch (cfg) {
        case 2: ai4k::launch_gemm<BN, 2, 1, LAY, XSH, false, 1, 4>(W, q, s, Y, M, M, N, K, stream); break;
        case 3: ai4k::launch_gemm<BN, 2, 1, LAY, XSH, false, 2, 4>(W, q, s, Y, M, M, N, K, stream); break;
        case 4: ai4k::launch_gemm<BN, 2, (BN <= 16 ? 1 : 2), LAY, XSH, false, 2, 4>(W, q, s, Y, M, M, N, K, stream); break;
        case 5:
            if constexpr (BN >= 32) { ai4k::launch_gemm<BN, 2, 1, LAY, XSH, false, 4, 2>(W, q, s, Y, M, M, N, K, stream); }
            else { ai4k::launch_gemm<BN, 2, 1, LAY, XSH, false, 2, 4>(W, q, s, Y, M, M, N, K, stream); }
            break;
        case 6:
            if constexpr (BN >= 32) { ai4k::launch_gemm<BN, 2, 1, LAY, XSH, false, 4, 4, false, 2>(W, q, s, Y, M, M, N, K, stream); }
            else { ai4k::launch_gemm<BN, 2, 1, LAY, XSH, false, 2, 4>(W, q, s, Y, M, M, N, K, stream); }
            break;
        case 8: ai4k::launch_gemm<BN, 2, 1, LAY, XSH, false, 2, 4, false, 1, true>(W, q, s, Y, M, M, N, K, stream); break;
        case 9:
            if constexpr (BN >= 32) { ai4k::launch_gemm<BN, 2, 1, LAY, XSH, false, 4, 4, false, 2, true>(W, q, s, Y, M, M, N, K, stream); }
            else { ai4k::launch_gemm<BN, 2, 1, LAY, XSH, false, 2, 4, false, 1, true>(W, q, s, Y, M, M, N, K, stream); }
            break;
        case 10:
            if constexpr (BN >= 32) { ai4k::launch_gemm<BN, 2, 1, LAY, XSH, false, 4, 2, false, 1, true>(W, q, s, Y, M, M, N, K, stream); }
            else { ai4k::launch_gemm<BN, 2, 1, LAY, XSH, false, 2, 4, false, 1, true>(W, q, s, Y, M, M, N, K, stream); }
            break;
        case 11: ai4k::launch_gemm<BN, 2, 1, LAY, XSH, false, 1, 8, false, 2, false, true>(W, q, s, Y, M, M, N, K, stream); break;
        case 12: ai4k::launch_gemm<BN, 3, 1, LAY, XSH, false, 1, 8, false, 2, false, true>(W, q, s, Y, M, M, N, K, stream); break;
        case 13: ai4k::launch_gemm<BN, 2, 1, LAY, XSH, false, 1, 4, false, 4, false, true>(W, q, s, Y, M, M, N, K, stream); break;
        case 14: ai4k::launch_gemm<BN, 3, (BN <= 16 ? 2 : 4), LAY, XSH, false, 1, 8, false, 2>(W, q, s, Y, M, M, N, K, stream); break;
        case 15: ai4k::launch_gemm<BN, 2, 1, LAY, XSH, false, 1, 8, false, 2>(W, q, s, Y, M, M, N, K, stream); break;
        case 16: ai4k::launch_gemm<BN, 2, 1, LAY, XSH, false, 1, 8, false, 2, false, true, true>(W, q, s, Y, M, M, N, K, stream); break;
        case 17: ai4k::launch_gemm<BN, 3, 1, LAY, XSH, false, 1, 8, false, 2, false, true, true>(W, q, s, Y, M, M, N, K, stream); break;
        case 18:
            if constexpr (ai4k::lay_rate<LAY>() == 2) {
                ai4k::launch_gemm<BN, 2, 1, LAY, XSH, false, 1, 8, false, 2, false, true, true, true>(W, q, s, Y, M, M, N, K, stream);
            } else {
                ai4k::launch_gemm<BN, 2, 1, LAY, XSH, false, 1, 8, false, 2, false, true, true>(W, q, s, Y, M, M, N, K, stream);
            }
            break;
        case 19:
            if constexpr (ai4k::lay_rate<LAY>() == 2) {
                ai4k::launch_gemm<BN, 3, 1, LAY, XSH, false, 1, 8, false, 2, false, true, true, true>(W, q, s, Y, M, M, N, K, stream);
            } else {
                ai4k::launch_gemm<BN, 3, 1, LAY, XSH, false, 1, 8, false, 2, false, true, true>(W, q, s, Y, M, M, N, K, stream);
            }
            break;
        case 20: ai4k::launch_gemm<BN, 2, 1, LAY, XSH, false, 1, 4, false, 4, false, true, true>(W, q, s, Y, M, M, N, K, stream); break;
        case 21:
            if constexpr (ai4k::lay_rate<LAY>() == 2) {
                ai4k::launch_gemm<BN, 2, 1, LAY, XSH, false, 1, 4, false, 4, false, true, true, true>(W, q, s, Y, M, M, N, K, stream);
            } else {
                ai4k::launch_gemm<BN, 2, 1, LAY, XSH, false, 1, 4, false, 4, false, true, true>(W, q, s, Y, M, M, N, K, stream);
            }
            break;
        case 22: ai4k::launch_gemm<BN, 2, 1, LAY, XSH, false, 1, 8, false, 2, false, true, false, false, false, true>(W, q, s, Y, M, M, N, K, stream); break;
        case 23: ai4k::launch_gemm<BN, 2, 1, LAY, XSH, false, 1, 4, false, 4, false, true, false, false, false, true>(W, q, s, Y, M, M, N, K, stream); break;
        case 24: ai4k::launch_gemm<BN, 2, 1, LAY, XSH, false, 1, 8, false, 2, false, true, false, false, false, false, true>(W, q, s, Y, M, M, N, K, stream); break;
        case 25: ai4k::launch_gemm<BN, 2, 1, LAY, XSH, false, 1, 8, false, 2, false, true, false, false, false, true, true>(W, q, s, Y, M, M, N, K, stream); break;
        case 26: ai4k::launch_gemm<BN, 2, 1, LAY, XSH, false, 1, 4, false, 4, false, true, false, false, false, false, true>(W, q, s, Y, M, M, N, K, stream); break;
        case 27: ai4k::launch_gemm<BN, 2, 1, LAY, XSH, false, 1, 4, false, 4, false, true, false, false, false, true, true>(W, q, s, Y, M, M, N, K, stream); break;
        case 28: ai4k::launch_gemm<BN, 2, 1, LAY, XSH, false, 1, 4, false, 4, false, true, false, false, false, false, true, true>(W, q, s, Y, M, M, N, K, stream); break;
        case 29: ai4k::launch_gemm<BN, 2, 1, LAY, XSH, false, 1, 8, false, 2, false, true, false, false, false, false, true, true>(W, q, s, Y, M, M, N, K, stream); break;
        case 7: ai4k::launch_gemm<BN, 2, 1, LAY, XSH>(W, q, s, Y, M, M, N, K, stream); break;
        default: ai4k::launch_gemm<BN, 3, (BN <= 16 ? 2 : 4), LAY, XSH>(W, q, s, Y, M, M, N, K, stream); break;
    }
}

template <int LAY, int XSH, int KT>
static void ai23_small_split(const uint8_t * W, const int8_t * q, const float * s, float * partial,
                              int M, int N, int K, int splits, cudaStream_t stream) {
    if (splits == 4) {
        ai4k::launch_mmvr<LAY, XSH, false, KT, 2, 4, 8, false, false, 4>(W, q, s, partial, M, M, N, K, stream);
    } else {
        ai4k::launch_mmvr<LAY, XSH, false, KT, 2, 4, 8, false, false, 2>(W, q, s, partial, M, M, N, K, stream);
    }
}

template <int LAY, int XSH>
static void ai4_mm_kernels(const uint8_t * W, const int8_t * q, const float * s, float * Y, int M, int N, int K, cudaStream_t stream, int sms) {
    if constexpr (LAY == ai4k::LAY_SB || LAY == ai4k::LAY_U) {
        if (sms && ai4_ampere_mmv(N)) {
            ai4_ampere_width<LAY, XSH, false>(W, q, s, Y, M, N, K, ai4_ampere_bn(N), sms, stream);
            return;
        }
    }
    if constexpr (ai4k::lay_rate<LAY>() == 2 || ai4k::lay_rate<LAY>() == 3) {
        static const int dense_tile = ai4_env_int("GGML_CUDA_AI23_DENSE_TILE", nullptr, 0);
        if (dense_tile && N > 8 && N <= 64) {
            if (N <= 16) { ai23_dense_tile<16, LAY, XSH>(W, q, s, Y, M, N, K, stream, dense_tile); }
            else if (N <= 32) { ai23_dense_tile<32, LAY, XSH>(W, q, s, Y, M, N, K, stream, dense_tile); }
            else { ai23_dense_tile<64, LAY, XSH>(W, q, s, Y, M, N, K, stream, dense_tile); }
            return;
        }
        static const bool narrow = ai4_env_int("GGML_CUDA_AI23_NARROW", nullptr, 0) != 0;
        if (sms && narrow && N > 8 && N <= 64) {
            ai4_ampere_width<LAY, XSH, false>(W, q, s, Y, M, N, K, ai4_ampere_bn(N), sms, stream);
            return;
        }
    }
    if (N <= 8) {
        if (ai4_mmvr_try<LAY, XSH, false>(W, q, s, Y, M, N, K, stream)) {
            return;
        }
        if (ai4_mmv_xsmem()) {
            ai4_mmv_warn_ignored();
            ai4k::launch_mmv<3, 3, false, LAY, XSH>(W, q, s, Y, M, M, N, K, stream);
        } else {
            switch (ai4_mmv_stages()) {
                case 3:
                    ai4_mmv_warn_ignored();
                    ai4k::launch_mmv<3, 6, true, LAY, XSH>(W, q, s, Y, M, M, N, K, stream);
                    break;
                case 4:
                    ai4_mmv_warn_ignored();
                    ai4k::launch_mmv<4, 6, true, LAY, XSH>(W, q, s, Y, M, M, N, K, stream);
                    break;
                default: {
                    const rpb_ai4_mmv k = ai4_mmv_knobs();
                    if (k.warps == rpb_ai4_warps_default && k.minb == rpb_ai4_minb_default) {
                        ai4k::launch_mmv<2, 6, true, LAY, XSH>(W, q, s, Y, M, M, N, K, stream);
                    } else {
                        ai4_launch_mmv_tuned<LAY, XSH, false>(W, q, s, Y, M, N, K, stream, k.warps, k.minb);
                    }
                    break;
                }
            }
        }
    } else if (N >= ai4_bn128_min()) {
        ai4k::launch_gemm<128, 3, 4, LAY, XSH>(W, q, s, Y, M, M, N, K, stream);
    } else {
        ai4k::launch_gemm<64, 3, 4, LAY, XSH>(W, q, s, Y, M, M, N, K, stream);
    }
}

// ai4-kern LAY for a type with IMMA kernels (AI4 / AI4_U / AI5_U / AI4_S / AI5_S, and AI2/AI3 _U/_S under
// GGML_CUDA_AI23_FAST=1); xsh = 1 unless the unit-128 layout has K % 256 != 0. The span layouts need K % 1024 == 0, so
// they are always xsh = 1.
static int ai4_lay_of(ggml_type t) {
    const int lay = ggml_ai4_layout(t), rate = ggml_ai4_rate(t);
    if (lay == 2) {
        return rate == 6 ? ai4k::LAY_S6 : rate == 5 ? ai4k::LAY_S5 : rate == 3 ? ai4k::LAY_S3 : rate == 2 ? ai4k::LAY_S2 : ai4k::LAY_S;
    }
    if (lay == 0) {
        return ai4k::LAY_SB;
    }
    return rate == 5 ? ai4k::LAY_U5 : rate == 3 ? ai4k::LAY_U3 : rate == 2 ? ai4k::LAY_U2 : ai4k::LAY_U;
}

// Rate-2 / rate-3 IMMA kernels (generic window decode, ai4k::kr_win). Opt-in while new: GGML_CUDA_AI23_FAST=1 runs them,
// unset keeps the rate-k reference path (reconstruct + SGEMM, host-routed MUL_MAT_ID).
static bool ai4_ai23_fast() {
    static const bool v = ai4_env_int("GGML_CUDA_AI23_FAST", nullptr, 0) != 0;
    return v;
}

template <int BN, int LAY, int XSH>
static void ai23_split_tile(const uint8_t * W, const int8_t * q, const float * s, float * partial,
                             int M, int N, int K, int splits, cudaStream_t stream) {
    static const int tile = ai4_env_int("GGML_CUDA_AI23_DENSE_TILE", nullptr, 0);
    constexpr bool RATE2 = ai4k::lay_rate<LAY>() == 2;
    if (tile >= 24 && tile <= 27) {
        if (tile == 25 || tile == 27) {
            ai4k::launch_gemm<BN, 2, 1, LAY, XSH, false, 1, 8, false, 2, false, true, false, false, true, true, true>(W, q, s, partial, M, M, N, K, stream, ai4k::idmap{}, 0, nullptr, splits);
        } else {
            ai4k::launch_gemm<BN, 2, 1, LAY, XSH, false, 1, 8, false, 2, false, true, false, false, true, false, true>(W, q, s, partial, M, M, N, K, stream, ai4k::idmap{}, 0, nullptr, splits);
        }
        return;
    }
    if (tile == 22 || tile == 23) {
        ai4k::launch_gemm<BN, 2, 1, LAY, XSH, false, 1, 8, false, 2, false, true, false, false, true, true>(W, q, s, partial, M, M, N, K, stream, ai4k::idmap{}, 0, nullptr, splits);
        return;
    }
    if constexpr (RATE2) {
        if (tile != 16 && tile != 17) {
            ai4k::launch_gemm<BN, 2, 1, LAY, XSH, false, 1, 8, false, 2, false, true, true, true, true>(W, q, s, partial, M, M, N, K, stream, ai4k::idmap{}, 0, nullptr, splits);
            return;
        }
    }
    ai4k::launch_gemm<BN, 2, 1, LAY, XSH, false, 1, 8, false, 2, false, true, true, false, true>(W, q, s, partial, M, M, N, K, stream, ai4k::idmap{}, 0, nullptr, splits);
}

template <int LAY, int XSH>
static void ai23_split_width(const uint8_t * W, const int8_t * q, const float * s, float * partial,
                              int M, int N, int K, int splits, cudaStream_t stream) {
    if (N <= 16) { ai23_split_tile<16, LAY, XSH>(W, q, s, partial, M, N, K, splits, stream); }
    else if (N <= 32) { ai23_split_tile<32, LAY, XSH>(W, q, s, partial, M, N, K, splits, stream); }
    else { ai23_split_tile<64, LAY, XSH>(W, q, s, partial, M, N, K, splits, stream); }
}

// W4A8 fast path: rotate+quantize the activations, then the AI4 decode-to-IMMA kernels
static void ai4_mul_mat_fast(ggml_backend_cuda_context & ctx, ggml_type t, const uint8_t * W, const float * X, float * Y,
                            int M, int N, int K) {
    cudaStream_t stream = ctx.stream();
    const int xsh = (!ggml_ai4_is_unit(t) || K % 256 == 0) ? 1 : 0;
    ggml_cuda_pool_alloc<int8_t> q(ctx.pool(), (size_t) N*K);
    ggml_cuda_pool_alloc<float>  s(ctx.pool(), (size_t) N*(K >> (7 + xsh)));
    static const int tile = ai4_env_int("GGML_CUDA_AI23_DENSE_TILE", nullptr, 0);
    const bool swap8 = (ggml_ai4_rate(t) == 2 || ggml_ai4_rate(t) == 3) && N > 8 && N <= 64 && (tile == 22 || tile == 23 || tile == 25 || tile == 27);
    if (swap8) { ai4_quantize_x<true>(X, N, K, xsh, q.get(), s.get(), stream); }
    else { ai4_quantize_x(X, N, K, xsh, q.get(), s.get(), stream); }
    static const int splits = ai4_env_int("GGML_CUDA_AI23_SPLIT_K", nullptr, 0);
    const int rate = ggml_ai4_rate(t);
    static const int small_splits = ai4_env_int("GGML_CUDA_AI23_SMALL_SPLIT", nullptr, 0);
    if ((rate == 2 || rate == 3) && N <= 8 && (K == 5120 || K == 17408) &&
            (small_splits == 2 || small_splits == 4)) {
        ggml_cuda_pool_alloc<float> partial(ctx.pool(), (size_t) M*N*small_splits);
        // All valid two/three-bit layouts use H256 at these fixed K values.
        switch (ai4_lay_of(t)) {
            case ai4k::LAY_U2:
                if (K == 5120) ai23_small_split<ai4k::LAY_U2, 1, 5120>(W, q.get(), s.get(), partial.get(), M, N, K, small_splits, stream);
                else ai23_small_split<ai4k::LAY_U2, 1, 17408>(W, q.get(), s.get(), partial.get(), M, N, K, small_splits, stream);
                break;
            case ai4k::LAY_U3:
                if (K == 5120) ai23_small_split<ai4k::LAY_U3, 1, 5120>(W, q.get(), s.get(), partial.get(), M, N, K, small_splits, stream);
                else ai23_small_split<ai4k::LAY_U3, 1, 17408>(W, q.get(), s.get(), partial.get(), M, N, K, small_splits, stream);
                break;
            case ai4k::LAY_S2:
                if (K == 5120) ai23_small_split<ai4k::LAY_S2, 1, 5120>(W, q.get(), s.get(), partial.get(), M, N, K, small_splits, stream);
                else ai23_small_split<ai4k::LAY_S2, 1, 17408>(W, q.get(), s.get(), partial.get(), M, N, K, small_splits, stream);
                break;
            case ai4k::LAY_S3:
                if (K == 5120) ai23_small_split<ai4k::LAY_S3, 1, 5120>(W, q.get(), s.get(), partial.get(), M, N, K, small_splits, stream);
                else ai23_small_split<ai4k::LAY_S3, 1, 17408>(W, q.get(), s.get(), partial.get(), M, N, K, small_splits, stream);
                break;
            default: GGML_ABORT("invalid small split layout");
        }
        ai4k::split_k_reduce<<<((int64_t) M*N + 255)/256, 256, 0, stream>>>(partial.get(), Y, (int64_t) M*N, small_splits);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if ((rate == 2 || rate == 3) && N > 8 && N <= 64 &&
            (splits == 2 || splits == 4 || splits == 8 || splits == 16)) {
        ggml_cuda_pool_alloc<float> partial(ctx.pool(), (size_t) M*N*splits);
        switch (ai4_lay_of(t)) {
#define AI23_SPLIT_U(L) \
            case L: \
                if (xsh) { ai23_split_width<L, 1>(W, q.get(), s.get(), partial.get(), M, N, K, splits, stream); } \
                else { ai23_split_width<L, 0>(W, q.get(), s.get(), partial.get(), M, N, K, splits, stream); } \
                break;
            AI23_SPLIT_U(ai4k::LAY_U2)
            AI23_SPLIT_U(ai4k::LAY_U3)
#undef AI23_SPLIT_U
            case ai4k::LAY_S2: ai23_split_width<ai4k::LAY_S2, 1>(W, q.get(), s.get(), partial.get(), M, N, K, splits, stream); break;
            case ai4k::LAY_S3: ai23_split_width<ai4k::LAY_S3, 1>(W, q.get(), s.get(), partial.get(), M, N, K, splits, stream); break;
            default: GGML_ABORT("unsupported AI23 split layout");
        }
        ai4k::split_k_reduce<<<((int64_t) M*N + 255)/256, 256, 0, stream>>>(partial.get(), Y, (int64_t) M*N, splits);
        return;
    }
    if (N >= 512 && ai4_ampere_sms(ctx) && ai4_prefill_v2() && (t == GGML_TYPE_AI4 || t == GGML_TYPE_AI4_U)) {
        GGML_ASSERT(M > 0 && K > 0 && (size_t) M <= SIZE_MAX / (size_t) K);
        ggml_cuda_pool_alloc<int8_t> decoded(ctx.pool(), (size_t) M*K);
        if (t == GGML_TYPE_AI4) {
            ai4k::launch_prefill_v2<ai4k::LAY_SB, 1>(W, q.get(), s.get(), decoded.get(), Y, M, M, N, K, stream);
        } else if (xsh) {
            ai4k::launch_prefill_v2<ai4k::LAY_U, 1>(W, q.get(), s.get(), decoded.get(), Y, M, M, N, K, stream);
        } else {
            ai4k::launch_prefill_v2<ai4k::LAY_U, 0>(W, q.get(), s.get(), decoded.get(), Y, M, M, N, K, stream);
        }
        return;
    }
    switch (ai4_lay_of(t)) {
#define AI4_MM_U(L) \
        case L: \
            if (xsh) { ai4_mm_kernels<L, 1>(W, q.get(), s.get(), Y, M, N, K, stream, ai4_ampere_sms(ctx)); } \
            else     { ai4_mm_kernels<L, 0>(W, q.get(), s.get(), Y, M, N, K, stream, ai4_ampere_sms(ctx)); } \
            break;
        AI4_MM_U(ai4k::LAY_U2)
        AI4_MM_U(ai4k::LAY_U3)
#undef AI4_MM_U
        case ai4k::LAY_S2:
            ai4_mm_kernels<ai4k::LAY_S2, 1>(W, q.get(), s.get(), Y, M, N, K, stream, ai4_ampere_sms(ctx));
            break;
        case ai4k::LAY_S3:
            ai4_mm_kernels<ai4k::LAY_S3, 1>(W, q.get(), s.get(), Y, M, N, K, stream, ai4_ampere_sms(ctx));
            break;
        case ai4k::LAY_S:
            ai4_mm_kernels<ai4k::LAY_S,  1>(W, q.get(), s.get(), Y, M, N, K, stream, ai4_ampere_sms(ctx));
            break;
        case ai4k::LAY_S5:
            ai4_mm_kernels<ai4k::LAY_S5, 1>(W, q.get(), s.get(), Y, M, N, K, stream, ai4_ampere_sms(ctx));
            break;
        case ai4k::LAY_S6:
            ai4_mm_kernels<ai4k::LAY_S6, 1>(W, q.get(), s.get(), Y, M, N, K, stream, ai4_ampere_sms(ctx));
            break;
        case ai4k::LAY_U5:
            if (xsh) {
                ai4_mm_kernels<ai4k::LAY_U5, 1>(W, q.get(), s.get(), Y, M, N, K, stream, ai4_ampere_sms(ctx));
            } else {
                ai4_mm_kernels<ai4k::LAY_U5, 0>(W, q.get(), s.get(), Y, M, N, K, stream, ai4_ampere_sms(ctx));
            }
            break;
        case ai4k::LAY_SB:
            ai4_mm_kernels<ai4k::LAY_SB, 1>(W, q.get(), s.get(), Y, M, N, K, stream, ai4_ampere_sms(ctx));
            break;
        default:
            if (xsh) {
                ai4_mm_kernels<ai4k::LAY_U, 1>(W, q.get(), s.get(), Y, M, N, K, stream, ai4_ampere_sms(ctx));
            } else {
                ai4_mm_kernels<ai4k::LAY_U, 0>(W, q.get(), s.get(), Y, M, N, K, stream, ai4_ampere_sms(ctx));
            }
            break;
    }
}

static bool ai4_is_type(ggml_type t) {
    return ggml_ai4_rate(t) != 0;
}

// Types that have no IMMA kernels: the rate-k reference path (rotated-basis reconstruct x int8-emulated activations +
// SGEMM) is always taken for them. AI5_U / AI5_S have rate-5 kernels, so they only take the reference path under
// GGML_CUDA_AI5_REF=1 (an A/B check of those kernels). The span types follow their rate: k = 4 and k = 5 have kernels;
// k = 2 and k = 3 (unit and span) have kernels under GGML_CUDA_AI23_FAST=1.
// AI6_S (k = 6 span, the AI+ output head) has rate-6 kernels and takes the reference path only under GGML_CUDA_AI6_REF=1.
static bool ai4_ref_only_type(ggml_type t) {
    static const bool v = ai4_env_int("GGML_CUDA_AI5_REF", nullptr, 0) != 0;
    static const bool v6 = ai4_env_int("GGML_CUDA_AI6_REF", nullptr, 0) != 0;
    if (t == GGML_TYPE_AI5_U || t == GGML_TYPE_AI5_S) {
        return v;
    }
    if (t == GGML_TYPE_AI2_U || t == GGML_TYPE_AI3_U || t == GGML_TYPE_AI2_S || t == GGML_TYPE_AI3_S) {
        return !ai4_ai23_fast();
    }
    if (t == GGML_TYPE_AI6_S) { return v6; }
    return t == GGML_TYPE_AI6_U;
}

static void ai4_reconstruct_any(const void * src, float * W, int64_t K, int64_t M, cudaStream_t stream, ggml_type t) {
    const int rate = ggml_ai4_rate(t), lay = ggml_ai4_layout(t);
    if (lay == 2) {
        GGML_ASSERT(aik_reconstruct_rate<float>(rate, src, W, K, M, true, stream, ggml_ai4_trellis_mul(t), true));
    } else if (lay == 1 && (rate != 4 || ggml_ai4_trellis_mul(t) != AI4_DEV_MUL)) {
        GGML_ASSERT(aik_reconstruct_rate<float>(rate, src, W, K, M, true, stream, ggml_ai4_trellis_mul(t)));
    } else if (lay == 1) {
        ai4u_reconstruct_launch<float>(src, W, K, M, stream);
    } else {
        ai4_reconstruct_launch<float>(src, W, K, M, stream);
    }
}

static bool ai4_fast_ok(const ggml_backend_cuda_context & ctx) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    return !ai4_ref() && GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_AMPERE;
}

// Largest multiple of 16 rows whose f32 image fits in the cap, in [16, M]. SLAB_ROWS wins over the MiB cap.
// SLAB_MIB <= 0 (or K == 0) is the old whole-tensor buffer. A reconstruct launch puts one 16-row group on
// gridDim.y, so a slab is also clamped to 65535 groups; the whole-tensor A/B path is not.
static int64_t ai4_ref_slab_rows(int64_t M, int64_t K) {
    static const int rows = ai4_env_int("GGML_CUDA_AI4_REF_SLAB_ROWS", nullptr, 0);
    static const int mib  = ai4_env_int("GGML_CUDA_AI4_REF_SLAB_MIB", nullptr, 256);
    const int64_t grid_max = (int64_t) 65535 * 16;
    if (rows > 0) {
        const int64_t r = ((int64_t) rows / 16) * 16;
        if (r >= 16) {
            return std::min<int64_t>(r, std::min<int64_t>(M, grid_max));
        }
    }
    if (mib <= 0 || K <= 0 || K > INT64_MAX / (int64_t) sizeof(float)) {
        return M;
    }
    const int64_t cap = (int64_t) mib << 20;
    const int64_t nb  = K * (int64_t) sizeof(float);
    int64_t n = (cap / nb) / 16 * 16;
    if (n < 16) {
        n = 16;
    }
    return std::min<int64_t>(n, std::min<int64_t>(M, grid_max));
}

// Reference SGEMM in row slabs so the f32 reconstruct buffer is one slab, not M*K. The pool keeps its high-water
// mark, and a whole output head (M ~ 2.5e5, K = 5120) was ~4.8 GiB. xe (T x K), when used, is built once.
// Weights are consecutive 16-row slabs (ggml-ai4.h); rotation is along K inside a row, so it does not cross a
// slab boundary. Byte offset m0 * ggml_row_size is therefore exactly rows [m0, m0+mm). dst is column-major
// M x T: ldc stays M and the C pointer moves by m0 elements. use_any: the final fallback's ai4_reconstruct_any
// (AI4 / AI4_U kernels). Ref-only types always go through aik_reconstruct_rate, including the unrotated A/B.
static void ai4_ref_gemm(ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
                         const float * X, float * Y, int64_t M, int64_t T, int64_t K, bool emul, bool use_any) {
    if (M == 0 || T == 0) {
        return;
    }
    GGML_ASSERT(M > 0 && T > 0 && K >= 0);
    GGML_ASSERT(M <= (int64_t) INT32_MAX && T <= (int64_t) INT32_MAX && K <= (int64_t) INT32_MAX);   // cuBLAS takes int
    GGML_ASSERT(!(use_any && emul));
    const int64_t rows = ai4_ref_slab_rows(M, K);
    GGML_ASSERT(rows >= 16 && rows % 16 == 0 && K <= INT64_MAX / rows);
    GGML_ASSERT(!emul || K <= 0 || T <= INT64_MAX / K);
    const int id = ggml_cuda_get_device();
    cudaStream_t stream = ctx.stream();
    // One buffer, live for the whole loop. Same stream: slab i's SGEMM is queued before slab i+1 overwrites it.
    const int64_t nw = rows * K > 0 ? rows * K : (int64_t) 1;   // K == 0: keep the pool off a zero-byte alloc
    const int64_t nx = emul && K > 0 ? T * K : (int64_t) 1;
    GGML_ASSERT((uint64_t) nw <= SIZE_MAX / sizeof(float) && (uint64_t) nx <= SIZE_MAX / sizeof(float));
    ggml_cuda_pool_alloc<float> w(ctx.pool(id), (size_t) nw);
    ggml_cuda_pool_alloc<float> xe(ctx.pool(id), (size_t) nx);
    if (emul) {
        aik_act_emul_launch(X, xe.get(), K, T, stream);
    }
    CUDA_CHECK(cudaGetLastError());
    const float * B = emul ? xe.get() : X;
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(), stream));
    const int64_t row_bytes = (int64_t) ggml_row_size(src0->type, K);
    GGML_ASSERT(row_bytes == (int64_t) src0->nb[1]);   // contiguous: nominal row size is the slab stride / 16
    GGML_ASSERT(row_bytes <= 0 || M <= INT64_MAX / row_bytes);
    const uint8_t * base = (const uint8_t *) src0->data;
    for (int64_t m0 = 0; m0 < M; ) {
        const int64_t mm = std::min<int64_t>(rows, M - m0);
        GGML_ASSERT(mm > 0 && mm % 16 == 0);   // last slab may be shorter than rows, still a whole number of groups
        const void * src = base + (size_t) (m0 * row_bytes);
        if (use_any) {
            ai4_reconstruct_any(src, w.get(), K, mm, stream, src0->type);
        } else {
            GGML_ASSERT(aik_reconstruct_rate<float>(ggml_ai4_rate(src0->type), src, w.get(), K, mm, !emul, stream,
                                                    ggml_ai4_trellis_mul(src0->type), ggml_ai4_is_span(src0->type)));
        }
        // C(mm x T) at row m0 of the M x T result. ldc = M, so column t lands at Y[t*M + m0].
        CUBLAS_CHECK(cublasSgemm(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                                 (int) mm, (int) T, (int) K,
                                 &alpha, w.get(), (int) K,
                                         B, (int) K,
                                 &beta,  Y + m0, (int) M));
        m0 += mm;
    }
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_ai4_id_needs_sync(int cc, ggml_type t) {
    return ai4_ref() || ai4_ref_only_type(t) || !GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_AMPERE;
}

bool ggml_cuda_ai4_supported(const ggml_tensor * op) {
    const ggml_tensor * a = op->src[0];
    const ggml_tensor * b = op->src[1];
    const int64_t kmod = ggml_ai4_is_unit(a->type) ? 128 : 1024;
    if (b->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32 || a->ne[0] % kmod != 0 || a->ne[1] % 16 != 0 ||
            a->ne[3] != 1 || !ggml_is_contiguous(a) || !ggml_is_contiguous(b) || !ggml_is_contiguous(op)) {
        return false;
    }
    if (op->op == GGML_OP_MUL_MAT) {
        return a->ne[2] == 1;
    }
    if (op->op == GGML_OP_MUL_MAT_ID) {
        const ggml_tensor * ids = op->src[2];
        return ids->type == GGML_TYPE_I32 && ids->nb[0] == sizeof(int32_t) && a->ne[2] <= ai4k::IDS_MAX_EXPERT &&
               b->ne[2] == ids->ne[1] && (b->ne[1] == 1 || b->ne[1] == ids->ne[0]) && b->ne[3] == 1;
    }
    return false;
}

void ggml_cuda_mul_mat_ai4(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(ai4_is_type(src0->type));
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst));
    GGML_ASSERT(src0->ne[2] == 1 && src0->ne[3] == 1);

    const bool unit = ggml_ai4_is_unit(src0->type);   // the span types need K % 1024 == 0, as the AI4 slab layout does
    const int64_t K = src0->ne[0];
    const int64_t M = src0->ne[1];
    GGML_ASSERT(K % (unit ? 128 : 1024) == 0 && M % 16 == 0);
    GGML_ASSERT(src1->ne[0] == K && dst->ne[0] == M);
    const int64_t T = src1->ne[1] * src1->ne[2] * src1->ne[3];

    if (ai4_ref_only_type(src0->type)) {
        // rate-k reference: rotated-basis reconstruct x int8-emulated rotated activations (the W4A8 path's arithmetic up to
        // summation order); GGML_CUDA_AI4_REF=1: original basis x f32 activations. Slabbed so the f32 buffer stays bounded.
        ai4_ref_gemm(ctx, src0, (const float *) src1->data, (float *) dst->data, M, T, K, !ai4_ref(), false);
        return;
    }

    if (ai4_fast_ok(ctx)) {
        const int rate = ggml_ai4_rate(src0->type);
        // With very few outputs, activation quantization can dominate the output norm.
        // Keep these small products in f32 without changing the normal matrix kernels.
        if ((rate == 2 || rate == 3) && M > 0 && M <= 128 && T > 0 && T <= 128/M) {
            ai4_ref_gemm(ctx, src0, (const float *) src1->data, (float *) dst->data, M, T, K, false, false);
            return;
        }
        // quantize_act's float4 loads need 16-B aligned rows (K % 128 == 0 keeps every row aligned)
        GGML_ASSERT(((uintptr_t) src1->data & 15) == 0);
        // the GEMM puts token tiles on gridDim.y: run longer batches in 65535-token pieces
        for (int64_t t0 = 0; t0 < T; t0 += 65535) {
            const int nt = (int) std::min<int64_t>(65535, T - t0);
            ai4_mul_mat_fast(ctx, src0->type, (const uint8_t *) src0->data, (const float *) src1->data + t0*K,
                            (float *) dst->data + t0*M, (int) M, nt, (int) K);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    // pre-Ampere, or GGML_CUDA_AI4_REF=1 on a type that has kernels: original-basis reconstruct + SGEMM, same slabs
    ai4_ref_gemm(ctx, src0, (const float *) src1->data, (float *) dst->data, M, T, K, false, true);
}

// GGML_CUDA_AI4_MMVID_PAIRS = P > 0: MUL_MAT_ID also takes the mmv-ID kernel (tiles of <= 8 pairs) past n_tok = 8 while the
// expected pairs per active expert (uniform top-k routing estimate from the shapes, no host sync) is <= P. Sparse MoE batches
// (512 experts top-10: about 1.05 pairs per active expert at 16 tokens) then read each active expert once at the decode kernel's
// bandwidth instead of running a 64-wide GEMM tile that is 1/64 full. 0 (default) = the n_tok <= 8 rule.
static int ai4_mmvid_pairs() {
    static const int v = ai4_env_int("GGML_CUDA_AI4_MMVID_PAIRS", nullptr, 0);
    return v;
}
static bool ai4_mmvid_route(int n_tok, int n_used, int n_expert, int rate) {
    if (n_tok <= 8) {
        return true;
    }
    const int p = getenv("GGML_CUDA_AI4_MMVID_PAIRS") ? ai4_mmvid_pairs() :
        (ai4_ai23_tuned() && (rate == 2 || rate == 3) ? 4 : 0);
    if (p <= 0 || n_expert <= 0) {
        return false;
    }
    const double act = n_expert*(1.0 - std::pow(1.0 - (double) n_used/n_expert, (double) n_tok));
    return (double) n_used*n_tok <= p*act;
}

template <int LAY, int XSH>
static void ai4_mm_id_kernels(const uint8_t * W, const int8_t * q, const float * s, float * Y, int M, int K, int n_tok,
                              int n_expert, int64_t n_pairs, const ai4k::idmap & idm, int bn, int max_tiles, cudaStream_t stream, int sms) {
    if constexpr (LAY == ai4k::LAY_SB || LAY == ai4k::LAY_U) {
        if (sms && ai4_ampere_mmv(n_tok)) {
            ai4_ampere_width<LAY, XSH, true>(W, q, s, Y, M, 0, K, bn, sms, stream, idm, max_tiles);
            return;
        }
    }
    if (bn == ai4k::MMV_ID_BN) {
        // one mmv block row per token tile of <= 8 pairs (= per active expert unless ids repeat within a token, or past
        // n_tok = 8 under GGML_CUDA_AI4_MMVID_PAIRS: an expert with more pairs gets further tiles)
        GGML_UNUSED(n_tok);
        GGML_UNUSED(n_expert);
        GGML_UNUSED(n_pairs);
        if (ai4_mmvr_try<LAY, XSH, true>(W, q, s, Y, M, n_tok, K, stream, idm, max_tiles)) {
            return;
        }
        const rpb_ai4_mmv k = ai4_mmv_knobs();
        if (k.warps == rpb_ai4_warps_default && k.minb == rpb_ai4_minb_default) {
            ai4k::launch_mmv<2, 6, true, LAY, XSH, true>(W, q, s, Y, M, M, 0, K, stream, idm, max_tiles);
        } else {
            ai4_launch_mmv_tuned<LAY, XSH, true>(W, q, s, Y, M, 0, K, stream, k.warps, k.minb, idm, max_tiles);
        }
    } else if (bn == 128) {
        ai4k::launch_gemm<128, 3, 4, LAY, XSH, true>(W, q, s, Y, M, M, 0, K, stream, idm, max_tiles);
    } else {
        ai4k::launch_gemm<64, 3, 4, LAY, XSH, true>(W, q, s, Y, M, M, 0, K, stream, idm, max_tiles);
    }
}

// MUL_MAT_ID: src0 [K, M, n_expert] AI4/AI4_U expert stack, src1 [K, ne11 (1 or n_used), n_tok] f32, ids [n_used, n_tok],
// dst [M, n_used, n_tok]. Fast path: all src1 rows quantized once, one routing CTA (ai4k::ids_helper) builds the per-expert
// token lists on the device (no host sync, graph-capturable), then mmv-ID (n_tok <= 8) or GEMM-ID over (active expert, token
// tile) blocks. Only the selected experts' bytes are read, so a stack whose unused experts are not resident (the scheduler
// copies only the used experts of a host-resident stack) works.
void ggml_cuda_mul_mat_id_ai4(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];
    GGML_ASSERT(ggml_cuda_ai4_supported(dst));

    const bool unit     = ggml_ai4_is_unit(src0->type);
    const int  K        = (int) src0->ne[0];
    const int  M        = (int) src0->ne[1];
    const int  n_expert = (int) src0->ne[2];
    const int  ne11     = (int) src1->ne[1];
    const int  n_used   = (int) ids->ne[0];
    const int  n_tok    = (int) ids->ne[1];
    const int64_t n_pairs = (int64_t) n_used*n_tok;
    const int64_t n_rows  = (int64_t) ne11*n_tok;
    cudaStream_t stream = ctx.stream();

    if (!ai4_fast_ok(ctx) || ai4_ref_only_type(src0->type)) {
        // reference: reconstruct each selected expert to f32, one SGEMV per (slot, token) pair (host-side routing)
        std::vector<int32_t> h((size_t) n_pairs);
        CUDA_CHECK(cudaMemcpy2DAsync(h.data(), n_used*sizeof(int32_t), ids->data, ids->nb[1], n_used*sizeof(int32_t), n_tok,
                                     cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        ggml_cuda_pool_alloc<float> w(ctx.pool(), (size_t) M*K);
        CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(), stream));
        const float alpha = 1.0f, beta = 0.0f;
        for (int e = 0; e < n_expert; ++e) {
            bool used = false;
            for (int64_t p = 0; p < n_pairs && !used; ++p) {
                used = h[p] == e;
            }
            if (!used) {
                continue;
            }
            ai4_reconstruct_any((const uint8_t *) src0->data + (int64_t) e*src0->nb[2], w.get(), K, M, stream, src0->type);
            for (int64_t p = 0; p < n_pairs; ++p) {
                if (h[p] != e) {
                    continue;
                }
                const int u = (int) (p % n_used), t = (int) (p / n_used);
                const float * x = (const float *) src1->data + ((int64_t) t*ne11 + u % ne11)*K;
                float * y = (float *) dst->data + ((int64_t) t*n_used + u)*M;
                CUBLAS_CHECK(cublasSgemv(ctx.cublas_handle(), CUBLAS_OP_T, K, M, &alpha, w.get(), K, x, 1, &beta, y, 1));
            }
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    GGML_ASSERT(((uintptr_t) src1->data & 15) == 0);
    static const int direct_max = std::min(8, ai4_env_int("GGML_CUDA_AI23_DIRECT", nullptr, ai4_ai23_tuned() ? 4 : 0));
    static const bool fused_x = ai4_env_int("GGML_CUDA_AI23_FUSED_X", nullptr, 0) != 0;
    const int rate = ggml_ai4_rate(src0->type);
    const bool direct = ai4_mmvr_on() && (rate == 2 || rate == 3) && n_tok <= direct_max;
    const bool fused = direct && fused_x && n_tok == 1;
    const int xsh = (!unit || K % 256 == 0) ? 1 : 0;
    ggml_cuda_pool_alloc<int8_t> q(ctx.pool(), (size_t) n_rows*K);
    ggml_cuda_pool_alloc<float>  s(ctx.pool(), (size_t) n_rows*(K >> (7 + xsh)));
    if (!fused) ai4_quantize_x((const float *) src1->data, n_rows, K, xsh, q.get(), s.get(), stream);

    // mmv-ID (n_tok <= 8) takes tiles of MMV_ID_BN pairs, so an expert repeated within a token still fits
    int bn = ai4_mmvid_route(n_tok, n_used, n_expert, ggml_ai4_rate(src0->type)) ? ai4k::MMV_ID_BN : (n_tok >= ai4_bn128_min() ? 128 : 64);
    const int sms = ai4_ampere_sms(ctx);
    if (sms && (src0->type == GGML_TYPE_AI4 || src0->type == GGML_TYPE_AI4_U)) {
        if (ai4_ampere_mmv(n_tok)) {
            bn = ai4_ampere_bn(n_tok);
        }
    }
    const int max_tiles = direct ? (int) n_pairs : ai4k::ids_max_tiles(n_expert, n_pairs, bn);
    GGML_ASSERT(max_tiles <= 65535);
    ggml_cuda_pool_alloc<int> routing(ctx.pool());
    ai4k::idmap idm{};
    if (direct) {
        idm.nb02 = src0->nb[2];
        idm.direct_ids = (const char *) ids->data;
        idm.direct_nb0 = ids->nb[0];
        idm.direct_nb1 = ids->nb[1];
        idm.direct_n_used = n_used;
        idm.direct_ne11 = ne11;
        idm.direct_n_expert = n_expert;
        idm.direct_x = fused ? (const float *) src1->data : nullptr;
    } else {
        routing.alloc((size_t) ai4k::ids_helper_ints(n_expert, n_pairs, max_tiles));
        idm = ai4k::launch_ids_helper(ids->data, ids->nb[0], ids->nb[1], n_used, n_tok, ne11, n_expert, bn,
                                    max_tiles, src0->nb[2], routing.get(), stream);
    }
    const uint8_t * W = (const uint8_t *) src0->data;
    float * Y = (float *) dst->data;
    switch (ai4_lay_of(src0->type)) {
#define AI4_MMID_U(L) \
        case L: \
            if (xsh) { ai4_mm_id_kernels<L, 1>(W, q.get(), s.get(), Y, M, K, n_tok, n_expert, n_pairs, idm, bn, max_tiles, stream, sms); } \
            else     { ai4_mm_id_kernels<L, 0>(W, q.get(), s.get(), Y, M, K, n_tok, n_expert, n_pairs, idm, bn, max_tiles, stream, sms); } \
            break;
        AI4_MMID_U(ai4k::LAY_U2)
        AI4_MMID_U(ai4k::LAY_U3)
#undef AI4_MMID_U
        case ai4k::LAY_S2:
            ai4_mm_id_kernels<ai4k::LAY_S2, 1>(W, q.get(), s.get(), Y, M, K, n_tok, n_expert, n_pairs, idm, bn, max_tiles, stream, sms);
            break;
        case ai4k::LAY_S3:
            ai4_mm_id_kernels<ai4k::LAY_S3, 1>(W, q.get(), s.get(), Y, M, K, n_tok, n_expert, n_pairs, idm, bn, max_tiles, stream, sms);
            break;
        case ai4k::LAY_S:
            ai4_mm_id_kernels<ai4k::LAY_S,  1>(W, q.get(), s.get(), Y, M, K, n_tok, n_expert, n_pairs, idm, bn, max_tiles, stream, sms);
            break;
        case ai4k::LAY_S5:
            ai4_mm_id_kernels<ai4k::LAY_S5, 1>(W, q.get(), s.get(), Y, M, K, n_tok, n_expert, n_pairs, idm, bn, max_tiles, stream, sms);
            break;
        case ai4k::LAY_S6:
            ai4_mm_id_kernels<ai4k::LAY_S6, 1>(W, q.get(), s.get(), Y, M, K, n_tok, n_expert, n_pairs, idm, bn, max_tiles, stream, sms);
            break;
        case ai4k::LAY_U5:
            if (xsh) {
                ai4_mm_id_kernels<ai4k::LAY_U5, 1>(W, q.get(), s.get(), Y, M, K, n_tok, n_expert, n_pairs, idm, bn, max_tiles, stream, sms);
            } else {
                ai4_mm_id_kernels<ai4k::LAY_U5, 0>(W, q.get(), s.get(), Y, M, K, n_tok, n_expert, n_pairs, idm, bn, max_tiles, stream, sms);
            }
            break;
        case ai4k::LAY_SB:
            ai4_mm_id_kernels<ai4k::LAY_SB, 1>(W, q.get(), s.get(), Y, M, K, n_tok, n_expert, n_pairs, idm, bn, max_tiles, stream, sms);
            break;
        default:
            if (xsh) {
                ai4_mm_id_kernels<ai4k::LAY_U, 1>(W, q.get(), s.get(), Y, M, K, n_tok, n_expert, n_pairs, idm, bn, max_tiles, stream, sms);
            } else {
                ai4_mm_id_kernels<ai4k::LAY_U, 0>(W, q.get(), s.get(), Y, M, K, n_tok, n_expert, n_pairs, idm, bn, max_tiles, stream, sms);
            }
            break;
    }
    CUDA_CHECK(cudaGetLastError());
}
