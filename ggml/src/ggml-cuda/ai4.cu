// AI+ unit types (AITk_U) CUDA backend entry: reference and fast-path matmul dispatch.
#include "ai4-mm.cuh"
#include "ai4.cuh"
#include "ai4-kern.cuh"
#include "ai4-ampere.cuh"
#include "magic2-v2.cuh"
#include "../ggml-aif25.h"
#include "rpb-knobs.h"
#include "quantize.cuh"
#include "moe-weighted-reduction.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

// Kernel selection (read once per process), for A/B runs:
//   GGML_CUDA_AI4_REF=1        reference path: reconstruct to f32 + cuBLAS SGEMM
//   GGML_CUDA_AI4_REF_SLAB_MIB=n   cap in MiB of that f32 buffer (default 256). Rows are reconstructed and multiplied
//                             in slabs of a multiple of 16. 0 or negative: one slab, the whole matrix (the old buffer).
//   GGML_CUDA_AI4_REF_SLAB_ROWS=n  test only. Slab height in rows; a positive value overrides the MiB cap. Rounded
//                             down to a multiple of 16 (the storage unit). Not a runtime tuning knob.
//   GGML_CUDA_AI4_BN128_MIN=n  N at which the GEMM switches from 64- to 128-wide token tiles (default 512)
//   GGML_CUDA_AI4_MMV_STAGES=2|3|4  cp.async ring depth of the N <= 8 kernel (default 2: fastest of 2/3/4 at verify
//                             widths 5-8)
//   GGML_CUDA_AI4_MMV_XSMEM=1  N <= 8 kernel stages the activations in shared memory (first version, 3 stages) instead
//                             of reading them into registers
//   GGML_CUDA_AI4_MMV_WARPS=2|4|8   warps per block of the stages-2 register kernel (default 4)
//   GGML_CUDA_AI4_MMV_MINB=2|4|6|8  launch_bounds min blocks of that kernel (default 6).
//                             8 warps with min blocks 8 does not fit sm_86 and uses 4 and 6.
//                             Stages 3/4 and XSMEM ignore both and keep 4 warps.
//   GGML_CUDA_AI4_AMPERE_MMV=1   rate-4 width 9-32 family on sm_80/sm_86 (default off)
//   GGML_CUDA_AI4_AMPERE_MMV_MIN_N/MAX_N   inclusive per-card tuning range within 1-64 (defaults 9/32)
//   GGML_CUDA_AI4_AMPERE_GEMM    retired; keeps baseline prefill after measured regressions
//   GGML_CUDA_AI4_PREFILL_V2=1  experimental rate-4 decode-once dense prefill, N >= 512 (default off)
//   GGML_CUDA_AI23_MMVR=1    rate-2/3 decode with fixed stage offsets (default off)
//   GGML_CUDA_AI23_MMVR_ID=1 rate-2/3 mmvr for expert stacks (MUL_MAT_ID) only (default: follows GGML_CUDA_AI23_MMVR)
//   GGML_CUDA_AI23_ID_REF=0  AIT2_U/AIT3_U MUL_MAT_ID run the device-routed IMMA kernels (default 1 = reference path)
//   GGML_CUDA_AI_ID_SKIP_ZERO=1  zero the outputs of skipped (out-of-stack) expert ids even without an expert bank
//                            (default: only when a model with hot-bank tensors is loaded)
//   GGML_CUDA_AI23_DIRECT=N  MUL_MAT_ID widths up to N (max 8) skip the routing helper (default 4 with
//                            GGML_CUDA_AI23_TUNED=1, else 0; lane measurement used 6)
//   GGML_CUDA_AI23_TUNED=1   select measured rate-2/3 decode and tile defaults (default off)
//   GGML_CUDA_AI23_MMVR_STAGES=4|8  deeper dense fixed-K four-warp weight prefetch (default 2)
//   GGML_CUDA_AI23_MIXED_STAGES=2|4|8  hot-route weight prefetch depth of the banked mixed MoE GEMV (default 4)
//   GGML_CUDA_AI23_MMVR_UNROLL=1|2|4|8  dense fixed-K loop unroll (tuned 8 at long K, 4 otherwise)
//   GGML_CUDA_AI23_MMVR_SINGLE_N=1  omit unused odd-column accumulators for dense N=1 (default off)
//   GGML_CUDA_AI23_MMVR_SINGLE_ID=1  omit unused odd-column accumulators for direct experts (default off)
//   GGML_CUDA_AI23_SMALL_SPLIT=2|4  split small-width fixed-K dense products (default off)
//   GGML_CUDA_AI23_FUSED_X=1  quantize activations inside direct one-token expert blocks (default off)
//   GGML_CUDA_AI23_DIRECT=1..8  skip expert grouping through this token width in MMVR (default off)
//   GGML_CUDA_AI23_ID_BN=32|64|128  override rate-2/3 routed prompt GEMM tile width (default automatic)
//   GGML_CUDA_AI23_ID_STAGES=2|3  routed rate-2/3 prompt staging depth (default 3)
//   GGML_CUDA_AI23_ID_SLABS=4|8  routed rate-2/3 prompt row slabs per block (default 8)
//   GGML_CUDA_AI23_ID_THREADS=256  smaller routed prompt helper for <=1024 experts (default 1024)
//   GGML_CUDA_AI23_ID_TIGHT_GRID=1  bound routed prompt tiles after each active expert's first pair (default off)
//   GGML_CUDA_AI23_ID_ROUTE_AGG=1  combine routed prompt counter updates within each warp (default off)
//   GGML_CUDA_AI23_ID_FIXED_LOAD=1  unroll fixed-size routed prompt staging copies (default off)
//   GGML_CUDA_AI23_ID_ZERO=1  zero skipped prompt columns in the routing helper (default off)
//   GGML_CUDA_AI23_DENSE_TILE=1..29  dense shared-activation tile controls for 9..64 tokens (default off)
//                             22/23 swap quantized activation halves to avoid LDS64 bank conflicts
//   GGML_CUDA_AI23_XPERM=1  vector activation loads for whole-row dense rate-2 kernels (default off)
//   GGML_CUDA_AI23_RESOURCE=1   print CUDA kernel resource and occupancy queries once (diagnostics only)
//   GGML_CUDA_AI23_SPLIT_K=2|4|8|16  split dense K chunks into partial tiles for 9..64 tokens (default off)
//   GGML_CUDA_AI23_NARROW_WARPS=2|4|8  override narrow dense K parallelism (default automatic)
//   GGML_CUDA_AI23_NARROW=1|2|3   one-slab dense tiles for 9..64 tokens at rates 2 and 3 (default off)
//                             2 fixes K/loop slots; 3 also uses whole-row rate-2 decode
//   GGML_CUDA_AI5_REF=1        AIT5_U takes the rate-k reference path (reconstruct + SGEMM) instead of its rate-5 IMMA
//                             kernels (an A/B check of those kernels)
//   GGML_CUDA_AI23_FAST=1      rates 2 and 3 run their IMMA kernels (default: rate-k reference path)
static int ai4_env_int(const char * name, const char * /*unused*/, int def) {
    const char * e = getenv(name);
    return e ? atoi(e) : def;
}
// fork: routed AI+ unit kernels; set GGML_CUDA_AIKERN=0 for the prior path.
static bool ai4_aikern() {
    static const bool v = ai4_env_int("GGML_CUDA_AIKERN", nullptr, 1) != 0;
    return v;
}
static bool ai4_ai23_tuned() {
    static const bool v = ai4_env_int("GGML_CUDA_AI23_TUNED", nullptr, 0) != 0;
    return v;
}
static bool ai4_ref() {
    static const bool v = ai4_env_int("GGML_CUDA_AI4_REF", nullptr, 0) != 0;
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
// Keep explicit development controls exact. Automatic choices share one resolver so the
// activation scratch layout and its consumer always agree.
struct ai23_dense_policy {
    int tile;
    int narrow;
    int splits;
};
static ai23_dense_policy ai23_dense_config(int M, int N, int K, int sms, int rate) {
    static const int tile = ai4_env_int("GGML_CUDA_AI23_DENSE_TILE", nullptr, 0);
    static const int narrow = ai4_env_int("GGML_CUDA_AI23_NARROW", nullptr, 0);
    static const int splits = ai4_env_int("GGML_CUDA_AI23_SPLIT_K", nullptr, 0);
    static const bool explicit_wide = getenv("GGML_CUDA_AI23_DENSE_TILE") ||
                                      getenv("GGML_CUDA_AI23_NARROW") ||
                                      getenv("GGML_CUDA_AI23_SPLIT_K");
    ai23_dense_policy p{tile, narrow, splits};
    if (!ai4_ai23_tuned() || explicit_wide || !sms || (rate != 2 && rate != 3) ||
            N <= 8 || N > 64 || (K != 5120 && K != 17408)) {
        return p;
    }
    // Eight-slab fixed-copy tiles cover 128 rows. The longest reduction needs
    // additional K blocks; small short-K grids use one-slab decode through BN32.
    const int blocks = (M + 127)/128;
    p.tile = 24;
    if (N > 32 || K == 17408) {
        if (4*blocks < 3*sms) p.splits = 4;
    } else if (N <= 16 || 4*blocks < 3*sms) {
        p.tile = 0;
        p.narrow = 2;
        if (N <= 16 && rate == 3 && 4*blocks >= 3*sms) {
            p.tile = 28;
            p.narrow = 0;
        }
    }
    return p;
}
static int ai4_ampere_bn(int N) {
    return N <= 8 ? 8 : N <= 16 ? 16 : N <= 32 ? 32 : 64;
}

// Only whole-row dense kernels consume the permuted activation scratch.
template <int BN, int STAGES = 3, int NTB = 4, int LAY = ai4k::LAY_AIT4_U, int XSH = 1, bool ID = false, int N_WARPS = 1, int M_SLABS = ai4k::GEMM_SLABS, bool PREDECODED = false, int MINB = 1, bool STREAM_A = false, bool J_MAJOR = false, bool X8 = false, bool WHOLE = false, bool SPLIT = false, bool SWAP8 = false, bool FIXED_LOAD = false, bool XOR_X = false>
static void ai23_launch_gemm(const uint8_t * W, const int8_t * Xq, const float * Xs, float * Y, int64_t ldy,
                             int M, int N, int K, cudaStream_t stream, const ai4k::idmap & idm = ai4k::idmap{}, int max_experts = 0,
                             const int8_t * decoded_weights = nullptr, int split_k = 1) {
    if constexpr (WHOLE && !ID) {
        if (ai4_env_int("GGML_CUDA_AI23_XPERM", nullptr, 0)) {
            ai4k::launch_gemm<BN, STAGES, NTB, LAY, XSH, ID, N_WARPS, M_SLABS, PREDECODED, MINB, STREAM_A, J_MAJOR, X8, WHOLE, SPLIT, SWAP8, FIXED_LOAD, true, XOR_X>(W, Xq, Xs, Y, ldy, M, N, K, stream, idm, max_experts, decoded_weights, split_k);
            return;
        }
    }
    ai4k::launch_gemm<BN, STAGES, NTB, LAY, XSH, ID, N_WARPS, M_SLABS, PREDECODED, MINB, STREAM_A, J_MAJOR, X8, WHOLE, SPLIT, SWAP8, FIXED_LOAD, false, XOR_X>(W, Xq, Xs, Y, ldy, M, N, K, stream, idm, max_experts, decoded_weights, split_k);
}

template <int BN, int LAY, int XSH, bool ID, int WARPS, int KT>
static void ai23_launch_narrow_kt(const uint8_t * W, const int8_t * q, const float * s, float * Y,
                                 int M, int N, int K, cudaStream_t stream, const ai4k::idmap & idm, int max_tiles, bool perm) {
    if constexpr (ai4k::lay_rate<LAY>() == 2) {
        if (perm) {
            if constexpr (!ID) {
                if (ai4_env_int("GGML_CUDA_AI23_XPERM", nullptr, 0)) {
                    ai4k::launch_ampere_mmv<BN, LAY, XSH, ID, WARPS, KT, true, true>(W, q, s, Y, M, M, N, K, stream, idm, max_tiles);
                    return;
                }
            }
            ai4k::launch_ampere_mmv<BN, LAY, XSH, ID, WARPS, KT, true>(W, q, s, Y, M, M, N, K, stream, idm, max_tiles);
            return;
        }
    }
    ai4k::launch_ampere_mmv<BN, LAY, XSH, ID, WARPS, KT>(W, q, s, Y, M, M, N, K, stream, idm, max_tiles);
}

template <int BN, int LAY, int XSH, bool ID, int WARPS>
static void ai4_launch_ampere_w(const uint8_t * W, const int8_t * q, const float * s, float * Y,
                               int M, int N, int K, cudaStream_t stream, const ai4k::idmap & idm, int max_tiles, int narrow) {
    if constexpr (ai4k::lay_rate<LAY>() == 2 || ai4k::lay_rate<LAY>() == 3) {
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
    const int narrow = ID ? ai4_env_int("GGML_CUDA_AI23_NARROW", nullptr, 0) :
                            ai23_dense_config(M, N, K, sms, ai4k::lay_rate<LAY>()).narrow;
    if constexpr (ai4k::lay_rate<LAY>() == 2 || ai4k::lay_rate<LAY>() == 3) {
        static const int warps = ai4_env_int("GGML_CUDA_AI23_NARROW_WARPS", nullptr, 0);
        if (warps == 8) {
            ai4_launch_ampere_w<BN, LAY, XSH, ID, 8>(W, q, s, Y, M, N, K, stream, idm, max_tiles, narrow);
            return;
        }
        if (warps == 4) {
            ai4_launch_ampere_w<BN, LAY, XSH, ID, 4>(W, q, s, Y, M, N, K, stream, idm, max_tiles, narrow);
            return;
        }
        if (warps == 2) {
            ai4_launch_ampere_w<BN, LAY, XSH, ID, 2>(W, q, s, Y, M, N, K, stream, idm, max_tiles, narrow);
            return;
        }
    }
    if (sms >= 80 && BN == 8 && K >= 1024) {
        ai4_launch_ampere_w<BN, LAY, XSH, ID, 8>(W, q, s, Y, M, N, K, stream, idm, max_tiles, narrow);
    } else if (sms > 28 && BN <= 32) {
        ai4_launch_ampere_w<BN, LAY, XSH, ID, 4>(W, q, s, Y, M, N, K, stream, idm, max_tiles, narrow);
    } else {
        ai4_launch_ampere_w<BN, LAY, XSH, ID, 2>(W, q, s, Y, M, N, K, stream, idm, max_tiles, narrow);
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
    static const int v = ai4_env_int("GGML_CUDA_AI4_BN128_MIN", nullptr, 512);
    return v;
}
static int ai4_mmv_stages() {
    static const int v = ai4_env_int("GGML_CUDA_AI4_MMV_STAGES", nullptr, 2);
    return v;
}
static bool ai4_mmv_xsmem() {
    static const bool v = ai4_env_int("GGML_CUDA_AI4_MMV_XSMEM", nullptr, 0) != 0;
    return v;
}

static_assert(ai4k::mmv_stage_bytes<true, ai4k::LAY_AIT4_U>() == rpb_ai4_stage_bytes(true, false), "ai4 stage bytes");
static_assert(ai4k::mmv_stage_bytes<true, ai4k::LAY_AIT5_U>() == rpb_ai4_stage_bytes(true, true), "ai4 k5 stage bytes");
static_assert(ai4k::mmv_stage_bytes<false, ai4k::LAY_AIT5_U>() == rpb_ai4_stage_bytes(false, true), "ai4 xsmem k5 stage bytes");
static_assert(ai4k::mmv_smem_bytes<2, true, ai4k::LAY_AIT4_U, 4>() == rpb_ai4_dynamic_smem(4, 2, true, false), "ai4 default dynamic smem");

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

// Activations: rotate + int8-quantize X [N rows of K] once. XSH = 1: H256 and one scale per 256 (track B's kernel, used
// when K % 256 == 0); XSH = 0: H128 and one scale per 128 (K % 256 != 0).
// gridDim.y carries the rows (<= 65535), so longer inputs run in pieces.
template <bool SWAP8 = false, bool XPERM = false, bool LINEAR = false>
static void ai4_quantize_x(const float * X, int64_t N, int K, int xsh, int8_t * q, float * s, cudaStream_t stream) {
    const int64_t ns = K >> (7 + xsh);
    for (int64_t r0 = 0; r0 < N; r0 += 65535) {
        const int nr = (int) std::min<int64_t>(65535, N - r0);
        if (xsh) {
            rq4::launch_quantize_act<SWAP8, XPERM, LINEAR>(X + r0*K, K, q + r0*K, s + r0*ns, K, nr, stream);
        } else {
            ai4k::launch_quantize_act128<SWAP8, XPERM, LINEAR>(X + r0*K, K, q + r0*K, s + r0*ns, K, nr, stream);
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
// GGML_CUDA_AI23_MMVR_ID=1 (opt-in): expert stacks (MUL_MAT_ID) take mmvr. Lane note: on a 3090 the rate-2/3 mmvr
// direct path was 28-42% faster than the mmv-ID kernel at widths 1-6 for 640x2560 / 2560x640 experts (kernel-only,
// no end-to-end speed test yet). Unset: follows GGML_CUDA_AI23_MMVR, as before.
static bool ai4_mmvr_on_id() {
    static const bool v = getenv("GGML_CUDA_AI23_MMVR_ID") ? ai4_env_int("GGML_CUDA_AI23_MMVR_ID", nullptr, 0) != 0 : (ai4_mmvr_on() || ai4_aikern());
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
    else { return XSH == 1 ? KT % 256 == 0 : KT % 256 != 0; }
}

template <int LAY, int XSH, bool ID, int KT, int STAGES = 2, int WARPS = 4, int MINB = 6, bool DIRECT = false, bool FUSED_X = false, bool SINGLE_N = false>
static void ai23_launch_mmvr(const uint8_t * W, const int8_t * q, const float * s, float * Y, int64_t ldy,
                             int M, int N, int K, cudaStream_t stream, const ai4k::idmap & idm, int max_tiles) {
    static_assert(!SINGLE_N || !ID || DIRECT, "single-column dense or direct scope");
    if constexpr (!ID && !SINGLE_N) {
        static const bool single_n = ai4_env_int("GGML_CUDA_AI23_MMVR_SINGLE_N", nullptr, 0) != 0;
        if (single_n && N == 1) {
            ai23_launch_mmvr<LAY, XSH, ID, KT, STAGES, WARPS, MINB, DIRECT, FUSED_X, true>(W, q, s, Y, ldy, M, N, K, stream, idm, max_tiles);
            return;
        }
    }
    if constexpr (ID && DIRECT && !SINGLE_N) {
        static const bool single_id = ai4_env_int("GGML_CUDA_AI23_MMVR_SINGLE_ID", nullptr, 0) != 0;
        if (single_id || idm.direct_zero) {
            ai23_launch_mmvr<LAY, XSH, ID, KT, STAGES, WARPS, MINB, DIRECT, FUSED_X, true>(W, q, s, Y, ldy, M, N, K, stream, idm, max_tiles);
            return;
        }
    }
    if constexpr (SINGLE_N && !DIRECT) GGML_ASSERT(N == 1);
    if constexpr (!ID && WARPS == 4 && (KT == 5120 || KT == 17408)) {
        static const int stages = ai4_env_int("GGML_CUDA_AI23_MMVR_STAGES", nullptr, 2);
        if (stages == 4) {
            ai4k::launch_mmvr<LAY, XSH, ID, KT, 4, WARPS, MINB, DIRECT, FUSED_X, 1, 1, SINGLE_N>(W, q, s, Y, ldy, M, N, K, stream, idm, max_tiles);
            return;
        }
        if (stages == 8) {
            ai4k::launch_mmvr<LAY, XSH, ID, KT, 8, WARPS, MINB, DIRECT, FUSED_X, 1, 1, SINGLE_N>(W, q, s, Y, ldy, M, N, K, stream, idm, max_tiles);
            return;
        }
    }
    if constexpr (!ID && ((WARPS == 4 && (KT == 5120 || KT == 17408)) ||
                          ((WARPS == 8 || WARPS == 16) && KT == 17408))) {
        static const int unroll = ai4_env_int("GGML_CUDA_AI23_MMVR_UNROLL", nullptr, ai4_ai23_tuned() ? (KT == 17408 ? 8 : 4) : 1);
        if (unroll == 2) {
            ai4k::launch_mmvr<LAY, XSH, ID, KT, STAGES, WARPS, MINB, DIRECT, FUSED_X, 1, 2, SINGLE_N>(W, q, s, Y, ldy, M, N, K, stream, idm, max_tiles);
            return;
        }
        if (unroll == 4) {
            ai4k::launch_mmvr<LAY, XSH, ID, KT, STAGES, WARPS, MINB, DIRECT, FUSED_X, 1, 4, SINGLE_N>(W, q, s, Y, ldy, M, N, K, stream, idm, max_tiles);
            return;
        }
        if (unroll == 8) {
            ai4k::launch_mmvr<LAY, XSH, ID, KT, STAGES, WARPS, MINB, DIRECT, FUSED_X, 1, 8, SINGLE_N>(W, q, s, Y, ldy, M, N, K, stream, idm, max_tiles);
            return;
        }
    }
    ai4k::launch_mmvr<LAY, XSH, ID, KT, STAGES, WARPS, MINB, DIRECT, FUSED_X, 1, 1, SINGLE_N>(W, q, s, Y, ldy, M, N, K, stream, idm, max_tiles);
}

static bool ai4_spw_enabled(int M, int K, const ai4k::idmap & idm, int max_tiles) {
    static const int mode = ai4_env_int("GGML_CUDA_AIKERN_SPW", nullptr, 2);
    if (!ai4_aikern() || mode == 0 || idm.direct_ids == nullptr || idm.direct_x != nullptr || M % 16 != 0) {
        return false;
    }
    const int n_tok = max_tiles/idm.direct_n_used;
    return mode != 2 || K == 640 || n_tok >= 4;
}

// fork AIPFIX D7 (defined after moe_mixed_group): cached AI+ experts, routes of one expert share a weight read
template <int LAY, int XSH, int KT>
static bool ai4_table_grouped_try(const int8_t * q, const float * s, float * Y, int M, int K, cudaStream_t stream,
                                  const ai4k::idmap & idm, int max_tiles);

// fork: each warp owns 16 rows and scans K without a cross-warp sum.
template <int LAY, int XSH, bool ID, int KT>
static bool ai4_spw_try(const uint8_t * W, const int8_t * q, const float * s, float * Y, int M, int K,
                        cudaStream_t stream, const ai4k::idmap & idm, int max_tiles) {
    if constexpr (!ID) {
        return false;
    } else {
        if (!ai4_spw_enabled(M, K, idm, max_tiles)) {
            return false;
        }
        if (ai4_table_grouped_try<LAY, XSH, KT>(q, s, Y, M, K, stream, idm, max_tiles)) {
            return true;
        }
        ai4k::launch_mmvr<LAY, XSH, ID, KT, 2, 4, 6, true, false, 1, 1, false, true>(W, q, s, Y, M, M, 1, K, stream, idm, max_tiles);
        return true;
    }
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
        int cfg = ai4_mmvr_cfg();
        if constexpr (!ID && (KT == 5120 || KT == 17408)) {
            // Small dense grids need the four-block register limit at long K;
            // shorter K keeps four warps and a larger block count.
            if (cfg == 0 && ai4_ai23_tuned()) cfg = KT == 17408 ? 84 : 48;
        }
        if constexpr (!ID && KT == 17408) {
            if (cfg == 162) {
                ai23_launch_mmvr<LAY, XSH, ID, KT, 2, 16, 2>(W, q, s, Y, M, M, N, K, stream, idm, max_tiles);
                return true;
            }
        }
        const bool two_warps = cfg == 212 || (cfg == 0 && ID && N > 1 && K >= 1024);
        if constexpr (ID) {
            if (ai4_spw_try<LAY, XSH, ID, KT>(W, q, s, Y, M, K, stream, idm, max_tiles)) {
                return true;
            }
            if (idm.direct_ids != nullptr) {
                if (idm.direct_x != nullptr) {
                    ai23_launch_mmvr<LAY, XSH, ID, KT, 2, 4, 6, true, true>(W, q, s, Y, M, M, N, K, stream, idm, max_tiles);
                    return true;
                }
                if (two_warps) {
                    ai23_launch_mmvr<LAY, XSH, ID, KT, 2, 2, 12, true>(W, q, s, Y, M, M, N, K, stream, idm, max_tiles);
                } else {
                    ai23_launch_mmvr<LAY, XSH, ID, KT, 2, 4, 6, true>(W, q, s, Y, M, M, N, K, stream, idm, max_tiles);
                }
                return true;
            }
        }
        if (two_warps) {
            ai23_launch_mmvr<LAY, XSH, ID, KT, 2, 2, 12>(W, q, s, Y, M, M, N, K, stream, idm, max_tiles);
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
    constexpr bool r4 = LAY == ai4k::LAY_AIT4_U && ID;
    if constexpr (rate != 2 && rate != 3 && !r4) {
        return false;
    } else {
        if (r4 ? !ai4_aikern() : !(ID ? ai4_mmvr_on_id() : ai4_mmvr_on())) {
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
        case 2: ai23_launch_gemm<BN, 2, 1, LAY, XSH, false, 1, 4>(W, q, s, Y, M, M, N, K, stream); break;
        case 3: ai23_launch_gemm<BN, 2, 1, LAY, XSH, false, 2, 4>(W, q, s, Y, M, M, N, K, stream); break;
        case 4: ai23_launch_gemm<BN, 2, (BN <= 16 ? 1 : 2), LAY, XSH, false, 2, 4>(W, q, s, Y, M, M, N, K, stream); break;
        case 5:
            if constexpr (BN >= 32) { ai23_launch_gemm<BN, 2, 1, LAY, XSH, false, 4, 2>(W, q, s, Y, M, M, N, K, stream); }
            else { ai23_launch_gemm<BN, 2, 1, LAY, XSH, false, 2, 4>(W, q, s, Y, M, M, N, K, stream); }
            break;
        case 6:
            if constexpr (BN >= 32) { ai23_launch_gemm<BN, 2, 1, LAY, XSH, false, 4, 4, false, 2>(W, q, s, Y, M, M, N, K, stream); }
            else { ai23_launch_gemm<BN, 2, 1, LAY, XSH, false, 2, 4>(W, q, s, Y, M, M, N, K, stream); }
            break;
        case 8: ai23_launch_gemm<BN, 2, 1, LAY, XSH, false, 2, 4, false, 1, true>(W, q, s, Y, M, M, N, K, stream); break;
        case 9:
            if constexpr (BN >= 32) { ai23_launch_gemm<BN, 2, 1, LAY, XSH, false, 4, 4, false, 2, true>(W, q, s, Y, M, M, N, K, stream); }
            else { ai23_launch_gemm<BN, 2, 1, LAY, XSH, false, 2, 4, false, 1, true>(W, q, s, Y, M, M, N, K, stream); }
            break;
        case 10:
            if constexpr (BN >= 32) { ai23_launch_gemm<BN, 2, 1, LAY, XSH, false, 4, 2, false, 1, true>(W, q, s, Y, M, M, N, K, stream); }
            else { ai23_launch_gemm<BN, 2, 1, LAY, XSH, false, 2, 4, false, 1, true>(W, q, s, Y, M, M, N, K, stream); }
            break;
        case 11: ai23_launch_gemm<BN, 2, 1, LAY, XSH, false, 1, 8, false, 2, false, true>(W, q, s, Y, M, M, N, K, stream); break;
        case 12: ai23_launch_gemm<BN, 3, 1, LAY, XSH, false, 1, 8, false, 2, false, true>(W, q, s, Y, M, M, N, K, stream); break;
        case 13: ai23_launch_gemm<BN, 2, 1, LAY, XSH, false, 1, 4, false, 4, false, true>(W, q, s, Y, M, M, N, K, stream); break;
        case 14: ai23_launch_gemm<BN, 3, (BN <= 16 ? 2 : 4), LAY, XSH, false, 1, 8, false, 2>(W, q, s, Y, M, M, N, K, stream); break;
        case 15: ai23_launch_gemm<BN, 2, 1, LAY, XSH, false, 1, 8, false, 2>(W, q, s, Y, M, M, N, K, stream); break;
        case 16: ai23_launch_gemm<BN, 2, 1, LAY, XSH, false, 1, 8, false, 2, false, true, true>(W, q, s, Y, M, M, N, K, stream); break;
        case 17: ai23_launch_gemm<BN, 3, 1, LAY, XSH, false, 1, 8, false, 2, false, true, true>(W, q, s, Y, M, M, N, K, stream); break;
        case 18:
            if constexpr (ai4k::lay_rate<LAY>() == 2) {
                ai23_launch_gemm<BN, 2, 1, LAY, XSH, false, 1, 8, false, 2, false, true, true, true>(W, q, s, Y, M, M, N, K, stream);
            } else {
                ai23_launch_gemm<BN, 2, 1, LAY, XSH, false, 1, 8, false, 2, false, true, true>(W, q, s, Y, M, M, N, K, stream);
            }
            break;
        case 19:
            if constexpr (ai4k::lay_rate<LAY>() == 2) {
                ai23_launch_gemm<BN, 3, 1, LAY, XSH, false, 1, 8, false, 2, false, true, true, true>(W, q, s, Y, M, M, N, K, stream);
            } else {
                ai23_launch_gemm<BN, 3, 1, LAY, XSH, false, 1, 8, false, 2, false, true, true>(W, q, s, Y, M, M, N, K, stream);
            }
            break;
        case 20: ai23_launch_gemm<BN, 2, 1, LAY, XSH, false, 1, 4, false, 4, false, true, true>(W, q, s, Y, M, M, N, K, stream); break;
        case 21:
            if constexpr (ai4k::lay_rate<LAY>() == 2) {
                ai23_launch_gemm<BN, 2, 1, LAY, XSH, false, 1, 4, false, 4, false, true, true, true>(W, q, s, Y, M, M, N, K, stream);
            } else {
                ai23_launch_gemm<BN, 2, 1, LAY, XSH, false, 1, 4, false, 4, false, true, true>(W, q, s, Y, M, M, N, K, stream);
            }
            break;
        case 22: ai23_launch_gemm<BN, 2, 1, LAY, XSH, false, 1, 8, false, 2, false, true, false, false, false, true>(W, q, s, Y, M, M, N, K, stream); break;
        case 23: ai23_launch_gemm<BN, 2, 1, LAY, XSH, false, 1, 4, false, 4, false, true, false, false, false, true>(W, q, s, Y, M, M, N, K, stream); break;
        case 24: ai23_launch_gemm<BN, 2, 1, LAY, XSH, false, 1, 8, false, 2, false, true, false, false, false, false, true>(W, q, s, Y, M, M, N, K, stream); break;
        case 25: ai23_launch_gemm<BN, 2, 1, LAY, XSH, false, 1, 8, false, 2, false, true, false, false, false, true, true>(W, q, s, Y, M, M, N, K, stream); break;
        case 26: ai23_launch_gemm<BN, 2, 1, LAY, XSH, false, 1, 4, false, 4, false, true, false, false, false, false, true>(W, q, s, Y, M, M, N, K, stream); break;
        case 27: ai23_launch_gemm<BN, 2, 1, LAY, XSH, false, 1, 4, false, 4, false, true, false, false, false, true, true>(W, q, s, Y, M, M, N, K, stream); break;
        case 28: ai23_launch_gemm<BN, 2, 1, LAY, XSH, false, 1, 4, false, 4, false, true, false, false, false, false, true, true>(W, q, s, Y, M, M, N, K, stream); break;
        case 29: ai23_launch_gemm<BN, 2, 1, LAY, XSH, false, 1, 8, false, 2, false, true, false, false, false, false, true, true>(W, q, s, Y, M, M, N, K, stream); break;
        case 7: ai23_launch_gemm<BN, 2, 1, LAY, XSH>(W, q, s, Y, M, M, N, K, stream); break;
        default: ai23_launch_gemm<BN, 3, (BN <= 16 ? 2 : 4), LAY, XSH>(W, q, s, Y, M, M, N, K, stream); break;
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
    if constexpr (LAY == ai4k::LAY_AIT4_U) {
        if (sms && ai4_ampere_mmv(N)) {
            ai4_ampere_width<LAY, XSH, false>(W, q, s, Y, M, N, K, ai4_ampere_bn(N), sms, stream);
            return;
        }
    }
    if constexpr (ai4k::lay_rate<LAY>() == 2 || ai4k::lay_rate<LAY>() == 3) {
        const ai23_dense_policy policy = ai23_dense_config(M, N, K, sms, ai4k::lay_rate<LAY>());
        const int dense_tile = policy.tile;
        if (dense_tile && N > 8 && N <= 64) {
            if (N <= 16) { ai23_dense_tile<16, LAY, XSH>(W, q, s, Y, M, N, K, stream, dense_tile); }
            else if (N <= 32) { ai23_dense_tile<32, LAY, XSH>(W, q, s, Y, M, N, K, stream, dense_tile); }
            else { ai23_dense_tile<64, LAY, XSH>(W, q, s, Y, M, N, K, stream, dense_tile); }
            return;
        }
        const bool narrow = policy.narrow != 0;
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
        ai23_launch_gemm<128, 3, 4, LAY, XSH>(W, q, s, Y, M, M, N, K, stream);
    } else {
        ai23_launch_gemm<64, 3, 4, LAY, XSH>(W, q, s, Y, M, M, N, K, stream);
    }
}

// ai4-kern LAY of an AI+ unit type; xsh = 1 unless K % 256 != 0.
static int ai4_lay_of(ggml_type t) {
    switch (t) {
        case GGML_TYPE_AIT2_U: return ai4k::LAY_AIT2_U;
        case GGML_TYPE_AIT2P_U: return ai4k::LAY_AIT2P_U;
        case GGML_TYPE_AIT3_U: return ai4k::LAY_AIT3_U;
        case GGML_TYPE_AIT4_U: return ai4k::LAY_AIT4_U;
        case GGML_TYPE_AIT5_U: return ai4k::LAY_AIT5_U;
        default: GGML_ABORT("not an AI+ unit type");
    }
}

// Rate-2 / rate-3 IMMA kernels; GGML_CUDA_AI23_FAST overrides the routed AI+ default.
static bool ai4_ai23_fast() {
    static const bool v = ai4_env_int("GGML_CUDA_AI23_FAST", nullptr, ai4_aikern() ? 1 : 0) != 0;
    return v;
}

template <int BN, int LAY, int XSH>
static void ai23_split_tile(const uint8_t * W, const int8_t * q, const float * s, float * partial,
                             int M, int N, int K, int splits, int tile, cudaStream_t stream) {
    constexpr bool RATE2 = ai4k::lay_rate<LAY>() == 2;
    if (tile >= 24 && tile <= 27) {
        if (tile == 25 || tile == 27) {
            ai23_launch_gemm<BN, 2, 1, LAY, XSH, false, 1, 8, false, 2, false, true, false, false, true, true, true>(W, q, s, partial, M, M, N, K, stream, ai4k::idmap{}, 0, nullptr, splits);
        } else {
            ai23_launch_gemm<BN, 2, 1, LAY, XSH, false, 1, 8, false, 2, false, true, false, false, true, false, true>(W, q, s, partial, M, M, N, K, stream, ai4k::idmap{}, 0, nullptr, splits);
        }
        return;
    }
    if (tile == 22 || tile == 23) {
        ai23_launch_gemm<BN, 2, 1, LAY, XSH, false, 1, 8, false, 2, false, true, false, false, true, true>(W, q, s, partial, M, M, N, K, stream, ai4k::idmap{}, 0, nullptr, splits);
        return;
    }
    if constexpr (RATE2) {
        if (tile != 16 && tile != 17) {
            ai23_launch_gemm<BN, 2, 1, LAY, XSH, false, 1, 8, false, 2, false, true, true, true, true>(W, q, s, partial, M, M, N, K, stream, ai4k::idmap{}, 0, nullptr, splits);
            return;
        }
    }
    ai23_launch_gemm<BN, 2, 1, LAY, XSH, false, 1, 8, false, 2, false, true, true, false, true>(W, q, s, partial, M, M, N, K, stream, ai4k::idmap{}, 0, nullptr, splits);
}

template <int LAY, int XSH>
static void ai23_split_width(const uint8_t * W, const int8_t * q, const float * s, float * partial,
                              int M, int N, int K, int splits, int tile, cudaStream_t stream) {
    if (N <= 16) { ai23_split_tile<16, LAY, XSH>(W, q, s, partial, M, N, K, splits, tile, stream); }
    else if (N <= 32) { ai23_split_tile<32, LAY, XSH>(W, q, s, partial, M, N, K, splits, tile, stream); }
    else { ai23_split_tile<64, LAY, XSH>(W, q, s, partial, M, N, K, splits, tile, stream); }
}

// AIT2P_U dense: mmvr (N <= 8) or the default GEMM; the other rate-2 variants decode the trellis only.
template <int XSH>
static void ai2p_mm_kernels(const uint8_t * W, const int8_t * q, const float * s, float * Y, int M, int N, int K, cudaStream_t stream) {
    constexpr int L = ai4k::LAY_AIT2P_U;
    if (N <= 8) {
        const ai4k::idmap idm{};
        if (ai4_mmvr_kt()) {
            if (ai4_mmvr_go<L, XSH, false,  640>(W, q, s, Y, M, N, K, stream, idm, 1)) return;
            if (ai4_mmvr_go<L, XSH, false, 2560>(W, q, s, Y, M, N, K, stream, idm, 1)) return;
        }
        ai4_mmvr_go<L, XSH, false, 0>(W, q, s, Y, M, N, K, stream, idm, 1);
    } else if (N >= ai4_bn128_min()) {
        ai23_launch_gemm<128, 3, 4, L, XSH>(W, q, s, Y, M, M, N, K, stream);
    } else {
        ai23_launch_gemm<64, 3, 4, L, XSH>(W, q, s, Y, M, M, N, K, stream);
    }
}

// W4A8 fast path: rotate+quantize the activations, then the AI+ decode-to-IMMA kernels
static void ai4_mul_mat_fast(ggml_backend_cuda_context & ctx, ggml_type t, const uint8_t * W, const float * X, float * Y,
                            int M, int N, int K) {
    cudaStream_t stream = ctx.stream();
    const int xsh = K % 256 == 0 ? 1 : 0;
    ggml_cuda_pool_alloc<int8_t> q(ctx.pool(), (size_t) N*K);
    ggml_cuda_pool_alloc<float>  s(ctx.pool(), (size_t) N*(K >> (7 + xsh)));
    if (t == GGML_TYPE_AIT2P_U) {
        ai4_quantize_x(X, N, K, xsh, q.get(), s.get(), stream);
        if (xsh) { ai2p_mm_kernels<1>(W, q.get(), s.get(), Y, M, N, K, stream); }
        else     { ai2p_mm_kernels<0>(W, q.get(), s.get(), Y, M, N, K, stream); }
        return;
    }
    const ai23_dense_policy policy = ai23_dense_config(M, N, K, ai4_ampere_sms(ctx), ggml_ai4_rate(t));
    const int tile = policy.tile;
    const bool swap8 = (ggml_ai4_rate(t) == 2 || ggml_ai4_rate(t) == 3) && N > 8 && N <= 64 && (tile == 22 || tile == 23 || tile == 25 || tile == 27);
    const int splits = policy.splits;
    const int narrow = policy.narrow;
    const bool split_active = splits == 2 || splits == 4 || splits == 8 || splits == 16;
    const bool whole = split_active ? (tile != 16 && tile != 17 && !(tile >= 22 && tile <= 27)) :
                       (tile == 18 || tile == 19 || tile == 21 || (!tile && narrow >= 3 && (K == 5120 || K == 17408) && ai4_ampere_sms(ctx)));
    const bool xperm = ggml_ai4_rate(t) == 2 && N > 8 && N <= 64 && whole &&
                       ai4_env_int("GGML_CUDA_AI23_XPERM", nullptr, 0);
    if (swap8) { ai4_quantize_x<true>(X, N, K, xsh, q.get(), s.get(), stream); }
    else if (xperm) { ai4_quantize_x<false, true>(X, N, K, xsh, q.get(), s.get(), stream); }
    else { ai4_quantize_x(X, N, K, xsh, q.get(), s.get(), stream); }
    const int rate = ggml_ai4_rate(t);
    static const int small_splits = ai4_env_int("GGML_CUDA_AI23_SMALL_SPLIT", nullptr, 0);
    if ((rate == 2 || rate == 3) && N <= 8 && (K == 5120 || K == 17408) &&
            (small_splits == 2 || small_splits == 4)) {
        ggml_cuda_pool_alloc<float> partial(ctx.pool(), (size_t) M*N*small_splits);
        // All valid two/three-bit layouts use H256 at these fixed K values.
        switch (ai4_lay_of(t)) {
            case ai4k::LAY_AIT2_U:
                if (K == 5120) ai23_small_split<ai4k::LAY_AIT2_U, 1, 5120>(W, q.get(), s.get(), partial.get(), M, N, K, small_splits, stream);
                else ai23_small_split<ai4k::LAY_AIT2_U, 1, 17408>(W, q.get(), s.get(), partial.get(), M, N, K, small_splits, stream);
                break;
            case ai4k::LAY_AIT3_U:
                if (K == 5120) ai23_small_split<ai4k::LAY_AIT3_U, 1, 5120>(W, q.get(), s.get(), partial.get(), M, N, K, small_splits, stream);
                else ai23_small_split<ai4k::LAY_AIT3_U, 1, 17408>(W, q.get(), s.get(), partial.get(), M, N, K, small_splits, stream);
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
                if (xsh) { ai23_split_width<L, 1>(W, q.get(), s.get(), partial.get(), M, N, K, splits, tile, stream); } \
                else { ai23_split_width<L, 0>(W, q.get(), s.get(), partial.get(), M, N, K, splits, tile, stream); } \
                break;
            AI23_SPLIT_U(ai4k::LAY_AIT2_U)
            AI23_SPLIT_U(ai4k::LAY_AIT3_U)
#undef AI23_SPLIT_U
            default: GGML_ABORT("unsupported AI23 split layout");
        }
        ai4k::split_k_reduce<<<((int64_t) M*N + 255)/256, 256, 0, stream>>>(partial.get(), Y, (int64_t) M*N, splits);
        return;
    }
    if (N >= 512 && ai4_ampere_sms(ctx) && ai4_prefill_v2() && t == GGML_TYPE_AIT4_U) {
        GGML_ASSERT(M > 0 && K > 0 && (size_t) M <= SIZE_MAX / (size_t) K);
        ggml_cuda_pool_alloc<int8_t> decoded(ctx.pool(), (size_t) M*K);
        if (xsh) {
            ai4k::launch_prefill_v2<ai4k::LAY_AIT4_U, 1>(W, q.get(), s.get(), decoded.get(), Y, M, M, N, K, stream);
        } else {
            ai4k::launch_prefill_v2<ai4k::LAY_AIT4_U, 0>(W, q.get(), s.get(), decoded.get(), Y, M, M, N, K, stream);
        }
        return;
    }
    switch (ai4_lay_of(t)) {
#define AI4_MM_U(L) \
        case L: \
            if (xsh) { ai4_mm_kernels<L, 1>(W, q.get(), s.get(), Y, M, N, K, stream, ai4_ampere_sms(ctx)); } \
            else     { ai4_mm_kernels<L, 0>(W, q.get(), s.get(), Y, M, N, K, stream, ai4_ampere_sms(ctx)); } \
            break;
        AI4_MM_U(ai4k::LAY_AIT2_U)
        AI4_MM_U(ai4k::LAY_AIT3_U)
        AI4_MM_U(ai4k::LAY_AIT4_U)
        AI4_MM_U(ai4k::LAY_AIT5_U)
#undef AI4_MM_U
        default: GGML_ABORT("not an AI+ unit type");
    }
}

static bool ai4_is_type(ggml_type t) {
    return ggml_ai4_rate(t) != 0;
}

// Types that take the rate-k reference path (rotated-basis reconstruct x int8-emulated activations + SGEMM): AIT5_U only
// under GGML_CUDA_AI5_REF=1 (an A/B check of its rate-5 kernels); AIT2_U / AIT3_U unless GGML_CUDA_AI23_FAST=1.
static bool ai4_ref_only_type(ggml_type t) {
    static const bool v = ai4_env_int("GGML_CUDA_AI5_REF", nullptr, 0) != 0;
    if (t == GGML_TYPE_AIT5_U) {
        return v;
    }
    if (t == GGML_TYPE_AIT2_U || t == GGML_TYPE_AIT2P_U || t == GGML_TYPE_AIT3_U) {
        return !ai4_ai23_fast();
    }
    return false;
}

// GGML_CUDA_AI23_ID_REF=0 (opt-in): MUL_MAT_ID over AIT2_U / AIT3_U expert stacks runs the device-routed IMMA kernels (mmv-ID / GEMM-ID):
// the reference path routes on the host (stream sync, no CUDA graphs) and runs one f32 SGEMV per (slot, token) pair,
// which made a prompt batch over an AI+ expert bank several times slower than the IQ expert stack it replaces.
// Default (unset or 1) keeps the reference path for MUL_MAT_ID; dense MUL_MAT keeps ai4_ref_only_type.
static bool ai4_id_ref_only_type(ggml_type t) {
    static const bool id_ref = ai4_env_int("GGML_CUDA_AI23_ID_REF", nullptr, 1) != 0;
    if (t == GGML_TYPE_AIT2_U || t == GGML_TYPE_AIT2P_U || t == GGML_TYPE_AIT3_U) {
        return id_ref && ai4_ref_only_type(t);
    }
    return ai4_ref_only_type(t);
}

// Pairs whose expert id is outside [0, n_expert) are skipped by the routing (an expert bank marks the slots another bank
// serves with -1); their dst columns are written as zeros so the routing-weight product downstream stays finite.
// grid (n_tok), block 256: one block per token checks its n_used ids and zeroes the skipped columns.
static __global__ void __launch_bounds__(256) k_ai4_zero_skipped(const char * __restrict__ ids, int64_t nb0, int64_t nb1,
                                                                   int n_used, int n_expert, int M, float * __restrict__ Y) {
    const int t = blockIdx.x;
    for (int u = 0; u < n_used; ++u) {
        const int e = *(const int *) (ids + u*nb0 + t*nb1);
        if (e >= 0 && e < n_expert) {
            continue;
        }
        float * y = Y + ((int64_t) t*n_used + u)*M;
        for (int i = threadIdx.x; i < M; i += blockDim.x) {
            y[i] = 0.0f;
        }
    }
}

// Skipped ids only come from an expert bank; without one the zeroing launch is left out (no change to other models).
static bool ai4_id_skip_on() {
    static const bool forced = ai4_env_int("GGML_CUDA_AI_ID_SKIP_ZERO", nullptr, 0) != 0;
    return forced || ggml_moe_bank_any();
}

static void ai4_zero_skipped(const ggml_tensor * ids, int n_used, int n_tok, int n_expert, int M, float * Y, cudaStream_t stream) {
    if (!ai4_id_skip_on()) {
        return;
    }
    for (int t0 = 0; t0 < n_tok; t0 += 65535) {
        const int nt = std::min(65535, n_tok - t0);
        k_ai4_zero_skipped<<<nt, 256, 0, stream>>>((const char *) ids->data + (int64_t) t0*ids->nb[1], ids->nb[0], ids->nb[1],
                                                   n_used, n_expert, M, Y + (int64_t) t0*n_used*M);
    }
}

static void ai4_reconstruct_any(const void * src, float * W, int64_t K, int64_t M, cudaStream_t stream, ggml_type t) {
    GGML_ASSERT(aik_reconstruct_rate<float>(ggml_ai4_rate(t), src, W, K, M, true, stream, ggml_ai4_trellis_mul(t)));
}

// fork AIPFIX ONEPATH check: the f32 image [M][K] of an AI+ unit matrix in the original basis (reference only)
void ggml_cuda_moe_ai_reconstruct(const void * src, ggml_type type, float * W, int64_t K, int64_t M, cudaStream_t stream) {
    ai4_reconstruct_any(src, W, K, M, stream, type);
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
// M x T: ldc stays M and the C pointer moves by m0 elements. use_any: the final fallback's original-basis reconstruct
// (ai4_reconstruct_any); otherwise aik_reconstruct_rate in the basis emul selects.
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
                                                    ggml_ai4_trellis_mul(src0->type)));
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
    return ai4_ref() || ai4_id_ref_only_type(t) || !GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_AMPERE;
}

bool ggml_cuda_ai4_supported(const ggml_tensor * op) {
    const ggml_tensor * a = op->src[0];
    const ggml_tensor * b = op->src[1];
    if (b->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32 || a->ne[0] % 128 != 0 || a->ne[1] % 16 != 0 ||
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

    const int64_t K = src0->ne[0];
    const int64_t M = src0->ne[1];
    GGML_ASSERT(K % 128 == 0 && M % 16 == 0);
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
static int ai23_id_bn() {
    static const int bn = [] {
        const int v = ai4_env_int("GGML_CUDA_AI23_ID_BN", nullptr, 0);
        if (v == 0 || v == 32 || v == 64 || v == 128) {
            return v;
        }
        GGML_LOG_WARN("invalid GGML_CUDA_AI23_ID_BN; using automatic tile width (expected 32, 64 or 128)\n");
        return 0;
    }();
    return bn;
}
static bool ai4_mmvid_route(int n_tok, int n_used, int n_expert, int rate) {
    if (n_tok <= 8) {
        return true;
    }
    const int p = getenv("GGML_CUDA_AI4_MMVID_PAIRS") ? ai4_mmvid_pairs() :
        (((ai4_ai23_tuned() && (rate == 2 || rate == 3)) || (ai4_aikern() && rate >= 2 && rate <= 4)) ? 4 : 0);
    if (p <= 0 || n_expert <= 0) {
        return false;
    }
    const double act = n_expert*(1.0 - std::pow(1.0 - (double) n_used/n_expert, (double) n_tok));
    return (double) n_used*n_tok <= p*act;
}

// fork: reuse dense staging and fragment schedules without changing routed rows or K order.
template <int BN, int STAGES, int LAY, int XSH, int SLABS = ai4k::GEMM_SLABS>
static void ai23_launch_id_shape(const uint8_t * W, const int8_t * q, const float * s, float * Y, int M, int K,
                                  cudaStream_t stream, const ai4k::idmap & idm, int max_tiles) {
    static const bool fixed_load = ai4_env_int("GGML_CUDA_AI23_ID_FIXED_LOAD", nullptr, 0) != 0;
    static const bool j_major = ai4_env_int("GGML_CUDA_AI23_ID_J_MAJOR", nullptr, 0) != 0;
    if (j_major) {
        if (fixed_load) {
            ai23_launch_gemm<BN, STAGES, 4, LAY, XSH, true, 1, SLABS,
                            false, 1, false, true, false, false, false, false, true>(
                    W, q, s, Y, M, M, 0, K, stream, idm, max_tiles);
        } else {
            ai23_launch_gemm<BN, STAGES, 4, LAY, XSH, true, 1, SLABS,
                            false, 1, false, true>(
                    W, q, s, Y, M, M, 0, K, stream, idm, max_tiles);
        }
        return;
    }
    if (fixed_load) {
        ai23_launch_gemm<BN, STAGES, 4, LAY, XSH, true, 1, SLABS,
                        false, 1, false, false, false, false, false, false, true>(
                W, q, s, Y, M, M, 0, K, stream, idm, max_tiles);
    } else {
        ai23_launch_gemm<BN, STAGES, 4, LAY, XSH, true, 1, SLABS>(
                W, q, s, Y, M, M, 0, K, stream, idm, max_tiles);
    }
}

template <int BN, int LAY, int XSH>
static void ai23_launch_id_gemm(const uint8_t * W, const int8_t * q, const float * s, float * Y, int M, int K,
                                 cudaStream_t stream, const ai4k::idmap & idm, int max_tiles) {
    if constexpr (ai4k::lay_rate<LAY>() == 2 || ai4k::lay_rate<LAY>() == 3) {
        static const int stages = [] {
            const int v = ai4_env_int("GGML_CUDA_AI23_ID_STAGES", nullptr, 3);
            if (v == 2 || v == 3) {
                return v;
            }
            GGML_LOG_WARN("invalid GGML_CUDA_AI23_ID_STAGES; using 3 (expected 2 or 3)\n");
            return 3;
        }();
        static const int slabs = [] {
            const int v = ai4_env_int("GGML_CUDA_AI23_ID_SLABS", nullptr, 8);
            if (v == 4 || v == 8) {
                return v;
            }
            GGML_LOG_WARN("invalid GGML_CUDA_AI23_ID_SLABS; using 8 (expected 4 or 8)\n");
            return 8;
        }();
        if (slabs == 4) {
            if (stages == 2) {
                ai23_launch_id_shape<BN, 2, LAY, XSH, 4>(W, q, s, Y, M, K, stream, idm, max_tiles);
            } else {
                ai23_launch_id_shape<BN, 3, LAY, XSH, 4>(W, q, s, Y, M, K, stream, idm, max_tiles);
            }
            return;
        }
        if (stages == 2) {
            ai23_launch_id_shape<BN, 2, LAY, XSH>(W, q, s, Y, M, K, stream, idm, max_tiles);
            return;
        }
    }
    if constexpr (ai4k::lay_rate<LAY>() == 2 || ai4k::lay_rate<LAY>() == 3) {
        ai23_launch_id_shape<BN, 3, LAY, XSH>(W, q, s, Y, M, K, stream, idm, max_tiles);
    } else {
        ai23_launch_gemm<BN, 3, 4, LAY, XSH, true>(W, q, s, Y, M, M, 0, K, stream, idm, max_tiles);
    }
}

template <int LAY, int XSH>
static void ai4_mm_id_kernels(const uint8_t * W, const int8_t * q, const float * s, float * Y, int M, int K, int n_tok,
                              int n_expert, int64_t n_pairs, const ai4k::idmap & idm, int bn, int max_tiles, cudaStream_t stream, int sms) {
    if constexpr (LAY == ai4k::LAY_AIT4_U) {
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
    } else if (bn == 32) {
        if constexpr (ai4k::lay_rate<LAY>() == 2 || ai4k::lay_rate<LAY>() == 3) {
            ai23_launch_id_gemm<32, LAY, XSH>(W, q, s, Y, M, K, stream, idm, max_tiles);
        } else {
            GGML_ABORT("32-column expert tiles require rate 2 or 3");
        }
    } else if (bn == 128) {
        ai23_launch_id_gemm<128, LAY, XSH>(W, q, s, Y, M, K, stream, idm, max_tiles);
    } else {
        ai23_launch_id_gemm<64, LAY, XSH>(W, q, s, Y, M, K, stream, idm, max_tiles);
    }
}

// MUL_MAT_ID: src0 [K, M, n_expert] AI+ unit expert stack, src1 [K, ne11 (1 or n_used), n_tok] f32, ids [n_used, n_tok],
// dst [M, n_used, n_tok]. Fast path: all src1 rows quantized once, one routing CTA (ai4k::ids_helper) builds the per-expert
// token lists on the device (no host sync, graph-capturable), then mmv-ID (n_tok <= 8) or GEMM-ID over (active expert, token
// tile) blocks. Only the selected experts' bytes are read, so a stack whose unused experts are not resident (the scheduler
// copies only the used experts of a host-resident stack) works.
void ggml_cuda_mul_mat_id_ai4(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];
    GGML_ASSERT(ggml_cuda_ai4_supported(dst));

    const int  K        = (int) src0->ne[0];
    const int  M        = (int) src0->ne[1];
    const int  n_expert = (int) src0->ne[2];
    const int  ne11     = (int) src1->ne[1];
    const int  n_used   = (int) ids->ne[0];
    const int  n_tok    = (int) ids->ne[1];
    const int64_t n_pairs = (int64_t) n_used*n_tok;
    const int64_t n_rows  = (int64_t) ne11*n_tok;
    cudaStream_t stream = ctx.stream();

    // skipped pairs (id outside [0, n_expert)) get zero columns; the kernels below never write them. Small batches on the
    // routed fast path zero them inside the routing helper (no extra launch per decode step).
    const bool ref_path = !ai4_fast_ok(ctx) || ai4_id_ref_only_type(src0->type);

    if (ref_path) {
        ai4_zero_skipped(ids, n_used, n_tok, n_expert, M, (float *) dst->data, stream);
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
    // direct: one block per (slot, token) pair reads its expert id itself, no routing helper; measured to width 6
    static const int direct_max = std::min(8, ai4_env_int("GGML_CUDA_AI23_DIRECT", nullptr, ai4_ai23_tuned() ? 4 : 0));
    static const bool fused_x = ai4_env_int("GGML_CUDA_AI23_FUSED_X", nullptr, 0) != 0;
    const int rate = ggml_ai4_rate(src0->type);
    const int sms = ai4_ampere_sms(ctx);
    // The Ampere kernel consumes grouped routing, including when sparse IDs qualify for direct routing.
    const bool ampere_grouped = sms && src0->type == GGML_TYPE_AIT4_U && ai4_ampere_mmv(n_tok);
    static const int aikern_direct_max = ai4_env_int("GGML_CUDA_AIKERN_DIRECT", nullptr, 64);
    const bool aikern_direct = !ampere_grouped && ai4_aikern() && rate >= 2 && rate <= 4 && (rate == 4 || ai4_mmvr_on_id());
    const bool direct = aikern_direct ?
        (n_tok <= aikern_direct_max && ai4_mmvid_route(n_tok, n_used, n_expert, rate)) :
        (ai4_mmvr_on_id() && (rate == 2 || rate == 3) && n_tok <= direct_max);
    // fork: one output column per direct block; zero skipped slots in that same kernel.
    const bool verify_compact = ai4_env_int("GGML_CUDA_AI23_VERIFY_COMPACT", nullptr, 1) != 0;
    const bool direct_zero = direct && verify_compact;
    const bool fused = direct && fused_x && n_tok == 1;
    const int xsh = K % 256 == 0 ? 1 : 0;
    ggml_cuda_pool_alloc<int8_t> q(ctx.pool(), (size_t) n_rows*K);
    ggml_cuda_pool_alloc<float>  s(ctx.pool(), (size_t) n_rows*(K >> (7 + xsh)));
    if (!fused) ai4_quantize_x((const float *) src1->data, n_rows, K, xsh, q.get(), s.get(), stream);

    // mmv-ID (n_tok <= 8) takes tiles of MMV_ID_BN pairs, so an expert repeated within a token still fits
    int bn = ai4_mmvid_route(n_tok, n_used, n_expert, ggml_ai4_rate(src0->type)) ? ai4k::MMV_ID_BN : (n_tok >= ai4_bn128_min() ? 128 : 64);
    // fork: keep decode and sparse MMV routing unchanged; size the helper and GEMM from the same tile width.
    if ((rate == 2 || rate == 3) && bn != ai4k::MMV_ID_BN && ai23_id_bn() != 0) {
        bn = ai23_id_bn();
    }
    if (ampere_grouped) {
        bn = ai4_ampere_bn(n_tok);
    }
    static const bool tight_grid = ai4_env_int("GGML_CUDA_AI23_ID_TIGHT_GRID", nullptr, 0) != 0;
    const bool tight = tight_grid && (rate == 2 || rate == 3) && bn != ai4k::MMV_ID_BN;
    const int max_tiles = direct ? (int) n_pairs : ai4k::ids_max_tiles(n_expert, n_pairs, bn, tight);
    static const bool prompt_zero = ai4_env_int("GGML_CUDA_AI23_ID_ZERO", nullptr, 0) != 0;
    const bool zero_in_helper = !direct && (n_tok <= 8 ||
        (prompt_zero && (rate == 2 || rate == 3) && bn != ai4k::MMV_ID_BN));
    if (!zero_in_helper && !direct_zero) {
        ai4_zero_skipped(ids, n_used, n_tok, n_expert, M, (float *) dst->data, stream);
    }
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
        idm.direct_zero = direct_zero;
    } else {
        routing.alloc((size_t) ai4k::ids_helper_ints(n_expert, n_pairs, max_tiles));
        static const bool route_agg = ai4_env_int("GGML_CUDA_AI23_ID_ROUTE_AGG", nullptr, 0) != 0;
        const bool aggregate = route_agg && (rate == 2 || rate == 3) && bn != ai4k::MMV_ID_BN;
        static const int route_threads = ai4_env_int("GGML_CUDA_AI23_ID_THREADS", nullptr, 1024);
        const bool compact = route_threads == 256 && (rate == 2 || rate == 3) && bn != ai4k::MMV_ID_BN;
        idm = ai4k::launch_ids_helper(ids->data, ids->nb[0], ids->nb[1], n_used, n_tok, ne11, n_expert, bn,
                                    max_tiles, src0->nb[2], routing.get(), stream,
                                    zero_in_helper && ai4_id_skip_on() ? (float *) dst->data : nullptr, M, aggregate, compact);
    }
    const uint8_t * W = (const uint8_t *) src0->data;
    float * Y = (float *) dst->data;
    switch (ai4_lay_of(src0->type)) {
#define AI4_MMID_U(L) \
        case L: \
            if (xsh) { ai4_mm_id_kernels<L, 1>(W, q.get(), s.get(), Y, M, K, n_tok, n_expert, n_pairs, idm, bn, max_tiles, stream, sms); } \
            else     { ai4_mm_id_kernels<L, 0>(W, q.get(), s.get(), Y, M, K, n_tok, n_expert, n_pairs, idm, bn, max_tiles, stream, sms); } \
            break;
        AI4_MMID_U(ai4k::LAY_AIT2_U)
        AI4_MMID_U(ai4k::LAY_AIT2P_U)
        AI4_MMID_U(ai4k::LAY_AIT3_U)
        AI4_MMID_U(ai4k::LAY_AIT4_U)
        AI4_MMID_U(ai4k::LAY_AIT5_U)
#undef AI4_MMID_U
        default: GGML_ABORT("not an AI+ unit type");
    }
    CUDA_CHECK(cudaGetLastError());
}

// fork: each route owns one slab and chooses its weight format before reading either stack.
#define MOE_MIXED_TYPES(X) \
    X(GGML_TYPE_Q2_0,    vec_dot_q2_0_q8_1,    VDR_Q2_0_Q8_1_MMVQ) \
    X(GGML_TYPE_Q4_0,    vec_dot_q4_0_q8_1,    VDR_Q4_0_Q8_1_MMVQ) \
    X(GGML_TYPE_Q2_K,    vec_dot_q2_K_q8_1,    VDR_Q2_K_Q8_1_MMVQ) \
    X(GGML_TYPE_Q3_K,    vec_dot_q3_K_q8_1,    VDR_Q3_K_Q8_1_MMVQ) \
    X(GGML_TYPE_Q4_K,    vec_dot_q4_K_q8_1,    VDR_Q4_K_Q8_1_MMVQ) \
    X(GGML_TYPE_Q5_K,    vec_dot_q5_K_q8_1,    VDR_Q5_K_Q8_1_MMVQ) \
    X(GGML_TYPE_Q6_K,    vec_dot_q6_K_q8_1,    VDR_Q6_K_Q8_1_MMVQ) \
    X(GGML_TYPE_IQ2_XXS, vec_dot_iq2_xxs_q8_1, VDR_IQ2_XXS_Q8_1_MMVQ) \
    X(GGML_TYPE_IQ2_XS,  vec_dot_iq2_xs_q8_1,  VDR_IQ2_XS_Q8_1_MMVQ) \
    X(GGML_TYPE_IQ2_S,   vec_dot_iq2_s_q8_1,   VDR_IQ2_S_Q8_1_MMVQ) \
    X(GGML_TYPE_IQ3_XXS, vec_dot_iq3_xxs_q8_1, VDR_IQ3_XXS_Q8_1_MMVQ) \
    X(GGML_TYPE_IQ3_S,   vec_dot_iq3_s_q8_1,   VDR_IQ3_S_Q8_1_MMVQ) \
    X(GGML_TYPE_IQ4_NL,  vec_dot_iq4_nl_q8_1,  VDR_IQ4_NL_Q8_1_MMVQ) \
    X(GGML_TYPE_IQ4_XS,  vec_dot_iq4_xs_q8_1,  VDR_IQ4_XS_Q8_1_MMVQ)

template <ggml_type T> struct moe_mixed_dot;
#define MOE_MIXED_DOT(T, F, V) \
    template <> struct moe_mixed_dot<T> { \
        static constexpr int vdr = V; \
        static __device__ __forceinline__ float dot(const void * w, const block_q8_1 * x, int k, int i) { return F(w, x, k, i); } \
    };
MOE_MIXED_TYPES(MOE_MIXED_DOT)
#undef MOE_MIXED_DOT

// NC routes of one expert (cold rows grouped by expert) share each weight read; every column keeps the
// one-route k order and reduction, so each column's sums are those of the NC = 1 call.
template <ggml_type T, int WARPS, int NC = 1>
static __device__ __forceinline__ void moe_mixed_base_rows(const char * w, const block_q8_1 * const * xs, float * const * ys, int K) {
    constexpr int qk = ggml_cuda_type_traits<T>::qk;
    constexpr int qi = ggml_cuda_type_traits<T>::qi;
    constexpr int vdr = moe_mixed_dot<T>::vdr;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    constexpr int rows = 16/WARPS;
    float sum[NC][rows] = {};
    for (int k = lane / (qi/vdr); k < K/qk; k += 32*vdr/qi) {
#pragma unroll
        for (int r = 0; r < rows; ++r) {
            const int row = warp + r*WARPS;
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                sum[c][r] += moe_mixed_dot<T>::dot(w, xs[c] + k*(qk/QK8_1), row*(K/qk) + k, vdr*(lane % (qi/vdr)));
            }
        }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
#pragma unroll
        for (int r = 0; r < rows; ++r) {
            sum[c][r] = warp_reduce_sum<32>(sum[c][r]);
            if (lane == 0) {
                ys[c][warp + r*WARPS] = sum[c][r];
            }
        }
    }
}

// Each register set keeps one original MMVQ lane; the butterfly order is unchanged (per column for NC > 1).
template <ggml_type T, int WARPS, int NC = 1>
static __device__ __forceinline__ void moe_mixed_down_rows(const char * w, const block_q8_1 * const * xs, float * const * ys, int K) {
    constexpr int qk = ggml_cuda_type_traits<T>::qk;
    constexpr int qi = ggml_cuda_type_traits<T>::qi;
    constexpr int vdr = moe_mixed_dot<T>::vdr;
    constexpr int rows = 4/WARPS;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int p = lane & 7;
    const int row0 = (warp*4 + lane/8)*rows;
    float sum[NC][4][rows] = {};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int v = p + 8*j;
        for (int k = v/(qi/vdr); k < K/qk; k += 32*vdr/qi) {
#pragma unroll
            for (int r = 0; r < rows; ++r) {
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    sum[c][j][r] += moe_mixed_dot<T>::dot(w, xs[c] + k*(qk/QK8_1), (row0 + r)*(K/qk) + k, vdr*(v%(qi/vdr)));
                }
            }
        }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
#pragma unroll
        for (int d = 2; d > 0; d >>= 1) {
            float next[4][rows];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
#pragma unroll
                for (int r = 0; r < rows; ++r) {
                    next[j][r] = sum[c][j][r] + sum[c][j ^ d][r];
                }
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
#pragma unroll
                for (int r = 0; r < rows; ++r) { sum[c][j][r] = next[j][r]; }
            }
        }
#pragma unroll
        for (int d = 4; d > 0; d >>= 1) {
#pragma unroll
            for (int r = 0; r < rows; ++r) {
                sum[c][0][r] += __shfl_xor_sync(0xffffffffu, sum[c][0][r], d);
            }
        }
#pragma unroll
        for (int r = 0; r < rows; ++r) {
            if (p == 0) { ys[c][row0 + r] = sum[c][0][r]; }
        }
    }
}

// One-token native MMVQ splits K across four warps (small_k only changes rows per block):
// merge the four K partitions before the lane butterfly. Used for IQ4_NL banks and cold rows.
template <ggml_type T>
static __device__ __forceinline__ void moe_mixed_rows_single(
        const char * w, const block_q8_1 * x, float * y, int K) {
    constexpr int qk = ggml_cuda_type_traits<T>::qk;
    constexpr int qi = ggml_cuda_type_traits<T>::qi;
    constexpr int vdr = moe_mixed_dot<T>::vdr;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    float sum[16] = {};
    for (int k = tid/(qi/vdr); k < K/qk; k += 128*vdr/qi) {
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            sum[r] += moe_mixed_dot<T>::dot(w, x + k*(qk/QK8_1), r*(K/qk) + k, vdr*(tid%(qi/vdr)));
        }
    }
    extern __shared__ __align__(16) uint8_t smem[];
    float (*partial)[16][32] = (float (*)[16][32]) smem;
    if (warp > 0) {
#pragma unroll
        for (int r = 0; r < 16; ++r) { partial[warp - 1][r][lane] = sum[r]; }
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int r = 0; r < 16; ++r) {
#pragma unroll
            for (int p = 0; p < 3; ++p) { sum[r] += partial[p][r][lane]; }
            sum[r] = warp_reduce_sum<32>(sum[r]);
            if (lane == 0) { y[r] = sum[r]; }
        }
    }
    __syncthreads(); // another slab may reuse partial in this block
}

// Each eighth occurrence owns the following <=8 matching routes. Scan immutable inputs,
// including the native-ID map on the first projection, never another CTA's route_ids stores.
static __device__ __forceinline__ void moe_mixed_group(const ai4k::idmap & idm, int h, ai4k::mmvr_group & group) {
    if (gridDim.z == 1) {
        // Decode top-k IDs are normally unique; duplicates remain independent, as before.
        if (threadIdx.x == 0) {
            group.n = 1;
            group.rows[0] = blockIdx.y % idm.direct_ne11;
            group.cols[0] = blockIdx.y;
        }
        __syncthreads();
        return;
    }
    if (threadIdx.x < 32) {
        const int lane = threadIdx.x;
        const int own = (int) blockIdx.z*idm.direct_n_used + blockIdx.y;
        const int routes = (int) gridDim.z*idm.direct_n_used;
        const unsigned lower_lanes = (1u << lane) - 1u;
        int before = 0, count = 0;
        for (int first = 0; first < routes; first += 32) {
            const int route = first + lane;
            int slot = -1;
            if (route < routes) {
                const int u = route % idm.direct_n_used, t = route / idm.direct_n_used;
                const int native = *(const int *) (idm.direct_ids + u*idm.direct_nb0 + t*idm.direct_nb1);
                slot = idm.direct_map ?
                        (native >= 0 && native < idm.direct_map_size ? (int) idm.direct_map[native] : -1) : native;
            }
            const unsigned matches = __ballot_sync(0xffffffffu, slot == h);
            const unsigned prior = __ballot_sync(0xffffffffu, slot == h && route < own);
            before += __popc(prior);
            if (first + 32 <= own) {
                continue;
            }
            if (before % ai4k::MMV_ID_BN != 0) {
                break;
            }
            const unsigned following = matches & ~prior;
            const int index = count + __popc(following & lower_lanes);
            if ((following & (1u << lane)) && index < ai4k::MMV_ID_BN) {
                const int u = route % idm.direct_n_used, t = route / idm.direct_n_used;
                group.rows[index] = t*idm.direct_ne11 + u % idm.direct_ne11;
                group.cols[index] = route;
            }
            count += __popc(following);
            if (count >= ai4k::MMV_ID_BN) {
                break;
            }
        }
        if (lane == 0) {
            group.n = min(count, ai4k::MMV_ID_BN);
        }
    }
    __syncthreads();
}

// fork AIPFIX D7: the slab-per-warp direct GEMV over cached AI+ experts (slot table), with the routes that picked the
// same expert in one block (up to MMV_ID_BN columns per weight read, as the bank's grouped hot path). Every column
// keeps the direct SPW K order, so the outputs are the ungrouped launch's bits. Filters, the paired second projection
// (upper half of x) and the fused SwiGLU work as in mmvr. GGML_MOE_CACHE_AI_GROUP=0 opts out.
template <int LAY, int XSH, int KT>
static __global__ void __launch_bounds__(128, 6) ai4_table_grouped(const int8_t * __restrict__ Xq,
        const float * __restrict__ Xs, float * Y, int M, int K, const ai4k::idmap idm) {
    constexpr int WARPS = 4;
    const bool second = idm.table2 && (int) blockIdx.x >= (int) gridDim.x/2;
    const int bx = (int) blockIdx.x - (second ? (int) gridDim.x/2 : 0);
    ai4k::idmap m = idm;
    float * y = Y;
    if (idm.table2) {
        m.table = second ? idm.table2 : idm.table;
        m.filter = second ? idm.filter2 : idm.filter;
        m.glu_pfilter = second ? idm.filter : idm.filter2;
        m.glu_prev = second ? idm.glu_prev : idm.glu_prev2;
        y = second ? idm.y2 : Y;
    }
    const int lane = threadIdx.x & 31;
    const int slab = bx*WARPS + (int) (threadIdx.x >> 5);
    const int u = blockIdx.y, t = blockIdx.z;
    const int64_t col = (int64_t) t*idm.direct_n_used + u;
    // every route clears its own tile in the other bank, filtered or not (mmvr does the same before its filter)
    if (m.glu && lane == 0 && slab < M/16) {
        ai4k::g_ai4k_glu_count[(m.filter_val ? ai4k::GLU_PAIR_TILES : 0) + col*(M/16) + slab] = 0;
    }
    const int e = *(const int *) (idm.direct_ids + u*idm.direct_nb0 + t*idm.direct_nb1);
    if (e < 0 || e >= idm.direct_n_expert) {
        if (idm.direct_zero && lane < 16 && slab < M/16) {
            y[col*M + slab*16 + lane] = 0.0f;
        }
        return;
    }
    if (m.filter && m.filter[e] != m.filter_val) {
        return;
    }
    __shared__ ai4k::mmvr_group group;
    moe_mixed_group(m, e, group);
    if (group.n == 0) {
        return;
    }
    ai4k::mmvr_body<LAY, XSH, true, KT, 2, WARPS, 6, true, false, 1, 1, false, true, true>(nullptr, Xq, Xs, y, M, M, 1,
            K, m, &group, bx);
}

static bool ai4_table_grouped_on() {
    static const bool on = ai4_env_int("GGML_MOE_CACHE_AI_GROUP", nullptr, 1) != 0;
    return on;
}

template <int LAY, int XSH, int KT>
static bool ai4_table_grouped_try(const int8_t * q, const float * s, float * Y, int M, int K, cudaStream_t stream,
                                  const ai4k::idmap & idm, int max_tiles) {
    // one token has nothing to share; the native-id map (bank) and fused activations keep their own launches
    if (!ai4_table_grouped_on() || !idm.table || idm.direct_map || idm.direct_x || idm.direct_n_used <= 0 ||
            max_tiles/idm.direct_n_used < 2 || M % 16 != 0) {
        return false;
    }
    constexpr int WARPS = 4;
    constexpr int smem = WARPS*2*ai4k::mmv_stage_bytes<true, LAY>();
    static bool attr_set[64] = {};
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    if (!attr_set[dev & 63]) {
        CUDA_CHECK(cudaFuncSetAttribute(ai4_table_grouped<LAY, XSH, KT>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        attr_set[dev & 63] = true;
    }
    const dim3 grid(((M/16 + WARPS - 1)/WARPS)*(idm.table2 ? 2 : 1), idm.direct_n_used, max_tiles/idm.direct_n_used);
    ai4_table_grouped<LAY, XSH, KT><<<grid, WARPS*32, smem, stream>>>(q, s, Y, M, K, idm);
    static bool logged = false;
    if (!logged) {
        logged = true;
        fprintf(stderr, "[ai4] grouped cached AI+ GEMV first launch: M=%d K=%d tokens=%d paired=%d\n", M, K,
                max_tiles/idm.direct_n_used, idm.table2 ? 1 : 0);
    }
    return true;
}

// Paired launch: blocks at x >= gridDim.x/2 compute this second projection (W == nullptr: unpaired).
struct moe_mixed_second {
    const uint8_t * W;
    float * Y;
    const char * base;
    const void * const * table;
    const uint8_t * skip;
    const uint8_t * resident;
    const float * glu_gate; // fused SwiGLU (glu == nullptr: none)
    const float * glu_up;
    float * glu;
    const float * red_w;    // fused routed sum of an unpaired launch (red_dst == nullptr: none)
    const int32_t * red_bank;
    int64_t red_bank_stride;
    float * red_dst;
    bool red_tree;
    bool pair_warp;         // hot SPW routes: warp pairs compute gate and up of one slab (see moe_mixed_mmv)
    bool cold_group;        // cold routes of one expert share a weight read (see moe_mixed_cold)
    bool res_both;          // paired: a cached route is resident only when both projections are (see P2LIST)
    bool hot_only;          // phase 1 skips cached routes too (P1HITG: the 3XS fused hit kernel computed them)
};

// Epilogue tiles are 16-row slices of one route's output, so launches that tile the rows differently (the SPW
// hit-first launch and the 16-row phase-2 launch) count into the same tiles. A block covers rows rows from row
// bx*rows (at most four slices).
static __device__ __forceinline__ int moe_mixed_slices(int bx, int rows, int M, int & first) {
    first = bx*rows/16;
    return min(rows/16, M/16 - first);
}

// Fused SwiGLU: each projection's store of a slice counts in once; the second one (in either phase launch)
// writes silu(gate)*up for that slice and resets the count for the next launch.
static constexpr int MOE_MIXED_GLU_TILES = 40*64*8;
static __device__ unsigned int g_moe_mixed_glu_count[MOE_MIXED_GLU_TILES];

static __device__ __forceinline__ void moe_mixed_glu(const moe_mixed_second & second, int bx, int rows, int M,
        const int * cols, int n, int own) {
    if (!second.glu) {
        return;
    }
    __threadfence();
    __syncthreads();
    int first;
    const int slices = moe_mixed_slices(bx, rows, M, first);
    for (int i = threadIdx.x; i < n*slices; i += blockDim.x) {
        const int col = cols ? cols[i / slices] : own;
        const int slice = first + i % slices;
        unsigned int * count = &g_moe_mixed_glu_count[col*(M/16) + slice];
        if (atomicAdd(count, 1u) != 1u) {
            continue;
        }
        *count = 0;
        const int64_t o = (int64_t) col*M + slice*16;
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            second.glu[o + r] = ggml_cuda_op_silu_single(__ldcg(second.glu_gate + o + r)) * __ldcg(second.glu_up + o + r);
        }
    }
}

// Fused routed sum (unpaired launches): each stored route slice counts in once per (token, slice); the block that
// completes a token's n_used routes (in either phase launch) writes that token's slice of the sum and resets the
// count. Same expression and order as moe_mixed_reduction_f32: products rounded alone, then serial or warp-tree
// hot and cold buckets, then hot + cold.
static constexpr int MOE_MIXED_RED_TILES = 4096*8;
static __device__ unsigned int g_moe_mixed_red_count[MOE_MIXED_RED_TILES];

static __device__ __forceinline__ void moe_mixed_reduce(const moe_mixed_second & second, const float * Y, int bx,
        int rows, int M, const int * cols, int n, int own, int n_used) {
    if (!second.red_dst) {
        return;
    }
    __shared__ unsigned int done;   // bit 4*token + slice - first
    __threadfence();
    if (threadIdx.x == 0) {
        done = 0;
    }
    __syncthreads();
    int first;
    const int slices = moe_mixed_slices(bx, rows, M, first);
    for (int i = threadIdx.x; i < n*slices; i += blockDim.x) {
        const int tok = (cols ? cols[i / slices] : own) / n_used;
        const int j = i % slices;
        unsigned int * count = &g_moe_mixed_red_count[tok*(M/16) + first + j];
        if (atomicAdd(count, 1u) == (unsigned int) n_used - 1) {
            *count = 0;
            atomicOr(&done, 1u << (4*tok + j));
        }
    }
    __syncthreads();
    if (done == 0) {
        return;
    }
    const bool tree = second.red_tree;
    const int lane = threadIdx.x & 31;
    const int step = tree ? blockDim.x >> 5 : blockDim.x;
    for (int i = tree ? threadIdx.x >> 5 : threadIdx.x; i < 32*16; i += step) {
        const int bit = i / 16;
        if (!(done >> bit & 1u)) {
            continue;
        }
        const int tok = bit / 4;
        const int r = (first + bit % 4)*16 + i % 16;
        const float * w = second.red_w + (int64_t) tok*n_used;
        const int32_t * bank = second.red_bank + tok*second.red_bank_stride;
        const float * y = Y + (int64_t) tok*n_used*M + r;
        float hot = 0.0f, cold = 0.0f;
        if (tree) {
            if (lane < n_used) {
                const float value = __fmul_rn(__ldcg(y + (int64_t) lane*M), w[lane]);
                if (bank[lane] >= 0) { hot = value; }
                else { cold = value; }
            }
            hot = warp_reduce_sum(hot);
            cold = warp_reduce_sum(cold);
        } else {
            for (int e = 0; e < n_used; ++e) {
                const float value = __fmul_rn(__ldcg(y + (int64_t) e*M), w[e]);
                if (bank[e] >= 0) { hot = __fadd_rn(hot, value); }
                else { cold = __fadd_rn(cold, value); }
            }
        }
        if (!tree || lane == 0) {
            second.red_dst[(int64_t) tok*M + r] = __fadd_rn(hot, cold);
        }
    }
}

// Epilogues of one block's stores: rows rows from tile bx of the routes cols[0..n) (cols == nullptr: route own).
static __device__ __forceinline__ void moe_mixed_done(const moe_mixed_second & second, const float * Y, int bx, int rows,
        int M, const int * cols, int n, int own, int n_used) {
    moe_mixed_glu(second, bx, rows, M, cols, n, own);
    moe_mixed_reduce(second, Y, bx, rows, M, cols, n, own, n_used);
}

// Bank slot of a route: the native expert through the optional native->bank map (-1: cold).
static __device__ __forceinline__ int moe_mixed_slot(const ai4k::idmap & idm, int u, int t) {
    const int native = *(const int32_t *) (idm.direct_ids + u*idm.direct_nb0 + t*idm.direct_nb1);
    return idm.direct_map ?
            (native >= 0 && native < idm.direct_map_size ? (int) idm.direct_map[native] : -1) : native;
}

// Cold rows of NC grouped routes: one weight read per k step serves every column (per-column sums unchanged).
template <ggml_type BASE, int WARPS, int NC>
static __device__ __forceinline__ void moe_mixed_cold_rows(const char * w, const block_q8_1 * const * xs,
        float * const * ys, int K) {
    if constexpr (BASE == GGML_TYPE_Q2_0 || BASE == GGML_TYPE_IQ4_NL) {
        moe_mixed_down_rows<BASE, WARPS, NC>(w, xs, ys, K);
    } else {
        moe_mixed_base_rows<BASE, WARPS, NC>(w, xs, ys, K);
    }
}

// The cold (base-weight) rows of route (u, t), plus one-token IQ4_NL bank rows; zero-fills skipped routes.
// With group (multi-token launches, specialized base type), the first cold route of an expert also computes the
// later routes of that expert (at most MMV_ID_BN), so their blocks exit and the expert's weights are read once.
template <ggml_type BASE, bool IQ4_BANK, int WARPS, bool SPW>
static __device__ __forceinline__ void moe_mixed_cold(
        const uint8_t * W, float * Y, int M, int K, const ai4k::idmap & idm, const char * base,
        const void * const * table, ggml_type type, size_t expert_stride, size_t row_stride, int n_expert,
        const block_q8_1 * x, const uint8_t * skip, const uint8_t * resident, const moe_mixed_second & second,
        int u, int t, int n_tok, int bx, int h, bool hot, int e, const int32_t * ids = nullptr, int64_t ids_stride = 0) {
    const int own = t*idm.direct_n_used + u;
    constexpr int slabs = SPW ? WARPS : 1;
    const int row0 = bx*16*slabs;
    float * out = Y + ((int64_t) t*idm.direct_n_used + u)*M + row0;
    if (!hot && (h >= 0 || e < 0 || e >= n_expert || (skip && skip[e] && (!resident || !resident[e])))) {
        if (threadIdx.x < 16*slabs && row0 + threadIdx.x < M) {
            out[threadIdx.x] = 0.0f;
        }
        moe_mixed_done(second, Y, bx, 16*slabs, M, nullptr, 1, own, idm.direct_n_used);
        return;
    }
    const char * w = hot ? (const char *) W + (size_t) h*idm.nb02 :
            table ? (const char *) table[e] : base + (size_t) e*expert_stride;
    w += (size_t) row0*row_stride;
    const int pad = GGML_PAD(K, MATRIX_ROW_PADDING)/QK8_1;
    if constexpr (BASE != GGML_TYPE_COUNT && !IQ4_BANK) {
        if (ids && second.cold_group && n_tok > 1 && !hot) {
            __shared__ int gcols[ai4k::MMV_ID_BN];
            __shared__ int gn;
            if (threadIdx.x < 32) {
                // Same expert = same classification (bank map, skip and resident flags are per expert).
                const int lane = threadIdx.x, routes = n_tok*idm.direct_n_used;
                int before = 0, n = 0;
                for (int first = 0; first < routes; first += 32) {
                    const int route = first + lane;
                    bool m = false;
                    if (route < routes) {
                        const int uu = route % idm.direct_n_used, tt = route / idm.direct_n_used;
                        m = ids[tt*ids_stride + uu] == e && moe_mixed_slot(idm, uu, tt) < 0;
                    }
                    before += __popc(__ballot_sync(0xffffffffu, m && route < own));
                    const unsigned after = __ballot_sync(0xffffffffu, m && route >= own);
                    const int i = n + __popc(after & ((1u << lane) - 1u));
                    if (m && route >= own && i < ai4k::MMV_ID_BN) {
                        gcols[i] = route;
                    }
                    n += __popc(after);
                }
                if (lane == 0) {
                    gn = before % ai4k::MMV_ID_BN == 0 ? min(n, ai4k::MMV_ID_BN) : 0;
                }
            }
            __syncthreads();
            const int n = gn;
            if (n == 0) {
                return; // an earlier route of this expert computes this one
            }
            const block_q8_1 * xs[ai4k::MMV_ID_BN];
            float * ys[ai4k::MMV_ID_BN];
            for (int c = 0; c < n; ++c) {
                const int uu = gcols[c] % idm.direct_n_used, tt = gcols[c] / idm.direct_n_used;
                xs[c] = x + ((int64_t) tt*idm.direct_ne11 + uu % idm.direct_ne11)*pad;
                ys[c] = Y + (int64_t) gcols[c]*M + row0;
            }
            for (int slab = 0; slab < slabs && row0 + slab*16 < M; ++slab) {
                const char * sw = w + (size_t) slab*16*row_stride;
                float * yo[ai4k::MMV_ID_BN];
                for (int c = 0; c < n; ++c) {
                    yo[c] = ys[c] + slab*16;
                }
                int c = 0;
                for (; c + 4 <= n; c += 4) {
                    moe_mixed_cold_rows<BASE, WARPS, 4>(sw, xs + c, yo + c, K);
                }
                if (c + 2 <= n) {
                    moe_mixed_cold_rows<BASE, WARPS, 2>(sw, xs + c, yo + c, K);
                    c += 2;
                }
                if (c < n) {
                    moe_mixed_cold_rows<BASE, WARPS, 1>(sw, xs + c, yo + c, K);
                }
            }
            moe_mixed_done(second, Y, bx, 16*slabs, M, gcols, n, own, idm.direct_n_used);
            return;
        }
    }
    const int64_t act_row = (int64_t) t*idm.direct_ne11 + u % idm.direct_ne11;
    x += act_row*pad;
    // SPW covers four slabs; retain each cold row's original MMVQ reduction.
    for (int slab = 0; slab < slabs && row0 + slab*16 < M; ++slab) {
        const char * sw = w + (size_t) slab*16*row_stride;
        float * sy = out + slab*16;
        if constexpr (IQ4_BANK && WARPS == 4) {
            if (hot && n_tok == 1) {
                moe_mixed_rows_single<GGML_TYPE_IQ4_NL>(sw, x, sy, K);
                continue;
            }
        }
        // Cold rows keep the one-warp order at every width: with the expert cache the unmixed
        // one-token path matches it bit for bit (native four-warp MMVQ does not).
        if constexpr (BASE == GGML_TYPE_COUNT) {
            switch (type) {
#define MOE_MIXED_CASE(T, F, V) case T: moe_mixed_cold_rows<T, WARPS, 1>(sw, &x, &sy, K); break;
            MOE_MIXED_TYPES(MOE_MIXED_CASE)
#undef MOE_MIXED_CASE
                default: __trap();
            }
        } else {
            moe_mixed_cold_rows<BASE, WARPS, 1>(sw, &x, &sy, K);
        }
    }
    moe_mixed_done(second, Y, bx, 16*slabs, M, nullptr, 1, own, idm.direct_n_used);
}

template <ggml_type BASE, bool IQ4_BANK, int LAY, int XSH, int KT, int WARPS, bool SINGLE_N, bool SPW, int STAGES>
static __global__ void __launch_bounds__(WARPS*32, 24/WARPS) moe_mixed_mmv(
        const uint8_t * W, const int8_t * q, const float * s, float * Y, int M, int K, ai4k::idmap idm,
        const char * base, const void * const * table, ggml_type type, size_t expert_stride, size_t row_stride,
        const int32_t * ids, int64_t ids_stride, int n_expert, const block_q8_1 * x,
        const uint8_t * skip, const uint8_t * resident, char * route_ids, size_t route_ids_stride,
        float * route_mask, int phase, int p2_n_tok, const moe_mixed_second second) {
    int bx = blockIdx.x;
    const uint8_t * const W0 = W;
    float * const Y0 = Y;
    // fork AIPFIX P2LIST: the other projection's resident flags (null: this projection's alone decide)
    const uint8_t * other = second.res_both ? second.resident : nullptr;
    if (second.W && bx >= (int) gridDim.x/2) {
        bx -= gridDim.x/2;
        other = second.res_both ? resident : nullptr;
        W = second.W; Y = second.Y; base = second.base; table = second.table;
        skip = second.skip; resident = second.resident; route_ids = nullptr;
    }
    if (p2_n_tok > 0) {
        // Compact phase 2: gridDim.y blocks per row tile split the routes this phase owns (copied, skipped
        // or invalid ones; at most 64), so a layer without misses costs a few hundred blocks, not one per route.
        __shared__ int list[64];
        __shared__ int n_list;
        if (threadIdx.x < 32) {
            const int n_routes = idm.direct_n_used*p2_n_tok;
            int n = 0;
            for (int c = 0; c < n_routes; c += 32) {
                const int i = c + (int) threadIdx.x;
                bool mine = false;
                if (i < n_routes) {
                    const int u = i % idm.direct_n_used, t = i / idm.direct_n_used;
                    const int h = moe_mixed_slot(idm, u, t);
                    const int e = ids[t*ids_stride + u];
                    const bool hot = h >= 0 && h < idm.direct_n_expert;
                    mine = !hot && !(h < 0 && e >= 0 && e < n_expert && resident[e] && (!other || other[e]));
                }
                const unsigned int b = __ballot_sync(0xffffffffu, mine);
                if (mine) {
                    list[n + __popc(b & ((1u << threadIdx.x) - 1))] = i;
                }
                n += __popc(b);
            }
            if (threadIdx.x == 0) {
                n_list = n;
            }
        }
        __syncthreads();
        for (int k = blockIdx.y; k < n_list; k += gridDim.y) {
            const int u = list[k] % idm.direct_n_used, t = list[k] / idm.direct_n_used;
            moe_mixed_cold<BASE, IQ4_BANK, WARPS, SPW>(W, Y, M, K, idm, base, table, type, expert_stride, row_stride,
                    n_expert, x, skip, resident, second, u, t, p2_n_tok, bx, moe_mixed_slot(idm, u, t), false,
                    ids[t*ids_stride + u]);
        }
        return;
    }
    const int u = blockIdx.y, t = blockIdx.z;
    const int h = moe_mixed_slot(idm, u, t);
    // Other slabs resolve the map themselves; these stores serve later projections.
    if (route_ids && bx == 0 && threadIdx.x == 0) {
        *(int32_t *) (route_ids + t*route_ids_stride + u*sizeof(int32_t)) = h;
        route_mask[t*idm.direct_n_used + u] = h < 0 ? 1.0f : 0.0f;
    }
    const bool hot = h >= 0 && h < idm.direct_n_expert;
    // Hit-first split: phase 1 runs before the copy wait (bank and resident routes),
    // phase 2 after it (all other routes); each route is written by exactly one phase.
    const int e = ids[t*ids_stride + u];
    const bool res = phase != 0 && !hot && h < 0 && e >= 0 && e < n_expert && resident[e] && (!other || other[e]);
    if ((phase == 1 && !hot && (!res || second.hot_only)) || (phase == 2 && (hot || res))) {
        return;
    }
    if constexpr (!IQ4_BANK) {
        if (hot) {
            if constexpr (SPW) {
                __shared__ ai4k::mmvr_group group;
                moe_mixed_group(idm, h, group);
                if (group.n == 0) {
                    return;
                }
                if (second.pair_warp) {
                    // Paired warps (as the fused 3XS hit kernel): block x owns WARPS/2 slabs of both projections,
                    // even warps the first, odd warps the second, so SwiGLU needs no cross-block count. The grid
                    // is the same size (2*ceil(slabs/WARPS) >= ceil(slabs/(WARPS/2))); each row keeps its
                    // one-warp k order and the epilogue expression, so the stores are unchanged.
                    const int warp = threadIdx.x >> 5;
                    const int slab = (int) blockIdx.x*(WARPS/2) + (warp >> 1);
                    float * Yw = warp & 1 ? second.Y : Y0;
                    ai4k::mmvr_body<LAY, XSH, true, KT, STAGES, WARPS, 24/WARPS, true, false, 1, 1, false, true, true>(
                            warp & 1 ? second.W : W0, q, s, Yw, M, M, 1, K, idm, &group, (int) blockIdx.x, slab);
                    if (second.glu) {
                        __syncthreads();
                        const int first = (int) blockIdx.x*(WARPS/2);
                        const int rows = 16*max(0, min(WARPS/2, M/16 - first));
                        for (int i = threadIdx.x; i < group.n*rows; i += blockDim.x) {
                            const int64_t o = (int64_t) group.cols[i / rows]*M + first*16 + i % rows;
                            second.glu[o] = ggml_cuda_op_silu_single(second.glu_gate[o]) * second.glu_up[o];
                        }
                    }
                    return;
                }
                ai4k::mmvr_body<LAY, XSH, true, KT, STAGES, WARPS, 24/WARPS, true, false, 1, 1, false, true, true>(W, q, s, Y, M, M, 1, K, idm, &group, bx);
                moe_mixed_done(second, Y, bx, 16*WARPS, M, group.cols, group.n, 0, idm.direct_n_used);
            } else {
                ai4k::mmvr_body<LAY, XSH, true, KT, 2, WARPS, 24/WARPS, true, false, 1, 1, SINGLE_N, false>(W, q, s, Y, M, M, 1, K, idm, nullptr, bx);
                moe_mixed_done(second, Y, bx, 16, M, nullptr, 1, t*idm.direct_n_used + u, idm.direct_n_used);
            }
            return;
        }
    }
    moe_mixed_cold<BASE, IQ4_BANK, WARPS, SPW>(W, Y, M, K, idm, base, table, type, expert_stride, row_stride,
            n_expert, x, skip, resident, second, u, t, gridDim.z, bx, h, hot, e, ids, ids_stride);
}

bool ggml_cuda_moe_mixed_supported(const ggml_tensor * bank, const ggml_tensor * base) {
    if (!bank || !base || bank->op != GGML_OP_MUL_MAT_ID || base->op != GGML_OP_MUL_MAT_ID ||
            !bank->src[0] || !bank->src[1] || !bank->src[2] || !base->src[0] || !base->src[2] ||
            ggml_get_op_params_i32(base, 3) == GGML_PREC_F32 ||
            base->src[1] != bank->src[1] || bank->type != GGML_TYPE_F32 || base->type != GGML_TYPE_F32 ||
            !ggml_are_same_shape(bank, base) ||
            !ggml_is_contiguous(bank) || !ggml_is_contiguous(base)) {
        return false;
    }
    const ggml_tensor * w = bank->src[0], * x = bank->src[1], * ids = bank->src[2], * cw = base->src[0], * ci = base->src[2];
    // AIT4_U banks: the K = 640 down projection only (a banked expert is AI+ in all three matrices).
    if ((w->type != GGML_TYPE_AIT2_U && w->type != GGML_TYPE_AIT2P_U && w->type != GGML_TYPE_AIT3_U && w->type != GGML_TYPE_IQ4_NL &&
            (w->type != GGML_TYPE_AIT4_U || w->ne[0] != 640)) ||
            (w->type == GGML_TYPE_IQ4_NL ? cw->type != GGML_TYPE_IQ4_NL : !ggml_cuda_ai4_supported(bank)) ||
            w->ne[3] != 1 || !ggml_is_contiguous(w) ||
            x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || x->ne[0] <= 0 || x->ne[0] > 65536 || x->ne[0] % 128 ||
            w->ne[1] <= 0 || w->ne[1] > 65536 || w->ne[1] % 16 || w->ne[2] <= 0 || w->ne[2] > INT32_MAX ||
            cw->ne[0] != w->ne[0] || cw->ne[1] != w->ne[1] || cw->ne[2] <= 0 || cw->ne[2] > INT32_MAX ||
            cw->ne[3] != 1 || !ggml_is_contiguous_2(cw) || cw->nb[1] != ggml_row_size(cw->type, cw->ne[0]) ||
            ids->type != GGML_TYPE_I32 || ci->type != GGML_TYPE_I32 || !ggml_are_same_shape(ids, ci) ||
            ids->ne[0] < 1 || ids->ne[0] > 64 || ids->ne[1] < 1 || ids->ne[1] > 8 || ids->ne[2] != 1 || ids->ne[3] != 1 ||
            ids->nb[0] != sizeof(int32_t) || ci->nb[0] != sizeof(int32_t) || ci->nb[1] % sizeof(int32_t) ||
            (x->ne[1] != 1 && x->ne[1] != ids->ne[0]) || x->ne[2] != ids->ne[1] || x->ne[3] != 1) {
        return false;
    }
    switch (cw->type) {
#define MOE_MIXED_SUPPORT(T, F, V) case T: return true;
        MOE_MIXED_TYPES(MOE_MIXED_SUPPORT)
#undef MOE_MIXED_SUPPORT
        default: return false;
    }
}

bool ggml_cuda_moe_mixed_pair_enabled() {
    static const bool v = ai4_env_int("GGML_MOE_BANK_MIXED_PAIR", nullptr, 1) != 0;
    return v;
}

bool ggml_cuda_moe_mixed_pair_supported(const ggml_tensor * bank, const ggml_tensor * base,
        const ggml_tensor * bank2, const ggml_tensor * base2) {
    if (!ggml_cuda_moe_mixed_supported(bank, base) || !ggml_cuda_moe_mixed_supported(bank2, base2)) {
        return false;
    }
    const ggml_tensor * w = bank->src[0], * w2 = bank2->src[0], * cw = base->src[0], * cw2 = base2->src[0];
    return w->type != GGML_TYPE_IQ4_NL && w2->type == w->type && cw2->type == cw->type &&
            ggml_are_same_shape(w, w2) && ggml_are_same_stride(w, w2) &&
            ggml_are_same_shape(cw, cw2) && ggml_are_same_stride(cw, cw2) &&
            bank2->src[1] == bank->src[1] && bank2->src[2] == bank->src[2] && base2->src[2] == base->src[2] &&
            ggml_are_same_shape(bank, bank2);
}

bool ggml_cuda_moe_mixed_reduce_enabled() {
    static const bool v = ai4_env_int("GGML_MOE_BANK_MIXED_REDUCE", nullptr, 1) != 0;
    return v;
}

bool ggml_cuda_moe_mixed_reduce_supported(const ggml_tensor * experts, const ggml_cuda_moe_mixed_reduce & red) {
    const ggml_tensor * w = red.weights, * b = red.bank_ids, * d = red.dst;
    const int64_t n_used = experts->ne[1], tokens = experts->ne[2];
    return w && b && d && experts->type == GGML_TYPE_F32 && ggml_is_contiguous(experts) && experts->ne[3] == 1 &&
            n_used >= 1 && n_used <= 32 && tokens >= 1 && tokens <= 8 && experts->ne[0] % 16 == 0 &&
            experts->ne[0]/16*tokens <= MOE_MIXED_RED_TILES &&
            w->type == GGML_TYPE_F32 && ggml_is_contiguous(w) && w->ne[0] == 1 && w->ne[1] == n_used &&
            w->ne[2] == tokens && w->ne[3] == 1 &&
            b->type == GGML_TYPE_I32 && b->nb[0] == sizeof(int32_t) && b->nb[1] % sizeof(int32_t) == 0 &&
            b->ne[0] == n_used && b->ne[1] == tokens && b->ne[2] == 1 && b->ne[3] == 1 &&
            d->type == GGML_TYPE_F32 && ggml_is_contiguous(d) && ggml_nelements(d) == experts->ne[0]*tokens &&
            d->data != experts->data && d->data != w->data && d->data != b->data;
}

bool ggml_cuda_moe_mixed_glu_enabled() {
    static const bool v = ai4_env_int("GGML_MOE_BANK_MIXED_GLU", nullptr, 1) != 0;
    return v;
}

bool ggml_cuda_moe_mixed_glu_supported(const ggml_tensor * dst, const ggml_tensor * dst2, const ggml_tensor * glu) {
    return glu && glu->op == GGML_OP_GLU && ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU &&
            ggml_get_op_params_i32(glu, 1) == 0 && glu != dst && glu != dst2 &&
            ((glu->src[0] == dst && glu->src[1] == dst2) || (glu->src[0] == dst2 && glu->src[1] == dst)) &&
            glu->type == GGML_TYPE_F32 && ggml_is_contiguous(glu) && ggml_are_same_shape(glu, dst) &&
            dst->ne[0] % 16 == 0 && dst->ne[0]/16*dst->ne[1]*dst->ne[2] <= MOE_MIXED_GLU_TILES;
}

const ggml_tensor * ggml_cuda_moe_mixed_base(const ggml_tensor * joined) {
    if (!joined || joined->op != GGML_OP_ADD || !joined->src[0] || !joined->src[1]) { return nullptr; }
    const ggml_tensor * hot = joined->src[0], * mul = joined->src[1];
    if (mul->op != GGML_OP_MUL || !mul->src[0] || !mul->src[1] ||
            !ggml_cuda_moe_mixed_supported(hot, mul->src[0]) || !ggml_are_same_shape(joined, hot) ||
            !ggml_are_same_shape(mul, hot) || !ggml_is_contiguous(joined) || !ggml_is_contiguous(mul)) { return nullptr; }
    const ggml_tensor * mask = mul->src[1];
    if (mask->type != GGML_TYPE_F32 || !ggml_is_contiguous(mask) || mask->op != GGML_OP_RESHAPE ||
            mask->ne[0] != 1 || mask->ne[1] != hot->ne[1] || mask->ne[2] != hot->ne[2] || mask->ne[3] != 1) { return nullptr; }
    const ggml_tensor * step = mask->src[0];
    if (!step || step->op != GGML_OP_UNARY || ggml_get_unary_op(step) != GGML_UNARY_OP_STEP) { return nullptr; }
    const ggml_tensor * scale = step->src[0];
    if (!scale || scale->op != GGML_OP_SCALE || ggml_get_op_params_f32(scale, 0) != -1.0f ||
            ggml_get_op_params_f32(scale, 1) != -0.5f) { return nullptr; }
    const ggml_tensor * cast = scale->src[0];
    if (!cast || cast->op != GGML_OP_CPY || cast->type != GGML_TYPE_F32 || cast->src[0] != hot->src[2]) { return nullptr; }
    return mul->src[0];
}

void ggml_cuda_moe_mixed_quantize_x(ggml_backend_cuda_context & ctx, const ggml_tensor * bank,
        ggml_cuda_pool_alloc<int8_t> & q, ggml_cuda_pool_alloc<float> & s) {
    if (bank->src[0]->type == GGML_TYPE_IQ4_NL) {
        return;
    }
    const ggml_tensor * x = bank->src[1];
    const int K = (int) x->ne[0];
    const int xsh = K % 256 == 0 ? 1 : 0;
    const size_t rows = (size_t) x->ne[1]*x->ne[2];
    q.alloc(rows*K);
    s.alloc(rows*(K >> (7 + xsh)));
    ai4_quantize_x((const float *) x->data, rows, K, xsh, q.get(), s.get(), ctx.stream());
}

// fork AIPFIX D: AI+ unit experts in the expert cache (see ai4-mm.cuh).
size_t ggml_cuda_moe_ai_act_scales(int64_t rows, int64_t K) {
    return GGML_PAD((size_t) (rows*K), 16);
}

size_t ggml_cuda_moe_ai_act_bytes(int64_t rows, int64_t K) {
    const int xsh = K % 256 == 0 ? 1 : 0;
    return ggml_cuda_moe_ai_act_scales(rows, K) + (size_t) rows*(K >> (7 + xsh))*sizeof(float);
}

void ggml_cuda_moe_ai_quantize(const float * x, int64_t rows, int64_t K, char * out, cudaStream_t stream) {
    const int xsh = K % 256 == 0 ? 1 : 0;
    ai4_quantize_x(x, rows, (int) K, xsh, (int8_t *) out, (float *) (out + ggml_cuda_moe_ai_act_scales(rows, K)), stream);
}

bool ggml_cuda_moe_ai_table_supported(ggml_type type, int64_t K, int64_t M, int64_t n_tok) {
    const int rate = ggml_ai4_rate(type);
    return ggml_ai4_is_unit(type) && rate >= 2 && rate <= 4 && (rate == 4 ? ai4_aikern() : ai4_mmvr_on_id()) &&
        K % 128 == 0 && M % 16 == 0 && n_tok >= 1 && n_tok <= 64;
}

bool ggml_cuda_moe_ai_table_mmv(const void * const * table, ggml_type type, const char * act, const int32_t * ids,
        float * dst, int64_t K, int64_t M, int64_t n_used, int64_t n_tok, int64_t ne11, int64_t ids_stride,
        int n_expert, const uint8_t * filter, int filter_val, cudaStream_t stream,
        const void * const * table2, float * dst2, const uint8_t * filter2,
        float * glu, const uint8_t * prev, const uint8_t * prev2) {
    if (!table || !ggml_cuda_moe_ai_table_supported(type, K, M, n_tok) || (table2 && !dst2) || (glu && !table2) ||
            (glu && M/16*n_used*n_tok > ai4k::GLU_PAIR_TILES)) {
        return false;
    }
    const int xsh = K % 256 == 0 ? 1 : 0;
    const int8_t * q = (const int8_t *) act;
    const float * s = (const float *) (act + ggml_cuda_moe_ai_act_scales(n_tok*ne11, K));
    ai4k::idmap idm{};
    idm.direct_ids = (const char *) ids;
    idm.direct_nb0 = sizeof(int32_t);
    idm.direct_nb1 = ids_stride*sizeof(int32_t);
    idm.direct_n_used = (int) n_used;
    idm.direct_ne11 = (int) ne11;
    idm.direct_n_expert = n_expert;
    idm.table = table;
    idm.filter = filter;
    idm.filter_val = filter_val;
    idm.table2 = table2;
    idm.filter2 = filter2;
    idm.y2 = dst2;
    idm.glu = glu;
    idm.glu_up = dst;
    idm.glu_gate = dst2;
    idm.glu_prev = prev;
    idm.glu_prev2 = prev2;
    const int max_tiles = (int) (n_used*n_tok);
    bool ok = false;
    switch (ai4_lay_of(type)) {
#define AI4_TABLE_U(L) \
        case L: \
            ok = xsh ? ai4_mmvr_try<L, 1, true>(nullptr, q, s, dst, (int) M, (int) n_tok, (int) K, stream, idm, max_tiles) \
                     : ai4_mmvr_try<L, 0, true>(nullptr, q, s, dst, (int) M, (int) n_tok, (int) K, stream, idm, max_tiles); \
            break;
        AI4_TABLE_U(ai4k::LAY_AIT2_U)
        AI4_TABLE_U(ai4k::LAY_AIT2P_U)
        AI4_TABLE_U(ai4k::LAY_AIT3_U)
        AI4_TABLE_U(ai4k::LAY_AIT4_U)
#undef AI4_TABLE_U
        default: break;
    }
    if (ok) {
        CUDA_CHECK(cudaGetLastError());
    }
    return ok;
}

// fork AIPFIX: the first nai blocks along x run the AI activation quantizer's CTAs, the rest quantize_q8_1's
// elements with 128 threads per block. Each part runs the same per-CTA / per-warp code as its own kernel.
template <bool XSH, bool SWZ>
static __global__ void __launch_bounds__(128) moe_mixed_quantize_dual(const float * __restrict__ x, int K,
        int8_t * __restrict__ q, float * __restrict__ s, void * __restrict__ q8, int padded, int nai) {
    const int tok = blockIdx.y;
    if ((int) blockIdx.x < nai) {
        if constexpr (XSH) {
            rq4::quantize_act_body(x, K, q, s, K, gridDim.y, blockIdx.x, tok);
        } else {
            ai4k::quantize_act128_body(x, K, q, s, K, gridDim.y, blockIdx.x, tok);
        }
        return;
    }
    const int i0 = ((int) blockIdx.x - nai)*128 + threadIdx.x;
    if (i0 >= padded) {
        return;
    }
    quantize_q8_1_elem<SWZ>(x, q8, (int64_t) tok*padded + i0, i0 < K ? (int64_t) tok*K + i0 : -1);
}

static bool moe_mixed_dualq_enabled() {
    static const bool v = ai4_env_int("GGML_MOE_BANK_MIXED_DUALQ", nullptr, 1) != 0;
    return v;
}

bool ggml_cuda_moe_mixed_quantize_dual(ggml_backend_cuda_context & ctx, const ggml_tensor * bank, const ggml_tensor * x,
        char * q8, ggml_type q8_type, ggml_cuda_pool_alloc<int8_t> & q, ggml_cuda_pool_alloc<float> & s) {
    const ggml_tensor * bx = bank->src[1];
    if (!moe_mixed_dualq_enabled() || bank->src[0]->type == GGML_TYPE_IQ4_NL || x->data != bx->data ||
            !ggml_are_same_shape(x, bx) || x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) ||
            !ggml_is_contiguous(bx) || x->ne[3] != 1) {
        return false;
    }
    const int K = (int) x->ne[0];
    const int64_t rows = x->ne[1]*x->ne[2];
    const int padded = GGML_PAD(K, MATRIX_ROW_PADDING);
    if (rows > 65535 || K % 128 != 0 || padded % 128 != 0) {
        return false;
    }
    const int xsh = K % 256 == 0 ? 1 : 0;
    q.alloc(rows*K);
    s.alloc(rows*(K >> (7 + xsh)));
    const int nai = xsh ? (K/RQ4_HB + 3)/4 : (K/128 + 7)/8;
    const dim3 grid(nai + padded/128, (unsigned) rows);
    const bool swz = q8_type == GGML_TYPE_IQ4_XS;
    const float * xd = (const float *) x->data;
    if (xsh) {
        if (swz) { moe_mixed_quantize_dual<true, true><<<grid, 128, 0, ctx.stream()>>>(xd, K, q.get(), s.get(), q8, padded, nai); }
        else { moe_mixed_quantize_dual<true, false><<<grid, 128, 0, ctx.stream()>>>(xd, K, q.get(), s.get(), q8, padded, nai); }
    } else {
        if (swz) { moe_mixed_quantize_dual<false, true><<<grid, 128, 0, ctx.stream()>>>(xd, K, q.get(), s.get(), q8, padded, nai); }
        else { moe_mixed_quantize_dual<false, false><<<grid, 128, 0, ctx.stream()>>>(xd, K, q.get(), s.get(), q8, padded, nai); }
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

void ggml_cuda_mul_mat_id_mixed(ggml_backend_cuda_context & ctx, const ggml_tensor * bank,
        const ggml_tensor * base, ggml_tensor * dst, const char * act_q8,
        const void * const * table, const uint8_t * skip, const uint8_t * resident,
        const ggml_tensor * route_map, ggml_tensor * route_mask,
        const int8_t * act_ai_q, const float * act_ai_s, int phase, const ggml_cuda_moe_mixed_pair * pair,
        const ggml_cuda_moe_mixed_reduce * red) {
    GGML_ASSERT(ggml_cuda_moe_mixed_supported(bank, base));
    // phase 3 = phase 1 over the bank routes only (fork AIPFIX P1HITG down: the 3XS hit kernel ran the cached ones)
    const bool p1_hot_only = phase == 3;
    phase = p1_hot_only ? 1 : phase;
    GGML_ASSERT(phase >= 0 && phase <= 2 && (phase == 0 || resident));
    GGML_ASSERT(!pair || (ggml_cuda_moe_mixed_pair_supported(bank, base, pair->bank, pair->base) &&
            (table == nullptr) == (pair->table == nullptr) && (phase == 0 || pair->resident) &&
            pair->dst->type == GGML_TYPE_F32 && ggml_is_contiguous(pair->dst) && ggml_are_same_shape(pair->dst, dst)));
    const ggml_tensor * w = bank->src[0], * x = bank->src[1], * ids = bank->src[2], * cw = base->src[0], * ci = base->src[2];
    const int K = (int) x->ne[0], M = (int) w->ne[1];
    GGML_ASSERT(!pair || !pair->glu || (pair->glu->type == GGML_TYPE_F32 && ggml_is_contiguous(pair->glu) &&
            ggml_are_same_shape(pair->glu, dst) && M/16*ids->ne[0]*ids->ne[1] <= MOE_MIXED_GLU_TILES &&
            (pair->glu->src[0]->data == dst->data) != (pair->glu->src[0]->data == pair->dst->data)));
    const int xsh = K % 256 == 0 ? 1 : 0;
    const size_t rows = (size_t) x->ne[1]*x->ne[2];
    // q8 is taken from the pool before q and s on every path: the pool frees in reverse order of allocation.
    ggml_cuda_pool_alloc<char> q8(ctx.pool());
    ggml_cuda_pool_alloc<int8_t> q(ctx.pool());
    ggml_cuda_pool_alloc<float> s(ctx.pool());
    GGML_ASSERT((act_ai_q == nullptr) == (act_ai_s == nullptr));
    if (!act_q8) {
        q8.alloc(rows*(GGML_PAD(K, MATRIX_ROW_PADDING)/QK8_1)*sizeof(block_q8_1));
    }
    if (!act_ai_q && !act_q8) {
        char * p = q8.get();
        if (ggml_cuda_moe_mixed_quantize_dual(ctx, bank, x, p, cw->type, q, s)) {
            act_ai_q = q.get();
            act_ai_s = s.get();
            act_q8 = p;
        }
    }
    if (!act_ai_q) {
        ggml_cuda_moe_mixed_quantize_x(ctx, bank, q, s);
        act_ai_q = q.get();
        act_ai_s = s.get();
    }
    if (!act_q8) {
        const int padded = GGML_PAD(K, MATRIX_ROW_PADDING);
        char * p = q8.get();
        quantize_row_q8_1_cuda((const float *) x->data, nullptr, p, cw->type, K,
                x->nb[1]/sizeof(float), x->nb[2]/sizeof(float), x->nb[3]/sizeof(float),
                padded, x->ne[1], x->ne[2], 1, ctx.stream());
        act_q8 = p;
    }
    ai4k::idmap idm{};
    idm.nb02 = w->nb[2]; idm.direct_ids = (const char *) ids->data;
    idm.direct_nb0 = ids->nb[0]; idm.direct_nb1 = ids->nb[1];
    idm.direct_n_used = (int) ids->ne[0]; idm.direct_ne11 = (int) x->ne[1]; idm.direct_n_expert = (int) w->ne[2];
    GGML_ASSERT((route_map == nullptr) == (route_mask == nullptr));
    if (route_map) {
        GGML_ASSERT(route_map->type == GGML_TYPE_F32 && ggml_is_contiguous(route_map));
        GGML_ASSERT(ggml_nelements(route_map) == cw->ne[2]);
        GGML_ASSERT(route_mask->type == GGML_TYPE_F32 && ggml_is_contiguous(route_mask));
        GGML_ASSERT(ggml_nelements(route_mask) == ggml_nelements(ids));
        idm.direct_ids = (const char *) ci->data;
        idm.direct_nb0 = ci->nb[0]; idm.direct_nb1 = ci->nb[1];
        idm.direct_map = (const float *) route_map->data;
        idm.direct_map_size = (int) cw->ne[2];
    }
    // The phase-2 launch holds no bank or resident routes: 16-row blocks of four warps spread its few copied
    // routes over four times the blocks of an SPW grid (each row's arithmetic is the same; the epilogues count
    // 16-row slices either way). GGML_MOE_BANK_MIXED_P2_FINE=0 keeps the SPW grid.
    static const bool p2_fine = ai4_env_int("GGML_MOE_BANK_MIXED_P2_FINE", nullptr, 1) != 0;
    const bool fine = p2_fine && phase == 2;
    // Hot K=640 downs: the slab-per-warp grid gives one warp 16 rows and all of K (2.5 units per warp, so the
    // 4-stage prefetch never fills; d9 traces: AIT2_U down 35.7 us vs Q2_0 sw 27.9, AIT4_U 55-105 vs IQ4_NL 35-44).
    // Default: the 16-row split-K blocks (M/16 x routes x tokens) that gate/up used before SPW. The k order changes
    // (KLD-gated, not identity). GGML_CUDA_AI23_DOWN_SPW=1 keeps the SPW grid for K=640.
    static const bool down_spw = ai4_env_int("GGML_CUDA_AI23_DOWN_SPW", nullptr, 0) != 0;
    const bool spw = !fine && w->type != GGML_TYPE_IQ4_NL && (K != 640 || down_spw) &&
            ai4_spw_enabled(M, K, idm, (int) ggml_nelements(ids));
    // Compact phase 2 (as the native miss list): four blocks per 16-row tile walk only the routes phase 2 owns,
    // instead of one block per route and tile that mostly exits. GGML_MOE_BANK_MIXED_P2_COMPACT=0 opts out.
    static const bool p2_compact_env = ai4_env_int("GGML_MOE_BANK_MIXED_P2_COMPACT", nullptr, 1) != 0;
    const bool p2_compact = p2_compact_env && fine && ggml_nelements(ids) <= 64;
    const int p2_n_tok = p2_compact ? (int) ids->ne[1] : 0;
    // Blocks per 16-row tile in compact phase 2: each block takes every gridDim.y-th miss route, so with four blocks
    // a layer's 1-3 misses run one route per block, serially per tile (d9: 12.2 us gate+up pair / 6.8 down mean,
    // 2.8 / 3.1 min, vs the native list kernel 4.3). Default 8 spreads misses over twice the blocks; the per-route
    // arithmetic and k order are unchanged (identity). GGML_MOE_BANK_MIXED_P2_Y=4 restores the old grid.
    static const int p2_y = std::min(64, std::max(1, ai4_env_int("GGML_MOE_BANK_MIXED_P2_Y", nullptr, 8)));
    // Slabs per hit-first block: 4 warps x 16 rows is 20 blocks per route for a 640-row projection (the fused 3XS
    // hit kernel runs 80); 2 warps doubles the block count at the same warps per SM and keeps each row's k order.
    // GGML_CUDA_AI23_MIXED_SPW_WARPS=2|4 (default 4).
    static const int spw_warps = ai4_env_int("GGML_CUDA_AI23_MIXED_SPW_WARPS", nullptr, 4) == 2 ? 2 : 4;
    const int slabs_per_block = spw ? spw_warps : 1;
    const dim3 grid((M/16 + slabs_per_block - 1)/slabs_per_block*(pair ? 2 : 1), p2_compact ? p2_y : ids->ne[0], p2_compact ? 1 : ids->ne[1]);
    moe_mixed_second second{};
    if (pair) {
        second = { (const uint8_t *) pair->bank->src[0]->data, (float *) pair->dst->data,
                   (const char *) pair->base->src[0]->data, pair->table, pair->skip, pair->resident,
                   nullptr, nullptr, nullptr };
        if (pair->glu) {
            const bool gate_first = pair->glu->src[0]->data == dst->data;
            second.glu_gate = (const float *) (gate_first ? dst : pair->dst)->data;
            second.glu_up = (const float *) (gate_first ? pair->dst : dst)->data;
            second.glu = (float *) pair->glu->data;
        }
    }
    // Hot gate+up of a paired SPW launch in one block (warp pairs, block-local SwiGLU); cold routes keep the
    // two-half grid. Default on; GGML_CUDA_AI23_MIXED_PAIR_WARP=0 opts out.
    static const bool pair_warp = ai4_env_int("GGML_CUDA_AI23_MIXED_PAIR_WARP", nullptr, 1) != 0;
    second.pair_warp = pair && spw && pair_warp;
    // Cold (base-type) routes grouped by expert in multi-token launches. Off by default: the paired coding-100K run
    // (f3b-speed) measured it 0.65 ms/round slower than ungrouped (p = 0.049). GGML_MOE_BANK_MIXED_COLD_GROUP=1 opts in.
    static const bool cold_group = ai4_env_int("GGML_MOE_BANK_MIXED_COLD_GROUP", nullptr, 0) != 0;
    second.cold_group = cold_group;
    second.res_both = pair && pair->res_both;
    second.hot_only = phase == 1 && (p1_hot_only || (pair && pair->hot_only));
    GGML_ASSERT(!red || (!pair && ggml_cuda_moe_mixed_reduce_supported(dst, *red)));
    if (red) {
        second.red_w = (const float *) red->weights->data;
        second.red_bank = (const int32_t *) red->bank_ids->data;
        second.red_bank_stride = red->bank_ids->nb[1]/sizeof(int32_t);
        second.red_dst = (float *) red->dst->data;
        second.red_tree = ggml_cuda_moe_mixed_reduction_tree(dst);
    }
    static const int trace = getenv("GGML_MOE_BANK_MIXED_TRACE") ? atoi(getenv("GGML_MOE_BANK_MIXED_TRACE")) : 0;
    const int cfg = ai4_mmvr_cfg();
    const bool iq4_hot_single = w->type == GGML_TYPE_IQ4_NL && ids->ne[1] == 1;
    // moe_mixed_rows_single partials (IQ4 hot and one-token cold rows).
    const size_t iq4_smem = ids->ne[1] == 1 ? 3*16*32*sizeof(float) : 0;
    const bool two_warps = !fine && !iq4_hot_single && (cfg == 212 || (cfg == 0 && ids->ne[1] > 1 && K >= 1024));
    static const bool single = ai4_env_int("GGML_CUDA_AI23_MMVR_SINGLE_ID", nullptr, 0) != 0 ||
                               ai4_env_int("GGML_CUDA_AI23_VERIFY_COMPACT", nullptr, 0) != 0;
    // Deeper hot-route weight prefetch; the K order and arithmetic are unchanged.
    static const int spw_stages_all = [] {
        const int v = ai4_env_int("GGML_CUDA_AI23_MIXED_STAGES", nullptr, 4);
        return v == 2 || v == 8 ? v : 4;
    }();
    // Phase 1 runs while the layer's miss copy streams from host memory (same as the native hit pass): a shallower
    // pipeline (smaller shared-memory stages, fewer loads in flight) leaves the copy kernel room. Same K order and
    // arithmetic. Default 0 keeps the common depth (no measured gain, AIPTAB r52); =2|4|8 opts in.
    static const int p1_stages = [] {
        const int v = ai4_env_int("GGML_MOE_BANK_MIXED_P1_STAGES", nullptr, 0);
        return v == 2 || v == 4 || v == 8 ? v : 0;
    }();
    const int spw_stages = phase == 1 && p1_stages ? p1_stages : spw_stages_all;
#define MOE_MIXED_LAUNCH(T, IQ4, L, XSH, KT, WARPS, SINGLE, SPW, ST) \
    moe_mixed_mmv<T, IQ4, L, XSH, KT, WARPS, SINGLE, SPW, ST><<<grid, WARPS*32, std::max((size_t) (WARPS*ST*ai4k::mmv_stage_bytes<true, L>()), iq4_smem), ctx.stream()>>>( \
            (const uint8_t *) w->data, act_ai_q, act_ai_s, (float *) dst->data, M, K, idm, \
            (const char *) cw->data, table, cw->type, cw->nb[2], cw->nb[1], (const int32_t *) ci->data, \
            ci->nb[1]/sizeof(int32_t), (int) cw->ne[2], (const block_q8_1 *) act_q8, skip, resident, \
            route_map && phase != 2 ? (char *) ids->data : nullptr, ids->nb[1], route_mask ? (float *) route_mask->data : nullptr, phase, p2_n_tok, second)
#define MOE_MIXED_SINGLE(T, L, XSH, KT, WARPS) \
    if (single) { MOE_MIXED_LAUNCH(T, false, L, XSH, KT, WARPS, true, false, 2); } \
    else        { MOE_MIXED_LAUNCH(T, false, L, XSH, KT, WARPS, false, false, 2); }
#define MOE_MIXED_WARPS(T, L, XSH, KT) \
    if (spw && spw_warps == 2)       { MOE_MIXED_LAUNCH(T, false, L, XSH, KT, 2, false, true, 4); } \
    else if (spw && spw_stages == 8) { MOE_MIXED_LAUNCH(T, false, L, XSH, KT, 4, false, true, 8); } \
    else if (spw && spw_stages == 4) { MOE_MIXED_LAUNCH(T, false, L, XSH, KT, 4, false, true, 4); } \
    else if (spw)       { MOE_MIXED_LAUNCH(T, false, L, XSH, KT, 4, false, true, 2); } \
    else if (two_warps) { MOE_MIXED_SINGLE(T, L, XSH, KT, 2); } \
    else                { MOE_MIXED_SINGLE(T, L, XSH, KT, 4); }
#define MOE_MIXED_SHAPE(L) \
    if (K == 640 && ai4_mmvr_kt())       { MOE_MIXED_WARPS(GGML_TYPE_COUNT, L, 0, 640); } \
    else if (K == 2560 && ai4_mmvr_kt()) { MOE_MIXED_WARPS(GGML_TYPE_COUNT, L, 1, 2560); } \
    else if (xsh)                      { MOE_MIXED_WARPS(GGML_TYPE_COUNT, L, 1, 0); } \
    else                               { MOE_MIXED_WARPS(GGML_TYPE_COUNT, L, 0, 0); }
    // IQ4 banks have no AI arithmetic; their unused layout, K and SINGLE choices need no variants.
    if (w->type == GGML_TYPE_IQ4_NL) {
        if (two_warps) { MOE_MIXED_LAUNCH(GGML_TYPE_IQ4_NL, true, ai4k::LAY_AIT2_U, 0, 0, 2, false, false, 2); }
        else           { MOE_MIXED_LAUNCH(GGML_TYPE_IQ4_NL, true, ai4k::LAY_AIT2_U, 0, 0, 4, false, false, 2); }
    // Limit format specializations to production pairs; other shapes retain the generic dispatch.
    } else if (ai4_mmvr_kt() && K == 2560 && M == 640 && w->type == GGML_TYPE_AIT2_U && cw->type == GGML_TYPE_IQ2_XXS) {
        MOE_MIXED_WARPS(GGML_TYPE_IQ2_XXS, ai4k::LAY_AIT2_U, 1, 2560);
    } else if (ai4_mmvr_kt() && K == 2560 && M == 640 && w->type == GGML_TYPE_AIT2_U && cw->type == GGML_TYPE_IQ2_XS) {
        MOE_MIXED_WARPS(GGML_TYPE_IQ2_XS, ai4k::LAY_AIT2_U, 1, 2560);
    } else if (ai4_mmvr_kt() && K == 2560 && M == 640 && w->type == GGML_TYPE_AIT2_U && cw->type == GGML_TYPE_IQ2_S) {
        MOE_MIXED_WARPS(GGML_TYPE_IQ2_S, ai4k::LAY_AIT2_U, 1, 2560);
    } else if (ai4_mmvr_kt() && K == 2560 && M == 640 && w->type == GGML_TYPE_AIT3_U && cw->type == GGML_TYPE_IQ3_XXS) {
        MOE_MIXED_WARPS(GGML_TYPE_IQ3_XXS, ai4k::LAY_AIT3_U, 1, 2560);
    } else if (ai4_mmvr_kt() && K == 2560 && M == 640 && w->type == GGML_TYPE_AIT3_U && cw->type == GGML_TYPE_IQ3_S) {
        MOE_MIXED_WARPS(GGML_TYPE_IQ3_S, ai4k::LAY_AIT3_U, 1, 2560);
    } else if (ai4_mmvr_kt() && K == 640 && M == 2560 && w->type == GGML_TYPE_AIT2_U && cw->type == GGML_TYPE_Q2_0) {
        MOE_MIXED_WARPS(GGML_TYPE_Q2_0, ai4k::LAY_AIT2_U, 0, 640);
    } else if (ai4_mmvr_kt() && K == 640 && M == 2560 && w->type == GGML_TYPE_AIT4_U && cw->type == GGML_TYPE_IQ4_NL) {
        MOE_MIXED_WARPS(GGML_TYPE_IQ4_NL, ai4k::LAY_AIT4_U, 0, 640);
    } else if (w->type == GGML_TYPE_AIT4_U) {
        if (ai4_mmvr_kt()) { MOE_MIXED_WARPS(GGML_TYPE_COUNT, ai4k::LAY_AIT4_U, 0, 640); }
        else               { MOE_MIXED_WARPS(GGML_TYPE_COUNT, ai4k::LAY_AIT4_U, 0, 0); }
    } else if (w->type == GGML_TYPE_AIT2_U) {
        MOE_MIXED_SHAPE(ai4k::LAY_AIT2_U);
    } else if (ai4_mmvr_kt() && K == 2560 && M == 640 && w->type == GGML_TYPE_AIT2P_U && cw->type == GGML_TYPE_IQ2_XXS) {
        MOE_MIXED_WARPS(GGML_TYPE_IQ2_XXS, ai4k::LAY_AIT2P_U, 1, 2560);
    } else if (ai4_mmvr_kt() && K == 2560 && M == 640 && w->type == GGML_TYPE_AIT2P_U && cw->type == GGML_TYPE_IQ2_XS) {
        MOE_MIXED_WARPS(GGML_TYPE_IQ2_XS, ai4k::LAY_AIT2P_U, 1, 2560);
    } else if (ai4_mmvr_kt() && K == 2560 && M == 640 && w->type == GGML_TYPE_AIT2P_U && cw->type == GGML_TYPE_IQ2_S) {
        MOE_MIXED_WARPS(GGML_TYPE_IQ2_S, ai4k::LAY_AIT2P_U, 1, 2560);
    } else if (ai4_mmvr_kt() && K == 640 && M == 2560 && w->type == GGML_TYPE_AIT2P_U && cw->type == GGML_TYPE_Q2_0) {
        MOE_MIXED_WARPS(GGML_TYPE_Q2_0, ai4k::LAY_AIT2P_U, 0, 640);
    } else if (w->type == GGML_TYPE_AIT2P_U) {
        MOE_MIXED_SHAPE(ai4k::LAY_AIT2P_U);
    } else {
        MOE_MIXED_SHAPE(ai4k::LAY_AIT3_U);
    }
#undef MOE_MIXED_SHAPE
#undef MOE_MIXED_WARPS
#undef MOE_MIXED_SINGLE
#undef MOE_MIXED_LAUNCH
    CUDA_CHECK(cudaGetLastError());
    if (trace) {
        fprintf(stderr, "[moe-bank-mixed] launch K=%d M=%d routes=%lld width=%lld bank=%s base=%s table=%d routing=%d spw=%d phase=%d pair=%d glu=%d reduce=%d pw=%d\n",
                K, M, (long long) ids->ne[0], (long long) ids->ne[1], ggml_type_name(w->type), ggml_type_name(cw->type), table != nullptr, route_map != nullptr, spw, phase, pair != nullptr, pair && pair->glu, red != nullptr, (int) second.pair_warp);
    }
}
#undef MOE_MIXED_TYPES
// Fractional pair path. Reuse the numerically checked pair decoder and existing
// activation rotation; keep the fractional rate separate from integer AI+ dispatch.
bool ggml_cuda_aif25_supported(const ggml_tensor * op) {
    const auto * w = op->src[0]; const auto * x = op->src[1];
    return op->op == GGML_OP_MUL_MAT && w->type == GGML_TYPE_AIF25_U &&
        x->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
        w->ne[0] > 0 && w->ne[0] <= INT32_MAX && w->ne[1] > 0 && w->ne[1] <= INT32_MAX &&
        w->ne[2] == 1 && w->ne[3] == 1 &&
        w->ne[0] % 128 == 0 && w->ne[1] % 16 == 0 &&
        ggml_is_contiguous(w) && ggml_is_contiguous(x) && ggml_is_contiguous(op);
}

void ggml_cuda_mul_mat_aif25(ggml_backend_cuda_context & ctx, const ggml_tensor * w,
                            const ggml_tensor * x, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_aif25_supported(dst));
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    GGML_ASSERT(GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_AMPERE);
    const int K = w->ne[0], M = w->ne[1];
    const int64_t N = ggml_nrows(x);
    const int xsh = K % 256 == 0 ? 1 : 0;
    auto stream = ctx.stream();
    ggml_cuda_pool_alloc<int8_t> q(ctx.pool(), (size_t)N*K);
    ggml_cuda_pool_alloc<float> scales(ctx.pool(), (size_t)N*(K >> (7+xsh)));
    // The pair kernel reads rotated columns in their original order.
    ai4_quantize_x<false, false, true>((const float *)x->data, N, K, xsh, q.get(), scales.get(), stream);
    // CUDA grid.y is limited to 65535; each block handles eight activation rows.
    for (int64_t n0 = 0; n0 < N; n0 += 8*65535) {
        const int nr = (int) std::min<int64_t>(N-n0, 8*65535);
        const dim3 grid(M/16, (nr+7)/8);
        const float * xs = scales.get() + (size_t)n0*(K >> (7+xsh));
        if (xsh) {
            magic2::mmv<5,2,AIF25_MUL,1><<<grid,128,0,stream>>>((const uint8_t *)w->data,
                q.get()+(size_t)n0*K, xs, (float *)dst->data+(size_t)n0*M, M,K,nr);
        } else {
            magic2::mmv<5,2,AIF25_MUL,0><<<grid,128,0,stream>>>((const uint8_t *)w->data,
                q.get()+(size_t)n0*K, xs, (float *)dst->data+(size_t)n0*M, M,K,nr);
        }
        CUDA_CHECK(cudaGetLastError());
    }
}
