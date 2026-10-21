// AI4 (Ampere INT4, formerly D2) CUDA backend entry: reference and fast-path matmul dispatch.
#include "ai4-mm.cuh"
#include "ai4.cuh"
#include "ai4-kern.cuh"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
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
//   GGML_CUDA_AI6_REF=1        AI6_S takes the rate-k reference path (reconstruct + SGEMM) instead of its rate-6 IMMA
//                             kernels (an A/B check of those kernels, like GGML_CUDA_AI5_REF for rate 5)
// The pre-rename GGML_CUDA_D2_* spellings (the format was called D2 until 2026-10-07) are still read when the AI4
// name is unset, so older run scripts keep selecting the same kernels.
static int ai4_env_int(const char * name, const char * legacy_name, int def) {
    const char * e = getenv(name);
    if (!e && legacy_name) {
        e = getenv(legacy_name);
    }
    return e ? atoi(e) : def;
}
static bool ai4_ref() {
    static const bool v = ai4_env_int("GGML_CUDA_AI4_REF", "GGML_CUDA_D2_REF", 0) != 0;
    return v;
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

// Activations: rotate + int8-quantize X [N rows of K] once. XSH = 1: H256 and one scale per 256 (track B's kernel, used by
// AI4 and by AI4_U with K % 256 == 0); XSH = 0: H128 and one scale per 128 (AI4_U with K % 256 != 0).
// gridDim.y carries the rows (<= 65535), so longer inputs run in pieces.
static void ai4_quantize_x(const float * X, int64_t N, int K, int xsh, int8_t * q, float * s, cudaStream_t stream) {
    const int64_t ns = K >> (7 + xsh);
    for (int64_t r0 = 0; r0 < N; r0 += 65535) {
        const int nr = (int) std::min<int64_t>(65535, N - r0);
        if (xsh) {
            rq4::launch_quantize_act(X + r0*K, K, q + r0*K, s + r0*ns, K, nr, stream);
        } else {
            ai4k::launch_quantize_act128(X + r0*K, K, q + r0*K, s + r0*ns, K, nr, stream);
        }
    }
}

template <int LAY, int XSH>
static void ai4_mm_kernels(const uint8_t * W, const int8_t * q, const float * s, float * Y, int M, int N, int K, cudaStream_t stream) {
    if (N <= 8) {
        if (ai4_mmv_xsmem()) {
            ai4k::launch_mmv<3, 3, false, LAY, XSH>(W, q, s, Y, M, M, N, K, stream);
        } else {
            switch (ai4_mmv_stages()) {
                case 3:  ai4k::launch_mmv<3, 6, true, LAY, XSH>(W, q, s, Y, M, M, N, K, stream); break;
                case 4:  ai4k::launch_mmv<4, 6, true, LAY, XSH>(W, q, s, Y, M, M, N, K, stream); break;
                default: ai4k::launch_mmv<2, 6, true, LAY, XSH>(W, q, s, Y, M, M, N, K, stream); break;
            }
        }
    } else if (N >= ai4_bn128_min()) {
        ai4k::launch_gemm<128, 3, 4, LAY, XSH>(W, q, s, Y, M, M, N, K, stream);
    } else {
        ai4k::launch_gemm<64, 3, 4, LAY, XSH>(W, q, s, Y, M, M, N, K, stream);
    }
}

// ai4-kern LAY for a type with IMMA kernels (AI4 / AI4_U / AI5_U / AI4_S / AI5_S / AI6_S); xsh = 1 unless the unit-128
// layout has K % 256 != 0. The span layouts need K % 1024 == 0, so they are always xsh = 1.
static int ai4_lay_of(ggml_type t) {
    const int lay = ggml_ai4_layout(t), rate = ggml_ai4_rate(t);
    if (lay == 2) {
        return rate == 6 ? ai4k::LAY_S6 : rate == 5 ? ai4k::LAY_S5 : ai4k::LAY_S;
    }
    if (lay == 0) {
        return ai4k::LAY_SB;
    }
    return rate == 5 ? ai4k::LAY_U5 : ai4k::LAY_U;
}

// W4A8 fast path: rotate+quantize the activations, then the AI4 decode-to-IMMA kernels
static void ai4_mul_mat_fast(ggml_backend_cuda_context & ctx, ggml_type t, const uint8_t * W, const float * X, float * Y,
                            int M, int N, int K) {
    cudaStream_t stream = ctx.stream();
    const int xsh = (!ggml_ai4_is_unit(t) || K % 256 == 0) ? 1 : 0;
    ggml_cuda_pool_alloc<int8_t> q(ctx.pool(), (size_t) N*K);
    ggml_cuda_pool_alloc<float>  s(ctx.pool(), (size_t) N*(K >> (7 + xsh)));
    ai4_quantize_x(X, N, K, xsh, q.get(), s.get(), stream);
    switch (ai4_lay_of(t)) {
        case ai4k::LAY_S:
            ai4_mm_kernels<ai4k::LAY_S,  1>(W, q.get(), s.get(), Y, M, N, K, stream);
            break;
        case ai4k::LAY_S5:
            ai4_mm_kernels<ai4k::LAY_S5, 1>(W, q.get(), s.get(), Y, M, N, K, stream);
            break;
        case ai4k::LAY_S6:
            ai4_mm_kernels<ai4k::LAY_S6, 1>(W, q.get(), s.get(), Y, M, N, K, stream);
            break;
        case ai4k::LAY_U5:
            if (xsh) {
                ai4_mm_kernels<ai4k::LAY_U5, 1>(W, q.get(), s.get(), Y, M, N, K, stream);
            } else {
                ai4_mm_kernels<ai4k::LAY_U5, 0>(W, q.get(), s.get(), Y, M, N, K, stream);
            }
            break;
        case ai4k::LAY_SB:
            ai4_mm_kernels<ai4k::LAY_SB, 1>(W, q.get(), s.get(), Y, M, N, K, stream);
            break;
        default:
            if (xsh) {
                ai4_mm_kernels<ai4k::LAY_U, 1>(W, q.get(), s.get(), Y, M, N, K, stream);
            } else {
                ai4_mm_kernels<ai4k::LAY_U, 0>(W, q.get(), s.get(), Y, M, N, K, stream);
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
// AI6_S (k = 6 span, the AI+ output head) has rate-6 kernels and takes the reference path only under GGML_CUDA_AI6_REF=1.
static bool ai4_ref_only_type(ggml_type t) {
    static const bool v = ai4_env_int("GGML_CUDA_AI5_REF", nullptr, 0) != 0;
    static const bool v6 = ai4_env_int("GGML_CUDA_AI6_REF", nullptr, 0) != 0;
    if (t == GGML_TYPE_AI5_U || t == GGML_TYPE_AI5_S) {
        return v;
    }
    if (t == GGML_TYPE_AI6_S) {
        return v6;
    }
    return t == GGML_TYPE_AI3_U || t == GGML_TYPE_AI2_U || t == GGML_TYPE_AI6_U ||
           t == GGML_TYPE_AI2_S || t == GGML_TYPE_AI3_S;
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

template <int LAY, int XSH>
static void ai4_mm_id_kernels(const uint8_t * W, const int8_t * q, const float * s, float * Y, int M, int K, int n_tok,
                              int n_expert, int64_t n_pairs, const ai4k::idmap & idm, int bn, int max_tiles, cudaStream_t stream) {
    if (n_tok <= 8) {
        // one mmv block row per token tile of <= 8 pairs (= per active expert unless ids repeat within a token)
        GGML_ASSERT(bn == ai4k::MMV_ID_BN);
        GGML_UNUSED(n_expert);
        GGML_UNUSED(n_pairs);
        ai4k::launch_mmv<2, 6, true, LAY, XSH, true>(W, q, s, Y, M, M, 0, K, stream, idm, max_tiles);
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
    const int xsh = (!unit || K % 256 == 0) ? 1 : 0;
    ggml_cuda_pool_alloc<int8_t> q(ctx.pool(), (size_t) n_rows*K);
    ggml_cuda_pool_alloc<float>  s(ctx.pool(), (size_t) n_rows*(K >> (7 + xsh)));
    ai4_quantize_x((const float *) src1->data, n_rows, K, xsh, q.get(), s.get(), stream);

    // mmv-ID (n_tok <= 8) takes tiles of MMV_ID_BN pairs, so an expert repeated within a token still fits
    const int bn = n_tok <= 8 ? ai4k::MMV_ID_BN : (n_tok >= ai4_bn128_min() ? 128 : 64);
    const int max_tiles = ai4k::ids_max_tiles(n_expert, n_pairs, bn);
    GGML_ASSERT(max_tiles <= 65535);
    ggml_cuda_pool_alloc<int> routing(ctx.pool(), (size_t) ai4k::ids_helper_ints(n_expert, n_pairs, max_tiles));
    const ai4k::idmap idm = ai4k::launch_ids_helper(ids->data, ids->nb[0], ids->nb[1], n_used, n_tok, ne11, n_expert, bn,
                                                    max_tiles, src0->nb[2], routing.get(), stream);
    const uint8_t * W = (const uint8_t *) src0->data;
    float * Y = (float *) dst->data;
    switch (ai4_lay_of(src0->type)) {
        case ai4k::LAY_S:
            ai4_mm_id_kernels<ai4k::LAY_S,  1>(W, q.get(), s.get(), Y, M, K, n_tok, n_expert, n_pairs, idm, bn, max_tiles, stream);
            break;
        case ai4k::LAY_S5:
            ai4_mm_id_kernels<ai4k::LAY_S5, 1>(W, q.get(), s.get(), Y, M, K, n_tok, n_expert, n_pairs, idm, bn, max_tiles, stream);
            break;
        case ai4k::LAY_S6:
            ai4_mm_id_kernels<ai4k::LAY_S6, 1>(W, q.get(), s.get(), Y, M, K, n_tok, n_expert, n_pairs, idm, bn, max_tiles, stream);
            break;
        case ai4k::LAY_U5:
            if (xsh) {
                ai4_mm_id_kernels<ai4k::LAY_U5, 1>(W, q.get(), s.get(), Y, M, K, n_tok, n_expert, n_pairs, idm, bn, max_tiles, stream);
            } else {
                ai4_mm_id_kernels<ai4k::LAY_U5, 0>(W, q.get(), s.get(), Y, M, K, n_tok, n_expert, n_pairs, idm, bn, max_tiles, stream);
            }
            break;
        case ai4k::LAY_SB:
            ai4_mm_id_kernels<ai4k::LAY_SB, 1>(W, q.get(), s.get(), Y, M, K, n_tok, n_expert, n_pairs, idm, bn, max_tiles, stream);
            break;
        default:
            if (xsh) {
                ai4_mm_id_kernels<ai4k::LAY_U, 1>(W, q.get(), s.get(), Y, M, K, n_tok, n_expert, n_pairs, idm, bn, max_tiles, stream);
            } else {
                ai4_mm_id_kernels<ai4k::LAY_U, 0>(W, q.get(), s.get(), Y, M, K, n_tok, n_expert, n_pairs, idm, bn, max_tiles, stream);
            }
            break;
    }
    CUDA_CHECK(cudaGetLastError());
}
