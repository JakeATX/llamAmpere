// [GFFN-P9] NVFP4 MoE prefill gate_up GEMM with the SITU epilogue (see gffn-p9.cuh; kernels in nvfp4pf-kern.cuh)
#include "gffn-p9.cuh"
#include "nvfp4pf.cuh"
#include "prefill-glue.cuh"

#include <atomic>
#include <climits>
#include <cstdlib>
#include <cstring>

static std::atomic<int64_t> g_gffn_p9_count{0};

bool ggml_cuda_gffn_p9_on() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_GFFN_P9");
        return e && *e && atoi(e) != 0;
    }();
    return on;
}

int64_t ggml_cuda_gffn_p9_count() {
    return g_gffn_p9_count.load();
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include "nk-pf-mm.cuh"
#include "nvfp4pf-kern.cuh"

bool ggml_cuda_gffn_p9_supported(int cc, const ggml_tensor * gu, const ggml_tensor * s_gu, const ggml_tensor * act,
        const ink_situ_params & p) {
    // the unfused run must be the NVFP4PF GEMM (ggml_cuda_nvfp4pf_try_mode) for the bytes to match; the launch takes
    // the kernel it would (NK_PF_MM when that is on, else the base kernel), each with the SITU epilogue
    if (ggml_cuda_nvfp4pf_mode() == 0 || !GGML_CUDA_CC_IS_NVIDIA(cc) || (cc != GGML_CUDA_CC_AMPERE && cc != 860)) {
        return false;
    }
    const ggml_tensor * src0 = gu->src[0];
    const ggml_tensor * src1 = gu->src[1];
    const ggml_tensor * ids  = gu->src[2];
    if (!src0 || !src1 || !ids || !s_gu || !act) {
        return false;
    }
    const int64_t K = src0->ne[0], M = src0->ne[1], n_expert = src0->ne[2], F = M/2;
    const int64_t ne11 = src1->ne[1], n_used = ids->ne[0], n_tok = ids->ne[1];
    // ggml_cuda_nvfp4pf_try_mode's checks on (src0, src1, ids, dst = gu)
    if (K <= 0 || M <= 0 || gu->ne[0] != M || gu->ne[1] != n_used || gu->ne[2] != n_tok ||
            src1->nb[0] != sizeof(float) || gu->nb[0] != sizeof(float)) {
        return false;
    }
    if (src0->type != GGML_TYPE_NVFP4 || src1->type != GGML_TYPE_F32 || gu->type != GGML_TYPE_F32 ||
            ids->type != GGML_TYPE_I32 || K % 128 != 0 || M % 16 != 0 || K > INT_MAX || M > INT_MAX || n_expert <= 0 ||
            n_expert > NVFP4PF_MAX_EXPERTS || n_tok < 512 || n_tok > INT_MAX || src0->ne[3] != 1 ||
            src1->ne[2] != n_tok || src1->ne[3] != 1 || ids->ne[2] != 1 || ids->ne[3] != 1) {
        return false;
    }
    if (src0->nb[0] != 36 || src0->nb[1] % 36 != 0 || src0->nb[1] < (size_t) (K/64)*36 || src0->nb[2] % 4 != 0 ||
            src1->nb[1] != (size_t) K*sizeof(float) || src1->nb[2] != (size_t) ne11*src1->nb[1] ||
            gu->nb[1] != (size_t) M*sizeof(float) || gu->nb[2] != (size_t) n_used*gu->nb[1]) {
        return false;
    }
    if (s_gu->type != GGML_TYPE_F32 || !ggml_is_contiguous(s_gu) || ggml_nelements(s_gu) != n_expert) {
        return false;
    }
    const int64_t n_rows = ne11*n_tok, n_pairs = n_used*n_tok;
    const int bn = n_pairs < 64*n_expert ? 64 : 128;
    if (n_rows > 65535 || n_pairs > INT_MAX || nvfp4pf_max_tiles((int) n_expert, n_pairs, bn) > 65535) {
        return false;
    }
    // the SITU output: contiguous f32 [F, n_used, T]
    if (act->type != GGML_TYPE_F32 || !ggml_is_contiguous(act) || act->ne[0] != F || act->ne[1] != n_used ||
            act->ne[2] != n_tok || act->ne[3] != 1) {
        return false;
    }
    // the route kernel zero-fills the columns of invalid ids as the unfused GEMM does; SITU(+0, +0) is +0 only with the
    // SCALE biases at +0 (bit pattern 0: a -0 bias keeps -0 through a negative scale; Inkling uses plain ggml_scale)
    const float b[4] = { p.ga_b, p.gb_b, p.ua_b, p.ub_b };
    uint32_t bits[4];
    memcpy(bits, b, sizeof(bits));
    return (bits[0] | bits[1] | bits[2] | bits[3]) == 0;
}

template <int KF, int BN>
static void gffn_p9_launch_mm(const char * w, int64_t stride_blk, const int8_t * xq, const float * xs, float * act,
        int M, int K, const nvfp4pf_map & map, int max_tiles, const float * s_exp, const ink_situ_params & p,
        cudaStream_t stream) {
    constexpr int smem = nvfp4pf_smem_bytes<KF, BN>();
    static_assert(smem + 2*BN*sizeof(int) <= 48*1024, "GFFN-P9: shared memory (+ static srow / scol) above the limit");
    k_nvfp4pf_mm<KF, BN, true><<<dim3((M + 63)/64, max_tiles), 256, smem, stream>>>(w, stride_blk, xq, xs, act, M/2, M,
        K, map, s_exp, 1.0f, p);
}

bool ggml_cuda_gffn_p9_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * gu, const ggml_tensor * s_gu,
        ggml_tensor * act, const ink_situ_params & p) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!ggml_cuda_gffn_p9_supported(cc, gu, s_gu, act, p)) {
        return false;
    }
    const ggml_tensor * src0 = gu->src[0];
    const ggml_tensor * src1 = gu->src[1];
    const ggml_tensor * ids  = gu->src[2];
    if (!src0->data || !src1->data || !ids->data || !s_gu->data || !act->data || (uintptr_t) src0->data % 4 != 0 ||
            (uintptr_t) src1->data % 16 != 0) {
        return false;
    }
    const int KF = ggml_cuda_nvfp4pf_mode();
    const int64_t K = src0->ne[0], M = src0->ne[1], n_expert = src0->ne[2];
    const int64_t ne11 = src1->ne[1], n_used = ids->ne[0], n_tok = ids->ne[1];
    const int64_t n_rows = ne11*n_tok, n_pairs = n_used*n_tok;
    const int bn = n_pairs < 64*n_expert ? 64 : 128;
    const int max_tiles = nvfp4pf_max_tiles((int) n_expert, n_pairs, bn);

    const bool use_nk = ggml_cuda_nk_pf_mm_enabled();   // ggml_cuda_nvfp4pf_try's kernel choice
    static std::atomic<int> logged{0};
    if (logged.exchange(1) == 0) {
        GGML_LOG_INFO("GFFN-P9: NVFP4PF k%d %s gate_up + SITU in one GEMM: %s K %lld F %lld experts %lld tokens %lld "
            "(bn %d)\n", KF, use_nk ? "NK" : "base", act->name, (long long) K, (long long) (M/2), (long long) n_expert,
            (long long) n_tok, bn);
    }

    cudaStream_t stream = ctx.stream();
    // the RMS_NORM producer's cached int8 x, as ggml_cuda_nvfp4pf_try_mode takes it
    const int8_t * pre_xq = nullptr;
    const float  * pre_xs = nullptr;
    const bool pre = ggml_cuda_prefill_nvfp4_take(ctx, src1, n_rows, (int) K, KF, pre_xq, pre_xs);
    ggml_cuda_pool_alloc<int8_t> xq(ctx.pool());
    ggml_cuda_pool_alloc<float>  xs(ctx.pool());
    if (!pre) {
        xq.alloc((size_t) n_rows*K);
        xs.alloc((size_t) n_rows*(K/KF));
    }
    const int8_t * xq_use = pre ? pre_xq : xq.get();
    const float  * xs_use = pre ? pre_xs : xs.get();
    ggml_cuda_pool_alloc<int> work(ctx.pool(), nvfp4pf_work_ints((int) n_expert, n_pairs, max_tiles));
    float * y = (float *) act->data;
    const int64_t stride_blk = src0->nb[1]/36;
    const char * w = (const char *) src0->data;
    const float * se = (const float *) s_gu->data;

    // invalid-id columns: F zeros (= SITU of the unfused run's zero gate / up rows, see the bias check)
    k_nvfp4pf_route<<<1, 256, 0, stream>>>((const char *) ids->data, ids->nb[0], ids->nb[1], (int) n_used, (int) n_tok,
        (int) ne11, (int) n_expert, bn, max_tiles, work.get(), y, (int) (M/2));
    CUDA_CHECK(cudaGetLastError());
    const nvfp4pf_map map = nvfp4pf_map_of(work.get(), (int) n_expert, n_pairs, src0->nb[2]);
    if (!pre) {
        const dim3 gq((unsigned) ((K/KF + 3)/4), (unsigned) n_rows);
        if (KF == 64) {
            k_nvfp4pf_xq<64><<<gq, 128, 0, stream>>>((const float *) src1->data, K, xq.get(), xs.get(), (int) K,
                (int) n_rows);
        } else {
            k_nvfp4pf_xq<32><<<gq, 128, 0, stream>>>((const float *) src1->data, K, xq.get(), xs.get(), (int) K,
                (int) n_rows);
        }
    }
    if (use_nk) {
        const nk_pf_map nk_map = { map.hdr, map.exps, map.off, map.rows, map.cols, map.tiles, map.nb02 };
        ggml_cuda_nk_pf_mm_launch_situ(KF, bn, w, stride_blk, xq_use, xs_use, y, (int) M, (int) K, nk_map, max_tiles,
            se, p, ctx.device, stream);
    } else if (KF == 64) {
        if (bn == 64) {
            gffn_p9_launch_mm<64, 64>(w, stride_blk, xq_use, xs_use, y, (int) M, (int) K, map, max_tiles, se, p,
                stream);
        } else {
            gffn_p9_launch_mm<64, 128>(w, stride_blk, xq_use, xs_use, y, (int) M, (int) K, map, max_tiles, se, p,
                stream);
        }
    } else {
        if (bn == 64) {
            gffn_p9_launch_mm<32, 64>(w, stride_blk, xq_use, xs_use, y, (int) M, (int) K, map, max_tiles, se, p,
                stream);
        } else {
            gffn_p9_launch_mm<32, 128>(w, stride_blk, xq_use, xs_use, y, (int) M, (int) K, map, max_tiles, se, p,
                stream);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    g_gffn_p9_count.fetch_add(1);
    return true;
}
#else
bool ggml_cuda_gffn_p9_supported(int, const ggml_tensor *, const ggml_tensor *, const ggml_tensor *,
        const ink_situ_params &) {
    return false;
}
bool ggml_cuda_gffn_p9_launch(ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *, ggml_tensor *,
        const ink_situ_params &) {
    return false;
}
#endif
