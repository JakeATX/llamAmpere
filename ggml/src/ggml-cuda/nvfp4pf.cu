// INKNVFP4K NVFP4PF: NVFP4 MoE prefill launches (kernels in nvfp4pf-kern.cuh), called from the fused scale-MMQ site
#include "nvfp4pf.cuh"

#include <atomic>
#include <climits>
#include <cstdlib>
#include <cstring>

int ggml_cuda_nvfp4pf_mode() {
    static const int mode = [] {
        const char * e = getenv("GGML_CUDA_NVFP4PF");
        if (!e) {
            return 0;
        }
        if (!strcmp(e, "k64") || !strcmp(e, "1")) {
            return 64;
        }
        return !strcmp(e, "k32") ? 32 : 0;
    }();
    return mode;
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include "nvfp4pf-kern.cuh"

template <int KF, int BN>
static void nvfp4pf_launch_mm(const char * w, int64_t stride_blk, const int8_t * xq, const float * xs, float * y, int M, int K,
        const nvfp4pf_map & map, int max_tiles, const float * s_exp, cudaStream_t stream) {
    constexpr int smem = nvfp4pf_smem_bytes<KF, BN>();
    static_assert(smem + 2*BN*sizeof(int) <= 48*1024, "NVFP4PF: shared memory (+ static srow / scol) above the default limit");
    k_nvfp4pf_mm<KF, BN><<<dim3((M + 63)/64, max_tiles), 256, smem, stream>>>(w, stride_blk, xq, xs, y, M, M, K, map, s_exp, 1.0f);
}

bool ggml_cuda_nvfp4pf_try(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        const ggml_tensor * ids, ggml_tensor * dst, const ggml_tensor * s_exp) {
    const int KF = ggml_cuda_nvfp4pf_mode();
    if (KF == 0 || !ids) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || (cc != GGML_CUDA_CC_AMPERE && cc != 860)) {
        return false;
    }
    const int64_t K = src0->ne[0], M = src0->ne[1], n_expert = src0->ne[2];
    const int64_t ne11 = src1->ne[1], n_used = ids->ne[0], n_tok = ids->ne[1];
    if (!src0->data || !src1->data || !dst->data || !ids->data || K <= 0 || M <= 0 || dst->ne[0] != M || dst->ne[1] != n_used ||
            dst->ne[2] != n_tok || src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float)) {
        return false;
    }
    if (src0->type != GGML_TYPE_NVFP4 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 ||
            K % 128 != 0 || M % 16 != 0 || K > INT_MAX || M > INT_MAX || n_expert <= 0 || n_expert > NVFP4PF_MAX_EXPERTS ||
            n_tok < 512 || n_tok > INT_MAX || src0->ne[3] != 1 || src1->ne[2] != n_tok || src1->ne[3] != 1 || ids->ne[2] != 1 ||
            ids->ne[3] != 1) {
        return false;
    }
    if (src0->nb[0] != 36 || src0->nb[1] % 36 != 0 || src0->nb[1] < (size_t) (K/64)*36 || src0->nb[2] % 4 != 0 ||
            (uintptr_t) src0->data % 4 != 0 || (uintptr_t) src1->data % 16 != 0 ||
            src1->nb[1] != (size_t) K*sizeof(float) || src1->nb[2] != (size_t) ne11*src1->nb[1] ||
            dst->nb[1] != (size_t) M*sizeof(float) || dst->nb[2] != (size_t) n_used*dst->nb[1]) {
        return false;
    }
    if (s_exp && (s_exp->type != GGML_TYPE_F32 || !ggml_is_contiguous(s_exp) || ggml_nelements(s_exp) != n_expert)) {
        return false;
    }
    const int64_t n_rows = ne11*n_tok, n_pairs = n_used*n_tok;
    const int bn = n_pairs < 64*n_expert ? 64 : 128;
    const int max_tiles = nvfp4pf_max_tiles((int) n_expert, n_pairs, bn);
    if (n_rows > 65535 || n_pairs > INT_MAX || max_tiles > 65535) {  // n_rows is the x quantize grid's y
        return false;
    }

    static std::atomic<int> logged{0};
    static const bool log_all = [] { const char * e = getenv("GGML_CUDA_NVFP4PF_LOG"); return e && atoi(e) != 0; }();
    if (log_all || logged.exchange(1) == 0) {
        GGML_LOG_WARN("NVFP4PF: s8 k%d MoE prefill: %s K %lld M %lld experts %lld tokens %lld (bn %d)\n", KF, dst->name,
            (long long) K, (long long) M, (long long) n_expert, (long long) n_tok, bn);
    }

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<int8_t> xq(ctx.pool(), (size_t) n_rows*K);
    ggml_cuda_pool_alloc<float>  xs(ctx.pool(), (size_t) n_rows*(K/KF));
    ggml_cuda_pool_alloc<int>    work(ctx.pool(), nvfp4pf_work_ints((int) n_expert, n_pairs, max_tiles));
    float * y = (float *) dst->data;
    const int64_t stride_blk = src0->nb[1]/36;
    const char * w = (const char *) src0->data;
    const float * se = s_exp ? (const float *) s_exp->data : nullptr;

    k_nvfp4pf_route<<<1, 256, 0, stream>>>((const char *) ids->data, ids->nb[0], ids->nb[1], (int) n_used, (int) n_tok, (int) ne11,
        (int) n_expert, bn, max_tiles, work.get(), y, (int) M);
    CUDA_CHECK(cudaGetLastError());
    const nvfp4pf_map map = nvfp4pf_map_of(work.get(), (int) n_expert, n_pairs, src0->nb[2]);
    const dim3 gq((unsigned) ((K/KF + 3)/4), (unsigned) n_rows);
    if (KF == 64) {
        k_nvfp4pf_xq<64><<<gq, 128, 0, stream>>>((const float *) src1->data, K, xq.get(), xs.get(), (int) K, (int) n_rows);
        if (bn == 64) {
            nvfp4pf_launch_mm<64, 64>(w, stride_blk, xq.get(), xs.get(), y, (int) M, (int) K, map, max_tiles, se, stream);
        } else {
            nvfp4pf_launch_mm<64, 128>(w, stride_blk, xq.get(), xs.get(), y, (int) M, (int) K, map, max_tiles, se, stream);
        }
    } else {
        k_nvfp4pf_xq<32><<<gq, 128, 0, stream>>>((const float *) src1->data, K, xq.get(), xs.get(), (int) K, (int) n_rows);
        if (bn == 64) {
            nvfp4pf_launch_mm<32, 64>(w, stride_blk, xq.get(), xs.get(), y, (int) M, (int) K, map, max_tiles, se, stream);
        } else {
            nvfp4pf_launch_mm<32, 128>(w, stride_blk, xq.get(), xs.get(), y, (int) M, (int) K, map, max_tiles, se, stream);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}
#else
bool ggml_cuda_nvfp4pf_try(ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *, const ggml_tensor *, ggml_tensor *,
        const ggml_tensor *) {
    return false;
}
#endif
