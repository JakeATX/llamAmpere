#pragma once

#include "common.cuh"
#include "inkglue.cuh"
#include "prefill-glue-math.h"

// [PREFILL-GLUE] Default-off switches. With both unset (or 0) no hook below changes anything: dispatch and numerics are
// those of rc/v0.7.4-small @ a0360f2e3. A fused producer always writes its normal f32 output; the quantized copy it
// adds is a cache a consumer matmul takes only on an exact match.
bool ggml_cuda_prefill_fuse_glu_actq_on();   // GGML_CUDA_PREFILL_FUSE_GLU_ACTQ=1 (default 0)
bool ggml_cuda_prefill_fuse_norm_actq_on();  // GGML_CUDA_PREFILL_FUSE_NORM_ACTQ=1 (default 0)
bool ggml_cuda_prefill_any_on();

// Context lifecycle / invalidation. invalidate drops every entry whose source memory overlaps [ptr, ptr + nbytes), except
// entries created after tick keep_after (the output of the fused launch that just ran).
void ggml_cuda_prefill_glue_free(ggml_backend_cuda_context * ctx);
void ggml_cuda_prefill_glue_reset(ggml_backend_cuda_context * ctx);
void ggml_cuda_prefill_invalidate_ptr(ggml_backend_cuda_context & ctx, const void * ptr, size_t nbytes,
        uint64_t keep_after = UINT64_MAX);

// Cached quantized activations (AI+, NVFP4PF, MMQ q8_1). take: true only when src1 (or a contiguous same-data view of
// it), its data pointer, the row count, K and the layout all match an entry of this graph eval.
bool ggml_cuda_prefill_ai4_take(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, int64_t n_rows, int K,
        int xsh, int variant, const int8_t * & q, const float * & s);
bool ggml_cuda_prefill_nvfp4_take(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, int64_t n_rows, int K,
        int kf, const int8_t * & xq, const float * & xs);
bool ggml_cuda_prefill_mmq_take(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, int64_t n_rows, int K,
        int64_t k_padded, int ds_layout, bool swizzle_iq4, const char * & q8);

// Graph-level matchers & launchers called from ggml-cuda.cu
bool ggml_cuda_prefill_situ_launch(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int last,
        const ggml_tensor * g, const ggml_tensor * u, ggml_tensor * out, const ink_situ_params & p);
bool ggml_cuda_try_fuse_prefill_glu(ggml_backend_cuda_context * ctx, ggml_cgraph * cgraph, int i);
int  ggml_cuda_try_fuse_prefill_norm(ggml_backend_cuda_context * ctx, ggml_cgraph * cgraph, int i);
// depth mix + norm: true = an entry for mul was created; quantize into q / s (kf = 0: AI+ H256, 32 / 64: NVFP4PF)
bool ggml_cuda_prefill_try_mix_norm_actq(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int last,
        const ggml_tensor * mul, int & kf, int8_t ** q, float ** s);

// Direct kernel launchers (also used by bench-prefill-glue). All write the f32 output and the quantized activation.
GGML_BACKEND_API void ggml_cuda_prefill_launch_glu_ai4(
        int glu_kind, const void * g, const void * u, float * out,
        int8_t * q, float * s, int K, int64_t rows, int64_t ne1,
        int64_t g_nb1, int64_t g_nb2, int64_t u_nb1, int64_t u_nb2, int xsh, int variant,
        const ink_situ_params & situ, float glu_limit, float oai_alpha, float oai_limit, cudaStream_t stream);

GGML_BACKEND_API void ggml_cuda_prefill_launch_glu_nvfp4pf(
        int glu_kind, const void * g, const void * u, float * out,
        int8_t * xq, float * xs, int K, int64_t rows, int64_t ne1,
        int64_t g_nb1, int64_t g_nb2, int64_t u_nb1, int64_t u_nb2, int kf,
        const ink_situ_params & situ, float glu_limit, float oai_alpha, float oai_limit, cudaStream_t stream);

GGML_BACKEND_API void ggml_cuda_prefill_launch_glu_mmq(
        int glu_kind, const void * g, const void * u, float * out,
        void * vy, int ds_layout, bool swizzle_iq4,
        int K, int K_padded, int64_t rows, int64_t ne1,
        int64_t g_nb1, int64_t g_nb2, int64_t u_nb1, int64_t u_nb2,
        const ink_situ_params & situ, float glu_limit, float oai_alpha, float oai_limit, cudaStream_t stream);

// x / r rows are contiguous (row stride K); add_dst (with r) and mul_dst are always written
GGML_BACKEND_API void ggml_cuda_prefill_launch_norm_ai4(
        const float * x, const float * r, float * add_dst, const float * w, float * mul_dst,
        int8_t * q, float * s, int64_t n_rows, int K, float eps, int xsh, int variant, cudaStream_t stream);

GGML_BACKEND_API void ggml_cuda_prefill_launch_norm_nvfp4pf(
        const float * x, const float * r, float * add_dst, const float * w, float * mul_dst,
        int8_t * xq, float * xs, int64_t n_rows, int K, float eps, int kf, cudaStream_t stream);

GGML_BACKEND_API void ggml_cuda_prefill_launch_norm_mmq(
        const float * x, const float * r, float * add_dst, const float * w, float * mul_dst,
        void * vy, int ds_layout, bool swizzle_iq4, int64_t n_rows, int K, int K_padded, float eps, cudaStream_t stream);
