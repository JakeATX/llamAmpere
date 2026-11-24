#pragma once

#include "common.cuh"
#include "inkglue.cuh"
#include "prefill-glue-math.h"

// Default-OFF independent kill switches for Prefill Glue & BF16 MLP Activation Narrowing.
// When all three remain unset or 0, runtime behavior is 100% identical to rc/v0.7.4-cand @ f868fb9cf.
bool ggml_cuda_prefill_fuse_glu_actq_on();   // GGML_CUDA_PREFILL_FUSE_GLU_ACTQ=1 (default 0)
bool ggml_cuda_prefill_fuse_norm_actq_on();  // GGML_CUDA_PREFILL_FUSE_NORM_ACTQ=1 (default 0)
bool ggml_cuda_prefill_bf16_act_on();        // GGML_CUDA_PREFILL_BF16_ACT=1 (default 0)
bool ggml_cuda_prefill_any_on();

// Context lifecycle / buffer invalidation
void ggml_cuda_prefill_glue_free(ggml_backend_cuda_context * ctx);
void ggml_cuda_prefill_glue_reset(ggml_backend_cuda_context * ctx);
void ggml_cuda_prefill_invalidate_ptr(ggml_backend_cuda_context & ctx, const void * ptr, size_t nbytes);

// Persistent prefill activation quantize cache (AI+, NVFP4PF, MMQ)
bool ggml_cuda_prefill_ai4_alloc(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, int64_t n_rows, int K,
        int xsh, int variant, int rem_uses, int8_t * & q, float * & s);
bool ggml_cuda_prefill_ai4_take(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, int64_t n_rows, int K,
        int xsh, int variant, const int8_t * & q, const float * & s);

bool ggml_cuda_prefill_nvfp4_alloc(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, int64_t n_rows, int K,
        int kf, int rem_uses, int8_t * & xq, float * & xs);
bool ggml_cuda_prefill_nvfp4_take(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, int64_t n_rows, int K,
        int kf, const int8_t * & xq, const float * & xs);

bool ggml_cuda_prefill_mmq_alloc(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, int64_t n_rows, int K,
        int64_t k_padded, int ds_layout, bool swizzle_iq4, int rem_uses, char * & q8);
bool ggml_cuda_prefill_mmq_take(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, int64_t n_rows, int K,
        int64_t k_padded, int ds_layout, bool swizzle_iq4, const char * & q8);

// Deferred MMQ_ID GLU/SITU consumer (quantizes directly using ids_src1 after mm_ids_helper)
bool ggml_cuda_prefill_glu_mmq_id_take(ggml_backend_cuda_context & ctx, const ggml_tensor * src1,
        const int32_t * ids_src1, void * vy, ggml_type w_type, int64_t ne10, int64_t ne10_padded,
        int64_t ne11_flat, int64_t ne12_flat, int64_t ne13_flat, cudaStream_t stream);

// BF16 transient MLP activation buffer tracker & converters (GGML_CUDA_PREFILL_BF16_ACT=1)
bool ggml_cuda_prefill_bf16_want_out(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx);
void ggml_cuda_prefill_bf16_mark(ggml_backend_cuda_context & ctx, const ggml_tensor * dst);
bool ggml_cuda_prefill_bf16_is_active(const ggml_backend_cuda_context & ctx, const ggml_tensor * src, int64_t * elem_off_out = nullptr);

GGML_BACKEND_API void ggml_cuda_prefill_bf16_to_ai4(const nv_bfloat16 * x, int64_t rows, int K, int xsh, int variant,
        int8_t * q, float * s, cudaStream_t stream);
GGML_BACKEND_API void ggml_cuda_prefill_bf16_to_nvfp4pf(const nv_bfloat16 * x, int64_t rows, int K, int kf,
        int8_t * xq, float * xs, cudaStream_t stream);
GGML_BACKEND_API void ggml_cuda_prefill_bf16_to_mmq(const nv_bfloat16 * x, const int32_t * ids, void * vy, ggml_type w_type,
        int64_t ne00, int64_t s01, int64_t s02, int64_t s03, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
        bool scatter, int n_expert_used, bool swizzle_iq4, cudaStream_t stream);
GGML_BACKEND_API void ggml_cuda_prefill_bf16_to_f32(const nv_bfloat16 * src, float * dst, size_t n, cudaStream_t stream);

// Graph-level matchers & launchers called from ggml-cuda.cu
int  ggml_cuda_ink_situ_spans(ggml_cgraph * cgraph, int start_idx, const ggml_tensor * src_tensor);
bool ggml_cuda_prefill_situ_launch(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int last,
        const ggml_tensor * g, const ggml_tensor * u, ggml_tensor * out, const ink_situ_params & p);
int  ggml_cuda_try_fuse_prefill_glu(ggml_backend_cuda_context * ctx, ggml_cgraph * cgraph, int i);
int  ggml_cuda_try_fuse_prefill_norm(ggml_backend_cuda_context * ctx, ggml_cgraph * cgraph, int i);
bool ggml_cuda_prefill_try_mix_norm_actq(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int last,
        ggml_tensor * mix, const ggml_tensor * w, ggml_tensor * mul, float eps, int & out_kind, void ** out0, void ** out1, bool & write_f32, const ggml_tensor ** out_mm);
void ggml_cuda_prefill_mix_norm_commit(ggml_backend_cuda_context & ctx, const ggml_tensor * mul, int64_t n_rows, int K, int kind, int param, void * p0, void * p1);

// Direct kernel launch wrappers exposed for bench-prefill-glue
GGML_BACKEND_API void ggml_cuda_prefill_launch_glu_ai4(
        int glu_kind, const void * g, const void * u, void * out,
        int8_t * q, float * s, int K, int64_t rows, int64_t ne1,
        int64_t g_nb1, int64_t g_nb2, int64_t u_nb1, int64_t u_nb2,
        bool src_bf16, int write_mode, int xsh, int variant,
        const ink_situ_params & situ, float glu_limit, float oai_alpha, float oai_limit, cudaStream_t stream);

GGML_BACKEND_API void ggml_cuda_prefill_launch_glu_nvfp4pf(
        int glu_kind, const void * g, const void * u, void * out,
        int8_t * xq, float * xs, int K, int64_t rows, int64_t ne1,
        int64_t g_nb1, int64_t g_nb2, int64_t u_nb1, int64_t u_nb2,
        bool src_bf16, int write_mode, int kf,
        const ink_situ_params & situ, float glu_limit, float oai_alpha, float oai_limit, cudaStream_t stream);

GGML_BACKEND_API void ggml_cuda_prefill_launch_glu_mmq(
        int glu_kind, const void * g, const void * u, void * out,
        const int32_t * ids, void * vy, int ds_layout, bool swizzle_iq4,
        int K, int K_padded, int64_t rows_dst, int64_t ne1,
        int64_t g_nb1, int64_t g_nb2, int64_t u_nb1, int64_t u_nb2,
        bool src_bf16, int write_mode,
        const ink_situ_params & situ, float glu_limit, float oai_alpha, float oai_limit, cudaStream_t stream);

GGML_BACKEND_API void ggml_cuda_prefill_launch_norm_ai4(
        const float * x, const float * r, float * add_dst, const float * w, int w_ne1, float * mul_dst,
        int8_t * q, float * s, int64_t n_rows, int K, int64_t x_s1, int64_t r_s1, float eps,
        bool write_f32, int xsh, int variant, cudaStream_t stream);

GGML_BACKEND_API void ggml_cuda_prefill_launch_norm_nvfp4pf(
        const float * x, const float * r, float * add_dst, const float * w, int w_ne1, float * mul_dst,
        int8_t * xq, float * xs, int64_t n_rows, int K, int64_t x_s1, int64_t r_s1, float eps,
        bool write_f32, int kf, cudaStream_t stream);

GGML_BACKEND_API void ggml_cuda_prefill_launch_norm_mmq(
        const float * x, const float * r, float * add_dst, const float * w, int w_ne1, float * mul_dst,
        void * vy, int ds_layout, bool swizzle_iq4, int64_t n_rows, int K, int K_padded, int64_t x_s1, int64_t r_s1, float eps,
        bool write_f32, cudaStream_t stream);
