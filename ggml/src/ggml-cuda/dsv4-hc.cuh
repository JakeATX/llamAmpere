#include "common.cuh"
#include "ggml.h"

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
// [HCPC] out != nullptr: the result goes to out (same shape, f32), dst is not written (GGML_CUDA_HC_POST_CPY)
void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * out = nullptr);

// [HCPN] GGML_CUDA_HC_POST_NORM / GGML_CUDA_HC_PRE_XN (opt-in, matched in ggml-cuda.cu).
// One pass over each row of the next hyper-connection norm: post (nullable) is an identity-comb DSV4_HC_POST whose
// output is norm's input, norm/mul are RMS_NORM -> MUL(gamma), dst_bf16 (nullable) receives the BF16 copy of the norm
// output. With row_scale == nullptr the f32 norm output is written to mul; otherwise (row_scale == mul->data) the
// scale of each norm row is written to the first float of that row instead, for ggml_cuda_op_dsv4_hc_pre_xn.
void ggml_cuda_op_dsv4_hc_post_norm(ggml_backend_cuda_context & ctx, const ggml_tensor * post, const ggml_tensor * norm,
                                    const ggml_tensor * mul, nv_bfloat16 * dst_bf16, float * row_scale);
// The gated DSV4_HC_PRE hc_pre, with its xn input rebuilt as (row_scale * x) * gamma from the norm input x.
void ggml_cuda_op_dsv4_hc_pre_xn(ggml_backend_cuda_context & ctx, ggml_tensor * hc_pre, const ggml_tensor * x,
                                 const ggml_tensor * gamma, const float * row_scale);
