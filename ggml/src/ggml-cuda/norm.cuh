#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// dst_override: write the result there instead of mul_tensor->data (same layout as mul_tensor)
void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor,
                                 float * dst_override = nullptr);

void ggml_cuda_op_rms_norm_scale_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * scale_tensor);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor);

// [#46] ADD -> RMS_NORM -> MUL(gamma) in one kernel (add keeps its output, the residual). With q8_dst != nullptr it also
// writes the MUL output as q8_1 in the MMVQ activation layout for weights of q8_type (quantize_row_q8_1_cuda's layout).
void ggml_cuda_op_add_rms_norm_mul(ggml_backend_cuda_context & ctx,
                                   ggml_tensor *               add,
                                   ggml_tensor *               rms_norm,
                                   ggml_tensor *               mul,
                                   void *                      q8_dst,
                                   ggml_type                   q8_type);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
