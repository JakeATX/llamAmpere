#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

// Optional decode-only conv/SiLU/QK normalization, using the existing history plan.
bool ggml_cuda_op_ssm_conv_norm(ggml_backend_cuda_context & ctx, ggml_tensor * conv,
        ggml_tensor * mixed, ggml_tensor * q, ggml_tensor * k, float eps);
