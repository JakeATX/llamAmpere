#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

// Optional decode-only conv/SiLU/QK normalization, using the existing history plan.
bool ggml_cuda_op_ssm_conv_norm(ggml_backend_cuda_context & ctx, ggml_tensor * conv,
        ggml_tensor * mixed, ggml_tensor * q, ggml_tensor * k, float eps);

// [INKGLUE] planned conv ring + SCALE(ADD(x, conv)) epilogue into out; res: then ADD(res, that) (the residual)
bool ggml_cuda_op_ssm_conv_ring_epi(ggml_backend_cuda_context & ctx, ggml_tensor * conv, const ggml_tensor * x,
        ggml_tensor * out, float scale, float bias, const ggml_tensor * res = nullptr);
