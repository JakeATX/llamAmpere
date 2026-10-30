#include "common.cuh"

void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst);

// dst = sum_k w_k * experts_k + shexp * sigmoid(gate_logit), bit-identical to the reduction followed by the
// separate sigmoid, mul and add kernels
void ggml_cuda_op_moe_weighted_combine(ggml_backend_cuda_context & ctx,
                                       const ggml_tensor *         experts,
                                       const ggml_tensor *         expert_scale,
                                       const ggml_tensor *         weights,
                                       const ggml_tensor *         gate_logit,
                                       const ggml_tensor *         shexp,
                                       ggml_tensor *               dst);

void ggml_cuda_op_moe_bank_joint_reduction(ggml_backend_cuda_context & ctx,
        const ggml_tensor * hot, const ggml_tensor * cold,
        const ggml_tensor * hot_weights, const ggml_tensor * cold_weights, ggml_tensor * dst);
