#pragma once

#include "common.cuh"

// [HCFUSED] Flash-Next hyper-connection mix as two ops (see ggml_dsv4_hc_down / ggml_dsv4_hc_up in ggml.h).
// Small batches only (1..8 tokens); n_embd a multiple of 512, R a multiple of 64, hc == 4, BF16 weights.
bool ggml_cuda_dsv4_hc_down_supported(const ggml_tensor * op);
bool ggml_cuda_dsv4_hc_up_supported(const ggml_tensor * op);

void ggml_cuda_op_dsv4_hc_down(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_up(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// [HCPD] the identity-comb DSV4_HC_POST that produces the HC_DOWN input, computed inside the down kernel (which also
// writes the hc_post output); bit-identical to running the two ops.
bool ggml_cuda_dsv4_hc_post_down_supported(const ggml_tensor * hpost, const ggml_tensor * down);
void ggml_cuda_op_dsv4_hc_post_down(ggml_backend_cuda_context & ctx, const ggml_tensor * hpost, ggml_tensor * down);

// [GDNGATES] gate = softplus(W_alpha y + dt) * a and beta = sigmoid(W_beta y) in one kernel, bit-identical to the
// unfused mul_mat_vec_f + add + softplus + mul and mul_mat_vec_f + sigmoid kernels.
bool ggml_cuda_gdn_gates_supported(const ggml_tensor * w_alpha, const ggml_tensor * w_beta, const ggml_tensor * y,
        const ggml_tensor * dt, const ggml_tensor * a, const ggml_tensor * gate, const ggml_tensor * beta);
void ggml_cuda_gdn_gates(ggml_backend_cuda_context & ctx, const ggml_tensor * w_alpha, const ggml_tensor * w_beta,
        const ggml_tensor * y, const ggml_tensor * dt, const ggml_tensor * a, ggml_tensor * gate, ggml_tensor * beta);

// [SHEXPGATE] dst = other + x * sigmoid(dot(w, y_t)) per token, w a [K] f32 vector (the shared-expert gate).
bool ggml_cuda_shexp_gate_supported(const ggml_tensor * w, const ggml_tensor * y, const ggml_tensor * x,
        const ggml_tensor * other, const ggml_tensor * dst);
void ggml_cuda_shexp_gate(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * y,
        const ggml_tensor * x, const ggml_tensor * other, ggml_tensor * dst);
