#include "common.cuh"

// GGML_OP_LOOP_GATE on CUDA (fork, Inkling 2 loop gate; see ggml.h): 2 kernels, the rank-R down products with the two
// sums of squares, then the gain/forget products and the gate.
void ggml_cuda_loop_gate(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_loop_gate_supported(const ggml_tensor * op);
