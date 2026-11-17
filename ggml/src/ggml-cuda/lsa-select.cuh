#include "common.cuh"

// GGML_OP_LSA_BLOCK_SELECT on CUDA (fork, opt-in LSA block pick; see ggml.h): one block per row ranks the blocks and
// marks the kb best eligible ones. Same selection as the CPU op, ties included (ties go to the lower block index).
void ggml_cuda_lsa_block_select(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_lsa_block_select_supported(const ggml_tensor * op);

// GGML_OP_LSA_UNIT_LSE on CUDA (fork, opt-in LSA pick; see ggml.h): index scoring + unit logsumexp in one kernel.
void ggml_cuda_lsa_unit_lse(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_lsa_unit_lse_supported(const ggml_tensor * op);
