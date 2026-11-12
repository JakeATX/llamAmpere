#include "common.cuh"

// GGML_OP_LSA_COMP_MERGE on CUDA (fork, opt-in LSA fast compensation; see ggml.h). Split over the mean blocks:
// one CTA per (block chunk, row, kv head) scores the chunk for all q heads of that kv head (each mean row is read
// once per row, not once per q head) and writes a partial (max, sum, weighted V sum); a second kernel merges the
// partials with the masked FA output and its log-sum-exp.
void ggml_cuda_lsa_comp_merge(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_lsa_comp_merge_supported(const ggml_tensor * op);
