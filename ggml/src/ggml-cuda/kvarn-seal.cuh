#include "common.cuh"

// GGML_OP_KVARN_SEAL on CUDA: seals (head, group) tiles of rotated fp16 K/V rows into KVarN records.
// Produces records identical to the CPU sealer (ggml-kvarn.h): same reduction contract, IEEE float
// division via __fdiv_rn (the build uses -use_fast_math), log domain in double.
void ggml_cuda_kvarn_seal(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_kvarn_seal_supported(const ggml_tensor * op);
