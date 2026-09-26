#include "common.cuh"

// GGML_OP_KVARN_SEAL on CUDA: seals rotated F16, Q8_0 or stored-domain TQ6_0 K/V tiles into KVarN records.
// Produces records identical to the CPU sealer (ggml-kvarn.h): same reduction contract, IEEE float
// division via __fdiv_rn (the build uses -use_fast_math), log domain in double.
void ggml_cuda_kvarn_seal(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_kvarn_seal_supported(const ggml_tensor * op);

// trellis body codebooks: every translation unit holding device copies registers their symbols at load time;
// init (once per device, on the first trellis seal or attention) overwrites them with the GGML_KVARN_TRELLIS_CB
// file when that override is set (the header defaults otherwise stay in place)
void ggml_cuda_kvarn_trellis_cb_register(const void * sym_k, const void * sym_v);
void ggml_cuda_kvarn_trellis_cb_init();
