#include "common.cuh"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0 (slot_rows == nullptr) or the cache base (slot_rows != nullptr)
    int64_t slot_stride; // between rollback slots (0 when K==1); unused when slot_rows != nullptr
    // [TAG_RECURRENT_ROLLBACK_RING] set_rows form: slot i of seq s goes to cache row
    // slot_rows[i * n_seqs + s] (row = D floats), i.e. data + row * D
    const int32_t * slot_rows = nullptr;
};

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache);
