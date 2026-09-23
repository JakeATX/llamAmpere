#include "common.cuh"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    void *    data;        // rollback slot 0 (slot_rows == nullptr) or the cache base (slot_rows != nullptr)
    ggml_type type;        // cache element type: F32, or BF16/F16/Q8_0 with --cache-type-s
    int64_t   slot_stride; // between rollback slots (0 when K==1); unused when slot_rows != nullptr
    // [TAG_RECURRENT_ROLLBACK_RING] set_rows form: slot i of seq s goes to cache row
    // slot_rows[i * n_seqs + s] (row = D elements), i.e. data + row * D
    const int32_t * slot_rows = nullptr;
};

// [#87] where the kernels read each sequence's input state. rows == nullptr: the op's own src[5]
// (sequence s at s * D floats). Otherwise row rows[s] of an F32 matrix with D-float rows spaced
// row_stride floats apart (the recurrent cache the skipped GET_ROWS would have gathered from).
struct ggml_cuda_gated_delta_net_state_src {
    const int32_t * rows       = nullptr;
    int64_t         row_stride = 0;
};

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache);
