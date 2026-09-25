#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11);

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

// Shared-quantize cache (q8_cache in common.cuh). ggml_cuda_q8_cacheable: whether a q8_1 activation of q8_bytes may be
// cached on the current stream. ggml_cuda_q8_cache_claim: keys the cache on (src1, type_src0, layout) for this graph
// eval and returns its buffer; the caller must fill it with quantize_row_q8_1_cuda's layout for type_src0 before any
// consumer runs (stream order). Used by ggml_cuda_mul_mat_vec_q and the fused ADD + RMS_NORM + MUL prefill [#46].
bool   ggml_cuda_q8_cacheable(const ggml_backend_cuda_context & ctx, size_t q8_bytes);
char * ggml_cuda_q8_cache_claim(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, ggml_type type_src0,
                                size_t q8_bytes, int64_t ne10_padded);

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
    const ggml_cuda_mm_fusion_args_host * fusion = nullptr, bool convrot = false);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

// Runs one batched matvec over slot-pool experts selected by device-side IDs.
// Output row c is W[ids[c]] times activation row act_ids[c].
// slot_stride_bytes is the original tensor nb[2], and act_q8 uses the padded q8_1 layout from quantize_row_q8_1_cuda.
void ggml_cuda_moe_cache_mmv(
    const void * pool, ggml_type type0, const char * act_q8,
    const int32_t * ids_dev, const int32_t * act_ids_dev,
    float * dst_dev, int64_t n_in, int64_t n_out, int64_t n_slots,
    int64_t slot_stride_bytes, int64_t n_hits, int64_t act_rows, cudaStream_t stream);

// Runs up * GLU(gate) for rows whose two independently cached experts are resident.
void ggml_cuda_moe_cache_mmv_fused(
    const void * up_pool, const void * gate_pool, ggml_type type0,
    const char * act_q8, const int32_t * up_ids_dev,
    const int32_t * gate_ids_dev, const int32_t * act_ids_dev,
    float * dst_dev, int64_t n_in, int64_t n_out,
    int64_t slot_stride_bytes, int64_t n_hits, int64_t act_rows,
    float up_min, float up_max, float gate_min, float gate_max,
    cudaStream_t stream);
