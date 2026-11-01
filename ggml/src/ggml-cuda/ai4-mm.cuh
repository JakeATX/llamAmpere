#pragma once
#include "common.cuh"

// AI+ unit weights (AITk_U): row-slab reconstruct to f32 (original basis) + cuBLAS SGEMM on the backend's
// handle (TF32 math). The matching weight-only control for a quantized type is GGML_CUDA_FORCE_CUBLAS_RT=1 plus
// GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 (dequant to f32 + the same cublasSgemm); without the second flag sm_86 runs the
// quantized cuBLAS path in fp16 compute. No mmvq/mmq/mmf.
void ggml_cuda_mul_mat_ai4(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// AI+ unit types: shapes and layouts the CUDA backend runs (MUL_MAT and MUL_MAT_ID)
bool ggml_cuda_ai4_supported(const ggml_tensor * op);

// MUL_MAT_ID over an AI+ unit expert stack (device-side routing, only the selected experts are read)
void ggml_cuda_mul_mat_id_ai4(ggml_backend_cuda_context & ctx, const ggml_tensor * dst);

// true when ggml_cuda_mul_mat_id_ai4 would take the host-routed reference path (GGML_CUDA_AI4_REF=1 or pre-Ampere)
bool ggml_cuda_ai4_id_needs_sync(int cc, ggml_type t);

// fork AIPFIX D: AI+ unit experts held in the expert cache. Activations [K, rows] f32 (contiguous) to the AI layout
// (int8 q then fp32 group scales at ggml_cuda_moe_ai_act_scales); act_bytes is the buffer size it needs.
size_t ggml_cuda_moe_ai_act_bytes(int64_t rows, int64_t K);
size_t ggml_cuda_moe_ai_act_scales(int64_t rows, int64_t K);
void ggml_cuda_moe_ai_quantize(const float * x, int64_t rows, int64_t K, char * out, cudaStream_t stream);
// Routed GEMV over expert pointers table[e] (one direct block per route and token, the stock AI+ MUL_MAT_ID direct
// kernel and order). ids [n_used, n_tok] with row stride ids_stride; act from ggml_cuda_moe_ai_quantize of the
// [K, ne11, n_tok] input; dst [M, n_used, n_tok]. With a filter only routes whose expert has filter[e] == filter_val
// are written. False (nothing launched) when the type or shape has no direct kernel.
bool ggml_cuda_moe_ai_table_supported(ggml_type type, int64_t K, int64_t M, int64_t n_tok);
bool ggml_cuda_moe_ai_table_mmv(const void * const * table, ggml_type type, const char * act, const int32_t * ids,
        float * dst, int64_t K, int64_t M, int64_t n_used, int64_t n_tok, int64_t ne11, int64_t ids_stride,
        int n_expert, const uint8_t * filter, int filter_val, cudaStream_t stream);

bool ggml_cuda_moe_mixed_supported(const ggml_tensor * bank, const ggml_tensor * base);
const ggml_tensor * ggml_cuda_moe_mixed_base(const ggml_tensor * joined);
// The caller retains these buffers through the mixed GEMV; IQ4_NL banks need only Q8 activations.
void ggml_cuda_moe_mixed_quantize_x(ggml_backend_cuda_context & ctx, const ggml_tensor * bank,
        ggml_cuda_pool_alloc<int8_t> & q, ggml_cuda_pool_alloc<float> & s);
// Same, plus the base experts' q8_1 activations of x (quantize_row_q8_1_cuda layout for weight type q8_type,
// ne0 padded to MATRIX_ROW_PADDING) into q8, from one launch. False (nothing done) when the bank or x does not fit;
// the caller then runs the two quantizers. GGML_MOE_BANK_MIXED_DUALQ=0 always declines (default 1).
bool ggml_cuda_moe_mixed_quantize_dual(ggml_backend_cuda_context & ctx, const ggml_tensor * bank, const ggml_tensor * x,
        char * q8, ggml_type q8_type, ggml_cuda_pool_alloc<int8_t> & q, ggml_cuda_pool_alloc<float> & s);
// Second projection of one paired launch (gate with up): same activations, routes, bank and base types
// and shapes; its blocks follow the first projection's along x. Route outputs are written by the first only.
struct ggml_cuda_moe_mixed_pair {
    const ggml_tensor * bank;
    const ggml_tensor * base;
    ggml_tensor * dst;
    const void * const * table;
    const uint8_t * skip;
    const uint8_t * resident;
    ggml_tensor * glu = nullptr; // SwiGLU of the two outputs, written by the same launch (nullptr: none)
};
// GGML_MOE_BANK_MIXED_PAIR=0 restores separate up and gate launches (default 1).
bool ggml_cuda_moe_mixed_pair_enabled();
bool ggml_cuda_moe_mixed_pair_supported(const ggml_tensor * bank, const ggml_tensor * base,
        const ggml_tensor * bank2, const ggml_tensor * base2);
// GGML_MOE_BANK_MIXED_GLU=0 leaves the SwiGLU of a paired launch to its own node (default 1).
bool ggml_cuda_moe_mixed_glu_enabled();
// glu is the plain SwiGLU of the paired outputs dst and dst2 (either one the gate).
bool ggml_cuda_moe_mixed_glu_supported(const ggml_tensor * dst, const ggml_tensor * dst2, const ggml_tensor * glu);
// Routed sum written by the same (unpaired) launch: dst = the weighted hot and cold bucket sums of the
// projection's routes, as ggml_cuda_op_moe_mixed_reduction computes them from the finished output.
struct ggml_cuda_moe_mixed_reduce {
    const ggml_tensor * weights;  // [1, n_used, tokens]
    const ggml_tensor * bank_ids; // [n_used, tokens], >= 0: hot bucket
    ggml_tensor * dst;            // [M, tokens]
};
// GGML_MOE_BANK_MIXED_REDUCE=0 leaves the routed sum to its own launch (default 1).
bool ggml_cuda_moe_mixed_reduce_enabled();
bool ggml_cuda_moe_mixed_reduce_supported(const ggml_tensor * experts, const ggml_cuda_moe_mixed_reduce & red);
void ggml_cuda_mul_mat_id_mixed(ggml_backend_cuda_context & ctx, const ggml_tensor * bank,
        const ggml_tensor * base, ggml_tensor * dst, const char * act_q8 = nullptr,
        const void * const * table = nullptr, const uint8_t * skip = nullptr, const uint8_t * resident = nullptr,
        const ggml_tensor * route_map = nullptr, ggml_tensor * route_mask = nullptr,
        const int8_t * act_ai_q = nullptr, const float * act_ai_s = nullptr, int phase = 0,
        const ggml_cuda_moe_mixed_pair * pair = nullptr, const ggml_cuda_moe_mixed_reduce * red = nullptr);
