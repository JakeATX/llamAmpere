#pragma once
#include "common.cuh"

// AI+ unit weights (AITk_U): row-slab reconstruct to f32 (original basis) + cuBLAS SGEMM on the backend's
// handle (TF32 math). The matching weight-only control for a quantized type is GGML_CUDA_FORCE_CUBLAS_RT=1 plus
// GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 (dequant to f32 + the same cublasSgemm); without the second flag sm_86 runs the
// quantized cuBLAS path in fp16 compute. No mmvq/mmq/mmf.
void ggml_cuda_mul_mat_ai4(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
#define GGML_CUDA_AI4_GRP_MAX 8   // = ai4k::MMV_GRP_MAX
// [INKDENSE] grouped decode mmv over n MUL_MATs that share src1 (false = not taken, nothing launched)
bool ggml_cuda_mul_mat_ai4_group(ggml_backend_cuda_context & ctx, ggml_tensor * const * dsts, int n);

// [OPT9] AI+ drop-in fusions, all default off (ai4.cu has the full switch list):
//   GGML_CUDA_AI_AUTO=1  GGML_CUDA_AI_FUSE_ACT=1  GGML_CUDA_AI_FUSE_GLU=1  GGML_CUDA_AI_MMV_GROUP_SPEC=1
bool ggml_cuda_ai_auto_enabled();
bool ggml_cuda_ai_fuse_act_enabled();
bool ggml_cuda_ai_fuse_glu_enabled();
bool ggml_cuda_ai_mmv_group_spec_enabled();
// the ai4 prep variant (0 PLAIN, 1 SWAP8, 2 XPERM, 3 LINEAR) an AI+ / AIF consumer at decode / verify width (<= 32 rows)
// reads, or -1 when it takes no int8 prep (reference paths, small-M f32 product, FUSED_X, wider batches)
int  ggml_cuda_ai_prep_variant_for(const ggml_backend_cuda_context & ctx, const ggml_tensor * op);
// fused producers: write the f32 output and the consumer's prep (variant) into ai_prep_cache; false = nothing launched
bool ggml_cuda_ai_rms_norm_mul_prep(ggml_backend_cuda_context & ctx, const ggml_tensor * norm, const ggml_tensor * mul,
                                    int variant);
bool ggml_cuda_ai_glu_prep(ggml_backend_cuda_context & ctx, const ggml_tensor * glu, int variant);
// dense AI+ gate + up + GLU (N <= 32) in one launch; with down_next (the GLU's AI+ consumer) and FUSE_ACT also its prep
bool ggml_cuda_mul_mat_ai4_dense_glu(ggml_backend_cuda_context & ctx, const ggml_tensor * gate, const ggml_tensor * up,
                                     ggml_tensor * glu, const ggml_tensor * down_next);

// AI+ unit types: shapes and layouts the CUDA backend runs (MUL_MAT and MUL_MAT_ID)
bool ggml_cuda_ai4_supported(const ggml_tensor * op);
// [PREFILL-GLUE] activation layout (AI4_PREP_*) ggml_cuda_mul_mat_ai4 quantizes src1 into for this shape
int  ggml_cuda_ai4_prefill_variant(const ggml_backend_cuda_context & ctx, ggml_type w_type, int M, int N, int K, bool is_id);

// MUL_MAT_ID over an AI+ unit expert stack (device-side routing, only the selected experts are read)
// y_into != nullptr: write the routed pairs into y_into (same layout as dst) and leave skipped pairs untouched
void ggml_cuda_mul_mat_id_ai4(ggml_backend_cuda_context & ctx, const ggml_tensor * dst, float * y_into = nullptr);

// true when ggml_cuda_mul_mat_id_ai4 would take the host-routed reference path (GGML_CUDA_AI4_REF=1 or pre-Ampere)
bool ggml_cuda_ai4_id_needs_sync(int cc, ggml_type t);

// fork AIPFIX D: AI+ unit experts held in the expert cache. Activations [K, rows] f32 (contiguous) to the AI layout
// (int8 q then fp32 group scales at ggml_cuda_moe_ai_act_scales); act_bytes is the buffer size it needs.
// fork AIPFIX ONEPATH check: f32 image [M][K] of an AI+ unit matrix (original basis), reference only
void ggml_cuda_moe_ai_reconstruct(const void * src, ggml_type type, float * W, int64_t K, int64_t M, cudaStream_t stream);
size_t ggml_cuda_moe_ai_act_bytes(int64_t rows, int64_t K);
size_t ggml_cuda_moe_ai_act_scales(int64_t rows, int64_t K);
void ggml_cuda_moe_ai_quantize(const float * x, int64_t rows, int64_t K, char * out, cudaStream_t stream);
// Routed GEMV over expert pointers table[e] (one direct block per route and token, the stock AI+ MUL_MAT_ID direct
// kernel and order). ids [n_used, n_tok] with row stride ids_stride; act from ggml_cuda_moe_ai_quantize of the
// [K, ne11, n_tok] input; dst [M, n_used, n_tok]. With a filter only routes whose expert has filter[e] == filter_val
// are written. False (nothing launched) when the type or shape has no direct kernel. With table2 the same launch
// also computes a second projection of the same type and shape (gate beside up) into dst2, filtered by filter2.
// With glu (and table2 = gate, table = up) the launch also writes glu = silu(dst2)*dst for every route whose two
// projections are stored by this launch or (prev / prev2 = the hit pass filters of up / gate, val 1) by that earlier
// pass; false (nothing launched) when the route tiles exceed the counter table. group = false keeps the routes of
// one expert in their own blocks (no grouped slab-per-warp launch).
bool ggml_cuda_moe_ai_table_supported(ggml_type type, int64_t K, int64_t M, int64_t n_tok);
bool ggml_cuda_moe_ai_table_mmv(const void * const * table, ggml_type type, const char * act, const int32_t * ids,
        float * dst, int64_t K, int64_t M, int64_t n_used, int64_t n_tok, int64_t ne11, int64_t ids_stride,
        int n_expert, const uint8_t * filter, int filter_val, cudaStream_t stream,
        const void * const * table2 = nullptr, float * dst2 = nullptr, const uint8_t * filter2 = nullptr,
        float * glu = nullptr, const uint8_t * prev = nullptr, const uint8_t * prev2 = nullptr, bool group = true);

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
    bool res_both = false;       // fork AIPFIX P2LIST: a cached route is resident only when both projections are
    bool hot_only = false;       // fork AIPFIX P1HITG: phase 1 takes the bank routes only (the 3XS hit kernel ran the cached ones)
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
bool ggml_cuda_aif25_supported(const ggml_tensor * op);
void ggml_cuda_mul_mat_aif25(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * x, ggml_tensor * dst);
// [KAIFUSE R2] MUL_MAT_ID over one-row experts of a fractional AI+ head (draft vocabulary shortlist)
bool ggml_cuda_aif_rows_supported(const ggml_tensor * op);
void ggml_cuda_mul_mat_id_aif_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
