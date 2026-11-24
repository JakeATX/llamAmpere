#include "common.cuh"
#include "convrot.cuh"
#include "mmq.cuh"
#include "mmq-prefill.cuh"
#include "quantize.cuh"
#include "dense-tc.cuh"
#include "mmid.cuh"
#include "prefill-glue.cuh"

#include <cstdint>
#include <cstdlib>

// Some opt-in callers bypass ggml_cuda_should_use_mmq() after doing their own
// activation preparation. Check that the selected architecture actually has
// a launchable table entry before entering MMQ so unsupported combinations
// can use their existing fallback instead of reaching J_best == 0 and aborting.
bool ggml_cuda_mmq_has_config(
        const ggml_type type, const int64_t nrows_x, const int cc, const size_t smpbo) {
    const bool fallback = ggml_cuda_mmq_needs_fallback(nrows_x);

    for (int J = 8; J <= 128; J += 8) {
        const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, fallback, cc, GGML_PREC_Q8);
        if (config.type != GGML_TYPE_COUNT && mmq_get_nbytes_shared(config, cc) <= smpbo) {
            return true;
        }
    }

    return false;
}

static void ggml_cuda_mul_mat_q_switch_type(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream, const ggml_prec prec_src1) {
    switch (args.type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_q_case<GGML_TYPE_Q1_0, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_TQ4_1S:
            mul_mat_q_case<GGML_TYPE_TQ4_1S, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_Q2_0:
            mul_mat_q_case<GGML_TYPE_Q2_0, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_PQ2_0:
            mul_mat_q_case<GGML_TYPE_PQ2_0, GGML_PREC_Q8>(ctx, args, stream);
            break;
#if !defined(GGML_USE_HIP)
        case GGML_TYPE_PTQ1_0:
            mul_mat_q_case<GGML_TYPE_PTQ1_0, GGML_PREC_Q8>(ctx, args, stream);
            break;
#endif
        case GGML_TYPE_Q4_0:
            mul_mat_q_case<GGML_TYPE_Q4_0, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_q_case<GGML_TYPE_Q4_1, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_q_case<GGML_TYPE_Q5_0, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_q_case<GGML_TYPE_Q5_1, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_q_case<GGML_TYPE_Q8_0, GGML_PREC_Q8>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_Q2_K:
            mul_mat_q_case<GGML_TYPE_Q2_K, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_q_case<GGML_TYPE_Q3_K, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_q_case<GGML_TYPE_Q4_K, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_q_case<GGML_TYPE_Q5_K, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_q_case<GGML_TYPE_Q6_K, GGML_PREC_Q8>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_IQ1_S:
            mul_mat_q_case<GGML_TYPE_IQ1_S, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_q_case<GGML_TYPE_IQ2_XXS, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_q_case<GGML_TYPE_IQ2_XS, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_q_case<GGML_TYPE_IQ2_S, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_q_case<GGML_TYPE_IQ3_XXS, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_q_case<GGML_TYPE_IQ3_S, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_q_case<GGML_TYPE_IQ4_XS, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_q_case<GGML_TYPE_IQ4_NL, GGML_PREC_Q8>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_MXFP4:
            // src1 at Q4 uses the native FP4 instructions, which are Blackwell-only
            if (prec_src1 == GGML_PREC_Q4) {
                mul_mat_q_case<GGML_TYPE_MXFP4, GGML_PREC_Q4>(ctx, args, stream);
                break;
            }
            mul_mat_q_case<GGML_TYPE_MXFP4, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_NVFP4:
            if (prec_src1 == GGML_PREC_Q4) {
                mul_mat_q_case<GGML_TYPE_NVFP4, GGML_PREC_Q4>(ctx, args, stream);
                break;
            }
            mul_mat_q_case<GGML_TYPE_NVFP4, GGML_PREC_Q8>(ctx, args, stream);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

// overrides the src1 precision requested by the graph, "auto" keeps the requested one
static ggml_prec ggml_cuda_mmq_get_prec_env() {
    const char * env_c = getenv("GGML_CUDA_MMQ_PREC");
    if (env_c == nullptr) {
        return GGML_PREC_UNDEFINED;
    }
    std::string env_cpp = env_c;
    for (char & c : env_cpp) {
        c = std::tolower(c);
    }
    if (env_cpp == "q4") {
        return GGML_PREC_Q4;
    }
    if (env_cpp == "q8") {
        return GGML_PREC_Q8;
    }
    if (env_cpp != "auto") {
        GGML_LOG_WARN("%s: Unknown value for GGML_CUDA_MMQ_PREC: '%s'. Available: 'q4', 'q8', 'auto'.\n", __func__, env_cpp.c_str());
    }
    return GGML_PREC_UNDEFINED;
}

// src1 is quantized to Q8_1 unless the FP4 types can use 4-bit activations, in which case they
// default to the native W4A4 instructions on Blackwell.
static ggml_prec ggml_cuda_mmq_get_prec_src1(const ggml_type type, const ggml_prec prec_requested, const int cc) {
    static const ggml_prec prec_env = ggml_cuda_mmq_get_prec_env();

    ggml_prec prec = prec_env;
    if (prec == GGML_PREC_UNDEFINED) {
        prec = prec_requested;
    }

    // Q4 only for the FP4 types on Blackwell
    GGML_ASSERT(prec == GGML_PREC_UNDEFINED || prec == GGML_PREC_Q8 || prec == GGML_PREC_Q4);
    const bool can_use_q4 = (type == GGML_TYPE_NVFP4 || type == GGML_TYPE_MXFP4) && blackwell_mma_available(cc);
    if (prec == GGML_PREC_Q8 || !can_use_q4) {
        return GGML_PREC_Q8;
    }
    return GGML_PREC_Q4;
}

// fork AIPFIX (LLAMA_MOE_COLD_SKIP): routes whose skip value is >= 0 are computed by another call (the hot bank).
// They get the out-of-range expert n_experts, so the ids helper gives them no compact row, and a zero output row.
// Unused compact rows sit past the last expert: skipped token slots point their inverse map at the last row,
// and the forward map of the unused rows reads source row 0.
static __global__ void mmq_route_skip(
        const int32_t * __restrict__ ids, const int si1, const float * __restrict__ skip, int32_t * __restrict__ ids_out,
        int32_t * __restrict__ ids_src1, float * __restrict__ dst, const int n_expert_used, const int n_experts,
        const int64_t ne0, const int64_t s1, const int64_t s2, const int src1_fill) {
    const int i   = blockIdx.x;
    const int it  = i / n_expert_used;
    const int iex = i % n_expert_used;
    const bool skipped = skip[i] >= 0.0f;
    if (threadIdx.x == 0) {
        ids_out[i]  = skipped ? n_experts : ids[(int64_t) it*si1 + iex];
        ids_src1[i] = src1_fill;
    }
    if (skipped) {
        float * row = dst + it*s2 + iex*s1;
        for (int64_t j = threadIdx.x; j < ne0; j += blockDim.x) {
            row[j] = 0.0f;
        }
    }
}

// fork: v0.6.3's MoE tile grid (default, see launch_mul_mat_q in mmq.cuh): ntx = ceil(ne12/J) column tiles per expert,
// tile jt of expert e covering compact columns [b[e] + jt*J, min(b[e] + (jt + 1)*J, b[e+1])), empty past b[e+1], in
// expert-major, jt-minor order. With nchannels_y == 1 this gives the same tile index and stream-k kbc space as v0.6.3,
// so the K split points and the fixup order are the same and the sums are bit-identical. The expert bounds come from
// the per-expert extra entries mm_ids_helper writes after the seam entries: entry nseam + e always ends at b[e+1]
// (its loop only stops on the expert's last column), and b[0] = 0 (valid ids are >= 0, skipped routes use n_experts).
static __global__ void mmq_tile_spec_per_expert(
        const ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec, ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec_v063,
        const int nseam, const int n_experts, const int ntx, const int J) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_experts*ntx) {
        return;
    }
    const int e    = i / ntx;
    const int jt   = i % ntx;
    const int b_lo = e == 0 ? 0 : tile_spec[nseam + e - 1].col_high;
    const int b_hi = tile_spec[nseam + e].col_high;
    const int col_low = b_lo + jt*J;
    if (col_low < b_hi) {
        tile_spec_v063[i] = {e, col_low, min(col_low + J, b_hi), /*padding =*/ 0};
    } else {
        tile_spec_v063[i] = {e, b_hi, b_hi, /*padding =*/ 0};
    }
}

// Tile width in ne11 (dense) / ne12 (MoE) direction with the fewest tiles that fits into shared memory, 0 if there is none.
static int ggml_cuda_mmq_get_J_best(
        const ggml_type type, const int64_t ncols, const bool fallback, const int cc, const size_t smpbo, const ggml_prec prec_src1,
        int * nthreads_best = nullptr) {
    int J_best        = 0;
    int ntiles_J_best = INT_MAX;

    for (int J = 8; J <= 128 && ntiles_J_best > 1; J += 8) {
        const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, fallback, cc, prec_src1);
        if (config.type == GGML_TYPE_COUNT) {
            continue;
        }

        if (mmq_get_nbytes_shared(config, cc) > smpbo) {
            continue;
        }

        const int ntiles_x = (ncols + config.J - 1) / config.J;

        if (ntiles_x < ntiles_J_best) {
            J_best = J;
            if (nthreads_best) {
                *nthreads_best = config.nthreads;
            }
            ntiles_J_best = ntiles_x;
        }
    }
    return J_best;
}

// fork: J_best is only usable if a tile of that width also exists for the given type (paired products share src1's tiles).
static bool ggml_cuda_mmq_J_valid(
        const ggml_type type, const int J, const bool fallback, const int cc, const size_t smpbo, const ggml_prec prec_src1,
        int * nthreads = nullptr) {
    const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, fallback, cc, prec_src1);
    if (config.type == GGML_TYPE_COUNT || config.J != J || mmq_get_nbytes_shared(config, cc) > smpbo) {
        return false;
    }
    if (nthreads) {
        *nthreads = config.nthreads;
    }
    return true;
}

// The column count ggml_cuda_mul_mat_q optimizes J_best against: ne11 dense; with ids ne12 (v0.6.3), or the average
// tokens per expert on RDNA3/RDNA4 and with GGML_CUDA_MMQ_ID_JAVG=1 (upstream averages everywhere). J_best sets the
// tile width and with it the stream-k partition, so the NVIDIA default stays v0.6.3's.
static int64_t ggml_cuda_mmq_ncols_opt(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const int cc) {
    if (!ids) {
        return src1->ne[1];
    }
    static const bool javg = [] { const char * e = getenv("GGML_CUDA_MMQ_ID_JAVG"); return e && atoi(e) > 0; }();
    if (!(GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc) || javg)) {
        return src1->ne[2];
    }
    const int64_t n_expert_used = ids->ne[0];
    // Each expert only sees ne12*n_expert_used/ne02 tokens on average.
    return (src1->ne[2]*n_expert_used + src0->ne[2] - 1) / src0->ne[2];
}

static void ggml_cuda_mul_mat_q_impl(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        const ggml_tensor * ids, ggml_tensor * dst, bool convrot, ggml_tensor * paired_dst,
        const ggml_cuda_mm_fusion_args_host * fusion, bool dst_bf16 = false, bool * out_wrote_bf16 = nullptr) {
    if (out_wrote_bf16) {
        *out_wrote_bf16 = false;
    }
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.
    GGML_ASSERT(!convrot || (!ids && mmq_get_q8_1_ds_layout(src0->type) == MMQ_Q8_1_DS_LAYOUT_D4));

    // Fused gate/up/GLU: src0 is the up weight, dst the GLU output.
    // Fused scale (no gate): dst is the MUL that applies the weight scale to the MUL_MAT.
    const ggml_tensor * gate = fusion ? fusion->gate : nullptr;
    ggml_cuda_mm_fusion_args_device fusion_device = {};
    if (fusion) {
        GGML_ASSERT(!fusion->x_bias && !fusion->gate_bias);
        // NVFP4 weight scales (only applied for NVFP4): one value per tensor, or one per expert with ids.
        for (const ggml_tensor * scale : { fusion->x_scale, fusion->gate_scale }) {
            GGML_ASSERT(!scale || (src0->type == GGML_TYPE_NVFP4 && scale->type == GGML_TYPE_F32 && ggml_is_contiguous(scale) &&
                ggml_nelements(scale) == (ids ? src0->ne[2] : 1)));
        }
        if (gate) {
            GGML_ASSERT(!fusion->x_scale == !fusion->gate_scale);
            GGML_ASSERT(gate->type == src0->type && ggml_are_same_shape(gate, src0) && ggml_are_same_stride(gate, src0));
            fusion_device.gate       = gate->data;
            fusion_device.gate_scale = fusion->gate_scale ? fusion->gate_scale->data : nullptr;
            fusion_device.glu_op     = fusion->glu_op;
            fusion_device.glu_limit  = fusion->glu_limit;
        } else {
            GGML_ASSERT(!fusion->gate_scale && fusion->x_scale);
        }
        fusion_device.x_scale = fusion->x_scale ? fusion->x_scale->data : nullptr;
    }

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();

    const int    id    = ggml_cuda_get_device();
    const int    cc    = ggml_cuda_info().devices[id].cc;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    const char  * src0_d = (const char  *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *)  dst->data;

    const ggml_tensor * paired_src0 = paired_dst ? paired_dst->src[0] : nullptr;

    // Preserve each original call's weight-padding contract.
    const auto clear_padding = [&](const ggml_tensor * weight) {
        if (ggml_backend_buffer_get_usage(weight->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
            const size_t size_data  = ggml_nbytes(weight);
            const size_t size_alloc = ggml_backend_buffer_get_alloc_size(weight->buffer, weight);
            if (size_alloc > size_data) {
                GGML_ASSERT(ggml_is_contiguously_allocated(weight));
                GGML_ASSERT(!weight->view_src);
                CUDA_CHECK(cudaMemsetAsync((char *) weight->data + size_data, 0, size_alloc - size_data, stream));
            }
        }
    };
    clear_padding(src0);
    if (paired_src0) {
        clear_padding(paired_src0);
    }
    if (gate) {
        clear_padding(gate); // upstream: the fused gate weight
    }
    GGML_ASSERT(!(paired_src0 && gate) && !(convrot && gate));

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const bool fallback = ggml_cuda_mmq_needs_fallback(ne01);

    // With fusion dst is the GLU output or the scale node, not a MUL_MAT; the src1 precision requested by the fused MUL_MATs is in the fusion args.
    const ggml_prec prec_src1 = ggml_cuda_mmq_get_prec_src1(src0->type, fusion ? fusion->prec_src1 : ggml_prec_get_src(dst, 1), cc);

    const bool use_native_fp4 = prec_src1 == GGML_PREC_Q4;
    const size_t y_block_size       = use_native_fp4 ? sizeof(block_fp4_mmq) : sizeof(block_q8_1_mmq);
    const size_t y_values_per_block = use_native_fp4 ? QK_FP4_MMQ            : QK8_1_MMQ;

    int J_best        = 0;
    int nthreads_best = 0;
    J_best = ggml_cuda_mmq_get_J_best(src0->type, ggml_cuda_mmq_ncols_opt(src0, src1, ids, cc), fallback, cc, smpbo, prec_src1, &nthreads_best);
    GGML_ASSERT(J_best > 0);

    // A tile of size J can read in at most J - 1 extra columns.
    // For simplicity, round up the padding of a full tile to a multiple of the number of bytes that nthreads can load in parallel.
    const auto src1_padding_for = [&](const int nthreads) {
        const size_t src1_load_chunk_size = nthreads * sizeof(int);
        return ((J_best * sizeof(block_q8_1_mmq) + src1_load_chunk_size - 1) / src1_load_chunk_size) * src1_load_chunk_size;
    };
    size_t src1_q8_1_padding = src1_padding_for(nthreads_best);
    if (paired_src0) {
        // fork MMQ_PAIR: the paired product reuses src1's q8_1 buffer, ids_dst and tile_spec, so it runs at the same J_best
        // (ggml_cuda_mul_mat_q_pair checked that the paired type has that tile).
        int nthreads_pair = 0;
        GGML_ASSERT(ggml_cuda_mmq_J_valid(paired_src0->type, J_best, fallback, cc, smpbo, prec_src1, &nthreads_pair));
        src1_q8_1_padding = std::max(src1_q8_1_padding, src1_padding_for(nthreads_pair));
    }

    const auto launch_products = [&](const mmq_args & args) {
        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream, prec_src1);
        if (paired_src0) {
            mmq_args paired_args = args;
            const size_t ts_pair = ggml_type_size(paired_src0->type);
            paired_args.x                  = (const char *) paired_src0->data;
            paired_args.type_x             = paired_src0->type;
            paired_args.stride_row_x       = paired_src0->nb[1] / ts_pair;
            paired_args.stride_channel_x   = paired_src0->nb[2] / ts_pair;
            paired_args.stride_sample_x    = paired_src0->nb[3] / ts_pair;
            paired_args.dst                = (float *) paired_dst->data;
            paired_args.nrows_dst          = paired_dst->nb[1] / sizeof(float);
            paired_args.stride_channel_dst = paired_dst->nb[2] / sizeof(float);
            paired_args.stride_sample_dst  = paired_dst->nb[3] / sizeof(float);
            ggml_cuda_mul_mat_q_switch_type(ctx, paired_args, stream, prec_src1);
        }
    };

    if (!ids) {
        const int ntj = (ne1 + J_best - 1) / J_best;

        const size_t nbytes_src1_q8_1 = ne13*ne12 * ne11*ne10_padded * y_block_size/y_values_per_block + src1_q8_1_padding;
        ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool());
        ggml_cuda_pool_alloc<float> src1_scale(ctx.pool());

        const int64_t s12 = use_native_fp4 ?
                                ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_FP4_MMQ * sizeof(int)) :
                                ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
        const int64_t s13 = ne12*s12;
        mmq_args args = {
            src0_d, src0->type, nullptr, nullptr, nullptr, dst_d, nullptr,
            ne00, ne01, ne1, s01, ne11, s1,
            ne02, ne12, s02, s12, s2,
            ne03, ne13, s03, s13, s3,
            ntj, J_best, fusion_device};
        args.ncols_max = ne1;
        const bool can_bf16 = dst_bf16 && !paired_src0 && !gate;
        args.dst_bf16 = can_bf16;
        args.out_wrote_bf16 = can_bf16 ? out_wrote_bf16 : nullptr;
        // fork: prefill can consume a swizzled producer buffer without quantizing it again.
        // fork: a paired product (paired_src0) reads the same q8_1 buffer in the default layout, so it keeps the default path.
        const bool use_mmq_prefill = !convrot && !use_native_fp4 && !paired_src0 && !gate && ggml_cuda_should_use_mmq_prefill(args);
        const char * prequantized = use_mmq_prefill ? ggml_cuda_mmq_prefill_q8_lookup(ctx, src1, dst, ne10_padded) : nullptr;
        if (prequantized == nullptr && !convrot && !use_native_fp4 && !paired_src0 && !gate && ne12 == 1 && ne13 == 1) {
            const int dsl = (int) mmq_get_q8_1_ds_layout(src0->type);
            ggml_cuda_prefill_mmq_take(ctx, src1, ne11, ne10, ne10_padded, dsl, use_mmq_prefill, prequantized);
        }
        if (prequantized == nullptr) {
            src1_q8_1.alloc(nbytes_src1_q8_1);
        }
        if (src0->type == GGML_TYPE_NVFP4 && use_native_fp4) {
            src1_scale.alloc(ne13*ne12*ne11);
        }
        args.y = (const int *) src1_q8_1.ptr;
        args.y_scale = src0->type == GGML_TYPE_NVFP4 && use_native_fp4 ? src1_scale.ptr : nullptr;

        if (prequantized != nullptr) {
            args.y = (const int *) prequantized;
        } else {
            const int64_t s11 = src1->nb[1] / ts_src1;
            const int64_t s12 = src1->nb[2] / ts_src1;
            const int64_t s13 = src1->nb[3] / ts_src1;
            const bool src_bf16 = !convrot && !use_native_fp4 && ggml_cuda_prefill_bf16_is_active(ctx, src1);
            if (convrot) {
                ggml_cuda_convrot_quantize_mmq_q8_1(
                    src1_d, src1_q8_1.get(), ne10, s11, s12, s13,
                    ne10_padded, ne11, ne12, ne13, stream);
            } else if (use_native_fp4) {
                static constexpr size_t align_float8 = 32;
                const bool use_aligned_float8 = ggml_cuda_is_aligned(src1, align_float8);
                static_assert(sizeof(block_fp4_mmq) == 4 * sizeof(block_q8_1));
                quantize_mmq_fp4_cuda(src1_d, nullptr, src1_q8_1.get(), src1_scale.ptr, src0->type, use_aligned_float8, ne10, s11, s12, s13, ne10_padded,
                                        ne11, ne12, ne13, stream);

            } else if (src_bf16) {
                ggml_cuda_prefill_bf16_to_mmq(
                    (const nv_bfloat16 *) src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13,
                    ne10_padded, ne11, ne12, ne13, false, 0, use_mmq_prefill, stream);
            } else {
                quantize_mmq_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded,
                                       ne11, ne12, ne13, stream, use_mmq_prefill);
            }
            CUDA_CHECK(cudaGetLastError());
        }

        if (use_mmq_prefill) {
            if (args.out_wrote_bf16) {
                *args.out_wrote_bf16 = args.dst_bf16;
            }
            ggml_cuda_mul_mat_q_prefill(args, stream);
            ctx.fusion_stats.mmq_prefill++;
        } else {
            launch_products(args);
        }
        return;
    }

    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;
    GGML_ASSERT(ne1 == n_expert_used);

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), ne_get_rows + J_best-1); // Needs to be padded for unconditional memory access.

    const int ntj = (ne_get_rows + J_best - 1) / J_best + ne02;
    ggml_cuda_pool_alloc<ggml_cuda_mm_id_tile_spec> tile_spec(ctx.pool(), ntj);

    // gate/up activations are broadcast across experts (ne11 == 1): quantize each token once and
    // scatter to its slots. ids_src1 then holds the inverse map (token slot -> compact row).
    const bool dedup_bcast = ne11 == 1 && n_expert_used > 1;

    {
        GGML_ASSERT(ids->nb[0] == ggml_element_size(ids));
        int si1  = ids->nb[1] / ggml_element_size(ids);
        const int sis1 = nb12 / nb11;
        const int32_t * ids_d = (const int32_t *) ids->data;

        const ggml_tensor * route_skip = dst->src[3];
        ggml_cuda_pool_alloc<int32_t> ids_skip(ctx.pool());
        if (route_skip && !paired_dst && route_skip->type == GGML_TYPE_F32 && ggml_is_contiguous(route_skip) &&
                route_skip->ne[0] == 1 && ggml_nelements(route_skip) == ne_get_rows) {
            ids_skip.alloc(ne_get_rows);
            mmq_route_skip<<<ne_get_rows, 128, 0, stream>>>(ids_d, si1, (const float *) route_skip->data, ids_skip.get(),
                ids_src1.get(), dst_d, n_expert_used, ne02, ne0, s1, s2, dedup_bcast ? ne_get_rows - 1 : 0);
            CUDA_CHECK(cudaGetLastError());
            ids_d = ids_skip.get();
            si1   = n_expert_used;
            ctx.fusion_stats.mmq_route_skip++;
            // Skipped routes belong to no expert, so the compact columns end before ne_get_rows and the tile_spec seams
            // past the last used column are written by no expert: zero them (col_low == col_high, the tile is skipped).
            CUDA_CHECK(cudaMemsetAsync(tile_spec.get(), 0, ntj*sizeof(ggml_cuda_mm_id_tile_spec), stream));
        }

        ggml_cuda_launch_mm_ids_helper(ids_d, ids_src1.get(), ids_dst.get(), tile_spec.get(),
            ne02, ne12, n_expert_used, ne11, J_best, si1, sis1, /*write_inverse =*/ dedup_bcast, stream, &ctx.pool());
        CUDA_CHECK(cudaGetLastError());
    }

    // fork: v0.6.3 per-expert tile grid unless GGML_CUDA_MMQ_ID_SK_UPSTREAM=1 (upstream #31047's linear layout has fewer
    // tiles, so stream-k splits K at other points and the sums differ in the last bits), see mmq_tile_spec_per_expert.
    const bool    tiles_v063 = !ggml_cuda_mmq_id_sk_upstream();
    const int64_t ntx_v063   = (ne12 + J_best - 1) / J_best;
    ggml_cuda_pool_alloc<ggml_cuda_mm_id_tile_spec> tile_spec_v063(ctx.pool());
    if (tiles_v063) {
        const int64_t n_v063 = ne02*ntx_v063;
        GGML_ASSERT(n_v063 < INT_MAX);
        tile_spec_v063.alloc(n_v063);
        const int nseam = (ne_get_rows + J_best - 1) / J_best;
        mmq_tile_spec_per_expert<<<(n_v063 + 255)/256, 256, 0, stream>>>(
            tile_spec.get(), tile_spec_v063.get(), nseam, ne02, ntx_v063, J_best);
        CUDA_CHECK(cudaGetLastError());
    }

    const size_t nbytes_src1_q8_1 = ne12*n_expert_used*ne10_padded * y_block_size/y_values_per_block + src1_q8_1_padding;
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);
    ggml_cuda_pool_alloc<float> src1_scale(ctx.pool());
    if (src0->type == GGML_TYPE_NVFP4 && use_native_fp4) {
        src1_scale.alloc(ne12*n_expert_used);
    }

    const int64_t ne11_flat = ne12*n_expert_used;
    const int64_t ne12_flat = 1;
    const int64_t ne13_flat = 1;

    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;

        if (use_native_fp4) {
            static constexpr size_t align_float8 = 32;
            const bool use_aligned_float8 = ggml_cuda_is_aligned(src1, align_float8);
            if (dedup_bcast) {
                quantize_scatter_mmq_fp4_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src1_scale.ptr, src0->type, use_aligned_float8, ne10,
                                        /*stride_token=*/s12, ne10_padded, ne12, ne11_flat, n_expert_used, stream);
            } else {
                quantize_mmq_fp4_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src1_scale.ptr, src0->type, use_aligned_float8, ne10, s11, s12, s13,
                                        ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
            }
        } else if (!dedup_bcast && ggml_cuda_prefill_glu_mmq_id_take(ctx, src1, ids_src1.get(), src1_q8_1.get(), src0->type, ne10, ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream)) {
            // [PREFILL-GLUE] GLU/SITU quantized directly in compact MoE sorted order using ids_src1
        } else if (ggml_cuda_prefill_bf16_is_active(ctx, src1)) {
            ggml_cuda_prefill_bf16_to_mmq((const nv_bfloat16 *) src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10,
                dedup_bcast ? s12 : s11, s12, s13, ne10_padded, dedup_bcast ? ne12 : ne11_flat, ne12_flat, ne13_flat,
                dedup_bcast, n_expert_used, false, stream);
        } else if (dedup_bcast) {
            quantize_scatter_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10,
                                    /*stride_token=*/s12, ne10_padded, ne12, ne11_flat, n_expert_used, stream);
        } else {
            quantize_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10, s11, s12, s13,
                                   ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    static_assert(QK_FP4_MMQ == 8 * QK_MXFP4, "QK_FP4_MMQ needs to be 8 * QK_MXFP4");
    const int64_t s12 = use_native_fp4 ? ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_FP4_MMQ * sizeof(int)) :
                                         ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const int64_t s13 = ne12*s12;

    // fork: GGML_CUDA_MMQ_ID_CAP (opt-in, off by default in v0.6.3) is superseded by the linearized tile_spec layout
    // (#31047), which has no idle blocks. GGML_CUDA_MMQ_ID_JAVG still works (ggml_cuda_mmq_ncols_opt).
    {
        static const bool warned = [] {
            if (getenv("GGML_CUDA_MMQ_ID_CAP") != nullptr) {
                GGML_LOG_WARN("%s: GGML_CUDA_MMQ_ID_CAP is ignored since the linearized MoE MMQ layout\n", __func__);
            }
            return true;
        }();
        GGML_UNUSED(warned);
    }

    // Note that ne02 is used instead of ne12 because the number of y channels determines the z dimension of the CUDA grid.
    // With tiles_v063 nchannels_y stays 1: the expert index is part of the linear tile index (e*ntx + jt).
    mmq_args args = {
        src0_d, src0->type, (const int *) src1_q8_1.get(), ids_dst.get(), tiles_v063 ? tile_spec_v063.get() : tile_spec.get(), dst_d,
        src1_scale.ptr,
        ne00, ne01, ne_get_rows, s01, ne_get_rows, s1,
        ne02,    1, s02, s12, s2,
        ne03, ne13, s03, s13, s3,
        tiles_v063 ? ne02*ntx_v063 : ntj, J_best, fusion_device, /*ncols_max =*/ ne12, /*tiles_v063 =*/ tiles_v063};
    const bool can_bf16 = dst_bf16 && !paired_src0 && !gate;
    args.dst_bf16 = can_bf16;
    args.out_wrote_bf16 = can_bf16 ? out_wrote_bf16 : nullptr;

    launch_products(args);
}

void ggml_cuda_mul_mat_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion, bool dst_bf16, bool * out_wrote_bf16) {
    ggml_cuda_mul_mat_q_impl(ctx, src0, src1, ids, dst, false, nullptr, fusion, dst_bf16, out_wrote_bf16);
}

void ggml_cuda_mul_mat_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        const ggml_tensor * ids, ggml_tensor * dst, bool convrot) {
    ggml_cuda_mul_mat_q_impl(ctx, src0, src1, ids, dst, convrot, nullptr, nullptr);
}

static int mmq_pair_mode() {
    static const int mode = [] {
        const char * e = getenv("GGML_CUDA_MMQ_PAIR");
        return e && strcmp(e, "1") == 0 ? 1 : e && strcmp(e, "2") == 0 ? 2 : 0;
    }();
    return mode;
}

bool ggml_cuda_mmq_pair_enabled() {
    return mmq_pair_mode() != 0;
}

static bool mmq_pair_type(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0: case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1: case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K: case GGML_TYPE_Q3_K: case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS: case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS: case GGML_TYPE_IQ3_S: case GGML_TYPE_IQ4_XS: case GGML_TYPE_IQ4_NL:
            return true;
        default:
            return false;
    }
}

static __global__ void mmq_pair_check(const float * actual, const float * reference, int64_t n) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i < n && __float_as_uint(actual[i]) != __float_as_uint(reference[i])) {
#if defined(__CUDA_ARCH__) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        // Validation is active even in Release builds. No host synchronization.
        asm volatile("trap;");
#endif
    }
}

bool ggml_cuda_mul_mat_q_pair(ggml_backend_cuda_context & ctx, ggml_tensor * first, ggml_tensor * second) {
    if (!ggml_cuda_mmq_pair_enabled() || first->op != second->op ||
            (first->op != GGML_OP_MUL_MAT && first->op != GGML_OP_MUL_MAT_ID) ||
            first->src[1] != second->src[1] || first->src[2] != second->src[2] ||
            first->type != GGML_TYPE_F32 || second->type != GGML_TYPE_F32 ||
            first->src[1]->type != GGML_TYPE_F32 ||
            !ggml_are_same_shape(first, second) || !ggml_is_contiguous(first) || !ggml_is_contiguous(second) ||
            !ggml_are_same_shape(first->src[0], second->src[0]) ||
            !mmq_pair_type(first->src[0]->type) || !mmq_pair_type(second->src[0]->type)) {
        return false;
    }
    const ggml_tensor * a = first->src[0];
    const ggml_tensor * b = second->src[0];
    const ggml_tensor * input = first->src[1];
    const ggml_tensor * ids = first->op == GGML_OP_MUL_MAT_ID ? first->src[2] : nullptr;
    const int64_t tokens = ids ? input->ne[2] : input->ne[1];
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    // quantize_mmq_q8_1[_scatter] depends on weight type ONLY through this
    // layout enum. Equal layouts have identical codes and metadata, even for
    // different weight formats. Native FP4 and rotated formats are excluded.
    if (tokens < 128 || !GGML_CUDA_CC_IS_NVIDIA(cc) ||
            mmq_get_q8_1_ds_layout(a->type) != mmq_get_q8_1_ds_layout(b->type) ||
            ggml_cuda_mmq_get_prec_src1(a->type, ggml_prec_get_src(first,  1), cc) != GGML_PREC_Q8 ||
            ggml_cuda_mmq_get_prec_src1(b->type, ggml_prec_get_src(second, 1), cc) != GGML_PREC_Q8) {
        return false;
    }
    {
        // both products run at the J_best picked for a (they share the q8_1 buffer and, with ids, tile_spec)
        const size_t smpbo    = ggml_cuda_info().devices[ctx.device].smpbo;
        const bool   fallback = ggml_cuda_mmq_needs_fallback(a->ne[1]);
        const int    J_best   = ggml_cuda_mmq_get_J_best(a->type, ggml_cuda_mmq_ncols_opt(a, input, ids, cc), fallback, cc, smpbo, GGML_PREC_Q8);
        if (J_best == 0 || ggml_cuda_mmq_needs_fallback(b->ne[1]) != fallback ||
                !ggml_cuda_mmq_J_valid(b->type, J_best, fallback, cc, smpbo, GGML_PREC_Q8)) {
            return false;
        }
    }

    ggml_cuda_pool_alloc<float> reference_first(ctx.pool());
    ggml_cuda_pool_alloc<float> reference_second(ctx.pool());
    const bool check = mmq_pair_mode() == 2;
    if (check) {
        ggml_tensor ref_first = *first;
        ggml_tensor ref_second = *second;
        ref_first.data = reference_first.alloc(ggml_nelements(first));
        ref_second.data = reference_second.alloc(ggml_nelements(second));
        ggml_cuda_mul_mat_q_impl(ctx, a, input, ids, &ref_first, false, nullptr, nullptr);
        ggml_cuda_mul_mat_q_impl(ctx, b, input, ids, &ref_second, false, nullptr, nullptr);
    }
    ggml_cuda_mul_mat_q_impl(ctx, a, input, ids, first, false, second, nullptr);
    if (check) {
        const int64_t n = ggml_nelements(first);
        mmq_pair_check<<<(n + 255)/256, 256, 0, ctx.stream()>>>((const float *) first->data, reference_first.get(), n);
        mmq_pair_check<<<(n + 255)/256, 256, 0, ctx.stream()>>>((const float *) second->data, reference_second.get(), n);
        CUDA_CHECK(cudaGetLastError());
    }
    ++ctx.fusion_stats.mmq_pair;
    return true;
}

bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    return false;
#endif // GGML_CUDA_FORCE_CUBLAS

    // [PFGAP-U3] dense quantized prompt matmuls take dequant + bf16 cuBLAS (FP32 accumulate) where detected faster
    if (n_experts == 0 && ggml_cuda_dense_tc_wants(type, ne11)) {
        return false;
    }

    bool mmq_supported;

    switch (type) {
#if !defined(GGML_USE_HIP)
        case GGML_TYPE_PTQ1_0:
            mmq_supported = turing_mma_available(cc);
            break;
#endif
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_PQ2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
// TQ4_1S is intentionally NOT listed here: should_use_mmq() must return false for it so the
// MoE MUL_MAT_ID path never routes it to an un-rotated MMQ. The dense Step-1 path calls
// ggml_cuda_mul_mat_q directly (after activation pre-rotation), bypassing this gate.
// -------------------------------------------------
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
// -------------------------------------------------
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
// -------------------------------------------------
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
            mmq_supported = true;
            break;
        default:
            mmq_supported = false;
            break;
    }

    if (!mmq_supported) {
        return false;
    }

    // MMQ tiles require at least 48 KiB per-block shared memory; fall back to BLAS otherwise.
    {
        const int    id    = ggml_cuda_get_device();
        const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
        if (smpbo < 48 * 1024) {
            return false;
        }
        if ((type == GGML_TYPE_PQ2_0 || type == GGML_TYPE_PTQ1_0) &&
                !ggml_cuda_mmq_has_config(type, 1, cc, smpbo)) {
            return false;
        }
    }

#if !defined(GGML_USE_HIP)
    if (type == GGML_TYPE_PTQ1_0) {
        // the fp16 dequantize + cuBLAS fallback is the source of PTQ1_0's extra error on CUDA, so
        // the MMQ tile path runs at every batch by default; the env var is the A/B knob for
        // deployments that prefer cuBLAS's ~7% at pp512 over the accuracy
        static const int64_t max_batch = [] {
            const char * s = getenv("GGML_CUDA_PTQ1_0_MMQ_MAX_BATCH");
            return s ? (int64_t) atoll(s) : (int64_t) MMQ_PTQ1_0_MAX_BATCH_SIZE;
        }();
        return ne11 <= max_batch;
    }
#endif

    if (turing_mma_available(cc)) {
        return true;
    }

    if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {
        // for MoE, mmq is faster even without native dp4a
        // TODO: check if cards older than pascal might benefit from this as well
        return cc >= GGML_CUDA_CC_PASCAL && n_experts > 0;
    }

#ifdef GGML_CUDA_FORCE_MMQ
    return true;
#endif //GGML_CUDA_FORCE_MMQ

    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }

    if (amd_mfma_available(cc)) {
        // As of ROCM 7.0 rocblas/tensile performs very poorly on CDNA3 and hipblaslt (via ROCBLAS_USE_HIPBLASLT)
        // performs better but is currently suffering from a crash on this architecture.
        // TODO: Revisit when hipblaslt is fixed on CDNA3
        if (GGML_CUDA_CC_IS_CDNA3(cc)) {
            return true;
        }
        if (n_experts > 64 || ne11 <= 128) {
            return true;
        }
        if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1) {
            return true;
        }
        if (ne11 <= 256 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) {
            return true;
        }
        return false;
    }

    if (amd_wmma_available(cc)) {
        if (GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4_M(cc)) {
            // High expert counts are almost always better on MMQ due to
            //     the synchronization overhead in the cuBLAS/hipBLAS path:
            // https://github.com/ggml-org/llama.cpp/pull/18202
            if (n_experts >= 64) {
                return true;
            }

            // For some quantization types MMQ can have lower peak TOPS than hipBLAS
            //     so it's only faster for sufficiently small batch sizes:
            switch (type) {
                case GGML_TYPE_Q2_K:
                    return ne11 <= 128;
                case GGML_TYPE_Q6_K:
                    return ne11 <= (GGML_CUDA_CC_IS_RDNA3_0(cc) ? 128 : 256);
                case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ2_S:
                    return GGML_CUDA_CC_IS_RDNA3_5(cc) || GGML_CUDA_CC_IS_RDNA4_M(cc) || ne11 <= 128;
                default:
                    return true;
            }
        }

        // For RDNA4 MMQ is consistently faster than dequantization + hipBLAS:
        // https://github.com/ggml-org/llama.cpp/pull/18537#issuecomment-3706422301
        return true;
    }

    // gfx900 (Vega 10), gfx909, and gfx90c lack native dp4a, losing to dequant + hipBLAS
    // for dense matrices; keep MMQ only for MoE, where the
    // hipBLAS path is much slower.
    if (cc == GGML_CUDA_CC_VEGA || GGML_CUDA_CC_IS_GCN_APU(cc)) {
        return n_experts > 0;
    }

    // MUSA: the MMQ kernels compute wrong values on PH1 (MTT S5000).
    if (cc == GGML_CUDA_CC_PH1) {
        return false;
    }

    return (!GGML_CUDA_CC_IS_CDNA(cc)) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
}

bool ggml_cuda_mmq_fusion_is_efficient(
        const ggml_type type, const ggml_prec prec_requested, const int64_t nrows_x, const int64_t ncols, const int64_t nchannels, const int device) {
    // fork: upstream's MMQ gate/up/GLU fusion is opt-in (GGML_CUDA_MMQ_FUSION=1). The fused kernel always tiles (one
    // block per tile) while the unfused MUL_MATs may split K over stream-k blocks, so results can differ from v0.6.3.
    static const bool fusion_on = [] { const char * e = getenv("GGML_CUDA_MMQ_FUSION"); return e && atoi(e) != 0; }();
    if (!fusion_on) {
        return false;
    }

    const int    cc    = ggml_cuda_info().devices[device].cc;
    const int    nsm   = ggml_cuda_info().devices[device].nsm;
    const size_t smpbo = ggml_cuda_info().devices[device].smpbo;

    // Same tile width as ggml_cuda_mul_mat_q; the fused kernel has no fallback and writes I/2 output rows per tile.
    const bool      fallback  = false;
    const ggml_prec prec_src1 = ggml_cuda_mmq_get_prec_src1(type, prec_requested, cc);
    const int       J_best    = ggml_cuda_mmq_get_J_best(type, ncols, fallback, cc, smpbo, prec_src1);
    if (J_best == 0) {
        return false;
    }
    const int64_t nrows_tile = ggml_cuda_mmq_get_config(type, J_best, fallback, cc, prec_src1).I / 2;
    const int64_t ntiles     = (ncols + J_best - 1)/J_best * ((nrows_x + nrows_tile - 1)/nrows_tile) * nchannels;
    // Same condition as launch_mul_mat_q uses to run the stream-k kernel with one block per tile.
    const int64_t tiles_nwaves             = (ntiles + nsm - 1) / nsm;
    const int64_t tiles_efficiency_percent = 100 * ntiles / (nsm*tiles_nwaves);
    return tiles_efficiency_percent >= 90;
}
