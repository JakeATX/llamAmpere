#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"
#include "convert.cuh"
#include "gated_delta_net_chunked_tc.cuh"

#include <cstdlib>
#include <algorithm>
#include <vector>

// store state element e (counted in elements from `state`). Plain types convert (RNE for bf16/f16).
// q8_0 is warp-collective: the calling warp's 32 lanes hold the 32 elements of one block
// (e = block start + lane); they agree on the block scale through a max-reduction and each lane
// writes its own quant -- the same arithmetic as quantize_f32_q8_0_block (the unfused cpy/set_rows).
template <typename state_t, int warp_size>
static __device__ __forceinline__ void gdn_store_state(void * state, const int64_t e, const float x) {
    if constexpr (std::is_same_v<state_t, block_q8_0>) {
        if constexpr (warp_size == QK8_0) {
            const float amax = warp_reduce_max<QK8_0>(fabsf(x));
            const float d    = amax / ((1 << 7) - 1);
            const float id   = d ? 1.0f/d : 0.0f;
            block_q8_0 * b = (block_q8_0 *) state + e / QK8_0;
            const int    j = (int) (e % QK8_0);
            b->qs[j] = roundf(x*id);
            if (j == 0) {
                b->d = d;
            }
        } else {
            GGML_UNUSED_VARS(state, e, x);
            NO_DEVICE_CODE; // never fused: needs 32-wide warps and S_v % 32 == 0 (ggml_cuda_gdn_cache_type_ok)
        }
    } else {
        ((state_t *) state)[e] = ggml_cuda_cast<state_t>(x);
    }
}

#include "gated_delta_net_chunked_argon.cuh"

// emit_ingredients_t (only meaningful when keep_rs_t): write per-token (k,v,g,beta) ingredients
// instead of a full [S_v,S_v] state snapshot per retained slot, plus one fixed-cost trailing
// final-state block -- see ggml_gated_delta_net's emit_mode==1/2 contract (ggml.h) and the CPU
// reference (ggml-cpu/ops.cpp). Never instantiated with keep_rs_t == false. ingr_w is the slot
// width per head (4*S_v, or emit_mode 2's compact 2*S_v + g width + 1), ingr_region the floats
// from `state` to the trailing final-state block.
// state_t is the type of the state output: float for the op's own dst tail, or the recurrent cache's
// type (f32/bf16/f16/q8_0, --cache-type-s) when the snapshot copy is fused (ggml_cuda_try_gdn_cache_fusion).
// The recurrence itself always runs in F32 registers; only the stored state is rounded. `state` is
// addressed in elements (gdn_store_state), so a q8_0 block is 32 consecutive elements of one column.
template <int S_v, bool KDA, bool keep_rs_t, bool emit_ingredients_t, typename state_t>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     void *        state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K,
                                     const int32_t * slot_rows,
                                     int           ingr_w,
                                     int64_t       ingr_region,
                                     const ggml_cuda_gated_delta_net_state_src state_src) {
    // ingredient slots and the trailing blocks live in the op's own F32 output (never fused)
    static_assert(!emit_ingredients_t || std::is_same_v<state_t, float>, "emit_mode 1/2 writes F32 only");

    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output layout (per-slot stride passed in as state_slot_stride) — same per-(seq,head)
    // offset scheme as before, except emit_ingredients_t slots are 4*S_v wide (k,v,g,beta)
    // instead of S_v*S_v (a full state matrix); state_out points at the base of THIS
    // (seq,head)'s slot-0 storage either way. `state` (unoffset) is kept around separately so
    // the trailing final-state block (always S_v*S_v-shaped, at a fixed offset past all K
    // ingredient slots) can be addressed too.
    // [#87] state_src.rows: this sequence's input state is cache row rows[sequence] (read after the
    // PDL sync below, the rows are an input of the graph); otherwise src[5] at sequence * D
    const int64_t state_in_offset = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    // state_out_e: element offset of THIS (seq,head)'s slot-0 storage in `state`
    const int64_t state_out_e = emit_ingredients_t ? (sequence * H + h_idx) * (int64_t) ingr_w : (sequence * H + h_idx) * S_v * S_v;
    float * const state_f     = (float *) state; // emit_ingredients_t: the op's own F32 output

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
    curr_state += (state_src.rows != nullptr
                       ? (int64_t) state_src.rows[sequence] * state_src.row_stride + h_idx * S_v * S_v
                       : state_in_offset) + col * S_v;
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // emit_ingredients_t==false: most-recent-first (slot 0 = final state), matching the
            // s_copy/rs_idx row-selection convention used elsewhere. emit_ingredients_t==true:
            // chronological (slot 0 = oldest of the K retained tokens) -- ingredients are only
            // ever consumed by this op's own replay call site, which needs a straight forward
            // prefix view, not a reversed one; see the CPU reference for the full rationale.
            const int target_slot = emit_ingredients_t ? ((int) t - ((int) n_tokens - K)) : ((int) n_tokens - 1 - t);
            if (target_slot >= 0 && target_slot < K) {
                if constexpr (emit_ingredients_t) {
                    // ingredients: k (offset 0, full vector), v (offset S_v, this warp's column
                    // only), g (offset 2*S_v, raw/non-exponentiated), beta (offset 3*S_v). k/g/beta
                    // don't vary by column, so only one warp (col == 0) needs to write them --
                    // every warp doing so redundantly would cost O(S_v) writes each, i.e. O(S_v^2)
                    // total per component, which is what this design is supposed to avoid.
                    // emit_mode 2 (compact) stores g and beta once: k, v, g (1 or S_v), beta.
                    float * ingr_slot = state_f + state_out_e + (int64_t) target_slot * state_slot_stride;
                    const bool compact = ingr_w != 4 * S_v;
                    if (col == 0) {
#pragma unroll
                        for (int r = 0; r < rows_per_lane; r++) {
                            const int i    = r * warp_size + lane;
                            ingr_slot[i]   = k_reg[r];
                            if constexpr (KDA) {
                                ingr_slot[2 * S_v + i] = g_t[i];
                            } else if (!compact) {
                                ingr_slot[2 * S_v + i] = *g_t;
                            }
                            if (!compact) {
                                ingr_slot[3 * S_v + i] = beta_val;
                            }
                        }
                        if (compact && lane == 0) {
                            if constexpr (!KDA) {
                                ingr_slot[2 * S_v] = *g_t;
                            }
                            ingr_slot[2 * S_v + (KDA ? S_v : 1)] = beta_val;
                        }
                    }
                    if (lane == 0) {
                        ingr_slot[S_v + col] = v_t[col];
                    }
                } else {
                    const int64_t snap_e = slot_rows == nullptr
                        ? state_out_e + (int64_t) target_slot * state_slot_stride
                        : (int64_t) slot_rows[target_slot * n_seqs + sequence] * (H * S_v * S_v) + h_idx * S_v * S_v;
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        const int i = r * warp_size + lane;
                        gdn_store_state<state_t, warp_size>(state, snap_e + col * S_v + i, s_shard[r]);
                    }
                }
            }

            if constexpr (emit_ingredients_t) {
                // state immediately before the K-token retained window starts, captured inline
                // (free relative to a second op call over the same prefix) when n_tokens > K.
                const int t_ckpt = (int) n_tokens - K - 1;
                if (t_ckpt >= 0 && t == t_ckpt) {
                    float * ckpt_slot = state_f + ingr_region +
                                         S_v * S_v * H * n_seqs + (sequence * H + h_idx) * S_v * S_v;
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        const int i = r * warp_size + lane;
                        ckpt_slot[col * S_v + i] = s_shard[r];
                    }
                }
            }
        }
    }

    if constexpr (emit_ingredients_t) {
        // fixed, once-per-call cost (not scaled by K): the true final state, for use as the next
        // decode's optimistic base state / this decode's own checkpoint update.
        float * final_slot = state_f + ingr_region + (sequence * H + h_idx) * S_v * S_v;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            final_slot[col * S_v + i] = s_shard[r];
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            gdn_store_state<state_t, warp_size>(state, state_out_e + col * S_v + i, s_shard[r]);
        }
    }
}


// same per-column math as gated_delta_net_cuda, but each warp owns NC state columns so the
// NC reduction chains interleave; multi-token, scalar-gate, S_v == 128 only (GGML_CUDA_SM86_GDN_COLS)
template <int S_v, bool keep_rs_t, int NC, bool PREFETCH, typename state_t>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda_ilp(const float * q,
                         const float * k,
                         const float * v,
                         const float * g,
                         const float * beta,
                         const float * curr_state,
                         float *       dst,
                         void *        state,
                         int64_t       H,
                         int64_t       n_tokens,
                         int64_t       n_seqs,
                         int64_t       sq1,
                         int64_t       sq2,
                         int64_t       sq3,
                         int64_t       sv1,
                         int64_t       sv2,
                         int64_t       sv3,
                         int64_t       sb1,
                         int64_t       sb2,
                         int64_t       sb3,
                         const uint3   neqk1_magic,
                         const uint3   rq3_magic,
                         float         scale,
                         int64_t       state_slot_stride,
                         int           K,
                         const int32_t * slot_rows,
                         const ggml_cuda_gated_delta_net_state_src state_src) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      lane     = threadIdx.x;
    const int      col0     = (blockIdx.z * blockDim.y + threadIdx.y) * NC;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float * attn_data = dst;

    const int64_t state_in_offset  = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_e      = (sequence * H + h_idx) * S_v * S_v; // element offset in `state`
    attn_data  += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = S_v / warp_size;

    float s_shard[NC][rows_per_lane];

    ggml_cuda_pdl_sync();
    // [#87] see gated_delta_net_cuda: cache row rows[sequence], or src[5] at sequence * D
    curr_state += (state_src.rows != nullptr
                       ? (int64_t) state_src.rows[sequence] * state_src.row_stride + h_idx * S_v * S_v
                       : state_in_offset) + col0 * S_v;
#pragma unroll
    for (int c = 0; c < NC; c++) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[c][r] = curr_state[c * S_v + r * warp_size + lane];
        }
    }

    const float * q_base = q + iq3 * sq3 + iq1 * sq1;
    const float * k_base = k + iq3 * sq3 + iq1 * sq1;
    const float * v_base = v + sequence * sv3 + h_idx * sv1 + col0;
    const float * b_base = beta + sequence * sb3 + h_idx * sb1;
    const float * g_base = g    + sequence * sb3 + h_idx * sb1;

    float k_reg[rows_per_lane];
    float q_reg[rows_per_lane];
    float v_reg[NC];
    float g_raw;
    float beta_val;

    auto load_tok = [&](int t, float * kr, float * qr, float * vr, float & gr, float & br) {
        const float * q_t = q_base + t * sq2;
        const float * k_t = k_base + t * sq2;
        const float * v_t = v_base + t * sv2;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            kr[r] = k_t[r * warp_size + lane];
            qr[r] = q_t[r * warp_size + lane];
        }
#pragma unroll
        for (int c = 0; c < NC; c++) {
            vr[c] = v_t[c];
        }
        gr = g_base[t * sb2];
        br = b_base[t * sb2];
    };

    if constexpr (PREFETCH) {
        load_tok(0, k_reg, q_reg, v_reg, g_raw, beta_val);
    }

    for (int t = 0; t < n_tokens; t++) {
        float kn_reg[rows_per_lane];
        float qn_reg[rows_per_lane];
        float vn_reg[NC];
        float gn_raw = 0.0f;
        float bn_val = 0.0f;
        if constexpr (PREFETCH) {
            if (t + 1 < n_tokens) {
                load_tok(t + 1, kn_reg, qn_reg, vn_reg, gn_raw, bn_val);
            }
        } else {
            load_tok(t, k_reg, q_reg, v_reg, g_raw, beta_val);
        }

        const float g_val = expf(g_raw);

        // kv[col] = sum_i S[i][col] * k[i]   (same order as the original kernel, per column)
        float kv_shard[NC];
#pragma unroll
        for (int c = 0; c < NC; c++) {
            kv_shard[c] = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard[c] += s_shard[c][r] * k_reg[r];
            }
        }
        float kv_col[NC];
#pragma unroll
        for (int c = 0; c < NC; c++) {
            kv_col[c] = warp_reduce_sum<warp_size>(kv_shard[c]);
        }

        float attn_partial[NC];
#pragma unroll
        for (int c = 0; c < NC; c++) {
            const float delta_col = (v_reg[c] - g_val * kv_col[c]) * beta_val;
            attn_partial[c] = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[c][r]    = g_val * s_shard[c][r] + k_reg[r] * delta_col;
                attn_partial[c] += s_shard[c][r] * q_reg[r];
            }
        }
        float attn_col[NC];
#pragma unroll
        for (int c = 0; c < NC; c++) {
            attn_col[c] = warp_reduce_sum<warp_size>(attn_partial[c]);
        }

        if (lane == 0) {
#pragma unroll
            for (int c = 0; c < NC; c++) {
                attn_data[col0 + c] = attn_col[c] * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                const int64_t snap_e = slot_rows == nullptr
                    ? state_out_e + (int64_t) target_slot * state_slot_stride
                    : (int64_t) slot_rows[target_slot * n_seqs + sequence] * (H * S_v * S_v) + h_idx * S_v * S_v;
#pragma unroll
                for (int c = 0; c < NC; c++) {
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        gdn_store_state<state_t, warp_size>(state, snap_e + (col0 + c) * S_v + r * warp_size + lane, s_shard[c][r]);
                    }
                }
            }
        }

        if constexpr (PREFETCH) {
            if (t + 1 < n_tokens) {
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    k_reg[r] = kn_reg[r];
                    q_reg[r] = qn_reg[r];
                }
#pragma unroll
                for (int c = 0; c < NC; c++) {
                    v_reg[c] = vn_reg[c];
                }
                g_raw    = gn_raw;
                beta_val = bn_val;
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int c = 0; c < NC; c++) {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                gdn_store_state<state_t, warp_size>(state, state_out_e + (col0 + c) * S_v + r * warp_size + lane, s_shard[c][r]);
            }
        }
    }
}

// [FNSTRATA] GGML_CUDA_GDN_R16=1 (opt-in; S_v 128, f32/bf16/f16 state, final state or K snapshot slots): the same math as the kernels
// above in a layout with far fewer warp shuffles. Above, a column sum is a per-lane partial over 4 rows (rows r*32 + lane,
// r = 0..3, in order) followed by a butterfly over the 32 lanes at offsets 16, 8, 4, 2, 1. Here one thread holds the rows
// of the four lanes j, j+8, j+16, j+24 (rows j + 8n, n = 0..15) of C columns, adds the offset-16 and offset-8 pairs in
// registers and shuffles only at offsets 4, 2, 1, inside its 8-lane group; one shuffle serves the 4 groups of a warp.
// IEEE addition is commutative, so every partial, every pair and every total is the same float, bit for bit. k and q of a
// token are staged in shared memory one token ahead, permuted so that a thread reads its 16 rows with four 128-bit loads.
// GGML_CUDA_GDN_R16=2 also runs the default kernel into scratch buffers and counts differing output and state words
// (synchronizes; testing only). GGML_CUDA_GDN_R16_COLS (2 or 4, default 2) and _WARPS (1, 2 or 4 per block, default 2).
template <int C, int NW, bool keep_rs_t, typename state_t>
__global__ void __launch_bounds__(32*NW)
gated_delta_net_cuda_r16(const float * q,
                         const float * k,
                         const float * v,
                         const float * g,
                         const float * beta,
                         const float * curr_state,
                         float *       dst,
                         void *        state,
                         int64_t       H,
                         int64_t       n_tokens,
                         int64_t       sq1,
                         int64_t       sq2,
                         int64_t       sq3,
                         int64_t       sv1,
                         int64_t       sv2,
                         int64_t       sv3,
                         int64_t       sb1,
                         int64_t       sb2,
                         int64_t       sb3,
                         const uint3   neqk1_magic,
                         const uint3   rq3_magic,
                         float         scale,
                         int64_t       n_seqs,
                         int64_t       state_slot_stride,
                         int           K,
                         const int32_t * slot_rows,
                         const ggml_cuda_gated_delta_net_state_src state_src) {
    constexpr int S_v = 128;
    constexpr int PAD = 20; // floats per row group in shared memory: 16 used; 20 keeps the 128-bit reads conflict-free
    __shared__ __align__(16) float s_k[2][8*PAD];
    __shared__ __align__(16) float s_q[2][8*PAD];

    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      lane     = threadIdx.x;
    const int      tid      = threadIdx.y*32 + lane;
    const int      j        = lane & 7; // rows j + 8n
    const int      col0     = (blockIdx.z*NW + threadIdx.y)*(4*C) + (lane >> 3)*C;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const int64_t state_in_offset = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_e     = (sequence * H + h_idx) * S_v * S_v;
    float * attn_data = dst + (sequence * n_tokens * H + h_idx) * S_v;

    // s[c][4r + m] = S[row r*32 + j + 8m][col0 + c]: element r of the original lane j + 8m
    float s[C][16];

    ggml_cuda_pdl_sync();
    curr_state += (state_src.rows != nullptr
                       ? (int64_t) state_src.rows[sequence] * state_src.row_stride + h_idx * S_v * S_v
                       : state_in_offset) + col0 * S_v;
#pragma unroll
    for (int c = 0; c < C; c++) {
#pragma unroll
        for (int n = 0; n < 16; n++) {
            s[c][n] = curr_state[c * S_v + j + 8*n];
        }
    }

    const float * q_base = q + iq3 * sq3 + iq1 * sq1;
    const float * k_base = k + iq3 * sq3 + iq1 * sq1;
    const float * v_base = v + sequence * sv3 + h_idx * sv1 + col0;
    const float * b_base = beta + sequence * sb3 + h_idx * sb1;
    const float * g_base = g    + sequence * sb3 + h_idx * sb1;

    // staging: 64 float4 per token (32 of k, then 32 of q), spread over the block
    constexpr int NLD = (64 + 32*NW - 1) / (32*NW);
    float4 ld[NLD];
    auto stage_load = [&](const int t) {
#pragma unroll
        for (int i = 0; i < NLD; i++) {
            const int f = tid + i*32*NW;
            if (f < 64) {
                ld[i] = *(const float4 *) ((f < 32 ? k_base : q_base) + t * sq2 + 4*(f & 31));
            }
        }
    };
    auto stage_store = [&](const int buf) {
#pragma unroll
        for (int i = 0; i < NLD; i++) {
            const int f = tid + i*32*NW;
            if (f < 64) {
                float * sm = f < 32 ? s_k[buf] : s_q[buf];
                const int e0 = 4*(f & 31); // row of ld[i].x; rows e go to [e % 8][e / 8]
                sm[((e0 + 0) & 7)*PAD + ((e0 + 0) >> 3)] = ld[i].x;
                sm[((e0 + 1) & 7)*PAD + ((e0 + 1) >> 3)] = ld[i].y;
                sm[((e0 + 2) & 7)*PAD + ((e0 + 2) >> 3)] = ld[i].z;
                sm[((e0 + 3) & 7)*PAD + ((e0 + 3) >> 3)] = ld[i].w;
            }
        }
    };

    float v_cur[C];
    float g_cur;
    float b_cur;
    stage_load(0);
#pragma unroll
    for (int c = 0; c < C; c++) {
        v_cur[c] = v_base[c];
    }
    g_cur = g_base[0];
    b_cur = b_base[0];
    stage_store(0);
    __syncthreads();

    for (int t = 0; t < n_tokens; t++) {
        const int buf = t & 1;
        float v_nxt[C];
        float g_nxt = 0.0f;
        float b_nxt = 0.0f;
        if (t + 1 < n_tokens) {
            stage_load(t + 1);
#pragma unroll
            for (int c = 0; c < C; c++) {
                v_nxt[c] = v_base[(t + 1) * sv2 + c];
            }
            g_nxt = g_base[(t + 1) * sb2];
            b_nxt = b_base[(t + 1) * sb2];
        }

        const float g_val = expf(g_cur);

        // kr[r] = k rows r*32 + j + 8m, m = 0..3 (x, y, z, w)
        float4 kr[4];
#pragma unroll
        for (int r = 0; r < 4; r++) {
            kr[r] = *(const float4 *) &s_k[buf][j*PAD + 4*r];
        }

        // kv[col] = sum_i S[i][col] * k[i]: the lane partials, then the butterfly in the original pairing
        float kv_col[C];
#pragma unroll
        for (int c = 0; c < C; c++) {
            float p0 = 0.0f;
            float p1 = 0.0f;
            float p2 = 0.0f;
            float p3 = 0.0f;
#pragma unroll
            for (int r = 0; r < 4; r++) {
                p0 += s[c][4*r + 0] * kr[r].x;
                p1 += s[c][4*r + 1] * kr[r].y;
                p2 += s[c][4*r + 2] * kr[r].z;
                p3 += s[c][4*r + 3] * kr[r].w;
            }
            float x = (p0 + p2) + (p1 + p3); // offset 16 (lanes j / j+16, j+8 / j+24), then offset 8
            x += __shfl_xor_sync(0xffffffff, x, 4, 32);
            x += __shfl_xor_sync(0xffffffff, x, 2, 32);
            x += __shfl_xor_sync(0xffffffff, x, 1, 32);
            kv_col[c] = x;
        }

        float4 qr[4];
#pragma unroll
        for (int r = 0; r < 4; r++) {
            qr[r] = *(const float4 *) &s_q[buf][j*PAD + 4*r];
        }

        float attn_col[C];
#pragma unroll
        for (int c = 0; c < C; c++) {
            const float delta_col = (v_cur[c] - g_val * kv_col[c]) * b_cur;
            float a0 = 0.0f;
            float a1 = 0.0f;
            float a2 = 0.0f;
            float a3 = 0.0f;
#pragma unroll
            for (int r = 0; r < 4; r++) {
                s[c][4*r + 0] = g_val * s[c][4*r + 0] + kr[r].x * delta_col;
                a0 += s[c][4*r + 0] * qr[r].x;
                s[c][4*r + 1] = g_val * s[c][4*r + 1] + kr[r].y * delta_col;
                a1 += s[c][4*r + 1] * qr[r].y;
                s[c][4*r + 2] = g_val * s[c][4*r + 2] + kr[r].z * delta_col;
                a2 += s[c][4*r + 2] * qr[r].z;
                s[c][4*r + 3] = g_val * s[c][4*r + 3] + kr[r].w * delta_col;
                a3 += s[c][4*r + 3] * qr[r].w;
            }
            float x = (a0 + a2) + (a1 + a3);
            x += __shfl_xor_sync(0xffffffff, x, 4, 32);
            x += __shfl_xor_sync(0xffffffff, x, 2, 32);
            x += __shfl_xor_sync(0xffffffff, x, 1, 32);
            attn_col[c] = x;
        }

        if (j == 0) {
#pragma unroll
            for (int c = 0; c < C; c++) {
                attn_data[col0 + c] = attn_col[c] * scale;
            }
        }
        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slots for the last K tokens, addressed exactly as in gated_delta_net_cuda_ilp
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                const int64_t snap_e = slot_rows == nullptr
                    ? state_out_e + (int64_t) target_slot * state_slot_stride
                    : (int64_t) slot_rows[target_slot * n_seqs + sequence] * (H * S_v * S_v) + h_idx * S_v * S_v;
#pragma unroll
                for (int c = 0; c < C; c++) {
#pragma unroll
                    for (int n = 0; n < 16; n++) {
                        gdn_store_state<state_t, 32>(state, snap_e + (int64_t) (col0 + c) * S_v + j + 8*n, s[c][n]);
                    }
                }
            }
        }

        if (t + 1 < n_tokens) {
            stage_store(buf ^ 1);
#pragma unroll
            for (int c = 0; c < C; c++) {
                v_cur[c] = v_nxt[c];
            }
            g_cur = g_nxt;
            b_cur = b_nxt;
        }
        __syncthreads();
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int c = 0; c < C; c++) {
#pragma unroll
            for (int n = 0; n < 16; n++) {
                gdn_store_state<state_t, 32>(state, state_out_e + (int64_t) (col0 + c) * S_v + j + 8*n, s[c][n]);
            }
        }
    } else {
        GGML_UNUSED(state_out_e);
    }
}

static __global__ void gdn_r16_diff(const uint32_t * __restrict__ a, const uint32_t * __restrict__ b, const int64_t n,
                                    unsigned long long * __restrict__ ndiff) {
    unsigned long long c = 0;
    for (int64_t i = blockIdx.x*(int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x*blockDim.x) {
        c += a[i] != b[i];
    }
    if (c) {
        atomicAdd(ndiff, c);
    }
}

// Four columns per physical warp, with 16 state floats per lane in either layout.
template <int SW, bool keep_rs_t, bool emit_ingredients_t>
__global__ void __launch_bounds__(128, 2)
gated_delta_net_cuda_subwarp(
        const float * q, const float * k, const float * v, const float * g, const float * beta,
        const float * curr_state, float * dst, float * state,
        int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3,
        uint3 neqk1_magic, uint3 rq3_magic, float scale,
        int64_t state_slot_stride, int K, const int32_t * slot_rows,
        int ingr_w, int64_t ingr_region, ggml_cuda_gated_delta_net_state_src state_src) {
    static_assert(SW == 16 || SW == 8, "unsupported subwarp width");
    constexpr int D = 128;
    constexpr int NC = SW / 8;
    constexpr int NR = D / SW;
    const int lane = threadIdx.x % SW;
    const int col0 = (blockIdx.z * blockDim.y + threadIdx.y) * 4 + (threadIdx.x / SW) * NC;
    const int64_t h = blockIdx.x;
    const int64_t seq = blockIdx.y;
    const int64_t head = seq * H + h;
    const uint32_t iq1 = fastmodulo(h, neqk1_magic);
    const uint32_t iq3 = fastdiv(seq, rq3_magic);

    q += iq3 * sq3 + iq1 * sq1 + lane;
    k += iq3 * sq3 + iq1 * sq1 + lane;
    v += seq * sv3 + h * sv1 + col0;
    g += seq * sb3 + h * sb1;
    beta += seq * sb3 + h * sb1;
    dst += (seq * n_tokens * H + h) * D + col0;
    const int64_t state_out_e = head * (emit_ingredients_t ? ingr_w : D * D);
    float s[NC][NR];

    ggml_cuda_pdl_sync();
    curr_state += (state_src.rows != nullptr
        ? (int64_t) state_src.rows[seq] * state_src.row_stride + h * D * D
        : head * D * D) + col0 * D + lane;
#pragma unroll
    for (int c = 0; c < NC; ++c) {
#pragma unroll
        for (int r = 0; r < NR; ++r) {
            s[c][r] = curr_state[c * D + r * SW];
        }
    }

    for (int t = 0; t < n_tokens; ++t) {
        float kr[NR], qr[NR], vr[NC];
#pragma unroll
        for (int r = 0; r < NR; ++r) {
            kr[r] = k[r * SW];
            qr[r] = q[r * SW];
        }
        if constexpr (NC == 2) {
            const float2 vv = *reinterpret_cast<const float2 *>(v);
            vr[0] = vv.x;
            vr[1] = vv.y;
        } else {
            vr[0] = *v;
        }
        const float g_raw = *g;
        const float decay = expf(g_raw);
        const float b = *beta;
        float out[NC];
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            // Independent local sums shorten the longer per-lane dot products.
            float p0 = 0.0f, p1 = 0.0f;
#pragma unroll
            for (int r = 0; r < NR; r += 2) {
                p0 += s[c][r    ] * kr[r    ];
                p1 += s[c][r + 1] * kr[r + 1];
            }
            const float kv = warp_reduce_sum<SW>(p0 + p1);
            const float delta = (vr[c] - decay * kv) * b;
            p0 = 0.0f;
            p1 = 0.0f;
#pragma unroll
            for (int r = 0; r < NR; r += 2) {
                s[c][r    ] = decay * s[c][r    ] + kr[r    ] * delta;
                s[c][r + 1] = decay * s[c][r + 1] + kr[r + 1] * delta;
                p0 += s[c][r    ] * qr[r    ];
                p1 += s[c][r + 1] * qr[r + 1];
            }
            out[c] = warp_reduce_sum<SW>(p0 + p1) * scale;
        }
        if (lane == 0) {
            if constexpr (NC == 2) {
                *reinterpret_cast<float2 *>(dst) = make_float2(out[0], out[1]);
            } else {
                *dst = out[0];
            }
        }

        if constexpr (keep_rs_t) {
            const int slot = emit_ingredients_t ? t - (n_tokens - K) : n_tokens - 1 - t;
            if (slot >= 0 && slot < K) {
                if constexpr (emit_ingredients_t) {
                    float * ingr = state + state_out_e + (int64_t) slot * state_slot_stride;
                    const bool compact = ingr_w != 4 * D;
                    if (col0 == 0) {
#pragma unroll
                        for (int r = 0; r < NR; ++r) {
                            const int i = r * SW + lane;
                            ingr[i] = kr[r];
                            if (!compact) {
                                ingr[2 * D + i] = g_raw;
                                ingr[3 * D + i] = b;
                            }
                        }
                        if (compact && lane == 0) {
                            ingr[2 * D] = g_raw;
                            ingr[2 * D + 1] = b;
                        }
                    }
                    if (lane == 0) {
                        if constexpr (NC == 2) {
                            *reinterpret_cast<float2 *>(ingr + D + col0) = make_float2(vr[0], vr[1]);
                        } else {
                            ingr[D + col0] = vr[0];
                        }
                    }
                } else {
                    const int64_t snap_e = slot_rows == nullptr
                        ? state_out_e + (int64_t) slot * state_slot_stride
                        : (int64_t) slot_rows[slot * n_seqs + seq] * (H * D * D) + h * D * D;
#pragma unroll
                    for (int c = 0; c < NC; ++c) {
#pragma unroll
                        for (int r = 0; r < NR; ++r) {
                            state[snap_e + (col0 + c) * D + r * SW + lane] = s[c][r];
                        }
                    }
                }
            }
            if constexpr (emit_ingredients_t) {
                if (t == n_tokens - K - 1) {
                    float * ckpt = state + ingr_region + D * D * H * n_seqs + head * D * D;
#pragma unroll
                    for (int c = 0; c < NC; ++c) {
#pragma unroll
                        for (int r = 0; r < NR; ++r) {
                            ckpt[(col0 + c) * D + r * SW + lane] = s[c][r];
                        }
                    }
                }
            }
        }
        q += sq2;
        k += sq2;
        v += sv2;
        g += sb2;
        beta += sb2;
        dst += D * H;
    }
    if constexpr (!keep_rs_t || emit_ingredients_t) {
        float * final_state = state + (emit_ingredients_t ? ingr_region + head * D * D : state_out_e);
#pragma unroll
        for (int c = 0; c < NC; ++c) {
#pragma unroll
            for (int r = 0; r < NR; ++r) {
                final_state[(col0 + c) * D + r * SW + lane] = s[c][r];
            }
        }
    }
}

static int ggml_cuda_gdn_subwarp() {
    static const int width = [] {
        const char * s = getenv("GGML_CUDA_GDN_SUBWARP");
        return s != nullptr && (strcmp(s, "16") == 0 || strcmp(s, "8") == 0) ? atoi(s) : 0;
    }();
    return width;
}

template <bool keep_rs_t, bool emit_ingredients_t>
static void launch_gated_delta_net_subwarp(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, void * state_d,
        int64_t S_v, int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, const int32_t * slot_rows,
        int ingr_w, int64_t ingr_region, ggml_cuda_gated_delta_net_state_src state_src, cudaStream_t stream) {
    GGML_ASSERT(S_v == 128);
    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic = init_fastdiv_values(rq3);
    const ggml_cuda_kernel_launch_params params(dim3(H, n_seqs, 8), dim3(32, 4, 1), 0, stream);
#define GDN_SUBWARP_LAUNCH(SW_) \
    ggml_cuda_kernel_launch(gated_delta_net_cuda_subwarp<SW_, keep_rs_t, emit_ingredients_t>, params, \
        q_d, k_d, v_d, g_d, b_d, s_d, dst_d, (float *) state_d, H, n_tokens, n_seqs, \
        sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, \
        state_slot_stride, K, slot_rows, ingr_w, ingr_region, state_src)
    if (ggml_cuda_gdn_subwarp() == 16) {
        GDN_SUBWARP_LAUNCH(16);
    } else {
        GDN_SUBWARP_LAUNCH(8);
    }
#undef GDN_SUBWARP_LAUNCH
}

// default 4 (qualified on sm86); 1 selects the original kernel
static int ggml_cuda_sm86_gdn_cols() {
    static const int cols = [] {
        const char * s = getenv("GGML_CUDA_SM86_GDN_COLS");
        const int v = s ? atoi(s) : 4;
        return (v == 1 || v == 2 || v == 4 || v == 8) ? v : 4;
    }();
    return cols;
}

static bool ggml_cuda_sm86_gdn_prefetch() {
    static const bool on = [] {
        const char * s = getenv("GGML_CUDA_SM86_GDN_PREFETCH");
        return s != nullptr && atoi(s) != 0;
    }();
    return on;
}

template <bool keep_rs_t, int NC, bool PREFETCH, typename state_t>
static void launch_gated_delta_net_ilp_inst(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, void * state_d,
        int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, const int32_t * slot_rows,
        const ggml_cuda_gated_delta_net_state_src state_src, cudaStream_t stream) {
    constexpr int S_v = 128;
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    GGML_ASSERT(S_v % (num_warps * NC) == 0);
    dim3 grid_dims(H, n_seqs, S_v / (num_warps * NC));
    dim3 block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    ggml_cuda_kernel_launch(gated_delta_net_cuda_ilp<S_v, keep_rs_t, NC, PREFETCH, state_t>, launch_params,
        q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
        n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
        sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, slot_rows, state_src);
}

// returns false when this path does not apply; caller falls back to the original kernel
template <bool keep_rs_t, typename state_t>
static bool launch_gated_delta_net_ilp(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, void * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, const int32_t * slot_rows,
        const ggml_cuda_gated_delta_net_state_src state_src, cudaStream_t stream) {
    const int  nc       = ggml_cuda_sm86_gdn_cols();
    const bool prefetch = ggml_cuda_sm86_gdn_prefetch();
    if (S_v != 128 || n_tokens < 2 || (nc == 1 && !prefetch)) {
        return false;
    }
#define GDN_ILP_LAUNCH(NC_, PF_) \
    launch_gated_delta_net_ilp_inst<keep_rs_t, NC_, PF_, state_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, \
        H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, slot_rows, state_src, stream)
    if (nc == 8) {
        if (prefetch) { GDN_ILP_LAUNCH(8, true); } else { GDN_ILP_LAUNCH(8, false); }
    } else if (nc == 4) {
        if (prefetch) { GDN_ILP_LAUNCH(4, true); } else { GDN_ILP_LAUNCH(4, false); }
    } else if (nc == 2) {
        if (prefetch) { GDN_ILP_LAUNCH(2, true); } else { GDN_ILP_LAUNCH(2, false); }
    } else {
        GDN_ILP_LAUNCH(1, true);
    }
#undef GDN_ILP_LAUNCH
    return true;
}

template <bool KDA, bool keep_rs_t, bool emit_ingredients_t, typename state_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, void * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, const int32_t * slot_rows,
        int ingr_w, int64_t ingr_region, const ggml_cuda_gated_delta_net_state_src state_src, cudaStream_t stream) {
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t, emit_ingredients_t, state_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, slot_rows, ingr_w, ingr_region, state_src);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t, emit_ingredients_t, state_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, slot_rows, ingr_w, ingr_region, state_src);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t, emit_ingredients_t, state_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, slot_rows, ingr_w, ingr_region, state_src);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t, emit_ingredients_t, state_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, slot_rows, ingr_w, ingr_region, state_src);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static int ggml_cuda_gdn_r16_mode() {
    static const int mode = [] {
        const char * s = getenv("GGML_CUDA_GDN_R16");
        return s ? atoi(s) : 0;
    }();
    return mode;
}

static int ggml_cuda_gdn_r16_cols() {
    static const int cols = [] {
        const char * s = getenv("GGML_CUDA_GDN_R16_COLS");
        return (s && atoi(s) == 4) ? 4 : 2;
    }();
    return cols;
}

static int ggml_cuda_gdn_r16_warps() {
    static const int warps = [] {
        const char * s = getenv("GGML_CUDA_GDN_R16_WARPS");
        const int v = s ? atoi(s) : 2;
        return (v == 1 || v == 4) ? v : 2;
    }();
    return warps;
}

template <int C, int NW, bool keep_rs_t, typename state_t>
static void launch_gated_delta_net_r16_inst(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, void * state_d,
        int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3, float scale, int64_t state_slot_stride, int K, const int32_t * slot_rows,
        const ggml_cuda_gated_delta_net_state_src state_src, cudaStream_t stream) {
    static_assert(128 % (4*C*NW) == 0, "columns per block must divide S_v");
    const dim3 grid_dims(H, n_seqs, 128 / (4*C*NW));
    const dim3 block_dims(32, NW, 1);
    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    ggml_cuda_kernel_launch(gated_delta_net_cuda_r16<C, NW, keep_rs_t, state_t>, launch_params,
        q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens,
        sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, n_seqs, state_slot_stride, K, slot_rows, state_src);
}

// GGML_CUDA_GDN_R16 (see gated_delta_net_cuda_r16). Returns false when the path does not apply.
template <bool keep_rs_t, typename state_t>
static bool launch_gated_delta_net_r16(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, void * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, const int32_t * slot_rows,
        const ggml_cuda_gated_delta_net_state_src state_src, ggml_cuda_pool * pool, cudaStream_t stream) {
    if constexpr (std::is_same_v<state_t, block_q8_0>) {
        // q8_0 state blocks are quantized warp-collectively over 32 consecutive rows; this layout spreads them
        GGML_UNUSED_VARS(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, slot_rows, state_src, pool, stream);
        return false;
    } else {
        const int mode = ggml_cuda_gdn_r16_mode();
        if (mode <= 0 || S_v != 128 || n_tokens < 2 || ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size != 32) {
            return false;
        }
        if ((((uintptr_t) q_d) | ((uintptr_t) k_d)) % 16 != 0 || (sq1 % 4) != 0 || (sq2 % 4) != 0 || (sq3 % 4) != 0) {
            return false;
        }
        const int nc = ggml_cuda_gdn_r16_cols();
        const int nw = ggml_cuda_gdn_r16_warps();
        auto run = [&](float * dst_o, void * state_o) {
#define GDN_R16_LAUNCH(C_, NW_) \
            launch_gated_delta_net_r16_inst<C_, NW_, keep_rs_t, state_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_o, state_o, H, n_tokens, n_seqs, \
                sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, slot_rows, state_src, stream)
            if (nc == 4) {
                if (nw == 1) { GDN_R16_LAUNCH(4, 1); } else if (nw == 4) { GDN_R16_LAUNCH(4, 4); } else { GDN_R16_LAUNCH(4, 2); }
            } else {
                if (nw == 1) { GDN_R16_LAUNCH(2, 1); } else if (nw == 4) { GDN_R16_LAUNCH(2, 4); } else { GDN_R16_LAUNCH(2, 2); }
            }
#undef GDN_R16_LAUNCH
        };
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        if (mode == 2 && pool != nullptr) {
            CUDA_CHECK(cudaStreamIsCapturing(stream, &cap));
        }
        if (mode == 2 && pool != nullptr && cap == cudaStreamCaptureStatusNone) {
            // reference first (the input state may be the output cache rows), into scratch; then this kernel for real.
            // Snapshot slots go to compact scratch slots (stride n_state, no slot rows) and are compared block by block.
            const int64_t n_attn  = n_seqs * n_tokens * H * S_v;
            const int64_t n_state = n_seqs * H * S_v * S_v;
            const int64_t n_slots = keep_rs_t ? std::min<int64_t>(K, n_tokens) : 1;
            ggml_cuda_pool_alloc<float>   r_dst(*pool, n_attn);
            ggml_cuda_pool_alloc<state_t> r_state(*pool, n_slots * n_state);
            if (!launch_gated_delta_net_ilp<keep_rs_t, state_t>(q_d, k_d, v_d, g_d, b_d, s_d, r_dst.get(), r_state.get(),
                    S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale,
                    n_state, K, nullptr, state_src, stream)) {
                launch_gated_delta_net<false, keep_rs_t, false, state_t>(q_d, k_d, v_d, g_d, b_d, s_d, r_dst.get(), r_state.get(),
                    S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale,
                    n_state, K, nullptr, 0, 0, state_src, stream);
            }
            run(dst_d, state_d);
            std::vector<int32_t> h_rows;
            if (keep_rs_t && slot_rows != nullptr) {
                h_rows.resize(n_slots * n_seqs);
                CUDA_CHECK(cudaMemcpyAsync(h_rows.data(), slot_rows, h_rows.size()*sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }
            ggml_cuda_pool_alloc<unsigned long long> nd(*pool, 2);
            CUDA_CHECK(cudaMemsetAsync(nd.get(), 0, 2*sizeof(unsigned long long), stream));
            const int64_t w_attn = n_attn;
            gdn_r16_diff<<<(int) std::min<int64_t>((w_attn + 255)/256, 2048), 256, 0, stream>>>(
                (const uint32_t *) dst_d, (const uint32_t *) r_dst.get(), w_attn, nd.get() + 0);
            // state blocks of one (slot, sequence): H*S_v*S_v elements, a multiple of 2 so whole 32-bit words
            const int64_t n_blk = H * S_v * S_v;
            const int64_t w_blk = n_blk * (int64_t) sizeof(state_t) / 4;
            for (int64_t sl = 0; sl < n_slots; sl++) {
                for (int64_t sq = 0; sq < n_seqs; sq++) {
                    const int64_t e_real = !keep_rs_t ? sq * n_blk
                        : (slot_rows == nullptr ? sq * n_blk + sl * state_slot_stride : (int64_t) h_rows[sl * n_seqs + sq] * n_blk);
                    const int64_t e_ref  = sl * n_state + sq * n_blk;
                    gdn_r16_diff<<<(int) std::min<int64_t>((w_blk + 255)/256, 2048), 256, 0, stream>>>(
                        (const uint32_t *) ((const state_t *) state_d + e_real), (const uint32_t *) (r_state.get() + e_ref), w_blk, nd.get() + 1);
                }
            }
            unsigned long long h[2];
            CUDA_CHECK(cudaMemcpyAsync(h, nd.get(), sizeof(h), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            static unsigned long long ncalls = 0, nbad = 0;
            ++ncalls;
            nbad += (h[0] | h[1]) != 0;
            if ((h[0] | h[1]) != 0) {
                GGML_LOG_WARN("%s: GDN_R16 check call %llu (cols %d warps %d n_tokens %lld H %lld n_seqs %lld state %d B slots %lld rows %d): differing attn words %llu state words %llu, bad calls %llu\n",
                    __func__, ncalls, nc, nw, (long long) n_tokens, (long long) H, (long long) n_seqs, (int) sizeof(state_t), (long long) n_slots, (int) !h_rows.empty(), h[0], h[1], nbad);
            } else if ((ncalls & (ncalls - 1)) == 0) {
                GGML_LOG_INFO("%s: GDN_R16 check call %llu (cols %d warps %d n_tokens %lld H %lld n_seqs %lld state %d B slots %lld rows %d): differing attn words %llu state words %llu, bad calls %llu\n",
                    __func__, ncalls, nc, nw, (long long) n_tokens, (long long) H, (long long) n_seqs, (int) sizeof(state_t), (long long) n_slots, (int) !h_rows.empty(), h[0], h[1], nbad);
            }
        } else {
            run(dst_d, state_d);
        }
        return true;
    }
}

// [#199] key-head-shared token-serial recurrence (multi-token prefill). On by default in v0.6; GGML_CUDA_GDN_KHSHARE=0 turns it off.
// The R = H / neqk1 value heads h = kh + i*neqk1 (i < R) read the same q/k (key head kh, see iq1 in
// the kernels above). One thread keeps a quarter of one state column for all R of those heads, so
// every q/k value it reads is used R times, and a column is reduced by 4 lanes (2 shuffles) instead
// of 32 (5 shuffles). q/k/v/g/beta for KHS_TT tokens at a time are staged in shared memory with
// cp.async, double-buffered, so there are 2 block barriers per KHS_TT tokens.
//
// Bit-identical to gated_delta_net_cuda / gated_delta_net_cuda_ilp: those keep column rows
// l + 32r in lane l. Here lane j (0..3) of a column holds the rows of the original lanes
// l = j + 4m (m < 8), i.e. rows j + 4m + 32r. Each per-l partial is the same fma chain over r, the
// xor-16/8/4 butterfly levels of warp_reduce_sum pair l with l^16, l^8, l^4 = m^4, m^2, m^1 and are
// folded in registers with the same pairing, and the last two levels (xor 2, xor 1) are shuffles.
#define KHS_TT  8                 // tokens per staged tile
#define KHS_PAD 36                // floats per j block of a staged q/k vector (32 + 4: conflict-free LDS.128)
#define KHS_VEC (4 * KHS_PAD)     // floats per staged q or k vector

static __device__ __forceinline__ void khs_cp_async_commit() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.commit_group;");
#else
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}

template <int N>
static __device__ __forceinline__ void khs_cp_async_wait() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.wait_group %0;" : : "n"(N));
#else
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}

// warp_reduce_sum<32> of the original layout: p[m] is the partial of original lane j + 4m
static __device__ __forceinline__ float khs_reduce(const float (&p)[8]) {
    const float a0 = p[0] + p[4];
    const float a1 = p[1] + p[5];
    const float a2 = p[2] + p[6];
    const float a3 = p[3] + p[7];
    const float b0 = a0 + a2;
    const float b1 = a1 + a3;
    float c = b0 + b1;
    c += __shfl_xor_sync(0xffffffff, c, 2, 32);
    c += __shfl_xor_sync(0xffffffff, c, 1, 32);
    return c;
}

// store one state column (128 rows; this lane holds rows j + 4m + 32r) at element offset e0 (row 0).
// q8_0: block r = rows 32r..32r+31, held by the 4 lanes of the column; same scale/rounding as gdn_store_state.
template <typename state_t>
static __device__ __forceinline__ void khs_store_col(void * state, const int64_t e0, const float (&sc)[8][4], const int j) {
    if constexpr (std::is_same_v<state_t, block_q8_0>) {
#pragma unroll
        for (int r = 0; r < 4; r++) {
            float amax = 0.0f;
#pragma unroll
            for (int m = 0; m < 8; m++) {
                amax = fmaxf(amax, fabsf(sc[m][r]));
            }
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2, 32));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1, 32));
            const float d  = amax / ((1 << 7) - 1);
            const float id = d ? 1.0f/d : 0.0f;
            block_q8_0 * b = (block_q8_0 *) state + (e0 + 32 * r) / QK8_0;
#pragma unroll
            for (int m = 0; m < 8; m++) {
                b->qs[j + 4 * m] = roundf(sc[m][r]*id);
            }
            if (j == 0) {
                b->d = d;
            }
        }
    } else {
#pragma unroll
        for (int r = 0; r < 4; r++) {
#pragma unroll
            for (int m = 0; m < 8; m++) {
                ((state_t *) state)[e0 + j + 4 * m + 32 * r] = ggml_cuda_cast<state_t>(sc[m][r]);
            }
        }
    }
}

template <int R, int WARPS, bool keep_rs_t, typename state_t>
__global__ void __launch_bounds__(WARPS * 32, 1)
gated_delta_net_cuda_khshare(const float * q,
                             const float * k,
                             const float * v,
                             const float * g,
                             const float * beta,
                             const float * curr_state,
                             float *       dst,
                             void *        state,
                             int64_t       H,
                             int64_t       n_tokens,
                             int64_t       n_seqs,
                             int64_t       sq1,
                             int64_t       sq2,
                             int64_t       sq3,
                             int64_t       sv1,
                             int64_t       sv2,
                             int64_t       sv3,
                             int64_t       sb1,
                             int64_t       sb2,
                             int64_t       sb3,
                             int           neqk1,
                             int           rq3,
                             float         scale,
                             int64_t       state_slot_stride,
                             int           K,
                             const int32_t * slot_rows,
                             const ggml_cuda_gated_delta_net_state_src state_src) {
    constexpr int S_v  = 128;
    constexpr int TT   = KHS_TT;
    constexpr int NCOL = WARPS * 8;   // state columns per block (8 per warp, 4 lanes each)
    constexpr int NTHR = WARPS * 32;

    __shared__ __align__(16) float kq_s[2][TT][2][KHS_VEC]; // [buf][token][k|q][j block]
    __shared__ float v_s[2][TT][R][NCOL];
    __shared__ float g_s[2][TT][R];
    __shared__ float b_s[2][TT][R];

    const int kh       = blockIdx.x;
    const int sequence = blockIdx.y;
    const int tid      = threadIdx.x;
    const int lane     = tid & 31;
    const int j        = lane & 3;
    const int wc       = (tid >> 5) * 8 + (lane >> 2); // column within the block
    const int col0     = blockIdx.z * NCOL;
    const int col      = col0 + wc;
    const int iq3      = sequence / rq3;

    const float * q_base = q + iq3 * sq3 + kh * sq1;
    const float * k_base = k + iq3 * sq3 + kh * sq1;

    float s[R][8][4];

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int i = 0; i < R; i++) {
        const int64_t h = kh + i * neqk1;
        // [#87] see gated_delta_net_cuda: cache row rows[sequence], or src[5] at sequence * D
        const float * cs = curr_state + (state_src.rows != nullptr
                ? (int64_t) state_src.rows[sequence] * state_src.row_stride + h * S_v * S_v
                : (int64_t) sequence * H * S_v * S_v + h * S_v * S_v) + col * S_v;
#pragma unroll
        for (int m = 0; m < 8; m++) {
#pragma unroll
            for (int r = 0; r < 4; r++) {
                s[i][m][r] = cs[j + 4 * m + 32 * r];
            }
        }
    }

    auto load_tile = [&](const int buf, const int t0) {
        const int nt = min((int64_t) TT, n_tokens - t0);
        for (int e = tid; e < nt * 2 * S_v; e += NTHR) {
            const int tt    = e / (2 * S_v);
            const int which = (e / S_v) & 1;
            const int i     = e % S_v;
            const float * src = (which ? q_base : k_base) + (int64_t) (t0 + tt) * sq2 + i;
            float * d = &kq_s[buf][tt][which][(i & 3) * KHS_PAD + ((i >> 2) & 7) * 4 + (i >> 5)];
            cp_async_ca_4(ggml_cuda_cvta_generic_to_shared(d), src);
        }
        for (int e = tid; e < nt * R * NCOL; e += NTHR) {
            const int tt = e / (R * NCOL);
            const int i  = (e / NCOL) % R;
            const int c  = e % NCOL;
            const float * src = v + sequence * sv3 + (int64_t) (t0 + tt) * sv2 + (kh + i * neqk1) * sv1 + col0 + c;
            cp_async_ca_4(ggml_cuda_cvta_generic_to_shared(&v_s[buf][tt][i][c]), src);
        }
        for (int e = tid; e < nt * R; e += NTHR) {
            const int tt = e / R;
            const int i  = e % R;
            const int64_t off = sequence * sb3 + (int64_t) (t0 + tt) * sb2 + (kh + i * neqk1) * sb1;
            cp_async_ca_4(ggml_cuda_cvta_generic_to_shared(&g_s[buf][tt][i]), g + off);
            cp_async_ca_4(ggml_cuda_cvta_generic_to_shared(&b_s[buf][tt][i]), beta + off);
        }
        khs_cp_async_commit();
    };

    const int n_tiles = (int) ((n_tokens + TT - 1) / TT);
    load_tile(0, 0);

    for (int tile = 0; tile < n_tiles; tile++) {
        const int buf = tile & 1;
        const int t0  = tile * TT;
        if (tile + 1 < n_tiles) {
            load_tile(buf ^ 1, t0 + TT);
            khs_cp_async_wait<1>();
        } else {
            khs_cp_async_wait<0>();
        }
        __syncthreads();

        const int nt = min((int64_t) TT, n_tokens - t0);
        for (int tt = 0; tt < nt; tt++) {
            const int t = t0 + tt;
            const float4 * k4 = (const float4 *) &kq_s[buf][tt][0][j * KHS_PAD];
            const float4 * q4 = (const float4 *) &kq_s[buf][tt][1][j * KHS_PAD];

            float g_val[R];
            float beta_val[R];
#pragma unroll
            for (int i = 0; i < R; i++) {
                g_val[i]    = expf(g_s[buf][tt][i]);
                beta_val[i] = b_s[buf][tt][i];
            }

            // kv[col] = sum_i S[i][col] * k[i]   (per original lane: same fma chain over r)
            float p[R][8];
#pragma unroll
            for (int m = 0; m < 8; m++) {
                const float4 kk = k4[m];
#pragma unroll
                for (int i = 0; i < R; i++) {
                    float acc = 0.0f;
                    acc += s[i][m][0] * kk.x;
                    acc += s[i][m][1] * kk.y;
                    acc += s[i][m][2] * kk.z;
                    acc += s[i][m][3] * kk.w;
                    p[i][m] = acc;
                }
            }
            float delta[R];
#pragma unroll
            for (int i = 0; i < R; i++) {
                const float kv_col = khs_reduce(p[i]);
                delta[i] = (v_s[buf][tt][i][wc] - g_val[i] * kv_col) * beta_val[i];
            }

            // S[i][col] = g * S[i][col] + k[i] * delta[col];  attn[col] = sum_i S[i][col] * q[i]
#pragma unroll
            for (int m = 0; m < 8; m++) {
                const float4 kk = k4[m];
                const float4 qq = q4[m];
                const float kr[4] = { kk.x, kk.y, kk.z, kk.w };
                const float qr[4] = { qq.x, qq.y, qq.z, qq.w };
#pragma unroll
                for (int i = 0; i < R; i++) {
                    float acc = 0.0f;
#pragma unroll
                    for (int r = 0; r < 4; r++) {
                        s[i][m][r] = g_val[i] * s[i][m][r] + kr[r] * delta[i];
                        acc += s[i][m][r] * qr[r];
                    }
                    p[i][m] = acc;
                }
            }
#pragma unroll
            for (int i = 0; i < R; i++) {
                const float attn_col = khs_reduce(p[i]);
                if (j == (i & 3)) {
                    dst[(((int64_t) sequence * n_tokens + t) * H + kh + i * neqk1) * S_v + col] = attn_col * scale;
                }
            }

            if constexpr (keep_rs_t) {
                // most-recent-first: slot 0 = final state (see gated_delta_net_cuda)
                const int target_slot = (int) n_tokens - 1 - t;
                if (target_slot >= 0 && target_slot < K) {
#pragma unroll
                    for (int i = 0; i < R; i++) {
                        const int64_t h = kh + i * neqk1;
                        const int64_t snap_e = slot_rows == nullptr
                            ? ((int64_t) sequence * H + h) * S_v * S_v + (int64_t) target_slot * state_slot_stride
                            : (int64_t) slot_rows[target_slot * n_seqs + sequence] * (H * S_v * S_v) + h * S_v * S_v;
                        khs_store_col<state_t>(state, snap_e + col * S_v, s[i], j);
                    }
                }
            }
        }
        __syncthreads(); // the next iteration's load_tile overwrites this buffer
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int i = 0; i < R; i++) {
            const int64_t h = kh + i * neqk1;
            khs_store_col<state_t>(state, ((int64_t) sequence * H + h) * S_v * S_v + col * S_v, s[i], j);
        }
    }
}

// Launch counter for the test harness (ggml_backend_cuda_fusion_count "gdn_khshare"). Process-wide
// rather than a fusion_stats field so common.cuh (and every file that includes it) stays untouched;
// the test reads it as a before/after delta.
int64_t ggml_cuda_gdn_khshare_count = 0;

static bool ggml_cuda_gdn_khshare_on() {
    static const bool on = [] {
        const char * s = getenv("GGML_CUDA_GDN_KHSHARE");
        return s == nullptr || atoi(s) != 0;
    }();
    return on;
}

// smallest token count that takes the key-head-shared kernel (decode/verify widths keep the ILP kernel)
static int64_t ggml_cuda_gdn_khshare_min() {
    static const int64_t v = [] {
        const char * s = getenv("GGML_CUDA_GDN_KHSHARE_MIN");
        return s ? (int64_t) atoll(s) : (int64_t) 64;
    }();
    return v;
}

// warps per block: 1, 2 or 4 (8 state columns per warp)
static int ggml_cuda_gdn_khshare_warps() {
    static const int v = [] {
        const char * s = getenv("GGML_CUDA_GDN_KHSHARE_WARPS");
        const int w = s ? atoi(s) : 4;
        return (w == 1 || w == 2 || w == 4) ? w : 4;
    }();
    return v;
}

template <int R, int WARPS, bool keep_rs_t, typename state_t>
static void launch_gated_delta_net_khshare_inst(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, void * state_d,
        int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, const int32_t * slot_rows,
        const ggml_cuda_gated_delta_net_state_src state_src, cudaStream_t stream) {
    constexpr int S_v = 128;
    dim3 grid_dims(neqk1, n_seqs, S_v / (WARPS * 8));
    dim3 block_dims(WARPS * 32, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    ggml_cuda_kernel_launch(gated_delta_net_cuda_khshare<R, WARPS, keep_rs_t, state_t>, launch_params,
        q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
        n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
        sb1, sb2, sb3, (int) neqk1, (int) rq3, scale, state_slot_stride, K, slot_rows, state_src);
}

// returns false when this path does not apply (off, decode width, shape); caller falls back
template <bool keep_rs_t, typename state_t>
static bool launch_gated_delta_net_khshare(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, void * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, const int32_t * slot_rows,
        const ggml_cuda_gated_delta_net_state_src state_src, cudaStream_t stream) {
    if (!ggml_cuda_gdn_khshare_on() || S_v != 128 || n_tokens < ggml_cuda_gdn_khshare_min() ||
        ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size != 32 ||
        ggml_cuda_info().devices[ggml_cuda_get_device()].cc < GGML_CUDA_CC_AMPERE ||
        neqk1 <= 0 || H % neqk1 != 0 || H / neqk1 > 4 || rq3 <= 0) {
        return false;
    }
    const int R     = (int) (H / neqk1);
    const int warps = ggml_cuda_gdn_khshare_warps();
#define GDN_KHS_LAUNCH(R_, W_) \
    launch_gated_delta_net_khshare_inst<R_, W_, keep_rs_t, state_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, \
        H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, slot_rows, state_src, stream)
#define GDN_KHS_WARPS(R_) \
    if (warps == 1) { GDN_KHS_LAUNCH(R_, 1); } else if (warps == 2) { GDN_KHS_LAUNCH(R_, 2); } else { GDN_KHS_LAUNCH(R_, 4); }
    switch (R) {
        case 1: GDN_KHS_WARPS(1); break;
        case 2: GDN_KHS_WARPS(2); break;
        case 3: GDN_KHS_WARPS(3); break;
        default: GDN_KHS_WARPS(4); break;
    }
#undef GDN_KHS_WARPS
#undef GDN_KHS_LAUNCH
    return true;
}


template <typename state_t>
static void gated_delta_net_dispatch(
        bool kda, bool keep_rs, bool emit_ingr,
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, void * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, const int32_t * slot_rows,
        int ingr_w, int64_t ingr_region, const ggml_cuda_gated_delta_net_state_src state_src, ggml_cuda_pool * pool, cudaStream_t stream) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    // Chunked tensor-core prompt kernel, enabled by default (valid math, not bit-identical);
    // returns false (falls through to the kernels below) when the shape, arch or state type does not apply
    // BF16 cache rounding amplifies small chunked-state differences; retain the precise path.
    if (!kda && pool != nullptr && !std::is_same_v<state_t, nv_bfloat16>) {
#define GDNA_TRY(KRS_, EMIT_) gdn_argon::launch_gated_delta_net_chunked_tc<KRS_, EMIT_, state_t>( \
            *pool, q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, \
            sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, slot_rows, ingr_w, ingr_region, state_src, stream)
        if (emit_ingr) {
            if constexpr (std::is_same_v<state_t, float>) {
                if (GDNA_TRY(true, true)) { return; }
            }
        } else if (keep_rs) {
            if (GDNA_TRY(true, false)) { return; }
        } else {
            if (GDNA_TRY(false, false)) { return; }
        }
#undef GDNA_TRY
    }
#endif
#define GDN_ARGS q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, \
        S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, \
        sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, slot_rows
    if constexpr (std::is_same_v<state_t, float>) {
        const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        // Every V row must permit float2 accesses, including strided/permuted views.
        const bool aligned_v = (reinterpret_cast<uintptr_t>(v_d) % alignof(float2)) == 0 &&
            sv1 % 2 == 0 && sv2 % 2 == 0 && sv3 % 2 == 0;
        if (ggml_cuda_gdn_subwarp() != 0 && S_v == 128 && n_tokens > 1 && !kda && aligned_v &&
            cc >= GGML_CUDA_CC_AMPERE && cc < GGML_CUDA_CC_ADA_LOVELACE &&
            ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size == 32) {
            if (emit_ingr) {
                launch_gated_delta_net_subwarp<true, true>(GDN_ARGS, ingr_w, ingr_region, state_src, stream);
            } else if (keep_rs) {
                launch_gated_delta_net_subwarp<true, false>(GDN_ARGS, ingr_w, ingr_region, state_src, stream);
            } else {
                launch_gated_delta_net_subwarp<false, false>(GDN_ARGS, ingr_w, ingr_region, state_src, stream);
            }
            return;
        }
    }
    if (emit_ingr) {
        if constexpr (std::is_same_v<state_t, float>) {
            if (kda) {
                launch_gated_delta_net<true, true, true, state_t>(GDN_ARGS, ingr_w, ingr_region, state_src, stream);
            } else {
                launch_gated_delta_net<false, true, true, state_t>(GDN_ARGS, ingr_w, ingr_region, state_src, stream);
            }
        } else {
            GGML_ABORT("gated_delta_net: emit_mode 1/2 is never fused into a non-F32 cache");
        }
    } else if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true, false, state_t>(GDN_ARGS, ingr_w, ingr_region, state_src, stream);
        } else {
            launch_gated_delta_net<true, false, false, state_t>(GDN_ARGS, ingr_w, ingr_region, state_src, stream);
        }
    } else if (keep_rs) {
        // [#199] key-head-shared prompt kernel after the opt-in R16 kernel, before ILP (bit-identical to ILP);
        // each launcher returns false when it does not apply
        if (launch_gated_delta_net_r16<true, state_t>(GDN_ARGS, state_src, pool, stream)) {
            return;
        }
        if (launch_gated_delta_net_khshare<true, state_t>(GDN_ARGS, state_src, stream)) {
            ggml_cuda_gdn_khshare_count++;
            return;
        }
        if (!launch_gated_delta_net_ilp<true, state_t>(GDN_ARGS, state_src, stream)) {
            launch_gated_delta_net<false, true, false, state_t>(GDN_ARGS, ingr_w, ingr_region, state_src, stream);
        }
    } else {
        // [#199] key-head-shared prompt kernel after the opt-in R16 kernel, before ILP (bit-identical to ILP);
        // each launcher returns false when it does not apply
        if (launch_gated_delta_net_r16<false, state_t>(GDN_ARGS, state_src, pool, stream)) {
            return;
        }
        if (launch_gated_delta_net_khshare<false, state_t>(GDN_ARGS, state_src, stream)) {
            ggml_cuda_gdn_khshare_count++;
            return;
        }
        if (!launch_gated_delta_net_ilp<false, state_t>(GDN_ARGS, state_src, stream)) {
            launch_gated_delta_net<false, false, false, state_t>(GDN_ARGS, ingr_w, ingr_region, state_src, stream);
        }
    }
#undef GDN_ARGS
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    // [#87] the GET_ROWS that gathered src[5] from the recurrent cache was skipped by the graph
    // evaluator (ggml_cuda_gdn_state_read_plan): read the source rows directly. Same floats, same
    // positions, so the launch computes exactly what it would have from the gathered copy.
    ggml_cuda_gated_delta_net_state_src state_src;
    for (auto & e : ctx.gdn_state_reads) {
        if (e.gdn == dst) {
            GGML_ASSERT(!e.used);
            s_d                  = e.base;
            state_src.rows       = e.rows;
            state_src.row_stride = e.row_stride;
            e.used               = true;
            ctx.fusion_stats.gdn_state_read++;
            break;
        }
    }

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) and emit_mode (0=full snapshots, 1=ingredients) are op params;
    // state holds s0 only [S_v, S_v, H, n_seqs].
    const int  K         = ggml_get_op_params_i32(dst, 0);
    const int  emit_mode = ggml_get_op_params_i32(dst, 1);
    const bool keep_rs   = (K > 1) || (emit_mode != 0);
    const bool emit_ingr = (emit_mode != 0);

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing.
    // emit_ingr slots are ingr_w wide per head (4*S_v, or emit_mode 2's compact width) instead
    // of S_v*S_v (a full state matrix); the fusion path (external cache) never engages for
    // emit_mode != 0 (see ggml_cuda_try_gdn_cache_fusion's explicit guard), so
    // cache->slot_stride is always the emit_mode==0 layout when cache != nullptr.
    const int     ingr_w      = emit_ingr ? (int) ggml_gated_delta_net_ingr_width(S_v, src_g->ne[0], emit_mode) : 0;
    const int64_t ingr_region = emit_ingr ? ggml_gated_delta_net_ingr_region(S_v, src_g->ne[0], H, n_seqs, K, emit_mode) : 0;
    void *    state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    ggml_type state_type        = GGML_TYPE_F32;
    int64_t   state_slot_stride = emit_ingr ? (ingr_w * H * n_seqs) : (S_v * S_v * H * n_seqs);
    const int32_t * slot_rows   = nullptr;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_type        = cache->type;
        state_slot_stride = cache->slot_stride;
        slot_rows         = cache->slot_rows;
    }

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (ggml_cuda_gdn_tc_enabled() && (cc == 800 || cc == 860) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_AMPERE &&
            S_v == 128 && n_tokens >= 512 && !kda && !keep_rs && state_type == GGML_TYPE_F32) {
        const int precision = ggml_cuda_gdn_tc_precision();
        const auto kernel = precision == 1 ? gated_delta_net_chunked_tc<1> : precision == 2 ? gated_delta_net_chunked_tc<2> : gated_delta_net_chunked_tc<3>;
        CUDA_SET_SHARED_MEMORY_LIMIT(kernel, GDN_TC_SHARED);
        const ggml_cuda_kernel_launch_params params(dim3(H, n_seqs, 128 / GDN_TC_COLS), dim3(WARP_SIZE, 4), GDN_TC_SHARED, stream);
        ggml_cuda_kernel_launch(kernel, params,
            q_d, k_d, v_d, g_d, b_d, s_d, dst_d, (float *) state_d, H, n_tokens, n_seqs,
            sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, init_fastdiv_values(neqk1), init_fastdiv_values(rq3),
            scale, slot_rows, state_src);
        return;
    }
#endif

#define GDN_DISPATCH(T_) \
    gated_delta_net_dispatch<T_>(kda, keep_rs, emit_ingr, q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, \
        S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, \
        sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, slot_rows, ingr_w, ingr_region, state_src, &ctx.pool(), stream)
    switch (state_type) {
        case GGML_TYPE_F32:  GDN_DISPATCH(float);       break;
        case GGML_TYPE_BF16: GDN_DISPATCH(nv_bfloat16); break;
        case GGML_TYPE_F16:  GDN_DISPATCH(half);        break;
        case GGML_TYPE_Q8_0:
            GGML_ASSERT(S_v % QK8_0 == 0 && ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size == QK8_0);
            GDN_DISPATCH(block_q8_0);
            break;
        default: GGML_ABORT("gated_delta_net: unsupported state cache type %s", ggml_type_name(state_type));
    }
#undef GDN_DISPATCH
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}
