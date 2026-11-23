// Bit-identical R16 Gated Delta Net verify and replay kernel (sm_80+, S_v=128, F32 state).
// Handles both masked ring prefix replay (--gdn-replay, emit_mode 1/2) and full snapshots.

#include "gated_delta_net.cuh"
#include "gated_delta_net_v16.cuh"

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace gdn_v16 {

template <int NW>
static __device__ __forceinline__ void block_sync() {
    if constexpr (NW == 1) {
        __syncwarp();
    } else {
        __syncthreads();
    }
}

template <int C, int NW, bool keep_rs_t, bool emit_ingredients_t>
__global__ void __launch_bounds__(32 * NW)
gated_delta_net_cuda_v16(const float * q,
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
    constexpr int S_v = 128;
    constexpr int PAD = 20; // 16 used + 4 pad keeps 128-bit LDS conflict-free
    __shared__ __align__(16) float s_k[2][8 * PAD];
    __shared__ __align__(16) float s_q[2][8 * PAD];

    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      lane     = threadIdx.x;
    const int      tid      = threadIdx.y * 32 + lane;
    const int      j        = lane & 7;
    const int      col0     = (blockIdx.z * NW + threadIdx.y) * (4 * C) + (lane >> 3) * C;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const int64_t state_in_offset = ((int64_t) sequence * H + h_idx) * S_v * S_v;
    const int64_t state_out_e     = emit_ingredients_t
        ? ((int64_t) sequence * H + h_idx) * (int64_t) ingr_w
        : state_in_offset;
    float * const state_f   = (float *) state;
    float *       attn_data = dst + ((int64_t) sequence * n_tokens * H + h_idx) * S_v;

    // s[c][4*r + m] = S[row 32*r + j + 8*m][col0 + c]
    float s[C][16];

    ggml_cuda_pdl_sync();
    float * fin_dst = state_src.final_dst;
    float * ckp_dst = state_src.ckpt_dst;
    if constexpr (emit_ingredients_t) {
        if (state_src.guard_in_place && gdn_direct_rows_conflict(curr_state, state_src, n_seqs, H * S_v * S_v)) {
            fin_dst = nullptr;
            ckp_dst = nullptr;
        }
    }

    const float * s_in = curr_state + (state_src.rows != nullptr
        ? (int64_t) state_src.rows[sequence] * state_src.row_stride + (int64_t) h_idx * S_v * S_v
        : state_in_offset) + (int64_t) col0 * S_v;
#pragma unroll
    for (int c = 0; c < C; c++) {
#pragma unroll
        for (int n = 0; n < 16; n++) {
            s[c][n] = s_in[(int64_t) c * S_v + j + 8 * n];
        }
    }

    const float * q_base = q + (int64_t) iq3 * sq3 + (int64_t) iq1 * sq1;
    const float * k_base = k + (int64_t) iq3 * sq3 + (int64_t) iq1 * sq1;
    const float * v_base = v + (int64_t) sequence * sv3 + (int64_t) h_idx * sv1 + col0;
    const float * b_base = beta + (int64_t) sequence * sb3 + (int64_t) h_idx * sb1;
    const float * g_base = g    + (int64_t) sequence * sb3 + (int64_t) h_idx * sb1;

    constexpr int NLD = (64 + 32 * NW - 1) / (32 * NW);
    float4 ld[NLD];
    auto stage_load_tok = [&](const int t) {
#pragma unroll
        for (int i = 0; i < NLD; i++) {
            const int f = tid + i * 32 * NW;
            if (f < 64) {
                ld[i] = *(const float4 *) ((f < 32 ? k_base : q_base) + (int64_t) t * sq2 + 4 * (f & 31));
            }
        }
    };
    auto stage_store_tok = [&](const int b) {
#pragma unroll
        for (int i = 0; i < NLD; i++) {
            const int f = tid + i * 32 * NW;
            if (f < 64) {
                float * sm = f < 32 ? s_k[b] : s_q[b];
                const int e0 = 4 * (f & 31);
                sm[((e0 + 0) & 7) * PAD + ((e0 + 0) >> 3)] = ld[i].x;
                sm[((e0 + 1) & 7) * PAD + ((e0 + 1) >> 3)] = ld[i].y;
                sm[((e0 + 2) & 7) * PAD + ((e0 + 2) >> 3)] = ld[i].z;
                sm[((e0 + 3) & 7) * PAD + ((e0 + 3) >> 3)] = ld[i].w;
            }
        }
    };
    auto store_state_block = [&](float * base_ptr) {
        float * dst_s = base_ptr + ((int64_t) sequence * H + h_idx) * S_v * S_v + (int64_t) col0 * S_v;
#pragma unroll
        for (int c = 0; c < C; c++) {
#pragma unroll
            for (int n = 0; n < 16; n++) {
                dst_s[(int64_t) c * S_v + j + 8 * n] = s[c][n];
            }
        }
    };

    float v_cur[C];
    float g_cur = 0.0f;
    float b_cur = 0.0f;
    int   buf = 0;
    bool  tok0_staged = false;

    // Phase 1: replay active prefix slots from the ingredient ring without loading q or writing attn.
    if constexpr (emit_ingredients_t) {
        const int n_prefix = state_src.prefix_n;
        if (n_prefix > 0) {
            const float * pw     = state_src.prefix_w + (int64_t) sequence * state_src.prefix_w_s1;
            const float * p_base = state_src.prefix + (int64_t) sequence * state_src.prefix_s1 +
                (int64_t) h_idx * state_src.prefix_ingr_w;
            const int64_t p_stride = (int64_t) state_src.prefix_ingr_w * H;

            // Prefix slots use 64 float2 loads because compact emit_mode 2 has stride 258 (8-byte aligned).
            float2 ld_p[NLD];
            auto stage_load_prefix = [&](const int p, const float w_p, float (&v_out)[C], float & g_out, float & b_out) {
                const float * slot = p_base + (int64_t) p * p_stride;
#pragma unroll
                for (int i = 0; i < NLD; i++) {
                    const int f = tid + i * 32 * NW;
                    if (f < 64) {
                        ld_p[i] = *(const float2 *) (slot + 2 * f);
                    }
                }
#pragma unroll
                for (int c = 0; c < C; c++) {
                    v_out[c] = slot[S_v + col0 + c];
                }
                g_out = slot[2 * S_v] * w_p;
                b_out = slot[state_src.prefix_off_b] * w_p;
            };
            auto stage_store_prefix = [&](const int b) {
#pragma unroll
                for (int i = 0; i < NLD; i++) {
                    const int f = tid + i * 32 * NW;
                    if (f < 64) {
                        const int e0 = 2 * f;
                        s_k[b][((e0 + 0) & 7) * PAD + ((e0 + 0) >> 3)] = ld_p[i].x;
                        s_k[b][((e0 + 1) & 7) * PAD + ((e0 + 1) >> 3)] = ld_p[i].y;
                    }
                }
            };

            int p = 0;
            while (p < n_prefix && pw[p] == 0.0f) {
                p++;
            }
            if (p < n_prefix) {
                stage_load_prefix(p, pw[p], v_cur, g_cur, b_cur);
                stage_store_prefix(0);
                block_sync<NW>();

                while (p < n_prefix) {
                    int p_nxt = p + 1;
                    while (p_nxt < n_prefix && pw[p_nxt] == 0.0f) {
                        p_nxt++;
                    }
                    float v_nxt[C];
                    float g_nxt = 0.0f;
                    float b_nxt = 0.0f;
                    if (p_nxt < n_prefix) {
                        stage_load_prefix(p_nxt, pw[p_nxt], v_nxt, g_nxt, b_nxt);
                    } else {
                        stage_load_tok(0);
#pragma unroll
                        for (int c = 0; c < C; c++) {
                            v_nxt[c] = v_base[c];
                        }
                        g_nxt = g_base[0];
                        b_nxt = b_base[0];
                    }

                    const float g_val = expf(g_cur);

                    float4 kr[4];
#pragma unroll
                    for (int r = 0; r < 4; r++) {
                        kr[r] = *(const float4 *) &s_k[buf][j * PAD + 4 * r];
                    }

                    float kv_col[C];
#pragma unroll
                    for (int c = 0; c < C; c++) {
                        float p0 = 0.0f;
                        float p1 = 0.0f;
                        float p2 = 0.0f;
                        float p3 = 0.0f;
#pragma unroll
                        for (int r = 0; r < 4; r++) {
                            p0 += s[c][4 * r + 0] * kr[r].x;
                            p1 += s[c][4 * r + 1] * kr[r].y;
                            p2 += s[c][4 * r + 2] * kr[r].z;
                            p3 += s[c][4 * r + 3] * kr[r].w;
                        }
                        float x = (p0 + p2) + (p1 + p3);
                        x += __shfl_xor_sync(0xffffffff, x, 4, 32);
                        x += __shfl_xor_sync(0xffffffff, x, 2, 32);
                        x += __shfl_xor_sync(0xffffffff, x, 1, 32);
                        kv_col[c] = x;
                    }

#pragma unroll
                    for (int c = 0; c < C; c++) {
                        const float delta_col = (v_cur[c] - g_val * kv_col[c]) * b_cur;
#pragma unroll
                        for (int r = 0; r < 4; r++) {
                            s[c][4 * r + 0] = g_val * s[c][4 * r + 0] + kr[r].x * delta_col;
                            s[c][4 * r + 1] = g_val * s[c][4 * r + 1] + kr[r].y * delta_col;
                            s[c][4 * r + 2] = g_val * s[c][4 * r + 2] + kr[r].z * delta_col;
                            s[c][4 * r + 3] = g_val * s[c][4 * r + 3] + kr[r].w * delta_col;
                        }
                    }

                    if (p_nxt < n_prefix) {
                        stage_store_prefix(buf ^ 1);
                    } else {
                        stage_store_tok(buf ^ 1);
                        tok0_staged = true;
                    }
#pragma unroll
                    for (int c = 0; c < C; c++) {
                        v_cur[c] = v_nxt[c];
                    }
                    g_cur = g_nxt;
                    b_cur = b_nxt;
                    buf ^= 1;
                    block_sync<NW>();
                    p = p_nxt;
                }
            }

            if (n_tokens <= K) {
                store_state_block(ckp_dst ? ckp_dst : state_f + ingr_region + (int64_t) S_v * S_v * H * n_seqs);
            }
        }
    }

    if (!tok0_staged) {
        stage_load_tok(0);
#pragma unroll
        for (int c = 0; c < C; c++) {
            v_cur[c] = v_base[c];
        }
        g_cur = g_base[0];
        b_cur = b_base[0];
        stage_store_tok(buf);
        block_sync<NW>();
    }

    // Phase 2: main batch tokens with double-buffered k/q staging.
    for (int t = 0; t < n_tokens; t++) {
        float v_nxt[C];
        float g_nxt = 0.0f;
        float b_nxt = 0.0f;
        if (t + 1 < n_tokens) {
            stage_load_tok(t + 1);
#pragma unroll
            for (int c = 0; c < C; c++) {
                v_nxt[c] = v_base[(int64_t) (t + 1) * sv2 + c];
            }
            g_nxt = g_base[(int64_t) (t + 1) * sb2];
            b_nxt = b_base[(int64_t) (t + 1) * sb2];
        }

        const float g_val = expf(g_cur);

        float4 kr[4];
#pragma unroll
        for (int r = 0; r < 4; r++) {
            kr[r] = *(const float4 *) &s_k[buf][j * PAD + 4 * r];
        }

        float kv_col[C];
#pragma unroll
        for (int c = 0; c < C; c++) {
            float p0 = 0.0f;
            float p1 = 0.0f;
            float p2 = 0.0f;
            float p3 = 0.0f;
#pragma unroll
            for (int r = 0; r < 4; r++) {
                p0 += s[c][4 * r + 0] * kr[r].x;
                p1 += s[c][4 * r + 1] * kr[r].y;
                p2 += s[c][4 * r + 2] * kr[r].z;
                p3 += s[c][4 * r + 3] * kr[r].w;
            }
            float x = (p0 + p2) + (p1 + p3);
            x += __shfl_xor_sync(0xffffffff, x, 4, 32);
            x += __shfl_xor_sync(0xffffffff, x, 2, 32);
            x += __shfl_xor_sync(0xffffffff, x, 1, 32);
            kv_col[c] = x;
        }

        float4 qr[4];
#pragma unroll
        for (int r = 0; r < 4; r++) {
            qr[r] = *(const float4 *) &s_q[buf][j * PAD + 4 * r];
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
                s[c][4 * r + 0] = g_val * s[c][4 * r + 0] + kr[r].x * delta_col;
                a0 += s[c][4 * r + 0] * qr[r].x;
                s[c][4 * r + 1] = g_val * s[c][4 * r + 1] + kr[r].y * delta_col;
                a1 += s[c][4 * r + 1] * qr[r].y;
                s[c][4 * r + 2] = g_val * s[c][4 * r + 2] + kr[r].z * delta_col;
                a2 += s[c][4 * r + 2] * qr[r].z;
                s[c][4 * r + 3] = g_val * s[c][4 * r + 3] + kr[r].w * delta_col;
                a3 += s[c][4 * r + 3] * qr[r].w;
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
        attn_data += (int64_t) S_v * H;

        if constexpr (keep_rs_t) {
            const int target_slot = emit_ingredients_t ? (t - ((int) n_tokens - K)) : ((int) n_tokens - 1 - t);
            if (target_slot >= 0 && target_slot < K) {
                if constexpr (emit_ingredients_t) {
                    float * ingr_slot = state_src.ingr_dst
                        ? state_src.ingr_dst + ((int64_t) sequence * K + target_slot) * ingr_w * H + (int64_t) h_idx * ingr_w
                        : state_f + state_out_e + (int64_t) target_slot * state_slot_stride;
                    const bool compact = ingr_w != 4 * S_v;
                    if (blockIdx.z == 0 && threadIdx.y == 0) {
#pragma unroll
                        for (int r = 0; r < 4; r++) {
                            const int i = r * 32 + lane;
                            ingr_slot[i] = s_k[buf][(i & 7) * PAD + (i >> 3)];
                            if (!compact) {
                                ingr_slot[2 * S_v + i] = g_cur;
                                ingr_slot[3 * S_v + i] = b_cur;
                            }
                        }
                        if (compact && lane == 0) {
                            ingr_slot[2 * S_v]     = g_cur;
                            ingr_slot[2 * S_v + 1] = b_cur;
                        }
                    }
                    if (j == 0) {
#pragma unroll
                        for (int c = 0; c < C; c++) {
                            ingr_slot[S_v + col0 + c] = v_cur[c];
                        }
                    }
                } else {
                    const int64_t snap_e = slot_rows == nullptr
                        ? state_out_e + (int64_t) target_slot * state_slot_stride
                        : (int64_t) slot_rows[(int64_t) target_slot * n_seqs + sequence] * (H * S_v * S_v) + (int64_t) h_idx * S_v * S_v;
#pragma unroll
                    for (int c = 0; c < C; c++) {
#pragma unroll
                        for (int n = 0; n < 16; n++) {
                            state_f[snap_e + (int64_t) (col0 + c) * S_v + j + 8 * n] = s[c][n];
                        }
                    }
                }
            }

            if constexpr (emit_ingredients_t) {
                if (n_tokens > K && t == n_tokens - K - 1) {
                    store_state_block(ckp_dst ? ckp_dst : state_f + ingr_region + (int64_t) S_v * S_v * H * n_seqs);
                }
            }
        }

        if (t + 1 < n_tokens) {
            stage_store_tok(buf ^ 1);
#pragma unroll
            for (int c = 0; c < C; c++) {
                v_cur[c] = v_nxt[c];
            }
            g_cur = g_nxt;
            b_cur = b_nxt;
            buf ^= 1;
            block_sync<NW>();
        }
    }

    if constexpr (emit_ingredients_t) {
        store_state_block(fin_dst ? fin_dst : state_f + ingr_region);
    } else if constexpr (!keep_rs_t) {
        store_state_block(state_f);
    }
}

static int v16_mode() {
    static const int mode = [] {
        const char * s = getenv("GGML_CUDA_GDN_V16");
        return s ? atoi(s) : 0;
    }();
    return mode;
}

static int v16_cols() {
    static const int cols = [] {
        const char * s = getenv("GGML_CUDA_GDN_V16_COLS");
        return (s && atoi(s) == 4) ? 4 : 2;
    }();
    return cols;
}

static int v16_warps() {
    static const int warps = [] {
        const char * s = getenv("GGML_CUDA_GDN_V16_WARPS");
        const int v = s ? atoi(s) : 2;
        return (v == 1 || v == 4) ? v : 2;
    }();
    return warps;
}

template <int C, int NW, bool keep_rs_t, bool emit_ingredients_t>
static void launch_inst(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, void * state_d,
        int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, const int32_t * slot_rows,
        int ingr_w, int64_t ingr_region,
        const ggml_cuda_gated_delta_net_state_src & state_src, cudaStream_t stream) {
    static_assert(128 % (4 * C * NW) == 0, "columns per block must divide S_v");
    const dim3 grid_dims(H, n_seqs, 128 / (4 * C * NW));
    const dim3 block_dims(32, NW, 1);

    static std::atomic<bool> logged{false};
    if (!logged.exchange(true, std::memory_order_relaxed)) {
        cudaFuncAttributes attr;
        CUDA_CHECK(cudaFuncGetAttributes(&attr, reinterpret_cast<const void *>(
            gated_delta_net_cuda_v16<C, NW, keep_rs_t, emit_ingredients_t>)));
        GGML_LOG_INFO("GDN V16: C=%d NW=%d emit=%d keep_rs=%d (H=%lld n_tokens=%lld prefix=%d, regs %d, local %zu B, grid %u x %u x %u)\n",
            C, NW, (int) emit_ingredients_t, (int) keep_rs_t,
            (long long) H, (long long) n_tokens, state_src.prefix_n,
            attr.numRegs, attr.localSizeBytes, grid_dims.x, grid_dims.y, grid_dims.z);
    }

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);
    const ggml_cuda_kernel_launch_params launch_params(grid_dims, block_dims, 0, stream);
    ggml_cuda_kernel_launch(gated_delta_net_cuda_v16<C, NW, keep_rs_t, emit_ingredients_t>, launch_params,
        q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, n_seqs,
        sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic,
        scale, state_slot_stride, K, slot_rows, ingr_w, ingr_region, state_src);
}

template <bool keep_rs_t, bool emit_ingredients_t>
static void launch_shape(
        const int nc, const int nw,
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, void * state_d,
        int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, const int32_t * slot_rows,
        int ingr_w, int64_t ingr_region,
        const ggml_cuda_gated_delta_net_state_src & state_src, cudaStream_t stream) {
#define GDN_V16_LAUNCH(C_, NW_) \
    launch_inst<C_, NW_, keep_rs_t, emit_ingredients_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, \
        H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, \
        scale, state_slot_stride, K, slot_rows, ingr_w, ingr_region, state_src, stream)
    if (nc == 4) {
        if (nw == 1) { GDN_V16_LAUNCH(4, 1); } else if (nw == 4) { GDN_V16_LAUNCH(4, 4); } else { GDN_V16_LAUNCH(4, 2); }
    } else {
        if (nw == 1) { GDN_V16_LAUNCH(2, 1); } else if (nw == 4) { GDN_V16_LAUNCH(2, 4); } else { GDN_V16_LAUNCH(2, 2); }
    }
#undef GDN_V16_LAUNCH
}

static bool host_direct_rows_conflict(
        const float * base, const ggml_cuda_gated_delta_net_state_src & ss,
        const std::vector<int32_t> & h_rows, const int64_t n_seqs, const int64_t D) {
    if (ss.guard_in_place == 2) {
        return true;
    }
    for (int64_t t = 0; t < n_seqs; ++t) {
        const float * r0 = base + (int64_t) h_rows[t] * ss.row_stride;
        for (int64_t s = 0; s < n_seqs; ++s) {
            for (int j = 0; j < 2; ++j) {
                const float * w = j == 0 ? ss.final_dst : ss.ckpt_dst;
                if (w == nullptr) {
                    continue;
                }
                const float * w0 = w + s * D;
                if (r0 < w0 + D && w0 < r0 + D && !(t == s && r0 == w0)) {
                    return true;
                }
            }
        }
    }
    return false;
}

static bool env_nonzero(const char * name) {
    const char * s = getenv(name);
    return s != nullptr && atoi(s) != 0;
}

static void v16_log_mode_warnings(const int mode) {
    static std::atomic<bool> warned{false};
    if (warned.exchange(true, std::memory_order_relaxed)) {
        return;
    }
    if (mode == 2) {
        GGML_LOG_WARN("GDN V16: GGML_CUDA_GDN_V16=2 is diagnostic-only (includes n_tokens=1 and long prefills)\n");
    } else if (mode == 3) {
        GGML_LOG_WARN("GDN V16: GGML_CUDA_GDN_V16=3 is diagnostic-only (runs reference kernel and bit-compares every output; synchronizes)\n");
    }
    if (env_nonzero("GGML_CUDA_GDN_R16") || env_nonzero("GGML_CUDA_GDN_SUBWARP") || env_nonzero("GGML_CUDA_GDN_REPLAY_COLS")) {
        GGML_LOG_WARN("GDN V16: GGML_CUDA_GDN_V16=%d overrides GGML_CUDA_GDN_R16 / GGML_CUDA_GDN_SUBWARP / GGML_CUDA_GDN_REPLAY_COLS\n", mode);
    }
}

struct diff_region {
    const char *     name;
    const uint32_t * a;
    const uint32_t * b;
    int64_t          n;
};

static void v16_check_bit_exact(
        const int nc, const int nw, const bool keep_rs, const bool emit_ingr,
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, void * state_d,
        const int64_t H, const int64_t n_tokens, const int64_t n_seqs,
        const int64_t sq1, const int64_t sq2, const int64_t sq3,
        const int64_t sv1, const int64_t sv2, const int64_t sv3,
        const int64_t sb1, const int64_t sb2, const int64_t sb3,
        const int64_t neqk1, const int64_t rq3,
        const float scale, const int64_t state_slot_stride, const int K, const int32_t * slot_rows,
        const int ingr_w, const int64_t ingr_region,
        const ggml_cuda_gated_delta_net_state_src & state_src, ggml_cuda_pool & pool, cudaStream_t stream) {
    constexpr int64_t S_v = 128;
    const int64_t n_attn  = n_seqs * n_tokens * H * S_v;
    const int64_t n_blk   = H * S_v * S_v;
    const int64_t n_state = n_seqs * n_blk;

    // Reference runs first into pool scratch because s_d may alias live output cache rows.
    ggml_cuda_pool_alloc<float> r_dst(pool, (size_t) n_attn);
    const bool has_ckpt = emit_ingr && ((state_src.prefix_n > 0 && n_tokens <= K) || n_tokens > K);
    const int64_t n_slots = !emit_ingr ? (keep_rs ? std::min<int64_t>(K, n_tokens) : 1) : 0;
    const int64_t r_state_elems = emit_ingr
        ? ingr_region + (has_ckpt ? 2 : 1) * n_state
        : n_slots * n_state;
    ggml_cuda_pool_alloc<float> r_state(pool, (size_t) r_state_elems);
    CUDA_CHECK(cudaMemsetAsync(r_state.get(), 0, (size_t) r_state_elems * sizeof(float), stream));

    ggml_cuda_gated_delta_net_state_src ref_ss = state_src;
    ref_ss.final_dst      = nullptr;
    ref_ss.ckpt_dst       = nullptr;
    ref_ss.ingr_dst       = nullptr;
    ref_ss.guard_in_place = 0;
    const int64_t ref_slot_stride = emit_ingr ? (int64_t) ingr_w * H * n_seqs : n_state;

    ggml_cuda_gated_delta_net_ref_f32(
        keep_rs, emit_ingr, q_d, k_d, v_d, g_d, b_d, s_d, r_dst.get(), r_state.get(),
        H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3,
        scale, ref_slot_stride, K, nullptr, ingr_w, ingr_region, ref_ss, stream);

    if (emit_ingr) {
        launch_shape<true, true>(nc, nw, q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
            H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3,
            scale, state_slot_stride, K, slot_rows, ingr_w, ingr_region, state_src, stream);
    } else if (keep_rs) {
        launch_shape<true, false>(nc, nw, q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
            H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3,
            scale, state_slot_stride, K, slot_rows, ingr_w, ingr_region, state_src, stream);
    } else {
        launch_shape<false, false>(nc, nw, q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
            H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3,
            scale, state_slot_stride, K, slot_rows, ingr_w, ingr_region, state_src, stream);
    }

    std::vector<diff_region> regs;
    regs.push_back({ "attn", (const uint32_t *) dst_d, (const uint32_t *) r_dst.get(), n_attn });

    const float * state_f = (const float *) state_d;
    if (emit_ingr) {
        bool conflict = false;
        if (state_src.guard_in_place) {
            std::vector<int32_t> h_rows((size_t) n_seqs);
            CUDA_CHECK(cudaMemcpyAsync(h_rows.data(), state_src.rows, h_rows.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            conflict = host_direct_rows_conflict(s_d, state_src, h_rows, n_seqs, n_blk);
        }
        // Only slots written by t = 0..n_tokens-1 (target_slot in [max(0, K - n_tokens), K)) are valid.
        const int64_t sl_beg = std::max<int64_t>(0, (int64_t) K - n_tokens);
        for (int64_t sl = sl_beg; sl < K; sl++) {
            for (int64_t sq = 0; sq < n_seqs; sq++) {
                const float * real_ingr = state_src.ingr_dst
                    ? state_src.ingr_dst + (sq * K + sl) * ingr_w * H
                    : state_f + sq * H * ingr_w + sl * state_slot_stride;
                const float * ref_ingr = r_state.get() + sq * H * ingr_w + sl * ref_slot_stride;
                regs.push_back({ "ingr", (const uint32_t *) real_ingr, (const uint32_t *) ref_ingr, H * ingr_w });
            }
        }
        const float * fin_real = (!conflict && state_src.final_dst) ? state_src.final_dst : (state_f + ingr_region);
        regs.push_back({ "final", (const uint32_t *) fin_real, (const uint32_t *) (r_state.get() + ingr_region), n_state });
        if (has_ckpt) {
            const float * ckp_real = (!conflict && state_src.ckpt_dst) ? state_src.ckpt_dst : (state_f + ingr_region + n_state);
            regs.push_back({ "ckpt", (const uint32_t *) ckp_real, (const uint32_t *) (r_state.get() + ingr_region + n_state), n_state });
        }
    } else {
        std::vector<int32_t> h_rows;
        if (keep_rs && slot_rows != nullptr) {
            h_rows.resize((size_t) (n_slots * n_seqs));
            CUDA_CHECK(cudaMemcpyAsync(h_rows.data(), slot_rows, h_rows.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        for (int64_t sl = 0; sl < n_slots; sl++) {
            for (int64_t sq = 0; sq < n_seqs; sq++) {
                const int64_t e_real = !keep_rs ? sq * n_blk
                    : (slot_rows == nullptr ? sq * n_blk + sl * state_slot_stride : (int64_t) h_rows[(size_t) (sl * n_seqs + sq)] * n_blk);
                const int64_t e_ref = sl * n_state + sq * n_blk;
                regs.push_back({ "snap", (const uint32_t *) (state_f + e_real), (const uint32_t *) (r_state.get() + e_ref), n_blk });
            }
        }
    }

    const size_t n_reg = regs.size();
    ggml_cuda_pool_alloc<unsigned long long> stats(pool, 2 * n_reg);
    std::vector<unsigned long long> init(2 * n_reg, 0);
    for (size_t i = 0; i < n_reg; ++i) {
        init[n_reg + i] = ULLONG_MAX;
    }
    CUDA_CHECK(cudaMemcpyAsync(stats.get(), init.data(), init.size() * sizeof(unsigned long long), cudaMemcpyHostToDevice, stream));
    for (size_t i = 0; i < n_reg; ++i) {
        const int blocks = (int) std::min<int64_t>((regs[i].n + 255) / 256, 2048);
        if (blocks > 0) {
            gdn_r16_diff<<<blocks, 256, 0, stream>>>(regs[i].a, regs[i].b, regs[i].n, stats.get() + i, stats.get() + n_reg + i);
        }
    }
    std::vector<unsigned long long> h(2 * n_reg);
    CUDA_CHECK(cudaMemcpyAsync(h.data(), stats.get(), h.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    unsigned long long total_diff = 0;
    size_t first_bad = n_reg;
    for (size_t i = 0; i < n_reg; ++i) {
        total_diff += h[i];
        if (h[i] != 0 && first_bad == n_reg) {
            first_bad = i;
        }
    }

    static std::atomic<unsigned long long> ncalls{0};
    static std::atomic<unsigned long long> nbad{0};
    const unsigned long long call_id = ++ncalls;
    if (total_diff != 0) {
        const unsigned long long bad_id = ++nbad;
        const unsigned long long idx = h[n_reg + first_bad];
        uint32_t w_v16 = 0, w_ref = 0;
        if (idx < (unsigned long long) regs[first_bad].n) {
            CUDA_CHECK(cudaMemcpyAsync(&w_v16, regs[first_bad].a + idx, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(&w_ref, regs[first_bad].b + idx, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        float f_v16 = 0.0f, f_ref = 0.0f;
        memcpy(&f_v16, &w_v16, sizeof(float));
        memcpy(&f_ref, &w_ref, sizeof(float));
        GGML_LOG_WARN("GDN V16 bit-check call %llu (C=%d NW=%d emit=%d keep_rs=%d n_tokens=%lld prefix=%d H=%lld n_seqs=%lld): "
            "MISMATCH total=%llu first=%s[%llu] v16=0x%08x (%.9g) ref=0x%08x (%.9g) bad_calls=%llu\n",
            call_id, nc, nw, (int) emit_ingr, (int) keep_rs, (long long) n_tokens, state_src.prefix_n,
            (long long) H, (long long) n_seqs, total_diff, regs[first_bad].name, idx, w_v16, f_v16, w_ref, f_ref, bad_id);
    } else if ((call_id & (call_id - 1)) == 0) {
        GGML_LOG_INFO("GDN V16 bit-check call %llu (C=%d NW=%d emit=%d keep_rs=%d n_tokens=%lld prefix=%d H=%lld n_seqs=%lld): "
            "0 differing words (bad_calls=%llu)\n",
            call_id, nc, nw, (int) emit_ingr, (int) keep_rs, (long long) n_tokens, state_src.prefix_n,
            (long long) H, (long long) n_seqs, nbad.load(std::memory_order_relaxed));
    }
}

} // namespace gdn_v16

bool ggml_cuda_gated_delta_net_v16_try(
        bool keep_rs, bool emit_ingr,
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, void * state_d,
        int64_t S_v, int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, const int32_t * slot_rows,
        int ingr_w, int64_t ingr_region,
        const ggml_cuda_gated_delta_net_state_src & state_src, ggml_cuda_pool * pool, cudaStream_t stream) {
    using namespace gdn_v16;
    const int mode = v16_mode();
    if (mode <= 0 || S_v != 128 || n_tokens < 1 || (emit_ingr && !keep_rs)) {
        return false;
    }
    if (state_src.prefix_n > 0 && !emit_ingr) {
        return false;
    }
    v16_log_mode_warnings(mode);
    if (mode == 1) {
        if (n_tokens + state_src.prefix_n < 2) {
            return false;
        }
        // Keep long non-prefix prefills on khshare / chunked paths.
        if (state_src.prefix_n == 0 && !emit_ingr && n_tokens >= 64) {
            return false;
        }
    }
    const auto & dev = ggml_cuda_info().devices[ggml_cuda_get_device()];
    if (dev.warp_size != 32 || dev.cc < GGML_CUDA_CC_AMPERE) {
        return false;
    }
    if ((((uintptr_t) q_d) | ((uintptr_t) k_d)) % 16 != 0 || (sq1 % 4) != 0 || (sq2 % 4) != 0 || (sq3 % 4) != 0) {
        return false;
    }
    if (state_src.prefix_n > 0) {
        if (state_src.prefix == nullptr || state_src.prefix_w == nullptr ||
                ((uintptr_t) state_src.prefix) % 8 != 0 ||
                (state_src.prefix_s1 % 2) != 0 || (state_src.prefix_ingr_w % 2) != 0) {
            return false;
        }
    }

    const int nc = v16_cols();
    const int nw = v16_warps();

    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    if (mode == 3 && pool != nullptr) {
        CUDA_CHECK(cudaStreamIsCapturing(stream, &cap));
    }
    if (mode == 3 && pool != nullptr && cap == cudaStreamCaptureStatusNone) {
        v16_check_bit_exact(nc, nw, keep_rs, emit_ingr, q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
            H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3,
            scale, state_slot_stride, K, slot_rows, ingr_w, ingr_region, state_src, *pool, stream);
        return true;
    }
    if (mode == 3 && pool != nullptr && cap != cudaStreamCaptureStatusNone) {
        static std::atomic<bool> cap_warned{false};
        if (!cap_warned.exchange(true, std::memory_order_relaxed)) {
            GGML_LOG_WARN("GDN V16: GGML_CUDA_GDN_V16=3 skipped bit-check during CUDA graph capture (set GGML_CUDA_DISABLE_GRAPHS=1 to check in-server)\n");
        }
    }

    if (emit_ingr) {
        launch_shape<true, true>(nc, nw, q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
            H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3,
            scale, state_slot_stride, K, slot_rows, ingr_w, ingr_region, state_src, stream);
    } else if (keep_rs) {
        launch_shape<true, false>(nc, nw, q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
            H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3,
            scale, state_slot_stride, K, slot_rows, ingr_w, ingr_region, state_src, stream);
    } else {
        launch_shape<false, false>(nc, nw, q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
            H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3,
            scale, state_slot_stride, K, slot_rows, ingr_w, ingr_region, state_src, stream);
    }
    return true;
}
