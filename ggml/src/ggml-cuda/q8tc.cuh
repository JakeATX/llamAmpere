#pragma once
// INKNVFP4K Q8TC: dense Q8_0 at T = 4..8 on f16 tensor cores (kernel + layout notes in q8tc-kern.cuh).
// Opt-in GGML_CUDA_Q8TC=1 (f16 activation) or 2 (HILO, ~f32 activation); default 0 = off until gated. Lowest width
// GGML_CUDA_Q8TC_MINT (default 4). Taken from inside the Q8A16 plain / INKGLUE group launches, so the same graph nodes as
// Q8A16 (grouped QKV, attn_output); members over 16384 rows (the output heads) stay on Q8A16.
#include "common.cuh"
#include "q8a16-kern.cuh"

int ggml_cuda_q8tc_mode();

// a: filled q8a16_args (members, no ids); w: first member weight (name picks the per-tensor mode), group: grouped launch.
// Launches and returns true when taken; false = nothing launched.
bool ggml_cuda_q8tc_try(ggml_backend_cuda_context & ctx, q8a16_args & a, int T, int nb, const ggml_tensor * w, bool group);

// Q8TC-MU (GGML_CUDA_Q8TC_MU=1): plain Q8_0 mul_mat with 9..32 columns as <= 8-column Q8TC launches; false = not taken.
bool ggml_cuda_q8tc_wide(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
// Q8TC-HLF: GGML_CUDA_Q8TC_IA=1 (mode 2 independent accumulators)
bool ggml_cuda_q8tc_ia();
// the checks of ggml_cuda_q8tc_try without the launch: fills a.blk_end, returns the mode for this member set (0 = refused)
int ggml_cuda_q8tc_prep(ggml_backend_cuda_context & ctx, q8a16_args & a, int T, int nb, const ggml_tensor * w, bool group);

// FUSE-WO (q8tc-fwo.cu, GGML_CUDA_Q8TC_FUSEWO=1, default off): the attn_output launch also runs the Inkling attn-out
// short-conv ring epilogue of its output channels (state windows, 4-tap conv, SCALE(ADD(x, conv)), residual ADD), the
// arithmetic of ssm_conv_ring_f32<false, 4, n_t, epi = true>; the mm output x is still stored.
struct q8tc_epi {
    ggml_cuda_conv_ring_args ring;
    const float * w;        // conv weight [4, channels], w_nb1 bytes per channel
    int           w_nb1;
    float *       out;      // [channels, T] contiguous, out_nb1 bytes per column
    int           out_nb1;
    float         scale;
    float         bias;
    const float * res;      // residual like out, or nullptr
};
bool ggml_cuda_q8tc_fusewo_on();
// src0 Q8_0 weight, src1 f32 [K, T], dst f32 [rows, T]: q8tc matmul + e epilogue; false = nothing launched
bool ggml_cuda_q8tc_fwo(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const q8tc_epi & e);
