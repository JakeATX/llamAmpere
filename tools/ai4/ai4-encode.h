#pragma once
// AI4 (Ampere INT4, formerly D2) GPU encoder / decoder entry points (tools/ai4/ai4-encode.cu). Host pointers in, host pointers out.
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// W: [M][K] f32 original-basis weights. G: optional LDLQ feedback [K/256][32][8][256] (G[c][a][k] = sum_b Ui[a][b]
// U[8c + b][k], Ui = inverse of U's 8x8 diagonal block, U = ai4_ldlq_prep factor), gok[K/256] = 1 where the factor
// exists. out: (M/16)*(K/1024)*8448 bytes. ms: kernel time. Returns 0 on success.
int ai4_cuda_encode(const float * W, int64_t K, int64_t M, const float * G, const uint8_t * gok, float lrms, float smul,
                   uint8_t * out, float * ms);
// AI4 bytes -> [M][K] f32 original basis, through the ggml CUDA backend's reconstruct kernel (ggml-cuda/ai4.cuh).
int ai4_cuda_reconstruct(const uint8_t * q, int64_t K, int64_t M, float * W);
size_t ai4_cuda_encode_smem_bytes(void);
// AIQ (private): full-K LDLQ, AI4 1024 layout. H: [K][K] f32 symmetric input Hessian (original basis). mode 0 full,
// 1 block-diagonal (256 blocks, rotated basis), 2 diagonal only (original basis). damp: added to the diagonal as
// damp * mean(diag). werr (optional): H-weighted relative reconstruction error in the rotated basis (undamped H).
int ai4_cuda_encode_fullk(const float * W, const float * H, int64_t K, int64_t M, int mode, double damp, float lrms,
                          float smul, uint8_t * out, float * ms, double * werr);
// EXLGAP: the same full-K LDLQ for the AI+ unit-128 layout at rate kbits (4 = AI4_U, 5 = AI5_U; K % 256, rotation block
// 256 = ai4u_rot_size(K) for those K; out (M/16)*(K/128)*(32 + 256*kbits) bytes), or kbits 0 = the AI4 1024 layout as above.
// refit 1: after the encode, multiply all fp16 group scales of each row by the closed-form c_r that minimizes the H'-weighted
// row error with the trellis fixed (EXL3 refit_scales, output side). werr_pre (optional, refit only): error before the refit.
// weight_metric 1: minimize row-scale-weighted tile error (rate 2..6 only); 0 keeps the legacy normalized metric.
int ai4_cuda_encode_fullk_rate(const float * W, const float * H, int64_t K, int64_t M, int mode, double damp, float lrms,
                               float smul, int kbits, uint32_t lmul, int refit, uint8_t * out, float * ms, double * werr,
                               double * werr_pre, int weight_metric
#ifdef __cplusplus
                               = 0
#endif
                               );
int ai4_cuda_werr(const float * W, const float * Wh, const float * H, int64_t K, int64_t M, double * werr);
// unit = 0: AI4 (1024 layout, K % 1024, as above). unit = 1: AI4_U (unit-128 layout, K % 128, rotation block rb =
// ai4u_rot_size(K); G is then [K/rb][rb/8][8][rb] and gok[K/rb]). out: (M/16)*(K/128)*1056 bytes for both.
int ai4_cuda_encode_layout(const float * W, int64_t K, int64_t M, const float * G, const uint8_t * gok, float lrms, float smul,
                          int unit, uint8_t * out, float * ms);
int ai4_cuda_reconstruct_layout(const uint8_t * q, int64_t K, int64_t M, int unit, float * W);
// AI+ unit-128 layout at rate kbits (2..6; 4 = AI4_U through the generalized kernel, 5 = AI5_U, 6 = AI6_U); G / gok as for
// unit = 1. lmul: the type's trellis constant (ggml_ai4_trellis_mul; 0x83DCD12D for everything but AI6_U / AI5_UN).
// out: (M/16)*(K/128)*(32 + 256*kbits) bytes.
// sfix (optional, private A/B control, do not ship): per-128-group scale table [M][K/128] f32 used in place of each
// group's own RMS (rank-1 / per-row scale fits); nullptr reproduces the shipped encoder's bytes exactly.
// weight_metric 1: branch cost e^2 * (row_scale / tile_max_scale)^2, instead of normalized e^2 (0, legacy).
int ai4_cuda_encode_rate(const float * W, int64_t K, int64_t M, const float * G, const uint8_t * gok, float lrms, float smul,
                         int kbits, uint32_t lmul, uint8_t * out, float * ms, const float * sfix
#ifdef __cplusplus
                         = nullptr
#endif
                         , int weight_metric
#ifdef __cplusplus
                         = 0
#endif
                         );
// rotated != 0: leave the output in the rotated basis (the values the decoder multiplies with rotated activations), which is
// bit-exact against aik_dequantize_slab_rotated_c; 0: original basis (unrotated).
int ai4_cuda_reconstruct_rate(const uint8_t * q, int64_t K, int64_t M, int kbits, uint32_t lmul, int rotated, float * W);
// the same for the AI+ span layout (AIk_S: K % 1024 == 0, one fp16 scale per row and 1024 K, no per-unit header);
// q is (M/16)*(K/128)*(4 + 256*kbits) bytes.
int ai4_cuda_reconstruct_span(const uint8_t * q, int64_t K, int64_t M, int kbits, uint32_t lmul, int rotated, float * W);

#ifdef __cplusplus
}
#endif
