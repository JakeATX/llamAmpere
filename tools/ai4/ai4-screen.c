// Private CPU reconstruction screening. Build as a shared library with -O3 -march=native -fopenmp.
#include "ggml-ai4.h"
#include <omp.h>

int ai4_screen_rotate(float * X, int K, int M, int inverse) {
    if (K <= 0 || K % 128 || M <= 0) return 2;
    const int rb = ai4u_rot_size(K);
#pragma omp parallel for schedule(static)
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < K; c += rb) {
            if (inverse) ai4_unrotate_n(X + (size_t) r*K + c, rb);
            else ai4_rotate_n(X + (size_t) r*K + c, rb);
        }
    }
    return 0;
}

// W and Q are [M,K]. H's diagonal supplies the existing block feedback; scoring uses a full H in Python.
int ai4_screen_encode(const float * W, int K, int M, const float * lam, int bits, float smul,
                      int weight_metric, int row_scale, float * Q) {
    if (M <= 0 || M % 16 || K <= 0 || K % 128 || bits < 2 || bits > 6 ||
        weight_metric < 0 || weight_metric > 1 || row_scale < 0 || row_scale > 1) return 2;
    const int rb = ai4u_rot_size(K), nb = K/rb;
    const uint32_t mul = bits == 6 ? 0xA1EEE321u : AI4_MUL;
    float * U = lam ? malloc((size_t) nb*rb*rb*sizeof(float)) : NULL;
    uint8_t * ok = lam ? malloc(nb) : NULL;
    float * fixed = row_scale ? malloc((size_t) M*(K/128)*sizeof(float)) : NULL;
    for (int r = 0; row_scale && r < M; ++r) {
        double ss = 0.0;
        for (int c = 0; c < K; ++c) { const double v = W[(size_t) r*K + c]; ss += v*v; }
        const float rms = sqrt(ss/K);
        for (int g = 0; g < K/128; ++g) fixed[(size_t) r*(K/128) + g] = rms;
    }
    for (int b = 0; lam && b < nb; ++b) ok[b] = ai4_ldlq_prep_n(lam + b*rb, U + (size_t) b*rb*rb, rb);
#pragma omp parallel
    {
        ai4_work * ws = ai4_work_new();
        uint8_t * q = malloc((size_t) K/128*AI4U_UNIT_BYTES(bits));
#pragma omp for schedule(dynamic)
        for (int r = 0; r < M; r += 16) {
            aik_quantize_slab_rb_cfm(ws, bits, mul, W + (size_t) r*K, K, lam, U, ok, q,
                                     rb, smul, row_scale ? fixed + (size_t) r*(K/128) : NULL, weight_metric);
            aik_dequantize_slab_c(bits, mul, q, Q + (size_t) r*K, K);
        }
        free(q);
        free(ws);
    }
    free(U);
    free(ok);
    free(fixed);
    return 0;
}

