#pragma once

// fork: decide per-token drops from normalized weights; a partial resident is kept.
#if defined(__CUDACC__) || defined(__HIPCC__)
#define GGML_SKIPMASS_HD __host__ __device__
#else
#define GGML_SKIPMASS_HD
#endif
static GGML_SKIPMASS_HD inline void ggml_moe_skipmass_select(
        const float * weights, const unsigned char * hot, int n, int mode, float threshold, unsigned char * drop) {
    if (n < 1 || n > 10) { return; }
    bool chosen[10] = {};
    float mass = 0.0f;
    for (int i = 0; i < n; ++i) { drop[i] = 0; }
    for (int i = 0; i < n; ++i) {
        if (!(weights[i] >= 0.0f && weights[i] <= 1.0f)) { return; }
    }
    for (int step = 0; step < n; ++step) {
        int best = -1;
        for (int i = n - 1; i >= 0; --i) {
            if (!chosen[i] && !hot[i] && (best < 0 || weights[i] < weights[best])) { best = i; }
        }
        if (best < 0) { break; }
        chosen[best] = true;
        if (mode == 1 ? weights[best] < threshold : mass + weights[best] <= threshold) {
            drop[best] = 1;
            mass += weights[best];
        }
    }
}
#undef GGML_SKIPMASS_HD
