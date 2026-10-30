#pragma once

// fork: decide per-token drops from normalized weights; a partial resident is kept.
#if defined(__CUDACC__) || defined(__HIPCC__)
#define GGML_SKIPMASS_HD __host__ __device__
#else
#define GGML_SKIPMASS_HD
#endif
static GGML_SKIPMASS_HD inline bool ggml_moe_skipmass_select(
        const float * weights, const unsigned char * hot, int n, int mode, float threshold, unsigned char * drop) {
    if (n < 1 || n > 10) { return false; }
    bool chosen[10] = {};
    bool changed = false;
    float mass = 0.0f;
    for (int i = 0; i < n; ++i) { drop[i] = 0; }
    for (int i = 0; i < n; ++i) {
        if (!(weights[i] >= 0.0f && weights[i] <= 1.0f)) { return false; }
    }
    if (mode == 1) {
        for (int i = 0; i < n; ++i) {
            drop[i] = !hot[i] && weights[i] < threshold;
            changed |= drop[i] != 0;
        }
        return changed;
    }
    for (int step = 0; step < n; ++step) {
        int best = -1;
        for (int i = n - 1; i >= 0; --i) {
            if (!chosen[i] && !hot[i] && (best < 0 || weights[i] < weights[best])) { best = i; }
        }
        if (best < 0) { break; }
        chosen[best] = true;
        // Remaining weights are at least this large and cannot fit either.
        if (!(mass + weights[best] <= threshold)) { break; }
        drop[best] = 1;
        changed = true;
        mass += weights[best];
    }
    return changed;
}
#undef GGML_SKIPMASS_HD
