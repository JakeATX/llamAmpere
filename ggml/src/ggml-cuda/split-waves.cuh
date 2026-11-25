#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>

// [WREAD08] Wave-efficient K-split count for the mid-T (9..64 rows) GEMM launches.
// units: CTAs of one K slice; nsm: SM count; occ: resident CTAs per SM; smax: upper split bound.
// Cost model: each split CTA does 1/s of the K work plus a fixed prologue/epilogue share
// (alpha = 1/8 of a full-K CTA, measured on an RTX 3090 for AIT5_U at M = 6144/10240/12288 and
// AIT3/AIT4 at M = 5120/17408), so cost(s) = waves(s) * (8 + s) / s with waves(s) = ceil(units * s / (nsm * occ)).
// Returns the s in [1, smax] with the lowest cost; ties keep the smaller s.
// The old rule (nsm / units) assumes everything fits one wave and keeps s = 1 as soon as units > nsm,
// which leaves 1.48 waves on a 46-SM card at M = 17408.
// The result depends on the SM count, so outputs are fixed per card model (as before), not across models.
static inline int ggml_cuda_split_for_waves(int units, int nsm, int occ, int smax) {
    if (units <= 0 || nsm <= 0) {
        return 1;
    }
    const int64_t slots = (int64_t) nsm * std::max(occ, 1);
    smax = std::max(smax, 1);
    int best = 1;
    int64_t best_num = 0, best_den = 1; // cost as a fraction num/den, compared exactly
    for (int s = 1; s <= smax; ++s) {
        const int64_t waves = ((int64_t) units * s + slots - 1) / slots;
        const int64_t num = waves * (8 + s), den = s;
        if (s == 1 || num * best_den < best_num * den) {
            best = s; best_num = num; best_den = den;
        }
    }
    return best;
}

// GGML_CUDA_SPLIT_WAVES=1: the wave rule above at the three mid-T split sites (ait k_ait_mmq mid-T, AIT5_U
// ai5_midt_splits, FEWPORT port 3 ai4_midt_u_splits). Default 0 = the old nsm / units rule (release behaviour) until
// the 4-user speed gate passes. It changes float summation order only (split count), not the math.
static inline bool ggml_cuda_split_waves_on() {
    static const bool v = [] {
        const char * e = getenv("GGML_CUDA_SPLIT_WAVES");
        return e != nullptr && *e != '\0' && atoi(e) != 0;
    }();
    return v;
}

// units = CTAs per K slice at occupancy 1 (the three mid-T kernels run one CTA per SM at these shapes)
static inline int ggml_cuda_midt_splits(int units, int nsm, int smax) {
    return ggml_cuda_split_waves_on() ? ggml_cuda_split_for_waves(units, nsm, 1, smax) : (units > 0 ? nsm / units : 1);
}
