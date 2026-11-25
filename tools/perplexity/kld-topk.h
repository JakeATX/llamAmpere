#pragma once
// Top-K agreement metrics for the KL-divergence scorer (llama-perplexity --kl-divergence).
//
//   KL@K   KL(base || q) over the base model's K most likely tokens, both distributions renormalized over those K.
//   EAR@K  sum over the same K tokens of min(p, q), both renormalized: the acceptance rate a speculative-sampling
//          verifier (the base model) would give a drafter that samples from q, restricted to those K tokens.
//          1 = identical predictions.
//   EAR    sum over the whole vocabulary of min(p, q) = 1 - total variation distance (no renormalization): the
//          acceptance rate of exact speculative sampling with q as the drafter.
//
// Header-only and free of llama.cpp types so that tests/test-kld-topk.cpp can check it against hand-made distributions.

#include <cmath>

namespace kld_topk {

constexpr int K = 10;

// Running top-K of the base log-probs, largest first. Ties keep the lower token id (pushes come in id order and only a
// strictly larger value moves ahead of an equal one).
struct top_k {
    int   n = 0;
    int   id[K];
    float lp[K];

    void push(int i, float v) {
        if (n == K && !(v > lp[K - 1])) {
            return;
        }
        int j = n < K ? n++ : K - 1;
        while (j > 0 && v > lp[j - 1]) {
            id[j] = id[j - 1];
            lp[j] = lp[j - 1];
            --j;
        }
        id[j] = i;
        lp[j] = v;
    }
};

// One token's term of the full-vocabulary EAR: min(p, q) from natural log-probs.
inline float accept(float lp_base, float lp_q) {
    return expf(lp_base < lp_q ? lp_base : lp_q);
}

struct result {
    double kl  = 0.0; // KL@K
    double ear = 0.0; // EAR@K
};

// lp_base[k], lp_q[k]: natural log-probs of the same n tokens under base and q (any common offset per side is fine:
// both sides are renormalized over the n tokens).
inline result over(const float * lp_base, const float * lp_q, int n) {
    result r;
    if (n <= 0) {
        return r;
    }
    double mb = lp_base[0], mq = lp_q[0];
    for (int k = 1; k < n; ++k) {
        mb = lp_base[k] > mb ? lp_base[k] : mb;
        mq = lp_q[k]    > mq ? lp_q[k]    : mq;
    }
    double sb = 0.0, sq = 0.0;
    for (int k = 0; k < n; ++k) {
        sb += exp(lp_base[k] - mb);
        sq += exp(lp_q[k]    - mq);
    }
    const double lb = mb + log(sb), lq = mq + log(sq);
    for (int k = 0; k < n; ++k) {
        const double a = lp_base[k] - lb, b = lp_q[k] - lq;
        const double p = exp(a), q = exp(b);
        r.kl  += p * (a - b);
        r.ear += p < q ? p : q;
    }
    return r;
}

} // namespace kld_topk
