#pragma once

// NMSE over the positions where both values are finite. Matching -inf (a mask)
// is ignored. NaN, +inf, or a -inf only one side has is a failing score:
// plain sum((a-b)^2)/sum(a^2) is NaN in those cases, and NaN > max_err is false.

#include <cmath>
#include <cstddef>

inline double nmse_masked_inf(const float * a, const float * b, size_t n) {
    double sse  = 0.0;
    double saa  = 0.0;
    size_t nfin = 0;

    for (size_t i = 0; i < n; i++) {
        const float ai = a[i];
        const float bi = b[i];

        if (std::isnan(ai) || std::isnan(bi) || ai == INFINITY || bi == INFINITY) {
            return INFINITY;
        }
        const bool a_mask = ai == -INFINITY;
        const bool b_mask = bi == -INFINITY;
        if (a_mask != b_mask) {
            return INFINITY;
        }
        if (a_mask) {
            continue;
        }

        // squares of finite floats can overflow float32; accumulate in double
        const double da = (double) ai;
        const double db = (double) bi;
        const double d  = da - db;
        sse += d*d;
        saa += da*da;
        nfin++;
    }

    // no finite positions (n == 0, or every position is a matching -inf)
    if (nfin == 0) {
        return 0.0;
    }
    // all-zero finite part: 0/0 would be NaN; the raw sum is 0 when they match
    if (saa == 0.0) {
        return sse;
    }
    const double score = sse/saa;
    // a NaN score would pass `err > max_err`
    return std::isfinite(score) ? score : INFINITY;
}
