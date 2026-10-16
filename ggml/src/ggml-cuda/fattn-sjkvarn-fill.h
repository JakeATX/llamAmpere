#pragma once

#include <cstdint>

// Costs are in 16-position strip units, not measured time. Two strips cover block setup or a combine launch.
static inline double ggml_cuda_sj_fill_cost(int parallel_blocks, int nsm, int occupancy, int ntiles_dst,
                                          int n_kv, int kv_splits, int64_t nrows) {
    const int64_t capacity = int64_t(nsm)*occupancy;
    const int64_t blocks = int64_t(ntiles_dst)*parallel_blocks;
    const int64_t waves = (blocks + capacity - 1)/capacity;
    const int64_t split_positions = int64_t(16)*kv_splits*parallel_blocks;
    const int64_t strips = (n_kv + split_positions - 1)/split_positions;
    // Estimate 16 partial-row reductions per strip; the combine has one 256-thread block per output row.
    const double combine = parallel_blocks > 1 ? 2.0 + ((nrows + int64_t(nsm) - 1)/nsm)*(parallel_blocks/16.0) : 0.0;
    return waves*(2.0 + strips) + combine;
}

static inline int ggml_cuda_sj_fill_split(int nsm, int occupancy, int ntiles_dst, int ntiles_kv,
                                         int n_kv, int kv_splits, int64_t nrows) {
    // Cover every SM when the KV tile limit permits it, then minimize the estimated cost.
    const int cover = (nsm - 1)/ntiles_dst + 1;
    const int first = cover < ntiles_kv ? cover : ntiles_kv;
    int best = first;
    double cost_best = ggml_cuda_sj_fill_cost(best, nsm, occupancy, ntiles_dst, n_kv, kv_splits, nrows);
    for (int p = first + 1; p <= ntiles_kv; ++p) {
        const double cost = ggml_cuda_sj_fill_cost(p, nsm, occupancy, ntiles_dst, n_kv, kv_splits, nrows);
        if (cost < cost_best) {
            best = p;
            cost_best = cost;
        }
    }
    return best;
}
