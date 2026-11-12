#pragma once

#include <stdint.h>

#if defined(__CUDACC__)
#define GGML_LSA_RECALL_HD __host__ __device__
#else
#define GGML_LSA_RECALL_HD
#endif

enum {
    GGML_LSA_RECALL_BUDGETS = 4,
    GGML_LSA_RECALL_HORIZONS = 3,
    GGML_LSA_RECALL_PLANES = GGML_LSA_RECALL_BUDGETS*GGML_LSA_RECALL_HORIZONS,
};

GGML_LSA_RECALL_HD inline int ggml_lsa_recall_budget_pct(int budget) {
    return budget == 0 ? 100 : budget == 1 ? 110 : budget == 2 ? 125 : 150;
}

// The visible length includes the tail; the sealed boundary is always page-aligned.
GGML_LSA_RECALL_HD inline bool ggml_lsa_workset_partial_page(uint32_t visible_end) {
    return visible_end % 128 != 0;
}

GGML_LSA_RECALL_HD inline bool ggml_lsa_history_contiguous(uint32_t prev_end, uint32_t prev_resolve,
        uint32_t qpos, uint32_t visible_end, uint32_t rows, uint32_t resolve) {
    return prev_resolve != 0 && prev_resolve + 1 == resolve && prev_end == qpos && visible_end == qpos + rows;
}

// Count physical record pages, once after OR across query rows. Existing card hits are never true positives.
// A newly sealed record is not available to the early predictor: its miss remains in the recall denominator.
struct ggml_lsa_recall_counts {
    uint32_t actual_new;
    uint32_t predicted_new;
    uint32_t hit;
    uint32_t newly_sealed;

    GGML_LSA_RECALL_HD void observe(bool resident, bool actual, bool predicted, bool old) {
        if (resident) {
            return;
        }
        const bool eligible_prediction = predicted && old;
        actual_new += actual;
        predicted_new += eligible_prediction;
        hit += actual && eligible_prediction;
        newly_sealed += actual && !old;
    }

    GGML_LSA_RECALL_HD void add(const ggml_lsa_recall_counts & b) {
        actual_new += b.actual_new;
        predicted_new += b.predicted_new;
        hit += b.hit;
        newly_sealed += b.newly_sealed;
    }
};

// Eight 16-cell strips in one logical 128-cell page. Shared-head masks give the same per-head and union counts;
// do not mistake that equality for evidence about independent per-head selection. Denominators include sink/tail.
struct ggml_lsa_workset_counts {
    uint32_t pages, strips, prev_pages, prev_strips, new_pages, new_strips;

    GGML_LSA_RECALL_HD void observe(uint32_t current, uint32_t previous) {
        pages += current != 0;
        prev_pages += previous != 0;
        new_pages += current != 0 && previous == 0;
        for (int b = 0; b < 8; ++b) {
            strips += (current >> b) & 1;
            prev_strips += (previous >> b) & 1;
            new_strips += ((current & ~previous) >> b) & 1;
        }
    }

    GGML_LSA_RECALL_HD void add(const ggml_lsa_workset_counts & b) {
        pages += b.pages;
        strips += b.strips;
        prev_pages += b.prev_pages;
        prev_strips += b.prev_strips;
        new_pages += b.new_pages;
        new_strips += b.new_strips;
    }
};

#undef GGML_LSA_RECALL_HD
