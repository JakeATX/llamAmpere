#pragma once

// [#69] sampled MTP chain: the in-graph draw of one chained draft step and its host-side re-derivation.
//
// The chain decode drafts every depth in one graph. With sampling on, each step draws its token on the GPU
// from the draft sampler chain top_k -> top_p -> min_p -> temperature, using a uniform the host drew before
// the round (inverse CDF instead of the dist sampler's own RNG). The host then re-derives every step from
// the emitted candidates in float64 with the sampler semantics of llama-sampler.cpp. It records the
// distribution each token came from (exact p/q holds by construction) and cuts the chain at the first step
// where the GPU disagreed, because later steps were conditioned on the GPU's token. The GPU draw therefore
// only has to agree almost always, never exactly: float rounding at a top-p or CDF boundary costs a shorter
// draft, not correctness.
//
// Header-only (ggml + llama types) so tests can build the graph on a backend and compare with the host.

#include "ggml.h"
#include "llama.h"

#include <cmath>
#include <cstdint>
#include <vector>

// one step's output row, F32: [0] drawn token id, [1] drawn position in the sorted candidates,
// [2, 2+k) candidate token ids sorted by logit (descending), [2+k, 2+2k) their raw (pre-temperature) logits
#define LLAMA_MTP_CHAIN_ROW(k) (2 + 2*(k))

// candidates per step: the graph's top-k width; requests with a larger top_k draft serially
#define LLAMA_MTP_CHAIN_TOP_K_MAX 64

// the [4] F32 sampling-parameter input
enum llama_mtp_chain_samp_idx {
    LLAMA_MTP_CHAIN_SAMP_INV_TEMP  = 0, // 1 / temperature
    LLAMA_MTP_CHAIN_SAMP_TOP_P     = 1, // top_p; 2 when top_p >= 1 (off)
    LLAMA_MTP_CHAIN_SAMP_LOG_MIN_P = 2, // log(min_p); -1e30 when min_p <= 0 (off)
    LLAMA_MTP_CHAIN_SAMP_N         = 4,
};

static inline void llama_mtp_chain_samp_pack(float temp, float top_p, float min_p, float * out) {
    out[LLAMA_MTP_CHAIN_SAMP_INV_TEMP]  = 1.0f / temp;
    out[LLAMA_MTP_CHAIN_SAMP_TOP_P]     = top_p >= 1.0f ? 2.0f : top_p;
    out[LLAMA_MTP_CHAIN_SAMP_LOG_MIN_P] = min_p > 0.0f ? logf(min_p) : -1e30f;
    out[3] = 0.0f;
}

// logits: F32 [n_v, 1], one step's draft logits (positions in the draft vocabulary map when id_map is set)
// samp:   F32 [LLAMA_MTP_CHAIN_SAMP_N], see llama_mtp_chain_samp_pack
// u:      F32 [1], the step's uniform in [0, 1)
// id_map: I32 [n_v] vocabulary ids of the logit positions, or nullptr when positions are token ids
// returns the step's output row F32 [LLAMA_MTP_CHAIN_ROW(k), 1]; *id_out is the drawn token id, I32 [1],
// for the next step's token embedding
static inline ggml_tensor * llama_mtp_chain_sample_graph(
        ggml_context * ctx,
        ggml_tensor  * logits,
        int32_t        k,
        ggml_tensor  * samp,
        ggml_tensor  * u,
        ggml_tensor  * id_map,
        ggml_tensor ** id_out) {
    const int64_t n_v = logits->ne[0];
    GGML_ASSERT(logits->ne[1] == 1 && k >= 1 && k <= n_v && k <= LLAMA_MTP_CHAIN_TOP_K_MAX);

    // top_k, then sort the k candidates by logit, descending
    ggml_tensor * cand = ggml_top_k(ctx, logits, k);                                        // I32 [k, 1], unordered
    ggml_tensor * lk   = ggml_get_rows(ctx, ggml_reshape_2d(ctx, logits, 1, n_v), cand);     // F32 [1, k]
    lk = ggml_reshape_2d(ctx, lk, k, 1);
    ggml_tensor * ord  = ggml_argsort(ctx, lk, GGML_SORT_ORDER_DESC);                       // I32 [k, 1]
    ggml_tensor * ls   = ggml_get_rows(ctx, ggml_reshape_2d(ctx, lk, 1, k), ord);            // F32 [1, k]
    ls = ggml_reshape_2d(ctx, ls, k, 1);
    ggml_tensor * pos  = ggml_get_rows(ctx, ggml_reshape_2d(ctx, cand, 1, k), ord);          // I32 [1, k]
    ggml_tensor * ids  = id_map == nullptr ? pos
        : ggml_get_rows(ctx, ggml_reshape_2d(ctx, id_map, 1, id_map->ne[0]), ggml_reshape_1d(ctx, pos, k));

    ggml_tensor * inv_t = ggml_view_1d(ctx, samp, 1, LLAMA_MTP_CHAIN_SAMP_INV_TEMP  * sizeof(float));
    ggml_tensor * top_p = ggml_view_1d(ctx, samp, 1, LLAMA_MTP_CHAIN_SAMP_TOP_P     * sizeof(float));
    ggml_tensor * lminp = ggml_view_1d(ctx, samp, 1, LLAMA_MTP_CHAIN_SAMP_LOG_MIN_P * sizeof(float));

    // top_p on the temperature-1 distribution of the candidates: keep while the mass before a candidate < top_p
    ggml_tensor * p1   = ggml_soft_max(ctx, ls);
    ggml_tensor * excl = ggml_sub(ctx, ggml_cumsum(ctx, p1), p1);
    ggml_tensor * keep = ggml_step(ctx, ggml_neg(ctx, ggml_sub(ctx, excl, top_p)));

    // min_p: keep logit >= max logit + log(min_p) (the sorted first candidate is the max)
    ggml_tensor * l0 = ggml_view_2d(ctx, ls, 1, 1, ls->nb[1], 0);
    keep = ggml_mul(ctx, keep, ggml_step(ctx, ggml_sub(ctx, ggml_sub(ctx, ls, l0), lminp)));

    // temperature over the kept prefix, then the inverse CDF: the first candidate whose running mass reaches
    // u * total, i.e. the number of candidates whose running mass is below it
    ggml_tensor * lt    = ggml_add(ctx, ggml_mul(ctx, ls, inv_t), ggml_scale_bias(ctx, keep, 1e30f, -1e30f));
    ggml_tensor * cdf   = ggml_cumsum(ctx, ggml_soft_max(ctx, lt));
    ggml_tensor * tot   = ggml_view_2d(ctx, cdf, 1, 1, cdf->nb[1], (size_t) (k - 1) * sizeof(float));
    ggml_tensor * thr   = ggml_mul(ctx, ggml_reshape_2d(ctx, u, 1, 1), tot);
    ggml_tensor * below = ggml_step(ctx, ggml_neg(ctx, ggml_sub(ctx, cdf, thr)));
    ggml_tensor * pick  = ggml_sum_rows(ctx, below);                                         // F32 [1, 1]

    ggml_tensor * pick_i = ggml_reshape_1d(ctx, ggml_cast(ctx, pick, GGML_TYPE_I32), 1);
    ggml_tensor * id     = ggml_get_rows(ctx, ids, pick_i);                                   // I32 [1, 1]

    ggml_tensor * idsf = ggml_reshape_2d(ctx, ggml_cast(ctx, ids, GGML_TYPE_F32), k, 1);
    ggml_tensor * head = ggml_concat(ctx, ggml_cast(ctx, id, GGML_TYPE_F32), pick, 0);
    ggml_tensor * row  = ggml_concat(ctx, head, ggml_concat(ctx, idsf, ls, 0), 0);          // F32 [2 + 2k, 1]

    if (id_out) {
        *id_out = ggml_reshape_1d(ctx, id, 1);
    }
    return row;
}

// Host re-derivation of one step from its output row, in float64, with the draft sampler chain's semantics
// (llama-sampler.cpp): top_p keeps through the first candidate whose running mass reaches top_p (skipped at
// top_p >= 1), min_p keeps logit >= max + log(min_p) (skipped at min_p <= 0), then logit / temp and the dist
// sampler's rule, the first candidate whose running mass reaches u * total. q receives the normalised
// distribution the draw came from, sorted like the row. Returns the drawn position (the token is row[2 + pos]),
// or -1 on a malformed row (non-finite logit, id out of range, order broken).
static inline int32_t llama_mtp_chain_rederive(
        const float * row, int32_t k, float temp, float top_p, float min_p, double u, int32_t n_vocab,
        std::vector<llama_token_data> & q) {
    q.clear();
    const float * ids = row + 2;
    const float * lg  = row + 2 + k;

    for (int32_t i = 0; i < k; ++i) {
        if (!std::isfinite(lg[i]) || ids[i] < 0.0f || ids[i] >= (float) n_vocab || (i > 0 && lg[i] > lg[i - 1])) {
            return -1;
        }
    }

    int32_t n = k;
    const double l0 = lg[0];

    if (top_p < 1.0f) {
        double sum = 0.0;
        for (int32_t i = 0; i < n; ++i) {
            sum += std::exp((double) lg[i] - l0);
        }
        double cum = 0.0;
        for (int32_t i = 0; i < n; ++i) {
            cum += std::exp((double) lg[i] - l0) / sum;
            if (cum >= (double) top_p) {
                n = i + 1;
                break;
            }
        }
    }

    if (min_p > 0.0f) {
        const double thr = l0 + std::log((double) min_p);
        int32_t i = 1; // the first candidate always matches
        while (i < n && (double) lg[i] >= thr) {
            ++i;
        }
        n = i;
    }

    q.resize(n);
    const double m = l0 / (double) temp;
    auto weight = [&](int32_t i) { return std::exp((double) lg[i] / (double) temp - m); };

    double tot = 0.0;
    for (int32_t i = 0; i < n; ++i) {
        tot += weight(i);
    }

    const double tgt = u * tot;
    double run = 0.0;
    int32_t pick = n - 1;
    for (int32_t i = 0; i < n; ++i) {
        run += weight(i);
        if (run >= tgt) {
            pick = i;
            break;
        }
    }

    for (int32_t i = 0; i < n; ++i) {
        q[i].id    = (llama_token) ids[i];
        q[i].logit = (float) ((double) lg[i] / (double) temp);
        q[i].p     = (float) (weight(i) / tot);
    }

    return pick;
}
