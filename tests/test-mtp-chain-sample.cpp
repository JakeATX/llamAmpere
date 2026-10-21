// [#69] The sampled MTP chain's in-graph draw (llama_mtp_chain_sample_graph) against its host re-derivation
// (llama_mtp_chain_rederive) and against llama's own sampler chain.
//
// Each case builds the one-step sampling graph on every available backend (CPU, and the GPU when present) over
// random logits, with and without a draft vocabulary map, and checks per trial:
//   - the row's candidates are the exact top-k of the logits (ids, and positions through the map), sorted by
//     logit, descending, with the logits copied unchanged;
//   - the GPU draw equals the host re-derivation's draw from the same row and uniform (a disagreement only cuts
//     a chain in production, so a small rate is tolerated; k = 1 must always agree);
//   - the host q equals the distribution llama's samplers produce: top_k -> top_p -> min_p -> temperature on the
//     full logits, then the softmax of the kept logits (same ids, probabilities within 1e-5).
// No model is needed.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "llama.h"

#include "../src/llama-mtp-chain-sample.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

struct samp_def {
    float temp;
    float top_p;
    float min_p;
};

static const samp_def k_samps[] = {
    { 1.0f, 0.95f, 0.00f }, // the ship corpus request
    { 0.7f, 1.00f, 0.05f },
    { 1.3f, 0.80f, 0.10f },
    { 1.0f, 1.00f, 0.00f },
};

static constexpr int32_t n_vocab_map = 300000; // id range of the map cases

struct totals {
    int64_t trials   = 0;
    int64_t agree    = 0;
    int64_t failures = 0;
};

// Preserve the actual CPU GGML_TOP_K order, including ties and its intentional first-two swap.
// Compare the raw payload and core sampler/RNG against ordinary backend candidate ingestion.
static bool run_qwen_cpu_reference(ggml_backend_t backend, bool use_map) {
    constexpr int32_t n_vocab = 32;
    constexpr int32_t k = 10;
    constexpr int32_t cap = 7;
    constexpr uint32_t seed = 1729u ^ 0x27d4eb2fu;
    ggml_init_params pg = { ggml_tensor_overhead() * 128 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(pg);
    ggml_tensor * logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_vocab, 1);
    ggml_tensor * samp = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, LLAMA_MTP_CHAIN_SAMP_N);
    ggml_tensor * u = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_tensor * id_map = use_map ? ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_vocab) : nullptr;
    for (ggml_tensor * tensor : { logits, samp, u, id_map }) {
        if (tensor) {
            ggml_set_input(tensor);
        }
    }
    ggml_tensor * id_out = nullptr;
    ggml_tensor * packed = llama_mtp_chain_sample_graph(ctx, logits, k, samp, u, id_map, &id_out, true);
    ggml_tensor * raw_topk = ggml_top_k(ctx, logits, k);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    for (ggml_tensor * tensor : { packed, id_out, raw_topk }) {
        ggml_set_output(tensor);
        ggml_build_forward_expand(gf, tensor);
    }
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        GGML_ASSERT(ggml_backend_supports_op(backend, ggml_graph_node(gf, i)));
    }
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    GGML_ASSERT(ggml_gallocr_alloc_graph(galloc, gf));
    std::vector<int32_t> map(n_vocab), topk(k);
    std::vector<float> vl(n_vocab);
    for (int i = 0; i < n_vocab; ++i) {
        map[i] = use_map ? (13*i + 7) % n_vocab : i;
    }
    bool ok = true;
    for (const auto & sd : k_samps) {
        for (const int32_t request_k : { 1, 5, 10, 40, 0 }) {
            for (bool suppress : { false, true }) {
                const auto make_sampler = [&] {
                    auto * sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
                    if (suppress) {
                        const llama_logit_bias bias = { map[0], -INFINITY };
                        llama_sampler_chain_add(sampler, llama_sampler_init_logit_bias(n_vocab, 1, &bias));
                    }
                    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(request_k));
                    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(sd.top_p, 0));
                    llama_sampler_chain_add(sampler, llama_sampler_init_min_p(sd.min_p, 0));
                    llama_sampler_chain_add(sampler, llama_sampler_init_temp(sd.temp));
                    llama_sampler_chain_add(sampler, llama_sampler_init_dist(seed));
                    return sampler;
                };
                llama_sampler * compact = make_sampler();
                llama_sampler * reference = make_sampler();
                std::mt19937 mirror(seed), expected_rng(seed);
                std::uniform_real_distribution<double> dist(0.0, 1.0);
                std::vector<double> host_u;
                std::vector<float> device_u;
                std::vector<float> row(LLAMA_MTP_CHAIN_ROW(k), 0.0f);
                std::vector<llama_token_data> candidates(k), q;
                float sp[LLAMA_MTP_CHAIN_SAMP_N];
                llama_mtp_chain_samp_pack(sd.temp, sd.top_p, sd.min_p, sp);
                for (int round = 0; round < 21; ++round) {
                    const auto initial = mirror;
                    llama_mtp_chain_qwen_uniforms(mirror, cap, host_u, device_u);
                    ok = ok && mirror == initial;
                    const int consumed = 1 + round % cap;
                    for (int j = 0; j < consumed; ++j) {
                        const double expected_u = dist(expected_rng);
                        ok = ok && host_u[j] == expected_u && device_u[j] == (float) expected_u;
                        for (int i = 0; i < n_vocab; ++i) {
                            const int rank = (13*i + 7*round + 3*j) % n_vocab;
                            switch ((round + j) % 4) {
                                case 0: vl[i] = 0.0f; break;
                                case 1: vl[i] = 3.0f - (float) (rank / 4); break;
                                case 2: vl[i] = rank < 12 ? 2.0f : -0.125f * rank; break;
                                default: vl[i] = 3.0f - 0.075f * rank; break;
                            }
                        }
                        ggml_backend_tensor_set(logits, vl.data(), 0, ggml_nbytes(logits));
                        ggml_backend_tensor_set(samp, sp, 0, sizeof(sp));
                        ggml_backend_tensor_set(u, &device_u[j], 0, sizeof(float));
                        if (id_map) {
                            ggml_backend_tensor_set(id_map, map.data(), 0, ggml_nbytes(id_map));
                        }
                        GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
                        ggml_backend_tensor_get(packed, row.data(), 0, ggml_nbytes(packed));
                        ggml_backend_tensor_get(raw_topk, topk.data(), 0, ggml_nbytes(raw_topk));
                        int32_t gpu_id = -1;
                        ggml_backend_tensor_get(id_out, &gpu_id, 0, sizeof(gpu_id));
                        bool raw_match = row[0] == (float) gpu_id;
                        for (int i = 0; i < k; ++i) {
                            const int32_t pos = topk[i];
                            GGML_ASSERT(pos >= 0 && pos < n_vocab);
                            raw_match = raw_match && row[2 + i] == (float) map[pos] &&
                                std::memcmp(&row[2 + k + i], &vl[pos], sizeof(float)) == 0;
                            candidates[i] = { map[pos], vl[pos], 0.0f };
                        }
                        if (!raw_match) {
                            printf("  FAIL Qwen CPU raw top10 order map=%d round=%d step=%d\n", use_map, round, j);
                            ok = false;
                        }
                        if ((round + j) % 4 == 0) {
                            const float saved = row[2 + k];
                            row[2 + k] = NAN;
                            ok = ok && llama_mtp_chain_rederive_qwen(row.data(), k, compact, n_vocab, q) == -1;
                            row[2 + k] = saved;
                        }
                        llama_token_data_array ref = { candidates.data(), candidates.size(), -1, false };
                        llama_sampler_apply(reference, &ref);
                        const llama_token expected_id = ref.data[ref.selected].id;
                        if (!ref.sorted) {
                            std::sort(ref.data, ref.data + ref.size, [](const llama_token_data & a, const llama_token_data & b) {
                                return a.p > b.p;
                            });
                        }
                        const int32_t pick = llama_mtp_chain_rederive_qwen(row.data(), k, compact, n_vocab, q);
                        bool match = pick >= 0 && (llama_token) row[2 + pick] == expected_id && q.size() == ref.size;
                        for (size_t i = 0; match && i < q.size(); ++i) {
                            match = q[i].id == ref.data[i].id &&
                                std::memcmp(&q[i].logit, &ref.data[i].logit, sizeof(float)) == 0 &&
                                std::memcmp(&q[i].p, &ref.data[i].p, sizeof(float)) == 0;
                        }
                        if (!match) {
                            printf("  FAIL Qwen CPU raw reference map=%d suppress=%d k=%d temp=%.2f top_p=%.2f min_p=%.2f round=%d step=%d\n",
                                   use_map, suppress, request_k, sd.temp, sd.top_p, sd.min_p, round, j);
                            ok = false;
                        }
                        // Last row models confidence failure: sampled and consumed, omitted from the draft.
                        if (j + 1 < consumed) {
                            llama_sampler_accept(compact, expected_id);
                            llama_sampler_accept(reference, expected_id);
                        }
                    }
                    llama_mtp_chain_qwen_commit(mirror, consumed);
                    ok = ok && mirror == expected_rng;
                    auto next_rng = expected_rng;
                    llama_mtp_chain_qwen_uniforms(mirror, 1, host_u, device_u);
                    ok = ok && host_u[0] == dist(next_rng);
                }
                llama_sampler_free(reference);
                llama_sampler_free(compact);
            }
        }
    }
    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    printf("Qwen CPU raw top10 sampler/RNG differential map=%d: %s\n", use_map, ok ? "OK" : "FAILED");
    return ok;
}

static bool run_case(ggml_backend_t backend, int64_t n_v, int32_t k, const samp_def & sd, bool use_map, totals & tot, int n_trials) {
    ggml_init_params pg = { ggml_tensor_overhead() * 96 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(pg);

    ggml_tensor * logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_v, 1);
    ggml_tensor * samp   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, LLAMA_MTP_CHAIN_SAMP_N);
    ggml_tensor * u      = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_tensor * id_map = use_map ? ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_v) : nullptr;
    for (ggml_tensor * t : { logits, samp, u, id_map }) {
        if (t) {
            ggml_set_input(t);
        }
    }

    ggml_tensor * id_out = nullptr;
    ggml_tensor * row = llama_mtp_chain_sample_graph(ctx, logits, k, samp, u, id_map, &id_out);
    ggml_set_output(row);
    ggml_set_output(id_out);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, row);
    ggml_build_forward_expand(gf, id_out);

    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        if (!ggml_backend_supports_op(backend, ggml_graph_node(gf, i))) {
            printf("  %s: op %s unsupported, skipping the case\n", ggml_backend_name(backend), ggml_op_desc(ggml_graph_node(gf, i)));
            ggml_free(ctx);
            return true;
        }
    }

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    GGML_ASSERT(ggml_gallocr_alloc_graph(galloc, gf));

    std::mt19937 rng(7 + 131*n_v + 17*k + (use_map ? 5 : 0) + (int) (sd.temp * 1000) + (int) (sd.min_p * 100000));
    std::normal_distribution<float>       nd(0.0f, 3.0f);
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);

    std::vector<int32_t> map(n_v);
    if (use_map) {
        std::vector<int32_t> all(n_vocab_map);
        std::iota(all.begin(), all.end(), 0);
        std::shuffle(all.begin(), all.end(), rng);
        std::copy(all.begin(), all.begin() + n_v, map.begin());
    } else {
        std::iota(map.begin(), map.end(), 0);
    }
    const int32_t n_vocab = use_map ? n_vocab_map : (int32_t) n_v;

    float sp[LLAMA_MTP_CHAIN_SAMP_N];
    llama_mtp_chain_samp_pack(sd.temp, sd.top_p, sd.min_p, sp);

    // llama's samplers, for the reference q (no dist: the draw is compared through the host re-derivation)
    llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(chain, llama_sampler_init_top_k(k));
    llama_sampler_chain_add(chain, llama_sampler_init_top_p(sd.top_p, 0));
    llama_sampler_chain_add(chain, llama_sampler_init_min_p(sd.min_p, 0));
    llama_sampler_chain_add(chain, llama_sampler_init_temp(sd.temp));

    std::vector<float> vl(n_v);
    std::vector<float> vrow(LLAMA_MTP_CHAIN_ROW(k));
    std::vector<int32_t> order(n_v);
    std::vector<llama_token_data> cand(n_v);
    std::vector<llama_token_data> q;
    bool ok = true;

    for (int t = 0; t < n_trials; ++t) {
        // distinct logits (ties at the top-k boundary would make the expected set ambiguous); every 4th trial is
        // peaked, the regime where top_p cuts early
        const float peak = (t % 4 == 0) ? 12.0f : 0.0f;
        for (int64_t i = 0; i < n_v; ++i) {
            vl[i] = nd(rng) + 1e-4f * (float) (i % 997);
        }
        vl[rng() % n_v] += peak;
        const float vu = (float) ((double) (rng() >> 8) * 0x1.0p-24);

        // every input each trial: the allocator may reuse an input's memory after its last use
        ggml_backend_tensor_set(logits, vl.data(), 0, ggml_nbytes(logits));
        ggml_backend_tensor_set(u, &vu, 0, sizeof(float));
        ggml_backend_tensor_set(samp, sp, 0, sizeof(sp));
        if (id_map) {
            ggml_backend_tensor_set(id_map, map.data(), 0, ggml_nbytes(id_map));
        }
        GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(row, vrow.data(), 0, ggml_nbytes(row));
        int32_t id_gpu = -1;
        ggml_backend_tensor_get(id_out, &id_gpu, 0, sizeof(int32_t));

        // exact top-k by position
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + k, order.end(), [&](int32_t a, int32_t b) { return vl[a] > vl[b]; });

        bool row_ok = true;
        for (int32_t i = 0; i < k; ++i) {
            const int32_t id = (int32_t) vrow[2 + i];
            if (id != map[order[i]] || vrow[2 + k + i] != vl[order[i]]) {
                row_ok = false;
            }
        }
        if (!row_ok || (int32_t) vrow[0] != id_gpu || vrow[1] < 0.0f || vrow[1] >= (float) k || id_gpu != (int32_t) vrow[2 + (int32_t) vrow[1]]) {
            printf("  FAIL %s n_v=%lld k=%d map=%d trial %d: candidate row mismatch\n", ggml_backend_name(backend), (long long) n_v, k, use_map, t);
            ok = false;
            continue;
        }

        const int32_t pick = llama_mtp_chain_rederive(vrow.data(), k, sd.temp, sd.top_p, sd.min_p, (double) vu, n_vocab, q);
        tot.trials++;
        if (pick < 0) {
            printf("  FAIL %s n_v=%lld k=%d map=%d trial %d: re-derivation rejected the row\n", ggml_backend_name(backend), (long long) n_v, k, use_map, t);
            ok = false;
            continue;
        }
        if (pick == (int32_t) vrow[1]) {
            tot.agree++;
        } else if (k == 1) {
            printf("  FAIL %s n_v=%lld k=1: draws differ\n", ggml_backend_name(backend), (long long) n_v);
            ok = false;
        }

        // reference q from llama's samplers over the full logits
        for (int64_t i = 0; i < n_v; ++i) {
            cand[i] = { map[i], vl[i], 0.0f };
        }
        llama_token_data_array arr = { cand.data(), (size_t) n_v, -1, false };
        llama_sampler_apply(chain, &arr);
        std::vector<llama_token_data> ref(arr.data, arr.data + arr.size);
        std::sort(ref.begin(), ref.end(), [](const llama_token_data & a, const llama_token_data & b) { return a.logit > b.logit; });
        double m = ref.empty() ? 0.0 : ref[0].logit, s = 0.0;
        for (const auto & c : ref) {
            s += std::exp((double) c.logit - m);
        }
        bool q_ok = ref.size() == q.size();
        for (size_t i = 0; q_ok && i < ref.size(); ++i) {
            const double p = std::exp((double) ref[i].logit - m) / s;
            q_ok = ref[i].id == q[i].id && std::fabs(p - (double) q[i].p) < 1e-5;
        }
        if (!q_ok) {
            printf("  FAIL %s n_v=%lld k=%d map=%d temp=%.2f top_p=%.2f min_p=%.2f trial %d: q differs from llama's samplers (%zu vs %zu kept)\n",
                   ggml_backend_name(backend), (long long) n_v, k, use_map, sd.temp, sd.top_p, sd.min_p, t, q.size(), ref.size());
            ok = false;
        }
    }

    llama_sampler_free(chain);
    ggml_gallocr_free(galloc);
    ggml_free(ctx);

    if (!ok) {
        tot.failures++;
    }
    return ok;
}

int main(int argc, char ** argv) {
    const bool quick = argc == 2 && strcmp(argv[1], "--quick") == 0;
    GGML_ASSERT(argc == 1 || quick);
    ggml_backend_load_all();

    std::vector<ggml_backend_t> backends;
    backends.push_back(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    if (ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)) {
        backends.push_back(ggml_backend_dev_init(dev, nullptr));
    }

    GGML_ASSERT(backends.front());
    bool ok = run_qwen_cpu_reference(backends.front(), false);
    ok = run_qwen_cpu_reference(backends.front(), true) && ok;
    for (ggml_backend_t backend : backends) {
        GGML_ASSERT(backend);
        totals tot;
        const std::vector<int64_t> vocab_sizes = quick ? std::vector<int64_t>{ 1000 } : std::vector<int64_t>{ 50, 1000, 32768, 248320 };
        for (int64_t n_v : vocab_sizes) {
            for (int32_t k : { 1, 5, 10, 20, 64 }) {
                if (k > n_v) {
                    continue;
                }
                for (const auto & sd : k_samps) {
                    for (bool use_map : { false, true }) {
                        if (use_map && n_v > 65536) {
                            continue; // maps are shortlists
                        }
                        ok = run_case(backend, n_v, k, sd, use_map, tot, quick ? 8 : 48) && ok;
                    }
                }
            }
        }
        const double rate = tot.trials ? (double) tot.agree / (double) tot.trials : 0.0;
        printf("%s: %lld trials, draw agreement %.4f, %lld failing cases\n", ggml_backend_name(backend),
               (long long) tot.trials, rate, (long long) tot.failures);
        if (tot.trials > 0 && rate < 0.995) {
            printf("  FAIL %s: draw agreement below 0.995\n", ggml_backend_name(backend));
            ok = false;
        }
        ggml_backend_free(backend);
    }

    printf("%s\n", ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
