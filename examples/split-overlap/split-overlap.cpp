// [SPLITOVL] llama-split-overlap: speculative verify loop with overlapped windows under -sm layer (2 cards).
//
// Arm "off": the usual loop. Draft n tokens, verify [last, d1..dn] in one decode, read it, roll back the rejected rows.
// Arm "on" (needs LLAMA_SPLIT_OVERLAP=1, -sm layer, 2+ cards): while window K is verified, window K+1 is already
// queued. Its tokens are the drafter's continuation under the assumption that K is fully accepted and that K's bonus
// token equals K+1's first token. Card A runs K+1 while card B finishes K. If K is fully accepted and the bonus
// matches, K+1 is kept and K+2 is queued; otherwise both are dropped (llama_synchronize + llama_memory_seq_rm back to
// the last accepted token; the recurrent rollback ring is sized for two windows) and the loop restarts with a pair.
//
// The sampler is the same in both arms and draws its random number from the token position (counter based), so for
// a given model and prompt both arms must produce the same tokens: greedy (SO_TEMP=0) and sampled.
//
// Env:
//   SO_MODE   off | on (default on; "on" falls back to off with a warning when the context refuses)
//   SO_N      drafts per window (default 3)
//   SO_DRAFT  ngram | ref (default ngram). ref: drafts are read from SO_REF (whitespace-separated token ids of an
//             earlier run's output) with a SO_MISS fraction of them corrupted (position hash), for a controlled
//             acceptance rate
//   SO_TEMP   0 = greedy (default), else temperature; SO_TOPK (default 40); SO_SEED (default 1)
//   SO_DUMP   write the generated token ids to this file; SO_CMP compare them with this file (exit 3 on a mismatch)
// CLI: the usual common args (-m, -f/-p, -n, -c, -ngl, -sm, -ts, -fa, ...).

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "../../src/llama-ext.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <string>
#include <vector>

static int env_int(const char * k, int d) { const char * e = getenv(k); return e && *e ? atoi(e) : d; }
static double env_f(const char * k, double d) { const char * e = getenv(k); return e && *e ? atof(e) : d; }
static std::string env_s(const char * k, const char * d) { const char * e = getenv(k); return e && *e ? e : d; }

static uint64_t mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

struct sampler {
    int    n_vocab = 0;
    double temp    = 0.0;
    int    top_k   = 40;
    uint64_t seed  = 1;
    std::vector<std::pair<float, int>> cand;

    // the token at absolute position pos, from the logits row that predicts it
    llama_token sample(const float * lg, int64_t pos) {
        if (temp <= 0.0) {
            int best = 0;
            for (int i = 1; i < n_vocab; ++i) {
                if (lg[i] > lg[best]) {
                    best = i;
                }
            }
            return best;
        }
        const int k = std::min(std::max(top_k, 1), n_vocab);
        cand.resize(n_vocab);
        for (int i = 0; i < n_vocab; ++i) {
            cand[i] = { lg[i], i };
        }
        std::partial_sort(cand.begin(), cand.begin() + k, cand.end(),
                [](const auto & a, const auto & b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
        const double mx = cand[0].first;
        double sum = 0.0;
        std::vector<double> p(k);
        for (int i = 0; i < k; ++i) {
            p[i] = std::exp((cand[i].first - mx) / temp);
            sum += p[i];
        }
        const double u = (double) (mix64(seed * 0x100000001b3ull ^ (uint64_t) pos) >> 11) * (1.0 / 9007199254740992.0) * sum;
        double acc = 0.0;
        for (int i = 0; i < k; ++i) {
            acc += p[i];
            if (u < acc) {
                return cand[i].second;
            }
        }
        return cand[k - 1].second;
    }
};

struct drafter {
    std::string kind = "ngram";
    std::vector<llama_token> ref;
    size_t n_prompt = 0;
    double miss = 0.0;
    int n_vocab = 0;

    // up to k tokens that follow hist (hist ends with the last token, already sampled)
    std::vector<llama_token> draft(const std::vector<llama_token> & hist, int k) const {
        std::vector<llama_token> out;
        if (k <= 0) {
            return out;
        }
        if (kind == "ref") {
            for (int j = 0; j < k; ++j) {
                const size_t g = hist.size() + j - n_prompt; // generated index of the token drafted
                if (hist.size() + j < n_prompt || g >= ref.size()) {
                    break;
                }
                llama_token t = ref[g];
                const double u = (double) (mix64(0x5eedull ^ (uint64_t) (hist.size() + j)) >> 11) * (1.0 / 9007199254740992.0);
                if (u < miss) {
                    t = (t + 1) % n_vocab;
                }
                out.push_back(t);
            }
            return out;
        }
        // n-gram lookup: newest earlier occurrence of the last 3/2/1 tokens, then copy what followed it
        const int n = (int) hist.size();
        for (int m = std::min(3, n - 1); m >= 1 && out.empty(); --m) {
            for (int s = n - m - 1; s >= 0; --s) {
                if (std::equal(hist.end() - m, hist.end(), hist.begin() + s)) {
                    for (int j = s + m; j < n && (int) out.size() < k; ++j) {
                        out.push_back(hist[j]);
                    }
                    break;
                }
            }
        }
        return out;
    }
};

struct window {
    std::vector<llama_token> tok;
    llama_pos p0 = 0;
    int slot = 0;
};

static void batch_fill(llama_batch & b, const std::vector<llama_token> & tok, llama_pos p0, bool all_out) {
    b.n_tokens = 0;
    for (size_t i = 0; i < tok.size(); ++i) {
        b.token[b.n_tokens]     = tok[i];
        b.pos[b.n_tokens]       = p0 + (llama_pos) i;
        b.n_seq_id[b.n_tokens]  = 1;
        b.seq_id[b.n_tokens][0] = 0;
        b.logits[b.n_tokens]    = all_out || i + 1 == tok.size();
        b.n_tokens++;
    }
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    const std::string mode = env_s("SO_MODE", "on");
    const int n_draft = std::max(1, env_int("SO_N", 3));
    const int n_win   = n_draft + 1;

    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model_params mparams = common_model_params_to_llama(params);
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (!model) {
        LOG_ERR("failed to load %s\n", params.model.path.c_str());
        return 1;
    }
    llama_context_params cparams = common_context_params_to_llama(params);
    // the rollback ring holds both in-flight windows (2 * n_win rows, a miss rolls back at most 2 * n_win - 1);
    // both arms use the same size so the default arm is the like-for-like baseline
    cparams.n_rs_seq = (uint32_t) (2 * n_win);
    cparams.n_batch  = std::max<uint32_t>(cparams.n_batch, cparams.n_rs_seq + 2);
    cparams.n_ubatch = std::max<uint32_t>(cparams.n_ubatch, (uint32_t) n_win);
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        LOG_ERR("failed to create the context\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    bool overlap = mode == "on";
    if (overlap && !llama_split_overlap_init(ctx)) {
        LOG_WRN("split overlap refused (set LLAMA_SPLIT_OVERLAP=1, -sm layer, 2 cards); running the off arm\n");
        overlap = false;
    }

    sampler smp;
    smp.n_vocab = n_vocab;
    smp.temp    = env_f("SO_TEMP", 0.0);
    smp.top_k   = env_int("SO_TOPK", 40);
    smp.seed    = (uint64_t) env_int("SO_SEED", 1);

    std::vector<llama_token> prompt = common_tokenize(ctx, params.prompt, true, true);
    if (prompt.size() < 2 || prompt.size() + params.n_predict + 4 * n_win > llama_n_ctx(ctx)) {
        LOG_ERR("prompt of %zu tokens does not fit ctx %u with -n %d\n", prompt.size(), llama_n_ctx(ctx), params.n_predict);
        return 1;
    }

    drafter dr;
    dr.kind = env_s("SO_DRAFT", "ngram");
    dr.n_prompt = prompt.size();
    dr.miss = env_f("SO_MISS", 0.0);
    dr.n_vocab = n_vocab;
    if (dr.kind == "ref") {
        std::ifstream f(env_s("SO_REF", ""));
        long v;
        while (f >> v) {
            dr.ref.push_back((llama_token) v);
        }
        if (dr.ref.empty()) {
            LOG_ERR("SO_DRAFT=ref needs SO_REF\n");
            return 1;
        }
    }

    const uint32_t n_bmax = std::max<uint32_t>(llama_n_batch(ctx), (uint32_t) (2 * n_win));
    llama_batch batch = llama_batch_init((int32_t) n_bmax, 0, 1);

    // prefill all but the last prompt token
    for (size_t i = 0; i + 1 < prompt.size(); i += n_bmax) {
        const size_t e = std::min(prompt.size() - 1, i + n_bmax);
        std::vector<llama_token> chunk(prompt.begin() + i, prompt.begin() + e);
        batch_fill(batch, chunk, (llama_pos) i, false);
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("prefill failed\n");
            return 1;
        }
    }
    llama_synchronize(ctx);

    llama_memory_t mem = llama_get_memory(ctx);
    std::vector<llama_token> hist = prompt; // committed tokens; hist.back() is not in the cache yet
    const size_t n_target = prompt.size() + (size_t) std::max(params.n_predict, 1);
    bool eog = false;

    uint64_t n_windows = 0, n_rows = 0, n_kept = 0, n_dropped = 0, n_rollback = 0;
    const int64_t t0 = ggml_time_us();

    auto done = [&]() { return eog || hist.size() >= n_target; };
    auto commit = [&](llama_token t) {
        hist.push_back(t);
        if (llama_vocab_is_eog(vocab, t)) {
            eog = true;
        }
    };

    if (!overlap) {
        while (!done()) {
            std::vector<llama_token> d = dr.draft(hist, n_draft);
            std::vector<llama_token> w = { hist.back() };
            w.insert(w.end(), d.begin(), d.end());
            const llama_pos p0 = (llama_pos) hist.size() - 1;
            batch_fill(batch, w, p0, true);
            if (llama_decode(ctx, batch) != 0) {
                LOG_ERR("decode failed\n");
                return 1;
            }
            n_windows++;
            n_rows += w.size();
            for (size_t i = 0; i < w.size() && !done(); ++i) {
                const llama_token t = smp.sample(llama_get_logits_ith(ctx, (int32_t) i), p0 + (llama_pos) i + 1);
                commit(t);
                if (i + 1 >= w.size() || t != w[i + 1]) {
                    break;
                }
            }
            // cache must hold positions [0, hist.size() - 1)
            if (!llama_memory_seq_rm(mem, 0, (llama_pos) hist.size() - 1, -1)) {
                LOG_ERR("seq_rm failed at %zu\n", hist.size() - 1);
                return 1;
            }
        }
    } else {
        std::deque<window> fl;   // in flight, oldest first
        int next_slot = 0;
        auto launch = [&](std::vector<llama_token> tok, llama_pos p0) -> bool {
            window w;
            w.tok  = std::move(tok);
            w.p0   = p0;
            w.slot = next_slot;
            next_slot ^= 1;
            batch_fill(batch, w.tok, p0, true);
            if (llama_split_overlap_decode(ctx, batch, w.slot) != 0) {
                return false;
            }
            n_windows++;
            n_rows += w.tok.size();
            fl.push_back(std::move(w));
            return true;
        };
        while (!done()) {
            if (fl.empty()) {
                // a pair: [last, c0..c_{n-1}] and the assumed continuation [c_n .. c_{2n}]
                std::vector<llama_token> c = dr.draft(hist, 2 * n_draft + 1);
                std::vector<llama_token> w0 = { hist.back() };
                const size_t n0 = std::min<size_t>(c.size(), n_draft);
                w0.insert(w0.end(), c.begin(), c.begin() + n0);
                const llama_pos p0 = (llama_pos) hist.size() - 1;
                if (!launch(w0, p0)) {
                    LOG_ERR("decode failed\n");
                    return 1;
                }
                if (c.size() > n0 && hist.size() + w0.size() < n_target) {
                    std::vector<llama_token> w1(c.begin() + n0, c.end());
                    if (!launch(w1, p0 + (llama_pos) w0.size())) {
                        LOG_ERR("decode failed\n");
                        return 1;
                    }
                }
            }
            window w = fl.front();
            fl.pop_front();
            const window * nx = fl.empty() ? nullptr : &fl.front();
            bool all = true;
            for (size_t i = 0; i < w.tok.size(); ++i) {
                if (done()) {
                    all = false;
                    break;
                }
                const float * lg = llama_split_overlap_logits_ith(ctx, w.slot, (int32_t) i);
                if (!lg) {
                    LOG_ERR("no logits for slot %d row %zu\n", w.slot, i);
                    return 1;
                }
                const llama_token t = smp.sample(lg, w.p0 + (llama_pos) i + 1);
                commit(t);
                const bool last = i + 1 >= w.tok.size();
                const llama_token expect = !last ? w.tok[i + 1] : (nx ? nx->tok[0] : LLAMA_TOKEN_NULL);
                if (t != expect) {
                    all = false;
                    break;
                }
            }
            if (all && nx && !done()) {
                // W fully accepted and its bonus is the next window's first token: keep the next window and queue
                // the one after it, assuming the next window is fully accepted too
                n_kept++;
                std::vector<llama_token> assumed = hist;
                assumed.insert(assumed.end(), nx->tok.begin() + 1, nx->tok.end());
                if (assumed.size() + 1 < n_target) {
                    std::vector<llama_token> c = dr.draft(assumed, n_draft + 1);
                    if (!c.empty()) {
                        if (!launch(c, nx->p0 + (llama_pos) nx->tok.size())) {
                            LOG_ERR("decode failed\n");
                            return 1;
                        }
                    }
                }
                continue;
            }
            // miss (or the pipeline ran dry): drop what is in flight and roll back to the last accepted token
            if (!fl.empty()) {
                n_dropped += fl.size();
                fl.clear();
            }
            llama_synchronize(ctx);
            n_rollback++;
            if (!llama_memory_seq_rm(mem, 0, (llama_pos) hist.size() - 1, -1)) {
                LOG_ERR("seq_rm failed at %zu (rollback ring too small?)\n", hist.size() - 1);
                return 1;
            }
        }
        llama_synchronize(ctx);
    }
    const int64_t t1 = ggml_time_us();

    std::vector<llama_token> gen(hist.begin() + prompt.size(), hist.end());
    if (gen.size() > (size_t) params.n_predict && params.n_predict > 0) {
        gen.resize(params.n_predict);
    }
    const double s = (t1 - t0) / 1e6;
    LOG_INF("\nSPLITOVL mode=%s n=%d draft=%s temp=%.2f prompt=%zu gen=%zu time=%.3f s tok/s=%.2f windows=%llu rows=%llu "
            "tok/window=%.3f kept=%llu dropped=%llu rollbacks=%llu\n",
            overlap ? "on" : "off", n_draft, dr.kind.c_str(), smp.temp, prompt.size(), gen.size(), s, gen.size() / s,
            (unsigned long long) n_windows, (unsigned long long) n_rows, n_windows ? (double) gen.size() / n_windows : 0.0,
            (unsigned long long) n_kept, (unsigned long long) n_dropped, (unsigned long long) n_rollback);

    const std::string dump = env_s("SO_DUMP", "");
    if (!dump.empty()) {
        std::ofstream f(dump);
        for (auto t : gen) {
            f << t << "\n";
        }
    }
    int rc = 0;
    const std::string cmp = env_s("SO_CMP", "");
    if (!cmp.empty()) {
        std::ifstream f(cmp);
        std::vector<llama_token> ref;
        long v;
        while (f >> v) {
            ref.push_back((llama_token) v);
        }
        size_t n = std::min(ref.size(), gen.size()), first = n;
        for (size_t i = 0; i < n; ++i) {
            if (ref[i] != gen[i]) {
                first = i;
                break;
            }
        }
        if (first == n && ref.size() == gen.size()) {
            LOG_INF("SPLITOVL IDENTICAL %zu tokens vs %s\n", n, cmp.c_str());
        } else {
            LOG_INF("SPLITOVL DIFF first mismatch at %zu (ref %zu, gen %zu tokens) vs %s\n", first, ref.size(), gen.size(), cmp.c_str());
            rc = 3;
        }
    }

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return rc;
}
