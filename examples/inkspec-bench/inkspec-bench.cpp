// INKSPEC verify cost model: time one llama_decode of W rows (all rows output logits, like a speculative verify)
// for W in ISB_WIDTHS, with the rows' routing either real (consecutive text tokens) or shared (one token repeated
// W times, so the rows pick almost the same experts). The row count per expert visit is recorded with ISB_COUNT=1
// (eval callback on ffn_moe_topk; that run's times are not valid). The KV cache only grows (no rollback), so the
// recurrent state stays consistent; fill moves from the prompt length up by sum(W) per step.
//
// Env: ISB_WIDTHS (default 1,2,3,4,5,6,8), ISB_REPS (default 40), ISB_WARM (default 4), ISB_MODES (default real,dup)
//      ISB_ALT=1   after every timed decode, decode 1 row in a second context of the same model (a foreign graph
//                  launch between two target launches, as the drafter does in a server round)
//      ISB_COUNT=1 count distinct experts per visit per decode (prints the mean per W and mode)
//      ISB_NPROMPT prompt tokens to prefill (default 1024; the rest of the prompt file feeds the rows)

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <sstream>
#include <string>
#include <vector>

struct count_state {
    double visits = 0;      // visit-tensors seen in this decode
    double distinct = 0;    // sum of distinct experts over those visits
    double picks = 0;       // sum of n_used*n_tok
};

static bool count_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    if (strncmp(t->name, "ffn_moe_topk-", 13) != 0) {
        return ask ? false : true;
    }
    for (const char * c = t->name + 13; *c; ++c) {
        if (*c < '0' || *c > '9') {
            return ask ? false : true;
        }
    }
    if (ask) {
        return true;
    }
    auto & st = *(count_state *) user_data;
    GGML_ASSERT(t->type == GGML_TYPE_I32);
    std::vector<uint8_t> raw(ggml_nbytes(t));
    ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
    std::set<int32_t> s;
    for (int64_t i1 = 0; i1 < t->ne[1]; ++i1)
    for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
        int32_t v;
        memcpy(&v, raw.data() + i0*t->nb[0] + i1*t->nb[1], sizeof(v));
        s.insert(v);
    }
    st.visits   += 1;
    st.distinct += s.size();
    st.picks    += (double) t->ne[0]*t->ne[1];
    return true;
}

static std::vector<std::string> split(const char * s, const char * def) {
    std::vector<std::string> out;
    std::stringstream ss(s ? s : def);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) {
            out.push_back(tok);
        }
    }
    return out;
}

static int env_int(const char * n, int def) {
    const char * s = getenv(n);
    return s ? atoi(s) : def;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    std::vector<int> widths;
    for (auto & w : split(getenv("ISB_WIDTHS"), "1,2,3,4,5,6,8")) {
        widths.push_back(atoi(w.c_str()));
    }
    const std::vector<std::string> modes = split(getenv("ISB_MODES"), "real,dup");
    const int reps    = env_int("ISB_REPS", 40);
    const int warm    = env_int("ISB_WARM", 4);
    const bool alt    = env_int("ISB_ALT", 0) != 0;
    const bool count  = env_int("ISB_COUNT", 0) != 0;
    const int nprompt = env_int("ISB_NPROMPT", 1024);
    GGML_ASSERT(reps >= 1 && warm >= 0);
    for (int w : widths) {
        GGML_ASSERT(w >= 1 && w <= 16);
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    count_state cst;
    if (count) {
        params.cb_eval = count_cb;
        params.cb_eval_user_data = &cst;
    }
    params.warmup = false;

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init\n", __func__);
        return 1;
    }
    llama_context * ctx2 = nullptr;
    if (alt) {
        auto cp = common_context_params_to_llama(params);
        cp.n_ctx = 4096;
        cp.cb_eval = nullptr;
        ctx2 = llama_init_from_model(model, cp);
        GGML_ASSERT(ctx2);
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, llama_vocab_get_add_bos(vocab), true);
    size_t need = nprompt;
    for (int w : widths) {
        need += (size_t) (reps + warm)*w*modes.size();
    }
    if (tokens.size() < need) {
        LOG_ERR("prompt has %zu tokens, need %zu\n", tokens.size(), need);
        return 1;
    }

    const size_t nb = llama_n_batch(ctx);
    for (int i = 0; i < nprompt; i += nb) {
        const int n = std::min<int>(nb, nprompt - i);
        if (llama_decode(ctx, llama_batch_get_one(tokens.data() + i, n))) {
            LOG_ERR("prefill failed\n");
            return 1;
        }
    }
    llama_synchronize(ctx);
    int pos = nprompt;      // next position in the target context
    size_t cur = nprompt;   // next unused text token
    int pos2 = 0;
    llama_token t2 = tokens[0];
    if (ctx2) {
        GGML_ASSERT(!llama_decode(ctx2, llama_batch_get_one(tokens.data(), 64)));
        llama_synchronize(ctx2);
        pos2 = 64;
    }

    llama_batch batch = llama_batch_init(16, 0, 1);
    printf("ISB mode W reps mean_ms median_ms p10_ms p90_ms distinct_per_visit picks_per_visit fill_start\n");
    for (const auto & mode : modes) {
        const bool dup = mode == "dup";
        for (int w : widths) {
            std::vector<double> ms;
            double dist = 0, picks = 0, visits = 0;
            const int fill0 = pos;
            for (int r = 0; r < warm + reps; ++r) {
                common_batch_clear(batch);
                for (int j = 0; j < w; ++j) {
                    common_batch_add(batch, dup ? tokens[cur] : tokens[cur + j], pos + j, { 0 }, true);
                }
                cst = count_state();
                const int64_t t0 = ggml_time_us();
                if (llama_decode(ctx, batch)) {
                    LOG_ERR("decode failed W=%d\n", w);
                    return 1;
                }
                llama_synchronize(ctx);
                const int64_t t1 = ggml_time_us();
                pos += w;
                cur += w;
                if (ctx2) {
                    llama_batch b2 = llama_batch_get_one(&t2, 1);
                    GGML_ASSERT(!llama_decode(ctx2, b2));
                    llama_synchronize(ctx2);
                    pos2++;
                    if (pos2 > 4000) {
                        llama_memory_clear(llama_get_memory(ctx2), true);
                        GGML_ASSERT(!llama_decode(ctx2, llama_batch_get_one(tokens.data(), 64)));
                        pos2 = 64;
                    }
                }
                if (r >= warm) {
                    ms.push_back((t1 - t0)/1e3);
                    dist += cst.distinct; picks += cst.picks; visits += cst.visits;
                }
            }
            std::vector<double> s = ms;
            std::sort(s.begin(), s.end());
            double mean = 0;
            for (double v : ms) {
                mean += v;
            }
            mean /= ms.size();
            printf("ISB %s %d %d %.3f %.3f %.3f %.3f %.2f %.2f %d\n", mode.c_str(), w, reps, mean, s[s.size()/2],
                   s[s.size()/10], s[s.size()*9/10], visits ? dist/visits : 0.0, visits ? picks/visits : 0.0, fill0);
            fflush(stdout);
        }
    }
    llama_batch_free(batch);
    if (ctx2) {
        llama_free(ctx2);
    }
    llama_backend_free();
    return 0;
}
