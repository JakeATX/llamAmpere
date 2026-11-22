// INKLSA quality probe: teacher-forced decode KLD of a long text against a dense reference run.
//
// The first INKKLD_PREFIX tokens of the prompt file are prefilled (or restored from INKKLD_STATE if it exists, else
// prefilled and saved there), then the next INKKLD_STEPS tokens are fed INKKLD_T at a time with logits for every row
// (T = 1: plain decode, T = 4: DSpark-verify-sized batches). INKKLD_MODE=base writes the logits to INKKLD_BASE (F32);
// INKKLD_MODE=test reads them and prints per-step KLD(base || test), top-1 agreement and the mean.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static int64_t envi(const char * n, int64_t d) { const char * s = getenv(n); return s && *s ? atoll(s) : d; }
static std::string envs(const char * n) { const char * s = getenv(n); return s ? s : ""; }

static void log_softmax(const float * x, int n, std::vector<double> & out) {
    out.resize(n);
    double mx = -INFINITY;
    for (int i = 0; i < n; ++i) mx = std::max(mx, (double) x[i]);
    double z = 0;
    for (int i = 0; i < n; ++i) z += std::exp(x[i] - mx);
    const double lz = mx + std::log(z);
    for (int i = 0; i < n; ++i) out[i] = x[i] - lz;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    const int64_t N0 = envi("INKKLD_PREFIX", 32768), M = envi("INKKLD_STEPS", 512), T = envi("INKKLD_T", 1);
    const std::string mode = envs("INKKLD_MODE"), base_path = envs("INKKLD_BASE"), state = envs("INKKLD_STATE");
    if ((mode != "base" && mode != "test") || base_path.empty()) {
        LOG_ERR("set INKKLD_MODE=base|test and INKKLD_BASE\n");
        return 1;
    }
    llama_backend_init();
    params.warmup = false;
    auto init = common_init_from_params(params);
    auto * model = init->model();
    auto * ctx = init->context();
    if (!model || !ctx) return 1;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<llama_token> toks = common_tokenize(ctx, params.prompt, llama_vocab_get_add_bos(vocab), true);
    if ((int64_t) toks.size() < N0 + M + 1) {
        LOG_ERR("prompt has %zu tokens, need %lld\n", toks.size(), (long long) (N0 + M + 1));
        return 1;
    }
    const int64_t t0 = ggml_time_us();
    bool loaded = false;
    if (!state.empty()) {
        FILE * f = fopen(state.c_str(), "rb");
        if (f) {
            fclose(f);
            std::vector<llama_token> st(N0);
            size_t n_st = 0;
            if (llama_state_seq_load_file(ctx, state.c_str(), 0, st.data(), st.size(), &n_st) && n_st == (size_t) N0) {
                loaded = true;
                LOG_INF("inklsa-kld: restored %zu tokens from %s in %.1f s\n", n_st, state.c_str(), (ggml_time_us() - t0)/1e6);
            } else {
                LOG_ERR("inklsa-kld: state restore failed\n");
                return 1;
            }
        }
    }
    if (!loaded) {
        const int64_t nb = llama_n_batch(ctx);
        for (int64_t i = 0; i < N0; i += nb) {
            const int64_t n = std::min(nb, N0 - i);
            if (llama_decode(ctx, llama_batch_get_one(toks.data() + i, n))) {
                LOG_ERR("prefill failed at %lld\n", (long long) i);
                return 1;
            }
        }
        LOG_INF("inklsa-kld: prefilled %lld tokens in %.1f s\n", (long long) N0, (ggml_time_us() - t0)/1e6);
        if (!state.empty()) {
            llama_state_seq_save_file(ctx, state.c_str(), 0, toks.data(), N0);
        }
    }
    FILE * fb = fopen(base_path.c_str(), mode == "base" ? "wb" : "rb");
    if (!fb) { LOG_ERR("cannot open %s\n", base_path.c_str()); return 1; }
    llama_batch batch = llama_batch_init(T, 0, 1);
    std::vector<float> ref(n_vocab);
    std::vector<double> lp, lq;
    double sum_kl = 0, max_kl = 0; int64_t n = 0, agree = 0;
    const int64_t t1 = ggml_time_us();
    for (int64_t s = 0; s < M; s += T) {
        common_batch_clear(batch);
        for (int64_t j = 0; j < T; ++j) {
            common_batch_add(batch, toks[N0 + s + j], N0 + s + j, {0}, true);
        }
        if (llama_decode(ctx, batch)) { LOG_ERR("decode failed at step %lld\n", (long long) s); return 1; }
        for (int64_t j = 0; j < T; ++j) {
            const float * lg = llama_get_logits_ith(ctx, (int32_t) j);
            if (mode == "base") {
                fwrite(lg, sizeof(float), n_vocab, fb);
                continue;
            }
            if (fread(ref.data(), sizeof(float), n_vocab, fb) != (size_t) n_vocab) { LOG_ERR("short base file\n"); return 1; }
            log_softmax(ref.data(), n_vocab, lp);
            log_softmax(lg, n_vocab, lq);
            double kl = 0; int am_p = 0, am_q = 0;
            for (int v = 0; v < n_vocab; ++v) {
                kl += std::exp(lp[v]) * (lp[v] - lq[v]);
                if (lp[v] > lp[am_p]) am_p = v;
                if (lq[v] > lq[am_q]) am_q = v;
            }
            sum_kl += kl; max_kl = std::max(max_kl, kl); agree += am_p == am_q; ++n;
            printf("STEP %lld kl %.6f top1 %d\n", (long long) (s + j), kl, am_p == am_q);
        }
    }
    const double dt = (ggml_time_us() - t1)/1e6;
    if (mode == "test") {
        printf("RESULT prefix %lld steps %lld T %lld kld_mean %.6f kld_max %.6f top1 %.4f time %.2f s\n", (long long) N0,
                (long long) n, (long long) T, sum_kl/n, max_kl, (double) agree/n, dt);
    } else {
        printf("RESULT base written prefix %lld steps %lld T %lld time %.2f s\n", (long long) N0, (long long) M, (long long) T, dt);
    }
    fclose(fb);
    llama_batch_free(batch);
    llama_perf_context_print(ctx);
    llama_backend_free();
    return 0;
}
