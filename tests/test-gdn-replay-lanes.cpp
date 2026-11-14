// [TAG_GDN_REPLAY_LANES] per-lane GDN replay on the static verify graph (LLAMA_GDN_REPLAY_LANES=1) against the split
// ubatches of the default (one ubatch per pending replay). The env is read once per process, so the test runs itself
// twice (=0 and =1) and compares the logits of every decoded token:
//   A. 4 seqs, verify rounds of n == n_rs_seq and n == n_rs_seq + 1 tokens, a different rollback per seq each round,
//      one seq restored from a state blob (stale) mid-way. With lanes the rounds must run as joint ubatches.
//   C. only seqs {0, 2, 3} verify while seq 1 is idle (bystander cell between them), with seq 1 at replay 0 or with a
//      pending rollback; then seq 1 decodes again. Unified and per-seq KV.
//   B. (B1) two long prompts with different pending replays and a small n_ubatch: no mixed ubatch narrower than
//      n_rs_seq (llama_memory_recurrent::replay_split::advance asserts it).
// Needs a GDN model, e.g. the generated qwen35-dense.gguf: test-gdn-replay-lanes -m <MODEL_DIR>/qwen35-dense.gguf
#include "arg.h"
#include "common.h"
#include "llama.h"

#include "../src/llama-context.h"
#include "../src/llama-memory-hybrid.h"
#include "../src/llama-memory-recurrent.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

static constexpr uint32_t K      = 4; // n_rs_seq
static constexpr int      N_SEQS = 4;

static llama_memory_recurrent * get_recr(llama_context * ctx) {
    llama_memory_i * mem = ctx->get_memory();
    if (auto * h = dynamic_cast<llama_memory_hybrid *>(mem)) {
        return h->get_mem_recr();
    }
    return dynamic_cast<llama_memory_recurrent *>(mem);
}

static llama_context * make_ctx(const common_params & params, llama_model * model, uint32_t n_ubatch, bool unified = false) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max  = N_SEQS;
    cparams.n_rs_seq   = K;
    cparams.gdn_replay = true;
    cparams.kv_unified = unified;
    cparams.n_ctx      = 2048;
    cparams.n_batch    = 512;
    cparams.n_ubatch   = n_ubatch;
    return llama_init_from_model(model, cparams);
}

// decodes n[s] tokens for every seq s from pos[s] on, all outputs; appends the logits in batch order
static bool decode_round(llama_context * ctx, std::vector<llama_pos> & pos, const std::vector<uint32_t> & n,
        uint32_t & rng, int n_vocab, std::vector<float> & out) {
    uint32_t total = 0;
    for (uint32_t c : n) {
        total += c;
    }
    llama_batch batch = llama_batch_init(total, 0, 1);
    for (int s = 0; s < (int) n.size(); ++s) {
        for (uint32_t i = 0; i < n[s]; ++i) {
            rng = rng * 1664525u + 1013904223u;
            common_batch_add(batch, (llama_token) ((rng >> 8) % (uint32_t) n_vocab), pos[s]++, { s }, true);
        }
    }
    bool ok = llama_decode(ctx, batch) == 0;
    for (int32_t i = 0; ok && i < batch.n_tokens; ++i) {
        const float * lg = llama_get_logits_ith(ctx, i);
        ok = lg != nullptr;
        if (ok) {
            out.insert(out.end(), lg, lg + n_vocab);
        }
    }
    llama_batch_free(batch);
    return ok;
}

static bool rollback(llama_context * ctx, std::vector<llama_pos> & pos, int s, uint32_t r) {
    if (r == 0) {
        return true;
    }
    pos[s] -= (llama_pos) r;
    return llama_memory_seq_rm(llama_get_memory(ctx), s, pos[s], -1);
}

// the child: runs all scenarios and writes [n_joint, n_bystand, n_floats, floats...] to path
static int run_child(const common_params & params, llama_model * model, const char * path) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<float> out;
    uint64_t n_joint   = 0;
    uint64_t n_bystand = 0;
    uint32_t rng = 1234;

    // A: verify rounds
    {
        llama_context_ptr ctx(make_ctx(params, model, 64));
        if (!ctx) {
            return 1;
        }
        std::vector<llama_pos> pos(N_SEQS, 0);
        bool ok = decode_round(ctx.get(), pos, { 12, 15, 18, 21 }, rng, n_vocab, out);
        for (int round = 0; ok && round < 8; ++round) {
            const uint32_t n = round % 2 == 0 ? K + 1 : K; // n > K (fold) and n == K
            ok = decode_round(ctx.get(), pos, { n, n, n, n }, rng, n_vocab, out);
            for (int s = 0; ok && s < N_SEQS; ++s) {
                ok = rollback(ctx.get(), pos, s, (uint32_t) (round * 3 + s) % std::min(n, K));
            }
            if (ok && round == 3) {
                // seq 2 goes through a state blob: restored with s_stale set
                std::vector<uint8_t> blob(llama_state_seq_get_size(ctx.get(), 2));
                const size_t got = llama_state_seq_get_data(ctx.get(), blob.data(), blob.size(), 2);
                ok = got > 0 && llama_memory_seq_rm(llama_get_memory(ctx.get()), 2, -1, -1) &&
                     llama_state_seq_set_data(ctx.get(), blob.data(), got, 2) == got;
            }
        }
        if (!ok) {
            fprintf(stderr, "scenario A failed\n");
            return 1;
        }
        llama_memory_recurrent * recr = get_recr(ctx.get());
        if (recr == nullptr || !recr->gdn_replay) {
            fprintf(stderr, "model has no gdn_replay recurrent memory\n");
            return 2;
        }
        n_joint = recr->lstats.n_joint;
    }

    // C: subset rounds around an idle seq 1 (unified KV splits non-sequentially, so {0, 2, 3} can share a ubatch)
    for (bool unified : { false, true }) {
        for (uint32_t r1 : { 0u, 2u }) {
            llama_context_ptr ctx(make_ctx(params, model, 64, unified));
            if (!ctx) {
                return 1;
            }
            // unified, r1 > 0: the joint {0, 2, 3} pass must not span seq 1 while its replay is pending (bystander cut)
            std::vector<float> & o = out;
            std::vector<llama_pos> pos(N_SEQS, 0);
            bool ok = decode_round(ctx.get(), pos, { 12, 15, 18, 21 }, rng, n_vocab, o);
            ok = ok && decode_round(ctx.get(), pos, { K + 1, K + 1, K + 1, K + 1 }, rng, n_vocab, o);
            for (int s = 0; ok && s < N_SEQS; ++s) {
                ok = rollback(ctx.get(), pos, s, s == 1 ? r1 : (uint32_t) (s + 1) % K);
            }
            for (int round = 0; ok && round < 4; ++round) {
                const uint32_t n = round % 2 == 0 ? K + 1 : K;
                ok = decode_round(ctx.get(), pos, { n, 0, n, n }, rng, n_vocab, o);
                for (int s : { 0, 2, 3 }) {
                    ok = ok && rollback(ctx.get(), pos, s, (uint32_t) (round + s) % std::min(n, K));
                }
            }
            ok = ok && decode_round(ctx.get(), pos, { 0, K + 1, 0, 0 }, rng, n_vocab, o);
            ok = ok && decode_round(ctx.get(), pos, { K + 1, K + 1, K + 1, K + 1 }, rng, n_vocab, o);
            if (!ok) {
                fprintf(stderr, "scenario C (unified %d, seq 1 rollback %u) failed\n", unified, r1);
                return 1;
            }
            llama_memory_recurrent * recr = get_recr(ctx.get());
            n_joint   += recr ? recr->lstats.n_joint   : 0;
            n_bystand += recr ? recr->lstats.n_bystand : 0;
        }
    }

    // B: two long prompts with different pending replays, n_ubatch 6 (width 3 < K) and 10 (width 5)
    for (uint32_t n_ubatch : { 6u, 10u }) {
        llama_context_ptr ctx(make_ctx(params, model, n_ubatch));
        if (!ctx) {
            return 1;
        }
        std::vector<llama_pos> pos(N_SEQS, 0);
        bool ok = decode_round(ctx.get(), pos, { K + 1, K + 1 }, rng, n_vocab, out);
        ok = ok && rollback(ctx.get(), pos, 0, 1) && rollback(ctx.get(), pos, 1, 3);
        ok = ok && decode_round(ctx.get(), pos, { 20, 20 }, rng, n_vocab, out);
        ok = ok && decode_round(ctx.get(), pos, { 7, 9 }, rng, n_vocab, out);
        if (!ok) {
            fprintf(stderr, "scenario B (n_ubatch %u) failed\n", n_ubatch);
            return 1;
        }
    }

    FILE * f = fopen(path, "wb");
    if (!f) {
        return 1;
    }
    const uint64_t n_floats = out.size();
    fwrite(&n_joint, sizeof(n_joint), 1, f);
    fwrite(&n_bystand, sizeof(n_bystand), 1, f);
    fwrite(&n_floats, sizeof(n_floats), 1, f);
    fwrite(out.data(), sizeof(float), out.size(), f);
    fclose(f);
    return 0;
}

#ifndef _WIN32
static bool spawn(char ** argv, const char * lanes, const std::string & path) {
    const pid_t pid = fork();
    if (pid == 0) {
        setenv("LLAMA_GDN_REPLAY_LANES", lanes, 1);
        setenv("LLAMA_TEST_LANES_DUMP", path.c_str(), 1);
        execv("/proc/self/exe", argv);
        _exit(127);
    }
    int status = 0;
    return pid > 0 && waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

static bool load(const std::string & path, uint64_t & n_joint, uint64_t & n_bystand, std::vector<float> & out) {
    FILE * f = fopen(path.c_str(), "rb");
    uint64_t n = 0;
    bool ok = f && fread(&n_joint, sizeof(n_joint), 1, f) == 1 && fread(&n_bystand, sizeof(n_bystand), 1, f) == 1 &&
              fread(&n, sizeof(n), 1, f) == 1;
    if (ok) {
        out.resize(n);
        ok = fread(out.data(), sizeof(float), n, f) == n;
    }
    if (f) {
        fclose(f);
    }
    return ok;
}
#endif

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    const char * dump = getenv("LLAMA_TEST_LANES_DUMP");
    if (dump == nullptr) {
#ifdef _WIN32
        fprintf(stderr, "skipped on Windows\n");
        return 0;
#else
        const std::string base = std::string("test-gdn-replay-lanes.") + std::to_string(getpid());
        const std::string p_off = base + ".off";
        const std::string p_on  = base + ".on";
        if (!spawn(argv, "0", p_off) || !spawn(argv, "1", p_on)) {
            fprintf(stderr, "FAIL: a child run failed\n");
            return 1;
        }
        uint64_t j_off = 0;
        uint64_t j_on  = 0;
        uint64_t b_off = 0;
        uint64_t b_on  = 0;
        std::vector<float> off;
        std::vector<float> on;
        const bool ok_load = load(p_off, j_off, b_off, off) && load(p_on, j_on, b_on, on);
        remove(p_off.c_str());
        remove(p_on.c_str());
        if (!ok_load || off.size() != on.size() || off.empty()) {
            fprintf(stderr, "FAIL: dumps missing or of different size\n");
            return 1;
        }
        struct diff_stats {
            double max_diff = 0.0;
            double max_abs  = 0.0;
            size_t n_diff   = 0;
        };
        auto compare = [](const std::vector<float> & a, const std::vector<float> & b) {
            diff_stats r;
            for (size_t i = 0; i < a.size(); ++i) {
                const double d = fabs((double) a[i] - (double) b[i]);
                r.max_diff = std::max(r.max_diff, std::isfinite(d) ? d : INFINITY);
                r.max_abs  = std::max(r.max_abs, fabs((double) a[i]));
                r.n_diff  += a[i] != b[i];
            }
            return r;
        };
        const diff_stats d = compare(off, on);
        printf("logits: %zu values, %zu differ, max diff %.3e (max |logit| %.3e); joint ubatches: off %llu, on %llu; "
               "bystander cuts: off %llu, on %llu\n",
                off.size(), d.n_diff, d.max_diff, d.max_abs, (unsigned long long) j_off, (unsigned long long) j_on,
                (unsigned long long) b_off, (unsigned long long) b_on);
        // CPU computes each row on its own, so joint and split must be bit-equal there. The generated model's logits
        // are small (a wrong lane mask moved them by ~3e-6), so a backend with other fp order needs a tight
        // LLAMA_TEST_LANES_TOL (absolute)
        const char *  tol_env = getenv("LLAMA_TEST_LANES_TOL");
        const double  tol     = tol_env ? atof(tol_env) : 0.0;
        // scenario C (unified, seq 1 with a pending replay) must hit the bystander cut with lanes on
        const bool ok = j_off == 0 && j_on > 0 && b_off == 0 && b_on > 0 && d.max_diff <= tol;
        printf("%s\n", ok ? "OK" : "FAIL");
        return ok ? 0 : 1;
#endif
    }

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    ggml_backend_load_all();
    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    if (model == nullptr) {
        fprintf(stderr, "failed to load the model\n");
        return 1;
    }
    return run_child(params, model, dump);
}
