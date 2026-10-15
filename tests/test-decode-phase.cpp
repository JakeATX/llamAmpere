// One-shot decode-phase tag. With a #217 keep map set, PROMPT and untagged calls stay
// bit-identical to a context with no map. A GENERATION call is where the map shows up.
// On CPU nothing is resident, so the map is visible in the logits.
//
// The fixture model is the qwen35moe arch test model: 2 experts, 2 used, so keep "1"
// drops route rank >= 1. Four tokens are 8 routed rows, under the 64-row width guard,
// so a missing tag cannot hide behind that guard.

#define LLAMA_ARCH_FIXTURE_ONLY
#include "test-llama-archs.cpp"
#include "../src/llama-ext.h"

#include <cstdlib>

static_assert(LLAMA_DECODE_PHASE_UNKNOWN == 0, "no tag is 0");
static_assert(LLAMA_DECODE_PHASE_PROMPT == 1, "prompt tag");
static_assert(LLAMA_DECODE_PHASE_GENERATION == 2, "generation tag");
static_assert(LLAMA_DECODE_PHASE_UNKNOWN == LLAMA_DECODE_PHASE_AUTOMATIC, "UNKNOWN aliases AUTOMATIC");

static int g_fail = 0;

static void expect(bool ok, const char * name) {
    if (ok) {
        fprintf(stdout, "PASS %s\n", name);
    } else {
        fprintf(stderr, "FAIL %s\n", name);
        g_fail++;
    }
}

static ggml_backend_dev_t cpu_dev() {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            return dev;
        }
    }
    return nullptr;
}

static llama_model_ptr load_model() {
    ggml_backend_dev_t cpu = cpu_dev();
    if (cpu == nullptr) {
        throw std::runtime_error("no CPU backend device");
    }
    ggml_backend_dev_t devs[2] = { cpu, nullptr };

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    model_params.devices = devs;
    model_params.progress_callback = silent_model_load_progress;

    gguf_context_ptr gguf = get_gguf_ctx(LLM_ARCH_QWEN35MOE, true);
    tensor_data_params tensor_params = { /*seed*/ 1, /*stdev*/ 0.1f };
    llama_model_ptr model(llama_model_init_from_user(gguf.get(), set_tensor_data, &tensor_params, model_params));
    if (!model) {
        throw std::runtime_error("failed to create qwen35moe model");
    }
    return model;
}

// keep_map is a string literal, or nullptr when the map is off. It must outlive the context.
static llama_context_ptr make_ctx(llama_model * model, const char * keep_map) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 32;
    cp.n_batch = 16;
    cp.n_ubatch = 16;
    cp.n_threads = 1;
    cp.n_threads_batch = 1;
    cp.moe_keep_map = keep_map;
    llama_context_ptr ctx(llama_init_from_model(model, cp));
    if (!ctx) {
        throw std::runtime_error("failed to create context");
    }
    return ctx;
}

enum step_tag { STEP_NONE = 0, STEP_PROMPT = 1, STEP_GEN = 2 };

static void decode_tokens(llama_context * ctx, const std::vector<llama_token> & toks, llama_pos pos0, step_tag tag) {
    if (tag == STEP_PROMPT) {
        llama_set_next_decode_phase(ctx, LLAMA_DECODE_PHASE_PROMPT);
    } else if (tag == STEP_GEN) {
        llama_set_next_decode_phase(ctx, LLAMA_DECODE_PHASE_GENERATION);
    }
    const int32_t n = (int32_t) toks.size();
    llama_batch batch = llama_batch_init(n, 0, 1);
    batch.n_tokens = n;
    for (int32_t i = 0; i < n; ++i) {
        batch.token[i] = toks[i];
        batch.pos[i] = pos0 + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
    }
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        throw std::runtime_error("llama_decode failed: " + std::to_string(rc));
    }
}

static std::vector<float> logits_of(llama_context * ctx, const llama_model * model, int32_t n_tokens) {
    llama_synchronize(ctx);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    if (n_vocab <= 0) {
        throw std::runtime_error("empty vocab");
    }
    std::vector<float> out((size_t) n_tokens * (size_t) n_vocab);
    for (int32_t i = 0; i < n_tokens; ++i) {
        const float * row = llama_get_logits_ith(ctx, i);
        if (row == nullptr) {
            throw std::runtime_error("null logits");
        }
        memcpy(out.data() + (size_t) i * (size_t) n_vocab, row, sizeof(float) * (size_t) n_vocab);
    }
    return out;
}

static bool same_floats(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && (a.empty() || memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

static int count_diff(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        return -1;
    }
    int n = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (memcmp(&a[i], &b[i], sizeof(float)) != 0) {
            n++;
        }
    }
    return n;
}

static bool all_zero(const std::vector<float> & a) {
    for (float x : a) {
        if (x != 0.0f) {
            return false;
        }
    }
    return true;
}

static std::vector<uint8_t> save_seq(llama_context * ctx) {
    const size_t n = llama_state_seq_get_size(ctx, 0);
    if (n == 0) {
        throw std::runtime_error("empty sequence state");
    }
    std::vector<uint8_t> buf(n);
    const size_t wrote = llama_state_seq_get_data(ctx, buf.data(), buf.size(), 0);
    if (wrote != n) {
        throw std::runtime_error("short sequence state write");
    }
    return buf;
}

static void load_seq(llama_context * ctx, const std::vector<uint8_t> & buf) {
    const size_t read = llama_state_seq_set_data(ctx, buf.data(), buf.size(), 0);
    if (read == 0) {
        throw std::runtime_error("sequence state restore failed");
    }
}

static bool same_bytes(const std::vector<uint8_t> & a, const std::vector<uint8_t> & b) {
    return a.size() == b.size() && (a.empty() || memcmp(a.data(), b.data(), a.size()) == 0);
}

// Last call's logits, plus the sequence state, against a context with no keep map.
static void check_exact(llama_model * model, const char * name,
        const std::vector<std::vector<llama_token>> & calls, const std::vector<step_tag> & tags) {
    auto mapped = make_ctx(model, "1");
    auto plain  = make_ctx(model, nullptr);
    llama_pos pos = 0;
    for (size_t i = 0; i < calls.size(); ++i) {
        decode_tokens(mapped.get(), calls[i], pos, tags[i]);
        decode_tokens(plain.get(), calls[i], pos, tags[i]);
        pos += (llama_pos) calls[i].size();
    }
    const auto & last = calls.back();
    const std::vector<float> lm = logits_of(mapped.get(), model, (int32_t) last.size());
    const std::vector<float> lp = logits_of(plain.get(), model, (int32_t) last.size());
    const bool logits_ok = same_floats(lm, lp) && !all_zero(lm);
    const bool state_ok = same_bytes(save_seq(mapped.get()), save_seq(plain.get()));
    if (!logits_ok || !state_ok) {
        fprintf(stderr, "  %s logits_eq=%d state_eq=%d diff=%d zero=%d\n", name,
                (int) same_floats(lm, lp), (int) state_ok, count_diff(lm, lp), (int) all_zero(lm));
    }
    expect(logits_ok && state_ok, name);
}

int main(int argc, char ** argv) {
    const char * save_path = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--save") == 0 && i + 1 < argc) {
            save_path = argv[++i];
        } else {
            fprintf(stderr, "usage: %s [--save gguf]\n", argv[0]);
            return 2;
        }
    }

    // one thread, so the two contexts take the same reduction order
    setenv("OMP_NUM_THREADS", "1", 1);
    setenv("OPENBLAS_NUM_THREADS", "1", 1);
    setenv("MKL_NUM_THREADS", "1", 1);

    llama_backend_init();
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_WARN) {
            fputs(text, stderr);
        }
    }, nullptr);

    try {
        llama_model_ptr model = load_model();
        if (save_path != nullptr) {
            llama_model_save_to_file(model.get(), save_path);
            fprintf(stdout, "saved %s\n", save_path);
        }

        const std::vector<llama_token> prompt = { 1, 2, 3, 4 };
        const std::vector<llama_token> tail   = { 5 };

        // (a) multi-row prompt tagged PROMPT
        check_exact(model.get(), "a multi-row PROMPT", { prompt }, { STEP_PROMPT });

        // (b) one-row prompt tail tagged PROMPT, after an exact prefix
        check_exact(model.get(), "b one-row PROMPT tail", { prompt, tail }, { STEP_PROMPT, STEP_PROMPT });

        // (c) untagged one-row, fresh and as a tail after a prompt
        check_exact(model.get(), "c untagged one-row", { tail }, { STEP_NONE });
        check_exact(model.get(), "c untagged one-row tail", { prompt, tail }, { STEP_PROMPT, STEP_NONE });

        // (d) one-row GENERATION differs from the same call with no map
        {
            auto mapped = make_ctx(model.get(), "1");
            auto plain  = make_ctx(model.get(), nullptr);
            decode_tokens(mapped.get(), tail, 0, STEP_GEN);
            decode_tokens(plain.get(), tail, 0, STEP_NONE);
            const std::vector<float> lm = logits_of(mapped.get(), model.get(), 1);
            const std::vector<float> lp = logits_of(plain.get(), model.get(), 1);
            const int diff = count_diff(lm, lp);
            fprintf(stdout, "  d differing logits: %d / %zu\n", diff, lm.size());
            expect(diff > 0 && !all_zero(lp), "d GENERATION differs from no map");
        }

        // (e) the tag resets: a GENERATION call, then an untagged call, is exact on the second call.
        // The first call's state is restored into a no-map context so the second call is the only difference.
        {
            auto mapped = make_ctx(model.get(), "1");
            auto plain  = make_ctx(model.get(), nullptr);
            decode_tokens(mapped.get(), std::vector<llama_token>{ 7 }, 0, STEP_GEN);
            const std::vector<uint8_t> state = save_seq(mapped.get());
            decode_tokens(mapped.get(), std::vector<llama_token>{ 8 }, 1, STEP_NONE);
            const std::vector<float> lm = logits_of(mapped.get(), model.get(), 1);
            load_seq(plain.get(), state);
            decode_tokens(plain.get(), std::vector<llama_token>{ 8 }, 1, STEP_NONE);
            const std::vector<float> lp = logits_of(plain.get(), model.get(), 1);
            const int diff = count_diff(lm, lp);
            if (diff != 0) {
                fprintf(stderr, "  e second-call differing logits: %d\n", diff);
            }
            expect(diff == 0, "e tag resets before the next call");
        }

        // (f) with no map, a GENERATION tag matches an untagged call
        {
            auto tagged = make_ctx(model.get(), nullptr);
            auto plain  = make_ctx(model.get(), nullptr);
            decode_tokens(tagged.get(), tail, 0, STEP_GEN);
            decode_tokens(plain.get(), tail, 0, STEP_NONE);
            const bool logits_ok = same_floats(
                    logits_of(tagged.get(), model.get(), 1),
                    logits_of(plain.get(), model.get(), 1));
            const bool state_ok = same_bytes(save_seq(tagged.get()), save_seq(plain.get()));
            expect(logits_ok && state_ok, "f no map, GENERATION matches untagged");
        }

        // Prompt-cache shape: PROMPT prefix plus a one-token PROMPT tail, then a few GENERATION
        // tokens. The state saved after the prompt must match a fresh PROMPT prefill, and must
        // match a prefill with no map (so a lossy prompt fails this, not only a mismatched pair).
        {
            auto first = make_ctx(model.get(), "1");
            auto fresh = make_ctx(model.get(), "1");
            auto plain = make_ctx(model.get(), nullptr);

            decode_tokens(first.get(), prompt, 0, STEP_PROMPT);
            decode_tokens(first.get(), tail, (llama_pos) prompt.size(), STEP_PROMPT);
            const std::vector<uint8_t> saved = save_seq(first.get());

            decode_tokens(first.get(), std::vector<llama_token>{ 6 }, 5, STEP_GEN);
            decode_tokens(first.get(), std::vector<llama_token>{ 7 }, 6, STEP_GEN);
            decode_tokens(first.get(), std::vector<llama_token>{ 8 }, 7, STEP_GEN);

            decode_tokens(fresh.get(), prompt, 0, STEP_PROMPT);
            decode_tokens(fresh.get(), tail, (llama_pos) prompt.size(), STEP_PROMPT);
            const std::vector<uint8_t> again = save_seq(fresh.get());

            decode_tokens(plain.get(), prompt, 0, STEP_PROMPT);
            decode_tokens(plain.get(), tail, (llama_pos) prompt.size(), STEP_PROMPT);
            const std::vector<uint8_t> exact = save_seq(plain.get());

            auto restored = make_ctx(model.get(), "1");
            load_seq(restored.get(), saved);
            const std::vector<uint8_t> roundtrip = save_seq(restored.get());

            fprintf(stdout, "  prompt state bytes: %zu\n", saved.size());
            expect(same_bytes(saved, again), "prompt state matches a fresh PROMPT prefill");
            expect(same_bytes(saved, exact), "prompt state matches a no-map prefill");
            expect(same_bytes(saved, roundtrip), "prompt state round-trips through set_data");
        }
    } catch (const std::exception & err) {
        fprintf(stderr, "FAIL exception: %s\n", err.what());
        g_fail++;
    }

    llama_backend_free();
    if (g_fail == 0) {
        fprintf(stdout, "all decode-phase checks passed\n");
        return 0;
    }
    fprintf(stderr, "%d decode-phase check(s) failed\n", g_fail);
    return 1;
}
