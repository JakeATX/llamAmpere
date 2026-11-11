// KV state save/load while block KV streaming (kv_stream_arena_mib > 0) is on.
//
// Generated 4-layer llama (head 128, 2 KV heads, f32 weights, no model file). Block KV streaming needs
// the CUDA backend, so:
//   * CPU part (always runs): a CPU-only model with kv_stream_arena_mib > 0 must be refused at context
//     creation (llama_init_from_model returns null) instead of running some silent fallback.
//   * GPU part (needs a CUDA device, else exit 77 = skipped): for arena 0 (baseline) and each streaming
//     arena size it decodes a prompt longer than the arena's resident KV, then
//       seq   : saves seq 0, clears, restores into the same context, decodes two more tokens,
//       fresh : restores that seq state into a fresh context and decodes the next token,
//       full  : round-trips the whole-context state (llama_state_get/set_data) into a fresh context,
//     comparing the logits bit for bit against an uninterrupted run with the same arena.
//     A restore that returns 0 must leave an empty cache that still decodes correctly (clean refusal).
//
// Prints one verdict line: PASS bit-exact / REFUSES cleanly / CORRUPTS.
//
//   test-kv-stream-state [arena_mib ...]      default arenas: 256 8

#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"
#include "gguf.h"

#include "../src/llama-arch.h"
#include "../src/llama-kv-cache.h"
#include "test-kv-writer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static const uint32_t n_vocab = 128, head_dim = 128, n_layer = 4;
static const uint32_t n_ctx = 8192;
// 7000 cells = 28 pages of 256 per layer, about 28 MiB of f16 KV: more than the small arenas keep resident, so a
// save/load under real streaming pressure is tested (LLAMA_KV_STREAM_TRACE=1 shows active tokens vs resident pages)
static const llama_pos n_prompt = 7000;

static void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static void tensor_data(ggml_tensor * tensor, void * /*userdata*/) {
    std::mt19937 gen(std::hash<std::string>()(tensor->name));
    std::normal_distribution<float> dis(0.0f, 0.1f);
    GGML_ASSERT(tensor->type == GGML_TYPE_F32);
    std::vector<float> tmp(ggml_nelements(tensor));
    for (auto & v : tmp) v = dis(gen);
    ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
}

static llama_model * make_model(bool gpu) {
    const llm_arch arch = LLM_ARCH_LLAMA;
    gguf_context * meta = gguf_init_empty();
    {
        test_kv_writer ms(arch, meta);
        ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,         llm_arch_name(arch));
        ms.add_kv(LLM_KV_VOCAB_SIZE,                   n_vocab);
        ms.add_kv(LLM_KV_CONTEXT_LENGTH,               uint32_t(8192));
        ms.add_kv(LLM_KV_EMBEDDING_LENGTH,             2*head_dim);
        ms.add_kv(LLM_KV_BLOCK_COUNT,                  n_layer);
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH,          uint32_t(512));
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,         uint32_t(2));
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV,      uint32_t(2));
        ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,  1e-5f);
        std::vector<std::string> tokens(n_vocab);
        for (uint32_t i = 0; i < n_vocab; ++i) tokens[i] = "tok_" + std::to_string(i);
        ms.add_kv(LLM_KV_TOKENIZER_MODEL,  "test");
        ms.add_kv(LLM_KV_TOKENIZER_LIST,   tokens);
        ms.add_kv(LLM_KV_TOKENIZER_SCORES, std::vector<float>(n_vocab, 0.0f));
    }
    llama_model_params mp = llama_model_default_params();
    ggml_backend_dev_t no_devices[] = { nullptr };
    if (gpu) {
        mp.n_gpu_layers = 99;
    } else {
        mp.devices = no_devices;
    }
    llama_model * model = llama_model_init_from_user(meta, tensor_data, nullptr, mp);
    gguf_free(meta);
    return model;
}

static llama_context * make_ctx(llama_model * model, uint32_t arena_mib) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = n_ctx;
    cp.n_batch         = 512;
    cp.n_ubatch        = 512;
    cp.n_seq_max       = 1;
    cp.n_threads       = 4;
    cp.n_threads_batch = 4;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.offload_kqv     = true;
    cp.kv_stream_arena_mib = arena_mib;
    return llama_init_from_model(model, cp);
}

struct ctx_ptr {
    llama_context * ctx = nullptr;
    explicit ctx_ptr(llama_context * c) : ctx(c) {}
    ~ctx_ptr() { if (ctx) llama_free(ctx); }
    ctx_ptr(const ctx_ptr &) = delete;
    ctx_ptr & operator=(const ctx_ptr &) = delete;
};

static llama_token token_at(llama_pos p) { return (llama_token) ((p*37 + 5) % n_vocab); }

// decode [from, to) in chunks of `width`; returns the logits of the last position
static std::vector<float> decode(llama_context * ctx, llama_pos from, llama_pos to, int width) {
    std::vector<float> last;
    for (llama_pos p = from; p < to;) {
        const int n = std::min<int>(width, to - p);
        llama_batch batch = llama_batch_init(n, 0, 1);
        batch.n_tokens = n;
        for (int j = 0; j < n; ++j) {
            batch.token[j]     = token_at(p + j);
            batch.pos[j]       = p + j;
            batch.n_seq_id[j]  = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j]    = j + 1 == n;
        }
        const int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        require(rc == 0, "llama_decode failed at pos " + std::to_string(p));
        p += n;
        const float * logits = llama_get_logits_ith(ctx, -1);
        require(logits != nullptr, "logits present");
        last.assign(logits, logits + n_vocab);
    }
    return last;
}

static bool streaming_attached(llama_context * ctx) {
    const auto * mem = llama_get_memory(ctx);
    const auto * kv  = dynamic_cast<const llama_kv_cache *>(mem);
    return kv != nullptr && kv->has_kv_stream_targets();
}

static float max_diff(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) return INFINITY;
    float d = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) d = std::max(d, std::fabs(a[i] - b[i]));
    return d;
}

static bool same(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0;
}

enum outcome { OUT_EXACT, OUT_REFUSED, OUT_CORRUPT };

static const char * outcome_name(outcome o) {
    return o == OUT_EXACT ? "PASS bit-exact" : o == OUT_REFUSED ? "REFUSES cleanly" : "CORRUPTS";
}

// compare continuation logits after a restore with the uninterrupted reference
static outcome judge(const char * what, uint32_t arena, const std::vector<float> & got, const std::vector<float> & ref) {
    if (same(got, ref)) {
        printf("  arena %4u %-28s bit-exact\n", arena, what);
        return OUT_EXACT;
    }
    printf("  arena %4u %-28s MISMATCH max|diff| = %g\n", arena, what, (double) max_diff(got, ref));
    return OUT_CORRUPT;
}

// a restore returned 0: the cache must be empty and must still decode the reference from scratch
static outcome judge_refusal(const char * what, uint32_t arena, llama_context * ctx, const std::vector<float> & ref) {
    const llama_pos pmax = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
    if (pmax != -1) {
        printf("  arena %4u %-28s refused but left cells behind (seq 0 pos_max %d)\n", arena, what, pmax);
        return OUT_CORRUPT;
    }
    decode(ctx, 0, n_prompt, 512);
    const auto again = decode(ctx, n_prompt, n_prompt + 1, 1);
    if (!same(again, ref)) {
        printf("  arena %4u %-28s refused, but the context decodes wrong afterwards (max|diff| %g)\n",
               arena, what, (double) max_diff(again, ref));
        return OUT_CORRUPT;
    }
    printf("  arena %4u %-28s refused cleanly (empty cache, re-decode bit-exact)\n", arena, what);
    return OUT_REFUSED;
}

static outcome worst(outcome a, outcome b) { return std::max(a, b); }

// returns false if this arena size cannot be used (context refused at creation)
static bool run_arena(llama_model * model, uint32_t arena, outcome & res) {
    std::vector<float> ref1, ref2;
    {
        ctx_ptr c(make_ctx(model, arena));
        if (!c.ctx) {
            printf("  arena %4u context creation refused (see log above), skipped\n", arena);
            return false;
        }
        if (arena > 0) {
            require(streaming_attached(c.ctx), "kv-stream context has a streaming target");
        }
        decode(c.ctx, 0, n_prompt, 512);
        ref1 = decode(c.ctx, n_prompt,     n_prompt + 1, 1);
        ref2 = decode(c.ctx, n_prompt + 1, n_prompt + 2, 1);
    }

    std::vector<uint8_t> seq_buf, full_buf;

    // seq: save, clear, restore into the same context
    {
        ctx_ptr c(make_ctx(model, arena));
        require(c.ctx != nullptr, "second context");
        decode(c.ctx, 0, n_prompt, 512);

        const size_t n = llama_state_seq_get_size(c.ctx, 0);
        require(n > 0, "llama_state_seq_get_size > 0");
        seq_buf.resize(n);
        require(llama_state_seq_get_data(c.ctx, seq_buf.data(), n, 0) == n, "llama_state_seq_get_data");

        const size_t nf = llama_state_get_size(c.ctx);
        require(nf > 0, "llama_state_get_size > 0");
        full_buf.resize(nf);
        require(llama_state_get_data(c.ctx, full_buf.data(), nf) == nf, "llama_state_get_data");

        llama_memory_clear(llama_get_memory(c.ctx), true);
        if (llama_state_seq_set_data(c.ctx, seq_buf.data(), seq_buf.size(), 0) == 0) {
            res = worst(res, judge_refusal("seq clear+restore", arena, c.ctx, ref1));
        } else {
            require(llama_memory_seq_pos_max(llama_get_memory(c.ctx), 0) == n_prompt - 1, "restored seq ends at the prompt");
            res = worst(res, judge("seq clear+restore (+1)", arena, decode(c.ctx, n_prompt,     n_prompt + 1, 1), ref1));
            res = worst(res, judge("seq clear+restore (+2)", arena, decode(c.ctx, n_prompt + 1, n_prompt + 2, 1), ref2));
        }
    }

    // fresh: the seq state into a context that never decoded
    {
        ctx_ptr c(make_ctx(model, arena));
        require(c.ctx != nullptr, "fresh context");
        if (llama_state_seq_set_data(c.ctx, seq_buf.data(), seq_buf.size(), 0) == 0) {
            res = worst(res, judge_refusal("seq into fresh ctx", arena, c.ctx, ref1));
        } else {
            res = worst(res, judge("seq into fresh ctx (+1)", arena, decode(c.ctx, n_prompt, n_prompt + 1, 1), ref1));
        }
    }

    // full: whole-context state into a fresh context
    {
        ctx_ptr c(make_ctx(model, arena));
        require(c.ctx != nullptr, "full-state context");
        if (llama_state_set_data(c.ctx, full_buf.data(), full_buf.size()) == 0) {
            res = worst(res, judge_refusal("full state into fresh ctx", arena, c.ctx, ref1));
        } else {
            res = worst(res, judge("full state into fresh (+1)", arena, decode(c.ctx, n_prompt, n_prompt + 1, 1), ref1));
        }
    }
    return true;
}

static bool have_gpu() {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            return true;
        }
    }
    return false;
}

int main(int argc, char ** argv) {
    std::vector<uint32_t> arenas;
    for (int i = 1; i < argc; ++i) {
        arenas.push_back((uint32_t) strtoul(argv[i], nullptr, 10));
    }
    if (arenas.empty()) {
        arenas = { 256, 8 };
    }

    llama_backend_init();
    int rc = 0;
    try {
        // CPU: kv-stream must be refused at context creation, never silently ignored
        {
            llama_model * model = make_model(false);
            require(model != nullptr, "cpu model");
            llama_context * ctx = make_ctx(model, 64);
            const bool refused = ctx == nullptr;
            if (ctx) llama_free(ctx);
            llama_model_free(model);
            require(refused, "CPU-only context with kv_stream_arena_mib > 0 must be refused at creation");
            printf("cpu: kv-stream context refused at creation (CUDA required) - ok\n");
        }

        if (!have_gpu()) {
            printf("SKIP: no GPU device; the save/load-with-streaming check needs CUDA\n");
            rc = 77;
        } else {
            llama_model * model = make_model(true);
            require(model != nullptr, "gpu model");

            outcome base = OUT_EXACT;
            require(run_arena(model, 0, base), "baseline (arena 0) context");
            printf("baseline (kv-stream off): %s\n", outcome_name(base));

            outcome res = OUT_EXACT;
            int n_run = 0;
            for (uint32_t a : arenas) {
                if (a > 0 && run_arena(model, a, res)) {
                    ++n_run;
                }
            }
            llama_model_free(model);
            require(n_run > 0, "no kv-stream arena size could be created");
            printf("RESULT kv-stream save/load: %s\n", outcome_name(res));
            if (base != OUT_EXACT) {
                printf("note: the baseline without kv-stream is not bit-exact either\n");
            }
            rc = res == OUT_CORRUPT || base == OUT_CORRUPT ? 1 : 0;
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "FAIL: %s\n", e.what());
        rc = 1;
    }
    llama_backend_free();
    return rc;
}
