// SJ-KVaRN prefill split independence (#197).
//
// The server restores a mid-prompt checkpoint and prefills the rest of the prompt from there, so the same
// prompt is evaluated with different ubatch boundaries than a cold run. This test prefills the same N tokens
//   cold:      [0, N) in n_ubatch chunks from 0
//   split c:   [0, c) then [c, N) (each in n_ubatch chunks)
//   trunc c:   [0, c + 300) with other tokens past c, seq_rm(c, -1), then [c, N)  (the checkpoint-restore path)
// and compares the last logits (bit-exact) for plain caches (controls) and the SJ-KVaRN pairs, with a tail
// long enough that nothing seals, a fixed tail and the adaptive tail. The SJ-KVaRN rows must be bit-identical.
// Default (restore grid, LLAMA_SJKVARN_SEAL_GRID unset/2): cuts on the n_ubatch grid only (where the server puts
// SJ-KVaRN checkpoints), and the trunc/state paths call llama_sj_kvarn_prefill_cut() before the cut like the server.
// LLAMA_SJKVARN_SEAL_GRID=1 (full grid schedule): any cut; =0 restores the v0.5.2 split-dependent behaviour.
//
//   test-sjkvarn-split          CPU (SJ_TEST_GPU=1: all layers offloaded)

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-sjkvarn.h"
#include "ggml-backend.h"
#include "llama.h"

#include "gguf.h"

#include "../src/llama-kv-cache.h"
#include "../src/llama-context.h"
#include "../src/llama-arch.h"
#include "../src/llama-model-saver.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static const uint32_t n_vocab = 128, n_embd = 512, n_layer = 4;
static uint32_t g_ub = 512; // n_batch = n_ubatch of every context
static int g_out_of_reach = 0; // [PCR1] restored cuts past the ring reach (inexact by design, info)

static std::string g_last_error;

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

static llama_model * make_model() {
    gguf_context * meta = gguf_init_empty();
    {
        llama_model_saver ms(LLM_ARCH_LLAMA, meta);
        ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,         llm_arch_name(LLM_ARCH_LLAMA));
        ms.add_kv(LLM_KV_VOCAB_SIZE,                   n_vocab);
        ms.add_kv(LLM_KV_CONTEXT_LENGTH,               uint32_t(8192));
        ms.add_kv(LLM_KV_EMBEDDING_LENGTH,             n_embd);
        ms.add_kv(LLM_KV_BLOCK_COUNT,                  n_layer);
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH,          uint32_t(768));
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
    if (!getenv("SJ_TEST_GPU")) {
        mp.devices = no_devices; // CPU only, also in a CUDA build (SJ_TEST_GPU=1: default devices, all layers offloaded)
    }
    llama_model * model = llama_model_init_from_user(meta, tensor_data, nullptr, mp);
    gguf_free(meta);
    return model;
}

struct sj_cfg {
    const char * name;
    uint32_t  bits_k, bits_v;
    ggml_type body;                 // F32 scalar, I16 trellis
    ggml_type staging = GGML_TYPE_TQ6_0;
    ggml_type sink    = GGML_TYPE_F16; // F16 = separate fp16 sink, COUNT = inherit staging
    uint32_t  tail = 256, tail_max = 512, flush_chunk = 0;
    uint32_t  edge_layers = 0, edge_bits_k = 4, edge_bits_v = 4;
    ggml_type edge_body = GGML_TYPE_F32;
    bool      plain = false;        // plain cache (no SJ-KVaRN), K/V type plain_type
    ggml_type plain_type = GGML_TYPE_F16;
};

static llama_context * make_ctx(llama_model * model, const sj_cfg & c, uint32_t n_seq = 1, bool unified = false) {
    llama_context_params cp = llama_context_default_params();
    // several sequences: every sequence gets the same 4096-position stream a single-sequence context has
    cp.n_ctx = unified ? 4096 : 4096*n_seq;
    cp.kv_unified = unified;
    cp.n_batch = g_ub;
    cp.n_ubatch = g_ub;
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    cp.n_seq_max = n_seq;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    if (c.plain) {
        cp.type_k = cp.type_v = c.plain_type;
    } else {
        cp.sj_kvarn_bits_k = c.bits_k;
        cp.sj_kvarn_bits_v = c.bits_v;
        cp.sj_kvarn_body_type = c.body;
        cp.sj_kvarn_staging_type = c.staging;
        cp.sj_kvarn_sink_type = c.sink;
        cp.sj_kvarn_sink = 128;
        cp.sj_kvarn_tail = c.tail;
        cp.sj_kvarn_tail_max = c.tail_max;
        cp.sj_kvarn_flush_chunk = c.flush_chunk;
        cp.sj_kvarn_edge_layers = c.edge_layers;
        cp.sj_kvarn_edge_bits_k = c.edge_bits_k;
        cp.sj_kvarn_edge_bits_v = c.edge_bits_v;
        cp.sj_kvarn_edge_body_type = c.edge_body;
    }
    return llama_init_from_model(model, cp);
}

static llama_token token_at(llama_pos p, int salt) { return (llama_token) ((p*37 + salt*11 + 5) % n_vocab); }

// decode [from, to) in chunks of `width`; returns the logits of the last position
static std::vector<float> decode(llama_context * ctx, llama_pos from, llama_pos to, int width, int salt = 0, llama_seq_id seq = 0) {
    std::vector<float> last;
    for (llama_pos p = from; p < to;) {
        const int n = std::min<int>(width, to - p);
        llama_batch batch = llama_batch_init(n, 0, 1);
        batch.n_tokens = n;
        for (int j = 0; j < n; ++j) {
            batch.token[j] = token_at(p + j, salt);
            batch.pos[j] = p + j;
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = seq;
            batch.logits[j] = j + 1 == n;
        }
        const int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        require(rc == 0, "decode");
        p += n;
        const float * logits = llama_get_logits_ith(ctx, -1);
        require(logits != nullptr, "logits present");
        last.assign(logits, logits + n_vocab);
    }
    return last;
}


static llama_kv_cache * cache_of(llama_context * ctx) {
    auto * kv = dynamic_cast<llama_kv_cache *>(llama_get_memory(ctx));
    require(kv != nullptr, "llama_kv_cache memory");
    return kv;
}

static float max_diff(const std::vector<float> & a, const std::vector<float> & b) {
    float d = 0.0f;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) d = std::max(d, std::fabs(a[i] - b[i]));
    return d;
}

static void prefill(llama_context * ctx, llama_pos from, llama_pos to, int salt = 0) { decode(ctx, from, to, (int) g_ub, salt); }

// returns the number of variants that differ from cold
static int run_cfg(llama_model * model, const sj_cfg & c, llama_pos N, const std::vector<llama_pos> & cuts) {
    int n_diff = 0;
    llama_context * ctx = make_ctx(model, c);
    require(ctx != nullptr, "context");
    std::vector<float> cold;
    {
        cold = decode(ctx, 0, N, (int) g_ub);
    }
    const uint32_t B_cold = c.plain ? 0 : cache_of(ctx)->get_sj_kvarn_sealed_end();
    printf("%-22s cold N=%d B=%u\n", c.name, (int) N, B_cold);
    for (llama_pos cut : cuts) {
        for (int mode = 0; mode < 4; ++mode) {
            llama_memory_clear(llama_get_memory(ctx), true);
            std::vector<float> got;
            bool in_reach = true;
            if (mode == 0) {
                prefill(ctx, 0, cut);
                got = decode(ctx, cut, N, (int) g_ub);
            } else if (mode == 2) {
                // [#197] server prompt-cache path: save the state of a longer prompt that shares [0, cut), load it
                // back, cut at the checkpoint and re-prefill (what a RAM/disk prompt-cache hit + checkpoint restore does)
                prefill(ctx, 0, cut);
                prefill(ctx, cut, std::min<llama_pos>(cut + 1400, 4000), 7); // far enough that B passes the cut
                std::vector<uint8_t> buf(llama_state_seq_get_size(ctx, 0));
                require(llama_state_seq_get_data(ctx, buf.data(), buf.size(), 0) == buf.size(), "state save");
                llama_memory_clear(llama_get_memory(ctx), true);
                require(llama_state_seq_set_data(ctx, buf.data(), buf.size(), 0) == buf.size(), "state load");
                in_reach = c.plain || cache_of(ctx)->sj_kvarn_cut_in_reach(0, cut);
                llama_sj_kvarn_prefill_cut(ctx, 0, cut); // [PCR1] what the server does before the cut
                const bool ok = llama_memory_seq_rm(llama_get_memory(ctx), 0, cut, -1);
                if (!ok) { printf("%-22s   state %d: seq_rm refused\n", c.name, (int) cut); continue; }
                got = decode(ctx, cut, N, (int) g_ub);
            } else if (mode == 3) {
                // [PCR1] a prompt that shares [0, cut), then 400 generated tokens one decode step at a time (the decode
                // seal schedule moves B past the cut), saved, loaded, cut and re-prefilled (a revisit of an edited
                // prompt after a turn, from the RAM/disk prompt cache)
                const llama_pos e = std::min<llama_pos>(cut + 200, 3600);
                prefill(ctx, 0, cut);
                prefill(ctx, cut, e, 7);
                decode(ctx, e, e + 400, 1, 9);
                std::vector<uint8_t> buf(llama_state_seq_get_size(ctx, 0));
                require(llama_state_seq_get_data(ctx, buf.data(), buf.size(), 0) == buf.size(), "state save");
                llama_memory_clear(llama_get_memory(ctx), true);
                require(llama_state_seq_set_data(ctx, buf.data(), buf.size(), 0) == buf.size(), "state load");
                in_reach = c.plain || cache_of(ctx)->sj_kvarn_cut_in_reach(0, cut);
                llama_sj_kvarn_prefill_cut(ctx, 0, cut);
                const bool ok = llama_memory_seq_rm(llama_get_memory(ctx), 0, cut, -1);
                if (!ok) { printf("%-22s   gen %d: seq_rm refused\n", c.name, (int) cut); continue; }
                got = decode(ctx, cut, N, (int) g_ub);
            } else {
                prefill(ctx, 0, cut);
                prefill(ctx, cut, std::min<llama_pos>(cut + 300, 4000), 7); // other tokens past the cut
                llama_sj_kvarn_prefill_cut(ctx, 0, cut); // [PCR1]
                const bool ok = llama_memory_seq_rm(llama_get_memory(ctx), 0, cut, -1);
                if (!ok) { printf("%-22s   trunc %d: seq_rm refused\n", c.name, (int) cut); continue; }
                got = decode(ctx, cut, N, (int) g_ub);
            }
            const uint32_t B = c.plain ? 0 : cache_of(ctx)->get_sj_kvarn_sealed_end();
            const bool same = got.size() == cold.size() && memcmp(got.data(), cold.data(), got.size()*sizeof(float)) == 0;
            n_diff += !same && in_reach;
            g_out_of_reach += !in_reach;
            printf("%-22s   %s %4d: %s max|d| = %.3g  B=%u%s\n", c.name, mode == 3 ? "gen  " : mode == 2 ? "state" : mode ? "trunc" : "split", (int) cut, same ? "identical" : "DIFFERS  ", max_diff(got, cold), B,
                   in_reach ? "" : "  (saved state ran past the ring reach: inexact by design, info)");
        }
    }
    llama_free(ctx);
    return n_diff;
}

// [#240] exact checkpoints (LLAMA_STATE_SEQ_FLAGS_SJ_EXACT): a checkpoint at any position c (the server puts them
// where the f16 cache does: 4 and 4 + n_ubatch before the prompt end) stores the seal state and the open ring rows.
// After a long answer past c (the ring wraps, B and an idle seal run past c), optionally a full state save/load (the
// RAM/disk prompt cache), the restore + cut at c + re-prefill of [c, N) must give the same logits as the run that
// made the checkpoint ([0, c) then [c, N)), bit for bit. Returns the number of differing variants.
static int run_exact(llama_model * model, const sj_cfg & c, llama_pos N, const std::vector<llama_pos> & cuts) {
    int n_diff = 0;
    llama_context * ctx = make_ctx(model, c);
    require(ctx != nullptr, "context");
    const llama_state_seq_flags fl = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_SJ_EXACT;
    for (llama_pos cut : cuts) {
        llama_memory_clear(llama_get_memory(ctx), true);
        prefill(ctx, 0, cut);
        const std::vector<float> ref = decode(ctx, cut, N, (int) g_ub);
        for (int mode = 0; mode < 2; ++mode) {
            llama_memory_clear(llama_get_memory(ctx), true);
            prefill(ctx, 0, cut);
            std::vector<uint8_t> ck(llama_state_seq_get_size_ext(ctx, 0, fl));
            require(!ck.empty() && llama_state_seq_get_data_ext(ctx, ck.data(), ck.size(), 0, fl) == ck.size(), "exact ckpt save");
            llama_pos ck_end = -1;
            const int32_t ck_B = llama_sj_kvarn_ckpt_sealed_end(ck.data(), ck.size(), &ck_end);
            require(ck_B >= 0 && ck_end == cut && ck_B <= cut, "exact ckpt trailer");
            // a long answer past the cut: prompt rows, then decode steps, then more rows until the ring has wrapped
            const llama_pos e = std::min<llama_pos>(cut + 300, 3900);
            prefill(ctx, cut, e, 7);
            decode(ctx, e, e + 40, 1, 9);
            const llama_pos e2 = std::min<llama_pos>(e + 40 + 1800, 4090);
            decode(ctx, e + 40, e2, (int) g_ub, 11);
            llama_sj_kvarn_compress_idle(ctx, 0, e2);
            if (mode == 1) { // RAM/disk prompt cache: the whole sequence state goes out and comes back
                std::vector<uint8_t> buf(llama_state_seq_get_size(ctx, 0));
                require(llama_state_seq_get_data(ctx, buf.data(), buf.size(), 0) == buf.size(), "state save");
                llama_memory_clear(llama_get_memory(ctx), true);
                require(llama_state_seq_set_data(ctx, buf.data(), buf.size(), 0) == buf.size(), "state load");
            }
            const int32_t B_live = llama_sj_kvarn_seq_sealed_end(ctx, 0);
            require(B_live >= ck_B, "live sealed end covers the checkpoint");
            require(llama_state_seq_set_data_ext(ctx, ck.data(), ck.size(), 0, fl) == ck.size(), "exact ckpt load");
            require(llama_sj_kvarn_seq_sealed_end(ctx, 0) == ck_B, "sealed end restored");
            require(!llama_sj_kvarn_prefill_cut(ctx, 0, cut), "prefill_cut keeps an exact restore");
            require(llama_memory_seq_rm(llama_get_memory(ctx), 0, cut, -1), "seq_rm at the checkpoint");
            const std::vector<float> got = decode(ctx, cut, N, (int) g_ub);
            const bool same = got.size() == ref.size() && memcmp(got.data(), ref.data(), got.size()*sizeof(float)) == 0;
            n_diff += !same;
            printf("%-22s   exact%s %4d: %s max|d| = %.3g  ckpt B=%d (live B %d before restore), %zu bytes\n", c.name,
                    mode ? "+state" : "      ", (int) cut, same ? "identical" : "DIFFERS  ", max_diff(got, ref), ck_B, B_live, ck.size());
        }
    }
    llama_free(ctx);
    return n_diff;
}

// [NOSEAL] #239 investigation switch LLAMA_SJKVARN_NOSEAL (read when a cache is built). Prefill N, then 200 decode steps
// (seals run during both), compare the last logits bit for bit:
//   "none" (whole-context ring, every layer still seals) == the production cache (the ring size alone changes nothing)
//   "all" == a tail long enough that nothing seals (every position read from tq6_0 staging rows either way)
//   "0-3" (every layer of this model, as a range) == "all"
//   "1" (one layer) differs from both production and "all" (only that layer stops sealing)
// Returns the number of failed checks.
static int run_noseal(llama_model * model, const sj_cfg & base, llama_pos N) {
    auto run = [&](const sj_cfg & c, const char * spec) {
        if (spec) { setenv("LLAMA_SJKVARN_NOSEAL", spec, 1); } else { unsetenv("LLAMA_SJKVARN_NOSEAL"); }
        llama_context * ctx = make_ctx(model, c);
        unsetenv("LLAMA_SJKVARN_NOSEAL");
        require(ctx != nullptr, "context");
        decode(ctx, 0, N, (int) g_ub);
        std::vector<float> l = decode(ctx, N, N + 200, 1, 3);
        llama_free(ctx);
        return l;
    };
    sj_cfg fixed = base; fixed.tail = 256; fixed.tail_max = 0;
    sj_cfg open  = base; open.tail = 2560; open.tail_max = 0; // N + 200 < sink + tail: nothing seals
    const auto prod = run(fixed, nullptr), none = run(fixed, "none"), all = run(fixed, "all"), rng = run(fixed, "0-3"),
               one = run(fixed, "1"), unsealed = run(open, nullptr);
    auto same = [](const std::vector<float> & a, const std::vector<float> & b) {
        return a.size() == b.size() && memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0;
    };
    int bad = 0;
    auto check = [&](const char * what, bool ok, float d) {
        printf("%-22s noseal %-34s %s (max|d| = %.3g)\n", base.name, what, ok ? "ok" : "FAILED", d);
        bad += !ok;
    };
    check("none == production", same(none, prod), max_diff(none, prod));
    check("all == nothing sealed", same(all, unsealed), max_diff(all, unsealed));
    check("0-3 == all", same(rng, all), max_diff(rng, all));
    check("1 differs from production", !same(one, prod), max_diff(one, prod));
    check("1 differs from all", !same(one, all), max_diff(one, all));
    return bad;
}

int main(int argc, char ** argv) {
    (void) argc; (void) argv;
    if (getenv("SJ_TEST_UB")) { g_ub = (uint32_t) atoi(getenv("SJ_TEST_UB")); }
    ggml_backend_load_all();
    llama_backend_init();
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level == GGML_LOG_LEVEL_ERROR) { g_last_error = text; fputs(text, stderr); }
    }, nullptr);
    llama_model * model = make_model();
    if (!model) {
        fprintf(stderr, "model init failed\n");
        return 1;
    }
    const llama_pos N = getenv("SJ_TEST_N") ? atoi(getenv("SJ_TEST_N")) : 2347;
    const char * grid_env = getenv("LLAMA_SJKVARN_SEAL_GRID");
    const bool full_grid = grid_env && atoi(grid_env) == 1;
    std::vector<llama_pos> cuts = { 1920, 1280, 1000, 2304 };
    if (!full_grid) { // restore grid: the server cuts SJ-KVaRN prompts on the n_ubatch grid only
        cuts.clear();
        for (llama_pos c = (llama_pos) g_ub; c < N; c += (llama_pos) g_ub) { cuts.push_back(c); }
    }
    if (getenv("SJ_TEST_CUTS")) { // comma list, e.g. grid points only (the server puts SJ-KVaRN checkpoints on the grid)
        cuts.clear();
        for (const char * q = getenv("SJ_TEST_CUTS"); *q; ) { cuts.push_back(atoi(q)); while (*q && *q != ',') ++q; if (*q) ++q; }
    }

    sj_cfg pf16 = { "plain f16", 0, 0, GGML_TYPE_F32 }; pf16.plain = true;
    sj_cfg pq8  = { "plain q8_0", 0, 0, GGML_TYPE_F32 }; pq8.plain = true; pq8.plain_type = GGML_TYPE_Q8_0;
    std::vector<sj_cfg> cfgs = { pf16, pq8 };
    const sj_cfg s44  = { "4/4",  4, 4, GGML_TYPE_F32 };
    const sj_cfg s33t = { "3/3t", 3, 3, GGML_TYPE_I16 };
    const sj_cfg s32t = { "3/2t", 3, 2, GGML_TYPE_I16 };
    static std::vector<std::string> names; names.reserve(64);
    for (const auto & base : { s44, s33t, s32t }) {
        sj_cfg a = base; names.push_back(std::string(base.name) + " noseal"); a.name = names.back().c_str(); a.tail = 2048; a.tail_max = 2560; cfgs.push_back(a);
        sj_cfg b = base; names.push_back(std::string(base.name) + " fixed256"); b.name = names.back().c_str(); b.tail = 256; b.tail_max = 0; cfgs.push_back(b);
        sj_cfg d = base; names.push_back(std::string(base.name) + " adapt256/512"); d.name = names.back().c_str(); d.tail = 256; d.tail_max = 512; cfgs.push_back(d);
    }
    if (getenv("SJ_TEST_FILTER")) {
        std::vector<sj_cfg> f;
        for (auto & c : cfgs) if (strstr(c.name, getenv("SJ_TEST_FILTER"))) f.push_back(c);
        cfgs = f;
    }
    int n_bad_plain = 0, n_bad_sj = 0;
    for (const auto & c : cfgs) {
        if (getenv("SJ_TEST_EXACT_ONLY") || getenv("SJ_TEST_NOSEAL_ONLY")) { break; } // [#240] exact-checkpoint cases only; [NOSEAL] never-seal checks only
        const int d = run_cfg(model, c, N, cuts);
        (c.plain ? n_bad_plain : n_bad_sj) += d;
    }
    // [#240] exact checkpoints at the f16 spots (4 and 4 + n_ubatch before the end, both off the grid); more cuts
    // (e.g. 1000,1024 for a grid line) with SJ_TEST_EXACT_CUTS
    if (!getenv("SJ_TEST_NOSEAL_ONLY")) {
        std::vector<llama_pos> ecuts = { N - 4, N - 4 - (llama_pos) g_ub };
        if (getenv("SJ_TEST_EXACT_CUTS")) {
            ecuts.clear();
            for (const char * q = getenv("SJ_TEST_EXACT_CUTS"); *q; ) { ecuts.push_back(atoi(q)); while (*q && *q != ',') ++q; if (*q) ++q; }
        }
        for (const auto & c : cfgs) {
            if (!c.plain) {
                n_bad_sj += run_exact(model, c, N, ecuts);
            }
        }
    }
    // [NOSEAL] the #239 investigation switch, per body codec
    for (const auto & base : { s44, s33t, s32t }) {
        if (getenv("SJ_TEST_EXACT_ONLY")) { break; } // [#240] exact-checkpoint cases only
        if (getenv("SJ_TEST_FILTER") && !strstr(base.name, getenv("SJ_TEST_FILTER"))) {
            continue;
        }
        n_bad_sj += run_noseal(model, base, N);
    }
    // the plain caches are controls only: their split (in)dependence is the backend's attention kernel (e.g. the CPU
    // f16 flash-attention path is not split-independent), outside SJ-KVaRN
    printf("%s: SJ-KVaRN %d differing; plain controls %d differing (info); %d restored cuts past the ring reach (info) (n_ubatch %u)\n", n_bad_sj ? "FAIL" : "OK", n_bad_sj, n_bad_plain, g_out_of_reach, g_ub);
    llama_model_free(model);
    llama_backend_free();
    return n_bad_sj ? 1 : 0;
}
