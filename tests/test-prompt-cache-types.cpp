// Prompt-cache state round trip over the KV cache type matrix.
//
// The server's prompt cache (RAM and disk tier) stores a slot as the bytes of
// llama_state_seq_get_data_ext() for the trunk context and for the MTP drafter
// context, and restores them with llama_state_seq_set_data_ext() into whatever
// slot asks for them - possibly another slot, possibly another process. This test
// does exactly that for every combination of
//
//   trunk K/V pair    (-ctk/-ctv)            x
//   drafter K/V pair  (--spec-draft-type-k/v) x
//   GDN state type    (--cache-type-s)
//
// and requires the next-token logits of the restored contexts to be bit-identical
// to the never-interrupted ones (trunk and drafter), the restored state to serialize
// back to the same bytes, and the recurrent-only checkpoint blob (what the server
// keeps per context checkpoint) to restore and re-serialize unchanged.
//
// Restore target: a fresh pair of contexts (the "server restarted" case) and a
// different sequence id (the "another slot" case).
//
// Block KV streaming (--kv-stream-arena-mib) is excluded on purpose: its state is
// not resident in the KV buffer and llama_state_* refuses it.
// SJ-KVaRN pairs are covered by test-sjkvarn-state.
//
// usage: test-prompt-cache-types -m model-with-mtp.gguf [--full] [--cpu] [--filter SUBSTR] [--ngl N]
//   default: every trunk pair with inherited drafter types and f32 state, every drafter
//            pair on the tq5_0/turbo4 trunk, every state type on four trunk pairs, and
//            the core cross product (4 trunk x 4 drafter x 4 state)
//   --full : the whole cross product trunk pairs x drafter pairs x state types

#include "common.h"
#include "llama.h"
#include "ggml.h"

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct kv_pair {
    ggml_type k;
    ggml_type v;
};

static std::string pair_name(const kv_pair & p) {
    return std::string(ggml_type_name(p.k)) + "/" + ggml_type_name(p.v);
}

// trunk K/V pairs: every KV type same-typed, plus every mixed pair the CUDA build
// instantiates (ggml/cmake/common.cmake: the default GGML_CUDA_FA_QUANTS list, the
// turbo cross product, the tq5/tq6 pairs) and the fused (tq5|tq6|q8)/turbo4 and
// tq6/tq5 pairs
static std::vector<kv_pair> trunk_pairs() {
    const ggml_type same[] = {
        GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1,
        GGML_TYPE_IQ4_NL, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO3_0,
        GGML_TYPE_TURBO4_0, GGML_TYPE_TQ5_0, GGML_TYPE_TQ6_0,
    };
    std::vector<kv_pair> res;
    // turbo2 is V-only: llama_init_from_model rejects it as a K type for every context, so pairs with
    // turbo2 K never reach the prompt cache and are left out (turbo2 V is covered below)
    for (ggml_type t : same) {
        if (t != GGML_TYPE_TURBO2_0) {
            res.push_back({ t, t });
        }
    }
    const kv_pair mixed[] = {
        { GGML_TYPE_F16,  GGML_TYPE_Q8_0 }, { GGML_TYPE_Q8_0, GGML_TYPE_F16 },
        { GGML_TYPE_BF16, GGML_TYPE_Q8_0 }, { GGML_TYPE_Q8_0, GGML_TYPE_BF16 },
    };
    for (const auto & p : mixed) {
        res.push_back(p);
    }
    const ggml_type turbo[] = { GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0 };
    for (ggml_type t : turbo) {
        const bool k_ok = t != GGML_TYPE_TURBO2_0;
        if (k_ok) { res.push_back({ t, GGML_TYPE_Q8_0 }); }
        res.push_back({ GGML_TYPE_Q8_0, t });
        res.push_back({ GGML_TYPE_F16, t });
        if (k_ok) { res.push_back({ t, GGML_TYPE_F16 }); }
        for (ggml_type o : turbo) {
            if (o != t && k_ok) {
                res.push_back({ t, o });
            }
        }
    }
    const ggml_type tq[] = { GGML_TYPE_TQ5_0, GGML_TYPE_TQ6_0 };
    for (ggml_type t : tq) {
        res.push_back({ t, GGML_TYPE_TURBO3_0 });
        res.push_back({ t, GGML_TYPE_TURBO4_0 });
        res.push_back({ t, GGML_TYPE_Q8_0 });
        res.push_back({ t, GGML_TYPE_F16 });
        res.push_back({ GGML_TYPE_Q8_0, t });
        res.push_back({ GGML_TYPE_F16, t });
    }
    res.push_back({ GGML_TYPE_TQ6_0, GGML_TYPE_TQ5_0 });
    return res;
}

static const ggml_type state_types[] = { GGML_TYPE_F32, GGML_TYPE_BF16, GGML_TYPE_F16, GGML_TYPE_Q8_0 };

struct options {
    std::string model;
    bool        full   = false;
    bool        cpu    = false;
    int         ngl    = 999;
    std::string filter;
    int         n_prompt = 300;
};

struct case_result {
    bool        ok = false;
    std::string why;
};

static const int N_SEQ = 2;

static llama_context * make_trunk(llama_model * model, const options & opt, const kv_pair & kv, ggml_type ts) {
    auto cp = llama_context_default_params();
    cp.n_ctx           = 1024;
    cp.n_batch         = 512;
    cp.n_ubatch        = 128; // several ubatches per prompt
    cp.n_seq_max       = N_SEQ;
    cp.kv_unified      = true;
    cp.n_rs_seq        = 4;   // MTP depth 4 rollback ring, as the server sets it
    cp.type_k          = kv.k;
    cp.type_v          = kv.v;
    cp.type_s          = ts;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.n_threads       = 8;
    cp.n_threads_batch = 8;
    cp.no_perf         = true;
    cp.offload_kqv     = !opt.cpu;
    return llama_init_from_model(model, cp);
}

static llama_context * make_draft(llama_model * model, const options & opt, llama_context * ctx_tgt, const kv_pair & kv, ggml_type ts) {
    auto cp = llama_context_default_params();
    cp.ctx_type        = LLAMA_CONTEXT_TYPE_MTP;
    cp.ctx_other       = ctx_tgt;
    cp.n_ctx           = llama_n_ctx(ctx_tgt);
    cp.n_batch         = 512;
    cp.n_ubatch        = 128;
    cp.n_seq_max       = N_SEQ;
    cp.kv_unified      = true;
    cp.n_rs_seq        = 0;
    cp.type_k          = kv.k;
    cp.type_v          = kv.v;
    cp.type_s          = ts;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.n_threads       = 8;
    cp.n_threads_batch = 8;
    cp.no_perf         = true;
    cp.offload_kqv     = !opt.cpu;
    return llama_init_from_model(model, cp);
}

static bool decode_trunk(llama_context * ctx, const std::vector<llama_token> & toks, llama_pos p0, llama_seq_id seq) {
    llama_batch b = llama_batch_init((int32_t) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        common_batch_add(b, toks[i], p0 + (llama_pos) i, { seq }, i + 1 == toks.size());
    }
    const bool ok = llama_decode(ctx, b) == 0;
    llama_batch_free(b);
    return ok;
}

// the MTP head takes (token[p+1], hidden[p]) at position p; the hidden rows here are
// deterministic pseudo-random: the round trip does not need meaningful ones
static bool decode_draft(llama_context * ctx, int32_t n_embd, const std::vector<llama_token> & toks, llama_pos p0, llama_seq_id seq) {
    const int32_t n = (int32_t) toks.size();
    llama_batch b = llama_batch_init(n, n_embd, 1);
    b.token = (llama_token *) malloc(sizeof(llama_token) * n);
    std::mt19937 rng(1234 + p0);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (int32_t i = 0; i < n; ++i) {
        b.token[i]     = toks[i];
        b.pos[i]       = p0 + i;
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = seq;
        b.logits[i]    = i + 1 == n;
        for (int32_t j = 0; j < n_embd; ++j) {
            b.embd[(size_t) i * n_embd + j] = nd(rng);
        }
    }
    b.n_tokens = n;
    const bool ok = llama_decode(ctx, b) == 0;
    free(b.token);
    b.token = nullptr;
    llama_batch_free(b);
    return ok;
}

static std::vector<uint8_t> save_seq(llama_context * ctx, llama_seq_id seq, llama_state_seq_flags flags = LLAMA_STATE_SEQ_FLAGS_NONE) {
    const size_t n = llama_state_seq_get_size_ext(ctx, seq, flags);
    std::vector<uint8_t> buf(n);
    if (n == 0) {
        return buf;
    }
    const size_t w = llama_state_seq_get_data_ext(ctx, buf.data(), n, seq, flags);
    if (w != n) {
        buf.clear();
    }
    return buf;
}

// byte comparison with a short description of the first difference
static bool same_bytes(const std::vector<uint8_t> & got, const std::vector<uint8_t> & want, const char * what, std::string & why) {
    if (got == want) {
        return true;
    }
    size_t first = SIZE_MAX, n_diff = 0, last = 0;
    for (size_t i = 0; i < std::min(got.size(), want.size()); ++i) {
        if (got[i] != want[i]) {
            if (first == SIZE_MAX) { first = i; }
            last = i;
            ++n_diff;
        }
    }
    char buf[256];
    snprintf(buf, sizeof(buf), "re-saved %s state bytes differ: %zu of %zu bytes, first at %zu, last at %zu (sizes %zu/%zu)",
             what, n_diff, want.size(), first, last, got.size(), want.size());
    why = buf;
    return false;
}

static std::vector<float> last_logits(llama_context * ctx, const llama_model * model) {
    const float * l = llama_get_logits_ith(ctx, -1);
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    return l ? std::vector<float>(l, l + n_vocab) : std::vector<float>();
}

static bool same_bits(const std::vector<float> & a, const std::vector<float> & b, double & max_diff) {
    max_diff = 0.0;
    if (a.size() != b.size() || a.empty()) {
        max_diff = INFINITY;
        return false;
    }
    bool same = memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
    for (size_t i = 0; i < a.size(); ++i) {
        max_diff = std::max(max_diff, (double) std::fabs(a[i] - b[i]));
    }
    return same;
}

static case_result run_case(llama_model * model, const options & opt, const kv_pair & tkv, const kv_pair & dkv, ggml_type ts,
                            const std::vector<llama_token> & prompt, llama_token next) {
    case_result r;
    const int32_t n_embd = llama_model_n_embd(model);
    const llama_pos n_p = (llama_pos) prompt.size();

    // drafter input is token[p+1] at p: feed prompt[1..n] at positions 0..n-1, then `next` at n
    std::vector<llama_token> dtoks(prompt.begin() + 1, prompt.end());
    dtoks.push_back(next);

    // reference: uninterrupted
    llama_context * a_t = make_trunk(model, opt, tkv, ts);
    if (!a_t) { r.why = "trunk context creation failed"; return r; }
    llama_context * a_d = make_draft(model, opt, a_t, dkv, ts);
    if (!a_d) { llama_free(a_t); r.why = "drafter context creation failed"; return r; }

    std::vector<uint8_t> st_t, st_d;
    std::vector<uint8_t> ck_t, ck_d; // context checkpoints (recurrent part only), as the server stores them
    std::vector<float> ref_t, ref_d;
    bool ok = decode_trunk(a_t, prompt, 0, 0) && decode_draft(a_d, n_embd, std::vector<llama_token>(dtoks.begin(), dtoks.end() - 1), 0, 0);
    if (!ok) { r.why = "prompt decode failed"; }
    if (ok) {
        st_t = save_seq(a_t, 0);
        st_d = save_seq(a_d, 0);
        ck_t = save_seq(a_t, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        ck_d = save_seq(a_d, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        if (st_t.empty()) { ok = false; r.why = "trunk state save returned 0 bytes"; }
        else if (st_d.empty()) { ok = false; r.why = "drafter state save returned 0 bytes"; }
    }
    if (ok) {
        ok = decode_trunk(a_t, { next }, n_p, 0) && decode_draft(a_d, n_embd, { next }, n_p - 1, 0);
        if (!ok) { r.why = "reference continuation decode failed"; }
        ref_t = last_logits(a_t, model);
        ref_d = last_logits(a_d, model);
    }
    llama_free(a_d);
    llama_free(a_t);
    if (!ok) { return r; }

    // restored: fresh contexts, other sequence id
    const llama_seq_id dst = 1;
    llama_context * b_t = make_trunk(model, opt, tkv, ts);
    llama_context * b_d = b_t ? make_draft(model, opt, b_t, dkv, ts) : nullptr;
    if (!b_t || !b_d) {
        llama_free(b_d); llama_free(b_t);
        r.why = "context re-creation failed";
        return r;
    }
    std::vector<float> got_t, got_d;
    // byte round trip on the saving sequence id (the blob records the cell's seq id, so a re-save from another
    // sequence differs from the original in exactly those fields): restore into seq 0, re-save, compare, clear
    if (llama_state_seq_set_data_ext(b_t, st_t.data(), st_t.size(), 0, LLAMA_STATE_SEQ_FLAGS_NONE) != st_t.size() ||
        llama_state_seq_set_data_ext(b_d, st_d.data(), st_d.size(), 0, LLAMA_STATE_SEQ_FLAGS_NONE) != st_d.size()) {
        ok = false; r.why = "state restore into seq 0 failed";
    } else if (!same_bytes(save_seq(b_t, 0), st_t, "trunk", r.why) || !same_bytes(save_seq(b_d, 0), st_d, "drafter", r.why)) {
        ok = false;
    } else if (!ck_t.empty() && llama_state_seq_set_data_ext(b_t, ck_t.data(), ck_t.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != ck_t.size()) {
        ok = false; r.why = "trunk checkpoint restore into seq 0 failed";
    } else if (!same_bytes(save_seq(b_t, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY), ck_t, "trunk checkpoint", r.why)) {
        ok = false;
    }
    llama_memory_seq_rm(llama_get_memory(b_t), 0, -1, -1);
    llama_memory_seq_rm(llama_get_memory(b_d), 0, -1, -1);
    if (!ok) {
        // reported below
    } else if (llama_state_seq_set_data_ext(b_t, st_t.data(), st_t.size(), dst, LLAMA_STATE_SEQ_FLAGS_NONE) != st_t.size()) {
        ok = false; r.why = "trunk state restore failed";
    } else if (llama_state_seq_set_data_ext(b_d, st_d.data(), st_d.size(), dst, LLAMA_STATE_SEQ_FLAGS_NONE) != st_d.size()) {
        ok = false; r.why = "drafter state restore failed";
    } else if (llama_memory_seq_pos_max(llama_get_memory(b_t), dst) != n_p - 1) {
        ok = false; r.why = "restored trunk pos_max mismatch";
    } else {
        const size_t sz_t = llama_state_seq_get_size_ext(b_t, dst, LLAMA_STATE_SEQ_FLAGS_NONE);
        const size_t sz_d = llama_state_seq_get_size_ext(b_d, dst, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (sz_t != st_t.size() || sz_d != st_d.size()) {
            ok = false; r.why = "re-saved state size differs";
        } else if (!ck_t.empty() && llama_state_seq_set_data_ext(b_t, ck_t.data(), ck_t.size(), dst, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != ck_t.size()) {
            ok = false; r.why = "trunk checkpoint restore failed";
        } else if (!ck_d.empty() && llama_state_seq_set_data_ext(b_d, ck_d.data(), ck_d.size(), dst, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != ck_d.size()) {
            ok = false; r.why = "drafter checkpoint restore failed";
        } else if (!decode_trunk(b_t, { next }, n_p, dst) || !decode_draft(b_d, n_embd, { next }, n_p - 1, dst)) {
            ok = false; r.why = "restored continuation decode failed";
        } else {
            got_t = last_logits(b_t, model);
            got_d = last_logits(b_d, model);
        }
    }
    llama_free(b_d);
    llama_free(b_t);
    if (!ok) { return r; }

    double dt = 0, dd = 0;
    const bool st = same_bits(ref_t, got_t, dt);
    const bool sd = same_bits(ref_d, got_d, dd);
    if (!st || !sd) {
        char buf[256];
        snprintf(buf, sizeof(buf), "logits differ after restore: trunk %s (max |d| %.3g), drafter %s (max |d| %.3g)",
                 st ? "identical" : "DIFF", dt, sd ? "identical" : "DIFF", dd);
        r.why = buf;
        return r;
    }
    r.ok = true;
    char buf[192];
    snprintf(buf, sizeof(buf), "trunk %.2f MiB, drafter %.2f MiB, checkpoint %.2f MiB", st_t.size() / 1048576.0, st_d.size() / 1048576.0, ck_t.size() / 1048576.0);
    r.why = buf;
    return r;
}

int main(int argc, char ** argv) {
    options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if ((a == "-m" || a == "--model") && i + 1 < argc) { opt.model = argv[++i]; }
        else if (a == "--full")                           { opt.full = true; }
        else if (a == "--cpu")                            { opt.cpu = true; opt.ngl = 0; }
        else if (a == "--ngl" && i + 1 < argc)            { opt.ngl = atoi(argv[++i]); }
        else if (a == "--filter" && i + 1 < argc)         { opt.filter = argv[++i]; }
        else if (a == "--n-prompt" && i + 1 < argc)       { opt.n_prompt = atoi(argv[++i]); }
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (opt.model.empty()) {
        fprintf(stderr, "usage: %s -m model-with-mtp.gguf [--full] [--cpu] [--filter SUBSTR]\n", argv[0]);
        return 2;
    }

    llama_backend_init();
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_ERROR) { fputs(text, stderr); }
    }, nullptr);

    auto mp = llama_model_default_params();
    mp.n_gpu_layers = opt.ngl;
    mp.load_mtp     = true;
    llama_model * model = llama_model_load_from_file(opt.model.c_str(), mp);
    if (!model) {
        fprintf(stderr, "failed to load %s\n", opt.model.c_str());
        return 1;
    }
    if (llama_model_n_layer_nextn(model) <= 0) {
        fprintf(stderr, "%s has no MTP layers\n", opt.model.c_str());
        return 1;
    }

    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::mt19937 rng(42);
    std::vector<llama_token> prompt(opt.n_prompt);
    for (auto & t : prompt) { t = (llama_token) (rng() % std::min<int32_t>(n_vocab, 100000)); }
    const llama_token next = (llama_token) (rng() % std::min<int32_t>(n_vocab, 100000));

    const auto trunks = trunk_pairs();
    std::vector<kv_pair> drafts = trunks;

    struct combo { kv_pair t; kv_pair d; ggml_type s; };
    std::vector<combo> combos;
    auto add = [&](const kv_pair & t, const kv_pair & d, ggml_type s) {
        for (const auto & c : combos) {
            if (c.t.k == t.k && c.t.v == t.v && c.d.k == d.k && c.d.v == d.v && c.s == s) { return; }
        }
        combos.push_back({ t, d, s });
    };
    if (opt.full) {
        for (const auto & t : trunks) for (const auto & d : drafts) for (ggml_type s : state_types) add(t, d, s);
    } else {
        const kv_pair prod { GGML_TYPE_TQ5_0, GGML_TYPE_TURBO4_0 };
        for (const auto & t : trunks) { add(t, t, GGML_TYPE_F32); }
        for (const auto & d : drafts) { add(prod, d, GGML_TYPE_F32); }
        const kv_pair core[] = { { GGML_TYPE_F16, GGML_TYPE_F16 }, { GGML_TYPE_Q8_0, GGML_TYPE_Q8_0 }, prod,
                                 { GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0 } };
        for (const auto & t : core) for (const auto & d : core) for (ggml_type s : state_types) add(t, d, s);
        for (const auto & t : trunks) for (ggml_type s : state_types) if (t.k == t.v) add(t, t, s);
    }

    int n_run = 0, n_fail = 0;
    for (const auto & c : combos) {
        char name[160];
        snprintf(name, sizeof(name), "trunk %-17s drafter %-17s state %-5s", pair_name(c.t).c_str(), pair_name(c.d).c_str(), ggml_type_name(c.s));
        if (!opt.filter.empty() && std::string(name).find(opt.filter) == std::string::npos) {
            continue;
        }
        const case_result r = run_case(model, opt, c.t, c.d, c.s, prompt, next);
        ++n_run;
        n_fail += !r.ok;
        printf("%s %s  %s\n", r.ok ? "PASS" : "FAIL", name, r.why.c_str());
        fflush(stdout);
    }
    printf("%d cases, %d failed (%s)\n", n_run, n_fail, opt.cpu ? "CPU" : "GPU");

    llama_model_free(model);
    llama_backend_free();
    return n_fail == 0 ? 0 : 1;
}
