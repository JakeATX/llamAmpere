// SJ-KVaRN sequence state save/restore (llama_state_seq_* / llama_state_* on an SJ-KVaRN cache).
//
// Generated 4-layer llama (head 256, 2 KV heads, CPU only, no model file). For every body type the cache
// ships with (4/4 scalar, 3/3 scalar, 3/3 trellis, 3/2 trellis, a tiered 3/2t + 4/4 edge, and the q8_0 / f16
// staging variants) it:
//   1. fills the cache past several sealed groups with the adaptive tail plus a partial tail group,
//   2. saves the sequence state, restores it into a fresh context, and compares the sealed records, the
//      sink rows (staging + fp16 sink) and the exact tail rows byte for byte against the source cache,
//   3. continues both caches with the same tokens through further seals, a rollback above B and the
//      adaptive-tail flush, requiring bit-identical logits and identical records afterwards,
//   4. round-trips the whole-context state and a state file,
//   5. checks that restoring into a cache with a different SJ-KVaRN config (body type, bits, tail, staging,
//      plain cache) fails and leaves the target cache untouched.
//
//   test-sjkvarn-state          all cases

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
    mp.devices = no_devices; // CPU only, also in a CUDA build
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
    bool      plain = false;        // plain f16 cache (no SJ-KVaRN)
};

static llama_context * make_ctx(llama_model * model, const sj_cfg & c) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = 128;
    cp.n_ubatch = 128;
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    cp.n_seq_max = 1;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    if (!c.plain) {
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
static std::vector<float> decode(llama_context * ctx, llama_pos from, llama_pos to, int width, int salt = 0) {
    std::vector<float> last;
    for (llama_pos p = from; p < to;) {
        const int n = std::min<int>(width, to - p);
        llama_batch batch = llama_batch_init(n, 0, 1);
        batch.n_tokens = n;
        for (int j = 0; j < n; ++j) {
            batch.token[j] = token_at(p + j, salt);
            batch.pos[j] = p + j;
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = 0;
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

static std::vector<int32_t> descriptor(llama_kv_cache * cache, llama_pos next_pos) {
    ggml_init_params ip = { 4*ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * desc = cache->build_input_sj_kvarn_desc(ctx);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
    require(buffer != nullptr, "allocate descriptor");
    llama_ubatch ubatch = {};
    ubatch.pos = &next_pos;
    cache->set_input_sj_kvarn_desc(desc, &ubatch);
    std::vector<int32_t> result(GGML_SJKVARN_DESC_N_ENTRIES);
    ggml_backend_tensor_get(desc, result.data(), 0, result.size()*sizeof(int32_t));
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return result;
}

static std::vector<uint8_t> tget(const ggml_tensor * t, size_t off, size_t size) {
    std::vector<uint8_t> v(size);
    if (size > 0) {
        ggml_backend_tensor_get(t, v.data(), off, size);
    }
    return v;
}

// The live contents of an SJ-KVaRN cache read straight from its tensors (independent of the state format):
// sealed records [0, (B-S)/G) per layer, sink rows [0, min(S,N)) (staging + fp16 sink), tail rows [B, N).
struct live_cache {
    uint32_t B = 0, N = 0;
    std::vector<std::vector<uint8_t>> sink_k, sink_v, f16_k, f16_v, tail_k, tail_v;
};

static live_cache snapshot(llama_context * ctx) {
    llama_kv_cache * kv = cache_of(ctx);
    const auto d = descriptor(kv, 0);
    const uint32_t S = d[GGML_SJKVARN_DESC_S], cap = d[GGML_SJKVARN_DESC_CAP];
    live_cache lc;
    lc.B = kv->get_sj_kvarn_sealed_end();
    lc.N = (uint32_t) (llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1);
    const bool f16_sink = d[GGML_SJKVARN_DESC_SINK_TYPE] == GGML_TYPE_F16;
    ggml_init_params ip = { 16*ggml_tensor_overhead(), nullptr, true };
    ggml_context * gctx = ggml_init(ip);
    llama_kv_cache::slot_info si;
    for (uint32_t il = 0; il < n_layer; ++il) {
        const ggml_tensor * k = kv->get_k(gctx, il, 0, si)->view_src;
        const ggml_tensor * v = kv->get_v(gctx, il, 0, si)->view_src;
        const uint32_t n_sink = std::min(S, lc.N);
        lc.sink_k.push_back(tget(k, 0, n_sink*k->nb[1]));
        lc.sink_v.push_back(tget(v, 0, n_sink*v->nb[1]));
        lc.f16_k.push_back(f16_sink ? tget(k, (S + cap)*k->nb[1], (size_t) n_sink*k->ne[0]*2) : std::vector<uint8_t>());
        lc.f16_v.push_back(f16_sink ? tget(v, (S + cap)*v->nb[1], (size_t) n_sink*v->ne[0]*2) : std::vector<uint8_t>());
        std::vector<uint8_t> tk, tv;
        for (uint32_t p = std::max(lc.B, S); p < lc.N; ++p) {
            const uint32_t row = S + (p - S) % cap;
            auto rk = tget(k, row*k->nb[1], k->nb[1]);
            auto rv = tget(v, row*v->nb[1], v->nb[1]);
            tk.insert(tk.end(), rk.begin(), rk.end());
            tv.insert(tv.end(), rv.begin(), rv.end());
        }
        lc.tail_k.push_back(tk);
        lc.tail_v.push_back(tv);
    }
    ggml_free(gctx);
    return lc;
}

// sealed records: the first n_rec*hkv records of each layer's pool; record size from the pool size
static std::vector<std::vector<uint8_t>> records(llama_context * ctx, uint32_t n_groups_pool) {
    llama_kv_cache * kv = cache_of(ctx);
    const auto d = descriptor(kv, 0);
    const uint32_t S = d[GGML_SJKVARN_DESC_S], G = d[GGML_SJKVARN_DESC_G];
    const uint32_t n_rec = (kv->get_sj_kvarn_sealed_end() - S)/G;
    std::vector<std::vector<uint8_t>> res;
    for (uint32_t il = 0; il < n_layer; ++il) {
        const ggml_tensor * body = kv->get_sj_kvarn_body(il);
        const size_t per_group = ggml_nbytes(body) / n_groups_pool; // rec_bytes*hkv
        res.push_back(tget(body, 0, (size_t) n_rec*per_group));
    }
    return res;
}

static std::vector<uint8_t> seq_state(llama_context * ctx, llama_state_seq_flags flags = 0) {
    const size_t n = llama_state_seq_get_size_ext(ctx, 0, flags);
    require(n > 0, "llama_state_seq_get_size > 0");
    std::vector<uint8_t> buf(n);
    require(llama_state_seq_get_data_ext(ctx, buf.data(), n, 0, flags) == n, "llama_state_seq_get_data");
    return buf;
}

static bool same_logits(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0;
}

static int run_case(llama_model * model, const sj_cfg & c, const std::vector<sj_cfg> & mismatches, const char * tmp_dir) {
    llama_context * src = make_ctx(model, c);
    require(src != nullptr, "source context");
    llama_context * dst = nullptr;
    llama_context * dst2 = nullptr;
    int result = 0;
    try {
        llama_kv_cache * kv_src = cache_of(src);
        require(kv_src->is_sj_kvarn(), "SJ-KVaRN cache");
        const uint32_t S = 128, G = 128;
        const uint32_t n_groups_pool = (kv_src->get_size() - S - c.tail)/G;

        // prefill past several groups, then single-token steps so the adaptive tail grows and flushes
        decode(src, 0, 1500, 128);
        decode(src, 1500, 1500 + 37, 1);
        uint32_t B0 = kv_src->get_sj_kvarn_sealed_end();
        const llama_pos N0 = 1537;
        require(B0 > S && (B0 - S) % G == 0, "records sealed before the save");
        require((N0 - B0) % G != 0, "partial tail group at the save");

        const auto blob = seq_state(src);
        const live_cache before = snapshot(src);
        const auto rec_before = records(src, n_groups_pool);

        dst = make_ctx(model, c);
        require(dst != nullptr, "destination context");
        decode(dst, 0, 300, 128, 7); // unrelated content that the restore must replace
        require(llama_state_seq_set_data(dst, blob.data(), blob.size(), 0) == blob.size(), "restore");
        llama_kv_cache * kv_dst = cache_of(dst);
        const live_cache after = snapshot(dst);
        require(after.B == before.B && after.N == before.N, "restored B / N");
        require(records(dst, n_groups_pool) == rec_before, "sealed records byte-identical");
        require(after.sink_k == before.sink_k && after.sink_v == before.sink_v, "sink rows byte-identical");
        require(after.f16_k == before.f16_k && after.f16_v == before.f16_v, "fp16 sink rows byte-identical");
        require(after.tail_k == before.tail_k && after.tail_v == before.tail_v, "tail rows byte-identical");
        require(descriptor(kv_dst, N0) == descriptor(kv_src, N0), "descriptor identical");
        require(seq_state(dst) == blob, "re-saved state byte-identical");
        size_t tail_bytes = 0, rec_bytes = 0;
        for (uint32_t il = 0; il < n_layer; ++il) {
            tail_bytes += before.tail_k[il].size() + before.tail_v[il].size() + before.sink_k[il].size() + before.sink_v[il].size() +
                          before.f16_k[il].size() + before.f16_v[il].size();
            rec_bytes += rec_before[il].size();
        }

        // continue both identically: decode steps (adaptive-tail flush), a rollback above B, a prefill ubatch
        uint32_t flushes = 0;
        uint32_t B_last = B0;
        for (llama_pos p = N0; p < N0 + 700; ++p) {
            const auto a = decode(src, p, p + 1, 1);
            const auto b = decode(dst, p, p + 1, 1);
            require(same_logits(a, b), "identical logits at step " + std::to_string(p));
            require(kv_src->get_sj_kvarn_sealed_end() == kv_dst->get_sj_kvarn_sealed_end(), "same sealed end at step " + std::to_string(p));
            if (kv_src->get_sj_kvarn_sealed_end() != B_last) { ++flushes; B_last = kv_src->get_sj_kvarn_sealed_end(); }
        }
        require(flushes > 0, "the adaptive tail flushed after the restore");
        const llama_pos n1 = N0 + 700;
        const uint32_t B1 = kv_src->get_sj_kvarn_sealed_end();
        require(B1 > S + 2*G, "body sealed past two groups");
        require(!llama_memory_seq_rm(llama_get_memory(dst), 0, S + G + 5, -1) && !llama_memory_seq_rm(llama_get_memory(src), 0, S + G + 5, -1),
                "unaligned rollback into the sealed body below the ring refused on both");
        require(llama_memory_seq_rm(llama_get_memory(src), 0, n1 - 9, -1) && llama_memory_seq_rm(llama_get_memory(dst), 0, n1 - 9, -1),
                "rollback above B on both");
        require(same_logits(decode(src, n1 - 9, n1 + 300, 128, 3), decode(dst, n1 - 9, n1 + 300, 128, 3)), "identical logits after rollback + prefill");
        require(records(src, n_groups_pool) == records(dst, n_groups_pool), "records sealed after the restore identical");
        require(seq_state(src) == seq_state(dst), "states identical after continuing");

        // whole-context state (llama_state_*) and a sequence state file
        {
            const size_t n = llama_state_get_size(src);
            std::vector<uint8_t> full(n);
            require(llama_state_get_data(src, full.data(), n) == n, "llama_state_get_data");
            dst2 = make_ctx(model, c);
            require(llama_state_set_data(dst2, full.data(), n) == n, "llama_state_set_data");
            require(seq_state(dst2) == seq_state(src), "whole-context restore identical");
            std::string fname = c.name;
            for (char & ch : fname) { if (ch == '/' || ch == ' ') ch = '_'; }
            const std::string file = std::string(tmp_dir) + "/sjkvarn-state-" + fname + ".bin";
            std::vector<llama_token> toks = { 1, 2, 3 };
            require(llama_state_seq_save_file(src, file.c_str(), 0, toks.data(), toks.size()) > 0, "llama_state_seq_save_file");
            llama_memory_clear(llama_get_memory(dst2), true);
            std::vector<llama_token> got(8);
            size_t n_got = 0;
            require(llama_state_seq_load_file(dst2, file.c_str(), 0, got.data(), got.size(), &n_got) > 0 && n_got == 3, "llama_state_seq_load_file");
            require(seq_state(dst2) == seq_state(src), "state file restore identical");
            const llama_pos n2 = n1 + 300;
            require(same_logits(decode(src, n2, n2 + 3, 1), decode(dst2, n2, n2 + 3, 1)), "identical logits after the file restore");
            std::remove(file.c_str());
            llama_free(dst2);
            dst2 = nullptr;
        }

        // mismatched configurations are refused and leave the target untouched
        int n_refused = 0;
        for (const auto & m : mismatches) {
            llama_context * other = make_ctx(model, m);
            require(other != nullptr, std::string("context ") + m.name);
            decode(other, 0, 200, 128, 5);
            const auto pos_max = llama_memory_seq_pos_max(llama_get_memory(other), 0);
            const size_t rc = llama_state_seq_set_data(other, blob.data(), blob.size(), 0);
            const bool kept = llama_memory_seq_pos_max(llama_get_memory(other), 0) == pos_max;
            if (!m.plain) {
                require(llama_sj_kvarn_sealed_end(other) == 128, "mismatch keeps the boundary");
            }
            llama_free(other);
            if (n_refused == 0) {
                printf("%-14s   refused into %s: %s", c.name, m.name, g_last_error.c_str());
            }
            require(rc == 0, std::string("restore into ") + m.name + " refused");
            // a plain cache drops the sequence on any failed restore (upstream behaviour); SJ-KVaRN validates first
            require(kept || m.plain, std::string("refused restore into ") + m.name + " leaves the cache");
            ++n_refused;
        }
        // truncation inside the sealed body (no re-quantisation):
        //  - above the intact ring rows any position works: the groups whose staging rows are still in the ring are
        //    reopened down to floor_g(c - tail), and they re-seal later into the same records;
        //  - below them only a group boundary works (whole records dropped), the kept records are bit-identical.
        {
            llama_context * ref = make_ctx(model, c);
            llama_context * cut = make_ctx(model, c);
            decode(ref, 0, 1500, 128); decode(ref, 1500, N0, 1);
            decode(cut, 0, 1500, 128); decode(cut, 1500, N0, 1);
            llama_kv_cache * kv_cut = cache_of(cut);
            require(kv_cut->get_sj_kvarn_sealed_end() == B0, "truncation case starts at the same sealed end");
            const uint32_t cap = descriptor(kv_cut, N0)[GGML_SJKVARN_DESC_CAP];
            const uint32_t lo  = std::max<uint32_t>(S, N0 > (llama_pos) cap ? N0 - cap : 0);
            const uint32_t lo_g = S + G*((lo - S + G - 1)/G);
            auto floor_g = [&](uint32_t p) { return p <= S ? S : S + G*((p - S)/G); };
            const auto rec_ref = records(ref, n_groups_pool);
            auto prefix_same = [&](llama_context * x, uint32_t upto) {
                const auto rx = records(x, n_groups_pool);
                for (uint32_t il = 0; il < n_layer; ++il) {
                    const size_t per = rec_ref[il].size() / ((B0 - S)/G);
                    const size_t n = (size_t) (upto - S)/G*per;
                    if (rx[il].size() < n || memcmp(rx[il].data(), rec_ref[il].data(), n) != 0) return false;
                }
                return true;
            };
            uint32_t n_cut_checks = 0;
            // (1) below the intact rows: unaligned refused, aligned cut drops whole records
            if (lo_g > S + G) {
                const llama_pos c_al = lo_g - G; // a boundary below the intact rows
                require(llama_sj_kvarn_rm_floor(cut, c_al + 5) == c_al && llama_sj_kvarn_rm_floor(cut, c_al) == c_al, "rm_floor below the ring");
                require(!llama_memory_seq_rm(llama_get_memory(cut), 0, c_al + 5, -1), "unaligned cut below the ring refused");
                require(llama_memory_seq_pos_max(llama_get_memory(cut), 0) == N0 - 1 && kv_cut->get_sj_kvarn_sealed_end() == B0,
                        "refused cut leaves the cache");
            }
            // (2) above the intact rows: reopen
            if (lo_g + G < B0) {
                const llama_pos c_re = lo_g + G + 37; // unaligned, inside the body, rows intact
                require(llama_sj_kvarn_rm_floor(cut, c_re) == c_re, "rm_floor keeps a reopenable position");
                require(llama_memory_seq_rm(llama_get_memory(cut), 0, c_re, -1), "unaligned cut inside the reopenable body");
                const uint32_t want = std::min<uint32_t>(B0, std::max<uint32_t>(lo_g, floor_g(c_re > (llama_pos) (S + c.tail) ? c_re - c.tail : S)));
                require(kv_cut->get_sj_kvarn_sealed_end() == want, "reopened down to max(lo_g, floor_g(c - tail))");
                require(llama_memory_seq_pos_max(llama_get_memory(cut), 0) == c_re - 1, "cut moves the frontier");
                require(prefix_same(cut, want), "kept records bit-identical");
                // re-decode the removed suffix; the reopened groups re-seal from the same staging rows
                decode(cut, c_re, N0, 128);
                if (c.tail_max > 0) {
                    require(llama_sj_kvarn_compress_idle(cut, 0, N0) >= 0, "idle seal");
                }
                const uint32_t reached = std::min<uint32_t>(kv_cut->get_sj_kvarn_sealed_end(), floor_g(c_re));
                require(reached > want, "reopened groups were sealed again");
                require(prefix_same(cut, reached), "reopened groups re-seal into the same records");
                n_cut_checks += 2;
                // back to the never-truncated contents for the next check
                llama_free(cut);
                cut = make_ctx(model, c);
                decode(cut, 0, 1500, 128); decode(cut, 1500, N0, 1);
                kv_cut = cache_of(cut);
            }
            // (3) group cut at a boundary below the intact rows
            {
                const llama_pos c_al = lo_g > S + G ? lo_g - G : S + G;
                require(llama_memory_seq_rm(llama_get_memory(cut), 0, c_al, -1), "aligned cut inside the body");
                require(kv_cut->get_sj_kvarn_sealed_end() == (uint32_t) std::min<llama_pos>(c_al, B0) || lo_g <= (uint32_t) c_al,
                        "group cut moves the sealed end to the boundary");
                require(prefix_same(cut, kv_cut->get_sj_kvarn_sealed_end()), "kept records bit-identical after a group cut");
                decode(cut, c_al, N0 + 40, 128);
                ++n_cut_checks;
            }
            // (4) cut at or below the sink drops every record
            require(llama_memory_seq_rm(llama_get_memory(cut), 0, 100, -1) && kv_cut->get_sj_kvarn_sealed_end() == S, "cut below the sink");
            decode(cut, 100, N0 + 40, 128);
            printf("%-14s   body truncation: ring rows intact from %u, %u cut checks passed\n", c.name, lo_g, n_cut_checks + 1);
            llama_free(ref);
            llama_free(cut);
        }

        // and a plain cache's state is refused by this SJ-KVaRN cache
        {
            sj_cfg plain = c; plain.plain = true; plain.name = "plain";
            llama_context * p = make_ctx(model, plain);
            decode(p, 0, 200, 128);
            const auto pb = seq_state(p);
            llama_free(p);
            require(llama_state_seq_set_data(dst, pb.data(), pb.size(), 0) == 0, "plain-cache state refused by the SJ-KVaRN cache");
        }
        printf("%-14s OK: saved B=%u N=%d (%u records/layer/head, %zu B records + %zu B rows, state %zu B = %.0f B/token); "
               "%u flushes after restore, %d mismatches refused\n",
               c.name, B0, N0, (B0 - S)/G, rec_bytes, tail_bytes, blob.size(), (double) blob.size()/N0, flushes, n_refused + 1);
    } catch (const std::exception & e) {
        fprintf(stderr, "%-14s FAIL: %s\n", c.name, e.what());
        result = 1;
    }
    if (dst2) llama_free(dst2);
    if (dst) llama_free(dst);
    llama_free(src);
    return result;
}

int main(int argc, char ** argv) {
    const char * tmp_dir = argc > 1 ? argv[1] : (getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp");
    ggml_backend_load_all();
    llama_backend_init();
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level == GGML_LOG_LEVEL_ERROR) g_last_error = text;
    }, nullptr);
    llama_model * model = make_model();
    if (!model) {
        fprintf(stderr, "model init failed\n");
        return 1;
    }

    const sj_cfg s44  = { "4/4",        4, 4, GGML_TYPE_F32 };
    const sj_cfg s33  = { "3/3",        3, 3, GGML_TYPE_F32 };
    const sj_cfg s33t = { "3/3t",       3, 3, GGML_TYPE_I16 };
    const sj_cfg s32t = { "3/2t",       3, 2, GGML_TYPE_I16 };
    sj_cfg tier = s32t; tier.name = "3/2t+edge4/4"; tier.edge_layers = 1; tier.edge_body = GGML_TYPE_F32;
    sj_cfg q8   = s44; q8.name = "4/4 q8_0"; q8.staging = GGML_TYPE_Q8_0; q8.sink = GGML_TYPE_COUNT;
    sj_cfg f16  = s33t; f16.name = "3/3t f16"; f16.staging = GGML_TYPE_F16; f16.sink = GGML_TYPE_COUNT;
    sj_cfg chunk = s33t; chunk.name = "3/3t chunk1"; chunk.flush_chunk = 1;
    sj_cfg fixed = s44; fixed.name = "4/4 fixed"; fixed.tail_max = 0;

    // config mismatches the 4/4 state must be refused by
    auto with = [](sj_cfg c, const char * name) { c.name = name; return c; };
    sj_cfg m_tail = with(s44, "tail 384");       m_tail.tail = 384;
    sj_cfg m_tmax = with(s44, "tail_max 1024");  m_tmax.tail_max = 1024;
    sj_cfg m_stg  = with(s44, "q8_0 staging");   m_stg.staging = GGML_TYPE_Q8_0; m_stg.sink = GGML_TYPE_COUNT;
    sj_cfg m_plain = with(s44, "plain f16");     m_plain.plain = true;

    const std::vector<std::pair<sj_cfg, std::vector<sj_cfg>>> cases = {
        { s44,   { s33, s33t, s32t, tier, m_tail, m_tmax, m_stg, m_plain } },
        { s33,   { s44, s33t } },
        { s33t,  { s33, s32t, f16 } },
        { s32t,  { s33t, tier } },
        { tier,  { s32t } },
        { q8,    { s44 } },
        { f16,   { s33t } },
        { chunk, { s33t } },
        { fixed, { s44 } },
    };
    int fails = 0;
    for (const auto & c : cases) {
        fails += run_case(model, c.first, c.second, tmp_dir);
    }
    printf("%s: %d/%zu cases passed\n", fails ? "FAIL" : "OK", (int) (cases.size() - fails), cases.size());
    llama_model_free(model);
    llama_backend_free();
    return fails ? 1 : 0;
}
