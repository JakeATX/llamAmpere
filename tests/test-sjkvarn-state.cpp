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
// A generated qwen2vl (M-RoPE) model checks image prompts, whose tokens share a position (run_mrope).
//
//   test-sjkvarn-state          all cases
//   test-sjkvarn-state --scalar-only [tmp_dir]   native scalar 4/4 state and multi-sequence checks
//   test-sjkvarn-state --resident [tmp_dir]   CUDA split-body state round trips
//   test-sjkvarn-state --device-bytes   CPU device-snapshot packed-byte regression
//   test-sjkvarn-state --exact-sink   CPU exact checkpoints below and across the sink boundary
//   test-sjkvarn-state --compact [tmp_dir]   CUDA compact attention on saved and truncated SJ state

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-sjkvarn.h"
#include "ggml-backend.h"
#include "llama.h"
#include "llama-cpp.h"

#include "gguf.h"

#include "../src/llama-kv-cache.h"
#include "../src/llama-context.h"
#include "../src/llama-arch.h"
#include "../src/llama-model-saver.h"
#include "../src/llama-model.h"
#include "../src/llama-ple-prefetch.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

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

static llama_model * make_model(bool resident = false, llm_arch arch = LLM_ARCH_LLAMA) {
    gguf_context * meta = gguf_init_empty();
    {
        llama_model_saver ms(arch, meta);
        ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,         llm_arch_name(arch));
        ms.add_kv(LLM_KV_VOCAB_SIZE,                   n_vocab);
        ms.add_kv(LLM_KV_CONTEXT_LENGTH,               uint32_t(8192));
        ms.add_kv(LLM_KV_EMBEDDING_LENGTH,             n_embd);
        ms.add_kv(LLM_KV_BLOCK_COUNT,                  n_layer);
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH,          uint32_t(768));
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,         uint32_t(2));
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV,      uint32_t(2));
        ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,  1e-5f);
        if (arch == LLM_ARCH_QWEN2VL) {
            const int32_t sections[4] = { 32, 48, 48, 0 }; // M-RoPE: time, y, x
            gguf_set_arr_data(meta, (std::string(llm_arch_name(arch)) + ".rope.dimension_sections").c_str(), GGUF_TYPE_INT32, sections, 4);
        }
        std::vector<std::string> tokens(n_vocab);
        for (uint32_t i = 0; i < n_vocab; ++i) tokens[i] = "tok_" + std::to_string(i);
        ms.add_kv(LLM_KV_TOKENIZER_MODEL,  "test");
        ms.add_kv(LLM_KV_TOKENIZER_LIST,   tokens);
        ms.add_kv(LLM_KV_TOKENIZER_SCORES, std::vector<float>(n_vocab, 0.0f));
    }
    llama_model_params mp = llama_model_default_params();
    ggml_backend_dev_t no_devices[] = { nullptr };
    if (resident) {
        mp.n_gpu_layers = 99;
    } else if (!getenv("SJ_TEST_GPU")) {
        mp.devices = no_devices; // CPU only, also in a CUDA build (SJ_TEST_GPU=1: default devices, all layers offloaded)
    }
    llama_model * model = llama_model_init_from_user(meta, tensor_data, nullptr, mp);
    gguf_free(meta);
    if (model && resident) {
        // The cache reads QSA eligibility from these per-layer values; the LLAMA graph ignores them.
        model->hparams.dsv4_compress_ratios.fill(0);
        model->hparams.dsv4_compress_ratios[0] = 16;
        model->hparams.dsv4_compress_ratios[2] = 16;
    }
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
    const char * transform = nullptr; // EXPERIMENTAL WUSH-KV transform file (--sjkvarn-transform)
    uint32_t resident = 0;
};

static llama_context * make_ctx(llama_model * model, const sj_cfg & c, uint32_t n_seq = 1, bool unified = false, uint32_t n_ctx = 4096) {
    llama_context_params cp = llama_context_default_params();
    // Several sequences each get the same position budget as a single-sequence context.
    cp.n_ctx = unified ? n_ctx : n_ctx*n_seq;
    cp.kv_unified = unified;
    cp.n_batch = 128;
    cp.n_ubatch = 128;
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    cp.n_seq_max = n_seq;
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
        cp.sj_kvarn_transform = c.transform;
        cp.kv_gpu_tokens = c.resident;
    }
    llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx && c.resident && !c.plain) {
        try {
            auto * cache = dynamic_cast<llama_kv_cache *>(llama_get_memory(ctx));
            require(cache != nullptr, "resident test uses llama_kv_cache");
            uint32_t n_split = 0, n_full = 0;
            for (uint32_t il = 0; il < n_layer; ++il) {
                require(model->hparams.n_embd_head_k(il) == 256, "resident test has 256-channel heads");
                const ggml_tensor * body = cache->get_sj_kvarn_body(il);
                require(body && body->buffer, "resident body allocated");
                const auto buft = ggml_backend_buffer_get_type(body->buffer);
                const bool split = strstr(ggml_backend_buft_name(buft), "CUDA_QSA_Resident_") != nullptr;
                require(split == (model->hparams.dsv4_compress_ratios[il] > 0), "only eligible QSA layers use split bodies");
                require(ggml_backend_buft_get_device(buft) == model->dev_layer(il), "body stays on the model layer device");
                require(ggml_backend_dev_type(model->dev_layer(il)) == GGML_BACKEND_DEVICE_TYPE_GPU, "resident state test runs on GPU");
                if (split) {
                    ++n_split;
                } else {
                    ++n_full;
                }
            }
            require(n_split == 2 && n_full == 2, "resident state test has two split and two full bodies");
        } catch (...) {
            llama_free(ctx);
            throw;
        }
    }
    return ctx;
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

static std::vector<int32_t> descriptor(llama_kv_cache * cache, llama_pos next_pos) {
    ggml_init_params ip = { 4*ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    llama_ubatch ubatch = {};
    ubatch.pos = &next_pos;
    ubatch.n_seqs_unq = 1;
    ggml_tensor * desc = cache->build_input_sj_kvarn_desc(ctx, ubatch);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
    require(buffer != nullptr, "allocate descriptor");
    llama_kv_cache::slot_info sinfo;
    sinfo.s0 = sinfo.s1 = 0;
    sinfo.strm  = { 0 };
    sinfo.idxs  = { { 0 } };
    cache->set_input_sj_kvarn_desc(desc, &ubatch, sinfo);
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
    si.s0 = si.s1 = 0;
    si.strm = { 0 };
    si.idxs = { { 0 } };
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

static void device_bytes_case(llama_model * model) {
    const sj_cfg cfg = { "4/4 device bytes", 4, 4, GGML_TYPE_F32 };
    llama_context * ctx = make_ctx(model, cfg);
    require(ctx != nullptr, "device-byte context");
    try {
        decode(ctx, 0, 129, 128);
        const auto host = seq_state(ctx);
        for (int reuse = 0; reuse < 2; ++reuse) {
            const auto device = seq_state(ctx, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
            decode(ctx, 129, 146, 17, 9);
            require(llama_state_seq_set_data_ext(ctx, device.data(), device.size(), 0,
                    LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) == device.size(), "restore packed-byte snapshot");
            require(seq_state(ctx) == host, "packed-byte snapshot matches host state");
            llama_memory_clear(llama_get_memory(ctx), true);
            require(llama_state_seq_set_data_ext(ctx, device.data(), device.size(), 0,
                    LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) == device.size(), "restore packed-byte snapshot after clear");
            require(seq_state(ctx) == host, "packed-byte snapshot survives clear");
        }
    } catch (...) {
        llama_free(ctx);
        throw;
    }
    llama_free(ctx);
}

static bool same_logits(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0;
}

static void exact_sink_case(llama_model * model) {
    const sj_cfg cfg = { "4/4 exact sink", 4, 4, GGML_TYPE_F32 };
    const auto flags = LLAMA_STATE_SEQ_FLAGS_SJ_EXACT;
    for (uint32_t n : { 1u, 33u, 127u, 128u, 129u }) {
        llama_context_ptr ctx(make_ctx(model, cfg));
        require(ctx != nullptr, "exact-sink context");
        decode(ctx.get(), 0, n, 128);
        const auto before = seq_state(ctx.get());
        const auto checkpoint = seq_state(ctx.get(), flags);
        const size_t off = sizeof(uint32_t) + sizeof(llama_seq_id);
        uint32_t magic;
        require(checkpoint.size() >= off + 8*sizeof(uint32_t), "exact checkpoint header fits");
        std::memcpy(&magic, checkpoint.data() + off, sizeof(magic));
        require(magic == LLAMA_SJ_KVARN_CKPT_MAGIC, "exact checkpoint section found");
        auto invalid = checkpoint;
        const uint32_t pending = std::max(n, 128u) + 1;
        std::memcpy(invalid.data() + off + 5*sizeof(uint32_t), &pending, sizeof(pending));
        g_last_error.clear();
        require(llama_state_seq_set_data_ext(ctx.get(), invalid.data(), invalid.size(), 0, flags) == 0,
                "pending seal beyond sink and sequence is refused");
        require(g_last_error.find("corrupt section") != std::string::npos, "invalid pending seal rejected by header validation");
        require(seq_state(ctx.get()) == before, "invalid pending seal leaves the cache unchanged");
        const auto expected = decode(ctx.get(), n, n + 4, 1, 7);
        for (int reuse = 0; reuse < 2; ++reuse) {
            require(llama_state_seq_set_data_ext(ctx.get(), checkpoint.data(), checkpoint.size(), 0, flags) == checkpoint.size(),
                    "restore exact checkpoint at " + std::to_string(n));
            require(llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, n, -1), "trim restored sequence");
            require(seq_state(ctx.get()) == before, "exact checkpoint restores prefix bytes");
            require(same_logits(decode(ctx.get(), n, n + 4, 1, 7), expected), "exact checkpoint continuation is bit-identical");
        }
        printf("OK: exact checkpoint n=%u restored twice with identical bytes and logits; invalid pending seal refused\n", n);
    }
}

#include "test-sjkvarn-compact-state.h"

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
        if (c.resident) {
            const auto device_blob = seq_state(src, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
            decode(src, N0, N0 + 256, 128, 9);
            require(llama_state_seq_set_data_ext(src, device_blob.data(), device_blob.size(), 0,
                    LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) == device_blob.size(), "on-device snapshot restores after further seals");
            require(seq_state(src) == blob, "on-device restore produces identical host state bytes");
            require(records(src, n_groups_pool) == rec_before, "on-device restore preserves split and full body bytes");
            const auto device_blob_reused = seq_state(src, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
            llama_memory_clear(llama_get_memory(src), true);
            require(llama_state_seq_set_data_ext(src, device_blob_reused.data(), device_blob_reused.size(), 0,
                    LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) == device_blob_reused.size(), "reused on-device snapshot restores after clear");
            require(seq_state(src) == blob, "reused on-device snapshot keeps identical host state bytes");
            require(records(src, n_groups_pool) == rec_before, "reused on-device snapshot keeps sealed body bytes");
        }

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

        if (std::string(c.name) == "4/4") {
            // The synthetic llama has one sequence per cell and no cell extensions.
            size_t off = 4*sizeof(uint32_t) + (size_t) N0*(sizeof(llama_pos) + sizeof(uint32_t) + sizeof(llama_seq_id));
            auto read_u32 = [&]() {
                require(off <= blob.size() && blob.size() - off >= sizeof(uint32_t), "state header fits");
                uint32_t value;
                std::memcpy(&value, blob.data() + off, sizeof(value));
                off += sizeof(value);
                return value;
            };
            require(read_u32() == 0x564b4a53u && read_u32() == 2, "SJ-KVaRN v2 state header");
            const uint32_t n_cfg = read_u32();
            off += (size_t) n_cfg*sizeof(int64_t);
            const uint32_t n_info = read_u32();
            off += (size_t) n_info*sizeof(int64_t);
            require(off <= blob.size() && blob.size() - off >= 5*sizeof(uint32_t), "SJ-KVaRN state fields fit");
            const struct { uint32_t field, value; const char * name; } corruptions[] = {
                { 1, S + 1, "misaligned previous sealed end" },
                { 2, kv_dst->get_size() + 1, "visible end beyond cache" },
                { 3, 2, "invalid draining flag" },
            };
            for (const auto & corruption : corruptions) {
                auto invalid = blob;
                std::memcpy(invalid.data() + off + corruption.field*sizeof(uint32_t), &corruption.value, sizeof(uint32_t));
                g_last_error.clear();
                require(llama_state_seq_set_data(dst, invalid.data(), invalid.size(), 0) == 0, std::string(corruption.name) + " refused");
                require(g_last_error.find("corrupt state") != std::string::npos, std::string(corruption.name) + " rejected by header validation");
                require(seq_state(dst) == blob, std::string(corruption.name) + " leaves the cache unchanged");
            }
            printf("%-14s OK: three corrupt state headers refused without changing the cache\n", c.name);
        }

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
                require(llama_sj_kvarn_rm_floor(cut, 0, c_al + 5) == c_al && llama_sj_kvarn_rm_floor(cut, 0, c_al) == c_al, "rm_floor below the ring");
                require(!llama_memory_seq_rm(llama_get_memory(cut), 0, c_al + 5, -1), "unaligned cut below the ring refused");
                require(llama_memory_seq_pos_max(llama_get_memory(cut), 0) == N0 - 1 && kv_cut->get_sj_kvarn_sealed_end() == B0,
                        "refused cut leaves the cache");
            }
            // (2) above the intact rows: reopen
            if (lo_g + G < B0) {
                const llama_pos c_re = lo_g + G + 37; // unaligned, inside the body, rows intact
                require(llama_sj_kvarn_rm_floor(cut, 0, c_re) == c_re, "rm_floor keeps a reopenable position");
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

// one span of a sequence decoded inside a mixed batch
struct span { llama_seq_id seq; llama_pos from, to; int salt; };

// decode several sequences together: every batch takes up to `width` positions of every span (mixed ubatches);
// returns each span's logits at its last position
// mixed = false: the same rounds, but every span in its own llama_decode (no batch mixing)
static std::vector<std::vector<float>> decode_mixed(llama_context * ctx, std::vector<span> spans, int width, bool mixed = true) {
    std::vector<std::vector<float>> last(spans.size());
    if (!mixed) {
        for (bool any = true; any;) {
            any = false;
            for (size_t i = 0; i < spans.size(); ++i) {
                auto & sp = spans[i];
                const int m = std::min<int>(width, sp.to - sp.from);
                if (m > 0) {
                    last[i] = decode(ctx, sp.from, sp.from + m, m, sp.salt, sp.seq);
                    sp.from += m;
                    any = true;
                }
            }
        }
        return last;
    }
    for (;;) {
        llama_batch batch = llama_batch_init(width*(int) spans.size(), 0, 1);
        std::vector<int> out(spans.size(), -1);
        int n = 0;
        for (size_t i = 0; i < spans.size(); ++i) {
            auto & sp = spans[i];
            const int m = std::min<int>(width, sp.to - sp.from);
            for (int j = 0; j < m; ++j) {
                batch.token[n] = token_at(sp.from + j, sp.salt);
                batch.pos[n] = sp.from + j;
                batch.n_seq_id[n] = 1;
                batch.seq_id[n][0] = sp.seq;
                batch.logits[n] = j + 1 == m;
                if (j + 1 == m) { out[i] = n; }
                ++n;
            }
            sp.from += m;
        }
        batch.n_tokens = n;
        if (n == 0) { llama_batch_free(batch); break; }
        const int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        require(rc == 0, "mixed decode");
        for (size_t i = 0; i < spans.size(); ++i) {
            if (out[i] >= 0) {
                const float * l = llama_get_logits_ith(ctx, out[i]);
                require(l != nullptr, "mixed logits present");
                last[i].assign(l, l + n_vocab);
            }
        }
    }
    return last;
}

static std::vector<uint8_t> seq_state_of(llama_context * ctx, llama_seq_id seq) {
    const size_t n = llama_state_seq_get_size_ext(ctx, seq, 0);
    require(n > 0, "llama_state_seq_get_size > 0");
    std::vector<uint8_t> buf(n);
    require(llama_state_seq_get_data_ext(ctx, buf.data(), n, seq, 0) == n, "llama_state_seq_get_data");
    return buf;
}

static float max_diff(const std::vector<float> & a, const std::vector<float> & b) {
    float m = a.size() == b.size() ? 0.0f : 1e30f;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) { m = std::max(m, std::fabs(a[i] - b[i])); }
    return m;
}


// control for the mixed-ubatch tolerance: the same mixed-vs-alone comparison on a plain f16 cache (no SJ-KVaRN)
static void plain_mixed_control(llama_model * model) {
    sj_cfg pc = { "plain f16", 0, 0, GGML_TYPE_F32 };
    pc.plain = true;
    llama_context * mx = make_ctx(model, pc, 2, false);
    llama_context * r1 = make_ctx(model, pc);
    const auto rm = decode_mixed(mx, { { 0, 0, 704, 0 }, { 1, 0, 704, 3 } }, 64);
    const auto a = decode(r1, 0, 704, 64, 3);
    printf("control: plain f16 cache, mixed ubatch vs alone: max|d| = %g\n", max_diff(rm[1], a));
    llama_free(mx); llama_free(r1);
}

// Several sequences, one stream each (paged record pool; `unified` = --kv-unified shared pool). Every sequence must
// give the logits a single-sequence context gives for the same tokens, through mixed ubatches, seals, a truncation
// into the sealed body, a per-sequence state save/restore into another sequence's stream, and seq_cp.
static int run_multi(llama_model * model, const sj_cfg & c, bool unified) {
    const std::string name = std::string(c.name) + (unified ? " x2 unified" : " x2");
    llama_context * ref0 = make_ctx(model, c);
    llama_context * ref1 = make_ctx(model, c);
    llama_context * mc   = make_ctx(model, c, 2, unified);
    llama_context * mc2  = make_ctx(model, c, 2, unified);
    int result = 0;
    try {
        require(ref0 && ref1 && mc && mc2, "contexts");
        require(cache_of(mc)->is_sj_kvarn() && cache_of(mc)->get_n_stream() == 2, "two SJ-KVaRN streams");
        const uint32_t S = 128;
        auto * m_ref0 = llama_get_memory(ref0);
        auto * m_mc   = llama_get_memory(mc);

        // Mixed ubatches change the shapes of the non-attention matmuls, whose CPU results are not batch-invariant
        // (~1e-7 differences at the first layer, with or without SJ-KVaRN), so a mixed batch is checked to a
        // tolerance in a separate context, and the exact checks interleave the sequences one decode call each.
        {
            llama_context * mx = make_ctx(model, c, 2, unified);
            llama_context * r1 = make_ctx(model, c);
            require(mx && r1, "contexts");
            const auto rm = decode_mixed(mx, { { 0, 0, 1536, 0 }, { 1, 0, 704, 3 } }, 64);
            const float d = max_diff(rm[1], decode(r1, 0, 704, 64, 3));
            require(d < 1e-3f, "mixed prefill within 1e-3 of the single sequence (max|d| " + std::to_string(d) + ")");
            // mixed multi-token ubatches with different lengths per stream (each FA node cuts the mask to its
            // stream's own n_kv: the per-stream mask view is not contiguous and is copied)
            const auto ru = decode_mixed(mx, { { 0, 1536, 1792, 0 }, { 1, 704, 960, 3 } }, 64);
            const float du = max_diff(ru[1], decode(r1, 704, 960, 64, 3));
            llama_free(mx); llama_free(r1);
            printf("%-18s mixed ubatch vs alone: max|d| = %g (equal lengths), %g (unequal lengths)\n", name.c_str(), d, du);
            require(du < 1e-3f, "mixed unequal-length chunks within 1e-3 (max|d| " + std::to_string(du) + ")");
        }
        // prefill both sequences, seq 1 shorter (span lengths are multiples of the chunk: same ubatch boundaries)
        auto r = decode_mixed(mc, { { 0, 0, 1536, 0 }, { 1, 0, 704, 3 } }, 64, false);
        require(same_logits(r[0], decode(ref0, 0, 1536, 64, 0)), "seq 0 prefill logits = single sequence");
        require(same_logits(r[1], decode(ref1, 0, 704, 64, 3)), "seq 1 prefill logits = single sequence");
        // interleaved single-token steps: the adaptive tail flushes per stream at different times
        uint32_t flush0 = 0, flush1 = 0;
        uint32_t b0 = llama_sj_kvarn_seq_sealed_end(mc, 0), b1 = llama_sj_kvarn_seq_sealed_end(mc, 1);
        for (llama_pos k = 0; k < 600; ++k) {
            r = decode_mixed(mc, { { 0, 1536 + k, 1537 + k, 0 }, { 1, 704 + k, 705 + k, 3 } }, 1, false);
            require(same_logits(r[0], decode(ref0, 1536 + k, 1537 + k, 1, 0)), "seq 0 step " + std::to_string(k));
            require(same_logits(r[1], decode(ref1, 704 + k, 705 + k, 1, 3)), "seq 1 step " + std::to_string(k));
            require(llama_sj_kvarn_seq_sealed_end(mc, 0) == llama_sj_kvarn_sealed_end(ref0), "seq 0 sealed end = single");
            require(llama_sj_kvarn_seq_sealed_end(mc, 1) == llama_sj_kvarn_sealed_end(ref1), "seq 1 sealed end = single");
            if ((uint32_t) llama_sj_kvarn_seq_sealed_end(mc, 0) != b0) { ++flush0; b0 = llama_sj_kvarn_seq_sealed_end(mc, 0); }
            if ((uint32_t) llama_sj_kvarn_seq_sealed_end(mc, 1) != b1) { ++flush1; b1 = llama_sj_kvarn_seq_sealed_end(mc, 1); }
        }
        require(flush0 > 0 && flush1 > 0, "both streams flushed");
        llama_pos n0 = 2136;
        const llama_pos n1 = 1304;

        // truncate seq 0 into its sealed body (reopen) on both, seq 1 untouched; re-prefill and continue
        const llama_pos cut = llama_sj_kvarn_rm_floor(ref0, 0, b0 - 50);
        require(llama_sj_kvarn_rm_floor(mc, 0, b0 - 50) == cut, "same rm floor");
        require(llama_memory_seq_rm(m_ref0, 0, cut, -1) && llama_memory_seq_rm(m_mc, 0, cut, -1), "truncate into the body on both");
        require(llama_sj_kvarn_seq_sealed_end(mc, 0) == llama_sj_kvarn_sealed_end(ref0), "same sealed end after the truncation");
        require(llama_sj_kvarn_seq_sealed_end(mc, 1) == (int32_t) b1, "seq 1 sealed end untouched by the truncation");
        require(same_logits(decode(mc, cut, n0, 128, 5, 0), decode(ref0, cut, n0, 128, 5)), "seq 0 re-prefill after the truncation");
        require(same_logits(decode(mc, n1, n1 + 1, 1, 3, 1), decode(ref1, n1, n1 + 1, 1, 3)), "seq 1 step after the re-prefill");

        // per-sequence state: save seq 0, restore it into seq 1 of another two-stream context that holds other data
        const auto blob = seq_state_of(mc, 0);
        decode_mixed(mc2, { { 0, 0, 900, 9 }, { 1, 0, 1700, 11 } }, 64); // mixed: content only
        require(llama_state_seq_set_data(mc2, blob.data(), blob.size(), 1) == blob.size(), "restore seq 0's state into seq 1");
        // (the per-cell meta carries the sequence id, 0 in the blob and 1 here, so compare the sizes only)
        require(seq_state_of(mc2, 1).size() == blob.size(), "re-saved state the same size");
        require(llama_sj_kvarn_seq_sealed_end(mc2, 1) == llama_sj_kvarn_seq_sealed_end(mc, 0), "restored sealed end");
        for (llama_pos k = 0; k < 300; ++k) {
            r = decode_mixed(mc2, { { 1, n0 + k, n0 + k + 1, 5 }, { 0, 900 + k, 901 + k, 9 } }, 1, false);
            const auto a = decode(ref0, n0 + k, n0 + k + 1, 1, 5);
            require(same_logits(r[0], a), "restored sequence step " + std::to_string(k));
        }

        // seq_cp: seq 1 of mc2 (the restored one) into seq 0; both continue like the single sequence
        n0 += 300;
        llama_memory_seq_rm(llama_get_memory(mc2), 0, -1, -1);
        llama_memory_seq_cp(llama_get_memory(mc2), 1, 0, -1, -1);
        const auto c1 = decode(mc2, n0, n0 + 1, 1, 5, 1);
        const auto c0 = decode(mc2, n0, n0 + 200, 128, 13, 0);
        llama_context * ref2 = make_ctx(model, c);
        require(ref2 != nullptr, "context");
        {
            const auto rb = seq_state_of(ref0, 0);
            require(llama_state_seq_set_data(ref2, rb.data(), rb.size(), 0) == rb.size(), "single restore");
        }
        const bool cp0 = same_logits(c0, decode(ref2, n0, n0 + 200, 128, 13));
        const bool cp1 = same_logits(c1, decode(ref0, n0, n0 + 1, 1, 5));
        llama_free(ref2);
        require(cp0, "seq_cp copy continues like the single sequence");
        require(cp1, "seq_cp source continues like the single sequence");
        printf("%-18s OK: interleaved prefill/steps identical (mixed within 1e-3), %u/%u flushes, truncation at %d, cross-stream restore, seq_cp\n",
               name.c_str(), flush0, flush1, cut);
        GGML_UNUSED(S);
    } catch (const std::exception & e) {
        fprintf(stderr, "%-18s FAIL: %s\n", name.c_str(), e.what());
        result = 1;
    }
    for (auto * x : { ref0, ref1, mc, mc2 }) { if (x) llama_free(x); }
    return result;
}

// decode an image of nx x ny embeddings at position p0 as mtmd does for M-RoPE (positions p0, p0 + y, p0 + x, 0), in chunks of `width`
static void decode_image(llama_context * ctx, llama_pos p0, int nx, int ny, int width, int salt) {
    const int n = nx*ny;
    std::vector<float> embd((size_t) n*n_embd);
    std::mt19937 gen(salt);
    std::normal_distribution<float> dis(0.0f, 0.1f);
    for (auto & v : embd) v = dis(gen);
    for (int i0 = 0; i0 < n; i0 += width) {
        const int m = std::min(width, n - i0);
        std::vector<llama_pos> pos((size_t) 4*m);
        std::vector<int32_t> n_seq_id(m, 1);
        llama_seq_id seq = 0;
        std::vector<llama_seq_id *> seq_id(m, &seq);
        std::vector<int8_t> logits(m, 0);
        for (int j = 0; j < m; ++j) {
            const int i = i0 + j;
            pos[j]       = p0;
            pos[j + m]   = p0 + i/nx;
            pos[j + 2*m] = p0 + i%nx;
            pos[j + 3*m] = 0;
        }
        llama_batch batch = { m, nullptr, embd.data() + (size_t) i0*n_embd, pos.data(), n_seq_id.data(), seq_id.data(), logits.data() };
        require(llama_decode(ctx, batch) == 0, "image decode");
    }
}

// M-RoPE image prompt: the image tokens share one position and the text after the image goes on max(nx, ny) positions later.
// The cache must match a plain f16 cache, and save/restore, rollback and a cut at rm_floor inside the image must keep two caches identical.
// p_img < 128: the image starts in the sink; p_img > 128: in the sealed body, off a group boundary, so rm_floor goes below the image.
static int run_mrope(llama_model * model, const sj_cfg & c, llama_pos p_img) {
    const std::string name = std::string(c.name) + " M-RoPE " + std::to_string(p_img);
    sj_cfg pc = c;
    pc.plain = true;
    llama_context * src = make_ctx(model, c);
    llama_context * ref = make_ctx(model, pc);
    llama_context * dst = nullptr;
    int result = 0;
    try {
        require(src && ref, "contexts");
        llama_kv_cache * kv = cache_of(src);
        require(kv->is_sj_kvarn(), "SJ-KVaRN cache");
        const uint32_t S = 128, G = 128;
        const int nx = 16, ny = 12;              // 192 tokens: two ubatches of 128
        const llama_pos p_txt = p_img + nx;      // max(nx, ny) positions after the image
        const uint32_t  c_txt = p_img + nx*ny;   // first cell after the image

        // before any seal: close to a plain f16 cache (the rotated rows round differently: max|d| < 1e-3 here; ring rows taken by position give ~0.1)
        for (llama_context * x : { src, ref }) {
            decode(x, 0, p_img, 50);
            decode_image(x, p_img, nx, ny, 128, 1);
        }
        const float d_img = max_diff(decode(src, p_txt, p_txt + 240, 128), decode(ref, p_txt, p_txt + 240, 128));
        require(kv->get_sj_kvarn_sealed_end() == S, "nothing sealed yet");
        require(llama_memory_seq_pos_max(llama_get_memory(src), 0) == p_txt + 239, "last position");
        require(kv->get_cells(0).used_max_p1() == c_txt + 240, "one cell per token");
        require(d_img < 0.02f, "image prompt vs plain f16 cache (max|d| " + std::to_string(d_img) + ")");

        // seal past the image cells
        decode(src, p_txt + 240, p_txt + 1000, 128);
        const llama_pos n0 = p_txt + 1000;
        require(kv->get_sj_kvarn_sealed_end() > c_txt, "image cells sealed");

        // save, restore into a context that holds other data, continue both
        const auto blob = seq_state(src);
        dst = make_ctx(model, c);
        require(dst != nullptr, "destination context");
        decode(dst, 0, 300, 128, 7);
        require(llama_state_seq_set_data(dst, blob.data(), blob.size(), 0) == blob.size(), "restore");
        require(seq_state(dst) == blob, "re-saved state byte-identical");
        for (llama_pos p = n0; p < n0 + 20; ++p) {
            require(same_logits(decode(src, p, p + 1, 1), decode(dst, p, p + 1, 1)), "identical logits at step " + std::to_string(p));
        }
        const llama_pos n1 = n0 + 20;
        require(llama_sj_kvarn_compress_idle(src, 0, n1) >= 0 && llama_sj_kvarn_compress_idle(dst, 0, n1) >= 0, "idle seal");
        require(kv->get_sj_kvarn_sealed_end() == cache_of(dst)->get_sj_kvarn_sealed_end(), "same sealed end");

        // rollback above the sealed end
        require(llama_memory_seq_rm(llama_get_memory(src), 0, n1 - 9, -1) && llama_memory_seq_rm(llama_get_memory(dst), 0, n1 - 9, -1), "rollback");
        require(same_logits(decode(src, n1 - 9, n1 + 100, 128, 3), decode(dst, n1 - 9, n1 + 100, 128, 3)), "identical logits after the rollback");

        // the first text cell after the image is sealed, below the ring and not on a group boundary: a cut there cannot split the image,
        // so rm_floor goes to the image start, or below it to the group boundary when the image starts inside the body
        const llama_pos want = p_img <= (llama_pos) S ? p_img : (llama_pos) (S + G*((p_img - S)/G));
        const llama_pos f = llama_sj_kvarn_rm_floor(src, 0, p_txt);
        require(f == want && llama_sj_kvarn_rm_floor(dst, 0, p_txt) == want, "rm_floor " + std::to_string(want) + " (got " + std::to_string(f) + ")");
        require(!llama_memory_seq_rm(llama_get_memory(src), 0, p_txt, -1), "unaligned cut in the sealed body refused");
        require(llama_memory_seq_rm(llama_get_memory(src), 0, f, -1) && llama_memory_seq_rm(llama_get_memory(dst), 0, f, -1), "cut at rm_floor");
        require(kv->get_cells(0).used_max_p1() == (uint32_t) f, "cut removes the whole image");
        for (llama_context * x : { src, dst }) {
            if (f < p_img) {
                decode(x, f, p_img, 50);
            }
            decode_image(x, p_img, nx, ny, 128, 1);
        }
        require(same_logits(decode(src, p_txt, p_txt + 400, 128, 5), decode(dst, p_txt, p_txt + 400, 128, 5)), "identical logits after the cut and re-prefill");
        printf("%-18s OK: max|d| vs plain f16 %g, sealed end %u, restore, rollback, rm_floor at the image start\n",
               name.c_str(), d_img, kv->get_sj_kvarn_sealed_end());
    } catch (const std::exception & e) {
        fprintf(stderr, "%-18s FAIL: %s\n", name.c_str(), e.what());
        result = 1;
    }
    for (auto * x : { src, ref, dst }) { if (x) llama_free(x); }
    return result;
}

// EXPERIMENTAL WUSH-KV transform file: per layer AK = Had D_k, AQ = AK^-T = Had D_k^-1, AV = Had D_v, AO = AV^-1 = D_v^-1 Had
// (Had = normalized 256-pt Sylvester Hadamard, D = positive diagonal), so q.k and the attention output are preserved exactly.
static std::string write_transform(const char * dir, const char * name, uint32_t seed) {
    const uint32_t D = 256, H = 2;
    std::vector<float> had(D*D);
    for (uint32_t i = 0; i < D; ++i) for (uint32_t j = 0; j < D; ++j) {
        had[i*D + j] = (__builtin_popcount(i & j) & 1 ? -1.0f : 1.0f) / 16.0f;
    }
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(0.5f, 2.0f);
    const std::string path = std::string(dir) + "/" + name;
    FILE * f = fopen(path.c_str(), "wb");
    require(f != nullptr, "cannot write " + path);
    const uint32_t hdr[4] = { 1, D, H, n_layer };
    fwrite("SJWU", 1, 4, f); fwrite(hdr, 4, 4, f);
    for (int32_t il = 0; il < (int32_t) n_layer; ++il) {
        std::vector<float> dk(H*D), dv(H*D);
        for (auto & x : dk) x = dis(gen);
        for (auto & x : dv) x = dis(gen);
        std::vector<float> aq(H*D*D), ak(H*D*D), av(H*D*D), ao(H*D*D);
        for (uint32_t h = 0; h < H; ++h) for (uint32_t i = 0; i < D; ++i) for (uint32_t j = 0; j < D; ++j) {
            const size_t o = (size_t) h*D*D + i*D + j;
            ak[o] = had[i*D + j]*dk[h*D + j];
            aq[o] = had[i*D + j]/dk[h*D + j];
            av[o] = had[i*D + j]*dv[h*D + j];
            ao[o] = had[i*D + j]/dv[h*D + i];
        }
        fwrite(&il, 4, 1, f);
        for (auto * m : { &aq, &ak, &av, &ao }) fwrite(m->data(), 4, m->size(), f);
    }
    fclose(f);
    return path;
}

static void scratch_views_case(llama_model * model, sj_cfg cfg) {
    llama_context_ptr ctx(make_ctx(model, cfg));
    require(ctx != nullptr, "scratch context created");
    auto * kv = cache_of(ctx.get());
    auto * buft = ggml_backend_buffer_get_type(kv->get_k_storage(0)->buffer);
    if (getenv("SJ_TEST_GPU")) {
        auto * device = ggml_backend_buft_get_device(buft);
        require(device && ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_GPU,
                "scratch views must use the requested GPU");
    }
    printf("scratch-views: %s buffer=%s\n", cfg.plain ? "plain" : "SJ-KVaRN", ggml_backend_buft_name(buft));
    constexpr uint32_t prompt_limit = 512;
    uint32_t protected_cells = prompt_limit;
    if (cfg.plain) {
        // A cold sequence in a reused cache need not occupy cells starting at zero.
        // Place a low logical position beyond the token-count loan boundary.
        llama_pos pos = 0;
        int32_t n_seq_id = 1;
        llama_seq_id seq_id = 0;
        llama_seq_id * seq_ids = &seq_id;
        llama_ubatch ubatch = {};
        ubatch.n_tokens = ubatch.n_seq_tokens = ubatch.n_seqs = ubatch.n_seqs_unq = ubatch.n_pos = 1;
        ubatch.pos = &pos;
        ubatch.n_seq_id = &n_seq_id;
        ubatch.seq_id = &seq_ids;
        llama_kv_cache::slot_info slot = {};
        slot.strm = {0};
        slot.idxs = {{768}};
        kv->apply_ubatch(slot, ubatch);
        require(kv->get_cells(0).used_max_p1() == 769, "offset physical cell fixture applied");
        protected_cells = kv->get_n_kv(slot);
        require(protected_cells == 1024, "masked attention pads beyond occupied physical cells");
    }
    for (int32_t first_unused : {2, 4}) {
        for (int32_t il = 0; il < 4; ++il) {
            for (auto * t : {kv->get_k_storage(il), kv->get_v_storage(il), cfg.plain ? nullptr : kv->get_sj_kvarn_body(il)}) {
                if (!t) { continue; }
                ggml_backend_tensor_memset(t, 0xa5, 0, ggml_nbytes(t));
            }
        }
        auto views = kv->b1_scratch_views(buft, first_unused, prompt_limit);
        require(!views.empty(), "scratch views exist");
        for (auto & view : views) { ggml_backend_buffer_clear(view.get(), 0x3c); }
        size_t touched = 0;
        for (int32_t il = 0; il < 4; ++il) {
            for (auto * t : {kv->get_k_storage(il), kv->get_v_storage(il), cfg.plain ? nullptr : kv->get_sj_kvarn_body(il)}) {
                if (!t) { continue; }
                std::vector<uint8_t> bytes(ggml_nbytes(t));
                ggml_backend_tensor_get(t, bytes.data(), 0, bytes.size());
                size_t protected_end = bytes.size();
                if (il >= first_unused) {
                    protected_end = 0;
                } else if (cfg.plain) {
                    protected_end = protected_cells * t->nb[1];
                } else if (t == kv->get_sj_kvarn_body(il)) {
                    // 512 cells: 128 sink cells followed by three sealed groups of 128.
                    protected_end = 3 * ggml_sj_kvarn_rec_bytes(256, 128, 4, 4) * 2;
                }
                for (size_t i = 0; i < bytes.size(); ++i) {
                    if (i < protected_end) { require(bytes[i] == 0xa5, "live KV bytes were preserved"); }
                    else if (bytes[i] == 0x3c) { ++touched; }
                    else { require(bytes[i] == 0xa5, "untouched padding preserved"); }
                }
            }
        }
        require(touched > 0, "borrowed views were written");
    }
    llama_context_ptr multi(make_ctx(model, cfg, 2));
    require(multi != nullptr, "multi scratch context created");
    require(cache_of(multi.get())->b1_scratch_views(buft, 0, 512).empty(), "multiple streams refuse borrowing");
}

static void deferred_mask_capacity_case() {
    ggml_context * ctx = ggml_init({ 1024*1024, nullptr, false });
    require(ctx != nullptr, "mask test context");
    auto * valid = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 8, 3);
    auto * wrong_type = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 3);
    auto * padded = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 10, 3);
    auto * strided = ggml_view_2d(ctx, padded, 8, 3, 10*sizeof(ggml_fp16_t), 0);
    ggml_tensor no_data = *valid;
    no_data.data = nullptr;
    require(!llama_kq_mask_deferred_set(nullptr, 8, { 0, 3, 8 }), "null mask refused");
    require(!llama_kq_mask_deferred_set(&no_data, 8, { 0, 3, 8 }), "unallocated mask refused");
    require(!llama_kq_mask_deferred_set(wrong_type, 8, { 0, 3, 8 }), "non-f16 mask refused");
    require(!llama_kq_mask_deferred_set(strided, 8, { 0, 3, 8 }), "strided mask refused");
    require(!llama_kq_mask_deferred_set(valid, 0, { 0, 0, 0 }), "zero width refused");
    require(!llama_kq_mask_deferred_set(valid, 7, { 0, 3, 7 }), "wrong width refused");
    require(!llama_kq_mask_deferred_set(valid, 8, { 0, 8 }), "missing row count refused");
    require(!llama_kq_mask_deferred_set(valid, 8, { 0, 3, 8, 8 }), "extra row count refused");
    require(!llama_kq_mask_deferred_set(valid, 8, { -1, 3, 8 }), "negative visible count refused");
    require(!llama_kq_mask_deferred_set(valid, 8, { 0, 3, 9 }), "oversized visible count refused");
    std::vector<int32_t> refused_counts;
    int64_t refused_width = 0;
    require(!llama_kq_mask_deferred_get(valid, refused_counts, &refused_width), "refused counts leave no record");
    require(!llama_kq_mask_deferred_get(nullptr, refused_counts, &refused_width), "null mask lookup refused");
    llama_kq_mask_deferred_fill_host(nullptr);
    require(llama_kq_mask_deferred_set(valid, 8, { 0, 3, 8 }), "register mask before layout change");
    auto * valid_data = static_cast<ggml_fp16_t *>(valid->data);
    std::fill(valid_data, valid_data + 24, ggml_fp32_to_fp16(1.0f));
    valid->nb[1] += sizeof(ggml_fp16_t);
    require(!llama_kq_mask_deferred_get(valid, refused_counts, &refused_width), "changed mask layout refused");
    llama_kq_mask_deferred_fill_host(valid);
    for (int i = 0; i < 24; ++i) {
        require(valid_data[i] == ggml_fp32_to_fp16(1.0f), "changed layout must not write host bytes");
    }
    valid->nb[1] -= sizeof(ggml_fp16_t);
    require(llama_kq_mask_deferred_get(valid, refused_counts, &refused_width), "layout refusal keeps pending record");
    require(refused_width == 8 && refused_counts == std::vector<int32_t>({ 0, 3, 8 }), "layout refusal keeps row counts");
    llama_kq_mask_deferred_fill_host(valid);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 8; ++c) {
            require(valid_data[r*8 + c] == ggml_fp32_to_fp16(c < refused_counts[r] ? 0.0f : -INFINITY), "restored layout mask bytes");
        }
    }
    printf("OK: changed deferred mask layout refused without host writes or lost counts\n");
    std::vector<ggml_tensor *> masks;
    for (int i = 0; i < 65; ++i) {
        auto * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 8, 3);
        masks.push_back(mask);
        require(llama_kq_mask_deferred_set(mask, 8, { 0, 3, 8 }) == (i < 64), "full list must refuse a new deferred mask");
    }
    for (int i = 0; i < 64; ++i) {
        std::vector<int32_t> counts;
        int64_t n_kv = 0;
        require(llama_kq_mask_deferred_get(masks[i], counts, &n_kv), "pending mask must survive capacity pressure");
        require(n_kv == 8 && counts == std::vector<int32_t>({ 0, 3, 8 }), "pending counts preserved");
        llama_kq_mask_deferred_fill_host(masks[i]);
        const auto * data = static_cast<const ggml_fp16_t *>(masks[i]->data);
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 8; ++c) {
                require(data[r*8 + c] == ggml_fp32_to_fp16(c < counts[r] ? 0.0f : -INFINITY), "host fallback mask bytes");
            }
        }
        require(!llama_kq_mask_deferred_get(masks[i], counts, &n_kv), "filled mask uses normal host copy");
    }
    std::vector<int32_t> counts;
    int64_t n_kv = 0;
    require(!llama_kq_mask_deferred_get(masks.back(), counts, &n_kv), "refused mask uses normal host fill");
    for (int pass = 0; pass < 2; ++pass) {
        for (int i = 0; i < 65; ++i) {
            require(llama_kq_mask_deferred_set(masks[i], 8, { 8, 1, 0 }) == (i < 64), "host-filled masks free their slots");
        }
        for (int i = 0; i < 64; ++i) {
            require(llama_kq_mask_deferred_get(masks[i], counts, &n_kv), "reused slot has new counts");
            require(n_kv == 8 && counts == std::vector<int32_t>({ 8, 1, 0 }), "old counts do not survive reuse");
            llama_kq_mask_deferred_fill_host(masks[i]);
            llama_kq_mask_deferred_fill_host(masks[i]);
            const auto * data = static_cast<const ggml_fp16_t *>(masks[i]->data);
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 8; ++c) {
                    require(data[r*8 + c] == ggml_fp32_to_fp16(c < counts[r] ? 0.0f : -INFINITY), "reused host mask bytes");
                }
            }
            require(!llama_kq_mask_deferred_get(masks[i], counts, &n_kv), "completed fill retires its slot");
        }
    }
    ggml_free(ctx);
    printf("OK: host-filled mask slots reused twice with exact bytes\n");
    printf("OK: invalid deferred masks refused without using pending slots\n");
    printf("OK: 64 pending masks preserved, 65th falls back, all host mask bytes exact\n");
}

static void ple_hash_pages_case(bool windowed = false, bool bounded = false, bool coalesced = false, size_t ubatch_size = 384, int populate = 0, size_t table_offset = 0, bool file_fault = false) {
#if defined(__linux__)
    for (uint32_t n_gram : { 4, 8 }) {
        for (llama_pos pos0 : { 0, 23 }) {
            const uint32_t per_gram = 2, vocab = coalesced ? 4093 : 1021, eos = 997;
            const uint32_t n_heads = (n_gram - 1)*per_gram, n_rows = n_heads*vocab;
            const size_t ubatch = coalesced || bounded ? ubatch_size : windowed ? 1 : 256;
            size_t chunk = coalesced ? ubatch : bounded ? std::min<size_t>(256, ubatch) : 256;
            const size_t n_tokens = 3*chunk + 17, n_pages = n_rows + (table_offset != 0), size = n_pages*4096;
            require(table_offset < 4096 && (coalesced || table_offset == 0), "hash page table offset");
            if (coalesced && !bounded) {
                chunk = std::max<size_t>(256, n_tokens);
            }
            const char * tmp_dir = getenv("TMPDIR");
            std::string path = std::string(tmp_dir ? tmp_dir : ".") + "/llama-ple-window-XXXXXX";
            const bool file_backed = coalesced || file_fault;
            std::unique_ptr<char, void(*)(char *)> cleanup(file_backed ? &path[0] : nullptr, [](char * p) { unlink(p); });
            std::unique_ptr<FILE, int(*)(FILE *)> file(nullptr, fclose);
            if (file_backed) {
                const int fd = mkstemp(&path[0]);
                require(fd >= 0, "coalesced page file");
                file.reset(fdopen(fd, "w+"));
                if (!file) { close(fd); }
                require(file != nullptr && ftruncate(fd, size) == 0, "coalesced page file size");
                // Sparse tmpfs reads do not populate the page cache; use real pages and require a cold start.
                const std::vector<char> data(4096, 0x5a);
                for (size_t off = 0; off < size; off += data.size()) {
                    require(pwrite(fd, data.data(), data.size(), off) == (ssize_t) data.size(), "coalesced page file contents");
                }
                require(fsync(fd) == 0 && posix_fadvise(fd, 0, size, POSIX_FADV_DONTNEED) == 0, "coalesced page file flushed and evicted");
            }
            void * pages = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                    file_backed ? MAP_SHARED : MAP_PRIVATE | MAP_ANONYMOUS, file_backed ? fileno(file.get()) : -1, 0);
            require(pages != MAP_FAILED, "hash page mapping");
            require(madvise(pages, size, MADV_NOHUGEPAGE) == 0, "hash pages stay separate");
            if (populate || bounded) {
                require(madvise(pages, size, MADV_SEQUENTIAL) == 0, "mapped population starts with read-around enabled");
            }
            std::unique_ptr<FILE, int(*)(FILE *)> pagemap(populate ? fopen("/proc/self/pagemap", "rb") : nullptr, fclose);
            std::vector<uint64_t> ptes(populate ? n_pages : 0);
            auto read_ptes = [&]() {
                require(pagemap != nullptr && pread(fileno(pagemap.get()), ptes.data(), ptes.size()*sizeof(uint64_t),
                        ((uintptr_t) pages/4096)*sizeof(uint64_t)) == (ssize_t) (ptes.size()*sizeof(uint64_t)), "read mapped page presence bits");
            };
            if (populate) {
                read_ptes();
                for (uint64_t pte : ptes) {
                    require(!(pte & (UINT64_C(1) << 63)), "mapped pages start without page-table entries");
                }
            }
            ggml_context * ctx = ggml_init({ 1024*1024, nullptr, true });
            require(ctx != nullptr, "hash page context");
            auto * table = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, n_rows);
            table->data = (char *) pages + table_offset;
            table->buffer = ggml_backend_cpu_buffer_from_ptr(pages, size);
            require(table->buffer != nullptr, "hash page buffer");
            llama_model_ptr model(llama_model_create(LLM_ARCH_QWEN4EXP, llama_model_default_params()));
            require(model != nullptr, "hash page model");
            model->per_layer_tok_embd = table;
            model->can_prefetch.insert(table);
            auto & hp = model->hparams;
            hp.ple_ngram_size = n_gram;
            hp.ple_n_heads = n_heads;
            hp.ple_heads_per_ngram = per_gram;
            hp.ple_eos_token_id = eos;
            for (uint32_t j = 0; j < n_gram; ++j) {
                hp.ple_layer_multipliers[j] = UINT64_C(0x9e3779b97f4a7c15) + j*UINT64_C(0xd1b54a32d192ed03);
            }
            for (uint32_t h = 0; h < n_heads; ++h) {
                hp.ple_head_offsets[h] = h*vocab;
                hp.ple_head_vocab_sizes[h] = vocab - h;
            }
            std::vector<llama_token> tokens(n_tokens);
            for (size_t i = 0; i < n_tokens; ++i) {
                tokens[i] = i % 127 == 0 ? eos : i % 131 == 0 ? LLAMA_TOKEN_NULL : (i*179 + i*i*13) % eos;
            }
            std::vector<unsigned char> expected(n_pages, 0), resident(n_pages);
            if (file_backed) {
                require(mincore(pages, size, resident.data()) == 0, "initial coalesced page residency");
                for (unsigned char page : resident) {
                    require(!(page & 1), "coalesced page file must start cold; set TMPDIR to a disk filesystem");
                }
            }
            std::vector<size_t> first_chunk(n_pages, n_tokens);
            for (size_t i = 0; i < n_tokens; ++i) {
                std::vector<llama_token> history(n_gram - 1);
                for (uint32_t s = 1; s < n_gram; ++s) {
                    history[n_gram - 1 - s] = i < s ? LLAMA_TOKEN_NULL : tokens[i - s];
                }
                std::vector<int32_t> rows(n_heads);
                llama_ple_hash_rows(hp, tokens[i], history.data(), rows.data());
                // Independent old graph formula: restart the hash for each ngram width.
                for (uint32_t g = 2; g <= n_gram; ++g) {
                    uint64_t mixed = (uint64_t) tokens[i]*hp.ple_layer_multipliers[0];
                    bool cut = false;
                    for (uint32_t s = 1; s < g; ++s) {
                        const llama_token t = history[n_gram - 1 - s];
                        cut = cut || t < 0 || t == (llama_token) eos;
                        mixed ^= (uint64_t) (cut ? (llama_token) eos : t)*hp.ple_layer_multipliers[s];
                    }
                    for (uint32_t k = 0; k < per_gram; ++k) {
                        const uint32_t h = (g - 2)*per_gram + k;
                        require(rows[h] == (int32_t) (hp.ple_head_offsets[h] + mixed % hp.ple_head_vocab_sizes[h]),
                                "incremental graph hash matches every old row index");
                    }
                }
                for (uint32_t g = 2; g <= n_gram; ++g) {
                    uint64_t mixed = (uint64_t) tokens[i]*hp.ple_layer_multipliers[0];
                    bool cut = false, known = true;
                    for (uint32_t j = 1; j < g; ++j) {
                        if (!cut && i < j && pos0 != 0) {
                            known = false;
                            break;
                        }
                        const llama_token t = cut || i < j ? (llama_token) eos : tokens[i - j];
                        cut = cut || t < 0 || t == (llama_token) eos;
                        mixed ^= (uint64_t) (cut ? (llama_token) eos : t)*hp.ple_layer_multipliers[j];
                    }
                    if (known) {
                        for (uint32_t k = 0; k < per_gram; ++k) {
                            const uint32_t h = (g - 2)*per_gram + k;
                            const size_t row = hp.ple_head_offsets[h] + mixed % hp.ple_head_vocab_sizes[h];
                            for (size_t page = row; page <= row + (table_offset != 0); ++page) {
                                expected[page] = 1;
                                first_chunk[page] = std::min(first_chunk[page], i/chunk);
                            }
                        }
                    }
                }
            }
            setenv("LLAMA_PLE_PREFETCH", "2", 1);
            setenv("LLAMA_PLE_PF_MIN", "1", 1);
            std::vector<size_t> first_read = first_chunk;
            size_t expected_reads = 0;
            if (coalesced) {
                const size_t gap = bounded ? 0 : 16;
                for (size_t c = 0; c < (n_tokens + chunk - 1)/chunk; ++c) {
                    size_t start = 0, end = 0;
                    for (size_t r = 0; r < n_pages; ++r) {
                        if (first_chunk[r] != c) { continue; }
                        if (end == 0 || r > end + gap || r - start >= 256) {
                            start = r;
                            end = r;
                        }
                        for (size_t p = end; p <= r; ++p) {
                            ++expected_reads;
                            first_read[p] = std::min(first_read[p], c);
                        }
                        end = r + 1;
                    }
                }
            }
            setenv("LLAMA_PLE_PF_MODE", coalesced ? "coalesce" : "fault", 1);
            setenv("LLAMA_PLE_PF_AHEAD", coalesced || (windowed && !bounded) ? "0" : "1000000", 1);
            setenv("LLAMA_PLE_PF_WINDOW_UBATCHES", bounded ? "1" : "0", 1);
            setenv("LLAMA_PLE_PF_COLD_THREADS", "0", 1);
            setenv("LLAMA_PLE_PF_POPULATE", std::to_string(populate).c_str(), 1);
            std::string startup;
            llama_log_set([](ggml_log_level, const char * text, void * data) {
                *static_cast<std::string *>(data) += text;
            }, &startup);
            auto pf = llama_ple_prefetch::create(*model, ubatch);
            llama_log_set(nullptr, nullptr);
            require(pf != nullptr, "hash page prefetch created");
            const std::string ahead_log = ", " + std::to_string(coalesced ? 0 : ubatch) + " tokens ahead\n";
            require(!bounded || startup.find(ahead_log) != std::string::npos,
                    "prefetch ubatch cap limits the effective token window");
            require(!coalesced || startup.find("coalesced pread") != std::string::npos, "file-backed coalescing active");
            require(!populate || startup.find("mapped reads use random-access mapping") != std::string::npos,
                    "mapped population disables mmap read-around before reading cold pages");
            if (bounded) {
                std::unique_ptr<FILE, int(*)(FILE *)> smaps(fopen("/proc/self/smaps", "r"), fclose);
                require(smaps != nullptr, "read bounded mapping advice");
                char line[4096];
                bool inside = false, random = false;
                while (fgets(line, sizeof(line), smaps.get())) {
                    unsigned long long first = 0, last = 0;
                    if (sscanf(line, "%llx-%llx", &first, &last) == 2) {
                        inside = (uintptr_t) table->data >= first && (uintptr_t) table->data < last;
                    } else if (inside && strncmp(line, "VmFlags:", 8) == 0) {
                        random = strstr(line, " rr ") && !strstr(line, " sr ");
                        break;
                    }
                }
                require(random, "bounded prefetch disables mapped read-around for fault and pread-only readers");
            }
            pf->hint(tokens.data(), tokens.size(), pos0);
            const size_t consume_size = coalesced || bounded ? std::min<size_t>(113, ubatch) : windowed ? 113 : 256;
            size_t future_checks = 0;
            for (size_t i = 0; i < n_tokens; i += consume_size) {
                const size_t count = std::min(consume_size, n_tokens - i);
                require(pf->consume(count, nullptr, 0), "hash page chunk consumed");
                require(mincore(pages, size, resident.data()) == 0, "consumed chunk residency");
                if (populate) { read_ptes(); }
                const size_t read_end = std::max(ubatch, i + count) + (bounded && !coalesced ? ubatch : 0);
                const size_t allowed = (read_end + chunk - 1)/chunk;
                for (size_t r = 0; r < n_pages; ++r) {
                    future_checks += expected[r] && first_read[r] >= allowed;
                    require(first_chunk[r] > (i + count - 1)/chunk || (resident[r] & 1), "consume waits for every page of its chunks");
                    require(!populate || first_chunk[r] > (i + count - 1)/chunk || (ptes[r] & (UINT64_C(1) << 63)),
                            "coalesced pread populates every consumed mapped page");
                    require(!windowed || first_read[r] < allowed || !(resident[r] & 1),
                            "pages beyond the bounded window stay untouched");
                }
            }
            require(!windowed || future_checks > 0, "window oracle checks unread future pages");
            std::string summary;
            llama_log_set([](ggml_log_level, const char * text, void * data) {
                if (strstr(text, "[PLEPF]") && strstr(text, " tokens: ")) {
                    *static_cast<std::string *>(data) = text;
                }
            }, &summary);
            pf->stop();
            llama_log_set(nullptr, nullptr);
            require(mincore(pages, size, resident.data()) == 0, "hash page residency");
            size_t n_expected = 0;
            for (size_t r = 0; r < n_pages; ++r) {
                require((resident[r] & 1) == (coalesced ? first_read[r] < n_tokens : expected[r]), "hash page set matches independent full hash");
                n_expected += expected[r];
            }
            require(n_expected > 0 && n_expected < n_pages, "hash test checks touched and untouched pages");
            require(!coalesced || !bounded || expected_reads == n_expected, "bounded coalescing reads each requested page once without gaps");
            const char * read_at = strstr(summary.c_str(), " ms, ");
            long long pages_read = -1;
            require(read_at && sscanf(read_at, " ms, %lld read", &pages_read) == 1, "prefetch page total logged");
            require(pages_read == (long long) (coalesced ? expected_reads : n_expected), "every requested page counted after readers join");
            if (populate == 2 || populate == 3) {
                const char * mapped_at = strstr(summary.c_str(), "; bulk mapped runs=");
                long long mapped_runs = 0;
                require(mapped_at && sscanf(mapped_at, "; bulk mapped runs=%lld", &mapped_runs) == 1 && mapped_runs > 0,
                        "bulk mapped-page path completed at least one run");
            }
            if (populate == 3) {
                const char * direct_at = strstr(summary.c_str(), "; direct mapped runs=");
                long long direct_runs = 0;
                require(direct_at && sscanf(direct_at, "; direct mapped runs=%lld", &direct_runs) == 1 && direct_runs > 0,
                        "direct mapped-page path bypassed scratch reads for at least one run");
            }
            if (bounded) {
                // Stop a partial window, then reuse the same prefetcher for a fresh call.
                for (int retry = 0; retry < 2; ++retry) {
                    pf->hint(tokens.data(), tokens.size(), pos0);
                    require(pf->consume(1, nullptr, 0), "bounded prefetch restarts after a partial-window stop");
                    pf->stop();
                }
            }
            if (coalesced) {
                if (populate == 3) {
                    size_t last_page = 0;
                    for (size_t page = 0; page < n_pages; ++page) {
                        if (first_read[page] < n_tokens) { last_page = page; }
                    }
                    require(ftruncate(fileno(file.get()), last_page*4096 + 1) == 0, "truncate within the last requested page");
                    pf->hint(tokens.data(), tokens.size(), pos0);
                    require(!pf->consume(tokens.size(), nullptr, 0), "direct mapping refuses an incomplete final requested page");
                    pf->stop();
                }
                require(ftruncate(fileno(file.get()), 0) == 0, "truncate prefetch file after successful reads");
                pf->hint(tokens.data(), tokens.size(), pos0);
                require(!pf->consume(tokens.size(), nullptr, 0), "incomplete file reads fall back without marking chunks ready");
                summary.clear();
                llama_log_set([](ggml_log_level, const char * text, void * data) {
                    if (strstr(text, "[PLEPF]") && strstr(text, " tokens: ")) {
                        *static_cast<std::string *>(data) = text;
                    }
                }, &summary);
                pf->stop();
                llama_log_set(nullptr, nullptr);
                const char * failed_read_at = strstr(summary.c_str(), " ms, ");
                long long failed_pages = -1;
                require(failed_read_at && sscanf(failed_read_at, " ms, %lld read", &failed_pages) == 1 && failed_pages == 0,
                        "incomplete file reads do not count successful pages");
            }
            pf.reset();
            model.reset();
            ggml_backend_buffer_free(table->buffer);
            ggml_free(ctx);
            munmap(pages, size);
        }
    }
    printf("OK: incremental 4/8-gram hashes match exact page sets across heads, EOS and call offsets\n");
    printf("OK: incremental graph hashes match every original row index with EOS and missing history\n");
    if (windowed) {
        printf("OK: batched page readers preserve exact pages and totals across short windows and a partial final batch\n");
        printf("OK: every consumed chunk has all required pages before stop joins readers\n");
    }
    if (bounded) {
        printf("OK: ubatch read-ahead cap preserves ready pages and leaves future pages untouched\n");
        printf("OK: bounded producer stops and restarts after partial-window consumes\n");
    }
    if (coalesced) {
        printf("OK: file-backed coalescing obeys ubatch windows, exact run pages and read totals\n");
        printf("OK: truncated prefetch reads fall back without completed chunks or page counts\n");
    }
    if (populate) {
        printf("OK: consumed coalesced pages have mapped page-table entries before stop\n");
    }
#else
    printf("SKIP: hash page test needs Linux\n");
#endif
}

static void ple_wide_rows_case() {
#if defined(__linux__)
    for (int width : { 2048, 4096, 65536, 65793 }) {
        const size_t n_tokens = width >= 65536 ? 65 : 3*256 + 17;
        const size_t size = ((n_tokens*width*sizeof(float) + 128 + 4095)/4096)*4096;
        void * pages = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        require(pages != MAP_FAILED, "wide prefetch pages");
        require(madvise(pages, size, MADV_NOHUGEPAGE) == 0, "wide prefetch uses separate pages");
        ggml_context * ctx = ggml_init({ 1024*1024, nullptr, true });
        require(ctx != nullptr, "wide prefetch context");
        auto * table = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, n_tokens);
        table->data = static_cast<char *>(pages) + 128;
        table->buffer = ggml_backend_cpu_buffer_from_ptr(pages, size);
        require(table->buffer != nullptr, "wide prefetch host buffer");
        llama_model_ptr model(llama_model_create(LLM_ARCH_QWEN4EXP, llama_model_default_params()));
        require(model != nullptr, "wide prefetch model");
        model->per_layer_tok_embd = table;
        model->can_prefetch.insert(table);
        model->hparams.ple_n_heads = 1;
        model->hparams.ple_ngram_size = 2;
        model->hparams.ple_heads_per_ngram = 1;
        model->hparams.ple_eos_token_id = n_tokens - 1;
        model->hparams.ple_layer_multipliers.fill(0);
        model->hparams.ple_layer_multipliers[0] = 1;
        model->hparams.ple_head_vocab_sizes[0] = n_tokens;
        model->hparams.ple_head_offsets[0] = 0;
        std::vector<llama_token> tokens(n_tokens);
        for (size_t i = 0; i < n_tokens; ++i) {
            tokens[i] = i;
        }
        setenv("LLAMA_PLE_PREFETCH", "1", 1);
        setenv("LLAMA_PLE_PF_MIN", "1", 1);
        setenv("LLAMA_PLE_PF_MODE", "fault", 1);
        setenv("LLAMA_PLE_PF_AHEAD", "1000000", 1);
        setenv("LLAMA_PLE_PF_COLD_THREADS", "0", 1);
        auto & hp = model->hparams;
        require(!llama_ple_prefetch::create(*model, 0), "prefetch refuses a zero-token ubatch before starting readers");
        hp.ple_ngram_size = hp.ple_layer_multipliers.size() + 1;
        require(!llama_ple_prefetch::create(*model, n_tokens), "prefetch refuses excess ngram width");
        hp.ple_ngram_size = 2;
        hp.ple_n_heads = hp.ple_head_vocab_sizes.size() + 1;
        hp.ple_heads_per_ngram = hp.ple_n_heads;
        require(!llama_ple_prefetch::create(*model, n_tokens), "prefetch refuses excess heads");
        hp.ple_n_heads = 1;
        hp.ple_heads_per_ngram = 0;
        require(!llama_ple_prefetch::create(*model, n_tokens), "prefetch refuses zero heads per ngram");
        hp.ple_heads_per_ngram = 2;
        require(!llama_ple_prefetch::create(*model, n_tokens), "prefetch refuses mismatched head count");
        hp.ple_heads_per_ngram = UINT32_MAX;
        hp.ple_ngram_size = 3;
        require(!llama_ple_prefetch::create(*model, n_tokens), "prefetch checks head product without wrapping");
        hp.ple_ngram_size = 2;
        hp.ple_heads_per_ngram = 1;
        hp.ple_head_vocab_sizes[0] = 0;
        require(!llama_ple_prefetch::create(*model, n_tokens), "prefetch refuses zero vocab size");
        hp.ple_head_vocab_sizes[0] = n_tokens;
        hp.ple_head_offsets[0] = 1;
        require(!llama_ple_prefetch::create(*model, n_tokens), "prefetch refuses rows past table end");
        hp.ple_head_offsets[0] = UINT32_MAX;
        hp.ple_head_vocab_sizes[0] = UINT32_MAX;
        require(!llama_ple_prefetch::create(*model, n_tokens), "prefetch checks row end without wrapping");
        hp.ple_head_offsets[0] = 0;
        hp.ple_head_vocab_sizes[0] = n_tokens;
        const size_t row_size = table->nb[1];
        const size_t end_offset = size - n_tokens*row_size;
        table->nb[1] = SIZE_MAX;
        require(!llama_ple_prefetch::create(*model, n_tokens), "prefetch refuses row stride past host buffer");
        table->nb[1] = row_size;
        table->ne[1] = INT64_MAX;
        require(!llama_ple_prefetch::create(*model, n_tokens), "prefetch refuses row span overflow");
        table->ne[1] = n_tokens;
        table->data = (void *) ((uintptr_t) pages - 1);
        require(!llama_ple_prefetch::create(*model, n_tokens), "prefetch refuses table before host buffer");
        table->data = static_cast<char *>(pages) + size;
        require(!llama_ple_prefetch::create(*model, n_tokens), "prefetch refuses table at host buffer end");
        table->data = static_cast<char *>(pages) + end_offset + 1;
        require(!llama_ple_prefetch::create(*model, n_tokens), "prefetch refuses row span one byte past host buffer");
        table->data = static_cast<char *>(pages) + end_offset;
        require(llama_ple_prefetch::create(*model, n_tokens) != nullptr, "prefetch accepts row span ending at host buffer end");
        table->data = static_cast<char *>(pages) + 128;
        for (const char * minimum : { "4294967296", "4294967297", "9223372036854775807", "9223372036854775808",
                "999999999999999999999999999999999999999999999999999999999999999999" }) {
            setenv("LLAMA_PLE_PF_MIN", minimum, 1);
            auto short_call = llama_ple_prefetch::create(*model, n_tokens);
            require(short_call != nullptr, "prefetch with large minimum created");
            short_call->hint(tokens.data(), tokens.size(), 0);
            require(!short_call->consume(n_tokens, nullptr, 0), "prefetch minimum does not wrap into a short call");
        }
        setenv("LLAMA_PLE_PF_MIN", "1", 1);
        for (const char * threads : { "-9223372036854775809", "-999999999999999999999999999999999999999999999999999999999999999999" }) {
            setenv("LLAMA_PLE_PREFETCH", threads, 1);
            require(!llama_ple_prefetch::create(*model, n_tokens), "negative overflowing thread setting keeps prefetch off");
        }
        setenv("LLAMA_PLE_PREFETCH", "1", 1);
        auto pf = llama_ple_prefetch::create(*model, n_tokens);
        require(pf != nullptr, "wide row page prefetch created");
        pf->hint(tokens.data(), tokens.size(), 0);
        require(pf->consume(n_tokens, nullptr, 0), "wide row pages consumed");
        pf->stop();
        std::vector<unsigned char> resident(size/4096);
        require(mincore(pages, size, resident.data()) == 0, "wide row page residency");
        for (unsigned char page : resident) {
            require((page & 1) != 0, "every wide row page was read");
        }
        pf.reset();
        model.reset();
        ggml_backend_buffer_free(table->buffer);
        ggml_free(ctx);
        munmap(pages, size);
    }
    printf("OK: invalid prefetch ngram heads and row ranges refused\n");
    printf("OK: prefetch row spans stay within host buffer without overflow\n");
    printf("OK: large prefetch minimum does not wrap into short calls\n");
    printf("OK: out-of-range prefetch settings saturate without enabling short calls or disabled readers\n");
    printf("OK: unaligned 8KiB and 16KiB prefetch rows cover every page\n");
    printf("OK: unaligned 256KiB rows cover full bitmap words and both boundary pages\n");
    printf("OK: ragged wide rows cover mid-word starts and shared boundary pages\n");
#else
    printf("SKIP: wide prefetch page test needs Linux\n");
#endif
}

static void ple_cold_recheck_case() {
#if defined(__linux__)
    ggml_context * ctx = ggml_init({ 1024*1024, nullptr, true });
    require(ctx != nullptr, "prefetch test context");
    const size_t size = 4*4096;
    void * pages = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    require(pages != MAP_FAILED, "prefetch test pages");
    auto * table = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 4);
    table->data = pages;
    table->buffer = ggml_backend_cpu_buffer_from_ptr(pages, size);
    require(table->buffer != nullptr, "prefetch host buffer");
    llama_model_ptr model_owner(llama_model_create(LLM_ARCH_QWEN4EXP, llama_model_default_params()));
    require(model_owner != nullptr, "prefetch test model");
    auto & model = *model_owner;
    model.per_layer_tok_embd = table;
    model.can_prefetch.insert(table);
    model.hparams.ple_n_heads = 1;
    model.hparams.ple_ngram_size = 2;
    model.hparams.ple_heads_per_ngram = 1;
    model.hparams.ple_eos_token_id = 3;
    model.hparams.ple_layer_multipliers.fill(0);
    model.hparams.ple_layer_multipliers[0] = 1;
    model.hparams.ple_head_vocab_sizes[0] = 4;
    model.hparams.ple_head_offsets[0] = 0;
    std::vector<llama_token> tokens(512, 0);
    std::fill(tokens.begin() + 256, tokens.end(), 1);
    setenv("LLAMA_PLE_PREFETCH", "1", 1);
    setenv("LLAMA_PLE_PF_MIN", "1", 1);
    setenv("LLAMA_PLE_PF_MODE", "fault", 1);
    setenv("LLAMA_PLE_PF_AHEAD", "1000000", 1);
    for (int mode = 0; mode < 3; ++mode) {
        require(madvise(pages, size, MADV_DONTNEED) == 0, "reset anonymous page residency");
        *(volatile char *) pages = 0;
        setenv("LLAMA_PLE_PF_RECHECK", mode == 0 ? "0" : "1", 1);
        setenv("LLAMA_PLE_PF_COLD_THREADS", mode == 2 ? "0" : "2", 1);
        std::atomic<int> boosts{0};
        llama_log_set([](ggml_log_level, const char * text, void * data) {
            if (strstr(text, "[PLEPF] cold table") && strstr(text, "readers at chunk 1")) {
                static_cast<std::atomic<int> *>(data)->fetch_add(1);
            }
        }, &boosts);
        auto pf = llama_ple_prefetch::create(model, 256);
        require(pf != nullptr, "create page-touch prefetch");
        pf->hint(tokens.data(), tokens.size(), 0);
        require(pf->consume(tokens.size(), nullptr, 0), "consume cached prefix and cold suffix");
        pf->stop();
        llama_log_set(nullptr, nullptr);
        require(boosts.load() == (mode == 1 ? 1 : 0), "late cold pages need reader boost only with recheck enabled");
    }
    ggml_backend_buffer_free(table->buffer);
    ggml_free(ctx);
    munmap(pages, size);
    printf("OK: cached first chunk, cold second chunk, recheck boost and cold-reader opt-out\n");
#else
    printf("SKIP: page residency test needs Linux\n");
#endif
}

int main(int argc, char ** argv) {
    if (argc > 1 && strcmp(argv[1], "--ple-populate-direct") == 0) {
        try {
            for (size_t offset : { 0, 64, 4092 }) {
                for (size_t chunk : { 384, 128, 1 }) {
                    ple_hash_pages_case(true, true, true, chunk, 3, offset);
                }
            }
            printf("OK: direct mapped-page population bypasses scratch reads with exact pages, PTEs and partial-EOF fallback\n");
            return 0;
        } catch (const std::exception & e) {
            fprintf(stderr, "prefetch direct populate pages: FAIL: %s\n", e.what());
            return 1;
        }
    }
    if (argc > 1 && strcmp(argv[1], "--ple-populate-bulk") == 0) {
        try {
            for (size_t offset : { 0, 64, 4092 }) {
                for (size_t chunk : { 384, 128, 1 }) {
                    ple_hash_pages_case(true, true, true, chunk, 2, offset);
                }
            }
            printf("OK: bulk mapped-page population preserves exact bounded pages, PTEs, totals and EOF fallback\n");
            return 0;
        } catch (const std::exception & e) {
            fprintf(stderr, "prefetch bulk populate pages: FAIL: %s\n", e.what());
            return 1;
        }
    }
    if (argc > 1 && strcmp(argv[1], "--ple-populate-unaligned") == 0) {
        try {
            for (size_t offset : { 64, 4092 }) {
                for (size_t chunk : { 384, 128, 1 }) {
                    ple_hash_pages_case(true, true, true, chunk, true, offset);
                }
            }
            printf("OK: unaligned PLE tables populate exact mapped pages within bounded windows\n");
            return 0;
        } catch (const std::exception & e) {
            fprintf(stderr, "prefetch unaligned populate pages: FAIL: %s\n", e.what());
            return 1;
        }
    }
    if (argc > 1 && strcmp(argv[1], "--ple-populate-pages") == 0) {
        try {
            for (size_t chunk : { 384, 128, 1 }) {
                ple_hash_pages_case(true, true, true, chunk, true);
            }
            return 0;
        } catch (const std::exception & e) {
            fprintf(stderr, "prefetch populate pages: FAIL: %s\n", e.what());
            return 1;
        }
    }
    if (argc > 1 && strcmp(argv[1], "--ple-read-failure") == 0) {
        try {
            ple_hash_pages_case(false, false, true);
            return 0;
        } catch (const std::exception & e) {
            fprintf(stderr, "prefetch read failure: FAIL: %s\n", e.what());
            return 1;
        }
    }
    if (argc > 1 && strcmp(argv[1], "--ple-coalesce-window") == 0) {
        try {
            for (size_t chunk : { 384, 128, 1 }) {
                ple_hash_pages_case(true, true, true, chunk);
            }
            printf("OK: coalesced windows cover 1, 128 and 384 token ubatches\n");
            printf("OK: bounded coalescing reads only requested pages without gap padding or rereads\n");
            return 0;
        } catch (const std::exception & e) {
            fprintf(stderr, "prefetch coalesced window: FAIL: %s\n", e.what());
            return 1;
        }
    }
    if (argc > 1 && (strcmp(argv[1], "--ple-hash-pages") == 0 || strcmp(argv[1], "--ple-batch-pages") == 0)) {
        try {
            ple_hash_pages_case(strcmp(argv[1], "--ple-batch-pages") == 0);
            if (strcmp(argv[1], "--ple-batch-pages") == 0) {
                for (size_t ubatch : { 384, 128, 1 }) {
                    ple_hash_pages_case(true, true, false, ubatch);
                    ple_hash_pages_case(true, true, false, ubatch, 0, 0, true);
                }
                printf("OK: fault windows cover 1, 128 and 384 token ubatches without whole-chunk overreads\n");
                printf("OK: disk-backed fault windows keep future pages cold with mapped read-around disabled\n");
            }
            return 0;
        } catch (const std::exception & e) {
            fprintf(stderr, "prefetch hash pages: FAIL: %s\n", e.what());
            return 1;
        }
    }
    if (argc > 1 && strcmp(argv[1], "--ple-wide-rows") == 0) {
        try {
            ple_wide_rows_case();
            return 0;
        } catch (const std::exception & e) {
            fprintf(stderr, "prefetch wide rows: FAIL: %s\n", e.what());
            return 1;
        }
    }
    if (argc > 1 && strcmp(argv[1], "--ple-cold-recheck") == 0) {
        try {
            ple_cold_recheck_case();
            return 0;
        } catch (const std::exception & e) {
            llama_log_set(nullptr, nullptr);
            fprintf(stderr, "prefetch recheck: FAIL: %s\n", e.what());
            return 1;
        }
    }
    if (argc > 1 && strcmp(argv[1], "--mask-capacity") == 0) {
        try {
            deferred_mask_capacity_case();
            return 0;
        } catch (const std::exception & e) {
            fprintf(stderr, "mask capacity: FAIL: %s\n", e.what());
            return 1;
        }
    }
    const bool scratch_views = argc > 1 && strcmp(argv[1], "--scratch-views") == 0;
    const bool resident = argc > 1 && strcmp(argv[1], "--resident") == 0;
    const bool scalar_only = argc > 1 && strcmp(argv[1], "--scalar-only") == 0;
    const bool device_bytes = argc > 1 && strcmp(argv[1], "--device-bytes") == 0;
    const bool exact_sink = argc > 1 && strcmp(argv[1], "--exact-sink") == 0;
    const bool compact = argc > 1 && strcmp(argv[1], "--compact") == 0;
    const int tmp_arg = resident || scalar_only || compact ? 2 : 1;
    const char * tmp_dir = argc > tmp_arg ? argv[tmp_arg] : (getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp");
    ggml_backend_load_all();
    llama_backend_init();
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level == GGML_LOG_LEVEL_ERROR) g_last_error = text;
    }, nullptr);
    llama_model * model = make_model(resident);
    if (!model) {
        fprintf(stderr, "model init failed\n");
        return 1;
    }

    const sj_cfg s44  = { "4/4",        4, 4, GGML_TYPE_F32 };
    if (exact_sink) {
        int result = 0;
        try {
            exact_sink_case(model);
        } catch (const std::exception & e) {
            fprintf(stderr, "exact sink: FAIL: %s\n", e.what());
            result = 1;
        }
        llama_model_free(model);
        llama_backend_free();
        return result;
    }
    if (compact) {
        int result = 0;
        try {
            compact_state_case(model, s44, tmp_dir);
        } catch (const std::exception & e) {
            fprintf(stderr, "compact state: FAIL: %s\n", e.what());
            result = 1;
        }
        llama_model_free(model);
        llama_backend_free();
        return result;
    }
    if (scratch_views) {
        int result = 0;
        try {
            scratch_views_case(model, s44);
            sj_cfg plain = s44;
            plain.plain = true;
            scratch_views_case(model, plain);
            printf("OK: empty future layers and unused cell suffixes preserve live bytes; multiple streams refuse borrowing\n");
        } catch (const std::exception & e) {
            fprintf(stderr, "scratch views: FAIL: %s\n", e.what());
            result = 1;
        }
        llama_model_free(model);
        llama_backend_free();
        return result;
    }
    if (device_bytes) {
        int result = 0;
        try {
            device_bytes_case(model);
            printf("OK: packed TQ6/F16 sink device snapshot, reuse and clear, exact host bytes\n");
        } catch (const std::exception & e) {
            fprintf(stderr, "device bytes: FAIL: %s\n", e.what());
            result = 1;
        }
        llama_model_free(model);
        llama_backend_free();
        return result;
    }
    if (resident) {
        int fails = 0;
        try {
            for (uint32_t cells : { 768u, 1024u }) {
                sj_cfg c = s44;
                c.name = cells == 768 ? "4/4 resident768" : "4/4 resident1024";
                c.resident = cells;
                fails += run_case(model, c, {}, tmp_dir);
            }
        } catch (const std::exception & e) {
            fprintf(stderr, "resident state: FAIL: %s\n", e.what());
            ++fails;
        }
        printf("%s: resident state round trips, split2/full2, GPU seals and continuation\n", fails ? "FAIL" : "OK");
        llama_model_free(model);
        llama_backend_free();
        return fails ? 1 : 0;
    }
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

    const std::string wpath1 = write_transform(tmp_dir, "test-sjkvarn-state-w1.sjwu", 1);
    const std::string wpath2 = write_transform(tmp_dir, "test-sjkvarn-state-w2.sjwu", 2);
    sj_cfg w1 = s44; w1.name = "4/4 wush"; w1.transform = wpath1.c_str();
    sj_cfg w2 = with(s44, "other transform"); w2.transform = wpath2.c_str();
    sj_cfg w1t = s33t; w1t.name = "3/3t wush"; w1t.transform = wpath1.c_str();

    const std::vector<std::pair<sj_cfg, std::vector<sj_cfg>>> cases = {
        { s44,   { s33, s33t, s32t, tier, m_tail, m_tmax, m_stg, m_plain, w1 } },
        { w1,    { s44, w2 } },
        { w1t,   { s33t } },
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
    int cases_run = 0;
    const bool multi_only = getenv("SJ_TEST_MULTI_ONLY") != nullptr; // debugging: only the multi-sequence checks
    for (const auto & c : cases) {
        if (multi_only) { break; }
        if (scalar_only && strcmp(c.first.name, s44.name) != 0) { continue; }
        if (getenv("SJ_TEST_CASE") && strcmp(getenv("SJ_TEST_CASE"), c.first.name) != 0) { continue; } // debugging
        ++cases_run;
        fails += run_case(model, c.first, c.second, tmp_dir);
    }
    if (!multi_only) { plain_mixed_control(model); }
    int fails_multi = 0;
    int multi_run = 0;
    for (const auto & c : { s44, s33t, s32t }) {
        if (scalar_only && c.body != GGML_TYPE_F32) { continue; }
        ++multi_run;
        fails_multi += run_multi(model, c, false);
    }
    ++multi_run;
    fails_multi += run_multi(model, s44, true);
    if (!scalar_only) {
        ++multi_run;
        fails_multi += run_multi(model, w1t, false); // transform with one FA node per stream
    }
    int fails_mrope = 0;
    int mrope_run = 0;
    if (!multi_only) {
        llama_model * model_mrope = make_model(false, LLM_ARCH_QWEN2VL);
        if (!model_mrope) {
            fprintf(stderr, "M-RoPE model init failed\n");
            mrope_run = 1;
            fails_mrope = 1;
        } else {
            ++mrope_run;
            fails_mrope += run_mrope(model_mrope, s44, 100);
            if (!scalar_only || f16.body == GGML_TYPE_F32) {
                ++mrope_run;
                fails_mrope += run_mrope(model_mrope, f16, 200);
            }
            llama_model_free(model_mrope);
        }
    }
    printf("%s: %d/%d cases passed, multi-sequence %d/%d, M-RoPE %d/%d\n", fails || fails_multi || fails_mrope ? "FAIL" : "OK",
           cases_run - fails, cases_run, multi_run - fails_multi, multi_run, mrope_run - fails_mrope, mrope_run);
    fails += fails_multi + fails_mrope;
    llama_model_free(model);
    llama_backend_free();
    return fails ? 1 : 0;
}
