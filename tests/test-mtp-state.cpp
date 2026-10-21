// Exercise real MTP cache restoration with a generated CPU model, split prefill, deferred rows, and adaptive draft widths.
// Restore into a fresh driver and a different destination sequence.
// --save-model PATH exports the fixture for other state tests.
// --bookkeep-output-only checks the experimental sampler output snapshot using only the CPU fixture.
// --bookkeep-sampler-only emits an exact common-sampler digest for fresh-process environment comparisons.
// --reserve-probe CTX PARALLEL [MODEL] measures production graph geometry without allocating weights or running kernels.
// --device-feedback[-cuda]-only and --qwen4exp-chain[-cuda]-only exercise the generated Qwen4exp helper.
// --qwen4exp-compact[-cuda]-only additionally uses compact QSA and TQ5/TURBO4 helper KV.
// --qwen4exp-pool-only checks incremental pool metadata with the differential validator enabled.
// --qwen4exp-pool-store[-cuda]-only checks specialized KV-only pool inputs and full-graph continuation.
// --qwen4exp-full-head[-cuda]-only checks batched full-head proposals and exact sequential suffix repair.

#include "common.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "llama.h"

#include "../ggml/src/ggml-impl.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "../src/llama-arch.h"
#include "../src/llama-batch.h"
#include "../src/llama-context.h"
#include "../src/llama-ext.h"
#include "../src/llama-graph.h"
#include "../src/llama-kv-cache.h"
#include "../src/llama-memory.h"
#include "../src/llama-memory-hybrid.h"
#include "../src/llama-memory-hybrid-idx.h"
#include "../src/llama-mtp-chain-sample.h"
#include "../src/llama-model.h"
#include "../src/llama-mmap.h"
#include "../src/llama-model-saver.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#ifndef _WIN32
#include <sys/resource.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

static constexpr uint32_t n_vocab = 32, n_embd = 256;
static size_t scratch_borrows = 0, scratch_releases = 0, scratch_fallbacks = 0, compute_shares = 0;
static size_t borrowed_host_pins = 0, borrowed_host_pin_failures = 0;
static bool bookkeep_sequential = false;
static size_t half_vocab_maps = 0;
static size_t half_vocab_compact_heads = 0;
static size_t pool_store_fills = 0;

static void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// Check the rendered owning batch, rather than the source row buffer. This
// catches borrowed-row reuse that otherwise changes inputs when batch width changes.
static uint64_t check_teacher_batch(const llama_batch_ext & batch, size_t width, uint64_t digest) {
    require(batch.n_embd == width, "teacher batch embedding width");
    auto observe = [&](const void * data, size_t size) {
        const auto * p = static_cast<const uint8_t *>(data);
        for (size_t i = 0; i < size; ++i) { digest = (digest ^ p[i]) * UINT64_C(1099511628211); }
    };
    for (const auto & token : batch.tokens) {
        require(token.has_embd && token.embd_off + width <= batch.embd.size(), "teacher row is retained");
        const auto * row = batch.embd.data() + token.embd_off;
        for (size_t j = 0; j < width; ++j) {
            require(row[j] == float((int) ((j + token.pos[0]) % 19) - 9) / 19.0f,
                    "serialized teacher row matches its logical position");
        }
        observe(&token.id, sizeof(token.id));
        observe(token.pos.data(), sizeof(llama_pos) * token.pos.size());
        observe(row, width * sizeof(float));
    }
    return digest;
static void embedding_input_u64(std::vector<uint8_t> & bytes, uint64_t value) {
    for (int j = 0; j < 8; ++j) { bytes.push_back(uint8_t(value >> (8*j))); }
}

static uint64_t embedding_input_digest(const std::vector<uint8_t> & bytes) {
    uint64_t result = UINT64_C(14695981039346656037);
    for (uint8_t value : bytes) { result = (result ^ value)*UINT64_C(1099511628211); }
    return result;
}

// Check the rendered owning batch, not the non-owning common_batch row views.
// The transcript contains actual row data and metadata without batch boundaries,
// so concatenated singleton materializations compare exactly with merged rows.
static llama_batch_ext * checked_embedding_batch(common_batch & input, const std::vector<float> & expected,
        size_t width, std::vector<uint8_t> * transcript = nullptr, size_t transcript_rows = SIZE_MAX) {
    const size_t rows = input.size();
    require(width > 0 && expected.size() >= rows*width, "embedding oracle has every intended logical row");
    auto * result = input.get();
    require(result && result->tokens.size() == rows && result->n_embd == width && result->embd.size() == rows*width,
            "rendered embedding batch owns the complete row matrix");
    for (size_t row = 0; row < rows; ++row) {
        const auto & want = input.tokens[row];
        const auto & actual = result->tokens[row];
        require(actual.has_embd && actual.embd_off == row*width && actual.embd_off + width <= result->embd.size(),
                "rendered embedding row has its own bounded offset");
        const float * data = result->embd.data() + actual.embd_off;
        require(data != want.embd.data && std::memcmp(data, expected.data() + row*width, width*sizeof(float)) == 0,
                "actual owning batch preserves every intended embedding bit after get");
        std::vector<llama_seq_id> seq_ids(actual.seq_ids.begin(), actual.seq_ids.end());
        std::sort(seq_ids.begin(), seq_ids.end());
        std::vector<llama_seq_id> expected_seq_ids { want.seq_id };
        expected_seq_ids.insert(expected_seq_ids.end(), want.seq_ids_extra.begin(), want.seq_ids_extra.end());
        std::sort(expected_seq_ids.begin(), expected_seq_ids.end());
        expected_seq_ids.erase(std::unique(expected_seq_ids.begin(), expected_seq_ids.end()), expected_seq_ids.end());
        require(actual.id == want.id && actual.pos == want.pos && actual.output == want.output &&
                    actual.decision_order == want.decision_order && seq_ids == expected_seq_ids,
                "actual owning batch preserves token, position, sequence and output metadata");
        if (transcript && row < transcript_rows) {
            embedding_input_u64(*transcript, uint32_t(actual.id));
            for (llama_pos pos : actual.pos) { embedding_input_u64(*transcript, uint32_t(pos)); }
            embedding_input_u64(*transcript, actual.output);
            embedding_input_u64(*transcript, uint32_t(actual.decision_order));
            embedding_input_u64(*transcript, seq_ids.size());
            for (llama_seq_id seq : seq_ids) { embedding_input_u64(*transcript, uint32_t(seq)); }
            embedding_input_u64(*transcript, width);
            for (size_t j = 0; j < width; ++j) {
                uint32_t bits;
                std::memcpy(&bits, data + j, sizeof(bits));
                for (int k = 0; k < 4; ++k) { transcript->push_back(uint8_t(bits >> (8*k))); }
            }
        }
    }
    return result;
}

static void tensor_data(ggml_tensor * tensor, void *) {
    std::mt19937 rng(std::hash<std::string>()(tensor->name));
    std::normal_distribution<float> dist(0.0f, 0.1f);
    require(tensor->type == GGML_TYPE_F32, "fixture tensor type");
    std::vector<float> data(ggml_nelements(tensor));
    for (auto & value : data) {
        value = dist(rng);
    }
    ggml_backend_tensor_set(tensor, data.data(), 0, ggml_nbytes(tensor));
}

struct model_geometry {
    uint32_t vocab = n_vocab, embd = n_embd, layers = 3, ff = 128;
    uint32_t heads = 1, heads_kv = 1, head_dim = n_embd, attention_interval = 2;
    uint32_t ssm_inner = 128, ssm_rank = 1, ssm_groups = 1;
    uint32_t rope_dim = n_embd;
    uint32_t kpool = 0;
    uint32_t indexer_top_k = 131072;
};

static llama_model * make_model(bool no_alloc = false, const model_geometry & g = {}, llm_arch arch = LLM_ARCH_QWEN35, ggml_backend_dev_t device = nullptr) {
    gguf_context * meta = gguf_init_empty();
    llama_model_saver ms(arch, meta);
    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE, llm_arch_name(arch));
    ms.add_kv(LLM_KV_VOCAB_SIZE, g.vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH, uint32_t(2048));
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH, g.embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT, g.layers);
    ms.add_kv(LLM_KV_NEXTN_PREDICT_LAYERS, uint32_t(1));
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, g.ff);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, g.heads);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, g.heads_kv);
    ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH, g.head_dim);
    ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH, g.head_dim);
    ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT, g.rope_dim);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, 1e-5f);
    ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS, std::vector<uint32_t>({64, 64, 64, 64}));
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL, g.attention_interval);
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL, uint32_t(4));
    ms.add_kv(LLM_KV_SSM_INNER_SIZE, g.ssm_inner);
    ms.add_kv(LLM_KV_SSM_STATE_SIZE, uint32_t(128));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK, g.ssm_rank);
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT, g.ssm_groups);
    if (arch == LLM_ARCH_QWEN4EXP) {
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT, uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_LOW_RANK, uint32_t(8));
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, uint32_t(1));
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, g.head_dim);
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K, g.indexer_top_k);
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS, std::vector<uint32_t>(g.layers, g.kpool));
        ms.add_kv(LLM_KV_EXPERT_COUNT, uint32_t(4));
        ms.add_kv(LLM_KV_EXPERT_USED_COUNT, uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, g.ff);
        ms.add_kv(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, g.ff);
    }
    ms.add_kv(LLM_KV_TOKENIZER_MODEL, "test");
    std::vector<std::string> tokens;
    for (uint32_t i = 0; i < g.vocab; ++i) {
        tokens.push_back("tok_" + std::to_string(i));
    }
    ms.add_kv(LLM_KV_TOKENIZER_LIST, tokens);
    ms.add_kv(LLM_KV_TOKENIZER_SCORES, std::vector<float>(g.vocab, 0.0f));
    llama_model_params mp = llama_model_default_params();
    ggml_backend_dev_t devices[] = { device, nullptr };
    mp.n_gpu_layers = device ? 999 : 0;
    mp.devices = devices;
    mp.load_mtp = true;
    mp.no_alloc = no_alloc;
    llama_model * model = llama_model_init_from_user(meta, tensor_data, nullptr, mp);
    gguf_free(meta);
    require(model != nullptr, "create fixture model");
    return model;
}

struct cache_config {
    const char * name;
    ggml_type k, v;
    uint32_t sj_k = 0, sj_v = 0;
    ggml_type body = GGML_TYPE_F32;
};

struct contexts {
    llama_context * tgt = nullptr;
    llama_context * dft = nullptr;
    common_speculative * spec = nullptr;

    contexts(llama_model * model, const cache_config & config, uint32_t batch_size = 128,
            common_speculative_type type = COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE, uint32_t ubatch_size = 0, uint32_t parallel = 2, uint32_t context_per_seq = 4096, llama_model * draft_model = nullptr,
            const char * draft_vocab_map = nullptr, bool backend_sampling = true,
            ggml_backend_sched_eval_callback eval_callback = nullptr, void * eval_user_data = nullptr, int draft_max = 4) {
        llama_context_params cp = llama_context_default_params();
        cp.cb_eval = eval_callback;
        cp.cb_eval_user_data = eval_user_data;
        cp.n_batch = batch_size;
        cp.n_ubatch = ubatch_size ? ubatch_size : batch_size;
        cp.n_seq_max = parallel;
        cp.n_ctx = context_per_seq * parallel;
        cp.kv_unified = false;
        cp.n_rs_seq = 4;
        cp.n_threads = cp.n_threads_batch = 2;
        cp.offload_kqv = model->n_devices() > 0;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.type_k = config.k;
        cp.type_v = config.v;
        if (config.sj_k) {
            cp.sj_kvarn_bits_k = config.sj_k;
            cp.sj_kvarn_bits_v = config.sj_v;
            cp.sj_kvarn_body_type = config.body;
            cp.sj_kvarn_staging_type = GGML_TYPE_TQ6_0;
            cp.sj_kvarn_sink_type = GGML_TYPE_F16;
            cp.sj_kvarn_sink = 128;
            cp.sj_kvarn_tail = 128;
            cp.sj_kvarn_tail_max = 256;
        }
        tgt = llama_init_from_model(model, cp);
        require(tgt != nullptr, "create target context");
        cp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        cp.ctx_other = tgt;
        cp.draft_vocab_map = draft_vocab_map;
        cp.n_rs_seq = 0;
        cp.sj_kvarn_bits_k = cp.sj_kvarn_bits_v = 0;
        if (config.sj_k) {
            cp.type_k = COMMON_SJKVARN_MTP_DRAFT_KV[0];
            cp.type_v = COMMON_SJKVARN_MTP_DRAFT_KV[1];
        }
        dft = llama_init_from_model(draft_model ? draft_model : model, cp);
        require(dft != nullptr, "create draft context");
        common_params_speculative params;
        params.types = { type };
        params.draft.ctx_tgt = tgt;
        params.draft.ctx_dft = dft;
        params.draft.n_max = draft_max;
        params.draft.n_min_adaptive = 3;
        params.draft.p_min = 0.0f;
        params.draft.chain = !bookkeep_sequential;
        params.draft.backend_sampling = backend_sampling;
        spec = common_speculative_init(params, parallel);
        require(spec != nullptr, "create MTP driver");
    }

    ~contexts() {
        common_speculative_free(spec);
        llama_free(dft);
        llama_free(tgt);
    }
};

static llama_token token_at(llama_pos pos) {
    return (pos * 7 + 3) % n_vocab;
}

static void process(contexts & ctx, const llama_tokens & tokens, llama_pos pos, llama_seq_id seq) {
    common_batch batch(ctx.tgt);
    for (llama_token token : tokens) {
        batch.add(token, pos++, seq, true);
    }
    require(llama_process(ctx.tgt, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0, "target decode");
    require(common_speculative_process(ctx.spec, batch), "MTP process");
}

static void prefill(contexts & ctx, llama_pos from, llama_pos to, llama_seq_id seq) {
    while (from < to) {
        llama_tokens tokens;
        const llama_pos end = std::min(from + (llama_pos) llama_n_batch(ctx.tgt), to);
        for (llama_pos pos = from; pos < end; ++pos) {
            tokens.push_back(token_at(pos));
        }
        process(ctx, tokens, from, seq);
        from = end;
    }
}

struct snapshot {
    std::vector<uint8_t> tgt, dft, spec;
};

static std::vector<uint8_t> save(llama_context * ctx, llama_seq_id seq) {
    const auto flags = LLAMA_STATE_SEQ_FLAGS_KEEP_LAYOUT;
    std::vector<uint8_t> state(llama_state_seq_get_size_ext(ctx, seq, flags));
    require(!state.empty(), "sequence state exists");
    require(llama_state_seq_get_data_ext(ctx, state.data(), state.size(), seq, flags) == state.size(), "save sequence state");
    return state;
}

static snapshot save(contexts & ctx, llama_seq_id seq) {
    snapshot state { save(ctx.tgt, seq), save(ctx.dft, seq), {} };
    require(common_speculative_get_state(ctx.spec, seq, state.spec) && !state.spec.empty(), "MTP snapshot contains carry and adaptive state");
    return state;
}

static void check_rejected_state(contexts & ctx, llama_seq_id seq, const std::vector<uint8_t> & valid) {
    for (int test = 0; test < 3; ++test) {
        auto invalid = valid;
        if (test == 0) {
            invalid.clear();
        } else if (test == 1) {
            invalid.pop_back();
        } else {
            invalid.front() ^= 0xff;
        }
        require(!common_speculative_restore_state(ctx.spec, seq, invalid), "invalid MTP state is rejected");
        std::vector<uint8_t> after;
        require(common_speculative_get_state(ctx.spec, seq, after) && after == valid, "invalid MTP state leaves sequence unchanged");
    }
}

static void restore(contexts & ctx, const snapshot & state, llama_seq_id seq) {
    const auto flags = LLAMA_STATE_SEQ_FLAGS_KEEP_LAYOUT;
    require(llama_state_seq_set_data_ext(ctx.tgt, state.tgt.data(), state.tgt.size(), seq, flags) == state.tgt.size(), "restore target state");
    require(llama_state_seq_set_data_ext(ctx.dft, state.dft.data(), state.dft.size(), seq, flags) == state.dft.size(), "restore draft state");
    require(common_speculative_restore_state(ctx.spec, seq, state.spec), "restore MTP host state");
    std::vector<uint8_t> round_trip;
    require(common_speculative_get_state(ctx.spec, seq, round_trip) && round_trip == state.spec, "MTP snapshot round trip across sequences");
    check_rejected_state(ctx, seq, round_trip);
}

struct schedule {
    std::vector<size_t> widths;
    std::vector<llama_tokens> tokens;

    bool operator==(const schedule & other) const {
        return widths == other.widths && tokens == other.tokens;
    }
};

static void require_schedule(const schedule & actual, const schedule & expected, const char * message) {
    if (actual == expected) {
        return;
    }
    fprintf(stderr, "%s: reference widths", message);
    for (size_t width : expected.widths) {
        fprintf(stderr, " %zu", width);
    }
    fprintf(stderr, "; restored widths");
    for (size_t width : actual.widths) {
        fprintf(stderr, " %zu", width);
    }
    fprintf(stderr, "\n");
    for (size_t round = 0; round < std::min(actual.tokens.size(), expected.tokens.size()); ++round) {
        const auto & got = actual.tokens[round];
        const auto & ref = expected.tokens[round];
        if (got != ref) {
            fprintf(stderr, "first differing draft round %zu: reference", round);
            for (llama_token token : ref) {
                fprintf(stderr, " %d", token);
            }
            fprintf(stderr, "; restored");
            for (llama_token token : got) {
                fprintf(stderr, " %d", token);
            }
            fprintf(stderr, "\n");
            break;
        }
    }
    require(false, message);
}

static schedule generate(contexts & ctx, llama_seq_id seq, llama_pos & pos, int first, int count, bool sampled_chain = false,
        bool require_chain_decode = true) {
    schedule result;
    common_params_sampling sampling;
    sampling.temp = sampled_chain ? 1.0f : 0.0f;
    if (sampled_chain) {
        sampling.seed = 9127;
        sampling.top_k = 10;
        sampling.top_p = 0.95f;
        sampling.min_p = 0.0f;
        sampling.min_keep = 1;
        sampling.dynatemp_range = 0.0f;
        sampling.samplers = { COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_TOP_P,
            COMMON_SAMPLER_TYPE_MIN_P, COMMON_SAMPLER_TYPE_TEMPERATURE };
    }
    for (int round = first; round < first + count; ++round) {
        llama_tokens prompt, draft;
        std::vector<std::vector<llama_token_data>> proposal_q;
        auto & dp = common_speculative_get_draft_params(ctx.spec, seq);
        dp = {};
        dp.drafting = true;
        dp.pos0 = pos;
        dp.id_last = token_at(pos);
        dp.prompt = &prompt;
        dp.result = &draft;
        dp.sampling = &sampling;
        dp.result_q = sampled_chain ? &proposal_q : nullptr;
        llama_synchronize(ctx.dft);
        const auto before = llama_perf_context(ctx.dft);
        common_speculative_draft(ctx.spec);
        require(!draft.empty(), "MTP drafted tokens");
        if (sampled_chain) {
            llama_synchronize(ctx.dft);
            const auto after = llama_perf_context(ctx.dft);
            require(!require_chain_decode || (after.n_eval == before.n_eval && after.n_p_eval >= before.n_p_eval + (int) draft.size()),
                    "sampled chain drafts in one multi-token decode without sequential single-token evaluations");
            require(proposal_q.size() == draft.size() &&
                    std::all_of(proposal_q.begin(), proposal_q.end(), [](const std::vector<llama_token_data> & q) { return !q.empty(); }),
                    "sampled chain returns a proposal distribution for every token");
        }
        result.widths.push_back(draft.size());
        result.tokens.push_back(draft);
        llama_tokens verify { dp.id_last };
        verify.insert(verify.end(), draft.begin(), draft.end());
        process(ctx, verify, pos, seq);
        // The sequential BOOKKEEP fixture also rejects the entire draft prefix.
        // The current input still commits one row; deferred speculative rows must
        // not survive to the next draft or to a saved/restored continuation.
        // Six consecutive full accepts are required to climb from width 3.
        // Follow them with partial and zero acceptance rather than resetting
        // the streak before this fixture has exercised width 4.
        const int accepted = bookkeep_sequential
            ? (round % 8 == 7 ? 0 : (round % 8 == 6 ? 1 : (int) draft.size()))
            : (round % 7 == 6 ? 1 : (int) draft.size());
        common_speculative_accept(ctx.spec, seq, accepted);
        pos += accepted + 1;
        require(llama_memory_seq_rm(llama_get_memory(ctx.tgt), seq, pos, -1), "target rollback");
        require(llama_memory_seq_rm(llama_get_memory(ctx.dft), seq, pos, -1), "draft rollback");
    }
    return result;
}

struct prefill_env_restore {
    std::string overlap = common_get_env("GGML_MTP_PREFILL_OVERLAP");
    std::string async = common_get_env("GGML_MTP_PREFILL_ASYNC");
    std::string device = common_get_env("GGML_MTP_PREFILL_DEVICE");
    std::string tight = common_get_env("LLAMA_MTP_RESERVE_TIGHT");
    std::string scratch = common_get_env("LLAMA_MTP_PREFILL_SCRATCH");
    std::string shared = common_get_env("LLAMA_SHARED_COMPUTE");
    std::string kv_only = common_get_env("LLAMA_MTP_UPDATE_KV_ONLY");
    std::string pool_store_only = common_get_env("LLAMA_MTP_POOL_STORE_ONLY");
    std::string eh_batch = common_get_env("LLAMA_MTP_EH_BATCH");

    ~prefill_env_restore() {
        common_set_env("GGML_MTP_PREFILL_OVERLAP", overlap);
        common_set_env("GGML_MTP_PREFILL_ASYNC", async);
        common_set_env("GGML_MTP_PREFILL_DEVICE", device);
        common_set_env("LLAMA_MTP_RESERVE_TIGHT", tight);
        common_set_env("LLAMA_MTP_PREFILL_SCRATCH", scratch);
        common_set_env("LLAMA_SHARED_COMPUTE", shared);
        common_set_env("LLAMA_MTP_UPDATE_KV_ONLY", kv_only);
        common_set_env("LLAMA_MTP_POOL_STORE_ONLY", pool_store_only);
        common_set_env("LLAMA_MTP_EH_BATCH", eh_batch);
    }

    void set(bool use_overlap, bool use_async, bool use_device, bool use_tight = false, bool use_scratch = false) {
        common_set_env("GGML_MTP_PREFILL_OVERLAP", use_overlap ? "1" : "0");
        common_set_env("GGML_MTP_PREFILL_ASYNC", use_async ? "1" : "0");
        common_set_env("GGML_MTP_PREFILL_DEVICE", use_device ? "1" : "0");
        common_set_env("LLAMA_MTP_RESERVE_TIGHT", use_tight ? "1" : "0");
        common_set_env("LLAMA_MTP_PREFILL_SCRATCH", use_scratch ? "1" : "0");
    }
};

struct feedback_env_restore {
    std::string feedback = common_get_env("LLAMA_MTP_DEVICE_FEEDBACK");
    std::string deferred = common_get_env("LLAMA_SPEC_DEFER_CATCHUP");
    std::string chain = common_get_env("LLAMA_MTP_QWEN4EXP_CHAIN");
    std::string compact = common_get_env("LLAMA_MTP_QSA_COMPACT");
    std::string sampled = common_get_env("LLAMA_SPEC_CHAIN_SAMPLED");
    std::string pq = common_get_env("LLAMA_SPEC_PQ");
    std::string full_head = common_get_env("LLAMA_MTP_CHAIN_FULL_HEAD");
    std::string test_ids = common_get_env("LLAMA_MTP_CHAIN_TEST_IDS");
    std::string vocab = common_get_env("LLAMA_SPEC_DRAFT_VOCAB");
    std::string compact_head = common_get_env("LLAMA_DRAFT_VOCAB_COMPACT");
    std::string compact_head_type = common_get_env("LLAMA_DRAFT_VOCAB_COMPACT_TYPE");
    std::string draft_temp_mult = common_get_env("LLAMA_SPEC_DRAFT_TEMP_MULT");
    ~feedback_env_restore() {
        common_set_env("LLAMA_MTP_DEVICE_FEEDBACK", feedback);
        common_set_env("LLAMA_SPEC_DEFER_CATCHUP", deferred);
        common_set_env("LLAMA_MTP_QWEN4EXP_CHAIN", chain);
        common_set_env("LLAMA_MTP_QSA_COMPACT", compact);
        common_set_env("LLAMA_SPEC_CHAIN_SAMPLED", sampled);
        common_set_env("LLAMA_SPEC_PQ", pq);
        common_set_env("LLAMA_MTP_CHAIN_FULL_HEAD", full_head);
        common_set_env("LLAMA_MTP_CHAIN_TEST_IDS", test_ids);
        common_set_env("LLAMA_SPEC_DRAFT_VOCAB", vocab);
        common_set_env("LLAMA_DRAFT_VOCAB_COMPACT", compact_head);
        common_set_env("LLAMA_DRAFT_VOCAB_COMPACT_TYPE", compact_head_type);
        common_set_env("LLAMA_SPEC_DRAFT_TEMP_MULT", draft_temp_mult);
    }
};

static void run_device_feedback(bool cuda) {
    prefill_env_restore prefill_env;
    prefill_env.set(false, false, false);
    feedback_env_restore feedback_env;
    common_set_env("LLAMA_SPEC_DEFER_CATCHUP", "0");
    common_set_env("LLAMA_MTP_QWEN4EXP_CHAIN", "0");
    auto * device = cuda ? ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) : nullptr;
    require(!cuda || device, "feedback GPU backend is available");
    llama_model_ptr model(make_model(false, {}, LLM_ARCH_QWEN4EXP, device));
    const cache_config config { "device feedback", GGML_TYPE_Q8_0, GGML_TYPE_Q8_0 };
    {
        common_set_env("LLAMA_MTP_DEVICE_FEEDBACK", "0");
        contexts reference(model.get(), config, 8, COMMON_SPECULATIVE_TYPE_DRAFT_MTP, 8, 1, 512);
        require(!llama_mtp_device_feedback_init(reference.dft), "feedback is opt-in");
        common_set_env("LLAMA_MTP_DEVICE_FEEDBACK", "1");
        contexts feedback(model.get(), config, 8, COMMON_SPECULATIVE_TYPE_DRAFT_MTP, 8, 1, 512);
        require(llama_mtp_device_feedback_init(feedback.dft), "Qwen4exp backend supports feedback");
        // This direct op comparison covers all vocabulary logits. The separate
        // common-driver checks below retain real backend top-k sampling.
        require(llama_set_sampler(reference.dft, 0, nullptr) && llama_set_sampler(feedback.dft, 0, nullptr),
                "detach backend samplers for full-vocabulary feedback comparison");
        const size_t width = llama_model_n_embd_out(model.get());
        std::vector<float> hidden(width), poison(width, 1000.0f), saved_hidden;
        for (size_t j = 0; j < width; ++j) {
            hidden[j] = float((int) (j % 17) - 8) / 17.0f;
        }
        llama_token token = 0;
        llama_pos pos = 0;
        int32_t n_seq_id = 1;
        llama_seq_id seq = 0;
        llama_seq_id * seq_id = &seq;
        int8_t output = 1;
        llama_batch batch = { 1, &token, hidden.data(), &pos, &n_seq_id, &seq_id, &output };
        require(llama_mtp_device_feedback_decode(feedback.dft, batch, true) == -4, "feedback rejects an unseeded previous row");
        std::vector<uint8_t> reference_state, feedback_state;
        for (int round = 0; round < 3; ++round) {
            if (round == 2) {
                require(llama_state_seq_set_data_ext(reference.dft, reference_state.data(), reference_state.size(), 0,
                            LLAMA_STATE_SEQ_FLAGS_KEEP_LAYOUT) == reference_state.size(), "restore reference helper state");
                require(llama_state_seq_set_data_ext(feedback.dft, feedback_state.data(), feedback_state.size(), 0,
                            LLAMA_STATE_SEQ_FLAGS_KEEP_LAYOUT) == feedback_state.size(), "restore feedback helper state");
                hidden = saved_hidden;
            }
            for (int step = 0; step < 4; ++step) {
                pos = (round == 0 ? 0 : 4) + step;
                token = token_at(pos);
                batch.embd = hidden.data();
                require(llama_decode(reference.dft, batch) == 0, "reference helper step");
                const float * expected = llama_get_logits_ith(reference.dft, 0);
                batch.embd = step == 0 ? hidden.data() : poison.data();
                require(llama_mtp_device_feedback_decode(feedback.dft, batch, step > 0) == 0, "feedback helper step");
                const float * actual = llama_get_logits_ith(feedback.dft, 0);
                require(std::memcmp(expected, actual, n_vocab * sizeof(float)) == 0, "feedback logits are bit-identical with poisoned host continuation rows");
                const float * next = llama_get_embeddings_nextn_ith(reference.dft, 0);
                std::copy(next, next + width, hidden.begin());
                pos += 2;
                require(llama_mtp_device_feedback_decode(feedback.dft, batch, true) == -4, "feedback rejects a nonconsecutive position");
            }
            if (round == 0) {
                reference_state = save(reference.dft, 0);
                feedback_state = save(feedback.dft, 0);
                saved_hidden = hidden;
            }
        }
    }
    common_set_env("LLAMA_MTP_DEVICE_FEEDBACK", "0");
    contexts reference(model.get(), config, 16, COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE, 16, 1, 512);
    common_set_env("LLAMA_MTP_DEVICE_FEEDBACK", "1");
    contexts feedback(model.get(), config, 16, COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE, 16, 1, 512);
    prefill(reference, 0, 8, 0);
    prefill(feedback, 0, 8, 0);
    llama_pos reference_pos = 8, feedback_pos = 8;
    const schedule initial = generate(reference, 0, reference_pos, 0, 3);
    require_schedule(generate(feedback, 0, feedback_pos, 0, 3), initial, "feedback preserves sequential proposal tokens");
    const snapshot cached = save(feedback, 0);
    require(cached.spec == save(reference, 0).spec, "feedback preserves the accepted-target hidden sidecar");
    const schedule expected = generate(reference, 0, reference_pos, 3, 3);
    contexts resumed(model.get(), config, 16, COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE, 16, 1, 512);
    restore(resumed, cached, 0);
    require_schedule(generate(resumed, 0, feedback_pos, 3, 3), expected, "feedback resumes from CPU sidecar without device carry state");
    printf("PASS Qwen4exp %s sequential feedback: bit-identical logits, poisoned host continuation, repeated rounds, state restore, target sidecar, position guards\n", cuda ? "GPU" : "CPU");
}

static void run_qwen4exp_pool(bool cuda = false, bool store_only = false) {
    require((common_get_env("LLAMA_MTP_POOL_INCREMENTAL") == "1" ||
                (!store_only && common_get_env("LLAMA_MTP_POOL_INCREMENTAL") == "0")) &&
                common_get_env("LLAMA_MTP_POOL_VALIDATE") == "1",
            "pool regression requires explicit LLAMA_MTP_POOL_INCREMENTAL=0/1 and LLAMA_MTP_POOL_VALIDATE=1");
    prefill_env_restore prefill_env;
    prefill_env.set(false, false, false);
    feedback_env_restore feedback_env;
    // fork: set these before any model/context graph is made; both gates are cached per process.
    common_set_env("LLAMA_MTP_UPDATE_KV_ONLY", "1");
    common_set_env("LLAMA_MTP_POOL_STORE_ONLY", store_only ? "1" : "0");
    common_set_env("LLAMA_MTP_EH_BATCH", "0");
    common_set_env("LLAMA_MTP_DEVICE_FEEDBACK", "0");
    common_set_env("LLAMA_MTP_QWEN4EXP_CHAIN", "0");
    common_set_env("LLAMA_MTP_QSA_COMPACT", "0");
    auto * device = cuda ? ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) : nullptr;
    require(!cuda || device, "pool GPU backend is available");
    model_geometry geometry;
    geometry.kpool = 4;
    geometry.indexer_top_k = 8;
    llama_model_ptr model(make_model(false, geometry, LLM_ARCH_QWEN4EXP, device));
    auto make_context = [&](uint32_t batch_size = 8) {
        auto cp = llama_context_default_params();
        cp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        cp.n_ctx = 128;
        cp.n_batch = cp.n_ubatch = batch_size;
        cp.n_seq_max = 2;
        cp.kv_unified = true;
        cp.n_threads = cp.n_threads_batch = 2;
        cp.offload_kqv = cuda;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.type_k = cp.type_v = GGML_TYPE_Q8_0;
        llama_context_ptr ctx(llama_init_from_model(model.get(), cp));
        require(ctx != nullptr, "create pool helper");
        return ctx;
    };
    auto pool_memory = [](llama_context * ctx) {
        auto * mem = dynamic_cast<llama_memory_hybrid_idx *>(llama_get_memory(ctx));
        require(mem && mem->get_mem_idx() && mem->get_kpool() == 4 && mem->get_kpool_by_order(),
                "Qwen4exp helper has pools of four cells in sequence order");
        require(mem->get_mem_idx()->get_n_stream() == 1, "pool helper shares a unified indexer stream");
        return mem;
    };
    const size_t width = llama_model_n_embd_out(model.get());
    size_t store_decodes = 0;
    uint64_t digest = UINT64_C(14695981039346656037);
    auto observe = [&](const void * data, size_t size) {
        // fork: all fresh-process modes record the same requested output/cache schedule.
        // Include byte counts so missing rows or cache observations change the digest.
        const uint64_t count = size;
        const auto bytes = [&](const void * source, size_t n) {
            const auto * value = (const uint8_t *) source;
            for (size_t i = 0; i < n; ++i) {
                digest = (digest ^ value[i])*UINT64_C(1099511628211);
            }
        };
        bytes(&count, sizeof(count));
        bytes(data, size);
    };
    auto decode = [&](llama_context * ctx, llama_seq_id seq, llama_pos from, llama_pos to, bool output) {
        common_batch batch(ctx);
        std::vector<float> hidden(size_t(to - from)*width);
        for (llama_pos pos = from; pos < to; ++pos) {
            float * hidden_row = hidden.data() + size_t(pos - from)*width;
            for (size_t j = 0; j < width; ++j) {
                hidden_row[j] = float((int) ((j + pos) % 19) - 9) / 19.0f;
            }
            const int row = batch.add(token_at(pos), pos, seq, output);
            require(batch.set_embd(row, { hidden_row, 1, width }), "attach deterministic pool helper input");
        }
        std::vector<uint8_t> merged_input, scalar_input;
        auto * rendered = checked_embedding_batch(batch, hidden, width, &merged_input);
        // This is an input-materialization oracle; the pool test still decodes
        // the original merged batch and retains all existing cache assertions.
        for (llama_pos pos = from; pos < to; ++pos) {
            const auto first = hidden.begin() + size_t(pos - from)*width;
            const std::vector<float> row_hidden(first, first + width);
            common_batch scalar(ctx);
            const int row = scalar.add(token_at(pos), pos, seq, output);
            require(scalar.set_embd(row, { row_hidden.data(), 1, width }), "attach scalar pool input oracle");
            checked_embedding_batch(scalar, row_hidden, width, &scalar_input);
        }
        require(scalar_input == merged_input, "scalar and merged pool batches serialize identical logical inputs");
        printf("DRAFTRW_POOL_INPUT seq=%d from=%d to=%d output=%d rows=%d scalar=%016llx merged=%016llx\n",
                seq, from, to, int(output), to - from,
                (unsigned long long) embedding_input_digest(scalar_input),
                (unsigned long long) embedding_input_digest(merged_input));
        // fork: validation compares the new inputs byte-for-byte with the generic fill, including padded rows.
        const size_t before = pool_store_fills;
        require(llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, rendered) == 0, "decode pool helper");
        llama_synchronize(ctx);
        if (!output) {
            require(store_only && pool_store_fills > before, "no-output decode used and validated the specialized pool fill");
            ++store_decodes;
            return std::vector<float>();
        }
        require(pool_store_fills == before, "logits decode keeps the full pool inputs");
        const float * logits = llama_get_logits(ctx);
        require(logits != nullptr, "pool helper returns logits");
        return std::vector<float>(logits, logits + (to - from)*n_vocab);
    };
    auto original = make_context();
    const int64_t timing_started = ggml_time_us();
    const auto timing_before = llama_perf_bookkeep_context(original.get());
    // A logits-producing helper uses the generic full graph under the same static env settings.
    llama_context_ptr reference = store_only ? make_context() : llama_context_ptr();
    auto append = [&](llama_seq_id seq, llama_pos from, llama_pos to, bool want_logits) {
        const bool output = !store_only || want_logits;
        std::vector<float> expected;
        if (reference) {
            expected = decode(reference.get(), seq, from, to, true);
        }
        auto actual = decode(original.get(), seq, from, to, output);
        if (reference) {
            require(save(original.get(), seq) == save(reference.get(), seq),
                    "KV-only pool writes preserve the full-graph helper cache bytes");
            require(!output || (actual.size() == expected.size() &&
                        std::memcmp(actual.data(), expected.data(), actual.size()*sizeof(float)) == 0),
                    "KV-only catch-up preserves subsequent full-graph logits exactly");
        }
        if (want_logits) { observe(actual.data(), actual.size()*sizeof(float)); }
        return actual;
    };
    auto * mem = pool_memory(original.get());
    auto * idx = mem->get_mem_idx();
    append(0, 0, 3, false);
    append(0, 3, 4, true);
    const int32_t reused = llama_perf_context(original.get()).n_reused;
    append(0, 4, 6, false);
    append(0, 6, 8, false);
    require(!store_only || llama_perf_context(original.get()).n_reused > reused,
            "specialized pool graph reuses its allocation across partial/complete boundaries");
    const auto shared = idx->get_cells(0).seq_pos_get(0);
    require(shared.size() == 8, "pool source has two complete pools");
    llama_memory_seq_cp(mem, 0, 1, -1, -1);
    if (reference) { llama_memory_seq_cp(llama_get_memory(reference.get()), 0, 1, -1, -1); }
    require(idx->get_cells(1).seq_pos_get(1) == shared, "sequence copy shares the physical pool representatives");
    append(0, 8, 9, true);
    append(1, 8, 9, false);
    require(llama_memory_seq_rm(mem, 1, -1, -1), "delete copied pool sequence");
    if (reference) { require(llama_memory_seq_rm(llama_get_memory(reference.get()), 1, -1, -1), "delete copied reference sequence"); }
    require(idx->get_cells(1).seq_pos_get(1).empty(), "copied pool sequence is empty");
    append(0, 9, 11, false);
    require(llama_memory_seq_rm(mem, 0, 6, -1), "rollback through a complete and partial pool");
    if (reference) { require(llama_memory_seq_rm(llama_get_memory(reference.get()), 0, 6, -1), "rollback reference pools"); }
    require(idx->get_cells(0).seq_pos_get(0).size() == 6 && llama_memory_seq_pos_max(mem, 0) == 5,
            "pool rollback preserves the prefix");
    append(0, 6, 8, false);
    append(0, 8, 11, false);
    const auto state = save(original.get(), 0);
    observe(state.data(), state.size());
    auto restore_pool = [&](llama_context * ctx, const std::vector<uint8_t> & data, llama_seq_id seq) {
        require(llama_state_seq_set_data_ext(ctx, data.data(), data.size(), seq,
                    LLAMA_STATE_SEQ_FLAGS_KEEP_LAYOUT) == data.size(), "restore pool cache bytes with the saved physical layout");
    };
    auto normalized_state = [&](llama_context * ctx, llama_seq_id seq) {
        auto data = save(ctx, seq);
        if (seq == 0) { return data; }
        // Serialized sequence IDs differ. Normalize through the public state API, never by masking bytes.
        auto normalized = make_context();
        restore_pool(normalized.get(), data, 0);
        return save(normalized.get(), 0);
    };
    auto resumed = make_context();
    auto same_seq = make_context();
    // Both restored trajectories start from the identical saved byte vector and use identical batch widths.
    restore_pool(resumed.get(), state, 1);
    restore_pool(same_seq.get(), state, 0);
    auto * restored = pool_memory(resumed.get());
    require(restored->get_mem_idx()->get_cells(0).seq_pos_get(0).empty(), "pool restore leaves the other sequence empty");
    require(restored->get_mem_idx()->get_cells(1).seq_pos_get(1) == idx->get_cells(0).seq_pos_get(0),
            "pool restore keeps the saved physical layout");
    require(normalized_state(resumed.get(), 1) == state && save(same_seq.get(), 0) == state,
            "same- and different-sequence pool restore round-trip every normalized cache byte exactly");
    llama_context_ptr restored_reference = store_only ? make_context() : llama_context_ptr();
    if (restored_reference) { restore_pool(restored_reference.get(), state, 0); }
    size_t restore_rows = 0, restore_exact_rows = 0, restore_top1_equal = 0, restored_store_calls = 0;
    double restore_kld_sum = 0.0, restore_kld_max = 0.0, restore_max_error = 0.0;
    llama_pos from = 11;
    for (llama_pos to : { 12, 15, 16, 20, 21, 24 }) {
        const bool want_logits = to != 16 && to != 21;
        const bool output = !store_only || want_logits;
        const auto expected = append(0, from, to, want_logits);
        const auto actual = decode(resumed.get(), 1, from, to, output);
        const auto same_actual = decode(same_seq.get(), 0, from, to, output);
        require(actual.size() == same_actual.size() && (actual.empty() ||
                    std::memcmp(actual.data(), same_actual.data(), actual.size()*sizeof(float)) == 0),
                "same- and different-sequence restored pool continuation logits are bit-identical");
        const auto normalized = normalized_state(resumed.get(), 1);
        require(normalized == save(same_seq.get(), 0),
                "same-state restored trajectories preserve every normalized cache byte after continuation");
        observe(normalized.data(), normalized.size());
        // Keep the causal boundary strict: historical re-pooling may round differently after restore,
        // but neither serialized physical membership nor any raw indexer key may change.
        auto * restored_idx = pool_memory(same_seq.get())->get_mem_idx();
        const auto & physical = idx->get_cells(0).seq_pos_get(0);
        require(physical == restored_idx->get_cells(0).seq_pos_get(0),
                "restored and unrestored pool continuation keeps the saved physical cells");
        auto * keys = idx->get_k_storage(model->hparams.n_layer());
        auto * restored_keys = restored_idx->get_k_storage(model->hparams.n_layer());
        require(keys->type == restored_keys->type && keys->nb[1] == restored_keys->nb[1],
                "restored indexer key geometry");
        const size_t raw_bytes = ggml_row_size(keys->type, model->hparams.indexer_head_size);
        require(keys->ne[0] == 2*model->hparams.indexer_head_size && restored_keys->ne[0] == keys->ne[0] &&
                    raw_bytes > 0 && 2*raw_bytes == keys->nb[1], "packed raw and pooled indexer key rows");
        std::vector<uint8_t> before_row(2*raw_bytes), after_row(2*raw_bytes);
        size_t changed_pooled_rows = 0;
        for (const auto & pc : physical) {
            ggml_backend_tensor_get(keys, before_row.data(), pc.second*keys->nb[1], before_row.size());
            ggml_backend_tensor_get(restored_keys, after_row.data(), pc.second*restored_keys->nb[1], after_row.size());
            require(std::memcmp(before_row.data(), after_row.data(), raw_bytes) == 0,
                    "restored continuation preserves every raw indexer key exactly");
            changed_pooled_rows += std::memcmp(before_row.data() + raw_bytes, after_row.data() + raw_bytes, raw_bytes) != 0;
        }
        printf("DRAFTRW_POOL_RESTORE_CACHE to=%d cells=%zu raw_equal=1 pooled_changed=%zu\n",
                to, physical.size(), changed_pooled_rows);
        if (!output) { restored_store_calls += 2; }
        if (restored_reference) {
            const auto full = decode(restored_reference.get(), 0, from, to, true);
            require(normalized == save(restored_reference.get(), 0),
                    "restored store-only pool writes preserve restored full-graph cache bytes exactly");
            require(!output || (same_actual.size() == full.size() &&
                        std::memcmp(same_actual.data(), full.data(), full.size()*sizeof(float)) == 0),
                    "restored store-only continuation matches restored full-graph logits exactly");
        }
        if (want_logits) {
            observe(actual.data(), actual.size()*sizeof(float));
            require(actual.size() == expected.size(), "restored pool output row count");
            // fork: state_read invalidates historical pools; re-pooling at a different width changes legacy rounding.
            // The KLD cell and exact cache/logit digest must match across fresh-process legacy/incremental/store-only runs.
            for (size_t row = 0; row < actual.size()/n_vocab; ++row) {
                const size_t first = row*n_vocab;
                const auto pa = expected.begin() + first;
                const auto qa = actual.begin() + first;
                const double ma = *std::max_element(pa, pa + n_vocab);
                const double mb = *std::max_element(qa, qa + n_vocab);
                double za = 0.0, zb = 0.0, kld = 0.0, max_error = 0.0;
                for (size_t j = first; j < first + n_vocab; ++j) {
                    require(std::isfinite(expected[j]) && std::isfinite(actual[j]), "finite restored pool logits");
                    max_error = std::max(max_error, std::abs(double(actual[j]) - expected[j]));
                    za += std::exp(expected[j] - ma);
                    zb += std::exp(actual[j] - mb);
                }
                for (size_t j = first; j < first + n_vocab; ++j) {
                    const double logp = expected[j] - ma - std::log(za);
                    const double logq = actual[j] - mb - std::log(zb);
                    kld += std::exp(logp)*(logp - logq);
                }
                require(std::isfinite(kld) && kld >= -1e-12, "finite restored pool KLD");
                const bool exact = std::memcmp(expected.data() + first, actual.data() + first, n_vocab*sizeof(float)) == 0;
                const bool top1_equal = std::max_element(pa, pa + n_vocab) - pa == std::max_element(qa, qa + n_vocab) - qa;
                ++restore_rows;
                restore_exact_rows += exact;
                restore_top1_equal += top1_equal;
                restore_kld_sum += kld;
                restore_kld_max = std::max(restore_kld_max, kld);
                restore_max_error = std::max(restore_max_error, max_error);
                printf("DRAFTRW_POOL_RESTORE_KLD pos=%d exact=%d top1_equal=%d kld=%.17g max_logit_error=%.17g\n",
                        from + int(row), int(exact), int(top1_equal), kld, max_error);
            }
        }
        from = to;
    }
    require(!store_only || restored_store_calls == 4,
            "restored pool cache exercises two real no-output continuations before exact output checks");
    require(!store_only || store_decodes >= 9, "store-only pool regression exercises real specialized fills throughout edits and restore");
    // fork: one real prefill ubatch creates 17 pools, crossing the sparse marker's stack16/hash threshold.
    // Repeated member marks also exercise hash deduplication.
    auto large = make_context(80);
    llama_context_ptr large_reference = store_only ? make_context(80) : llama_context_ptr();
    std::vector<float> large_expected;
    if (large_reference) { decode(large_reference.get(), 0, 0, 68, true); }
    const size_t large_store_before = pool_store_fills;
    decode(large.get(), 0, 0, 68, !store_only);
    require(!store_only || pool_store_fills == large_store_before + 1, "large sparse-mark prefill executes as one store-only ubatch");
    require(pool_memory(large.get())->get_mem_idx()->get_cells(0).seq_pos_get(0).size() == 68,
            "large pool fixture prefills 17 complete pools in one ubatch");
    if (large_reference) {
        require(save(large.get(), 0) == save(large_reference.get(), 0),
                "sparse hash marks preserve the generic full-graph prefill cache bytes");
        large_expected = decode(large_reference.get(), 0, 68, 72, true);
    }
    const auto large_actual = decode(large.get(), 0, 68, 72, true);
    observe(large_actual.data(), large_actual.size()*sizeof(float));
    require(!large_reference || (large_actual.size() == large_expected.size() &&
                std::memcmp(large_actual.data(), large_expected.data(), large_actual.size()*sizeof(float)) == 0),
            "sparse hash prefill preserves subsequent full-graph logits exactly");
    const char * timing = std::getenv("LLAMA_BOOKKEEP_TIMING");
    if (timing && std::atoi(timing) != 0) {
        const auto after = llama_perf_bookkeep_context(original.get());
        const uint64_t detail_us =
            after.memory_init_layout_host_us - timing_before.memory_init_layout_host_us +
            after.graph_prepare_host_us - timing_before.graph_prepare_host_us +
            after.set_inputs_host_us - timing_before.set_inputs_host_us +
            after.peer_sync_host_us - timing_before.peer_sync_host_us +
            after.graph_submit_host_us - timing_before.graph_submit_host_us;
        require(after.graph_rebuild_calls + after.graph_reuse_calls >
                timing_before.graph_rebuild_calls + timing_before.graph_reuse_calls,
                "catch-up detail counts graph preparation attempts");
        require(detail_us > 0 && detail_us <= uint64_t(ggml_time_us() - timing_started + 10),
                "catch-up detail stages are active and do not overlap");
        uint64_t backend_us = 0;
        for (size_t i : { 0u, 1u, 2u, 4u, 5u, 6u }) {
            require(after.backend_submit[i] >= timing_before.backend_submit[i], "backend counters are monotonic");
            backend_us += after.backend_submit[i] - timing_before.backend_submit[i];
        }
        require(backend_us > 0 && backend_us <= uint64_t(ggml_time_us() - timing_started + 10),
                "backend host intervals are active and do not overlap");
        require(after.backend_submit[3] - timing_before.backend_submit[3] <=
                after.backend_submit[2] - timing_before.backend_submit[2],
                "user-input synchronization is nested in input copies");
        require(after.backend_submit[cuda ? 8 : 7] > timing_before.backend_submit[cuda ? 8 : 7],
                "backend counters execute the selected CPU or CUDA provider");
    }
    printf("DRAFTRW_POOL digest=%016llx\n", (unsigned long long) digest);
    printf("DRAFTRW_POOL_RESTORE rows=%zu exact_rows=%zu top1_equal=%zu kld_mean=%.17g kld_max=%.17g max_logit_error=%.17g\n",
            restore_rows, restore_exact_rows, restore_top1_equal, restore_kld_sum/restore_rows, restore_kld_max, restore_max_error);
    printf("PASS Qwen4exp %s pool fixture%s: partial boundaries, suffix rollback/reappend, shared representatives, copied sequence deletion, normalized restore bytes, identical restored cache/logits, post-restore KV-only continuation, stack16/hash prefill; unrestored rounding is a separate KLD cell requiring fresh-process legacy comparison (%zu specialized fills)\n",
            cuda ? "GPU" : "CPU", store_only ? " store-only" : "", store_decodes);
}

static void run_qwen4exp_chain(bool cuda, bool compact = false) {
    prefill_env_restore prefill_env;
    prefill_env.set(false, false, false);
    feedback_env_restore feedback_env;
    common_set_env("LLAMA_SPEC_DEFER_CATCHUP", "0");
    common_set_env("LLAMA_MTP_DEVICE_FEEDBACK", "0");
    common_set_env("LLAMA_MTP_QWEN4EXP_CHAIN", "0");
    common_set_env("LLAMA_MTP_QSA_COMPACT", compact ? "1" : "0");
    auto * device = cuda ? ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) : nullptr;
    require(!cuda || device, "chain GPU backend is available");
    model_geometry geometry;
    geometry.kpool = 4;
    // Two selected pools force real ranking once catch-up plus depth completes the third pool.
    geometry.indexer_top_k = 8;
    llama_model_ptr model(make_model(false, geometry, LLM_ARCH_QWEN4EXP, device));
    require(!llama_model_supports_mtp_chain(model.get()), "Qwen4exp chain is opt-in");
    const cache_config config { "Qwen4exp chain", compact ? GGML_TYPE_TQ5_0 : GGML_TYPE_Q8_0,
        compact ? GGML_TYPE_TURBO4_0 : GGML_TYPE_Q8_0 };
    const int depth = 3, top_k = 10;
    const float top_p = 0.95f;
    const float uniforms[depth] = { 0.173f, 0.619f, 0.937f };
    const size_t width = llama_model_n_embd_out(model.get());
    for (int catchup : { 0, 2 }) {
        common_set_env("LLAMA_MTP_QWEN4EXP_CHAIN", "0");
        contexts reference(model.get(), config, 16, COMMON_SPECULATIVE_TYPE_DRAFT_MTP, 16, 1, 512);
        contexts chained(model.get(), config, 16, COMMON_SPECULATIVE_TYPE_DRAFT_MTP, 16, 1, 512);
        // The scalar oracle reads full raw logits. Backend top-k sampling skips
        // that download, so detach it before any oracle decode.
        require(llama_set_sampler(reference.dft, 0, nullptr), "detach raw-logit reference backend sampler");
        // These direct graph calls use packed chain outputs, not the driver's
        // single-row top-k backend sampler installed while chain mode was off.
        require(llama_set_sampler(chained.dft, 0, nullptr), "detach direct-chain backend sampler");
        prefill(reference, 0, 7, 0);
        prefill(chained, 0, 7, 0);
        std::vector<float> seed((catchup + 1) * width), poison(width, 1000.0f);
        const float * last = llama_get_embeddings_nextn_ith(reference.tgt, 6);
        std::copy(last, last + width, seed.begin());
        if (catchup) {
            common_batch target(reference.tgt);
            for (int j = 0; j < catchup; ++j) {
                target.add(token_at(7 + j), 7 + j, 0, true);
            }
            require(llama_process(reference.tgt, LLAMA_PROCESS_TYPE_DECODE, target.get()) == 0, "chain target catch-up residuals");
            for (int j = 0; j < catchup; ++j) {
                const float * row = llama_get_embeddings_nextn_ith(reference.tgt, j);
                std::copy(row, row + width, seed.begin() + (j + 1) * width);
            }
        }
        const auto reference_state = save(reference.dft, 0);
        const auto chain_state = save(chained.dft, 0);
        common_set_env("LLAMA_MTP_QWEN4EXP_CHAIN", "1");
        require(llama_model_supports_mtp_chain(model.get()), "Qwen4exp chain capability is enabled");
        for (int trial = 0; trial < 3; ++trial) {
            std::vector<uint8_t> scalar_input, merged_input;
            if (trial == 1) {
                require(llama_memory_seq_rm(llama_get_memory(reference.dft), 0, 7, -1), "rollback sequential helper");
                require(llama_memory_seq_rm(llama_get_memory(chained.dft), 0, 7, -1), "rollback chained helper");
            } else if (trial == 2) {
                require(llama_state_seq_set_data_ext(reference.dft, reference_state.data(), reference_state.size(), 0,
                            LLAMA_STATE_SEQ_FLAGS_KEEP_LAYOUT) == reference_state.size(), "restore sequential helper");
                require(llama_state_seq_set_data_ext(chained.dft, chain_state.data(), chain_state.size(), 0,
                            LLAMA_STATE_SEQ_FLAGS_KEEP_LAYOUT) == chain_state.size(), "restore chained helper");
            }
            if (catchup) {
                common_batch front(reference.dft);
                for (int j = 0; j < catchup; ++j) {
                    const int row = front.add(token_at(7 + j), 7 + j, 0, false);
                    require(front.set_embd(row, { seed.data() + j * width, 1, width }), "attach catch-up residual");
                }
                require(llama_process(reference.dft, LLAMA_PROCESS_TYPE_DECODE,
                            checked_embedding_batch(front, seed, width, &scalar_input)) == 0, "sequential KV-only catch-up");
            }
            std::vector<float> hidden(seed.end() - width, seed.end()), expected_logits(depth * n_vocab), expected_hidden(depth * width);
            std::vector<llama_token> picks(depth);
            llama_token token = token_at(7 + catchup);
            for (int j = 0; j < depth; ++j) {
                common_batch step(reference.dft);
                const int row = step.add(token, 7 + catchup + j, 0, true);
                require(step.set_embd(row, { hidden.data(), 1, width }), "attach sequential helper residual");
                require(llama_process(reference.dft, LLAMA_PROCESS_TYPE_DECODE,
                            checked_embedding_batch(step, hidden, width, j == 0 ? &scalar_input : nullptr)) == 0,
                        "sequential reference step");
                const float * logits = llama_get_logits(reference.dft);
                std::copy(logits, logits + n_vocab, expected_logits.begin() + j * n_vocab);
                std::vector<int> order(n_vocab);
                for (uint32_t k = 0; k < n_vocab; ++k) { order[k] = k; }
                std::sort(order.begin(), order.end(), [&](int a, int b) { return logits[a] > logits[b]; });
                std::vector<float> candidate_row(LLAMA_MTP_CHAIN_ROW(top_k));
                for (int k = 0; k < top_k; ++k) {
                    candidate_row[2 + k] = order[k];
                    candidate_row[2 + top_k + k] = logits[order[k]];
                }
                std::vector<llama_token_data> q;
                const int pick = llama_mtp_chain_rederive(candidate_row.data(), top_k, 1.0f, top_p, 0.0f,
                        uniforms[j], n_vocab, q);
                require(pick >= 0, "sample the sequential reference with the chain uniform");
                picks[j] = order[pick];
                token = picks[j];
                const float * next = llama_get_embeddings_nextn(reference.dft);
                std::copy(next, next + width, hidden.begin());
                std::copy(hidden.begin(), hidden.end(), expected_hidden.begin() + j * width);
            }
            common_batch batch(chained.dft);
            std::vector<float> chain_input(size_t(catchup + depth)*width);
            for (int j = 0; j < catchup + depth; ++j) {
                const int row = batch.add(j <= catchup ? token_at(7 + j) : 0, 7 + j, 0, j >= catchup);
                const float * h = j <= catchup ? seed.data() + j * width : poison.data();
                std::copy(h, h + width, chain_input.begin() + size_t(j)*width);
                require(batch.set_embd(row, { chain_input.data() + size_t(j)*width, 1, width }), "attach chain seed or poisoned placeholder");
            }
            llama_set_mtp_chain_sampling(chained.dft, top_k, 1.0f, top_p, 0.0f, uniforms, depth);
            llama_set_mtp_chain(chained.dft, true);
            // Later host rows are deliberately poisoned: the chain replaces
            // them with feedback. Only catch-up plus the common seed is shared
            // with the scalar input; every rendered row is still checked above.
            auto * rendered = checked_embedding_batch(batch, chain_input, width, &merged_input, catchup + 1);
            require(scalar_input == merged_input, "scalar and merged chain receive identical serialized catch-up and seed inputs");
            printf("DRAFTRW_CHAIN_INPUT catchup=%d trial=%d rows=%d scalar=%016llx merged=%016llx\n",
                    catchup, trial, catchup + 1, (unsigned long long) embedding_input_digest(scalar_input),
                    (unsigned long long) embedding_input_digest(merged_input));
            require(llama_process(chained.dft, LLAMA_PROCESS_TYPE_DECODE, rendered) == 0, "sampled Qwen4exp chain");
            const float * rows = llama_get_logits(chained.dft);
            const float * residuals = llama_get_embeddings_nextn(chained.dft);
            for (int j = 0; j < depth; ++j) {
                const float * row = rows + j * LLAMA_MTP_CHAIN_ROW(top_k);
                const float * logits = expected_logits.data() + j * n_vocab;
                std::vector<int> order(n_vocab);
                for (uint32_t k = 0; k < n_vocab; ++k) { order[k] = k; }
                std::sort(order.begin(), order.end(), [&](int a, int b) { return logits[a] > logits[b]; });
                // Qwen payloads retain backend top-k order; the legacy graph-uniform
                // reference operates on a sorted copy of those same candidates.
                std::vector<int> candidate_order(top_k), actual_ids(top_k), expected_ids(order.begin(), order.begin() + top_k);
                for (int k = 0; k < top_k; ++k) {
                    const float id = row[2 + k];
                    require(std::isfinite(id) && id >= 0 && id < n_vocab && id == std::floor(id),
                            "chain candidate IDs are valid");
                    candidate_order[k] = k;
                    actual_ids[k] = (llama_token) id;
                    require(cuda || row[2 + top_k + k] == logits[actual_ids[k]], "CPU chain candidate logits are bit-identical");
                }
                std::sort(actual_ids.begin(), actual_ids.end());
                std::sort(expected_ids.begin(), expected_ids.end());
                require(actual_ids == expected_ids, "chain top-k candidate set matches the sequential helper");
                std::sort(candidate_order.begin(), candidate_order.end(), [&](int a, int b) {
                    return row[2 + top_k + a] > row[2 + top_k + b];
                });
                std::vector<float> sorted_row(row, row + LLAMA_MTP_CHAIN_ROW(top_k));
                for (int k = 0; k < top_k; ++k) {
                    sorted_row[2 + k] = row[2 + candidate_order[k]];
                    sorted_row[2 + top_k + k] = row[2 + top_k + candidate_order[k]];
                }
                std::vector<llama_token_data> q;
                const int pick = llama_mtp_chain_rederive(sorted_row.data(), top_k, 1.0f, top_p, 0.0f, uniforms[j], n_vocab, q);
                require(pick >= 0 && (llama_token) sorted_row[2 + pick] == (llama_token) row[0],
                        "host re-derives the sampled chain token");
                require((llama_token) row[0] == picks[j], "chain conditions on the sequential sampled token");
                std::vector<float> reference_row(LLAMA_MTP_CHAIN_ROW(top_k));
                for (int k = 0; k < top_k; ++k) {
                    reference_row[2 + k] = order[k];
                    reference_row[2 + top_k + k] = logits[order[k]];
                }
                std::vector<llama_token_data> reference_q;
                const int reference_pick = llama_mtp_chain_rederive(reference_row.data(), top_k, 1.0f, top_p, 0.0f,
                        uniforms[j], n_vocab, reference_q);
                require(reference_pick >= 0 && reference_row[2 + reference_pick] == sorted_row[2 + pick],
                        "reference and chain draw the same candidate");
                require(reference_q.size() == q.size(), "chain preserves top-p support");
                double kl = 0.0, error = 0.0, norm = 0.0;
                for (size_t k = 0; k < q.size(); ++k) {
                    require(reference_q[k].id == q[k].id, "chain preserves sampled proposal IDs");
                    kl += reference_q[k].p * std::log(double(reference_q[k].p) / q[k].p);
                }
                for (size_t k = 0; k < width; ++k) {
                    const double expected = expected_hidden[j * width + k], actual = residuals[j * width + k];
                    error += (expected - actual) * (expected - actual);
                    norm += expected * expected;
                }
                const double nmse = error / std::max(norm, 1e-30);
                printf("CHAIN_KLD %s catchup=%d trial=%d step=%d kl=%.9g hidden_nmse=%.9g\n", cuda ? "GPU" : "CPU", catchup, trial, j, kl, nmse);
                require(kl <= 1e-6 && nmse <= 1e-8, "chain preserves proposal math within the rounding budget");
                require(cuda || error == 0.0, "CPU chain preserves the raw helper residual exactly");
            }
            llama_set_mtp_chain(chained.dft, false);
            llama_set_mtp_chain_sampling(chained.dft, 0, 1.0f, 1.0f, 0.0f, nullptr, 0);
        }
        require(llama_perf_context(chained.dft).n_reused > 0, "chain graph is reused across rollback and restore");
    }
    // Separate model objects engage the production common-driver chain (ctx_other then shares no KV).
    // Save before drafting so the prompt restore also covers target carry and deferred catch-up rows.
    llama_model_ptr helper(make_model(false, geometry, LLM_ARCH_QWEN4EXP, device));
    common_set_env("LLAMA_MTP_QWEN4EXP_CHAIN", "1");
    common_set_env("LLAMA_SPEC_CHAIN_SAMPLED", "1");
    common_set_env("LLAMA_SPEC_PQ", "1");
    contexts original(model.get(), config, 16, COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE, 16, 1, 512, helper.get());
    prefill(original, 0, 7, 0);
    const snapshot cached = save(original, 0);
    llama_pos original_pos = 7;
    const schedule expected = generate(original, 0, original_pos, 0, 3, true);
    contexts resumed(model.get(), config, 16, COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE, 16, 1, 512, helper.get());
    restore(resumed, cached, 0);
    llama_pos resumed_pos = 7;
    require_schedule(generate(resumed, 0, resumed_pos, 0, 3, true), expected, "sampled common chain driver resumes prompt carry and deferred catch-up");
    printf("PASS Qwen4exp %s sampled chain%s: k-pool boundaries/ranking, front-only catch-up, poisoned placeholders, nonzero uniforms/top-p, proposal KLD, raw residuals, rollback, restore, graph reuse\n", cuda ? "GPU" : "CPU", compact ? " compact TQ5/TURBO4" : "");
}

// The source head must have a row decoder for init_draft_vocab to build its compact copy.
// F32 intentionally lacks to_float, so this generated fixture uses identical F16-rounded heads.
// These owners are declared after the model owners and before all contexts: contexts die first,
// then the original pointers are restored before the F16 buffers or models can be destroyed.
struct full_head_f16_output {
    llama_model & model;
    ggml_tensor * original;
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * head = nullptr;

    full_head_f16_output(llama_model & model, const std::vector<ggml_fp16_t> & weights) :
        model(model), original(model.output) {
        require(original && original->type == GGML_TYPE_F32 && original->buffer &&
                original != model.tok_embd && original->ne[2] == 1 && original->ne[3] == 1 &&
                ggml_is_contiguous(original) && weights.size() == size_t(ggml_nelements(original)),
                "generated fixture has a separate allocated full F32 output head");
        require(ggml_get_type_traits(GGML_TYPE_F16)->to_float != nullptr,
                "F16 fixture source head has a real compact-row decoder");
        for (const auto & layer : model.layers) {
            require(!layer.nextn.shared_head_head, "Qwen4exp fixture compact source is model.output");
        }
        ctx.reset(ggml_init({ ggml_tensor_overhead(), nullptr, true }));
        require(ctx != nullptr, "allocate fixture F16 head metadata");
        head = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, original->ne[0], original->ne[1]);
        ggml_set_name(head, ggml_get_name(original));
        buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_buffer_get_type(original->buffer)));
        require(buffer != nullptr, "allocate fixture F16 head in original backend");
        ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_tensor_set(head, weights.data(), 0, ggml_nbytes(head));
        for (auto & named : model.tensors_by_name) {
            if (named.second == original) { named.second = head; }
        }
        model.output = head;
    }

    ~full_head_f16_output() {
        model.output = original;
        for (auto & named : model.tensors_by_name) {
            if (named.second == head) { named.second = original; }
        }
    }
};

struct full_head_graph_stats {
    size_t batched = 0, provisional = 0, scalar = 0;
};

static bool full_head_fixture_eval(ggml_tensor * tensor, bool ask, void * user_data) {
    const bool batched = tensor->op == GGML_OP_MUL_MAT && tensor->ne[0] == n_vocab && tensor->ne[1] == 4 &&
            std::strncmp(tensor->name, "mtp_chain_full_logits", std::strlen("mtp_chain_full_logits")) == 0;
    const bool provisional = tensor->op == GGML_OP_MUL_MAT && tensor->ne[0] == n_vocab / 2 && tensor->ne[1] == 1 &&
            tensor->src[0] && tensor->src[0]->ne[0] == n_embd && tensor->src[0]->ne[1] == n_vocab / 2 &&
            std::strstr(tensor->src[0]->name, "draft_vocab_compact") != nullptr;
    // This fixture installs an F16 full head without output scaling or LoRA, so the
    // ordinary helper projection is this exact named MUL_MAT, not a provisional head.
    const bool scalar = tensor->op == GGML_OP_MUL_MAT && tensor->ne[0] == n_vocab && tensor->ne[1] == 1 &&
            tensor->src[0] && tensor->src[0]->type == GGML_TYPE_F16 &&
            tensor->src[0]->ne[0] == n_embd && tensor->src[0]->ne[1] == n_vocab &&
            std::strcmp(tensor->name, "result_output") == 0;
    if (ask) { return batched || provisional || scalar; }
    auto & stats = *static_cast<full_head_graph_stats *>(user_data);
    stats.batched += batched;
    stats.provisional += provisional;
    stats.scalar += scalar;
    return true;
}

static void require_full_head_graph(const full_head_graph_stats & before, const full_head_graph_stats & after,
        bool cuda, const char * phase, int fault, int round, bool enabled) {
    const size_t batched = after.batched - before.batched, provisional = after.provisional - before.provisional;
    const size_t expected_batched = enabled ? 1 : 0, expected_provisional = enabled ? 3 : 0;
    if (batched != expected_batched || provisional != expected_provisional) {
        fprintf(stderr, "FULL_HEAD_GRAPH_FAILURE %s phase=%s fault=%d round=%d batched=%zu expected_batched=%zu provisional=%zu expected_provisional=%zu\n",
                cuda ? "GPU" : "CPU", phase, fault, round, batched, expected_batched, provisional, expected_provisional);
    }
    require(batched == expected_batched && provisional == expected_provisional,
            "full-head fixture executes actual batched and compact provisional projections (or explicit sequential fallback)");
}

struct full_head_round {
    llama_tokens tokens;
    std::vector<std::vector<llama_token_data>> q;
    int n_eval = 0;
    int n_p_eval = 0;
};

static common_params_sampling full_head_sampling() {
    common_params_sampling sampling;
    sampling.seed = 9127;
    sampling.temp = 1.0f;
    sampling.top_k = 10;
    sampling.top_p = 0.95f;
    sampling.min_p = 0.0f;
    sampling.min_keep = 1;
    sampling.dynatemp_range = 0.0f;
    sampling.samplers = { COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_TOP_P,
        COMMON_SAMPLER_TYPE_MIN_P, COMMON_SAMPLER_TYPE_TEMPERATURE };
    return sampling;
}

static void prime_full_head_eval_counter(contexts & ctx) {
    // Perf counts clamp zero to one. Seed the scalar counter without touching the driver's sampler or carry.
    const auto state = save(ctx.dft, 0);
    auto * memory = llama_get_memory(ctx.dft);
    const llama_pos original_max = llama_memory_seq_pos_max(memory, 0);
    const llama_pos pos = original_max + 1;
    std::vector<float> zero(llama_model_n_embd_out(llama_get_model(ctx.dft)), 0.0f);
    common_batch batch(ctx.dft);
    const int row = batch.add(token_at(pos), pos, 0, true);
    require(batch.set_embd(row, { zero.data(), 1, zero.size() }), "attach counter-prime residual");
    require(llama_process(ctx.dft, LLAMA_PROCESS_TYPE_DECODE,
                checked_embedding_batch(batch, zero, zero.size())) == 0, "prime scalar eval counter");
    llama_synchronize(ctx.dft);
    // Empty KEEP_LAYOUT sections skip cache metadata restoration. Explicitly
    // remove the priming row before loading that empty state.
    if (original_max < 0) {
        require(llama_memory_seq_rm(memory, 0, -1, -1), "remove counter prime from originally empty helper");
    }
    require(llama_state_seq_set_data_ext(ctx.dft, state.data(), state.size(), 0,
                LLAMA_STATE_SEQ_FLAGS_KEEP_LAYOUT) == state.size(), "restore helper after counter prime");
    require(llama_memory_seq_pos_max(memory, 0) == original_max, "counter prime restores the original helper extent");
}

static full_head_round draft_full_head_round(contexts & ctx, llama_pos pos, const common_params_sampling & sampling, int depth) {
    full_head_round result;
    llama_tokens prompt;
    auto & dp = common_speculative_get_draft_params(ctx.spec, 0);
    dp = {};
    dp.drafting = true;
    dp.n_max = depth;
    dp.pos0 = pos;
    dp.id_last = token_at(pos);
    dp.prompt = &prompt;
    dp.result = &result.tokens;
    dp.result_q = &result.q;
    dp.sampling = &sampling;
    llama_synchronize(ctx.dft);
    // Suffix repair detaches its temporary backend top-k sampler and requests a scheduler reserve.
    // Drain that pending work outside chain mode: the generic reserve probes width one, whereas
    // this fixture's immutable TEST_IDS list describes the four scored rows, not reserve probes.
    // The following real draft still has to execute the audited width-four batched graph.
    require(!ctx.dft->get_cparams().mtp_chain && ctx.dft->get_cparams().mtp_chain_top_k == 0,
            "full-head fixture prepares the scheduler outside sampled chain mode");
    ctx.dft->sched_reserve();
    const auto before = llama_perf_context(ctx.dft);
    common_speculative_draft(ctx.spec);
    llama_synchronize(ctx.dft);
    const auto after = llama_perf_context(ctx.dft);
    result.n_eval = after.n_eval - before.n_eval;
    result.n_p_eval = after.n_p_eval;
    require(result.tokens.size() == (size_t) depth && result.q.size() == result.tokens.size(),
            "full-head repair fills the entire requested draft cap");
    for (size_t j = 0; j < result.q.size(); ++j) {
        double total = 0.0;
        bool selected = false;
        require(!result.q[j].empty(), "full-head draft returns exact proposal q for every step");
        for (const auto & candidate : result.q[j]) {
            require(candidate.id >= 0 && candidate.id < (llama_token) n_vocab &&
                    std::isfinite(candidate.p) && candidate.p > 0.0f, "valid full-vocabulary proposal candidate");
            total += candidate.p;
            selected = selected || candidate.id == result.tokens[j];
        }
        require(selected && std::fabs(total - 1.0) <= 1e-6, "proposal q is normalized and contains the sampled token");
    }
    return result;
}

static void finish_full_head_round(contexts & ctx, llama_pos & pos, const full_head_round & round, int accepted) {
    llama_tokens verify { token_at(pos) };
    verify.insert(verify.end(), round.tokens.begin(), round.tokens.end());
    process(ctx, verify, pos, 0);
    common_speculative_accept(ctx.spec, 0, accepted);
    pos += accepted + 1;
    require(llama_memory_seq_rm(llama_get_memory(ctx.tgt), 0, pos, -1), "full-head target rollback");
    require(llama_memory_seq_rm(llama_get_memory(ctx.dft), 0, pos, -1), "full-head helper rollback");
}

static void require_full_head_round(const full_head_round & actual, const full_head_round & expected,
        bool cuda, int fault, int round) {
    require(actual.tokens == expected.tokens, "full-head repair preserves sampled tokens and RNG continuation");
    require(actual.q.size() == expected.q.size(), "full-head repair preserves proposal row count");
    for (size_t j = 0; j < expected.q.size(); ++j) {
        float p[n_vocab] = {}, q[n_vocab] = {};
        for (const auto & candidate : expected.q[j]) { p[candidate.id] = candidate.p; }
        for (const auto & candidate : actual.q[j]) { q[candidate.id] = candidate.p; }
        double kl = 0.0, max_error = 0.0;
        for (uint32_t id = 0; id < n_vocab; ++id) {
            require((p[id] > 0.0f) == (q[id] > 0.0f), "full-head repair preserves exact top-k/top-p support");
            if (p[id] > 0.0f) { kl += p[id] * std::log(double(p[id]) / q[id]); }
            max_error = std::max(max_error, std::fabs(double(p[id]) - q[id]));
        }
        printf("FULL_HEAD_KLD %s fault=%d round=%d step=%zu kl=%.9g max_p_error=%.9g\n",
                cuda ? "GPU" : "CPU", fault, round, j, kl, max_error);
        require(std::fabs(kl) <= 1e-6 && max_error <= 1e-5, "batched full-head proposal math stays within the rounding budget");
    }
}

static void run_qwen4exp_full_head(bool cuda) {
#ifndef _WIN32
    prefill_env_restore prefill_env;
    prefill_env.set(false, false, false);
    feedback_env_restore env;
    common_set_env("LLAMA_SPEC_DEFER_CATCHUP", "0");
    common_set_env("LLAMA_MTP_DEVICE_FEEDBACK", "0");
    common_set_env("LLAMA_MTP_QSA_COMPACT", "0");
    common_set_env("LLAMA_SPEC_CHAIN_SAMPLED", "1");
    common_set_env("LLAMA_SPEC_PQ", "1");
    common_set_env("LLAMA_SPEC_DRAFT_TEMP_MULT", "1");
    common_set_env("LLAMA_MTP_CHAIN_FULL_HEAD", "1");
    common_set_env("LLAMA_MTP_CHAIN_TEST_IDS", "");
    common_set_env("LLAMA_MTP_QWEN4EXP_CHAIN", "0");
    common_set_env("LLAMA_DRAFT_VOCAB_COMPACT", "1");
    common_set_env("LLAMA_DRAFT_VOCAB_COMPACT_TYPE", "f32");
    auto * device = cuda ? ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) : nullptr;
    require(!cuda || device, "full-head repair GPU backend is available");
    model_geometry geometry;
    geometry.kpool = 4;
    geometry.indexer_top_k = 8;
    llama_model_ptr model(make_model(false, geometry, LLM_ARCH_QWEN4EXP, device));
    llama_model_ptr helper(make_model(false, geometry, LLM_ARCH_QWEN4EXP, device));
    require(model->output && helper->output && model->output->type == GGML_TYPE_F32 &&
            helper->output->type == GGML_TYPE_F32 && ggml_nbytes(model->output) == ggml_nbytes(helper->output),
            "full-head fixture models start with matching F32 head shapes");
    std::vector<float> full_head_weights(ggml_nelements(model->output)), helper_head_weights(full_head_weights.size());
    ggml_backend_tensor_get(model->output, full_head_weights.data(), 0, ggml_nbytes(model->output));
    ggml_backend_tensor_get(helper->output, helper_head_weights.data(), 0, ggml_nbytes(helper->output));
    require(full_head_weights == helper_head_weights, "target and helper generated full-head weights match exactly");
    std::vector<ggml_fp16_t> rounded_head(full_head_weights.size());
    ggml_fp32_to_fp16_row(full_head_weights.data(), rounded_head.data(), rounded_head.size());
    full_head_f16_output model_head(*model, rounded_head), helper_head(*helper, rounded_head);
    const cache_config config { "Qwen4exp full-head repair", GGML_TYPE_Q8_0, GGML_TYPE_Q8_0 };
    const auto sampling = full_head_sampling();
    constexpr int depth = 4, prompt_size = 7, rounds = 3;
    const auto report_counter_mismatch = [cuda](const char * phase, int fault, int round,
            const full_head_round & actual, int expected_repair, int expected_prompt,
            const common_params_sampling & params, bool backend_sampling) {
        if (actual.n_eval != expected_repair || actual.n_p_eval != expected_prompt) {
            fprintf(stderr, "FULL_HEAD_COUNTER_FAILURE %s phase=%s fault=%d round=%d n_eval=%d expected_repair=%d n_p_eval=%d expected_prompt=%d backend_sampling=%d top_k=%d min_keep=%d top_p=%.9g min_p=%.9g temp=%.9g dynatemp_range=%.9g chain_optin=1 full_head=1 pq=1 sampled=1 n_seq=1 tokens=%zu q_rows=%zu\n",
                    cuda ? "GPU" : "CPU", phase, fault, round, actual.n_eval, expected_repair,
                    actual.n_p_eval, expected_prompt, int(backend_sampling), params.top_k, params.min_keep,
                    double(params.top_p), double(params.min_p), double(params.temp), double(params.dynatemp_range),
                    actual.tokens.size(), actual.q.size());
        }
    };
    const int accepted[rounds] = { depth, 1, depth };
    std::vector<full_head_round> expected;
    // Prime every helper while empty, before prefill or prompt restore. Priming
    // a filled seven-row context completes an indexer pool and changes cache
    // history, while the chain context still has its prompt deferred.
    {
        contexts reference(model.get(), config, 16, COMMON_SPECULATIVE_TYPE_DRAFT_MTP, 16, 1, 512, helper.get(), "none");
        prime_full_head_eval_counter(reference);
        prefill(reference, 0, prompt_size, 0);
        llama_pos pos = prompt_size;
        for (int round = 0; round < rounds; ++round) {
            expected.push_back(draft_full_head_round(reference, pos, sampling, depth));
            finish_full_head_round(reference, pos, expected.back(), accepted[round]);
        }
    }

    // Exclude every reference draw and scramble map positions to expose mapped IDs and renormalized q.
    const auto close_file = [](FILE * file) { fclose(file); };
    std::unique_ptr<FILE, decltype(close_file)> map(tmpfile(), close_file);
    require(map != nullptr, "create provisional vocabulary fixture");
    std::vector<llama_token> map_ids;
    for (int id = n_vocab - 1; id >= 0 && map_ids.size() < n_vocab / 2; --id) {
        bool drawn = false;
        for (const auto & round : expected) {
            drawn = drawn || std::find(round.tokens.begin(), round.tokens.end(), id) != round.tokens.end();
        }
        if (!drawn) { map_ids.push_back(id); }
    }
    require(map_ids.size() == n_vocab / 2, "fixture has enough unused tokens for its provisional map");
    fprintf(map.get(), "llama-mtp-vocab-v1 %u %zu\n", n_vocab, map_ids.size());
    for (llama_token id : map_ids) { fprintf(map.get(), "%d\n", id); }
    require(fflush(map.get()) == 0, "flush provisional vocabulary fixture");
    const std::string map_path = "/proc/self/fd/" + std::to_string(fileno(map.get()));
    const auto in_map = [&](llama_token id) { return std::find(map_ids.begin(), map_ids.end(), id) != map_ids.end(); };
    require(!in_map(expected[0].tokens[0]), "fixture full-head sample lies outside the provisional map");

    // -2 uses the actual mapped provisional sampler; -1 forces a matching prefix; 0..3 force each first mismatch.
    for (int fault : { -2, -1, 0, 1, 2, 3 }) {
        std::string ids;
        if (fault != -2) {
            for (int j = 0; j < depth; ++j) {
                const llama_token id = j == fault ? (expected[0].tokens[j] + 1) % n_vocab : expected[0].tokens[j];
                if (j) { ids += ','; }
                ids += std::to_string(id);
            }
        }
        common_set_env("LLAMA_MTP_CHAIN_TEST_IDS", ids);
        common_set_env("LLAMA_MTP_QWEN4EXP_CHAIN", "1");
        const size_t maps_before = half_vocab_maps, compact_before = half_vocab_compact_heads;
        full_head_graph_stats graph_stats;
        contexts candidate(model.get(), config, 16, COMMON_SPECULATIVE_TYPE_DRAFT_MTP, 16, 1, 512,
                helper.get(), map_path.c_str(), true, full_head_fixture_eval, &graph_stats);
        require(half_vocab_maps == maps_before + 1, "candidate really loaded the provisional map");
        require(half_vocab_compact_heads == compact_before + 1, "candidate really created its resident compact head");
        prime_full_head_eval_counter(candidate);
        prefill(candidate, 0, prompt_size, 0);
        const snapshot cached = save(candidate, 0);
        llama_pos pos = prompt_size;
        for (int round = 0; round < rounds; ++round) {
            const auto graph_before = graph_stats;
            const auto actual = draft_full_head_round(candidate, pos, sampling, depth);
            require_full_head_round(actual, expected[round], cuda, fault, round);
            require(std::none_of(actual.tokens.begin(), actual.tokens.end(), in_map),
                    "sequential suffix repair samples outside the loaded provisional map");
            if (round == 0) {
                const int repair = fault == -1 ? 0 : depth - (fault == -2 ? 0 : fault) - 1;
                report_counter_mismatch("candidate", fault, round, actual, repair, prompt_size + depth, sampling, true);
                require_full_head_graph(graph_before, graph_stats, cuda, "candidate", fault, round, true);
                require(actual.n_eval == repair, "forced first mismatch performs exactly the required scalar suffix repairs");
                // All seven prompt rows are deferred, including pos0 with its zero pending_h carry.
                // Only the hidden copy is shifted; the final target hidden separately feeds query zero.
                // The absolute counter avoids the API's zero-to-one prompt clamp: seven plus four = eleven.
                require(actual.n_p_eval == prompt_size + depth, "full-head driver ran the multi-row chain with all deferred prompt rows");
                require(!in_map(actual.tokens[0]), "loaded map does not truncate full-head proposals or repaired draws");
                printf("FULL_HEAD_REPAIR %s fault=%d scalar_steps=%d chain_tokens=%d\n",
                        cuda ? "GPU" : "CPU", fault, actual.n_eval, actual.n_p_eval);
            } else {
                require_full_head_graph(graph_before, graph_stats, cuda, "candidate", fault, round, true);
            }
            finish_full_head_round(candidate, pos, actual, accepted[round]);
        }

        if (fault == 0) {
            const size_t compact_before = half_vocab_compact_heads;
            full_head_graph_stats resumed_stats;
            contexts resumed(model.get(), config, 16, COMMON_SPECULATIVE_TYPE_DRAFT_MTP, 16, 1, 512,
                    helper.get(), map_path.c_str(), true, full_head_fixture_eval, &resumed_stats);
            require(half_vocab_compact_heads == compact_before + 1, "restored candidate really created its resident compact head");
            prime_full_head_eval_counter(resumed);
            restore(resumed, cached, 0);
            llama_pos resumed_pos = prompt_size;
            for (int round = 0; round < rounds; ++round) {
                const auto graph_before = resumed_stats;
                const auto actual = draft_full_head_round(resumed, resumed_pos, sampling, depth);
                require_full_head_round(actual, expected[round], cuda, fault, round);
                if (round == 0) {
                    report_counter_mismatch("restored", fault, round, actual, depth - 1, prompt_size + depth, sampling, true);
                    require_full_head_graph(graph_before, resumed_stats, cuda, "restored", fault, round, true);
                    require(actual.n_eval == depth - 1 && actual.n_p_eval == prompt_size + depth,
                            "fresh prompt restore retains the real chain, catch-up, and suffix-repair paths");
                } else {
                    require_full_head_graph(graph_before, resumed_stats, cuda, "restored", fault, round, true);
                }
                finish_full_head_round(resumed, resumed_pos, actual, accepted[round]);
            }
        }
    }

    // Unsupported sampler shapes or caps above four must keep full-head sequential fallback despite the loaded map.
    for (int unsupported : { 0, 1, 2, 3 }) {
        auto fallback_sampling = sampling;
        if (unsupported == 0) { fallback_sampling.min_keep = 2; }
        if (unsupported == 1) { fallback_sampling.top_k = 5; }
        const bool backend_sampling = unsupported != 2;
        const int case_depth = unsupported == 3 ? 5 : depth;
        std::vector<full_head_round> fallback_expected;
        common_set_env("LLAMA_MTP_CHAIN_TEST_IDS", "");
        common_set_env("LLAMA_MTP_QWEN4EXP_CHAIN", "0");
        {
            contexts reference(model.get(), config, 16, COMMON_SPECULATIVE_TYPE_DRAFT_MTP, 16, 1, 512,
                    helper.get(), "none", backend_sampling, nullptr, nullptr, case_depth);
            prime_full_head_eval_counter(reference);
            prefill(reference, 0, prompt_size, 0);
            llama_pos pos = prompt_size;
            for (int round = 0; round < rounds; ++round) {
                fallback_expected.push_back(draft_full_head_round(reference, pos, fallback_sampling, case_depth));
                finish_full_head_round(reference, pos, fallback_expected.back(), accepted[round]);
            }
        }
        common_set_env("LLAMA_MTP_QWEN4EXP_CHAIN", "1");
        {
            const size_t maps_before = half_vocab_maps, compact_before = half_vocab_compact_heads;
            full_head_graph_stats fallback_stats;
            contexts fallback(model.get(), config, 16, COMMON_SPECULATIVE_TYPE_DRAFT_MTP, 16, 1, 512,
                    helper.get(), map_path.c_str(), backend_sampling, full_head_fixture_eval, &fallback_stats, case_depth);
            require(half_vocab_maps == maps_before + 1, "fallback really loaded the provisional map");
            require(half_vocab_compact_heads == compact_before + 1, "fallback really created its resident compact head");
            prime_full_head_eval_counter(fallback);
            prefill(fallback, 0, prompt_size, 0);
            llama_pos pos = prompt_size;
            int scalar_eval_work = 0, reference_scalar_eval_work = 0;
            for (int round = 0; round < rounds; ++round) {
                const auto graph_before = fallback_stats;
                const auto actual = draft_full_head_round(fallback, pos, fallback_sampling, case_depth);
                require_full_head_round(actual, fallback_expected[round], cuda, -3 - unsupported, round);
                // Widths up to four enter chain mode and defer the seven prompt rows.
                // Rejecting the sampler flushes those rows, then queues query zero before
                // llama_synchronize(): the perf API classifies the combined eight rows as
                // prompt work. The other three scalar queries count as n_eval. This one-row
                // reclassification persists in the absolute prompt count on later rounds.
                // Cap five is rejected by the constructor, so it never defers the prompt.
                const bool deferred_prompt = unsupported != 3;
                const int expected_scalar = case_depth - int(deferred_prompt && round == 0);
                const int expected_prompt = fallback_expected[round].n_p_eval + int(deferred_prompt);
                report_counter_mismatch("fallback", -3 - unsupported, round, actual, expected_scalar,
                        expected_prompt, fallback_sampling, backend_sampling);
                require_full_head_graph(graph_before, fallback_stats, cuda, "fallback", -3 - unsupported, round, false);
                const size_t scalar_steps = fallback_stats.scalar - graph_before.scalar;
                if (scalar_steps != size_t(case_depth)) {
                    fprintf(stderr, "FULL_HEAD_SCALAR_GRAPH_FAILURE %s fault=%d round=%d scalar=%zu expected=%d\n",
                            cuda ? "GPU" : "CPU", -3 - unsupported, round, scalar_steps, case_depth);
                }
                require(scalar_steps == size_t(case_depth),
                        "unsupported sampled request or cap executes every actual full-head scalar projection");
                require(actual.n_eval == expected_scalar && actual.n_p_eval == expected_prompt,
                        "sequential fallback accounts for exactly the deferred prompt and scalar queries");
                require(fallback_expected[round].n_eval == case_depth,
                        "unmapped sequential reference accounts for every scalar query");
                scalar_eval_work += actual.n_eval;
                reference_scalar_eval_work += fallback_expected[round].n_eval;
                require(scalar_eval_work + actual.n_p_eval ==
                        reference_scalar_eval_work + fallback_expected[round].n_p_eval,
                        "fallback and reference perform identical cumulative prompt plus scalar work");
                printf("FULL_HEAD_FALLBACK %s unsupported=%d round=%d scalar_projections=%zu n_eval=%d n_p_eval=%d cumulative_work=%d\n",
                        cuda ? "GPU" : "CPU", unsupported, round, scalar_steps, actual.n_eval, actual.n_p_eval,
                        scalar_eval_work + actual.n_p_eval);
                finish_full_head_round(fallback, pos, actual, accepted[round]);
            }
        }
    }
    printf("PASS Qwen4exp %s batched full head: full-vocabulary q/KLD, actual mapped guesses, every repair depth, token identity, RNG continuation, deferred catch-up, partial rejection, fresh prompt restore\n", cuda ? "GPU" : "CPU");
#else
    (void) cuda;
    require(false, "full-head fixture requires /proc/self/fd for its temporary vocabulary map");
#endif
}

static void run_prefill_identity(llama_model * model, const cache_config & config, uint32_t parallel = 2, bool borrow = true) {
    const size_t releases_before = scratch_releases;
    const uint32_t context_per_seq = borrow ? 4096 : 512;
    prefill_env_restore env;

    env.set(false, false, false);
    contexts reference(model, config, 128, COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE, 32, parallel, context_per_seq);
    env.set(true, false, false);
    contexts coalesced(model, config, 128, COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE, 32, parallel, context_per_seq);
    env.set(false, true, false);
    contexts asynchronous(model, config, 128, COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE, 32, parallel, context_per_seq);
    env.set(false, false, true);
    contexts device(model, config, 128, COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE, 32, parallel, context_per_seq);
    env.set(false, true, true);
    contexts combined(model, config, 128, COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE, 32, parallel, context_per_seq);
    env.set(false, false, false, true, false);
    contexts tight(model, config, 128, COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE, 32, parallel, context_per_seq);
    env.set(false, false, false, false, true);
    contexts scratch(model, config, 128, COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE, 32, parallel, context_per_seq);
    env.set(false, false, false, true, true);
    contexts tight_scratch(model, config, 128, COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE, 32, parallel, context_per_seq);

    struct arm {
        const char * name;
        contexts * ctx;
        bool overlap, async, device;
        bool tight = false, scratch = false;
        size_t borrowed = 0;
        size_t fallbacks = 0;
    };
    std::vector<arm> arms {
        { "legacy", &reference, false, false, false },
        { "overlap", &coalesced, true, false, false },
        { "async", &asynchronous, false, true, false },
        { "device", &device, false, false, true },
        { "async+device", &combined, false, true, true },
        { "tight", &tight, false, false, false, true, false },
        { "scratch", &scratch, false, false, false, false, true },
        { "tight+scratch", &tight_scratch, false, false, false, true, true },
    };
    auto check = [](bool condition, const arm & variant, const char * message) {
        if (!condition) {
            throw std::runtime_error(std::string(variant.name) + ": " + message);
        }
    };
    auto check_state = [&](llama_seq_id seq) {
        const snapshot a = save(reference, seq);
        for (size_t i = 1; i < arms.size(); ++i) {
            const auto & variant = arms[i];
            const snapshot b = save(*variant.ctx, seq);
            check(a.tgt == b.tgt, variant, "target state is bit-identical");
            check(a.dft == b.dft, variant, "drafter KV is bit-identical");
            check(a.spec == b.spec, variant, "carry and deferred state is bit-identical");
        }
    };
    auto process_all = [&](common_batch & batch) {
        for (auto & variant : arms) {
            env.set(variant.overlap, variant.async, variant.device, variant.tight, variant.scratch);
            common_batch input(variant.ctx->tgt);
            input.tokens = batch.tokens;
            require(llama_process(variant.ctx->tgt, LLAMA_PROCESS_TYPE_DECODE, input.get()) == 0, "identity target decode");
            const size_t borrowed_before = scratch_borrows;
            const size_t fallbacks_before = scratch_fallbacks;
            require(common_speculative_process(variant.ctx->spec, input), "identity MTP process");
            variant.borrowed += scratch_borrows - borrowed_before;
            variant.fallbacks += scratch_fallbacks - fallbacks_before;
        }
        if (std::any_of(batch.tokens.begin(), batch.tokens.end(), [](const common_batch::token & t) { return t.output; })) {
            for (size_t i = 1; i < arms.size(); ++i) {
                check(std::memcmp(llama_get_logits_ith(reference.tgt, -1), llama_get_logits_ith(arms[i].ctx->tgt, -1),
                            n_vocab * sizeof(float)) == 0, arms[i], "target logits are bit-identical");
            }
        }
    };

    std::vector<llama_pos> pos(parallel, 0);
    // Cross target/draft ubatches and the deferred-row limit, including one-token prompt tails.
    for (int width : { 128, 67, 3, 1, 127 }) {
        for (llama_seq_id seq = 0; seq < (llama_seq_id) parallel; ++seq) {
            common_batch batch(reference.tgt);
            for (int i = 0; i < width; ++i) {
                batch.add(token_at(pos[seq]), pos[seq], seq, width != 128 && i == width - 1);
                ++pos[seq];
            }
            process_all(batch);
            if (width == 128 && seq == 0) {
                snapshot copied_reference;
                for (const auto & variant : arms) {
                    env.set(variant.overlap, variant.async, variant.device, variant.tight, variant.scratch);
                    auto & ctx = *variant.ctx;
                    const snapshot source = save(ctx, 0), empty = save(ctx, 1);
                    const size_t releases = scratch_releases;
                    llama_memory_seq_cp(llama_get_memory(ctx.tgt), 0, 1, -1, -1);
                    llama_memory_seq_cp(llama_get_memory(ctx.dft), 0, 1, -1, -1);
                    check(!borrow || !variant.scratch || scratch_releases > releases, variant, "sequence copy released borrowed KV scratch");
                    ctx.tgt->memory_update(false);
                    ctx.dft->memory_update(false);
                    require(common_speculative_restore_state(ctx.spec, 1, source.spec), "copy MTP host state");
                    const snapshot copied = save(ctx, 1), unchanged = save(ctx, 0);
                    if (variant.ctx == &reference) {
                        copied_reference = copied;
                    }
                    check(copied.tgt == copied_reference.tgt && copied.dft == copied_reference.dft && copied.spec == source.spec,
                            variant, "copied target, draft KV and host state match the legacy sequence copy");
                    check(llama_memory_seq_pos_max(llama_get_memory(ctx.tgt), 0) == llama_memory_seq_pos_max(llama_get_memory(ctx.tgt), 1) &&
                            llama_memory_seq_pos_max(llama_get_memory(ctx.dft), 0) == llama_memory_seq_pos_max(llama_get_memory(ctx.dft), 1),
                            variant, "sequence copy preserves target and draft positions");
                    check(unchanged.tgt == source.tgt && unchanged.dft == source.dft && unchanged.spec == source.spec,
                            variant, "sequence copy preserves the source slot");
                    require(llama_memory_seq_rm(llama_get_memory(ctx.tgt), 1, -1, -1), "clear copied target sequence");
                    require(llama_memory_seq_rm(llama_get_memory(ctx.dft), 1, -1, -1), "clear copied draft sequence");
                    require(common_speculative_restore_state(ctx.spec, 1, empty.spec), "restore empty copied MTP slot");
                }
                check_state(0);
                check_state(1);
            }
            for (llama_seq_id check_seq = 0; check_seq < (llama_seq_id) parallel; ++check_seq) {
                check_state(check_seq);
            }
        }
    }

    // Unequal sequence spans exercise dense-row reordering with sparse logits.
    common_batch batch(reference.tgt);
    for (llama_seq_id seq : { 1, 0 }) {
        const int width = seq == 1 ? 17 : 80;
        for (int i = 0; i < width; ++i) {
            batch.add(token_at(pos[seq]), pos[seq], seq, i == width - 1);
            ++pos[seq];
        }
    }
    process_all(batch);
    check_state(0);
    check_state(1);
    for (const auto & variant : arms) {
        if (variant.scratch) {
            // Applied-batch plans can fit short tails even in the small cache.
            // That case must either decline all loans or exercise the real
            // retry when a subsequent graph outgrows its borrowed chunks.
            check(borrow ? variant.borrowed > 0 : (variant.borrowed == 0 || variant.fallbacks > 0),
                    variant, "prefill exercised the expected KV scratch loan or fallback");
        }
    }

    std::vector<snapshot> cached;
    for (const auto & variant : arms) {
        cached.push_back(save(*variant.ctx, 0));
    }
    std::vector<schedule> expected(parallel);
    for (llama_seq_id seq = 0; seq < (llama_seq_id) parallel; ++seq) {
        for (size_t i = 0; i < arms.size(); ++i) {
            const auto & variant = arms[i];
            env.set(variant.overlap, variant.async, variant.device, variant.tight, variant.scratch);
            common_speculative_begin(variant.ctx->spec, seq, llama_tokens(pos[seq], 0));
            llama_pos pos_got = pos[seq];
            const schedule got = generate(*variant.ctx, seq, pos_got, 0, 8);
            if (i == 0) {
                expected[seq] = got;
            } else {
                require_schedule(got, expected[seq], variant.name);
            }
        }
        check_state(seq);
    }
    require(!borrow || scratch_releases > releases_before, "generation released borrowed KV scratch");
    for (size_t i = 1; i < arms.size(); ++i) {
        const auto & variant = arms[i];
        env.set(false, false, false);
        contexts resumed(model, config, 128, COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE, 32, parallel, context_per_seq);
        restore(resumed, cached[i], 1);
        common_speculative_begin(resumed.spec, 1, llama_tokens(pos[0], 0));
        llama_pos pos_restored = pos[0];
        require_schedule(generate(resumed, 1, pos_restored, 0, 8), expected[0], "optimized prompt snapshot restores with switches off into another sequence");
        env.set(variant.overlap, variant.async, variant.device, variant.tight, variant.scratch);
        contexts resumed_on(model, config, 128, COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE, 32, parallel, context_per_seq);
        restore(resumed_on, cached[0], 1);
        common_speculative_begin(resumed_on.spec, 1, llama_tokens(pos[0], 0));
        pos_restored = pos[0];
        require_schedule(generate(resumed_on, 1, pos_restored, 0, 8), expected[0], "legacy prompt snapshot restores with switch on into another sequence");
    }

    // Reuse one slot at position zero while another slot retains its generated target state.
    std::vector<snapshot> other;
    for (const auto & variant : arms) {
        other.push_back(save(*variant.ctx, 1));
        require(llama_memory_seq_rm(llama_get_memory(variant.ctx->tgt), 0, -1, -1), "clear restarted target sequence");
        require(llama_memory_seq_rm(llama_get_memory(variant.ctx->dft), 0, -1, -1), "clear restarted draft sequence");
    }
    llama_pos restarted = 0;
    for (int width : { 128, 3 }) {
        common_batch input(reference.tgt);
        for (int i = 0; i < width; ++i) {
            input.add(token_at(restarted), restarted, 0, i == width - 1);
            ++restarted;
        }
        process_all(input);
        check_state(0);
        check_state(1);
    }
    for (size_t i = 0; i < other.size(); ++i) {
        check(save(arms[i].ctx->tgt, 1) == other[i].tgt, arms[i], "restarted slot preserves the other target sequence");
    }
    printf("PASS %s (%u slots, scratch %s): overlap, async, device, async+device, tight, scratch and tight+scratch match legacy target, draft KV, carry, deferred rows, logits and %u draft rounds; seq_cp, cross-switch cache restore and slot restart\n", config.name, parallel, borrow ? "loan" : "small-cache fallback", parallel * 8);
}



// Diagnose the real architecture's KV-only + output call versus a merged
// catch-up/output call, with identical teacher inputs and hidden rows. This
// measures arithmetic drift separately from stochastic trajectory divergence.
static void run_qwen4exp_defer_math(bool cuda = false, const char * model_path = nullptr) {
    model_geometry geometry;
    geometry.kpool = 4;
    geometry.indexer_top_k = 8;
    auto * device = cuda ? ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) : nullptr;
    require(!cuda || device, "defer arithmetic GPU backend is available");
    auto mp = llama_model_default_params();
    ggml_backend_dev_t devices[] = { device, nullptr };
    mp.devices = devices;
    mp.n_gpu_layers = cuda ? 999 : 0;
    mp.load_mtp = true;
    llama_model_ptr model(model_path ? llama_model_load_from_file(model_path, mp) :
            make_model(false, geometry, LLM_ARCH_QWEN4EXP, device));
    require(model && model->arch == LLM_ARCH_QWEN4EXP && model->hparams.n_layer_nextn == 1,
            "defer arithmetic model is single-head Qwen4exp");
    const int vocab_size = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    printf("BOOKKEEP_DEFER_MATH_MODEL projection=%s head=%s embedding_width=%d model=%s\n",
            ggml_type_name(model->layers[model->hparams.n_layer()].nextn.eh_proj->type),
            ggml_type_name(model->output->type), llama_model_n_embd_out(model.get()),
            model_path ? model_path : "generated_f32");
    const char * f16_kv_env = std::getenv("LLAMA_BOOKKEEP_DEFER_F16_KV");
    const bool f16_kv = !model_path || (f16_kv_env && std::strcmp(f16_kv_env, "1") == 0);
    auto make_context = [&]() {
        auto cp = llama_context_default_params();
        cp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        cp.n_ctx = 128;
        cp.n_batch = cp.n_ubatch = 8;
        cp.n_seq_max = 1;
        cp.n_threads = cp.n_threads_batch = 2;
        cp.offload_kqv = cuda;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.type_k = f16_kv ? GGML_TYPE_F16 : GGML_TYPE_TQ5_0;
        cp.type_v = f16_kv ? GGML_TYPE_F16 : GGML_TYPE_TURBO4_0;
        llama_context_ptr ctx(llama_init_from_model(model.get(), cp));
        require(ctx != nullptr, "create defer arithmetic helper");
        llama_set_embeddings_nextn(ctx.get(), true, true);
        return ctx;
    };
    auto reference = make_context();
    auto merged = make_context();
    const auto backend_before = llama_perf_bookkeep_context(reference.get());
    uint64_t teacher_digest[2] = { UINT64_C(14695981039346656037), UINT64_C(14695981039346656037) };
    std::array<std::array<std::array<uint64_t, 11>, 9>, 2> backend_cells{};
    const size_t width = llama_model_n_embd_out(model.get());
    auto decode = [&](llama_context * ctx, llama_pos from, int count, bool output) {
        common_batch batch(ctx);
        // common_batch borrows each embedding until get()/decode. Retain a
        // distinct stable row per logical position, identical across batchings.
        std::vector<float> hidden(width * count);
        for (int i = 0; i < count; ++i) {
            const auto pos = from + i;
            float * row_hidden = hidden.data() + width * i;
            for (size_t j = 0; j < width; ++j) {
                row_hidden[j] = float((int) ((j + pos) % 19) - 9) / 19.0f;
            }
            const int row = batch.add(token_at(pos), pos, 0, output && i == count - 1);
            require(batch.set_embd(row, { row_hidden, 1, width }), "attach defer arithmetic teacher row");
        }
        auto * rendered = batch.get();
        const int index = ctx == reference.get() ? 0 : 1;
        require(ctx == reference.get() || ctx == merged.get(), "known teacher context");
        teacher_digest[index] = check_teacher_batch(*rendered, width, teacher_digest[index]);
        const auto before = llama_perf_bookkeep_context(ctx);
        require(llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, rendered) == 0, "defer arithmetic decode");
        const auto after = llama_perf_bookkeep_context(ctx);
        require(count > 0 && count <= 8, "bounded teacher batch width");
        auto & cell = backend_cells[output][count];
        ++cell[0];
        for (size_t i = 0; i < 10; ++i) {
            cell[i + 1] += after.backend_submit[i] - before.backend_submit[i];
        }
        if (!output) {
            llama_synchronize(ctx);
            return std::vector<float>();
        }
        const auto * logits = llama_get_logits_ith(ctx, -1);
        require(logits != nullptr, "defer arithmetic output row");
        return std::vector<float>(logits, logits + vocab_size);
    };
    llama_pos pos = 0;
    while (pos < 37) {
        const int count = std::min(8, 37 - pos);
        decode(reference.get(), pos, count, false);
        decode(merged.get(), pos, count, false);
        pos += count;
    }
    size_t rows = 0, exact_rows = 0, exact_states = 0, same_top1 = 0;
    double sum_kld = 0, max_kld = 0, max_logit_error = 0;
    auto compare = [&](const std::vector<float> & a, const std::vector<float> & b) {
        require(a.size() == (size_t) vocab_size && b.size() == (size_t) vocab_size, "complete defer arithmetic distributions");
        const double amax = *std::max_element(a.begin(), a.end());
        const double bmax = *std::max_element(b.begin(), b.end());
        double az = 0, bz = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            require(std::isfinite(a[i]) && std::isfinite(b[i]), "finite defer arithmetic logits");
            az += std::exp(double(a[i]) - amax);
            bz += std::exp(double(b[i]) - bmax);
            max_logit_error = std::max(max_logit_error, std::abs(double(a[i]) - b[i]));
        }
        double kld = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            const double logp = double(a[i]) - amax - std::log(az);
            const double logq = double(b[i]) - bmax - std::log(bz);
            kld += std::exp(logp)*(logp - logq);
        }
        require(std::isfinite(kld) && kld >= -1e-12, "finite nonnegative defer arithmetic KLD");
        kld = std::max(0.0, kld);
        sum_kld += kld;
        max_kld = std::max(max_kld, kld);
        exact_rows += std::memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0;
        same_top1 += std::max_element(a.begin(), a.end()) - a.begin() ==
                     std::max_element(b.begin(), b.end()) - b.begin();
        printf("BOOKKEEP_DEFER_MATH_ROW row=%zu kld=%.17g\n", rows, kld);
        ++rows;
    };
    for (int accepted : { 0, 1, 3, 4, 2, 0, 4, 1, 3, 2, 0, 4 }) {
        if (accepted > 0) { decode(reference.get(), pos, accepted, false); }
        compare(decode(reference.get(), pos + accepted, 1, true), decode(merged.get(), pos, accepted + 1, true));
        pos += accepted + 1;
        require(llama_memory_seq_pos_max(llama_get_memory(reference.get()), 0) == pos - 1 &&
                llama_memory_seq_pos_max(llama_get_memory(merged.get()), 0) == pos - 1,
                "defer arithmetic committed positions match");
        exact_states += save(reference.get(), 0) == save(merged.get(), 0);
        // Continue at identical teacher inputs to expose accumulated cache drift.
        compare(decode(reference.get(), pos, 1, true), decode(merged.get(), pos, 1, true));
        ++pos;
    }
    require(rows == 24, "nonempty complete defer arithmetic cell");
    require(teacher_digest[0] == teacher_digest[1], "separate and merged serialized teacher transcripts are identical");
    printf("BOOKKEEP_DEFER_INPUTS separate=%016llx merged=%016llx identical=1\n",
            (unsigned long long) teacher_digest[0], (unsigned long long) teacher_digest[1]);
    printf("BOOKKEEP_DEFER_MATH architecture=qwen4exp backend=%s dtype=%s kv=%s rows=%zu exact_rows=%zu exact_states=%zu/12 vocab=%d top1_equal=%zu/24 mean_kld=%.17g max_kld=%.17g max_logit_error=%.17g teacher=deterministic_synthetic quality_review_required=1\n",
            cuda ? "CUDA" : "CPU", model_path ? "GGUF" : "F32", f16_kv ? "F16" : "TQ5_0/TURBO4_0",
            rows, exact_rows, exact_states, vocab_size, same_top1, sum_kld/rows, max_kld, max_logit_error);
    const auto backend_after = llama_perf_bookkeep_context(reference.get());
    static const char * names[] = { "session_us", "transition_us", "input_copy_us", "user_input_sync_nested_us",
        "cpu_compute_us", "gpu_submit_us", "other_compute_us", "cpu_splits", "gpu_splits", "other_splits" };
    printf("BOOKKEEP_DEFER_BACKEND");
    for (size_t i = 0; i < 10; ++i) {
        printf(" %s=%llu", names[i], (unsigned long long) (backend_after.backend_submit[i] - backend_before.backend_submit[i]));
    }
    printf("\n");
    for (size_t output = 0; output < backend_cells.size(); ++output) {
        for (size_t count = 1; count < backend_cells[output].size(); ++count) {
            const auto & cell = backend_cells[output][count];
            if (cell[0] == 0) { continue; }
            printf("BOOKKEEP_DEFER_BACKEND_CELL output=%zu width=%zu calls=%llu", output, count,
                    (unsigned long long) cell[0]);
            for (size_t i = 0; i < 10; ++i) {
                printf(" %s=%llu", names[i], (unsigned long long) cell[i + 1]);
            }
            printf("\n");
        }
    }
}

static void run_case(llama_model * model, const cache_config & config, bool deferred) {
    contexts reference(model, config);
    const llama_pos boundary = 512 + (deferred ? 3 : 0);
    const llama_pos prompt_end = boundary + (deferred ? 5 : 128);
    prefill(reference, 0, boundary, 0);
    const snapshot prefix = save(reference, 0);
    prefill(reference, boundary, prompt_end, 0);
    const snapshot prompt = save(reference, 0);
    common_speculative_begin(reference.spec, 0, llama_tokens(prompt_end, 0));
    llama_pos pos_ref = prompt_end;
    const schedule expected = generate(reference, 0, pos_ref, 0, 8);
    const snapshot middle = save(reference, 0);
    const llama_pos middle_pos = pos_ref;
    const schedule expected_tail = generate(reference, 0, pos_ref, 8, 8);
    require(std::find(expected.widths.begin(), expected.widths.end(), 3) != expected.widths.end() &&
            std::find(expected.widths.begin(), expected.widths.end(), 4) != expected.widths.end(), "schedule exercises adaptive climb");

    contexts resumed(model, config);
    std::vector<uint8_t> cold;
    require(common_speculative_get_state(resumed.spec, 0, cold), "save empty sequence state");
    prefill(resumed, 0, 259, 0);
    std::vector<uint8_t> other_before, other_after;
    require(common_speculative_get_state(resumed.spec, 0, other_before), "save other sequence carry");
    restore(resumed, prefix, 1);
    require(common_speculative_get_state(resumed.spec, 0, other_after) && other_before == other_after, "restore preserves another sequence carry and deferred rows");
    // Clear the other sequence after checking isolation, so its deferred rows cannot change the draft batch shape.
    require(llama_memory_seq_rm(llama_get_memory(resumed.tgt), 0, -1, -1), "clear other target sequence");
    require(llama_memory_seq_rm(llama_get_memory(resumed.dft), 0, -1, -1), "clear other draft sequence");
    require(common_speculative_restore_state(resumed.spec, 0, cold), "restore empty sequence state");
    prefill(resumed, boundary, prompt_end, 1);
    common_speculative_begin(resumed.spec, 1, llama_tokens(prompt_end, 0));
    llama_pos pos_got = prompt_end;
    require_schedule(generate(resumed, 1, pos_got, 0, 8), expected, "prefill restore draft tokens and round widths");

    contexts cached_prompt(model, config);
    restore(cached_prompt, prompt, 1);
    common_speculative_begin(cached_prompt.spec, 1, llama_tokens(prompt_end, 0));
    pos_got = prompt_end;
    require_schedule(generate(cached_prompt, 1, pos_got, 0, 8), expected, "complete prompt restore draft tokens and round widths");

    contexts continued(model, config);
    restore(continued, middle, 1);
    pos_got = middle_pos;
    require_schedule(generate(continued, 1, pos_got, 8, 8), expected_tail, "generation restore draft tokens and adaptive round widths");
    printf("PASS %s: %s, fresh driver and sequence 0 -> 1, 24 matching draft rounds\n", config.name, deferred ? "deferred carry" : "batch carry");
}

static void run_deferred_overflow(llama_model * model, const cache_config & config) {
    contexts reference(model, config, 64);
    prefill(reference, 0, 124, 0);
    require(llama_memory_seq_pos_max(llama_get_memory(reference.dft), 0) == 63, "sixty real rows remain deferred");
    const snapshot prefix = save(reference, 0);
    prefill(reference, 124, 188, 0);
    common_speculative_begin(reference.spec, 0, llama_tokens(188, 0));
    llama_pos pos = 188;
    const schedule expected = generate(reference, 0, pos, 0, 8);

    contexts resumed(model, config, 64);
    std::vector<uint8_t> cold;
    require(common_speculative_get_state(resumed.spec, 0, cold), "save empty overflow sequence state");
    restore(resumed, prefix, 0);
    restore(resumed, prefix, 1);
    // Each snapshot fits the decode capacity; together their 120 deferred rows need more than one flush batch.
    prefill(resumed, 124, 188, 0);
    require(llama_memory_seq_pos_max(llama_get_memory(resumed.dft), 0) == 187, "overflow flush retains first sequence rows");
    require(llama_memory_seq_pos_max(llama_get_memory(resumed.dft), 1) == 123, "overflow flush retains all other sequence rows");
    prefill(resumed, 124, 188, 1);
    common_speculative_begin(resumed.spec, 0, llama_tokens(188, 0));
    pos = 188;
    require_schedule(generate(resumed, 0, pos, 0, 8), expected, "overflow restore first sequence draft tokens and round widths");

    require(llama_memory_seq_rm(llama_get_memory(resumed.tgt), 0, -1, -1), "clear first overflow target sequence");
    require(llama_memory_seq_rm(llama_get_memory(resumed.dft), 0, -1, -1), "clear first overflow draft sequence");
    require(common_speculative_restore_state(resumed.spec, 0, cold), "restore empty overflow sequence state");
    common_speculative_begin(resumed.spec, 1, llama_tokens(188, 0));
    pos = 188;
    require_schedule(generate(resumed, 1, pos, 0, 8), expected, "overflow restore other sequence draft tokens and round widths");

    common_params_sampling sampling;
    sampling.temp = 0.0f;
    llama_tokens draft_prompt, expected_drafts[2], drafts[2];
    auto prepare = [&](contexts & ctx, llama_seq_id seq, llama_tokens & result) {
        auto & dp = common_speculative_get_draft_params(ctx.spec, seq);
        dp = {};
        dp.drafting = true;
        dp.pos0 = 124;
        dp.id_last = token_at(124);
        dp.prompt = &draft_prompt;
        dp.result = &result;
        dp.sampling = &sampling;
    };
    contexts fresh(model, config, 64);
    for (llama_seq_id seq = 0; seq < 2; ++seq) {
        prefill(fresh, 0, 124, seq);
        common_speculative_begin(fresh.spec, seq, llama_tokens(124, 0));
        prepare(fresh, seq, expected_drafts[seq]);
    }
    common_speculative_draft(fresh.spec);
    for (llama_seq_id seq = 0; seq < 2; ++seq) {
        require(expected_drafts[seq].size() == 3, "fresh simultaneous sequences draft at adaptive floor");
    }

    contexts simultaneous(model, config, 64);
    for (llama_seq_id seq = 0; seq < 2; ++seq) {
        restore(simultaneous, prefix, seq);
        common_speculative_begin(simultaneous.spec, seq, llama_tokens(124, 0));
        prepare(simultaneous, seq, drafts[seq]);
    }
    common_speculative_draft(simultaneous.spec);
    for (llama_seq_id seq = 0; seq < 2; ++seq) {
        require_schedule({ { drafts[seq].size() }, { drafts[seq] } },
                         { { expected_drafts[seq].size() }, { expected_drafts[seq] } },
                         "simultaneous restored deferred drafts match fresh tokens and width");
    }
    printf("PASS %s: two restored 60-row deferred states with a 64-row decode capacity, 16 matching draft rounds plus two simultaneous drafts\n", config.name);
}

static void run_draft_vocab_reserve(llama_model * model) {
    const ggml_init_params ip { 3 * ggml_tensor_overhead(), nullptr, true };
    ggml_context_ptr meta(ggml_init(ip));
    require(meta != nullptr, "create draft shortlist tensor metadata");
    auto * head = ggml_new_tensor_2d(meta.get(), GGML_TYPE_F32, 5120, 248320);
    auto * compact = ggml_new_tensor_2d(meta.get(), GGML_TYPE_F32, 5120, 65536);
    auto * ids = ggml_new_tensor_1d(meta.get(), GGML_TYPE_I32, 65536);
    for (uint32_t rows : { 1u, 4u }) {
        for (bool reserve : { false, true }) {
            llm_graph_result result(32);
            llm_graph_params params {};
            params.arch = model->arch;
            params.hparams = model->hparams;
            params.gtype = LLM_GRAPH_TYPE_DECODER_MTP;
            params.res = &result;
            params.n_outputs = rows;
            params.is_reserve = reserve;
            params.draft_vocab_ids = ids;
            params.draft_vocab_compact = compact;
            params.draft_vocab_compact_src = head;
            std::vector<int8_t> output(rows, 1);
            std::vector<int32_t> n_seq_id(rows, 1);
            std::vector<llama_seq_id> seq_id(rows);
            std::vector<llama_seq_id *> seq_ids(rows);
            for (uint32_t i = 0; i < rows; ++i) {
                seq_id[i] = i;
                seq_ids[i] = &seq_id[i];
            }
            params.ubatch.n_tokens = rows;
            params.ubatch.output = output.data();
            params.ubatch.n_seq_id = n_seq_id.data();
            params.ubatch.seq_id = seq_ids.data();
            auto * cur = ggml_new_tensor_2d(result.get_ctx(), GGML_TYPE_F32, 5120, rows);
            llm_graph_context graph(params);
            auto * logits = graph.build_draft_vocab_logits(head, nullptr, cur);
            if (reserve) {
                require(logits && logits->ne[0] == 65536 && logits->ne[1] == rows &&
                        ggml_nbytes(logits) == rows * 256 * 1024 && result.t_logits_ids == ids,
                        "reserve without samplers uses the compact 64K draft head");
            } else {
                require(!logits && !result.t_logits_ids, "decode without a backend sampler keeps the full-head fallback");
            }
            params.n_outputs = 0;
            llm_graph_context kv_only(params);
            require(!kv_only.build_draft_vocab_logits(head, nullptr, cur), "KV-only reserve does not build draft logits");
        }
    }
    printf("PASS draft shortlist: reserve without samplers uses 64K x 1/4 logits (256 KiB per row); decode fallback and KV-only behavior stay intact\n");
}

static void run_scratch_target_lifetime(llama_model * model) {
    prefill_env_restore env;
    env.set(false, false, false, false, true);
    common_set_env("LLAMA_SHARED_COMPUTE", "0");
    common_set_env("LLAMA_MTP_UPDATE_KV_ONLY", "1");
    contexts ctx(model, { "scratch target lifetime", GGML_TYPE_Q8_0, GGML_TYPE_Q8_0 },
            128, COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE, 32);
    const size_t borrowed = scratch_borrows;
    prefill(ctx, 0, 128, 0);
    require(scratch_borrows > borrowed, "target lifetime fixture acquired a KV scratch loan");
    common_speculative_free(ctx.spec);
    ctx.spec = nullptr;
    const size_t released = scratch_releases;
    llama_free(ctx.tgt);
    ctx.tgt = nullptr;
    require(scratch_releases > released, "destroying target released its draft scratch loan");
    const size_t after_destroy = scratch_borrows;
    std::vector<float> hidden(32*n_embd, 0.0f);
    common_batch input(ctx.dft);
    for (llama_pos pos = 128; pos < 160; ++pos) {
        const int idx = input.add(token_at(pos), pos, 0, false);
        require(input.set_embd(idx, { hidden.data() + size_t(pos - 128)*n_embd, 1, n_embd }), "attach draft hidden row");
    }
    require(llama_process(ctx.dft, LLAMA_PROCESS_TYPE_DECODE,
                checked_embedding_batch(input, hidden, n_embd)) == 0, "draft catch-up succeeds after target destruction");
    require(llama_memory_seq_pos_max(llama_get_memory(ctx.dft), 0) == 159 && scratch_borrows == after_destroy,
            "draft uses private compute after its scratch target is destroyed");
    printf("PASS scratch lifetime: target destruction releases the active loan; the independent draft still decodes without shared compute\n");
}

// Snapshot must be the first output access: copy borrowed arrays before the legacy
// getters can synchronize or lazily reorder them and hide an incorrect snapshot.
static llama_sampler_output check_sampler_output(llama_context * ctx, int32_t idx) {
    const auto out = llama_get_sampler_output_ith(ctx, idx);
    require(out.probs || out.probs_count == 0, "snapshot absent probs count");
    require(out.logits || out.logits_count == 0, "snapshot absent sampled logits count");
    require(out.raw_logits || out.raw_logits_count == 0, "snapshot absent raw logits count");
    require(out.candidates || out.candidates_count == 0, "snapshot absent candidates count");
    require(out.probs_count <= n_vocab && out.logits_count <= n_vocab &&
            out.raw_logits_count <= n_vocab && out.candidates_count <= n_vocab, "snapshot bounded counts");
    std::vector<float> probs, logits, raw;
    std::vector<llama_token> candidates;
    if (out.probs_count) { probs.assign(out.probs, out.probs + out.probs_count); }
    if (out.logits_count) { logits.assign(out.logits, out.logits + out.logits_count); }
    if (out.raw_logits_count) { raw.assign(out.raw_logits, out.raw_logits + out.raw_logits_count); }
    // The public candidate getter promises the complete vocabulary when count is zero.
    if (out.candidates) {
        const auto count = out.candidates_count ? out.candidates_count : n_vocab;
        candidates.assign(out.candidates, out.candidates + count);
    }
    require(out.token == llama_get_sampled_token_ith(ctx, idx), "snapshot sampled token identity");
    require(out.probs == llama_get_sampled_probs_ith(ctx, idx), "snapshot probs pointer identity");
    require(out.logits == llama_get_sampled_logits_ith(ctx, idx), "snapshot sampled logits pointer identity");
    require(out.candidates == llama_get_sampled_candidates_ith(ctx, idx), "snapshot candidates pointer identity");
    require(out.probs_count == (out.probs ? llama_get_sampled_probs_count_ith(ctx, idx) : 0),
            "snapshot guarded probs count");
    // Legacy sampled-logit count can be n_vocab even when no sampled-logit buffer exists.
    require(out.logits_count == (out.logits ? llama_get_sampled_logits_count_ith(ctx, idx) : 0),
            "snapshot guarded sampled logits count");
    require(out.candidates_count == ((out.probs || out.logits) ?
            llama_get_sampled_candidates_count_ith(ctx, idx) : 0), "snapshot guarded candidates count");
    if (!probs.empty()) {
        require(std::memcmp(probs.data(), out.probs, probs.size() * sizeof(float)) == 0, "snapshot probs bytes");
    }
    if (!logits.empty()) {
        require(std::memcmp(logits.data(), out.logits, logits.size() * sizeof(float)) == 0, "snapshot sampled logits bytes");
    }
    if (!candidates.empty()) {
        require(std::memcmp(candidates.data(), out.candidates, candidates.size() * sizeof(llama_token)) == 0,
                "snapshot candidates bytes");
    }
    const bool needs_raw = out.token == LLAMA_TOKEN_NULL && !out.probs && !out.logits;
    require((out.raw_logits != nullptr) == needs_raw, "snapshot raw logits only for CPU sampling");
    if (needs_raw) {
        require(out.raw_logits == llama_get_logits_ith(ctx, idx), "snapshot raw logits pointer identity");
        require(out.raw_logits_count == n_vocab, "snapshot raw logits vocabulary size");
        require(std::memcmp(raw.data(), out.raw_logits, raw.size() * sizeof(float)) == 0, "snapshot raw logits bytes");
    }
    // Do not call the raw getter for compact token-only sampling: no raw allocation is required.
    return out;
}

static void run_sampler_output(llama_model * model) {
    enum output_mode { RAW, TOP_K, DIST, TOP_K_DIST, GREEDY };
    const char * names[] = { "raw", "top-k", "dist", "top-k+dist", "compact greedy" };
    constexpr int32_t k = 5;
    for (output_mode mode : { RAW, TOP_K, DIST, TOP_K_DIST, GREEDY }) {
        std::vector<llama_sampler_ptr> chains;
        std::vector<llama_sampler_seq_config> configs;
        if (mode != RAW) {
            for (llama_seq_id seq = 0; seq < 2; ++seq) {
                chains.emplace_back(llama_sampler_chain_init(llama_sampler_chain_default_params()));
                if (mode == TOP_K || mode == TOP_K_DIST) {
                    llama_sampler_chain_add(chains.back().get(), llama_sampler_init_top_k(k));
                }
                if (mode == DIST || mode == TOP_K_DIST) {
                    llama_sampler_chain_add(chains.back().get(), llama_sampler_init_dist(1234 + seq));
                }
                if (mode == GREEDY) {
                    llama_sampler_chain_add(chains.back().get(), llama_sampler_init_greedy());
                }
                configs.push_back({ seq, chains.back().get() });
            }
        }
        const bool multi_output = mode == RAW || mode == GREEDY;
        const int n_positions = multi_output ? 3 : 2;
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = 256;
        cp.n_batch = cp.n_ubatch = 8;
        cp.n_outputs_max = 4;
        cp.n_outputs_max_per_seq = multi_output ? 2 : 1;
        cp.n_seq_max = 2;
        cp.n_rs_seq = 4;
        cp.n_threads = cp.n_threads_batch = 2;
        cp.kv_unified = false;
        cp.offload_kqv = false;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.type_k = cp.type_v = GGML_TYPE_Q8_0;
        cp.samplers = configs.data();
        cp.n_samplers = configs.size();
        // The context borrows the chains, so it must be destroyed before them.
        llama_context_ptr ctx(llama_init_from_model(model, cp));
        require(ctx != nullptr, "create CPU sampler output context");
        for (int round = 0; round < 2; ++round) {
            common_batch batch(ctx.get());
            std::vector<int32_t> rows;
            for (int pos = 0; pos < n_positions; ++pos) {
                for (int slot = 0; slot < 2; ++slot) {
                    const llama_seq_id seq = (slot + round) % 2;
                    const auto idx = batch.add((pos + seq + round + 1) % n_vocab,
                            round * n_positions + pos, seq, pos > 0);
                    if (pos > 0) { rows.push_back(idx); }
                }
            }
            require(llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0, "CPU sampler output decode");
            // Negative lookup is deliberately the first access after each decode.
            const auto last = check_sampler_output(ctx.get(), -1);
            for (size_t row = 0; row < rows.size(); ++row) {
                const auto out = check_sampler_output(ctx.get(), rows[row]);
                const auto neg = check_sampler_output(ctx.get(), int32_t(row) - int32_t(rows.size()));
                require(out.token == neg.token && out.probs == neg.probs && out.logits == neg.logits &&
                        out.candidates == neg.candidates && out.raw_logits == neg.raw_logits,
                        "snapshot positive and negative row mapping");
                require(out.probs_count == neg.probs_count && out.logits_count == neg.logits_count &&
                        out.candidates_count == neg.candidates_count && out.raw_logits_count == neg.raw_logits_count,
                        "snapshot positive and negative row counts");
                if (row + 1 == rows.size()) {
                    require(last.token == out.token && last.probs == out.probs && last.logits == out.logits &&
                            last.raw_logits == out.raw_logits, "snapshot first negative lookup maps last row");
                }
                require(out.candidates != nullptr, "snapshot candidates include full vocabulary fallback");
                if (mode == RAW) {
                    require(out.token == LLAMA_TOKEN_NULL && !out.probs && !out.logits && out.raw_logits,
                            "snapshot exercises raw branch");
                } else if (mode == TOP_K) {
                    require(out.token == LLAMA_TOKEN_NULL && !out.probs && out.logits &&
                            out.logits_count == k && out.candidates_count == k, "snapshot exercises partial top-k branch");
                } else {
                    require(out.token >= 0 && out.token < int32_t(n_vocab), "snapshot exercises backend sampled token");
                    if (mode == GREEDY) {
                        require(!out.probs && !out.logits && !out.raw_logits, "snapshot exercises token-only branch");
                    } else {
                        const auto count = mode == DIST ? n_vocab : uint32_t(k);
                        require(out.probs && out.logits && out.probs_count == count && out.logits_count == count,
                                "snapshot exercises backend distribution branch");
                        require(out.candidates_count == (mode == DIST ? 0 : uint32_t(k)),
                                "snapshot full and compact distribution candidates");
                    }
                }
                const auto count = out.candidates_count ? out.candidates_count : n_vocab;
                for (uint32_t i = 0; i < count; ++i) {
                    require(out.candidates[i] >= 0 && out.candidates[i] < int32_t(n_vocab), "snapshot candidate token range");
                    if (!out.candidates_count) {
                        require(out.candidates[i] == int32_t(i), "snapshot full vocabulary candidate fallback");
                    }
                }
            }
        }
        printf("PASS CPU sampler output snapshot %s, two sequences, positive/negative rows\n", names[mode]);
    }
}

// Preserve order and floating-point bits without hashing struct padding or addresses.
static std::vector<uint32_t> sampler_words(common_sampler * sampler, llama_token token) {
    const auto * cur = common_sampler_get_candidates(sampler, false);
    require(cur && cur->size >= 2 && cur->size <= 5 && cur->selected >= 0 &&
            size_t(cur->selected) < cur->size, "common sampler has selected compact candidates");
    std::vector<uint32_t> words = { uint32_t(token), uint32_t(cur->size), uint32_t(cur->selected), uint32_t(cur->sorted) };
    for (size_t i = 0; i < cur->size; ++i) {
        const auto & c = cur->data[i];
        require(c.id >= 5 && c.id < int32_t(n_vocab), "common sampler excludes repeated bans");
        require(c.p >= 0.0f && c.p <= 1.0f, "common sampler normalized candidate probability");
        uint32_t logit, prob;
        std::memcpy(&logit, &c.logit, sizeof(logit));
        std::memcpy(&prob, &c.p, sizeof(prob));
        words.insert(words.end(), { uint32_t(c.id), logit, prob });
    }
    return words;
}

static void sampler_digest(uint64_t & digest, const std::vector<uint32_t> & words) {
    for (const auto word : words) {
        for (int shift = 0; shift < 32; shift += 8) {
            digest = (digest ^ ((word >> shift) & 0xff)) * UINT64_C(1099511628211);
        }
    }
}

static void run_common_sampler(llama_model * model) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 128;
    cp.n_batch = cp.n_ubatch = 4;
    cp.n_seq_max = 1;
    cp.n_rs_seq = 4;
    cp.n_threads = cp.n_threads_batch = 2;
    cp.offload_kqv = false;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k = cp.type_v = GGML_TYPE_Q8_0;
    llama_context_ptr ctx(llama_init_from_model(model, cp));
    require(ctx != nullptr, "create CPU common sampler context");
    common_batch batch(ctx.get());
    for (int pos = 0; pos < 3; ++pos) { batch.add(token_at(pos), pos, 0, true); }
    require(llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0, "common sampler fixture decode");

    common_params_sampling params;
    params.seed = 12345;
    params.top_k = 5;
    params.top_p = 0.9f;
    params.min_keep = 2;
    params.temp = 0.85f;
    params.no_perf = true;
    params.samplers = { COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_TOP_P, COMMON_SAMPLER_TYPE_TEMPERATURE };
    const float ban = -std::numeric_limits<float>::infinity();
    // Production has the same five -inf request/EOG bans repeated in its merged list.
    for (int repeat = 0; repeat < 2; ++repeat) {
        for (llama_token token = 0; token < 5; ++token) { params.logit_bias.push_back({ token, ban }); }
    }
    common_sampler_ptr sampler(common_sampler_init(model, params));
    require(sampler != nullptr && common_sampler_pq_supported(sampler.get()), "initialize CPU common sampler with p/q");
    uint64_t digest = UINT64_C(14695981039346656037);
    const auto draw = [&](common_sampler * smpl, int idx) {
        const auto token = common_sampler_sample(smpl, ctx.get(), idx);
        const auto words = sampler_words(smpl, token);
        const auto * cur = common_sampler_get_candidates(smpl, false);
        require(cur->data[cur->selected].id == token, "common sample selects returned token");
        common_sampler_accept(smpl, token, true);
        require(common_sampler_last(smpl) == token, "common accept updates history");
        return words;
    };
    common_sampler_accept(sampler.get(), 7, false);
    require(common_sampler_last(sampler.get()) == 7, "common prompt accept updates history");
    std::vector<std::vector<uint32_t>> initial;
    for (int i = 0; i < 8; ++i) {
        initial.push_back(draw(sampler.get(), i % 3));
        sampler_digest(digest, initial.back());
    }
    // Sample before inspecting clone candidates: the existing clone aliases its source's cur_p.data.
    common_sampler_ptr clone(common_sampler_clone(sampler.get()));
    common_sampler_ptr copied(common_sampler_init(model, params));
    common_sampler_copy(sampler.get(), copied.get());
    for (int i = 0; i < 4; ++i) {
        const auto words = draw(sampler.get(), (i + 1) % 3);
        require(words == draw(clone.get(), (i + 1) % 3), "common clone ordinary draw and candidate bytes");
        require(words == draw(copied.get(), (i + 1) % 3), "common copy ordinary draw and candidate bytes");
        sampler_digest(digest, words);
    }
    common_sampler_reset(sampler.get());
    common_sampler_accept(sampler.get(), 7, false);
    for (int i = 0; i < 8; ++i) {
        const auto words = draw(sampler.get(), i % 3);
        require(words == initial[i], "common reset reproduces dist draws and ordered candidate bytes");
        sampler_digest(digest, words);
    }
    // p/q copy/reset state is outside this test: neither API currently restores that RNG and its stats.
    for (int path = 0; path < 3; ++path) {
        common_sampler_ptr probe(common_sampler_clone(sampler.get()));
        const auto predicted = common_sampler_sample(probe.get(), ctx.get(), 0);
        const auto * p = common_sampler_get_candidates(probe.get(), false);
        std::vector<llama_token_data> q(p->data, p->data + p->size);
        sampler_digest(digest, sampler_words(probe.get(), predicted));
        llama_token draft = predicted;
        if (path == 0) {
            // q == p gives acceptance probability one, including when the target draw differs.
            for (const auto & c : q) {
                if (c.id != predicted && c.p > 0.0f && c.p < 1.0f) { draft = c.id; break; }
            }
            require(draft != predicted, "p/q fixture has a positive alternative candidate");
        } else if (path == 2) {
            draft = 0;
            q = { { draft, 0.0f, 1.0f } };
        }
        const auto before = common_sampler_get_pq_stats(sampler.get());
        const auto accepted = common_sampler_sample_and_accept_n_pq(sampler.get(), ctx.get(), { 0, 1 }, { draft }, { q });
        const auto after = common_sampler_get_pq_stats(sampler.get());
        require(after.n_pos == before.n_pos + 1, "p/q CPU candidate path executed");
        require(accepted.size() == (path == 2 ? 1 : 2), "p/q accept includes bonus, reject stops immediately");
        if (path == 0) {
            require(accepted.front() == draft && after.n_extra == before.n_extra + 1, "p/q residual acceptance executed");
        } else if (path == 1) {
            require(accepted.front() == draft && after.n_match == before.n_match + 1, "p/q id-match acceptance executed");
        } else {
            require(accepted.front() != draft && after.n_rej == before.n_rej + 1, "p/q residual rejection executed");
        }
        sampler_digest(digest, { uint32_t(path), uint32_t(accepted.size()) });
        for (const auto token : accepted) { sampler_digest(digest, { uint32_t(token) }); }
        sampler_digest(digest, sampler_words(sampler.get(), accepted.back()));
        require(common_sampler_last(sampler.get()) == accepted.back(), "p/q updates accepted history");
    }
    const auto & stats = common_sampler_get_pq_stats(sampler.get());
    require(stats.n_pos == 3 && stats.n_match == 1 && stats.n_extra == 1 && stats.n_rej == 1,
            "all three deterministic p/q outcomes exercised");
    const char * timing = std::getenv("LLAMA_BOOKKEEP_TIMING");
    if (timing && std::atoi(timing) != 0) {
        const auto perf = common_sampler_get_prep_perf(sampler.get());
        require(perf.prepare_calls > 0 && perf.prepare_us >= perf.fused_topk_us,
                "candidate preparation timer includes fused selection");
        const bool fused = common_get_env("LLAMA_BOOKKEEP_SAMPLER") == "1";
        require((perf.fused_topk_calls > 0) == fused, "fused selection timer follows the actual path");
    }
    printf("BOOKKEEP_SAMPLER digest=%016llx pq_pos=%zu match=%zu extra=%zu reject=%zu\n",
            (unsigned long long) digest, stats.n_pos, stats.n_match, stats.n_extra, stats.n_rej);
}

// fork (bookkeep): compare production range removal with a physical-cell scan.
static void run_kv_removal_metadata(llama_model * model) {
    llama_kv_cache cache(*model, model->hparams, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, false, false, true,
            16, 3, 1, 0, LLAMA_SWA_TYPE_NONE, nullptr, [](uint32_t il) { return il == 1; }, nullptr, nullptr);
    const std::vector<uint32_t> slots = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 12 };
    const llama_pos pos_by_slot[] = { 30, 0, 8, 4, 5, 2, 9, 3, 1, 5, 7, 12, 6, -1, 15, 5 };
    std::vector<llama_token> tokens(slots.size());
    std::vector<llama_pos> positions(3*slots.size());
    std::vector<int32_t> n_seq_id(slots.size());
    std::vector<std::vector<llama_seq_id>> seq_data(slots.size());
    std::vector<llama_seq_id *> seq_id(slots.size());
    llama_seq_id unique[] = { 0, 1 };
    for (size_t j = 0; j < slots.size(); ++j) {
        const uint32_t i = slots[j];
        tokens[j] = i + 1;
        positions[j] = pos_by_slot[i];
        positions[j + slots.size()] = 200 + i;
        positions[j + 2*slots.size()] = 100 + i;
        seq_data[j] = i == 0 || i == 3 || i == 15 ? std::vector<llama_seq_id>{1} : std::vector<llama_seq_id>{0};
        if (i == 6 || i == 9) {
            seq_data[j].push_back(1);
        }
        n_seq_id[j] = seq_data[j].size();
        seq_id[j] = seq_data[j].data();
    }
    llama_ubatch seed = {};
    seed.n_tokens = seed.n_seq_tokens = slots.size();
    seed.n_seqs = 1;
    seed.n_seqs_unq = 2;
    seed.n_pos = 3;
    seed.token = tokens.data();
    seed.pos = positions.data();
    seed.n_seq_id = n_seq_id.data();
    seed.seq_id = seq_id.data();
    seed.seq_id_unq = unique;
    llama_kv_cache::slot_info placement;
    placement.s0 = placement.s1 = 0;
    placement.strm = {0};
    placement.idxs = {slots};

    llama_pos probe_pos = 40;
    int32_t probe_n_seq = 1;
    llama_seq_id probe_seq = 0;
    llama_seq_id * probe_seq_ptr = &probe_seq;
    llama_ubatch probe = {};
    probe.n_tokens = probe.n_seq_tokens = probe.n_seqs = probe.n_seqs_unq = probe.n_pos = 1;
    probe.pos = &probe_pos;
    probe.n_seq_id = &probe_n_seq;
    probe.seq_id = &probe_seq_ptr;
    probe.seq_id_unq = &probe_seq;

    auto compare = [&](const llama_kv_cells & expected, uint32_t head) {
        const auto & actual = cache.get_cells(0);
        require(actual.size() == expected.size() && actual.get_used() == expected.get_used() &&
                actual.used_min() == expected.used_min() && actual.used_max_p1() == expected.used_max_p1(), "KV used-cell metadata");
        for (uint32_t i = 0; i < expected.size(); ++i) {
            require(actual.pos_data()[i] == expected.pos_data()[i] &&
                    actual.seq_get_all(i) == expected.seq_get_all(i), "KV positions and sequence memberships");
            if (!expected.is_empty(i)) {
                const auto & a = actual.ext_get(i);
                const auto & b = expected.ext_get(i);
                require(a.x == b.x && a.y == b.y && a.tok == b.tok &&
                        actual.get_shift(i) == expected.get_shift(i), "KV token and spatial metadata");
            }
        }
        for (llama_seq_id s = 0; s < 3; ++s) {
            require(actual.seq_pos_get(s) == expected.seq_pos_get(s) &&
                    actual.seq_pos_min(s) == expected.seq_pos_min(s) &&
                    actual.seq_pos_max(s) == expected.seq_pos_max(s), "KV sequence position index");
        }
        // The expected head is empty; these cases keep it below the sparse-cache restart threshold.
        const auto next = cache.find_slot(probe, false);
        require(!next.empty() && next.head() == head, "KV physical head selects the same next allocation");
    };
    auto remove = [&](llama_kv_cells & expected, uint32_t & head, llama_seq_id s, llama_pos p0, llama_pos p1) {
        require(cache.seq_rm(s, p0, p1), "production KV removal succeeds");
        p0 = std::max<llama_pos>(p0, 0);
        p1 = p1 < 0 ? std::numeric_limits<llama_pos>::max() : p1;
        uint32_t freed = expected.size();
        for (uint32_t i = 0; i < expected.size(); ++i) {
            if (!expected.pos_in(i, p0, p1)) {
                continue;
            }
            bool empty = false;
            if (s < 0) {
                expected.rm(i);
                empty = true;
            } else if (expected.seq_has(i, s)) {
                empty = expected.seq_rm(i, s);
            }
            if (empty && freed == expected.size()) {
                freed = i;
            }
        }
        head = std::min(head, freed);
        compare(expected, head);
    };
    struct removal_case { const char * name; llama_seq_id seq; llama_pos p0, p1; };
    const removal_case cases[] = {
        { "absent suffix", 0, 31, -1 },
        { "empty range", 0, 5, 5 },
        { "reversed range", 0, 9, 7 },
        { "position order differs from slot order", 0, 7, 9 },
        { "duplicate positions and an alias", 0, 5, 6 },
        { "only aliased membership removed", 0, 9, 10 },
        { "upper endpoint excluded", 0, 6, 7 },
        { "freed slot beyond old head", 0, 15, 16 },
        { "negative lower endpoint", 0, -1, 1 },
        { "absent sequence", 2, 0, -1 },
        { "full specific sequence", 0, -1, -1 },
        { "second sequence and aliased cells", 1, 5, 6 },
        { "wildcard range", -1, 5, 6 },
        { "wildcard clear", -1, -1, -1 },
    };
    for (const auto & c : cases) {
        cache.clear(false);
        cache.apply_ubatch(placement, seed);
        llama_kv_cells expected = cache.get_cells(0);
        uint32_t head = 13;
        try {
            compare(expected, head);
            remove(expected, head, c.seq, c.p0, c.p1);
        } catch (const std::exception & error) {
            throw std::runtime_error(std::string(c.name) + ": " + error.what());
        }
    }
    cache.clear(false);
    cache.apply_ubatch(placement, seed);
    llama_kv_cells expected = cache.get_cells(0);
    uint32_t head = 13;
    remove(expected, head, 0, 7, 9);
    remove(expected, head, 0, 5, 6);
    remove(expected, head, 1, 5, 6);
    remove(expected, head, 0, 9, 10);
    remove(expected, head, 1, 9, 10);
    printf("PASS KV removal metadata: production seq_rm matches scan control for shuffled cells, duplicate positions, aliases, ranges and composed head updates\n");
}

static void run_scratch_empty_region(llama_model * model) {
    llama_kv_cache cache(*model, model->hparams, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, false, false, true,
            1, 1, 1, 0, LLAMA_SWA_TYPE_NONE, nullptr, [](uint32_t il) { return il == 1; }, nullptr, nullptr);
    const auto layers = cache.get_layer_ids();
    require(layers.size() == 1, "empty scratch fixture has one attention layer");
    const auto * key = cache.get_k_storage(layers[0]);
    const size_t alignment = ggml_backend_buffer_get_alignment(key->buffer);
    require(ggml_nbytes(key) % alignment != 0, "empty scratch region ends between backend alignment boundaries");
    bool canceled = false;
    auto chunks = cache.scratch_borrow(ggml_backend_buffer_get_type(key->buffer), alignment, 1, 0, 0,
            [](void * user) { *static_cast<bool *>(user) = true; }, &canceled);
    const bool empty = chunks.empty();
    chunks.clear();
    cache.scratch_release();
    require(empty && !canceled, "zero-size KV suffix cannot lend occupied bytes after alignment");
    printf("PASS scratch bounds: an empty unaligned KV suffix cannot be borrowed\n");
}

static void run_scratch_clear(llama_model * model) {
    prefill_env_restore env;
    env.set(false, false, false);
    const cache_config config { "scratch clear", GGML_TYPE_Q8_0, GGML_TYPE_Q8_0 };
    contexts reference(model, config, 128, COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE, 32);
    prefill(reference, 0, 128, 0);
    env.set(false, false, false, false, true);
    contexts borrowed(model, config, 128, COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE, 32);
    auto * hybrid = dynamic_cast<llama_memory_hybrid *>(llama_get_memory(borrowed.tgt));
    require(hybrid != nullptr, "scratch clear fixture has hybrid target memory");
    auto * cache = hybrid->get_mem_attn();
    std::vector<ggml_tensor *> storage_tensors;
    for (auto il : cache->get_layer_ids()) {
        storage_tensors.push_back(cache->get_k_storage(il));
        storage_tensors.push_back(cache->get_v_storage(il));
    }
    for (auto * storage : storage_tensors) {
        std::vector<uint8_t> bytes(ggml_nbytes(storage));
        ggml_backend_tensor_get(storage, bytes.data(), 0, bytes.size());
        require(std::all_of(bytes.begin(), bytes.end(), [](uint8_t value) { return value == 0; }), "plain KV starts zero");
    }
    const size_t loans = scratch_borrows;
    prefill(borrowed, 0, 128, 0);
    require(scratch_borrows > loans, "scratch clear fixture acquired a KV loan");
    llama_synchronize(borrowed.dft);
    struct poisoned_range { ggml_tensor * tensor; size_t offset, size; };
    std::vector<poisoned_range> ranges;
    for (auto * storage : storage_tensors) {
        const size_t alignment = ggml_backend_buffer_get_alignment(storage->buffer);
        require(alignment && alignment <= 256, "scratch clear fixture uses bounded CPU alignment");
        std::vector<uint8_t> bytes(ggml_nbytes(storage));
        ggml_backend_tensor_get(storage, bytes.data(), 0, bytes.size());
        size_t count = 0;
        for (size_t offset = 0; offset + alignment <= bytes.size() && count < 4; offset += alignment) {
            if (offset % storage->nb[2] / storage->nb[1] >= 640 &&
                    std::any_of(bytes.begin() + offset, bytes.begin() + offset + alignment, [](uint8_t value) { return value != 0; })) {
                // Only the loan wrote these unused rows; a whole alignment block stays inside its borrowed view.
                ranges.push_back({ storage, offset, alignment });
                ++count;
            }
        }
    }
    require(!ranges.empty(), "MTP scratch wrote into unwritten target KV rows");
    std::vector<uint8_t> poison(256, 0xff);
    for (const auto & range : ranges) {
        ggml_backend_tensor_set(range.tensor, poison.data(), range.offset, range.size);
    }
    const size_t releases = scratch_releases;
    common_params_sampling sampling;
    sampling.temp = 0.0f;
    llama_tokens expected, actual, prompt;
    auto draft = [&](contexts & ctx, llama_tokens & tokens) {
        common_speculative_begin(ctx.spec, 0, llama_tokens(128, 0));
        auto & dp = common_speculative_get_draft_params(ctx.spec, 0);
        dp = {};
        dp.drafting = true;
        dp.pos0 = 128;
        dp.id_last = token_at(128);
        dp.prompt = &prompt;
        dp.result = &tokens;
        dp.sampling = &sampling;
        common_speculative_draft(ctx.spec);
    };
    draft(reference, expected);
    draft(borrowed, actual);
    require(!actual.empty() && actual == expected, "draft tokens survive poisoned scratch release");
    require(scratch_releases > releases, "generation released poisoned KV scratch");
    for (const auto & range : ranges) {
        std::vector<uint8_t> bytes(range.size);
        ggml_backend_tensor_get(range.tensor, bytes.data(), range.offset, range.size);
        require(std::all_of(bytes.begin(), bytes.end(), [](uint8_t value) { return value == 0; }),
                "released scratch bytes are zero before target attention can read padded KV");
    }
    require(save(reference.tgt, 0) == save(borrowed.tgt, 0), "scratch clear preserves written target KV");
    printf("PASS scratch clear: %zu NaN-poisoned borrowed KV ranges are zero after generation releases the loan; tokens and written KV match legacy\n", ranges.size());
}

static void run_prefill_async_noalloc(llama_model * model) {
    struct env_restore {
        std::string async = common_get_env("GGML_MTP_PREFILL_ASYNC");
        std::string kv_only = common_get_env("LLAMA_MTP_UPDATE_KV_ONLY");
        std::string tight = common_get_env("LLAMA_MTP_RESERVE_TIGHT");
        std::string scratch = common_get_env("LLAMA_MTP_PREFILL_SCRATCH");
        ~env_restore() {
            common_set_env("GGML_MTP_PREFILL_ASYNC", async);
            common_set_env("LLAMA_MTP_UPDATE_KV_ONLY", kv_only);
            common_set_env("LLAMA_MTP_RESERVE_TIGHT", tight);
            common_set_env("LLAMA_MTP_PREFILL_SCRATCH", scratch);
        }
    } env;
    common_set_env("GGML_MTP_PREFILL_ASYNC", "1");
    common_set_env("LLAMA_MTP_UPDATE_KV_ONLY", "1");
    common_set_env("LLAMA_MTP_RESERVE_TIGHT", "0");
    common_set_env("LLAMA_MTP_PREFILL_SCRATCH", "0");
    llama_model_ptr measured_model(make_model(true));
    const cache_config config { "async no_alloc", GGML_TYPE_Q8_0, GGML_TYPE_Q8_0 };
    contexts measured(measured_model.get(), config, 128, COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE, 32);
    contexts allocated(model, config, 128, COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE, 32);
    auto compute_bytes = [](const llama_context * ctx) {
        size_t result = 0;
        for (const auto & entry : llama_get_memory_breakdown(ctx)) {
            result += entry.second.compute;
        }
        return result;
    };
    const size_t expected = compute_bytes(measured.dft);
    const size_t actual = compute_bytes(allocated.dft);
    printf("ASYNC no_alloc draft compute: measured %zu bytes, allocated %zu bytes\n", expected, actual);
    require(actual > 0 && expected >= actual, "async no_alloc measurement covers catch-up and generation compute buffers");
    printf("PASS ASYNC: no_alloc compute measurement covers the allocated draft scheduler\n");
}

static void run_eagle_boundary_restore(llama_model * model) {
    // Only exercise EAGLE's boundary serializer with the hybrid fixture; no EAGLE graph is decoded.
    model->target_layer_ids = { 0, 1, 2 };
    contexts ctx(model, { "EAGLE boundary", GGML_TYPE_Q8_0, GGML_TYPE_Q8_0 }, 128, COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3);
    std::vector<uint8_t> got;
    require(!common_speculative_get_state(ctx.spec, 1, got), "fresh EAGLE sequence has no boundary state");
    const llama_pos pos = 123;
    std::vector<float> carry(n_embd);
    for (uint32_t i = 0; i < n_embd; ++i) {
        carry[i] = (float) i / n_embd;
    }
    std::vector<uint8_t> state(sizeof(pos) + carry.size() * sizeof(float));
    std::memcpy(state.data(), &pos, sizeof(pos));
    std::memcpy(state.data() + sizeof(pos), carry.data(), carry.size() * sizeof(float));
    require(common_speculative_restore_state(ctx.spec, 1, state), "restore EAGLE boundary into a fresh driver");
    require(common_speculative_get_state(ctx.spec, 1, got) && got == state, "EAGLE boundary restore round trip");
    require(!common_speculative_get_state(ctx.spec, 0, got), "EAGLE boundary restore leaves another sequence cold");
    auto truncated = state;
    truncated.pop_back();
    require(!common_speculative_restore_state(ctx.spec, 1, truncated), "truncated EAGLE boundary is rejected");
    require(!common_speculative_restore_state(ctx.spec, 1, {}), "missing EAGLE boundary cannot replace a saved boundary");
    require(common_speculative_get_state(ctx.spec, 1, got) && got == state, "invalid EAGLE boundary leaves state unchanged");
    printf("PASS EAGLE: boundary state restores into a fresh driver; invalid snapshots preserve it\n");
}

static void model_ampere_reserve(ggml_cgraph * graph, const char * phase) {
    // Replay CPU placement with the Ampere attention allocation size; this does not measure a CUDA scheduler.
    struct model_state { bool supported = true; } state;
    ggml_backend_buffer_type buft = *ggml_backend_cpu_buffer_type();
    buft.context = &state;
    buft.iface.get_name = [](ggml_backend_buffer_type_t) { return "MODELED_Ampere_single_buffer"; };
    buft.iface.get_alignment = [](ggml_backend_buffer_type_t) { return size_t(128); };
    buft.iface.get_alloc_size_n = nullptr;
    buft.iface.get_alloc_size = [](ggml_backend_buffer_type_t bt, const ggml_tensor * tensor) {
        size_t size = ggml_nbytes(tensor);
        if (tensor->op != GGML_OP_FLASH_ATTN_EXT) {
            return size;
        }
        const auto * q = tensor->src[0];
        const auto * k = tensor->src[1];
        if (q->ne[0] != 256 || q->ne[1] <= 8 || q->ne[3] != 1 || k->ne[2] <= 0 || q->ne[2] % k->ne[2] != 0) {
            static_cast<model_state *>(bt->context)->supported = false;
            return size;
        }
        const bool sj = tensor->src[6] != nullptr;
        const size_t n_kv = sj ? ggml_get_op_params_i32(tensor, 6) : k->ne[1];
        const size_t budget = size_t(256) << 20;
        const size_t floor = size_t(2) * n_kv * 256 * sizeof(ggml_fp16_t) +
            size_t(q->ne[1]) * (q->ne[2] / k->ne[2]) * 256 * sizeof(float);
        if (!sj || floor <= budget) {
            size = ((size + 127) / 128) * 128 + std::max(budget, floor);
        }
        return size;
    };
    printf("RESERVE_PROBE MODELED_Ampere begin %s: single buffer, alignment 128, FA budget 256 MiB, CPU graph placement retained\n", phase);
    fflush(stdout);
    ggml_gallocr_t alloc = ggml_gallocr_new(&buft);
    size_t size = 0;
    ggml_gallocr_reserve_n_size(alloc, graph, nullptr, nullptr, &size);
    ggml_gallocr_free(alloc);
    printf("RESERVE_PROBE MODELED_Ampere %s compute = %.6f MiB (%zu bytes), FA shapes %s\n",
            phase, size / 1048576.0, size, state.supported ? "modeled" : "include unsupported cases");
}

static void run_reserve_probe(uint32_t context_size, uint32_t parallel, const char * model_path) {
    require(context_size >= 1024 && (parallel == 1 || parallel == 4), "reserve probe needs context >= 1024 and parallel 1 or 4");
    const model_geometry production { 248320, 5120, 65, 17408, 24, 4, 256, 4, 6144, 48, 16, 64 };
    llama_model_params mp = llama_model_default_params();
    ggml_backend_dev_t devices[] = { nullptr };
    mp.devices = devices;
    mp.no_alloc = true;
    mp.load_mode = LLAMA_LOAD_MODE_NONE;
    mp.load_mtp = true;
    llama_model_ptr model(model_path ? llama_model_load_from_file(model_path, mp) : make_model(true, production));
    require(model != nullptr, "load reserve probe metadata");
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = context_size;
    cp.n_batch = 4096;
    cp.n_ubatch = 1024;
    cp.n_seq_max = parallel;
    cp.n_rs_seq = 4;
    cp.kv_unified = true;
    cp.n_threads = cp.n_threads_batch = 2;
    cp.offload_kqv = model->n_devices() > 0;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k = cp.type_v = GGML_TYPE_TQ6_0;
    cp.sj_kvarn_bits_k = cp.sj_kvarn_bits_v = 4;
    cp.sj_kvarn_body_type = GGML_TYPE_F32;
    cp.sj_kvarn_staging_type = GGML_TYPE_TQ6_0;
    cp.sj_kvarn_sink_type = GGML_TYPE_F16;
    cp.sj_kvarn_sink = 128;
    cp.sj_kvarn_tail = 4096;
    cp.sj_kvarn_tail_max = 8192;
    const auto outputs = common_speculative_get_output_limits(cp.n_batch, parallel, 4);
    cp.n_outputs_max = outputs.total;
    cp.n_outputs_max_per_seq = outputs.per_seq;
    printf("RESERVE_PROBE CPU no_alloc: context=%u parallel=%u batch=%u ubatch=%u target_outputs=%u tight=%s\n",
            context_size, parallel, cp.n_batch, cp.n_ubatch, cp.n_outputs_max, common_get_env("LLAMA_MTP_RESERVE_TIGHT").c_str());
    fflush(stdout);
    llama_context_ptr target(llama_init_from_model(model.get(), cp));
    require(target != nullptr, "create production geometry target");
    llama_set_embeddings_nextn(target.get(), true, false);
    target->sched_reserve();
    {
        auto mctx = target->get_memory()->init_full();
        auto * graph = target->graph_reserve(1024, parallel, outputs.total, mctx.get(), true);
        require(graph != nullptr, "reserve probe target graph exists");
        model_ampere_reserve(graph, "target-prefill");
    }

    cp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    cp.ctx_other = target.get();
    cp.n_rs_seq = 0;
    cp.sj_kvarn_bits_k = cp.sj_kvarn_bits_v = 0;
    cp.type_k = GGML_TYPE_TQ5_0;
    cp.type_v = GGML_TYPE_TURBO4_0;
    cp.n_outputs_max = parallel;
    cp.n_outputs_max_per_seq = 1;
    cp.draft_vocab_map = model_path ? "auto" : "docs/mtp-vocab/atx_65536.txt";
    llama_context_ptr draft(llama_init_from_model(model.get(), cp));
    require(draft != nullptr, "create production geometry draft");
    llama_set_embeddings_nextn(draft.get(), true, true);
    draft->sched_reserve();

    auto print_memory = [](const char * name, const llama_context * ctx) {
        size_t compute = 0;
        for (const auto & entry : llama_get_memory_breakdown(ctx)) {
            compute += entry.second.compute;
        }
        printf("RESERVE_PROBE CPU no_alloc %s compute = %.6f MiB (%zu bytes)\n", name, compute / 1048576.0, compute);
    };
    print_memory("target", target.get());
    print_memory("draft", draft.get());

    auto print_graph = [&](uint32_t rows, uint32_t n_outputs, const char * phase) {
        auto mctx = draft->get_memory()->init_full();
        auto * graph = draft->graph_reserve(rows, parallel, n_outputs, mctx.get(), true);
        require(graph != nullptr, "reserve probe graph exists");
        printf("RESERVE_PROBE graph %s rows=%u outputs=%u nodes=%d leaves=%d\n", phase, rows, n_outputs,
                graph->n_nodes, graph->n_leafs);
        auto print_tensor = [&](const ggml_tensor * tensor, int index) {
            printf("TENSOR %s %d %s %s %s %lld %lld %lld %lld %zu view=%s\n", phase, index, tensor->name,
                    ggml_op_name(tensor->op), ggml_type_name(tensor->type), (long long) tensor->ne[0],
                    (long long) tensor->ne[1], (long long) tensor->ne[2], (long long) tensor->ne[3],
                    ggml_nbytes(tensor), tensor->view_src ? tensor->view_src->name : "none");
        };
        for (int i = 0; i < graph->n_nodes; ++i) {
            print_tensor(graph->nodes[i], i);
        }
        for (int i = 0; i < graph->n_leafs; ++i) {
            const auto * tensor = graph->leafs[i];
            if (!tensor->buffer || ggml_backend_buffer_get_usage(tensor->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                print_tensor(tensor, -i - 1);
            }
        }
        model_ampere_reserve(graph, phase);
    };
    print_graph(1024, parallel, "legacy-prefill");
    if (common_get_env("LLAMA_MTP_RESERVE_TIGHT") == "1") {
        print_graph(1024, 0, "tight-prefill");
        print_graph(64 + parallel, parallel, "tight-generation");
    }
    printf("PASS RESERVE_PROBE: real CPU no_alloc model graph and allocator measurements; no CUDA execution\n");
}

static void run_failed_helper(llama_model * model) {
    prefill_env_restore env;
    common_set_env("LLAMA_SHARED_COMPUTE", "1");
    llama_model_ptr helper_model(make_model());
    auto cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_batch = cp.n_ubatch = 32;
    cp.n_threads = cp.n_threads_batch = 2;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context_ptr target(llama_init_from_model(model, cp));
    require(target != nullptr, "create failed-helper target");
    cp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    cp.ctx_other = target.get();
    // The CPU-only fixture rejects a device cache after selecting its compute donor.
    cp.moe_cache_size = 1;
    const size_t before = compute_shares;
    require(llama_init_from_model(helper_model.get(), cp) == nullptr, "helper initialization fails after donor selection");
    require(compute_shares == before + 1, "failed helper reached donor selection");
    cp.moe_cache_size = 0;
    llama_context_ptr helper(llama_init_from_model(helper_model.get(), cp));
    require(helper != nullptr, "retry helper initialization");
    require(compute_shares == before + 2, "target accepts shared helper after initialization failure");
    target->sched_reserve();
    helper.reset();
    target->sched_reserve();
    printf("PASS failed helper initialization, shared retry, target reserve and teardown\n");
}

// The second GGUF must reuse identical storage and retain it after the donor model is freed.
static void run_cuda_readonly_pin() {
#ifndef _WIN32
    constexpr size_t bytes = 32 * 1024 * 1024;
    auto reg = ggml_backend_reg_by_name("CUDA");
    require(reg != nullptr, "CUDA registration backend exists");
    auto pin = (bool (*)(void *, size_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_host_pin");
    auto unpin = (void (*)(void *)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_host_unpin");
    require(pin && unpin, "read-only host registration entry points exist");
    FILE * file = tmpfile();
    require(file != nullptr, "create large read-only mapping fixture");
    require(ftruncate(fileno(file), bytes) == 0, "size large read-only mapping fixture");
    llama_file source(file);
    {
        llama_mmap mapped(&source, 0);
        mapped.remap_private_writable(&source, 0, bytes);
        auto * data = (unsigned char *) mapped.addr();
        data[0] = 123;
        const bool pinned = pin(data, bytes);
        if (pinned) { unpin(data); }
        require(pinned, "register 32 MiB private file mapping without a second tensor allocation");
        require(data[0] == 123, "registration preserves private tensor contents");
        rewind(file);
        require(fgetc(file) == 0, "private mapped writes do not change model file");
    }
    fclose(file);
    printf("PASS CUDA registration of 32 MiB private dense mapping; model file unchanged\n");
#endif
}

static void run_weight_borrow(llama_model * source, bool cuda = false, bool pin_host = false) {
    const auto close_file = [](FILE * f) { fclose(f); };
    std::unique_ptr<FILE, decltype(close_file)> file(tmpfile(), close_file);
    require(file != nullptr, "create borrowed-weight fixture file");
    llama_model_saver saver(source);
    saver.add_kv_from_model();
    saver.add_tensors_from_model();
    const bool recurrent[] = { true, false, false };
    gguf_set_arr_data(saver.gguf_ctx, saver.llm_kv(LLM_KV_ATTENTION_RECURRENT_LAYERS).c_str(), GGUF_TYPE_BOOL, recurrent, 3);
    saver.save(file.get());
    fflush(file.get());
    llama_model_params mp = llama_model_default_params();
    ggml_backend_dev_t devices[] = { nullptr };
    mp.devices = cuda ? nullptr : devices;
    mp.n_gpu_layers = cuda ? 999 : 0;
    mp.load_mode = LLAMA_LOAD_MODE_MMAP;
    mp.load_mtp = true;
    rewind(file.get());
    llama_model_ptr donor(cuda ? llama_model_load_from_file_ptr(file.get(), mp) : make_model());
    require(donor != nullptr, "load donor fixture");
    mp.tensor_borrow_from = donor.get();
#ifndef _WIN32
    if (pin_host && !cuda) {
        llama_mlock overflow;
        overflow.init(donor->tok_embd->data);
        overflow.grow_to(SIZE_MAX);
        require(overflow.locked_size() == 0, "overflowing lock size is refused");
        struct rlimit original;
        require(getrlimit(RLIMIT_MEMLOCK, &original) == 0, "read lock limit");
        auto denied = original;
        denied.rlim_cur = 0;
        require(setrlimit(RLIMIT_MEMLOCK, &denied) == 0, "deny host locking");
        rewind(file.get());
        llama_model_ptr failed(llama_model_load_from_file_ptr(file.get(), mp));
        const int restored = setrlimit(RLIMIT_MEMLOCK, &original);
        require(restored == 0, "restore host lock limit");
        require(!failed && borrowed_host_pin_failures > 0, "failed pin refuses helper loading");
    }
#endif
    const size_t pins_before = borrowed_host_pins;
    rewind(file.get());
    llama_model_ptr borrower(llama_model_load_from_file_ptr(file.get(), mp));
    require(borrower != nullptr, "load borrower fixture");
    require(borrower->output == donor->output && borrower->tok_embd == donor->tok_embd, "identical weights share tensor and storage");
    if (pin_host) {
        require(borrowed_host_pins > pins_before, "host storage was pinned during helper load");
        const size_t pins_after = borrowed_host_pins;
        rewind(file.get());
        llama_model_ptr repeated(llama_model_load_from_file_ptr(file.get(), mp));
        require(repeated && borrowed_host_pins == pins_after, "repeated borrowing reuses donor lock");
    }
    {
        prefill_env_restore env;
        common_set_env("LLAMA_SHARED_COMPUTE", "1");
        const cache_config config { "separate-file q8", GGML_TYPE_Q8_0, GGML_TYPE_Q8_0 };
        const size_t before = compute_shares;
        contexts reference(donor.get(), config);
        contexts separate(donor.get(), config, 128, COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE, 0, 2, 4096, borrower.get());
        require(compute_shares >= before + 2, "same-device contexts activate serialized compute sharing");
        prefill(reference, 0, 128, 0);
        prefill(separate, 0, 128, 0);
        separate.tgt->set_embeddings_nextn(false, false);
        separate.tgt->set_embeddings_nextn(true, false);
        separate.tgt->sched_reserve();
        prefill(reference, 128, 256, 0);
        prefill(separate, 128, 256, 0);
        const auto a = save(reference, 0), b = save(separate, 0);
        require(a.tgt == b.tgt && a.dft == b.dft && a.spec == b.spec, "separate-file state survives donor allocator reset");
    }
    float expected = 0;
    ggml_backend_tensor_get(donor->output, &expected, 0, sizeof(expected));
    const float changed = expected + 1.0f;
    ggml_backend_tensor_set(donor->output, &changed, 0, sizeof(changed));
    rewind(file.get());
    llama_model_ptr different(llama_model_load_from_file_ptr(file.get(), mp));
    require(different != nullptr && different->output != donor->output, "different bytes refuse weight sharing");
    require(different->tok_embd == donor->tok_embd, "other identical weight can still share");
    float actual = 0;
    ggml_backend_tensor_get(different->output, &actual, 0, sizeof(actual));
    require(actual == expected, "fallback loads helper bytes");
    donor.reset();
    ggml_backend_tensor_get(borrower->output, &actual, 0, sizeof(actual));
    require(actual == changed, "borrowed weight survives donor model destruction");
    printf("PASS %s separate-file weight borrowing, unequal-byte fallback, allocator replacement, donor lifetime\n", cuda ? "CUDA" : "CPU");
}

int main(int argc, char ** argv) {
    const bool reserve_probe = (argc == 4 || argc == 5) && std::strcmp(argv[1], "--reserve-probe") == 0;
    common_log_set_verbosity_thold(LOG_LEVEL_ERROR);
    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        if (std::strstr(text, "draft vocabulary shortlist: 16 of 32 tokens ") != nullptr) { ++half_vocab_maps; }
        if (std::strstr(text, "compact draft head: type f32, n_sel 16,") != nullptr) { ++half_vocab_compact_heads; }
        if (std::strstr(text, "MTP pool store-only inputs validated (") != nullptr) { ++pool_store_fills; }
        if (std::strstr(text, "pinned borrowed host ") != nullptr) { ++borrowed_host_pins; }
        if (std::strstr(text, "cannot pin borrowed host tensor ") != nullptr) { ++borrowed_host_pin_failures; }
        if (std::strstr(text, "MTP context will share the target context's compute buffers") != nullptr) {
            ++compute_shares;
        }
        if (std::strstr(text, "MTP prefill compute =") != nullptr) {
            ++scratch_borrows;
        }
        if (std::strstr(text, "released MTP KV scratch;") != nullptr) {
            ++scratch_releases;
        }
        if (std::strstr(text, "retrying with serialized compute") != nullptr) {
            ++scratch_fallbacks;
        }
        if (level >= (user_data ? GGML_LOG_LEVEL_INFO : GGML_LOG_LEVEL_ERROR)) {
            fputs(text, stderr);
        }
    }, reserve_probe ? &argc : nullptr);
    const bool pin_host = argc == 2 && (std::strcmp(argv[1], "--host-pin-only") == 0 || std::strcmp(argv[1], "--host-pin-cuda-only") == 0);
    const bool cuda_sharing = argc == 2 && (std::strcmp(argv[1], "--cuda-sharing-only") == 0 || std::strcmp(argv[1], "--host-pin-cuda-only") == 0);
    const bool feedback_cuda = argc == 2 && std::strcmp(argv[1], "--device-feedback-cuda-only") == 0;
    const bool chain_cuda = argc == 2 && std::strcmp(argv[1], "--qwen4exp-chain-cuda-only") == 0;
    const bool compact_cuda = argc == 2 && std::strcmp(argv[1], "--qwen4exp-compact-cuda-only") == 0;
    const bool full_head_cuda = argc == 2 && std::strcmp(argv[1], "--qwen4exp-full-head-cuda-only") == 0;
    const bool pool_store_cuda = argc == 2 && std::strcmp(argv[1], "--qwen4exp-pool-store-cuda-only") == 0;
    if (pin_host) { common_set_env("LLAMA_MTP_PIN_BORROWED_HOST", "1"); }
    const bool pool_cuda = argc == 2 && std::strcmp(argv[1], "--qwen4exp-pool-cuda-only") == 0;
    const bool defer_math_cuda = argc == 3 && std::strcmp(argv[1], "--bookkeep-defer-math-cuda") == 0;
    if (cuda_sharing || feedback_cuda || chain_cuda || compact_cuda || full_head_cuda || pool_store_cuda || pool_cuda ||
            defer_math_cuda) { ggml_backend_load_all(); }
    llama_backend_init();
    int failed = 0;
    llama_model * model = nullptr;
    try {
        if (reserve_probe) {
            run_reserve_probe(std::stoul(argv[2]), std::stoul(argv[3]), argc == 5 ? argv[4] : nullptr);
            llama_backend_free();
            return 0;
        }
        if (feedback_cuda || (argc == 2 && std::strcmp(argv[1], "--device-feedback-only") == 0)) {
            run_device_feedback(feedback_cuda);
            llama_backend_free();
            return 0;
        }
        const bool pool_store = argc == 2 && std::strcmp(argv[1], "--qwen4exp-pool-store-only") == 0;
        if (pool_store || pool_store_cuda || pool_cuda || (argc == 2 && std::strcmp(argv[1], "--qwen4exp-pool-only") == 0)) {
            run_qwen4exp_pool(pool_store_cuda || pool_cuda, pool_store || pool_store_cuda);
            llama_backend_free();
            return 0;
        }
        if (defer_math_cuda || (argc == 2 && std::strcmp(argv[1], "--bookkeep-defer-math-only") == 0)) {
            run_qwen4exp_defer_math(defer_math_cuda, defer_math_cuda ? argv[2] : nullptr);
            llama_backend_free();
            return 0;
        }
        if (chain_cuda || (argc == 2 && std::strcmp(argv[1], "--qwen4exp-chain-only") == 0)) {
            run_qwen4exp_chain(chain_cuda);
            llama_backend_free();
            return 0;
        }
        if (compact_cuda || (argc == 2 && std::strcmp(argv[1], "--qwen4exp-compact-only") == 0)) {
            run_qwen4exp_chain(compact_cuda, true);
            llama_backend_free();
            return 0;
        }
        if (full_head_cuda || (argc == 2 && std::strcmp(argv[1], "--qwen4exp-full-head-only") == 0)) {
            run_qwen4exp_full_head(full_head_cuda);
            llama_backend_free();
            return 0;
        }
        model = make_model();
        if (argc == 3 && std::strcmp(argv[1], "--save-model") == 0) {
            llama_model_saver saver(model);
            saver.add_kv_from_model();
            saver.add_tensors_from_model();
            // Include the MTP block in the fixture's recurrent-layer metadata.
            const bool recurrent[] = { true, false, false };
            gguf_set_arr_data(saver.gguf_ctx, saver.llm_kv(LLM_KV_ATTENTION_RECURRENT_LAYERS).c_str(), GGUF_TYPE_BOOL, recurrent, 3);
            saver.save(argv[2]);
            llama_model_free(model);
            llama_backend_free();
            return 0;
        }
        if (argc == 2 && std::strcmp(argv[1], "--bookkeep-sampler-only") == 0) {
            run_common_sampler(model);
            llama_model_free(model);
            llama_backend_free();
            return 0;
        }
        if (argc == 2 && std::strcmp(argv[1], "--bookkeep-output-only") == 0) {
            run_sampler_output(model);
            llama_model_free(model);
            llama_backend_free();
            return 0;
        }
        if (argc == 2 && std::strcmp(argv[1], "--bookkeep-metadata-only") == 0) {
            run_kv_removal_metadata(model);
            llama_model_free(model);
            llama_backend_free();
            return 0;
        }
        if (argc == 2 && std::strcmp(argv[1], "--bookkeep-defer-only") == 0) {
            require(std::getenv("LLAMA_SPEC_DEFER_CATCHUP") && std::atoi(std::getenv("LLAMA_SPEC_DEFER_CATCHUP")) != 0,
                    "sequential deferred test requires LLAMA_SPEC_DEFER_CATCHUP=1");
            bookkeep_sequential = true;
            const cache_config config = { "SJ-KVaRN 4/4 sequential", GGML_TYPE_F16, GGML_TYPE_F16, 4, 4 };
            run_case(model, config, true);
            run_deferred_overflow(model, config);
            llama_model_free(model);
            llama_backend_free();
            return 0;
        }
        if (argc == 2 && std::strcmp(argv[1], "--failed-helper-only") == 0) {
            run_failed_helper(model);
            llama_model_free(model);
            llama_backend_free();
            return 0;
        }
        if (cuda_sharing) { require(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) != nullptr, "CUDA backend is available"); }
        if (cuda_sharing && pin_host) { run_cuda_readonly_pin(); }
        run_weight_borrow(model, cuda_sharing, pin_host);
        if (pin_host || cuda_sharing || (argc == 2 && std::strcmp(argv[1], "--weight-borrow-only") == 0)) {
            llama_model_free(model);
            llama_backend_free();
            return 0;
        }
        run_failed_helper(model);
        const cache_config configs[] = {
            { "q8_0/q8_0", GGML_TYPE_Q8_0, GGML_TYPE_Q8_0 },
            { "tq5_0/turbo4", GGML_TYPE_TQ5_0, GGML_TYPE_TURBO4_0 },
            { "SJ-KVaRN 4/4", GGML_TYPE_F16, GGML_TYPE_F16, 4, 4 },
            { "SJ-KVaRN 3/3t", GGML_TYPE_F16, GGML_TYPE_F16, 3, 3, GGML_TYPE_I16 },
            { "SJ-KVaRN 3/2t", GGML_TYPE_F16, GGML_TYPE_F16, 3, 2, GGML_TYPE_I16 },
        };
        for (const auto & config : configs) {
            try {
                run_prefill_identity(model, config);
            } catch (const std::exception & error) {
                fprintf(stderr, "FAIL %s (prefill identity): %s\n", config.name, error.what());
                ++failed;
            }
            for (bool deferred : { false, true }) {
                try {
                    run_case(model, config, deferred);
                } catch (const std::exception & error) {
                    fprintf(stderr, "FAIL %s (%s): %s\n", config.name, deferred ? "deferred" : "batch", error.what());
                    ++failed;
                }
            }
            try {
                run_deferred_overflow(model, config);
            } catch (const std::exception & error) {
                fprintf(stderr, "FAIL %s (deferred overflow): %s\n", config.name, error.what());
                ++failed;
            }
        }
        for (size_t index : { size_t(0), size_t(2) }) {
            try {
                run_prefill_identity(model, configs[index], 4);
                run_prefill_identity(model, configs[index], 2, false);
            } catch (const std::exception & error) {
                fprintf(stderr, "FAIL %s (four-slot prefill identity): %s\n", configs[index].name, error.what());
                ++failed;
            }
        }
        try {
            run_draft_vocab_reserve(model);
        } catch (const std::exception & error) {
            fprintf(stderr, "FAIL draft shortlist reserve: %s\n", error.what());
            ++failed;
        }
        try {
            run_scratch_target_lifetime(model);
        } catch (const std::exception & error) {
            fprintf(stderr, "FAIL scratch target lifetime: %s\n", error.what());
            ++failed;
        }
        try {
            run_kv_removal_metadata(model);
        } catch (const std::exception & error) {
            fprintf(stderr, "FAIL KV removal metadata: %s\n", error.what());
            ++failed;
        }
        try {
            run_scratch_empty_region(model);
        } catch (const std::exception & error) {
            fprintf(stderr, "FAIL scratch empty region: %s\n", error.what());
            ++failed;
        }
        try {
            run_scratch_clear(model);
        } catch (const std::exception & error) {
            fprintf(stderr, "FAIL scratch clear: %s\n", error.what());
            ++failed;
        }
        try {
            run_prefill_async_noalloc(model);
        } catch (const std::exception & error) {
            fprintf(stderr, "FAIL ASYNC no_alloc: %s\n", error.what());
            ++failed;
        }
        try {
            run_eagle_boundary_restore(model);
        } catch (const std::exception & error) {
            fprintf(stderr, "FAIL EAGLE boundary restore: %s\n", error.what());
            ++failed;
        }
    } catch (const std::exception & error) {
        fprintf(stderr, "FAIL fixture: %s\n", error.what());
        ++failed;
    }
    llama_model_free(model);
    llama_backend_free();
    return failed ? 1 : 0;
}
