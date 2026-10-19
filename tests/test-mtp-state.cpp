// Exercise real MTP cache restoration with a generated CPU model, split prefill, deferred rows, and adaptive draft widths.
// Restore into a fresh driver and a different destination sequence.
// --save-model PATH exports the fixture for other state tests.
// --reserve-probe CTX PARALLEL [MODEL] measures production graph geometry without allocating weights or running kernels.

#include "common.h"
#include "log.h"
#include "speculative.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "llama.h"

#include "../ggml/src/ggml-impl.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "../src/llama-arch.h"
#include "../src/llama-context.h"
#include "../src/llama-ext.h"
#include "../src/llama-graph.h"
#include "../src/llama-kv-cache.h"
#include "../src/llama-memory.h"
#include "../src/llama-memory-hybrid.h"
#include "../src/llama-model.h"
#include "../src/llama-mmap.h"
#include "../src/llama-model-saver.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
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

static void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
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
};

static llama_model * make_model(bool no_alloc = false, const model_geometry & g = {}) {
    gguf_context * meta = gguf_init_empty();
    llama_model_saver ms(LLM_ARCH_QWEN35, meta);
    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE, llm_arch_name(LLM_ARCH_QWEN35));
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
    ms.add_kv(LLM_KV_TOKENIZER_MODEL, "test");
    std::vector<std::string> tokens;
    for (uint32_t i = 0; i < g.vocab; ++i) {
        tokens.push_back("tok_" + std::to_string(i));
    }
    ms.add_kv(LLM_KV_TOKENIZER_LIST, tokens);
    ms.add_kv(LLM_KV_TOKENIZER_SCORES, std::vector<float>(g.vocab, 0.0f));
    llama_model_params mp = llama_model_default_params();
    ggml_backend_dev_t devices[] = { nullptr };
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
            common_speculative_type type = COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE, uint32_t ubatch_size = 0, uint32_t parallel = 2, uint32_t context_per_seq = 4096, llama_model * draft_model = nullptr) {
        llama_context_params cp = llama_context_default_params();
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
        params.draft.n_max = 4;
        params.draft.n_min_adaptive = 3;
        params.draft.p_min = 0.0f;
        params.draft.chain = true;
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

static schedule generate(contexts & ctx, llama_seq_id seq, llama_pos & pos, int first, int count) {
    schedule result;
    common_params_sampling sampling;
    sampling.temp = 0.0f;
    for (int round = first; round < first + count; ++round) {
        llama_tokens prompt, draft;
        auto & dp = common_speculative_get_draft_params(ctx.spec, seq);
        dp = {};
        dp.drafting = true;
        dp.pos0 = pos;
        dp.id_last = token_at(pos);
        dp.prompt = &prompt;
        dp.result = &draft;
        dp.sampling = &sampling;
        common_speculative_draft(ctx.spec);
        require(!draft.empty(), "MTP drafted tokens");
        result.widths.push_back(draft.size());
        result.tokens.push_back(draft);
        llama_tokens verify { dp.id_last };
        verify.insert(verify.end(), draft.begin(), draft.end());
        process(ctx, verify, pos, seq);
        const int accepted = round % 7 == 6 ? 1 : (int) draft.size();
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

    ~prefill_env_restore() {
        common_set_env("GGML_MTP_PREFILL_OVERLAP", overlap);
        common_set_env("GGML_MTP_PREFILL_ASYNC", async);
        common_set_env("GGML_MTP_PREFILL_DEVICE", device);
        common_set_env("LLAMA_MTP_RESERVE_TIGHT", tight);
        common_set_env("LLAMA_MTP_PREFILL_SCRATCH", scratch);
        common_set_env("LLAMA_SHARED_COMPUTE", shared);
        common_set_env("LLAMA_MTP_UPDATE_KV_ONLY", kv_only);
    }

    void set(bool use_overlap, bool use_async, bool use_device, bool use_tight = false, bool use_scratch = false) {
        common_set_env("GGML_MTP_PREFILL_OVERLAP", use_overlap ? "1" : "0");
        common_set_env("GGML_MTP_PREFILL_ASYNC", use_async ? "1" : "0");
        common_set_env("GGML_MTP_PREFILL_DEVICE", use_device ? "1" : "0");
        common_set_env("LLAMA_MTP_RESERVE_TIGHT", use_tight ? "1" : "0");
        common_set_env("LLAMA_MTP_PREFILL_SCRATCH", use_scratch ? "1" : "0");
    }
};

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
    std::vector<float> hidden(n_embd, 0.0f);
    common_batch input(ctx.dft);
    for (llama_pos pos = 128; pos < 160; ++pos) {
        const int idx = input.add(token_at(pos), pos, 0, false);
        require(input.set_embd(idx, { hidden.data(), 1, n_embd }), "attach draft hidden row");
    }
    require(llama_process(ctx.dft, LLAMA_PROCESS_TYPE_DECODE, input.get()) == 0, "draft catch-up succeeds after target destruction");
    require(llama_memory_seq_pos_max(llama_get_memory(ctx.dft), 0) == 159 && scratch_borrows == after_destroy,
            "draft uses private compute after its scratch target is destroyed");
    printf("PASS scratch lifetime: target destruction releases the active loan; the independent draft still decodes without shared compute\n");
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
    if (pin_host) { common_set_env("LLAMA_MTP_PIN_BORROWED_HOST", "1"); }
    if (cuda_sharing) { ggml_backend_load_all(); }
    llama_backend_init();
    int failed = 0;
    llama_model * model = nullptr;
    try {
        if (reserve_probe) {
            run_reserve_probe(std::stoul(argv[2]), std::stoul(argv[3]), argc == 5 ? argv[4] : nullptr);
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
